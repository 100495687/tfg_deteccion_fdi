"""Inferencia online de P. Envuelve, sin modificar,
`evaluacion_final_retrospectiva_test.puntuar_p_limpio` (que a su vez reutiliza
`windowing.ventanear` + `normalization.aplicar_zscore` + `modelo.reconstruir` +
`base_relative.calcular_relative_kw`), aplicada a una ventana de exactamente 360 minutos
extraida del buffer de streaming.

Anclaje verificado en el codigo offline (`evaluate_replay.py::puntuar_pool`,
`fusion_p_histgb.cargar_p_congelado`): `wnd.ventanear(particion, 360, 60)` indexa de forma
posicional desde el inicio de la particion que se le pasa -- nunca desde un huso horario ni
desde "cada hora en punto". Para ser equivalente, el motor online ventanea exclusivamente
sobre lecturas de streaming (nunca del bootstrap): la ventana i-esima cubre los minutos de
streaming [60*i, 60*i+360), con decision en el minuto 60*i+359 -- exactamente la misma
rejilla que `ventanear` produciria sobre la particion de streaming completa.
"""
from __future__ import annotations

import pandas as pd

from edge_deployment.core.detector_state import DetectorState, P_STRIDE_MIN, P_WINDOW_MIN
from edge_deployment.core.streaming_preprocessing import get_window, has_complete_window


def p_decision_due(state: DetectorState) -> bool:
    return state.stream_minutes_ingested >= P_WINDOW_MIN and state.stream_minutes_ingested % P_STRIDE_MIN == 0


def evaluar_p_si_corresponde(state: DetectorState, modelo_p, params_norm_p, threshold_p: float) -> dict | None:
    if not p_decision_due(state):
        return None
    if not has_complete_window(state, P_WINDOW_MIN):
        # ventana afectada por un hueco -- no evaluable hasta acumular 360 min contiguos
        return None

    from src import evaluacion_final_retrospectiva_test as eftt  # import perezoso: evita ciclos, mismo patron que replay_pilot

    df_360 = get_window(state, P_WINDOW_MIN)
    scores, ts = eftt.puntuar_p_limpio(df_360, modelo_p, params_norm_p)
    assert len(scores) == 1, f"se esperaba exactamente 1 ventana de P, se obtuvieron {len(scores)}"
    ts_decision = pd.Timestamp(ts[0])
    assert ts_decision == df_360.index[-1], "el timestamp de decision de P no coincide con el fin de la ventana"

    score = float(scores[0])
    active = bool(score > threshold_p)
    return {"score_p": score, "timestamp": ts_decision, "active_p": active}
