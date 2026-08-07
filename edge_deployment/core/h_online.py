"""Inferencia online de H. Reutiliza literalmente, sobre una rejilla acotada
(el `historial_15min` del estado, no la serie completa):
  - `predictor_causal_lags.construir_features_kw` (lags + rolling, siempre con shift previo)
  - `predictor_causal_lags._calendario`
  - `predictor_causal_lags.calcular_score`
  - el merge con el perfil historico congelado (`profile_train_final.joblib`)

Solo llama a `modelo.predict` (nunca `fit`). Las columnas se pasan en el orden congelado
(`features_h_final.json` == `optimizacion_histgb_periodic.PERIODIC_BASE_FEATURES`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edge_deployment.core.detector_state import DetectorState, MAX_LAG_H_BINS
from edge_deployment.core.streaming_preprocessing import historial_15min_to_arrays, historial_contiguous_tail


def h_decision_due(bin_cerrado) -> bool:
    return bin_cerrado is not None


def evaluar_h_si_corresponde(state: DetectorState, modelo_h, perfil_h, features_h_orden: list[str],
                              threshold_h: float, bin_cerrado) -> dict | None:
    if not h_decision_due(bin_cerrado):
        return None
    if not historial_contiguous_tail(state, MAX_LAG_H_BINS + 1):
        # contexto insuficiente o no contiguo (hueco reciente) -- no evaluable
        return None

    from src import predictor_causal_lags as pcl  # import perezoso, mismo patron que el resto del motor

    kw, ts_start, ts_end = historial_15min_to_arrays(state)
    feats_kw = pcl.construir_features_kw(kw)
    cal = pcl._calendario(ts_start)
    df = pd.concat([feats_kw, cal], axis=1)
    df["target_kw"] = kw
    df["ts_start"] = ts_start
    df["ts_end"] = ts_end
    df = df.merge(perfil_h, on=["es_finde", "slot_15min"], how="left")

    fila = df.iloc[[-1]]
    assert pd.Timestamp(fila["ts_end"].iloc[0]) == pd.Timestamp(bin_cerrado.ts_end), \
        "la ultima fila del historial no corresponde al bin recien cerrado"
    if fila[features_h_orden].isna().to_numpy().any():
        state.recent_warnings.append(f"features de H con NaN pese a contexto contiguo suficiente en {bin_cerrado.ts_end}")
        return None

    pred = modelo_h.predict(fila[features_h_orden].to_numpy(dtype=np.float64))
    _, positive = pcl.calcular_score(pred, fila["target_kw"].to_numpy())
    score = float(positive[0])
    active = bool(score > threshold_h)
    ts_decision = pd.Timestamp(bin_cerrado.ts_end)
    return {"score_h": score, "timestamp": ts_decision, "active_h": active, "predicted_kw": float(pred[0]), "observed_kw": float(bin_cerrado.kw)}
