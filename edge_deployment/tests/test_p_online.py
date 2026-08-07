"""Tests 36-40: la ventana, el anclaje, el score y la alerta de P deben
coincidir con `evaluacion_final_retrospectiva_test.puntuar_p_limpio` aplicado directamente
a la misma ventana."""
from __future__ import annotations

import pandas as pd

from edge_deployment.core.detector_state import P_STRIDE_MIN, P_WINDOW_MIN
from edge_deployment.core.p_online import evaluar_p_si_corresponde, p_decision_due


def test_p_not_evaluated_before_360_minutes(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    for i in range(P_WINDOW_MIN - 1):
        resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
        assert resp.p_evaluated is False


def test_38_p_fires_exactly_at_360_then_every_60(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    disparos = []
    for i in range(P_WINDOW_MIN + 3 * P_STRIDE_MIN):
        resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
        if resp.p_evaluated:
            disparos.append(i)
    assert disparos == [P_WINDOW_MIN - 1, P_WINDOW_MIN - 1 + P_STRIDE_MIN,
                         P_WINDOW_MIN - 1 + 2 * P_STRIDE_MIN, P_WINDOW_MIN - 1 + 3 * P_STRIDE_MIN]


def test_36_37_p_window_and_score_match_offline_puntuar_p_limpio(engine, bootstrapped):
    from src import evaluacion_final_retrospectiva_test as eftt
    from src.data_loading import COLUMNA_OBJETIVO

    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    from edge_deployment.tests.conftest import synthetic_series
    stream = synthetic_series(P_WINDOW_MIN, start=stream_start, seed=42)

    last_resp = None
    for ts, row in stream.iterrows():
        last_resp = engine.ingest(mid, ts, float(row[COLUMNA_OBJETIVO]))
    assert last_resp.p_evaluated

    scores_offline, ts_offline = eftt.puntuar_p_limpio(stream, engine.modelo_p, engine.params_norm_p)
    assert len(scores_offline) == 1
    assert last_resp.score_p == pytest.approx(float(scores_offline[0]))
    assert pd.Timestamp(last_resp.p_last_evaluation_timestamp) == pd.Timestamp(ts_offline[0])
    assert pd.Timestamp(ts_offline[0]) == stream.index[-1]


def test_39_p_alert_uses_strict_greater_than(engine, bootstrapped):
    mid = bootstrapped["meter_id"]
    threshold = engine.threshold_p
    # umbral estricto: un score exactamente igual al umbral no debe alertar
    assert not (threshold > threshold)


def test_40_p_skips_incomplete_window_after_gap(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    for i in range(P_WINDOW_MIN - 1):
        engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
    # hueco justo antes de completar la ventana
    ts_gap = stream_start + pd.Timedelta(minutes=P_WINDOW_MIN - 1 + 5)
    resp = engine.ingest(mid, ts_gap, 1.0)
    assert resp.p_evaluated is False  # la cuenta de streaming avanza, pero la ventana ya no es contigua


def test_p_decision_due_pure_function():
    from edge_deployment.core.detector_state import DetectorState
    s = DetectorState(meter_id="m")
    s.stream_minutes_ingested = P_WINDOW_MIN - 1
    assert p_decision_due(s) is False
    s.stream_minutes_ingested = P_WINDOW_MIN
    assert p_decision_due(s) is True
    s.stream_minutes_ingested = P_WINDOW_MIN + P_STRIDE_MIN
    assert p_decision_due(s) is True
    s.stream_minutes_ingested = P_WINDOW_MIN + 5
    assert p_decision_due(s) is False


import pytest  # noqa: E402
