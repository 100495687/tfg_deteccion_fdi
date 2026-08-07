"""Tests 41-43: fusion OR y propagacion causal de P."""
from __future__ import annotations

import pandas as pd
import pytest

from edge_deployment.core.fusion_online import fusionar


def test_43_or_equals_p_or_h_when_both_ready():
    for p in (True, False):
        for h in (True, False):
            r = fusionar(p, True, h, True)
            assert r["alert_or"] == (p or h)


def test_detection_source_categories_when_both_ready():
    assert fusionar(True, True, True, True)["detection_source"] == "both"
    assert fusionar(True, True, False, True)["detection_source"] == "P_only"
    assert fusionar(False, True, True, True)["detection_source"] == "H_only"
    assert fusionar(False, True, False, True)["detection_source"] == "none"


def test_partial_availability_when_only_one_branch_ready():
    r = fusionar(False, True, None, False)  # solo P listo
    assert r["detection_source"] == "partial_availability"
    r2 = fusionar(False, False, True, True)  # solo H listo
    assert r2["detection_source"] == "partial_availability"


def test_none_when_nothing_ready():
    r = fusionar(False, False, None, False)
    assert r["alert_or"] is False and r["detection_source"] == "none"


def test_41_42_p_propagation_uses_last_decision_leq_now(engine, bootstrapped):
    """`DetectorState.active_p_ffill` debe ser equivalente a
    `fusion_p_histgb.estado_ffill` consultado en el instante actual: False antes de la
    primera decision, y despues, el ultimo valor conocido (nunca uno futuro)."""
    from src import fusion_p_histgb as fus

    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    state = engine._states[mid]
    assert state.active_p_ffill() is False  # antes de la primera decision de P

    from edge_deployment.core.detector_state import P_WINDOW_MIN
    for i in range(P_WINDOW_MIN):
        engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
    assert state.p_last_evaluation_timestamp is not None

    ts_p = pd.DatetimeIndex([state.p_last_evaluation_timestamp])
    score_p = pd.array([state.last_score_p], dtype=float)
    objetivo = pd.DatetimeIndex([state.p_last_evaluation_timestamp + pd.Timedelta(minutes=30)])
    activo_offline = fus.estado_ffill(score_p.to_numpy(), engine.threshold_p, ts_p.to_numpy(), objetivo.to_numpy())
    assert bool(activo_offline[0]) == state.active_p_ffill()


def test_causality_p_state_never_uses_a_future_decision(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    from edge_deployment.core.detector_state import P_WINDOW_MIN
    for i in range(P_WINDOW_MIN - 1):
        resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
        assert resp.alert_p_ffill is False  # todavia no hay ninguna decision de P
