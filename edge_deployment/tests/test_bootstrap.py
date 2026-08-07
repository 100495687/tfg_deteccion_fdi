"""Tests 9-14 + Caso I (bootstrap insuficiente) y Caso J (reset)."""
from __future__ import annotations

import pandas as pd
import pytest

from edge_deployment.tests.conftest import synthetic_series


def test_9_engine_starts_empty(engine):
    status = engine.get_status("meter_never_seen")
    assert status["engine_status"] == "empty"


def test_10_bootstrap_loads_correct_context(bootstrapped):
    report = bootstrapped["report"]
    assert report["h_ready"] is True
    assert report["p_ready"] is False  # P nunca usa contexto de bootstrap


def test_11_insufficient_context_produces_warming_up(engine):
    corto = synthetic_series(700, start="2021-01-01", seed=5)  # muy por debajo de 673 bins
    report = engine.bootstrap("meter_short_bootstrap", corto)
    assert report["h_ready"] is False
    assert len(report["warnings"]) >= 1
    status = engine.get_status("meter_short_bootstrap")
    assert status["engine_status"] == "warming_up"
    engine.reset("meter_short_bootstrap")


def test_12_p_can_become_ready_before_h_in_status_flags(bootstrapped):
    """Justo tras el bootstrap, H ya esta listo (dispone de 682 bins) pero P todavia no
    (necesita 360 minutos de streaming que el bootstrap no le proporciona)."""
    assert bootstrapped["report"]["h_ready"] is True
    assert bootstrapped["report"]["p_ready"] is False


def test_13_unavailable_branch_not_treated_as_normal(bootstrapped):
    eng, mid, stream_start = bootstrapped["engine"], bootstrapped["meter_id"], bootstrapped["stream_start"]
    resp = eng.ingest(mid, stream_start, 1.0)
    assert resp.detection_source == "partial_availability", (
        "con P todavia no listo, alert_or no debe presentarse como equivalente a la arquitectura completa")


def test_bootstrap_rejects_non_monotonic_index(engine):
    serie = synthetic_series(100, start="2021-02-01", seed=2)
    desordenada = serie.iloc[::-1]
    with pytest.raises(ValueError):
        engine.bootstrap("meter_bad_order", desordenada)


def test_bootstrap_rejects_duplicated_timestamps(engine):
    serie = synthetic_series(100, start="2021-03-01", seed=3)
    con_dup = pd.concat([serie, serie.iloc[[5]]])
    with pytest.raises(ValueError):
        engine.bootstrap("meter_bad_dup", con_dup)


def test_bootstrap_rejects_gaps(engine):
    serie = synthetic_series(100, start="2021-04-01", seed=4)
    con_hueco = pd.concat([serie.iloc[:50], serie.iloc[55:]])
    with pytest.raises(ValueError):
        engine.bootstrap("meter_bad_gap", con_hueco)


def test_bootstrap_rejects_nan(engine):
    serie = synthetic_series(100, start="2021-05-01", seed=6)
    serie = serie.copy()
    serie.iloc[10, 0] = float("nan")
    with pytest.raises(ValueError):
        engine.bootstrap("meter_bad_nan", serie)


def test_bootstrap_must_be_strictly_before_stream_first_reading(bootstrapped):
    """La primera lectura de streaming debe llegar exactamente un minuto despues del ultimo
    timestamp de bootstrap -- si no, se detecta como hueco (nunca se asume continuidad)."""
    eng, mid, stream_start = bootstrapped["engine"], bootstrapped["meter_id"], bootstrapped["stream_start"]
    resp = eng.ingest(mid, stream_start + pd.Timedelta(minutes=5), 1.0)
    assert resp.accepted  # se acepta (timestamp > ultimo aceptado)
    assert "hueco" in " ".join(resp.warnings).lower()


class TestReset:
    def test_j_reset_empties_state(self, bootstrapped):
        eng, mid, stream_start = bootstrapped["engine"], bootstrapped["meter_id"], bootstrapped["stream_start"]
        eng.ingest(mid, stream_start, 1.0)
        assert eng.get_status(mid)["n_accepted"] == 1
        r = eng.reset(mid)
        assert r["reset"] is True and r["existed"] is True
        assert eng.get_status(mid)["engine_status"] == "empty"
        assert eng.get_status(mid)["n_accepted"] == 0

    def test_j_reset_does_not_reload_or_modify_models(self, engine):
        modelo_h_antes = engine.modelo_h
        modelo_p_antes = engine.modelo_p
        engine.reset("meter_inexistente")
        assert engine.modelo_h is modelo_h_antes
        assert engine.modelo_p is modelo_p_antes

    def test_j_reset_requires_new_bootstrap_to_become_ready_again(self, bootstrapped):
        eng, mid = bootstrapped["engine"], bootstrapped["meter_id"]
        eng.reset(mid)
        assert eng.get_status(mid)["h_ready"] is False
