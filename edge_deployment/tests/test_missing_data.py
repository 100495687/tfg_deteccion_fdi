"""Tests 22-24 + Casos E/F/G/H: huecos temporales -- nunca se
rellenan con cero ni se interpola, y la recuperacion es causal (nunca retroactiva)."""
from __future__ import annotations

import pandas as pd
import pytest

from edge_deployment.core.detector_state import P_WINDOW_MIN


def test_22_gap_is_detected(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    engine.ingest(mid, stream_start, 1.0)
    resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=10), 1.0)
    assert resp.accepted
    assert any("hueco" in w.lower() for w in resp.warnings)
    assert engine.get_status(mid)["n_gaps"] == 1


def test_23_gap_never_filled_with_zero():
    """El motor nunca inserta un valor 0.0 sintetico: el bucket con el hueco se descarta
    entero (ver streaming_preprocessing.append_reading), nunca se promedia con ceros."""
    from edge_deployment.core.detector_state import DetectorState
    from edge_deployment.core.streaming_preprocessing import append_reading

    state = DetectorState(meter_id="m")
    ts0 = pd.Timestamp("2024-01-01")
    for i in range(10):
        append_reading(state, ts0 + pd.Timedelta(minutes=i), 5.0)
    append_reading(state, ts0 + pd.Timedelta(minutes=20), 5.0)  # hueco
    assert len(state.historial_15min) == 0  # el bucket de 10 nunca se cerro con ceros de relleno


def test_24_no_interpolation_source_check():
    from pathlib import Path
    core_dir = Path(__file__).resolve().parents[1] / "core"
    for f in core_dir.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert ".interpolate(" not in text
        assert "fillna(0" not in text.replace(" ", "")


def test_case_e_nan_reading_rejected_state_unchanged(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    engine.ingest(mid, stream_start, 1.0)
    before = engine.get_status(mid)["buffer_1min_size"]
    resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=1), float("nan"))
    assert not resp.accepted and resp.rejection_reason == "nan_value"
    assert engine.get_status(mid)["buffer_1min_size"] == before


def test_case_f_infinite_reading_rejected(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    resp = engine.ingest(mid, stream_start, float("inf"))
    assert not resp.accepted and resp.rejection_reason == "infinite_value"


def test_case_g_negative_power_same_policy_as_offline(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    resp = engine.ingest(mid, stream_start, -1.0)
    assert resp.accepted  # el pipeline offline tampoco filtra negativos (ver test_streaming_validation)


def test_case_h_missing_minute_blocks_affected_p_window_then_recovers(engine, bootstrapped):
    """El disparo de P sigue el contador global de minutos de streaming (identico al indexado
    posicional offline) -- un hueco no lo reinicia. Lo que si debe ocurrir es que
    P nunca evalue mientras su ventana de 360 min siga conteniendo el hueco, y que en cuanto
    dispare de nuevo, su ventana sea integramente posterior al hueco."""
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    for i in range(100):
        engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
    ts_gap = stream_start + pd.Timedelta(minutes=150)  # hueco de 51 min
    engine.ingest(mid, ts_gap, 1.0)

    resp = None
    for i in range(1, 400):  # suficientes minutos post-hueco para alcanzar el siguiente disparo valido
        resp = engine.ingest(mid, ts_gap + pd.Timedelta(minutes=i), 1.0)
        if resp.p_evaluated:
            break
    assert resp is not None and resp.p_evaluated is True
    ventana_inicio = pd.Timestamp(resp.p_last_evaluation_timestamp) - pd.Timedelta(minutes=P_WINDOW_MIN - 1)
    assert ventana_inicio >= ts_gap, "la ventana de P que disparo no debe contener minutos anteriores al hueco"


def test_case_h_recovery_p_ready_again(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    for i in range(P_WINDOW_MIN):
        engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
    assert engine.get_status(mid)["p_ready"] is True

    ts_gap = stream_start + pd.Timedelta(minutes=P_WINDOW_MIN + 20)
    engine.ingest(mid, ts_gap, 1.0)
    assert engine.get_status(mid)["p_ready"] is False  # la ventana ya no es contigua

    for i in range(1, P_WINDOW_MIN):
        engine.ingest(mid, ts_gap + pd.Timedelta(minutes=i), 1.0)
    assert engine.get_status(mid)["p_ready"] is True


def test_case_h_recovery_h_ready_again_after_gap_shrinks_history(engine):
    """Un hueco no vacia el historial de H (los bins ya cerrados siguen siendo validos),
    solo descarta el bucket parcial en curso -- H sigue listo inmediatamente."""
    from edge_deployment.tests.conftest import synthetic_series
    boot = synthetic_series(10230, start="2025-01-01", seed=21)
    engine.bootstrap("meter_h_gap", boot)
    stream_start = boot.index.max() + pd.Timedelta(minutes=1)
    assert engine.get_status("meter_h_gap")["h_ready"] is True
    resp = engine.ingest("meter_h_gap", stream_start + pd.Timedelta(minutes=10), 1.0)  # hueco de 10 min
    assert resp.accepted
    assert engine.get_status("meter_h_gap")["h_ready"] is True
    engine.reset("meter_h_gap")
