"""Tests 19, 25-26: buffer de 1 minuto acotado, ordenado, ventanas exactas."""
from __future__ import annotations

import pandas as pd

from edge_deployment.core.detector_state import BUFFER_1MIN_MAXLEN, DetectorState, P_WINDOW_MIN
from edge_deployment.core.streaming_preprocessing import append_reading, get_window, has_complete_window


def _fill(state, n, start="2020-01-01 00:00:00"):
    ts0 = pd.Timestamp(start)
    for i in range(n):
        append_reading(state, ts0 + pd.Timedelta(minutes=i), 1.0 + 0.01 * i)


def test_25_buffer_orders_readings_chronologically():
    state = DetectorState(meter_id="m")
    _fill(state, 20)
    ts = [t for t, _ in state.buffer_1min]
    assert ts == sorted(ts)


def test_26_buffer_bounded_maxlen():
    state = DetectorState(meter_id="m")
    _fill(state, BUFFER_1MIN_MAXLEN + 100)
    assert len(state.buffer_1min) == BUFFER_1MIN_MAXLEN


def test_get_window_none_if_incomplete():
    state = DetectorState(meter_id="m")
    _fill(state, P_WINDOW_MIN - 1)
    assert get_window(state, P_WINDOW_MIN) is None


def test_get_window_returns_expected_shape_and_columns():
    state = DetectorState(meter_id="m")
    _fill(state, P_WINDOW_MIN)
    from src.data_loading import COLUMNA_OBJETIVO
    w = get_window(state, P_WINDOW_MIN)
    assert w is not None
    assert len(w) == P_WINDOW_MIN
    assert COLUMNA_OBJETIVO in w.columns and "imputado" in w.columns
    assert isinstance(w.index, pd.DatetimeIndex)


def test_get_window_is_the_trailing_slice():
    state = DetectorState(meter_id="m")
    _fill(state, P_WINDOW_MIN + 10)
    from src.data_loading import COLUMNA_OBJETIVO
    w = get_window(state, P_WINDOW_MIN)
    valores_directos = [v for _, v in list(state.buffer_1min)[-P_WINDOW_MIN:]]
    assert list(w[COLUMNA_OBJETIVO]) == valores_directos


def test_has_complete_window_false_with_gap_in_trailing_window():
    state = DetectorState(meter_id="m")
    _fill(state, P_WINDOW_MIN)
    # simula un hueco: inserta una lectura con salto de 10 minutos, seguida de menos de
    # P_WINDOW_MIN lecturas contiguas nuevas
    ts_gap = list(state.buffer_1min)[-1][0] + pd.Timedelta(minutes=10)
    append_reading(state, ts_gap, 2.0)
    assert has_complete_window(state, P_WINDOW_MIN) is False


def test_has_complete_window_recovers_after_enough_contiguous_readings():
    state = DetectorState(meter_id="m")
    _fill(state, P_WINDOW_MIN)
    ts_gap = list(state.buffer_1min)[-1][0] + pd.Timedelta(minutes=10)
    append_reading(state, ts_gap, 2.0)
    for i in range(1, P_WINDOW_MIN):
        append_reading(state, ts_gap + pd.Timedelta(minutes=i), 2.0)
    assert has_complete_window(state, P_WINDOW_MIN) is True


def test_validate_continuity_pure_function():
    from edge_deployment.core.streaming_preprocessing import validate_continuity
    ts0 = pd.Timestamp("2020-01-01")
    contiguo = [(ts0 + pd.Timedelta(minutes=i), 1.0) for i in range(5)]
    assert validate_continuity(contiguo) is True
    con_hueco = contiguo[:3] + [(ts0 + pd.Timedelta(minutes=10), 1.0)]
    assert validate_continuity(con_hueco) is False
