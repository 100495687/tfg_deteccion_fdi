"""Tests 27-30: la agregacion online a 15 min debe reproducir exactamente
`predictor_causal_lags.construir_serie_15min` (bins de 15 muestras consecutivas, sin
solape, media aritmetica, timestamp = ultima muestra del bin)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edge_deployment.core.detector_state import DetectorState
from edge_deployment.core.streaming_preprocessing import append_reading
from edge_deployment.tests.conftest import synthetic_series


def test_27_28_bin_closes_exactly_at_15_readings_and_matches_offline():
    from src import predictor_causal_lags as pcl

    serie = synthetic_series(45, start="2022-01-01", seed=7)  # exactamente 3 bins
    state = DetectorState(meter_id="m")
    for ts, row in serie.iterrows():
        append_reading(state, ts, float(row.iloc[0]))

    assert len(state.historial_15min) == 3
    offline = pcl.construir_serie_15min(serie)
    for i, b in enumerate(state.historial_15min):
        assert b.kw == pytest.approx(offline["kw"][i])
        assert pd.Timestamp(b.ts_start) == pd.Timestamp(offline["ts_start"][i])
        assert pd.Timestamp(b.ts_end) == pd.Timestamp(offline["ts_end"][i])


def test_29_bin_timestamp_is_last_sample_not_first():
    serie = synthetic_series(15, start="2022-02-01", seed=8)
    state = DetectorState(meter_id="m")
    for ts, row in serie.iterrows():
        append_reading(state, ts, float(row.iloc[0]))
    b = state.historial_15min[0]
    assert pd.Timestamp(b.ts_end) == serie.index[-1]
    assert pd.Timestamp(b.ts_start) == serie.index[0]


def test_30_incomplete_bin_never_closed():
    serie = synthetic_series(14, start="2022-03-01", seed=9)  # una lectura menos de un bin
    state = DetectorState(meter_id="m")
    for ts, row in serie.iterrows():
        append_reading(state, ts, float(row.iloc[0]))
    assert len(state.historial_15min) == 0
    assert len(state.open_bucket_values) == 14


def test_bin_is_arithmetic_mean():
    ts0 = pd.Timestamp("2022-04-01")
    state = DetectorState(meter_id="m")
    valores = list(range(1, 16))
    for i, v in enumerate(valores):
        append_reading(state, ts0 + pd.Timedelta(minutes=i), float(v))
    assert state.historial_15min[0].kw == pytest.approx(np.mean(valores))


def test_gap_discards_partial_bucket_without_closing_it():
    ts0 = pd.Timestamp("2022-05-01")
    state = DetectorState(meter_id="m")
    for i in range(10):
        append_reading(state, ts0 + pd.Timedelta(minutes=i), 1.0)
    assert len(state.open_bucket_values) == 10
    append_reading(state, ts0 + pd.Timedelta(minutes=25), 1.0)  # salto -> hueco
    assert len(state.open_bucket_values) == 1  # se reinicia, no se cierra el parcial de 10
    assert len(state.historial_15min) == 0
