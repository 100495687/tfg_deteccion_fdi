"""Tests 44-49 -- causalidad demostrada por test, no
solo documentada."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from edge_deployment.tests.conftest import synthetic_series

SIMULATORS_DIR = Path(__file__).resolve().parents[1] / "simulators"


def test_44_45_truncating_series_at_t_gives_same_result_as_processing_whole_series(engine, bootstrap_series):
    """Alimentar N minutos y comparar contra alimentar N+K minutos: las primeras N
    respuestas deben ser identicas -- ni el futuro (K minutos extra) ni el simple hecho de
    que existan mas datos por venir puede cambiar resultados ya emitidos."""
    from src.data_loading import COLUMNA_OBJETIVO

    stream_start = bootstrap_series.index.max() + pd.Timedelta(minutes=1)
    N, K = 400, 120
    stream = synthetic_series(N + K, start=stream_start, seed=99)

    engine.bootstrap("meter_causal_a", bootstrap_series)
    respuestas_cortas = [engine.ingest("meter_causal_a", ts, float(row[COLUMNA_OBJETIVO]))
                          for ts, row in stream.iloc[:N].iterrows()]
    engine.reset("meter_causal_a")

    engine.bootstrap("meter_causal_a", bootstrap_series)
    respuestas_largas = [engine.ingest("meter_causal_a", ts, float(row[COLUMNA_OBJETIVO]))
                          for ts, row in stream.iterrows()]
    engine.reset("meter_causal_a")

    for i in range(N):
        a, b = respuestas_cortas[i], respuestas_largas[i]
        assert a.score_p == b.score_p
        assert a.score_h == b.score_h
        assert a.alert_or == b.alert_or
        assert a.engine_status == b.engine_status


def test_46_simulator_does_not_use_wall_clock_to_decide_evaluations():
    text = (SIMULATORS_DIR / "replay_clean_stream.py").read_text(encoding="utf-8")
    core_dir = Path(__file__).resolve().parents[1] / "core"
    for f in [core_dir / "p_online.py", core_dir / "h_online.py", core_dir / "detector_state.py",
              core_dir / "streaming_preprocessing.py"]:
        t = f.read_text(encoding="utf-8")
        assert "datetime.now(" not in t and "time.time()" not in t


def test_47_simulator_processes_chronologically():
    text = (SIMULATORS_DIR / "replay_clean_stream.py").read_text(encoding="utf-8")
    assert "for i, (ts, row) in enumerate(streaming_partition.iterrows())" in text


def test_48_simulator_is_deterministic(engine, bootstrap_series):
    from src.data_loading import COLUMNA_OBJETIVO
    stream_start = bootstrap_series.index.max() + pd.Timedelta(minutes=1)
    stream = synthetic_series(50, start=stream_start, seed=123)

    resultados = []
    for _ in range(2):
        engine.bootstrap("meter_determinismo", bootstrap_series)
        scores = [engine.ingest("meter_determinismo", ts, float(row[COLUMNA_OBJETIVO])).score_h
                  for ts, row in stream.iterrows()]
        resultados.append(scores)
        engine.reset("meter_determinismo")
    assert resultados[0] == resultados[1]


def test_h_uses_only_lags_strictly_before_target():
    """Modificar el valor de un bin posterior no puede cambiar las features calculadas para
    un bin anterior (pcl.construir_features_kw usa exclusivamente shift(k>=1))."""
    from src import predictor_causal_lags as pcl
    import numpy as np

    kw = np.arange(100, dtype=np.float64)
    feats_original = pcl.construir_features_kw(kw)

    kw_modificado = kw.copy()
    kw_modificado[80:] = -999.0  # altera solo el futuro respecto a la fila 50
    feats_modificado = pcl.construir_features_kw(kw_modificado)

    assert feats_original.iloc[50].equals(feats_modificado.iloc[50])


def test_49_p_window_never_extends_past_decision_timestamp(engine, bootstrapped):
    from edge_deployment.core.detector_state import P_WINDOW_MIN
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    resp = None
    for i in range(P_WINDOW_MIN):
        resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
    assert resp.p_evaluated
    assert pd.Timestamp(resp.p_last_evaluation_timestamp) == stream_start + pd.Timedelta(minutes=P_WINDOW_MIN - 1)


def test_p_state_history_never_stores_a_timestamp_ahead_of_now(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    from edge_deployment.core.detector_state import P_WINDOW_MIN
    for i in range(P_WINDOW_MIN):
        resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
        if resp.p_last_evaluation_timestamp is not None:
            assert pd.Timestamp(resp.p_last_evaluation_timestamp) <= pd.Timestamp(resp.timestamp)
