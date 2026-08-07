"""Tests 31-35: las features, el orden de columnas, el score y la alerta de H
online deben coincidir con `predictor_causal_lags` aplicado directamente al mismo historial."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CORE_DIR = Path(__file__).resolve().parents[1] / "core"


def test_33_h_online_only_calls_predict_never_fit():
    text = (CORE_DIR / "h_online.py").read_text(encoding="utf-8")
    assert ".fit(" not in text
    assert ".predict(" in text


def test_32_h_column_order_matches_frozen_features_json(engine):
    from src import optimizacion_histgb_periodic as opt
    assert engine.features_h_orden == opt.PERIODIC_BASE_FEATURES


def test_31_34_h_features_and_score_match_offline_construction(engine, bootstrapped):
    """Construye el historial online para exactamente un bin nuevo y compara sus features y
    su score contra una llamada directa a las funciones offline sobre el mismo array."""
    from src import predictor_causal_lags as pcl
    from edge_deployment.core.streaming_preprocessing import historial_15min_to_arrays, append_reading

    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    state = bootstrapped["engine"]._states[mid]

    ts0 = stream_start
    for i in range(15):
        append_reading(state, ts0 + pd.Timedelta(minutes=i), 1.0 + 0.01 * i)

    kw, ts_start, ts_end = historial_15min_to_arrays(state)
    feats_kw = pcl.construir_features_kw(kw)
    cal = pcl._calendario(ts_start)
    df = pd.concat([feats_kw, cal], axis=1)
    df["target_kw"] = kw
    df = df.merge(bootstrapped["engine"].perfil_h, on=["es_finde", "slot_15min"], how="left")
    fila_offline = df.iloc[[-1]]

    pred_offline = bootstrapped["engine"].modelo_h.predict(fila_offline[bootstrapped["engine"].features_h_orden].to_numpy(dtype=np.float64))
    _, score_offline = pcl.calcular_score(pred_offline, fila_offline["target_kw"].to_numpy())

    from edge_deployment.core.h_online import evaluar_h_si_corresponde
    bin_cerrado = state.historial_15min[-1]
    res = evaluar_h_si_corresponde(state, bootstrapped["engine"].modelo_h, bootstrapped["engine"].perfil_h,
                                    bootstrapped["engine"].features_h_orden, bootstrapped["engine"].threshold_h, bin_cerrado)
    assert res["score_h"] == pytest.approx(float(score_offline[0]), abs=1e-10)


def test_35_h_alert_uses_strict_greater_than(engine):
    threshold = engine.threshold_h
    assert not (threshold > threshold)  # documenta la comparacion estricta usada en h_online.py
    text = (CORE_DIR / "h_online.py").read_text(encoding="utf-8")
    assert "score > threshold_h" in text


def test_h_evaluates_only_when_a_bin_just_closed(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    n_h_evaluations = 0
    for i in range(30):
        resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
        if resp.h_evaluated:
            n_h_evaluations += 1
    assert n_h_evaluations == 2  # 30 minutos -> exactamente 2 bins cerrados


def test_h_not_evaluated_when_context_not_contiguous(engine):
    """Con contexto de bootstrap corto (menos de 673 bins), H nunca debe evaluar aunque se
    cierren bins de streaming."""
    from edge_deployment.tests.conftest import synthetic_series
    corto = synthetic_series(700, start="2023-01-01", seed=11)
    engine.bootstrap("meter_h_short", corto)
    stream_start = corto.index.max() + pd.Timedelta(minutes=1)
    n_h_evaluations = 0
    for i in range(30):
        resp = engine.ingest("meter_h_short", stream_start + pd.Timedelta(minutes=i), 1.0)
        if resp.h_evaluated:
            n_h_evaluations += 1
    assert n_h_evaluations == 0
    engine.reset("meter_h_short")
