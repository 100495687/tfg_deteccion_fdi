"""Fase 3: carga de trabajo sintetica compartida por las matrices de
memoria/CPU y la prueba prolongada. No usa el periodo pre-test real (evita depender de
core/period_selection.py, que no forma parte de la imagen -- ver build_docker_context.py) --
en su lugar reutiliza el mismo generador sintetico de edge_deployment/tests/conftest.py,
duplicado deliberadamente (fixture de pruebas, no logica del detector). El objetivo de estos
experimentos es caracterizar recursos (memoria/CPU/latencia) bajo restricciones, no repetir la
validacion de scores exactos contra el dataset real -- eso ya se hizo, con datos reales, en
equivalence.py y debe haber pasado (100% alertas) antes de llegar aqui.

Aun asi, cada contenedor se compara contra una referencia fija (motor directo, mismo patron de
Fase 1/2) para detectar cualquier cambio de alerta inducido por una restriccion de recursos
("Las restricciones pueden aumentar la latencia, pero no deben cambiar scores o
alertas").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BOOTSTRAP_MINUTES = 10230  # MAX_LAG_H_BINS(672)+H_BOOTSTRAP_MARGIN_BINS(10) * 15 -- H ready de inmediato
DEFAULT_STREAM_MINUTES = 500  # > P_WINDOW_MIN(360) -- P tambien llega a evaluar al menos una vez


def synthetic_series(n_minutes: int, start: str, seed: int, base: float = 1.0, amp: float = 0.5) -> pd.DataFrame:
    from src.data_loading import COLUMNA_OBJETIVO
    idx = pd.date_range(start, periods=n_minutes, freq="1min")
    t = np.arange(n_minutes)
    rng = np.random.default_rng(seed)
    valores = base + amp * np.sin(2 * np.pi * t / 1440.0) + 0.05 * rng.standard_normal(n_minutes)
    valores = np.clip(valores, 0.05, None)
    return pd.DataFrame({COLUMNA_OBJETIVO: valores, "imputado": False}, index=idx)


def build_bootstrap_and_stream(bootstrap_minutes: int = BOOTSTRAP_MINUTES, stream_minutes: int = DEFAULT_STREAM_MINUTES,
                                seed_bootstrap: int = 1, seed_stream: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_df = synthetic_series(bootstrap_minutes, "2020-01-01 00:00:00", seed_bootstrap)
    stream_start = bootstrap_df.index.max() + pd.Timedelta(minutes=1)
    stream_df = synthetic_series(stream_minutes, stream_start.isoformat(), seed_stream)
    return bootstrap_df, stream_df


def compute_reference(bootstrap_df: pd.DataFrame, stream_df: pd.DataFrame, meter_id: str = "ref_meter") -> list[dict]:
    """Motor directo (sin HTTP), la misma referencia que Fase 1/2 usan para probar
    equivalencia -- se instancia una unica vez por proceso (host, sin Docker)."""
    from edge_deployment.clients.direct_engine_client import DirectEngineClient
    from src.data_loading import COLUMNA_OBJETIVO

    direct = DirectEngineClient()
    direct.bootstrap(meter_id, bootstrap_df)
    filas = []
    for ts, row in stream_df.iterrows():
        d = direct.ingest(meter_id, ts, float(row[COLUMNA_OBJETIVO]))
        filas.append({"timestamp": ts, "accepted": d["accepted"], "p_evaluated": d["p_evaluated"], "h_evaluated": d["h_evaluated"],
                       "score_p": d["score_p"], "score_h": d["score_h"], "alert_p": d["alert_p"], "alert_h": d["alert_h"],
                       "alert_p_ffill": d["alert_p_ffill"], "alert_or": d["alert_or"], "detection_source": d["detection_source"]})
    return filas


def replay_via_http(base_url: str, bootstrap_df: pd.DataFrame, stream_df: pd.DataFrame, meter_id: str = "matrix_meter",
                     timeout: float = 30.0) -> tuple[list[dict], dict]:
    import time

    import httpx

    from src.data_loading import COLUMNA_OBJETIVO

    client = httpx.Client(base_url=base_url, timeout=timeout)
    t0 = time.perf_counter()
    r = client.post("/bootstrap", json={"meter_id": meter_id, "readings": [
        {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(bootstrap_df.index, bootstrap_df[COLUMNA_OBJETIVO])
    ]})
    bootstrap_ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise RuntimeError(f"bootstrap fallo: {r.status_code} {r.text[:500]}")

    filas, latencias_ms, errores, perdidas = [], [], 0, 0
    for ts, row in stream_df.iterrows():
        t0 = time.perf_counter()
        try:
            r = client.post("/readings", json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})
        except Exception:
            perdidas += 1
            continue
        dt_ms = (time.perf_counter() - t0) * 1000
        latencias_ms.append(dt_ms)
        if r.status_code >= 500:
            errores += 1
            continue
        body = r.json()
        filas.append({"timestamp": ts, "accepted": body.get("accepted"), "p_evaluated": body.get("p_evaluated"),
                       "h_evaluated": body.get("h_evaluated"), "score_p": body.get("score_p"), "score_h": body.get("score_h"),
                       "alert_p": body.get("alert_p"), "alert_h": body.get("alert_h"), "alert_p_ffill": body.get("alert_p_ffill"),
                       "alert_or": body.get("alert_or"), "detection_source": body.get("detection_source"),
                       "round_trip_ms": dt_ms})
    client.close()
    meta = {"bootstrap_ms": bootstrap_ms, "n_sent": len(stream_df), "n_ok": len(filas), "n_errors_5xx": errores,
            "n_lost": perdidas, "latencias_ms": latencias_ms}
    return filas, meta


def compare_to_reference(reference: list[dict], actual: list[dict]) -> dict:
    fields = ["accepted", "p_evaluated", "h_evaluated", "alert_p", "alert_h", "alert_p_ffill", "alert_or", "detection_source"]
    n = min(len(reference), len(actual))
    if n == 0:
        return {"n_compared": 0, "all_100pct": False, "match_pct": {}}
    match_pct = {}
    for f in fields:
        iguales = sum(1 for i in range(n) if reference[i][f] == actual[i][f])
        match_pct[f] = 100.0 * iguales / n
    diffs_p = [abs((actual[i]["score_p"] or 0) - (reference[i]["score_p"] or 0)) for i in range(n)
               if actual[i]["score_p"] is not None and reference[i]["score_p"] is not None]
    diffs_h = [abs((actual[i]["score_h"] or 0) - (reference[i]["score_h"] or 0)) for i in range(n)
               if actual[i]["score_h"] is not None and reference[i]["score_h"] is not None]
    return {
        "n_compared": n, "match_pct": match_pct, "all_100pct": all(v == 100.0 for v in match_pct.values()),
        "score_p_max_abs_diff": max(diffs_p) if diffs_p else None, "score_h_max_abs_diff": max(diffs_h) if diffs_h else None,
        "n_length_mismatch": len(reference) != len(actual),
    }
