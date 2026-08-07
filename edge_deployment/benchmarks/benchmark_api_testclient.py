"""Benchmark TestClient: separa la latencia total via TestClient (Pydantic +
rutas + serializacion + motor) de la latencia interna del motor (`engine_processing_time_ms`,
ya devuelta por la propia respuesta) para aislar el overhead que anade la capa HTTP/Pydantic.
Sin tracemalloc (igual que en Fase 1). Un calentamiento previo (200 lecturas) antes de medir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

WARMUP_READINGS = 200
DEFAULT_N_READINGS = 10000


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    arr = np.array(values, dtype=float)
    return {"n": int(len(arr)), "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)), "max_ms": float(arr.max())}


def run_benchmark(n_readings: int = DEFAULT_N_READINGS, warmup: int = WARMUP_READINGS) -> dict:
    from fastapi.testclient import TestClient

    from edge_deployment.api.main import app
    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    max_days = max(8, (n_readings + warmup) // 1440 + 2)
    periodo = seleccionar_periodo(max_stream_days=max_days)
    stream = periodo["streaming_partition"]
    assert len(stream) >= n_readings + warmup, "periodo de streaming insuficiente para el benchmark solicitado"

    meter_id = "bench_testclient"
    with TestClient(app) as client:
        r = client.post("/bootstrap", json={"meter_id": meter_id, "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(periodo["bootstrap_partition"].index, periodo["bootstrap_partition"][COLUMNA_OBJETIVO])
        ]})
        assert r.status_code == 200, r.text

        for ts, row in stream.iloc[:warmup].iterrows():
            client.post("/readings", json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})

        filas = []
        for ts, row in stream.iloc[warmup:warmup + n_readings].iterrows():
            t0 = time.perf_counter()
            r = client.post("/readings", json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})
            testclient_total_ms = (time.perf_counter() - t0) * 1000
            body = r.json()
            filas.append({
                "timestamp": ts, "testclient_total_ms": testclient_total_ms,
                "api_processing_time_ms": body.get("api_processing_time_ms"),
                "engine_processing_time_ms": body.get("engine_processing_time_ms"),
                "p_evaluated": body.get("p_evaluated"), "h_evaluated": body.get("h_evaluated"),
            })
        client.post(f"/reset/{meter_id}")

    df = pd.DataFrame(filas)
    df["pydantic_routing_overhead_ms"] = df["testclient_total_ms"] - df["engine_processing_time_ms"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "api_testclient_latency.csv", index=False)

    resumen = {
        "n_readings": len(df), "warmup_readings": warmup,
        "testclient_total": _percentiles(df["testclient_total_ms"].tolist()),
        "engine_internal": _percentiles(df["engine_processing_time_ms"].tolist()),
        "pydantic_routing_overhead": _percentiles(df["pydantic_routing_overhead_ms"].tolist()),
        "by_branch": {
            "no_model": _percentiles(df.loc[~df["p_evaluated"] & ~df["h_evaluated"], "testclient_total_ms"].tolist()),
            "h_only": _percentiles(df.loc[df["h_evaluated"] & ~df["p_evaluated"], "testclient_total_ms"].tolist()),
            "p_and_h": _percentiles(df.loc[df["p_evaluated"] & df["h_evaluated"], "testclient_total_ms"].tolist()),
        },
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "api_testclient_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    return resumen


def run_memory_profile(days_checkpoints: tuple[int, ...] = (1, 3, 7)) -> pd.DataFrame:
    """Memoria RSS del proceso (via `psutil`, nunca `tracemalloc` durante un
    benchmark de latencia/memoria a esta escala -- ver Fase 1) antes/despues de cargar
    modelos, tras el bootstrap y en varios puntos de streaming. Confirma que la memoria no
    crece linealmente con el numero de lecturas (los buffers de `DetectorState` estan
    acotados, ver Fase 1)."""
    import os

    import psutil
    from fastapi.testclient import TestClient

    from edge_deployment.api.main import app
    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    proc = psutil.Process(os.getpid())
    filas = []

    def _snap(etapa: str) -> None:
        filas.append({"stage": etapa, "rss_mb": proc.memory_info().rss / 1e6,
                      "fecha_utc": datetime.now(timezone.utc).isoformat()})

    _snap("before_models_loaded")
    periodo = seleccionar_periodo(max_stream_days=max(days_checkpoints))
    stream = periodo["streaming_partition"]
    meter_id = "mem_profile"

    with TestClient(app) as client:
        _snap("after_models_loaded")
        r = client.post("/bootstrap", json={"meter_id": meter_id, "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(periodo["bootstrap_partition"].index, periodo["bootstrap_partition"][COLUMNA_OBJETIVO])
        ]})
        assert r.status_code == 200, r.text
        _snap("after_bootstrap")

        idx = 0
        for dias in sorted(days_checkpoints):
            objetivo = dias * 1440
            for ts, row in stream.iloc[idx:objetivo].iterrows():
                client.post("/readings", json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})
            idx = objetivo
            status = client.get(f"/status/{meter_id}").json()
            filas.append({"stage": f"after_{dias}_days", "rss_mb": proc.memory_info().rss / 1e6,
                          "fecha_utc": datetime.now(timezone.utc).isoformat(),
                          "buffer_1min_size": status["buffer_1min_size"], "history_15min_size": status["history_15min_size"]})
        client.post(f"/reset/{meter_id}")

    df = pd.DataFrame(filas)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "api_memory_usage.csv", index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(EDGE_DIR / "config" / "api.yaml"))
    parser.add_argument("--n-readings", type=int, default=DEFAULT_N_READINGS)
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()
    r = run_benchmark(n_readings=args.n_readings)
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    if not args.skip_memory:
        run_memory_profile()
    return 0


if __name__ == "__main__":
    sys.exit(main())
