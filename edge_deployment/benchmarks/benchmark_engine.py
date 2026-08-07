"""Benchmark preliminar de rendimiento. Docker no se implementa todavia en
esta fase -- estas cifras son preliminares (motor Python puro sobre el equipo de
desarrollo) y no sustituyen el benchmark edge-cloud posterior en contenedor.

Mide: tiempo de carga de modelos, tiempo de bootstrap, latencia por ingest (separada por
si evaluo H, P, ambos o ninguno), throughput acelerado y memoria aproximada del estado
(via `tracemalloc`, libreria estandar -- sin nuevas dependencias).

Comando:
    python -m edge_deployment.benchmarks.benchmark_engine --max-days 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None, "n_observations": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)), "max_ms": float(arr.max()), "n_observations": int(len(arr)),
    }


def run_benchmark(max_stream_days: int = 7, memory_sample_minutes: int = 1500) -> dict:
    """Dos pasadas separadas a proposito: la latencia se mide sin `tracemalloc` activo
    (que distorsiona brutalmente los tiempos, ver `results/logs/` -- confirmado
    empiricamente: ~20-25x mas lento con tracemalloc activo), y la memoria se mide en una
    segunda pasada corta, mas barata, exclusivamente con `tracemalloc` activo. Ambas pasadas
    reutilizan el mismo motor ya cargado (el coste de carga de modelos solo se paga una vez)."""
    from edge_deployment.core.detector_engine import DetectorEngine
    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    t0 = time.perf_counter()
    engine = DetectorEngine()
    t_load = (time.perf_counter() - t0) * 1000

    periodo = seleccionar_periodo(max_stream_days=max_stream_days)

    # --- Pasada 1: latencia (sin tracemalloc) ---
    t0 = time.perf_counter()
    engine.bootstrap("meter_bench", periodo["bootstrap_partition"])
    t_bootstrap = (time.perf_counter() - t0) * 1000

    lat_none, lat_h_only, lat_p_only, lat_both, lat_all = [], [], [], [], []
    resp = None
    for ts, row in periodo["streaming_partition"].iterrows():
        t0 = time.perf_counter()
        resp = engine.ingest("meter_bench", ts, float(row[COLUMNA_OBJETIVO]))
        dt_ms = (time.perf_counter() - t0) * 1000
        lat_all.append(dt_ms)
        if resp.p_evaluated and resp.h_evaluated:
            lat_both.append(dt_ms)
        elif resp.p_evaluated:
            lat_p_only.append(dt_ms)
        elif resp.h_evaluated:
            lat_h_only.append(dt_ms)
        else:
            lat_none.append(dt_ms)
    engine.reset("meter_bench")

    total_seconds = sum(lat_all) / 1000.0
    throughput = len(periodo["streaming_partition"]) / total_seconds if total_seconds > 0 else None

    # --- Pasada 2: memoria aproximada del estado (con tracemalloc, sub-muestra mas corta) ---
    engine.bootstrap("meter_bench", periodo["bootstrap_partition"])
    sub_stream = periodo["streaming_partition"].iloc[:memory_sample_minutes]
    tracemalloc.start()
    memoria_muestras = []
    resp_mem = None
    for i, (ts, row) in enumerate(sub_stream.iterrows()):
        resp_mem = engine.ingest("meter_bench", ts, float(row[COLUMNA_OBJETIVO]))
        if i % 100 == 0:
            current, peak = tracemalloc.get_traced_memory()
            memoria_muestras.append({
                "minute_index": i, "timestamp": ts, "current_bytes": current, "peak_bytes": peak,
                "buffer_1min_size": resp_mem.buffer_1min_size, "buffer_15min_size": resp_mem.buffer_15min_size,
            })
    current_final, peak_final = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engine.reset("meter_bench")
    if resp_mem is not None:
        resp = resp_mem  # buffer sizes finales para el resumen (misma cola: rejilla ya llena)

    execution_profile = pd.DataFrame([
        {"operation": "model_load", "n_observations": 1, "mean_ms": t_load, "median_ms": t_load, "p95_ms": t_load, "p99_ms": t_load, "max_ms": t_load},
        {"operation": "bootstrap", "n_observations": 1, "mean_ms": t_bootstrap, "median_ms": t_bootstrap, "p95_ms": t_bootstrap, "p99_ms": t_bootstrap, "max_ms": t_bootstrap},
        {"operation": "ingest_no_model_evaluated", **_percentiles(lat_none)},
        {"operation": "ingest_h_only_evaluated", **_percentiles(lat_h_only)},
        {"operation": "ingest_p_only_evaluated", **_percentiles(lat_p_only)},
        {"operation": "ingest_p_and_h_evaluated", **_percentiles(lat_both)},
        {"operation": "ingest_overall", **_percentiles(lat_all)},
    ])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    execution_profile.to_csv(TABLES_DIR / "execution_profile.csv", index=False)
    pd.DataFrame({"timestamp": periodo["streaming_partition"].index, "processing_time_ms": lat_all}).to_csv(
        TABLES_DIR / "preliminary_latency.csv", index=False)
    pd.DataFrame(memoria_muestras).to_csv(TABLES_DIR / "state_memory_usage.csv", index=False)

    resumen = {
        "model_load_ms": t_load, "bootstrap_ms": t_bootstrap,
        "throughput_readings_per_second_accelerated": throughput,
        "memory_current_bytes_final": int(current_final), "memory_peak_bytes": int(peak_final),
        "buffer_1min_final_size": resp.buffer_1min_size, "buffer_15min_final_size": resp.buffer_15min_size,
        "n_stream_minutes": int(len(periodo["streaming_partition"])), "max_stream_days": max_stream_days,
        "note": "cifras preliminares (motor Python puro, sin contenedor); no sustituyen el benchmark edge-cloud en Docker",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "performance_summary.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)

    engine.reset("meter_bench")
    return {"execution_profile": execution_profile, "resumen": resumen}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(EDGE_DIR / "config" / "edge_engine.yaml"))
    parser.add_argument("--max-days", type=int, default=7)
    args = parser.parse_args()

    r = run_benchmark(max_stream_days=args.max_days)
    print(r["execution_profile"].to_string(index=False))
    print(json.dumps(r["resumen"], indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
