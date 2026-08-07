"""Fase 3: benchmark de latencia/throughput dentro de un contenedor optimized real
(arrancado por este script, contenedor nuevo, sin tracemalloc activo -- solo se mide RSS via
psutil/docker stats en pasadas separadas, patron ya establecido en Fase 2). Separa por
categoria usando los campos que la API ya devuelve (p_evaluated/h_evaluated, sin reimplementar
nada): bootstrap, lectura_sin_inferencia (ni P ni H evaluan), lectura_solo_H,
lectura_P_y_H (H siempre evalua junto con P en la arquitectura P_OR_H -- ver Fase 1 -- por eso
no existe la categoria "solo P").
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from edge_deployment.docker_tools import container_control as cc

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

PORT = 18310
N_READINGS = 3000
WARMUP = 100


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    arr = np.array(values, dtype=float)
    return {"n": int(len(arr)), "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)), "max_ms": float(arr.max())}


def _categorize(row: dict) -> str:
    if row.get("h_evaluated"):
        return "lectura_P_y_H" if row.get("p_evaluated") else "lectura_solo_H"
    return "lectura_sin_inferencia"


def run_benchmark(n_readings: int = N_READINGS, warmup: int = WARMUP) -> dict:
    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    max_days = max(3, (n_readings + warmup) // 1440 + 2)
    periodo = seleccionar_periodo(max_stream_days=max_days)
    stream = periodo["streaming_partition"]
    assert len(stream) >= n_readings + warmup

    name = cc.new_run_id("latency_bench")
    print(f"[latency] arrancando contenedor optimized ({name})...")
    cc.start_container("fdia-edge:optimized", name, PORT)
    filas = []
    bootstrap_ms = None
    errores = perdidas = 0
    try:
        ok, dt = cc.wait_ready(PORT, timeout=90)
        if not ok:
            raise RuntimeError(f"contenedor no listo. Logs:\n{cc.container_logs(name, 60)}")

        client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30.0)
        meter_id = "bench_docker_latency"
        t0 = time.perf_counter()
        r = client.post("/bootstrap", json={"meter_id": meter_id, "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)}
            for ts, v in zip(periodo["bootstrap_partition"].index, periodo["bootstrap_partition"][COLUMNA_OBJETIVO])
        ]})
        bootstrap_ms = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, r.text

        for ts, row in stream.iloc[:warmup].iterrows():
            client.post("/readings", json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})

        t_start = time.perf_counter()
        for ts, row in stream.iloc[warmup:warmup + n_readings].iterrows():
            t0 = time.perf_counter()
            try:
                r = client.post("/readings", json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})
            except Exception:
                perdidas += 1
                continue
            round_trip_ms = (time.perf_counter() - t0) * 1000
            if r.status_code >= 500:
                errores += 1
                continue
            body = r.json()
            filas.append({
                "timestamp": ts, "http_status": r.status_code, "round_trip_ms": round_trip_ms,
                "api_processing_time_ms": body.get("api_processing_time_ms"),
                "engine_processing_time_ms": body.get("engine_processing_time_ms"),
                "p_evaluated": body.get("p_evaluated"), "h_evaluated": body.get("h_evaluated"),
                "category": _categorize(body),
            })
        elapsed = time.perf_counter() - t_start
        client.post(f"/reset/{meter_id}")
    finally:
        cc.stop_and_remove(name)

    df = pd.DataFrame(filas)
    df["http_overhead_ms"] = df["round_trip_ms"] - df["engine_processing_time_ms"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "docker_latency.csv", index=False)

    by_category = {}
    for cat, g in df.groupby("category"):
        by_category[cat] = {
            "round_trip": _percentiles(g["round_trip_ms"].tolist()),
            "engine_internal": _percentiles(g["engine_processing_time_ms"].tolist()),
            "http_overhead": _percentiles(g["http_overhead_ms"].tolist()),
        }

    resumen = {
        "n_readings": len(df), "warmup_readings": warmup, "errors_5xx": errores, "lost_requests": perdidas,
        "bootstrap_latency_ms": bootstrap_ms,
        "overall_round_trip": _percentiles(df["round_trip_ms"].tolist()),
        "overall_engine_internal": _percentiles(df["engine_processing_time_ms"].tolist()),
        "overall_http_overhead": _percentiles(df["http_overhead_ms"].tolist()),
        "by_category": by_category,
        "category_counts": df["category"].value_counts().to_dict(),
        "throughput_readings_per_second": len(df) / elapsed if elapsed > 0 else None,
        "elapsed_seconds": elapsed,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "docker_latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"[latency] n={len(df)} throughput={resumen['throughput_readings_per_second']:.2f}/s "
          f"errors5xx={errores} lost={perdidas} bootstrap={bootstrap_ms:.1f}ms")
    for cat, d in by_category.items():
        print(f"  {cat}: n={d['round_trip']['n']} round_trip_median={d['round_trip']['median_ms']:.2f}ms "
              f"p95={d['round_trip']['p95_ms']:.2f}ms")
    return resumen


if __name__ == "__main__":
    run_benchmark()
