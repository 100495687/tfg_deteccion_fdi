"""Benchmark Uvicorn real: arranca un servidor Uvicorn de verdad (subproceso,
`--workers 1`, ver README_API.md) y envia lecturas via httpx sobre un socket TCP real --
mide round-trip completo, overhead HTTP aproximado (round_trip - tiempo interno del motor) y
throughput. Alcance mas modesto que el benchmark TestClient (que pide >=10000
lecturas/7 dias para TestClient; aqui, por el coste de un servidor real, se usa un periodo
mas corto por defecto -- documentado en api_final_report.md).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]  # deteccion_fraude_no_supervisada/ -- cwd correcto para `python -m edge_deployment...`
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

HOST = "127.0.0.1"
PORT = 8123
DEFAULT_N_READINGS = 2000
WARMUP_READINGS = 100


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    arr = np.array(values, dtype=float)
    return {"n": int(len(arr)), "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)), "max_ms": float(arr.max())}


def _start_server() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "edge_deployment.api.main:app", "--host", HOST, "--port", str(PORT),
         "--workers", "1", "--log-level", "warning"],
        cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _wait_ready(base_url: str, timeout: float = 90.0) -> bool:
    import httpx
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{base_url}/ready", timeout=2.0)
            if r.status_code == 200 and r.json().get("ready"):
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def run_benchmark(n_readings: int = DEFAULT_N_READINGS, warmup: int = WARMUP_READINGS) -> dict:
    import httpx

    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    max_days = max(3, (n_readings + warmup) // 1440 + 2)
    periodo = seleccionar_periodo(max_stream_days=max_days)
    stream = periodo["streaming_partition"]
    assert len(stream) >= n_readings + warmup

    base_url = f"http://{HOST}:{PORT}"
    proc = _start_server()
    errores = 0
    perdidas = 0
    try:
        if not _wait_ready(base_url):
            salida = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"el servidor Uvicorn no respondio ready=true a tiempo.\n{salida[-2000:]}")

        client = httpx.Client(base_url=base_url, timeout=30.0)
        meter_id = "bench_uvicorn"
        r = client.post("/bootstrap", json={"meter_id": meter_id, "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(periodo["bootstrap_partition"].index, periodo["bootstrap_partition"][COLUMNA_OBJETIVO])
        ]})
        assert r.status_code == 200, r.text

        for ts, row in stream.iloc[:warmup].iterrows():
            client.post("/readings", json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})

        filas = []
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
            body = r.json()
            filas.append({
                "timestamp": ts, "http_status": r.status_code, "round_trip_ms": round_trip_ms,
                "api_processing_time_ms": body.get("api_processing_time_ms"),
                "engine_processing_time_ms": body.get("engine_processing_time_ms"),
            })
        elapsed = time.perf_counter() - t_start
        client.post(f"/reset/{meter_id}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    df = pd.DataFrame(filas)
    df["http_overhead_ms"] = df["round_trip_ms"] - df["engine_processing_time_ms"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "api_uvicorn_latency.csv", index=False)

    throughput = pd.DataFrame([{
        "n_readings": len(df), "elapsed_seconds": elapsed,
        "throughput_readings_per_second": len(df) / elapsed if elapsed > 0 else None,
        "n_http_errors_5xx": errores, "n_lost_requests": perdidas,
    }])
    throughput.to_csv(TABLES_DIR / "api_throughput.csv", index=False)

    resumen = {
        "n_readings": len(df), "warmup_readings": warmup, "errors_5xx": errores, "lost_requests": perdidas,
        "round_trip": _percentiles(df["round_trip_ms"].tolist()),
        "engine_internal": _percentiles(df["engine_processing_time_ms"].tolist()),
        "http_overhead": _percentiles(df["http_overhead_ms"].tolist()),
        "throughput_readings_per_second": len(df) / elapsed if elapsed > 0 else None,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "api_uvicorn_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    return resumen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(EDGE_DIR / "config" / "api.yaml"))
    parser.add_argument("--n-readings", type=int, default=DEFAULT_N_READINGS)
    args = parser.parse_args()
    r = run_benchmark(n_readings=args.n_readings)
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
