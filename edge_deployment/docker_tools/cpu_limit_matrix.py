"""Fase 3: matriz de CPU, con la memoria "estable" recomendada (se recalcula tras
la matriz de memoria -- ver docker_final_report.md; por defecto usa el minimo_estable_mb de
docker_memory_limit_tiers.json si existe, si no cae a 1024MB como valor conservador). Mismo
patron que memory_limit_matrix.py: un contenedor nuevo por configuracion, referencia fija
(motor directo) para detectar cualquier cambio de alerta inducido por la restriccion de CPU.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.docker_tools import synthetic_workload as sw

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

CPU_LIMITS = [2.0, 1.0, 0.5, 0.25]
BASE_PORT = 18450
READY_TIMEOUT_S = 150  # limites de CPU muy bajos pueden alargar bastante la carga inicial (P+H+datos)


def _stable_memory_mb() -> int:
    p = REPORTS_DIR / "docker_memory_limit_tiers.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("minimo_estable_mb"):
            return int(d["minimo_estable_mb"])
    return 1024


def _run_one_mem_config_with_cpu(cpu: float, mem_mb: int, port: int, bootstrap_df, stream_df, reference: list[dict]) -> dict:
    import time

    name = cc.new_run_id(f"cpulimit_{cpu}")
    row = {"run_id": name, "cpu_limit": cpu, "memory_limit_mb": mem_mb,
           "startup_success": False, "ready_success": False, "bootstrap_success": False, "streaming_success": False,
           "peak_cpu_pct": None, "mean_cpu_pct": None, "throttled": None,
           "latency_p50_ms": None, "latency_p95_ms": None, "latency_p99_ms": None, "throughput_rps": None,
           "equivalence_passed": None, "failure_stage": None, "failure_reason": None,
           "fecha_utc": datetime.now(timezone.utc).isoformat()}
    sampler = cc.StatsSampler(name, interval_s=1.0)
    try:
        cc.start_container("fdia-edge:optimized", name, port, mem_limit=f"{mem_mb}m", cpus=cpu)
        row["startup_success"] = True
        sampler.start()
        ok_ready, t_ready = cc.wait_ready(port, timeout=READY_TIMEOUT_S)
        if not ok_ready:
            state = cc.inspect_state(name)
            row["failure_stage"] = "ready"
            row["failure_reason"] = "oom_killed_during_startup" if state.get("OOMKilled") else "ready_timeout"
            return row
        row["ready_success"] = True
        row["time_to_ready_s"] = round(t_ready, 2)

        t0 = time.perf_counter()
        try:
            actual, meta = sw.replay_via_http(f"http://127.0.0.1:{port}", bootstrap_df, stream_df, meter_id=f"cc_{cpu}", timeout=45.0)
        except Exception as e:
            row["failure_stage"] = "bootstrap_or_streaming"
            row["failure_reason"] = f"exception: {e}"
            return row
        elapsed = time.perf_counter() - t0
        row["bootstrap_success"] = meta["bootstrap_ms"] is not None
        row["streaming_success"] = meta["n_ok"] == meta["n_sent"] and meta["n_errors_5xx"] == 0
        row["bootstrap_ms"] = round(meta["bootstrap_ms"], 1)
        row["n_ok"], row["n_errors_5xx"], row["n_lost"] = meta["n_ok"], meta["n_errors_5xx"], meta["n_lost"]
        if meta["latencias_ms"]:
            row["latency_p50_ms"] = round(float(np.percentile(meta["latencias_ms"], 50)), 2)
            row["latency_p95_ms"] = round(float(np.percentile(meta["latencias_ms"], 95)), 2)
            row["latency_p99_ms"] = round(float(np.percentile(meta["latencias_ms"], 99)), 2)
            row["throughput_rps"] = round(meta["n_ok"] / elapsed, 2) if elapsed > 0 else None

        comp = sw.compare_to_reference(reference, actual)
        row["equivalence_passed"] = comp["all_100pct"]
        row["score_p_max_abs_diff"] = comp["score_p_max_abs_diff"]
        row["score_h_max_abs_diff"] = comp["score_h_max_abs_diff"]
        if not row["streaming_success"] or not row["equivalence_passed"]:
            row["failure_stage"] = row["failure_stage"] or "streaming_quality"
    finally:
        samples = sampler.stop()
        if samples:
            cpu_vals = [s["cpu_pct"] for s in samples if s["cpu_pct"] is not None]
            row["peak_cpu_pct"] = round(max(cpu_vals), 1) if cpu_vals else None
            row["mean_cpu_pct"] = round(sum(cpu_vals) / len(cpu_vals), 1) if cpu_vals else None
            # heuristica de throttling: CPU% sostenido cerca del limite asignado (100*cpu) sugiere
            # que el contenedor esta topando con la cuota de cgroup durante buena parte de la carga
            limit_pct = cpu * 100
            row["throttled"] = bool(cpu_vals) and (sum(1 for v in cpu_vals if v >= 0.95 * limit_pct) / len(cpu_vals)) > 0.3
        cc.stop_and_remove(name)
    return row


def run_matrix(cpus: list[float] = CPU_LIMITS) -> list[dict]:
    mem_mb = _stable_memory_mb()
    print(f"[cpu-matrix] usando memoria estable fija = {mem_mb}MB para toda la matriz de CPU")
    print("[cpu-matrix] generando carga sintetica y referencia (motor directo)...")
    bootstrap_df, stream_df = sw.build_bootstrap_and_stream()
    reference = sw.compute_reference(bootstrap_df, stream_df)

    rows = []
    port = BASE_PORT
    for cpu in cpus:
        print(f"[cpu-matrix] cpu={cpu} mem={mem_mb}MB (puerto {port})...")
        row = _run_one_mem_config_with_cpu(cpu, mem_mb, port, bootstrap_df, stream_df, reference)
        print(f"  ready={row['ready_success']} streaming={row['streaming_success']} "
              f"p50={row['latency_p50_ms']}ms p95={row['latency_p95_ms']}ms peak_cpu={row['peak_cpu_pct']}% "
              f"throttled={row['throttled']} equiv={row['equivalence_passed']}")
        rows.append(row)
        port += 1

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "docker_cpu_limit_matrix.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[cpu-matrix] -> {out}")
    return rows


if __name__ == "__main__":
    run_matrix()
