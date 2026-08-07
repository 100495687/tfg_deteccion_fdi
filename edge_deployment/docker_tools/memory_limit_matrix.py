"""Fase 3: matriz de memoria a 1 CPU. Salvaguardas del usuario aplicadas: cada
limite arranca un contenedor nuevo (run_id unico, sin reutilizar contenedores calentados);
nunca se recomienda el limite que "apenas arranca" (eso lo decide docker_final_report.md, no
este script, que solo mide y clasifica startup/functional/stable de forma automatica).
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.docker_tools import synthetic_workload as sw

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

MEMORY_LIMITS_MB = [2048, 1536, 1280, 1024, 896, 768, 640, 512]
BASE_PORT = 18400
READY_TIMEOUT_S = 90


def _pct(values: list[float], p: float) -> float | None:
    return float(np.percentile(values, p)) if values else None


def run_one(mem_mb: int, port: int, bootstrap_df, stream_df, reference: list[dict]) -> dict:
    name = cc.new_run_id(f"memlimit_{mem_mb}")
    row = {"run_id": name, "memory_limit_mb": mem_mb, "cpu_limit": 1.0,
           "startup_success": False, "ready_success": False, "bootstrap_success": False, "streaming_success": False,
           "peak_memory_mb_docker_stats": None, "oom_killed": False, "exit_code": None,
           "latency_p50_ms": None, "latency_p95_ms": None, "latency_p99_ms": None, "throughput_rps": None,
           "equivalence_passed": None, "failure_stage": None, "failure_reason": None,
           "n_stream_readings": len(stream_df), "fecha_utc": datetime.now(timezone.utc).isoformat()}
    sampler = cc.StatsSampler(name, interval_s=1.0)
    try:
        cc.start_container("fdia-edge:optimized", name, port, mem_limit=f"{mem_mb}m", cpus=1.0)
        row["startup_success"] = True
        sampler.start()
        ok_ready, t_ready = cc.wait_ready(port, timeout=READY_TIMEOUT_S)
        state = cc.inspect_state(name)
        if not ok_ready:
            row["failure_stage"] = "ready"
            row["oom_killed"] = bool(state.get("OOMKilled"))
            row["exit_code"] = state.get("ExitCode")
            row["failure_reason"] = "oom_killed_during_startup" if row["oom_killed"] else "ready_timeout"
            return row
        row["ready_success"] = True
        row["time_to_ready_s"] = round(t_ready, 2)

        t0 = time.perf_counter()
        try:
            actual, meta = sw.replay_via_http(f"http://127.0.0.1:{port}", bootstrap_df, stream_df, meter_id=f"mm_{mem_mb}", timeout=30.0)
        except Exception as e:
            state = cc.inspect_state(name)
            row["failure_stage"] = "bootstrap_or_streaming"
            row["oom_killed"] = bool(state.get("OOMKilled"))
            row["exit_code"] = state.get("ExitCode")
            row["failure_reason"] = f"exception: {e}"
            return row
        elapsed = time.perf_counter() - t0
        row["bootstrap_success"] = meta["bootstrap_ms"] is not None
        row["streaming_success"] = meta["n_ok"] == meta["n_sent"] and meta["n_errors_5xx"] == 0
        row["bootstrap_ms"] = round(meta["bootstrap_ms"], 1)
        row["n_ok"] = meta["n_ok"]
        row["n_errors_5xx"] = meta["n_errors_5xx"]
        row["n_lost"] = meta["n_lost"]
        if meta["latencias_ms"]:
            row["latency_p50_ms"] = round(_pct(meta["latencias_ms"], 50), 2)
            row["latency_p95_ms"] = round(_pct(meta["latencias_ms"], 95), 2)
            row["latency_p99_ms"] = round(_pct(meta["latencias_ms"], 99), 2)
            row["throughput_rps"] = round(meta["n_ok"] / elapsed, 2) if elapsed > 0 else None

        comp = sw.compare_to_reference(reference, actual)
        row["equivalence_passed"] = comp["all_100pct"]
        row["equivalence_match_pct_min"] = min(comp["match_pct"].values()) if comp["match_pct"] else None
        row["score_p_max_abs_diff"] = comp["score_p_max_abs_diff"]
        row["score_h_max_abs_diff"] = comp["score_h_max_abs_diff"]

        state = cc.inspect_state(name)
        row["oom_killed"] = bool(state.get("OOMKilled"))
        row["exit_code"] = state.get("ExitCode")
        if not row["streaming_success"] or not row["equivalence_passed"]:
            row["failure_stage"] = row["failure_stage"] or "streaming_quality"
            row["failure_reason"] = row["failure_reason"] or (
                "errores_5xx_o_perdidas" if not row["streaming_success"] else "diferencia_de_alertas_o_scores")
    finally:
        samples = sampler.stop()
        if samples:
            mem_vals = [s["mem_mb"] for s in samples if s["mem_mb"] is not None]
            row["peak_memory_mb_docker_stats"] = round(max(mem_vals), 1) if mem_vals else None
            row["mean_memory_mb_docker_stats"] = round(sum(mem_vals) / len(mem_vals), 1) if mem_vals else None
        cc.stop_and_remove(name)
    return row


def run_matrix(limits: list[int] = MEMORY_LIMITS_MB, out_name: str = "docker_memory_limit_matrix.csv",
               tiers_name: str = "docker_memory_limit_tiers.json") -> list[dict]:
    print("[mem-matrix] generando carga sintetica (bootstrap 10230min + 500min streaming) y referencia (motor directo)...")
    bootstrap_df, stream_df = sw.build_bootstrap_and_stream()
    reference = sw.compute_reference(bootstrap_df, stream_df)
    print(f"[mem-matrix] referencia calculada: {len(reference)} lecturas")

    rows = []
    port = BASE_PORT
    for mem_mb in limits:
        print(f"[mem-matrix] limite={mem_mb}MB @1CPU (puerto {port})...")
        row = run_one(mem_mb, port, bootstrap_df, stream_df, reference)
        print(f"  startup={row['startup_success']} ready={row['ready_success']} "
              f"bootstrap={row['bootstrap_success']} streaming={row['streaming_success']} "
              f"peak_mem={row['peak_memory_mb_docker_stats']}MB oom={row['oom_killed']} "
              f"equiv={row['equivalence_passed']} stage_fallo={row['failure_stage']}")
        rows.append(row)
        port += 1

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / out_name
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[mem-matrix] -> {out}")

    tiers = _classify_tiers(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / tiers_name, "w", encoding="utf-8") as f:
        json.dump(tiers, f, indent=2, ensure_ascii=False, default=str)
    print(f"[mem-matrix] tiers: {tiers}")
    return rows


def _classify_tiers(rows: list[dict]) -> dict:
    """minimo_arranque: el limite mas bajo que arranca y llega a ready. minimo_funcional: el
    limite mas bajo que completa bootstrap+streaming con equivalencia OK (aunque sea con
    latencia alta o sin margen). minimo_estable: el limite mas bajo funcional cuyo pico de RAM
    deja margen razonable (<90% del limite, heuristica documentada -- no se recomienda el que
    apenas arranca ni el que opera sin margen, salvaguarda del usuario)."""
    ordenados = sorted(rows, key=lambda r: r["memory_limit_mb"])
    min_arranque = next((r["memory_limit_mb"] for r in ordenados if r["ready_success"]), None)
    funcionales = [r for r in ordenados if r["ready_success"] and r["bootstrap_success"] and r["streaming_success"] and r["equivalence_passed"]]
    min_funcional = funcionales[0]["memory_limit_mb"] if funcionales else None
    estables = [r for r in funcionales if r["peak_memory_mb_docker_stats"] is not None
                and r["peak_memory_mb_docker_stats"] < 0.90 * r["memory_limit_mb"]]
    min_estable = estables[0]["memory_limit_mb"] if estables else None
    return {"minimo_arranque_mb": min_arranque, "minimo_funcional_mb": min_funcional, "minimo_estable_mb": min_estable,
            "n_funcionales": len(funcionales), "n_estables": len(estables)}


if __name__ == "__main__":
    run_matrix()
