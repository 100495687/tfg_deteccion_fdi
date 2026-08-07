"""Fase 3: equivalencia local vs docker. Reutiliza literalmente
`clients.api_stream_client.run_equivalence` (Fase 2, sin modificar): DirectEngineClient
(motor sin HTTP, ya probado bit-identico al offline en Fase 1) sirve de referencia unica y se
compara, lectura a lectura, contra cada entorno real por HTTP -- A: API local (Fase 2,
Uvicorn nativo), B: fdia-edge:baseline, C: fdia-edge:optimized. Por transitividad (A/B/C
identicos a la referencia directa) se concluye que A, B y C son equivalentes entre si, sin
tener que comparar 3x3.

Primero un smoke test de 3 dias en los tres entornos (salvaguarda: no seguir con benchmarks de
recursos si aparece una diferencia de alertas). Si el smoke test pasa, se ejecuta 30 dias solo
en optimized (el entorno que de verdad se va a recomendar), tal como pide el enunciado.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from edge_deployment.docker_tools import container_control as cc

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

LOCAL_HOST, LOCAL_PORT = "127.0.0.1", 18300
DOCKER_BASELINE_PORT = 18301
DOCKER_OPTIMIZED_PORT = 18302
DOCKER_OPTIMIZED_PORT_30D = 18303


def _start_local_uvicorn() -> subprocess.Popen:
    """Mismo patron que benchmarks/benchmark_api_uvicorn.py (Fase 2, sin modificar) -- servidor
    Uvicorn real, --workers 1, arrancado como subproceso nativo (sin Docker)."""
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "edge_deployment.api.main:app", "--host", LOCAL_HOST,
         "--port", str(LOCAL_PORT), "--workers", "1", "--log-level", "warning"],
        cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _run_one_environment(env_name: str, base_url: str, days: int, out_suffix: str) -> dict:
    from edge_deployment.clients.api_stream_client import run_equivalence
    print(f"[docker-equiv] {env_name}: comparando {days} dias contra {base_url} ...")
    result = run_equivalence(max_stream_days=days, base_url=base_url, overwrite=True, out_suffix=out_suffix)
    df = result["df"].copy()
    df.insert(0, "environment", env_name)
    return {"df": df, "resumen": result["resumen"]}


def run_smoke(days: int = 3) -> dict:
    resultados = {}

    print("[docker-equiv] A: arrancando API local (Fase 2, sin Docker)...")
    proc = _start_local_uvicorn()
    try:
        ok, dt = cc.wait_ready(LOCAL_PORT, timeout=90)
        if not ok:
            salida = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"API local no respondio ready=true a tiempo.\n{salida[-2000:]}")
        resultados["A_local_api"] = _run_one_environment("A_local_api", f"http://{LOCAL_HOST}:{LOCAL_PORT}", days, "_docker_A_local")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    for env_name, image, port, suffix in [
        ("B_docker_baseline", "fdia-edge:baseline", DOCKER_BASELINE_PORT, "_docker_B_baseline"),
        ("C_docker_optimized", "fdia-edge:optimized", DOCKER_OPTIMIZED_PORT, "_docker_C_optimized"),
    ]:
        name = cc.new_run_id(f"equiv_{env_name}")
        print(f"[docker-equiv] {env_name}: arrancando contenedor {image}...")
        cc.start_container(image, name, port)
        try:
            ok, dt = cc.wait_ready(port, timeout=90)
            if not ok:
                raise RuntimeError(f"{env_name} no alcanzo ready=true. Logs:\n{cc.container_logs(name, 60)}")
            resultados[env_name] = _run_one_environment(env_name, f"http://127.0.0.1:{port}", days, suffix)
        finally:
            cc.stop_and_remove(name)

    combined = pd.concat([r["df"] for r in resultados.values()], ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES_DIR / "container_equivalence.csv"
    combined.to_csv(out_csv, index=False)

    match_fields = ["accepted", "engine_status", "p_ready", "h_ready", "p_evaluated", "h_evaluated",
                     "alert_p", "alert_p_ffill", "alert_h", "alert_or", "detection_source"]
    per_env_summary = {}
    all_100 = True
    for env_name, r in resultados.items():
        mp = r["resumen"]["match_pct"]
        env_all_100 = all(mp[f] == 100.0 for f in match_fields)
        all_100 = all_100 and env_all_100
        per_env_summary[env_name] = {
            "n_readings": r["resumen"]["n_readings"], "match_pct": mp,
            "all_alert_fields_100pct": env_all_100,
            "score_p_max_abs_diff": r["resumen"]["score_p_max_abs_diff"],
            "score_h_max_abs_diff": r["resumen"]["score_h_max_abs_diff"],
        }

    summary = {
        "phase": "smoke_3day" if days <= 7 else f"{days}day",
        "days": days, "per_environment": per_env_summary,
        "all_environments_100pct_alerts": all_100,
        "gate_passed_continue_to_resource_benchmarks": all_100,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / f"container_equivalence_summary_{'smoke' if days <= 7 else str(days) + 'd'}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"[docker-equiv] -> {out_csv}")
    print(json.dumps({k: v["all_alert_fields_100pct"] for k, v in per_env_summary.items()}, indent=2))
    print(f"[docker-equiv] TODOS los entornos 100% en alertas: {all_100}")
    return summary


def run_30day_optimized_only(days: int = 30) -> dict:
    name = cc.new_run_id("equiv_30d_optimized")
    print(f"[docker-equiv] 30 dias, solo optimized: arrancando contenedor...")
    cc.start_container("fdia-edge:optimized", name, DOCKER_OPTIMIZED_PORT_30D)
    try:
        ok, dt = cc.wait_ready(DOCKER_OPTIMIZED_PORT_30D, timeout=90)
        if not ok:
            raise RuntimeError(f"optimized (30d) no alcanzo ready=true. Logs:\n{cc.container_logs(name, 60)}")
        t0 = time.time()
        r = _run_one_environment("C_docker_optimized_30d", f"http://127.0.0.1:{DOCKER_OPTIMIZED_PORT_30D}", days, "_docker_C_optimized_30d")
        elapsed = time.time() - t0
    finally:
        cc.stop_and_remove(name)

    mp = r["resumen"]["match_pct"]
    match_fields = ["accepted", "engine_status", "p_ready", "h_ready", "p_evaluated", "h_evaluated",
                     "alert_p", "alert_p_ffill", "alert_h", "alert_or", "detection_source"]
    all_100 = all(mp[f] == 100.0 for f in match_fields)
    summary = {
        "phase": "30day_optimized_only", "days": days, "n_readings": r["resumen"]["n_readings"],
        "elapsed_seconds": elapsed, "match_pct": mp, "all_alert_fields_100pct": all_100,
        "score_p_max_abs_diff": r["resumen"]["score_p_max_abs_diff"],
        "score_h_max_abs_diff": r["resumen"]["score_h_max_abs_diff"],
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "container_equivalence_summary_30d.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    r["df"].to_csv(TABLES_DIR / "container_equivalence_30d_optimized.csv", index=False)
    print(f"[docker-equiv] 30 dias optimized: n={summary['n_readings']} all_100pct={all_100} "
          f"elapsed={elapsed:.1f}s")
    return summary


if __name__ == "__main__":
    smoke = run_smoke(days=3)
    if smoke["gate_passed_continue_to_resource_benchmarks"]:
        print("[docker-equiv] smoke test OK (100% alertas en los 3 entornos) -> lanzando 30 dias en optimized...")
        run_30day_optimized_only(days=30)
    else:
        print("[docker-equiv] SMOKE TEST FALLO: hay diferencias de alertas. NO se continua "
              "con 30 dias ni con benchmarks de recursos (salvaguarda del usuario).")
