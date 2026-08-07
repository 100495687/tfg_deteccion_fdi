"""Fase 4: equivalencia legacy vs autonomo.

  A. Motor legacy   -- DetectorEngine(use_legacy_loader=True), reconstruye params_norm desde
                        data_raw.csv/episodes_master_val.csv (host, tiene acceso a los CSV).
  B. Motor autonomo -- DetectorEngine() (por defecto desde Fase 4), carga params_norm_p
                        congelado, nunca abre los CSV.
  C. API local autonoma -- Uvicorn nativo (sin Docker), autonomous_loader por defecto.
  D. Docker autonomo    -- fdia-edge:optimized, sin montar ningun CSV.

Mismo bootstrap/periodo pre-test/timestamps/lecturas/modelos/umbrales en las 4 rutas
(`core.period_selection.seleccionar_periodo`, igual que Fase 1-3). A y B se comparan
directamente (motor sin HTTP) -> `autonomous_loader_equivalence.csv`. C y D se comparan cada
uno contra el motor directo autonomo (B, el mismo patron ya usado en Fase 2/3 -- reutiliza
literalmente `clients.api_stream_client.run_equivalence`, que desde Fase 4 usa el
autonomous_loader por defecto porque `DirectEngineClient()`/`DetectorEngine()` ya no reciben
`use_legacy_loader=True`) -> `autonomous_container_equivalence.csv`.
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

LOCAL_HOST, LOCAL_PORT = "127.0.0.1", 18800
DOCKER_PORT = 18801
DOCKER_PORT_30D = 18802


def _row(ts, a: dict, b: dict) -> dict:
    return {
        "timestamp": ts,
        "legacy_accepted": a["accepted"], "autonomous_accepted": b["accepted"],
        "legacy_p_ready": a["p_ready"], "autonomous_p_ready": b["p_ready"],
        "legacy_h_ready": a["h_ready"], "autonomous_h_ready": b["h_ready"],
        "legacy_p_evaluated": a["p_evaluated"], "autonomous_p_evaluated": b["p_evaluated"],
        "legacy_h_evaluated": a["h_evaluated"], "autonomous_h_evaluated": b["h_evaluated"],
        "legacy_score_p": a["score_p"], "autonomous_score_p": b["score_p"],
        "legacy_score_h": a["score_h"], "autonomous_score_h": b["score_h"],
        "legacy_alert_p": a["alert_p"], "autonomous_alert_p": b["alert_p"],
        "legacy_alert_p_ffill": a["alert_p_ffill"], "autonomous_alert_p_ffill": b["alert_p_ffill"],
        "legacy_alert_h": a["alert_h"], "autonomous_alert_h": b["alert_h"],
        "legacy_alert_or": a["alert_or"], "autonomous_alert_or": b["alert_or"],
        "legacy_detection_source": a["detection_source"], "autonomous_detection_source": b["detection_source"],
        "legacy_buffer_1min_size": a["buffer_1min_size"], "autonomous_buffer_1min_size": b["buffer_1min_size"],
        "legacy_rejection_reason": a.get("rejection_reason"), "autonomous_rejection_reason": b.get("rejection_reason"),
    }


def run_loader_equivalence(days: int = 3, overwrite: bool = True) -> dict:
    """A vs B: motor legacy vs motor autonomo, sin HTTP, mismo periodo pre-test real."""
    from edge_deployment.clients.direct_engine_client import DirectEngineClient
    from edge_deployment.core.detector_engine import DetectorEngine
    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    print("[loader-equiv] cargando motor legacy (reconstruye params_norm desde CSV)...")
    t0 = time.perf_counter()
    engine_legacy = DetectorEngine(use_legacy_loader=True)
    t_legacy = time.perf_counter() - t0

    print("[loader-equiv] cargando motor autonomo (params_norm_p congelado)...")
    t0 = time.perf_counter()
    engine_auto = DetectorEngine()
    t_auto = time.perf_counter() - t0

    params_norm_legacy = engine_legacy.params_norm_p
    params_norm_auto = engine_auto.params_norm_p
    params_norm_exact_match = params_norm_legacy == params_norm_auto
    print(f"[loader-equiv] params_norm legacy={params_norm_legacy} autonomo={params_norm_auto} "
          f"exact_match={params_norm_exact_match} (carga: legacy={t_legacy:.2f}s autonomo={t_auto:.3f}s)")

    periodo = seleccionar_periodo(max_stream_days=days)
    a = DirectEngineClient(engine=engine_legacy)
    b = DirectEngineClient(engine=engine_auto)

    meter_a, meter_b = "loader_equiv_legacy", "loader_equiv_autonomous"
    rep_a = a.bootstrap(meter_a, periodo["bootstrap_partition"])
    rep_b = b.bootstrap(meter_b, periodo["bootstrap_partition"])
    assert rep_a["h_ready"] == rep_b["h_ready"], "bootstrap legacy y autonomo difieren en h_ready"

    filas = []
    for ts, row in periodo["streaming_partition"].iterrows():
        v = float(row[COLUMNA_OBJETIVO])
        da = a.ingest(meter_a, ts, v)
        db = b.ingest(meter_b, ts, v)
        filas.append(_row(ts, da, db))

    df = pd.DataFrame(filas)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES_DIR / "autonomous_loader_equivalence.csv"
    if out_csv.exists() and not overwrite:
        raise SystemExit(f"{out_csv} ya existe. Usa overwrite=True para reemplazarlo.")
    df.to_csv(out_csv, index=False)

    from edge_deployment.clients.api_stream_client import _series_equal_nan_aware

    campos = ["accepted", "p_ready", "h_ready", "p_evaluated", "h_evaluated", "alert_p",
              "alert_p_ffill", "alert_h", "alert_or", "detection_source", "buffer_1min_size"]
    # alert_p/alert_h/alert_or/detection_source son None/NaN cuando P o H todavia no han
    # evaluado (la mayoria de las filas, dado que P evalua cada 60min y H cada 15min) -- NaN
    # == NaN es False en pandas/IEEE754, una comparacion ingenua cuenta "ambos ausentes" como
    # mismatch (bug real, encontrado y corregido en Fase 2 para api_stream_client.py; se
    # reutiliza aqui la misma correccion en vez de duplicarla con otro fallo).
    match_pct = {c: float(_series_equal_nan_aware(df[f"legacy_{c}"], df[f"autonomous_{c}"]).mean() * 100) for c in campos}
    diff_p = (df["autonomous_score_p"].astype(float) - df["legacy_score_p"].astype(float)).abs()
    diff_h = (df["autonomous_score_h"].astype(float) - df["legacy_score_h"].astype(float)).abs()

    a.reset(meter_a)
    b.reset(meter_b)

    resumen = {
        "days": days, "n_readings": len(df),
        "params_norm_legacy": params_norm_legacy, "params_norm_autonomous": params_norm_auto,
        "params_norm_exact_match": params_norm_exact_match,
        "load_time_legacy_s": t_legacy, "load_time_autonomous_s": t_auto,
        "match_pct": match_pct, "all_100pct": all(v == 100.0 for v in match_pct.values()),
        "score_p_max_abs_diff": float(diff_p.max()), "score_p_exactly_equal_pct": float((diff_p == 0).mean() * 100),
        "score_h_max_abs_diff": float(diff_h.max()), "score_h_exactly_equal_pct": float((diff_h == 0).mean() * 100),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "autonomous_loader_equivalence_summary.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"[loader-equiv] -> {out_csv}  all_100pct={resumen['all_100pct']} "
          f"score_p_max_diff={resumen['score_p_max_abs_diff']:.2e} score_h_max_diff={resumen['score_h_max_abs_diff']:.2e}")
    print(json.dumps(match_pct, indent=2))
    return resumen


def _start_local_autonomous_uvicorn() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "edge_deployment.api.main:app", "--host", LOCAL_HOST,
         "--port", str(LOCAL_PORT), "--workers", "1", "--log-level", "warning"],
        cwd=str(BASE_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def run_container_equivalence(days: int = 3) -> dict:
    """C vs D: API local autonoma vs Docker autonomo (sin montar ningun CSV), cada una
    comparada contra el motor directo autonomo (referencia B), reutilizando literalmente
    `clients.api_stream_client.run_equivalence` (Fase 2, sin modificar -- desde Fase 4 su
    DirectEngineClient() interno ya usa el autonomous_loader por defecto)."""
    from edge_deployment.clients.api_stream_client import run_equivalence

    print("[container-equiv] C: arrancando API local autonoma (sin Docker)...")
    proc = _start_local_autonomous_uvicorn()
    try:
        ok, dt = cc.wait_ready(LOCAL_PORT, timeout=90)
        if not ok:
            salida = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"API local no respondio ready=true a tiempo.\n{salida[-2000:]}")
        r_c = run_equivalence(max_stream_days=days, base_url=f"http://{LOCAL_HOST}:{LOCAL_PORT}",
                               overwrite=True, out_suffix="_phase4_C_local_autonomous")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("[container-equiv] D: arrancando Docker autonomo (fdia-edge:optimized, SIN montar ningun CSV)...")
    name = cc.new_run_id("autonomous_docker")
    cc.start_container("fdia-edge:optimized", name, DOCKER_PORT, mount_legacy_data=False)
    try:
        ok, dt = cc.wait_ready(DOCKER_PORT, timeout=90)
        if not ok:
            raise RuntimeError(f"Docker autonomo no alcanzo ready=true. Logs:\n{cc.container_logs(name, 60)}")
        r_d = run_equivalence(max_stream_days=days, base_url=f"http://127.0.0.1:{DOCKER_PORT}",
                               overwrite=True, out_suffix="_phase4_D_docker_autonomous")
    finally:
        cc.stop_and_remove(name)

    df_c = r_c["df"].copy(); df_c.insert(0, "environment", "C_local_api_autonomous")
    df_d = r_d["df"].copy(); df_d.insert(0, "environment", "D_docker_autonomous")
    combined = pd.concat([df_c, df_d], ignore_index=True)
    out_csv = TABLES_DIR / "autonomous_container_equivalence.csv"
    combined.to_csv(out_csv, index=False)

    all_100 = r_c["resumen"]["all_categorical_100pct"] and r_d["resumen"]["all_categorical_100pct"]
    resumen = {
        "days": days,
        "C_local_api_autonomous": {"n_readings": r_c["resumen"]["n_readings"], "match_pct": r_c["resumen"]["match_pct"],
                                    "all_categorical_100pct": r_c["resumen"]["all_categorical_100pct"]},
        "D_docker_autonomous": {"n_readings": r_d["resumen"]["n_readings"], "match_pct": r_d["resumen"]["match_pct"],
                                 "all_categorical_100pct": r_d["resumen"]["all_categorical_100pct"],
                                 "no_csv_mounted": True},
        "all_environments_100pct": all_100,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "autonomous_container_equivalence_summary.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"[container-equiv] -> {out_csv}  all_environments_100pct={all_100}")
    return resumen


def run_30day_docker_autonomous(days: int = 30) -> dict:
    from edge_deployment.clients.api_stream_client import run_equivalence

    name = cc.new_run_id("autonomous_docker_30d")
    print(f"[container-equiv-30d] arrancando Docker autonomo para {days} dias (sin montar ningun CSV)...")
    cc.start_container("fdia-edge:optimized", name, DOCKER_PORT_30D, mount_legacy_data=False)
    try:
        ok, dt = cc.wait_ready(DOCKER_PORT_30D, timeout=90)
        if not ok:
            raise RuntimeError(f"Docker autonomo (30d) no alcanzo ready=true. Logs:\n{cc.container_logs(name, 60)}")
        r = run_equivalence(max_stream_days=days, base_url=f"http://127.0.0.1:{DOCKER_PORT_30D}",
                             overwrite=True, out_suffix="_phase4_D_docker_autonomous_30d")
    finally:
        cc.stop_and_remove(name)

    r["df"].to_csv(TABLES_DIR / "autonomous_container_equivalence_30d.csv", index=False)
    resumen = {"days": days, "n_readings": r["resumen"]["n_readings"],
               "all_categorical_100pct": r["resumen"]["all_categorical_100pct"],
               "match_pct": r["resumen"]["match_pct"], "no_csv_mounted": True,
               "fecha_utc": datetime.now(timezone.utc).isoformat()}
    with open(REPORTS_DIR / "autonomous_container_equivalence_summary_30d.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"[container-equiv-30d] n={resumen['n_readings']} all_100pct={resumen['all_categorical_100pct']}")
    return resumen


if __name__ == "__main__":
    loader_summary = run_loader_equivalence(days=3)
    if not loader_summary["all_100pct"]:
        print("[autonomous-equiv] DETENIDO: A vs B (legacy vs autonomo) no coincide al 100%. "
              "No se adopta el cargador autonomo. No se continua con C/D.")
        sys.exit(1)

    container_summary = run_container_equivalence(days=3)
    if not container_summary["all_environments_100pct"]:
        print("[autonomous-equiv] DETENIDO: C/D no coinciden al 100% contra la referencia. "
              "No se adopta el cargador autonomo para produccion.")
        sys.exit(1)

    print("[autonomous-equiv] smoke test OK en las 4 rutas -> lanzando 30 dias en Docker autonomo...")
    run_30day_docker_autonomous(days=30)
