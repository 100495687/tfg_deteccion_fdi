"""Fase 4: repite solo las metricas que pueden cambiar al eliminar la lectura de
CSV (tiempo hasta /ready, pico de memoria durante el arranque, RAM al quedar ready, RAM
estable, tamano de imagen, latencia inicial, minimos de memoria) -- no repite toda la Fase 3.
Compara explicitamente contra los CSV ya guardados de Fase 3 (docker_startup_metrics.csv,
docker_image_metrics.csv, docker_memory_limit_matrix.csv), que quedan intactos (no se
sobreescriben, salvaguarda del usuario) y sirven de columna "legacy (Fase 3, con montajes)"
en las tablas de comparacion.
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.docker_tools.measure_startup import _one_startup_run, _image_size_true_bytes, N_STARTUP_RUNS

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

STARTUP_PORT_BASE = 18900
PROBE_SCRIPT = EDGE_DIR / "docker_tools" / "_memory_stage_probe_autonomous.py"


def measure_autonomous_startup(n_runs: int = N_STARTUP_RUNS) -> pd.DataFrame:
    print(f"[autonomous-startup] {n_runs} arranques del contenedor optimized, SIN montar ningun CSV...")
    rows = []
    port = STARTUP_PORT_BASE
    for i in range(n_runs):
        print(f"[autonomous-startup] run {i+1}/{n_runs} (puerto {port})...")
        row = _one_startup_run("fdia-edge:optimized", "optimized_autonomous", i, port)
        print(f"  ready={row['time_to_ready_s']}s ram_docker_stats={row['ram_at_ready_mb_docker_stats']}MB "
              f"rss={row['process_rss_mb_at_ready_via_metrics']}MB")
        rows.append(row)
        port += 1
    return pd.DataFrame(rows)


def build_startup_comparison() -> pd.DataFrame:
    legacy_path = TABLES_DIR / "docker_startup_metrics.csv"
    if not legacy_path.exists():
        raise FileNotFoundError(f"faltan las medidas de Fase 3 ({legacy_path}) -- necesarias como referencia 'legacy'")
    legacy_df = pd.read_csv(legacy_path)
    legacy_opt = legacy_df[legacy_df.image == "optimized"]

    auto_df = measure_autonomous_startup()

    def _stats(df: pd.DataFrame, col: str) -> dict:
        return {"mean": float(df[col].mean()), "std": float(df[col].std()), "min": float(df[col].min()), "max": float(df[col].max())}

    comparison_rows = []
    for metric, col in [("time_to_ready_s", "time_to_ready_s"),
                         ("ram_at_ready_mb_docker_stats", "ram_at_ready_mb_docker_stats"),
                         ("process_rss_mb_at_ready_via_metrics", "process_rss_mb_at_ready_via_metrics")]:
        s_legacy = _stats(legacy_opt, col)
        s_auto = _stats(auto_df, col)
        comparison_rows.append({
            "metric": metric,
            "legacy_mean_phase3_with_csv_mounts": s_legacy["mean"], "legacy_std": s_legacy["std"],
            "autonomous_mean_phase4_no_mounts": s_auto["mean"], "autonomous_std": s_auto["std"],
            "delta": s_auto["mean"] - s_legacy["mean"],
            "delta_pct": 100.0 * (s_auto["mean"] - s_legacy["mean"]) / s_legacy["mean"] if s_legacy["mean"] else None,
        })

    img_legacy = pd.read_csv(TABLES_DIR / "docker_image_metrics.csv")
    img_legacy_opt = img_legacy[img_legacy.image == "optimized"].iloc[0]
    img_auto_bytes = _image_size_true_bytes("fdia-edge:optimized")
    comparison_rows.append({
        "metric": "image_size_bytes_true",
        "legacy_mean_phase3_with_csv_mounts": float(img_legacy_opt["size_bytes_true_docker_save"]), "legacy_std": None,
        "autonomous_mean_phase4_no_mounts": float(img_auto_bytes), "autonomous_std": None,
        "delta": img_auto_bytes - img_legacy_opt["size_bytes_true_docker_save"],
        "delta_pct": 100.0 * (img_auto_bytes - img_legacy_opt["size_bytes_true_docker_save"]) / img_legacy_opt["size_bytes_true_docker_save"],
    })

    df = pd.DataFrame(comparison_rows)
    out = TABLES_DIR / "autonomous_startup_comparison.csv"
    df.to_csv(out, index=False)
    auto_df.to_csv(TABLES_DIR / "autonomous_startup_runs_raw.csv", index=False)
    print(f"[autonomous-startup] -> {out}")
    print(df.to_string(index=False))
    return df


def measure_autonomous_memory_decomposition() -> list[dict]:
    args = ["docker", "run", "--rm", "--name", cc.new_run_id("memprobe_auto"),
            "-v", f"{PROBE_SCRIPT}:/app/_memory_stage_probe_autonomous.py:ro",
            "--entrypoint", "python", "fdia-edge:optimized", "/app/_memory_stage_probe_autonomous.py"]
    print("[autonomous-memory] ejecutando sonda de memoria autonoma (SIN montar ningun CSV)...")
    r = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"sonda autonoma fallo (exit={r.returncode}):\nSTDOUT:{r.stdout[-3000:]}\nSTDERR:{r.stderr[-3000:]}")
    rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip().startswith("{")]
    for i in range(1, len(rows)):
        rows[i]["rss_delta_mb"] = round(rows[i]["rss_mb"] - rows[i - 1]["rss_mb"], 2)
    rows[0]["rss_delta_mb"] = None
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "autonomous_memory_decomposition.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "rss_mb", "rss_delta_mb", "t_elapsed_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"[autonomous-memory] -> {out}")
    for row in rows:
        delta = f"(+{row['rss_delta_mb']})" if row["rss_delta_mb"] else ""
        print(f"  {row['stage']}: {row['rss_mb']}MB {delta}")
    return rows


def build_memory_comparison(autonomous_stages: list[dict]) -> pd.DataFrame:
    legacy_path = TABLES_DIR / "docker_memory_decomposition.csv"
    legacy_df = pd.read_csv(legacy_path)
    legacy_peak_load = float(legacy_df[legacy_df.stage.str.contains("p_y_h_cargados")]["rss_mb"].iloc[0])
    legacy_steady = float(legacy_df[legacy_df.stage.str.contains("streaming_tras_3000")]["rss_mb"].iloc[0])

    auto_df = pd.DataFrame(autonomous_stages)
    auto_peak_load = float(auto_df[auto_df.stage.str.contains("p_y_h_cargados")]["rss_mb"].iloc[0])
    auto_steady = float(auto_df[auto_df.stage.str.contains("streaming_tras_3000")]["rss_mb"].iloc[0])

    rows = [
        {"metric": "rss_mb_peak_after_P_and_H_load", "legacy_phase3_with_csv": legacy_peak_load,
         "autonomous_phase4_no_csv": auto_peak_load, "delta": auto_peak_load - legacy_peak_load,
         "delta_pct": 100.0 * (auto_peak_load - legacy_peak_load) / legacy_peak_load},
        {"metric": "rss_mb_steady_after_3000_readings", "legacy_phase3_with_csv": legacy_steady,
         "autonomous_phase4_no_csv": auto_steady, "delta": auto_steady - legacy_steady,
         "delta_pct": 100.0 * (auto_steady - legacy_steady) / legacy_steady},
    ]
    df = pd.DataFrame(rows)
    out = TABLES_DIR / "autonomous_memory_comparison.csv"
    df.to_csv(out, index=False)
    print(f"[autonomous-memory] -> {out}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    build_startup_comparison()
    stages = measure_autonomous_memory_decomposition()
    build_memory_comparison(stages)
