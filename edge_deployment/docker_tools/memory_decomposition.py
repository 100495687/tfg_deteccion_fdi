"""Fase 3: ejecuta _memory_stage_probe.py dentro de un contenedor optimized nuevo
(entrypoint sobreescrito, el script se monta de solo lectura -- nunca se hornea en la imagen
real). Un solo contenedor efimero (--rm, sin -d): se lanza en primer plano y se captura stdout
directamente, sin pasar por docker logs.
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from edge_deployment.docker_tools import container_control as cc

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
PROBE_SCRIPT = EDGE_DIR / "docker_tools" / "_memory_stage_probe.py"


def run() -> list[dict]:
    args = [
        "docker", "run", "--rm", "--name", cc.new_run_id("memprobe"),
        "-v", f"{cc.DATA_RAW_HOST}:/app/data/raw/data_raw.csv:ro",
        "-v", f"{cc.EPISODES_HOST}:/app/results/tables/episodes_master_val.csv:ro",
        "-v", f"{PROBE_SCRIPT}:/app/_memory_stage_probe.py:ro",
        "--entrypoint", "python",
        "fdia-edge:optimized", "/app/_memory_stage_probe.py",
    ]
    print("[mem-decomp] ejecutando sonda de memoria por etapas en un contenedor nuevo...")
    r = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"sonda de memoria fallo (exit={r.returncode}):\nSTDOUT:{r.stdout[-3000:]}\nSTDERR:{r.stderr[-3000:]}")

    rows = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise RuntimeError(f"la sonda no produjo ninguna linea JSON valida. STDOUT:\n{r.stdout[-3000:]}")

    for i in range(1, len(rows)):
        rows[i]["rss_delta_mb"] = round(rows[i]["rss_mb"] - rows[i - 1]["rss_mb"], 2)
    rows[0]["rss_delta_mb"] = None

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "docker_memory_decomposition.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "rss_mb", "rss_delta_mb", "t_elapsed_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"[mem-decomp] -> {out}")
    for row in rows:
        delta = f"(+{row['rss_delta_mb']})" if row["rss_delta_mb"] else ""
        print(f"  {row['stage']}: {row['rss_mb']}MB {delta}")
    return rows


if __name__ == "__main__":
    run()
