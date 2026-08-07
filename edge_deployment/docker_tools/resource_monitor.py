"""Fase 3: CPU/RAM (docker stats, muestreo continuo) a lo largo de todo el ciclo de
vida de un unico contenedor optimized sin restricciones (referencia/cloud emulado)
-- arranque, carga de modelos, bootstrap, streaming, inferencia H, inferencia P+H, estado
estable. Las fases se etiquetan por rango temporal (marcas de tiempo de cada transicion) y se
cruzan con las muestras de `docker stats` despues, sin detener el muestreo entre fases (para no
perder continuidad de la serie).
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.docker_tools import synthetic_workload as sw

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
PORT = 18500
STREAM_MINUTES = 1000  # suficientes evaluaciones de P (stride 60min tras 360min de warmup) y H (cada 15min)


def run() -> dict:
    from src.data_loading import COLUMNA_OBJETIVO

    bootstrap_df, stream_df = sw.build_bootstrap_and_stream(stream_minutes=STREAM_MINUTES)
    name = cc.new_run_id("resource_monitor")
    print(f"[resource] arrancando contenedor optimized sin restricciones ({name})...")
    sampler = cc.StatsSampler(name, interval_s=0.5)
    marks = {}
    per_reading_phase = []
    try:
        marks["t_container_start"] = time.time()
        cc.start_container("fdia-edge:optimized", name, PORT)
        sampler.start()
        ok_ready, _ = cc.wait_ready(PORT, timeout=90)
        marks["t_ready_models_loaded"] = time.time()
        if not ok_ready:
            raise RuntimeError(f"contenedor no listo. Logs:\n{cc.container_logs(name, 60)}")

        import httpx
        client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30.0)
        marks["t_bootstrap_start"] = time.time()
        r = client.post("/bootstrap", json={"meter_id": "resmon", "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(bootstrap_df.index, bootstrap_df[COLUMNA_OBJETIVO])
        ]})
        assert r.status_code == 200, r.text
        marks["t_bootstrap_end"] = time.time()

        marks["t_streaming_start"] = time.time()
        for ts, row in stream_df.iterrows():
            t_req = time.time()
            r = client.post("/readings", json={"meter_id": "resmon", "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})
            body = r.json() if r.status_code == 200 else {}
            phase = "streaming_p_y_h" if body.get("p_evaluated") else ("streaming_solo_h" if body.get("h_evaluated") else "streaming_sin_inferencia")
            per_reading_phase.append({"t": t_req, "phase": phase})
        marks["t_streaming_end"] = time.time()

        # "estado estable": ultima porcion del streaming, ya con buffers en su tamano maximo (deques acotados, Fase 1)
        marks["t_steady_state_start"] = per_reading_phase[int(len(per_reading_phase) * 0.8)]["t"] if per_reading_phase else marks["t_streaming_end"]
        marks["t_steady_state_end"] = marks["t_streaming_end"]

        client.post("/reset/resmon")
        time.sleep(1)
    finally:
        samples = sampler.stop()
        cc.stop_and_remove(name)

    rows = _assign_phases(samples, marks, per_reading_phase)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "docker_resource_timeline.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t_wall", "phase", "cpu_pct", "mem_mb", "pids"])
        w.writeheader()
        w.writerows(rows)

    summary_rows = _summarize_by_phase(rows)
    out2 = TABLES_DIR / "docker_resource_summary.csv"
    with open(out2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "n_samples", "cpu_pct_mean", "cpu_pct_max", "mem_mb_mean", "mem_mb_max"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[resource] -> {out}\n[resource] -> {out2}")
    for r in summary_rows:
        print(f"  {r['phase']}: n={r['n_samples']} cpu_mean={r['cpu_pct_mean']}% cpu_max={r['cpu_pct_max']}% "
              f"mem_mean={r['mem_mb_mean']}MB mem_max={r['mem_mb_max']}MB")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "docker_resource_monitor_marks.json", "w", encoding="utf-8") as f:
        json.dump({**marks, "fecha_utc": datetime.now(timezone.utc).isoformat()}, f, indent=2, default=str)
    return {"timeline_rows": rows, "summary_rows": summary_rows, "marks": marks}


def _assign_phases(samples: list[dict], marks: dict, per_reading_phase: list[dict]) -> list[dict]:
    rows = []
    for s in samples:
        t = s["t_wall"]
        if t < marks["t_ready_models_loaded"]:
            phase = "1_arranque_y_carga_modelos"
        elif t < marks["t_bootstrap_end"]:
            phase = "2_bootstrap"
        elif t < marks["t_steady_state_start"]:
            phase = _nearest_reading_phase(t, per_reading_phase) or "3_streaming"
        else:
            phase = "4_estado_estable"
        rows.append({"t_wall": t, "phase": phase, "cpu_pct": s["cpu_pct"], "mem_mb": s["mem_mb"], "pids": s["pids"]})
    return rows


def _nearest_reading_phase(t: float, per_reading_phase: list[dict]) -> str | None:
    if not per_reading_phase:
        return None
    best = min(per_reading_phase, key=lambda r: abs(r["t"] - t))
    return best["phase"] if abs(best["t"] - t) < 2.0 else "3_streaming"


def _summarize_by_phase(rows: list[dict]) -> list[dict]:
    by_phase: dict[str, list[dict]] = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(r)
    out = []
    for phase, rs in sorted(by_phase.items()):
        cpu_vals = [r["cpu_pct"] for r in rs if r["cpu_pct"] is not None]
        mem_vals = [r["mem_mb"] for r in rs if r["mem_mb"] is not None]
        out.append({
            "phase": phase, "n_samples": len(rs),
            "cpu_pct_mean": round(sum(cpu_vals) / len(cpu_vals), 2) if cpu_vals else None,
            "cpu_pct_max": round(max(cpu_vals), 2) if cpu_vals else None,
            "mem_mb_mean": round(sum(mem_vals) / len(mem_vals), 1) if mem_vals else None,
            "mem_mb_max": round(max(mem_vals), 1) if mem_vals else None,
        })
    return out


if __name__ == "__main__":
    run()
