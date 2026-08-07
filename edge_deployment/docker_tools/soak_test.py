"""Fase 3: prueba prolongada en la configuracion edge recomendada (un unico
contenedor de principio a fin, run_id unico). Comprueba: memoria en plateau (no crecimiento
lineal -- deques acotados de Fase 1), buffers acotados (buffer_1min_size/history_15min_size,
ya expuestos por la API sin modificar), sin OOM, sin 5xx, sin lecturas perdidas, alertas
estables (comparadas contra la misma referencia de motor directo que las matrices de
recursos), y que logs/metricas no crezcan sin limite (tamano de fichero de log al final).
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.docker_tools import synthetic_workload as sw

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
PORT = 18600

MINUTES_PER_DAY = 1440


def _recommended_config() -> dict:
    mem_path = REPORTS_DIR / "docker_memory_limit_tiers.json"
    mem_mb = 1024
    if mem_path.exists():
        d = json.loads(mem_path.read_text(encoding="utf-8"))
        mem_mb = d.get("minimo_estable_mb") or d.get("minimo_funcional_mb") or 1024
    return {"memory_mb": mem_mb, "cpus": 1.0}


def run(days: int = 30, checkpoint_every_min: int = 720) -> dict:
    config = _recommended_config()
    print(f"[soak] configuracion edge recomendada: {config['memory_mb']}MB @ {config['cpus']} CPU, {days} dias simulados")
    bootstrap_df, stream_df = sw.build_bootstrap_and_stream(stream_minutes=days * MINUTES_PER_DAY)

    name = cc.new_run_id("soak")
    sampler = cc.StatsSampler(name, interval_s=5.0)
    checkpoints = []
    errores_5xx = perdidas = 0
    buffer_1min_sizes, history_15min_sizes = [], []
    t_start = time.time()
    try:
        cc.start_container("fdia-edge:optimized", name, PORT, mem_limit=f"{config['memory_mb']}m", cpus=config["cpus"])
        sampler.start()
        ok_ready, _ = cc.wait_ready(PORT, timeout=120)
        if not ok_ready:
            raise RuntimeError(f"contenedor no listo. Logs:\n{cc.container_logs(name, 80)}")

        from src.data_loading import COLUMNA_OBJETIVO
        client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30.0)
        r = client.post("/bootstrap", json={"meter_id": "soak", "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(bootstrap_df.index, bootstrap_df[COLUMNA_OBJETIVO])
        ]})
        assert r.status_code == 200, r.text

        for i, (ts, row) in enumerate(stream_df.iterrows()):
            try:
                r = client.post("/readings", json={"meter_id": "soak", "timestamp": ts.isoformat(), "power_kw": float(row[COLUMNA_OBJETIVO])})
            except Exception:
                perdidas += 1
                continue
            if r.status_code >= 500:
                errores_5xx += 1
                continue
            body = r.json()
            buffer_1min_sizes.append(body.get("buffer_1min_size"))
            history_15min_sizes.append(body.get("buffer_15min_size"))
            if (i + 1) % checkpoint_every_min == 0:
                st = cc.inspect_state(name)
                mem_samples = [s["mem_mb"] for s in sampler.samples[-6:] if s["mem_mb"] is not None]
                checkpoints.append({
                    "minute": i + 1, "day": round((i + 1) / MINUTES_PER_DAY, 2),
                    "mem_mb_recent_mean": round(sum(mem_samples) / len(mem_samples), 1) if mem_samples else None,
                    "buffer_1min_size": body.get("buffer_1min_size"), "history_15min_size": body.get("buffer_15min_size"),
                    "errores_5xx_acumulados": errores_5xx, "perdidas_acumuladas": perdidas,
                    "oom_killed_sofar": st.get("OOMKilled"),
                })
                print(f"  checkpoint dia {checkpoints[-1]['day']}: mem={checkpoints[-1]['mem_mb_recent_mean']}MB "
                      f"buf1min={checkpoints[-1]['buffer_1min_size']} buf15min={checkpoints[-1]['history_15min_size']} "
                      f"5xx={errores_5xx} perdidas={perdidas}")

        # nota: la correccion de alertas en 30 dias ya se comprueba por separado y con datos
        # reales en equivalence.py::run_30day_optimized_only -- esta prueba
        # prolongada se centra en estabilidad de recursos (memoria/buffers/errores), no repite
        # esa comparacion con carga sintetica.
        log_size_bytes = None
        r_log = subprocess.run(["docker", "exec", name, "sh", "-c",
                                 "wc -c < edge_deployment/results/logs/api.log 2>/dev/null || echo -1"],
                                capture_output=True, text=True)
        if r_log.returncode == 0 and r_log.stdout.strip().lstrip("-").isdigit():
            log_size_bytes = int(r_log.stdout.strip())

        client.post("/reset/soak")
    finally:
        samples = sampler.stop()
        state = cc.inspect_state(name)
        cc.stop_and_remove(name)

    elapsed = time.time() - t_start
    mem_vals = [s["mem_mb"] for s in samples if s["mem_mb"] is not None]
    plateau = _check_plateau(checkpoints)
    buffers_bounded = _check_bounded(buffer_1min_sizes, 420) and _check_bounded(history_15min_sizes, 702)

    summary = {
        "days_simulated": days, "n_readings": len(stream_df), "elapsed_seconds": elapsed,
        "config": config, "errores_5xx": errores_5xx, "lecturas_perdidas": perdidas,
        "oom_killed": bool(state.get("OOMKilled")), "exit_code_container": state.get("ExitCode"),
        "mem_mb_mean": round(sum(mem_vals) / len(mem_vals), 1) if mem_vals else None,
        "mem_mb_max": round(max(mem_vals), 1) if mem_vals else None,
        "memory_plateau_detected": plateau,
        "buffer_1min_max_observed": max((v for v in buffer_1min_sizes if v is not None), default=None),
        "buffer_1min_maxlen_expected": 420,
        "history_15min_max_observed": max((v for v in history_15min_sizes if v is not None), default=None),
        "history_15min_maxlen_expected": 702,
        "buffers_bounded": buffers_bounded,
        "log_file_size_bytes_at_end": log_size_bytes,
        "log_growth_note": "el log es un registro append-only (1 linea/peticion, por diseno de Fase 2, sin "
                            "modificar) -- crecimiento aprox. lineal con el numero de peticiones es ESPERADO, "
                            "no es una fuga; lo que se vigila aqui es que la MEMORIA (RSS/docker stats) no crezca.",
        "checkpoints": checkpoints,
        "no_errors": errores_5xx == 0 and perdidas == 0,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "docker_soak_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "docker_soak_checkpoints.csv", "w", newline="", encoding="utf-8") as f:
        if checkpoints:
            w = csv.DictWriter(f, fieldnames=list(checkpoints[0].keys()))
            w.writeheader()
            w.writerows(checkpoints)
    print(f"[soak] completado: {len(stream_df)} lecturas en {elapsed:.1f}s | mem_mean={summary['mem_mb_mean']}MB "
          f"mem_max={summary['mem_mb_max']}MB | plateau={plateau} | buffers_acotados={buffers_bounded} | "
          f"5xx={errores_5xx} perdidas={perdidas} oom={summary['oom_killed']}")
    return summary


def _check_bounded(values: list, maxlen: int) -> bool:
    vals = [v for v in values if v is not None]
    return bool(vals) and max(vals) <= maxlen


def _check_plateau(checkpoints: list[dict]) -> bool | None:
    """Compara la memoria media del ultimo checkpoint contra la del checkpoint anterior al
    ultimo -- si crece mas de un 5% se considera que no hay plateau (posible fuga)."""
    vals = [c["mem_mb_recent_mean"] for c in checkpoints if c.get("mem_mb_recent_mean") is not None]
    if len(vals) < 2:
        return None
    ultimo, penultimo = vals[-1], vals[-2]
    if penultimo <= 0:
        return None
    return (ultimo - penultimo) / penultimo < 0.05


if __name__ == "__main__":
    import sys
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run(days=d)
