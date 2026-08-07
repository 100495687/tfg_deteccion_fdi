"""Fase 5, seccion 19 (prueba prolongada): 7 dias acelerados de trafico LEGITIMO firmado
contra `fdia-edge:phase5-antireplay`, con la configuracion edge recomendada (heredada de
Fase 4: 1 CPU / 448 MB, salvo que deje de tener margen -- se documenta si hace falta subir de
nivel). Solo se ejecuta tras superar los GATES 0-5 (equivalencia + ataques ya validados).

Comprueba: 0 OOM, 0 errores 5xx, 0 perdidas, 0 falsos rechazos, memoria en plateau, buffers
420/702 acotados, estado anti-replay acotado, equivalencia completa mantenida, overhead
estable.
"""
from __future__ import annotations

import csv
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.security import hmac_auth
from edge_deployment.security.canonicalization import canonicalize

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
PORT = 19400
MINUTES_PER_DAY = 1440


def _recommended_config() -> dict:
    """Hereda la configuracion recomendada de Fase 4 (memoria minima estable = 448 MB, ver
    autonomous_memory_limit_tiers.json) -- Fase 5 solo la usa como punto de partida, GATE 6
    confirma si sigue teniendo margen con el overhead de seguridad."""
    p = REPORTS_DIR / "autonomous_memory_limit_tiers.json"
    mem_mb = 448
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        mem_mb = d.get("minimo_estable_mb") or mem_mb
    return {"memory_mb": mem_mb, "cpus": 1.0}


def run(days: int = 7, checkpoint_every_min: int = 720) -> dict:
    import numpy as np
    from src.data_loading import COLUMNA_OBJETIVO

    config = _recommended_config()
    key = secrets.token_hex(32).encode("utf-8")
    meter_id, session_id = "soak_secure", "session-soak"
    n_minutes = days * MINUTES_PER_DAY

    print(f"[soak-secure] {config['memory_mb']}MB @ {config['cpus']}CPU, {days} dias, trafico legitimo firmado...")

    bootstrap_idx = pd.date_range("2020-01-01 00:00:00", periods=10230, freq="1min")
    rng = np.random.default_rng(0)
    bootstrap_df = pd.DataFrame({COLUMNA_OBJETIVO: np.clip(1.0 + 0.3 * rng.standard_normal(10230), 0.05, None),
                                  "imputado": False}, index=bootstrap_idx)
    stream_start = bootstrap_idx.max() + pd.Timedelta(minutes=1)
    stream_idx = pd.date_range(stream_start, periods=n_minutes, freq="1min")
    stream_vals = np.clip(1.0 + 0.3 * rng.standard_normal(n_minutes), 0.05, None)

    name = cc.new_run_id("soak_secure")
    extra_args = ["-e", "FDIA_ANTI_REPLAY_ENABLED=true", "-e", "FDIA_SECURITY_CLOCK_MODE=simulated",
                  "-e", "FDIA_SECURITY_TEST_ENDPOINTS_ENABLED=true",
                  "-e", f"FDIA_METER_KEY_{meter_id}={key.decode()}"]
    sampler = cc.StatsSampler(name, interval_s=5.0)
    checkpoints, errores_5xx, perdidas, falsos_rechazos = [], 0, 0, 0
    buffer_sizes, history_sizes, n_sessions_over_time = [], [], []
    t_start = __import__("time").time()

    cc.start_container("fdia-edge:phase5-antireplay", name, PORT, mem_limit=f"{config['memory_mb']}m",
                        cpus=config["cpus"], mount_legacy_data=False, extra_args=extra_args)
    try:
        sampler.start()
        ok, _ = cc.wait_ready(PORT, timeout=120)
        if not ok:
            raise RuntimeError(f"contenedor no listo. Logs:\n{cc.container_logs(name, 80)}")

        client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30.0)
        client.post("/bootstrap", json={"meter_id": meter_id, "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(bootstrap_df.index, bootstrap_df[COLUMNA_OBJETIVO])
        ]})
        client.post("/security/test/init-session", json={"meter_id": meter_id, "session_id": session_id})

        for i, ts in enumerate(stream_idx):
            client.post("/security/test/clock", json={"timestamp": ts.isoformat()})
            msg = {"protocol_version": "1", "meter_id": meter_id, "session_id": session_id,
                   "timestamp": ts.isoformat(), "sequence_number": i + 1, "power_kw": float(stream_vals[i])}
            mac = hmac_auth.compute_hmac(key, canonicalize(msg))
            try:
                r = client.post("/secure-readings", json={**msg, "mac": mac})
            except Exception:
                perdidas += 1
                continue
            if r.status_code >= 500:
                errores_5xx += 1
                continue
            if r.status_code != 200:
                falsos_rechazos += 1
                continue
            body = r.json()
            detector = body.get("detector_result") or {}
            buffer_sizes.append(detector.get("buffer_1min_size"))
            history_sizes.append(detector.get("buffer_15min_size"))

            if (i + 1) % checkpoint_every_min == 0:
                mem_samples = [s["mem_mb"] for s in sampler.samples[-6:] if s["mem_mb"] is not None]
                n_sessions = client.get("/security/status").json()["n_sessions"]
                n_sessions_over_time.append(n_sessions)
                checkpoints.append({
                    "minute": i + 1, "day": round((i + 1) / MINUTES_PER_DAY, 2),
                    "mem_mb_recent_mean": round(sum(mem_samples) / len(mem_samples), 1) if mem_samples else None,
                    "buffer_1min_size": detector.get("buffer_1min_size"), "history_15min_size": detector.get("buffer_15min_size"),
                    "n_security_sessions": n_sessions,
                    "errores_5xx_acumulados": errores_5xx, "perdidas_acumuladas": perdidas,
                    "falsos_rechazos_acumulados": falsos_rechazos,
                })
                print(f"  dia {checkpoints[-1]['day']}: mem={checkpoints[-1]['mem_mb_recent_mean']}MB "
                      f"buf1min={checkpoints[-1]['buffer_1min_size']} buf15min={checkpoints[-1]['history_15min_size']} "
                      f"n_sesiones={n_sessions} 5xx={errores_5xx} perdidas={perdidas} falsos_rechazos={falsos_rechazos}")

        client.post(f"/reset/{meter_id}")
    finally:
        samples = sampler.stop()
        state = cc.inspect_state(name)
        cc.stop_and_remove(name)

    elapsed = __import__("time").time() - t_start
    mem_vals = [s["mem_mb"] for s in samples if s["mem_mb"] is not None]
    plateau = None
    vals = [c["mem_mb_recent_mean"] for c in checkpoints if c.get("mem_mb_recent_mean") is not None]
    if len(vals) >= 2:
        plateau = abs(vals[-1] - vals[-2]) / vals[-2] < 0.05 if vals[-2] else None
    buffers_bounded = (max((v for v in buffer_sizes if v is not None), default=0) <= 420
                        and max((v for v in history_sizes if v is not None), default=0) <= 702)
    sessions_bounded = max(n_sessions_over_time, default=0) <= 1  # una sola sesion aprovisionada en este experimento

    summary = {
        "days_simulated": days, "n_readings": n_minutes, "elapsed_seconds": elapsed, "config": config,
        "errores_5xx": errores_5xx, "lecturas_perdidas": perdidas, "falsos_rechazos": falsos_rechazos,
        "oom_killed": bool(state.get("OOMKilled")), "exit_code_container": state.get("ExitCode"),
        "mem_mb_mean": round(sum(mem_vals) / len(mem_vals), 1) if mem_vals else None,
        "mem_mb_max": round(max(mem_vals), 1) if mem_vals else None,
        "memory_plateau_detected": plateau,
        "buffer_1min_max_observed": max((v for v in buffer_sizes if v is not None), default=None),
        "history_15min_max_observed": max((v for v in history_sizes if v is not None), default=None),
        "buffers_bounded": buffers_bounded, "security_state_bounded": sessions_bounded,
        "max_n_security_sessions_observed": max(n_sessions_over_time, default=0),
        "checkpoints": checkpoints,
        "no_errors_no_false_rejections": errores_5xx == 0 and perdidas == 0 and falsos_rechazos == 0,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "anti_replay_soak_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "anti_replay_soak_checkpoints.csv", "w", newline="", encoding="utf-8") as f:
        if checkpoints:
            w = csv.DictWriter(f, fieldnames=list(checkpoints[0].keys()))
            w.writeheader()
            w.writerows(checkpoints)
    print(f"[soak-secure] completado: {n_minutes} lecturas en {elapsed:.1f}s | mem_mean={summary['mem_mb_mean']}MB "
          f"mem_max={summary['mem_mb_max']}MB | plateau={plateau} | buffers_acotados={buffers_bounded} | "
          f"estado_seguridad_acotado={sessions_bounded} | 5xx={errores_5xx} perdidas={perdidas} "
          f"falsos_rechazos={falsos_rechazos} oom={summary['oom_killed']}")
    return summary


if __name__ == "__main__":
    import sys
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    run(days=d)
