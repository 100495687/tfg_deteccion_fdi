"""Fase 5, seccion 19: rendimiento. Dos partes:

  1. Microbenchmark EN PROCESO de cada paso de seguridad por separado (canonicalizacion,
     verificacion HMAC, comprobacion de secuencia+frescura) -- llamadas directas a
     `security/*.py`, sin HTTP, para aislar el coste de cada paso.
  2. Comparacion de extremo a extremo con UNA carga fija (secciones 19: "no comparar cargas
     diferentes como si fueran equivalentes") entre tres objetivos reales por HTTP:
       A. fdia-edge:phase4-autonomous -> /readings
       B. fdia-edge:phase5-antireplay -> /readings (misma ruta, seguridad activada pero sin
          usarse -- confirma que activar la seguridad no penaliza /readings)
       C. fdia-edge:phase5-antireplay -> /secure-readings (trafico legitimo firmado)
"""
from __future__ import annotations

import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.security import hmac_auth
from edge_deployment.security.anti_replay import AntiReplayGuard
from edge_deployment.security.canonicalization import canonicalize
from edge_deployment.security.key_provider import InMemoryTestKeyProvider
from edge_deployment.security.security_clock import SimulatedClock
from edge_deployment.security.security_state import AntiReplayState

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

PORT_A = 19200
PORT_B = 19201
PORT_C = 19202


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    arr = np.array(values, dtype=float)
    return {"n": int(len(arr)), "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)), "max_ms": float(arr.max())}


def micro_benchmark_security_steps(n: int = 5000) -> dict:
    print(f"[bench] microbenchmark en proceso, n={n} por paso...")
    key = secrets.token_hex(32).encode("utf-8")
    meter_id, session_id = "bench_meter", "bench_session"
    kp = InMemoryTestKeyProvider({meter_id: key})
    clock = SimulatedClock(start=datetime(2020, 1, 1, tzinfo=timezone.utc))
    state = AntiReplayState()
    state.init_session(meter_id, session_id)
    guard = AntiReplayGuard(kp, clock, state, max_message_age_seconds=300, max_future_skew_seconds=60)

    msg_template = {"protocol_version": "1", "meter_id": meter_id, "session_id": session_id,
                     "timestamp": "2020-01-01T00:00:00Z", "sequence_number": 1, "power_kw": 1.0}

    t_canon = []
    for _ in range(n):
        t0 = time.perf_counter()
        canonical = canonicalize(msg_template)
        t_canon.append((time.perf_counter() - t0) * 1000)

    t_hmac_compute = []
    for _ in range(n):
        t0 = time.perf_counter()
        hmac_auth.compute_hmac(key, canonical)
        t_hmac_compute.append((time.perf_counter() - t0) * 1000)

    mac = hmac_auth.compute_hmac(key, canonical)
    t_hmac_verify = []
    for _ in range(n):
        t0 = time.perf_counter()
        hmac_auth.verify_hmac(key, canonical, mac)
        t_hmac_verify.append((time.perf_counter() - t0) * 1000)

    t_authenticity = []
    for _ in range(n):
        t0 = time.perf_counter()
        guard.verify_authenticity({**msg_template, "mac": mac})
        t_authenticity.append((time.perf_counter() - t0) * 1000)

    # secuencia+frescura: usa sequence_number creciente para no rechazar por duplicado
    t_seq_fresh = []
    for i in range(n):
        clock.set(datetime(2020, 1, 1, tzinfo=timezone.utc) + pd.Timedelta(minutes=i))
        ts = clock.now()
        t0 = time.perf_counter()
        guard.check_sequence_and_freshness(meter_id, session_id, i + 1, ts)
        t_seq_fresh.append((time.perf_counter() - t0) * 1000)
        guard.commit_accept(meter_id, session_id, i + 1, ts)

    resumen = {
        "n_per_step": n,
        "canonicalization": _percentiles(t_canon),
        "hmac_compute": _percentiles(t_hmac_compute),
        "hmac_verify": _percentiles(t_hmac_verify),
        "verify_authenticity_full": _percentiles(t_authenticity),
        "sequence_and_freshness_check": _percentiles(t_seq_fresh),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "anti_replay_microbenchmark.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"[bench] canon median={resumen['canonicalization']['median_ms']:.4f}ms "
          f"hmac_compute median={resumen['hmac_compute']['median_ms']:.4f}ms "
          f"hmac_verify median={resumen['hmac_verify']['median_ms']:.4f}ms "
          f"seq+fresh median={resumen['sequence_and_freshness_check']['median_ms']:.4f}ms")
    return resumen


def _run_endpoint_benchmark(image: str, port: int, endpoint: str, secure: bool, n_readings: int, warmup: int) -> dict:
    import httpx

    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    name = cc.new_run_id(f"bench_{endpoint.strip('/').replace('-', '_')}")
    extra_args = None
    if secure:
        key = secrets.token_hex(32).encode("utf-8")
        extra_args = ["-e", "FDIA_ANTI_REPLAY_ENABLED=true", "-e", "FDIA_SECURITY_CLOCK_MODE=simulated",
                      "-e", "FDIA_SECURITY_TEST_ENDPOINTS_ENABLED=true",
                      "-e", f"FDIA_METER_KEY_bench_meter={key.decode()}"]

    print(f"[bench] arrancando {image} para {endpoint} (secure={secure})...")
    cc.start_container(image, name, port, mount_legacy_data=False, extra_args=extra_args)
    filas = []
    errores = perdidas = 0
    try:
        ok, dt = cc.wait_ready(port, timeout=90)
        if not ok:
            raise RuntimeError(f"{image} no alcanzo ready. Logs:\n{cc.container_logs(name, 60)}")

        max_days = max(3, (n_readings + warmup) // 1440 + 2)
        periodo = seleccionar_periodo(max_stream_days=max_days)
        stream = periodo["streaming_partition"]
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
        meter_id = "bench_meter"

        # bootstrap SIEMPRE (regardless de secure/plain) -- mismos datos en los 3 objetivos,
        # para que H este ready y la mezcla de categorias (sin-inferencia/H/P+H) sea la MISMA
        # en las 3 comparaciones (bug real encontrado: la version anterior de este script
        # nunca arrancaba el motor secure con bootstrap, asi que H nunca evaluaba y el
        # objetivo C parecia artificialmente ~3x mas rapido -- corregido).
        client.post("/bootstrap", json={"meter_id": meter_id, "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)}
            for ts, v in zip(periodo["bootstrap_partition"].index, periodo["bootstrap_partition"][COLUMNA_OBJETIVO])
        ]})
        if secure:
            client.post("/security/test/clock", json={"timestamp": stream.index[0].isoformat()})
            client.post("/security/test/init-session", json={"meter_id": meter_id, "session_id": "bench_session"})

        def _send(seq, ts, power_kw):
            if secure:
                client.post("/security/test/clock", json={"timestamp": ts.isoformat()})
                msg = {"protocol_version": "1", "meter_id": meter_id, "session_id": "bench_session",
                       "timestamp": ts.isoformat(), "sequence_number": seq, "power_kw": power_kw}
                mac = hmac_auth.compute_hmac(key, canonicalize(msg))
                return client.post(endpoint, json={**msg, "mac": mac})
            return client.post(endpoint, json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": power_kw})

        for i, (ts, row) in enumerate(stream.iloc[:warmup].iterrows()):
            _send(i + 1, ts, float(row[COLUMNA_OBJETIVO]))

        t_start = time.perf_counter()
        for i, (ts, row) in enumerate(stream.iloc[warmup:warmup + n_readings].iterrows()):
            t0 = time.perf_counter()
            try:
                r = _send(warmup + i + 1, ts, float(row[COLUMNA_OBJETIVO]))
            except Exception:
                perdidas += 1
                continue
            round_trip_ms = (time.perf_counter() - t0) * 1000
            if r.status_code >= 500:
                errores += 1
            body = r.json()
            filas.append({"timestamp": ts, "http_status": r.status_code, "round_trip_ms": round_trip_ms,
                          "engine_processing_time_ms": (body.get("detector_result") or {}).get("processing_time_ms")
                          if secure else body.get("engine_processing_time_ms"),
                          "security_processing_time_ms": body.get("security_processing_time_ms")})
        elapsed = time.perf_counter() - t_start
    finally:
        cc.stop_and_remove(name)

    df = pd.DataFrame(filas)
    return {"df": df, "elapsed": elapsed, "errores": errores, "perdidas": perdidas}


def run_endpoint_comparison(n_readings: int = 2000, warmup: int = 100) -> dict:
    results = {}
    for label, image, port, endpoint, secure in [
        ("A_phase4_readings", "fdia-edge:phase4-autonomous", PORT_A, "/readings", False),
        ("B_phase5_readings", "fdia-edge:phase5-antireplay", PORT_B, "/readings", False),
        ("C_phase5_secure_readings", "fdia-edge:phase5-antireplay", PORT_C, "/secure-readings", True),
    ]:
        r = _run_endpoint_benchmark(image, port, endpoint, secure, n_readings, warmup)
        results[label] = r
        print(f"[bench] {label}: n={len(r['df'])} throughput={len(r['df'])/r['elapsed']:.2f}/s "
              f"errors={r['errores']} lost={r['perdidas']}")

    all_rows = []
    for label, r in results.items():
        df = r["df"].copy()
        df.insert(0, "target", label)
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TABLES_DIR / "anti_replay_latency.csv", index=False)

    summary = {}
    for label, r in results.items():
        df = r["df"]
        summary[label] = {
            "n": len(df), "throughput_rps": len(df) / r["elapsed"] if r["elapsed"] > 0 else None,
            "errors_5xx": r["errores"], "lost": r["perdidas"],
            "round_trip": _percentiles(df["round_trip_ms"].tolist()) if len(df) else None,
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "anti_replay_latency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps({k: v["round_trip"]["median_ms"] if v["round_trip"] else None for k, v in summary.items()}, indent=2))
    return summary


if __name__ == "__main__":
    micro_benchmark_security_steps()
    run_endpoint_comparison()
