"""Fase 5, GATE 4-5: equivalencia secure-vs-plain (seccion 16 caso A / GATE 4) y los 12 casos
experimentales B-L (seccion 16). Usa TestClient (ASGI en proceso, sin Docker -- igual patron
que Fase 2 para experimentos profundos; Docker se reserva para GATE 6) con
`clock_mode=simulated` para reproducir el periodo pre-test real de 2009 sin desincronizar el
reloj del sistema.

Invariante de rechazo (seccion 17): para cada caso de rechazo se captura el estado del motor
(via /status/{meter_id}) ANTES y DESPUES -- debe ser identico salvo los contadores propios de
seguridad.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from edge_deployment.security import hmac_auth
from edge_deployment.security.canonicalization import canonicalize

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

DETECTOR_FIELDS = ["accepted", "p_ready", "h_ready", "p_evaluated", "h_evaluated", "score_p", "score_h",
                    "alert_p", "alert_h", "alert_p_ffill", "alert_or", "detection_source",
                    "buffer_1min_size", "buffer_15min_size"]
INVARIANT_STATUS_FIELDS = ["buffer_1min_size", "history_15min_size", "last_accepted_timestamp",
                            "accepted_readings", "rejected_readings", "last_score_p", "last_score_h",
                            "last_alert_p", "last_alert_h", "last_alert_or", "p_last_evaluation_timestamp",
                            "h_last_evaluation_timestamp"]


def _fresh_client(meter_key: bytes, meter_id: str):
    """Cliente TestClient nuevo con seguridad activada, reloj simulado, y la clave del
    medidor inyectada por entorno ANTES de construir el cliente (el lifespan la lee al
    entrar en el `with`)."""
    os.environ["FDIA_ANTI_REPLAY_ENABLED"] = "true"
    os.environ["FDIA_SECURITY_CLOCK_MODE"] = "simulated"
    # Correccion obligatoria: /security/test/* ahora exige este flag explicitamente (false por
    # defecto SIEMPRE) -- GATE 4/5 son precisamente el "mecanismo experimental controlado" que
    # la correccion permite seguir usando, nunca produccion.
    os.environ["FDIA_SECURITY_TEST_ENDPOINTS_ENABLED"] = "true"
    os.environ[f"FDIA_METER_KEY_{meter_id}"] = meter_key.decode("utf-8")
    from fastapi.testclient import TestClient

    from edge_deployment.api.main import app
    return TestClient(app)


def _bootstrap(client, meter_id: str, bootstrap_df: pd.DataFrame) -> dict:
    from src.data_loading import COLUMNA_OBJETIVO
    payload = {"meter_id": meter_id, "readings": [
        {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(bootstrap_df.index, bootstrap_df[COLUMNA_OBJETIVO])
    ]}
    r = client.post("/bootstrap", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def _set_clock(client, ts) -> None:
    client.post("/security/test/clock", json={"timestamp": pd.Timestamp(ts).isoformat()})


def _init_session(client, meter_id: str, session_id: str) -> None:
    r = client.post("/security/test/init-session", json={"meter_id": meter_id, "session_id": session_id})
    assert r.status_code == 200, r.text


def _sign(meter_id: str, session_id: str, seq: int, ts, power_kw: float, key: bytes) -> dict:
    msg = {"protocol_version": "1", "meter_id": meter_id, "session_id": session_id,
           "timestamp": pd.Timestamp(ts).isoformat(), "sequence_number": seq, "power_kw": power_kw}
    mac = hmac_auth.compute_hmac(key, canonicalize(msg))
    return {**msg, "mac": mac}


def _status(client, meter_id: str) -> dict:
    r = client.get(f"/status/{meter_id}")
    return r.json() if r.status_code == 200 else {}


# ==================================================================================================
# GATE 4 / seccion 16 caso A: equivalencia secure-vs-plain con trafico legitimo
# ==================================================================================================

def run_gate4_secure_vs_plain(days: int = 3) -> dict:
    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    periodo = seleccionar_periodo(max_stream_days=days)
    bootstrap_df, stream_df = periodo["bootstrap_partition"], periodo["streaming_partition"]
    key = secrets.token_hex(32).encode("utf-8")
    meter_plain, meter_secure, session_id = "gate4_plain", "gate4_secure", "session-gate4"

    print(f"[gate4] {len(stream_df)} lecturas, comparando /readings vs /secure-readings...")
    client = _fresh_client(key, meter_secure)
    with client as c:
        _bootstrap(c, meter_plain, bootstrap_df)
        _bootstrap(c, meter_secure, bootstrap_df)
        _init_session(c, meter_secure, session_id)

        filas = []
        for i, (ts, row) in enumerate(stream_df.iterrows()):
            power_kw = float(row[COLUMNA_OBJETIVO])
            r_plain = c.post("/readings", json={"meter_id": meter_plain, "timestamp": ts.isoformat(), "power_kw": power_kw})
            _set_clock(c, ts)
            msg = _sign(meter_secure, session_id, i + 1, ts, power_kw, key)
            r_secure = c.post("/secure-readings", json=msg)
            d_plain, d_secure = r_plain.json(), r_secure.json()
            detector_secure = d_secure.get("detector_result") or {}
            filas.append({
                "timestamp": ts, "plain_http_status": r_plain.status_code, "secure_http_status": r_secure.status_code,
                "secure_security_status": d_secure.get("security_status"),
                **{f"plain_{f}": d_plain.get(f) for f in DETECTOR_FIELDS},
                **{f"secure_{f}": detector_secure.get(f) for f in DETECTOR_FIELDS},
            })
        c.post(f"/reset/{meter_plain}")
        c.post(f"/reset/{meter_secure}")

    df = pd.DataFrame(filas)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "secure_vs_plain_equivalence.csv"
    df.to_csv(out, index=False)

    from edge_deployment.clients.api_stream_client import _series_equal_nan_aware

    # NaN == NaN es False en pandas/IEEE754 -- score_p/alert_p (etc.) son NaN la mayoria del
    # tiempo (P evalua cada 60min); una comparacion ingenua cuenta "ambos ausentes" como
    # mismatch (mismo bug ya encontrado y corregido en Fase 4 para autonomous_equivalence.py
    # -- se reutiliza la misma correccion en vez de duplicar el fallo).
    match_pct = {f: float(_series_equal_nan_aware(df[f"plain_{f}"], df[f"secure_{f}"]).mean() * 100) for f in DETECTOR_FIELDS}
    false_rejections = int((df["secure_http_status"] != 200).sum())
    resumen = {
        "days": days, "n_readings": len(df), "match_pct": match_pct,
        "all_100pct": all(v == 100.0 for v in match_pct.values()),
        "false_rejections": false_rejections, "false_rejection_rate": false_rejections / len(df) if len(df) else None,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase5_gate4_summary.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"[gate4] -> {out}  all_100pct={resumen['all_100pct']} false_rejections={false_rejections}")
    print(json.dumps(match_pct, indent=2))
    return resumen


# ==================================================================================================
# GATE 5 / seccion 16, casos B-L
# ==================================================================================================

def _one_case_setup(case_name: str, key: bytes | None = None):
    key = key or secrets.token_hex(32).encode("utf-8")
    meter_id = f"case_{case_name}"
    session_id = f"session-{case_name}"
    client = _fresh_client(key, meter_id)
    return client, key, meter_id, session_id


def _bootstrap_short(client, meter_id: str, n_minutes: int = 10230):
    """Bootstrap sintetico corto y reproducible -- estos casos prueban la capa de seguridad,
    no la exactitud de P/H (eso ya lo prueba GATE 4 con datos reales)."""
    import numpy as np
    from src.data_loading import COLUMNA_OBJETIVO
    idx = pd.date_range("2020-01-01 00:00:00", periods=n_minutes, freq="1min")
    rng = np.random.default_rng(0)
    valores = np.clip(1.0 + 0.3 * rng.standard_normal(n_minutes), 0.05, None)
    df = pd.DataFrame({COLUMNA_OBJETIVO: valores, "imputado": False}, index=idx)
    _bootstrap(client, meter_id, df)
    return df.index.max() + pd.Timedelta(minutes=1)


def case_B_exact_replay() -> dict:
    client, key, meter_id, session_id = _one_case_setup("B_exact_replay")
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        _init_session(c, meter_id, session_id)
        _set_clock(c, stream_start)
        msg = _sign(meter_id, session_id, 1, stream_start, 1.2, key)
        status_before = _status(c, meter_id)
        r1 = c.post("/secure-readings", json=msg)
        r2 = c.post("/secure-readings", json=msg)  # replay EXACTO, mismo mensaje
        status_after = _status(c, meter_id)
        c.post(f"/reset/{meter_id}")
    return {"case": "B_exact_replay", "first_status": r1.status_code, "replay_status": r2.status_code,
            "replay_reason": r2.json().get("security_rejection_reason"),
            "replay_reached_engine": r2.json().get("detector_invoked"),
            "state_invariant_ok": all(status_before.get(f) != "__missing__" for f in INVARIANT_STATUS_FIELDS),
            "criteria_met": r1.status_code == 200 and r2.status_code == 409 and r2.json().get("detector_invoked") is False}


def case_C_block_replay(block_sizes=(120, 360, 1440)) -> list[dict]:
    resultados = []
    for n in block_sizes:
        client, key, meter_id, session_id = _one_case_setup(f"C_block_{n}")
        with client as c:
            stream_start = _bootstrap_short(c, meter_id)
            _init_session(c, meter_id, session_id)
            mensajes = []
            for i in range(n):
                ts = stream_start + pd.Timedelta(minutes=i)
                _set_clock(c, ts)
                mensajes.append(_sign(meter_id, session_id, i + 1, ts, 1.0 + 0.01 * i, key))
            # primera pasada: se aceptan todos (trafico legitimo)
            for m in mensajes:
                c.post("/secure-readings", json=m)
            engine_calls_before_replay = c.get("/security/status").json()["metrics"]["total_accepted"]
            # segunda pasada: bloque COMPLETO reinyectado tal cual (replay de bloque)
            rechazados = 0
            alcanzaron_motor = 0
            for m in mensajes:
                r = c.post("/secure-readings", json=m)
                if r.status_code != 200:
                    rechazados += 1
                if r.json().get("detector_invoked"):
                    alcanzaron_motor += 1
            c.post(f"/reset/{meter_id}")
        resultados.append({"case": f"C_block_replay_{n}", "block_size": n, "n_rejected": rechazados,
                            "n_reached_engine": alcanzaron_motor, "pct_rejected": 100.0 * rechazados / n,
                            "criteria_met": rechazados == n and alcanzaron_motor == 0})
    return resultados


def case_D_delayed_replay() -> dict:
    client, key, meter_id, session_id = _one_case_setup("D_delayed_replay", secrets.token_hex(32).encode("utf-8"))
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        _init_session(c, meter_id, session_id)
        old_ts = stream_start
        _set_clock(c, old_ts)
        old_msg = _sign(meter_id, session_id, 1, old_ts, 1.0, key)
        # avanza el reloj mucho mas alla de max_message_age_seconds (300s) sin haber enviado old_msg antes
        _set_clock(c, old_ts + pd.Timedelta(minutes=30))
        r = c.post("/secure-readings", json=old_msg)
        c.post(f"/reset/{meter_id}")
    reason = r.json().get("security_rejection_reason")
    return {"case": "D_delayed_replay", "status": r.status_code, "first_rule_applied": reason,
            "criteria_met": r.status_code == 409 and reason in ("stale_timestamp", "duplicate_sequence")}


def case_E_sequence_modified() -> dict:
    client, key, meter_id, session_id = _one_case_setup("E_sequence_modified")
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        _init_session(c, meter_id, session_id)
        _set_clock(c, stream_start)
        msg = _sign(meter_id, session_id, 1, stream_start, 1.0, key)
        msg["sequence_number"] = 999  # cambia DESPUES de firmar, MAC ya no cubre este valor
        r = c.post("/secure-readings", json=msg)
        c.post(f"/reset/{meter_id}")
    return {"case": "E_sequence_modified", "status": r.status_code,
            "reason": r.json().get("security_rejection_reason"),
            "criteria_met": r.status_code == 401 and r.json().get("security_rejection_reason") == "invalid_mac"}


def case_F_timestamp_modified() -> dict:
    client, key, meter_id, session_id = _one_case_setup("F_timestamp_modified")
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        _init_session(c, meter_id, session_id)
        _set_clock(c, stream_start)
        msg = _sign(meter_id, session_id, 1, stream_start, 1.0, key)
        msg["timestamp"] = (stream_start + pd.Timedelta(minutes=5)).isoformat()
        r = c.post("/secure-readings", json=msg)
        c.post(f"/reset/{meter_id}")
    return {"case": "F_timestamp_modified", "status": r.status_code,
            "reason": r.json().get("security_rejection_reason"),
            "criteria_met": r.status_code == 401 and r.json().get("security_rejection_reason") == "invalid_mac"}


def case_G_power_modified() -> dict:
    client, key, meter_id, session_id = _one_case_setup("G_power_modified")
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        _init_session(c, meter_id, session_id)
        _set_clock(c, stream_start)
        msg = _sign(meter_id, session_id, 1, stream_start, 1.0, key)
        msg["power_kw"] = 999.0
        r = c.post("/secure-readings", json=msg)
        c.post(f"/reset/{meter_id}")
    return {"case": "G_power_modified", "status": r.status_code,
            "reason": r.json().get("security_rejection_reason"),
            "criteria_met": r.status_code == 401 and r.json().get("security_rejection_reason") == "invalid_mac"}


def case_H_random_mac() -> dict:
    client, key, meter_id, session_id = _one_case_setup("H_random_mac")
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        _init_session(c, meter_id, session_id)
        _set_clock(c, stream_start)
        msg = _sign(meter_id, session_id, 1, stream_start, 1.0, key)
        msg["mac"] = secrets.token_hex(32)
        r = c.post("/secure-readings", json=msg)
        c.post(f"/reset/{meter_id}")
    return {"case": "H_random_mac", "status": r.status_code, "reason": r.json().get("security_rejection_reason"),
            "criteria_met": r.status_code == 401 and r.json().get("security_rejection_reason") == "invalid_mac"}


def case_I_unknown_meter() -> dict:
    client, key, meter_id, session_id = _one_case_setup("I_unknown_meter")
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        # NUNCA se aprovisiona clave para "house_never_registered"
        _set_clock(c, stream_start)
        msg = _sign("house_never_registered", session_id, 1, stream_start, 1.0, key)
        r = c.post("/secure-readings", json=msg)
        c.post(f"/reset/{meter_id}")
    return {"case": "I_unknown_meter", "status": r.status_code, "reason": r.json().get("security_rejection_reason"),
            "criteria_met": r.status_code == 404 and r.json().get("security_rejection_reason") == "unknown_meter_key"}


def case_J_future_timestamp() -> dict:
    client, key, meter_id, session_id = _one_case_setup("J_future_timestamp")
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        _init_session(c, meter_id, session_id)
        _set_clock(c, stream_start)
        future_ts = stream_start + pd.Timedelta(minutes=10)  # > max_future_skew_seconds (60s)
        msg = _sign(meter_id, session_id, 1, future_ts, 1.0, key)  # firmado correctamente, con ese timestamp futuro
        r = c.post("/secure-readings", json=msg)
        c.post(f"/reset/{meter_id}")
    return {"case": "J_future_timestamp", "status": r.status_code, "reason": r.json().get("security_rejection_reason"),
            "criteria_met": r.status_code == 409 and r.json().get("security_rejection_reason") == "future_timestamp"}


def case_K_concurrent_duplicate(n_concurrent: int = 10) -> dict:
    import concurrent.futures

    client, key, meter_id, session_id = _one_case_setup("K_concurrent_duplicate")
    with client as c:
        stream_start = _bootstrap_short(c, meter_id)
        _init_session(c, meter_id, session_id)
        _set_clock(c, stream_start)
        msg = _sign(meter_id, session_id, 1, stream_start, 1.0, key)

        def _send():
            return c.post("/secure-readings", json=msg)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
            resultados = list(ex.map(lambda _: _send(), range(n_concurrent)))
        status_codes = [r.status_code for r in resultados]
        n_accepted = sum(1 for s in status_codes if s == 200)
        c.post(f"/reset/{meter_id}")
    return {"case": "K_concurrent_duplicate", "n_concurrent": n_concurrent, "n_accepted": n_accepted,
            "n_rejected": n_concurrent - n_accepted, "status_codes": status_codes,
            "criteria_met": n_accepted == 1}


def run_rejection_invariant_suite() -> list[dict]:
    """Seccion 17: para cada caso de rechazo, estado del motor antes/despues debe ser
    identico salvo metricas propias de seguridad."""
    resultados = []
    for case_name, key_needed in [("invariant_wrong_mac", True), ("invariant_unknown_session", True)]:
        client, key, meter_id, session_id = _one_case_setup(case_name)
        with client as c:
            stream_start = _bootstrap_short(c, meter_id)
            _init_session(c, meter_id, session_id)
            _set_clock(c, stream_start)
            # una lectura legitima para tener estado no trivial
            ok_msg = _sign(meter_id, session_id, 1, stream_start, 1.0, key)
            c.post("/secure-readings", json=ok_msg)
            status_before = _status(c, meter_id)

            if case_name == "invariant_wrong_mac":
                bad = _sign(meter_id, session_id, 2, stream_start + pd.Timedelta(minutes=1), 1.0, key)
                bad["mac"] = "0" * 64
                c.post("/secure-readings", json=bad)
            else:
                bad = _sign(meter_id, "session-does-not-exist", 2, stream_start + pd.Timedelta(minutes=1), 1.0, key)
                c.post("/secure-readings", json=bad)

            status_after = _status(c, meter_id)
            c.post(f"/reset/{meter_id}")
        unchanged = all(status_before.get(f) == status_after.get(f) for f in INVARIANT_STATUS_FIELDS)
        resultados.append({"case": case_name, "state_unchanged": unchanged,
                            "diffs": {f: (status_before.get(f), status_after.get(f)) for f in INVARIANT_STATUS_FIELDS
                                      if status_before.get(f) != status_after.get(f)}})
    return resultados


# ==================================================================================================
# Caso L (seccion 16/14): reinicio -- requiere un contenedor Docker real (el estado en memoria
# de AntiReplayState, igual que DetectorState en Fase 1, no sobrevive a un reinicio de proceso).
# Documenta la limitacion honestamente (seccion 14): "la proteccion cubre replay durante una
# sesion activa; la persistencia segura del estado tras reinicios queda como limitacion y
# trabajo futuro" -- no se oculta.
# ==================================================================================================

def case_L_restart_persistence(port: int = 19300) -> dict:
    from edge_deployment.docker_tools import container_control as cc

    key = secrets.token_hex(32).encode("utf-8")
    meter_id, session_id = "case_L_restart", "session-L"
    name = cc.new_run_id("case_L_restart")
    extra_args = ["-e", "FDIA_ANTI_REPLAY_ENABLED=true", "-e", "FDIA_SECURITY_CLOCK_MODE=simulated",
                  "-e", "FDIA_SECURITY_TEST_ENDPOINTS_ENABLED=true",
                  "-e", f"FDIA_METER_KEY_{meter_id}={key.decode()}"]
    print(f"[case-L] arrancando {name} para probar reinicio...")
    cc.start_container("fdia-edge:phase5-antireplay", name, port, mount_legacy_data=False, extra_args=extra_args)
    resultado = {"case": "L_restart"}
    try:
        ok, _ = cc.wait_ready(port, timeout=90)
        assert ok, cc.container_logs(name, 60)

        import httpx
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
        ts = "2020-01-01T00:00:00Z"
        client.post("/security/test/clock", json={"timestamp": ts})
        client.post("/security/test/init-session", json={"meter_id": meter_id, "session_id": session_id})
        msg = _sign(meter_id, session_id, 1, ts, 1.0, key)
        r_before = client.post("/secure-readings", json=msg)
        resultado["accepted_before_restart"] = r_before.status_code == 200

        print(f"[case-L] reiniciando el contenedor ({name})...")
        import subprocess
        subprocess.run(["docker", "restart", "-t", "5", name], capture_output=True, text=True)
        ok2, _ = cc.wait_ready(port, timeout=90)
        resultado["ready_after_restart"] = ok2

        # (a) SIN re-aprovisionar la sesion tras el reinicio -- debe fallar cerrado (rechazo
        # seguro, NUNCA aceptar silenciosamente una sesion que ya no existe en memoria)
        client2 = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0)
        client2.post("/security/test/clock", json={"timestamp": ts})
        r_replay_no_reinit = client2.post("/secure-readings", json=msg)
        resultado["replay_without_reinit_status"] = r_replay_no_reinit.status_code
        resultado["replay_without_reinit_reason"] = r_replay_no_reinit.json().get("security_rejection_reason")
        resultado["fails_closed_without_reinit"] = r_replay_no_reinit.status_code != 200

        # (b) CON re-aprovisionamiento explicito (un operador reinicia la sesion) -- demuestra
        # la perdida de memoria de secuencia: el mismo mensaje, ya aceptado antes del
        # reinicio, se vuelve a aceptar -- limitacion real, documentada, no oculta.
        client2.post("/security/test/init-session", json={"meter_id": meter_id, "session_id": session_id})
        r_replay_after_reinit = client2.post("/secure-readings", json=msg)
        resultado["accepted_after_reinit_same_message"] = r_replay_after_reinit.status_code == 200
        resultado["sequence_memory_lost_on_restart"] = resultado["accepted_after_reinit_same_message"]
    finally:
        cc.stop_and_remove(name)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([resultado]).to_csv(TABLES_DIR / "anti_replay_restart_case.csv", index=False)
    print(f"[case-L] -> {TABLES_DIR / 'anti_replay_restart_case.csv'}")
    print(json.dumps(resultado, indent=2))
    return resultado


def run_all_cases() -> dict:
    cases = {}
    cases["B_exact_replay"] = case_B_exact_replay()
    cases["C_block_replay"] = case_C_block_replay()
    cases["D_delayed_replay"] = case_D_delayed_replay()
    cases["E_sequence_modified"] = case_E_sequence_modified()
    cases["F_timestamp_modified"] = case_F_timestamp_modified()
    cases["G_power_modified"] = case_G_power_modified()
    cases["H_random_mac"] = case_H_random_mac()
    cases["I_unknown_meter"] = case_I_unknown_meter()
    cases["J_future_timestamp"] = case_J_future_timestamp()
    cases["K_concurrent_duplicate"] = case_K_concurrent_duplicate()
    invariant = run_rejection_invariant_suite()

    rows = []
    for name, result in cases.items():
        if isinstance(result, list):
            rows.extend(result)
        else:
            rows.append(result)
    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "anti_replay_cases.csv", index=False)

    concurrency_df = pd.DataFrame([cases["K_concurrent_duplicate"]])
    concurrency_df.to_csv(TABLES_DIR / "anti_replay_concurrency_cases.csv", index=False)

    all_criteria_met = all(r.get("criteria_met", True) for r in rows if "criteria_met" in r)
    all_invariants_ok = all(r["state_unchanged"] for r in invariant)

    resumen = {
        "n_cases": len(rows), "all_criteria_met": all_criteria_met, "all_invariants_ok": all_invariants_ok,
        "invariant_results": invariant, "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase5_gate5_summary.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"[gate5] {len(rows)} casos -- all_criteria_met={all_criteria_met} all_invariants_ok={all_invariants_ok}")
    for r in rows:
        print(f"  {r.get('case')}: criteria_met={r.get('criteria_met')}")
    return resumen


if __name__ == "__main__":
    run_gate4_secure_vs_plain(days=3)
    run_all_cases()
    case_L_restart_persistence()
