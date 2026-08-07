"""Fase 5, correccion obligatoria (punto 1): "Anade tests dentro de la imagen Docker de
produccion" -- verifica, contra la imagen `fdia-edge:phase5-antireplay` REAL ejecutandose en un
contenedor (no TestClient en proceso), que `/security/test/init-session` y `/security/test/clock`
estan bloqueados en produccion, y que el mecanismo experimental controlado sigue funcionando
cuando se activa explicitamente.

Dos contenedores, deliberadamente distintos (seccion "No basta con impedir unicamente el reloj
simulado"):
  1. "produccion": FDIA_ANTI_REPLAY_ENABLED=true, FDIA_SECURITY_TEST_ENDPOINTS_ENABLED SIN FIJAR
     (por tanto `false`, el default real de un despliegue sin variables de prueba) ->
     init-session y clock deben devolver 404, incluso con anti_replay_enabled=true y clock_mode
     por defecto (system, no simulado) -- 404 tiene que llegar ANTES de cualquier otra
     comprobacion.
  2. "experimental": ambos flags a true (el "mecanismo experimental controlado" que la
     correccion permite seguir usando fuera de produccion) -> confirma que /secure-readings
     sigue aceptando trafico legitimo con sesiones asi aprovisionadas, y que una segunda
     llamada a init-session para la MISMA sesion devuelve 409 (no se puede re-inicializar/
     resetear una sesion ya aprovisionada, ni siquiera con el mecanismo habilitado).
"""
from __future__ import annotations

import csv
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import httpx

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.security import hmac_auth
from edge_deployment.security.canonicalization import canonicalize

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
IMAGE = "fdia-edge:phase5-antireplay"

PORT_PROD = 19600
PORT_EXPERIMENTAL = 19601


def _check_production_endpoints_disabled(port: int = PORT_PROD) -> dict:
    name = cc.new_run_id("prodcheck")
    # deliberadamente SIN FDIA_SECURITY_TEST_ENDPOINTS_ENABLED -- asi se comporta un
    # despliegue real que solo activa anti_replay_enabled (nunca los endpoints de prueba)
    extra_args = ["-e", "FDIA_ANTI_REPLAY_ENABLED=true"]
    print(f"[prodcheck] arrancando {IMAGE} en modo produccion (test_endpoints_enabled NO fijado)...")
    cc.start_container(IMAGE, name, port, mount_legacy_data=False, extra_args=extra_args)
    result = {"scenario": "production_test_endpoints_disabled"}
    try:
        ok, _ = cc.wait_ready(port, timeout=90)
        if not ok:
            raise RuntimeError(f"{IMAGE} no alcanzo ready. Logs:\n{cc.container_logs(name, 60)}")
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15.0)

        r_init = client.post("/security/test/init-session", json={"meter_id": "attacker_meter", "session_id": "s1"})
        result["init_session_status"] = r_init.status_code
        result["init_session_returns_404"] = r_init.status_code == 404

        r_clock = client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        result["set_clock_status"] = r_clock.status_code
        result["set_clock_returns_404"] = r_clock.status_code == 404

        # /readings y /health/ready siguen intactos -- la correccion es aditiva, no rompe nada
        r_health = client.get("/health")
        result["health_still_ok"] = r_health.status_code == 200
        r_status = client.get("/security/status")
        result["security_status_still_reports"] = r_status.status_code == 200 and r_status.json().get("anti_replay_enabled") is True
    finally:
        cc.stop_and_remove(name)
    result["criteria_met"] = bool(result.get("init_session_returns_404") and result.get("set_clock_returns_404")
                                   and result.get("health_still_ok"))
    print(f"[prodcheck] init-session={result['init_session_status']} clock={result['set_clock_status']} "
          f"criteria_met={result['criteria_met']}")
    return result


def _check_experimental_mechanism_still_works(port: int = PORT_EXPERIMENTAL) -> dict:
    name = cc.new_run_id("expcheck")
    key = secrets.token_hex(32).encode("utf-8")
    meter_id, session_id = "exp_meter", "exp_session"
    extra_args = ["-e", "FDIA_ANTI_REPLAY_ENABLED=true", "-e", "FDIA_SECURITY_CLOCK_MODE=simulated",
                  "-e", "FDIA_SECURITY_TEST_ENDPOINTS_ENABLED=true",
                  "-e", f"FDIA_METER_KEY_{meter_id}={key.decode()}"]
    print(f"[expcheck] arrancando {IMAGE} con el mecanismo experimental habilitado explicitamente...")
    cc.start_container(IMAGE, name, port, mount_legacy_data=False, extra_args=extra_args)
    result = {"scenario": "experimental_mechanism_explicitly_enabled"}
    try:
        ok, _ = cc.wait_ready(port, timeout=90)
        if not ok:
            raise RuntimeError(f"{IMAGE} no alcanzo ready. Logs:\n{cc.container_logs(name, 60)}")
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15.0)

        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        r_init1 = client.post("/security/test/init-session", json={"meter_id": meter_id, "session_id": session_id})
        result["first_init_status"] = r_init1.status_code
        result["first_init_ok"] = r_init1.status_code == 200

        # correccion: ni siquiera con el mecanismo habilitado se puede re-inicializar una sesion existente
        r_init2 = client.post("/security/test/init-session", json={"meter_id": meter_id, "session_id": session_id})
        result["reinit_status"] = r_init2.status_code
        result["reinit_blocked"] = r_init2.status_code == 409

        msg = {"protocol_version": "1", "meter_id": meter_id, "session_id": session_id,
               "timestamp": "2020-01-01T00:00:00Z", "sequence_number": 1, "power_kw": 1.42}
        mac = hmac_auth.compute_hmac(key, canonicalize(msg))
        r_secure = client.post("/secure-readings", json={**msg, "mac": mac})
        result["secure_readings_status"] = r_secure.status_code
        result["secure_readings_works"] = r_secure.status_code == 200
    finally:
        cc.stop_and_remove(name)
    result["criteria_met"] = bool(result.get("first_init_ok") and result.get("reinit_blocked")
                                   and result.get("secure_readings_works"))
    print(f"[expcheck] first_init={result['first_init_status']} reinit={result['reinit_status']} "
          f"secure_readings={result['secure_readings_status']} criteria_met={result['criteria_met']}")
    return result


def run() -> dict:
    prod = _check_production_endpoints_disabled()
    exp = _check_experimental_mechanism_still_works()

    rows = [
        {"scenario": prod["scenario"], "check": "init_session_returns_404", "result": prod["init_session_returns_404"]},
        {"scenario": prod["scenario"], "check": "set_clock_returns_404", "result": prod["set_clock_returns_404"]},
        {"scenario": prod["scenario"], "check": "health_still_ok", "result": prod["health_still_ok"]},
        {"scenario": exp["scenario"], "check": "first_init_ok", "result": exp["first_init_ok"]},
        {"scenario": exp["scenario"], "check": "reinit_blocked_409", "result": exp["reinit_blocked"]},
        {"scenario": exp["scenario"], "check": "secure_readings_works", "result": exp["secure_readings_works"]},
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "anti_replay_production_endpoint_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "check", "result"])
        w.writeheader()
        w.writerows(rows)

    summary = {"production_scenario": prod, "experimental_scenario": exp,
               "all_criteria_met": bool(prod["criteria_met"] and exp["criteria_met"]),
               "fecha_utc": datetime.now(timezone.utc).isoformat()}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "anti_replay_production_check_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[prodcheck] -> {TABLES_DIR / 'anti_replay_production_endpoint_check.csv'}")
    print(f"[prodcheck] all_criteria_met={summary['all_criteria_met']}")
    return summary


if __name__ == "__main__":
    run()
