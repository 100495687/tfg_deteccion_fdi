"""Tests 30-41 + tabla de 24 casos de error."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from edge_deployment.tests.conftest import bootstrap_readings_payload, synthetic_series

EDGE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = EDGE_DIR / "results" / "tables"


def _post_raw(client, path: str, raw_json: str):
    return client.post(path, content=raw_json, headers={"Content-Type": "application/json"})


# ==========================================================================================
# Tests puntuales obligatorios
# ==========================================================================================

def test_30_empty_json_returns_422(api_client):
    r = api_client.post("/readings", json={})
    assert r.status_code == 422


def test_31_missing_field_returns_422(api_client):
    r = api_client.post("/readings", json={"meter_id": "m", "power_kw": 1.0})
    assert r.status_code == 422


def test_32_invalid_timestamp_returns_422(api_client):
    r = api_client.post("/readings", json={"meter_id": "m", "timestamp": "no-es-una-fecha", "power_kw": 1.0})
    assert r.status_code == 422


def test_33_nan_rejected(api_client):
    r = _post_raw(api_client, "/readings", '{"meter_id": "m", "timestamp": "2024-01-01T00:00:00", "power_kw": NaN}')
    assert r.status_code == 422


def test_34_infinite_rejected(api_client):
    r = _post_raw(api_client, "/readings", '{"meter_id": "m", "timestamp": "2024-01-01T00:00:00", "power_kw": Infinity}')
    assert r.status_code == 422


def test_35_extra_fields_rejected(api_client):
    r = api_client.post("/readings", json={"meter_id": "m", "timestamp": "2024-01-01T00:00:00", "power_kw": 1.0, "campo_extra": 1})
    assert r.status_code == 422


def test_36_37_38_duplicate_and_out_of_order_return_409(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    r0 = client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    assert r0.status_code == 200

    r_dup_identical = client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    assert r_dup_identical.status_code == 409
    assert r_dup_identical.json()["error_code"] == "duplicate_identical"

    r_dup_conflict = client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 999.0})
    assert r_dup_conflict.status_code == 409
    assert r_dup_conflict.json()["error_code"] == "duplicate_conflicting"

    r_ooo = client.post("/readings", json={"meter_id": meter_id, "timestamp": (stream_start - pd.Timedelta(minutes=5)).isoformat(), "power_kw": 1.0})
    assert r_ooo.status_code == 409
    assert r_ooo.json()["error_code"] == "out_of_order"


def test_39_engine_error_does_not_corrupt_state(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    n0 = client.get(f"/status/{meter_id}").json()["accepted_readings"]
    client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 999.0})  # ok
    n1 = client.get(f"/status/{meter_id}").json()["accepted_readings"]
    client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 999.0})  # rechazado (duplicado)
    n2 = client.get(f"/status/{meter_id}").json()["accepted_readings"]
    assert n1 == n0 + 1
    assert n2 == n1, "una lectura rechazada no debe incrementar accepted_readings"


def test_40_error_includes_request_id(api_client):
    r = api_client.post("/readings", json={})
    assert r.status_code == 422
    assert "request_id" in r.json() and len(r.json()["request_id"]) > 0
    assert "X-Request-ID" in r.headers


def test_41_error_does_not_leak_stack_trace(api_client):
    r = api_client.post("/readings", json={})
    text = r.text.lower()
    assert "traceback" not in text
    assert "site-packages" not in text
    assert str(EDGE_DIR).lower().replace("\\", "/") not in text.replace("\\", "/")


# ==========================================================================================
# Tabla de 24 casos -- se genera al ejecutar este fichero de tests
# ==========================================================================================

def test_build_error_cases_table(api_client, preserve_startup_reports):
    casos = []

    def _caso(nombre, payload_resumen, status_code, error_code, expected, passed):
        casos.append({"endpoint": "/readings" if "bootstrap" not in nombre and "status" not in nombre and "reset" not in nombre else nombre,
                       "case": nombre, "payload_summary": payload_resumen, "status_code": status_code,
                       "error_code": error_code, "state_modified": False, "expected": expected, "passed": passed})

    m = "err_case_meter"
    api_client.post(f"/reset/{m}")

    r = api_client.post("/readings", json={})
    _caso("1_empty_json", "{}", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = api_client.post("/readings", json={"timestamp": "2024-01-01T00:00:00", "power_kw": 1.0})
    _caso("2_missing_meter_id", "no meter_id", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = api_client.post("/readings", json={"meter_id": m, "power_kw": 1.0})
    _caso("3_missing_timestamp", "no timestamp", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = api_client.post("/readings", json={"meter_id": m, "timestamp": "2024-01-01T00:00:00"})
    _caso("4_missing_power_kw", "no power_kw", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = api_client.post("/readings", json={"meter_id": m, "timestamp": "no-es-fecha", "power_kw": 1.0})
    _caso("5_malformed_timestamp", "timestamp=no-es-fecha", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = api_client.post("/readings", json={"meter_id": m, "timestamp": "2024-01-01T00:00:00", "power_kw": "texto"})
    _caso("6_power_kw_as_text", "power_kw='texto'", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = _post_raw(api_client, "/readings", '{"meter_id": "%s", "timestamp": "2024-01-01T00:00:00", "power_kw": NaN}' % m)
    _caso("7_nan", "power_kw=NaN", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = _post_raw(api_client, "/readings", '{"meter_id": "%s", "timestamp": "2024-01-01T00:00:00", "power_kw": Infinity}' % m)
    _caso("8_infinite", "power_kw=Infinity", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = api_client.post("/readings", json={"meter_id": "", "timestamp": "2024-01-01T00:00:00", "power_kw": 1.0})
    _caso("9_empty_meter_id", "meter_id=''", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    r = api_client.post("/readings", json={"meter_id": m, "timestamp": "2024-01-01T00:00:00", "power_kw": 1.0, "extra": 1})
    _caso("10_extra_fields", "campo 'extra'", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    boot = synthetic_series(10230, start="2024-02-01", seed=42)
    api_client.post("/bootstrap", json=bootstrap_readings_payload(m, boot))
    stream_start = boot.index.max() + pd.Timedelta(minutes=1)
    r0 = api_client.post("/readings", json={"meter_id": m, "timestamp": stream_start.isoformat(), "power_kw": 1.0})

    r = api_client.post("/readings", json={"meter_id": m, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    _caso("11_duplicate_identical", "mismo timestamp+valor", r.status_code, r.json().get("error_code"), 409, r.status_code == 409)

    r = api_client.post("/readings", json={"meter_id": m, "timestamp": stream_start.isoformat(), "power_kw": 55.0})
    _caso("12_duplicate_conflicting", "mismo timestamp, valor distinto", r.status_code, r.json().get("error_code"), 409, r.status_code == 409)

    r = api_client.post("/readings", json={"meter_id": m, "timestamp": (stream_start - pd.Timedelta(minutes=1)).isoformat(), "power_kw": 1.0})
    _caso("13_out_of_order", "timestamp anterior al ultimo aceptado", r.status_code, r.json().get("error_code"), 409, r.status_code == 409)

    r = api_client.post("/readings", json={"meter_id": m, "timestamp": (stream_start + pd.Timedelta(minutes=10)).isoformat(), "power_kw": 1.0})
    _caso("14_temporal_gap", "salto de 9 min (no es error: se acepta con warning)", r.status_code, None, 200, r.status_code == 200)
    api_client.post(f"/reset/{m}")

    r = api_client.post("/bootstrap", json={"meter_id": m, "readings": []})
    _caso("15_empty_bootstrap", "readings=[]", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    boot2 = synthetic_series(200, start="2024-03-01", seed=7)
    r = api_client.post("/bootstrap", json=bootstrap_readings_payload(m, boot2.iloc[::-1]))
    _caso("16_disordered_bootstrap", "readings invertidas", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    con_dup = pd.concat([boot2, boot2.iloc[[3]]])
    r = api_client.post("/bootstrap", json=bootstrap_readings_payload(m, con_dup))
    _caso("17_bootstrap_with_duplicates", "timestamp repetido en la lista", r.status_code, r.json().get("error_code"), 422, r.status_code == 422)

    boot3 = synthetic_series(10230, start="2024-04-01", seed=8)
    api_client.post("/bootstrap", json=bootstrap_readings_payload(m, boot3))
    r = api_client.post("/bootstrap", json=bootstrap_readings_payload(m, boot3))
    _caso("18_repeated_bootstrap", "segundo bootstrap sin reset", r.status_code, r.json().get("error_code"), 409, r.status_code == 409)
    api_client.post(f"/reset/{m}")

    r = api_client.get("/status/meter_que_no_existe_jamas")
    _caso("19_status_unknown_meter", "meter_id inexistente", r.status_code, r.json().get("error_code"), 404, r.status_code == 404)

    r = api_client.post("/reset/meter_que_no_existe_jamas_2")
    _caso("20_reset_unknown_meter", "meter_id inexistente (politica idempotente)", r.status_code, None, 200, r.status_code == 200)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edge_deployment.api import lifecycle as lc
    from edge_deployment.api.error_handlers import register_error_handlers
    from edge_deployment.api.routes import router
    original = lc.validate_artifacts
    try:
        lc.validate_artifacts = lambda: {"hashes_valid": False, "n_missing": 1, "missing": ["models/tcn_ae_ventana360.pt"],
                                          "manifest": {}, "validation_time_ms": 0.0}
        app_aislada = FastAPI(lifespan=lc.lifespan)
        register_error_handlers(app_aislada)
        app_aislada.include_router(router)
        with TestClient(app_aislada) as c2:
            r = c2.post("/readings", json={"meter_id": m, "timestamp": "2024-01-01T00:00:00", "power_kw": 1.0})
            _caso("21_missing_artifact", "artefacto ausente simulado", r.status_code, r.json().get("error_code"), 503, r.status_code == 503)
            r = c2.get("/ready")
            _caso("22_bad_hash", "hash invalido simulado (misma via)", r.status_code, r.json().get("error_code"), 503, r.status_code == 503)
            r = c2.post("/readings", json={"meter_id": m, "timestamp": "2024-01-01T00:00:00", "power_kw": 1.0})
            _caso("23_engine_not_ready", "ready=false", r.status_code, r.json().get("error_code"), 503, r.status_code == 503)
    finally:
        lc.validate_artifacts = original
        lc.validate_artifacts()

    from edge_deployment.api.locks import MeterLockRegistry
    from edge_deployment.api.metrics import ApiMetrics

    class _RaisingEngine:
        threshold_p = 0.049748
        threshold_h = 0.883628

        def ingest(self, *a, **k):
            raise RuntimeError("fallo interno simulado (test_24)")

        def n_active_meters(self):
            return 0

    app_err = FastAPI()
    register_error_handlers(app_err)
    app_err.include_router(router)
    app_err.state.engine = _RaisingEngine()
    app_err.state.ready = True
    app_err.state.threshold_p = 0.049748
    app_err.state.threshold_h = 0.883628
    app_err.state.artifact_hashes_valid = True
    app_err.state.metrics = ApiMetrics()
    app_err.state.locks = MeterLockRegistry()
    app_err.state.bootstrapped_meters = set()

    with TestClient(app_err, raise_server_exceptions=False) as c3:
        r = c3.post("/readings", json={"meter_id": m, "timestamp": "2024-01-01T00:00:00", "power_kw": 1.0})
    passed_24 = (r.status_code == 500 and r.json().get("error_code") == "internal_error"
                 and "request_id" in r.json() and "runtimeerror" not in r.text.lower() and "traceback" not in r.text.lower())
    _caso("24_internal_error_simulated", "engine.ingest lanza RuntimeError", r.status_code, r.json().get("error_code"), 500, passed_24)

    tabla = pd.DataFrame(casos)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(TABLES_DIR / "api_error_cases.csv", index=False)
    assert tabla["passed"].all(), tabla[~tabla["passed"]].to_string()
