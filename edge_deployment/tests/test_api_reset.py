"""Tests 27-29. Politica de /reset sobre un meter_id inexistente: idempotente
(devuelve 200 con reset=true, state_exists=false) en vez de 404 -- documentada en
README_API.md y en api_final_report.md, pregunta 18."""
from __future__ import annotations

import pandas as pd


def test_27_reset_empties_state(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    r = client.post(f"/reset/{meter_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["reset"] is True
    assert body["state_exists"] is False
    assert client.get(f"/status/{meter_id}").status_code == 404


def test_28_reset_does_not_unload_models(api_bootstrapped):
    client = api_bootstrapped["client"]
    m0 = client.get("/metrics").json()["model_load_time_ms"]
    client.post(f"/reset/{api_bootstrapped['meter_id']}")
    r = client.get("/ready")
    assert r.json()["models_loaded"] is True
    m1 = client.get("/metrics").json()["model_load_time_ms"]
    assert m0 == m1


def test_29_reset_does_not_reload_models(api_bootstrapped):
    client = api_bootstrapped["client"]
    r = client.post(f"/reset/{api_bootstrapped['meter_id']}")
    assert r.json()["models_reloaded"] is False


def test_reset_idempotent_on_unknown_meter(api_client):
    r = api_client.post("/reset/meter_que_nunca_existio")
    assert r.status_code == 200
    assert r.json() == {"meter_id": "meter_que_nunca_existio", "reset": True, "models_reloaded": False, "state_exists": False}


def test_reset_allows_rebootstrap(api_client, bootstrap_series, meter_id):
    from edge_deployment.tests.conftest import bootstrap_readings_payload
    payload = bootstrap_readings_payload(meter_id, bootstrap_series)
    r1 = api_client.post("/bootstrap", json=payload)
    assert r1.status_code == 200
    api_client.post(f"/reset/{meter_id}")
    r2 = api_client.post("/bootstrap", json=payload)
    assert r2.status_code == 200, "tras un reset debe poder volver a bootstrapearse el mismo meter_id"
    api_client.post(f"/reset/{meter_id}")
