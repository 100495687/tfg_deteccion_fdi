"""Tests 11-15."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from edge_deployment.tests.conftest import bootstrap_readings_payload, synthetic_series

API_DIR = Path(__file__).resolve().parents[1] / "api"


def test_11_bootstrap_calls_engine(api_client, bootstrap_series, meter_id):
    r = api_client.post("/bootstrap", json=bootstrap_readings_payload(meter_id, bootstrap_series))
    assert r.status_code == 200
    body = r.json()
    assert body["bootstrap_completed"] is True
    assert body["readings_received"] == len(bootstrap_series)
    api_client.post(f"/reset/{meter_id}")


def test_12_bootstrap_does_not_reimplement_aggregation():
    text = (API_DIR / "routes.py").read_text(encoding="utf-8")
    assert "construir_serie_15min" not in text
    assert "rolling(" not in text and ".shift(" not in text


def test_13_bootstrap_generates_no_retrospective_alerts(api_client, bootstrap_series, meter_id):
    r = api_client.post("/bootstrap", json=bootstrap_readings_payload(meter_id, bootstrap_series))
    body = r.json()
    assert "alert" not in "".join(body.keys()).lower()
    api_client.post(f"/reset/{meter_id}")


def test_14_bootstrap_keeps_p_not_ready(api_bootstrapped):
    assert api_bootstrapped["report"]["p_ready"] is False


def test_15_bootstrap_allows_h_ready(api_bootstrapped):
    assert api_bootstrapped["report"]["h_ready"] is True


def test_bootstrap_duplicate_rejected_with_409(api_client, bootstrap_series, meter_id):
    r1 = api_client.post("/bootstrap", json=bootstrap_readings_payload(meter_id, bootstrap_series))
    assert r1.status_code == 200
    r2 = api_client.post("/bootstrap", json=bootstrap_readings_payload(meter_id, bootstrap_series))
    assert r2.status_code == 409
    assert r2.json()["error_code"] == "bootstrap_duplicate"
    api_client.post(f"/reset/{meter_id}")


def test_bootstrap_disordered_rejected_with_422(api_client, meter_id):
    serie = synthetic_series(200, start="2022-06-01", seed=2)
    invertida = serie.iloc[::-1]
    payload = bootstrap_readings_payload(meter_id, invertida)
    r = api_client.post("/bootstrap", json=payload)
    assert r.status_code == 422


def test_bootstrap_with_duplicates_rejected_with_422(api_client, meter_id):
    serie = synthetic_series(200, start="2022-07-01", seed=3)
    con_dup = pd.concat([serie, serie.iloc[[5]]])
    payload = bootstrap_readings_payload(meter_id, con_dup)
    r = api_client.post("/bootstrap", json=payload)
    assert r.status_code == 422


def test_bootstrap_empty_readings_rejected_with_422(api_client, meter_id):
    r = api_client.post("/bootstrap", json={"meter_id": meter_id, "readings": []})
    assert r.status_code == 422
