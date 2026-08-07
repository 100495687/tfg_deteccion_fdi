"""Test 25 + 404 para meter inexistente."""
from __future__ import annotations

import pandas as pd


def test_25_status_does_not_modify_state(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    s1 = client.get(f"/status/{meter_id}").json()
    s2 = client.get(f"/status/{meter_id}").json()
    assert s1["accepted_readings"] == s2["accepted_readings"]
    assert s1["buffer_1min_size"] == s2["buffer_1min_size"]
    assert s1["last_accepted_timestamp"] == s2["last_accepted_timestamp"]


def test_status_returns_404_for_unknown_meter(api_client):
    r = api_client.get("/status/no_existe_este_meter")
    assert r.status_code == 404
    assert r.json()["error_code"] == "meter_not_found"


def test_status_returns_200_and_expected_fields(api_bootstrapped):
    client, meter_id = api_bootstrapped["client"], api_bootstrapped["meter_id"]
    r = client.get(f"/status/{meter_id}")
    assert r.status_code == 200
    body = r.json()
    for campo in ["meter_id", "state_exists", "engine_status", "stream_anchor_timestamp",
                  "current_bucket_start", "current_bucket_sample_count", "p_ready", "h_ready",
                  "last_alert_or", "buffer_1min_size", "history_15min_size", "accepted_readings",
                  "rejected_readings", "duplicates", "out_of_order", "gaps"]:
        assert campo in body
