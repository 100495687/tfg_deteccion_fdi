"""Tests 6-7."""
from __future__ import annotations

import time


def test_6_health_returns_200(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "fdia-edge-detector"


def test_7_health_is_fast_and_does_not_run_inference(api_client):
    t0 = time.perf_counter()
    r = api_client.get("/health")
    dt_ms = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200
    assert dt_ms < 200, f"/health tardo {dt_ms:.1f}ms -- sugiere que esta ejecutando inferencia"
