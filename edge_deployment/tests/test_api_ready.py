"""Tests 8-10."""
from __future__ import annotations


def test_8_ready_returns_200_when_prepared(api_client):
    r = api_client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_9_ready_returns_503_when_not_prepared(preserve_startup_reports):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edge_deployment.api import lifecycle as lc
    from edge_deployment.api.error_handlers import register_error_handlers
    from edge_deployment.api.routes import router

    original = lc.validate_artifacts
    try:
        lc.validate_artifacts = lambda: {"hashes_valid": False, "n_missing": 1, "missing": ["x"],
                                          "manifest": {}, "validation_time_ms": 0.0}
        app_aislada = FastAPI(lifespan=lc.lifespan)
        register_error_handlers(app_aislada)
        app_aislada.include_router(router)
        with TestClient(app_aislada) as client:
            r = client.get("/ready")
            assert r.status_code == 503
    finally:
        lc.validate_artifacts = original


def test_10_ready_exposes_correct_thresholds(api_client):
    r = api_client.get("/ready").json()
    assert r["threshold_p"] == 0.049748
    assert abs(r["threshold_h"] - 0.883628) < 1e-5
    assert r["architecture"] == "P_OR_H"
    assert r["workers_supported"] == 1
    assert r["state_backend"] == "in_memory"
