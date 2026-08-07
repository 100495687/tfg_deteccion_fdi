"""Test 26, 61-63."""
from __future__ import annotations

from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1] / "api"


def test_26_metrics_does_not_modify_state(api_bootstrapped):
    client, meter_id = api_bootstrapped["client"], api_bootstrapped["meter_id"]
    s0 = client.get(f"/status/{meter_id}").json()["accepted_readings"]
    client.get("/metrics")
    client.get("/metrics")
    s1 = client.get(f"/status/{meter_id}").json()["accepted_readings"]
    assert s0 == s1


def test_metrics_returns_200_and_expected_fields(api_client):
    r = api_client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    for campo in ["service_start_time", "uptime_seconds", "model_load_time_ms", "bootstrap_requests",
                  "reading_requests", "accepted_readings", "rejected_readings", "duplicate_identical",
                  "duplicate_conflicting", "out_of_order", "p_evaluations", "h_evaluations",
                  "alerts_p", "alerts_h", "alerts_or", "internal_errors", "active_meter_states",
                  "approximate_memory_mb"]:
        assert campo in body


def test_61_metrics_are_bounded():
    text = (API_DIR / "metrics.py").read_text(encoding="utf-8")
    assert "maxlen=" in text
    assert "deque" in text


def test_62_logs_do_not_include_full_windows():
    """Comprueba el contenido real de una linea de log de una lectura (logger aislado,
    nunca toca el logger compartido de la sesion) -- debe ser corta y no contener arrays
    completos de ventanas (360 muestras de P) ni listas de miles de lecturas de bootstrap."""
    import io
    import logging

    from edge_deployment.api import logging_config as lc

    logger = logging.getLogger("test_62_isolated_logger")
    logger.handlers.clear()
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    lc.log_reading_event(logger, request_id="rid-1", meter_id="m1", timestamp="2024-01-01T00:00:00",
                          accepted=True, rejection_reason=None, p_evaluated=True, h_evaluated=True,
                          alert_or=False, engine_time_ms=1.2, api_time_ms=2.3, status_code=200)
    handler.flush()
    linea = buffer.getvalue().strip()
    assert len(linea) < 500, "una linea de log de una lectura no deberia superar unos pocos cientos de caracteres"
    assert linea.count(",") < 30, "sugiere que se volco una lista/array completo en el log"


def test_63_models_not_included_in_responses(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    r = client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    body_text = r.text.lower()
    for token in ["histgradientboosting", "tcnae", "state_dict", "joblib", "torch.nn"]:
        assert token not in body_text
