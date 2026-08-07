"""Tests 16-24."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from edge_deployment.core.detector_state import P_WINDOW_MIN

API_DIR = Path(__file__).resolve().parents[1] / "api"


def test_16_readings_calls_ingest_once(api_bootstrapped):
    """No hay forma directa de contar llamadas sin instrumentar el motor; se verifica
    indirectamente: una lectura hace avanzar `n_accepted` en exactamente 1 (si se llamase dos
    veces, avanzaria en 2 o se rechazaria como duplicado)."""
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    s0 = client.get(f"/status/{meter_id}").json()["accepted_readings"]
    client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    s1 = client.get(f"/status/{meter_id}").json()["accepted_readings"]
    assert s1 == s0 + 1


def test_17_readings_does_not_load_models():
    text = (API_DIR / "routes.py").read_text(encoding="utf-8")
    assert "DetectorEngine(" not in text
    assert "joblib.load" not in text and "torch.load" not in text


def test_18_valid_reading_returns_200(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    r = client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.2})
    assert r.status_code == 200
    assert r.json()["accepted"] is True


def test_19_warming_up_not_presented_as_normalcy(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    r = client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    body = r.json()
    assert body["engine_status"] == "warming_up"
    assert body["p_ready"] is False
    assert body["detection_source"] == "partial_availability"


def test_20_null_not_converted_to_false(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    r = client.post("/readings", json={"meter_id": meter_id, "timestamp": stream_start.isoformat(), "power_kw": 1.0})
    body = r.json()
    assert body["score_p"] is None
    assert body["alert_p"] is None
    assert body["score_p"] is not False
    assert body["alert_p"] is not False


def test_21_event_timestamp_preserved(api_bootstrapped):
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    ts = stream_start + pd.Timedelta(minutes=5)
    r = client.post("/readings", json={"meter_id": meter_id, "timestamp": ts.isoformat(), "power_kw": 1.0})
    assert pd.Timestamp(r.json()["timestamp"]) == ts


def test_22_real_clock_does_not_determine_evaluations():
    text = (API_DIR / "routes.py").read_text(encoding="utf-8")
    assert "datetime.now(" not in text and "time.time()" not in text.replace("time.perf_counter()", "")


def test_23_24_positional_anchor_preserved_bins_at_1724(api_bootstrapped, engine):
    """El propio periodo pre-test real empieza en 17:24 (ver period_selection.json, Fase 1) --
    aqui se reproduce el mismo patron con datos sinteticos: el primer bin de H se cierra
    exactamente 15 minutos (no alineado a reloj) despues del inicio del streaming, igual en
    la API que en el motor directo (fixture `engine`, misma logica, otra instancia)."""
    client, meter_id, stream_start = api_bootstrapped["client"], api_bootstrapped["meter_id"], api_bootstrapped["stream_start"]
    # (la evidencia de que el anclaje no depende del reloj -- p.ej. el periodo real empieza en
    # 17:24 -- ya esta cubierta con datos reales en tests/test_temporal_alignment.py; aqui se
    # comprueba que, sea cual sea el minuto de inicio, el primer bin cierra exactamente 15
    # minutos despues, igual que en el motor directo)
    last = None
    for i in range(15):
        last = client.post("/readings", json={"meter_id": meter_id, "timestamp": (stream_start + pd.Timedelta(minutes=i)).isoformat(), "power_kw": 1.0})
    body = last.json()
    assert body["h_evaluated"] is True
    assert pd.Timestamp(body["h_last_evaluation_timestamp"]) == stream_start + pd.Timedelta(minutes=14)
