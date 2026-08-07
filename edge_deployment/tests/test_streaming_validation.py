"""Tests 15-21 + Casos B/C/D/E/F/G: validacion de lecturas antes
de tocar ningun buffer."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from edge_deployment.core.schemas import Reading
from edge_deployment.core.validation import validate_reading


def test_15_valid_reading_accepted():
    r = Reading("m1", pd.Timestamp("2020-01-01 00:01:00"), 1.5)
    res = validate_reading(r, pd.Timestamp("2020-01-01 00:00:00"), 1.0)
    assert res.accepted and res.rejection_reason is None


def test_16_nan_rejected():
    r = Reading("m1", pd.Timestamp("2020-01-01 00:01:00"), float("nan"))
    res = validate_reading(r, None, None)
    assert not res.accepted and res.rejection_reason == "nan_value"


def test_17_infinite_rejected():
    r = Reading("m1", pd.Timestamp("2020-01-01 00:01:00"), float("inf"))
    res = validate_reading(r, None, None)
    assert not res.accepted and res.rejection_reason == "infinite_value"
    r2 = Reading("m1", pd.Timestamp("2020-01-01 00:01:00"), float("-inf"))
    assert not validate_reading(r2, None, None).accepted


def test_18_duplicate_identical_rejected():
    last_ts = pd.Timestamp("2020-01-01 00:05:00")
    r = Reading("m1", last_ts, 2.0)
    res = validate_reading(r, last_ts, 2.0)
    assert not res.accepted and res.rejection_reason == "duplicate_identical"


def test_19_duplicate_conflicting_rejected():
    last_ts = pd.Timestamp("2020-01-01 00:05:00")
    r = Reading("m1", last_ts, 3.5)
    res = validate_reading(r, last_ts, 2.0)
    assert not res.accepted and res.rejection_reason == "duplicate_conflicting"


def test_20_out_of_order_rejected():
    last_ts = pd.Timestamp("2020-01-01 00:05:00")
    r = Reading("m1", last_ts - pd.Timedelta(minutes=1), 2.0)
    res = validate_reading(r, last_ts, 2.0)
    assert not res.accepted and res.rejection_reason == "out_of_order"


def test_21_rejected_reading_never_reported_as_modifying_state():
    r = Reading("", pd.Timestamp("2020-01-01 00:01:00"), 1.0)
    res = validate_reading(r, None, None)
    assert not res.accepted and res.rejection_reason == "empty_meter_id"


def test_empty_meter_id_rejected():
    r = Reading("   ", pd.Timestamp("2020-01-01 00:01:00"), 1.0)
    assert not validate_reading(r, None, None).accepted


def test_non_numeric_power_rejected():
    r = Reading("m1", pd.Timestamp("2020-01-01 00:01:00"), "not_a_number")
    assert not validate_reading(r, None, None).accepted


def test_invalid_timestamp_rejected():
    r = Reading("m1", pd.NaT, 1.0)
    res = validate_reading(r, None, None)
    assert not res.accepted and res.rejection_reason == "invalid_timestamp"


def test_negative_power_policy_matches_offline_pipeline():
    """El pipeline offline (src/data_loading.py) nunca filtra potencia negativa -- solo
    interpola NaN. Para ser equivalente, este motor tampoco la rechaza (decision explicita,
    documentada en validation.py, no un olvido)."""
    from pathlib import Path
    data_loading_src = (Path(__file__).resolve().parents[2] / "src" / "data_loading.py").read_text(encoding="utf-8")
    assert "< 0" not in data_loading_src and "negativ" not in data_loading_src.lower()

    r = Reading("m1", pd.Timestamp("2020-01-01 00:01:00"), -0.5)
    res = validate_reading(r, None, None)
    assert res.accepted, "la politica debe coincidir con el pipeline offline (no rechaza negativos)"


def test_first_ever_reading_always_accepted_regardless_of_value():
    r = Reading("m1", pd.Timestamp("2020-01-01 00:01:00"), 2.0)
    assert validate_reading(r, None, None).accepted


class TestEngineLevelValidation:
    def test_rejected_reading_does_not_change_buffer_size(self, bootstrapped):
        eng, mid, stream_start = bootstrapped["engine"], bootstrapped["meter_id"], bootstrapped["stream_start"]
        eng.ingest(mid, stream_start, 1.0)
        size_before = eng.get_status(mid)["buffer_1min_size"]
        resp = eng.ingest(mid, stream_start, 999.0)  # mismo timestamp, valor distinto -> duplicate_conflicting
        assert not resp.accepted and resp.rejection_reason == "duplicate_conflicting"
        assert eng.get_status(mid)["buffer_1min_size"] == size_before

    def test_rejected_reading_does_not_increment_stream_minutes(self, bootstrapped):
        eng, mid, stream_start = bootstrapped["engine"], bootstrapped["meter_id"], bootstrapped["stream_start"]
        eng.ingest(mid, stream_start, 1.0)
        n_before = eng.get_status(mid)["n_accepted"]
        eng.ingest(mid, stream_start - pd.Timedelta(minutes=5), 1.0)  # out of order
        assert eng.get_status(mid)["n_accepted"] == n_before

    def test_nan_ingest_rejected_end_to_end(self, bootstrapped):
        eng, mid, stream_start = bootstrapped["engine"], bootstrapped["meter_id"], bootstrapped["stream_start"]
        resp = eng.ingest(mid, stream_start, float("nan"))
        assert not resp.accepted and resp.rejection_reason == "nan_value"

    def test_infinite_ingest_rejected_end_to_end(self, bootstrapped):
        eng, mid, stream_start = bootstrapped["engine"], bootstrapped["meter_id"], bootstrapped["stream_start"]
        resp = eng.ingest(mid, stream_start, float("inf"))
        assert not resp.accepted and resp.rejection_reason == "infinite_value"
