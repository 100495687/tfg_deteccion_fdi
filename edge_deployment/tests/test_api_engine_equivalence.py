"""Tests 45-53, sobre la tabla ya persistida por
`clients/api_stream_client.py` (no se recalcula aqui)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

EDGE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
SUMMARY_PATH = REPORTS_DIR / "api_equivalence_summary.json"

requires_run = pytest.mark.skipif(not SUMMARY_PATH.exists(), reason="ejecutar clients/api_stream_client.py primero")


@requires_run
class TestApiEngineEquivalence:
    def _summary(self) -> dict:
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_45_same_readings_accepted(self):
        assert self._summary()["match_pct"]["accepted"] == 100.0

    def test_46_same_readings_rejected(self):
        assert self._summary()["match_pct"]["rejection_reason"] == 100.0

    def test_47_same_scores_p(self):
        s = self._summary()
        assert s["score_p_exactly_equal_pct"] == 100.0
        assert s["score_p_max_abs_diff"] == 0.0

    def test_48_same_scores_h(self):
        s = self._summary()
        assert s["score_h_exactly_equal_pct"] == 100.0
        assert s["score_h_max_abs_diff"] == 0.0

    def test_49_same_alerts_p(self):
        assert self._summary()["match_pct"]["alert_p"] == 100.0

    def test_50_same_alerts_h(self):
        assert self._summary()["match_pct"]["alert_h"] == 100.0

    def test_51_same_p_propagated(self):
        assert self._summary()["match_pct"]["alert_p_ffill"] == 100.0

    def test_52_same_or(self):
        assert self._summary()["match_pct"]["alert_or"] == 100.0

    def test_53_same_buffers(self):
        s = self._summary()
        assert s["match_pct"]["buffer_1min_size"] == 100.0
        assert s["match_pct"]["history_15min_size"] == 100.0

    def test_all_categorical_match_true(self):
        assert self._summary()["all_categorical_100pct"] is True

    def test_http_status_matches_accepted_semantics(self):
        assert self._summary()["http_status_match_pct"] == 100.0

    def test_table_reloadable(self):
        df = pd.read_csv(TABLES_DIR / "api_engine_equivalence.csv")
        assert len(df) == self._summary()["n_readings"]
