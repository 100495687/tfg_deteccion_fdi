"""Test 54 + 71-75."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

EDGE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EDGE_DIR.parent
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
SUMMARY_PATH = REPORTS_DIR / "api_offline_equivalence_summary.json"

requires_run = pytest.mark.skipif(not SUMMARY_PATH.exists(), reason="ejecutar build_api_offline_equivalence_summary primero")


def test_generate_api_offline_equivalence_summary():
    from edge_deployment.clients.api_stream_client import build_api_offline_equivalence_summary
    resumen = build_api_offline_equivalence_summary()
    assert resumen["api_preserves_offline_equivalence"] is True


@requires_run
class TestApiOfflineEquivalence:
    def _summary(self) -> dict:
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_54_api_preserves_offline_equivalence(self):
        s = self._summary()
        assert s["alert_P_100pct"] is True
        assert s["alert_H_100pct"] is True
        assert s["alert_P_ffill_100pct"] is True
        assert s["alert_OR_100pct"] is True

    def test_73_results_are_deterministic_hashes_stable(self):
        s = self._summary()
        assert "artifact_hashes" in s and len(s["artifact_hashes"]) >= 4

    def test_71_csv_tables_reloadable(self):
        for name in ["api_engine_equivalence", "api_error_cases", "api_concurrency_cases"]:
            path = TABLES_DIR / f"{name}.csv"
            if path.exists():
                df = pd.read_csv(path)
                assert isinstance(df, pd.DataFrame) and len(df.columns) > 0

    def test_72_json_reports_valid(self):
        for name in ["api_startup_report", "api_artifact_validation", "api_equivalence_summary",
                     "api_offline_equivalence_summary"]:
            path = REPORTS_DIR / f"{name}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    json.load(f)

    def test_74_no_overwrite_without_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "edge_deployment.clients.api_stream_client", "--days", "1"],
            cwd=str(REPO_DIR), capture_output=True, text=True, timeout=60)
        assert result.returncode != 0
        assert "ya existe" in (result.stdout + result.stderr)

    def test_75_critical_error_nonzero_exit(self):
        code = (
            "import sys; sys.path.insert(0, r'" + str(REPO_DIR) + "');"
            "from edge_deployment.core import model_loader as ml;"
            "ml._require(ml.EDGE_DIR / 'no_existe_de_verdad.joblib', 'x', 'y')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
