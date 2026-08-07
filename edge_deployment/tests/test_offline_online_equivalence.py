"""Tests 50-59, 65-68: sobre los resultados ya persistidos de la ejecucion de
equivalencia (no se recalculan aqui)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

EDGE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
SUMMARY_PATH = REPORTS_DIR / "equivalence_summary.json"

requires_run = pytest.mark.skipif(not SUMMARY_PATH.exists(), reason="ejecutar replay_clean_stream.py primero")


@requires_run
class TestEquivalenceResults:
    def _load(self):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_50_p_scores_equivalent_within_tolerance_or_documented(self):
        d = self._load()
        # score_p no es bit-identico (ruido de punto flotante dependiente del tamano de lote
        # en el forward pass del TCN-AE, ver reports/final_report.md) -- se exige una
        # correlacion casi perfecta y una diferencia maxima varios ordenes de magnitud por
        # debajo del espaciado entre umbrales relevantes
        assert d["metrics"]["score_p"]["correlation"] > 0.999999
        assert d["metrics"]["score_p"]["max_abs_diff"] < 1e-5

    def test_51_h_scores_bit_identical(self):
        d = self._load()
        assert d["metrics"]["score_h"]["mae"] == 0.0
        assert d["metrics"]["score_h"]["max_abs_diff"] == 0.0

    def test_52_alertas_p_coinciden_100pct(self):
        assert self._load()["metrics"]["alerts"]["match_pct_P"] == 100.0

    def test_53_alertas_h_coinciden_100pct(self):
        assert self._load()["metrics"]["alerts"]["match_pct_H"] == 100.0

    def test_54_p_propagada_coincide_100pct(self):
        assert self._load()["metrics"]["alerts"]["match_pct_P_ffill"] == 100.0

    def test_55_or_coincide_100pct(self):
        assert self._load()["metrics"]["alerts"]["match_pct_OR"] == 100.0

    def test_49_timestamps_evaluacion_coinciden_sin_faltantes_ni_extra(self):
        t = self._load()["metrics"]["timestamps"]
        assert t["n_h_missing_online"] == 0 and t["n_h_extra_online"] == 0
        assert t["n_p_missing_online"] == 0 and t["n_p_extra_online"] == 0

    def test_28_agregacion_15min_bit_identica(self):
        assert self._load()["metrics"]["aggregation"]["pct_exactly_equal"] == 100.0

    def test_criterio_principal_cumplido(self):
        d = self._load()
        assert d["criterio_principal_cumplido"] is True
        assert d["status"] == "EQUIVALENT"

    def test_65_no_overwrite_without_flag(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "edge_deployment.simulators.replay_clean_stream", "--max-days", "1"],
            cwd=str(EDGE_DIR.parent), capture_output=True, text=True, timeout=60)
        assert result.returncode != 0
        assert "ya existe" in (result.stdout + result.stderr)

    def test_66_csv_tables_reloadable(self):
        for name in ["offline_online_equivalence", "p_online_decisions", "h_online_decisions",
                     "simulation_latency", "h_offline_reference", "p_offline_reference",
                     "input_artifacts_inventory"]:
            df = pd.read_csv(TABLES_DIR / f"{name}.csv")
            assert isinstance(df, pd.DataFrame) and len(df.columns) > 0

    def test_67_json_reports_valid(self):
        for name in ["equivalence_summary", "resampling_semantics", "period_selection", "bootstrap_report"]:
            path = REPORTS_DIR / f"{name}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    json.load(f)  # no debe lanzar

    def test_68_missing_manifest_causes_nonzero_exit(self):
        import subprocess
        import sys
        code = (
            "import sys; sys.path.insert(0, r'" + str(EDGE_DIR.parent) + "');"
            "from edge_deployment.core import model_loader as ml;"
            "ml._require(ml.EDGE_DIR / 'no_existe.joblib', 'x', 'y')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
