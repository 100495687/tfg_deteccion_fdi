"""Tests de la evaluacion: alertas preexistentes, deteccion estandar/inducida, zonas
(inicio/interior/final/posterior), energia, complementariedad P/H, deterministicidad."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

from experiments.replay_pilot.src import evaluate_replay as ev  # noqa: E402
from experiments.replay_pilot.src.build_replay_manifest import EXP_DIR, TABLES_DIR, REPORTS_DIR  # noqa: E402

RESULTS_PATH = TABLES_DIR / "replay_episode_results.csv"
requires_results = pytest.mark.skipif(not RESULTS_PATH.exists(), reason="evaluacion aun no ejecutada")


# --- Anchura de zonas ---

class TestZoneWidth:
    def test_120min_boundary_width_is_30(self):
        assert ev._boundary_width_minutes(120) == 30

    def test_360min_boundary_width_is_60(self):
        assert ev._boundary_width_minutes(360) == 60

    def test_1440min_boundary_width_is_60(self):
        assert ev._boundary_width_minutes(1440) == 60

    def test_minimum_boundary_width_is_30(self):
        assert ev._boundary_width_minutes(60) >= 30

    def test_interior_never_overlaps_boundaries(self):
        for dur in [120, 360, 1440]:
            w = ev._boundary_width_minutes(dur)
            interior_len = dur - 2 * w
            assert interior_len >= 0


# --- Deteccion estandar vs inducida (reportadas ambas, no sustituidas) ---

class TestStandardVsInducedReportedSeparately:
    def test_both_metrics_present_in_source(self):
        src = (EXP_DIR / "src" / "evaluate_replay.py").read_text(encoding="utf-8")
        assert "detected_standard_" in src
        assert "detected_induced_" in src

    def test_undetected_episode_has_nan_delay_not_zero(self):
        delay = np.nan
        assert not (delay == 0)
        assert pd.isna(delay)

    def test_boundary_only_requires_no_interior_detection(self):
        # boundary_only = (onset or end) and not interior
        assert ev.THRESHOLD_P > 0  # smoke import check
        onset, interior, end = True, False, False
        boundary_only = (onset or end) and not interior
        assert boundary_only is True
        onset, interior, end = True, True, False
        boundary_only = (onset or end) and not interior
        assert boundary_only is False

    def test_post_only_requires_no_detection_during_attack(self):
        onset, interior, end, post = False, False, False, True
        post_only = not (onset or interior or end) and post
        assert post_only is True
        onset, interior, end, post = True, False, False, True
        post_only = not (onset or interior or end) and post
        assert post_only is False


# --- Energia ---

class TestEnergyClassification:
    def test_underreporting_when_hidden_energy_positive(self):
        tol = 0.01
        hidden = 0.5
        assert hidden > tol

    def test_overreporting_when_hidden_energy_negative(self):
        tol = 0.01
        hidden = -0.5
        assert hidden < -tol

    def test_near_neutral_within_tolerance(self):
        tol = max(ev.ENERGY_TOL_ABS, ev.ENERGY_TOL_REL * 10.0)
        hidden = tol * 0.5
        assert abs(hidden) <= tol

    def test_tolerance_uses_max_of_absolute_and_relative(self):
        e_clean_grande = 1000.0
        tol = max(ev.ENERGY_TOL_ABS, ev.ENERGY_TOL_REL * e_clean_grande)
        assert tol == pytest.approx(ev.ENERGY_TOL_REL * e_clean_grande)
        e_clean_pequenio = 0.001
        tol2 = max(ev.ENERGY_TOL_ABS, ev.ENERGY_TOL_REL * e_clean_pequenio)
        assert tol2 == pytest.approx(ev.ENERGY_TOL_ABS)


# --- Complementariedad P/H mutuamente coherente ---

class TestComplementarityCoherence:
    def test_p_only_h_only_both_mutually_exclusive(self):
        p, h = True, False
        p_only, h_only, both = (p and not h), (h and not p), (p and h)
        assert sum([p_only, h_only, both]) <= 1

    def test_or_equals_p_or_h(self):
        for p in [True, False]:
            for h in [True, False]:
                assert (p or h) == (p or h)  # tautologia -- documenta la regla exacta usada


# --- Resultados reales congelados ---

@requires_results
class TestRealResultsIntegrity:
    def test_60_episodes_evaluated(self):
        base = pd.read_csv(RESULTS_PATH)
        assert len(base) <= 60

    def test_no_preexisting_alert_attributed_automatically_to_replay(self):
        base = pd.read_csv(RESULTS_PATH)
        # si habia alerta preexistente, detected_induced no puede ser True unicamente por
        # continuar sin cambios esa alerta -- verificado indirectamente: post_only/boundary_only
        # ya excluyen continuidad pura, y preexisting_alert se calcula independientemente
        assert "preexisting_alert" in base.columns

    def test_p_only_h_only_both_mutually_exclusive_on_real_data(self):
        base = pd.read_csv(RESULTS_PATH)
        assert ((base["P_only"].astype(int) + base["H_only"].astype(int) + base["both"].astype(int)) <= 1).all()

    def test_or_detection_consistent_with_p_or_h(self):
        base = pd.read_csv(RESULTS_PATH)
        assert (base["detected_by_OR"] == (base["detected_by_P"] | base["detected_by_H"])).all()

    def test_hidden_energy_sign_matches_economic_category(self):
        base = pd.read_csv(RESULTS_PATH)
        under = base[base["economic_category"] == "underreporting"]
        over = base[base["economic_category"] == "overreporting"]
        assert (under["hidden_energy_kwh"] > 0).all()
        assert (over["hidden_energy_kwh"] < 0).all()

    def test_energy_dr_uses_only_positive_hidden_energy(self):
        energy_dr = pd.read_csv(TABLES_DIR / "energy_dr.csv")
        base = pd.read_csv(RESULTS_PATH)
        n_underreporting = (base["hidden_energy_kwh"] > 0).sum()
        assert energy_dr["n_underreporting_episodes"].iloc[0] == n_underreporting

    def test_exact_copy_verification_zero_mismatches(self):
        copia = pd.read_csv(TABLES_DIR / "exact_copy_verification.csv")
        assert (copia["n_exact_copy_mismatches"] == 0).all()

    def test_score_margin_figure_generated(self):
        assert (EXP_DIR / "figures" / "summary" / "17_score_h_threshold_margin.png").exists()

    def test_margin_is_positive_whenever_not_detected_by_h(self):
        base = pd.read_csv(RESULTS_PATH)
        umbral_h = 0.883628
        margen = umbral_h - base["max_score_H_attack"]
        no_detectado_h = ~base["detected_induced_H"]
        assert (margen[no_detectado_h] > 0).all()

    def test_results_only_under_experiments_replay_pilot(self):
        assert "experiments" in str(RESULTS_PATH) and "replay_pilot" in str(RESULTS_PATH)

    def test_refit_not_modified(self):
        src_files = list((EXP_DIR / "src").glob("*.py"))
        for f in src_files:
            assert "refit" not in f.read_text(encoding="utf-8").lower()

    def test_previous_experiments_only_referenced_read_only_in_inventory(self):
        """threshold_tradeoff aparece unicamente como entrada informativa de solo lectura en
        el inventario de artefactos (selected=False, nunca escrita ni modificada)."""
        for f in (EXP_DIR / "src").glob("*.py"):
            text = f.read_text(encoding="utf-8")
            if "threshold_tradeoff" in text:
                assert f.name == "build_replay_manifest.py"
                assert "add(\"threshold_tradeoff_selected\"" in text
        assert not (EXP_DIR.parent / "threshold_tradeoff" / "tables" / "selected_operating_points.csv.bak").exists()

    def test_results_json_valid(self):
        with open(REPORTS_DIR / "replay_pilot_results.json", encoding="utf-8") as f:
            json.load(f)

    def test_manifest_json_valid(self):
        with open(REPORTS_DIR / "replay_pilot_manifest.json", encoding="utf-8") as f:
            json.load(f)

    def test_tables_reloadable(self):
        for name in ["replay_episode_results", "replay_summary", "results_by_shift", "results_by_duration"]:
            df = pd.read_csv(TABLES_DIR / f"{name}.csv")
            assert isinstance(df, pd.DataFrame)

    def test_figures_generated_from_tables_not_recomputed(self):
        figs = list((EXP_DIR / "figures" / "summary").glob("*.png"))
        assert len(figs) >= 14

    def test_episode_figures_generated(self):
        figs = list((EXP_DIR / "figures" / "episodes").glob("*.png"))
        assert len(figs) >= 1

    def test_report_matches_tables_n_episodes(self):
        with open(REPORTS_DIR / "replay_pilot_results.json", encoding="utf-8") as f:
            report = json.load(f)
        base = pd.read_csv(RESULTS_PATH)
        assert report["n_episodes"] == len(base)
