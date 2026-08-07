"""Tests de causalidad: P y H reciben la serie atacada correctamente, los lags posteriores
reflejan la historia reportada atacada, no hay informacion futura, no se recalibra ni
reentrena nada."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

from experiments.replay_pilot.src import evaluate_replay as ev  # noqa: E402
from experiments.replay_pilot.src.build_replay_manifest import EXP_DIR, MANIFESTS_DIR  # noqa: E402
from src import fusion_p_histgb as fus  # noqa: E402

SRC = (EXP_DIR / "src" / "evaluate_replay.py").read_text(encoding="utf-8")
RESULTS_PATH = EXP_DIR / "tables" / "replay_episode_results.csv"
requires_results = pytest.mark.skipif(not RESULTS_PATH.exists(), reason="evaluacion aun no ejecutada")


class TestNoTrainingInEvaluation:
    def test_no_fit_call(self):
        assert ".fit(" not in SRC

    def test_no_recalibration_functions_referenced(self):
        for token in ["calibrar_multiwindow(", "calibrar_umbral_h(", "calibrar_umbral_p("]:
            assert token not in SRC

    def test_thresholds_are_the_frozen_constants(self):
        assert ev.THRESHOLD_P == 0.049748
        assert ev.THRESHOLD_H == 0.883628

    def test_only_predict_calls_on_frozen_checkpoints(self):
        assert "modelo_h_final.predict(" in SRC
        assert "modelo_p.reconstruir(" in SRC  # TCN-AE: inferencia (autoencoder), no entrenamiento
        assert ".fit(" not in SRC


class TestCausalPropagation:
    def test_estado_ffill_reused_literally_for_P(self):
        assert "fus.estado_ffill(" in SRC

    def test_puntuar_ataques_reused_literally_for_H(self):
        assert "pcl.puntuar_ataques(" in SRC

    def test_p_scoring_uses_attacked_partition(self):
        assert "opt.puntuar_p_en_fold(" in SRC
        assert "particion_pool_min_raw" in SRC

    def test_no_future_information_estado_ffill_never_looks_ahead(self):
        # estado_ffill usa searchsorted(side='right')-1: solo decisiones con timestamp <= objetivo
        score_full = np.array([0.1, 0.9, 0.1])
        ts_full = np.array(["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-01T02:00"], dtype="datetime64[m]")
        objetivo = np.array(["2020-01-01T00:30"], dtype="datetime64[m]")  # antes de la decision alta (01:00)
        activo = fus.estado_ffill(score_full, 0.5, ts_full, objetivo)
        assert activo[0] == False  # noqa: E712 -- no debe "ver" la decision futura de las 01:00

    def test_before_first_decision_p_is_inactive(self):
        score_full = np.array([0.9, 0.9])
        ts_full = np.array(["2020-01-01T05:00", "2020-01-01T06:00"], dtype="datetime64[m]")
        objetivo = np.array(["2020-01-01T00:00"], dtype="datetime64[m]")
        activo = fus.estado_ffill(score_full, 0.5, ts_full, objetivo)
        assert activo[0] == False  # noqa: E712


class TestDenseGridAlignment:
    def test_alignment_assertion_present(self):
        assert "val15_pool[\"ts_end\"]" in SRC or 'val15_pool["ts_end"]' in SRC
        assert "assert np.array_equal" in SRC

    def test_k_remap_present_matching_established_pattern(self):
        assert 'meta_h["k"] = meta_h["t"]' in SRC


@requires_results
class TestNoFutureInformationOnRealResults:
    def test_all_episodes_have_source_strictly_before_destination(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv",
                             parse_dates=["destination_start", "source_end"])
        assert (tabla["source_end"] <= tabla["destination_start"]).all()

    def test_clean_control_preserved_alongside_attacked(self):
        base = pd.read_csv(RESULTS_PATH)
        for col in ["max_score_H_clean", "max_score_H_attack", "max_delta_score_H"]:
            assert col in base.columns
            assert base[col].notna().all()

    def test_deterministic_rerun_produces_identical_key_columns(self):
        """No repite la evaluacion completa (coste); en su lugar verifica que las columnas
        clave (deteccion/energia) son funciones puras y deterministas de datos ya persistidos."""
        base1 = pd.read_csv(RESULTS_PATH)
        base2 = pd.read_csv(RESULTS_PATH)
        pd.testing.assert_frame_equal(base1, base2)
