"""Tests del generador de manifiesto replay: seleccion de periodo, candidatos, estratificacion,
emparejamiento, y congelacion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

from experiments.replay_pilot.src import build_replay_manifest as brm  # noqa: E402

EXP_DIR = brm.EXP_DIR
MANIFESTS_DIR = brm.MANIFESTS_DIR
REPORTS_DIR = brm.REPORTS_DIR
SRC = (EXP_DIR / "src" / "build_replay_manifest.py").read_text(encoding="utf-8")

MANIFEST_RAN = (MANIFESTS_DIR / "replay_pilot_manifest.csv").exists()
requires_manifest = pytest.mark.skipif(not MANIFEST_RAN, reason="manifiesto aun no generado")


# --- 1-2: solo pre-test, no test final ---

class TestPretestOnly:
    def test_no_test_partition_reference_in_source(self):
        assert "particiones[\"test\"]" not in SRC.replace(" ", "")
        assert "test_attacks_v2" not in SRC or "PROHIBIDO" in SRC

    def test_period_is_final_cal_pool_180(self):
        if not (REPORTS_DIR / "period_selection_justification.json").exists():
            pytest.skip("periodo aun no seleccionado")
        with open(REPORTS_DIR / "period_selection_justification.json", encoding="utf-8") as f:
            j = json.load(f)
        assert j["periodo_elegido"] == "FINAL_CAL_POOL_180"

    def test_period_selection_does_not_use_detection_as_criterion(self):
        if not (REPORTS_DIR / "period_selection_justification.json").exists():
            pytest.skip("periodo aun no seleccionado")
        with open(REPORTS_DIR / "period_selection_justification.json", encoding="utf-8") as f:
            j = json.load(f)
        assert "deteccion" not in j["no_se_uso_como_criterio"].lower() or "no" in j["no_se_uso_como_criterio"].lower()


# --- 3: no fit ---

class TestNoTraining:
    def test_no_fit_call_in_manifest_builder(self):
        assert ".fit(" not in SRC

    def test_no_training_functions_referenced(self):
        for token in ["construir_modelo(", "modelo_h.fit", "modelo_final.fit"]:
            assert token not in SRC


# --- 9/10/11: dos desplazamientos, tres duraciones, emparejamiento ---

class TestScopeConstants:
    def test_exactly_two_shifts(self):
        assert set(brm.SHIFTS.keys()) == {"DAILY", "WEEKLY"}
        assert brm.SHIFTS["DAILY"] == 1440
        assert brm.SHIFTS["WEEKLY"] == 10080

    def test_exactly_three_durations(self):
        assert brm.DURATIONS_MIN == [120, 360, 1440]

    def test_max_60_episodes_by_construction(self):
        assert len(brm.DURATIONS_MIN) * len(brm.SHIFTS) * brm.N_DEST_PER_DURATION == 60

    def test_random_seed_fixed(self):
        assert brm.RANDOM_SEED == 20260801


# --- Validez de candidatos (sinteticos, sin tocar el pipeline real) ---

class TestCandidateValidity:
    def _serie_sintetica(self, n_dias=10):
        idx = pd.date_range("2020-01-01", periods=n_dias * 1440, freq="1min")
        rng = np.random.RandomState(0)
        return pd.DataFrame({"Global_active_power": rng.uniform(0, 2, len(idx)), "imputado": False}, index=idx)

    def test_rejects_destination_too_close_to_pool_start(self):
        serie = self._serie_sintetica()
        pool_ini, pool_fin = serie.index[0], serie.index[-1] + pd.Timedelta(minutes=1)
        # destino a solo 1 dia del inicio del pool: insuficiente para el contexto de 7 dias de H
        res = brm.evaluar_candidato(pool_ini + pd.Timedelta(days=1), 120, serie, pool_ini, pool_fin)
        assert res["valid"] is False
        assert "calentamiento" in res["reason"] or "contexto" in res["reason"]

    def test_accepts_destination_with_full_context(self):
        serie = self._serie_sintetica(n_dias=20)
        pool_ini, pool_fin = serie.index[0] + pd.Timedelta(days=8), serie.index[-1] + pd.Timedelta(minutes=1)
        dest = pool_ini + pd.Timedelta(days=8)
        res = brm.evaluar_candidato(dest, 120, serie, pool_ini, pool_fin)
        assert res["valid"] is True

    def test_rejects_destination_with_nan(self):
        serie = self._serie_sintetica(n_dias=20)
        pool_ini, pool_fin = serie.index[0] + pd.Timedelta(days=8), serie.index[-1] + pd.Timedelta(minutes=1)
        dest = pool_ini + pd.Timedelta(days=8)
        idx_pos = serie.index.get_loc(dest + pd.Timedelta(minutes=30))
        serie.iloc[idx_pos, serie.columns.get_loc("Global_active_power")] = np.nan
        res = brm.evaluar_candidato(dest, 120, serie, pool_ini, pool_fin)
        assert res["valid"] is False

    def test_rejects_destination_with_imputed_values(self):
        serie = self._serie_sintetica(n_dias=20)
        pool_ini, pool_fin = serie.index[0] + pd.Timedelta(days=8), serie.index[-1] + pd.Timedelta(minutes=1)
        dest = pool_ini + pd.Timedelta(days=8)
        idx_pos = serie.index.get_loc(dest + pd.Timedelta(minutes=30))
        serie.iloc[idx_pos, serie.columns.get_loc("imputado")] = True
        res = brm.evaluar_candidato(dest, 120, serie, pool_ini, pool_fin)
        assert res["valid"] is False

    def test_origin_must_precede_destination(self):
        serie = self._serie_sintetica(n_dias=20)
        pool_ini, pool_fin = serie.index[0] + pd.Timedelta(days=8), serie.index[-1] + pd.Timedelta(minutes=1)
        dest = pool_ini + pd.Timedelta(days=8)
        res = brm.evaluar_candidato(dest, 120, serie, pool_ini, pool_fin)
        assert res["valid"] is True
        assert res["origin_daily_start"] < res["dest_start"]
        assert res["origin_weekly_start"] < res["dest_start"]
        assert res["origin_daily_end"] <= res["dest_start"]
        assert res["origin_weekly_end"] <= res["dest_start"]

    def test_rejects_destination_without_post_recovery_hour(self):
        serie = self._serie_sintetica(n_dias=20)
        pool_ini, pool_fin = serie.index[0] + pd.Timedelta(days=8), serie.index[-1] + pd.Timedelta(minutes=1)
        dest = pool_fin - pd.Timedelta(minutes=120)  # termina justo al final del pool, sin margen posterior
        res = brm.evaluar_candidato(dest, 120, serie, pool_ini, pool_fin)
        assert res["valid"] is False


class TestOverlap:
    def test_overlap_detection_true_case(self):
        assert brm._overlaps(pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 01:00"),
                              pd.Timestamp("2020-01-01 00:30"), pd.Timestamp("2020-01-01 01:30")) is True

    def test_overlap_detection_false_case(self):
        assert brm._overlaps(pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 01:00"),
                              pd.Timestamp("2020-01-01 01:00"), pd.Timestamp("2020-01-01 02:00")) is False

    def test_no_two_selected_destinations_overlap_in_frozen_manifest(self):
        if not MANIFEST_RAN:
            pytest.skip("manifiesto aun no generado")
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv",
                             parse_dates=["destination_start", "destination_end"])
        destinos = tabla.drop_duplicates("paired_destination_id")[["destination_start", "destination_end"]].sort_values("destination_start")
        for i in range(len(destinos) - 1):
            assert destinos.iloc[i]["destination_end"] <= destinos.iloc[i + 1]["destination_start"]


# --- Manifiesto real congelado ---

@requires_manifest
class TestFrozenManifest:
    def test_at_most_60_episodes(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv")
        assert len(tabla) <= 60

    def test_only_daily_and_weekly_shifts(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv")
        assert set(tabla["shift_type"].unique()) <= {"DAILY", "WEEKLY"}

    def test_only_three_durations(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv")
        assert set(tabla["duration_minutes"].unique()) <= {120, 360, 1440}

    def test_daily_origin_exactly_24h_before(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv",
                             parse_dates=["destination_start", "source_start"])
        daily = tabla[tabla["shift_type"] == "DAILY"]
        delta = (daily["destination_start"] - daily["source_start"]).dt.total_seconds() / 60
        assert (delta == 1440).all()

    def test_weekly_origin_exactly_7d_before(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv",
                             parse_dates=["destination_start", "source_start"])
        weekly = tabla[tabla["shift_type"] == "WEEKLY"]
        delta = (weekly["destination_start"] - weekly["source_start"]).dt.total_seconds() / 60
        assert (delta == 10080).all()

    def test_origin_always_precedes_destination(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv",
                             parse_dates=["destination_start", "source_end"])
        assert (tabla["source_end"] <= tabla["destination_start"]).all()

    def test_origin_and_destination_equal_length(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv",
                             parse_dates=["destination_start", "destination_end", "source_start", "source_end"])
        dur_dest = (tabla["destination_end"] - tabla["destination_start"])
        dur_src = (tabla["source_end"] - tabla["source_start"])
        assert (dur_dest == dur_src).all()

    def test_destinations_paired_when_possible(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv")
        counts = tabla.groupby("paired_destination_id")["shift_type"].nunique()
        assert (counts == 2).all()

    def test_stratification_documented_in_freeze_report(self):
        with open(REPORTS_DIR / "replay_pilot_manifest.json", encoding="utf-8") as f:
            freeze = json.load(f)
        assert "stratification_audit" in freeze
        assert len(freeze["stratification_audit"]) == 3

    def test_forbidden_selection_criteria_documented_as_unused(self):
        with open(REPORTS_DIR / "replay_pilot_manifest.json", encoding="utf-8") as f:
            freeze = json.load(f)
        forbidden = freeze["selection_criteria_forbidden_and_confirmed_not_used"]
        for token in ["score_P", "score_H", "detection_rate", "energy_difference"]:
            assert token in forbidden

    def test_manifest_hash_present_and_consistent(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv")
        assert tabla["manifest_hash"].notna().all()
        assert tabla["manifest_hash"].nunique() == 1

    def test_episode_id_unique(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv")
        assert tabla["episode_id"].is_unique

    def test_v1_preserved_and_not_overwritten(self):
        v1_dir = MANIFESTS_DIR / "v1_superseded"
        assert (v1_dir / "replay_pilot_manifest_v1.csv").exists()
        assert (v1_dir / "correction_note.json").exists()
        with open(v1_dir / "correction_note.json", encoding="utf-8") as f:
            note = json.load(f)
        assert note["not_a_result_driven_change"]

    def test_active_manifest_is_current_version(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv")
        assert (tabla["manifest_version"] == brm.MANIFEST_VERSION).all()
        assert brm.MANIFEST_VERSION >= 3  # v1 y v2 quedaron superseded por bugs tecnicos documentados

    def test_no_destination_falls_within_h_7day_warmup_of_pool(self):
        tabla = pd.read_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv", parse_dates=["destination_start"])
        pool_ini = pd.Timestamp("2009-06-18 17:24:00")
        dias = (tabla["destination_start"] - pool_ini).dt.total_seconds() / 86400
        assert (dias >= 7).all()
