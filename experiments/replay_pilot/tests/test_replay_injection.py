"""Tests de la inyeccion replay: copia exacta, sin escalado/suavizado/interpolacion/ruido,
independencia entre episodios."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

from experiments.replay_pilot.src import inject_replay as inj  # noqa: E402
from experiments.replay_pilot.src.build_replay_manifest import EXP_DIR  # noqa: E402

SRC = (EXP_DIR / "src" / "inject_replay.py").read_text(encoding="utf-8")


def _serie_sintetica(n_dias=10, seed=0):
    idx = pd.date_range("2020-01-01", periods=n_dias * 1440, freq="1min")
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"Global_active_power": rng.uniform(0, 2, len(idx))}, index=idx)


def _manifest_sintetico(serie, dur=120, shift_min=1440):
    dest_start = serie.index[0] + pd.Timedelta(days=8)
    source_start = dest_start - pd.Timedelta(minutes=shift_min)
    return pd.DataFrame([{
        "episode_id": "T1", "paired_destination_id": "T1_pair", "shift_type": "DAILY" if shift_min == 1440 else "WEEKLY",
        "shift_minutes": shift_min, "duration_minutes": dur,
        "destination_start": dest_start, "destination_end": dest_start + pd.Timedelta(minutes=dur),
        "source_start": source_start, "source_end": source_start + pd.Timedelta(minutes=dur),
    }])


class TestNoTransformations:
    def test_no_scaling_operations_in_source(self):
        for token in [" * escala", "scale_factor", "* rho", "* factor"]:
            assert token not in SRC

    def test_no_smoothing_or_interpolation_in_source(self):
        for token in ["interpolate(", "smooth", "savgol", "gaussian_filter", "rolling("]:
            assert token not in SRC

    def test_no_noise_addition_in_source(self):
        for token in ["np.random.normal", "rng.normal", "+ ruido", "+ noise"]:
            assert token not in SRC

    def test_no_energy_or_score_based_source_selection_in_source(self):
        for token in ["score_P", "score_H", "hidden_energy_kwh >", "seleccionar_origen_similar"]:
            assert token not in SRC


class TestExactCopy:
    def test_replay_values_equal_source_values_bit_for_bit(self):
        serie = _serie_sintetica()
        manifest = _manifest_sintetico(serie)
        pool_ini = serie.index[0] + pd.Timedelta(days=8)
        ataques = inj.construir_ataques_replay(manifest, serie, pool_ini)
        a = ataques["T1"]
        origen_esperado = serie.loc[manifest.iloc[0]["source_start"]:manifest.iloc[0]["source_end"] - pd.Timedelta(minutes=1),
                                     "Global_active_power"].to_numpy()
        assert np.array_equal(a["y_tramo"], origen_esperado)

    def test_replay_length_matches_duration(self):
        serie = _serie_sintetica()
        manifest = _manifest_sintetico(serie, dur=360)
        pool_ini = serie.index[0] + pd.Timedelta(days=8)
        ataques = inj.construir_ataques_replay(manifest, serie, pool_ini)
        assert len(ataques["T1"]["y_tramo"]) == 360

    def test_verificar_copia_exacta_reports_zero_mismatches(self):
        serie = _serie_sintetica()
        manifest = _manifest_sintetico(serie)
        pool_ini = serie.index[0] + pd.Timedelta(days=8)
        ataques = inj.construir_ataques_replay(manifest, serie, pool_ini)
        tabla = inj.verificar_copia_exacta(ataques, serie)
        assert (tabla["n_exact_copy_mismatches"] == 0).all()

    def test_verificar_copia_exacta_detects_tampering(self):
        serie = _serie_sintetica()
        manifest = _manifest_sintetico(serie)
        pool_ini = serie.index[0] + pd.Timedelta(days=8)
        ataques = inj.construir_ataques_replay(manifest, serie, pool_ini)
        ataques["T1"]["y_tramo"] = ataques["T1"]["y_tramo"] + 1.0  # simula manipulacion
        tabla = inj.verificar_copia_exacta(ataques, serie)
        assert (tabla["n_exact_copy_mismatches"] > 0).all()

    def test_attack_dict_offset_relative_to_pool_start(self):
        serie = _serie_sintetica()
        pool_ini = serie.index[0] + pd.Timedelta(days=8)
        manifest = _manifest_sintetico(serie)
        ataques = inj.construir_ataques_replay(manifest, serie, pool_ini)
        expected_offset = int((manifest.iloc[0]["destination_start"] - pool_ini) / pd.Timedelta(minutes=1))
        assert ataques["T1"]["offset_global"] == expected_offset

    def test_family_labeled_replay(self):
        serie = _serie_sintetica()
        pool_ini = serie.index[0] + pd.Timedelta(days=8)
        manifest = _manifest_sintetico(serie)
        ataques = inj.construir_ataques_replay(manifest, serie, pool_ini)
        assert ataques["T1"]["family"] == "replay"


class TestIndependentInjection:
    def test_each_episode_built_from_independent_dict_entry(self):
        serie = _serie_sintetica()
        pool_ini = serie.index[0] + pd.Timedelta(days=8)
        m1 = _manifest_sintetico(serie, dur=120, shift_min=1440)
        m2 = _manifest_sintetico(serie, dur=120, shift_min=10080)
        m2["episode_id"] = "T2"
        manifest = pd.concat([m1, m2], ignore_index=True)
        ataques = inj.construir_ataques_replay(manifest, serie, pool_ini)
        assert set(ataques.keys()) == {"T1", "T2"}
        assert not np.shares_memory(ataques["T1"]["y_tramo"], ataques["T2"]["y_tramo"])

    def test_attack_dict_does_not_mutate_source_series(self):
        serie = _serie_sintetica()
        pool_ini = serie.index[0] + pd.Timedelta(days=8)
        manifest = _manifest_sintetico(serie)
        original = serie["Global_active_power"].copy()
        inj.construir_ataques_replay(manifest, serie, pool_ini)
        assert serie["Global_active_power"].equals(original)
