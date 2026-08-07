"""Tests 1-6, 60-61, 69-70: sin fit, umbrales congelados, hashes, sin
dependencia de FastAPI/Docker, resultados solo dentro de edge_deployment/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edge_deployment.core import model_loader as ml

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
SIMULATORS_DIR = Path(__file__).resolve().parents[1] / "simulators"
BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
ALL_SRC_FILES = (list(CORE_DIR.glob("*.py")) + list(SIMULATORS_DIR.glob("*.py"))
                 + [f for f in BENCHMARKS_DIR.glob("*.py") if not f.name.startswith("benchmark_api")])
# benchmark_api_*.py (Fase 2, results/reports/api_final_report.md) importa FastAPI/httpx
# legitimamente -- fuera del alcance de estos tests, que auditan el motor de Fase 1
# (core/, simulators/, y los benchmarks de Fase 1 en esta misma carpeta compartida).


def test_1_no_fit_called_anywhere_in_engine_source():
    """Fase 4: `core/freeze_params_norm.py` menciona '.fit()' en prosa (documentando que
    nunca se llama, con parentesis vacios) -- una llamada real siempre pasa argumentos
    (`modelo.fit(X, y)`), nunca `.fit()` vacio; ese es el patron que distingue mencion de
    llamada aqui (mismo criterio que edge_deployment/tests/test_phase4_autonomous.py)."""
    for f in ALL_SRC_FILES:
        for line in f.read_text(encoding="utf-8").splitlines():
            sin_espacios = line.replace(" ", "")
            assert ".fit(" not in sin_espacios or ".fit()" in sin_espacios, \
                f"{f.name} contiene una posible llamada real a .fit(): {line!r}"


def test_2_no_recalibration_of_thresholds():
    for f in ALL_SRC_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "calibrar_umbral" not in text and "recalibrar" not in text


def test_3_checkpoints_not_modified_by_loading_twice():
    art1 = ml.load_frozen_artifacts()
    art2 = ml.load_frozen_artifacts()
    rng = np.random.default_rng(0)
    X = rng.random((10, len(art1["features_h_orden"])))
    p1, p2 = art1["modelo_h"].predict(X), art2["modelo_h"].predict(X)
    assert np.array_equal(p1, p2), "el modelo H no es determinista entre cargas -- posible modificacion del checkpoint"


def test_4_model_h_hash_stable_across_two_loads():
    h1 = ml._sha256(ml.H_MODEL_PATH)
    ml.load_frozen_artifacts()
    h2 = ml._sha256(ml.H_MODEL_PATH)
    assert h1 == h2


def test_5_thresholds_are_the_frozen_ones():
    art = ml.load_frozen_artifacts()
    with open(ml.H_THRESHOLD_PATH, encoding="utf-8") as f:
        expected_h = json.load(f)["umbral_h"]
    with open(ml.P_THRESHOLD_PATH, encoding="utf-8") as f:
        expected_p = json.load(f)["umbral_p"]
    assert art["threshold_h"] == pytest.approx(expected_h)
    assert art["threshold_p"] == pytest.approx(expected_p)


def test_6_score_direction_matches_offline_convention():
    """score = max(0, (pred-real)/max(|pred|,eps)) -- mayor score cuando el consumo reportado
    es menor que el predicho (igual que pcl.calcular_score, base para H y para P)."""
    from src import predictor_causal_lags as pcl
    pred = np.array([1.0, 1.0])
    real_bajo, real_alto = np.array([0.5, 0.5]), np.array([1.5, 1.5])
    _, score_bajo = pcl.calcular_score(pred, real_bajo)
    _, score_alto = pcl.calcular_score(pred, real_alto)
    assert score_bajo[0] > score_alto[0]


def test_7_only_pretest_period_used_in_engine_code():
    for f in ALL_SRC_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "particiones[\"test\"]" not in text and "particion_test" not in text


def test_8_test_partition_reading_function_never_referenced():
    """`evaluacion_final_retrospectiva_test.construir_ataques_desde_manifiesto` es la unica
    funcion del repo que lee la particion de test -- el motor online no debe referenciarla."""
    for f in ALL_SRC_FILES:
        text = f.read_text(encoding="utf-8")
        assert "construir_ataques_desde_manifiesto" not in text


def test_60_hashes_recorded_in_manifest():
    manifiesto = ml.build_artifact_manifest()
    assert "hashes" in manifiesto and len(manifiesto["hashes"]) >= 4
    assert all(isinstance(v, str) and len(v) == 64 for v in manifiesto["hashes"].values())


def test_61_results_only_under_edge_deployment():
    assert "edge_deployment" in str(ml.INVENTORY_PATH)
    assert "edge_deployment" in str(ml.MANIFEST_PATH)


def test_62_63_64_refit_replay_pilot_threshold_tradeoff_not_imported():
    """edge_deployment/ no importa ni lee archivos de refit/, experiments/replay_pilot/ ni
    experiments/threshold_tradeoff/ -- a diferencia de replay_pilot (que si referencia
    threshold_tradeoff de solo lectura), este motor es completamente independiente de esos
    experimentos. Se permite mencionarlos en comentarios/docstrings (p.ej. para citar de
    donde viene un patron ya validado), pero nunca importarlos ni abrir sus archivos."""
    for f in ALL_SRC_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "from experiments" not in text and "import experiments" not in text
        assert "from refit" not in text and "import refit" not in text
        assert "experiments/replay_pilot/" not in text.replace("\\", "/")
        assert "experiments/threshold_tradeoff/" not in text.replace("\\", "/")


def test_69_no_fastapi_import():
    for f in ALL_SRC_FILES:
        text = f.read_text(encoding="utf-8")
        assert "import fastapi" not in text.lower() and "from fastapi" not in text.lower()


def test_70_no_docker_dependency():
    for f in ALL_SRC_FILES:
        text = f.read_text(encoding="utf-8")
        assert "import docker" not in text.lower() and "dockerfile" not in text.lower()


def test_missing_artifact_raises_clear_error(tmp_path, monkeypatch):
    fake_path = tmp_path / "no_existe.joblib"
    with pytest.raises(ml.MissingArtifactError) as exc:
        ml._require(fake_path, "artefacto de prueba", "test_missing_artifact_raises_clear_error")
    msg = str(exc.value)
    assert "no_existe.joblib" in msg and "artefacto de prueba" in msg and "test_missing_artifact_raises_clear_error" in msg


def test_inventory_all_required_artifacts_found():
    df = ml.build_input_artifacts_inventory()
    faltantes = df[df["required"] & ~df["found"]]
    assert len(faltantes) == 0, f"faltan artefactos requeridos: {faltantes['artifact_name'].tolist()}"
