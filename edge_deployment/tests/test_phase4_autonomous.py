"""Fase 4: autonomia de artefactos y robustez del arranque edge.

Cubre: carga de params_norm_p, coincidencia de claves/shapes/dtypes con legacy, que el
autonomous_loader nunca abre los 2 CSV, que Docker no contiene ni monta datasets, comparacion
real de SHA-256 contra el manifiesto v2, degradacion correcta ante artefacto ausente/corrupto
sin crash, equivalencia de scores API/legacy, y que Docker autonomo conserva las alertas (sobre
resultados ya persistidos, mismo patron que test_offline_online_equivalence.py -- no se
recalcula la equivalencia aqui).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import joblib
import pytest

EDGE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = EDGE_DIR.parent
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
DOCKER_CONTEXT_DIR = EDGE_DIR / "docker_context"


# ==============================================================================================
# params_norm_p: carga, claves, shapes, dtypes (coincide con legacy)
# ==============================================================================================

def test_params_norm_p_artifact_can_be_loaded():
    from edge_deployment.core.model_loader import PARAMS_NORM_P_PATH
    assert PARAMS_NORM_P_PATH.exists(), f"falta el artefacto congelado: {PARAMS_NORM_P_PATH}"
    params_norm = joblib.load(PARAMS_NORM_P_PATH)
    assert isinstance(params_norm, dict)
    assert set(params_norm.keys()) == {"media", "std"}


def test_params_norm_p_metadata_documents_keys_shapes_dtypes_and_hashes():
    path = EDGE_DIR / "models" / "params_norm_p_metadata.json"
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    assert set(meta["keys"]) == {"media", "std"}
    assert meta["shapes"] == {"media": [], "std": []}
    assert "float" in meta["dtypes"]["media"].lower()
    assert len(meta["sha256_artifact"]) == 64
    assert "data_raw_csv" in meta["source_csv_hashes"] and "episodes_master_val_csv" in meta["source_csv_hashes"]
    assert meta["deterministic_confirmed_2_of_2_runs"] is True
    assert "no fit" in meta["no_fit_no_recalibration"].lower() or "nunca llama" in meta["no_fit_no_recalibration"].lower()


def test_params_norm_p_keys_shapes_dtypes_match_legacy_reconstruction():
    """Reconstruye params_norm por la ruta legacy una vez (lento, pero es exactamente el
    mismo calculo que genero el artefacto congelado -- confirma que no diverge)."""
    from edge_deployment.core import model_loader as ml
    legacy = ml.load_frozen_artifacts()
    auto = joblib.load(ml.PARAMS_NORM_P_PATH)
    assert set(legacy["params_norm_p"].keys()) == set(auto.keys())
    for k in auto:
        assert type(legacy["params_norm_p"][k]) is type(auto[k])
    assert legacy["params_norm_p"] == auto


# ==============================================================================================
# El autonomous_loader nunca abre los 2 CSV
# ==============================================================================================

def test_autonomous_loader_never_reads_data_raw_or_episodes_csv(monkeypatch):
    calls: list[str] = []
    import pandas as pd
    original_read_csv = pd.read_csv

    def spy_read_csv(*args, **kwargs):
        if args:
            calls.append(str(args[0]))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy_read_csv)

    from edge_deployment.core import model_loader as ml
    ml.load_frozen_artifacts_autonomous()

    forbidden = [c for c in calls if "data_raw.csv" in c.replace("\\", "/") or "episodes_master_val.csv" in c.replace("\\", "/")]
    assert forbidden == [], f"el autonomous_loader abrio CSV prohibidos: {forbidden}"


def test_load_frozen_artifacts_autonomous_source_never_imports_base_relative_at_module_level():
    text = (EDGE_DIR / "core" / "model_loader.py").read_text(encoding="utf-8")
    # la funcion autonoma en si (no el fichero completo, que si conserva legacy_loader)
    start = text.index("def load_frozen_artifacts_autonomous")
    end = text.index("\ndef ", start + 1) if "\ndef " in text[start + 1:] else len(text)
    body = text[start:end]
    assert "base_relative" not in body
    assert "cargar_y_limpiar" not in body


# ==============================================================================================
# artifact_manifest_v2.json: comparacion real de SHA-256
# ==============================================================================================

def test_manifest_v2_exists_and_is_valid_against_real_artifacts():
    from edge_deployment.core import manifest_v2
    assert manifest_v2.MANIFEST_V2_PATH.exists()
    r = manifest_v2.validate_manifest_v2()
    assert r["integrity_valid"] is True
    assert r["n_checked"] == len(manifest_v2.ARTIFACT_PATHS) == 8
    assert r["n_missing"] == 0 and r["n_mismatched"] == 0


def test_manifest_v2_detects_missing_artifact():
    from edge_deployment.core import manifest_v2
    real = json.loads(manifest_v2.MANIFEST_V2_PATH.read_text(encoding="utf-8"))
    tmpdir = Path(tempfile.mkdtemp())
    try:
        broken = json.loads(json.dumps(real))
        broken["artifacts"]["model_H"]["path"] = "no_existe/en_ningun_sitio.joblib"
        p = tmpdir / "manifest_missing.json"
        p.write_text(json.dumps(broken), encoding="utf-8")
        r = manifest_v2.validate_manifest_v2(manifest_path=p)
        assert r["integrity_valid"] is False
        assert "model_H" in r["missing"]
        assert r["mismatched"] == []
    finally:
        shutil.rmtree(tmpdir)


def test_manifest_v2_detects_one_byte_corruption_via_hash_before_deserializing():
    """El hash debe fallar antes de deserializar -- se corrompe una copia temporal, nunca el
    artefacto original."""
    from edge_deployment.core import manifest_v2
    real = json.loads(manifest_v2.MANIFEST_V2_PATH.read_text(encoding="utf-8"))
    tmpdir = Path(tempfile.mkdtemp())
    try:
        real_path = manifest_v2.BASE_DIR / real["artifacts"]["params_norm_P"]["path"]
        original_bytes = real_path.read_bytes()
        corrupt_copy = tmpdir / "params_norm_p_corrupt.joblib"
        data = bytearray(original_bytes)
        data[0] ^= 0xFF
        corrupt_copy.write_bytes(bytes(data))

        broken = json.loads(json.dumps(real))
        broken["artifacts"]["params_norm_P"]["path"] = str(corrupt_copy)
        p = tmpdir / "manifest_corrupt.json"
        p.write_text(json.dumps(broken), encoding="utf-8")

        r = manifest_v2.validate_manifest_v2(manifest_path=p)
        assert r["integrity_valid"] is False
        assert len(r["mismatched"]) == 1
        assert r["mismatched"][0]["artifact"] == "params_norm_P"
        assert r["mismatched"][0]["expected_sha256"] != r["mismatched"][0]["actual_sha256"]

        # el original nunca se toco
        assert real_path.read_bytes() == original_bytes
    finally:
        shutil.rmtree(tmpdir)


# ==============================================================================================
# Degradacion correcta ante fallo de integridad (API real, sin mocks del resultado)
# ==============================================================================================

def _app_with_redirected_manifest(monkeypatch, manifest_path: Path):
    from edge_deployment.api import lifecycle as lc
    from edge_deployment.core import manifest_v2

    def fake_validate_artifacts():
        import time
        t0 = time.perf_counter()
        integrity = manifest_v2.validate_manifest_v2(manifest_path=manifest_path)
        dt = (time.perf_counter() - t0) * 1000
        missing_and_mismatched = list(integrity["missing"]) + [m["artifact"] for m in integrity["mismatched"]]
        return {"hashes_valid": integrity["integrity_valid"],
                "n_missing": integrity["n_missing"] + integrity["n_mismatched"],
                "missing": missing_and_mismatched, "manifest": integrity, "validation_time_ms": dt}

    monkeypatch.setattr(lc, "validate_artifacts", fake_validate_artifacts)
    return lc


def test_missing_artifact_yields_health_200_ready_503_readings_503_no_crash(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edge_deployment.api.error_handlers import register_error_handlers
    from edge_deployment.api.routes import router
    from edge_deployment.core import manifest_v2

    real = json.loads(manifest_v2.MANIFEST_V2_PATH.read_text(encoding="utf-8"))
    tmpdir = Path(tempfile.mkdtemp())
    try:
        broken = json.loads(json.dumps(real))
        broken["artifacts"]["model_P_checkpoint"]["path"] = "no_existe.pt"
        p = tmpdir / "manifest.json"
        p.write_text(json.dumps(broken), encoding="utf-8")
        lc = _app_with_redirected_manifest(monkeypatch, p)

        app = FastAPI(lifespan=lc.lifespan)
        register_error_handlers(app)
        app.include_router(router)
        with TestClient(app) as client:
            r_health = client.get("/health")
            r_ready = client.get("/ready")
            r_reading = client.post("/readings", json={"meter_id": "m1", "timestamp": "2020-01-01T00:00:00", "power_kw": 1.0})
            assert r_health.status_code == 200
            assert r_ready.status_code == 503
            assert r_reading.status_code == 503
            for r in (r_ready, r_reading):
                body = r.json()
                assert "error_code" in body and "message" in body
                assert "Traceback" not in json.dumps(body)
    finally:
        shutil.rmtree(tmpdir)


def test_corrupted_artifact_yields_health_200_ready_503_readings_503_no_crash(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edge_deployment.api.error_handlers import register_error_handlers
    from edge_deployment.api.routes import router
    from edge_deployment.core import manifest_v2

    real = json.loads(manifest_v2.MANIFEST_V2_PATH.read_text(encoding="utf-8"))
    tmpdir = Path(tempfile.mkdtemp())
    try:
        real_path = manifest_v2.BASE_DIR / real["artifacts"]["threshold_H_json"]["path"]
        original_bytes = real_path.read_bytes()
        corrupt_copy = tmpdir / "threshold_h_corrupt.json"
        data = bytearray(original_bytes)
        data[0] ^= 0xFF
        corrupt_copy.write_bytes(bytes(data))

        broken = json.loads(json.dumps(real))
        broken["artifacts"]["threshold_H_json"]["path"] = str(corrupt_copy)
        p = tmpdir / "manifest.json"
        p.write_text(json.dumps(broken), encoding="utf-8")
        lc = _app_with_redirected_manifest(monkeypatch, p)

        app = FastAPI(lifespan=lc.lifespan)
        register_error_handlers(app)
        app.include_router(router)
        with TestClient(app) as client:
            r_health = client.get("/health")
            r_ready = client.get("/ready")
            r_reading = client.post("/readings", json={"meter_id": "m1", "timestamp": "2020-01-01T00:00:00", "power_kw": 1.0})
            assert r_health.status_code == 200
            assert r_ready.status_code == 503
            assert r_reading.status_code == 503

        assert real_path.read_bytes() == original_bytes  # original nunca modificado
    finally:
        shutil.rmtree(tmpdir)


# ==============================================================================================
# Docker no contiene ni monta datasets
# ==============================================================================================

def test_docker_context_contains_no_csv_or_dataset():
    if not DOCKER_CONTEXT_DIR.exists():
        pytest.skip("docker_context/ no generado en esta ejecucion (build_docker_context.py)")
    forbidden = list(DOCKER_CONTEXT_DIR.rglob("*.csv"))
    assert forbidden == [], f"el contexto Docker contiene CSV: {forbidden}"


def test_docker_context_includes_params_norm_p_and_manifest_v2():
    if not DOCKER_CONTEXT_DIR.exists():
        pytest.skip("docker_context/ no generado en esta ejecucion (build_docker_context.py)")
    assert (DOCKER_CONTEXT_DIR / "edge_deployment" / "models" / "params_norm_p.joblib").exists()
    assert (DOCKER_CONTEXT_DIR / "edge_deployment" / "models" / "artifact_manifest_v2.json").exists()


def test_dockerfile_run_command_never_mounts_data_or_episodes_csv():
    for fname in ("Dockerfile", "Dockerfile.baseline"):
        text = (EDGE_DIR / fname).read_text(encoding="utf-8")
        assert "data_raw.csv" not in text
        assert "episodes_master_val.csv" not in text


# ==============================================================================================
# no fit, no recalibracion, no test final (mismo patron de Fase 1, extendido a los ficheros nuevos)
# ==============================================================================================

NEW_CORE_FILES = [EDGE_DIR / "core" / "model_loader.py", EDGE_DIR / "core" / "manifest_v2.py",
                  EDGE_DIR / "core" / "freeze_params_norm.py", EDGE_DIR / "api" / "lifecycle.py"]


def test_no_fit_call_in_phase4_files():
    """Ignora menciones en prosa/comentarios/docstrings de '.fit()' (p.ej. "nunca entra en
    .fit()", "sin `.fit()`, sin recalibracion" -- documentando que no se llama, siempre con
    parentesis vacios). Una llamada real siempre pasa argumentos (`modelo.fit(X, y)`), nunca
    `.fit()` vacio -- ese es el patron que distingue mencion de llamada aqui."""
    for f in NEW_CORE_FILES:
        for line in f.read_text(encoding="utf-8").splitlines():
            sin_espacios = line.replace(" ", "")
            if ".fit(" in sin_espacios and ".fit()" not in sin_espacios:
                raise AssertionError(f"{f.name} contiene una posible llamada real a .fit(): {line!r}")


def test_no_recalibration_in_phase4_files():
    for f in NEW_CORE_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "calibrar_umbral" not in text and "recalibrar" not in text


def test_no_test_partition_referenced_in_phase4_files():
    for f in NEW_CORE_FILES:
        text = f.read_text(encoding="utf-8")
        assert "particiones[\"test\"]" not in text.lower()
        assert "construir_ataques_desde_manifiesto" not in text


# ==============================================================================================
# Resultados ya persistidos (no se recalculan aqui, mismo patron que
# test_offline_online_equivalence.py)
# ==============================================================================================

LOADER_SUMMARY = REPORTS_DIR / "autonomous_loader_equivalence_summary.json"
CONTAINER_SUMMARY = REPORTS_DIR / "autonomous_container_equivalence_summary.json"
SUMMARY_30D = REPORTS_DIR / "autonomous_container_equivalence_summary_30d.json"

requires_loader_run = pytest.mark.skipif(not LOADER_SUMMARY.exists(), reason="ejecutar autonomous_equivalence.py primero")
requires_container_run = pytest.mark.skipif(not CONTAINER_SUMMARY.exists(), reason="ejecutar autonomous_equivalence.py primero")
requires_30d_run = pytest.mark.skipif(not SUMMARY_30D.exists(), reason="ejecutar autonomous_equivalence.py primero")


@requires_loader_run
def test_legacy_and_autonomous_params_norm_exact_match():
    with open(LOADER_SUMMARY, encoding="utf-8") as f:
        d = json.load(f)
    assert d["params_norm_exact_match"] is True
    assert d["params_norm_legacy"] == d["params_norm_autonomous"]


@requires_loader_run
def test_legacy_and_autonomous_engines_produce_same_scores_and_alerts():
    with open(LOADER_SUMMARY, encoding="utf-8") as f:
        d = json.load(f)
    assert d["all_100pct"] is True
    assert d["score_p_max_abs_diff"] == 0.0
    assert d["score_h_max_abs_diff"] == 0.0
    for campo in ("alert_p", "alert_p_ffill", "alert_h", "alert_or", "detection_source"):
        assert d["match_pct"][campo] == 100.0


@requires_container_run
def test_local_autonomous_api_preserves_all_alerts():
    with open(CONTAINER_SUMMARY, encoding="utf-8") as f:
        d = json.load(f)
    assert d["C_local_api_autonomous"]["all_categorical_100pct"] is True


@requires_container_run
def test_docker_autonomous_preserves_all_alerts_without_any_csv_mount():
    with open(CONTAINER_SUMMARY, encoding="utf-8") as f:
        d = json.load(f)
    assert d["D_docker_autonomous"]["all_categorical_100pct"] is True
    assert d["D_docker_autonomous"]["no_csv_mounted"] is True
    assert d["all_environments_100pct"] is True


@requires_30d_run
def test_docker_autonomous_30days_preserves_all_alerts():
    with open(SUMMARY_30D, encoding="utf-8") as f:
        d = json.load(f)
    assert d["all_categorical_100pct"] is True
    assert d["n_readings"] == 43200
    assert d["no_csv_mounted"] is True
