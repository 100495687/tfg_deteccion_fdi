"""Tests 1-5, 55-60, 64-70: arranque, carga unica de modelos, independencia de
FastAPI/core, sin Docker todavia."""
from __future__ import annotations

from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1] / "api"
CORE_DIR = Path(__file__).resolve().parents[1] / "core"
EDGE_DIR = Path(__file__).resolve().parents[1]
ALL_API_FILES = list(API_DIR.glob("*.py")) + list((Path(__file__).resolve().parents[1] / "clients").glob("*.py"))


def test_1_fastapi_starts(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200


def test_2_engine_created_in_lifespan(api_client):
    r = api_client.get("/ready")
    assert r.status_code == 200
    assert r.json()["engine_loaded"] is True


def test_3_engine_instantiated_once(api_client):
    """Dos peticiones consecutivas deben ver el mismo objeto motor (mismo `id`) -- se
    verifica indirectamente: bootstrap + reset + una segunda peticion no fuerzan una nueva
    carga (model_load_time_ms en /metrics no cambia entre peticiones)."""
    m1 = api_client.get("/metrics").json()["model_load_time_ms"]
    api_client.get("/health")
    api_client.get("/ready")
    m2 = api_client.get("/metrics").json()["model_load_time_ms"]
    assert m1 == m2


def test_4_5_p_and_h_loaded_once(api_client):
    """El propio `api_startup_report.json` solo se escribe una vez, durante el lifespan --
    confirma que no hay una segunda carga de P/H en ningun endpoint."""
    import json
    path = EDGE_DIR / "results" / "reports" / "api_startup_report.json"
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["ready"] is True
    assert "model_load_time_ms" in report


def test_55_no_fit_called_anywhere_in_api_source():
    for f in ALL_API_FILES:
        assert ".fit(" not in f.read_text(encoding="utf-8")


def test_56_no_threshold_recalibration_in_api_source():
    for f in ALL_API_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "calibrar_umbral" not in text and "recalibrar" not in text


def test_57_test_partition_never_referenced_in_api_source():
    for f in ALL_API_FILES:
        text = f.read_text(encoding="utf-8")
        assert "particiones[\"test\"]" not in text.lower()
        assert "construir_ataques_desde_manifiesto" not in text


def test_58_fastapi_not_imported_from_core():
    for f in CORE_DIR.glob("*.py"):
        text = f.read_text(encoding="utf-8").lower()
        assert "import fastapi" not in text and "from fastapi" not in text


def test_59_core_still_works_without_fastapi(engine, bootstrapped):
    """El motor directo (fixture de Fase 1) sigue funcionando exactamente igual con la API
    ya cargada en el mismo proceso de tests -- ninguna dependencia cruzada."""
    resp = engine.ingest(bootstrapped["meter_id"], bootstrapped["stream_start"], 1.0)
    assert resp.accepted is True


def test_60_single_worker_documented():
    readme = (EDGE_DIR / "README_API.md")
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "un unico worker" in text.lower() or "workers 1" in text.lower() or "--workers 1" in text


def test_64_hashes_validated_at_startup():
    import json
    path = EDGE_DIR / "results" / "reports" / "api_artifact_validation.json"
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    assert d["hashes_valid"] is True


def test_65_missing_artifact_yields_ready_false(preserve_startup_reports):
    """Instancia una app FastAPI independiente (mismo router/lifespan) para no interferir con
    el `app` compartido de la sesion de tests (el fixture `api_client` mantiene su propio
    lifespan abierto durante toda la sesion; anidar un segundo lifespan sobre el mismo objeto
    `app` dejaria `ready=false` al salir del bloque `with` interno, rompiendo el resto de
    tests que dependen de `api_client`). `preserve_startup_reports` restaura
    api_startup_report.json/api_artifact_validation.json (que el lifespan de la app aislada
    tambien escribe, mismas rutas compartidas) al terminar, sea cual sea el orden de tests."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edge_deployment.api import lifecycle as lc
    from edge_deployment.api.error_handlers import register_error_handlers
    from edge_deployment.api.routes import router

    original = lc.validate_artifacts
    try:
        lc.validate_artifacts = lambda: {"hashes_valid": False, "n_missing": 1, "missing": ["fake.joblib"],
                                          "manifest": {}, "validation_time_ms": 0.0}
        app_aislada = FastAPI(lifespan=lc.lifespan)
        register_error_handlers(app_aislada)
        app_aislada.include_router(router)
        with TestClient(app_aislada) as client:
            r = client.get("/ready")
            assert r.status_code == 503
            assert r.json()["error_code"] == "engine_not_ready"
    finally:
        lc.validate_artifacts = original


def test_66_results_only_under_edge_deployment():
    from edge_deployment.clients.api_stream_client import REPORTS_DIR, TABLES_DIR
    assert "edge_deployment" in str(REPORTS_DIR) and "edge_deployment" in str(TABLES_DIR)


def test_67_68_69_refit_replay_pilot_threshold_tradeoff_not_referenced():
    for f in ALL_API_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "refit" not in text
        assert "replay_pilot" not in text
        assert "threshold_tradeoff" not in text


def test_70_docker_kept_out_of_api_source():
    """Fase 2 comprobaba que Fase 3 (contenerizacion) no habia empezado; ahora que Fase 3 ya
    esta implementada (edge_deployment/Dockerfile, Dockerfile.baseline existen por diseno),
    lo que sigue importando de la Fase 2 es que el codigo de la API (api/, core/) siga sin
    saber nada de Docker -- toda la logica de contenedores vive en Dockerfile*/
    docker_entrypoint.sh/docker_tools/, nunca en api/ ni core/ (sin modificar)."""
    for f in ALL_API_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "dockerfile" not in text and "docker-compose" not in text
        assert "import docker" not in text
