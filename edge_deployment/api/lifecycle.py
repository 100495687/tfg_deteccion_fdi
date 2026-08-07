"""Ciclo de vida de la aplicacion (seccion 6). Los modelos se cargan EXACTAMENTE una vez
aqui, nunca dentro de un endpoint ni de una dependencia por-peticion. `DetectorEngine` se
instancia una sola vez y se guarda en `app.state.engine`; `ready` solo pasa a `true` si la
validacion de artefactos y la carga terminan sin errores.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI

from edge_deployment.api.locks import MeterLockRegistry
from edge_deployment.api.logging_config import get_logger
from edge_deployment.api.metrics import ApiMetrics
from edge_deployment.api.security_lifecycle import init_security
from edge_deployment.core.detector_engine import DetectorEngine

EDGE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = EDGE_DIR / "results" / "reports"

SERVICE_NAME = "fdia-edge-detector"
SERVICE_VERSION = "1.0.0"


def validate_artifacts() -> dict:
    """Fase 4, seccion 3-4: usa `manifest_v2.validate_manifest_v2()` -- comparacion REAL
    hash_calculado==hash_esperado contra `artifact_manifest_v2.json` (los 8 artefactos que
    necesita el autonomous_loader: checkpoint P, modelo H, perfil H, params_norm_p, ambos
    umbrales, features H, config base -- NO incluye data_raw.csv ni episodes_master_val.csv,
    que el camino de produccion ya no abre). Antes (Fase 3) esto solo comprobaba PRESENCIA de
    artefactos via `model_loader.build_input_artifacts_inventory/build_artifact_manifest`
    (que registraban hashes pero nunca los comparaban contra una referencia); ahora un
    artefacto con un byte alterado tambien produce `hashes_valid=False` sin necesidad de
    llegar a `joblib.load`/`torch.load` (`manifest_v2` solo lee bytes crudos para el hash).
    Se comprueba ANTES de construir el DetectorEngine, igual que el patron ya establecido en
    Fase 3 para el caso de artefacto ausente (evita lanzar `ArtifactIntegrityError` sin
    capturar durante un arranque normal)."""
    from edge_deployment.core import manifest_v2

    t0 = time.perf_counter()
    integrity = manifest_v2.validate_manifest_v2()
    dt = (time.perf_counter() - t0) * 1000
    missing_and_mismatched = list(integrity["missing"]) + [m["artifact"] for m in integrity["mismatched"]]
    return {"hashes_valid": integrity["integrity_valid"],
            "n_missing": integrity["n_missing"] + integrity["n_mismatched"],
            "missing": missing_and_mismatched, "manifest": integrity, "validation_time_ms": dt}


def build_detector_engine() -> tuple[DetectorEngine, float]:
    """Instancia UNICA de `DetectorEngine` -- esta es la unica linea de todo el codigo de la
    API que construye un motor. Fase 4: `DetectorEngine()` sin argumentos usa por defecto el
    autonomous_loader (`use_legacy_loader=False`) -- la API de produccion nunca abre
    data_raw.csv ni episodes_master_val.csv."""
    t0 = time.perf_counter()
    engine = DetectorEngine()
    dt = (time.perf_counter() - t0) * 1000
    return engine, dt


def _guardar_json(nombre: str, payload: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / nombre, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = get_logger()
    t_start = time.perf_counter()

    app.state.service_name = SERVICE_NAME
    app.state.service_version = SERVICE_VERSION
    app.state.metrics = ApiMetrics()
    app.state.locks = MeterLockRegistry()
    app.state.engine = None
    app.state.threshold_p = None
    app.state.threshold_h = None
    app.state.bootstrapped_meters = set()  # seguimiento a nivel API de bootstrap ya completado (seccion 8.3)

    # Fase 5: unica linea permitida para inicializar el componente OPCIONAL de seguridad
    # (secciones 4/6) -- nunca lanza, nunca condiciona `ready` del detector; independiente de
    # la validacion/carga de artefactos que sigue debajo, sin modificar.
    init_security(app)

    validation = validate_artifacts()
    startup_report = {
        "service": SERVICE_NAME, "version": SERVICE_VERSION,
        "artifact_validation": {"hashes_valid": validation["hashes_valid"], "n_missing": validation["n_missing"],
                                 "missing": validation["missing"], "validation_time_ms": validation["validation_time_ms"]},
    }
    _guardar_json("api_artifact_validation.json", {"hashes_valid": validation["hashes_valid"], "manifest": validation["manifest"]})

    if not validation["hashes_valid"]:
        app.state.artifact_hashes_valid = False
        app.state.ready = False
        logger.error("startup: artefactos faltantes %s -- ready=false", validation["missing"])
        startup_report["ready"] = False
        _guardar_json("api_startup_report.json", startup_report)
        yield
    else:
        engine, load_ms = build_detector_engine()
        app.state.engine = engine
        app.state.metrics.model_load_time_ms = load_ms
        app.state.artifact_hashes_valid = True
        app.state.threshold_p = engine.threshold_p
        app.state.threshold_h = engine.threshold_h
        app.state.ready = True

        startup_report.update({
            "model_load_time_ms": load_ms, "total_startup_time_ms": (time.perf_counter() - t_start) * 1000,
            "threshold_p": engine.threshold_p, "threshold_h": engine.threshold_h, "architecture": "P_OR_H",
            "ready": True, "fecha_utc": datetime.now(timezone.utc).isoformat(),
        })
        _guardar_json("api_startup_report.json", startup_report)
        logger.info("startup completado: ready=true, model_load_ms=%.1f", load_ms)
        yield

    app.state.ready = False
