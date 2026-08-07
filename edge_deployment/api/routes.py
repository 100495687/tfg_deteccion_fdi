"""Endpoints. Cada endpoint se limita a: validar (Pydantic ya lo hizo), adquirir
lock si corresponde, llamar exactamente una vez al metodo de `DetectorEngine` ya existente, y
traducir la respuesta a JSON -- ninguna logica de agregacion/scoring/propagacion/fusion vive
aqui.
"""
from __future__ import annotations

import time

import pandas as pd
import psutil
from fastapi import APIRouter, Depends, Request

from edge_deployment.api.dependencies import get_engine, get_locks, get_metrics, get_request_id, require_ready
from edge_deployment.api.error_handlers import ConflictError, NotFoundError, NotReadyError, UnprocessableError
from edge_deployment.api.logging_config import get_logger, log_reading_event
from edge_deployment.api.schemas import (
    BootstrapRequest, BootstrapResponse, HealthResponse, MeterStatusResponse, MetricsResponse,
    ReadingRequest, ReadingResponse, ReadyResponse, ResetResponse,
)
from edge_deployment.core.detector_engine import DetectorEngine
from edge_deployment.core.model_loader import H_MODEL_PATH, H_THRESHOLD_PATH, P_CHECKPOINT_PATH, P_THRESHOLD_PATH
from src.data_loading import COLUMNA_OBJETIVO

router = APIRouter()
logger = get_logger()

SERVICE_NAME = "fdia-edge-detector"
SERVICE_VERSION = "1.0.0"


# ==========================================================================================
# 8.1 /health -- vivo, sin inferencia, sin validar hashes
# ==========================================================================================

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME, version=SERVICE_VERSION)


# ==========================================================================================
# 8.2 /ready
# ==========================================================================================

@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    st = request.app.state
    payload = ReadyResponse(
        ready=bool(getattr(st, "ready", False)), engine_loaded=st.engine is not None,
        models_loaded=st.engine is not None, artifact_hashes_valid=bool(getattr(st, "artifact_hashes_valid", False)),
        architecture="P_OR_H", threshold_p=st.threshold_p or 0.0, threshold_h=st.threshold_h or 0.0,
        workers_supported=1, state_backend="in_memory",
    )
    request.state._ready_payload = payload
    if not payload.ready:
        raise NotReadyError("El motor no esta preparado", state_modified=False)
    return payload


# ==========================================================================================
# Adaptador de formato: lista JSON -> DataFrame que ya espera DetectorEngine.bootstrap
# (identico formato al usado por streaming_preprocessing.get_window / particiones offline)
# ==========================================================================================

def _bootstrap_readings_to_dataframe(payload: BootstrapRequest) -> pd.DataFrame:
    idx = pd.DatetimeIndex([r.timestamp for r in payload.readings])
    valores = [r.power_kw for r in payload.readings]
    return pd.DataFrame({COLUMNA_OBJETIVO: valores, "imputado": False}, index=idx)


# ==========================================================================================
# 8.3 /bootstrap
# ==========================================================================================

@router.post("/bootstrap", response_model=BootstrapResponse, status_code=200)
def bootstrap(payload: BootstrapRequest, request: Request, _: None = Depends(require_ready),
              engine: DetectorEngine = Depends(get_engine), metrics=Depends(get_metrics),
              locks=Depends(get_locks)) -> BootstrapResponse:
    t0 = time.perf_counter()
    lock = locks.get_lock(payload.meter_id)
    with lock:
        if payload.meter_id in request.app.state.bootstrapped_meters:
            raise ConflictError("bootstrap ya completado para este meter_id (llama antes a /reset)",
                                 error_code="bootstrap_duplicate", meter_id=payload.meter_id, state_modified=False)
        df = _bootstrap_readings_to_dataframe(payload)
        try:
            reporte = engine.bootstrap(payload.meter_id, df)
        except ValueError as e:
            raise UnprocessableError(str(e), meter_id=payload.meter_id, state_modified=False) from e
        request.app.state.bootstrapped_meters.add(payload.meter_id)

    metrics.record_bootstrap()
    dt_ms = (time.perf_counter() - t0) * 1000
    stream_anchor = None
    if reporte.get("last_bootstrap_timestamp"):
        stream_anchor = pd.Timestamp(reporte["last_bootstrap_timestamp"]) + pd.Timedelta(minutes=1)
    return BootstrapResponse(
        meter_id=payload.meter_id, bootstrap_completed=True,
        readings_received=reporte["number_of_readings"], readings_accepted=reporte["number_of_readings"],
        p_ready=reporte["p_ready"], h_ready=reporte["h_ready"], stream_anchor_timestamp=stream_anchor,
        warnings=reporte.get("warnings", []), processing_time_ms=dt_ms,
    )


# ==========================================================================================
# 8.4 /readings -- endpoint principal
# ==========================================================================================

@router.post("/readings", response_model=ReadingResponse, status_code=200)
def readings(payload: ReadingRequest, request: Request, _: None = Depends(require_ready),
             engine: DetectorEngine = Depends(get_engine), metrics=Depends(get_metrics),
             locks=Depends(get_locks), request_id: str = Depends(get_request_id)) -> ReadingResponse:
    t0 = time.perf_counter()
    lock = locks.get_lock(payload.meter_id)
    with lock:
        resp = engine.ingest(payload.meter_id, payload.timestamp, payload.power_kw)
    api_dt_ms = (time.perf_counter() - t0) * 1000

    metrics.record_reading(resp, api_dt_ms)
    status_code = 200 if resp.accepted else 409
    metrics.record_status_code(status_code)
    log_reading_event(logger, request_id=request_id, meter_id=payload.meter_id, timestamp=payload.timestamp,
                       accepted=resp.accepted, rejection_reason=resp.rejection_reason, p_evaluated=resp.p_evaluated,
                       h_evaluated=resp.h_evaluated, alert_or=resp.alert_or, engine_time_ms=resp.processing_time_ms,
                       api_time_ms=api_dt_ms, status_code=status_code)

    if not resp.accepted:
        # duplicate_identical / duplicate_conflicting / out_of_order son las unicas razones de
        # rechazo alcanzables aqui: NaN/infinito/meter_id vacio ya los filtra Pydantic (422)
        raise ConflictError(f"lectura rechazada por el motor: {resp.rejection_reason}", error_code=resp.rejection_reason,
                             meter_id=payload.meter_id, timestamp=payload.timestamp, state_modified=False)

    return ReadingResponse(
        meter_id=resp.meter_id, timestamp=resp.timestamp, accepted=resp.accepted, rejection_reason=resp.rejection_reason,
        engine_status=resp.engine_status, warming_up=resp.warming_up, p_ready=resp.p_ready, h_ready=resp.h_ready,
        p_evaluated=resp.p_evaluated, h_evaluated=resp.h_evaluated, score_p=resp.score_p, score_h=resp.score_h,
        alert_p=resp.alert_p, alert_p_ffill=resp.alert_p_ffill, alert_h=resp.alert_h, alert_or=resp.alert_or,
        detection_source=resp.detection_source, p_last_evaluation_timestamp=resp.p_last_evaluation_timestamp,
        h_last_evaluation_timestamp=resp.h_last_evaluation_timestamp, buffer_1min_size=resp.buffer_1min_size,
        buffer_15min_size=resp.buffer_15min_size, engine_processing_time_ms=resp.processing_time_ms,
        api_processing_time_ms=api_dt_ms, warnings=resp.warnings,
    )


# ==========================================================================================
# 8.5 /status/{meter_id} -- nunca modifica estado
# ==========================================================================================

@router.get("/status/{meter_id}", response_model=MeterStatusResponse)
def status(meter_id: str, request: Request, _: None = Depends(require_ready), engine: DetectorEngine = Depends(get_engine),
           metrics=Depends(get_metrics), locks=Depends(get_locks)) -> MeterStatusResponse:
    lock = locks.get_lock(meter_id)
    with lock:  # lock breve de lectura: snapshot atomico, no muta nada
        s = engine.get_status(meter_id)
    metrics.record_status_request()
    if not s["state_exists"]:
        raise NotFoundError(f"no existe estado para meter_id={meter_id}", meter_id=meter_id, state_modified=False)
    return MeterStatusResponse(
        meter_id=s["meter_id"], state_exists=s["state_exists"], engine_status=s["engine_status"],
        last_accepted_timestamp=s["last_accepted_timestamp"], stream_anchor_timestamp=s["stream_anchor_timestamp"],
        current_bucket_start=s["current_bucket_start"], current_bucket_sample_count=s["current_bucket_sample_count"],
        p_ready=s["p_ready"], h_ready=s["h_ready"], p_last_evaluation_timestamp=s["p_last_evaluation_timestamp"],
        h_last_evaluation_timestamp=s["h_last_evaluation_timestamp"], last_score_p=s["score_p"], last_score_h=s["score_h"],
        last_alert_p=s["alert_p_ffill"], last_alert_h=s["alert_h"], last_alert_or=s["last_alert_or"],
        buffer_1min_size=s["buffer_1min_size"], history_15min_size=s["buffer_15min_size"],
        accepted_readings=s["n_accepted"], rejected_readings=s["n_rejected"],
        duplicates=s["n_duplicates_identical"] + s["n_duplicates_conflicting"], out_of_order=s["n_out_of_order"],
        gaps=s["n_gaps"], warnings=[],
    )


# ==========================================================================================
# 8.6 /metrics -- JSON (Prometheus queda para una fase posterior)
# ==========================================================================================

@router.get("/metrics", response_model=MetricsResponse)
def metrics_endpoint(request: Request, engine: DetectorEngine = Depends(get_engine), metrics=Depends(get_metrics)) -> MetricsResponse:
    n_meters = engine.n_active_meters() if engine is not None else 0
    try:
        mem_mb = psutil.Process().memory_info().rss / 1e6
    except Exception:
        mem_mb = None
    snap = metrics.snapshot(n_active_meters=n_meters, approximate_memory_mb=mem_mb)
    return MetricsResponse(**{k: v for k, v in snap.items() if k != "status_codes"})


# ==========================================================================================
# 8.7 /reset/{meter_id} -- idempotente (ver README_API.md, decision documentada)
# ==========================================================================================

@router.post("/reset/{meter_id}", response_model=ResetResponse)
def reset(meter_id: str, request: Request, _: None = Depends(require_ready), engine: DetectorEngine = Depends(get_engine),
          locks=Depends(get_locks)) -> ResetResponse:
    lock = locks.get_lock(meter_id)
    with lock:
        r = engine.reset(meter_id)
        request.app.state.bootstrapped_meters.discard(meter_id)
    return ResetResponse(meter_id=meter_id, reset=True, models_reloaded=False, state_exists=False)


# ==========================================================================================
# Verificacion de artefactos congelados usados ("verificarla de nuevo, no modificarla")
# ==========================================================================================

def _verificar_umbrales_congelados() -> None:
    assert P_CHECKPOINT_PATH.exists() and H_MODEL_PATH.exists()
    assert P_THRESHOLD_PATH.exists() and H_THRESHOLD_PATH.exists()
