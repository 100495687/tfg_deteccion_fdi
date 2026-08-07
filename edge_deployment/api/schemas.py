"""Esquemas Pydantic. Validan unicamente forma/tipo/estructura del JSON
(json valido, campos obligatorios, tipos, timestamp parseable, numero finito, longitud de
strings) -- nunca reimplementan la logica de negocio del motor (duplicados, fuera de orden,
huecos, disponibilidad P/H, continuidad...), que sigue siendo responsabilidad exclusiva de
`DetectorEngine`/`DetectorState` (ver `core/validation.py`, `core/streaming_preprocessing.py`).
"""
from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_METER_ID_LEN = 128


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validar_finito(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("power_kw debe ser un numero finito (ni NaN ni infinito)")
    return v


class ReadingRequest(_StrictModel):
    meter_id: str = Field(min_length=1, max_length=MAX_METER_ID_LEN)
    timestamp: datetime
    power_kw: float = Field(allow_inf_nan=False)

    @field_validator("power_kw")
    @classmethod
    def _power_kw_finito(cls, v: float) -> float:
        return _validar_finito(v)


class BootstrapReading(_StrictModel):
    timestamp: datetime
    power_kw: float = Field(allow_inf_nan=False)

    @field_validator("power_kw")
    @classmethod
    def _power_kw_finito(cls, v: float) -> float:
        return _validar_finito(v)


class BootstrapRequest(_StrictModel):
    meter_id: str = Field(min_length=1, max_length=MAX_METER_ID_LEN)
    readings: list[BootstrapReading] = Field(min_length=1)


class ReadingResponse(BaseModel):
    meter_id: str
    timestamp: datetime
    accepted: bool
    rejection_reason: str | None
    engine_status: str
    warming_up: bool
    p_ready: bool
    h_ready: bool
    p_evaluated: bool
    h_evaluated: bool
    score_p: float | None
    score_h: float | None
    alert_p: bool | None
    alert_p_ffill: bool | None
    alert_h: bool | None
    alert_or: bool | None
    detection_source: str | None
    p_last_evaluation_timestamp: datetime | None
    h_last_evaluation_timestamp: datetime | None
    buffer_1min_size: int
    buffer_15min_size: int
    engine_processing_time_ms: float
    api_processing_time_ms: float
    warnings: list[str] = Field(default_factory=list)


class BootstrapResponse(BaseModel):
    meter_id: str
    bootstrap_completed: bool
    readings_received: int
    readings_accepted: int
    p_ready: bool
    h_ready: bool
    stream_anchor_timestamp: datetime | None
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    engine_loaded: bool
    models_loaded: bool
    artifact_hashes_valid: bool
    architecture: str
    threshold_p: float
    threshold_h: float
    workers_supported: int
    state_backend: str


class MeterStatusResponse(BaseModel):
    meter_id: str
    state_exists: bool
    engine_status: str
    last_accepted_timestamp: datetime | None
    stream_anchor_timestamp: datetime | None
    current_bucket_start: datetime | None
    current_bucket_sample_count: int
    p_ready: bool
    h_ready: bool
    p_last_evaluation_timestamp: datetime | None
    h_last_evaluation_timestamp: datetime | None
    last_score_p: float | None
    last_score_h: float | None
    last_alert_p: bool | None
    last_alert_h: bool | None
    last_alert_or: bool | None
    buffer_1min_size: int
    history_15min_size: int
    accepted_readings: int
    rejected_readings: int
    duplicates: int
    out_of_order: int
    gaps: int
    warnings: list[str] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    service_start_time: datetime
    uptime_seconds: float
    model_load_time_ms: float
    bootstrap_requests: int
    reading_requests: int
    status_requests: int
    accepted_readings: int
    rejected_readings: int
    duplicate_identical: int
    duplicate_conflicting: int
    out_of_order: int
    temporal_gaps: int
    invalid_values: int
    p_evaluations: int
    h_evaluations: int
    alerts_p: int
    alerts_h: int
    alerts_or: int
    internal_errors: int
    latency_api_mean_ms: float | None
    latency_api_p50_ms: float | None
    latency_api_p95_ms: float | None
    latency_api_p99_ms: float | None
    latency_engine_mean_ms: float | None
    active_meter_states: int
    approximate_memory_mb: float | None


class ResetResponse(BaseModel):
    meter_id: str
    reset: bool
    models_reloaded: bool
    state_exists: bool


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    meter_id: str | None = None
    timestamp: datetime | None = None
    state_modified: bool
    request_id: str
