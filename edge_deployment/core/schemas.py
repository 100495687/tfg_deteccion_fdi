"""Estructuras de entrada/salida del motor online. Sin dependencia de ningun framework
web: FastAPI, en una fase posterior, se limitara a validar el
JSON de entrada y llamar a `DetectorEngine.ingest(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

ENGINE_STATUSES = ("empty", "warming_up", "ready", "degraded", "invalid_input", "temporal_gap", "error")

REJECTION_REASONS = (
    None, "invalid_timestamp", "empty_meter_id", "non_numeric_power", "nan_value", "infinite_value",
    "negative_power", "duplicate_identical", "duplicate_conflicting", "out_of_order", "temporal_gap_detected",
)


@dataclass
class Reading:
    """Una lectura de un minuto. Solo se soporta un `meter_id` activo en esta primera
    version, pero la estructura queda preparada para asociar estado a meter_id."""
    meter_id: str
    timestamp: datetime
    power_kw: float


@dataclass
class IngestResponse:
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
    p_last_evaluation_timestamp: datetime | None
    h_last_evaluation_timestamp: datetime | None
    buffer_1min_size: int
    buffer_15min_size: int
    processing_time_ms: float
    warnings: list[str] = field(default_factory=list)
    detection_source: str | None = None  # P_only | H_only | both | none | partial_availability
    aggregate_15min_kw: float | None = None  # agregado de 15 min recien cerrado, solo si h_evaluated

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["timestamp"] = str(d["timestamp"])
        d["p_last_evaluation_timestamp"] = str(d["p_last_evaluation_timestamp"]) if d["p_last_evaluation_timestamp"] is not None else None
        d["h_last_evaluation_timestamp"] = str(d["h_last_evaluation_timestamp"]) if d["h_last_evaluation_timestamp"] is not None else None
        return d
