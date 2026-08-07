"""Fase 5, seccion 8/13: esquemas Pydantic del mensaje seguro y sus respuestas. Mismo patron
que `edge_deployment/api/schemas.py` (Fase 2, sin modificar): `extra="forbid"`, validacion
solo de forma/tipo -- nunca logica de negocio (eso es responsabilidad de `AntiReplayGuard` y,
mas alla, de `DetectorEngine`).
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_ID_LEN = 128
PROTOCOL_VERSION = "1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SecureReadingRequest(_StrictModel):
    protocol_version: str = Field(min_length=1, max_length=8)
    meter_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    session_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    timestamp: datetime
    sequence_number: int = Field(ge=0)
    power_kw: float = Field(allow_inf_nan=False)
    mac: str = Field(min_length=1, max_length=128)

    @field_validator("power_kw")
    @classmethod
    def _power_kw_finito(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("power_kw debe ser un numero finito (ni NaN ni infinito)")
        return v


class SecureReadingResponse(BaseModel):
    accepted: bool
    security_status: str
    security_rejection_reason: str | None
    meter_id: str
    session_id: str | None
    sequence_number: int | None
    detector_invoked: bool
    detector_result: dict[str, Any] | None
    security_processing_time_ms: float
    total_processing_time_ms: float
    request_id: str


class SecurityStatusResponse(BaseModel):
    anti_replay_enabled: bool
    n_sessions: int
    metrics: dict[str, Any]


class SecurityReadyResponse(BaseModel):
    anti_replay_enabled: bool
    security_ready: bool
    detail: str


class InitSessionRequest(_StrictModel):
    """Aprovisionamiento EXPLICITO de sesion (seccion 14) -- solo para administracion/pruebas."""
    meter_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    session_id: str = Field(min_length=1, max_length=MAX_ID_LEN)


class SetClockRequest(_StrictModel):
    """Solo tiene efecto si el reloj configurado es SimulatedClock (seccion 11) -- para
    reproducir el periodo pre-test historico sin desincronizar el reloj real del sistema."""
    timestamp: datetime
