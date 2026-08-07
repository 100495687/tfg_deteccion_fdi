"""Fase 5: `AntiReplayGuard` -- orquesta autenticidad (HMAC), sesion, secuencia y frescura.
NUNCA llama a `DetectorEngine` (eso es responsabilidad exclusiva de la capa API, Gate 2, que
ya posee el motor y el registro de locks) -- este modulo solo produce un VEREDICTO de
seguridad y, tras la aceptacion del motor, permite confirmar el avance de estado.

Orden de comprobacion (seccion 12, pasos 1-8), cada metodo se detiene en el primer fallo y
registra esa como "la primera regla aplicada" (seccion 16, caso D):
  1-2. esquema + localizar clave         -> `verify_authenticity` (sin lock, sin estado)
  3-4. canonicalizar + verificar HMAC    -> `verify_authenticity` (sin lock, sin estado)
  6.   sesion                            -> `check_sequence_and_freshness` (CON lock, sin lock aqui)
  7.   secuencia                         -> `check_sequence_and_freshness`
  8.   frescura                          -> `check_sequence_and_freshness`

`check_sequence_and_freshness` NUNCA muta `AntiReplayState` -- solo `commit_accept` (paso 10,
llamado por la API tras confirmar que `DetectorEngine.ingest()` acepto) y `record_rejection`
(para contadores, nunca para `highest_accepted_sequence`) lo hacen. La API es responsable de
adquirir el lock del `meter_id` (reutilizando `edge_deployment/api/locks.py`, sin modificar)
ANTES de llamar a `check_sequence_and_freshness` y de mantenerlo hasta despues de
`commit_accept`/`record_rejection` -- ver seccion 12.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from edge_deployment.security import hmac_auth
from edge_deployment.security.canonicalization import CanonicalizationError, canonicalize
from edge_deployment.security.key_provider import KeyProvider
from edge_deployment.security.security_clock import Clock
from edge_deployment.security.security_state import AntiReplayState

# Motivos de rechazo (seccion 13/16) -- cadenas estables, usadas en respuestas HTTP, metricas y tests.
REASON_UNKNOWN_METER_KEY = "unknown_meter_key"
REASON_INVALID_MAC = "invalid_mac"
REASON_UNKNOWN_SESSION = "unknown_session"
REASON_DUPLICATE_SEQUENCE = "duplicate_sequence"
REASON_STALE_TIMESTAMP = "stale_timestamp"
REASON_FUTURE_TIMESTAMP = "future_timestamp"
REASON_DETECTOR_REJECTED = "detector_rejected"

# codigo HTTP sugerido por motivo (seccion 13)
HTTP_STATUS_BY_REASON = {
    REASON_UNKNOWN_METER_KEY: 404,
    REASON_INVALID_MAC: 401,
    REASON_UNKNOWN_SESSION: 409,
    REASON_DUPLICATE_SEQUENCE: 409,
    REASON_STALE_TIMESTAMP: 409,
    REASON_FUTURE_TIMESTAMP: 409,
    REASON_DETECTOR_REJECTED: 409,
}


@dataclass
class AuthResult:
    ok: bool
    reason: str | None
    canonical_bytes: bytes | None = None


@dataclass
class SequenceCheckResult:
    ok: bool
    reason: str | None


class AntiReplayGuard:
    def __init__(self, key_provider: KeyProvider, clock: Clock, state: AntiReplayState,
                 max_message_age_seconds: float, max_future_skew_seconds: float) -> None:
        self.key_provider = key_provider
        self.clock = clock
        self.state = state
        self.max_message_age_seconds = max_message_age_seconds
        self.max_future_skew_seconds = max_future_skew_seconds

    # ------------------------------------------------------------------------------------
    # Pasos 1-4: esquema (asumido ya validado por Pydantic antes de llegar aqui), clave, HMAC.
    # Sin lock -- no toca AntiReplayState.
    # ------------------------------------------------------------------------------------

    def verify_authenticity(self, message: dict) -> AuthResult:
        meter_id = message["meter_id"]
        key = self.key_provider.get_key(meter_id)
        if key is None:
            return AuthResult(ok=False, reason=REASON_UNKNOWN_METER_KEY)

        try:
            canonical_bytes = canonicalize(message)
        except CanonicalizationError:
            # un mensaje que ya paso la validacion de esquema Pydantic pero no se puede
            # canonicalizar (caso extremo) se trata como MAC invalido, nunca como excepcion
            # sin capturar hacia el llamador HTTP.
            return AuthResult(ok=False, reason=REASON_INVALID_MAC)

        if not hmac_auth.verify_hmac(key, canonical_bytes, message.get("mac", "")):
            return AuthResult(ok=False, reason=REASON_INVALID_MAC)

        return AuthResult(ok=True, reason=None, canonical_bytes=canonical_bytes)

    # ------------------------------------------------------------------------------------
    # Pasos 6-8: sesion, secuencia, frescura. DEBE llamarse con el lock del meter_id ya
    # adquirido por el llamador (API). No muta estado.
    # ------------------------------------------------------------------------------------

    def check_sequence_and_freshness(self, meter_id: str, session_id: str, sequence_number: int,
                                      timestamp: datetime) -> SequenceCheckResult:
        session = self.state.get(meter_id, session_id)
        if session is None:
            return SequenceCheckResult(ok=False, reason=REASON_UNKNOWN_SESSION)

        # secuencia (seccion 10): estricta, sequence_number > highest_accepted_sequence.
        if session.highest_accepted_sequence is not None and sequence_number <= session.highest_accepted_sequence:
            return SequenceCheckResult(ok=False, reason=REASON_DUPLICATE_SEQUENCE)

        # frescura (seccion 11): el tiempo de referencia es el reloj del edge (self.clock),
        # nunca un campo elegido por el cliente.
        now = self.clock.now()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=now.tzinfo)
        age_seconds = (now - timestamp).total_seconds()
        if age_seconds > self.max_message_age_seconds:
            return SequenceCheckResult(ok=False, reason=REASON_STALE_TIMESTAMP)
        if age_seconds < -self.max_future_skew_seconds:
            return SequenceCheckResult(ok=False, reason=REASON_FUTURE_TIMESTAMP)

        return SequenceCheckResult(ok=True, reason=None)

    # ------------------------------------------------------------------------------------
    # Paso 10: avanza el estado -- SOLO tras confirmar que DetectorEngine acepto la lectura.
    # ------------------------------------------------------------------------------------

    def commit_accept(self, meter_id: str, session_id: str, sequence_number: int, timestamp: datetime) -> None:
        self.state.record_accept(meter_id, session_id, sequence_number, timestamp, now=self.clock.now())

    def record_rejection(self, meter_id: str, session_id: str, reason: str) -> None:
        self.state.record_rejection(meter_id, session_id, reason, now=self.clock.now())

    # ------------------------------------------------------------------------------------
    # Aprovisionamiento explicito de sesiones (seccion 14) -- fuera del flujo normal de
    # mensajes, nunca se crea una sesion silenciosamente al recibir un session_id nuevo.
    # ------------------------------------------------------------------------------------

    def init_session(self, meter_id: str, session_id: str) -> None:
        self.state.init_session(meter_id, session_id)
