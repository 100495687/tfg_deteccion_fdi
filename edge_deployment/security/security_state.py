"""Fase 5, seccion 10: estado anti-replay, completamente INDEPENDIENTE de `DetectorState`
(nunca importado, nunca referenciado desde `core/`). Por (meter_id, session_id) guarda
UNICAMENTE: `highest_accepted_sequence`, `last_accepted_timestamp`, contadores de
aceptados/rechazos por motivo, y `last_update_time`. Nunca guarda ventanas P, lags H, lecturas
completas, scores, alertas, claves, ni payloads historicos (verificado por test:
`test_anti_replay_state_never_stores_readings_or_keys`).

Politica de secuencia (seccion 10): `sequence_number > highest_accepted_sequence` -- estricta,
sin ventana deslizante. RFC 4302 (IP Authentication Header) permite una ventana deslizante
para tolerar cierto reordenamiento entre pares IPsec; aqui NO se implementa esa ventana --
DetectorEngine ya procesa una serie temporal estrictamente ordenada (Fase 1), y una politica
estricta es la que mejor concuerda con esa semantica. Documentado como decision de diseno, no
como limitacion de RFC 4302 mal aplicada.

Aprovisionamiento de sesiones: `init_session()` es un mecanismo EXPLICITO, separado del flujo
normal de mensajes -- un `session_id` que llega en un mensaje pero nunca fue aprovisionado via
`init_session()` se rechaza (`unknown_session`), nunca se crea silenciosamente. Esto acota el
crecimiento del estado: un atacante que envia session_id arbitrarios no puede hacer crecer
`AntiReplayState` sin limite (test: `test_state_bounded_by_provisioned_sessions_only`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AntiReplayCounters:
    accepted_secure_messages: int = 0
    replay_rejections: int = 0
    invalid_mac_rejections: int = 0
    stale_rejections: int = 0
    future_rejections: int = 0
    session_rejections: int = 0


@dataclass
class SessionState:
    highest_accepted_sequence: int | None = None
    last_accepted_timestamp: datetime | None = None
    last_update_time: datetime | None = None
    counters: AntiReplayCounters = field(default_factory=AntiReplayCounters)


class AntiReplayState:
    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], SessionState] = {}

    def init_session(self, meter_id: str, session_id: str) -> SessionState:
        """Aprovisionamiento EXPLICITO -- unico punto que crea una entrada nueva."""
        key = (meter_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = SessionState()
        return self._sessions[key]

    def get(self, meter_id: str, session_id: str) -> SessionState | None:
        """Nunca crea una entrada -- devuelve None si la sesion no fue aprovisionada."""
        return self._sessions.get((meter_id, session_id))

    def session_exists(self, meter_id: str, session_id: str) -> bool:
        return (meter_id, session_id) in self._sessions

    def record_accept(self, meter_id: str, session_id: str, sequence_number: int,
                       timestamp: datetime, now: datetime) -> None:
        """Avanza el estado -- SOLO debe llamarse tras confirmar que DetectorEngine acepto
        la lectura (seccion 10, invariante central de esta fase)."""
        st = self.init_session(meter_id, session_id)
        st.highest_accepted_sequence = sequence_number
        st.last_accepted_timestamp = timestamp
        st.last_update_time = now
        st.counters.accepted_secure_messages += 1

    def record_rejection(self, meter_id: str, session_id: str, reason: str, now: datetime) -> None:
        """Registra el rechazo en los contadores de una sesion YA aprovisionada -- NUNCA
        crea una entrada nueva (a diferencia de una version anterior de este metodo, que
        llamaba a `init_session` aqui: un atacante enviando pares (meter_id, session_id)
        arbitrarios para forzar rechazos de autenticidad/sesion desconocida podia hacer
        crecer el estado sin limite, violando la cota exigida en la seccion 10/18). Los
        rechazos de mensajes SIN sesion aprovisionada (unknown_meter_key, invalid_mac,
        unknown_session) se cuentan solo en las metricas GLOBALES
        (`security_metrics.SecurityMetrics`, acotadas por diseno), nunca aqui. NUNCA toca
        `highest_accepted_sequence` ni `last_accepted_timestamp` (el estado de secuencia solo
        avanza en `record_accept`)."""
        st = self._sessions.get((meter_id, session_id))
        if st is None:
            return
        st.last_update_time = now
        field_map = {
            "duplicate_sequence": "replay_rejections", "stale_sequence": "replay_rejections",
            "invalid_mac": "invalid_mac_rejections",
            "stale_timestamp": "stale_rejections", "future_timestamp": "future_rejections",
            "unknown_session": "session_rejections", "session_conflict": "session_rejections",
        }
        attr = field_map.get(reason)
        if attr:
            setattr(st.counters, attr, getattr(st.counters, attr) + 1)

    def n_sessions(self) -> int:
        return len(self._sessions)

    def reset_session(self, meter_id: str, session_id: str) -> None:
        self._sessions.pop((meter_id, session_id), None)
