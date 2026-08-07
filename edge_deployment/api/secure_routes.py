"""Fase 5: `POST /secure-readings` + `/security/status` + `/security/ready` -- ruta PARALELA a
`POST /readings` (`edge_deployment/api/routes.py`, Fase 1-4, sin tocar). Nunca hace una
peticion HTTP interna a `/readings`: llama a la MISMA interfaz publica de `DetectorEngine`
(`engine.ingest(...)`) que ya usa `routes.py`, reutilizando el MISMO `MeterLockRegistry`
(`request.app.state.locks`) para que la comprobacion de secuencia + la llamada al motor + la
actualizacion de `AntiReplayState` sean atomicas respecto al mismo `meter_id` (seccion 12).

Orden exacto (seccion 12, pasos 1-12):
  1-4. schema (Pydantic) -> clave -> canonicalizar -> HMAC       [SIN lock]
  5.   adquirir el lock del meter_id UNA vez
  6-8. sesion -> secuencia -> frescura                            [CON lock]
  9.   engine.ingest() UNA vez
  10.  AntiReplayState.commit_accept() SOLO si el motor acepto
  11.  liberar el lock (automatico al salir del `with`)
  12.  devolver la respuesta

Ademas de `/secure-readings`, se incluyen 2 endpoints de ADMINISTRACION/PRUEBA, claramente
separados del flujo normal (seccion 14: "permitir inicializar una sesion mediante un mecanismo
explicito de administracion o prueba, separado del flujo normal"): `POST
/security/test/init-session` (aprovisiona una sesion) y `POST /security/test/clock` (solo
funciona si el reloj configurado es `SimulatedClock` -- necesario para repetir el periodo
pre-test historico de 2009 sin desincronizar el reloj real del sistema).

CORRECCION OBLIGATORIA (post-aceptacion inicial de Fase 5): estos 2 endpoints ahora exigen
`security.test_endpoints_enabled=true` (variable de entorno `FDIA_SECURITY_TEST_ENDPOINTS_ENABLED`,
`false` por defecto SIEMPRE, independiente de `anti_replay_enabled`) -- en produccion devuelven
**404** (no simplemente 503 como antes, y no basta con que el reloj no sea simulado: un
atacante no debe poder ni siquiera invocar `init-session` para provisionar o RE-provisionar
una sesion). `init_session_admin` ademas rechaza explicitamente re-inicializar una sesion que
YA existe (409) -- ni un operador de pruebas, ni mucho menos un atacante, pueden resetear el
`highest_accepted_sequence` de una sesion activa a traves de este mecanismo.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from edge_deployment.api.dependencies import get_request_id
from edge_deployment.api.secure_error_handlers import SecurityRejectionError
from edge_deployment.security.anti_replay import HTTP_STATUS_BY_REASON, REASON_DETECTOR_REJECTED, AntiReplayGuard
from edge_deployment.security.canonicalization import quantize_power_kw_float
from edge_deployment.security.secure_schemas import (
    InitSessionRequest,
    SecureReadingRequest,
    SecureReadingResponse,
    SecurityReadyResponse,
    SecurityStatusResponse,
    SetClockRequest,
)
from edge_deployment.security.security_clock import SimulatedClock
from edge_deployment.security.security_metrics import SecurityMetrics

secure_router = APIRouter()


def _require_anti_replay_enabled(request: Request, meter_id: str | None, session_id: str | None,
                                  sequence_number: int | None) -> None:
    if not getattr(request.app.state, "anti_replay_enabled", False):
        raise SecurityRejectionError(503, "anti_replay_not_enabled", meter_id, session_id, sequence_number)


def _require_test_endpoints_enabled(request: Request) -> None:
    """Correccion obligatoria: en produccion (`test_endpoints_enabled=false`, SIEMPRE por
    defecto) los endpoints /security/test/* deben devolver 404 -- ni registrarse
    condicionalmente (FastAPI registra las rutas en el arranque del proceso, antes de que la
    configuracion este disponible) ni depender unicamente de que el reloj no sea simulado
    (eso dejaba `init-session` accesible incluso en produccion). Se usa `HTTPException` de
    FastAPI (no `SecurityRejectionError`) para que la respuesta sea un 404 estandar, no el
    formato especifico de rechazo de `/secure-readings`."""
    if not getattr(request.app.state, "security_test_endpoints_enabled", False):
        raise HTTPException(status_code=404, detail="Not Found")


@secure_router.post("/secure-readings", response_model=SecureReadingResponse)
def secure_readings(payload: SecureReadingRequest, request: Request, response: Response,
                     request_id: str = Depends(get_request_id)) -> SecureReadingResponse:
    t0 = time.perf_counter()
    st = request.app.state
    _require_anti_replay_enabled(request, payload.meter_id, payload.session_id, payload.sequence_number)
    if not getattr(st, "ready", False):
        raise SecurityRejectionError(503, "detector_not_ready", payload.meter_id, payload.session_id, payload.sequence_number)

    guard: AntiReplayGuard = st.security_guard
    sec_metrics: SecurityMetrics = st.security_metrics

    # pasos 1-4: SIN lock, no toca AntiReplayState todavia
    message = {
        "protocol_version": payload.protocol_version, "meter_id": payload.meter_id,
        "session_id": payload.session_id, "timestamp": payload.timestamp,
        "sequence_number": payload.sequence_number, "power_kw": payload.power_kw, "mac": payload.mac,
    }
    auth = guard.verify_authenticity(message)
    if not auth.ok:
        sec_ms = (time.perf_counter() - t0) * 1000
        sec_metrics.record_reject(auth.reason, sec_ms, sec_ms)
        guard.record_rejection(payload.meter_id, payload.session_id, auth.reason)  # no-op si la sesion no existe
        raise SecurityRejectionError(HTTP_STATUS_BY_REASON.get(auth.reason, 401), auth.reason,
                                      payload.meter_id, payload.session_id, payload.sequence_number)

    # paso 5: UN solo lock, el MISMO registro que /readings (routes.py, sin modificar)
    locks = st.locks
    lock = locks.get_lock(payload.meter_id)
    engine = st.engine
    detector_invoked = False
    engine_ms = 0.0

    with lock:
        # pasos 6-8: sesion, secuencia, frescura -- CON el lock ya adquirido
        seq_check = guard.check_sequence_and_freshness(payload.meter_id, payload.session_id,
                                                         payload.sequence_number, payload.timestamp)
        if not seq_check.ok:
            sec_ms = (time.perf_counter() - t0) * 1000
            sec_metrics.record_reject(seq_check.reason, sec_ms, sec_ms)
            guard.record_rejection(payload.meter_id, payload.session_id, seq_check.reason)
            raise SecurityRejectionError(HTTP_STATUS_BY_REASON.get(seq_check.reason, 409), seq_check.reason,
                                          payload.meter_id, payload.session_id, payload.sequence_number)

        # paso 9: UNA sola llamada a DetectorEngine.ingest() -- misma interfaz publica que /readings.
        # Correccion obligatoria: se entrega el valor CUANTIZADO (el mismo que cubre el HMAC),
        # NUNCA payload.power_kw crudo -- ver canonicalization.py, nota de aliasing.
        quantized_power_kw = quantize_power_kw_float(payload.power_kw)
        t_engine0 = time.perf_counter()
        resp = engine.ingest(payload.meter_id, payload.timestamp, quantized_power_kw)
        detector_invoked = True
        engine_ms = (time.perf_counter() - t_engine0) * 1000

        # paso 10: el estado anti-replay SOLO avanza si el motor acepto
        if resp.accepted:
            guard.commit_accept(payload.meter_id, payload.session_id, payload.sequence_number, payload.timestamp)
        else:
            guard.record_rejection(payload.meter_id, payload.session_id, REASON_DETECTOR_REJECTED)
    # paso 11: lock liberado automaticamente al salir del `with`

    total_ms = (time.perf_counter() - t0) * 1000
    security_ms = total_ms - engine_ms

    # metricas SEPARADAS de las del detector (seccion 20) -- nunca se llama a
    # request.app.state.metrics (Fase 2, exclusivo de /readings)
    if resp.accepted:
        sec_metrics.record_accept(security_ms, engine_ms, total_ms)
    else:
        sec_metrics.record_reject(REASON_DETECTOR_REJECTED, security_ms, total_ms)

    response.status_code = 200 if resp.accepted else 409
    return SecureReadingResponse(
        accepted=resp.accepted, security_status="authenticated_fresh",
        security_rejection_reason=None if resp.accepted else REASON_DETECTOR_REJECTED,
        meter_id=payload.meter_id, session_id=payload.session_id, sequence_number=payload.sequence_number,
        detector_invoked=detector_invoked, detector_result=resp.to_dict(),
        security_processing_time_ms=security_ms, total_processing_time_ms=total_ms, request_id=request_id,
    )


@secure_router.get("/security/status", response_model=SecurityStatusResponse)
def security_status(request: Request) -> SecurityStatusResponse:
    st = request.app.state
    metrics_snapshot = st.security_metrics.snapshot() if hasattr(st, "security_metrics") else {}
    n_sessions = st.security_state.n_sessions() if hasattr(st, "security_state") else 0
    return SecurityStatusResponse(anti_replay_enabled=bool(getattr(st, "anti_replay_enabled", False)),
                                   n_sessions=n_sessions, metrics=metrics_snapshot)


@secure_router.get("/security/ready", response_model=SecurityReadyResponse)
def security_ready(request: Request) -> SecurityReadyResponse:
    """NUNCA hace que `/ready` (routes.py) dependa de esto -- es un endpoint NUEVO,
    independiente. El detector original sigue `ready` aunque la seguridad no este
    configurada."""
    st = request.app.state
    enabled = bool(getattr(st, "anti_replay_enabled", False))
    if not enabled:
        return SecurityReadyResponse(anti_replay_enabled=False, security_ready=False, detail="anti_replay_enabled=false")
    detector_ready = bool(getattr(st, "ready", False))
    detail = "ready" if detector_ready else "detector not ready (engine no preparado)"
    return SecurityReadyResponse(anti_replay_enabled=True, security_ready=detector_ready, detail=detail)


# ==========================================================================================
# Administracion/prueba (seccion 14) -- NUNCA se usan en el flujo normal de mensajes.
# ==========================================================================================

@secure_router.post("/security/test/init-session")
def init_session_admin(payload: InitSessionRequest, request: Request) -> dict:
    st = request.app.state
    _require_test_endpoints_enabled(request)  # 404 en produccion, primero y siempre
    _require_anti_replay_enabled(request, payload.meter_id, payload.session_id, None)
    if st.security_state.session_exists(payload.meter_id, payload.session_id):
        # correccion obligatoria: ni un operador de pruebas ni un atacante pueden
        # re-inicializar/resetear una sesion YA aprovisionada a traves de este mecanismo --
        # solo el primer aprovisionamiento tiene efecto.
        raise SecurityRejectionError(409, "session_already_provisioned", payload.meter_id, payload.session_id, None)
    st.security_guard.init_session(payload.meter_id, payload.session_id)
    return {"initialized": True, "meter_id": payload.meter_id, "session_id": payload.session_id}


@secure_router.post("/security/test/clock")
def set_clock_admin(payload: SetClockRequest, request: Request) -> dict:
    """Solo funciona si `test_endpoints_enabled=true` Y `clock_mode: simulated` -- en
    produccion (ambos `false`/`system` por defecto) devuelve 404 antes de comprobar nada mas,
    nunca puede manipular el reloj real."""
    st = request.app.state
    _require_test_endpoints_enabled(request)  # 404 en produccion, primero y siempre
    _require_anti_replay_enabled(request, None, None, None)
    clock = st.security_clock
    if not isinstance(clock, SimulatedClock):
        raise SecurityRejectionError(503, "clock_not_simulated", None)
    clock.set(payload.timestamp)
    return {"clock_set_to": clock.now().isoformat()}
