"""Fase 5: manejo de errores de `/secure-readings`, SEPARADO de
`edge_deployment/api/error_handlers.py` (Fase 2, intacto) porque el cuerpo de respuesta de
seguridad tiene una forma distinta (seccion 13: `accepted`/`security_status`/
`security_rejection_reason`/`detector_invoked`/`detector_result`, no
`error_code`/`meter_id`/`timestamp`/`state_modified`). Nunca incluye stack trace.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class SecurityRejectionError(Exception):
    """Rechazo de la capa de seguridad -- NUNCA se lanza despues de invocar
    `DetectorEngine.ingest()` con exito (ver `secure_routes.py`: el resultado del motor se
    devuelve directamente, nunca como excepcion)."""

    def __init__(self, status_code: int, reason: str, meter_id: str | None,
                 session_id: str | None = None, sequence_number: int | None = None) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason
        self.meter_id = meter_id
        self.session_id = session_id
        self.sequence_number = sequence_number


def register_secure_error_handlers(app: FastAPI) -> None:
    from edge_deployment.api.logging_config import get_logger, log_error_event

    logger = get_logger()

    @app.exception_handler(SecurityRejectionError)
    async def _security_rejection_handler(request: Request, exc: SecurityRejectionError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        log_error_event(logger, request_id=request_id, endpoint=request.url.path, method=request.method,
                         status_code=exc.status_code, error_type=f"security_{exc.reason}", meter_id=exc.meter_id)
        body = {
            "accepted": False, "security_status": "rejected", "security_rejection_reason": exc.reason,
            "meter_id": exc.meter_id, "session_id": exc.session_id, "sequence_number": exc.sequence_number,
            "detector_invoked": False, "detector_result": None, "request_id": request_id,
        }
        return JSONResponse(status_code=exc.status_code, content=body)
