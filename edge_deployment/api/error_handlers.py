"""Mapeo de errores a HTTP. Nunca devuelve stack trace, rutas absolutas ni
contenido de modelos al cliente -- esos detalles solo van a los logs.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from edge_deployment.api.logging_config import get_logger, log_error_event


class ApiError(Exception):
    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, meter_id: str | None = None, timestamp=None, state_modified: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.meter_id = meter_id
        self.timestamp = timestamp
        self.state_modified = state_modified


class NotReadyError(ApiError):
    status_code = 503
    error_code = "engine_not_ready"


class NotFoundError(ApiError):
    status_code = 404
    error_code = "meter_not_found"


class ConflictError(ApiError):
    status_code = 409

    def __init__(self, message: str, error_code: str = "conflict", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.error_code = error_code


class UnprocessableError(ApiError):
    status_code = 422
    error_code = "invalid_structure"


def _error_body(error_code: str, message: str, request: Request, meter_id: str | None, timestamp, state_modified: bool) -> dict:
    request_id = getattr(request.state, "request_id", "unknown")
    return {
        "error_code": error_code, "message": message, "meter_id": meter_id,
        "timestamp": str(timestamp) if timestamp is not None else None,
        "state_modified": state_modified, "request_id": request_id,
    }


def register_error_handlers(app: FastAPI) -> None:
    logger = get_logger()

    @app.exception_handler(ApiError)
    async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        if exc.status_code >= 500:
            app.state.metrics.record_error()
        log_error_event(logger, request_id=getattr(request.state, "request_id", "unknown"),
                         endpoint=request.url.path, method=request.method, status_code=exc.status_code,
                         error_type=exc.error_code, meter_id=exc.meter_id)
        return JSONResponse(status_code=exc.status_code,
                             content=_error_body(exc.error_code, exc.message, request, exc.meter_id, exc.timestamp, exc.state_modified))

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        log_error_event(logger, request_id=getattr(request.state, "request_id", "unknown"),
                         endpoint=request.url.path, method=request.method, status_code=422, error_type="validation_error")
        errores = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422,
                             content=_error_body("validation_error", errores, request, None, None, False))

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        log_error_event(logger, request_id=getattr(request.state, "request_id", "unknown"),
                         endpoint=request.url.path, method=request.method, status_code=exc.status_code, error_type="http_error")
        return JSONResponse(status_code=exc.status_code,
                             content=_error_body("http_error", str(exc.detail), request, None, None, False))

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        app.state.metrics.record_error()
        logger.exception("request_id=%s unhandled exception on %s %s",
                          getattr(request.state, "request_id", "unknown"), request.method, request.url.path)
        return JSONResponse(status_code=500,
                             content=_error_body("internal_error", "An internal error occurred", request, None, None, False))
