"""Dependencias FastAPI: acceso a los objetos unicos creados en el lifespan y al
request_id asignado por el middleware. Nunca construyen nada nuevo -- solo leen
`app.state`."""
from __future__ import annotations

import uuid

from fastapi import Request

from edge_deployment.api.error_handlers import NotReadyError
from edge_deployment.core.detector_engine import DetectorEngine


def new_request_id(existing: str | None) -> str:
    return existing if existing else str(uuid.uuid4())


def get_engine(request: Request) -> DetectorEngine:
    return request.app.state.engine


def get_metrics(request: Request):
    return request.app.state.metrics


def get_locks(request: Request):
    return request.app.state.locks


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def require_ready(request: Request) -> None:
    if not getattr(request.app.state, "ready", False):
        raise NotReadyError("El motor todavia no esta preparado (ready=false)")
