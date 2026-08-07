"""Aplicacion FastAPI (Fase 2). Envuelve `DetectorEngine` (Fase 1, sin modificar su logica
matematica) como microservicio HTTP con un unico worker (seccion 7 -- el estado temporal en
memoria no es compartible entre procesos):

    uvicorn edge_deployment.api.main:app --host 127.0.0.1 --port 8000 --workers 1
"""
from __future__ import annotations

from fastapi import FastAPI, Request

from edge_deployment.api.dependencies import new_request_id
from edge_deployment.api.error_handlers import register_error_handlers
from edge_deployment.api.lifecycle import SERVICE_NAME, SERVICE_VERSION, lifespan
from edge_deployment.api.routes import router
from edge_deployment.api.secure_error_handlers import register_secure_error_handlers
from edge_deployment.api.secure_routes import secure_router

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)
register_error_handlers(app)
app.include_router(router)

# Fase 5: ruta PARALELA y opcional (POST /secure-readings, /security/*) -- registrada
# ADEMAS del router original, nunca en su lugar. Eliminar estas dos lineas (y
# edge_deployment/security/) basta para volver a la Fase 4 funcionalmente intacta.
register_secure_error_handlers(app)
app.include_router(secure_router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Asigna un request_id por peticion (seccion 12): usa X-Request-ID si el cliente lo
    envia, si no genera un UUID. Se guarda en request.state para que routes.py y
    error_handlers.py lo usen sin recalcularlo."""
    request.state.request_id = new_request_id(request.headers.get("X-Request-ID"))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response
