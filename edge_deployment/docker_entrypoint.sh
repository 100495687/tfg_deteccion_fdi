#!/bin/sh
# Fase 3: plomeria de contenedor UNICAMENTE -- no toca edge_deployment/api ni
# edge_deployment/core. api/logging_config.py (Fase 2, sin modificar) escribe en
# edge_deployment/results/logs/api.log via FileHandler; este script lo vuelca tambien a
# stdout (requisito "logs a stdout" de la imagen optimized) con `tail -F`, que espera a que
# el fichero exista si el arranque aun no ha escrito la primera linea.
set -e
mkdir -p edge_deployment/results/logs
tail -F edge_deployment/results/logs/api.log 2>/dev/null &
exec uvicorn edge_deployment.api.main:app --host 0.0.0.0 --port 8000 --workers 1
