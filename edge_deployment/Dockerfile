# Fase 3 -- imagen OPTIMIZED (fdia-edge:optimized). Aplica UNICAMENTE optimizaciones de
# empaquetado sobre la misma base que Dockerfile.baseline: NO cambia el modelo, los umbrales
# ni la precision numerica (mismo edge_deployment/docker_context/, copiado sin modificar).
#
# Optimizaciones aplicadas (seccion 2 del encargo):
#   - python:3.14-slim (misma version que Dockerfile.baseline, sin cambiar de familia)
#   - pip install --no-cache-dir + limpieza de bytecode de las dependencias
#   - usuario no-root (fdia) para ejecutar el proceso
#   - codigo y modelos copiados SIN permiso de escritura para ese usuario -- solo
#     edge_deployment/results/{reports,logs} son escribibles (ahi es donde la API regenera
#     reportes de arranque/logs en cada arranque, Fase 1/2 sin modificar)
#   - un unico worker (--workers 1, igual que baseline -- Fase 2)
#   - HEALTHCHECK sobre /health
#   - logs tambien a stdout (docker_entrypoint.sh, plomeria de contenedor, no toca api/core)
#   - sin herramientas de desarrollo, sin datasets ni resultados
#
# Fase 4 (autonomia de artefactos): `edge_deployment/models/` (params_norm_p.joblib,
# artifact_manifest_v2.json) pasa a ser de SOLO LECTURA -- el autonomous_loader (por defecto)
# ya no regenera ningun manifiesto en el arranque (a diferencia de Fase 1-3, que
# reescribian edge_deployment/models/artifact_manifest.json y
# results/tables/input_artifacts_inventory.csv en cada arranque via la ruta legacy). Tampoco
# se crean puntos de montaje para data/raw ni results/tables: el contenedor de produccion no
# los necesita (ver docker_context/RUNTIME_MOUNTS.md).
#
# Build (contexto = edge_deployment/docker_context/, generado por build_docker_context.py):
#   python -m edge_deployment.build_docker_context
#   docker build -f edge_deployment/Dockerfile -t fdia-edge:optimized edge_deployment/docker_context

FROM python:3.14-slim

WORKDIR /app

COPY requirements-edge.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    -r requirements-edge.txt \
    && find /usr/local/lib -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib -name "*.pyc" -delete

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY edge_deployment/ ./edge_deployment/
COPY docker_entrypoint.sh ./docker_entrypoint.sh

# Unicos directorios de escritura en tiempo de ejecucion: reportes de arranque y logs.
# Codigo y TODOS los modelos/artefactos congelados (incluye edge_deployment/models/, Fase 4:
# params_norm_p.joblib, artifact_manifest_v2.json) quedan de solo lectura para el usuario de
# ejecucion -- ningun comentario dentro del bloque RUN (la continuacion con `\` mezclada con
# lineas `#` es fragil de parsear, evitado deliberadamente tras un fallo de build real).
RUN groupadd --system fdia \
    && useradd --system --gid fdia --home-dir /app --shell /usr/sbin/nologin fdia \
    && chmod +x docker_entrypoint.sh \
    && mkdir -p edge_deployment/results/reports edge_deployment/results/logs \
    && chown -R fdia:fdia edge_deployment/results \
    && chown -R root:root src configs models edge_deployment/api edge_deployment/core \
                          edge_deployment/config edge_deployment/models edge_deployment/security \
                          edge_deployment/__init__.py \
    && chmod -R a-w src configs models edge_deployment/api edge_deployment/core \
                     edge_deployment/config edge_deployment/models edge_deployment/security \
                     edge_deployment/__init__.py

USER fdia

# matplotlib (import transitivo incondicional, ver docker_context, nunca dibuja nada en
# tiempo de ejecucion) intenta crear su cache en $HOME/.config, no escribible para `fdia` --
# se redirige a /tmp (escribible) para evitar el warning, sin afectar ningun resultado.
ENV MPLCONFIGDIR=/tmp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["./docker_entrypoint.sh"]
