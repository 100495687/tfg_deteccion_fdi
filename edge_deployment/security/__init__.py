"""Fase 5 -- capa anti-replay OPCIONAL y AISLADA, anterior a `DetectorEngine`.

Nada en este paquete modifica P, H, OR, `DetectorEngine`, `DetectorState`, ni ningun umbral o
artefacto congelado de Fases 1-4. Este paquete se puede eliminar por completo sin afectar al
detector original -- ver `edge_deployment/api/routes.py` (POST /readings, sin tocar) frente a
`edge_deployment/api/secure_routes.py` (POST /secure-readings, nuevo, Fase 5).
"""
