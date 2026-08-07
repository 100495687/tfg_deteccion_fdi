"""Fase 5: inicializacion del componente OPCIONAL de seguridad -- unica pieza que
`edge_deployment/api/lifecycle.py` (Fase 1-4, intacto salvo una linea que llama a
`init_security`) tiene permitido invocar. Nunca falla el arranque del detector: si la
seguridad esta desactivada o mal configurada, el detector sigue estando `ready` igual que en
Fase 4 -- solo `/secure-readings` queda inoperante (503).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import FastAPI

from edge_deployment.security.anti_replay import AntiReplayGuard
from edge_deployment.security.key_provider import EnvironmentKeyProvider, KeyProvider
from edge_deployment.security.security_clock import Clock, SimulatedClock, SystemClock
from edge_deployment.security.security_metrics import SecurityMetrics
from edge_deployment.security.security_state import AntiReplayState

EDGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = EDGE_DIR / "config" / "api.yaml"


def _load_security_config(config_path: Path) -> dict:
    defaults = {"anti_replay_enabled": False, "key_provider": "environment", "key_env_prefix": "FDIA_METER_KEY_",
                "max_message_age_seconds": 300, "max_future_skew_seconds": 60, "clock_mode": "system",
                "test_endpoints_enabled": False}
    if not config_path.exists():
        return defaults
    with open(config_path, encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    section = full.get("security", {}) or {}
    defaults.update({k: v for k, v in section.items() if k in defaults})

    # Override por entorno (mismo patron que EnvironmentKeyProvider) -- practico para
    # Docker/tests sin editar el YAML; el valor por defecto sigue siendo `false` (seccion 6).
    env_enabled = os.environ.get("FDIA_ANTI_REPLAY_ENABLED")
    if env_enabled is not None:
        defaults["anti_replay_enabled"] = env_enabled.strip().lower() in ("1", "true", "yes")
    env_clock_mode = os.environ.get("FDIA_SECURITY_CLOCK_MODE")
    if env_clock_mode is not None:
        defaults["clock_mode"] = env_clock_mode.strip().lower()
    # Correccion obligatoria post-aceptacion: los endpoints /security/test/* (init-session,
    # clock) son un mecanismo de administracion/prueba (seccion 14) que NUNCA debe quedar
    # accesible en un despliegue de produccion -- independiente de `anti_replay_enabled`
    # (que protege /secure-readings, no estos dos endpoints de administracion). Por defecto
    # `false` SIEMPRE; solo los scripts de experimentos de esta misma fase lo activan
    # explicitamente via esta variable de entorno, nunca vía el YAML de produccion.
    env_test_endpoints = os.environ.get("FDIA_SECURITY_TEST_ENDPOINTS_ENABLED")
    if env_test_endpoints is not None:
        defaults["test_endpoints_enabled"] = env_test_endpoints.strip().lower() in ("1", "true", "yes")
    return defaults


def _build_key_provider(config: dict) -> KeyProvider:
    if config["key_provider"] == "environment":
        return EnvironmentKeyProvider(key_env_prefix=config["key_env_prefix"])
    raise ValueError(f"key_provider no soportado: {config['key_provider']!r} (solo 'environment' esta cableado en la API; "
                      f"FileKeyProvider/InMemoryTestKeyProvider se usan directamente en tests/experimentos)")


def _build_clock(config: dict) -> Clock:
    if config.get("clock_mode") == "simulated":
        return SimulatedClock()
    return SystemClock()


def init_security(app: FastAPI, config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Se llama UNA vez desde `lifecycle.lifespan()`, siempre -- construye el componente
    incluso si `anti_replay_enabled=false` (para que `/security/status` pueda informar de
    ello), pero `/secure-readings` comprueba el flag en cada peticion, no aqui."""
    config = _load_security_config(config_path)

    app.state.anti_replay_enabled = bool(config["anti_replay_enabled"])
    app.state.security_test_endpoints_enabled = bool(config["test_endpoints_enabled"])
    app.state.security_config = config
    app.state.security_key_provider = _build_key_provider(config)
    app.state.security_clock = _build_clock(config)
    app.state.security_state = AntiReplayState()
    app.state.security_metrics = SecurityMetrics()
    app.state.security_guard = AntiReplayGuard(
        key_provider=app.state.security_key_provider, clock=app.state.security_clock,
        state=app.state.security_state, max_message_age_seconds=config["max_message_age_seconds"],
        max_future_skew_seconds=config["max_future_skew_seconds"],
    )
    return config
