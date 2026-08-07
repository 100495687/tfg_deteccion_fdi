"""Fase 5, seccion 9: interfaz `KeyProvider` + 3 implementaciones. Reglas estrictas (seccion 9,
verificadas por tests -- ver test_phase5_anti_replay.py): ninguna clave en Git, ninguna clave
en la imagen Docker, ninguna clave por defecto, ninguna clave en logs, ninguna clave en
respuestas HTTP, ninguna clave dentro de `AntiReplayState`, ninguna clave en
`context_manifest.json` ni en `artifact_manifest_v2.json` (ningun fichero de Fase 5 se anade a
esos manifiestos).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class KeyProvider(ABC):
    @abstractmethod
    def get_key(self, meter_id: str) -> bytes | None:
        """Devuelve la clave HMAC (bytes) para `meter_id`, o `None` si no hay clave
        aprovisionada -- nunca lanza, nunca genera una clave por defecto."""


class EnvironmentKeyProvider(KeyProvider):
    """Lee `{key_env_prefix}{meter_id}` como variable de entorno. El valor se trata como
    secreto UTF-8 arbitrario (no requiere codificacion hexadecimal) -- consistente con el
    ejemplo de configuracion de la seccion 9 (`key_env_prefix: FDIA_METER_KEY_`)."""

    def __init__(self, key_env_prefix: str = "FDIA_METER_KEY_") -> None:
        self.key_env_prefix = key_env_prefix

    def get_key(self, meter_id: str) -> bytes | None:
        value = os.environ.get(f"{self.key_env_prefix}{meter_id}")
        if value is None or value == "":
            return None
        return value.encode("utf-8")


class FileKeyProvider(KeyProvider):
    """Lee `{keys_dir}/{meter_id}.key` (un fichero por medidor, montado como volumen de solo
    lectura en despliegue real -- nunca dentro de `docker_context/`, nunca en el repositorio)."""

    def __init__(self, keys_dir: Path) -> None:
        self.keys_dir = Path(keys_dir)

    def get_key(self, meter_id: str) -> bytes | None:
        path = self.keys_dir / f"{meter_id}.key"
        if not path.exists():
            return None
        content = path.read_bytes().strip()
        return content or None


class InMemoryTestKeyProvider(KeyProvider):
    """SOLO para tests/experimentos -- nunca en produccion (ni la API ni Docker la
    instancian; ver `api/security_lifecycle.py`, Fase 5 Gate 2)."""

    def __init__(self, keys: dict[str, bytes] | None = None) -> None:
        self._keys: dict[str, bytes] = dict(keys) if keys else {}

    def set_key(self, meter_id: str, key: bytes) -> None:
        self._keys[meter_id] = key

    def get_key(self, meter_id: str) -> bytes | None:
        return self._keys.get(meter_id)
