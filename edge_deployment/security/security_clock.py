"""Fase 5, seccion 11: reloj inyectable -- nunca `datetime.now()` directamente dentro de la
logica de dominio de seguridad. `SystemClock` para despliegue real; `SimulatedClock` para
tests y para reproducir el periodo pre-test historico sin desincronizar el reloj del sistema
(las lecturas de 2009 siempre pareceran "stale" frente a un reloj real -- por eso los
experimentos de equivalencia/replay usan SimulatedClock, nunca SystemClock).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime:
        """Debe devolver un datetime CON tzinfo (UTC)."""


class SystemClock(Clock):
    """Reloj real del edge -- el tiempo de recepcion procede de este reloj, nunca de un campo
    elegido por el cliente (seccion 11)."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class SimulatedClock(Clock):
    """Reloj controlado manualmente -- para tests y para reproducir datos historicos
    (periodo pre-test) sin tocar el reloj del sistema operativo."""

    def __init__(self, start: datetime | None = None) -> None:
        self._current = start or datetime(1970, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._current

    def set(self, dt: datetime) -> None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        self._current = dt

    def advance(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)
