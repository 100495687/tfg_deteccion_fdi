"""Control de concurrencia por meter_id.

Los endpoints se declaran `def` (sincronos) -- Starlette/FastAPI los ejecuta en
un threadpool, por lo que dos peticiones concurrentes para el mismo meter_id pueden correr en
hilos distintos de verdad. Un `threading.Lock` por meter_id evita que dos hilos muten el mismo
`DetectorState` a la vez (bootstrap/ingest/reset son las secciones criticas); health/ready/
metrics no necesitan lock porque no tocan estado por-meter.
"""
from __future__ import annotations

import threading


class MeterLockRegistry:
    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def get_lock(self, meter_id: str) -> threading.Lock:
        with self._registry_lock:
            lock = self._locks.get(meter_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[meter_id] = lock
            return lock

    def n_registered(self) -> int:
        with self._registry_lock:
            return len(self._locks)
