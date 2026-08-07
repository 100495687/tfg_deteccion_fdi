"""Fase 5, seccion 18-19: metricas de seguridad, SEPARADAS de `edge_deployment/api/metrics.py`
(metricas del detector, sin modificar). Guarda latencias acotadas (`deque(maxlen=...)`, mismo
patron que `ApiMetrics` de Fase 2) y contadores globales por motivo de rechazo.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class SecurityMetrics:
    reservoir_size: int = 5000
    total_processed: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    rejected_messages_reaching_engine: int = 0  # invariante: debe permanecer en 0 (seccion 18)
    detector_state_changes_after_rejection: int = 0  # invariante: debe permanecer en 0
    _security_latency_ms: deque = field(default_factory=lambda: deque(maxlen=5000))
    _engine_latency_ms: deque = field(default_factory=lambda: deque(maxlen=5000))
    _total_latency_ms: deque = field(default_factory=lambda: deque(maxlen=5000))

    def record_accept(self, security_latency_ms: float, engine_latency_ms: float, total_latency_ms: float) -> None:
        self.total_processed += 1
        self.total_accepted += 1
        self._security_latency_ms.append(security_latency_ms)
        self._engine_latency_ms.append(engine_latency_ms)
        self._total_latency_ms.append(total_latency_ms)

    def record_reject(self, reason: str, security_latency_ms: float, total_latency_ms: float) -> None:
        self.total_processed += 1
        self.total_rejected += 1
        self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1
        self._security_latency_ms.append(security_latency_ms)
        self._total_latency_ms.append(total_latency_ms)

    def snapshot(self) -> dict:
        return {
            "total_processed": self.total_processed, "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected, "rejections_by_reason": dict(self.rejections_by_reason),
            "rejected_messages_reaching_engine": self.rejected_messages_reaching_engine,
            "detector_state_changes_after_rejection": self.detector_state_changes_after_rejection,
            "n_security_latency_samples": len(self._security_latency_ms),
        }

    def latency_samples_ms(self) -> dict[str, list[float]]:
        return {"security_ms": list(self._security_latency_ms), "engine_ms": list(self._engine_latency_ms),
                "total_ms": list(self._total_latency_ms)}
