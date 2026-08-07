"""Recolector de metricas acotado. Contadores simples + un reservoir
`deque(maxlen=...)` para latencias -- nunca se guardan todas las lecturas indefinidamente.
No usa Prometheus todavia (JSON basta para esta fase).
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99))}


class ApiMetrics:
    def __init__(self, reservoir_size: int = 10000) -> None:
        self._lock = threading.Lock()
        self.service_start_time = datetime.now(timezone.utc)
        self.model_load_time_ms = 0.0
        self.bootstrap_requests = 0
        self.reading_requests = 0
        self.status_requests = 0
        self.accepted_readings = 0
        self.rejected_readings = 0
        self.duplicate_identical = 0
        self.duplicate_conflicting = 0
        self.out_of_order = 0
        self.temporal_gaps = 0
        self.invalid_values = 0
        self.p_evaluations = 0
        self.h_evaluations = 0
        self.alerts_p = 0
        self.alerts_h = 0
        self.alerts_or = 0
        self.internal_errors = 0
        self._api_latencies: deque = deque(maxlen=reservoir_size)
        self._engine_latencies: deque = deque(maxlen=reservoir_size)
        self._status_codes: dict[int, int] = {}

    def record_status_code(self, code: int) -> None:
        with self._lock:
            self._status_codes[code] = self._status_codes.get(code, 0) + 1

    def record_bootstrap(self) -> None:
        with self._lock:
            self.bootstrap_requests += 1

    def record_status_request(self) -> None:
        with self._lock:
            self.status_requests += 1

    def record_error(self) -> None:
        with self._lock:
            self.internal_errors += 1

    def record_reading(self, resp, api_time_ms: float) -> None:
        rejection_invalid = {"nan_value", "infinite_value", "invalid_timestamp", "empty_meter_id", "non_numeric_power"}
        with self._lock:
            self.reading_requests += 1
            if resp.accepted:
                self.accepted_readings += 1
            else:
                self.rejected_readings += 1
                if resp.rejection_reason == "duplicate_identical":
                    self.duplicate_identical += 1
                elif resp.rejection_reason == "duplicate_conflicting":
                    self.duplicate_conflicting += 1
                elif resp.rejection_reason == "out_of_order":
                    self.out_of_order += 1
                elif resp.rejection_reason in rejection_invalid:
                    self.invalid_values += 1
            if resp.p_evaluated:
                self.p_evaluations += 1
            if resp.h_evaluated:
                self.h_evaluations += 1
            if resp.alert_p:
                self.alerts_p += 1
            if resp.alert_h:
                self.alerts_h += 1
            if resp.alert_or:
                self.alerts_or += 1
            if any("hueco" in w.lower() for w in resp.warnings):
                self.temporal_gaps += 1
            self._api_latencies.append(api_time_ms)
            self._engine_latencies.append(resp.processing_time_ms)

    def snapshot(self, n_active_meters: int, approximate_memory_mb: float | None) -> dict:
        with self._lock:
            api_lat = list(self._api_latencies)
            eng_lat = list(self._engine_latencies)
            status_codes = dict(self._status_codes)
            base = {
                "service_start_time": self.service_start_time, "model_load_time_ms": self.model_load_time_ms,
                "bootstrap_requests": self.bootstrap_requests, "reading_requests": self.reading_requests,
                "status_requests": self.status_requests, "accepted_readings": self.accepted_readings,
                "rejected_readings": self.rejected_readings, "duplicate_identical": self.duplicate_identical,
                "duplicate_conflicting": self.duplicate_conflicting, "out_of_order": self.out_of_order,
                "temporal_gaps": self.temporal_gaps, "invalid_values": self.invalid_values,
                "p_evaluations": self.p_evaluations, "h_evaluations": self.h_evaluations,
                "alerts_p": self.alerts_p, "alerts_h": self.alerts_h, "alerts_or": self.alerts_or,
                "internal_errors": self.internal_errors,
            }
        uptime = (datetime.now(timezone.utc) - self.service_start_time).total_seconds()
        api_p = _percentiles(api_lat)
        eng_p = _percentiles(eng_lat)
        return {
            **base, "uptime_seconds": uptime,
            "latency_api_mean_ms": api_p["mean"], "latency_api_p50_ms": api_p["p50"],
            "latency_api_p95_ms": api_p["p95"], "latency_api_p99_ms": api_p["p99"],
            "latency_engine_mean_ms": eng_p["mean"],
            "active_meter_states": n_active_meters, "approximate_memory_mb": approximate_memory_mb,
            "status_codes": status_codes,
        }
