"""Logging estructurado. Registra request_id/endpoint/status/meter_id/tiempos
-- nunca modelos, arrays completos, ventanas de 360 muestras ni las miles de lecturas de un
bootstrap. Los logs quedan en edge_deployment/results/logs/api.log.
"""
from __future__ import annotations

import logging
from pathlib import Path

EDGE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = EDGE_DIR / "results" / "logs"
LOGGER_NAME = "fdia_edge_api"

_configured = False


def get_logger(level: str = "INFO") -> logging.Logger:
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    if not _configured:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOGS_DIR / "api.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def log_reading_event(logger: logging.Logger, *, request_id: str, meter_id: str, timestamp, accepted: bool,
                       rejection_reason: str | None, p_evaluated: bool, h_evaluated: bool, alert_or,
                       engine_time_ms: float, api_time_ms: float, status_code: int) -> None:
    logger.info(
        "request_id=%s endpoint=/readings method=POST status_code=%s meter_id=%s timestamp=%s "
        "accepted=%s rejection_reason=%s p_evaluated=%s h_evaluated=%s alert_or=%s "
        "engine_time_ms=%.3f api_time_ms=%.3f",
        request_id, status_code, meter_id, timestamp, accepted, rejection_reason,
        p_evaluated, h_evaluated, alert_or, engine_time_ms, api_time_ms,
    )


def log_error_event(logger: logging.Logger, *, request_id: str, endpoint: str, method: str, status_code: int,
                     error_type: str, meter_id: str | None = None) -> None:
    logger.warning(
        "request_id=%s endpoint=%s method=%s status_code=%s meter_id=%s error_type=%s",
        request_id, endpoint, method, status_code, meter_id, error_type,
    )


def log_generic_event(logger: logging.Logger, *, request_id: str, endpoint: str, method: str, status_code: int) -> None:
    logger.info("request_id=%s endpoint=%s method=%s status_code=%s", request_id, endpoint, method, status_code)
