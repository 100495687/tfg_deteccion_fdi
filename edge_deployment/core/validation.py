"""Validacion de lecturas antes de que modifiquen cualquier estado.

Politica de potencia negativa ("si el pipeline la considera invalida"): el
pipeline offline (`src/data_loading.py::cargar_y_limpiar`) nunca filtra valores negativos --
solo interpola NaN. Por tanto, para ser equivalente al pipeline offline, este modulo no
rechaza potencia negativa por si sola (se documenta explicitamente, no es un olvido).

Una lectura rechazada nunca modifica ningun buffer ni estado.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from edge_deployment.core.schemas import Reading

# clasificacion, no rechazo -- documentado explicitamente (ver docstring del modulo)
REJECT_NEGATIVE_POWER = False


@dataclass
class ValidationResult:
    accepted: bool
    rejection_reason: str | None
    classification: str  # "ok" | "duplicate_identical" | "duplicate_conflicting" | "out_of_order" | ...


def validate_reading(reading: Reading, last_accepted_timestamp, last_accepted_value: float | None) -> ValidationResult:
    if reading.timestamp is None or pd.isna(reading.timestamp):
        return ValidationResult(False, "invalid_timestamp", "invalid_timestamp")
    ts = pd.Timestamp(reading.timestamp)

    if reading.meter_id is None or str(reading.meter_id).strip() == "":
        return ValidationResult(False, "empty_meter_id", "empty_meter_id")

    if reading.power_kw is None or isinstance(reading.power_kw, bool):
        return ValidationResult(False, "non_numeric_power", "non_numeric_power")
    try:
        val = float(reading.power_kw)
    except (TypeError, ValueError):
        return ValidationResult(False, "non_numeric_power", "non_numeric_power")

    if math.isnan(val):
        return ValidationResult(False, "nan_value", "nan_value")
    if math.isinf(val):
        return ValidationResult(False, "infinite_value", "infinite_value")
    if REJECT_NEGATIVE_POWER and val < 0:
        return ValidationResult(False, "negative_power", "negative_power")

    if last_accepted_timestamp is not None:
        last_ts = pd.Timestamp(last_accepted_timestamp)
        if ts == last_ts:
            if last_accepted_value is not None and val == float(last_accepted_value):
                return ValidationResult(False, "duplicate_identical", "duplicate_identical")
            return ValidationResult(False, "duplicate_conflicting", "duplicate_conflicting")
        if ts < last_ts:
            return ValidationResult(False, "out_of_order", "out_of_order")

    return ValidationResult(True, None, "ok")
