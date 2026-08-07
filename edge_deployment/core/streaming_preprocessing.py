"""Buffer de 1 minuto y agregacion causal a 15 minutos.

La agregacion online debe reproducir exactamente la agregacion offline
(`predictor_causal_lags.construir_serie_15min`): bins de 15 muestras consecutivas sin
solape, valor = media aritmetica, timestamp del bin = timestamp de la ultima muestra
(`ts_end`), sin redondeo ni conversion de unidades, sin zona horaria (timestamps naive,
igual que el resto del pipeline). Los bins son posicionales (la muestra 0 de la particion
pasada, no un anclaje a :00/:15/:30/:45 de reloj) -- ver
`edge_deployment/results/reports/resampling_semantics.json`.

Un bucket de 15 minutos solo se cierra cuando se han recibido sus 15 lecturas
consecutivas (sin huecos). Un hueco descarta el bucket parcial en curso (nunca se
rellena con cero ni se interpola) y reinicia la acumulacion desde la siguiente lectura
aceptada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edge_deployment.core.detector_state import Bin15Min, DetectorState, P_WINDOW_MIN, RESOLUCION_H_MIN
from src.data_loading import COLUMNA_OBJETIVO

BIN_SIZE = RESOLUCION_H_MIN  # 15


def append_reading(state: DetectorState, timestamp: pd.Timestamp, value: float) -> dict:
    """Anade una lectura ya validada y ya aceptada (ver validation.py) al estado. Devuelve
    informacion de lo ocurrido (hueco detectado, bin cerrado) para que el motor la use en
    la respuesta."""
    gap_detected = False
    if state.last_accepted_timestamp is not None:
        esperado = pd.Timestamp(state.last_accepted_timestamp) + pd.Timedelta(minutes=1)
        if pd.Timestamp(timestamp) != esperado:
            gap_detected = True
            state.n_gaps += 1
            state.degraded = True
            # el bucket de 15 min parcial en curso queda invalidado -- nunca se cierra con huecos
            state.open_bucket_values = []
            state.recent_warnings.append(f"hueco temporal detectado antes de {timestamp}: bucket de 15min en curso descartado")

    state.buffer_1min.append((pd.Timestamp(timestamp), float(value)))
    state.last_accepted_timestamp = pd.Timestamp(timestamp)
    state.last_accepted_value = float(value)
    state.n_accepted += 1
    state.stream_minutes_ingested += 1

    state.open_bucket_values.append((pd.Timestamp(timestamp), float(value)))
    bin_cerrado = None
    if len(state.open_bucket_values) == BIN_SIZE:
        ts_start = state.open_bucket_values[0][0]
        ts_end = state.open_bucket_values[-1][0]
        kw = float(np.mean([v for _, v in state.open_bucket_values]))
        bin_cerrado = Bin15Min(ts_start=ts_start, ts_end=ts_end, kw=kw)
        state.historial_15min.append(bin_cerrado)
        state.open_bucket_values = []
        state.open_bucket_index += 1

    if state.degraded and has_complete_window(state, P_WINDOW_MIN):
        state.degraded = False

    return {"gap_detected": gap_detected, "bin_closed": bin_cerrado}


def validate_continuity(entries: list[tuple]) -> bool:
    """True si una lista de (timestamp, valor) es estrictamente contigua (paso de 1 minuto)."""
    if len(entries) < 2:
        return True
    ts = [t for t, _ in entries]
    return all((ts[i] - ts[i - 1]) == pd.Timedelta(minutes=1) for i in range(1, len(ts)))


def has_complete_window(state: DetectorState, n_minutes: int) -> bool:
    """True si el buffer tiene al menos `n_minutes` lecturas y las ultimas `n_minutes` son
    contiguas (sin huecos) -- asi P vuelve a estar ready automaticamente en cuanto se
    acumula una ventana completa tras un hueco. Delega en
    `DetectorState.has_complete_window` (misma logica que usa `state.p_ready()`)."""
    return state.has_complete_window(n_minutes)


def get_window(state: DetectorState, n_minutes: int) -> pd.DataFrame | None:
    """Devuelve las ultimas `n_minutes` lecturas como DataFrame compatible con
    `windowing.ventanear` (columnas COLUMNA_OBJETIVO + imputado, DatetimeIndex), o None si
    no hay ventana completa disponible."""
    if not has_complete_window(state, n_minutes):
        return None
    cola = list(state.buffer_1min)[-n_minutes:]
    idx = pd.DatetimeIndex([t for t, _ in cola])
    valores = [v for _, v in cola]
    return pd.DataFrame({COLUMNA_OBJETIVO: valores, "imputado": False}, index=idx)


def historial_contiguous_tail(state: DetectorState, n_bins: int) -> bool:
    """True si los ultimos `n_bins` agregados de 15 min son contiguos (cada uno de exactamente
    15 min, y el siguiente empieza justo 1 min despues del anterior). Delega en
    `DetectorState.historial_contiguous_tail` (misma logica que usa `state.h_ready()`)."""
    return state.historial_contiguous_tail(n_bins)


def trim_history(state: DetectorState) -> None:
    """Las deques con `maxlen` ya se autolimitan; esta funcion solo protege el bucket
    parcial en curso (nunca deberia superar BIN_SIZE, pero se recorta defensivamente)."""
    if len(state.open_bucket_values) > BIN_SIZE:
        state.open_bucket_values = state.open_bucket_values[-BIN_SIZE:]


def historial_15min_to_arrays(state: DetectorState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serializa `historial_15min` (deque de Bin15Min) a tres arrays paralelos (kw, ts_start,
    ts_end), en orden cronologico -- entrada de `predictor_causal_lags.construir_features_kw`."""
    bins = list(state.historial_15min)
    kw = np.array([b.kw for b in bins], dtype=np.float64)
    ts_start = np.array([b.ts_start for b in bins], dtype="datetime64[ns]")
    ts_end = np.array([b.ts_end for b in bins], dtype="datetime64[ns]")
    return kw, ts_start, ts_end
