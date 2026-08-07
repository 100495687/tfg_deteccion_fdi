"""Fase 3: se ejecuta dentro del contenedor optimized (montado como fichero extra
de solo lectura, nunca horneado en la imagen -- no es parte del servicio, es un script de
diagnostico). Importa en el mismo orden que el arranque real de la API y mide RSS (psutil,
igual que /metrics) tras cada etapa. Imprime una linea JSON por etapa a stdout -- el host la
recoge desde `docker run` (sin -d, captura directa, sin pasar por `docker logs`).

Usa series sinteticas (mismo generador que edge_deployment/tests/conftest.py::synthetic_series,
duplicado aqui deliberadamente: es un fixture de pruebas, no logica del detector) en vez del
periodo pre-test real para las etapas de bootstrap/streaming -- la etapa de carga de P/H si usa
los artefactos y montajes reales (unica dependencia de datos que el motor tiene en el arranque).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import psutil

PROC = psutil.Process()


def _rss_mb() -> float:
    return PROC.memory_info().rss / 1e6


def _emit(stage: str, t0: float) -> None:
    print(json.dumps({"stage": stage, "rss_mb": round(_rss_mb(), 2), "t_elapsed_s": round(time.perf_counter() - t0, 3)}), flush=True)


def synthetic_series(n_minutes: int, start: str = "2020-01-01 00:00:00", seed: int = 0,
                      base: float = 1.0, amp: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_minutes, freq="1min")
    t = np.arange(n_minutes)
    rng = np.random.default_rng(seed)
    valores = base + amp * np.sin(2 * np.pi * t / 1440.0) + 0.05 * rng.standard_normal(n_minutes)
    valores = np.clip(valores, 0.05, None)
    from src.data_loading import COLUMNA_OBJETIVO
    return pd.DataFrame({COLUMNA_OBJETIVO: valores, "imputado": False}, index=idx)


def main() -> int:
    t0 = time.perf_counter()
    _emit("1_python_minimo", t0)

    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    import pydantic  # noqa: F401
    _emit("2_fastapi_uvicorn", t0)

    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    _emit("3_numpy_pandas_sklearn", t0)

    import torch  # noqa: F401
    _emit("4_pytorch", t0)

    from edge_deployment.core import model_loader as ml
    artefactos = ml.load_frozen_artifacts()
    _emit("5_6_p_y_h_cargados_via_load_frozen_artifacts", t0)
    # nota: en esta arquitectura P se carga antes que H dentro de load_frozen_artifacts() (ver
    # model_loader.py, sin modificar) y la carga de P incluye, inevitablemente, la lectura de
    # data_raw.csv + episodes_master_val.csv (para params_norm) -- no es posible separar
    # "P cargado" de "datos leidos" sin tocar ese codigo, así que se reportan como una sola etapa.

    from edge_deployment.core.detector_engine import DetectorEngine
    engine = DetectorEngine()
    _emit("7_detector_engine_construido", t0)

    from src.data_loading import COLUMNA_OBJETIVO
    bootstrap_df = synthetic_series(10230, start="2020-01-01 00:00:00", seed=1)
    meter_id = "mem_probe_meter"
    engine.bootstrap(meter_id, bootstrap_df)
    _emit("8_tras_bootstrap", t0)

    stream_start = bootstrap_df.index.max() + pd.Timedelta(minutes=1)
    stream_df = synthetic_series(3000, start=stream_start.isoformat(), seed=2)
    for i, (ts, row) in enumerate(stream_df.iterrows()):
        engine.ingest(meter_id, ts, float(row[COLUMNA_OBJETIVO]))
        if i in (499, 999, 1999, 2999):
            _emit(f"9_streaming_tras_{i+1}_lecturas", t0)

    engine.reset(meter_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
