"""Fase 4: version autonoma de _memory_stage_probe.py (Fase 3). Se ejecuta dentro
de un contenedor optimized nuevo, sin ningun volumen montado (a diferencia de la sonda de
Fase 3, que llamaba a `load_frozen_artifacts()` legacy explicitamente en la etapa 5-6). Aqui
la etapa 5-6 llama a `load_frozen_artifacts_autonomous()` -- nunca abre data_raw.csv ni
episodes_master_val.csv. Mismo patron de emision de una linea JSON por etapa a stdout.
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
    artefactos = ml.load_frozen_artifacts_autonomous()  # Fase 4: nunca abre CSV
    _emit("5_6_p_y_h_cargados_via_load_frozen_artifacts_autonomous", t0)

    from edge_deployment.core.detector_engine import DetectorEngine
    engine = DetectorEngine(artifact_manifest=artefactos)  # reutiliza lo ya cargado, no recarga
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
