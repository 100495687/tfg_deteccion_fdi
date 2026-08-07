"""Cliente 'directo' (ruta A del experimento de equivalencia): llama a
`DetectorEngine` sin pasar por HTTP en ningun momento. Referencia de comportamiento contra la
que se comparan la ruta B (TestClient) y la ruta C (Uvicorn real)."""
from __future__ import annotations

import time

import pandas as pd

from edge_deployment.core.detector_engine import DetectorEngine


class DirectEngineClient:
    def __init__(self, engine: DetectorEngine | None = None) -> None:
        self.engine = engine or DetectorEngine()

    def bootstrap(self, meter_id: str, historical_readings: pd.DataFrame) -> dict:
        return self.engine.bootstrap(meter_id, historical_readings)

    def ingest(self, meter_id: str, timestamp, power_kw: float) -> dict:
        t0 = time.perf_counter()
        resp = self.engine.ingest(meter_id, timestamp, power_kw)
        client_ms = (time.perf_counter() - t0) * 1000
        d = resp.to_dict()
        d["client_round_trip_ms"] = client_ms
        d["http_status"] = 200 if resp.accepted else 409
        return d

    def get_status(self, meter_id: str) -> dict:
        return self.engine.get_status(meter_id)

    def reset(self, meter_id: str) -> dict:
        return self.engine.reset(meter_id)
