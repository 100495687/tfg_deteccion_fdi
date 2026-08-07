from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))

from edge_deployment.core.detector_engine import DetectorEngine  # noqa: E402
from edge_deployment.core.detector_state import MAX_LAG_H_BINS, H_BOOTSTRAP_MARGIN_BINS  # noqa: E402
from src.data_loading import COLUMNA_OBJETIVO  # noqa: E402

BOOTSTRAP_BINS = MAX_LAG_H_BINS + H_BOOTSTRAP_MARGIN_BINS
BOOTSTRAP_MINUTES = BOOTSTRAP_BINS * 15  # 10230


def synthetic_series(n_minutes: int, start: str = "2020-01-01 00:00:00", seed: int = 0,
                      base: float = 1.0, amp: float = 0.5) -> pd.DataFrame:
    """Serie sintetica 1-min, continua y sin NaN -- suficiente para ejercitar el motor sin
    depender del dataset real (rapido, deterministico)."""
    idx = pd.date_range(start, periods=n_minutes, freq="1min")
    t = np.arange(n_minutes)
    rng = np.random.default_rng(seed)
    valores = base + amp * np.sin(2 * np.pi * t / 1440.0) + 0.05 * rng.standard_normal(n_minutes)
    valores = np.clip(valores, 0.05, None)
    return pd.DataFrame({COLUMNA_OBJETIVO: valores, "imputado": False}, index=idx)


@pytest.fixture(scope="session")
def engine() -> DetectorEngine:
    return DetectorEngine()


@pytest.fixture(scope="session")
def bootstrap_series() -> pd.DataFrame:
    return synthetic_series(BOOTSTRAP_MINUTES, start="2020-01-01 00:00:00", seed=1)


@pytest.fixture
def meter_id(request) -> str:
    return f"meter_{request.node.name}"


@pytest.fixture
def bootstrapped(engine, bootstrap_series, meter_id):
    """Motor con un contador recien bootstrapeado (H ready, P todavia warming_up) listo para
    empezar a recibir streaming."""
    report = engine.bootstrap(meter_id, bootstrap_series)
    stream_start = bootstrap_series.index.max() + pd.Timedelta(minutes=1)
    yield {"engine": engine, "meter_id": meter_id, "report": report, "stream_start": stream_start}
    engine.reset(meter_id)


# ==========================================================================================
# Fase 2 (API) -- fixtures compartidas por edge_deployment/tests/test_api_*.py
# ==========================================================================================

@pytest.fixture(scope="session")
def api_client():
    """TestClient con el lifespan real ya ejecutado (modelos cargados una vez para toda la
    sesion de tests) -- evita recargar P/H en cada test."""
    from fastapi.testclient import TestClient

    from edge_deployment.api.main import app
    with TestClient(app) as client:
        yield client


def bootstrap_readings_payload(meter_id: str, df: pd.DataFrame) -> dict:
    return {"meter_id": meter_id, "readings": [
        {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(df.index, df[COLUMNA_OBJETIVO])
    ]}


@pytest.fixture
def preserve_startup_reports():
    """Algunos tests (p.ej. simular un artefacto ausente) hacen que una app FastAPI aislada
    ejecute su propio lifespan, que escribe los mismos ficheros compartidos
    (`api_startup_report.json`, `api_artifact_validation.json`) que la app real de la sesion.
    Este fixture guarda su contenido byte a byte antes del test y lo restaura despues,
    independientemente de lo que el test haga -- evita que el orden de ejecucion de pytest
    deje esos reportes compartidos en un estado simulado de fallo para otros tests."""
    from edge_deployment.api.lifecycle import REPORTS_DIR
    rutas = [REPORTS_DIR / "api_startup_report.json", REPORTS_DIR / "api_artifact_validation.json"]
    originales = {r: (r.read_text(encoding="utf-8") if r.exists() else None) for r in rutas}
    yield
    for r, contenido in originales.items():
        if contenido is not None:
            r.write_text(contenido, encoding="utf-8")


@pytest.fixture
def api_bootstrapped(api_client, bootstrap_series, meter_id):
    """Contador bootstrapeado a traves de la API (no del motor directo) -- usado por los
    tests de /readings, /status, /metrics, /reset."""
    r = api_client.post("/bootstrap", json=bootstrap_readings_payload(meter_id, bootstrap_series))
    assert r.status_code == 200, r.text
    stream_start = bootstrap_series.index.max() + pd.Timedelta(minutes=1)
    yield {"client": api_client, "meter_id": meter_id, "report": r.json(), "stream_start": stream_start}
    api_client.post(f"/reset/{meter_id}")
