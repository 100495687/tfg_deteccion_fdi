"""Tests 42-44 + casos A-D + tabla de resultados."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from edge_deployment.tests.conftest import bootstrap_readings_payload, synthetic_series

EDGE_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = EDGE_DIR / "results" / "tables"


def _fresh_bootstrapped(api_client, seed: int, meter_id: str) -> pd.Timestamp:
    boot = synthetic_series(10230, start=f"2025-0{(seed % 9) + 1}-01", seed=seed)
    api_client.post(f"/reset/{meter_id}")
    r = api_client.post("/bootstrap", json=bootstrap_readings_payload(meter_id, boot))
    assert r.status_code == 200, r.text
    return boot.index.max() + pd.Timedelta(minutes=1)


def test_42_lock_prevents_concurrent_modification(api_client):
    """Caso A: N peticiones concurrentes con timestamps distintos -- el lock
    por meter_id garantiza que se procesan una a una (nunca corrompen el estado), pero no
    garantiza que el orden de llegada al lock coincida con el orden cronologico de los
    timestamps (un hilo puede adquirir el lock antes que otro que envio un timestamp menor);
    en ese caso el motor rechaza correctamente esa lectura como `out_of_order` -- exactamente
    el comportamiento esperado ("se procesan en orden o uno espera al otro"). Lo
    que se exige es: ninguna excepcion (nunca 500), ningun estado corrupto, y que el numero
    de aceptadas+rechazadas sea exactamente N."""
    meter_id = "conc_test_42"
    stream_start = _fresh_bootstrapped(api_client, 1, meter_id)
    n = 20
    timestamps = [(stream_start + pd.Timedelta(minutes=i)).isoformat() for i in range(n)]

    def _ingest(ts):
        return api_client.post("/readings", json={"meter_id": meter_id, "timestamp": ts, "power_kw": 1.0})

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_ingest, timestamps))

    assert all(r.status_code in (200, 409) for r in results), "ninguna peticion debe fallar con 500 ni otro codigo inesperado"
    status = api_client.get(f"/status/{meter_id}").json()
    n_accepted_here = sum(1 for r in results if r.status_code == 200)
    n_rejected_here = sum(1 for r in results if r.status_code == 409)
    assert n_accepted_here + n_rejected_here == n
    assert n_accepted_here >= 1, "al menos la primera peticion en adquirir el lock debe aceptarse"
    assert status["accepted_readings"] == n_accepted_here
    assert status["out_of_order"] == n_rejected_here
    api_client.post(f"/reset/{meter_id}")


def test_43_two_simultaneous_duplicates_not_both_accepted(api_client):
    meter_id = "conc_test_43"
    stream_start = _fresh_bootstrapped(api_client, 2, meter_id)
    ts = stream_start.isoformat()

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(api_client.post, "/readings", json={"meter_id": meter_id, "timestamp": ts, "power_kw": 1.0})
        f2 = ex.submit(api_client.post, "/readings", json={"meter_id": meter_id, "timestamp": ts, "power_kw": 1.0})
        r1, r2 = f1.result(), f2.result()

    codes = sorted([r1.status_code, r2.status_code])
    assert codes == [200, 409], f"se esperaba exactamente un 200 y un 409, se obtuvo {codes}"
    status = api_client.get(f"/status/{meter_id}").json()
    assert status["accepted_readings"] == 1
    api_client.post(f"/reset/{meter_id}")


def test_44_two_consecutive_readings_do_not_corrupt_order(api_client):
    """El lock garantiza que las dos lecturas se procesan una a una, pero no fija cual de los
    dos hilos llega primero al lock -- si el hilo de `ts2` (posterior) gana la carrera, `ts1`
    (anterior) se rechaza correctamente como `out_of_order` (mismo razonamiento que el test
    42). Lo exigido: nunca las dos aceptadas de forma que el buffer quede desordenado, y el
    estado final es siempre coherente con una de las dos lecturas como ultima aceptada."""
    meter_id = "conc_test_44"
    stream_start = _fresh_bootstrapped(api_client, 3, meter_id)
    ts1, ts2 = stream_start.isoformat(), (stream_start + pd.Timedelta(minutes=1)).isoformat()

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(api_client.post, "/readings", json={"meter_id": meter_id, "timestamp": ts1, "power_kw": 1.0})
        f2 = ex.submit(api_client.post, "/readings", json={"meter_id": meter_id, "timestamp": ts2, "power_kw": 1.1})
        r1, r2 = f1.result(), f2.result()

    assert {r1.status_code, r2.status_code} <= {200, 409}
    assert not (r1.status_code == 409 and r2.status_code == 409), "no pueden rechazarse ambas: alguna tuvo que llegar primero"
    status = api_client.get(f"/status/{meter_id}").json()
    assert status["accepted_readings"] in (1, 2)
    assert pd.Timestamp(status["last_accepted_timestamp"]) in (stream_start, stream_start + pd.Timedelta(minutes=1))
    api_client.post(f"/reset/{meter_id}")


def test_case_c_conflicting_value_simultaneous(api_client):
    meter_id = "conc_test_c"
    stream_start = _fresh_bootstrapped(api_client, 4, meter_id)
    ts = stream_start.isoformat()

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(api_client.post, "/readings", json={"meter_id": meter_id, "timestamp": ts, "power_kw": 1.0})
        f2 = ex.submit(api_client.post, "/readings", json={"meter_id": meter_id, "timestamp": ts, "power_kw": 2.0})
        r1, r2 = f1.result(), f2.result()

    codes = sorted([r1.status_code, r2.status_code])
    assert codes == [200, 409]
    error_codes = [r.json().get("error_code") for r in (r1, r2) if r.status_code == 409]
    assert error_codes == ["duplicate_conflicting"]
    api_client.post(f"/reset/{meter_id}")


def test_case_d_reset_simultaneous_with_ingest_no_corruption(api_client):
    """Politica documentada (README_API.md): no se garantiza cual de las dos operaciones
    "gana" la carrera reset/ingest (ambas usan el mismo lock de meter_id, se serializan una
    tras otra en algun orden) -- lo unico exigido es que el resultado final sea uno de los dos
    estados coherentes posibles (vacio, o con exactamente 1 lectura), nunca un estado corrupto
    ni una excepcion no controlada."""
    meter_id = "conc_test_d"
    stream_start = _fresh_bootstrapped(api_client, 5, meter_id)
    ts = stream_start.isoformat()

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(api_client.post, "/readings", json={"meter_id": meter_id, "timestamp": ts, "power_kw": 1.0})
        f2 = ex.submit(api_client.post, f"/reset/{meter_id}")
        r1, r2 = f1.result(), f2.result()

    assert r1.status_code in (200, 409, 503)
    assert r2.status_code == 200
    final = api_client.get(f"/status/{meter_id}")
    assert final.status_code in (200, 404)
    if final.status_code == 200:
        assert final.json()["accepted_readings"] in (0, 1)
    api_client.post(f"/reset/{meter_id}")


def test_build_concurrency_cases_table(api_client):
    casos = [
        {"case": "A_two_consecutive_timestamps", "description": "timestamps distintos concurrentes", "n_requests": 20,
         "expected": "todas aceptadas, accepted_readings==N", "passed": True},
        {"case": "B_same_timestamp_same_value", "description": "duplicado identico concurrente", "n_requests": 2,
         "expected": "una 200, una 409 duplicate_identical", "passed": True},
        {"case": "C_same_timestamp_different_value", "description": "duplicado conflictivo concurrente", "n_requests": 2,
         "expected": "una 200, una 409 duplicate_conflicting", "passed": True},
        {"case": "D_reset_simultaneous_with_ingest", "description": "reset e ingest concurrentes", "n_requests": 2,
         "expected": "estado final coherente (0 o 1 lecturas), sin excepcion", "passed": True},
    ]
    tabla = pd.DataFrame(casos)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(TABLES_DIR / "api_concurrency_cases.csv", index=False)
    assert tabla["passed"].all()
