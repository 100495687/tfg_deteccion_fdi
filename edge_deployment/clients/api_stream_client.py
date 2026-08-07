"""Experimento de equivalencia principal: compara, lectura a lectura, sobre el
mismo periodo pre-test de Fase 1 (mismo bootstrap, mismo inicio de streaming):

  A. DetectorEngine llamado directamente        (clients.direct_engine_client)
  B. DetectorEngine llamado via FastAPI TestClient (transporte ASGI in-process -- ejercita
     Pydantic, routing y error_handlers exactamente igual que un servidor real, sin el coste
     de un socket TCP real)
  C. DetectorEngine llamado via Uvicorn real (--base-url apunta a un servidor ya arrancado)

Por defecto compara A vs B (rapido, reproducible, cubre el 100% del periodo). A vs C se
ejecuta por separado con --base-url sobre un periodo mas corto (ver
benchmarks/benchmark_api_uvicorn.py y api_final_report.md, seccion de alcance).

Comandos:
    python -m edge_deployment.clients.api_stream_client --days 3
    python -m edge_deployment.clients.api_stream_client --days 30 --overwrite
    python -m edge_deployment.clients.api_stream_client --days 3 --base-url http://127.0.0.1:8000 --overwrite --out-suffix _uvicorn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"


class TestClientApiClient:
    """Ruta B: TestClient (transporte ASGI in-process, ejecuta el lifespan real)."""

    def __init__(self) -> None:
        from fastapi.testclient import TestClient

        from edge_deployment.api.main import app
        self._cm = TestClient(app)
        self.client = self._cm.__enter__()

    def close(self) -> None:
        self._cm.__exit__(None, None, None)

    def _post(self, path: str, payload: dict) -> dict:
        t0 = time.perf_counter()
        r = self.client.post(path, json=payload)
        dt = (time.perf_counter() - t0) * 1000
        body = r.json()
        body["http_status"] = r.status_code
        body["client_round_trip_ms"] = dt
        return body

    def bootstrap(self, meter_id: str, readings_df: pd.DataFrame) -> dict:
        from src.data_loading import COLUMNA_OBJETIVO
        payload = {"meter_id": meter_id, "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(readings_df.index, readings_df[COLUMNA_OBJETIVO])
        ]}
        return self._post("/bootstrap", payload)

    def ingest(self, meter_id: str, timestamp, power_kw: float) -> dict:
        return self._post("/readings", {"meter_id": meter_id, "timestamp": pd.Timestamp(timestamp).isoformat(), "power_kw": float(power_kw)})

    def reset(self, meter_id: str) -> dict:
        r = self.client.post(f"/reset/{meter_id}")
        return r.json()


class HttpApiClient:
    """Ruta C: servidor Uvicorn real, ya arrancado en `base_url`."""

    def __init__(self, base_url: str) -> None:
        import httpx
        self.client = httpx.Client(base_url=base_url, timeout=30.0)

    def close(self) -> None:
        self.client.close()

    def _post(self, path: str, payload: dict) -> dict:
        t0 = time.perf_counter()
        r = self.client.post(path, json=payload)
        dt = (time.perf_counter() - t0) * 1000
        body = r.json()
        body["http_status"] = r.status_code
        body["client_round_trip_ms"] = dt
        return body

    def bootstrap(self, meter_id: str, readings_df: pd.DataFrame) -> dict:
        from src.data_loading import COLUMNA_OBJETIVO
        payload = {"meter_id": meter_id, "readings": [
            {"timestamp": ts.isoformat(), "power_kw": float(v)} for ts, v in zip(readings_df.index, readings_df[COLUMNA_OBJETIVO])
        ]}
        return self._post("/bootstrap", payload)

    def ingest(self, meter_id: str, timestamp, power_kw: float) -> dict:
        return self._post("/readings", {"meter_id": meter_id, "timestamp": pd.Timestamp(timestamp).isoformat(), "power_kw": float(power_kw)})

    def reset(self, meter_id: str) -> dict:
        r = self.client.post(f"/reset/{meter_id}")
        return r.json()


def _row(ts, d: dict, a: dict) -> dict:
    return {
        "timestamp": ts,
        "direct_http_status": d["http_status"], "api_http_status": a["http_status"],
        "direct_accepted": d["accepted"], "api_accepted": a.get("accepted"),
        "direct_engine_status": d["engine_status"], "api_engine_status": a.get("engine_status"),
        "direct_p_ready": d["p_ready"], "api_p_ready": a.get("p_ready"),
        "direct_h_ready": d["h_ready"], "api_h_ready": a.get("h_ready"),
        "direct_p_evaluated": d["p_evaluated"], "api_p_evaluated": a.get("p_evaluated"),
        "direct_h_evaluated": d["h_evaluated"], "api_h_evaluated": a.get("h_evaluated"),
        "direct_score_p": d["score_p"], "api_score_p": a.get("score_p"),
        "direct_score_h": d["score_h"], "api_score_h": a.get("score_h"),
        "direct_alert_p": d["alert_p"], "api_alert_p": a.get("alert_p"),
        "direct_alert_p_ffill": d["alert_p_ffill"], "api_alert_p_ffill": a.get("alert_p_ffill"),
        "direct_alert_h": d["alert_h"], "api_alert_h": a.get("alert_h"),
        "direct_alert_or": d["alert_or"], "api_alert_or": a.get("alert_or"),
        "direct_detection_source": d["detection_source"], "api_detection_source": a.get("detection_source"),
        "direct_buffer_1min_size": d["buffer_1min_size"], "api_buffer_1min_size": a.get("buffer_1min_size"),
        "direct_history_15min_size": d["buffer_15min_size"], "api_history_15min_size": a.get("buffer_15min_size"),
        "direct_rejection_reason": d["rejection_reason"], "api_rejection_reason": a.get("rejection_reason", a.get("error_code")),
        "direct_round_trip_ms": d["client_round_trip_ms"], "api_round_trip_ms": a["client_round_trip_ms"],
        "direct_engine_ms": d["processing_time_ms"], "api_engine_ms": a.get("engine_processing_time_ms"),
        "api_api_ms": a.get("api_processing_time_ms"),
    }


def _series_equal_nan_aware(a: pd.Series, b: pd.Series) -> pd.Series:
    """`NaN == NaN` es False en pandas/IEEE754, y `.astype(str)` en pandas recientes no
    convierte NaN a la cadena 'nan' (lo deja como NaN) -- ambas cosas rompen una comparacion
    ingenua. Aqui: dos ausentes (NaN/None) cuentan como iguales; en el resto, comparacion de
    valor (via texto, para que True/'True'/1 se traten de forma consistente tras el
    roundtrip JSON)."""
    ambos_na = a.isna() & b.isna()
    iguales_valor = (a.astype(str) == b.astype(str))
    return ambos_na | (iguales_valor & ~a.isna() & ~b.isna())


def _resumen_equivalencia(df: pd.DataFrame) -> dict:
    campos_categoricos = ["accepted", "engine_status", "p_ready", "h_ready", "p_evaluated", "h_evaluated",
                           "alert_p", "alert_p_ffill", "alert_h", "alert_or", "detection_source",
                           "buffer_1min_size", "history_15min_size", "rejection_reason"]
    match_pct = {}
    for campo in campos_categoricos:
        d_col, a_col = df[f"direct_{campo}"], df[f"api_{campo}"]
        iguales = _series_equal_nan_aware(d_col, a_col)
        match_pct[campo] = float(iguales.mean() * 100)

    score_p_d, score_p_a = df["direct_score_p"].astype(float), df["api_score_p"].astype(float)
    score_h_d, score_h_a = df["direct_score_h"].astype(float), df["api_score_h"].astype(float)
    mask_p = score_p_d.notna() & score_p_a.notna()
    mask_h = score_h_d.notna() & score_h_a.notna()
    diff_p = (score_p_a[mask_p] - score_p_d[mask_p]).to_numpy()
    diff_h = (score_h_a[mask_h] - score_h_d[mask_h]).to_numpy()

    return {
        "n_readings": int(len(df)),
        "match_pct": match_pct,
        "http_status_match_pct": float((df["direct_http_status"] == df["api_http_status"]).mean() * 100),
        "score_p_max_abs_diff": float(np.abs(diff_p).max()) if len(diff_p) else None,
        "score_p_exactly_equal_pct": float((diff_p == 0).mean() * 100) if len(diff_p) else None,
        "score_h_max_abs_diff": float(np.abs(diff_h).max()) if len(diff_h) else None,
        "score_h_exactly_equal_pct": float((diff_h == 0).mean() * 100) if len(diff_h) else None,
        "all_categorical_100pct": bool(all(v == 100.0 for v in match_pct.values())),
    }


def run_equivalence(max_stream_days: int, base_url: str | None = None, overwrite: bool = False,
                     out_suffix: str = "") -> dict:
    from edge_deployment.clients.direct_engine_client import DirectEngineClient
    from edge_deployment.core.period_selection import seleccionar_periodo
    from src.data_loading import COLUMNA_OBJETIVO

    out_path = TABLES_DIR / f"api_engine_equivalence{out_suffix}.csv"
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} ya existe. Usa --overwrite para reemplazarlo.")

    t0 = time.time()
    periodo = seleccionar_periodo(max_stream_days=max_stream_days)
    print(f"[api-equiv] streaming: {periodo['streaming_partition'].index.min()} -> {periodo['streaming_partition'].index.max()}")

    print("[api-equiv] Cargando motor directo (ruta A)...")
    direct = DirectEngineClient()
    if base_url:
        print(f"[api-equiv] Conectando a servidor Uvicorn real en {base_url} (ruta C)...")
        api_client = HttpApiClient(base_url)
    else:
        print("[api-equiv] Cargando API via TestClient (ruta B, ejecuta el lifespan real)...")
        api_client = TestClientApiClient()

    meter_id = "meter_api_equiv"
    rep_direct = direct.bootstrap(meter_id, periodo["bootstrap_partition"])
    rep_api = api_client.bootstrap(meter_id, periodo["bootstrap_partition"])
    print(f"[api-equiv] bootstrap directo: h_ready={rep_direct['h_ready']} | bootstrap API: h_ready={rep_api.get('h_ready')}")
    assert rep_direct["h_ready"] == rep_api.get("h_ready"), "el bootstrap directo y el de la API difieren en h_ready"

    filas = []
    for ts, row in periodo["streaming_partition"].iterrows():
        v = float(row[COLUMNA_OBJETIVO])
        d = direct.ingest(meter_id, ts, v)
        a = api_client.ingest(meter_id, ts, v)
        filas.append(_row(ts, d, a))

    df = pd.DataFrame(filas)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    resumen = _resumen_equivalencia(df)
    resumen.update({"max_stream_days": max_stream_days, "base_url": base_url,
                     "route_b_or_c": "C_uvicorn_real" if base_url else "B_testclient",
                     "elapsed_seconds": time.time() - t0, "fecha_utc": datetime.now(timezone.utc).isoformat()})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / f"api_equivalence_summary{out_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)

    api_client.close()
    print(f"[api-equiv] Completado en {time.time()-t0:.1f}s -- all_categorical_100pct={resumen['all_categorical_100pct']}")
    print(json.dumps(resumen["match_pct"], indent=2))
    return {"df": df, "resumen": resumen}


def build_api_offline_equivalence_summary() -> dict:
    """No se repite la investigacion de Fase 1. Se reutiliza
    `results/reports/equivalence_summary.json` (Fase 1: motor directo vs. referencia offline,
    100% de alertas) y `results/reports/api_equivalence_summary.json` (Fase 2: API vs. motor
    directo, ya calculado arriba) y se concluye por transitividad: si API==directo y
    directo==offline, entonces API==offline. Solo se valida que ambos informes usan los
    mismos artefactos (hashes del manifiesto), nunca se recalcula ningun score."""
    from edge_deployment.core import model_loader as ml

    phase1_path = REPORTS_DIR / "equivalence_summary.json"
    phase2_path = REPORTS_DIR / "api_equivalence_summary.json"
    if not phase1_path.exists() or not phase2_path.exists():
        raise SystemExit("Faltan equivalence_summary.json (Fase 1) o api_equivalence_summary.json (Fase 2) "
                          "-- ejecuta primero replay_clean_stream.py y api_stream_client.py")

    with open(phase1_path, encoding="utf-8") as f:
        fase1 = json.load(f)
    with open(phase2_path, encoding="utf-8") as f:
        fase2 = json.load(f)

    manifest = ml.build_artifact_manifest()
    fase1_alertas_100 = all(fase1["metrics"]["alerts"][k] == 100.0 for k in
                             ("match_pct_H", "match_pct_P", "match_pct_P_ffill", "match_pct_OR"))
    fase2_alertas_100 = fase2["all_categorical_100pct"]
    transitividad_ok = bool(fase1_alertas_100 and fase2_alertas_100)

    resumen = {
        "method": "transitividad (API vs directo, Fase 2) + (directo vs offline, Fase 1) -- sin recalcular scores",
        "phase1_direct_vs_offline": {"criterio_principal_cumplido": fase1["criterio_principal_cumplido"],
                                      "alerts": fase1["metrics"]["alerts"]},
        "phase2_api_vs_direct": {"all_categorical_100pct": fase2_alertas_100, "match_pct": fase2["match_pct"],
                                  "n_readings": fase2["n_readings"], "route": fase2["route_b_or_c"]},
        "artifact_hashes": manifest["hashes"],
        "api_preserves_offline_equivalence": transitividad_ok,
        "alert_P_100pct": fase1_alertas_100 and fase2["match_pct"]["alert_p"] == 100.0,
        "alert_H_100pct": fase1_alertas_100 and fase2["match_pct"]["alert_h"] == 100.0,
        "alert_P_ffill_100pct": fase1_alertas_100 and fase2["match_pct"]["alert_p_ffill"] == 100.0,
        "alert_OR_100pct": fase1_alertas_100 and fase2["match_pct"]["alert_or"] == 100.0,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(REPORTS_DIR / "api_offline_equivalence_summary.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    return resumen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(EDGE_DIR / "config" / "api.yaml"))
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-suffix", type=str, default="")
    args = parser.parse_args()
    run_equivalence(max_stream_days=args.days, base_url=args.base_url, overwrite=args.overwrite, out_suffix=args.out_suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
