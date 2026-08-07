"""Simulador acelerado de streaming + comparacion online/offline.
Envia una lectura por minuto en orden al `DetectorEngine`, usando el timestamp del
propio evento (nunca el reloj real), y compara la salida contra la referencia offline
congelada -- reutilizada literalmente:
  - H offline: `congelacion_final_or_pretest.predecir_pool_limpio`
  - P offline: `evaluacion_final_retrospectiva_test.puntuar_p_limpio`
  - propagacion/fusion offline: `fusion_p_histgb.estado_ffill`

No entrena ni recalibra nada. No usa test (unicamente el periodo pre-test seleccionado en
`period_selection.py`).

Comando:
    python -m edge_deployment.simulators.replay_clean_stream \
        --config edge_deployment/config/edge_engine.yaml --period pretest_equivalence
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
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
LOGS_DIR = EDGE_DIR / "results" / "logs"


def run_streaming_simulation(engine, meter_id: str, streaming_partition, diagnostic: bool = False) -> dict:
    from src.data_loading import COLUMNA_OBJETIVO

    filas_p, filas_h, latencias, buffer_sizes = [], [], [], []
    for i, (ts, row) in enumerate(streaming_partition.iterrows()):
        resp = engine.ingest(meter_id, ts, float(row[COLUMNA_OBJETIVO]))
        assert resp.accepted, f"lectura de streaming limpia rechazada inesperadamente en {ts}: {resp.rejection_reason}"
        latencias.append({"timestamp": ts, "processing_time_ms": resp.processing_time_ms,
                           "p_evaluated": resp.p_evaluated, "h_evaluated": resp.h_evaluated})
        if resp.p_evaluated:
            filas_p.append({"timestamp": resp.p_last_evaluation_timestamp, "score_p_online": resp.score_p, "alert_p_online": resp.alert_p})
        if resp.h_evaluated:
            filas_h.append({
                "timestamp": resp.h_last_evaluation_timestamp, "score_h_online": resp.score_h,
                "alert_h_online": resp.alert_h, "aggregate_15m_online": resp.aggregate_15min_kw,
                "alert_p_ffill_online": resp.alert_p_ffill, "alert_or_online": resp.alert_or,
                "detection_source_online": resp.detection_source,
            })
        if diagnostic and (i % 200 == 0):
            buffer_sizes.append({"timestamp": ts, "buffer_1min_size": resp.buffer_1min_size, "buffer_15min_size": resp.buffer_15min_size})

    return {
        "p_online": pd.DataFrame(filas_p), "h_online": pd.DataFrame(filas_h),
        "latencies": pd.DataFrame(latencias), "buffer_sizes": pd.DataFrame(buffer_sizes),
    }


def build_offline_reference(periodo: dict, artefactos: dict) -> dict:
    from src import congelacion_final_or_pretest as cfg
    from src import evaluacion_final_retrospectiva_test as eftt
    from src import fusion_p_histgb as fus

    positive_h, ts_h, df_pool = cfg.predecir_pool_limpio(
        artefactos["modelo_h"], artefactos["perfil_h"], periodo["particion_entrenamiento"],
        periodo["streaming_partition"], periodo["final_cal_pool_ini"])
    score_p, ts_p = eftt.puntuar_p_limpio(periodo["streaming_partition"], artefactos["modelo_p"], artefactos["params_norm_p"])

    active_h = positive_h > artefactos["threshold_h"]
    active_p = score_p > artefactos["threshold_p"]
    active_p_ffill = fus.estado_ffill(score_p, artefactos["threshold_p"], ts_p, ts_h)
    active_or = active_p_ffill | active_h

    h_df = pd.DataFrame({
        "timestamp": pd.DatetimeIndex(ts_h), "score_h_offline": positive_h, "alert_h_offline": active_h,
        "aggregate_15m_offline": df_pool["target_kw"].to_numpy(),
        "alert_p_ffill_offline": active_p_ffill, "alert_or_offline": active_or,
    })
    p_df = pd.DataFrame({"timestamp": pd.DatetimeIndex(ts_p), "score_p_offline": score_p, "alert_p_offline": active_p})
    return {"h_offline": h_df, "p_offline": p_df}


def comparar_equivalencia(online: dict, offline: dict, rtol: float, atol: float) -> dict:
    merged_h = offline["h_offline"].merge(online["h_online"], on="timestamp", how="outer", indicator=True)
    merged_p = offline["p_offline"].merge(online["p_online"], on="timestamp", how="outer", indicator=True)

    both_h = merged_h[merged_h["_merge"] == "both"].sort_values("timestamp").reset_index(drop=True)
    both_p = merged_p[merged_p["_merge"] == "both"].sort_values("timestamp").reset_index(drop=True)

    diff_agg = (both_h["aggregate_15m_online"] - both_h["aggregate_15m_offline"]).to_numpy()
    diff_score_h = (both_h["score_h_online"] - both_h["score_h_offline"]).to_numpy()
    diff_score_p = (both_p["score_p_online"] - both_p["score_p_offline"]).to_numpy()

    metrics = {
        "timestamps": {
            "n_h_offline": int(len(offline["h_offline"])), "n_h_online": int(len(online["h_online"])),
            "n_h_matched": int(len(both_h)), "n_h_missing_online": int((merged_h["_merge"] == "left_only").sum()),
            "n_h_extra_online": int((merged_h["_merge"] == "right_only").sum()),
            "n_p_offline": int(len(offline["p_offline"])), "n_p_online": int(len(online["p_online"])),
            "n_p_matched": int(len(both_p)), "n_p_missing_online": int((merged_p["_merge"] == "left_only").sum()),
            "n_p_extra_online": int((merged_p["_merge"] == "right_only").sum()),
        },
        "aggregation": {
            "mae": float(np.abs(diff_agg).mean()) if len(diff_agg) else None,
            "rmse": float(np.sqrt((diff_agg ** 2).mean())) if len(diff_agg) else None,
            "max_abs_diff": float(np.abs(diff_agg).max()) if len(diff_agg) else None,
            "pct_exactly_equal": float((diff_agg == 0).mean() * 100) if len(diff_agg) else None,
            "n_bins_not_matching": int((diff_agg != 0).sum()) if len(diff_agg) else None,
        },
        "score_h": {
            "mae": float(np.abs(diff_score_h).mean()) if len(diff_score_h) else None,
            "rmse": float(np.sqrt((diff_score_h ** 2).mean())) if len(diff_score_h) else None,
            "max_abs_diff": float(np.abs(diff_score_h).max()) if len(diff_score_h) else None,
            "pct_isclose": float(np.isclose(both_h["score_h_online"], both_h["score_h_offline"], rtol=rtol, atol=atol).mean() * 100) if len(both_h) else None,
            "correlation": float(np.corrcoef(both_h["score_h_online"], both_h["score_h_offline"])[0, 1]) if len(both_h) > 1 else None,
        },
        "score_p": {
            "mae": float(np.abs(diff_score_p).mean()) if len(diff_score_p) else None,
            "rmse": float(np.sqrt((diff_score_p ** 2).mean())) if len(diff_score_p) else None,
            "max_abs_diff": float(np.abs(diff_score_p).max()) if len(diff_score_p) else None,
            "pct_isclose": float(np.isclose(both_p["score_p_online"], both_p["score_p_offline"], rtol=rtol, atol=atol).mean() * 100) if len(both_p) else None,
            "correlation": float(np.corrcoef(both_p["score_p_online"], both_p["score_p_offline"])[0, 1]) if len(both_p) > 1 else None,
        },
        "alerts": {
            "match_pct_H": float((both_h["alert_h_online"] == both_h["alert_h_offline"]).mean() * 100) if len(both_h) else None,
            "match_pct_P": float((both_p["alert_p_online"] == both_p["alert_p_offline"]).mean() * 100) if len(both_p) else None,
            "match_pct_P_ffill": float((both_h["alert_p_ffill_online"] == both_h["alert_p_ffill_offline"]).mean() * 100) if len(both_h) else None,
            "match_pct_OR": float((both_h["alert_or_online"] == both_h["alert_or_offline"]).mean() * 100) if len(both_h) else None,
            "n_mismatches_H": int((both_h["alert_h_online"] != both_h["alert_h_offline"]).sum()) if len(both_h) else None,
            "n_mismatches_P": int((both_p["alert_p_online"] != both_p["alert_p_offline"]).sum()) if len(both_p) else None,
            "n_mismatches_P_ffill": int((both_h["alert_p_ffill_online"] != both_h["alert_p_ffill_offline"]).sum()) if len(both_h) else None,
            "n_mismatches_OR": int((both_h["alert_or_online"] != both_h["alert_or_offline"]).sum()) if len(both_h) else None,
        },
    }
    return {"metrics": metrics, "both_h": both_h, "both_p": both_p, "merged_h": merged_h, "merged_p": merged_p}


def criterio_principal_cumplido(metrics: dict) -> bool:
    t = metrics["timestamps"]
    a = metrics["alerts"]
    return (
        t["n_h_missing_online"] == 0 and t["n_h_extra_online"] == 0 and
        t["n_p_missing_online"] == 0 and t["n_p_extra_online"] == 0 and
        metrics["aggregation"]["pct_exactly_equal"] == 100.0 and
        a["match_pct_H"] == 100.0 and a["match_pct_P"] == 100.0 and
        a["match_pct_P_ffill"] == 100.0 and a["match_pct_OR"] == 100.0
    )


def run_equivalence(max_stream_days: int, diagnostic: bool = False, overwrite: bool = False) -> dict:
    from edge_deployment.core import model_loader as ml
    from edge_deployment.core.detector_engine import DetectorEngine
    from edge_deployment.core.period_selection import seleccionar_periodo

    out_path = TABLES_DIR / "offline_online_equivalence.csv"
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} ya existe. Usa --overwrite para reemplazarlo.")

    t0 = time.time()
    print("[edge] Construyendo inventario y manifiesto de artefactos...")
    ml.build_input_artifacts_inventory()
    ml.build_artifact_manifest()

    print("[edge] Seleccionando periodo pre-test (bootstrap + streaming)...")
    periodo = seleccionar_periodo(max_stream_days=max_stream_days)
    print(f"  bootstrap: {periodo['bootstrap_partition'].index.min()} -> {periodo['bootstrap_partition'].index.max()}")
    print(f"  streaming: {periodo['streaming_partition'].index.min()} -> {periodo['streaming_partition'].index.max()}"
          f" ({periodo['reporte']['streaming_period']['n_days']:.1f} dias)")

    print("[edge] Cargando motor (artefactos congelados, solo inferencia)...")
    engine = DetectorEngine()
    artefactos = {"modelo_h": engine.modelo_h, "perfil_h": engine.perfil_h, "modelo_p": engine.modelo_p,
                  "params_norm_p": engine.params_norm_p, "threshold_h": engine.threshold_h, "threshold_p": engine.threshold_p}

    print("[edge] Bootstrap...")
    bootstrap_report = engine.bootstrap("meter_pretest", periodo["bootstrap_partition"])
    print(f"  p_ready={bootstrap_report['p_ready']} h_ready={bootstrap_report['h_ready']}")

    print(f"[edge] Ejecutando simulacion de streaming ({len(periodo['streaming_partition'])} min)...")
    online = run_streaming_simulation(engine, "meter_pretest", periodo["streaming_partition"], diagnostic=diagnostic)
    print(f"  P evaluado {len(online['p_online'])} veces | H evaluado {len(online['h_online'])} veces")

    print("[edge] Construyendo referencia offline (predecir_pool_limpio + puntuar_p_limpio)...")
    offline = build_offline_reference(periodo, artefactos)

    print("[edge] Comparando online vs offline...")
    cfg_dict = yaml.safe_load((EDGE_DIR / "config" / "edge_engine.yaml").read_text(encoding="utf-8"))
    rtol, atol = cfg_dict["equivalence"]["rtol"], cfg_dict["equivalence"]["atol"]
    resultado = comparar_equivalencia(online, offline, rtol, atol)
    metrics = resultado["metrics"]
    criterio_ok = criterio_principal_cumplido(metrics)
    print(f"  alert match H={metrics['alerts']['match_pct_H']} P={metrics['alerts']['match_pct_P']} "
          f"P_ffill={metrics['alerts']['match_pct_P_ffill']} OR={metrics['alerts']['match_pct_OR']}")
    print(f"  CRITERIO PRINCIPAL DE EQUIVALENCIA CUMPLIDO: {criterio_ok}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tabla_final = resultado["both_h"].drop(columns=["_merge"]).merge(
        resultado["both_p"].drop(columns=["_merge"]), on="timestamp", how="outer", suffixes=("", "_p"))
    tabla_final = tabla_final.sort_values("timestamp").reset_index(drop=True)
    tabla_final.to_csv(out_path, index=False)
    online["p_online"].to_csv(TABLES_DIR / "p_online_decisions.csv", index=False)
    online["h_online"].to_csv(TABLES_DIR / "h_online_decisions.csv", index=False)
    # nota de nombres: el benchmark dedicado (benchmarks/benchmark_engine.py) escribe
    # results/tables/preliminary_latency.csv; esta es la latencia observada
    # durante la propia corrida de equivalencia, guardada con nombre distinto para no chocar
    online["latencies"].to_csv(TABLES_DIR / "simulation_latency.csv", index=False)
    offline["h_offline"].to_csv(TABLES_DIR / "h_offline_reference.csv", index=False)
    offline["p_offline"].to_csv(TABLES_DIR / "p_offline_reference.csv", index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    equivalence_summary = {
        "status": "EQUIVALENT" if criterio_ok else "NOT_EQUIVALENT",
        "criterio_principal_cumplido": criterio_ok, "rtol": rtol, "atol": atol,
        "max_stream_days": max_stream_days, "n_stream_minutes": int(len(periodo["streaming_partition"])),
        "elapsed_seconds": time.time() - t0, "metrics": metrics,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(REPORTS_DIR / "equivalence_summary.json", "w", encoding="utf-8") as f:
        json.dump(equivalence_summary, f, indent=2, ensure_ascii=False, default=str)

    resampling_semantics = {
        "frequency": "15min", "anchor": "posicional (0-indexado desde el inicio de la particion pasada a "
        "construir_serie_15min / ventanear), NUNCA alineado a :00/:15/:30/:45 de reloj",
        "label": "el timestamp del bin/ventana es el de su ULTIMA muestra (ts_end / window `fin`), no el de inicio",
        "closed": "ambos extremos pertenecen al bin (bins de exactamente 15/360 muestras consecutivas, sin solape en streaming)",
        "operation": "media aritmetica (H: pcl.construir_serie_15min; P: valores crudos ventaneados, el score ya "
        "promedia dentro de la ventana)",
        "min_muestras_bin": {"H": 15, "P": 360},
        "incomplete_bins": "nunca se cierran/evaluan -- un hueco descarta el bucket parcial en curso (streaming_preprocessing.append_reading)",
        "timezone": "timestamps naive, sin zona horaria (identico al resto del pipeline)",
        "rounding": "ninguno", "unit_conversion": "ninguna (kW tal cual Global_active_power)",
        "verified_against": "period_selection.py: bootstrap termina en 2009-06-18 17:23:00 y streaming empieza en "
        "2009-06-18 17:24:00 (no alineados a cuarto de hora de reloj) -- confirma anclaje posicional, no de reloj.",
    }
    with open(REPORTS_DIR / "resampling_semantics.json", "w", encoding="utf-8") as f:
        json.dump(resampling_semantics, f, indent=2, ensure_ascii=False)

    print(f"[edge] Completado en {time.time()-t0:.1f}s -- resultados en {TABLES_DIR} y {REPORTS_DIR}")
    return {"periodo": periodo, "online": online, "offline": offline, "resultado": resultado,
            "equivalence_summary": equivalence_summary, "engine": engine}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(EDGE_DIR / "config" / "edge_engine.yaml"))
    parser.add_argument("--period", type=str, default="pretest_equivalence")
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_equivalence(max_stream_days=args.max_days, diagnostic=args.diagnostic, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
