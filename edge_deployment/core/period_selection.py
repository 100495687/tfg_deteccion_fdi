"""Seleccion del periodo pre-test de equivalencia.

Reutiliza literalmente `congelacion_final_or_pretest.verificar_decision_previa` y
`.preparar_periodos` -- el mismo FINAL_TRAIN / FINAL_CAL_POOL_180 ya usado por
threshold_tradeoff y replay_pilot. La razon de eleccion es exclusivamente tecnica: es el
unico periodo pre-test con checkpoints H y P ya congelados y con una funcion offline
(`predecir_pool_limpio` / `puntuar_p_limpio`) ya validada para producir la referencia sin
reentrenar nada. No se elige por tasa de deteccion, alertas ni comportamiento del score.

Bootstrap: el motor online necesita, antes de empezar a transmitir, el mismo contexto de 7
dias (+margen) que `predecir_pool_limpio` toma de `particion_entrenamiento` para los lags de
H (`pcl.MAX_LAG=672` bins de 15 min + 10 de margen = 682*15 = 10230 minutos). P, en cambio, no
usa ese contexto: la referencia offline (`puntuar_p_limpio`) ventanea desde el inicio de
`particion_final_pool`, sin ningun prefijo -- para que el motor online sea equivalente debe
igualmente empezar a ventanear P desde el inicio del streaming, nunca desde el bootstrap.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

DEFAULT_MAX_STREAM_DAYS = 30
MIN_STREAM_DAYS = 14


def seleccionar_periodo(max_stream_days: int | None = DEFAULT_MAX_STREAM_DAYS) -> dict:
    from src import congelacion_final_or_pretest as cfg
    from src import predictor_causal_lags as pcl

    payload_decision = cfg.verificar_decision_previa()  # solo verifica, no recalcula nada con test
    periodos = cfg.preparar_periodos()
    particion_entrenamiento = periodos["particion_entrenamiento"]
    particion_final_pool = periodos["particion_final_pool"]
    final_cal_pool_ini = periodos["final_cal_pool_ini"]
    fin_pretest = periodos["fin_pretest"]

    bootstrap_context_min = (pcl.MAX_LAG + 10) * 15  # identico margen a predecir_pool_limpio
    bootstrap_partition = particion_entrenamiento.iloc[-bootstrap_context_min:]
    assert bootstrap_partition.index.max() < final_cal_pool_ini, "el bootstrap debe ser estrictamente anterior al streaming"
    assert len(bootstrap_partition) == bootstrap_context_min, "el bootstrap no tiene la longitud exacta esperada (contexto insuficiente en FINAL_TRAIN)"
    assert bootstrap_context_min % 15 == 0, "el contexto de bootstrap debe ser multiplo exacto de 15 minutos para que los bins de H se alineen con el inicio del streaming"

    if max_stream_days is not None:
        stream_end = final_cal_pool_ini + pd.Timedelta(days=max_stream_days)
        streaming_partition = particion_final_pool.loc[particion_final_pool.index < stream_end]
    else:
        streaming_partition = particion_final_pool

    n_stream_days = (streaming_partition.index.max() - streaming_partition.index.min()).total_seconds() / 86400.0
    cobertura_completa = bool(len(streaming_partition) == len(streaming_partition.index.to_series().diff().fillna(pd.Timedelta(minutes=1))))
    huecos = int((streaming_partition.index.to_series().diff().dropna() != pd.Timedelta(minutes=1)).sum())

    warnings = []
    if n_stream_days < MIN_STREAM_DAYS:
        warnings.append(f"streaming cubre solo {n_stream_days:.1f} dias, por debajo del minimo recomendado ({MIN_STREAM_DAYS})")
    if huecos:
        warnings.append(f"se detectaron {huecos} huecos de mas de 1 minuto en la particion de streaming (dataset ya interpolado en origen)")

    reporte = {
        "bootstrap_period": {"start": str(bootstrap_partition.index.min()), "end": str(bootstrap_partition.index.max()),
                              "n_minutes": int(len(bootstrap_partition))},
        "streaming_period": {"start": str(streaming_partition.index.min()), "end": str(streaming_partition.index.max()),
                              "n_minutes": int(len(streaming_partition)), "n_days": n_stream_days},
        "split": "FINAL_CAL_POOL_180 (pre-test, ver src/congelacion_final_or_pretest.py)",
        "selection_reason": "unico periodo pre-test con checkpoints H y P ya congelados y con funcion offline "
                             "validada (predecir_pool_limpio/puntuar_p_limpio) para reproducir la referencia sin "
                             "reentrenar -- seleccion por disponibilidad tecnica, no por comportamiento del score",
        "available_artifacts": {"model_H": True, "model_P": True, "profiles_H": True, "thresholds": True},
        "test_absence_check": {"test_intacto": bool(payload_decision.get("test_intacto")),
                                "unico_dato_leido_de_test": "timestamp de arranque (fin_pretest), igual que congelacion_final_or_pretest"},
        "duration_days": n_stream_days, "coverage_complete": cobertura_completa, "warnings": warnings,
        "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "period_selection.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)

    return {
        "particion_entrenamiento": particion_entrenamiento, "particion_final_pool": particion_final_pool,
        "bootstrap_partition": bootstrap_partition, "streaming_partition": streaming_partition,
        "final_cal_pool_ini": final_cal_pool_ini, "fin_pretest": fin_pretest, "stream_start": final_cal_pool_ini,
        "reporte": reporte,
    }


if __name__ == "__main__":
    r = seleccionar_periodo()
    print(f"bootstrap: {r['bootstrap_partition'].index.min()} -> {r['bootstrap_partition'].index.max()} "
          f"({len(r['bootstrap_partition'])} min)")
    print(f"streaming: {r['streaming_partition'].index.min()} -> {r['streaming_partition'].index.max()} "
          f"({len(r['streaming_partition'])} min, {r['reporte']['streaming_period']['n_days']:.1f} dias)")
