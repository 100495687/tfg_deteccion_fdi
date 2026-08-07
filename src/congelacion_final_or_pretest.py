"""Congelacion tecnica final de la arquitectura ya seleccionada por la auditoria
(OR_PMULTI_HMULTIWINDOW = P_MULTI_SEASON OR HistGB_PERIODIC/H_MULTIWINDOW_180).

Esto no es un experimento de seleccion ni de optimizacion: la decision metodologica ya
esta cerrada (models/auditoria_final_p_vs_or/configuracion_final_decision_pretest.json).
Este modulo unicamente:
  1) verifica esa decision (sin volver a compararla con nada);
  2) entrena un unico modelo HistGB_PERIODIC final sobre FINAL_TRAIN limpio;
  3) calibra un unico umbral final de H mediante H_MULTIWINDOW_180 (procedimiento ya
     validado, reutilizado literalmente via calibracion_temporal_h.calibrar_multiwindow)
     sobre FINAL_CAL_POOL_180 limpio;
  4) congela threshold_P = 0.049748 (ya fijado, no se recalibra);
  5) escribe los artefactos congelados en models/final_or_pretest/, con sus hashes SHA-256
     y una verificacion de determinismo (doble carga -> predicciones bit-identicas).

Test permanece completamente cerrado: la unica informacion leida de la particion de test
es su timestamp de arranque (para delimitar FINAL_CAL_POOL_180). No se genera ningun
ataque, no se calcula ningun score ni se ejecuta ninguna inferencia sobre test.

Ejecutable de forma aislada:
    python -m src.congelacion_final_or_pretest
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import calibracion_temporal_h as cth
from src import optimizacion_histgb_periodic as opt
from src import predictor_causal_lags as pcl
from src import robustez_temporal_p as rtp
from src.data_loading import cargar_y_limpiar
from src.experimento_stride import contar_eventos
from src.memoria_finita_relative_kw import _duracion_eventos
from src.splitting import split_temporal

BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "models" / "final_or_pretest"
AUD_MODELOS_DIR = BASE_DIR / "models" / "auditoria_final_p_vs_or"
AUD_TABLAS_DIR = BASE_DIR / "results" / "tables" / "auditoria_final_p_vs_or"

RESOLUCION_MIN = 15
CAL_POOL_DIAS = 180
THRESHOLD_P_FINAL = 0.049748


# ==========================================================================================
# 5) Reproduccion / verificacion previa obligatoria (solo artefactos ya guardados en disco)
# ==========================================================================================

def verificar_decision_previa() -> dict:
    with open(AUD_MODELOS_DIR / "configuracion_final_decision_pretest.json", encoding="utf-8") as f:
        payload = json.load(f)

    checks = {
        "selected_configuration": payload["selected_configuration"] == "OR_PMULTI_HMULTIWINDOW",
        "umbral_final_p": abs(payload["umbral_final_p"] - THRESHOLD_P_FINAL) < 1e-6,
        "features_h": payload["features_h"] == opt.PERIODIC_BASE_FEATURES,
        "hiperparametros_h": payload["hiperparametros_h"] == opt.CONFIG_HISTGB_PERIODIC_ACTUAL,
        "regla_or": payload["regla_or"] == "active_P_MULTI_SEASON_ffill OR active_H_MULTIWINDOW_180",
        "hashes_disponibles": bool(payload.get("metadatos_previos", {}).get("hash_modelo_HistGB_PERIODIC")),
        "test_intacto": payload.get("test_intacto") is True,
    }
    if not all(checks.values()):
        fallos = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"la decision previa no se reproduce dentro de lo esperado -- congelacion detenida. Fallan: {fallos}")

    tabla_principal = pd.read_csv(AUD_TABLAS_DIR / "tabla_principal.csv")
    fila_or = tabla_principal[tabla_principal["configuracion"] == "OR_PMULTI_HMULTIWINDOW"].iloc[0]
    checks_num = {
        "fa_ponderada_or": np.isclose(fila_or["fa_ponderada"], 0.050, atol=0.01),
        "dr_ramp_or": np.isclose(fila_or["dr_ramp_pct"], 15.9, atol=1.0),
        "dr_energ_or": np.isclose(fila_or["dr_energ_pct"], 21.5, atol=1.0),
        "folds_saturados_or": int(fila_or["folds_saturados"]) == 0,
    }
    tabla_criterios = pd.read_csv(AUD_TABLAS_DIR / "cumplimiento_regla_fusion.csv")
    checks_num["catorce_de_catorce"] = bool(tabla_criterios["cumple"].all()) and len(tabla_criterios) == 14
    if not all(checks_num.values()):
        fallos = [k for k, v in checks_num.items() if not v]
        raise RuntimeError(f"las cifras de la auditoria final no se reproducen dentro de tolerancia -- congelacion detenida. Fallan: {fallos}")

    return payload


# ==========================================================================================
# 6) Periodos finales
# ==========================================================================================

def preparar_periodos() -> dict:
    df_min = cargar_y_limpiar()
    particiones = split_temporal(df_min)
    serie_completa_min = pd.concat([particiones["train"], particiones["val"]])
    fin_pretest = particiones["test"].index[0]  # unico dato leido de test: su timestamp de arranque
    final_cal_pool_ini = fin_pretest - pd.Timedelta(days=CAL_POOL_DIAS)

    particion_entrenamiento = serie_completa_min.loc[serie_completa_min.index < final_cal_pool_ini]
    particion_final_pool = serie_completa_min.loc[
        (serie_completa_min.index >= final_cal_pool_ini) & (serie_completa_min.index < fin_pretest)]

    assert particion_entrenamiento.index.max() < final_cal_pool_ini, "FINAL_TRAIN debe preceder a FINAL_CAL_POOL_180"
    assert particion_final_pool.index.min() >= final_cal_pool_ini and particion_final_pool.index.max() < fin_pretest, \
        "FINAL_CAL_POOL_180 debe preceder a test, sin solapamiento"
    return {
        "particion_entrenamiento": particion_entrenamiento, "particion_final_pool": particion_final_pool,
        "final_cal_pool_ini": final_cal_pool_ini, "fin_pretest": fin_pretest,
    }


# ==========================================================================================
# 7) Entrenamiento final de H (exclusivamente sobre FINAL_TRAIN limpio)
# ==========================================================================================

def entrenar_h_final(particion_entrenamiento: pd.DataFrame):
    cols = opt.PERIODIC_BASE_FEATURES
    kw_train = pcl.construir_serie_15min(particion_entrenamiento)
    feats_train = pcl.construir_features_kw(kw_train["kw"])
    cal_train = pcl._calendario(kw_train["ts_start"])
    df_train = pd.concat([feats_train, cal_train], axis=1)
    df_train["target_kw"] = kw_train["kw"]
    df_train["ts_start"] = kw_train["ts_start"]
    df_train["ts_end"] = kw_train["ts_end"]
    df_train = df_train.iloc[pcl.MAX_LAG:].reset_index(drop=True)

    perfil_final = pcl.construir_perfil_train({"kw": kw_train["kw"][pcl.MAX_LAG:], "ts_start": kw_train["ts_start"][pcl.MAX_LAG:]})
    df_train = df_train.merge(perfil_final, on=["es_finde", "slot_15min"], how="left").dropna(subset=cols)

    modelo_final = opt.construir_modelo(opt.CONFIG_HISTGB_PERIODIC_ACTUAL)
    modelo_final.fit(df_train[cols].to_numpy(dtype=np.float64), df_train["target_kw"].to_numpy())

    metadatos = {
        "n_observaciones_entrenamiento": int(len(df_train)),
        "fecha_inicio_entrenamiento": str(pd.Timestamp(df_train["ts_start"].iloc[0])),
        "fecha_fin_entrenamiento": str(pd.Timestamp(df_train["ts_end"].iloc[-1])),
        "features_orden": cols, "hiperparametros": opt.CONFIG_HISTGB_PERIODIC_ACTUAL,
        "sklearn_version": sklearn.__version__,
    }
    return modelo_final, perfil_final, metadatos


# ==========================================================================================
# 8) Prediccion limpia de FINAL_CAL_POOL_180 (sin ataques)
# ==========================================================================================

def predecir_pool_limpio(modelo_final, perfil_final: pd.DataFrame, particion_entrenamiento: pd.DataFrame,
                          particion_final_pool: pd.DataFrame, final_cal_pool_ini: pd.Timestamp):
    cols = opt.PERIODIC_BASE_FEATURES
    serie_para_pool = pd.concat([particion_entrenamiento.iloc[-(pcl.MAX_LAG + 10) * RESOLUCION_MIN:], particion_final_pool])
    kw_pool = pcl.construir_serie_15min(serie_para_pool)
    feats_pool = pcl.construir_features_kw(kw_pool["kw"])
    cal_pool = pcl._calendario(kw_pool["ts_start"])
    df_pool_full = pd.concat([feats_pool, cal_pool], axis=1)
    df_pool_full["target_kw"] = kw_pool["kw"]
    df_pool_full["ts_start"] = kw_pool["ts_start"]
    df_pool_full["ts_end"] = kw_pool["ts_end"]
    df_pool_full = df_pool_full.merge(perfil_final, on=["es_finde", "slot_15min"], how="left")
    df_pool = df_pool_full[df_pool_full["ts_start"] >= final_cal_pool_ini].dropna(subset=cols).reset_index(drop=True)

    pred_pool = modelo_final.predict(df_pool[cols].to_numpy(dtype=np.float64))
    _, positive_pool = pcl.calcular_score(pred_pool, df_pool["target_kw"].to_numpy())
    ts_pool = df_pool["ts_end"].to_numpy()
    return positive_pool, ts_pool, df_pool


# ==========================================================================================
# 9) Calibracion final H_MULTIWINDOW_180 -- reutiliza literalmente cth.calibrar_multiwindow
# ==========================================================================================

def calibrar_h_final(positive_pool: np.ndarray, ts_pool: np.ndarray, fin_pretest: pd.Timestamp):
    tabla_mw, sel_mw = cth.calibrar_multiwindow(positive_pool, ts_pool, fin_pretest)
    status = "FINAL_FREEZE_FAILED" if sel_mw["deficiente"] else "READY_FOR_TEST"
    return tabla_mw, sel_mw, status


# ==========================================================================================
# 10) Comprobaciones operativas del H final sobre FINAL_CAL_POOL_180 limpio
# ==========================================================================================

def comprobaciones_operativas(positive_pool: np.ndarray, ts_pool: np.ndarray, umbral_h: float, fin_pretest: pd.Timestamp) -> dict:
    activo = positive_pool > umbral_h
    sat = rtp.calcular_saturacion(activo, ts_pool, RESOLUCION_MIN)
    fa_agregada = contar_eventos(activo) / CAL_POOL_DIAS
    dur_min = _duracion_eventos(activo, resolucion_min=RESOLUCION_MIN)
    dur_h = dur_min / 60.0

    bloques = cth._bloques_multiwindow(ts_pool, fin_pretest)
    fa_bloques = []
    for b in range(cth.N_BLOQUES_MULTIWINDOW):
        mb = bloques == b
        fa_bloques.append(contar_eventos(activo[mb]) / cth.DIAS_POR_BLOQUE)

    return {
        "fa_agregada": fa_agregada, "fa_por_bloque": fa_bloques,
        "active_fraction": sat["active_fraction"], "duracion_media_h": float(dur_h.mean()),
        "duracion_mediana_h": float(np.median(dur_h)), "duracion_p95_h": float(np.percentile(dur_h, 95)),
        "duracion_maxima_h": sat["longest_clean_event_hours"], "seven_day_max_active_fraction": sat["seven_day_max_active_fraction"],
        "days_over_25pct_active": sat["days_over_25pct_active"], "days_over_50pct_active": sat["days_over_50pct_active"],
        "consecutive_days_over_50": sat["consecutive_days_over_50"], "falla_saturacion": sat["falla_saturacion"],
    }


# ==========================================================================================
# 12-13) Artefactos finales + JSON principal
# ==========================================================================================

def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def congelar_artefactos(modelo_final, perfil_final: pd.DataFrame, metadatos_h: dict, tabla_mw: pd.DataFrame,
                         sel_mw: dict, status: str, comprobaciones: dict, periodos: dict, payload_decision: dict) -> dict:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    cols = opt.PERIODIC_BASE_FEATURES

    joblib.dump(modelo_final, MODELOS_DIR / "histgb_periodic_final.joblib")
    joblib.dump(perfil_final, MODELOS_DIR / "profile_train_final.joblib")

    with open(MODELOS_DIR / "features_h_final.json", "w", encoding="utf-8") as f:
        json.dump({"features_orden": cols}, f, indent=2, ensure_ascii=False)
    with open(MODELOS_DIR / "hyperparameters_h_final.json", "w", encoding="utf-8") as f:
        json.dump(opt.CONFIG_HISTGB_PERIODIC_ACTUAL, f, indent=2, ensure_ascii=False, default=str)
    with open(MODELOS_DIR / "threshold_h_final.json", "w", encoding="utf-8") as f:
        json.dump({"umbral_h": float(sel_mw["umbral"]), "banda": sel_mw["banda"], "deficiente": bool(sel_mw["deficiente"]),
                   "procedimiento": "H_MULTIWINDOW_180"}, f, indent=2, ensure_ascii=False, default=str)
    with open(MODELOS_DIR / "threshold_p_final.json", "w", encoding="utf-8") as f:
        json.dump({"umbral_p": THRESHOLD_P_FINAL, "procedimiento": "P_MULTI_SEASON", "recalibrado_en_esta_fase": False}, f, indent=2, ensure_ascii=False)

    tabla_mw.to_csv(MODELOS_DIR / "calibration_multiwindow_blocks.csv", index=False)
    pd.DataFrame([comprobaciones]).drop(columns=["fa_por_bloque"]).assign(
        **{f"fa_bloque_{i}": v for i, v in enumerate(comprobaciones["fa_por_bloque"])}
    ).to_csv(MODELOS_DIR / "calibration_multiwindow_metrics.csv", index=False)

    transformaciones = {
        "score_p": "relative_kw = TCN-AE congelado, sin transformar", "score_h": "causal_relative_kw = max(0, (pred-real)/max(|pred|,0.1))",
        "forward_fill_p": "causal, np.searchsorted(ts, objetivo, side='right')-1, sin backfill, se propaga solo el estado booleano",
        "regla_fusion": "active_OR = active_P_ffill OR active_H", "sin_cooldown": True, "sin_AND": True, "sin_ponderaciones": True,
        "sin_confirmaciones_temporales": True, "sin_tolerancias_temporales": True, "sin_reglas_por_duracion": True,
    }
    with open(MODELOS_DIR / "transformaciones_finales.json", "w", encoding="utf-8") as f:
        json.dump(transformaciones, f, indent=2, ensure_ascii=False)

    payload_principal = {
        "selected_configuration": "OR_PMULTI_HMULTIWINDOW", "status": status,
        "P": {"checkpoint": "models/tcn_ae_ventana360.pt", "score": "relative_kw", "threshold": THRESHOLD_P_FINAL,
              "resolution_minutes": 60, "forward_fill": "causal"},
        "H": {"model_path": "models/final_or_pretest/histgb_periodic_final.joblib", "score": "causal_relative_kw",
              "threshold": float(sel_mw["umbral"]), "resolution_minutes": 15, "features_orden": cols,
              "hiperparametros": opt.CONFIG_HISTGB_PERIODIC_ACTUAL,
              "entrenamiento": {"n_observaciones": metadatos_h["n_observaciones_entrenamiento"],
                                "fecha_inicio": metadatos_h["fecha_inicio_entrenamiento"], "fecha_fin": metadatos_h["fecha_fin_entrenamiento"]},
              "calibracion": {"procedimiento": "H_MULTIWINDOW_180", "banda": sel_mw["banda"], "deficiente": bool(sel_mw["deficiente"]),
                              "fa_agregada": comprobaciones["fa_agregada"], "fa_por_bloque": comprobaciones["fa_por_bloque"]},
              "procedimiento": "H_MULTIWINDOW_180"},
        "fusion": {"rule": "OR", "common_resolution_minutes": 15,
                   "event_definition": "event_start_t = active_OR_t AND NOT active_OR_{t-1}",
                   "induced_detection_definition": "event_start AND NOT active_clean_local (misma logica ya validada, fusion_p_histgb.simular_episodios_generico)"},
        "operational_limits": {"FA": {"limite_admisible": 0.15, "obtenido_agregado": comprobaciones["fa_agregada"]},
                               "active_fraction": {"limite": rtp.LIM_ACTIVE_FRACTION, "obtenido": comprobaciones["active_fraction"]},
                               "longest_event": {"limite_horas": rtp.LIM_LONGEST_CLEAN_HOURS, "obtenido_horas": comprobaciones["duracion_maxima_h"]},
                               "seven_day_max_active_fraction": {"limite": rtp.LIM_7DAY_MAX_ACTIVE, "obtenido": comprobaciones["seven_day_max_active_fraction"]}},
        "periodos": {"final_train_fin_exclusivo": str(periodos["final_cal_pool_ini"]), "final_cal_pool_180_ini": str(periodos["final_cal_pool_ini"]),
                     "final_cal_pool_180_fin_exclusivo": str(periodos["fin_pretest"]), "test_inicio": str(periodos["fin_pretest"])},
        "test": {"untouched": True, "inference_executed": False, "attacks_generated": False, "scores_calculated": False},
        "decision_previa": {"archivo": "models/auditoria_final_p_vs_or/configuracion_final_decision_pretest.json",
                            "criterios_cumplidos": payload_decision.get("criterios_cumplidos")},
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELOS_DIR / "configuracion_final_or_pretest.json", "w", encoding="utf-8") as f:
        json.dump(payload_principal, f, indent=2, ensure_ascii=False, default=str)

    with open(MODELOS_DIR / "environment_versions.json", "w", encoding="utf-8") as f:
        json.dump({"sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
                   "numpy_version": np.__version__, "pandas_version": pd.__version__}, f, indent=2, ensure_ascii=False)

    return payload_principal


def verificar_determinismo() -> dict:
    """Carga el modelo dos veces de forma independiente y comprueba predicciones bit-identicas
    sobre un lote sintetico (mismo numero/orden de features que el modelo congelado)."""
    cols = opt.PERIODIC_BASE_FEATURES
    rng = np.random.default_rng(0)
    X = rng.random((50, len(cols)))
    m1 = joblib.load(MODELOS_DIR / "histgb_periodic_final.joblib")
    m2 = joblib.load(MODELOS_DIR / "histgb_periodic_final.joblib")
    p1, p2 = m1.predict(X), m2.predict(X)
    idénticas = bool(np.array_equal(p1, p2))
    n_features_carga = int(m1.n_features_in_)
    return {"predicciones_bit_identicas": idénticas, "n_features_coincide": n_features_carga == len(cols)}


def calcular_y_guardar_hashes() -> dict:
    archivos = [
        "histgb_periodic_final.joblib", "profile_train_final.joblib", "features_h_final.json",
        "hyperparameters_h_final.json", "threshold_h_final.json", "threshold_p_final.json",
        "calibration_multiwindow_blocks.csv", "calibration_multiwindow_metrics.csv",
        "transformaciones_finales.json", "configuracion_final_or_pretest.json", "environment_versions.json",
    ]
    hashes = {nombre: _sha256(MODELOS_DIR / nombre) for nombre in archivos if (MODELOS_DIR / nombre).exists()}
    with open(MODELOS_DIR / "hashes_finales.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)
    return hashes


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("Fase 0: verificando la decision previa de la auditoria (sin recalcular nada con test)...")
    payload_decision = verificar_decision_previa()
    print("OK -- decision previa reproducida: OR_PMULTI_HMULTIWINDOW, 14/14 criterios, umbral_p=0.049748")

    print("\nPreparando FINAL_TRAIN y FINAL_CAL_POOL_180...")
    periodos = preparar_periodos()
    print(f"  FINAL_TRAIN: hasta {periodos['final_cal_pool_ini']}")
    print(f"  FINAL_CAL_POOL_180: {periodos['final_cal_pool_ini']} -> {periodos['fin_pretest']}")
    print(f"  test (solo timestamp de arranque leido): {periodos['fin_pretest']}")

    print("\nEntrenando el modelo H final (unico .fit, exclusivamente sobre FINAL_TRAIN limpio)...")
    modelo_final, perfil_final, metadatos_h = entrenar_h_final(periodos["particion_entrenamiento"])
    print(f"  n_observaciones={metadatos_h['n_observaciones_entrenamiento']}  "
          f"{metadatos_h['fecha_inicio_entrenamiento']} -> {metadatos_h['fecha_fin_entrenamiento']}")

    print("\nPrediciendo (limpio, sin ataques) sobre FINAL_CAL_POOL_180...")
    positive_pool, ts_pool, df_pool = predecir_pool_limpio(
        modelo_final, perfil_final, periodos["particion_entrenamiento"], periodos["particion_final_pool"], periodos["final_cal_pool_ini"])
    print(f"  n_puntos_pool={len(positive_pool)}")

    print("\nCalibrando H_MULTIWINDOW_180 (procedimiento ya validado, reutilizado literalmente)...")
    tabla_mw, sel_mw, status = calibrar_h_final(positive_pool, ts_pool, periodos["fin_pretest"])
    print(f"  umbral_h={sel_mw['umbral']:.6f}  banda={sel_mw['banda']}  deficiente={sel_mw['deficiente']}  status={status}")

    if status == "FINAL_FREEZE_FAILED":
        MODELOS_DIR.mkdir(parents=True, exist_ok=True)
        payload_fail = {"selected_configuration": "OR_PMULTI_HMULTIWINDOW", "status": status,
                        "motivo": f"ningun umbral de H_MULTIWINDOW_180 cumplio la banda principal ni la secundaria sobre FINAL_CAL_POOL_180 (banda obtenida: {sel_mw['banda']})",
                        "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(), "test_intacto": True}
        with open(MODELOS_DIR / "configuracion_final_or_pretest.json", "w", encoding="utf-8") as f:
            json.dump(payload_fail, f, indent=2, ensure_ascii=False, default=str)
        with open(MODELOS_DIR / "freeze_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"status": status, "fecha_utc": datetime.now(timezone.utc).isoformat()}, f, indent=2, ensure_ascii=False)
        print("\n>>> FINAL_FREEZE_FAILED -- no se abre test. Ver configuracion_final_or_pretest.json para el detalle.")
        return {"status": status}

    print("\nComprobaciones operativas del H final sobre FINAL_CAL_POOL_180 limpio (no es una nueva seleccion)...")
    comprobaciones = comprobaciones_operativas(positive_pool, ts_pool, float(sel_mw["umbral"]), periodos["fin_pretest"])
    print(f"  FA_agregada={comprobaciones['fa_agregada']:.4f}  active_fraction={comprobaciones['active_fraction']:.4f}  "
          f"falla_saturacion={comprobaciones['falla_saturacion']}")

    print("\nCongelando artefactos finales en models/final_or_pretest/...")
    payload_principal = congelar_artefactos(modelo_final, perfil_final, metadatos_h, tabla_mw, sel_mw, status, comprobaciones, periodos, payload_decision)

    print("Verificando determinismo (doble carga del modelo, predicciones bit-identicas)...")
    det = verificar_determinismo()
    assert det["predicciones_bit_identicas"], "el modelo congelado NO es determinista entre cargas -- congelacion detenida"
    assert det["n_features_coincide"], "el numero de features del modelo cargado no coincide con PERIODIC_BASE_FEATURES"
    print(f"  determinismo verificado: {det}")

    print("Calculando y guardando hashes SHA-256...")
    hashes = calcular_y_guardar_hashes()

    manifiesto = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(), "status": status, "n_archivos_congelados": len(hashes),
        "determinismo_verificado": det["predicciones_bit_identicas"], "test_intacto": True,
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
    }
    with open(MODELOS_DIR / "freeze_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)

    print(f"\n>>> STATUS FINAL: {status}")
    print("\nCONFIRMACION: en ningun momento se ha leido consumo de test, calculado scores de test, "
          "generado ataques de test ni ejecutado inferencia sobre test (unicamente su timestamp de arranque).")
    return {"status": status, "payload": payload_principal, "hashes": hashes, "comprobaciones": comprobaciones, "sel_mw": sel_mw}


if __name__ == "__main__":
    main()
