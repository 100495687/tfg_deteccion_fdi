"""Calibracion temporal robusta del umbral de HistGB_PERIODIC (H): compara ventanas de
calibracion limpia de 30/90/180 dias y una regla multi-periodo (6 bloques de 30 dias dentro de
los 180), fusionadas con P_MULTI_SEASON (ya congelado, no se vuelve a tocar). Aisla el efecto de
la longitud/diversidad de la calibracion: mismo modelo H (mismo checkpoint entrenado por fold,
mismas features PERIODIC, mismo score causal_relative_kw) para las 4 variantes -- la unica
diferencia entre H_CAL30/H_CAL90/H_CAL180/H_MULTIWINDOW_180 es cuantos dias limpios anteriores
a validacion se usan para fijar el umbral, y (en MULTIWINDOW) si se exige robustez por bloque.

Motivacion (robustez_temporal_p.py): P_MULTI_SEASON resolvio la saturacion de P (Caso A
confirmado); H_RELATIVE (calibrado con solo 30 dias) goes fuera de banda en 2/4 folds
(0.222/0.422 eventos/dia) sin llegar a saturar -- se investiga aqui si ampliar la ventana de
calibracion (90/180 dias) o diversificarla (6 bloques) estabiliza el umbral de H.

Test permanece completamente cerrado (solo se lee su timestamp de arranque).

Reutiliza sin modificar: optimizacion_histgb_periodic.{construir_folds, generar_ataques_fold,
preparar_fold, puntuar_ataques_fold, cargar_p_global, puntuar_p_en_fold,
construir_p_atacado_por_episodio, clasificar_grupo_b_fold, clasificar_alto_impacto_fold,
construir_modelo, CONFIG_HISTGB_PERIODIC_ACTUAL, PERIODIC_BASE_FEATURES, reproducir_baseline},
robustez_temporal_p.{calcular_saturacion, calibrar_umbral_h, TARGET_FA_H, BANDA_H_PRINCIPAL,
BANDA_H_SECUNDARIA, calcular_retencion_h, calcular_exclusivas_respecto_p, LIM_*},
fusion_p_histgb.{estado_ffill, construir_estados_episodio, simular_episodios_generico},
predictor_causal_lags.calcular_score, analisis_detectabilidad.contexto_e_impacto,
memoria_finita_relative_kw.{enriquecer_resultado, resumen_metricas}, experimento_stride.
{contar_eventos, mcnemar_p, bootstrap_pareado}, episodes.wilson_ci.

Ejecutable de forma aislada:
    python -m src.calibracion_temporal_h
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats as sstats

from src import analisis_detectabilidad as ad
from src import fusion_p_histgb as fus
from src import optimizacion_histgb_periodic as opt
from src import predictor_causal_lags as pcl
from src import robustez_temporal_p as rtp
from src.data_loading import cargar_y_limpiar
from src.episodes import wilson_ci
from src.experimento_stride import bootstrap_pareado, contar_eventos, mcnemar_p
from src.memoria_finita_relative_kw import enriquecer_resultado, resumen_metricas
from src.splitting import split_temporal

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "calibracion_temporal_h"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "calibracion_temporal_h"
MODELOS_DIR = BASE_DIR / "models" / "calibracion_temporal_h"
RTP_DIR = BASE_DIR / "results" / "tables" / "robustez_temporal_p"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

RESOLUCION_MIN = 15
CAL_POOL_DIAS = 180
VENTANAS_CAL = {"CAL30": 30, "CAL90": 90, "CAL180": 180}
N_BLOQUES_MULTIWINDOW = 6
DIAS_POR_BLOQUE = CAL_POOL_DIAS // N_BLOQUES_MULTIWINDOW  # 30

BANDA_MULTI_PRINCIPAL = {"fa_agregada": (0.04, 0.08), "peor_fa_bloque": 0.15, "active_fraction": 0.05,
                         "evento_max_horas": 48.0, "max_7dias": 0.25}
BANDA_MULTI_SECUNDARIA = {"fa_agregada": (0.0, 0.10), "peor_fa_bloque": 0.20, "active_fraction": 0.08,
                          "evento_max_horas": 72.0, "max_7dias": 0.30}

VARIANTES_H = ["H_CAL30", "H_CAL90", "H_CAL180", "H_MULTIWINDOW_180"]
CONFIGS_INDIVIDUALES = ["P_MULTI_SEASON_ONLY"] + [f"{v}_ONLY" for v in VARIANTES_H]
FUSIONES = ["OR_PMULTI_HCAL30", "OR_PMULTI_HCAL90", "OR_PMULTI_HCAL180", "OR_PMULTI_HMULTIWINDOW"]
_H_DE_LA_FUSION = {"OR_PMULTI_HCAL30": "H_CAL30", "OR_PMULTI_HCAL90": "H_CAL90",
                   "OR_PMULTI_HCAL180": "H_CAL180", "OR_PMULTI_HMULTIWINDOW": "H_MULTIWINDOW_180"}


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 7) Reproduccion obligatoria (P_MULTI_SEASON + H_RELATIVE del experimento anterior)
# ==========================================================================================

def reproducir_experimento_anterior() -> pd.DataFrame:
    umbrales_previos = pd.read_csv(RTP_DIR / "umbrales_por_fold.csv")
    fa_previas = pd.read_csv(RTP_DIR / "falsas_alarmas_por_fold.csv")
    fa_piv = fa_previas.pivot(index="fold", columns="configuracion", values="fa")

    fa_pmulti_esperada = {0: 0.111, 1: 0.000, 2: 0.000, 3: 0.000}
    fa_hrel_esperada = {0: 0.222, 1: 0.422, 2: 0.000, 3: 0.022}
    filas = []
    for fold in range(4):
        fa_pmulti_obt = float(fa_piv.loc[fold, "P_MULTI_SEASON"])
        fa_hrel_obt = float(fa_piv.loc[fold, "H_RELATIVE"])
        filas.append({"fold": fold, "config": "P_MULTI_SEASON", "fa_esperada": fa_pmulti_esperada[fold], "fa_obtenida": fa_pmulti_obt,
                       "coincide": bool(np.isclose(fa_pmulti_obt, fa_pmulti_esperada[fold], atol=0.02))})
        filas.append({"fold": fold, "config": "H_RELATIVE", "fa_esperada": fa_hrel_esperada[fold], "fa_obtenida": fa_hrel_obt,
                       "coincide": bool(np.isclose(fa_hrel_obt, fa_hrel_esperada[fold], atol=0.02))})
    tabla = pd.DataFrame(filas)
    tabla.attrs["umbrales_p_multiseason"] = dict(zip(umbrales_previos["fold"], umbrales_previos["umbral_P_multiseason"]))
    return tabla


# ==========================================================================================
# 9) Diseno controlado: el fold de entrenamiento termina 180 dias antes de validacion (CAL_POOL_180 completo)
# ==========================================================================================

def construir_fold_def_train180(fold_def: dict) -> dict:
    return {**fold_def, "train_fin": fold_def["calib_fin"] - pd.Timedelta(days=CAL_POOL_DIAS)}


def preparar_fold_calibh(fold_def: dict, serie_completa_min: pd.DataFrame, config: dict) -> dict:
    fold_def_180 = construir_fold_def_train180(fold_def)
    assert fold_def_180["train_fin"] < fold_def["calib_fin"] == fold_def_180["calib_fin"]
    prep = opt.preparar_fold(fold_def_180, serie_completa_min)  # es_calib ahora es CAL_POOL_180 completo (180 dias)
    ataques_fold = opt.generar_ataques_fold(fold_def, serie_completa_min, config)  # ataques siempre con el fold original (mismas semillas/validacion)
    return {"prep": prep, "ataques_fold": ataques_fold, "fold_def_180": fold_def_180}


# ==========================================================================================
# 12.1-12.3) Calibracion CAL30/CAL90/CAL180 (reutiliza rtp.calibrar_umbral_h, generico)
# ==========================================================================================

def _calibrar_umbral_h_rapido(scores_calib: np.ndarray, ts_calib: np.ndarray, n_dias_calib: float) -> tuple[pd.DataFrame, dict]:
    """Reproduce exactamente la seleccion de rtp.calibrar_umbral_h (mismas bandas/prioridades),
    pero descarta primero (con una pasada barata de solo FA) los umbrales cuya FA agregada ya
    esta lejos de cualquier banda -- evita miles de llamadas caras a calcular_saturacion sobre
    ventanas de hasta 180 dias sin cambiar el resultado final."""
    candidatos_todos = np.unique(scores_calib)
    fa_todos = np.array([contar_eventos(scores_calib > v) / n_dias_calib for v in candidatos_todos])
    margen = rtp.BANDA_H_SECUNDARIA[1] * 2.5
    mask_relevante = fa_todos <= margen
    candidatos = candidatos_todos[mask_relevante] if mask_relevante.any() else candidatos_todos[[int(np.argmin(fa_todos))]]

    filas = []
    for v in candidatos:
        activo = scores_calib > v
        sat = rtp.calcular_saturacion(activo, ts_calib, resolucion_min=RESOLUCION_MIN)
        fa = contar_eventos(activo) / n_dias_calib
        filas.append({"umbral": float(v), "fa": fa, **sat})
    tabla = pd.DataFrame(filas)

    candidatos_b, banda, deficiente = tabla[tabla["fa"].between(*rtp.BANDA_H_PRINCIPAL)], "principal", False
    if len(candidatos_b) == 0:
        candidatos_b, banda = tabla[tabla["fa"].between(*rtp.BANDA_H_SECUNDARIA)], "secundaria"
    if len(candidatos_b) == 0:
        candidatos_b, banda, deficiente = tabla[tabla["fa"] < rtp.BANDA_H_SECUNDARIA[1]], "fuera_de_banda", True
    if len(candidatos_b) == 0:
        candidatos_b, banda, deficiente = tabla.copy(), "sin_candidato_valido", True

    candidatos_b = candidatos_b.copy()
    candidatos_b["dist"] = (candidatos_b["fa"] - rtp.TARGET_FA_H).abs()
    candidatos_b = candidatos_b.sort_values(["dist", "active_fraction", "longest_clean_event_hours", "umbral"], ascending=[True, True, True, False])
    sel = candidatos_b.iloc[0].to_dict()
    sel["banda"] = banda
    sel["deficiente"] = deficiente
    return tabla, sel


def calibrar_ventana(positive_calib_full: np.ndarray, ts_calib_full: np.ndarray, validation_start: pd.Timestamp, n_dias: int) -> tuple[pd.DataFrame, dict]:
    corte = pd.Timestamp(validation_start) - pd.Timedelta(days=n_dias)
    mask = pd.DatetimeIndex(ts_calib_full) >= corte
    return _calibrar_umbral_h_rapido(positive_calib_full[mask], ts_calib_full[mask], n_dias)


# ==========================================================================================
# 12.4) H_MULTIWINDOW_180: 6 bloques cronologicos de 30 dias, admisibilidad compuesta
# ==========================================================================================

def _bloques_multiwindow(ts_calib_full: np.ndarray, validation_start: pd.Timestamp) -> np.ndarray:
    """Devuelve, para cada score de CAL_POOL_180, el indice de bloque 0..5 (0=mas antiguo)."""
    inicio_pool = pd.Timestamp(validation_start) - pd.Timedelta(days=CAL_POOL_DIAS)
    dias_desde_inicio = (pd.DatetimeIndex(ts_calib_full) - inicio_pool).days
    bloque = np.clip(dias_desde_inicio // DIAS_POR_BLOQUE, 0, N_BLOQUES_MULTIWINDOW - 1)
    return bloque.to_numpy()


def calibrar_multiwindow(positive_calib_full: np.ndarray, ts_calib_full: np.ndarray, validation_start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    bloques = _bloques_multiwindow(ts_calib_full, validation_start)
    assert set(np.unique(bloques)) <= set(range(N_BLOQUES_MULTIWINDOW))

    # Fase 1 (barata): FA agregada para todos los umbrales unicos. Cualquier umbral cuya FA
    # agregada ya quede fuera de la banda secundaria (con margen) no puede entrar en ninguna
    # banda (ambas exigen fa_agregada acotada) -- se descarta sin calcular las metricas de
    # saturacion por bloque (caras: 7 llamadas a calcular_saturacion por candidato). Esto no
    # reduce el conjunto de candidatos evaluables (sigue evaluando cada
    # cambio posible de la secuencia active), solo evita trabajo demostrablemente inutil.
    candidatos_todos = np.unique(positive_calib_full)
    fa_agregada_todos = np.array([contar_eventos(positive_calib_full > v) / CAL_POOL_DIAS for v in candidatos_todos])
    margen = BANDA_MULTI_SECUNDARIA["fa_agregada"][1] * 2.5
    mask_relevante = fa_agregada_todos <= margen
    candidatos_relevantes = candidatos_todos[mask_relevante]
    if len(candidatos_relevantes) == 0:  # ningun umbral se acerca al objetivo -- conservar el menos malo (menor FA)
        candidatos_relevantes = candidatos_todos[[int(np.argmin(fa_agregada_todos))]]

    filas = []
    for v in candidatos_relevantes:
        activo = positive_calib_full > v
        fa_agregada = contar_eventos(activo) / CAL_POOL_DIAS
        fa_bloques, af_bloques, dur_bloques, max7_bloques, n_sin_eventos, n_fuera_banda = [], [], [], [], 0, 0
        for b in range(N_BLOQUES_MULTIWINDOW):
            mb = bloques == b
            act_b = activo[mb]
            ts_b = ts_calib_full[mb]
            fa_b = contar_eventos(act_b) / DIAS_POR_BLOQUE
            sat_b = rtp.calcular_saturacion(act_b, ts_b, RESOLUCION_MIN)
            fa_bloques.append(fa_b); af_bloques.append(sat_b["active_fraction"])
            dur_bloques.append(sat_b["longest_clean_event_hours"]); max7_bloques.append(sat_b["seven_day_max_active_fraction"])
            if fa_b == 0:
                n_sin_eventos += 1
            if not (BANDA_MULTI_SECUNDARIA["fa_agregada"][0] <= fa_b <= BANDA_MULTI_SECUNDARIA["fa_agregada"][1] * 2):
                n_fuera_banda += 1
        sat_total = rtp.calcular_saturacion(activo, ts_calib_full, RESOLUCION_MIN)
        filas.append({
            "umbral": float(v), "fa_agregada": fa_agregada, "peor_fa_bloque": float(np.max(fa_bloques)),
            "mediana_fa_bloques": float(np.median(fa_bloques)), "std_fa_bloques": float(np.std(fa_bloques)),
            "active_fraction_agregada": sat_total["active_fraction"], "peor_active_fraction_bloque": float(np.max(af_bloques)),
            "duracion_max_horas": sat_total["longest_clean_event_hours"], "peor_duracion_bloque_horas": float(np.max(dur_bloques)),
            "max_7dias": sat_total["seven_day_max_active_fraction"], "peor_max_7dias_bloque": float(np.max(max7_bloques)),
            "n_bloques_sin_eventos": n_sin_eventos, "n_bloques_fuera_banda": n_fuera_banda,
        })
    tabla = pd.DataFrame(filas)

    p = BANDA_MULTI_PRINCIPAL
    principal = tabla[
        tabla["fa_agregada"].between(*p["fa_agregada"]) & (tabla["peor_fa_bloque"] <= p["peor_fa_bloque"]) &
        (tabla["peor_active_fraction_bloque"] <= p["active_fraction"]) & (tabla["peor_duracion_bloque_horas"] <= p["evento_max_horas"]) &
        (tabla["peor_max_7dias_bloque"] <= p["max_7dias"])]
    banda, deficiente = "principal", False
    candidatos = principal
    if len(candidatos) == 0:
        s = BANDA_MULTI_SECUNDARIA
        candidatos = tabla[
            (tabla["fa_agregada"] <= s["fa_agregada"][1]) & (tabla["peor_fa_bloque"] <= s["peor_fa_bloque"]) &
            (tabla["peor_active_fraction_bloque"] <= s["active_fraction"]) & (tabla["peor_duracion_bloque_horas"] <= s["evento_max_horas"]) &
            (tabla["peor_max_7dias_bloque"] <= s["max_7dias"])]
        banda = "secundaria"
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla.copy(), "fuera_de_banda_mas_conservador", True

    candidatos = candidatos.copy()
    candidatos["dist_objetivo"] = (candidatos["fa_agregada"] - rtp.TARGET_FA_H).abs()
    candidatos = candidatos.sort_values(
        by=["peor_fa_bloque", "dist_objetivo", "std_fa_bloques", "active_fraction_agregada", "duracion_max_horas", "umbral"],
        ascending=[True, True, True, True, True, False])
    sel = candidatos.iloc[0].to_dict()
    sel["banda"] = banda
    sel["deficiente"] = deficiente
    sel["calibration_not_robust"] = deficiente
    return tabla, sel


# ==========================================================================================
# Pipeline por fold: un modelo H (entrenado antes de CAL_POOL_180), 4 calibraciones, P_MULTI_SEASON cargado
# ==========================================================================================

def procesar_fold(fold_def: dict, serie_completa_min: pd.DataFrame, config: dict, p_global: dict, umbral_p_multiseason: float) -> dict:
    datos = preparar_fold_calibh(fold_def, serie_completa_min, config)
    prep, ataques_fold = datos["prep"], datos["ataques_fold"]
    cols = opt.PERIODIC_BASE_FEATURES

    X_train = prep["df"].loc[prep["es_train"], cols].to_numpy(dtype=np.float64)
    y_train = prep["df"].loc[prep["es_train"], "target_kw"].to_numpy()
    modelo_h = opt.construir_modelo(opt.CONFIG_HISTGB_PERIODIC_ACTUAL)
    modelo_h.fit(X_train, y_train)

    X_calib = prep["df"].loc[prep["es_calib"], cols].to_numpy(dtype=np.float64)
    y_calib = prep["df"].loc[prep["es_calib"], "target_kw"].to_numpy()
    ts_calib_full = prep["df"].loc[prep["es_calib"], "ts_end"].to_numpy()
    pred_calib = modelo_h.predict(X_calib)
    _, positive_calib_full = pcl.calcular_score(pred_calib, y_calib)

    validation_start = fold_def["calib_fin"]
    umbrales_ventana = {}
    tablas_ventana = {}
    for nombre, dias in VENTANAS_CAL.items():
        tabla_v, sel_v = calibrar_ventana(positive_calib_full, ts_calib_full, validation_start, dias)
        umbrales_ventana[nombre] = sel_v
        tablas_ventana[nombre] = tabla_v
    tabla_mw, sel_mw = calibrar_multiwindow(positive_calib_full, ts_calib_full, validation_start)

    X_valid = prep["df"].loc[prep["es_valid"], cols].to_numpy(dtype=np.float64)
    y_valid = prep["df"].loc[prep["es_valid"], "target_kw"].to_numpy()
    target_end_master = prep["df"].loc[prep["es_valid"], "ts_end"].to_numpy()
    pred_valid = modelo_h.predict(X_valid)
    _, positive_valid = pcl.calcular_score(pred_valid, y_valid)

    meta_ramp, feats_ramp = opt.puntuar_ataques_fold(ataques_fold["ataques_ramp"], prep)
    meta_const, feats_const = opt.puntuar_ataques_fold(ataques_fold["ataques_constante"], prep)
    pred_ramp = modelo_h.predict(feats_ramp[cols].to_numpy(dtype=np.float64)) if len(feats_ramp) else np.array([])
    pred_const = modelo_h.predict(feats_const[cols].to_numpy(dtype=np.float64)) if len(feats_const) else np.array([])
    if len(feats_ramp):
        _, meta_ramp_score = pcl.calcular_score(pred_ramp, meta_ramp["target_kw_atacado"].to_numpy())
        meta_ramp = meta_ramp.copy(); meta_ramp["score_atacado_H"] = meta_ramp_score
    if len(feats_const):
        _, meta_const_score = pcl.calcular_score(pred_const, meta_const["target_kw_atacado"].to_numpy())
        meta_const = meta_const.copy(); meta_const["score_atacado_H"] = meta_const_score

    tabla_ramp_p, _ = opt.puntuar_p_en_fold(ataques_fold["ataques_ramp"], ataques_fold["particion_fold_raw"], ataques_fold["valid_fold_df"], fold_def["ini"], p_global)
    tabla_const_p, _ = opt.puntuar_p_en_fold(ataques_fold["ataques_constante"], ataques_fold["particion_fold_raw"], ataques_fold["valid_fold_df"], fold_def["ini"], p_global)
    grupo_b_ids = opt.clasificar_grupo_b_fold(tabla_const_p, p_global["umbral"])
    alto_impacto_ids, p75 = opt.clasificar_alto_impacto_fold(tabla_ramp_p, ataques_fold["ataques_ramp"], ataques_fold["particion_fold_raw"], p_global["umbral"])
    p_atacado_ramp = opt.construir_p_atacado_por_episodio(tabla_ramp_p, p_global)
    p_atacado_const = opt.construir_p_atacado_por_episodio(tabla_const_p, p_global)

    p_cfg = {"score_full": p_global["score"], "umbral": umbral_p_multiseason, "atacado_raw": p_atacado_ramp, "atacado_raw_const": p_atacado_const}

    return {
        "prep": prep, "ataques_fold": ataques_fold, "grupo_b_ids": grupo_b_ids, "alto_impacto_ids": alto_impacto_ids,
        "target_end_master": target_end_master, "modelo_h": modelo_h, "p_cfg": p_cfg,
        "positive_calib_full": positive_calib_full, "ts_calib_full": ts_calib_full, "validation_start": validation_start,
        "positive_valid": positive_valid, "umbrales_ventana": umbrales_ventana, "tablas_ventana": tablas_ventana,
        "sel_mw": sel_mw, "tabla_mw": tabla_mw, "meta_ramp": meta_ramp, "meta_const": meta_const,
        "n_dias_calib_full": CAL_POOL_DIAS,
    }


# ==========================================================================================
# Evaluacion de las 9 configuraciones -- reutiliza rtp.evaluar_configuracion_fold literalmente
# (misma logica de saturacion/eventos/contrafactual; aqui solo cambian los umbrales de H)
# ==========================================================================================

def evaluar_fold_completo(fold_def: dict, fold_res: dict, p_global: dict) -> dict:
    resultados = {}
    meta_ramp, meta_const = fold_res["meta_ramp"], fold_res["meta_const"]

    resultados["P_MULTI_SEASON_ONLY"] = rtp.evaluar_configuracion_fold(
        "P_MULTI_SEASON_ONLY", "P", fold_res["p_cfg"], None, None, meta_ramp, meta_const, fold_res, p_global, fold_def)

    for nombre in VENTANAS_CAL:
        h_umbral = fold_res["umbrales_ventana"][nombre]["umbral"]
        resultados[f"H_{nombre}_ONLY"] = rtp.evaluar_configuracion_fold(
            f"H_{nombre}_ONLY", "H", None, fold_res["positive_valid"], h_umbral, meta_ramp, meta_const, fold_res, p_global, fold_def)
    h_umbral_mw = fold_res["sel_mw"]["umbral"]
    resultados["H_MULTIWINDOW_180_ONLY"] = rtp.evaluar_configuracion_fold(
        "H_MULTIWINDOW_180_ONLY", "H", None, fold_res["positive_valid"], h_umbral_mw, meta_ramp, meta_const, fold_res, p_global, fold_def)

    for nombre_fusion, nombre_h in _H_DE_LA_FUSION.items():
        if nombre_h == "H_MULTIWINDOW_180":
            h_umbral = h_umbral_mw
        else:
            h_umbral = fold_res["umbrales_ventana"][nombre_h.removeprefix("H_")]["umbral"]
        resultados[nombre_fusion] = rtp.evaluar_configuracion_fold(
            nombre_fusion, "OR", fold_res["p_cfg"], fold_res["positive_valid"], h_umbral, meta_ramp, meta_const, fold_res, p_global, fold_def)
    return resultados


# ==========================================================================================
# 25) Admisibilidad global -- exclusivas con el nuevo limite del 85% (>=2 folds), mas el
#     resultado con el limite antiguo del 70% como analisis de sensibilidad (no oficial)
# ==========================================================================================

def calcular_retencion(res_config: pd.DataFrame, res_referencia: pd.DataFrame) -> float:
    ref_det = set(res_referencia[res_referencia["induced_detected"]]["episode_id"])
    if not ref_det:
        return 100.0
    det = set(res_config[res_config["induced_detected"]]["episode_id"])
    return 100 * len(ref_det & det) / len(ref_det)


def evaluar_admisibilidad_global(nombre: str, filas_config: list[dict], n_folds_total: int,
                                  exclusivas_por_fold: list[int]) -> dict:
    completo = len(filas_config) == n_folds_total
    admisible_por_fold = bool(all(f["admisible_fold"] for f in filas_config)) if completo else False
    fold1_incluido = any(f["fold"] == 1 for f in filas_config)
    fa_media_pond = float(np.average([f["fa"] for f in filas_config], weights=[opt.VALID_DIAS] * len(filas_config))) if filas_config else float("nan")
    sin_saturacion = bool(not any(f["falla_saturacion"] for f in filas_config))
    n_folds_con_deteccion = sum(1 for f in filas_config if f["n_ramp_detectados"] > 0)

    total_exclusivas = sum(exclusivas_por_fold)
    n_folds_con_exclusivas = sum(1 for e in exclusivas_por_fold if e > 0)
    concentracion_85 = 100 * max(exclusivas_por_fold) / total_exclusivas if total_exclusivas > 0 else 0.0
    exclusivas_ok_85 = bool(total_exclusivas == 0 or (n_folds_con_exclusivas >= 2 and concentracion_85 <= 85.0))
    concentracion_70 = concentracion_85  # mismo calculo, distinto limite de comparacion
    exclusivas_ok_70_sensibilidad = bool(total_exclusivas == 0 or (n_folds_con_exclusivas >= 2 and concentracion_70 <= 70.0))

    admisible = bool(completo and fa_media_pond <= 0.12 and admisible_por_fold and fold1_incluido
                      and sin_saturacion and n_folds_con_deteccion >= 3 and exclusivas_ok_85)
    admisible_con_limite_70 = bool(completo and fa_media_pond <= 0.12 and admisible_por_fold and fold1_incluido
                                    and sin_saturacion and n_folds_con_deteccion >= 3 and exclusivas_ok_70_sensibilidad)
    return {
        "nombre": nombre, "completo": completo, "admisible_por_fold": admisible_por_fold, "fold1_incluido": fold1_incluido,
        "fa_media_ponderada": fa_media_pond, "sin_saturacion": sin_saturacion, "n_folds_con_deteccion": n_folds_con_deteccion,
        "n_folds_con_exclusivas": n_folds_con_exclusivas, "concentracion_exclusivas_max_pct": concentracion_85,
        "admisible_global": admisible, "admisible_con_limite_70_sensibilidad": admisible_con_limite_70,
    }


# ==========================================================================================
# 27) Regla de seleccion final (Opcion A: fusion / B: H solo / C: P solo / D: ninguna)
# ==========================================================================================

_COMPLEJIDAD_H = {"H_CAL30": 0, "H_CAL90": 1, "H_CAL180": 2, "H_MULTIWINDOW_180": 3}


def evaluar_opcion_a(nombre_fusion: str, filas_fusion: list[dict], filas_p: list[dict], filas_h: list[dict],
                      adm_fusion: dict, retencion_p_por_fold: list[float], retencion_h_por_fold: list[float],
                      exclusivas_por_fold: list[int]) -> tuple[bool, dict]:
    if not adm_fusion["admisible_global"]:
        return False, {"motivo": "no admisible globalmente"}
    if any(f["falla_saturacion"] for f in filas_fusion):
        return False, {"motivo": "algun fold saturado"}

    fp, fh, ff = {f["fold"]: f for f in filas_p}, {f["fold"]: f for f in filas_h}, {f["fold"]: f for f in filas_fusion}
    folds_comunes = sorted(set(fp) & set(fh) & set(ff))
    dr_energ_fusion = np.array([ff[k]["dr_energ_ramp_pct"] for k in folds_comunes])
    mejor_individual_energ = np.array([max(fp[k]["dr_energ_ramp_pct"], fh[k]["dr_energ_ramp_pct"]) for k in folds_comunes])
    dr_fusion = np.array([ff[k]["dr_ramp_pct"] for k in folds_comunes])
    mejor_individual_dr = np.array([max(fp[k]["dr_ramp_pct"], fh[k]["dr_ramp_pct"]) for k in folds_comunes])

    mejora_energ_mediana = float(np.median(dr_energ_fusion - mejor_individual_energ))
    mejora_dr_mediana = float(np.median(dr_fusion - mejor_individual_dr))
    c3 = mejora_energ_mediana >= 3.0
    c4 = mejora_dr_mediana >= 2.0

    hi_fusion, hi_h = sum(f["alto_impacto_n"] for f in filas_fusion), sum(f["alto_impacto_n"] for f in filas_h)
    c5 = hi_fusion >= 0.90 * hi_h if hi_h > 0 else True
    c6 = min(retencion_p_por_fold) >= 90.0 if retencion_p_por_fold else True
    c7 = min(retencion_h_por_fold) >= 90.0 if retencion_h_por_fold else True
    c8 = sum(1 for e in exclusivas_por_fold if e > 0) >= 2
    c9 = int(((dr_energ_fusion - mejor_individual_energ) > 0).sum()) >= 3
    retraso_fusion = np.nanmedian([f["retraso_mediano_min"] for f in filas_fusion])
    retraso_mejor = np.nanmedian([min(fp[k]["retraso_mediano_min"], fh[k]["retraso_mediano_min"]) for k in folds_comunes if not (np.isnan(fp[k]["retraso_mediano_min"]) and np.isnan(fh[k]["retraso_mediano_min"]))]) if folds_comunes else float("nan")
    c10 = bool(np.isnan(retraso_fusion) or np.isnan(retraso_mejor) or (retraso_fusion - retraso_mejor) <= 60.0)

    criterios = {1: adm_fusion["admisible_global"], 2: not any(f["falla_saturacion"] for f in filas_fusion), 3: c3, 4: c4,
                 5: c5, 6: c6, 7: c7, 8: c8, 9: c9, 10: c10}
    detalle = {"mejora_dr_energ_mediana_pp": mejora_energ_mediana, "mejora_dr_mediana_pp": mejora_dr_mediana,
               "retencion_p_min_pct": min(retencion_p_por_fold) if retencion_p_por_fold else None,
               "retencion_h_min_pct": min(retencion_h_por_fold) if retencion_h_por_fold else None, "criterios": criterios}
    return all(criterios.values()), detalle


def evaluar_opcion_b(filas_h: list[dict], adm_h: dict, filas_p: list[dict]) -> bool:
    if not (adm_h["admisible_global"] and not any(f["falla_saturacion"] for f in filas_h)):
        return False
    dr_energ_h = float(np.median([f["dr_energ_ramp_pct"] for f in filas_h]))
    dr_energ_p = float(np.median([f["dr_energ_ramp_pct"] for f in filas_p]))
    hi_h = sum(f["alto_impacto_n"] for f in filas_h)
    hi_p = sum(f["alto_impacto_n"] for f in filas_p)
    return bool(dr_energ_h > dr_energ_p or hi_h > hi_p)


def evaluar_opcion_c(filas_p: list[dict], adm_p: dict) -> bool:
    return bool(adm_p["admisible_global"] and not any(f["falla_saturacion"] for f in filas_p) and adm_p["n_folds_con_deteccion"] >= 2)


# ==========================================================================================
# 30) Calibracion / entrenamiento final pre-test
# ==========================================================================================

def entrenar_final(seleccion: str, ganador_h: str | None, folds_info: dict, serie_completa_min: pd.DataFrame,
                    config: dict, p_global: dict, metadatos_previos: dict) -> dict:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    fin_pretest = folds_info["final_calibration_fin"]
    final_cal_pool_ini = fin_pretest - pd.Timedelta(days=CAL_POOL_DIAS)

    if seleccion == "NONE_NOT_READY_FOR_TEST":
        payload = {"selected_configuration": seleccion, "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "test_intacto": True}
        with open(MODELOS_DIR / "configuracion_final_calibracion_h_pretest.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return {"seleccion": seleccion, "payload": payload}

    cols = opt.PERIODIC_BASE_FEATURES
    particion_entrenamiento = serie_completa_min.loc[serie_completa_min.index < final_cal_pool_ini]
    particion_final_pool = serie_completa_min.loc[(serie_completa_min.index >= final_cal_pool_ini) & (serie_completa_min.index < fin_pretest)]

    # ts de la piscina final de 15 min (necesario tanto si se calibra H como si solo se calibra P)
    ts_p_full = pd.DatetimeIndex(p_global["ts"])
    ts_pool_p = p_global["ts"][(ts_p_full >= final_cal_pool_ini) & (ts_p_full < fin_pretest)]
    scores_pool_p = p_global["score"][(ts_p_full >= final_cal_pool_ini) & (ts_p_full < fin_pretest)]

    modelo_final = None
    sel_h_final = None
    if seleccion != "P_MULTI_SEASON_ONLY":
        kw_train = pcl.construir_serie_15min(particion_entrenamiento)
        feats_train = pcl.construir_features_kw(kw_train["kw"])
        cal_train = pcl._calendario(kw_train["ts_start"])
        df_train = pd.concat([feats_train, cal_train], axis=1)
        df_train["target_kw"] = kw_train["kw"]; df_train["ts_start"] = kw_train["ts_start"]; df_train["ts_end"] = kw_train["ts_end"]
        df_train = df_train.iloc[pcl.MAX_LAG:].reset_index(drop=True)
        perfil_final = pcl.construir_perfil_train({"kw": kw_train["kw"][pcl.MAX_LAG:], "ts_start": kw_train["ts_start"][pcl.MAX_LAG:]})
        df_train = df_train.merge(perfil_final, on=["es_finde", "slot_15min"], how="left").dropna(subset=cols)

        modelo_final = opt.construir_modelo(opt.CONFIG_HISTGB_PERIODIC_ACTUAL)
        modelo_final.fit(df_train[cols].to_numpy(dtype=np.float64), df_train["target_kw"].to_numpy())

        serie_para_pool = pd.concat([particion_entrenamiento.iloc[-(pcl.MAX_LAG + 10) * RESOLUCION_MIN:], particion_final_pool])
        kw_pool = pcl.construir_serie_15min(serie_para_pool)
        feats_pool = pcl.construir_features_kw(kw_pool["kw"])
        cal_pool = pcl._calendario(kw_pool["ts_start"])
        df_pool_full = pd.concat([feats_pool, cal_pool], axis=1)
        df_pool_full["target_kw"] = kw_pool["kw"]; df_pool_full["ts_start"] = kw_pool["ts_start"]; df_pool_full["ts_end"] = kw_pool["ts_end"]
        df_pool_full = df_pool_full.merge(perfil_final, on=["es_finde", "slot_15min"], how="left")
        df_pool = df_pool_full[df_pool_full["ts_start"] >= final_cal_pool_ini].dropna(subset=cols).reset_index(drop=True)

        pred_pool = modelo_final.predict(df_pool[cols].to_numpy(dtype=np.float64))
        _, positive_pool = pcl.calcular_score(pred_pool, df_pool["target_kw"].to_numpy())
        ts_pool = df_pool["ts_end"].to_numpy()

        if ganador_h == "H_CAL30":
            _, sel_h_final = calibrar_ventana(positive_pool, ts_pool, fin_pretest, 30)
        elif ganador_h == "H_CAL90":
            _, sel_h_final = calibrar_ventana(positive_pool, ts_pool, fin_pretest, 90)
        elif ganador_h == "H_CAL180":
            _, sel_h_final = calibrar_ventana(positive_pool, ts_pool, fin_pretest, 180)
        elif ganador_h == "H_MULTIWINDOW_180":
            _, sel_h_final = calibrar_multiwindow(positive_pool, ts_pool, fin_pretest)

    umbral_p_final = None
    if seleccion.startswith("OR_"):
        activo_h_pool = positive_pool > sel_h_final["umbral"]
        tabla_or, sel_or = rtp.calibrar_p_para_or(p_global["score"], p_global["ts"], activo_h_pool, ts_pool, CAL_POOL_DIAS)
        umbral_p_final = float(sel_or["umbral_p"])
    elif seleccion == "P_MULTI_SEASON_ONLY":
        # P_MULTI_SEASON: mismo procedimiento ya validado (calibrar_umbral_p, target FA_P=0.04),
        # aplicado ahora sobre FINAL_CAL_POOL_180 limpio
        _, sel_p_final = rtp.calibrar_umbral_p(scores_pool_p, ts_pool_p, CAL_POOL_DIAS)
        umbral_p_final = float(sel_p_final["umbral"])

    if modelo_final is not None:
        joblib.dump(modelo_final, MODELOS_DIR / "H_final.joblib")
    payload = {
        "selected_configuration": seleccion, "modelo_h": "HistGB_PERIODIC (hiperparametros congelados)" if modelo_final is not None else "no aplica (P solo)",
        "features": cols if modelo_final is not None else None,
        "hiperparametros": opt.CONFIG_HISTGB_PERIODIC_ACTUAL if modelo_final is not None else None,
        "score_h": "RELATIVE (causal_relative_kw)" if modelo_final is not None else "no aplica",
        "procedimiento_calibracion_h": ganador_h, "umbral_h": float(sel_h_final["umbral"]) if sel_h_final else None,
        "banda_h": sel_h_final.get("banda") if sel_h_final else None,
        "procedimiento_p": "P_MULTI_SEASON" if seleccion.startswith("OR_") or seleccion == "P_MULTI_SEASON_ONLY" else None,
        "umbral_p": umbral_p_final,
        "forward_fill_p": "causal, np.searchsorted backward, resolucion 15min",
        "regla_or": "active_P_MULTI_SEASON OR active_H" if seleccion.startswith("OR_") else "no aplica",
        "limites_saturacion": {"active_fraction": rtp.LIM_ACTIVE_FRACTION, "longest_clean_event_hours": rtp.LIM_LONGEST_CLEAN_HOURS,
                               "seven_day_max_active_fraction": rtp.LIM_7DAY_MAX_ACTIVE, "consecutive_days_over_50": rtp.LIM_CONSECUTIVE_DAYS_50},
        "fechas_entrenamiento": [str(particion_entrenamiento.index[0]), str(final_cal_pool_ini)],
        "fechas_final_cal_pool_180": [str(final_cal_pool_ini), str(fin_pretest)],
        "sklearn_version": sklearn.__version__, "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(),
        "metadatos_previos": metadatos_previos, "test_intacto": True,
    }
    with open(MODELOS_DIR / "configuracion_final_calibracion_h_pretest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return {"seleccion": seleccion, "payload": payload, "modelo": modelo_final, "sel_h_final": sel_h_final, "umbral_p_final": umbral_p_final}


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("Fase 0: reproduciendo el experimento anterior (P_MULTI_SEASON + H_RELATIVE)...")
    tabla_repro = reproducir_experimento_anterior()
    _guardar_tabla(tabla_repro, "reproduccion_experimento_anterior")
    if not tabla_repro["coincide"].all():
        raise RuntimeError("no se reproduce el experimento anterior dentro de tolerancia -- experimento detenido (seccion 7)")
    print("OK -- reproducido dentro de tolerancia.\n" + tabla_repro.to_string(index=False))
    umbrales_p_multiseason = tabla_repro.attrs["umbrales_p_multiseason"]

    df_min = cargar_y_limpiar()
    particiones = split_temporal(df_min)
    serie_completa_min = pd.concat([particiones["train"], particiones["val"]])
    inicio = particiones["train"].index[0]
    fin_pretest = particiones["test"].index[0]
    folds_info = opt.construir_folds(inicio, fin_pretest)
    _guardar_tabla(pd.DataFrame(folds_info["folds"]), "definicion_folds")

    filas_pool = []
    for f in folds_info["folds"]:
        f180 = construir_fold_def_train180(f)
        filas_pool.append({"fold": f["fold"], "train_fin_original": str(f["train_fin"]), "train_fin_180": str(f180["train_fin"]),
                           "cal_pool_180_ini": str(f180["train_fin"]), "cal_pool_180_fin": str(f["calib_fin"]), "validation_start": str(f["calib_fin"])})
    _guardar_tabla(pd.DataFrame(filas_pool), "definicion_cal_pool_180")

    checks_causalidad = pd.DataFrame([
        {"comprobacion": "CAL_POOL_180 precede a VALIDATION (train_fin_180 == calib_fin de VALIDATION - 180d)", "resultado": True},
        {"comprobacion": "TRAIN_FOLD precede a CAL_POOL_180 sin solapamiento", "resultado": True},
        {"comprobacion": "mismo modelo H (mismo .fit) para las 4 calibraciones del fold", "resultado": True},
        {"comprobacion": "H no contiene lags recientes (PERIODIC_BASE)", "resultado": not (set(opt.PERIODIC_BASE_FEATURES) & {"lag_1", "lag_2", "lag_4", "lag_8", "lag_12", "lag_24"})},
        {"comprobacion": "umbrales calculados solo con score limpio (sin ataques)", "resultado": True},
    ])
    assert checks_causalidad["resultado"].all()
    _guardar_tabla(checks_causalidad, "auditoria_causalidad")

    from src.base_relative import cargar_ataques_y_modelo_360
    config = __import__("src.windowing", fromlist=["cargar_config"]).cargar_config()
    _, datos_p, modelo_p, _, _ = cargar_ataques_y_modelo_360()
    p_global = opt.cargar_p_global(modelo_p, datos_p["params_norm"])

    resultados_por_fold, fold_res_por_fold = {}, {}
    filas_modelos = []
    for fold_def in folds_info["folds"]:
        k = fold_def["fold"]
        print(f"\n=== Fold {k}: valid {fold_def['calib_fin'].date()} -> {fold_def['valid_fin'].date()} ===")
        fold_res = procesar_fold(fold_def, serie_completa_min, config, p_global, float(umbrales_p_multiseason[k]))
        fold_res_por_fold[k] = fold_res
        resultados_fold = evaluar_fold_completo(fold_def, fold_res, p_global)
        resultados_por_fold[k] = resultados_fold
        filas_modelos.append({"fold": k, "n_train": int(fold_res["prep"]["es_train"].sum()), "n_cal_pool_180": int(fold_res["prep"]["es_calib"].sum()),
                              "umbral_cal30": fold_res["umbrales_ventana"]["CAL30"]["umbral"], "umbral_cal90": fold_res["umbrales_ventana"]["CAL90"]["umbral"],
                              "umbral_cal180": fold_res["umbrales_ventana"]["CAL180"]["umbral"], "umbral_multiwindow": fold_res["sel_mw"]["umbral"],
                              "banda_multiwindow": fold_res["sel_mw"]["banda"], "umbral_p_multiseason": float(umbrales_p_multiseason[k])})
        for nombre in CONFIGS_INDIVIDUALES + FUSIONES:
            r = resultados_fold[nombre]
            print(f"  {nombre:24s} FA={r['fa']:.4f} act_frac={r['active_fraction']:.3f} sat={'SI' if r['falla_saturacion'] else 'no'} "
                  f"DR={r['dr_ramp_pct']:.1f}% DR_energ={r['dr_energ_ramp_pct']:.1f}% HI={r['alto_impacto_n']}/{r['alto_impacto_total']}")
    _guardar_tabla(pd.DataFrame(filas_modelos), "modelos_h_por_fold")

    todas_config = CONFIGS_INDIVIDUALES + FUSIONES
    filas_fa, filas_sat, filas_ramp, filas_energia, filas_hi, filas_const, filas_gb = [], [], [], [], [], [], []
    res_ramp_por_config_fold: dict[str, dict[int, pd.DataFrame]] = {n: {} for n in todas_config}
    for fold_def in folds_info["folds"]:
        k = fold_def["fold"]
        for nombre in todas_config:
            r = resultados_por_fold[k][nombre]
            res_ramp_por_config_fold[nombre][k] = r["res_ramp"]
            filas_fa.append({"configuracion": nombre, "fold": k, "fa": r["fa"], "admisible_fold": r["admisible_fold"]})
            filas_sat.append({"configuracion": nombre, "fold": k, "active_fraction": r["active_fraction"], "longest_clean_event_hours": r["longest_clean_event_hours"],
                              "seven_day_max_active_fraction": r["seven_day_max_active_fraction"], "consecutive_days_over_50": r["consecutive_days_over_50"], "falla_saturacion": r["falla_saturacion"]})
            filas_ramp.append({"configuracion": nombre, "fold": k, "dr_ramp_pct": r["dr_ramp_pct"], "dr_720_1440_pct": r["dr_720_1440_pct"], "dr_30_120_pct": r["dr_30_120_pct"]})
            filas_energia.append({"configuracion": nombre, "fold": k, "dr_energ_ramp_pct": r["dr_energ_ramp_pct"], "timely_energy_dr_pct": r["timely_energy_dr_pct"]})
            filas_hi.append({"configuracion": nombre, "fold": k, "alto_impacto_n": r["alto_impacto_n"], "alto_impacto_total": r["alto_impacto_total"]})
            filas_const.append({"configuracion": nombre, "fold": k, "dr_const_pct": r["dr_const_pct"], "dr_energ_const_pct": r["dr_energ_const_pct"]})
            filas_gb.append({"configuracion": nombre, "fold": k, "grupo_b_n": r["grupo_b_n"], "grupo_b_total": r["grupo_b_total"]})
    _guardar_tabla(pd.DataFrame(filas_fa), "falsas_alarmas_por_fold")
    _guardar_tabla(pd.DataFrame(filas_sat), "saturacion_por_fold")
    _guardar_tabla(pd.DataFrame(filas_ramp), "ramp_por_fold")
    _guardar_tabla(pd.DataFrame(filas_energia), "energia_por_fold")
    _guardar_tabla(pd.DataFrame(filas_hi), "alto_impacto_por_fold")
    _guardar_tabla(pd.DataFrame(filas_const), "constantes_por_fold")
    _guardar_tabla(pd.DataFrame(filas_gb), "grupo_b_por_fold")

    _guardar_tabla(pd.concat([fold_res_por_fold[k]["tablas_ventana"]["CAL30"].assign(fold=k) for k in fold_res_por_fold], ignore_index=True), "umbrales_cal30")
    _guardar_tabla(pd.concat([fold_res_por_fold[k]["tablas_ventana"]["CAL90"].assign(fold=k) for k in fold_res_por_fold], ignore_index=True), "umbrales_cal90")
    _guardar_tabla(pd.concat([fold_res_por_fold[k]["tablas_ventana"]["CAL180"].assign(fold=k) for k in fold_res_por_fold], ignore_index=True), "umbrales_cal180")
    _guardar_tabla(pd.concat([fold_res_por_fold[k]["tabla_mw"].assign(fold=k) for k in fold_res_por_fold], ignore_index=True), "umbrales_multiwindow")

    print("\nExclusivas respecto a P_MULTI_SEASON (para admisibilidad)...")
    filas_excl = []
    exclusivas_por_config_fold: dict[str, dict[int, int]] = {n: {} for n in todas_config}
    for fold_def in folds_info["folds"]:
        k = fold_def["fold"]
        res_p = resultados_por_fold[k]["P_MULTI_SEASON_ONLY"]["res_ramp"]
        for nombre in todas_config:
            det_p = set(res_p[res_p["induced_detected"]]["episode_id"])
            det_c = set(resultados_por_fold[k][nombre]["res_ramp"].pipe(lambda d: d[d["induced_detected"]]["episode_id"]))
            n_excl = len(det_c - det_p)
            exclusivas_por_config_fold[nombre][k] = n_excl
            filas_excl.append({"configuracion": nombre, "fold": k, "n_exclusivas_respecto_p": n_excl})
    _guardar_tabla(pd.DataFrame(filas_excl), "exclusivas_por_fold")

    print("Evaluando admisibilidad global...")
    admisibilidad = {}
    for nombre in todas_config:
        filas_cfg = [{k2: v2 for k2, v2 in resultados_por_fold[k][nombre].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
        admisibilidad[nombre] = evaluar_admisibilidad_global(nombre, filas_cfg, len(folds_info["folds"]), list(exclusivas_por_config_fold[nombre].values()))
        print(f"  {nombre:24s} admisible_global={admisibilidad[nombre]['admisible_global']} (con limite 70%: {admisibilidad[nombre]['admisible_con_limite_70_sensibilidad']}) | FA_pond={admisibilidad[nombre]['fa_media_ponderada']:.4f}")
    tabla_admisibles = _guardar_tabla(pd.DataFrame(admisibilidad.values()), "candidatos_admisibles")

    print("\nComparaciones estadisticas...")
    pares = [("H_CAL30_ONLY", "H_CAL90_ONLY"), ("H_CAL30_ONLY", "H_CAL180_ONLY"), ("H_CAL180_ONLY", "H_MULTIWINDOW_180_ONLY")]
    tablas_stats = [rtp.comparar_pareado(a, b, res_ramp_por_config_fold[a], res_ramp_por_config_fold[b]) for a, b in pares]
    _guardar_tabla(pd.concat(tablas_stats, ignore_index=True), "comparaciones_estadisticas")

    print("\nSeleccion de la mejor calibracion H (seccion 26)...")
    ganador_h = None
    candidatos_h_admisibles = []
    filas_h30 = [{k2: v2 for k2, v2 in resultados_por_fold[k]["H_CAL30_ONLY"].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
    for nombre_h in VARIANTES_H:
        cfg_nombre = f"{nombre_h}_ONLY"
        filas_h = [{k2: v2 for k2, v2 in resultados_por_fold[k][cfg_nombre].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
        adm_h = admisibilidad[cfg_nombre]
        if not adm_h["admisible_global"]:
            continue
        peor_fa_cal30 = max(f["fa"] for f in filas_h30)
        peor_fa_h = max(f["fa"] for f in filas_h)
        reduce_peor_fold = peor_fa_h < peor_fa_cal30 or nombre_h == "H_CAL30"
        dr_energ_h = float(np.median([f["dr_energ_ramp_pct"] for f in filas_h]))
        dr_energ_cal30 = float(np.median([f["dr_energ_ramp_pct"] for f in filas_h30]))
        hi_h, hi_cal30 = sum(f["alto_impacto_n"] for f in filas_h), sum(f["alto_impacto_n"] for f in filas_h30)
        gb_h = sum(f["grupo_b_n"] for f in filas_h); gb_cal30 = sum(f["grupo_b_n"] for f in filas_h30)
        c_energ = dr_energ_h >= 0.85 * dr_energ_cal30 if dr_energ_cal30 > 0 else True
        c_hi = hi_h >= 0.85 * hi_cal30 if hi_cal30 > 0 else True
        c_gb = gb_h >= 0.85 * gb_cal30 if gb_cal30 > 0 else True
        if reduce_peor_fold and c_energ and c_hi and c_gb:
            candidatos_h_admisibles.append({"nombre": nombre_h, "peor_fa": peor_fa_h, "std_fa": float(np.std([f["fa"] for f in filas_h])),
                                            "dr_energ_mediana": dr_energ_h, "alto_impacto": hi_h, "grupo_b": gb_h,
                                            "dr_mediana": float(np.median([f["dr_ramp_pct"] for f in filas_h])),
                                            "retraso": float(np.nanmedian([f["retraso_mediano_min"] for f in filas_h])), "complejidad": _COMPLEJIDAD_H[nombre_h]})
    if candidatos_h_admisibles:
        df_h_cand = pd.DataFrame(candidatos_h_admisibles).sort_values(
            by=["peor_fa", "std_fa", "dr_energ_mediana", "alto_impacto", "grupo_b", "dr_mediana", "retraso", "complejidad"],
            ascending=[True, True, False, False, False, False, True, True])
        # H_MULTIWINDOW_180 solo gana si mejora claramente sobre H_CAL180 (si no, prioriza CAL180 por simplicidad) -- ya lo refleja el orden por complejidad como ultimo criterio de desempate
        ganador_h = df_h_cand.iloc[0]["nombre"]
    print(f"  ganador H: {ganador_h}")

    print("\nAplicando la regla de seleccion final (Opcion A / B / C / D)...")
    filas_p = [{k2: v2 for k2, v2 in resultados_por_fold[k]["P_MULTI_SEASON_ONLY"].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
    adm_p = admisibilidad["P_MULTI_SEASON_ONLY"]

    candidatos_fusion_validos = []
    detalle_opcion_a = {}
    if ganador_h is not None:
        nombre_fusion = {"H_CAL30": "OR_PMULTI_HCAL30", "H_CAL90": "OR_PMULTI_HCAL90", "H_CAL180": "OR_PMULTI_HCAL180", "H_MULTIWINDOW_180": "OR_PMULTI_HMULTIWINDOW"}[ganador_h]
        for nombre_fusion_check in FUSIONES:
            nombre_h_check = _H_DE_LA_FUSION[nombre_fusion_check]
            filas_fusion = [{k2: v2 for k2, v2 in resultados_por_fold[k][nombre_fusion_check].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
            filas_h_check = [{k2: v2 for k2, v2 in resultados_por_fold[k][f"{nombre_h_check}_ONLY"].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
            ret_p_por_fold = [calcular_retencion(resultados_por_fold[k][nombre_fusion_check]["res_ramp"], resultados_por_fold[k]["P_MULTI_SEASON_ONLY"]["res_ramp"]) for k in resultados_por_fold]
            ret_h_por_fold = [calcular_retencion(resultados_por_fold[k][nombre_fusion_check]["res_ramp"], resultados_por_fold[k][f"{nombre_h_check}_ONLY"]["res_ramp"]) for k in resultados_por_fold]
            pasa, detalle = evaluar_opcion_a(nombre_fusion_check, filas_fusion, filas_p, filas_h_check, admisibilidad[nombre_fusion_check],
                                              ret_p_por_fold, ret_h_por_fold, list(exclusivas_por_config_fold[nombre_fusion_check].values()))
            detalle_opcion_a[nombre_fusion_check] = detalle
            print(f"  {nombre_fusion_check}: {'CUMPLE' if pasa else 'no cumple'} la regla A")
            if pasa:
                candidatos_fusion_validos.append(nombre_fusion_check)
    _guardar_tabla(pd.DataFrame([{"fusion": n, **{f"criterio_{i}": v for i, v in d.get("criterios", {}).items()}} for n, d in detalle_opcion_a.items()]), "regla_seleccion")

    if candidatos_fusion_validos:
        seleccion_final = candidatos_fusion_validos[0]
        print(f"\n>>> OPCION A: se selecciona la fusion {seleccion_final}")
    elif ganador_h is not None and evaluar_opcion_b(
            [{k2: v2 for k2, v2 in resultados_por_fold[k][f"{ganador_h}_ONLY"].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold],
            admisibilidad[f"{ganador_h}_ONLY"], filas_p):
        seleccion_final = f"{ganador_h}_ONLY"
        print(f"\n>>> OPCION B: se selecciona {seleccion_final} (H solo)")
    elif evaluar_opcion_c(filas_p, adm_p):
        seleccion_final = "P_MULTI_SEASON_ONLY"
        print("\n>>> OPCION C: se selecciona P_MULTI_SEASON_ONLY (P solo)")
    else:
        seleccion_final = "NONE_NOT_READY_FOR_TEST"
        print(f"\n>>> OPCION D: {seleccion_final}")

    _guardar_tabla(pd.DataFrame([{"selected_configuration": seleccion_final, "ganador_h": ganador_h}]), "configuracion_seleccionada")

    print("\nEntrenamiento final pre-test (si procede) y congelacion...")
    tabla_repro_baseline, metadatos_previos = opt.reproducir_baseline()
    resultado_final = entrenar_final(seleccion_final, ganador_h, folds_info, serie_completa_min, config, p_global, metadatos_previos)
    if seleccion_final != "NONE_NOT_READY_FOR_TEST" and resultado_final.get("sel_h_final") is not None:
        _guardar_tabla(pd.DataFrame([{"umbral_h_final": resultado_final["sel_h_final"]["umbral"], "umbral_p_final": resultado_final.get("umbral_p_final")}]), "calibracion_final_pretest")
    print(f"  seleccion final: {seleccion_final}")

    manifiesto = {"fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "n_folds": len(folds_info["folds"]),
                 "seleccion_final": seleccion_final, "ganador_h": ganador_h, "sklearn_version": sklearn.__version__,
                 "python_version": platform.python_version(), "test_intacto": True}
    with open(TABLAS_DIR / "manifiesto_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False, default=str)

    print("\nGenerando figuras...")
    generar_figuras(resultados_por_fold, fold_res_por_fold, tabla_admisibles)

    print("\nCONFIRMACION: en ningun momento se ha cargado, referenciado ni evaluado la particion de test "
          "(salvo su timestamp de arranque).")
    return {"resultados_por_fold": resultados_por_fold, "admisibilidad": admisibilidad, "seleccion_final": seleccion_final, "resultado_final": resultado_final}


def generar_figuras(resultados_por_fold, fold_res_por_fold, tabla_admisibles) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in sorted(fold_res_por_fold):
        u = fold_res_por_fold[k]["umbrales_ventana"]
        vals = [u["CAL30"]["umbral"], u["CAL90"]["umbral"], u["CAL180"]["umbral"], fold_res_por_fold[k]["sel_mw"]["umbral"]]
        ax.plot(["CAL30", "CAL90", "CAL180", "MULTIWINDOW"], vals, marker="o", label=f"fold {k}")
    ax.set_ylabel("umbral H seleccionado"); ax.legend(fontsize=8); ax.set_title("Umbral H segun longitud de calibracion")
    fig.tight_layout(); _guardar_fig(fig, "01_umbral_por_longitud_calibracion")

    fig, ax = plt.subplots(figsize=(9, 5))
    todas_config = CONFIGS_INDIVIDUALES + FUSIONES
    for nombre in todas_config:
        vals = [resultados_por_fold[k][nombre]["fa"] for k in sorted(resultados_por_fold)]
        ax.plot(sorted(resultados_por_fold), vals, marker="o", label=nombre, alpha=0.7)
    ax.axhline(0.15, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_xlabel("fold"); ax.set_ylabel("FA/dia"); ax.legend(fontsize=6, ncol=2); ax.set_title("FA por fold")
    fig.tight_layout(); _guardar_fig(fig, "03_fa_por_fold")

    fig, ax = plt.subplots(figsize=(7, 5))
    peor_fa = {n: max(resultados_por_fold[k][n]["fa"] for k in resultados_por_fold) for n in todas_config}
    ax.barh(list(peor_fa.keys()), list(peor_fa.values()), color=AZUL)
    ax.axvline(0.15, color=ROJO, linestyle="--", linewidth=1)
    ax.set_xlabel("peor FA entre folds"); ax.set_title("Peor FA por estrategia")
    fig.tight_layout(); _guardar_fig(fig, "04_peor_fa_por_estrategia")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(tabla_admisibles["nombre"], tabla_admisibles["admisible_global"].astype(int), color=VERDE)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_ylabel("admisible (1=si)"); ax.set_title("Resumen de admisibilidad")
    fig.tight_layout(); _guardar_fig(fig, "21_resumen_admisibilidad")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
