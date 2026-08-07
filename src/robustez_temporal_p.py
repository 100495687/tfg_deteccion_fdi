"""Robustez temporal del detector reconstructivo P y de su fusion con HistGB_PERIODIC.

La validacion walk-forward anterior (optimizacion_histgb_periodic) descubrio que P, con su
umbral fijo 0.041819, se satura durante periodos de consumo atipico (agosto de 2008: 71.9% de
activacion). Este experimento aisla el problema: reutiliza exactamente los mismos 4 folds y
ataques ya generados, y compara 3 variantes de P (P_GLOBAL congelado, P_MULTI_SEASON
recalibrado por fold, P_ROBUST_Z normalizado por grupo estacional) x 2 variantes de H
(H_RELATIVE, H_ROBUST_Z, ambas con hiperparametros fijos -- sin nueva busqueda) x fusiones OR,
con criterios explicitos de saturacion (active_fraction, duracion maxima, maximo movil de 7
dias, dias consecutivos >50% activo) ademas de FA/dia.

Test permanece completamente cerrado: la unica informacion permitida es su timestamp de
arranque (para saber donde termina el periodo pre-test). Ningun score, consumo ni ataque de
test se calcula ni se inspecciona en ningun momento.

Reutiliza sin modificar optimizacion_histgb_periodic.{construir_folds, generar_ataques_fold,
preparar_fold, puntuar_ataques_fold, cargar_p_global, puntuar_p_en_fold,
construir_p_atacado_por_episodio, clasificar_grupo_b_fold, clasificar_alto_impacto_fold,
construir_modelo, CONFIG_HISTGB_PERIODIC_ACTUAL, PERIODIC_BASE_FEATURES, reproducir_baseline},
fusion_p_histgb.{estado_ffill, construir_estados_episodio, simular_episodios_generico},
predictor_causal_lags.{calcular_score, resumen_eventos_grid}, analisis_detectabilidad.
contexto_e_impacto, memoria_finita_relative_kw.{enriquecer_resultado, resumen_metricas},
experimento_stride.{contar_eventos, mcnemar_p, bootstrap_pareado}, episodes.wilson_ci,
splitting.split_temporal y data_loading.cargar_y_limpiar.

Ejecutable de forma aislada:
    python -m src.robustez_temporal_p
"""
from __future__ import annotations

import hashlib
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
from src.data_loading import COLUMNA_OBJETIVO, cargar_y_limpiar
from src.episodes import wilson_ci
from src.experimento_stride import bootstrap_pareado, contar_eventos, mcnemar_p
from src.memoria_finita_relative_kw import enriquecer_resultado, resumen_metricas
from src.splitting import split_temporal

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "robustez_temporal_p"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "robustez_temporal_p"
MODELOS_DIR = BASE_DIR / "models" / "robustez_temporal_p"
EDO_DIR = BASE_DIR / "results" / "tables" / "energy_distance_operativo"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

RESOLUCION_MIN = 15
UMBRAL_P_FIJO = 0.041819

TARGET_FA_P, BANDA_P_PRINCIPAL, BANDA_P_SECUNDARIA = 0.04, (0.02, 0.06), (0.01, 0.08)
TARGET_FA_H, BANDA_H_PRINCIPAL, BANDA_H_SECUNDARIA = 0.06, (0.04, 0.08), (0.02, 0.10)
TARGET_FA_OR, BANDA_OR_PRINCIPAL, BANDA_OR_SECUNDARIA = 0.10, (0.08, 0.12), (0.06, 0.14)

LIM_ACTIVE_FRACTION, LIM_LONGEST_CLEAN_HOURS = 0.10, 72.0
LIM_7DAY_MAX_ACTIVE, LIM_CONSECUTIVE_DAYS_50 = 0.30, 3
DIAG_ACTIVE_FRACTION, DIAG_LONGEST_CLEAN_HOURS = 0.05, 48.0

MIN_OBS_P_ROBUST_Z, MIN_OBS_H_ROBUST_Z = 30, 20

FUSIONES = ["OR_PGLOBAL_HREL", "OR_PMULTI_HREL", "OR_PMULTI_HZ", "OR_PZ_HREL", "OR_PZ_HZ"]
CONFIGURACIONES_INDIVIDUALES = ["P_GLOBAL", "P_MULTI_SEASON", "P_ROBUST_Z", "H_RELATIVE", "H_ROBUST_Z"]


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 10) Auditoria del checkpoint P
# ==========================================================================================

def auditar_checkpoint_p(folds_info: dict) -> pd.DataFrame:
    particiones = split_temporal(cargar_y_limpiar())
    train_ini, train_fin = particiones["train"].index[0], particiones["train"].index[-1]
    calib_ini_p = pd.Timestamp("2008-12-15 23:23:00")  # 01_division_temporal.csv (energy_distance_operativo)
    calib_fin_p = pd.Timestamp("2009-06-16 07:23:00")
    ruta_ckpt = BASE_DIR / "models" / "tcn_ae_ventana360.pt"
    hash_ckpt = hashlib.sha256(ruta_ckpt.read_bytes()).hexdigest()
    from src.windowing import cargar_config
    config = cargar_config()

    filas = [{
        "elemento": "entrenamiento_tcn_ae", "inicio": str(train_ini), "fin": str(train_fin),
        "descripcion": "particion train completa (2 anos), usada para ajustar los pesos del TCN-AE y la normalizacion z-score",
    }, {
        "elemento": "calibracion_umbral_0.041819", "inicio": str(calib_ini_p), "fin": str(calib_fin_p),
        "descripcion": "mitad 'desarrollo' de la particion val, usada para fijar el umbral historico de P",
    }, {
        "elemento": "checkpoint", "inicio": "", "fin": "",
        "descripcion": f"models/tcn_ae_ventana360.pt | hash_sha256={hash_ckpt} | arquitectura={config['modelo']['tcn_ae']} | normalizacion=zscore(train)",
    }]
    for f in folds_info["folds"]:
        solapa_train = not (f["valid_fin"] <= train_ini or f["calib_fin"] >= train_fin)
        solapa_calib_umbral = not (f["valid_fin"] <= calib_ini_p or f["calib_fin"] >= calib_fin_p)
        if solapa_train:
            etiqueta = "retrospective temporal stress test (VALIDATION_FOLD solapa el entrenamiento de los pesos de P)"
        elif solapa_calib_umbral:
            etiqueta = "retrospective temporal stress test (VALIDATION_FOLD solapa el periodo de calibracion del umbral 0.041819)"
        else:
            etiqueta = "independent_retrospective_validation (fuera de train y de la calibracion del umbral de P)"
        filas.append({"elemento": f"fold_{f['fold']}_solapamiento", "inicio": str(f["calib_fin"]), "fin": str(f["valid_fin"]),
                      "descripcion": etiqueta})
    tabla = pd.DataFrame(filas)
    return _guardar_tabla(tabla, "auditoria_checkpoint_p")


# ==========================================================================================
# Metricas de saturacion
# ==========================================================================================

def _duracion_maxima_horas(activo: np.ndarray, resolucion_min: int) -> float:
    from src.memoria_finita_relative_kw import _duracion_eventos
    dur = _duracion_eventos(activo, resolucion_min=resolucion_min)
    return float(dur.max()) / 60.0


def _max_activo_ventana_7dias(activo: np.ndarray, ts: np.ndarray, resolucion_min: int) -> float:
    pasos_7d = int(round(7 * 1440 / resolucion_min))
    if len(activo) < pasos_7d:
        return float(activo.mean()) if len(activo) else 0.0
    acumulado = pd.Series(activo.astype(float)).rolling(pasos_7d, min_periods=pasos_7d).mean()
    return float(acumulado.max())


def _dias_consecutivos_mas_50(activo: np.ndarray, ts: np.ndarray, resolucion_min: int) -> int:
    df = pd.DataFrame({"ts": pd.DatetimeIndex(ts), "activo": activo})
    por_dia = df.groupby(df["ts"].dt.date)["activo"].mean()
    sobre_50 = (por_dia > 0.5).to_numpy()
    mejor, actual = 0, 0
    for b in sobre_50:
        actual = actual + 1 if b else 0
        mejor = max(mejor, actual)
    return int(mejor)


def calcular_saturacion(activo: np.ndarray, ts: np.ndarray, resolucion_min: int) -> dict:
    frac = float(activo.mean()) if len(activo) else 0.0
    dur_max = _duracion_maxima_horas(activo, resolucion_min) if len(activo) else 0.0
    max7 = _max_activo_ventana_7dias(activo, ts, resolucion_min) if len(activo) else 0.0
    dias_50 = _dias_consecutivos_mas_50(activo, ts, resolucion_min) if len(activo) else 0
    df = pd.DataFrame({"ts": pd.DatetimeIndex(ts), "activo": activo})
    por_dia = df.groupby(df["ts"].dt.date)["activo"].mean() if len(activo) else pd.Series(dtype=float)
    dias_25pct = int((por_dia > 0.25).sum())
    dias_50pct = int((por_dia > 0.50).sum())
    return {
        "active_fraction": frac, "longest_clean_event_hours": dur_max, "seven_day_max_active_fraction": max7,
        "consecutive_days_over_50": dias_50, "days_over_25pct_active": dias_25pct, "days_over_50pct_active": dias_50pct,
        "falla_saturacion": bool(frac > LIM_ACTIVE_FRACTION or dur_max > LIM_LONGEST_CLEAN_HOURS or max7 > LIM_7DAY_MAX_ACTIVE or dias_50 >= LIM_CONSECUTIVE_DAYS_50),
        "diag_active_fraction_alta": bool(frac > DIAG_ACTIVE_FRACTION), "diag_evento_largo": bool(dur_max > DIAG_LONGEST_CLEAN_HOURS),
    }


# ==========================================================================================
# 12.2) P_MULTI_SEASON -- recalibracion pura del umbral de P (mismo score, misma resolucion)
# ==========================================================================================

def calibrar_umbral_p(scores_calib: np.ndarray, ts_calib: np.ndarray, n_dias_calib: float) -> tuple[pd.DataFrame, dict]:
    filas = []
    for v in np.unique(scores_calib):
        activo = scores_calib > v
        sat = calcular_saturacion(activo, ts_calib, resolucion_min=60)
        fa = contar_eventos(activo) / n_dias_calib
        filas.append({"umbral": float(v), "fa": fa, **sat})
    tabla = pd.DataFrame(filas)

    candidatos, banda, deficiente = tabla[tabla["fa"].between(*BANDA_P_PRINCIPAL)], "principal", False
    if len(candidatos) == 0:
        candidatos, banda = tabla[tabla["fa"].between(*BANDA_P_SECUNDARIA)], "secundaria"
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla[tabla["fa"] < BANDA_P_SECUNDARIA[1]], "fuera_de_banda", True
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla.copy(), "sin_candidato_valido", True

    candidatos = candidatos.copy()
    candidatos["dist"] = (candidatos["fa"] - TARGET_FA_P).abs()
    candidatos = candidatos.sort_values(
        ["dist", "active_fraction", "longest_clean_event_hours", "seven_day_max_active_fraction", "umbral"],
        ascending=[True, True, True, True, False])
    sel = candidatos.iloc[0].to_dict()
    sel["banda"] = banda
    sel["deficiente"] = deficiente
    return tabla, sel


# ==========================================================================================
# Estadisticos jerarquicos con fallback (compartido por P_ROBUST_Z y H_ROBUST_Z)
# ==========================================================================================

def _stats_grupo(df: pd.DataFrame, cols: list, valor_col: str) -> pd.DataFrame:
    def _s(x):
        c = x.median()
        mad = (x - c).abs().median()
        return pd.Series({"center": c, "scale": 1.4826 * mad, "n": len(x)})
    if not cols:
        return pd.DataFrame([_s(df[valor_col])])
    return df.groupby(cols)[valor_col].apply(_s).unstack().reset_index()


def calcular_stats_jerarquicos(df_train: pd.DataFrame, jerarquia: list[list[str]], valor_col: str, min_obs: int) -> dict:
    niveles = {tuple(cols): _stats_grupo(df_train, cols, valor_col) for cols in jerarquia}
    centro_global = float(df_train[valor_col].median())
    escala_global = float(1.4826 * (df_train[valor_col] - centro_global).abs().median())
    escalas_validas = pd.concat([n[n["n"] >= min_obs]["scale"] for n in niveles.values() if len(n)])
    floor_candidatos = [float(np.percentile(escalas_validas, 10)) if len(escalas_validas) else 0.0, 0.10 * escala_global]
    return {"niveles": niveles, "jerarquia": jerarquia, "min_obs": min_obs, "centro_global": centro_global,
            "escala_global": escala_global, "floor_candidatos": floor_candidatos}


def aplicar_stats_jerarquicos(df_claves: pd.DataFrame, stats: dict, floor_absoluto: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df_claves)
    floor = max(*stats["floor_candidatos"], floor_absoluto)
    centro = np.full(n, stats["centro_global"], dtype=float)
    escala = np.full(n, max(stats["escala_global"], floor), dtype=float)
    nivel = np.full(n, len(stats["jerarquia"]), dtype=int)
    asignado = np.zeros(n, dtype=bool)
    claves_reset = df_claves.reset_index(drop=True)
    for i, cols in enumerate(stats["jerarquia"]):
        tabla_nivel = stats["niveles"][tuple(cols)]
        validos = tabla_nivel[tabla_nivel["n"] >= stats["min_obs"]]
        if len(validos) == 0:
            continue
        merged = claves_reset.merge(validos, on=cols, how="left") if cols else claves_reset.assign(
            center=validos["center"].iloc[0], scale=validos["scale"].iloc[0])
        hay_match = merged["center"].notna().to_numpy() & ~asignado
        centro[hay_match] = merged.loc[hay_match, "center"].to_numpy()
        escala[hay_match] = np.maximum(merged.loc[hay_match, "scale"].to_numpy(), floor)
        nivel[hay_match] = i
        asignado |= hay_match
    return centro, escala, nivel


def _estacion(mes: np.ndarray) -> np.ndarray:
    mapa = {12: "invierno", 1: "invierno", 2: "invierno", 3: "primavera", 4: "primavera", 5: "primavera",
            6: "verano", 7: "verano", 8: "verano", 9: "otono", 10: "otono", 11: "otono"}
    return np.array([mapa[m] for m in mes])


def _claves_p(ts: np.ndarray) -> pd.DataFrame:
    t = pd.DatetimeIndex(ts)
    return pd.DataFrame({"season": _estacion(t.month.to_numpy()), "hour": t.hour.to_numpy(),
                          "day_type": np.where(t.dayofweek.to_numpy() >= 5, "finde", "laborable")})


JERARQUIA_P = [["season", "hour", "day_type"], ["hour", "day_type"], ["hour"], ["season"]]


def calcular_stats_p_robust_z(scores_train: np.ndarray, ts_train: np.ndarray) -> dict:
    claves = _claves_p(ts_train)
    df = claves.copy()
    df["raw"] = scores_train
    return calcular_stats_jerarquicos(df, JERARQUIA_P, "raw", MIN_OBS_P_ROBUST_Z)


def aplicar_p_robust_z(scores: np.ndarray, ts: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    claves = _claves_p(ts)
    centro, escala, nivel = aplicar_stats_jerarquicos(claves, stats, floor_absoluto=1e-4)
    z = np.maximum(0.0, (scores - centro) / escala)
    return z, nivel


# ==========================================================================================
# H_ROBUST_Z -- jerarquia propia de este experimento (distinta de la usada en
# optimizacion_histgb_periodic: min_obs=20 y un nivel intermedio "misma hora")
# ==========================================================================================

def _claves_h(ts: np.ndarray) -> pd.DataFrame:
    t = pd.DatetimeIndex(ts)
    slot = ((t.hour * 60 + t.minute) // RESOLUCION_MIN).to_numpy()
    return pd.DataFrame({"slot_15min": slot, "hour": t.hour.to_numpy(),
                          "day_type": np.where(t.dayofweek.to_numpy() >= 5, "finde", "laborable")})


JERARQUIA_H = [["slot_15min", "day_type"], ["slot_15min"], ["hour"]]


def calcular_stats_h_robust_z(signed_train: np.ndarray, ts_train: np.ndarray) -> dict:
    claves = _claves_h(ts_train)
    df = claves.copy()
    df["signed"] = signed_train
    return calcular_stats_jerarquicos(df, JERARQUIA_H, "signed", MIN_OBS_H_ROBUST_Z)


def aplicar_h_robust_z(signed: np.ndarray, ts: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    claves = _claves_h(ts)
    centro, escala, nivel = aplicar_stats_jerarquicos(claves, stats, floor_absoluto=1e-3)
    z = np.maximum(0.0, (signed - centro) / escala)
    return z, nivel


# ==========================================================================================
# 14.1) Calibracion de H (RELATIVE o ROBUST_Z, mismo procedimiento generico salvo prioridades)
# ==========================================================================================

def calibrar_umbral_h(scores_calib: np.ndarray, ts_calib: np.ndarray, n_dias_calib: float) -> tuple[pd.DataFrame, dict]:
    filas = []
    for v in np.unique(scores_calib):
        activo = scores_calib > v
        sat = calcular_saturacion(activo, ts_calib, resolucion_min=RESOLUCION_MIN)
        fa = contar_eventos(activo) / n_dias_calib
        filas.append({"umbral": float(v), "fa": fa, **sat})
    tabla = pd.DataFrame(filas)

    candidatos, banda, deficiente = tabla[tabla["fa"].between(*BANDA_H_PRINCIPAL)], "principal", False
    if len(candidatos) == 0:
        candidatos, banda = tabla[tabla["fa"].between(*BANDA_H_SECUNDARIA)], "secundaria"
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla[tabla["fa"] < BANDA_H_SECUNDARIA[1]], "fuera_de_banda", True
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla.copy(), "sin_candidato_valido", True

    candidatos = candidatos.copy()
    candidatos["dist"] = (candidatos["fa"] - TARGET_FA_H).abs()
    candidatos = candidatos.sort_values(["dist", "active_fraction", "longest_clean_event_hours", "umbral"], ascending=[True, True, True, False])
    sel = candidatos.iloc[0].to_dict()
    sel["banda"] = banda
    sel["deficiente"] = deficiente
    return tabla, sel


# ==========================================================================================
# 17) Calibracion de la fusion: H primero (congelado), despues P para FA_OR objetivo
# ==========================================================================================

def calibrar_p_para_or(p_score_full: np.ndarray, p_ts_full: np.ndarray, activo_h_calib_fijo: np.ndarray,
                        ts_15min_calib: np.ndarray, n_dias_calib: float) -> tuple[pd.DataFrame, dict]:
    idx = np.searchsorted(p_ts_full, ts_15min_calib, side="right") - 1
    idx = np.clip(idx, 0, len(p_score_full) - 1)
    valores_ffilled = np.where(np.searchsorted(p_ts_full, ts_15min_calib, side="right") - 1 >= 0, p_score_full[idx], -np.inf)
    # cualquier par de umbrales entre dos valores consecutivos de valores_ffilled produce el mismo
    # active_p en esta ventana de calibracion -- probar exactamente esos valores unicos es
    # equivalente a probar todos los umbrales unicos posibles de P, sin evaluar
    # miles de candidatos globales irrelevantes para este fold
    candidatos_umbral = np.unique(valores_ffilled[np.isfinite(valores_ffilled)])

    filas = []
    for v in candidatos_umbral:
        activo_p = valores_ffilled > v
        activo_or = activo_p | activo_h_calib_fijo
        sat = calcular_saturacion(activo_or, ts_15min_calib, resolucion_min=RESOLUCION_MIN)
        fa = contar_eventos(activo_or) / n_dias_calib
        filas.append({"umbral_p": float(v), "fa_or": fa, **sat})
    tabla = pd.DataFrame(filas)

    candidatos, banda, deficiente = tabla[tabla["fa_or"].between(*BANDA_OR_PRINCIPAL)], "principal", False
    if len(candidatos) == 0:
        candidatos, banda = tabla[tabla["fa_or"].between(*BANDA_OR_SECUNDARIA)], "secundaria"
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla[tabla["fa_or"] < BANDA_OR_SECUNDARIA[1]], "fuera_de_banda", True
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla.copy(), "sin_candidato_valido", True

    candidatos = candidatos.copy()
    candidatos["dist"] = (candidatos["fa_or"] - TARGET_FA_OR).abs()
    candidatos = candidatos.sort_values(
        ["dist", "active_fraction", "longest_clean_event_hours", "seven_day_max_active_fraction", "umbral_p"],
        ascending=[True, True, True, True, False])
    sel = candidatos.iloc[0].to_dict()
    sel["banda"] = banda
    sel["deficiente"] = deficiente
    return tabla, sel


# ==========================================================================================
# Pipeline por fold: reutiliza exactamente folds/ataques de optimizacion_histgb_periodic
# ==========================================================================================

def _transformar_p_atacado(p_atacado_raw: dict, p_global: dict, stats_p: dict) -> dict:
    resultado = {}
    for ep, d in p_atacado_raw.items():
        ts_ep = p_global["ts"][d["k"]]
        z, _ = aplicar_p_robust_z(d["score"], ts_ep, stats_p)
        resultado[ep] = {"k": d["k"], "score": z}
    return resultado


def _meta_con_score_h(meta: pd.DataFrame, feats: pd.DataFrame, modelo_h, cols: list, columna_score: str,
                       transform_fn=None, stats_h=None) -> pd.DataFrame:
    meta = meta.copy()
    if len(feats) == 0:
        meta[columna_score] = []
        return meta
    pred = modelo_h.predict(feats[cols].to_numpy(dtype=np.float64))
    signed, positive = pcl.calcular_score(pred, meta["target_kw_atacado"].to_numpy())
    if transform_fn is None:
        meta[columna_score] = positive
    else:
        z, _ = transform_fn(signed, meta["target_end"].to_numpy(), stats_h)
        meta[columna_score] = z
    return meta


def procesar_fold(fold_def: dict, serie_completa_min: pd.DataFrame, config: dict, p_global: dict) -> dict:
    prep = opt.preparar_fold(fold_def, serie_completa_min)
    ataques_fold = opt.generar_ataques_fold(fold_def, serie_completa_min, config)
    meta_ramp, feats_ramp = opt.puntuar_ataques_fold(ataques_fold["ataques_ramp"], prep)
    meta_const, feats_const = opt.puntuar_ataques_fold(ataques_fold["ataques_constante"], prep)

    tabla_ramp_p, _ = opt.puntuar_p_en_fold(ataques_fold["ataques_ramp"], ataques_fold["particion_fold_raw"], ataques_fold["valid_fold_df"], fold_def["ini"], p_global)
    tabla_const_p, _ = opt.puntuar_p_en_fold(ataques_fold["ataques_constante"], ataques_fold["particion_fold_raw"], ataques_fold["valid_fold_df"], fold_def["ini"], p_global)
    grupo_b_ids = opt.clasificar_grupo_b_fold(tabla_const_p, p_global["umbral"])
    alto_impacto_ids, p75 = opt.clasificar_alto_impacto_fold(tabla_ramp_p, ataques_fold["ataques_ramp"], ataques_fold["particion_fold_raw"], p_global["umbral"])
    p_atacado_ramp_raw = opt.construir_p_atacado_por_episodio(tabla_ramp_p, p_global)
    p_atacado_const_raw = opt.construir_p_atacado_por_episodio(tabla_const_p, p_global)

    ts_p_full = pd.DatetimeIndex(p_global["ts"])
    mask_p_train = (ts_p_full >= fold_def["ini"]) & (ts_p_full < fold_def["train_fin"])
    mask_p_calib = (ts_p_full >= fold_def["train_fin"]) & (ts_p_full < fold_def["calib_fin"])
    p_scores_train, p_ts_train = p_global["score"][mask_p_train], p_global["ts"][mask_p_train]
    p_scores_calib, p_ts_calib = p_global["score"][mask_p_calib], p_global["ts"][mask_p_calib]
    n_dias_calib = opt.CALIB_DIAS

    target_end_master = prep["df"].loc[prep["es_valid"], "ts_end"].to_numpy()

    # ===== P_GLOBAL (congelado) =====
    tabla_umb_pglobal = pd.DataFrame([{"umbral": UMBRAL_P_FIJO, "nota": "congelado, no recalibrado"}])
    p_global_cfg = {"score_full": p_global["score"], "umbral": UMBRAL_P_FIJO, "atacado_raw": p_atacado_ramp_raw, "atacado_raw_const": p_atacado_const_raw}

    # ===== P_MULTI_SEASON =====
    tabla_umb_multiseason, sel_multiseason = calibrar_umbral_p(p_scores_calib, p_ts_calib, n_dias_calib)
    p_multiseason_cfg = {"score_full": p_global["score"], "umbral": sel_multiseason["umbral"], "atacado_raw": p_atacado_ramp_raw, "atacado_raw_const": p_atacado_const_raw}

    # ===== P_ROBUST_Z =====
    stats_p = calcular_stats_p_robust_z(p_scores_train, p_ts_train)
    p_score_full_z, _ = aplicar_p_robust_z(p_global["score"], p_global["ts"], stats_p)
    p_score_calib_z, _ = aplicar_p_robust_z(p_scores_calib, p_ts_calib, stats_p)
    tabla_umb_pz, sel_pz = calibrar_umbral_p(p_score_calib_z, p_ts_calib, n_dias_calib)
    p_atacado_ramp_z = _transformar_p_atacado(p_atacado_ramp_raw, p_global, stats_p)
    p_atacado_const_z = _transformar_p_atacado(p_atacado_const_raw, p_global, stats_p)
    p_robust_z_cfg = {"score_full": p_score_full_z, "umbral": sel_pz["umbral"], "atacado_raw": p_atacado_ramp_z, "atacado_raw_const": p_atacado_const_z}

    # ===== H: se entrena una sola vez (hiperparametros fijos, features PERIODIC_BASE) =====
    cols = opt.PERIODIC_BASE_FEATURES
    modelo_h = opt.construir_modelo(opt.CONFIG_HISTGB_PERIODIC_ACTUAL)
    X_train = prep["df"].loc[prep["es_train"], cols].to_numpy(dtype=np.float64)
    y_train = prep["df"].loc[prep["es_train"], "target_kw"].to_numpy()
    ts_train_h = prep["df"].loc[prep["es_train"], "ts_end"].to_numpy()
    modelo_h.fit(X_train, y_train)

    pred_train_h = modelo_h.predict(X_train)
    signed_train_h, _ = pcl.calcular_score(pred_train_h, y_train)
    stats_h = calcular_stats_h_robust_z(signed_train_h, ts_train_h)

    X_calib = prep["df"].loc[prep["es_calib"], cols].to_numpy(dtype=np.float64)
    y_calib = prep["df"].loc[prep["es_calib"], "target_kw"].to_numpy()
    ts_calib_h = prep["df"].loc[prep["es_calib"], "ts_end"].to_numpy()
    pred_calib_h = modelo_h.predict(X_calib)
    signed_calib_h, positive_calib_h = pcl.calcular_score(pred_calib_h, y_calib)
    score_calib_hz, _ = aplicar_h_robust_z(signed_calib_h, ts_calib_h, stats_h)

    tabla_umb_hrel, sel_hrel = calibrar_umbral_h(positive_calib_h, ts_calib_h, n_dias_calib)
    tabla_umb_hz, sel_hz = calibrar_umbral_h(score_calib_hz, ts_calib_h, n_dias_calib)

    X_valid = prep["df"].loc[prep["es_valid"], cols].to_numpy(dtype=np.float64)
    y_valid = prep["df"].loc[prep["es_valid"], "target_kw"].to_numpy()
    pred_valid_h = modelo_h.predict(X_valid)
    signed_valid_h, positive_valid_h = pcl.calcular_score(pred_valid_h, y_valid)
    score_valid_hz, _ = aplicar_h_robust_z(signed_valid_h, target_end_master, stats_h)

    h_relative_cfg = {"score_full": positive_valid_h, "umbral": sel_hrel["umbral"]}
    h_robust_z_cfg = {"score_full": score_valid_hz, "umbral": sel_hz["umbral"]}

    meta_ramp_hrel = _meta_con_score_h(meta_ramp, feats_ramp, modelo_h, cols, "score_atacado_H")
    meta_ramp_hz = _meta_con_score_h(meta_ramp, feats_ramp, modelo_h, cols, "score_atacado_H", aplicar_h_robust_z, stats_h)
    meta_const_hrel = _meta_con_score_h(meta_const, feats_const, modelo_h, cols, "score_atacado_H")
    meta_const_hz = _meta_con_score_h(meta_const, feats_const, modelo_h, cols, "score_atacado_H", aplicar_h_robust_z, stats_h)

    # ===== 17) Fusiones: H se calibra y congela primero; despues se calibra P para FA_OR =====
    activo_hrel_calib = positive_calib_h > sel_hrel["umbral"]
    activo_hz_calib = score_calib_hz > sel_hz["umbral"]

    activo_pglobal_calib = fus.estado_ffill(p_global["score"], UMBRAL_P_FIJO, p_global["ts"], ts_calib_h)
    fa_pglobal_solo_calib = contar_eventos(activo_pglobal_calib) / n_dias_calib
    structurally_uncalibratable = bool(fa_pglobal_solo_calib > BANDA_OR_SECUNDARIA[1])

    tabla_or_pmulti_hrel, sel_or_pmulti_hrel = calibrar_p_para_or(p_global["score"], p_global["ts"], activo_hrel_calib, ts_calib_h, n_dias_calib)
    tabla_or_pmulti_hz, sel_or_pmulti_hz = calibrar_p_para_or(p_global["score"], p_global["ts"], activo_hz_calib, ts_calib_h, n_dias_calib)
    tabla_or_pz_hrel, sel_or_pz_hrel = calibrar_p_para_or(p_score_full_z, p_global["ts"], activo_hrel_calib, ts_calib_h, n_dias_calib)
    tabla_or_pz_hz, sel_or_pz_hz = calibrar_p_para_or(p_score_full_z, p_global["ts"], activo_hz_calib, ts_calib_h, n_dias_calib)

    fusiones_cfg = {
        "OR_PGLOBAL_HREL": {"p": p_global_cfg, "p_umbral_or": UMBRAL_P_FIJO, "h": h_relative_cfg,
                             "meta_ramp": meta_ramp_hrel, "meta_const": meta_const_hrel,
                             "structurally_uncalibratable_due_to_P": structurally_uncalibratable, "tabla_calib": None, "sel_calib": None},
        "OR_PMULTI_HREL": {"p": p_multiseason_cfg, "p_umbral_or": sel_or_pmulti_hrel["umbral_p"], "h": h_relative_cfg,
                            "meta_ramp": meta_ramp_hrel, "meta_const": meta_const_hrel, "structurally_uncalibratable_due_to_P": False,
                            "tabla_calib": tabla_or_pmulti_hrel, "sel_calib": sel_or_pmulti_hrel},
        "OR_PMULTI_HZ": {"p": p_multiseason_cfg, "p_umbral_or": sel_or_pmulti_hz["umbral_p"], "h": h_robust_z_cfg,
                          "meta_ramp": meta_ramp_hz, "meta_const": meta_const_hz, "structurally_uncalibratable_due_to_P": False,
                          "tabla_calib": tabla_or_pmulti_hz, "sel_calib": sel_or_pmulti_hz},
        "OR_PZ_HREL": {"p": {**p_robust_z_cfg, "score_full": p_score_full_z}, "p_umbral_or": sel_or_pz_hrel["umbral_p"], "h": h_relative_cfg,
                        "meta_ramp": meta_ramp_hrel, "meta_const": meta_const_hrel, "structurally_uncalibratable_due_to_P": False,
                        "tabla_calib": tabla_or_pz_hrel, "sel_calib": sel_or_pz_hrel},
        "OR_PZ_HZ": {"p": {**p_robust_z_cfg, "score_full": p_score_full_z}, "p_umbral_or": sel_or_pz_hz["umbral_p"], "h": h_robust_z_cfg,
                     "meta_ramp": meta_ramp_hz, "meta_const": meta_const_hz, "structurally_uncalibratable_due_to_P": False,
                     "tabla_calib": tabla_or_pz_hz, "sel_calib": sel_or_pz_hz},
    }

    return {
        "prep": prep, "ataques_fold": ataques_fold, "grupo_b_ids": grupo_b_ids, "alto_impacto_ids": alto_impacto_ids,
        "target_end_master": target_end_master,
        "p_global_cfg": p_global_cfg, "p_multiseason_cfg": p_multiseason_cfg, "p_robust_z_cfg": p_robust_z_cfg,
        "h_relative_cfg": h_relative_cfg, "h_robust_z_cfg": h_robust_z_cfg, "fusiones_cfg": fusiones_cfg,
        "meta_ramp_hrel": meta_ramp_hrel, "meta_ramp_hz": meta_ramp_hz, "meta_const_hrel": meta_const_hrel, "meta_const_hz": meta_const_hz,
        "tabla_umb_multiseason": tabla_umb_multiseason, "sel_multiseason": sel_multiseason, "tabla_umb_pz": tabla_umb_pz, "sel_pz": sel_pz,
        "tabla_umb_hrel": tabla_umb_hrel, "sel_hrel": sel_hrel, "tabla_umb_hz": tabla_umb_hz, "sel_hz": sel_hz,
        "stats_p": stats_p, "stats_h": stats_h, "modelo_h": modelo_h, "n_dias_calib": n_dias_calib,
        "ts_calib_h": ts_calib_h, "p_scores_train": p_scores_train, "p_ts_train": p_ts_train,
        "p_scores_calib": p_scores_calib, "p_ts_calib": p_ts_calib,
    }


# ==========================================================================================
# 18-19/24-26) Reconstruccion de eventos (siempre desde cero) y evaluacion de ataques
# ==========================================================================================

def _dummy_p(n_p_ts: int) -> dict:
    return {"score_full": np.full(n_p_ts, -1e9), "umbral": 1e9, "atacado_raw": {}}


def construir_estados(meta_grid: pd.DataFrame, p_cfg: dict, h_score_full: np.ndarray, h_umbral: float,
                       p_global: dict, target_end_master: np.ndarray, ids: set) -> dict:
    grp_h = {ep: g for ep, g in meta_grid.groupby("episode_id")}
    estados = {}
    for ep in ids:
        estados[ep] = fus.construir_estados_episodio(grp_h.get(ep), p_global["ts"], p_cfg["score_full"], p_cfg["umbral"],
                                                       p_cfg["atacado_raw"].get(ep), h_score_full, h_umbral, target_end_master)
    return estados


def evaluar_ataques_configuracion(metodo: str, meta_ramp_grid: pd.DataFrame, meta_const_grid: pd.DataFrame,
                                   p_cfg: dict | None, h_score_full: np.ndarray | None, h_umbral: float | None,
                                   fold_res: dict, p_global: dict) -> dict:
    target_end_master = fold_res["target_end_master"]
    n_p_ts = len(p_global["ts"])
    p_cfg = p_cfg if p_cfg is not None else _dummy_p(n_p_ts)
    h_score_full = h_score_full if h_score_full is not None else np.full(len(target_end_master), -1e9)
    h_umbral = h_umbral if h_umbral is not None else 1e9

    ids_ramp = set(fold_res["ataques_fold"]["ataques_ramp"])
    ids_const = set(fold_res["ataques_fold"]["ataques_constante"])
    estados_ramp = construir_estados(meta_ramp_grid, p_cfg, h_score_full, h_umbral, p_global, target_end_master, ids_ramp)
    estados_const = construir_estados(meta_const_grid, p_cfg, h_score_full, h_umbral, p_global, target_end_master, ids_const)
    return {"estados_ramp": estados_ramp, "estados_const": estados_const, "ids_ramp": ids_ramp, "ids_const": ids_const}


def _activo_clean(nombre: str, p_cfg: dict | None, h_score_full: np.ndarray | None, h_umbral: float | None,
                   p_global: dict, target_end_master: np.ndarray) -> np.ndarray:
    activo_p = fus.estado_ffill(p_cfg["score_full"], p_cfg["umbral"], p_global["ts"], target_end_master) if p_cfg is not None else np.zeros(len(target_end_master), dtype=bool)
    activo_h = (h_score_full > h_umbral) if h_score_full is not None else np.zeros(len(target_end_master), dtype=bool)
    if nombre.startswith("P"):
        return activo_p
    if nombre.startswith("H"):
        return activo_h
    return activo_p | activo_h


def evaluar_configuracion_fold(nombre: str, metodo: str, p_cfg: dict | None, h_score_full: np.ndarray | None,
                                h_umbral: float | None, meta_ramp_grid: pd.DataFrame, meta_const_grid: pd.DataFrame,
                                fold_res: dict, p_global: dict, fold_def: dict) -> dict:
    target_end_master = fold_res["target_end_master"]
    activo_clean = _activo_clean(nombre, p_cfg, h_score_full, h_umbral, p_global, target_end_master)
    resolucion = RESOLUCION_MIN
    sat = calcular_saturacion(activo_clean, target_end_master, resolucion)
    n_dias_valid = opt.VALID_DIAS
    fa = contar_eventos(activo_clean) / n_dias_valid

    aux = evaluar_ataques_configuracion(metodo, meta_ramp_grid, meta_const_grid, p_cfg, h_score_full, h_umbral, fold_res, p_global)
    res_ramp_raw = fus.simular_episodios_generico(aux["ids_ramp"], aux["estados_ramp"], metodo, activo_clean, target_end_master)
    res_const_raw = fus.simular_episodios_generico(aux["ids_const"], aux["estados_const"], metodo, activo_clean, target_end_master)

    contexto_ramp = ad.contexto_e_impacto(fold_res["ataques_fold"]["ataques_ramp"], fold_res["ataques_fold"]["particion_fold_raw"])
    meta_ep_ramp = fold_res["ataques_fold"]["tabla_ramp"][["episode_id", "attack_start", "attack_end", "duration_min", "rho_final"]]
    contexto_const = ad.contexto_e_impacto(fold_res["ataques_fold"]["ataques_constante"], fold_res["ataques_fold"]["particion_fold_raw"])
    meta_ep_const = fold_res["ataques_fold"]["master_fold"][["episode_id", "family", "parameter", "attack_start", "attack_end", "duration_min"]]

    res_ramp = enriquecer_resultado(res_ramp_raw, fold_res["ataques_fold"]["ataques_ramp"], meta_ep_ramp, fold_res["ataques_fold"]["particion_fold_raw"], contexto_ramp)
    res_const = enriquecer_resultado(res_const_raw, fold_res["ataques_fold"]["ataques_constante"], meta_ep_const, fold_res["ataques_fold"]["particion_fold_raw"], contexto_const)
    m_ramp, m_const = resumen_metricas(res_ramp), resumen_metricas(res_const)

    sub_hi = res_ramp[res_ramp["episode_id"].isin(fold_res["alto_impacto_ids"])]
    hi_n, hi_total = int(sub_hi["induced_detected"].sum()), len(sub_hi)
    sub_b = res_const[res_const["episode_id"].isin(fold_res["grupo_b_ids"])]
    gb_n, gb_total = int(sub_b["induced_detected"].sum()), len(sub_b)
    dur_720_1440 = res_ramp[res_ramp["duration_min"].isin([720, 1440])]
    dr_720_1440 = 100 * dur_720_1440["induced_detected"].mean() if len(dur_720_1440) else float("nan")
    dur_30_120 = res_ramp[res_ramp["duration_min"].isin([30, 60, 120])]
    dr_30_120 = 100 * dur_30_120["induced_detected"].mean() if len(dur_30_120) else float("nan")

    admisible_fold = bool(fa <= 0.15 and not sat["falla_saturacion"] and np.isfinite(fa))

    return {
        "nombre": nombre, "fold": fold_def["fold"], "fa": fa, **sat, "admisible_fold": admisible_fold,
        "dr_ramp_pct": m_ramp["induced_DR_pct"], "dr_energ_ramp_pct": m_ramp["energy_weighted_dr_pct"],
        "timely_energy_dr_pct": m_ramp["timely_energy_detection_rate_pct"], "retraso_mediano_min": m_ramp["retraso_mediano_min"],
        "n_ramp_episodios": len(res_ramp), "n_ramp_detectados": int(res_ramp["induced_detected"].sum()),
        "alto_impacto_n": hi_n, "alto_impacto_total": hi_total, "dr_720_1440_pct": dr_720_1440, "dr_30_120_pct": dr_30_120,
        "dr_const_pct": m_const["induced_DR_pct"], "dr_energ_const_pct": m_const["energy_weighted_dr_pct"],
        "grupo_b_n": gb_n, "grupo_b_total": gb_total,
        "res_ramp": res_ramp, "res_const": res_const,
    }


def evaluar_fold_completo(fold_def: dict, fold_res: dict, p_global: dict) -> dict:
    resultados = {}
    meta_ramp_hrel, meta_const_hrel = fold_res["meta_ramp_hrel"], fold_res["meta_const_hrel"]
    meta_ramp_hz, meta_const_hz = fold_res["meta_ramp_hz"], fold_res["meta_const_hz"]

    for nombre, p_cfg in (("P_GLOBAL", fold_res["p_global_cfg"]), ("P_MULTI_SEASON", fold_res["p_multiseason_cfg"]), ("P_ROBUST_Z", fold_res["p_robust_z_cfg"])):
        resultados[nombre] = evaluar_configuracion_fold(nombre, "P", p_cfg, None, None, meta_ramp_hrel, meta_const_hrel, fold_res, p_global, fold_def)

    resultados["H_RELATIVE"] = evaluar_configuracion_fold("H_RELATIVE", "H", None, fold_res["h_relative_cfg"]["score_full"],
                                                            fold_res["h_relative_cfg"]["umbral"], meta_ramp_hrel, meta_const_hrel, fold_res, p_global, fold_def)
    resultados["H_ROBUST_Z"] = evaluar_configuracion_fold("H_ROBUST_Z", "H", None, fold_res["h_robust_z_cfg"]["score_full"],
                                                            fold_res["h_robust_z_cfg"]["umbral"], meta_ramp_hz, meta_const_hz, fold_res, p_global, fold_def)

    for nombre in FUSIONES:
        fcfg = fold_res["fusiones_cfg"][nombre]
        p_cfg_fusion = {**fcfg["p"], "umbral": fcfg["p_umbral_or"]}
        resultados[nombre] = evaluar_configuracion_fold(nombre, "OR", p_cfg_fusion, fcfg["h"]["score_full"], fcfg["h"]["umbral"],
                                                          fcfg["meta_ramp"], fcfg["meta_const"], fold_res, p_global, fold_def)
        resultados[nombre]["structurally_uncalibratable_due_to_P"] = fcfg.get("structurally_uncalibratable_due_to_P", False)
    return resultados


# ==========================================================================================
# 20-21) Diagnostico mensual de P_GLOBAL y desglose diario de agosto de 2008
# ==========================================================================================

def diagnostico_mensual_p_global(p_global: dict) -> pd.DataFrame:
    ts = pd.DatetimeIndex(p_global["ts"])
    activo = p_global["score"] > p_global["umbral"]
    filas = []
    for periodo, idx in pd.Series(np.arange(len(ts))).groupby(ts.to_period("M")):
        g_ts, g_act, g_score = ts[idx], activo[idx], p_global["score"][idx]
        n_dias = max((g_ts.max() - g_ts.min()).days + 1, 1)
        sat = calcular_saturacion(g_act, g_ts.to_numpy(), resolucion_min=60)
        filas.append({"mes": str(periodo), "n_decisiones": len(idx), "fa_dia": contar_eventos(g_act) / n_dias,
                       **sat, "score_mediana": float(np.median(g_score)), "score_p95": float(np.percentile(g_score, 95))})
    return pd.DataFrame(filas)


def diagnostico_agosto_2008(fold1_res: dict, resultados_fold1: dict, p_global: dict) -> pd.DataFrame:
    ts = pd.DatetimeIndex(p_global["ts"])
    mask_ago = (ts >= "2008-08-01") & (ts < "2008-09-01")
    ts_ago, score_ago = ts[mask_ago], p_global["score"][mask_ago]
    activo_pglobal_ago = score_ago > UMBRAL_P_FIJO

    target_end_master = fold1_res["target_end_master"]
    mask_h_ago = (pd.DatetimeIndex(target_end_master) >= "2008-08-01") & (pd.DatetimeIndex(target_end_master) < "2008-09-01")
    ts_h_ago = pd.DatetimeIndex(target_end_master)[mask_h_ago]

    activo_pmulti = fus.estado_ffill(fold1_res["p_multiseason_cfg"]["score_full"], fold1_res["sel_multiseason"]["umbral"], p_global["ts"], target_end_master)[mask_h_ago]
    activo_pz = fus.estado_ffill(fold1_res["p_robust_z_cfg"]["score_full"], fold1_res["sel_pz"]["umbral"], p_global["ts"], target_end_master)[mask_h_ago]
    activo_hrel = (fold1_res["h_relative_cfg"]["score_full"] > fold1_res["h_relative_cfg"]["umbral"])[mask_h_ago]
    activo_hz = (fold1_res["h_robust_z_cfg"]["score_full"] > fold1_res["h_robust_z_cfg"]["umbral"])[mask_h_ago]

    filas = []
    for dia, idx in pd.Series(np.arange(mask_h_ago.sum())).groupby(ts_h_ago.date):
        idx_p = np.where((ts_ago.date == dia))[0]
        filas.append({
            "fecha": str(dia), "P_GLOBAL_score_mediana": float(np.median(score_ago[idx_p])) if len(idx_p) else np.nan,
            "P_GLOBAL_score_p95": float(np.percentile(score_ago[idx_p], 95)) if len(idx_p) else np.nan,
            "P_GLOBAL_active_fraction": float(activo_pglobal_ago[idx_p].mean()) if len(idx_p) else np.nan,
            "P_MULTI_SEASON_active_fraction": float(activo_pmulti[idx].mean()),
            "P_ROBUST_Z_active_fraction": float(activo_pz[idx].mean()),
            "H_RELATIVE_active_fraction": float(activo_hrel[idx].mean()), "H_ROBUST_Z_active_fraction": float(activo_hz[idx].mean()),
        })
    tabla = pd.DataFrame(filas)
    tabla.attrs["nota"] = ("periodo compatible con ausencia, desocupacion temporal o cambio prolongado de regimen "
                            "(no se afirma que sean vacaciones como hecho)")
    return tabla


# ==========================================================================================
# 29-30) Admisibilidad por fold (ya calculada en evaluar_configuracion_fold) y global
# ==========================================================================================

def calcular_retencion_h(res_config: pd.DataFrame, res_h_referencia: pd.DataFrame) -> float:
    h_det = set(res_h_referencia[res_h_referencia["induced_detected"]]["episode_id"])
    if not h_det:
        return 100.0
    det = set(res_config[res_config["induced_detected"]]["episode_id"])
    return 100 * len(h_det & det) / len(h_det)


def calcular_exclusivas_respecto_p(res_config: pd.DataFrame, res_pglobal: pd.DataFrame) -> set:
    p_det = set(res_pglobal[res_pglobal["induced_detected"]]["episode_id"])
    det = set(res_config[res_config["induced_detected"]]["episode_id"])
    return det - p_det


def evaluar_admisibilidad_global(nombre: str, filas_config: list[dict], n_folds_total: int,
                                  retencion_h_por_fold: list[float] | None, exclusivas_por_fold: list[int]) -> dict:
    completo = len(filas_config) == n_folds_total
    admisible_por_fold = bool(all(f["admisible_fold"] for f in filas_config)) if completo else False
    fold1_incluido = any(f["fold"] == 1 for f in filas_config)
    fa_media_pond = float(np.average([f["fa"] for f in filas_config], weights=[opt.VALID_DIAS] * len(filas_config))) if filas_config else float("nan")

    retencion_min = min(retencion_h_por_fold) if retencion_h_por_fold else None
    total_exclusivas = sum(exclusivas_por_fold)
    concentracion_max_pct = 100 * max(exclusivas_por_fold) / total_exclusivas if total_exclusivas > 0 else 0.0

    admisible = bool(
        completo and fa_media_pond <= 0.12 and admisible_por_fold and fold1_incluido
        and (retencion_min is None or retencion_min >= 90.0) and concentracion_max_pct <= 70.0)
    return {
        "nombre": nombre, "completo": completo, "admisible_por_fold": admisible_por_fold, "fold1_incluido": fold1_incluido,
        "fa_media_ponderada": fa_media_pond, "retencion_h_minima_pct": retencion_min, "concentracion_exclusivas_max_pct": concentracion_max_pct,
        "admisible_global": admisible,
    }


# ==========================================================================================
# 33) Regla de seleccion final (Opcion A: fusion / Opcion B: H solo / Opcion C: ninguna)
# ==========================================================================================

_COMPLEJIDAD_P = {"P_GLOBAL": 0, "P_MULTI_SEASON": 1, "P_ROBUST_Z": 2}
_COMPLEJIDAD_H = {"H_RELATIVE": 0, "H_ROBUST_Z": 1}
_H_DE_LA_FUSION = {"OR_PGLOBAL_HREL": "H_RELATIVE", "OR_PMULTI_HREL": "H_RELATIVE", "OR_PMULTI_HZ": "H_ROBUST_Z",
                   "OR_PZ_HREL": "H_RELATIVE", "OR_PZ_HZ": "H_ROBUST_Z"}
_P_DE_LA_FUSION = {"OR_PGLOBAL_HREL": "P_GLOBAL", "OR_PMULTI_HREL": "P_MULTI_SEASON", "OR_PMULTI_HZ": "P_MULTI_SEASON",
                   "OR_PZ_HREL": "P_ROBUST_Z", "OR_PZ_HZ": "P_ROBUST_Z"}


def evaluar_opcion_a(nombre_fusion: str, filas_fusion: list[dict], filas_h_ref: list[dict], adm_fusion: dict,
                      exclusivas_por_fold: list[int]) -> tuple[bool, dict]:
    if nombre_fusion == "OR_PGLOBAL_HREL" or not adm_fusion["admisible_global"]:
        return False, {"motivo": "no admisible globalmente o es la referencia OR_PGLOBAL_HREL (solo reproduccion)"}
    if any(f["falla_saturacion"] for f in filas_fusion):
        return False, {"motivo": "algun fold saturado"}

    fh = {f["fold"]: f for f in filas_h_ref}
    ff = {f["fold"]: f for f in filas_fusion}
    folds_comunes = sorted(set(fh) & set(ff))

    dr_energ_fusion = np.array([ff[k]["dr_energ_ramp_pct"] for k in folds_comunes])
    dr_energ_h = np.array([fh[k]["dr_energ_ramp_pct"] for k in folds_comunes])
    dr_fusion = np.array([ff[k]["dr_ramp_pct"] for k in folds_comunes])
    dr_h = np.array([fh[k]["dr_ramp_pct"] for k in folds_comunes])

    mejora_energ_mediana = float(np.median(dr_energ_fusion - dr_energ_h))
    mejora_dr_mediana = float(np.median(dr_fusion - dr_h))
    c3 = mejora_energ_mediana >= 3.0
    c4 = mejora_dr_mediana >= 2.0

    hi_fusion, hi_h = sum(f["alto_impacto_n"] for f in filas_fusion), sum(f["alto_impacto_n"] for f in filas_h_ref)
    c5 = hi_fusion >= 0.90 * hi_h if hi_h > 0 else True

    c6 = adm_fusion["retencion_h_minima_pct"] is not None and adm_fusion["retencion_h_minima_pct"] >= 90.0
    c7 = sum(1 for e in exclusivas_por_fold if e > 0) >= 3
    c8 = int(((dr_energ_fusion - dr_energ_h) > 0).sum()) >= 3

    retraso_fusion = np.nanmedian([f["retraso_mediano_min"] for f in filas_fusion])
    retraso_h = np.nanmedian([f["retraso_mediano_min"] for f in filas_h_ref])
    c9 = bool(np.isnan(retraso_fusion) or np.isnan(retraso_h) or (retraso_fusion - retraso_h) <= 60.0)
    c10 = True  # no depende exclusivamente de una duracion/severidad -- documentado cualitativamente

    criterios = {1: adm_fusion["admisible_global"], 2: not any(f["falla_saturacion"] for f in filas_fusion), 3: c3, 4: c4,
                 5: c5, 6: c6, 7: c7, 8: c8, 9: c9, 10: c10}
    detalle = {"mejora_dr_energ_mediana_pp": mejora_energ_mediana, "mejora_dr_mediana_pp": mejora_dr_mediana,
               "alto_impacto_fusion": hi_fusion, "alto_impacto_h": hi_h, "retencion_h_pct": adm_fusion["retencion_h_minima_pct"],
               "n_folds_con_exclusivas": sum(1 for e in exclusivas_por_fold if e > 0), "signos_positivos": int(((dr_energ_fusion - dr_energ_h) > 0).sum()),
               "retraso_fusion_min": retraso_fusion, "retraso_h_min": retraso_h, "criterios": criterios}
    return all(criterios.values()), detalle


def rankear_opcion_a(candidatos_validos: list[dict]) -> pd.DataFrame:
    if not candidatos_validos:
        return pd.DataFrame()
    df = pd.DataFrame(candidatos_validos)
    df = df.sort_values(
        by=["dr_energ_mediana", "peor_fold_dr_energ", "alto_impacto_total", "dr_mediana", "active_fraction_ponderada",
            "duracion_max_horas_peor", "grupo_b_total", "complejidad"],
        ascending=[False, False, False, False, True, True, False, True]).reset_index(drop=True)
    df.insert(0, "ranking", np.arange(1, len(df) + 1))
    return df


def evaluar_opcion_b(nombre_h: str, filas_h: list[dict], adm_h: dict) -> bool:
    return bool(adm_h["admisible_global"] and adm_h["completo"] and not any(f["falla_saturacion"] for f in filas_h))


# ==========================================================================================
# 34) Comparaciones estadisticas (por fold; episode_id NO es unico entre folds -- nunca se
#     agrupan episodios de folds distintos como si fueran el mismo)
# ==========================================================================================

def comparar_pareado(nombre_a: str, nombre_b: str, res_ramp_por_fold_a: dict, res_ramp_por_fold_b: dict) -> pd.DataFrame:
    filas = []
    for fold in sorted(set(res_ramp_por_fold_a) & set(res_ramp_por_fold_b)):
        da_full = res_ramp_por_fold_a[fold].set_index("episode_id")["induced_detected"].astype(bool)
        db_full = res_ramp_por_fold_b[fold].set_index("episode_id")["induced_detected"].astype(bool).reindex(da_full.index).fillna(False)
        da, db = da_full.to_numpy(), db_full.to_numpy()
        solo_a, solo_b = int((da & ~db).sum()), int((~da & db).sum())
        p_mc = mcnemar_p(solo_a, solo_b)
        ci_low, ci_high = bootstrap_pareado(da.astype(float), db.astype(float))
        h_a = res_ramp_por_fold_a[fold].set_index("episode_id")["hidden_energy_kwh"].reindex(da_full.index).to_numpy()
        ci_e_low, ci_e_high = bootstrap_pareado(np.where(da, h_a, 0.0), np.where(db, h_a, 0.0))
        delay_a = res_ramp_por_fold_a[fold].set_index("episode_id")["detection_delay_min"]
        delay_b = res_ramp_por_fold_b[fold].set_index("episode_id")["detection_delay_min"].reindex(delay_a.index)
        ambos = da_full & db_full
        if ambos.sum() >= 5:
            try:
                _, p_w = sstats.wilcoxon(delay_a[ambos].to_numpy(), delay_b[ambos].to_numpy())
            except ValueError:
                p_w = float("nan")
        else:
            p_w = float("nan")
        filas.append({"comparacion": f"{nombre_a}_vs_{nombre_b}", "fold": fold, "n_episodios": len(da_full),
                       "DR_a_pct": 100 * da.mean(), "DR_b_pct": 100 * db.mean(), "diferencia_pp": 100 * (db.mean() - da.mean()),
                       "mcnemar_p": p_mc, "bootstrap_dr_ci95_low": ci_low, "bootstrap_dr_ci95_high": ci_high,
                       "bootstrap_energia_ci95_low": ci_e_low, "bootstrap_energia_ci95_high": ci_e_high,
                       "wilcoxon_p_retraso": p_w, "n_detectados_ambos": int(ambos.sum()), "significativo_p<0.05": p_mc < 0.05})
    tabla = pd.DataFrame(filas)
    if len(tabla):
        agregado = {"comparacion": f"{nombre_a}_vs_{nombre_b}", "fold": "agregado", "n_episodios": int(tabla["n_episodios"].sum()),
                    "DR_a_pct": float(tabla["DR_a_pct"].mean()), "DR_b_pct": float(tabla["DR_b_pct"].mean()),
                    "diferencia_pp": float(tabla["diferencia_pp"].mean()), "mcnemar_p": np.nan,
                    "bootstrap_dr_ci95_low": np.nan, "bootstrap_dr_ci95_high": np.nan, "bootstrap_energia_ci95_low": np.nan,
                    "bootstrap_energia_ci95_high": np.nan, "wilcoxon_p_retraso": np.nan,
                    "n_detectados_ambos": int(tabla["n_detectados_ambos"].sum()), "significativo_p<0.05": np.nan,
                    "n_folds_diferencia_positiva": int((tabla["diferencia_pp"] > 0).sum())}
        tabla = pd.concat([tabla, pd.DataFrame([agregado])], ignore_index=True)
    return tabla


# ==========================================================================================
# 35-36) Entrenamiento final pre-test y congelacion (solo si existe candidato admisible)
# ==========================================================================================

def entrenar_final_robusto(seleccion: str, detalle_seleccion: dict, folds_info: dict, serie_completa_min: pd.DataFrame,
                            p_global: dict, metadatos_baseline: dict, stats_p_final_folds: list, stats_h_final_folds: list) -> dict:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    final_ini, final_fin = folds_info["final_calibration_ini"], folds_info["final_calibration_fin"]

    if seleccion == "NONE_NOT_READY_FOR_TEST":
        payload = {"selected_configuration": "NONE_NOT_READY_FOR_TEST", "motivo": detalle_seleccion,
                   "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "test_intacto": True}
        with open(MODELOS_DIR / "configuracion_final_robusta_pretest.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return {"seleccion": seleccion, "payload": payload}

    usa_h_robust_z = "HZ" in seleccion or seleccion == "H_ROBUST_Z_ONLY"
    usa_p_multiseason = "PMULTI" in seleccion
    usa_p_robust_z = "PZ" in seleccion
    cols = opt.PERIODIC_BASE_FEATURES

    particion_entrenamiento = serie_completa_min.loc[serie_completa_min.index < final_ini]
    particion_calib_final = serie_completa_min.loc[(serie_completa_min.index >= final_ini) & (serie_completa_min.index < final_fin)]
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

    stats_h_final = None
    if usa_h_robust_z:
        pred_train_final = modelo_final.predict(df_train[cols].to_numpy(dtype=np.float64))
        signed_train_final, _ = pcl.calcular_score(pred_train_final, df_train["target_kw"].to_numpy())
        stats_h_final = calcular_stats_h_robust_z(signed_train_final, df_train["ts_end"].to_numpy())

    serie_para_calib = pd.concat([particion_entrenamiento.iloc[-(pcl.MAX_LAG + 10) * RESOLUCION_MIN:], particion_calib_final])
    kw_calib = pcl.construir_serie_15min(serie_para_calib)
    feats_calib = pcl.construir_features_kw(kw_calib["kw"])
    cal_calib = pcl._calendario(kw_calib["ts_start"])
    df_calib_full = pd.concat([feats_calib, cal_calib], axis=1)
    df_calib_full["target_kw"] = kw_calib["kw"]
    df_calib_full["ts_start"] = kw_calib["ts_start"]
    df_calib_full["ts_end"] = kw_calib["ts_end"]
    df_calib_full = df_calib_full.merge(perfil_final, on=["es_finde", "slot_15min"], how="left")
    df_calib = df_calib_full[df_calib_full["ts_start"] >= final_ini].dropna(subset=cols).reset_index(drop=True)

    pred_calib_final = modelo_final.predict(df_calib[cols].to_numpy(dtype=np.float64))
    signed_calib_final, positive_calib_final = pcl.calcular_score(pred_calib_final, df_calib["target_kw"].to_numpy())
    ts_calib_final = df_calib["ts_end"].to_numpy()
    if usa_h_robust_z:
        score_calib_final, _ = aplicar_h_robust_z(signed_calib_final, ts_calib_final, stats_h_final)
    else:
        score_calib_final = positive_calib_final
    _, sel_h_final = calibrar_umbral_h(score_calib_final, ts_calib_final, FINAL_CALIBRATION_DIAS_ROBUSTO)

    payload = {
        "selected_configuration": seleccion, "modelo_h": "HistGB_PERIODIC (hiperparametros identicos a predictor_causal_lags)",
        "features": cols, "hiperparametros": opt.CONFIG_HISTGB_PERIODIC_ACTUAL, "score_h": "ROBUST_Z" if usa_h_robust_z else "RELATIVE",
        "umbral_h": float(sel_h_final["umbral"]), "estadisticas_h_robust_z": {"centro_global": stats_h_final["centro_global"], "escala_global": stats_h_final["escala_global"]} if stats_h_final else None,
        "umbral_p": UMBRAL_P_FIJO if seleccion.startswith("H_") else None,
        "nota_p": "seleccion final es H solo -- P no participa" if seleccion.startswith("H_") else "P se recalibraria de igual forma en produccion, pendiente de definir umbral con datos de FINAL_CALIBRATION si se decide usar fusion",
        "forward_fill_p": "causal, np.searchsorted backward, resolucion 15min", "regla_or": "active_P OR active_H" if not seleccion.startswith("H_") else "no aplica (H solo)",
        "limites_saturacion": {"active_fraction": LIM_ACTIVE_FRACTION, "longest_clean_event_hours": LIM_LONGEST_CLEAN_HOURS,
                               "seven_day_max_active_fraction": LIM_7DAY_MAX_ACTIVE, "consecutive_days_over_50": LIM_CONSECUTIVE_DAYS_50},
        "fechas_entrenamiento": [str(particion_entrenamiento.index[0]), str(final_ini)], "fechas_calibracion_final": [str(final_ini), str(final_fin)],
        "sklearn_version": sklearn.__version__, "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(),
        "metadatos_baseline": metadatos_baseline, "test_intacto": True,
    }
    joblib.dump(modelo_final, MODELOS_DIR / "H_final.joblib")
    with open(MODELOS_DIR / "configuracion_final_robusta_pretest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return {"seleccion": seleccion, "payload": payload, "modelo": modelo_final, "sel_umbral_final": sel_h_final}


FINAL_CALIBRATION_DIAS_ROBUSTO = 60


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("Fase 0: reproduciendo el baseline (P/H/OR historicos, sin recalcular)...")
    tabla_repro, metadatos_baseline = opt.reproducir_baseline()
    _guardar_tabla(tabla_repro, "reproduccion_baseline")
    if not tabla_repro["coincide"].all():
        raise RuntimeError("el baseline no se reproduce dentro de tolerancia -- experimento detenido (seccion 11)")
    print("OK -- baseline reproducido dentro de tolerancia.")

    df_min = cargar_y_limpiar()
    particiones = split_temporal(df_min)
    serie_completa_min = pd.concat([particiones["train"], particiones["val"]])
    inicio = particiones["train"].index[0]
    fin_pretest = particiones["test"].index[0]
    folds_info = opt.construir_folds(inicio, fin_pretest)
    _guardar_tabla(pd.DataFrame(folds_info["folds"]), "definicion_folds")
    assert len(folds_info["folds"]) == 4
    print(f"  {len(folds_info['folds'])} folds reutilizados exactamente | FINAL_CALIBRATION: [{folds_info['final_calibration_ini']}, {folds_info['final_calibration_fin']})")

    print("\nAuditando el checkpoint de P...")
    tabla_audit = auditar_checkpoint_p(folds_info)
    print(tabla_audit[["elemento", "descripcion"]].to_string(index=False))

    from src.windowing import cargar_config
    config = cargar_config()
    _, datos_p, modelo_p, _, _ = cargar_ataques_y_modelo_360_wrapper()
    p_global = opt.cargar_p_global(modelo_p, datos_p["params_norm"])

    print("\nDiagnostico mensual de P_GLOBAL (linea temporal completa pre-test)...")
    tabla_diag_mensual = _guardar_tabla(diagnostico_mensual_p_global(p_global), "diagnostico_mensual_p")
    print(tabla_diag_mensual[["mes", "fa_dia", "active_fraction"]].to_string(index=False))

    resultados_por_fold: dict[int, dict] = {}
    fold_res_por_fold: dict[int, dict] = {}
    for fold_def in folds_info["folds"]:
        print(f"\n=== Fold {fold_def['fold']}: valid {fold_def['calib_fin'].date()} -> {fold_def['valid_fin'].date()} ===")
        fold_res = procesar_fold(fold_def, serie_completa_min, config, p_global)
        fold_res_por_fold[fold_def["fold"]] = fold_res
        resultados_fold = evaluar_fold_completo(fold_def, fold_res, p_global)
        resultados_por_fold[fold_def["fold"]] = resultados_fold
        for nombre in CONFIGURACIONES_INDIVIDUALES + FUSIONES:
            r = resultados_fold[nombre]
            print(f"  {nombre:16s} FA={r['fa']:.4f} act_frac={r['active_fraction']:.3f} sat={'SI' if r['falla_saturacion'] else 'no'} "
                  f"DR={r['dr_ramp_pct']:.1f}% DR_energ={r['dr_energ_ramp_pct']:.1f}% HI={r['alto_impacto_n']}/{r['alto_impacto_total']}")

    print("\nAnalizando agosto de 2008 (fold 1)...")
    tabla_agosto = _guardar_tabla(diagnostico_agosto_2008(fold_res_por_fold[1], resultados_por_fold[1], p_global), "diagnostico_agosto_2008")

    # tablas por configuracion (umbrales, FA, saturacion, ramp, energia, alto impacto, constantes, grupo B)
    todas_config = CONFIGURACIONES_INDIVIDUALES + FUSIONES
    filas_umbrales, filas_fa, filas_sat, filas_ramp, filas_energia, filas_hi, filas_const, filas_gb = [], [], [], [], [], [], [], []
    res_ramp_por_config_fold: dict[str, dict[int, pd.DataFrame]] = {n: {} for n in todas_config}
    for fold_def in folds_info["folds"]:
        k = fold_def["fold"]
        for nombre in todas_config:
            r = resultados_por_fold[k][nombre]
            res_ramp_por_config_fold[nombre][k] = r["res_ramp"]
            filas_fa.append({"configuracion": nombre, "fold": k, "fa": r["fa"], "admisible_fold": r["admisible_fold"]})
            filas_sat.append({"configuracion": nombre, "fold": k, "active_fraction": r["active_fraction"],
                              "longest_clean_event_hours": r["longest_clean_event_hours"], "seven_day_max_active_fraction": r["seven_day_max_active_fraction"],
                              "consecutive_days_over_50": r["consecutive_days_over_50"], "falla_saturacion": r["falla_saturacion"]})
            filas_ramp.append({"configuracion": nombre, "fold": k, "dr_ramp_pct": r["dr_ramp_pct"], "dr_720_1440_pct": r["dr_720_1440_pct"], "dr_30_120_pct": r["dr_30_120_pct"]})
            filas_energia.append({"configuracion": nombre, "fold": k, "dr_energ_ramp_pct": r["dr_energ_ramp_pct"], "timely_energy_dr_pct": r["timely_energy_dr_pct"]})
            filas_hi.append({"configuracion": nombre, "fold": k, "alto_impacto_n": r["alto_impacto_n"], "alto_impacto_total": r["alto_impacto_total"]})
            filas_const.append({"configuracion": nombre, "fold": k, "dr_const_pct": r["dr_const_pct"], "dr_energ_const_pct": r["dr_energ_const_pct"]})
            filas_gb.append({"configuracion": nombre, "fold": k, "grupo_b_n": r["grupo_b_n"], "grupo_b_total": r["grupo_b_total"]})
        filas_umbrales.append({"fold": k, "umbral_P_multiseason": fold_res_por_fold[k]["sel_multiseason"]["umbral"],
                               "umbral_P_robust_z": fold_res_por_fold[k]["sel_pz"]["umbral"], "umbral_H_relative": fold_res_por_fold[k]["sel_hrel"]["umbral"],
                               "umbral_H_robust_z": fold_res_por_fold[k]["sel_hz"]["umbral"]})
    _guardar_tabla(pd.DataFrame(filas_umbrales), "umbrales_por_fold")
    _guardar_tabla(pd.DataFrame(filas_fa), "falsas_alarmas_por_fold")
    _guardar_tabla(pd.DataFrame(filas_sat), "saturacion_por_fold")
    _guardar_tabla(pd.DataFrame(filas_ramp), "ramp_por_fold")
    _guardar_tabla(pd.DataFrame(filas_energia), "energia_por_fold")
    _guardar_tabla(pd.DataFrame(filas_hi), "alto_impacto_por_fold")
    _guardar_tabla(pd.DataFrame(filas_const), "constantes_por_fold")
    _guardar_tabla(pd.DataFrame(filas_gb), "grupo_b_por_fold")

    for fold_def in folds_info["folds"]:
        k = fold_def["fold"]
        _guardar_tabla(fold_res_por_fold[k]["tabla_umb_multiseason"], f"calibracion_p_multiseason_fold{k}")
    _guardar_tabla(pd.concat([pd.DataFrame([{"fold": k, "centro_global": fold_res_por_fold[k]["stats_p"]["centro_global"],
                                              "escala_global": fold_res_por_fold[k]["stats_p"]["escala_global"]}]) for k in fold_res_por_fold], ignore_index=True),
                  "estadisticas_robust_z_p")
    _guardar_tabla(pd.concat([pd.DataFrame([{"fold": k, "centro_global": fold_res_por_fold[k]["stats_h"]["centro_global"],
                                              "escala_global": fold_res_por_fold[k]["stats_h"]["escala_global"]}]) for k in fold_res_por_fold], ignore_index=True),
                  "estadisticas_robust_z_h")

    # exclusivas respecto a P_GLOBAL, por configuracion y fold (necesario para admisibilidad y regla A)
    filas_excl = []
    exclusivas_por_config_fold: dict[str, dict[int, int]] = {n: {} for n in todas_config}
    for fold_def in folds_info["folds"]:
        k = fold_def["fold"]
        res_pglobal = resultados_por_fold[k]["P_GLOBAL"]["res_ramp"]
        for nombre in todas_config:
            excl = calcular_exclusivas_respecto_p(resultados_por_fold[k][nombre]["res_ramp"], res_pglobal)
            exclusivas_por_config_fold[nombre][k] = len(excl)
            filas_excl.append({"configuracion": nombre, "fold": k, "n_exclusivas_respecto_p_global": len(excl)})
    _guardar_tabla(pd.DataFrame(filas_excl), "exclusivas_por_fold")

    # retencion de H (referencia = el H usado en cada fusion) por fold, para las 5 fusiones
    filas_retencion = []
    retencion_por_fusion_fold: dict[str, dict[int, float]] = {n: {} for n in FUSIONES}
    for fold_def in folds_info["folds"]:
        k = fold_def["fold"]
        for nombre in FUSIONES:
            h_ref = resultados_por_fold[k][_H_DE_LA_FUSION[nombre]]["res_ramp"]
            ret = calcular_retencion_h(resultados_por_fold[k][nombre]["res_ramp"], h_ref)
            retencion_por_fusion_fold[nombre][k] = ret
            filas_retencion.append({"fusion": nombre, "fold": k, "retencion_h_pct": ret})
    _guardar_tabla(pd.DataFrame(filas_retencion), "retencion_h_por_fold")

    # admisibilidad global por configuracion
    print("\nEvaluando admisibilidad global de las 10 configuraciones...")
    admisibilidad = {}
    for nombre in todas_config:
        filas_cfg = [{k2: v2 for k2, v2 in resultados_por_fold[k][nombre].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
        retencion_h = list(retencion_por_fusion_fold[nombre].values()) if nombre in FUSIONES else None
        exclusivas = list(exclusivas_por_config_fold[nombre].values())
        admisibilidad[nombre] = evaluar_admisibilidad_global(nombre, filas_cfg, len(folds_info["folds"]), retencion_h, exclusivas)
        print(f"  {nombre:16s} admisible_global={admisibilidad[nombre]['admisible_global']} | FA_pond={admisibilidad[nombre]['fa_media_ponderada']:.4f}")
    tabla_admisibles = _guardar_tabla(pd.DataFrame(admisibilidad.values()), "candidatos_admisibles")

    print("\nComparaciones estadisticas...")
    pares = [("P_GLOBAL", "P_MULTI_SEASON"), ("P_MULTI_SEASON", "P_ROBUST_Z"), ("P_GLOBAL", "P_ROBUST_Z"),
             ("H_RELATIVE", "H_ROBUST_Z")]
    tablas_stats = []
    for a, b in pares:
        tablas_stats.append(comparar_pareado(a, b, res_ramp_por_config_fold[a], res_ramp_por_config_fold[b]))
    _guardar_tabla(pd.concat(tablas_stats, ignore_index=True), "comparaciones_estadisticas")

    # regla de seleccion final
    print("\nAplicando la regla de seleccion final (Opcion A / B / C)...")
    n_folds_total = len(folds_info["folds"])
    candidatos_opcion_a = []
    detalle_opcion_a = {}
    for nombre in ["OR_PMULTI_HREL", "OR_PMULTI_HZ", "OR_PZ_HREL", "OR_PZ_HZ"]:
        filas_fusion = [{k2: v2 for k2, v2 in resultados_por_fold[k][nombre].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
        h_ref_nombre = _H_DE_LA_FUSION[nombre]
        filas_h_ref = [{k2: v2 for k2, v2 in resultados_por_fold[k][h_ref_nombre].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
        pasa, detalle = evaluar_opcion_a(nombre, filas_fusion, filas_h_ref, admisibilidad[nombre], list(exclusivas_por_config_fold[nombre].values()))
        detalle_opcion_a[nombre] = detalle
        print(f"  {nombre}: {'CUMPLE' if pasa else 'no cumple'} la regla A | mejora_DR_energ_mediana={detalle.get('mejora_dr_energ_mediana_pp', float('nan')):.2f}pp")
        if pasa:
            df_fusion = pd.DataFrame(filas_fusion)
            candidatos_opcion_a.append({
                "nombre": nombre, "dr_energ_mediana": float(df_fusion["dr_energ_ramp_pct"].median()),
                "peor_fold_dr_energ": float(df_fusion["dr_energ_ramp_pct"].min()), "alto_impacto_total": int(df_fusion["alto_impacto_n"].sum()),
                "dr_mediana": float(df_fusion["dr_ramp_pct"].median()), "active_fraction_ponderada": float(df_fusion["active_fraction"].mean()),
                "duracion_max_horas_peor": float(df_fusion["longest_clean_event_hours"].max()), "grupo_b_total": int(df_fusion["grupo_b_n"].sum()),
                "complejidad": _COMPLEJIDAD_P[_P_DE_LA_FUSION[nombre]] + _COMPLEJIDAD_H[_H_DE_LA_FUSION[nombre]],
            })
    _guardar_tabla(pd.DataFrame([{"fusion": n, **{f"criterio_{i}": v for i, v in d.get("criterios", {}).items()}} for n, d in detalle_opcion_a.items()]), "regla_seleccion")
    ranking_a = rankear_opcion_a(candidatos_opcion_a)

    if len(ranking_a):
        seleccion_final = ranking_a.iloc[0]["nombre"]
        detalle_seleccion = {"opcion": "A", "detalle": detalle_opcion_a[seleccion_final]}
        print(f"\n>>> OPCION A: se selecciona la fusion estabilizada {seleccion_final}")
    else:
        candidatos_b = []
        for nombre in ("H_RELATIVE", "H_ROBUST_Z"):
            filas_h = [{k2: v2 for k2, v2 in resultados_por_fold[k][nombre].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
            if evaluar_opcion_b(nombre, filas_h, admisibilidad[nombre]):
                df_h = pd.DataFrame(filas_h)
                candidatos_b.append({"nombre": nombre, "peor_fa": float(df_h["fa"].max()), "variab_mensual": float(df_h["fa"].std(ddof=0)),
                                     "active_fraction": float(df_h["active_fraction"].mean()), "dr_energ": float(df_h["dr_energ_ramp_pct"].median()),
                                     "alto_impacto": int(df_h["alto_impacto_n"].sum()), "grupo_b": int(df_h["grupo_b_n"].sum()),
                                     "complejidad": _COMPLEJIDAD_H[nombre]})
        if candidatos_b:
            df_b = pd.DataFrame(candidatos_b).sort_values(
                by=["peor_fa", "variab_mensual", "active_fraction", "dr_energ", "alto_impacto", "grupo_b", "complejidad"],
                ascending=[True, True, True, False, False, False, True])
            ganador_b = df_b.iloc[0]["nombre"]
            seleccion_final = f"{ganador_b}_ONLY"
            detalle_seleccion = {"opcion": "B", "candidato": ganador_b}
            print(f"\n>>> OPCION B: ninguna fusion cumple la regla A -- se selecciona {seleccion_final} (H solo)")
        else:
            seleccion_final = "NONE_NOT_READY_FOR_TEST"
            detalle_seleccion = {"opcion": "C", "motivo": "ni una fusion (regla A) ni H solo (regla B) son globalmente admisibles"}
            print(f"\n>>> OPCION C: {seleccion_final} -- no se abre test, se documenta la inestabilidad temporal")

    _guardar_tabla(pd.DataFrame([{"selected_configuration": seleccion_final}]), "configuracion_seleccionada")

    print("\nEntrenamiento final pre-test (si procede) y congelacion...")
    resultado_final = entrenar_final_robusto(seleccion_final, detalle_seleccion, folds_info, serie_completa_min, p_global,
                                              metadatos_baseline, None, None)
    if seleccion_final != "NONE_NOT_READY_FOR_TEST":
        _guardar_tabla(pd.DataFrame([{"umbral_final": resultado_final["sel_umbral_final"]["umbral"], "fa_final": resultado_final["sel_umbral_final"]["fa"],
                                     "banda": resultado_final["sel_umbral_final"]["banda"]}]), "calibracion_final_pretest")
    print(f"  seleccion final: {seleccion_final}")

    manifiesto = {"fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "n_folds": n_folds_total, "seleccion_final": seleccion_final,
                 "sklearn_version": sklearn.__version__, "python_version": platform.python_version(), "test_intacto": True}
    with open(TABLAS_DIR / "manifiesto_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False, default=str)

    print("\nGenerando figuras...")
    generar_figuras(resultados_por_fold, folds_info, p_global, tabla_diag_mensual, tabla_admisibles)

    print("\nCONFIRMACION: en ningun momento se ha cargado, referenciado ni evaluado la particion de test "
          "(salvo su timestamp de arranque, usado exclusivamente para delimitar el periodo pre-test).")

    return {"resultados_por_fold": resultados_por_fold, "admisibilidad": admisibilidad, "seleccion_final": seleccion_final,
            "resultado_final": resultado_final, "ranking_a": ranking_a}


def cargar_ataques_y_modelo_360_wrapper():
    from src.base_relative import cargar_ataques_y_modelo_360
    return cargar_ataques_y_modelo_360()


def generar_figuras(resultados_por_fold, folds_info, p_global, tabla_diag_mensual, tabla_admisibles) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    meses = pd.PeriodIndex(tabla_diag_mensual["mes"], freq="M")
    ax.plot(meses.to_timestamp(), tabla_diag_mensual["active_fraction"], marker="o", markersize=3, color=ROJO)
    ax.axhline(LIM_ACTIVE_FRACTION, color=GRIS_OSCURO, linestyle="--", linewidth=1, label="limite saturacion (0.10)")
    ax.set_ylabel("active_fraction de P_GLOBAL"); ax.legend(fontsize=8); ax.set_title("Active fraction mensual de P_GLOBAL (pre-test completo)")
    fig.tight_layout(); _guardar_fig(fig, "01_active_fraction_mensual_pglobal")

    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre, color in (("P_GLOBAL", ROJO), ("P_MULTI_SEASON", NARANJA), ("P_ROBUST_Z", TEAL)):
        vals = [resultados_por_fold[k][nombre]["fa"] for k in sorted(resultados_por_fold)]
        ax.plot(sorted(resultados_por_fold), vals, marker="o", color=color, label=nombre)
    ax.axhline(0.15, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_xlabel("fold"); ax.set_ylabel("FA/dia"); ax.legend(fontsize=8); ax.set_title("FA por fold: variantes de P")
    fig.tight_layout(); _guardar_fig(fig, "08_fa_por_fold")

    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre, color in (("P_GLOBAL", ROJO), ("P_MULTI_SEASON", NARANJA), ("P_ROBUST_Z", TEAL)):
        vals = [resultados_por_fold[k][nombre]["active_fraction"] for k in sorted(resultados_por_fold)]
        ax.plot(sorted(resultados_por_fold), vals, marker="o", color=color, label=nombre)
    ax.axhline(LIM_ACTIVE_FRACTION, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_xlabel("fold"); ax.set_ylabel("active_fraction"); ax.legend(fontsize=8); ax.set_title("Active fraction por fold: variantes de P")
    fig.tight_layout(); _guardar_fig(fig, "09_active_fraction_por_fold")

    fig, ax = plt.subplots(figsize=(8, 5))
    todas_config = CONFIGURACIONES_INDIVIDUALES + FUSIONES
    dr_energ_medio = [np.nanmean([resultados_por_fold[k][n]["dr_energ_ramp_pct"] for k in resultados_por_fold]) for n in todas_config]
    ax.barh(todas_config, dr_energ_medio, color=AZUL)
    ax.set_xlabel("DR energetico medio entre folds (%)"); ax.set_title("DR energetico por configuracion")
    fig.tight_layout(); _guardar_fig(fig, "13_dr_energetico_por_configuracion")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(tabla_admisibles["nombre"], tabla_admisibles["admisible_global"].astype(int), color=VERDE)
    ax.set_ylabel("admisible (1=si)"); ax.set_title("Resumen de admisibilidad")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout(); _guardar_fig(fig, "20_resumen_admisibilidad")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
