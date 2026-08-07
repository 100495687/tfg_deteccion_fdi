"""Auditoria final de decision: P_MULTI_SEASON_ONLY frente a la fusion estable
OR_PMULTI_HMULTIWINDOW (P_MULTI_SEASON OR H_MULTIWINDOW_180), ambas ya congeladas en
calibracion_temporal_h.py. No se entrena ningun modelo con hiperparametros, features,
bandas de calibracion o datos distintos de los ya congelados; no se busca ningun umbral
nuevo; no se prueba ninguna ventana/fusion/regla nueva. Unico objetivo: aplicar de forma
literal la regla de seleccion de 14 criterios a dos configuraciones fijas.

Nota de reproduccion: el experimento anterior
(calibracion_temporal_h.py) no persistio en disco los modelos H por fold ni las tablas de
episodios individuales -- solo metricas agregadas por fold (CSV). Para poder reconstruir
los eventos "desde cero" y desglosar por episodio (duracion/severidad/energia/
retraso), esta auditoria reproduce esas puntuaciones llamando literalmente, sin modificar,
a calibracion_temporal_h.procesar_fold/evaluar_fold_completo (mismos hiperparametros
opt.CONFIG_HISTGB_PERIODIC_ACTUAL, mismas features PERIODIC_BASE_FEATURES, mismo
procedimiento de calibracion multiwindow, mismos folds/ataques/semillas). Antes de
comparar nada, se verifica que esta reproduccion coincide exactamente (dentro de
tolerancia) con los numeros ya publicados y guardados en
results/tables/calibracion_temporal_h/falsas_alarmas_por_fold.csv -- si no coincide, la
auditoria se detiene. Esto no es una recalibracion ni un nuevo entrenamiento: es la misma
funcion determinista ya validada, usada unicamente para poder inspeccionar el detalle por
episodio que no se guardo la primera vez.

Test permanece completamente cerrado (solo se lee su timestamp de arranque).

Ejecutable de forma aislada:
    python -m src.auditoria_final_p_vs_or
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats as sstats

from src import calibracion_temporal_h as cth
from src import fusion_p_histgb as fus
from src import optimizacion_histgb_periodic as opt
from src import robustez_temporal_p as rtp
from src.data_loading import cargar_y_limpiar
from src.experimento_stride import bootstrap_pareado, contar_eventos, mcnemar_p
from src.splitting import split_temporal

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "auditoria_final_p_vs_or"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "auditoria_final_p_vs_or"
MODELOS_DIR = BASE_DIR / "models" / "auditoria_final_p_vs_or"
CTH_DIR = BASE_DIR / "results" / "tables" / "calibracion_temporal_h"
RTP_DIR = BASE_DIR / "results" / "tables" / "robustez_temporal_p"
CTH_MODELOS_DIR = BASE_DIR / "models" / "calibracion_temporal_h"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

RESOLUCION_MIN = 15
P_ONLY, H_ONLY, OR_STABLE = "P_MULTI_SEASON_ONLY", "H_MULTIWINDOW_180_ONLY", "OR_PMULTI_HMULTIWINDOW"
CONFIGS = [P_ONLY, H_ONLY, OR_STABLE]
UMBRAL_P_FINAL = 0.049748

DURACIONES = [30, 60, 120, 240, 360, 720, 1440]


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 6) Reproduccion obligatoria (usando exclusivamente artefactos ya guardados en disco)
# ==========================================================================================

FA_ESPERADA = {
    P_ONLY: {0: 0.1111, 1: 0.0000, 2: 0.0000, 3: 0.0000},
    H_ONLY: {0: 0.0222, 1: 0.0222, 2: 0.0000, 3: 0.0444},
    OR_STABLE: {0: 0.1333, 1: 0.0222, 2: 0.0000, 3: 0.0444},
}


def reproducir_configuraciones() -> pd.DataFrame:
    fa_prev = pd.read_csv(CTH_DIR / "falsas_alarmas_por_fold.csv")
    sat_prev = pd.read_csv(CTH_DIR / "saturacion_por_fold.csv")
    piv = fa_prev.pivot(index="fold", columns="configuracion", values="fa")
    filas = []
    for cfg, esperado in FA_ESPERADA.items():
        for fold, v in esperado.items():
            obt = float(piv.loc[fold, cfg])
            filas.append({"config": cfg, "fold": fold, "fa_esperada": v, "fa_obtenida": obt,
                           "coincide": bool(np.isclose(obt, v, atol=0.02))})
    tabla = pd.DataFrame(filas)
    tabla.attrs["fa_ponderada"] = {cfg: float(np.mean(list(v.values()))) for cfg, v in FA_ESPERADA.items()}
    sat_sub = sat_prev[sat_prev["configuracion"].isin(CONFIGS)]
    tabla.attrs["sin_saturacion_confirmado"] = bool(not sat_sub["falla_saturacion"].any())
    return tabla


# ==========================================================================================
# 7-8) Datos, folds, linea temporal comun
# ==========================================================================================

def preparar_datos():
    df_min = cargar_y_limpiar()
    particiones = split_temporal(df_min)
    serie_completa_min = pd.concat([particiones["train"], particiones["val"]])
    inicio = particiones["train"].index[0]
    fin_pretest = particiones["test"].index[0]  # unico dato leido de test: su timestamp de arranque
    folds_info = opt.construir_folds(inicio, fin_pretest)
    from src.base_relative import cargar_ataques_y_modelo_360
    config = __import__("src.windowing", fromlist=["cargar_config"]).cargar_config()
    _, datos_p, modelo_p, _, _ = cargar_ataques_y_modelo_360()
    p_global = opt.cargar_p_global(modelo_p, datos_p["params_norm"])
    umbrales_p_multiseason = dict(zip(
        pd.read_csv(RTP_DIR / "umbrales_por_fold.csv")["fold"],
        pd.read_csv(RTP_DIR / "umbrales_por_fold.csv")["umbral_P_multiseason"]))
    return serie_completa_min, folds_info, fin_pretest, config, p_global, umbrales_p_multiseason


# ==========================================================================================
# 9) Reconstruccion de las 3 configuraciones por fold (reutiliza cth.procesar_fold/
#    evaluar_fold_completo literalmente, sin modificarlos) -- eventos reconstruidos desde
#    las secuencias booleanas en cth (event_start = active_t AND NOT active_{t-1})
# ==========================================================================================

def ejecutar_fold(fold_def: dict, serie_completa_min: pd.DataFrame, config: dict, p_global: dict,
                   umbral_p_multiseason: float) -> tuple[dict, dict]:
    fold_res = cth.procesar_fold(fold_def, serie_completa_min, config, p_global, umbral_p_multiseason)
    resultados_9 = cth.evaluar_fold_completo(fold_def, fold_res, p_global)
    resultados = {c: resultados_9[c] for c in CONFIGS}
    return fold_res, resultados


def activos_crudos_fold(fold_res: dict, p_global: dict) -> dict[str, np.ndarray]:
    """Devuelve, por configuracion, el array booleano 'activo' de la rama limpia (15 min),
    reutilizando literalmente rtp._activo_clean (misma logica que evaluar_configuracion_fold)."""
    target_end_master = fold_res["target_end_master"]
    h_score, h_umbral = fold_res["positive_valid"], fold_res["sel_mw"]["umbral"]
    activo_p = rtp._activo_clean(P_ONLY, fold_res["p_cfg"], None, None, p_global, target_end_master)
    activo_h = rtp._activo_clean(H_ONLY, None, h_score, h_umbral, p_global, target_end_master)
    activo_or = rtp._activo_clean(OR_STABLE, fold_res["p_cfg"], h_score, h_umbral, p_global, target_end_master)
    assert np.array_equal(activo_or, activo_p | activo_h), "OR_STABLE debe coincidir EXACTAMENTE con P | H (seccion 8)"
    return {P_ONLY: activo_p, H_ONLY: activo_h, OR_STABLE: activo_or, "ts": target_end_master}


# ==========================================================================================
# 15) Complementariedad P/H y 16) mejoras por fold
# ==========================================================================================

def complementariedad_fold(fold: int, res_p: pd.DataFrame, res_h: pd.DataFrame, res_or: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    dp = set(res_p[res_p["induced_detected"]]["episode_id"])
    dh = set(res_h[res_h["induced_detected"]]["episode_id"])
    dor = set(res_or[res_or["induced_detected"]]["episode_id"])
    solo_p, solo_h, ambos = dp - dh, dh - dp, dp & dh
    union_teorica = dp | dh
    perdidas_or = union_teorica - dor
    ganancias_artificiales = dor - union_teorica
    retencion_p = 100 * len(dp & dor) / len(dp) if dp else 100.0
    retencion_h = 100 * len(dh & dor) / len(dh) if dh else 100.0

    resumen = {
        "fold": fold, "n_solo_p": len(solo_p), "n_solo_h": len(solo_h), "n_ambos": len(ambos),
        "n_union_teorica": len(union_teorica), "n_or_operativo": len(dor),
        "n_perdidas_or_vs_union": len(perdidas_or), "n_ganancias_artificiales": len(ganancias_artificiales),
        "retencion_p_pct": retencion_p, "retencion_h_pct": retencion_h,
    }

    filas_excl_h = []
    idx_h = res_h.set_index("episode_id")
    for ep in solo_h:
        row = idx_h.loc[ep]
        bloqueado_por_p_clean = False  # solo_h => P no lo detecta; comprobamos si P.raw_detected==True (bloqueado por su propia alarma limpia previa)
        if ep in res_p["episode_id"].values:
            fila_p = res_p.set_index("episode_id").loc[ep]
            bloqueado_por_p_clean = bool(fila_p.get("raw_detected", False)) and not bool(fila_p.get("induced_detected", False))
        filas_excl_h.append({
            "fold": fold, "episode_id": ep, "duration_min": row.get("duration_min"), "rho_final": row.get("rho_final"),
            "hidden_energy_kwh": row.get("hidden_energy_kwh"), "alto_impacto": bool(row.get("alto_impacto", False)) if "alto_impacto" in row else None,
            "detection_delay_min": row.get("detection_delay_min"), "bloqueado_por_alarma_limpia_previa_de_P": bloqueado_por_p_clean,
        })
    return resumen, pd.DataFrame(filas_excl_h)


def deltas_fold(fold: int, r_p: dict, r_h: dict, r_or: dict) -> dict:
    d_dr = r_or["dr_ramp_pct"] - r_p["dr_ramp_pct"]
    d_energ = r_or["dr_energ_ramp_pct"] - r_p["dr_energ_ramp_pct"]
    d_hi = r_or["alto_impacto_n"] - r_p["alto_impacto_n"]
    d_gb = r_or["grupo_b_n"] - r_p["grupo_b_n"]
    d_delay = (r_or["retraso_mediano_min"] - r_p["retraso_mediano_min"]) if np.isfinite(r_or["retraso_mediano_min"]) and np.isfinite(r_p["retraso_mediano_min"]) else float("nan")
    return {
        "fold": fold, "delta_DR_pp": d_dr, "signo_DR": int(np.sign(d_dr)),
        "delta_energy_DR_pp": d_energ, "signo_energy_DR": int(np.sign(d_energ)),
        "delta_high_impact_n": d_hi, "delta_groupB_n": d_gb, "delta_delay_min": d_delay,
    }


# ==========================================================================================
# 11-14) Desgloses por duracion y severidad
# ==========================================================================================

def resultados_por_duracion(resultados_por_fold: dict) -> pd.DataFrame:
    filas = []
    for fold, res in resultados_por_fold.items():
        for cfg in CONFIGS:
            rr = res[cfg]["res_ramp"]
            for dur in DURACIONES:
                sub = rr[rr["duration_min"] == dur]
                if len(sub) == 0:
                    continue
                filas.append({"fold": fold, "config": cfg, "duration_min": dur, "n": len(sub),
                               "dr_pct": 100 * sub["induced_detected"].mean(),
                               "hidden_energy_total_kwh": float(sub["hidden_energy_kwh"].sum()),
                               "hidden_energy_detectada_kwh": float(sub.loc[sub["induced_detected"], "hidden_energy_kwh"].sum())})
    return pd.DataFrame(filas)


def resultados_por_severidad(resultados_por_fold: dict) -> pd.DataFrame:
    filas = []
    for fold, res in resultados_por_fold.items():
        rho_ref = res[P_ONLY]["res_ramp"][["episode_id", "rho_final"]].drop_duplicates("episode_id")
        if len(rho_ref) < 3:
            continue
        terciles = pd.qcut(rho_ref["rho_final"].rank(method="first"), 3, labels=["baja", "media", "alta"])
        mapa_tercil = dict(zip(rho_ref["episode_id"], terciles))
        for cfg in CONFIGS:
            rr = res[cfg]["res_ramp"].copy()
            rr["severidad"] = rr["episode_id"].map(mapa_tercil)
            for terc, sub in rr.groupby("severidad", observed=True):
                if len(sub) == 0:
                    continue
                filas.append({"fold": fold, "config": cfg, "severidad": terc, "n": len(sub),
                               "dr_pct": 100 * sub["induced_detected"].mean()})
    return pd.DataFrame(filas)


# ==========================================================================================
# 18) Admisibilidad (sin el limite de concentracion, que es un criterio
#     aparte, el 11, de la regla de 14 puntos)
# ==========================================================================================

def evaluar_admisibilidad_18(nombre: str, filas_config: list[dict], n_folds_total: int) -> dict:
    completo = len(filas_config) == n_folds_total
    admisible_por_fold = bool(all(f["fa"] <= 0.15 and not f["falla_saturacion"] and np.isfinite(f["fa"]) for f in filas_config)) if completo else False
    fa_media_pond = float(np.mean([f["fa"] for f in filas_config])) if filas_config else float("nan")
    sin_saturacion = bool(not any(f["falla_saturacion"] for f in filas_config))
    admisible_global = bool(completo and fa_media_pond <= 0.12 and admisible_por_fold and sin_saturacion)
    return {"nombre": nombre, "completo": completo, "admisible_por_fold": admisible_por_fold,
            "fa_media_ponderada": fa_media_pond, "sin_saturacion": sin_saturacion, "admisible_global": admisible_global}


# ==========================================================================================
# 19) Regla literal de seleccion de la fusion (14 criterios)
# ==========================================================================================

def _dr_energ_agregado(filas: list[dict], res_ramp_por_fold: dict[int, pd.DataFrame]) -> float:
    total_hidden = sum(float(res_ramp_por_fold[f["fold"]]["hidden_energy_kwh"].sum()) for f in filas)
    detectada = sum(float(res_ramp_por_fold[f["fold"]].loc[res_ramp_por_fold[f["fold"]]["induced_detected"], "hidden_energy_kwh"].sum()) for f in filas)
    return 100 * detectada / total_hidden if total_hidden > 1e-9 else float("nan")


def _dr_agregado(res_ramp_por_fold: dict[int, pd.DataFrame]) -> float:
    n_tot = sum(len(df) for df in res_ramp_por_fold.values())
    n_det = sum(int(df["induced_detected"].sum()) for df in res_ramp_por_fold.values())
    return 100 * n_det / n_tot if n_tot else float("nan")


def _retencion_global(res_ref_por_fold: dict[int, pd.DataFrame], res_or_por_fold: dict[int, pd.DataFrame]) -> float:
    det_ref = sum(int(df["induced_detected"].sum()) for df in res_ref_por_fold.values())
    if det_ref == 0:
        return 100.0
    det_ambos = 0
    for f, df_ref in res_ref_por_fold.items():
        ids_ref = set(df_ref[df_ref["induced_detected"]]["episode_id"])
        ids_or = set(res_or_por_fold[f][res_or_por_fold[f]["induced_detected"]]["episode_id"])
        det_ambos += len(ids_ref & ids_or)
    return 100 * det_ambos / det_ref


def evaluar_regla_fusion(filas_or: list[dict], filas_p: list[dict], filas_h: list[dict], adm_or: dict,
                          res_ramp_p: dict[int, pd.DataFrame], res_ramp_h: dict[int, pd.DataFrame], res_ramp_or: dict[int, pd.DataFrame],
                          exclusivas_h_por_fold: list[int], deltas: list[dict],
                          filas_const_p: list[dict], filas_const_or: list[dict],
                          exclusivas_h_detalle: pd.DataFrame) -> dict:
    # Los 14 criterios se numeran exactamente en el mismo orden en que aparecen aqui.
    # "Retencion de H por OR" no es uno de los 14 criterios oficiales (solo se calcula como dato
    # informativo, ver retencion_h_global en `detalle`) -- no ocupa ningun puesto de la regla.
    c1 = bool(adm_or["admisible_global"])  # 1) globalmente admisible
    c2 = bool(not any(f["falla_saturacion"] for f in filas_or))  # 2) cero folds saturados

    dr_energ_or_agg = _dr_energ_agregado(filas_or, res_ramp_or)
    dr_energ_p_agg = _dr_energ_agregado(filas_p, res_ramp_p)
    mejora_energ_pp = dr_energ_or_agg - dr_energ_p_agg
    c3 = bool(mejora_energ_pp >= 3.0)  # 3) mejora DR energetico agregado >= 3pp

    dr_or_agg = _dr_agregado(res_ramp_or)
    dr_p_agg = _dr_agregado(res_ramp_p)
    mejora_dr_pp = dr_or_agg - dr_p_agg
    c4 = bool(mejora_dr_pp >= 2.0)  # 4) mejora DR global agregado >= 2pp

    retencion_p_global = _retencion_global(res_ramp_p, res_ramp_or)
    c5 = bool(retencion_p_global >= 90.0)  # 5) conserva >=90% de las detecciones de P

    hi_p, hi_or = sum(f["alto_impacto_n"] for f in filas_p), sum(f["alto_impacto_n"] for f in filas_or)
    c6 = bool(hi_or >= 0.90 * hi_p) if hi_p > 0 else True  # 6) conserva >=90% del alto impacto de P

    n_folds_con_exclusivas_h = sum(1 for e in exclusivas_h_por_fold if e > 0)
    c7 = bool(n_folds_con_exclusivas_h >= 2)  # 7) H aporta exclusivas en >=2 folds

    n_signo_pos_dr = sum(1 for d in deltas if d["signo_DR"] > 0)
    c8 = bool(n_signo_pos_dr >= 3)  # 8) mejora DR signo positivo en >=3/4 folds
    n_signo_pos_energ = sum(1 for d in deltas if d["signo_energy_DR"] > 0)
    c9 = bool(n_signo_pos_energ >= 3)  # 9) mejora DR energ signo positivo en >=3/4 folds

    peor_delta_dr = min(d["delta_DR_pp"] for d in deltas)
    c10 = bool(peor_delta_dr >= -1.0)  # 10) ningun fold pierde >1pp de DR respecto a P

    total_excl_h = sum(exclusivas_h_por_fold)
    concentracion_h = 100 * max(exclusivas_h_por_fold) / total_excl_h if total_excl_h > 0 else 0.0
    c11 = bool(total_excl_h == 0 or concentracion_h <= 85.0)  # 11) exclusivas de H no concentradas >85% en 1 fold

    delays_or = pd.concat([df.loc[df["induced_detected"], "detection_delay_min"] for df in res_ramp_or.values()])
    delays_p = pd.concat([df.loc[df["induced_detected"], "detection_delay_min"] for df in res_ramp_p.values()])
    retraso_or_agg = float(delays_or.median()) if len(delays_or) else float("nan")
    retraso_p_agg = float(delays_p.median()) if len(delays_p) else float("nan")
    empeora_retraso = (retraso_or_agg - retraso_p_agg) if np.isfinite(retraso_or_agg) and np.isfinite(retraso_p_agg) else 0.0
    c12 = bool(not np.isfinite(empeora_retraso) or empeora_retraso <= 60.0)  # 12) retraso mediano agregado no empeora >60min

    gb_p = sum(f["dr_energ_const_pct"] for f in filas_const_p if np.isfinite(f["dr_energ_const_pct"])) / max(1, sum(1 for f in filas_const_p if np.isfinite(f["dr_energ_const_pct"])))
    gb_or = sum(f["dr_energ_const_pct"] for f in filas_const_or if np.isfinite(f["dr_energ_const_pct"])) / max(1, sum(1 for f in filas_const_or if np.isfinite(f["dr_energ_const_pct"])))
    c13 = bool((gb_or - gb_p) >= -5.0)  # 13) no empeora de forma material Grupo B (tolerancia operativa de 5pp)

    n_duraciones_excl_h = exclusivas_h_detalle["duration_min"].nunique() if len(exclusivas_h_detalle) else 0
    c14 = bool(n_duraciones_excl_h >= 2)  # 14) la mejora no depende unicamente de una duracion (>=2 duraciones distintas entre las exclusivas de H)

    retencion_h_global = _retencion_global(res_ramp_h, res_ramp_or)  # informativo, no es uno de los 14 criterios

    criterios = {1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 6: c6, 7: c7, 8: c8, 9: c9, 10: c10,
                 11: c11, 12: c12, 13: c13, 14: c14}
    detalle = {
        "criterios": criterios, "cumple_todos": bool(all(criterios.values())),
        "dr_energ_or_agregado_pct": dr_energ_or_agg, "dr_energ_p_agregado_pct": dr_energ_p_agg, "mejora_energ_pp": mejora_energ_pp,
        "dr_or_agregado_pct": dr_or_agg, "dr_p_agregado_pct": dr_p_agg, "mejora_dr_pp": mejora_dr_pp,
        "alto_impacto_p": hi_p, "alto_impacto_or": hi_or,
        "retencion_p_global_pct": retencion_p_global, "retencion_h_global_pct": retencion_h_global,
        "n_folds_con_exclusivas_h": n_folds_con_exclusivas_h, "n_folds_signo_positivo_dr": n_signo_pos_dr,
        "n_folds_signo_positivo_energ": n_signo_pos_energ, "peor_delta_dr_pp": peor_delta_dr,
        "concentracion_exclusivas_h_max_pct": concentracion_h, "retraso_or_agregado_min": retraso_or_agg,
        "retraso_p_agregado_min": retraso_p_agg, "empeora_retraso_min": empeora_retraso,
        "grupo_b_dr_energ_p_medio_pct": gb_p, "grupo_b_dr_energ_or_medio_pct": gb_or,
        "n_duraciones_distintas_en_exclusivas_h": n_duraciones_excl_h,
    }
    return detalle


# ==========================================================================================
# 20) Regla de decision final
# ==========================================================================================

def decidir(detalle_regla_a: dict, adm_p: dict, filas_p: list[dict]) -> str:
    if detalle_regla_a["cumple_todos"]:
        return "OR_PMULTI_HMULTIWINDOW"
    p_sigue_valido = bool(adm_p["admisible_global"] and not any(f["falla_saturacion"] for f in filas_p))
    if p_sigue_valido:
        return "P_MULTI_SEASON_ONLY"
    return "NONE_NOT_READY_FOR_TEST"


# ==========================================================================================
# 21) Comparaciones estadisticas adicionales (FA por dia, active_fraction por bloques de 7 dias)
# ==========================================================================================

def _eventos_por_dia(activo: np.ndarray, ts: np.ndarray) -> pd.Series:
    event_start = activo & ~np.r_[False, activo[:-1]]
    dias = pd.DatetimeIndex(ts).date
    return pd.Series(event_start.astype(int), index=dias).groupby(level=0).sum()


def bootstrap_fa_por_dia(activo_a: np.ndarray, ts_a: np.ndarray, activo_b: np.ndarray, ts_b: np.ndarray,
                          n_boot: int = 3000, seed: int = 42) -> tuple[float, float, float]:
    ea, eb = _eventos_por_dia(activo_a, ts_a), _eventos_por_dia(activo_b, ts_b)
    idx_comun = ea.index.union(eb.index)
    ea, eb = ea.reindex(idx_comun, fill_value=0).to_numpy(), eb.reindex(idx_comun, fill_value=0).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(ea)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = eb[idx].mean() - ea[idx].mean()
    return float(eb.mean() - ea.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _active_fraction_por_bloque(activo: np.ndarray, dias_por_bloque: int = 7, resolucion_min: int = RESOLUCION_MIN) -> np.ndarray:
    pasos_por_bloque = dias_por_bloque * 24 * 60 // resolucion_min
    n_bloques = len(activo) // pasos_por_bloque
    if n_bloques == 0:
        return np.array([activo.mean()]) if len(activo) else np.array([])
    return activo[:n_bloques * pasos_por_bloque].reshape(n_bloques, pasos_por_bloque).mean(axis=1)


def bootstrap_active_fraction_7dias(activo_a: np.ndarray, activo_b: np.ndarray, n_boot: int = 3000, seed: int = 42) -> tuple[float, float, float]:
    ba, bb = _active_fraction_por_bloque(activo_a), _active_fraction_por_bloque(activo_b)
    n = min(len(ba), len(bb))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ba, bb = ba[:n], bb[:n]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = bb[idx].mean() - ba[idx].mean()
    return float(bb.mean() - ba.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ==========================================================================================
# 22) Congelacion final (no recalibra nada -- solo escribe la decision ya tomada)
# ==========================================================================================

def congelar_decision(seleccion: str, detalle_regla_a: dict, umbrales_h_por_fold: dict, metadatos_previos: dict) -> dict:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    criterios_ok = [i for i, v in detalle_regla_a["criterios"].items() if v] if detalle_regla_a else []
    criterios_fail = [i for i, v in detalle_regla_a["criterios"].items() if not v] if detalle_regla_a else []
    payload = {
        "selected_configuration": seleccion,
        "motivo": ("cumple literalmente los 14 criterios de la seccion 19" if seleccion == OR_STABLE else
                   "la fusion incumple al menos uno de los 14 criterios; P_MULTI_SEASON sigue siendo admisible" if seleccion == P_ONLY else
                   "ni la fusion ni P_MULTI_SEASON son admisibles"),
        "criterios_cumplidos": criterios_ok, "criterios_incumplidos": criterios_fail,
        "umbral_final_p": UMBRAL_P_FINAL, "procedimiento_p": "P_MULTI_SEASON",
        "modelo_h": "HistGB_PERIODIC (hiperparametros congelados)" if seleccion == OR_STABLE else None,
        "procedimiento_h": "H_MULTIWINDOW_180" if seleccion == OR_STABLE else None,
        "umbral_final_h_por_fold": umbrales_h_por_fold if seleccion == OR_STABLE else None,
        "features_h": opt.PERIODIC_BASE_FEATURES if seleccion == OR_STABLE else None,
        "hiperparametros_h": opt.CONFIG_HISTGB_PERIODIC_ACTUAL if seleccion == OR_STABLE else None,
        "forward_fill_p": "causal, np.searchsorted backward, resolucion 15min, sin backfill",
        "regla_or": "active_P_MULTI_SEASON_ffill OR active_H_MULTIWINDOW_180" if seleccion == OR_STABLE else "no aplica",
        "limites_saturacion": {"active_fraction": rtp.LIM_ACTIVE_FRACTION, "longest_clean_event_hours": rtp.LIM_LONGEST_CLEAN_HOURS,
                               "seven_day_max_active_fraction": rtp.LIM_7DAY_MAX_ACTIVE, "consecutive_days_over_50": rtp.LIM_CONSECUTIVE_DAYS_50},
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(),
        "metadatos_previos": metadatos_previos, "modelos_reutilizados_sin_reentrenar": True,
        "no_se_recalibro_ningun_umbral_nuevo": True, "test_intacto": True,
    }
    with open(MODELOS_DIR / "configuracion_final_decision_pretest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    with open(TABLAS_DIR / "configuracion_final_decision_pretest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return payload


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("Fase 0: reproduciendo P_MULTI_SEASON_ONLY / H_MULTIWINDOW_180_ONLY / OR_PMULTI_HMULTIWINDOW...")
    tabla_repro = reproducir_configuraciones()
    _guardar_tabla(tabla_repro, "reproduccion_configuraciones")
    if not tabla_repro["coincide"].all():
        raise RuntimeError("no se reproducen las configuraciones congeladas dentro de tolerancia -- auditoria detenida (seccion 6)")
    if not tabla_repro.attrs["sin_saturacion_confirmado"]:
        raise RuntimeError("saturacion detectada en artefactos previos -- auditoria detenida (seccion 6)")
    print("OK -- reproducido dentro de tolerancia. FA ponderada esperada:", tabla_repro.attrs["fa_ponderada"])

    serie_completa_min, folds_info, fin_pretest, config, p_global, umbrales_p_multiseason = preparar_datos()
    _guardar_tabla(pd.DataFrame(folds_info["folds"]), "definicion_folds")

    filas_cfg_congeladas = [
        {"config": P_ONLY, "score": "relative_kw (TCN-AE congelado)", "umbral_final_pretest": UMBRAL_P_FINAL, "recalibrado_en_esta_auditoria": False},
        {"config": H_ONLY, "score": "causal_relative_kw (HistGB_PERIODIC)", "umbral_final_pretest": "ya congelado en calibracion_temporal_h.py", "recalibrado_en_esta_auditoria": False},
        {"config": OR_STABLE, "score": "P OR H", "umbral_final_pretest": "hereda P y H", "recalibrado_en_esta_auditoria": False},
    ]
    _guardar_tabla(pd.DataFrame(filas_cfg_congeladas), "configuraciones_congeladas")

    resultados_por_fold: dict[int, dict] = {}
    fold_res_por_fold: dict[int, dict] = {}
    activos_por_fold: dict[int, dict] = {}
    for fold_def in folds_info["folds"]:
        k = fold_def["fold"]
        print(f"\n=== Fold {k}: valid {fold_def['calib_fin'].date()} -> {fold_def['valid_fin'].date()} ===")
        fold_res, resultados = ejecutar_fold(fold_def, serie_completa_min, config, p_global, float(umbrales_p_multiseason[k]))
        fold_res_por_fold[k] = fold_res
        resultados_por_fold[k] = resultados
        activos_por_fold[k] = activos_crudos_fold(fold_res, p_global)
        for cfg in CONFIGS:
            r = resultados[cfg]
            print(f"  {cfg:24s} FA={r['fa']:.4f} act_frac={r['active_fraction']:.3f} sat={'SI' if r['falla_saturacion'] else 'no'} "
                  f"DR={r['dr_ramp_pct']:.1f}% DR_energ={r['dr_energ_ramp_pct']:.1f}% HI={r['alto_impacto_n']}/{r['alto_impacto_total']}")
            fa_esp = FA_ESPERADA[cfg][k]
            if not np.isclose(r["fa"], fa_esp, atol=0.02):
                raise RuntimeError(f"reproduccion por fold NO coincide para {cfg} fold {k}: esperado {fa_esp}, obtenido {r['fa']} -- auditoria detenida")

    print("\nReproduccion por fold verificada (coincide con los artefactos guardados) -- se continua con la auditoria.")

    filas_fa, filas_sat, filas_ramp, filas_energia, filas_hi, filas_const, filas_gb = [], [], [], [], [], [], []
    for k, resultados in resultados_por_fold.items():
        for cfg in CONFIGS:
            r = resultados[cfg]
            filas_fa.append({"configuracion": cfg, "fold": k, "fa": r["fa"], "admisible_fold": bool(r["fa"] <= 0.15 and not r["falla_saturacion"])})
            filas_sat.append({"configuracion": cfg, "fold": k, "active_fraction": r["active_fraction"], "longest_clean_event_hours": r["longest_clean_event_hours"],
                              "seven_day_max_active_fraction": r["seven_day_max_active_fraction"], "consecutive_days_over_50": r["consecutive_days_over_50"], "falla_saturacion": r["falla_saturacion"]})
            filas_ramp.append({"configuracion": cfg, "fold": k, "dr_ramp_pct": r["dr_ramp_pct"], "dr_720_1440_pct": r["dr_720_1440_pct"], "dr_30_120_pct": r["dr_30_120_pct"]})
            filas_energia.append({"configuracion": cfg, "fold": k, "dr_energ_ramp_pct": r["dr_energ_ramp_pct"], "timely_energy_dr_pct": r["timely_energy_dr_pct"]})
            filas_hi.append({"configuracion": cfg, "fold": k, "alto_impacto_n": r["alto_impacto_n"], "alto_impacto_total": r["alto_impacto_total"]})
            filas_const.append({"configuracion": cfg, "fold": k, "dr_const_pct": r["dr_const_pct"], "dr_energ_const_pct": r["dr_energ_const_pct"]})
            filas_gb.append({"configuracion": cfg, "fold": k, "grupo_b_n": r["grupo_b_n"], "grupo_b_total": r["grupo_b_total"]})
    _guardar_tabla(pd.DataFrame(filas_fa), "falsas_alarmas_por_fold")
    _guardar_tabla(pd.DataFrame(filas_sat), "saturacion_por_fold")
    _guardar_tabla(pd.DataFrame(filas_ramp), "resultados_ramp_por_fold")
    _guardar_tabla(pd.DataFrame(filas_energia), "resultados_energia_por_fold")
    _guardar_tabla(pd.DataFrame(filas_hi), "resultados_alto_impacto")
    _guardar_tabla(pd.DataFrame(filas_const), "resultados_constantes")
    _guardar_tabla(pd.DataFrame(filas_gb), "resultados_grupo_b")
    _guardar_tabla(resultados_por_duracion(resultados_por_fold), "resultados_por_duracion")
    _guardar_tabla(resultados_por_severidad(resultados_por_fold), "resultados_por_severidad")

    print("\nComplementariedad P/H por fold...")
    filas_comp, filas_excl_h = [], []
    res_ramp_p = {k: resultados_por_fold[k][P_ONLY]["res_ramp"] for k in resultados_por_fold}
    res_ramp_h = {k: resultados_por_fold[k][H_ONLY]["res_ramp"] for k in resultados_por_fold}
    res_ramp_or = {k: resultados_por_fold[k][OR_STABLE]["res_ramp"] for k in resultados_por_fold}
    for k in resultados_por_fold:
        resumen, det = complementariedad_fold(k, res_ramp_p[k], res_ramp_h[k], res_ramp_or[k])
        filas_comp.append(resumen)
        filas_excl_h.append(det)
    tabla_comp = _guardar_tabla(pd.DataFrame(filas_comp), "complementariedad_p_h")
    tabla_excl_h = _guardar_tabla(pd.concat(filas_excl_h, ignore_index=True) if filas_excl_h else pd.DataFrame(), "exclusivas_h_por_fold")
    _guardar_tabla(tabla_comp[["fold", "n_union_teorica", "n_or_operativo", "n_perdidas_or_vs_union", "n_ganancias_artificiales"]], "union_teorica_vs_or")

    print("Mejoras por fold (OR vs P)...")
    filas_deltas = []
    for k in resultados_por_fold:
        filas_deltas.append(deltas_fold(k, resultados_por_fold[k][P_ONLY], resultados_por_fold[k][H_ONLY], resultados_por_fold[k][OR_STABLE]))
    _guardar_tabla(pd.DataFrame(filas_deltas), "mejoras_por_fold")

    print("\nAdmisibilidad (seccion 18)...")
    admisibilidad = {}
    filas_por_cfg = {}
    for cfg in CONFIGS:
        filas_cfg = [{k2: v2 for k2, v2 in resultados_por_fold[k][cfg].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
        filas_por_cfg[cfg] = filas_cfg
        admisibilidad[cfg] = evaluar_admisibilidad_18(cfg, filas_cfg, len(folds_info["folds"]))
        print(f"  {cfg:24s} admisible_global={admisibilidad[cfg]['admisible_global']} FA_pond={admisibilidad[cfg]['fa_media_ponderada']:.4f}")

    print("\nComparaciones estadisticas (seccion 21)...")
    tabla_pvsor = rtp.comparar_pareado("P_MULTI_SEASON_ONLY", "OR_PMULTI_HMULTIWINDOW", res_ramp_p, res_ramp_or)
    tabla_hvsor = rtp.comparar_pareado("H_MULTIWINDOW_180_ONLY", "OR_PMULTI_HMULTIWINDOW", res_ramp_h, res_ramp_or)
    filas_fa_boot, filas_af_boot = [], []
    for k in resultados_por_fold:
        act = activos_por_fold[k]
        diff_fa, lo_fa, hi_fa = bootstrap_fa_por_dia(act[P_ONLY], act["ts"], act[OR_STABLE], act["ts"])
        filas_fa_boot.append({"fold": k, "comparacion": "P_vs_OR", "diff_fa_dia": diff_fa, "ci95_low": lo_fa, "ci95_high": hi_fa})
        diff_af, lo_af, hi_af = bootstrap_active_fraction_7dias(act[P_ONLY], act[OR_STABLE])
        filas_af_boot.append({"fold": k, "comparacion": "P_vs_OR", "diff_active_fraction": diff_af, "ci95_low": lo_af, "ci95_high": hi_af})
    tabla_fa_boot = pd.DataFrame(filas_fa_boot)
    tabla_af_boot = pd.DataFrame(filas_af_boot)
    tabla_stats = pd.concat([
        tabla_pvsor.assign(tipo="mcnemar_bootstrap_wilcoxon"), tabla_hvsor.assign(tipo="mcnemar_bootstrap_wilcoxon"),
        tabla_fa_boot.assign(tipo="bootstrap_fa_por_dia"), tabla_af_boot.assign(tipo="bootstrap_active_fraction_7dias"),
    ], ignore_index=True)
    _guardar_tabla(tabla_stats, "comparaciones_estadisticas")

    print("\nAplicando la regla literal de 14 criterios (seccion 19) a OR_PMULTI_HMULTIWINDOW...")
    exclusivas_h_por_fold = [int((tabla_comp[tabla_comp["fold"] == k]["n_solo_h"]).iloc[0]) for k in resultados_por_fold]
    detalle_regla_a = evaluar_regla_fusion(
        filas_por_cfg[OR_STABLE], filas_por_cfg[P_ONLY], filas_por_cfg[H_ONLY], admisibilidad[OR_STABLE],
        res_ramp_p, res_ramp_h, res_ramp_or, exclusivas_h_por_fold, filas_deltas,
        filas_por_cfg[P_ONLY], filas_por_cfg[OR_STABLE], tabla_excl_h)
    tabla_criterios = pd.DataFrame([{"criterio": i, "cumple": v} for i, v in detalle_regla_a["criterios"].items()])
    _guardar_tabla(tabla_criterios, "cumplimiento_regla_fusion")
    for i, v in detalle_regla_a["criterios"].items():
        print(f"  criterio {i:2d}: {'CUMPLE' if v else 'NO CUMPLE'}")

    seleccion_final = decidir(detalle_regla_a, admisibilidad[P_ONLY], filas_por_cfg[P_ONLY])
    print(f"\n>>> DECISION FINAL: {seleccion_final}")
    _guardar_tabla(pd.DataFrame([{"selected_configuration": seleccion_final, "cumple_los_14_criterios": detalle_regla_a["cumple_todos"]}]), "configuracion_seleccionada")

    tabla_principal = construir_tabla_principal(resultados_por_fold, admisibilidad, tabla_comp, filas_deltas)
    _guardar_tabla(tabla_principal, "tabla_principal")

    umbrales_h_por_fold = {int(k): float(fold_res_por_fold[k]["sel_mw"]["umbral"]) for k in fold_res_por_fold}
    _, metadatos_previos = opt.reproducir_baseline()
    payload_final = congelar_decision(seleccion_final, detalle_regla_a, umbrales_h_por_fold, metadatos_previos)
    print(f"  artefacto final escrito: selected_configuration={payload_final['selected_configuration']}")

    manifiesto = {"fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "n_folds": len(folds_info["folds"]),
                 "seleccion_final": seleccion_final, "sklearn_version": sklearn.__version__,
                 "python_version": platform.python_version(), "test_intacto": True}
    with open(TABLAS_DIR / "manifiesto_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False, default=str)

    print("\nGenerando figuras...")
    generar_figuras(resultados_por_fold, filas_deltas, tabla_criterios, tabla_comp, admisibilidad)

    print("\nCONFIRMACION: en ningun momento se ha cargado, referenciado ni evaluado la particion de test "
          "(salvo su timestamp de arranque).")
    return {"resultados_por_fold": resultados_por_fold, "admisibilidad": admisibilidad, "detalle_regla_a": detalle_regla_a,
            "seleccion_final": seleccion_final, "tabla_principal": tabla_principal}


def construir_tabla_principal(resultados_por_fold, admisibilidad, tabla_comp, filas_deltas) -> pd.DataFrame:
    filas = []
    for cfg in CONFIGS:
        filas_cfg = [{k2: v2 for k2, v2 in resultados_por_fold[k][cfg].items() if k2 not in ("res_ramp", "res_const")} for k in resultados_por_fold]
        fa_pond = float(np.mean([f["fa"] for f in filas_cfg]))
        peor_fa = max(f["fa"] for f in filas_cfg)
        af_pond = float(np.mean([f["active_fraction"] for f in filas_cfg]))
        peor_af = max(f["active_fraction"] for f in filas_cfg)
        dur_max = max(f["longest_clean_event_hours"] for f in filas_cfg)
        peor_7d = max(f["seven_day_max_active_fraction"] for f in filas_cfg)
        n_sat = sum(1 for f in filas_cfg if f["falla_saturacion"])
        dr_ramp = float(np.mean([f["dr_ramp_pct"] for f in filas_cfg]))
        dr_energ = float(np.mean([f["dr_energ_ramp_pct"] for f in filas_cfg]))
        timely = float(np.nanmean([f["timely_energy_dr_pct"] for f in filas_cfg]))
        hi_n, hi_tot = sum(f["alto_impacto_n"] for f in filas_cfg), sum(f["alto_impacto_total"] for f in filas_cfg)
        dr_30_120 = float(np.nanmean([f["dr_30_120_pct"] for f in filas_cfg]))
        dr_720_1440 = float(np.nanmean([f["dr_720_1440_pct"] for f in filas_cfg]))
        dr_const = float(np.nanmean([f["dr_const_pct"] for f in filas_cfg]))
        dr_energ_const = float(np.nanmean([f["dr_energ_const_pct"] for f in filas_cfg]))
        gb_n, gb_tot = sum(f["grupo_b_n"] for f in filas_cfg), sum(f["grupo_b_total"] for f in filas_cfg)
        retraso = float(np.nanmedian([f["retraso_mediano_min"] for f in filas_cfg]))

        if cfg == P_ONLY:
            excl = int(tabla_comp["n_solo_p"].sum()); ret_p, ret_h = np.nan, np.nan
        elif cfg == H_ONLY:
            excl = int(tabla_comp["n_solo_h"].sum()); ret_p, ret_h = np.nan, np.nan
        else:
            excl = int(tabla_comp["n_ganancias_artificiales"].sum())
            ret_p = float(np.mean(tabla_comp["retencion_p_pct"])); ret_h = float(np.mean(tabla_comp["retencion_h_pct"]))

        folds_mejora_dr = sum(1 for d in filas_deltas if d["signo_DR"] > 0) if cfg == OR_STABLE else np.nan
        folds_mejora_energ = sum(1 for d in filas_deltas if d["signo_energy_DR"] > 0) if cfg == OR_STABLE else np.nan

        filas.append({
            "configuracion": cfg, "fa_ponderada": fa_pond, "peor_fa_fold": peor_fa, "active_fraction_ponderada": af_pond,
            "peor_active_fraction": peor_af, "duracion_maxima_h": dur_max, "peor_max_7dias": peor_7d, "folds_saturados": n_sat,
            "dr_ramp_pct": dr_ramp, "dr_energ_pct": dr_energ, "timely_energy_dr_pct": timely, "alto_impacto": f"{hi_n}/{hi_tot}",
            "dr_30_120_pct": dr_30_120, "dr_720_1440_pct": dr_720_1440, "dr_const_pct": dr_const, "dr_energ_const_pct": dr_energ_const,
            "grupo_b": f"{gb_n}/{gb_tot}", "retraso_mediano_min": retraso, "detecciones_exclusivas": excl,
            "retencion_p_pct": ret_p, "retencion_h_pct": ret_h, "folds_con_mejora_dr": folds_mejora_dr,
            "folds_con_mejora_energ": folds_mejora_energ, "admisible_global": admisibilidad[cfg]["admisible_global"],
        })
    return pd.DataFrame(filas)


def generar_figuras(resultados_por_fold, filas_deltas, tabla_criterios, tabla_comp, admisibilidad) -> None:
    folds = sorted(resultados_por_fold)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, color in zip(CONFIGS, [AZUL, NARANJA, VERDE]):
        ax.plot(folds, [resultados_por_fold[k][cfg]["fa"] for k in folds], marker="o", label=cfg, color=color)
    ax.axhline(0.15, color=ROJO, linestyle="--", linewidth=1)
    ax.set_xlabel("fold"); ax.set_ylabel("FA/dia"); ax.legend(fontsize=8); ax.set_title("FA por fold")
    fig.tight_layout(); _guardar_fig(fig, "01_fa_por_fold")

    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, color in zip(CONFIGS, [AZUL, NARANJA, VERDE]):
        ax.plot(folds, [resultados_por_fold[k][cfg]["active_fraction"] for k in folds], marker="o", label=cfg, color=color)
    ax.set_xlabel("fold"); ax.set_ylabel("active_fraction"); ax.legend(fontsize=8); ax.set_title("Active fraction por fold")
    fig.tight_layout(); _guardar_fig(fig, "02_active_fraction_por_fold")

    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, color in zip(CONFIGS, [AZUL, NARANJA, VERDE]):
        ax.plot(folds, [resultados_por_fold[k][cfg]["dr_ramp_pct"] for k in folds], marker="o", label=cfg, color=color)
    ax.set_xlabel("fold"); ax.set_ylabel("DR global (%)"); ax.legend(fontsize=8); ax.set_title("DR global por fold")
    fig.tight_layout(); _guardar_fig(fig, "03_dr_global_por_fold")

    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, color in zip(CONFIGS, [AZUL, NARANJA, VERDE]):
        ax.plot(folds, [resultados_por_fold[k][cfg]["dr_energ_ramp_pct"] for k in folds], marker="o", label=cfg, color=color)
    ax.set_xlabel("fold"); ax.set_ylabel("DR energetico (%)"); ax.legend(fontsize=8); ax.set_title("DR energetico por fold")
    fig.tight_layout(); _guardar_fig(fig, "04_dr_energetico_por_fold")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(d["fold"]) for d in filas_deltas], [d["delta_DR_pp"] for d in filas_deltas], color=AZUL)
    ax.axhline(0, color=GRIS_OSCURO, linewidth=1)
    ax.set_xlabel("fold"); ax.set_ylabel("Delta DR (OR - P), pp"); ax.set_title("Delta DR global OR menos P")
    fig.tight_layout(); _guardar_fig(fig, "05_delta_dr_or_menos_p")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(d["fold"]) for d in filas_deltas], [d["delta_energy_DR_pp"] for d in filas_deltas], color=NARANJA)
    ax.axhline(0, color=GRIS_OSCURO, linewidth=1)
    ax.set_xlabel("fold"); ax.set_ylabel("Delta DR energetico (OR - P), pp"); ax.set_title("Delta DR energetico OR menos P")
    fig.tight_layout(); _guardar_fig(fig, "06_delta_dr_energetico_or_menos_p")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(folds)); w = 0.25
    for i, cfg in enumerate(CONFIGS):
        vals = [resultados_por_fold[k][cfg]["alto_impacto_n"] for k in folds]
        ax.bar(x + i * w, vals, width=w, label=cfg)
    ax.set_xticks(x + w); ax.set_xticklabels([str(k) for k in folds]); ax.legend(fontsize=7); ax.set_title("Alto impacto detectado por fold")
    fig.tight_layout(); _guardar_fig(fig, "07_alto_impacto")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["P_solo_exclusivas", "H_solo_exclusivas", "ambos"], [tabla_comp["n_solo_p"].sum(), tabla_comp["n_solo_h"].sum(), tabla_comp["n_ambos"].sum()], color=[AZUL, NARANJA, VERDE])
    ax.set_ylabel("n episodios (suma de folds)"); ax.set_title("Complementariedad P/H (ramp)")
    fig.tight_layout(); _guardar_fig(fig, "11_exclusivas_h_por_fold")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(tabla_comp["fold"].astype(str), tabla_comp["retencion_p_pct"], color=AZUL)
    ax.set_ylim(0, 105); ax.set_xlabel("fold"); ax.set_ylabel("retencion de P por OR (%)"); ax.set_title("Retencion de P")
    fig.tight_layout(); _guardar_fig(fig, "12_retencion_p")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(d["fold"]) for d in filas_deltas], [d["delta_delay_min"] for d in filas_deltas], color=MORADO)
    ax.axhline(0, color=GRIS_OSCURO, linewidth=1)
    ax.set_xlabel("fold"); ax.set_ylabel("Delta retraso mediano (OR - P), min"); ax.set_title("Retraso: OR vs P")
    fig.tight_layout(); _guardar_fig(fig, "13_retraso")

    fig, ax = plt.subplots(figsize=(8, 6))
    criterios_ok = tabla_criterios.set_index("criterio")["cumple"].astype(int)
    ax.barh(criterios_ok.index.astype(str), criterios_ok.values, color=[VERDE if v else ROJO for v in criterios_ok.values])
    ax.set_xlabel("cumple (1=si)"); ax.set_ylabel("criterio"); ax.set_title("Resumen de los 14 criterios de seleccion de la fusion")
    fig.tight_layout(); _guardar_fig(fig, "14_resumen_14_criterios")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
