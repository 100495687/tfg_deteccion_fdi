"""Calibracion operativa de Energy distance sobre los residuos del TCN-AE: convierte los scores
diagnosticos `ED_signed`/`ED_positive` (ver `diagnostico_energy_distance.py`) en dos detectores
puntuales (umbral fijo por ventana, sin memoria) -- EDS y EDP -- calibrados a 0.06 falsos
eventos/dia solo en la primera mitad de val, y los compara con el detector puntual actual `P`
(`relative_kw` con su umbral consolidado) sobre la segunda mitad de val, fuera de muestra.

A diferencia de SM/EWMA (`memoria_finita_relative_kw.py`), P/EDS/EDP no tienen memoria: el
estado de alarma de una ventana depende solo de su propio score, nunca de ventanas anteriores.
Por eso la deteccion inducida no necesita reconstruir un historial de acumulador -- el estado
"limpio" en cualquier k se calcula una unica vez de forma global (`score_clean[k] > umbral`) y
se reutiliza para las tres variantes. Un evento de alarma es una transicion de inactivo a
activo (nunca cada ventana suelta), igual que en `memoria_finita_relative_kw.py`.

No reentrena nada, no cambia ventana/stride/relative_kw/checkpoint/umbral consolidado de P, no
reconstruye una referencia de Energy distance distinta (reutiliza literalmente la de
`diagnostico_energy_distance.construir_referencia_limpia`), no genera ataques nuevos. Reutiliza
sin modificar base_relative.cargar_ataques_y_modelo_360, diagnostico_energy_distance.{cargar_todo,
construir_referencia_limpia, construir_secuencia_val, calcular_residuos_por_minuto,
energy_distance_lote, puntuar_ventanas_ramp}, experimento_ramp.{construir_episodios_ramp,
_elegir_representativos, RHO_FINALES, COLOR_GRUPO}, experimento_stride.{_ventanas_solapadas,
calibrar_umbral_por_eventos, contar_eventos, mcnemar_p, bootstrap_pareado}, analisis_detectabilidad.
{contexto_e_impacto, _energia_antes_despues}, episodes.wilson_ci y memoria_finita_relative_kw.
{_duracion_eventos, enriquecer_resultado, resumen_metricas, tabla_complementariedad}.

Ejecutable de forma aislada:
    python -m src.energy_distance_operativo
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

from src import analisis_detectabilidad as ad
from src import diagnostico_energy_distance as ded
from src.base_relative import calcular_relative_kw
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import wilson_ci
from src.experimento_ramp import COLOR_GRUPO, RHO_FINALES, _elegir_representativos, construir_episodios_ramp
from src.experimento_stride import (
    _ventana_atacada,
    _ventanas_solapadas,
    bootstrap_pareado,
    calibrar_umbral_por_eventos,
    contar_eventos,
    mcnemar_p,
)
from src.memoria_finita_relative_kw import _duracion_eventos, enriquecer_resultado, resumen_metricas
from src.normalization import aplicar_zscore

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "energy_distance_operativo"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "energy_distance_operativo"
RAMP_DIR = BASE_DIR / "results" / "tables" / "experimento_ramp"
DET_DIR = BASE_DIR / "results" / "tables" / "analisis_detectabilidad"
DED_DIR = BASE_DIR / "results" / "tables" / "diagnostico_energy_distance"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
COLOR_METODO = {"P": GRIS_OSCURO, "EDS": AZUL, "EDP": NARANJA}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDE = 60
OBJETIVO_FA = 0.06
METODOS = ["P", "EDS", "EDP"]
COLUMNA_SCORE = {"P": "relative_kw", "EDS": "ED_signed", "EDP": "ED_positive"}
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
DURACION_BUCKETS = {"30_120min": (30, 120), "240_360min": (240, 360), "720_1440min": (720, 1440)}


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 1) Setup: referencia + secuencia limpia de val (reutilizadas tal cual del diagnostico) +
#    umbral consolidado de P (reproducido con la misma formula que en todos los experimentos
#    anteriores -- nunca recalibrado aqui)
# ==========================================================================================

def _tercil(x: pd.Series) -> pd.Series:
    """Tercil de una serie mediante rangos (method='first'): garantiza exactamente 3 grupos de
    igual tamano incluso con valores empatados (frecuente en consumo casi nulo tras un ataque),
    evitando el fallo de `qcut` por bordes de bin duplicados."""
    return pd.qcut(x.rank(method="first"), 3, labels=["bajo", "medio", "alto"])


def cargar_todo() -> dict:
    ctx = ded.cargar_todo()
    referencia = ded.construir_referencia_limpia(ctx["particion_train"], ctx["modelo"], ctx["params_norm"])
    df_val, signed_val, positive_val = ded.construir_secuencia_val(ctx, referencia)

    n_dias_val_total = len(ctx["particion_val_raw"]) / 1440.0
    umbral_relative_kw, _, _ = calibrar_umbral_por_eventos(df_val["relative_kw"].to_numpy(), n_dias_val_total, OBJETIVO_FA)

    ctx.update({
        "referencia": referencia, "df_val": df_val, "signed_val": signed_val, "positive_val": positive_val,
        "n_dias_val_total": n_dias_val_total, "umbral_relative_kw": umbral_relative_kw,
    })
    return ctx


def dividir_desarrollo_evaluacion(ctx: dict) -> dict:
    """Misma convencion exacta (corte = n_val // 2) que memoria_finita_relative_kw y
    cusum_final_normalizacion_movil -- verificado por test contra su 00_particion_temporal.csv."""
    df_val = ctx["df_val"]
    n_val = len(df_val)
    corte = n_val // 2
    ts = pd.to_datetime(df_val["timestamp"])
    dev_ini, dev_fin = ts.iloc[0], ts.iloc[corte - 1]
    eval_ini, eval_fin = ts.iloc[corte], ts.iloc[-1]
    n_dias_dev = corte * STRIDE / 1440.0
    n_dias_eval = (n_val - corte) * STRIDE / 1440.0
    tabla = pd.DataFrame([{
        "desarrollo_inicio": dev_ini, "desarrollo_fin": dev_fin, "n_ventanas_desarrollo": corte, "n_dias_desarrollo": n_dias_dev,
        "evaluacion_inicio": eval_ini, "evaluacion_fin": eval_fin, "n_ventanas_evaluacion": n_val - corte, "n_dias_evaluacion": n_dias_eval,
    }])
    _guardar_tabla(tabla, "01_division_temporal")
    return {"corte_idx": corte, "dev_ini": dev_ini, "dev_fin": dev_fin, "eval_ini": eval_ini, "eval_fin": eval_fin,
            "n_dias_dev": n_dias_dev, "n_dias_eval": n_dias_eval}


# ==========================================================================================
# 4/5) Eventos por transicion + calibracion de umbrales ED (solo desarrollo)
# ==========================================================================================

def resumen_eventos(activo: np.ndarray, n_dias: float) -> dict:
    n_eventos = contar_eventos(activo)
    duraciones = _duracion_eventos(activo)
    ts_eventos = np.where(activo & ~np.concatenate([[False], activo[:-1]]))[0]
    tiempo_medio = float(np.diff(ts_eventos).mean() * STRIDE) if len(ts_eventos) > 1 else float("nan")
    return {
        "n_eventos": n_eventos, "n_dias": n_dias, "eventos_dia": n_eventos / n_dias if n_dias else float("nan"),
        "ventanas_alarma_dia": float(activo.sum()) / n_dias if n_dias else float("nan"),
        "duracion_media_evento_min": float(duraciones.mean()), "duracion_mediana_evento_min": float(np.median(duraciones)),
        "duracion_p95_evento_min": float(np.percentile(duraciones, 95)), "duracion_max_evento_min": float(duraciones.max()),
        "tiempo_medio_entre_eventos_min": tiempo_medio,
    }


def calibrar_umbrales(ctx: dict, div: dict) -> dict:
    df_val = ctx["df_val"]
    corte = div["corte_idx"]
    umbrales = {}
    filas = []
    for metodo in METODOS:
        col = COLUMNA_SCORE[metodo]
        scores_dev = df_val[col].to_numpy()[:corte]
        if metodo == "P":
            h = ctx["umbral_relative_kw"]  # consolidado, nunca recalibrado
        else:
            h, _, _ = calibrar_umbral_por_eventos(scores_dev, div["n_dias_dev"], OBJETIVO_FA)
        activo_dev = scores_dev > h
        pct_limpio = float(sstats.percentileofscore(scores_dev, h))
        fila = {"metodo": metodo, "umbral": h, "percentil_limpio_equivalente": pct_limpio}
        fila.update(resumen_eventos(activo_dev, div["n_dias_dev"]))
        filas.append(fila)
        umbrales[metodo] = h
    tabla = _guardar_tabla(pd.DataFrame(filas), "02_umbrales_calibrados")
    return {"umbrales": umbrales, "tabla": tabla}


# ==========================================================================================
# 6) Falsas alarmas fuera de muestra (desarrollo + evaluacion, umbral congelado)
# ==========================================================================================

def falsas_alarmas_dev_eval(ctx: dict, div: dict, umbrales: dict) -> pd.DataFrame:
    df_val = ctx["df_val"]
    corte = div["corte_idx"]
    ts = pd.to_datetime(df_val["timestamp"])
    filas = []
    for metodo in METODOS:
        col = COLUMNA_SCORE[metodo]
        scores = df_val[col].to_numpy()
        h = umbrales[metodo]
        for etiqueta, ini, fin, n_dias in (("desarrollo", 0, corte, div["n_dias_dev"]), ("evaluacion", corte, len(df_val), div["n_dias_eval"])):
            activo = scores[ini:fin] > h
            fila = {"metodo": metodo, "periodo": etiqueta, "umbral": h}
            r = resumen_eventos(activo, n_dias)
            if r["n_eventos"] > 0:
                lo = sstats.chi2.ppf(0.025, 2 * r["n_eventos"]) / 2 / n_dias
            else:
                lo = 0.0
            hi = sstats.chi2.ppf(0.975, 2 * (r["n_eventos"] + 1)) / 2 / n_dias
            r["ic95_poisson_low"] = lo
            r["ic95_poisson_high"] = hi
            r["diferencia_vs_objetivo"] = r["eventos_dia"] - OBJETIVO_FA
            fila.update(r)
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    dev_rate = tabla[tabla["periodo"] == "desarrollo"].set_index("metodo")["eventos_dia"]
    eval_rate = tabla[tabla["periodo"] == "evaluacion"].set_index("metodo")["eventos_dia"]
    tabla["razon_eval_dev"] = tabla["metodo"].map(lambda m: eval_rate[m] / dev_rate[m] if dev_rate[m] else float("nan"))
    _guardar_tabla(tabla, "03_falsas_alarmas_desarrollo_evaluacion")

    filas_ctx = []
    for metodo in METODOS:
        col = COLUMNA_SCORE[metodo]
        sub = pd.DataFrame({"score": df_val[col].to_numpy()[corte:], "activo": df_val[col].to_numpy()[corte:] > umbrales[metodo],
                             "mes": ts.dt.month.to_numpy()[corte:], "hora": ts.dt.hour.to_numpy()[corte:],
                             "consumo_medio_kw": df_val["consumo_medio_kw"].to_numpy()[corte:]})
        terciles = _tercil(sub["consumo_medio_kw"])
        for mes, g in sub.groupby("mes"):
            filas_ctx.append({"metodo": metodo, "agrupacion": "mes", "valor": mes, "n_ventanas": len(g), "pct_ventanas_activas": 100 * g["activo"].mean()})
        for hora, g in sub.groupby("hora"):
            filas_ctx.append({"metodo": metodo, "agrupacion": "hora", "valor": hora, "n_ventanas": len(g), "pct_ventanas_activas": 100 * g["activo"].mean()})
        for terc, g in sub.groupby(terciles, observed=True):
            filas_ctx.append({"metodo": metodo, "agrupacion": "tercil_consumo", "valor": terc, "n_ventanas": len(g), "pct_ventanas_activas": 100 * g["activo"].mean()})
    _guardar_tabla(pd.DataFrame(filas_ctx), "03b_falsas_alarmas_distribucion_eval")
    return tabla


# ==========================================================================================
# 7) Estabilidad del score limpio (desarrollo vs evaluacion)
# ==========================================================================================

def _mad(x): return float(np.median(np.abs(x - np.median(x))))


def _skew(x):
    m, s = x.mean(), x.std()
    return float(np.mean((x - m) ** 3) / s ** 3) if s else 0.0


def estabilidad_score(ctx: dict, div: dict) -> pd.DataFrame:
    df_val = ctx["df_val"]
    corte = div["corte_idx"]
    ts = pd.to_datetime(df_val["timestamp"])
    filas = []
    for metodo in METODOS:
        col = COLUMNA_SCORE[metodo]
        for etiqueta, ini, fin in (("desarrollo", 0, corte), ("evaluacion", corte, len(df_val))):
            x = df_val[col].to_numpy()[ini:fin]
            t = ts.iloc[ini:fin]
            pcts = np.percentile(x, [90, 95, 99, 99.5, 99.9])
            mediana_mensual = pd.Series(x).groupby(t.dt.month.to_numpy()).median()
            mediana_horaria = pd.Series(x).groupby(t.dt.hour.to_numpy()).median()
            serie = pd.Series(x)
            autocorr = {f"autocorr_{h}h": float(serie.autocorr(lag=h)) for h in (1, 6, 12, 24, 168)}
            fila = {
                "metodo": metodo, "periodo": etiqueta, "n": len(x), "media": float(x.mean()), "mediana": float(np.median(x)),
                "desviacion": float(x.std()), "mad": _mad(x), "p90": float(pcts[0]), "p95": float(pcts[1]), "p99": float(pcts[2]),
                "p99_5": float(pcts[3]), "p99_9": float(pcts[4]), "asimetria": _skew(x),
                "rango_mensual_mediana": float(mediana_mensual.max() - mediana_mensual.min()),
                "rango_horario_mediana": float(mediana_horaria.max() - mediana_horaria.min()),
                "pearson_vs_consumo": float(sstats.pearsonr(x, df_val["consumo_medio_kw"].to_numpy()[ini:fin])[0]),
            }
            fila.update(autocorr)
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    _guardar_tabla(tabla, "04_estabilidad_ED")
    return tabla


# ==========================================================================================
# 8) Conjuntos de episodios de evaluacion (R, RC, B, HI) -- solo segunda mitad de val
# ==========================================================================================

def construir_conjuntos_eval(ctx: dict, div: dict) -> dict:
    tabla_ramp_fresca, ataques_ramp = construir_episodios_ramp(ctx["master_csv"], ctx["particion_val_raw"], ctx["inicio_particion"])
    for a in ataques_ramp.values():
        a["y_tramo"] = a["y_tramo_ramp"]  # alias para reutilizar ded.puntuar_ventanas_ramp sin modificarlo

    reduccion = ctx["master_csv"][ctx["master_csv"]["family"] == "reduccion_constante"]
    ataques_constante = {ep: ctx["ataques_originales"][ep] for ep in reduccion["episode_id"]}

    def _en_eval(tabla_meta: pd.DataFrame) -> set:
        return set(tabla_meta[(tabla_meta["attack_start"] >= div["eval_ini"]) & (tabla_meta["attack_end"] <= div["eval_fin"])]["episode_id"])

    R_eval = _en_eval(tabla_ramp_fresca)
    RC_eval = _en_eval(reduccion[["episode_id", "attack_start", "attack_end"]])

    det_full = pd.read_csv(DET_DIR / "episodios_detectabilidad_impacto.csv")
    grupo_b_ids = set(det_full[(det_full["family"] == "reduccion_constante") & (det_full["detectability_group"] == "B")]["episode_id"])
    B_eval = RC_eval & grupo_b_ids

    alto_impacto_full = pd.read_csv(RAMP_DIR / "ramp_no_detectados_alto_impacto.csv")
    HI_eval = set(alto_impacto_full["episode_id"]) & R_eval

    resumen = pd.DataFrame([
        {"conjunto": "R_ramp", "n_total": len(tabla_ramp_fresca), "n_evaluacion": len(R_eval)},
        {"conjunto": "RC_todas_constantes", "n_total": len(reduccion), "n_evaluacion": len(RC_eval)},
        {"conjunto": "B_grupo_b", "n_total": len(grupo_b_ids), "n_evaluacion": len(B_eval)},
        {"conjunto": "HI_alto_impacto_no_detectado_por_P_original", "n_total": len(alto_impacto_full), "n_evaluacion": len(HI_eval)},
    ])
    _guardar_tabla(resumen, "00_conjuntos_episodios_evaluacion")

    return {
        "tabla_ramp_fresca": tabla_ramp_fresca, "ataques_ramp": ataques_ramp, "ataques_constante": ataques_constante,
        "R_eval": R_eval, "RC_eval": RC_eval, "B_eval": B_eval, "HI_eval": HI_eval,
    }


# ==========================================================================================
# 9) Puntuacion de ventanas atacadas (relative_kw + ED_signed + ED_positive, 1 llamada
#    batched por conjunto de ataques) + evaluacion contrafactual inducida por eventos
# ==========================================================================================

def puntuar_ventanas_atacadas(ataques: dict, particion_val_raw: np.ndarray, grid_60: dict, modelo, params_norm: dict,
                               referencia: dict) -> pd.DataFrame:
    """Genérico: `ded.puntuar_ventanas_ramp` solo necesita offset_global/duracion/y_tramo, por
    lo que sirve igual para ataques ramp_lineal o reduccion_constante (no se reimplementa)."""
    meta, signed, positive = ded.puntuar_ventanas_ramp(ataques, particion_val_raw, grid_60, modelo, params_norm)
    meta = meta.rename(columns={"relative_kw_ramp": "relative_kw"})
    meta["ED_signed"] = ded.energy_distance_lote(signed, referencia["reference_signed"])
    meta["ED_positive"] = ded.energy_distance_lote(positive, referencia["reference_positive"])
    return meta


def simular_episodios_puntual(ataques: dict, ids: set, attacked_scores_idx: dict, score_clean_full: np.ndarray,
                               umbral: float, fin_grid: np.ndarray) -> pd.DataFrame:
    """Deteccion inducida basada en eventos (transicion inactivo->activo), sin memoria: el
    estado limpio se calcula una unica vez de forma global (`score_clean_full > umbral`)."""
    active_clean_full = score_clean_full > umbral
    filas = []
    for ep in ids:
        a = ataques[ep]
        ks_local = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, len(score_clean_full)))
        k_before = ks_local[0] - 1
        s_att = np.array([attacked_scores_idx[ep][k] for k in ks_local])
        active_att = s_att > umbral

        prev_active = bool(active_clean_full[k_before]) if k_before >= 0 else False
        event_start = np.empty(len(ks_local), dtype=bool)
        for i in range(len(ks_local)):
            event_start[i] = bool(active_att[i]) and not prev_active
            prev_active = bool(active_att[i])

        active_clean_local = active_clean_full[ks_local]
        induced = event_start & ~active_clean_local
        idx_ind = np.where(induced)[0]
        ts_induced = pd.Timestamp(fin_grid[ks_local[idx_ind[0]]]) if len(idx_ind) else pd.NaT

        filas.append({
            "episode_id": ep, "n_affected_windows": len(ks_local),
            "max_score_attacked": float(s_att.max()), "max_score_clean": float(score_clean_full[ks_local].max()),
            "raw_detected": bool(active_att.any()), "induced_detected": bool(induced.any()),
            "first_alarm_timestamp": ts_induced,
        })
    return pd.DataFrame(filas)


def _indexar_scores(tabla_ventanas: pd.DataFrame, columna: str) -> dict[str, dict[int, float]]:
    return {ep: dict(zip(g["k"], g[columna])) for ep, g in tabla_ventanas.groupby("episode_id")}


def evaluar_conjunto(metodo: str, ataques: dict, ids: set, meta_ventanas: pd.DataFrame, df_val: pd.DataFrame,
                      umbral: float, fin_grid: np.ndarray, meta_episodio: pd.DataFrame, particion_val_raw: np.ndarray,
                      contexto_impacto: pd.DataFrame) -> pd.DataFrame:
    col = COLUMNA_SCORE[metodo]
    idx = _indexar_scores(meta_ventanas, col)
    score_clean_full = df_val[col].to_numpy()
    sim = simular_episodios_puntual(ataques, ids, idx, score_clean_full, umbral, fin_grid)
    return enriquecer_resultado(sim, ataques, meta_episodio, particion_val_raw, contexto_impacto)


# ==========================================================================================
# 12) Comparacion con los 145 casos diagnosticos (P90 limpio, del experimento anterior)
# ==========================================================================================

def comparar_con_casos_p90(dfs_ramp_eval: dict[str, pd.DataFrame], umbrales: dict) -> pd.DataFrame:
    diag = pd.read_csv(DED_DIR / "casos_ed_aporta_sobre_relative_kw.csv")
    r_eval_ids = set(dfs_ramp_eval["P"]["episode_id"])
    diag_eval = diag[diag["episode_id"].isin(r_eval_ids)].copy()

    filas = []
    for metodo in ("EDS", "EDP"):
        col_max = "max_ED_signed_ramp" if metodo == "EDS" else "max_ED_positive_ramp"
        det = dfs_ramp_eval[metodo].set_index("episode_id")["induced_detected"].reindex(diag_eval["episode_id"]).fillna(False).to_numpy()
        energia = diag_eval["hidden_energy_kwh"].to_numpy()
        distancia = umbrales[metodo] - diag_eval[col_max].to_numpy()
        filas.append({
            "metodo": metodo, "n_casos_diagnosticos_disponibles_en_evaluacion": len(diag_eval),
            "n_detectados_por_umbral_operativo": int(det.sum()),
            "pct_retenido": 100 * det.mean() if len(diag_eval) else float("nan"),
            "energia_total_kwh": float(energia.sum()), "energia_retenida_kwh": float(energia[det].sum()) if det.any() else 0.0,
            "pct_energia_retenida": 100 * energia[det].sum() / energia.sum() if energia.sum() > 1e-9 else float("nan"),
            "distancia_media_ED_max_a_umbral": float(distancia.mean()), "distancia_mediana_ED_max_a_umbral": float(np.median(distancia)),
        })
    tabla_resumen = pd.DataFrame(filas)
    _guardar_tabla(tabla_resumen, "09_retencion_casos_p90")

    diag_eval["retenido_EDS"] = diag_eval["episode_id"].isin(set(dfs_ramp_eval["EDS"].loc[dfs_ramp_eval["EDS"]["induced_detected"], "episode_id"]))
    diag_eval["retenido_EDP"] = diag_eval["episode_id"].isin(set(dfs_ramp_eval["EDP"].loc[dfs_ramp_eval["EDP"]["induced_detected"], "episode_id"]))
    _guardar_tabla(diag_eval, "09b_casos_p90_detalle_evaluacion")
    return tabla_resumen


# ==========================================================================================
# 14) Complementariedad P vs EDS / P vs EDP
# ==========================================================================================

def tabla_complementariedad(df_p: pd.DataFrame, df_acc: pd.DataFrame) -> dict:
    p = df_p.set_index("episode_id")["induced_detected"].astype(bool)
    a = df_acc.set_index("episode_id")["induced_detected"].astype(bool).reindex(p.index).fillna(False)
    ambos = int((p & a).sum()); solo_p = int((p & ~a).sum()); solo_a = int((~p & a).sum()); ninguno = int((~p & ~a).sum())
    retention = 100 * ambos / p.sum() if p.sum() else float("nan")
    exclusive = 100 * solo_a / (~p).sum() if (~p).sum() else float("nan")
    return {"detectado_ambos": ambos, "solo_puntual": solo_p, "solo_ed": solo_a, "detectado_ninguno": ninguno,
            "retention_rate_pct": retention, "exclusive_recovery_rate_pct": exclusive}


def construir_complementariedad(subconjuntos: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    filas = []
    for nombre_sub, dfs in subconjuntos.items():
        for metodo_ed in ("EDS", "EDP"):
            fila = {"subconjunto": nombre_sub, "comparacion": f"P_vs_{metodo_ed}"}
            fila.update(tabla_complementariedad(dfs["P"], dfs[metodo_ed]))
            filas.append(fila)
    return pd.DataFrame(filas)


# ==========================================================================================
# 15) Comparaciones estadisticas pareadas (McNemar + bootstrap + Wilcoxon)
# ==========================================================================================

def comparacion_pareada(dfs: dict[str, pd.DataFrame], hidden_col: str, subconjunto: str) -> pd.DataFrame:
    pares = [("P", "EDS"), ("P", "EDP"), ("EDS", "EDP")]
    filas = []
    for a, b in pares:
        da_full = dfs[a].set_index("episode_id")["induced_detected"].astype(bool)
        db_full = dfs[b].set_index("episode_id")["induced_detected"].astype(bool).reindex(da_full.index).fillna(False)
        da, db = da_full.to_numpy(), db_full.to_numpy()
        n = len(da)
        solo_a, solo_b = int((da & ~db).sum()), int((~da & db).sum())
        p_mcnemar = mcnemar_p(solo_a, solo_b)
        ci_low, ci_high = bootstrap_pareado(da.astype(float), db.astype(float))

        if hidden_col in dfs[a].columns:
            h = dfs[a].set_index("episode_id")[hidden_col].reindex(da_full.index).to_numpy()
            ci_e_low, ci_e_high = bootstrap_pareado(np.where(da, h, 0.0), np.where(db, h, 0.0))
            dr_e_a = 100 * (h * da).sum() / h.sum() if h.sum() > 1e-9 else float("nan")
            dr_e_b = 100 * (h * db).sum() / h.sum() if h.sum() > 1e-9 else float("nan")
        else:
            ci_e_low = ci_e_high = dr_e_a = dr_e_b = float("nan")

        delay_a = dfs[a].set_index("episode_id")["detection_delay_min"]
        delay_b = dfs[b].set_index("episode_id")["detection_delay_min"].reindex(delay_a.index)
        ambos_det = da_full & db_full
        if ambos_det.sum() >= 5:
            ra, rb = delay_a[ambos_det].to_numpy(), delay_b[ambos_det].to_numpy()
            try:
                _, p_wilcoxon = sstats.wilcoxon(ra, rb)
            except ValueError:
                p_wilcoxon = float("nan")
        else:
            p_wilcoxon = float("nan")

        filas.append({
            "subconjunto": subconjunto, "comparacion": f"{a}_vs_{b}", "n_episodios": n,
            "DR_a_pct": 100 * da.mean() if n else float("nan"), "DR_b_pct": 100 * db.mean() if n else float("nan"),
            "diferencia_DR_pp": 100 * (db.mean() - da.mean()) if n else float("nan"),
            "mcnemar_p": p_mcnemar, "bootstrap_dr_ci95_low": ci_low, "bootstrap_dr_ci95_high": ci_high,
            "dr_energetico_a_pct": dr_e_a, "dr_energetico_b_pct": dr_e_b,
            "bootstrap_energia_ci95_low": ci_e_low, "bootstrap_energia_ci95_high": ci_e_high,
            "wilcoxon_p_retraso": p_wilcoxon, "n_detectados_ambos": int(ambos_det.sum()),
            "significativo_p<0.05": p_mcnemar < 0.05,
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# 18) Criterios de exito (fijados antes de mirar los resultados de evaluacion)
# ==========================================================================================

CRITERIOS_EXITO_TEXTO = {
    1: "Recupera >= 15% de las rampas de alto impacto no detectadas por P",
    2: "Recupera >= 10% de las rampas largas (720-1440min) no detectadas por P",
    3: "Mejora >= 3pp el DR energetico de ramp frente a P (o cobertura energetica exclusiva equivalente)",
    4: "Mantiene una tasa de falsas alarmas fuera de muestra operativamente comparable con P",
    5: "No genera una inflacion > 2x el objetivo de 0.06 eventos/dia",
    6: "Retraso mediano no peor que el de P en mas de 120 minutos",
    7: "La mejora aparece en mas de una severidad o duracion",
    8: "Las detecciones exclusivas no se concentran unicamente en un mes/hora/tercil de consumo",
    9: "EDS o EDP aporta informacion claramente no redundante (deteccion exclusiva > 0)",
    10: "Los casos diagnosticos del P90 mantienen una fraccion relevante (>=30%) al aplicar el umbral operativo",
}


def evaluar_criterios(metodo: str, pct_recupera_alto_impacto: float, pct_recupera_largas: float,
                       dif_dr_energia: float, fa_eval: float, inflacion_fa: float, dif_retraso: float,
                       mejora_multi: bool, concentracion_exclusiva: bool, deteccion_exclusiva: bool,
                       pct_retenido_p90: float) -> pd.DataFrame:
    resultados = {
        1: pct_recupera_alto_impacto >= 15.0, 2: pct_recupera_largas >= 10.0, 3: dif_dr_energia >= 3.0,
        4: abs(fa_eval - OBJETIVO_FA) <= OBJETIVO_FA, 5: inflacion_fa <= 2.0,
        6: (dif_retraso <= 120.0) if not np.isnan(dif_retraso) else False,
        7: mejora_multi, 8: not concentracion_exclusiva, 9: deteccion_exclusiva, 10: pct_retenido_p90 >= 30.0,
    }
    tabla = pd.DataFrame([{"metodo": metodo, "criterio_num": k, "criterio": CRITERIOS_EXITO_TEXTO[k], "cumplido": v} for k, v in resultados.items()])
    tabla.attrs["n_cumplidos"] = sum(resultados.values())
    tabla.attrs["mayoria"] = sum(resultados.values()) >= 6
    return tabla


# ==========================================================================================
# 16) Analisis del consumo medio (solo diagnostico, no implementa umbrales por tercil)
# ==========================================================================================

def analisis_por_tercil(dfs_ramp_eval: dict[str, pd.DataFrame], ataques_ramp: dict, particion_val_raw: np.ndarray, hi_ids: set) -> dict:
    filas_dr, filas_migracion = [], []
    for metodo, df in dfs_ramp_eval.items():
        consumo_clean, consumo_atacado = [], []
        for ep in df["episode_id"]:
            a = ataques_ramp[ep]
            clean = particion_val_raw[a["offset_global"]:a["offset_global"] + a["duracion"]]
            consumo_clean.append(float(clean.mean()))
            consumo_atacado.append(float(np.asarray(a["y_tramo_ramp"]).mean()))
        df = df.copy()
        df["consumo_medio_clean"] = consumo_clean
        df["consumo_medio_atacado"] = consumo_atacado
        terciles_clean = _tercil(df["consumo_medio_clean"])
        terciles_atacado = _tercil(df["consumo_medio_atacado"])
        df["tercil_clean"], df["tercil_atacado"] = terciles_clean, terciles_atacado
        for terc, g in df.groupby(terciles_clean, observed=True):
            filas_dr.append({"metodo": metodo, "tercil_consumo": terc, "n_episodios": len(g), "DR_pct": 100 * g["induced_detected"].mean(),
                              "n_alto_impacto": int(g["episode_id"].isin(hi_ids).sum()),
                              "n_alto_impacto_recuperado": int((g["episode_id"].isin(hi_ids) & g["induced_detected"]).sum())})
        migracion = pd.crosstab(df["tercil_clean"], df["tercil_atacado"])
        migracion.insert(0, "metodo", metodo)
        filas_migracion.append(migracion.reset_index())
    tabla_dr = _guardar_tabla(pd.DataFrame(filas_dr), "13_resultados_por_tercil_consumo")
    tabla_migracion = pd.concat(filas_migracion, ignore_index=True)
    _guardar_tabla(tabla_migracion, "13b_migracion_tercil")
    return {"dr_por_tercil": tabla_dr, "migracion": tabla_migracion}


def concentracion_exclusiva_por_tercil(df_p: pd.DataFrame, df_ed: pd.DataFrame, ataques: dict, particion_val_raw: np.ndarray) -> pd.DataFrame:
    p = df_p.set_index("episode_id")["induced_detected"].astype(bool)
    a = df_ed.set_index("episode_id")["induced_detected"].astype(bool).reindex(p.index).fillna(False)
    solo_ed_ids = list(p.index[(~p) & a])
    if len(solo_ed_ids) < 3:
        return pd.DataFrame([{"tercil_consumo": "sin_casos", "n": len(solo_ed_ids)}])
    consumo = []
    for ep in solo_ed_ids:
        at = ataques[ep]
        consumo.append(float(particion_val_raw[at["offset_global"]:at["offset_global"] + at["duracion"]].mean()))
    tabla = pd.DataFrame({"episode_id": solo_ed_ids, "consumo_medio_clean": consumo})
    tabla["tercil_consumo"] = _tercil(tabla["consumo_medio_clean"])
    return tabla.groupby("tercil_consumo", observed=True).size().reset_index(name="n")


# ==========================================================================================
# 17) Casos visuales
# ==========================================================================================

def figura_caso(episode_id: str, ataques: dict, particion_val_raw: np.ndarray, meta_ventanas: pd.DataFrame,
                 df_val: pd.DataFrame, umbrales: dict, inicio_particion, nombre_archivo: str, etiqueta: str) -> None:
    a = ataques[episode_id]
    offset, dur = a["offset_global"], a["duracion"]
    y_tramo = a.get("y_tramo_ramp", a.get("y_tramo"))
    margen = max(60, dur // 4)
    ini, fin = max(0, offset - margen), offset + dur + margen
    x_limpio = particion_val_raw[ini:fin].astype(np.float64)
    x_atacado = x_limpio.copy(); x_atacado[offset - ini:offset - ini + dur] = y_tramo
    t = [inicio_particion + pd.Timedelta(minutes=m) for m in range(ini, fin)]
    energia_acum = np.cumsum(x_limpio[offset - ini:offset - ini + dur] - y_tramo) / 60.0
    t_ataque = [inicio_particion + pd.Timedelta(minutes=offset + m) for m in range(dur)]

    g = meta_ventanas[meta_ventanas["episode_id"] == episode_id].sort_values("k")
    t_score = pd.to_datetime(g["window_end"])
    k_ini = int(g["k"].iloc[0])
    clean_rel = df_val.set_index("k")["relative_kw"].reindex(g["k"]).to_numpy()
    clean_eds = df_val.set_index("k")["ED_signed"].reindex(g["k"]).to_numpy()
    clean_edp = df_val.set_index("k")["ED_positive"].reindex(g["k"]).to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=False)
    axes[0].plot(t, x_limpio, color=GRIS, linewidth=1.1, label="limpio")
    axes[0].plot(t, x_atacado, color=ROJO, linewidth=1.2, alpha=0.85, label="atacado")
    axes[0].axvspan(inicio_particion + pd.Timedelta(minutes=offset), inicio_particion + pd.Timedelta(minutes=offset + dur), color=VERDE, alpha=0.08)
    axes[0].set_ylabel("kW"); axes[0].legend(fontsize=7.5); axes[0].set_title(f"{episode_id} | {etiqueta}", fontsize=9.5)

    axes[1].plot(t_ataque, energia_acum, color=NARANJA, linewidth=1.3); axes[1].set_ylabel("energia oculta acum. (kWh)")

    axes[2].plot(t_score, clean_rel, color=GRIS, marker="o", markersize=3, label="relative_kw limpio")
    axes[2].plot(t_score, g["relative_kw"], color=ROJO, marker="o", markersize=3, label="relative_kw atacado")
    axes[2].axhline(umbrales["P"], color=GRIS_OSCURO, linestyle="--", linewidth=1, label="umbral P")
    axes[2].set_ylabel("relative_kw"); axes[2].legend(fontsize=7)

    axes[3].plot(t_score, clean_eds, color=AZUL_CLARO, marker="o", markersize=3, label="ED_signed limpio")
    axes[3].plot(t_score, g["ED_signed"], color=AZUL, marker="o", markersize=3, label="ED_signed atacado")
    axes[3].plot(t_score, clean_edp, color="#f0b27a", marker="s", markersize=3, label="ED_positive limpio")
    axes[3].plot(t_score, g["ED_positive"], color=NARANJA, marker="s", markersize=3, label="ED_positive atacado")
    axes[3].axhline(umbrales["EDS"], color=AZUL_OSCURO, linestyle="--", linewidth=1, label="umbral EDS")
    axes[3].axhline(umbrales["EDP"], color="#a04000", linestyle="--", linewidth=1, label="umbral EDP")
    axes[3].set_ylabel("energy distance"); axes[3].legend(fontsize=6.5)

    fig.tight_layout(); _guardar_fig(fig, nombre_archivo)


def _cuenta_buckets_con_mejora(filas_bucket: list[dict], metodo: str, campo_bucket: str) -> int:
    """Numero de buckets (duracion o severidad) donde `metodo` supera a P en DR ponderado por
    energia -- usado para el criterio 7 ('la mejora aparece en mas de una duracion/severidad')."""
    piv = pd.DataFrame(filas_bucket).pivot(index=campo_bucket, columns="metodo", values="energy_weighted_dr_pct")
    if "P" not in piv.columns or metodo not in piv.columns:
        return 0
    return int((piv[metodo].fillna(0) > piv["P"].fillna(0)).sum())


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("[1] Cargando modelo, referencia y secuencia limpia de val (reutilizadas del diagnostico)...")
    ctx = cargar_todo()
    print(f"  umbral consolidado de P (relative_kw): {ctx['umbral_relative_kw']:.6f}")

    print("\n[3] Dividiendo val en desarrollo/evaluacion...")
    div = dividir_desarrollo_evaluacion(ctx)
    print(f"  desarrollo: {div['dev_ini']} -> {div['dev_fin']} ({div['n_dias_dev']:.1f} dias)")
    print(f"  evaluacion: {div['eval_ini']} -> {div['eval_fin']} ({div['n_dias_eval']:.1f} dias)")

    print("\n[5] Calibrando umbrales EDS/EDP (SOLO desarrollo, objetivo 0.06 eventos/dia)...")
    cal = calibrar_umbrales(ctx, div)
    umbrales = cal["umbrales"]
    print(cal["tabla"][["metodo", "umbral", "percentil_limpio_equivalente", "eventos_dia", "duracion_max_evento_min"]].to_string(index=False))

    print("\n[6] Falsas alarmas desarrollo/evaluacion (umbral congelado)...")
    tabla_fa = falsas_alarmas_dev_eval(ctx, div, umbrales)
    print(tabla_fa[["metodo", "periodo", "eventos_dia", "ic95_poisson_low", "ic95_poisson_high", "razon_eval_dev"]].to_string(index=False))

    print("\n[7] Estabilidad del score limpio (desarrollo vs evaluacion)...")
    tabla_estab = estabilidad_score(ctx, div)
    print(tabla_estab[["metodo", "periodo", "p90", "p99", "p99_9", "autocorr_1h", "autocorr_6h"]].to_string(index=False))

    print("\n[8] Construyendo conjuntos de episodios de evaluacion (R, RC, B, HI)...")
    conj = construir_conjuntos_eval(ctx, div)
    print(f"  R_eval={len(conj['R_eval'])} | RC_eval={len(conj['RC_eval'])} | B_eval={len(conj['B_eval'])} | HI_eval={len(conj['HI_eval'])}")

    print("\nPuntuando ventanas atacadas (ramp + constante), calculando ED contra la referencia congelada...")
    meta_ramp = puntuar_ventanas_atacadas(conj["ataques_ramp"], ctx["particion_val_raw"], ctx["grid_60"], ctx["modelo"], ctx["params_norm"], ctx["referencia"])
    meta_const = puntuar_ventanas_atacadas(conj["ataques_constante"], ctx["particion_val_raw"], ctx["grid_60"], ctx["modelo"], ctx["params_norm"], ctx["referencia"])
    _guardar_tabla(meta_ramp, "trayectorias_ventanas_ramp")
    _guardar_tabla(meta_const, "trayectorias_ventanas_constante")

    contexto_ramp = ad.contexto_e_impacto(conj["ataques_ramp"], ctx["particion_val_raw"])
    contexto_const = ad.contexto_e_impacto(conj["ataques_constante"], ctx["particion_val_raw"])
    meta_ep_ramp = conj["tabla_ramp_fresca"][["episode_id", "attack_start", "attack_end", "duration_min", "rho_final"]]
    meta_ep_const = ctx["master_csv"][ctx["master_csv"]["family"] == "reduccion_constante"][
        ["episode_id", "family", "parameter", "attack_start", "attack_end", "duration_min"]]

    print("\n[9] Evaluando P/EDS/EDP sobre ramp y constante (evaluacion, deteccion inducida por eventos)...")
    fin_grid = ctx["grid_60"]["fin"]
    dfs_ramp_eval, dfs_const_eval = {}, {}
    for metodo in METODOS:
        dfs_ramp_eval[metodo] = evaluar_conjunto(metodo, conj["ataques_ramp"], conj["R_eval"], meta_ramp, ctx["df_val"],
                                                  umbrales[metodo], fin_grid, meta_ep_ramp, ctx["particion_val_raw"], contexto_ramp)
        dfs_const_eval[metodo] = evaluar_conjunto(metodo, conj["ataques_constante"], conj["RC_eval"], meta_const, ctx["df_val"],
                                                   umbrales[metodo], fin_grid, meta_ep_const, ctx["particion_val_raw"], contexto_const)
        ramp_temp = pd.read_csv(RAMP_DIR / "episodios_ramp.csv")[["episode_id", "grupo_temporal"]]
        dfs_ramp_eval[metodo] = dfs_ramp_eval[metodo].merge(ramp_temp, on="episode_id", how="left")
        _guardar_tabla(dfs_ramp_eval[metodo], f"05_resultados_{metodo}_ramp_eval")
        _guardar_tabla(dfs_const_eval[metodo], f"05_resultados_{metodo}_constante_eval")

    print("\n[10] Metricas principales sobre ramp (evaluacion)...")
    filas_glob = []
    for metodo, df in dfs_ramp_eval.items():
        fila = {"metodo": metodo}; fila.update(resumen_metricas(df)); filas_glob.append(fila)
    tabla_ramp_global = _guardar_tabla(pd.DataFrame(filas_glob), "06_ramp_global")
    print(tabla_ramp_global.to_string(index=False))

    filas_dur, filas_rho, filas_grupo = [], [], []
    for metodo, df in dfs_ramp_eval.items():
        for dur, g in df.groupby("duration_min"):
            fila = {"metodo": metodo, "duration_min": dur}; fila.update(resumen_metricas(g)); filas_dur.append(fila)
        for rho, g in df.groupby("rho_final"):
            fila = {"metodo": metodo, "rho_final": rho}; fila.update(resumen_metricas(g)); filas_rho.append(fila)
        for grupo in ("C", "D"):
            g = df[df["grupo_temporal"] == grupo]
            if len(g):
                fila = {"metodo": metodo, "grupo_temporal": grupo}; fila.update(resumen_metricas(g)); filas_grupo.append(fila)
    _guardar_tabla(pd.DataFrame(filas_dur), "07_ramp_por_duracion")
    _guardar_tabla(pd.DataFrame(filas_rho), "07_ramp_por_severidad")
    _guardar_tabla(pd.DataFrame(filas_grupo), "07_ramp_por_grupo_temporal")

    print("\n[11] Rampas de alto impacto (evaluacion)...")
    filas_hi = []
    for metodo, df in dfs_ramp_eval.items():
        sub = df[df["episode_id"].isin(conj["HI_eval"])]
        n_rec = int(sub["induced_detected"].sum())
        filas_hi.append({
            "metodo": metodo, "n_total": len(sub), "n_detectados": n_rec,
            "pct_recuperado": 100 * n_rec / len(sub) if len(sub) else float("nan"),
            "energia_total_kwh": float(sub["hidden_energy_kwh"].sum()),
            "energia_recuperada_kwh": float(sub.loc[sub["induced_detected"], "hidden_energy_kwh"].sum()),
            "pct_energia_cubierta": 100 * sub.loc[sub["induced_detected"], "hidden_energy_kwh"].sum() / sub["hidden_energy_kwh"].sum() if sub["hidden_energy_kwh"].sum() > 1e-9 else float("nan"),
            "retraso_mediano_min": float(sub.loc[sub["induced_detected"], "detection_delay_min"].median()) if n_rec else float("nan"),
            "hidden_energy_before_detection_media_kwh": float(sub["hidden_energy_before_detection_kwh"].mean()) if len(sub) else float("nan"),
        })
    tabla_hi = _guardar_tabla(pd.DataFrame(filas_hi), "08_rampas_alto_impacto")
    print(f"  n alto impacto en evaluacion: {len(conj['HI_eval'])}")
    print(tabla_hi.to_string(index=False))
    detalle_hi = dfs_ramp_eval["P"][dfs_ramp_eval["P"]["episode_id"].isin(conj["HI_eval"])][["episode_id", "duration_min", "rho_final", "grupo_temporal"]].copy()
    for metodo in ("EDS", "EDP"):
        detalle_hi[f"detectado_{metodo}"] = detalle_hi["episode_id"].map(
            dfs_ramp_eval[metodo].set_index("episode_id")["induced_detected"])
    _guardar_tabla(detalle_hi, "ramp_alto_impacto_detectado_energy_distance")

    print("\n[12] Comparando con los 145 casos diagnosticos (P90 limpio, experimento anterior)...")
    tabla_p90 = comparar_con_casos_p90(dfs_ramp_eval, umbrales)
    print(tabla_p90.to_string(index=False))

    print("\n[13] Evaluando reducciones constantes (todas + Grupo B)...")
    filas_rc = []
    for metodo, df in dfs_const_eval.items():
        fila = {"metodo": metodo}; fila.update(resumen_metricas(df)); filas_rc.append(fila)
    tabla_rc = _guardar_tabla(pd.DataFrame(filas_rc), "10_resultados_constantes")
    print(tabla_rc.to_string(index=False))

    filas_b = []
    for metodo, df in dfs_const_eval.items():
        sub = df[df["episode_id"].isin(conj["B_eval"])]
        n_rec = int(sub["induced_detected"].sum())
        filas_b.append({"metodo": metodo, "n_grupo_b": len(sub), "n_recuperados": n_rec,
                         "pct_recuperado": 100 * n_rec / len(sub) if len(sub) else float("nan"),
                         "retraso_mediano_min": float(sub.loc[sub["induced_detected"], "detection_delay_min"].median()) if n_rec else float("nan")})
    tabla_b = _guardar_tabla(pd.DataFrame(filas_b), "11_grupo_b_constante")
    print(tabla_b.to_string(index=False))

    print("\n[14] Complementariedad P vs EDS/EDP...")
    hi_df = {m: dfs_ramp_eval[m][dfs_ramp_eval[m]["episode_id"].isin(conj["HI_eval"])] for m in METODOS}
    largas_df = {m: dfs_ramp_eval[m][dfs_ramp_eval[m]["duration_min"].isin([720, 1440])] for m in METODOS}
    grupos_cd_df = {m: dfs_ramp_eval[m][dfs_ramp_eval[m]["grupo_temporal"].isin(["C", "D"])] for m in METODOS}
    b_df = {m: dfs_const_eval[m][dfs_const_eval[m]["episode_id"].isin(conj["B_eval"])] for m in METODOS}
    subconjuntos_comp = {"ramp_global": dfs_ramp_eval, "rampas_largas": largas_df, "alto_impacto": hi_df,
                          "grupos_C_D": grupos_cd_df, "constantes_todas": dfs_const_eval, "grupo_b": b_df}
    tabla_comp = _guardar_tabla(construir_complementariedad(subconjuntos_comp), "12_complementariedad")
    print(tabla_comp.to_string(index=False))

    print("\n[15] Comparaciones estadisticas pareadas (P vs EDS, P vs EDP, EDS vs EDP)...")
    tablas_comparacion = []
    for nombre_sub, dfs in subconjuntos_comp.items():
        hidden_col = "hidden_energy_kwh"
        tablas_comparacion.append(comparacion_pareada(dfs, hidden_col, nombre_sub))
    tabla_comparaciones = _guardar_tabla(pd.concat(tablas_comparacion, ignore_index=True), "14_comparaciones_estadisticas")
    print(tabla_comparaciones[tabla_comparaciones["subconjunto"] == "ramp_global"].to_string(index=False))

    print("\n[16] Analisis diagnostico por tercil de consumo (SIN implementar umbrales por tercil)...")
    tercil = analisis_por_tercil(dfs_ramp_eval, conj["ataques_ramp"], ctx["particion_val_raw"], conj["HI_eval"])
    print(tercil["dr_por_tercil"].to_string(index=False))
    conc_eds = concentracion_exclusiva_por_tercil(dfs_ramp_eval["P"], dfs_ramp_eval["EDS"], conj["ataques_ramp"], ctx["particion_val_raw"])
    conc_edp = concentracion_exclusiva_por_tercil(dfs_ramp_eval["P"], dfs_ramp_eval["EDP"], conj["ataques_ramp"], ctx["particion_val_raw"])
    print("Concentracion tercil detecciones exclusivas EDS:"); print(conc_eds.to_string(index=False))
    print("Concentracion tercil detecciones exclusivas EDP:"); print(conc_edp.to_string(index=False))

    print("\n[18] Evaluando criterios de exito (fijados antes de mirar resultados)...")
    fa_eval_tabla = tabla_fa[tabla_fa["periodo"] == "evaluacion"].set_index("metodo")
    dr_energ_p = tabla_ramp_global.set_index("metodo").loc["P", "energy_weighted_dr_pct"]
    retraso_p = tabla_ramp_global.set_index("metodo").loc["P", "retraso_mediano_min"]
    duraciones_largas_ids = dfs_ramp_eval["P"][dfs_ramp_eval["P"]["duration_min"].isin([720, 1440])]["episode_id"]
    hi_no_det_p = dfs_ramp_eval["P"][dfs_ramp_eval["P"]["episode_id"].isin(conj["HI_eval"]) & ~dfs_ramp_eval["P"]["induced_detected"]]
    largas_no_det_p = dfs_ramp_eval["P"][dfs_ramp_eval["P"]["episode_id"].isin(duraciones_largas_ids) & ~dfs_ramp_eval["P"]["induced_detected"]]

    tablas_criterios = []
    for metodo in ("EDS", "EDP"):
        pct_ai = 100 * dfs_ramp_eval[metodo].set_index("episode_id").loc[hi_no_det_p["episode_id"], "induced_detected"].mean() if len(hi_no_det_p) else float("nan")
        pct_largas = 100 * dfs_ramp_eval[metodo].set_index("episode_id").loc[largas_no_det_p["episode_id"], "induced_detected"].mean() if len(largas_no_det_p) else float("nan")
        dif_dr_energia = tabla_ramp_global.set_index("metodo").loc[metodo, "energy_weighted_dr_pct"] - dr_energ_p
        fa_eval = fa_eval_tabla.loc[metodo, "eventos_dia"]
        inflacion = fa_eval / OBJETIVO_FA
        retraso_metodo = tabla_ramp_global.set_index("metodo").loc[metodo, "retraso_mediano_min"]
        dif_retraso = retraso_metodo - retraso_p if not (np.isnan(retraso_metodo) or np.isnan(retraso_p)) else float("nan")
        n_buckets_mejora = _cuenta_buckets_con_mejora(filas_dur, metodo, "duration_min") + \
            _cuenta_buckets_con_mejora(filas_rho, metodo, "rho_final")
        mejora_multi = n_buckets_mejora > 1
        conc_tabla = conc_eds if metodo == "EDS" else conc_edp
        concentracion = len(conc_tabla) == 1 and conc_tabla.iloc[0].get("tercil_consumo") != "sin_casos" and conc_tabla["n"].iloc[0] > 0
        deteccion_exclusiva = tabla_comp[(tabla_comp["subconjunto"] == "ramp_global") & (tabla_comp["comparacion"] == f"P_vs_{metodo}")]["solo_ed"].iloc[0] > 0
        pct_retenido_p90 = tabla_p90.set_index("metodo").loc[metodo, "pct_retenido"]

        criterios = evaluar_criterios(metodo, pct_ai, pct_largas, dif_dr_energia, fa_eval, inflacion, dif_retraso,
                                       bool(mejora_multi), bool(concentracion), bool(deteccion_exclusiva), pct_retenido_p90)
        tablas_criterios.append(criterios)
        print(f"  {metodo}: {criterios.attrs['n_cumplidos']}/10 criterios | mayoria={criterios.attrs['mayoria']}")
    tabla_criterios = pd.concat(tablas_criterios, ignore_index=True)
    _guardar_tabla(tabla_criterios, "15_criterios_exito")

    print("\n[17] Generando casos visuales...")
    p_det = set(dfs_ramp_eval["P"].loc[dfs_ramp_eval["P"]["induced_detected"], "episode_id"])
    eds_det = set(dfs_ramp_eval["EDS"].loc[dfs_ramp_eval["EDS"]["induced_detected"], "episode_id"])
    edp_det = set(dfs_ramp_eval["EDP"].loc[dfs_ramp_eval["EDP"]["induced_detected"], "episode_id"])
    resultado_ramp = dfs_ramp_eval["P"].set_index("episode_id")
    casos = {
        "solo_EDS": sorted(eds_det - p_det - edp_det)[:3], "solo_EDP": sorted(edp_det - p_det - eds_det)[:3],
        "solo_P": sorted(p_det - eds_det - edp_det)[:3], "ambos_P_ED": sorted((p_det & (eds_det | edp_det)))[:3],
        "alto_impacto_ninguno": sorted(conj["HI_eval"] - p_det - eds_det - edp_det)[:3],
    }
    for etiqueta, ids in casos.items():
        for i, ep in enumerate(ids):
            figura_caso(ep, conj["ataques_ramp"], ctx["particion_val_raw"], meta_ramp, ctx["df_val"], umbrales,
                        ctx["inicio_particion"], f"16_caso_{etiqueta}_{i + 1}", etiqueta)
    for ep, prefijo, etiqueta in (("ramp_lineal__1440_0.30_37", "19", "caso_largo"), ("ramp_lineal__0120_0.50_08", "20", "caso_corto")):
        if ep in conj["R_eval"]:
            figura_caso(ep, conj["ataques_ramp"], ctx["particion_val_raw"], meta_ramp, ctx["df_val"], umbrales,
                        ctx["inicio_particion"], prefijo, etiqueta)

    print("\nGenerando figuras 1-15...")
    generar_figuras(ctx, div, cal, tabla_fa, tabla_ramp_global, pd.DataFrame(filas_dur), pd.DataFrame(filas_rho),
                     tabla_hi, tabla_p90, tabla_comp, tabla_comparaciones)

    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "ctx": ctx, "div": div, "umbrales": umbrales, "conj": conj, "meta_ramp": meta_ramp, "meta_const": meta_const,
        "dfs_ramp_eval": dfs_ramp_eval, "dfs_const_eval": dfs_const_eval, "tabla_ramp_global": tabla_ramp_global,
        "tabla_hi": tabla_hi, "tabla_p90": tabla_p90, "tabla_comp": tabla_comp, "tabla_comparaciones": tabla_comparaciones,
        "tabla_criterios": tabla_criterios,
    }


def generar_figuras(ctx, div, cal, tabla_fa, tabla_ramp_global, tabla_dur, tabla_rho, tabla_hi, tabla_p90,
                     tabla_comp, tabla_comparaciones) -> None:
    df_val = ctx["df_val"]; corte = div["corte_idx"]

    # 1) distribucion ED dev/eval
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, col in enumerate(["ED_signed", "ED_positive"]):
        axes[i].hist(df_val[col].to_numpy()[:corte], bins=60, color=GRIS, alpha=0.5, density=True, label="desarrollo")
        axes[i].hist(df_val[col].to_numpy()[corte:], bins=60, color=AZUL, alpha=0.5, density=True, label="evaluacion")
        axes[i].set_title(col); axes[i].legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "01_distribucion_ED_dev_eval")

    # 2) cola superior de ED limpia
    fig, ax = plt.subplots(figsize=(7, 5))
    for etiqueta, ini, fin, color in (("desarrollo", 0, corte, GRIS), ("evaluacion", corte, len(df_val), AZUL)):
        x = np.sort(df_val["ED_signed"].to_numpy()[ini:fin])
        p = np.linspace(90, 100, 200)
        ax.plot(p, np.percentile(x, p), color=color, label=etiqueta)
    ax.axhline(cal["umbrales"]["EDS"], color=ROJO, linestyle="--", label="umbral EDS")
    ax.set_xlabel("percentil"); ax.set_ylabel("ED_signed"); ax.legend(fontsize=8); ax.set_title("Cola superior de ED_signed")
    fig.tight_layout(); _guardar_fig(fig, "02_cola_superior_ED")

    # 3) falsas alarmas por metodo
    fig, ax = plt.subplots(figsize=(7, 5))
    ev = tabla_fa[tabla_fa["periodo"] == "evaluacion"]
    ax.bar(ev["metodo"], ev["eventos_dia"], color=[COLOR_METODO[m] for m in ev["metodo"]])
    ax.axhline(OBJETIVO_FA, color=GRIS_OSCURO, linestyle="--", label="objetivo 0.06")
    ax.set_ylabel("eventos falsos/dia (evaluacion)"); ax.legend(fontsize=8); ax.set_title("Falsas alarmas por metodo")
    fig.tight_layout(); _guardar_fig(fig, "03_falsas_alarmas_por_metodo")

    fa_dist = pd.read_csv(TABLAS_DIR / "03b_falsas_alarmas_distribucion_eval.csv")
    # 4) por mes
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METODOS:
        sub = fa_dist[(fa_dist["metodo"] == m) & (fa_dist["agrupacion"] == "mes")].sort_values("valor")
        ax.plot(sub["valor"], sub["pct_ventanas_activas"], marker="o", color=COLOR_METODO[m], label=m)
    ax.set_xlabel("mes"); ax.set_ylabel("% ventanas activas"); ax.legend(fontsize=8); ax.set_title("Falsas alarmas por mes (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "04_falsas_alarmas_por_mes")

    # 5) por hora
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METODOS:
        sub = fa_dist[(fa_dist["metodo"] == m) & (fa_dist["agrupacion"] == "hora")].sort_values("valor")
        ax.plot(sub["valor"], sub["pct_ventanas_activas"], marker="o", color=COLOR_METODO[m], label=m)
    ax.set_xlabel("hora"); ax.set_ylabel("% ventanas activas"); ax.legend(fontsize=8); ax.set_title("Falsas alarmas por hora (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "05_falsas_alarmas_por_hora")

    # 6) por tercil de consumo
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.25
    terciles = ["bajo", "medio", "alto"]
    for i, m in enumerate(METODOS):
        sub = fa_dist[(fa_dist["metodo"] == m) & (fa_dist["agrupacion"] == "tercil_consumo")].set_index("valor").reindex(terciles)
        ax.bar(np.arange(3) + i * width, sub["pct_ventanas_activas"], width=width, color=COLOR_METODO[m], label=m)
    ax.set_xticks(np.arange(3) + width); ax.set_xticklabels(terciles)
    ax.set_ylabel("% ventanas activas"); ax.legend(fontsize=8); ax.set_title("Falsas alarmas por tercil de consumo (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "06_falsas_alarmas_por_consumo")

    # 7) DR ramp
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_ramp_global["metodo"], tabla_ramp_global["induced_DR_pct"], color=[COLOR_METODO[m] for m in tabla_ramp_global["metodo"]])
    ax.set_ylabel("induced_DR (%)"); ax.set_title("DR de ramp (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "07_DR_ramp")

    # 8) DR por duracion
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in METODOS:
        sub = tabla_dur[tabla_dur["metodo"] == m].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", color=COLOR_METODO[m], label=m)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)"); ax.legend(fontsize=8); ax.set_title("DR por duracion")
    fig.tight_layout(); _guardar_fig(fig, "08_DR_por_duracion")

    # 9) DR por severidad
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METODOS:
        sub = tabla_rho[tabla_rho["metodo"] == m].sort_values("rho_final", ascending=False)
        ax.plot(sub["rho_final"].astype(str), sub["induced_DR_pct"], marker="o", color=COLOR_METODO[m], label=m)
    ax.set_xlabel("rho_final"); ax.set_ylabel("induced_DR (%)"); ax.legend(fontsize=8); ax.set_title("DR por severidad")
    fig.tight_layout(); _guardar_fig(fig, "09_DR_por_severidad")

    # 10) DR energetico
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_ramp_global["metodo"], tabla_ramp_global["energy_weighted_dr_pct"], color=[COLOR_METODO[m] for m in tabla_ramp_global["metodo"]])
    ax.set_ylabel("DR ponderado por energia (%)"); ax.set_title("DR energetico de ramp")
    fig.tight_layout(); _guardar_fig(fig, "10_DR_energetico")

    # 11) recuperacion de alto impacto
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_hi["metodo"], tabla_hi["pct_recuperado"], color=[COLOR_METODO[m] for m in tabla_hi["metodo"]])
    ax.set_ylabel("% recuperado"); ax.set_title("Rampas de alto impacto no detectadas por P (original)")
    fig.tight_layout(); _guardar_fig(fig, "11_recuperacion_alto_impacto")

    # 12) retencion de casos P90
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_p90["metodo"], tabla_p90["pct_retenido"], color=[COLOR_METODO[m] for m in tabla_p90["metodo"]])
    ax.set_ylabel("% retenido"); ax.set_title("Retencion de los 145 casos diagnosticos (P90)")
    fig.tight_layout(); _guardar_fig(fig, "12_retencion_casos_p90")

    # 13) complementariedad
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = tabla_comp[tabla_comp["subconjunto"] == "ramp_global"]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["detectado_ambos"], width=0.2, color=VERDE, label="ambos")
    ax.bar(x, sub["solo_puntual"], width=0.2, color=GRIS_OSCURO, label="solo P")
    ax.bar(x + 0.2, sub["solo_ed"], width=0.2, color=AZUL, label="solo ED")
    ax.set_xticks(x); ax.set_xticklabels(sub["comparacion"]); ax.legend(fontsize=8); ax.set_title("Complementariedad (ramp global)")
    fig.tight_layout(); _guardar_fig(fig, "13_complementariedad")

    # 14) retraso
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_ramp_global["metodo"], tabla_ramp_global["retraso_mediano_min"], color=[COLOR_METODO[m] for m in tabla_ramp_global["metodo"]])
    ax.set_ylabel("retraso mediano (min)"); ax.set_title("Retraso de deteccion (ramp)")
    fig.tight_layout(); _guardar_fig(fig, "14_retraso")

    # 15) energia oculta antes de detectar
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_ramp_global["metodo"], tabla_ramp_global["hidden_energy_before_detection_total_kwh"], color=[COLOR_METODO[m] for m in tabla_ramp_global["metodo"]])
    ax.set_ylabel("energia oculta antes de deteccion (kWh)"); ax.set_title("Energia oculta antes de detectar (total, ramp)")
    fig.tight_layout(); _guardar_fig(fig, "15_energia_oculta_antes_deteccion")

    print(f"  figuras 1-15 guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
