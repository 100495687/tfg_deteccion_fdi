"""Experimento: acumulacion temporal mediante CUSUM sobre el score `relative_kw`.

Comprueba si un CUSUM unilateral, aplicado sobre la misma secuencia observable que ya usa el
detector puntual (relative_kw, ventana=360, stride=60, checkpoint existente), recupera fraudes
moderados y sostenidos que hoy elevan el score sin cruzar el umbral puntual (Grupo B del
analisis de detectabilidad).

Restriccion importante: CUSUM solo ve `s_t = relative_kw_score_t` de la rama que esta
evaluando (limpia o atacada). Nunca usa score_delta, el score limpio de la misma ventana
atacada, energia realmente ocultada, ni metadatos del ataque -- esas variables solo se usan
despues para analizar resultados.

No entrena nada, no cambia ventana/stride/relative_kw/ataques/episodios/normalizacion.
Reutiliza sin modificar funciones de analisis_detectabilidad (cargar_todo, contexto_e_impacto,
construir_tabla_episodios, clasificar_grupos, _energia_antes_despues), de experimento_stride
(_ventanas_solapadas, mcnemar_p, bootstrap_pareado, contar_eventos) y episodes.wilson_ci.

Ejecutable de forma aislada:
    python -m src.experimento_cusum
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analisis_detectabilidad as ad
from src.episodes import wilson_ci
from src.experimento_stride import _ventanas_solapadas, bootstrap_pareado, mcnemar_p

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "experimento_cusum"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "experimento_cusum"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDE = 60
SCORE = "relative_kw"
EPS_Z = 1e-6  # suelo del denominador de la estandarizacion robusta (evita division por ~0
              # cuando la mediana de desviaciones absolutas del score limpio fuese casi nula)
VALORES_K = [0.00, 0.25, 0.50, 0.75, 1.00]
OBJETIVOS_EVENTOS_DIA = {"sec_0.05": 0.05, "principal_0.06": 0.06, "sec_0.10": 0.10, "sec_0.25": 0.25}
OBJETIVO_PRINCIPAL = "principal_0.06"
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
COLOR_K = {0.00: "#d0e3f7", 0.25: "#8fb8e8", 0.50: AZUL, 0.75: AZUL_OSCURO, 1.00: "#0a2e57"}


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 3) Estandarizacion robusta del score, ajustada UNICAMENTE con scores limpios
# ==========================================================================================

def ajustar_estandarizacion(scores_clean: np.ndarray) -> dict:
    median_clean = float(np.median(scores_clean))
    mad_clean = float(np.median(np.abs(scores_clean - median_clean)))
    robust_scale = 1.4826 * mad_clean
    denom = max(robust_scale, EPS_Z)
    return {"median_clean": median_clean, "mad_clean": mad_clean, "robust_scale": robust_scale,
            "denom_efectivo": denom, "epsilon": EPS_Z}


def estandarizar(scores: np.ndarray, params: dict) -> np.ndarray:
    return (scores - params["median_clean"]) / params["denom_efectivo"]


# ==========================================================================================
# 4) CUSUM unilateral (con reset tras alarma) -- nucleo puro, sin informacion contrafactual
# ==========================================================================================

def simular_cusum(z: np.ndarray, k_param: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """C_t = max(0, C_{t-1} + z_t - k_param); alarma si C_t > h; tras alarma, C_t se resetea a
    0 ANTES del siguiente paso (el valor devuelto en C[t] es el que disparo la alarma, previo
    al reset -- asi se puede inspeccionar cuanto llego a acumular). Recursion secuencial pura
    en `z`: nunca usa score_delta ni el score limpio de la ventana atacada."""
    n = len(z)
    C = np.empty(n, dtype=np.float64)
    alarm = np.zeros(n, dtype=bool)
    c_prev = 0.0
    for t in range(n):
        c_t = c_prev + z[t] - k_param
        if c_t < 0.0:
            c_t = 0.0
        if c_t > h:
            alarm[t] = True
            C[t] = c_t
            c_prev = 0.0  # politica de reset: tras una alarma, el acumulador vuelve a 0
        else:
            C[t] = c_t
            c_prev = c_t
    return C, alarm


def _excursiones(C: np.ndarray, stride: int) -> np.ndarray:
    """Duraciones (min) de rachas maximales consecutivas con C_t > 0."""
    activo = C > 0
    duraciones, contador = [], 0
    for a in activo:
        if a:
            contador += 1
        else:
            if contador > 0:
                duraciones.append(contador * stride)
            contador = 0
    if contador > 0:
        duraciones.append(contador * stride)
    return np.array(duraciones) if duraciones else np.array([0.0])


# ==========================================================================================
# 5-6) Barrido de k, calibracion de h por objetivo de falsos eventos/dia (bisection, SOLO
#      sobre la secuencia limpia; nunca se calibra usando ataques)
# ==========================================================================================

def calibrar_h(z_clean: np.ndarray, k_param: float, objetivo: float, n_dias: float,
                max_iter: int = 40) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Busqueda por biseccion (la tasa de eventos/dia decrece monotonamente con h para un
    CUSUM con reset). Expande el limite superior hasta acotar el objetivo, despues bisecta."""
    h_lo, h_hi = 0.0, 1.0
    tasa_hi = simular_cusum(z_clean, k_param, h_hi)[1].sum() / n_dias
    intentos = 0
    while tasa_hi > objetivo and intentos < 30:
        h_hi *= 2
        tasa_hi = simular_cusum(z_clean, k_param, h_hi)[1].sum() / n_dias
        intentos += 1

    lo, hi = h_lo, h_hi
    mejor_h, mejor_C, mejor_alarm = h_hi, None, None
    mejor_dif = np.inf
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        C_mid, alarm_mid = simular_cusum(z_clean, k_param, mid)
        tasa_mid = alarm_mid.sum() / n_dias
        dif = abs(tasa_mid - objetivo)
        if dif < mejor_dif:
            mejor_h, mejor_dif, mejor_C, mejor_alarm = mid, dif, C_mid, alarm_mid
        if tasa_mid > objetivo:
            lo = mid
        else:
            hi = mid
    tasa_final = mejor_alarm.sum() / n_dias
    return mejor_h, tasa_final, mejor_C, mejor_alarm


def tabla_calibracion(z_clean: np.ndarray, n_dias: float, fin_grid: np.ndarray) -> tuple[pd.DataFrame, dict]:
    filas = []
    resultados = {}  # (k_param, objetivo_nombre) -> dict con h, C_clean_full, alarm_clean_full
    for k_param in VALORES_K:
        for objetivo_nombre, objetivo_valor in OBJETIVOS_EVENTOS_DIA.items():
            h, tasa_eventos, C_full, alarm_full = calibrar_h(z_clean, k_param, objetivo_valor, n_dias)
            excursiones = _excursiones(C_full, STRIDE)
            ts_alarma = pd.to_datetime(fin_grid)[alarm_full]
            if len(ts_alarma) > 1:
                tiempo_medio_min = float(np.diff(ts_alarma).astype("timedelta64[m]").astype(float).mean())
            else:
                tiempo_medio_min = float("nan")
            filas.append({
                "k": k_param, "objetivo_nombre": objetivo_nombre, "objetivo_valor": objetivo_valor,
                "h": h, "tasa_eventos_dia_real": tasa_eventos, "tasa_ventanas_dia_real": float(alarm_full.sum()) / n_dias,
                "n_eventos_totales": int(alarm_full.sum()),
                "duracion_media_excursion_min": float(excursiones.mean()), "duracion_max_excursion_min": float(excursiones.max()),
                "tiempo_medio_entre_alarmas_min": tiempo_medio_min,
            })
            resultados[(k_param, objetivo_nombre)] = {"h": h, "C_clean_full": C_full, "alarm_clean_full": alarm_full}
    return pd.DataFrame(filas), resultados


# ==========================================================================================
# 7-9) Estado previo al ataque + ramas limpia/atacada + deteccion inducida contrafactual
# ==========================================================================================

def _indexar_scores_atacados(tabla_ventanas: pd.DataFrame) -> dict[str, dict[int, float]]:
    d = {}
    for episode_id, g in tabla_ventanas.groupby("episode_id"):
        d[episode_id] = dict(zip(g["k"], g["attacked_score"]))
    return d


def simular_episodios(ataques: dict, scores_atacados_idx: dict, n_ventanas_grid: int,
                       C_clean_full: np.ndarray, alarm_clean_full: np.ndarray, fin_grid: np.ndarray,
                       params_estandarizacion: dict, k_param: float, h: float) -> pd.DataFrame:
    """Para cada episodio: recupera el estado limpio de CUSUM en la ultima ventana anterior a
    la primera ventana afectada (0.0 si el episodio empieza al inicio de la serie), y desde
    ahi simula la rama ATACADA sobre exactamente las mismas ventanas que la rama limpia (que
    ya esta precalculada en C_clean_full/alarm_clean_full -- no se recalcula)."""
    filas = []
    for episode_id, a in ataques.items():
        ks = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_ventanas_grid))
        k_before = ks[0] - 1
        c_prev = float(C_clean_full[k_before]) if k_before >= 0 else 0.0

        sub = scores_atacados_idx[episode_id]
        C_att, alarm_att = [], []
        for k in ks:
            s_att = sub[k]
            z_att = (s_att - params_estandarizacion["median_clean"]) / params_estandarizacion["denom_efectivo"]
            c_t = c_prev + z_att - k_param
            c_t = max(0.0, c_t)
            es_alarma = c_t > h
            C_att.append(c_t); alarm_att.append(es_alarma)
            c_prev = 0.0 if es_alarma else c_t
        C_att = np.array(C_att); alarm_att = np.array(alarm_att, dtype=bool)

        C_lim = C_clean_full[ks]
        alarm_lim = alarm_clean_full[ks]
        induced = alarm_att & ~alarm_lim
        idx_ind = np.where(induced)[0]
        ts_induced = pd.Timestamp(fin_grid[ks[idx_ind[0]]]) if len(idx_ind) else pd.NaT

        filas.append({
            "episode_id": episode_id, "n_affected_windows": len(ks),
            "max_C_attacked": float(C_att.max()), "max_C_clean": float(C_lim.max()),
            "raw_detected": bool(alarm_att.any()),
            "induced_detected": bool(induced.any()),
            "first_induced_alarm_timestamp": ts_induced,
            "preexisting_alarm": bool((alarm_att & alarm_lim).any()),
            "suppressed_alarm": bool((alarm_lim & ~alarm_att).any()),
        })
    return pd.DataFrame(filas)


def enriquecer_episodios_cusum(sim: pd.DataFrame, ataques: dict, master_csv: pd.DataFrame,
                                particion_val_raw: np.ndarray, contexto_impacto: pd.DataFrame,
                                k_param: float, h: float) -> pd.DataFrame:
    meta_cols = ["episode_id", "family", "parameter", "duration_min", "repetition_idx", "attack_start", "attack_end"]
    df = master_csv[meta_cols].merge(sim, on="episode_id").merge(contexto_impacto, on="episode_id")

    sin_induced = ~df["induced_detected"]
    df["induced_detection_delay_min"] = (df["first_induced_alarm_timestamp"] - df["attack_start"]) / pd.Timedelta(minutes=1)
    df.loc[sin_induced, "induced_detection_delay_min"] = np.nan
    durante = (df["first_induced_alarm_timestamp"] <= df["attack_end"]).astype(object)
    despues = (df["first_induced_alarm_timestamp"] > df["attack_end"]).astype(object)
    durante[sin_induced] = np.nan
    despues[sin_induced] = np.nan
    df["detected_during_attack"] = durante
    df["detected_after_attack"] = despues

    filas_energia = [ad._energia_antes_despues(row, ataques, particion_val_raw) for row in df.itertuples()]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(filas_energia)], axis=1)

    df["k_param"] = k_param
    df["h"] = h
    return df


# ==========================================================================================
# 11) Resumen de metricas (episodios + energia), reutilizable por grupo
# ==========================================================================================

def resumen_metricas(grupo: pd.DataFrame) -> dict:
    n = len(grupo)
    n_ind = int(grupo["induced_detected"].sum())
    ci = wilson_ci(n_ind, n) if n else (float("nan"), float("nan"))
    con_ind = grupo[grupo["induced_detected"] == True]  # noqa: E712
    total_hidden = grupo["hidden_energy_kwh"].sum()
    detected_hidden = grupo.loc[grupo["induced_detected"], "hidden_energy_kwh"].sum()
    after_total = grupo["hidden_energy_after_detection_kwh"].sum()
    return {
        "n_episodios": n, "n_detectados": n_ind,
        "induced_DR_pct": 100 * n_ind / n if n else float("nan"),
        "induced_DR_ci95_low": ci[0], "induced_DR_ci95_high": ci[1],
        "energy_weighted_dr_pct": 100 * detected_hidden / total_hidden if total_hidden > ad.EPS_ENERGIA else float("nan"),
        "timely_energy_detection_rate_pct": 100 * after_total / total_hidden if total_hidden > ad.EPS_ENERGIA else float("nan"),
        "retraso_mediano_min": float(con_ind["induced_detection_delay_min"].median()) if len(con_ind) else float("nan"),
        "pct_detectado_durante_ataque": 100 * (con_ind["detected_during_attack"] == True).mean() if len(con_ind) else float("nan"),  # noqa: E712
        "hidden_energy_before_detection_total_kwh": float(grupo["hidden_energy_before_detection_kwh"].sum()),
        "pct_recuperable_medio": float(grupo["recoverable_hidden_energy_pct"].mean()),
    }


def tablas_resumen(df: pd.DataFrame, umbrales_tabla: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tablas = {}
    filas = []
    for k_param, grupo in df.groupby("k_param"):
        for etiqueta, g in (("todas", grupo), ("sin_recorte_picos", grupo[grupo["family"] != "recorte_picos"])):
            fila = {"k": k_param, "familias": etiqueta}
            fila.update(resumen_metricas(g))
            u = umbrales_tabla[(umbrales_tabla["k"] == k_param) & (umbrales_tabla["objetivo_nombre"] == OBJETIVO_PRINCIPAL)].iloc[0]
            fila["h"] = u["h"]; fila["falsos_eventos_dia"] = u["tasa_eventos_dia_real"]; fila["falsas_ventanas_dia"] = u["tasa_ventanas_dia_real"]
            filas.append(fila)
    tablas["global"] = pd.DataFrame(filas)

    for nombre_grupo, col in (("familia", "family"), ("duracion", "duration_min"), ("severidad", "parameter")):
        filas = []
        for (k_param, val), grupo in df.groupby(["k_param", col]):
            fila = {"k": k_param, col: val}
            fila.update(resumen_metricas(grupo))
            filas.append(fila)
        tablas[f"por_{nombre_grupo}"] = pd.DataFrame(filas)
    return tablas


# ==========================================================================================
# 12) Grupo B (responde pero queda lejos del umbral puntual) -- objetivo principal
# ==========================================================================================

def analizar_grupo_b(df_cusum_principal: pd.DataFrame, grupo_b_ids: set) -> dict[str, pd.DataFrame]:
    sub = df_cusum_principal[df_cusum_principal["episode_id"].isin(grupo_b_ids)]
    tablas = {}
    filas = []
    for k_param, g in sub.groupby("k_param"):
        fila = {"k": k_param}
        fila.update(resumen_metricas(g))
        filas.append(fila)
    tablas["global"] = pd.DataFrame(filas)
    for nombre_grupo, col in (("duracion", "duration_min"), ("severidad", "parameter"), ("familia", "family")):
        filas = []
        for (k_param, val), g in sub.groupby(["k_param", col]):
            fila = {"k": k_param, col: val}
            fila.update(resumen_metricas(g))
            filas.append(fila)
        tablas[f"por_{nombre_grupo}"] = pd.DataFrame(filas)
    return tablas


# ==========================================================================================
# 13) Ataques no detectados de alto impacto (reutiliza el CSV del analisis de detectabilidad)
# ==========================================================================================

def analizar_alto_impacto(df_cusum_principal: pd.DataFrame, alto_impacto_ids: set) -> pd.DataFrame:
    sub = df_cusum_principal[df_cusum_principal["episode_id"].isin(alto_impacto_ids)]
    filas = []
    for k_param, g in sub.groupby("k_param"):
        n = len(g)
        n_det = int(g["induced_detected"].sum())
        hidden_total = g["hidden_energy_kwh"].sum()
        hidden_det = g.loc[g["induced_detected"], "hidden_energy_kwh"].sum()
        rc = g[g["family"] == "reduccion_constante"]
        d720_1440 = g[g["duration_min"].isin([720, 1440])]
        con_ind = g[g["induced_detected"] == True]  # noqa: E712
        filas.append({
            "k": k_param, "n_episodios": n, "n_recuperados": n_det,
            "pct_recuperados": 100 * n_det / n if n else float("nan"),
            "pct_energia_cubierta": 100 * hidden_det / hidden_total if hidden_total > ad.EPS_ENERGIA else float("nan"),
            "n_reduccion_constante": len(rc), "n_reduccion_constante_recuperados": int(rc["induced_detected"].sum()),
            "n_720_1440": len(d720_1440), "n_720_1440_recuperados": int(d720_1440["induced_detected"].sum()),
            "retraso_mediano_min": float(con_ind["induced_detection_delay_min"].median()) if len(con_ind) else float("nan"),
            "hidden_energy_before_detection_media_kwh": float(g["hidden_energy_before_detection_kwh"].mean()),
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# 14) Comparacion pareada CUSUM vs detector puntual
# ==========================================================================================

def comparacion_pareada(df_cusum_principal: pd.DataFrame, df_punto: pd.DataFrame, alto_impacto_ids: set) -> pd.DataFrame:
    punto = df_punto.set_index("episode_id")["induced_detected"].astype(bool)
    filas = []
    for k_param, g in df_cusum_principal.groupby("k_param"):
        cusum = g.set_index("episode_id")["induced_detected"].astype(bool)
        meta = g.set_index("episode_id")[["family", "duration_min"]]
        comb = pd.DataFrame({"cusum": cusum, "puntual": punto.reindex(cusum.index)}).join(meta)
        comb["grupo_b"] = comb.index.isin(GRUPO_B_IDS_GLOBAL)
        comb["alto_impacto"] = comb.index.isin(alto_impacto_ids)

        subconjuntos = {
            "global": comb, "30min": comb[comb["duration_min"] == 30], "60min": comb[comb["duration_min"] == 60],
            "120min": comb[comb["duration_min"] == 120], "240_360min": comb[comb["duration_min"].isin([240, 360])],
            "720_1440min": comb[comb["duration_min"].isin([720, 1440])],
            "grupo_b": comb[comb["grupo_b"]], "alto_impacto": comb[comb["alto_impacto"]],
        }
        for fam in comb["family"].unique():
            subconjuntos[f"familia_{fam}"] = comb[comb["family"] == fam]

        for nombre_sub, sub in subconjuntos.items():
            det_c, det_p = sub["cusum"].to_numpy(), sub["puntual"].to_numpy()
            solo_c = int((det_c & ~det_p).sum()); solo_p = int((~det_c & det_p).sum())
            ambos = int((det_c & det_p).sum()); ninguno = int((~det_c & ~det_p).sum())
            p = mcnemar_p(solo_c, solo_p)
            ci_low, ci_high = bootstrap_pareado(det_c.astype(float), det_p.astype(float)) if len(sub) else (float("nan"), float("nan"))
            filas.append({
                "k": k_param, "subconjunto": nombre_sub, "n_episodios": len(sub),
                "detectado_solo_cusum": solo_c, "detectado_solo_puntual": solo_p,
                "detectado_con_ambos": ambos, "detectado_con_ninguno": ninguno,
                "DR_cusum_pct": 100 * det_c.mean() if len(sub) else float("nan"),
                "DR_puntual_pct": 100 * det_p.mean() if len(sub) else float("nan"),
                "mcnemar_p_valor": p, "bootstrap_ci95_diferencia_low": ci_low, "bootstrap_ci95_diferencia_high": ci_high,
                "diferencia_significativa_p<0.05": p < 0.05,
            })
    return pd.DataFrame(filas)


GRUPO_B_IDS_GLOBAL: set = set()  # se rellena en main() antes de llamar a comparacion_pareada


# ==========================================================================================
# 16) Tabla de decision para elegir k (NO solo por DR global)
# ==========================================================================================

def tabla_decision(tablas_resumen_principal: pd.DataFrame, tabla_grupo_b: pd.DataFrame,
                    tabla_alto_impacto: pd.DataFrame, pareada: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for k_param in VALORES_K:
        glob = tablas_resumen_principal[(tablas_resumen_principal["k"] == k_param) & (tablas_resumen_principal["familias"] == "sin_recorte_picos")].iloc[0]
        gb = tabla_grupo_b[tabla_grupo_b["k"] == k_param].iloc[0]
        ai = tabla_alto_impacto[tabla_alto_impacto["k"] == k_param].iloc[0]
        retencion = pareada[(pareada["k"] == k_param) & (pareada["subconjunto"] == "global")].iloc[0]
        # retencion de lo que el puntual ya detectaba: ambos / (ambos+solo_puntual)
        ya_detectado_puntual = retencion["detectado_con_ambos"] + retencion["detectado_solo_puntual"]
        pct_retenido = 100 * retencion["detectado_con_ambos"] / ya_detectado_puntual if ya_detectado_puntual else float("nan")
        filas.append({
            "k": k_param, "h": glob["h"], "falsos_eventos_dia": glob["falsos_eventos_dia"],
            "episode_DR_pct": glob["induced_DR_pct"], "energy_weighted_DR_pct": glob["energy_weighted_dr_pct"],
            "timely_energy_DR_pct": glob["timely_energy_detection_rate_pct"],
            "grupo_b_recuperacion_pct": gb["induced_DR_pct"], "alto_impacto_recuperacion_pct": ai["pct_recuperados"],
            "retraso_mediano_min": glob["retraso_mediano_min"],
            "pct_retenido_de_lo_ya_detectado_por_puntual": pct_retenido,
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# 15/17) Figuras de ejemplos temporales
# ==========================================================================================

def figura_evolucion_temporal(episode_id: str, ataques: dict, particion_val_raw: np.ndarray, grid_60: dict,
                               scores_atacados_idx: dict, scores_limpios_60: np.ndarray, params_est: dict,
                               C_clean_full: np.ndarray, k_param: float, h: float, inicio_particion,
                               nombre_archivo: str, titulo_extra: str) -> None:
    a = ataques[episode_id]
    offset, dur = a["offset_global"], a["duracion"]
    n_ventanas_grid = grid_60["X"].shape[0]
    ks = sorted(_ventanas_solapadas(offset, dur, STRIDE, n_ventanas_grid))
    fin = pd.to_datetime(grid_60["fin"])

    sub = scores_atacados_idx[episode_id]
    s_att = np.array([sub[k] for k in ks])
    s_lim = scores_limpios_60[ks]
    z_att = (s_att - params_est["median_clean"]) / params_est["denom_efectivo"]

    k_before = ks[0] - 1
    c_prev = float(C_clean_full[k_before]) if k_before >= 0 else 0.0
    C_att = []
    for zt in z_att:
        c_t = max(0.0, c_prev + zt - k_param)
        C_att.append(c_t)
        c_prev = 0.0 if c_t > h else c_t
    C_att = np.array(C_att)
    C_lim = C_clean_full[ks]
    t = fin[ks]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8), sharex=True)
    axes[0].plot(t, s_lim, color=GRIS, marker="o", markersize=3, label="relative_kw limpio")
    axes[0].plot(t, s_att, color=ROJO, marker="o", markersize=3, label="relative_kw atacado")
    axes[0].set_ylabel("relative_kw"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"{episode_id} | {a['family']} ({a['parameter']}) | {dur} min | k={k_param} | {titulo_extra}", fontsize=9.5)

    axes[1].plot(t, (s_lim - params_est["median_clean"]) / params_est["denom_efectivo"], color=GRIS, marker="o", markersize=3, label="z_t limpio")
    axes[1].plot(t, z_att, color=ROJO, marker="o", markersize=3, label="z_t atacado")
    axes[1].axhline(k_param, color=GRIS_OSCURO, linestyle=":", linewidth=1, label=f"k={k_param}")
    axes[1].set_ylabel("z_t"); axes[1].legend(fontsize=8)

    axes[2].plot(t, C_lim, color=GRIS, marker="o", markersize=3, label="C_t limpio")
    axes[2].plot(t, C_att, color=ROJO, marker="o", markersize=3, label="C_t atacado")
    axes[2].axhline(h, color=GRIS_OSCURO, linestyle="--", linewidth=1, label="h")
    axes[2].axvspan(inicio_particion + pd.Timedelta(minutes=offset), inicio_particion + pd.Timedelta(minutes=offset + dur),
                     color=VERDE, alpha=0.08)
    axes[2].set_ylabel("C_t"); axes[2].legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, nombre_archivo)


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    global GRUPO_B_IDS_GLOBAL
    print("Cargando episodios exactos, rejilla stride=60 y scores relative_kw (reutilizados, sin reentrenar)...")
    ctx = ad.cargar_todo()
    ataques, master_csv = ctx["ataques"], ctx["master_csv"]
    particion_val_raw = ctx["particion_val_raw"]
    serie_antes = particion_val_raw.copy()
    tabla_ventanas, scores_limpios_60 = ctx["tabla_ventanas"], ctx["scores_limpios_60"]
    grid_60 = ctx["grid_60"]
    n_dias = ctx["n_dias"]
    n_ventanas_grid = grid_60["X"].shape[0]
    inicio_particion = ctx["particion_val"].index[0]

    # ---- detector puntual de referencia (360/60/relative_kw, 0.06 eventos/dia) ----
    umbral_punto, _, _ = ad.calibrar_umbral_por_eventos(scores_limpios_60, n_dias, 0.06)
    contexto_impacto = ad.contexto_e_impacto(ataques, particion_val_raw)
    df_punto = ad.clasificar_grupos(ad.construir_tabla_episodios(
        tabla_ventanas, ataques, master_csv, particion_val_raw, contexto_impacto, umbral_punto))
    grupo_b_ids = set(df_punto[df_punto["detectability_group"] == "B"]["episode_id"])
    GRUPO_B_IDS_GLOBAL = grupo_b_ids
    alto_impacto = pd.read_csv(BASE_DIR / "results" / "tables" / "analisis_detectabilidad" / "09_ataques_no_detectados_alto_impacto.csv")
    alto_impacto_ids = set(alto_impacto["episode_id"])
    print(f"Referencia puntual: umbral={umbral_punto:.5f} | DR={100*df_punto['induced_detected'].mean():.2f}% | "
          f"Grupo B={len(grupo_b_ids)} episodios | alto impacto={len(alto_impacto_ids)} episodios\n")

    # ---- [3] estandarizacion robusta (SOLO scores limpios) ----
    params_est = ajustar_estandarizacion(scores_limpios_60)
    z_clean_full = estandarizar(scores_limpios_60, params_est)
    assert not np.any(np.isnan(z_clean_full)) and not np.any(np.isinf(z_clean_full))
    print(f"[3] median_clean={params_est['median_clean']:.5f} | mad_clean={params_est['mad_clean']:.5f} | "
          f"robust_scale={params_est['robust_scale']:.5f} (denom efectivo={params_est['denom_efectivo']:.5f})")
    desc_z = pd.DataFrame([{
        "median_clean": params_est["median_clean"], "mad_clean": params_est["mad_clean"],
        "robust_scale": params_est["robust_scale"], "epsilon": EPS_Z,
        "z_media": float(z_clean_full.mean()), "z_std": float(z_clean_full.std()), "z_mediana": float(np.median(z_clean_full)),
        "z_p95": float(np.percentile(z_clean_full, 95)), "z_p99": float(np.percentile(z_clean_full, 99)),
        "z_min": float(z_clean_full.min()), "z_max": float(z_clean_full.max()),
    }])
    _guardar_tabla(desc_z, "00_estandarizacion_descriptivos")

    # ---- [5-6] barrido de k, calibracion de h ----
    print("\n[5-6] Calibrando h por (k, objetivo de falsos eventos/dia)...")
    tabla_calib, calib_idx = tabla_calibracion(z_clean_full, n_dias, grid_60["fin"])
    _guardar_tabla(tabla_calib, "01_calibracion_kh")
    print(tabla_calib[tabla_calib["objetivo_nombre"] == OBJETIVO_PRINCIPAL].to_string(index=False))

    # ---- [7-9] simulacion por episodio, para las 20 combinaciones (k, objetivo) ----
    print("\n[7-9] Simulando ramas limpia/atacada por episodio (20 combinaciones k x objetivo)...")
    scores_atacados_idx = _indexar_scores_atacados(tabla_ventanas)
    tablas_por_combo = []
    for k_param in VALORES_K:
        for objetivo_nombre in OBJETIVOS_EVENTOS_DIA:
            info = calib_idx[(k_param, objetivo_nombre)]
            sim = simular_episodios(ataques, scores_atacados_idx, n_ventanas_grid, info["C_clean_full"],
                                     info["alarm_clean_full"], grid_60["fin"], params_est, k_param, info["h"])
            enr = enriquecer_episodios_cusum(sim, ataques, master_csv, particion_val_raw, contexto_impacto, k_param, info["h"])
            enr["objetivo_nombre"] = objetivo_nombre
            tablas_por_combo.append(enr)
    resultados_todo = pd.concat(tablas_por_combo, ignore_index=True)
    _guardar_tabla(resultados_todo, "02_episode_detection_results_cusum")
    print(f"OK -- {len(resultados_todo)} filas ({len(ataques)} episodios x {len(VALORES_K)} k x {len(OBJETIVOS_EVENTOS_DIA)} objetivos)")

    df_cusum_principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL].copy()

    # ---- [11] tablas resumen (SOLO el objetivo principal -- resultados_todo mezcla los 4
    #      objetivos de falsos eventos/dia, y cada uno tiene su propio h; agregarlos juntos
    #      mezclaria episodios evaluados con umbrales distintos) ----
    print("\n[11] Construyendo tablas de metricas (global/familia/duracion/severidad)...")
    tablas_res = tablas_resumen(df_cusum_principal, tabla_calib)
    for nombre, t in tablas_res.items():
        _guardar_tabla(t, f"03_resumen_{nombre}")
    print(tablas_res["global"][["k", "familias", "n_detectados", "induced_DR_pct", "energy_weighted_dr_pct",
                                 "timely_energy_detection_rate_pct", "retraso_mediano_min"]].to_string(index=False))

    # ---- [12] Grupo B ----
    print("\n[12] Analizando recuperacion del Grupo B (objetivo principal del experimento)...")
    tablas_gb = analizar_grupo_b(df_cusum_principal, grupo_b_ids)
    for nombre, t in tablas_gb.items():
        _guardar_tabla(t, f"04_grupo_b_{nombre}")
    print(tablas_gb["global"].to_string(index=False))

    # ---- [13] alto impacto ----
    print("\n[13] Analizando recuperacion de ataques no detectados de alto impacto...")
    tabla_ai = analizar_alto_impacto(df_cusum_principal, alto_impacto_ids)
    _guardar_tabla(tabla_ai, "alto_impacto_recuperado_por_cusum")
    print(tabla_ai.to_string(index=False))

    # ---- [14] comparacion pareada ----
    print("\n[14] Comparacion pareada CUSUM vs detector puntual (McNemar + bootstrap)...")
    pareada = comparacion_pareada(df_cusum_principal, df_punto, alto_impacto_ids)
    _guardar_tabla(pareada, "05_comparacion_pareada")
    print(pareada[pareada["subconjunto"].isin(["global", "grupo_b", "alto_impacto"])].to_string(index=False))

    # ---- [16] tabla de decision ----
    print("\n[16] Tabla de decision para elegir k...")
    decision = tabla_decision(tablas_res["global"], tablas_gb["global"], tabla_ai, pareada)
    _guardar_tabla(decision, "06_tabla_decision_k")
    print(decision.to_string(index=False))

    candidatos = decision[(decision["falsos_eventos_dia"] - 0.06).abs() < 0.03]
    candidatos = candidatos[candidatos["pct_retenido_de_lo_ya_detectado_por_puntual"] >= 90.0]
    if len(candidatos):
        k_recomendado = float(candidatos.sort_values("grupo_b_recuperacion_pct", ascending=False).iloc[0]["k"])
    else:
        k_recomendado = float(decision.sort_values("grupo_b_recuperacion_pct", ascending=False).iloc[0]["k"])
    print(f"\nk recomendado (mejora Grupo B, mantiene ~0.06 eventos/dia, retiene >=90% de lo ya detectado): {k_recomendado}")

    # ---- [15/17] figuras de ejemplos y evolucion temporal, con el k recomendado ----
    print("\n[15/17] Seleccionando casos de ejemplo y generando figuras...")
    info_rec = calib_idx[(k_recomendado, OBJETIVO_PRINCIPAL)]
    df_rec = df_cusum_principal[df_cusum_principal["k_param"] == k_recomendado].set_index("episode_id")
    df_pt = df_punto.set_index("episode_id")
    comun = df_rec.index.intersection(df_pt.index)
    solo_cusum = [e for e in comun if df_rec.loc[e, "induced_detected"] and not df_pt.loc[e, "induced_detected"]]
    solo_puntual = [e for e in comun if df_pt.loc[e, "induced_detected"] and not df_rec.loc[e, "induced_detected"]]
    ninguno = [e for e in comun if not df_rec.loc[e, "induced_detected"] and not df_pt.loc[e, "induced_detected"] and e in grupo_b_ids]

    def _elegir(lista, n=3):
        return sorted(lista)[:n] if len(lista) <= n else [lista[i] for i in np.linspace(0, len(lista) - 1, n).astype(int)]

    casos = {"solo_cusum": _elegir(solo_cusum), "solo_puntual": _elegir(solo_puntual), "ninguno": _elegir(ninguno)}
    log_detallado = []
    for etiqueta, ids in casos.items():
        for i, ep_id in enumerate(ids):
            nombre_archivo = f"14_evolucion_{etiqueta}_{i+1}"
            figura_evolucion_temporal(ep_id, ataques, particion_val_raw, grid_60, scores_atacados_idx, scores_limpios_60,
                                       params_est, info_rec["C_clean_full"], k_recomendado, info_rec["h"],
                                       inicio_particion, nombre_archivo, etiqueta)
            a = ataques[ep_id]
            ks = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_ventanas_grid))
            fin = pd.to_datetime(grid_60["fin"])
            c_prev = float(info_rec["C_clean_full"][ks[0] - 1]) if ks[0] - 1 >= 0 else 0.0
            for k in ks:
                s_att = scores_atacados_idx[ep_id][k]
                z_att = (s_att - params_est["median_clean"]) / params_est["denom_efectivo"]
                c_t = max(0.0, c_prev + z_att - k_recomendado)
                alarm_att = c_t > info_rec["h"]
                log_detallado.append({
                    "seleccion": etiqueta, "episode_id": ep_id, "k": k, "window_end": fin[k],
                    "relative_kw": s_att, "z_t": z_att, "incremento_z_menos_k": z_att - k_recomendado,
                    "C_t_limpio": float(info_rec["C_clean_full"][k]), "C_t_atacado": c_t, "h": info_rec["h"],
                    "alarma_limpia": bool(info_rec["alarm_clean_full"][k]), "alarma_atacada": bool(alarm_att),
                    "attack_start": a["attack_start"], "attack_end": a["attack_end"],
                })
                c_prev = 0.0 if alarm_att else c_t
    _guardar_tabla(pd.DataFrame(log_detallado), "07_log_acumulacion_casos")
    print(f"  {sum(len(v) for v in casos.values())} casos de ejemplo, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")

    # ---- [17] figuras restantes ----
    print("\nGenerando figuras restantes...")
    tabla_dur = tablas_res["por_duracion"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k_param in VALORES_K:
        sub = tabla_dur[tabla_dur["k"] == k_param].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", markersize=4, color=COLOR_K[k_param], label=f"CUSUM k={k_param}")
    sub_pt = df_punto.groupby("duration_min")["induced_detected"].mean() * 100
    ax.plot(sub_pt.index, sub_pt.values, marker="s", markersize=5, color=ROJO, linewidth=2, linestyle="--", label="puntual")
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)")
    ax.set_title("DR puntual vs CUSUM, por duracion (0.06 eventos/dia)")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout(); _guardar_fig(fig, "01_dr_puntual_vs_cusum_duracion")

    tabla_fam = tablas_res["por_familia"]
    familias = sorted(tabla_fam["family"].unique())
    x = np.arange(len(familias)); ancho = 0.8 / (len(VALORES_K) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, k_param in enumerate(VALORES_K):
        sub = tabla_fam[tabla_fam["k"] == k_param].set_index("family").reindex(familias)
        ax.bar(x + (i - len(VALORES_K) / 2) * ancho, sub["induced_DR_pct"], width=ancho, color=COLOR_K[k_param], label=f"k={k_param}")
    sub_pt_fam = df_punto.groupby("family")["induced_detected"].mean().reindex(familias) * 100
    ax.bar(x + (len(VALORES_K) / 2) * ancho, sub_pt_fam.values, width=ancho, color=ROJO, label="puntual")
    ax.set_xticks(x); ax.set_xticklabels(familias, rotation=15, ha="right")
    ax.set_ylabel("induced_DR (%)"); ax.set_title("DR puntual vs CUSUM, por familia")
    ax.legend(fontsize=7.5, ncols=2)
    fig.tight_layout(); _guardar_fig(fig, "02_dr_puntual_vs_cusum_familia")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    tabla_glob = tablas_res["global"]
    sin_picos_glob = tabla_glob[tabla_glob["familias"] == "sin_recorte_picos"]
    ax.bar(["puntual"] + [f"k={k}" for k in VALORES_K],
           [100 * df_punto[df_punto["family"] != "recorte_picos"]["induced_detected"].mean()] +
           [sin_picos_glob[sin_picos_glob["k"] == k]["energy_weighted_dr_pct"].iloc[0] for k in VALORES_K],
           color=[ROJO] + [COLOR_K[k] for k in VALORES_K])
    ax.set_ylabel("DR ponderado por energia (%) -- sin recorte_picos")
    ax.set_title("Cobertura energetica por metodo")
    fig.tight_layout(); _guardar_fig(fig, "03_cobertura_energetica_por_metodo")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    tabla_gb_dur = tablas_gb["por_duracion"]
    for k_param in VALORES_K:
        sub = tabla_gb_dur[tabla_gb_dur["k"] == k_param].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", color=COLOR_K[k_param], label=f"k={k_param}")
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("% recuperado del Grupo B")
    ax.set_title("Recuperacion del Grupo B por duracion y k")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "04_recuperacion_grupo_b_duracion")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar([f"k={k}" for k in VALORES_K], tabla_ai.set_index("k").reindex(VALORES_K)["pct_recuperados"], color=[COLOR_K[k] for k in VALORES_K])
    ax.set_ylabel("% recuperado de los 371 de alto impacto")
    ax.set_title("Recuperacion de ataques no detectados de alto impacto")
    fig.tight_layout(); _guardar_fig(fig, "05_recuperacion_alto_impacto")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k_param in VALORES_K:
        sub = tabla_calib[tabla_calib["k"] == k_param].sort_values("tasa_eventos_dia_real")
        dr_k = [tablas_res["global"][(tablas_res["global"]["k"] == k_param) & (tablas_res["global"]["familias"] == "sin_recorte_picos")]["induced_DR_pct"].iloc[0]] * len(sub)
        ax.scatter(sub["tasa_eventos_dia_real"], dr_k, color=COLOR_K[k_param], label=f"k={k_param}", s=30)
    ax.axvline(0.06, color=GRIS, linestyle="--", linewidth=1)
    ax.set_xlabel("falsos eventos/dia (calibracion)"); ax.set_ylabel("induced_DR (%) a 0.06/dia (referencia)")
    ax.set_title("DR frente a falsos eventos/dia (puntos de calibracion evaluados por k)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "06_dr_vs_falsos_eventos_dia")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k_param in VALORES_K:
        sub = resultados_todo[(resultados_todo["k_param"] == k_param) & (resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL) & (resultados_todo["induced_detected"])]
        ax.scatter([tabla_calib[(tabla_calib["k"] == k_param) & (tabla_calib["objetivo_nombre"] == OBJETIVO_PRINCIPAL)]["tasa_eventos_dia_real"].iloc[0]] * len(sub),
                   sub["induced_detection_delay_min"], color=COLOR_K[k_param], alpha=0.3, s=10, label=f"k={k_param}" if k_param == VALORES_K[0] else None)
    medianas = [resultados_todo[(resultados_todo["k_param"] == k) & (resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL) & (resultados_todo["induced_detected"])]["induced_detection_delay_min"].median() for k in VALORES_K]
    ax.plot([tabla_calib[(tabla_calib["k"] == k) & (tabla_calib["objetivo_nombre"] == OBJETIVO_PRINCIPAL)]["tasa_eventos_dia_real"].iloc[0] for k in VALORES_K], medianas, color=GRIS_OSCURO, marker="D", linewidth=1.5, label="mediana")
    ax.set_xlabel("falsos eventos/dia real"); ax.set_ylabel("retraso (min)")
    ax.set_title("Retraso frente a falsos eventos/dia, por k")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "07_retraso_vs_falsos_eventos_dia")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(["puntual"] + [f"k={k}" for k in VALORES_K],
           [df_punto["hidden_energy_before_detection_kwh"].sum()] +
           [tablas_res["global"][(tablas_res["global"]["k"] == k) & (tablas_res["global"]["familias"] == "todas")]["hidden_energy_before_detection_total_kwh"].iloc[0] for k in VALORES_K],
           color=[ROJO] + [COLOR_K[k] for k in VALORES_K])
    ax.set_ylabel("energia oculta ANTES de la alarma (kWh, total)")
    ax.set_title("Energia oculta antes de detectar, por metodo")
    fig.tight_layout(); _guardar_fig(fig, "08_energia_antes_de_detectar")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    tabla_glob_todas = tablas_res["global"][tablas_res["global"]["familias"] == "sin_recorte_picos"].sort_values("k")
    ax.plot(tabla_glob_todas["k"], tabla_glob_todas["induced_DR_pct"], marker="o", color=AZUL, label="episode DR")
    ax.plot(tabla_glob_todas["k"], tabla_glob_todas["energy_weighted_dr_pct"], marker="o", color=NARANJA, label="energy-weighted DR")
    ax.set_xlabel("k"); ax.set_ylabel("DR (%) -- sin recorte_picos")
    ax.set_title("Rendimiento por valor de k (0.06 eventos/dia)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "09_rendimiento_por_k")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for k_param in VALORES_K:
        C_full = calib_idx[(k_param, OBJETIVO_PRINCIPAL)]["C_clean_full"]
        ax.hist(C_full, bins=60, color=COLOR_K[k_param], alpha=0.4, density=True, label=f"k={k_param}")
    ax.set_xlabel("C_t (limpio)"); ax.set_ylabel("densidad")
    ax.set_title("Distribucion de CUSUM sobre datos limpios (0.06 eventos/dia)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "10_distribucion_cusum_limpio")

    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")
    assert np.array_equal(serie_antes, particion_val_raw), "la serie limpia original no debe modificarse"

    return {
        "ctx": ctx, "df_punto": df_punto, "grupo_b_ids": grupo_b_ids, "alto_impacto_ids": alto_impacto_ids,
        "params_est": params_est, "tabla_calib": tabla_calib, "calib_idx": calib_idx,
        "resultados_todo": resultados_todo, "df_cusum_principal": df_cusum_principal,
        "tablas_res": tablas_res, "tablas_gb": tablas_gb, "tabla_ai": tabla_ai, "pareada": pareada,
        "decision": decision, "k_recomendado": k_recomendado,
    }


if __name__ == "__main__":
    main()
