"""Acumulacion temporal con memoria corta sobre `relative_kw`: suma movil (SM) y EWMA del
exceso positivo, comparadas con el detector puntual sobre ataques `ramp_lineal`, el Grupo B
de `reduccion_constante` y un control con todas las reducciones constantes.

A diferencia de CUSUM, SM y EWMA nunca se resetean -- su memoria evoluciona segun su propia
formula (ventana deslizante finita para SM, decaimiento exponencial para EWMA), lo que evita
el problema de deriva de semanas que hizo fallar a CUSUM. Un "evento" de alarma es una
transicion de inactivo a activo (no cada ventana por encima de h) -- se reutiliza literalmente
`experimento_stride.contar_eventos` tanto para calibrar h como para medir falsos eventos/dia.

No reentrena nada, no cambia ventana/stride/relative_kw/umbral puntual, no genera ataques
nuevos: la reconstruccion de los 1050 episodios ramp via
`experimento_ramp.construir_episodios_ramp` es determinista, no es "generar" episodios sino
reproducir los mismos. Reutiliza sin modificar base_relative.{cargar_ataques_y_modelo_360,
calcular_relative_kw}, experimento_ramp.construir_episodios_ramp, funciones de
experimento_stride (_ventanas_solapadas, calibrar_umbral_por_eventos, contar_eventos,
mcnemar_p, bootstrap_pareado, construir_rejilla), stride_relative.puntuar_stride_relative,
analisis_detectabilidad._energia_antes_despues, episodes.wilson_ci y
diagnostico_no_estacionariedad.construir_secuencia_continua.

Ejecutable de forma aislada:
    python -m src.memoria_finita_relative_kw
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

from src import analisis_detectabilidad as ad
from src.base_relative import cargar_ataques_y_modelo_360, calcular_relative_kw
from src.data_loading import COLUMNA_OBJETIVO
from src.diagnostico_no_estacionariedad import construir_secuencia_continua
from src.episodes import wilson_ci
from src.experimento_ramp import construir_episodios_ramp
from src.experimento_stride import (
    _ventanas_solapadas,
    bootstrap_pareado,
    calibrar_umbral_por_eventos,
    construir_rejilla,
    contar_eventos,
    mcnemar_p,
)
from src.normalization import aplicar_zscore
from src.stride_relative import puntuar_stride_relative

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "memoria_finita_relative_kw"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "memoria_finita_relative_kw"
DET_DIR = BASE_DIR / "results" / "tables" / "analisis_detectabilidad"
RAMP_DIR = BASE_DIR / "results" / "tables" / "experimento_ramp"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
COLOR_METODO = {"P": GRIS_OSCURO, "SM": AZUL, "EWMA": NARANJA}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDE = 60
SCORE = "relative_kw"
EPS_ENERGIA = 1e-9
L_VALUES = [3, 6, 12]
Q_VALUES = [75, 85, 90]
METODOS = ["SM", "EWMA"]
OBJETIVO_FA = 0.06


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


def alpha_de_L(L: int) -> float:
    return 2.0 / (L + 1.0)


# ==========================================================================================
# 1/3) Setup + division desarrollo/evaluacion
# ==========================================================================================

def cargar_todo() -> dict:
    config, datos, modelo, ataques_originales, master_csv = cargar_ataques_y_modelo_360()
    params_norm = datos["params_norm"]
    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    n_dias_val = len(particion_val_raw) / 1440.0
    inicio_particion = particion_val.index[0]

    base = construir_secuencia_continua()  # checkpoint existente, sin reentrenar
    d_train, d_val = base["por_particion"]["train"], base["por_particion"]["val"]
    scores_train, scores_val = d_train["relative_kw"], d_val["relative_kw"]
    ts_train, ts_val = pd.DatetimeIndex(d_train["timestamps"]), pd.DatetimeIndex(d_val["timestamps"])
    scores_full = np.concatenate([scores_train, scores_val])
    n_train = len(scores_train)

    grid_60 = construir_rejilla(particion_val, STRIDE)
    umbral_punto, _, _ = calibrar_umbral_por_eventos(scores_val, n_dias_val, OBJETIVO_FA)

    return {
        "config": config, "datos": datos, "modelo": modelo, "ataques_originales": ataques_originales,
        "master_csv": master_csv, "particion_val_raw": particion_val_raw, "n_dias_val": n_dias_val,
        "inicio_particion": inicio_particion, "scores_train": scores_train, "scores_val": scores_val,
        "ts_train": ts_train, "ts_val": ts_val, "scores_full": scores_full, "n_train": n_train,
        "grid_60": grid_60, "params_norm": params_norm, "umbral_punto": umbral_punto,
    }


def dividir_desarrollo_evaluacion(ctx: dict) -> dict:
    n_val = len(ctx["scores_val"])
    corte = n_val // 2
    ts_val = ctx["ts_val"]
    dev_ini, dev_fin = ts_val[0], ts_val[corte - 1]
    eval_ini, eval_fin = ts_val[corte], ts_val[-1]
    n_dias_dev = corte * STRIDE / 1440.0
    n_dias_eval = (n_val - corte) * STRIDE / 1440.0
    _guardar_tabla(pd.DataFrame([{
        "desarrollo_inicio": dev_ini, "desarrollo_fin": dev_fin, "n_ventanas_desarrollo": corte, "n_dias_desarrollo": n_dias_dev,
        "evaluacion_inicio": eval_ini, "evaluacion_fin": eval_fin, "n_ventanas_evaluacion": n_val - corte, "n_dias_evaluacion": n_dias_eval,
    }]), "00_particion_temporal")
    return {"corte_idx": corte, "dev_ini": dev_ini, "dev_fin": dev_fin, "eval_ini": eval_ini, "eval_fin": eval_fin,
            "n_dias_dev": n_dias_dev, "n_dias_eval": n_dias_eval}


# ==========================================================================================
# 4) Conjuntos de episodios R (ramp), B (Grupo B constante), C (control: todas las constantes)
# ==========================================================================================

def construir_conjuntos(ctx: dict, div: dict) -> dict:
    tabla_ramp, ataques_ramp = construir_episodios_ramp(ctx["master_csv"], ctx["particion_val_raw"], ctx["inicio_particion"])
    for a in ataques_ramp.values():
        a["y_tramo"] = a["y_tramo_ramp"]  # alias para reutilizar puntuar_stride_relative sin modificarlo

    reduccion = ctx["master_csv"][ctx["master_csv"]["family"] == "reduccion_constante"]
    ataques_constante = {ep: ctx["ataques_originales"][ep] for ep in reduccion["episode_id"]}

    det_full = pd.read_csv(DET_DIR / "episodios_detectabilidad_impacto.csv")
    grupo_b_ids = set(det_full[(det_full["family"] == "reduccion_constante") & (det_full["detectability_group"] == "B")]["episode_id"])

    def _filtrar(tabla_meta: pd.DataFrame) -> tuple[set, set]:
        dev = set(tabla_meta[(tabla_meta["attack_start"] >= div["dev_ini"]) & (tabla_meta["attack_end"] <= div["dev_fin"])]["episode_id"])
        ev = set(tabla_meta[(tabla_meta["attack_start"] >= div["eval_ini"]) & (tabla_meta["attack_end"] <= div["eval_fin"])]["episode_id"])
        return dev, ev

    ramp_dev, ramp_eval = _filtrar(tabla_ramp)
    const_meta = reduccion[["episode_id", "attack_start", "attack_end"]]
    const_dev_all, const_eval_all = _filtrar(const_meta)
    b_dev, b_eval = const_dev_all & grupo_b_ids, const_eval_all & grupo_b_ids

    resumen = pd.DataFrame([
        {"conjunto": "R_ramp", "n_total": len(tabla_ramp), "n_desarrollo": len(ramp_dev), "n_evaluacion": len(ramp_eval)},
        {"conjunto": "C_todas_constantes", "n_total": len(reduccion), "n_desarrollo": len(const_dev_all), "n_evaluacion": len(const_eval_all)},
        {"conjunto": "B_grupo_b", "n_total": len(grupo_b_ids), "n_desarrollo": len(b_dev), "n_evaluacion": len(b_eval)},
    ])
    _guardar_tabla(resumen, "00_conjuntos_episodios")

    return {
        "tabla_ramp": tabla_ramp, "ataques_ramp": ataques_ramp, "ataques_constante": ataques_constante,
        "grupo_b_ids": grupo_b_ids, "R_dev": ramp_dev, "R_eval": ramp_eval,
        "C_dev": const_dev_all, "C_eval": const_eval_all, "B_dev": b_dev, "B_eval": b_eval,
    }


# ==========================================================================================
# 5) Nivel secundario del score (SOLO clean de desarrollo) + exceso observable
# ==========================================================================================

def calcular_niveles_secundarios(scores_val: np.ndarray, corte: int) -> dict[int, float]:
    scores_dev = scores_val[:corte]
    return {q: float(np.percentile(scores_dev, q)) for q in Q_VALUES}


def exceso(scores: np.ndarray, u_q: float) -> np.ndarray:
    return np.maximum(0.0, scores - u_q)


# ==========================================================================================
# 7/8) Construccion de SM y EWMA (trayectoria global limpia, sin reset)
# ==========================================================================================

def estadistico_sm(e: np.ndarray, L: int) -> np.ndarray:
    return pd.Series(e).rolling(window=L, min_periods=1).sum().to_numpy()


def estadistico_ewma(e: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA_t = alpha*e_t + (1-alpha)*EWMA_{t-1}, con EWMA_{-1}=0 exactamente (no la primera
    observacion, que es el comportamiento por defecto de pandas .ewm(adjust=False)). Se
    consigue anteponiendo un 0 sintetico como "EWMA_{-1}" y descartando esa posicion tras
    calcular -- diferencia solo relevante en los primerisimos pasos de la serie (aqui, el
    arranque de train, sin efecto practico en val, pero exacta segun la formula del encargo)."""
    con_semilla = np.concatenate([[0.0], e])
    return pd.Series(con_semilla).ewm(alpha=alpha, adjust=False).mean().to_numpy()[1:]


def calcular_estadistico(e_full: np.ndarray, metodo: str, L: int) -> np.ndarray:
    return estadistico_sm(e_full, L) if metodo == "SM" else estadistico_ewma(e_full, alpha_de_L(L))


def _duracion_eventos(activo: np.ndarray, resolucion_min: int = STRIDE) -> np.ndarray:
    duraciones, contador = [], 0
    for a in activo:
        if a:
            contador += 1
        else:
            if contador > 0:
                duraciones.append(contador * resolucion_min)
            contador = 0
    if contador > 0:
        duraciones.append(contador * resolucion_min)
    return np.array(duraciones) if duraciones else np.array([0.0])


# ==========================================================================================
# 12) Calibracion de h (SOLO desarrollo) -- el estadistico no depende de h, asi que se
#      reutiliza calibrar_umbral_por_eventos (barrido de percentiles + contar_eventos)
# ==========================================================================================

def calibrar_todas_las_configuraciones(ctx: dict, div: dict, u_qs: dict) -> tuple[pd.DataFrame, dict]:
    corte = div["corte_idx"]
    n_train = ctx["n_train"]
    filas = []
    trayectorias = {}
    for metodo in METODOS:
        for q in Q_VALUES:
            e_full = exceso(ctx["scores_full"], u_qs[q])
            for L in L_VALUES:
                stat_full = calcular_estadistico(e_full, metodo, L)
                stat_val = stat_full[n_train:]
                stat_dev = stat_val[:corte]
                h, tasa_dev, _ = calibrar_umbral_por_eventos(stat_dev, div["n_dias_dev"], OBJETIVO_FA)
                activo_dev = stat_dev > h
                eventos = contar_eventos(activo_dev)
                duraciones = _duracion_eventos(activo_dev)
                ts_eventos = np.where(activo_dev & ~np.concatenate([[False], activo_dev[:-1]]))[0]
                tiempo_medio = float(np.diff(ts_eventos).mean() * STRIDE) if len(ts_eventos) > 1 else float("nan")
                filas.append({
                    "metodo": metodo, "q": q, "u_q": u_qs[q], "L": L,
                    "alpha": alpha_de_L(L) if metodo == "EWMA" else float("nan"), "h": h,
                    "n_eventos_falsos_desarrollo": eventos, "n_dias_desarrollo": div["n_dias_dev"],
                    "eventos_dia_desarrollo": eventos / div["n_dias_dev"], "ventanas_alarma_dia_desarrollo": float(activo_dev.sum()) / div["n_dias_dev"],
                    "duracion_media_evento_min": float(duraciones.mean()), "duracion_mediana_evento_min": float(np.median(duraciones)),
                    "duracion_p95_evento_min": float(np.percentile(duraciones, 95)), "duracion_max_evento_min": float(duraciones.max()),
                    "tiempo_medio_entre_eventos_min": tiempo_medio,
                })
                trayectorias[(metodo, q, L)] = {"stat_full": stat_full, "h": h}
    tabla = _guardar_tabla(pd.DataFrame(filas), "01_configuraciones_exploradas")
    return tabla, trayectorias


# ==========================================================================================
# 9/10/11) Simulacion por episodio (rama limpia continua + rama atacada, sin reset)
# ==========================================================================================

def _indexar_scores_atacados(tabla_ventanas: pd.DataFrame) -> dict[str, dict[int, float]]:
    return {ep: dict(zip(g["k"], g["attacked_score"])) for ep, g in tabla_ventanas.groupby("episode_id")}


def simular_episodios(ataques: dict, ids: set, attacked_scores_idx: dict, n_train: int, n_ventanas_grid: int,
                       e_full: np.ndarray, stat_full_clean: np.ndarray, h: float, metodo: str, u_q: float, L: int,
                       fin_grid: np.ndarray) -> pd.DataFrame:
    active_clean_full = stat_full_clean > h
    alpha = alpha_de_L(L)
    filas = []
    for ep in ids:
        a = ataques[ep]
        ks_local = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_ventanas_grid))
        ks_global = np.array([n_train + k for k in ks_local])
        k_before = int(ks_global[0]) - 1

        s_att = np.array([attacked_scores_idx[ep][k] for k in ks_local])
        e_att = np.maximum(0.0, s_att - u_q)

        if metodo == "SM":
            hist_ini = max(0, k_before - L + 2)
            historial = e_full[hist_ini:k_before + 1]
            serie_local = np.concatenate([historial, e_att])
            stat_local = pd.Series(serie_local).rolling(window=L, min_periods=1).sum().to_numpy()[-len(ks_local):]
        else:
            prev = float(stat_full_clean[k_before]) if k_before >= 0 else 0.0
            stat_local = np.empty(len(ks_local))
            for i, e in enumerate(e_att):
                prev = alpha * e + (1.0 - alpha) * prev
                stat_local[i] = prev

        active_local = stat_local > h
        prev_active = bool(active_clean_full[k_before]) if k_before >= 0 else False
        event_start = np.empty(len(ks_local), dtype=bool)
        for i in range(len(ks_local)):
            event_start[i] = bool(active_local[i]) and not prev_active
            prev_active = bool(active_local[i])

        active_clean_local = active_clean_full[ks_global]
        induced = event_start & ~active_clean_local
        idx_ind = np.where(induced)[0]
        ts_induced = pd.Timestamp(fin_grid[ks_local[idx_ind[0]]]) if len(idx_ind) else pd.NaT

        filas.append({
            "episode_id": ep, "n_affected_windows": len(ks_local),
            "max_stat_attacked": float(stat_local.max()), "max_stat_clean": float(stat_full_clean[ks_global].max()),
            "raw_detected": bool(active_local.any()), "induced_detected": bool(induced.any()),
            "first_alarm_timestamp": ts_induced,
        })
    return pd.DataFrame(filas)


def enriquecer_resultado(sim: pd.DataFrame, ataques: dict, master_csv_meta: pd.DataFrame, particion_val_raw: np.ndarray,
                          contexto_impacto: pd.DataFrame) -> pd.DataFrame:
    df = master_csv_meta.merge(sim, on="episode_id").merge(contexto_impacto, on="episode_id")
    sin_induced = ~df["induced_detected"]
    df["detection_delay_min"] = (df["first_alarm_timestamp"] - df["attack_start"]) / pd.Timedelta(minutes=1)
    df.loc[sin_induced, "detection_delay_min"] = np.nan
    durante = (df["first_alarm_timestamp"] <= df["attack_end"]).astype(object)
    despues = (df["first_alarm_timestamp"] > df["attack_end"]).astype(object)
    durante[sin_induced] = np.nan; despues[sin_induced] = np.nan
    df["detected_during_attack"] = durante; df["detected_after_attack"] = despues
    filas_energia = [ad._energia_antes_despues(row, ataques, particion_val_raw) for row in
                      df.rename(columns={"first_alarm_timestamp": "first_induced_alarm_timestamp"}).itertuples()]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(filas_energia)], axis=1)
    return df


# ==========================================================================================
# 13) Seleccion de configuraciones (SOLO datos de desarrollo: R_dev + B_dev)
# ==========================================================================================

def evaluar_config_en_desarrollo(ctx, div, conjuntos, tabla_ventanas_ramp, tabla_ventanas_const, u_qs, trayectorias,
                                  contexto_ramp, contexto_const, df_p_ramp_dev, df_p_b_dev) -> pd.DataFrame:
    n_train, n_grid = ctx["n_train"], ctx["grid_60"]["X"].shape[0]
    fin_grid = ctx["grid_60"]["fin"]
    idx_ramp = _indexar_scores_atacados(tabla_ventanas_ramp)
    idx_const = _indexar_scores_atacados(tabla_ventanas_const)

    filas = []
    for metodo in METODOS:
        for q in Q_VALUES:
            e_full = exceso(ctx["scores_full"], u_qs[q])
            for L in L_VALUES:
                info = trayectorias[(metodo, q, L)]
                sim_r = simular_episodios(conjuntos["ataques_ramp"], conjuntos["R_dev"], idx_ramp, n_train, n_grid,
                                           e_full, info["stat_full"], info["h"], metodo, u_qs[q], L, fin_grid)
                sim_b = simular_episodios(conjuntos["ataques_constante"], conjuntos["B_dev"], idx_const, n_train, n_grid,
                                           e_full, info["stat_full"], info["h"], metodo, u_qs[q], L, fin_grid)
                dr_ramp = 100 * sim_r["induced_detected"].mean() if len(sim_r) else float("nan")
                dr_b = 100 * sim_b["induced_detected"].mean() if len(sim_b) else float("nan")
                no_det_p_ramp = set(df_p_ramp_dev[df_p_ramp_dev["induced_detected"] == False]["episode_id"])  # noqa: E712
                rec_no_det = sim_r[sim_r["episode_id"].isin(no_det_p_ramp)]
                pct_recupera_no_det = 100 * rec_no_det["induced_detected"].mean() if len(rec_no_det) else float("nan")
                det_p_ramp = set(df_p_ramp_dev[df_p_ramp_dev["induced_detected"] == True]["episode_id"])  # noqa: E712
                retencion = sim_r[sim_r["episode_id"].isin(det_p_ramp)]
                retention_rate = 100 * retencion["induced_detected"].mean() if len(retencion) else float("nan")
                filas.append({
                    "metodo": metodo, "q": q, "L": L, "dr_ramp_dev_pct": dr_ramp, "dr_grupo_b_dev_pct": dr_b,
                    "pct_recupera_ramp_no_detectada_por_p_dev": pct_recupera_no_det, "retention_rate_ramp_dev_pct": retention_rate,
                    "h": info["h"],
                })
    return pd.DataFrame(filas)


def seleccionar_configuraciones_finales(tabla_seleccion_dev: pd.DataFrame) -> dict:
    seleccion = {}
    for metodo in METODOS:
        sub = tabla_seleccion_dev[tabla_seleccion_dev["metodo"] == metodo].copy()
        # criterio compuesto (solo desarrollo): prioriza recuperar ramp no detectada por P y
        # Grupo B sin perder retencion, con desempate por simplicidad (L mas pequenio)
        sub["score_seleccion"] = (
            sub["pct_recupera_ramp_no_detectada_por_p_dev"].fillna(0)
            + sub["dr_grupo_b_dev_pct"].fillna(0)
            + 0.5 * sub["dr_ramp_dev_pct"].fillna(0)
            + 0.2 * sub["retention_rate_ramp_dev_pct"].fillna(0)
            - 0.5 * sub["L"]  # penaliza complejidad/memoria mas larga a igualdad de lo demas
        )
        mejor = sub.sort_values("score_seleccion", ascending=False).iloc[0]
        seleccion[metodo] = {"metodo": metodo, "q": int(mejor["q"]), "L": int(mejor["L"]),
                              "alpha": alpha_de_L(int(mejor["L"])) if metodo == "EWMA" else float("nan"), "h": float(mejor["h"])}
    tabla = pd.DataFrame(seleccion.values())
    _guardar_tabla(tabla, "03_configuraciones_congeladas")
    return seleccion


# ==========================================================================================
# 15) Metricas principales
# ==========================================================================================

def resumen_metricas(grupo: pd.DataFrame) -> dict:
    n = len(grupo)
    n_ind = int(grupo["induced_detected"].sum())
    ci = wilson_ci(n_ind, n) if n else (float("nan"), float("nan"))
    con_ind = grupo[grupo["induced_detected"] == True]  # noqa: E712
    hidden_col = "hidden_energy_ramp_kwh" if "hidden_energy_ramp_kwh" in grupo.columns else "hidden_energy_kwh"
    total_hidden = grupo[hidden_col].sum()
    detected_hidden = grupo.loc[grupo["induced_detected"], hidden_col].sum()
    after_total = grupo["hidden_energy_after_detection_kwh"].sum()
    return {
        "n_episodios": n, "n_detectados": n_ind, "induced_DR_pct": 100 * n_ind / n if n else float("nan"),
        "induced_DR_ci95_low": ci[0], "induced_DR_ci95_high": ci[1],
        "energy_weighted_dr_pct": 100 * detected_hidden / total_hidden if total_hidden > EPS_ENERGIA else float("nan"),
        "timely_energy_detection_rate_pct": 100 * after_total / total_hidden if total_hidden > EPS_ENERGIA else float("nan"),
        "retraso_mediano_min": float(con_ind["detection_delay_min"].median()) if len(con_ind) else float("nan"),
        "retraso_p75_min": float(con_ind["detection_delay_min"].quantile(0.75)) if len(con_ind) else float("nan"),
        "retraso_p90_min": float(con_ind["detection_delay_min"].quantile(0.90)) if len(con_ind) else float("nan"),
        "pct_detectado_durante_ataque": 100 * (con_ind["detected_during_attack"] == True).mean() if len(con_ind) else float("nan"),  # noqa: E712
        "hidden_energy_before_detection_total_kwh": float(grupo["hidden_energy_before_detection_kwh"].sum()),
        "pct_recuperable_medio": float(grupo["recoverable_hidden_energy_pct"].mean()),
    }


def tabla_complementariedad(df_p: pd.DataFrame, df_acc: pd.DataFrame) -> dict:
    p = df_p.set_index("episode_id")["induced_detected"].astype(bool)
    a = df_acc.set_index("episode_id")["induced_detected"].astype(bool).reindex(p.index)
    ambos = int((p & a).sum()); solo_p = int((p & ~a).sum()); solo_a = int((~p & a).sum()); ninguno = int((~p & ~a).sum())
    retention = 100 * ambos / p.sum() if p.sum() else float("nan")
    exclusive = 100 * solo_a / (~p).sum() if (~p).sum() else float("nan")
    return {"detectado_ambos": ambos, "solo_puntual": solo_p, "solo_acumulador": solo_a, "detectado_ninguno": ninguno,
            "retention_rate_pct": retention, "exclusive_recovery_rate_pct": exclusive}


def comparacion_pareada(dfs: dict[str, pd.DataFrame], hidden_col: str) -> pd.DataFrame:
    pares = [("P", "SM"), ("P", "EWMA"), ("SM", "EWMA")]
    filas = []
    for a, b in pares:
        meta = dfs[a].set_index("episode_id")[["family", "duration_min"]] if "family" in dfs[a].columns else \
            dfs[a].set_index("episode_id")[["duration_min"]]
        det_a = dfs[a].set_index("episode_id")["induced_detected"].astype(bool)
        det_b = dfs[b].set_index("episode_id")["induced_detected"].astype(bool).reindex(det_a.index)
        hid = dfs[a].set_index("episode_id")[hidden_col] if hidden_col in dfs[a].columns else None
        delay_a = dfs[a].set_index("episode_id")["detection_delay_min"]
        delay_b = dfs[b].set_index("episode_id")["detection_delay_min"].reindex(delay_a.index)

        comb = pd.DataFrame({"a": det_a, "b": det_b}).join(meta)
        subconjuntos = {"global": comb, "duraciones_largas_720_1440": comb[comb["duration_min"].isin([720, 1440])]}
        for nombre_sub, sub in subconjuntos.items():
            if len(sub) == 0:
                continue
            da, db = sub["a"].to_numpy(), sub["b"].to_numpy()
            solo_a, solo_b = int((da & ~db).sum()), int((~da & db).sum())
            p_mcnemar = mcnemar_p(solo_a, solo_b)
            ci_low, ci_high = bootstrap_pareado(da.astype(float), db.astype(float))
            if hid is not None:
                h = hid.reindex(sub.index).to_numpy()
                ci_e_low, ci_e_high = bootstrap_pareado(np.where(da, h, 0.0), np.where(db, h, 0.0))
            else:
                ci_e_low = ci_e_high = float("nan")
            ambos_det = da & db
            if ambos_det.sum() >= 5:
                ra = delay_a.reindex(sub.index).to_numpy()[ambos_det]; rb = delay_b.reindex(sub.index).to_numpy()[ambos_det]
                try:
                    _, p_wilcoxon = sstats.wilcoxon(ra, rb)
                except ValueError:
                    p_wilcoxon = float("nan")
            else:
                p_wilcoxon = float("nan")
            filas.append({
                "comparacion": f"{a}_vs_{b}", "subconjunto": nombre_sub, "n_episodios": len(sub),
                "DR_a_pct": 100 * da.mean(), "DR_b_pct": 100 * db.mean(), "diferencia_DR_pp": 100 * (db.mean() - da.mean()),
                "mcnemar_p": p_mcnemar, "bootstrap_dr_ci95_low": ci_low, "bootstrap_dr_ci95_high": ci_high,
                "bootstrap_energia_ci95_low": ci_e_low, "bootstrap_energia_ci95_high": ci_e_high,
                "wilcoxon_p_retraso": p_wilcoxon, "n_detectados_ambos": int(ambos_det.sum()),
                "significativo_p<0.05": p_mcnemar < 0.05,
            })
    return pd.DataFrame(filas)


# ==========================================================================================
# 24) Criterios de exito (fijados antes de evaluar la segunda mitad)
# ==========================================================================================

CRITERIOS_EXITO_TEXTO = {
    1: "Recupera >= 10% de las rampas no detectadas por P",
    2: "Recupera >= 15% de las rampas no detectadas de alto impacto",
    3: "Mejora >= 5pp el DR de ramp frente a P",
    4: "Mejora >= 5pp el DR ponderado por energia de ramp frente a P",
    5: "Recupera >= 5% del Grupo B de reduccion_constante",
    6: "Mantiene una tasa de falsas alarmas operativamente compatible con 0.06 eventos/dia",
    7: "Retraso mediano < 180 min",
    8: "Conserva >= 85% de las reducciones constantes detectadas por P",
    9: "La mejora no depende solo de ventanas casi identicas por solapamiento",
    10: "La mejora aparece en mas de una duracion o severidad",
}


def evaluar_criterios(nombre: str, pct_recupera_no_det: float, pct_recupera_alto_impacto: float,
                       dif_dr_ramp: float, dif_dr_energia_ramp: float, pct_recupera_b: float,
                       fa_eval: float, retraso_mediano: float, retention_rate_control: float,
                       mejora_sobrevive_downsample: bool, mejora_multi_duracion: bool) -> pd.DataFrame:
    resultados = {
        1: pct_recupera_no_det >= 10.0, 2: pct_recupera_alto_impacto >= 15.0, 3: dif_dr_ramp >= 5.0,
        4: dif_dr_energia_ramp >= 5.0, 5: pct_recupera_b >= 5.0, 6: abs(fa_eval - 0.06) <= 0.06,
        7: (retraso_mediano < 180.0) if not np.isnan(retraso_mediano) else False,
        8: retention_rate_control >= 85.0, 9: mejora_sobrevive_downsample, 10: mejora_multi_duracion,
    }
    tabla = pd.DataFrame([{"criterio_num": k, "criterio": CRITERIOS_EXITO_TEXTO[k], "cumplido": v} for k, v in resultados.items()])
    tabla.attrs["n_cumplidos"] = sum(resultados.values())
    tabla.attrs["mayoria"] = sum(resultados.values()) >= 6
    return tabla


# ==========================================================================================
# 14) Detector puntual de referencia (misma logica ventana-a-ventana que en todos los
#      experimentos anteriores; se evalua UNA vez sobre todos los episodios de cada familia)
# ==========================================================================================

def evaluar_puntual(tabla_ventanas: pd.DataFrame, ataques: dict, master_csv_meta: pd.DataFrame,
                     particion_val_raw: np.ndarray, contexto_impacto: pd.DataFrame, umbral: float) -> pd.DataFrame:
    t = tabla_ventanas.copy()
    t["window_end"] = pd.to_datetime(t["window_end"])
    t["clean_is_alarm"] = t["clean_score"] > umbral
    t["attacked_is_alarm"] = t["attacked_score"] > umbral
    t["attack_induced_alarm"] = t["attacked_is_alarm"] & ~t["clean_is_alarm"]
    agg = t.groupby("episode_id").agg(
        n_affected_windows=("k", "size"), induced_detected=("attack_induced_alarm", "any"),
    ).reset_index()
    ts = t[t["attack_induced_alarm"]].groupby("episode_id")["window_end"].min().rename("first_alarm_timestamp")
    agg = agg.merge(ts, on="episode_id", how="left")
    sim = agg
    return enriquecer_resultado(sim, ataques, master_csv_meta, particion_val_raw, contexto_impacto)


# ==========================================================================================
# 21) Influencia del solapamiento (diagnostico, SIN recalibrar): submuestrea cada 6a decision
# ==========================================================================================

def diagnostico_solapamiento(ataques: dict, ids_exclusivas: set, attacked_scores_idx: dict, n_train: int,
                              n_grid: int, e_full: np.ndarray, stat_full_clean: np.ndarray, h: float,
                              metodo: str, u_q: float, L: int, fin_grid: np.ndarray, factor: int = 6) -> dict:
    if not ids_exclusivas:
        return {"n_episodios": 0, "pct_sobrevive_downsample": float("nan")}
    ataques_sub = {ep: ataques[ep] for ep in ids_exclusivas}
    sim_completa = simular_episodios(ataques_sub, ids_exclusivas, attacked_scores_idx, n_train, n_grid, e_full,
                                      stat_full_clean, h, metodo, u_q, L, fin_grid)

    filas_resumen = []
    for ep in ids_exclusivas:
        a = ataques[ep]
        ks_local = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_grid))
        n_windows_atacadas_por_pct = []
        for k in ks_local:
            w_start = k * STRIDE
            solape_ini = max(a["offset_global"], w_start)
            solape_fin = min(a["offset_global"] + a["duracion"], w_start + TAMANO_VENTANA)
            n_windows_atacadas_por_pct.append(max(0, solape_fin - solape_ini) / TAMANO_VENTANA)
        ks_decimados = ks_local[::factor]
        duracion_real_min = (max(ks_local) - min(ks_local)) * STRIDE + TAMANO_VENTANA
        filas_resumen.append({
            "episode_id": ep, "n_ventanas_completas": len(ks_local), "n_ventanas_decimadas": len(ks_decimados),
            "duracion_real_cubierta_min": duracion_real_min, "pct_medio_muestras_atacadas_por_ventana": 100 * float(np.mean(n_windows_atacadas_por_pct)),
        })
    tabla_resumen = pd.DataFrame(filas_resumen)

    ataques_decimados = {}
    for ep in ids_exclusivas:
        a = ataques[ep]
        ks_local = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_grid))
        ks_dec = ks_local[::factor]
        if len(ks_dec) < 1:
            continue
        offset_dec = ks_dec[0] * STRIDE
        fin_dec = ks_dec[-1] * STRIDE + TAMANO_VENTANA
        ataques_decimados[ep] = {"offset_global": offset_dec, "duracion": fin_dec - offset_dec}

    n_sobrevive = 0
    for ep in ataques_decimados:
        a_ep = ataques[ep]
        ks_local = sorted(_ventanas_solapadas(a_ep["offset_global"], a_ep["duracion"], STRIDE, n_grid))
        ks_dec = ks_local[::factor]
        s_att = np.array([attacked_scores_idx[ep][k] for k in ks_dec])
        e_att = np.maximum(0.0, s_att - u_q)
        if metodo == "SM":
            stat_dec = pd.Series(e_att).rolling(window=min(L, len(e_att)), min_periods=1).sum().to_numpy()
        else:
            alpha = alpha_de_L(L)
            prev, stat_dec = 0.0, []
            for e in e_att:
                prev = alpha * e + (1 - alpha) * prev
                stat_dec.append(prev)
            stat_dec = np.array(stat_dec)
        if (stat_dec > h).any():
            n_sobrevive += 1

    pct_sobrevive = 100 * n_sobrevive / len(ids_exclusivas) if ids_exclusivas else float("nan")
    return {"n_episodios": len(ids_exclusivas), "n_sobrevive_downsample": n_sobrevive,
            "pct_sobrevive_downsample": pct_sobrevive, "tabla_resumen": tabla_resumen}


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("[1] Cargando modelo, ataques existentes, ramp y rejilla val (checkpoint existente)...")
    ctx = cargar_todo()
    serie_train_antes = ctx["scores_train"].copy(); serie_val_antes = ctx["particion_val_raw"].copy()

    print("[3] Dividiendo val en desarrollo/evaluacion...")
    div = dividir_desarrollo_evaluacion(ctx)
    print(f"  desarrollo: {div['dev_ini']} -> {div['dev_fin']} ({div['n_dias_dev']:.1f} dias)")
    print(f"  evaluacion: {div['eval_ini']} -> {div['eval_fin']} ({div['n_dias_eval']:.1f} dias)")

    print("\n[4] Construyendo conjuntos R (ramp), B (Grupo B) y C (todas las constantes)...")
    conjuntos = construir_conjuntos(ctx, div)
    print(f"  R: {len(conjuntos['R_dev'])} dev / {len(conjuntos['R_eval'])} eval")
    print(f"  B: {len(conjuntos['B_dev'])} dev / {len(conjuntos['B_eval'])} eval")
    print(f"  C: {len(conjuntos['C_dev'])} dev / {len(conjuntos['C_eval'])} eval")

    print("\nPuntuando ramp y reduccion_constante (2 llamadas batched al modelo)...")
    tabla_ventanas_ramp, _ = puntuar_stride_relative(conjuntos["ataques_ramp"], ctx["particion_val_raw"], ctx["grid_60"],
                                                       ctx["modelo"], ctx["params_norm"], STRIDE)
    tabla_ventanas_const, _ = puntuar_stride_relative(conjuntos["ataques_constante"], ctx["particion_val_raw"], ctx["grid_60"],
                                                        ctx["modelo"], ctx["params_norm"], STRIDE)
    _guardar_tabla(tabla_ventanas_ramp, "trayectorias_ventanas_ramp")
    _guardar_tabla(tabla_ventanas_const, "trayectorias_ventanas_constante")

    contexto_ramp = ad.contexto_e_impacto(conjuntos["ataques_ramp"], ctx["particion_val_raw"])
    contexto_const = ad.contexto_e_impacto(conjuntos["ataques_constante"], ctx["particion_val_raw"])
    meta_ramp = conjuntos["tabla_ramp"][["episode_id", "attack_start", "attack_end", "duration_min", "rho_final"]]
    meta_const = ctx["master_csv"][ctx["master_csv"]["family"] == "reduccion_constante"][
        ["episode_id", "family", "parameter", "attack_start", "attack_end", "duration_min"]]

    df_p_ramp_full = evaluar_puntual(tabla_ventanas_ramp, conjuntos["ataques_ramp"], meta_ramp, ctx["particion_val_raw"], contexto_ramp, ctx["umbral_punto"])
    df_p_const_full = evaluar_puntual(tabla_ventanas_const, conjuntos["ataques_constante"], meta_const, ctx["particion_val_raw"], contexto_const, ctx["umbral_punto"])
    _guardar_tabla(df_p_ramp_full, "05_resultados_P_ramp_completo")
    _guardar_tabla(df_p_const_full, "05_resultados_P_constante_completo")

    print("\n[5] Calculando niveles secundarios (P75/P85/P90, solo desarrollo limpio)...")
    u_qs = calcular_niveles_secundarios(ctx["scores_val"], div["corte_idx"])
    print(f"  u_q: {u_qs}")

    print("\n[12] Calibrando h para las 18 configuraciones (solo desarrollo)...")
    tabla_calib, trayectorias = calibrar_todas_las_configuraciones(ctx, div, u_qs)
    print(tabla_calib[["metodo", "q", "L", "h", "eventos_dia_desarrollo", "duracion_max_evento_min"]].to_string(index=False))

    print("\n[13] Evaluando las 18 configuraciones sobre R_dev y B_dev (solo desarrollo)...")
    df_p_ramp_dev = df_p_ramp_full[df_p_ramp_full["episode_id"].isin(conjuntos["R_dev"])]
    df_p_b_dev = df_p_const_full[df_p_const_full["episode_id"].isin(conjuntos["B_dev"])]
    tabla_seleccion_dev = evaluar_config_en_desarrollo(ctx, div, conjuntos, tabla_ventanas_ramp, tabla_ventanas_const,
                                                        u_qs, trayectorias, contexto_ramp, contexto_const, df_p_ramp_dev, df_p_b_dev)
    _guardar_tabla(tabla_seleccion_dev, "02_evaluacion_desarrollo_18_configs")
    seleccion = seleccionar_configuraciones_finales(tabla_seleccion_dev)
    for m, s in seleccion.items():
        print(f"  {m}: q=P{s['q']} L={s['L']}h h={s['h']:.4f}")

    print("\n[14/15] Simulando P, SM y EWMA sobre los episodios de EVALUACION...")
    idx_ramp = _indexar_scores_atacados(tabla_ventanas_ramp)
    idx_const = _indexar_scores_atacados(tabla_ventanas_const)
    n_train, n_grid = ctx["n_train"], ctx["grid_60"]["X"].shape[0]
    fin_grid = ctx["grid_60"]["fin"]

    dfs_ramp_eval, dfs_const_eval = {"P": df_p_ramp_full[df_p_ramp_full["episode_id"].isin(conjuntos["R_eval"])]}, \
        {"P": df_p_const_full[df_p_const_full["episode_id"].isin(conjuntos["C_eval"])]}
    for metodo in METODOS:
        s = seleccion[metodo]; u_q = u_qs[s["q"]]
        e_full = exceso(ctx["scores_full"], u_q)
        info = trayectorias[(metodo, s["q"], s["L"])]

        sim_r = simular_episodios(conjuntos["ataques_ramp"], conjuntos["R_eval"], idx_ramp, n_train, n_grid, e_full,
                                   info["stat_full"], s["h"], metodo, u_q, s["L"], fin_grid)
        dfs_ramp_eval[metodo] = enriquecer_resultado(sim_r, conjuntos["ataques_ramp"], meta_ramp, ctx["particion_val_raw"], contexto_ramp)

        sim_c = simular_episodios(conjuntos["ataques_constante"], conjuntos["C_eval"], idx_const, n_train, n_grid, e_full,
                                   info["stat_full"], s["h"], metodo, u_q, s["L"], fin_grid)
        dfs_const_eval[metodo] = enriquecer_resultado(sim_c, conjuntos["ataques_constante"], meta_const, ctx["particion_val_raw"], contexto_const)

        _guardar_tabla(dfs_ramp_eval[metodo], f"05_resultados_{metodo}_ramp_eval")
        _guardar_tabla(dfs_const_eval[metodo], f"05_resultados_{metodo}_constante_eval")

    print("\n[15] Metricas principales sobre ramp (evaluacion)...")
    filas_glob = []
    for metodo, df in dfs_ramp_eval.items():
        fila = {"metodo": metodo}; fila.update(resumen_metricas(df)); filas_glob.append(fila)
    tabla_ramp_global = _guardar_tabla(pd.DataFrame(filas_glob), "06_ramp_global")
    print(tabla_ramp_global.to_string(index=False))

    filas_dur, filas_rho = [], []
    for metodo, df in dfs_ramp_eval.items():
        for dur, g in df.groupby("duration_min"):
            fila = {"metodo": metodo, "duration_min": dur}; fila.update(resumen_metricas(g)); filas_dur.append(fila)
        for rho, g in df.groupby("rho_final"):
            fila = {"metodo": metodo, "rho_final": rho}; fila.update(resumen_metricas(g)); filas_rho.append(fila)
    _guardar_tabla(pd.DataFrame(filas_dur), "06_ramp_por_duracion")
    _guardar_tabla(pd.DataFrame(filas_rho), "06_ramp_por_severidad")

    print("\n[17] Rampas no detectadas por P de alto impacto (subconjunto de evaluacion)...")
    alto_impacto_full = pd.read_csv(RAMP_DIR / "ramp_no_detectados_alto_impacto.csv")
    alto_impacto_eval_ids = set(alto_impacto_full["episode_id"]) & conjuntos["R_eval"]
    filas_ai = []
    for metodo, df in dfs_ramp_eval.items():
        sub = df[df["episode_id"].isin(alto_impacto_eval_ids)]
        n_rec = int(sub["induced_detected"].sum())
        filas_ai.append({"metodo": metodo, "n_alto_impacto": len(sub), "n_recuperados": n_rec,
                          "pct_recuperado": 100 * n_rec / len(sub) if len(sub) else float("nan"),
                          "energia_recuperada_kwh": float(sub.loc[sub["induced_detected"], "hidden_energy_kwh"].sum()),
                          "hidden_energy_before_detection_media_kwh": float(sub["hidden_energy_before_detection_kwh"].mean())})
    tabla_ai = _guardar_tabla(pd.DataFrame(filas_ai), "ramp_alto_impacto_recuperado_memoria_finita")
    print(f"  n alto impacto en evaluacion: {len(alto_impacto_eval_ids)}")
    print(tabla_ai.to_string(index=False))

    print("\n[19] Grupo B de reduccion_constante (evaluacion)...")
    filas_b = []
    for metodo, df in dfs_const_eval.items():
        sub = df[df["episode_id"].isin(conjuntos["B_eval"])]
        n_rec = int(sub["induced_detected"].sum())
        filas_b.append({"metodo": metodo, "n_grupo_b": len(sub), "n_recuperados": n_rec,
                         "pct_recuperado": 100 * n_rec / len(sub) if len(sub) else float("nan"),
                         "energy_weighted_dr_pct": resumen_metricas(sub)["energy_weighted_dr_pct"],
                         "retraso_mediano_min": resumen_metricas(sub)["retraso_mediano_min"]})
    tabla_b = _guardar_tabla(pd.DataFrame(filas_b), "07_grupo_b_evaluacion")
    print(tabla_b.to_string(index=False))

    print("\n[20] Control: todas las reducciones constantes (evaluacion)...")
    filas_c = []
    for metodo, df in dfs_const_eval.items():
        fila = {"metodo": metodo}; fila.update(resumen_metricas(df)); filas_c.append(fila)
    tabla_c = _guardar_tabla(pd.DataFrame(filas_c), "08_todas_constantes_evaluacion")
    print(tabla_c.to_string(index=False))

    print("\n[7] Falsas alarmas fuera de muestra (h congelado, sin recalibrar)...")
    filas_fa = []
    for metodo in METODOS:
        s = seleccion[metodo]; info = trayectorias[(metodo, s["q"], s["L"])]
        stat_eval = info["stat_full"][n_train:][div["corte_idx"]:]
        activo_eval = stat_eval > s["h"]
        n_eventos = contar_eventos(activo_eval)
        tasa = n_eventos / div["n_dias_eval"]
        lo = sstats.chi2.ppf(0.025, 2 * n_eventos) / 2 / div["n_dias_eval"] if n_eventos else 0.0
        hi = sstats.chi2.ppf(0.975, 2 * (n_eventos + 1)) / 2 / div["n_dias_eval"]
        dur = _duracion_eventos(activo_eval)
        filas_fa.append({"metodo": metodo, "h": s["h"], "n_eventos_evaluacion": n_eventos, "n_dias_evaluacion": div["n_dias_eval"],
                          "eventos_dia_evaluacion": tasa, "ic95_poisson_low": lo, "ic95_poisson_high": hi,
                          "ventanas_alarma_dia_evaluacion": float(activo_eval.sum()) / div["n_dias_eval"],
                          "duracion_media_evento_min": float(dur.mean()), "duracion_max_evento_min": float(dur.max()),
                          "diferencia_vs_objetivo": tasa - 0.06})
    es_alarma_p_eval = ctx["scores_val"][div["corte_idx"]:] > ctx["umbral_punto"]
    n_eventos_p = contar_eventos(es_alarma_p_eval)
    filas_fa.append({"metodo": "P", "h": ctx["umbral_punto"], "n_eventos_evaluacion": n_eventos_p, "n_dias_evaluacion": div["n_dias_eval"],
                      "eventos_dia_evaluacion": n_eventos_p / div["n_dias_eval"],
                      "ic95_poisson_low": sstats.chi2.ppf(0.025, 2 * n_eventos_p) / 2 / div["n_dias_eval"] if n_eventos_p else 0.0,
                      "ic95_poisson_high": sstats.chi2.ppf(0.975, 2 * (n_eventos_p + 1)) / 2 / div["n_dias_eval"],
                      "ventanas_alarma_dia_evaluacion": float(es_alarma_p_eval.sum()) / div["n_dias_eval"],
                      "duracion_media_evento_min": float("nan"), "duracion_max_evento_min": float("nan"), "diferencia_vs_objetivo": float("nan")})
    tabla_fa = pd.DataFrame(filas_fa)
    tabla_fa = pd.concat([tabla_fa[tabla_fa["metodo"] == "P"], tabla_fa[tabla_fa["metodo"] != "P"]], ignore_index=True)
    _guardar_tabla(tabla_fa, "04_falsas_alarmas_desarrollo_evaluacion")
    print(tabla_fa[["metodo", "h", "eventos_dia_evaluacion", "diferencia_vs_objetivo"]].to_string(index=False))

    print("\n[22] Complementariedad con el detector puntual...")
    comp_ramp = {m: tabla_complementariedad(dfs_ramp_eval["P"], dfs_ramp_eval[m]) for m in METODOS}
    comp_const = {m: tabla_complementariedad(dfs_const_eval["P"], dfs_const_eval[m]) for m in METODOS}
    tabla_comp = pd.DataFrame({
        **{f"ramp_{m}": comp_ramp[m] for m in METODOS}, **{f"constante_{m}": comp_const[m] for m in METODOS},
    }).T.reset_index().rename(columns={"index": "conjunto_metodo"})
    _guardar_tabla(tabla_comp, "09_complementariedad")
    print(tabla_comp.to_string(index=False))

    print("\n[23] Comparaciones estadisticas (McNemar + bootstrap + Wilcoxon)...")
    pareada_ramp = comparacion_pareada(dfs_ramp_eval, "hidden_energy_kwh")
    pareada_const = comparacion_pareada(dfs_const_eval, "hidden_energy_kwh")
    pareada_ramp["conjunto"] = "ramp"; pareada_const["conjunto"] = "constante"
    pareada = pd.concat([pareada_ramp, pareada_const], ignore_index=True)
    _guardar_tabla(pareada, "10_comparaciones_estadisticas")
    print(pareada[pareada["subconjunto"] == "global"].to_string(index=False))

    print("\n[21] Influencia del solapamiento (diagnostico, sin recalibrar)...")
    filas_solap = []
    for metodo in METODOS:
        s = seleccion[metodo]; u_q = u_qs[s["q"]]; info = trayectorias[(metodo, s["q"], s["L"])]
        e_full = exceso(ctx["scores_full"], u_q)
        p_ids = set(dfs_ramp_eval["P"][dfs_ramp_eval["P"]["induced_detected"] == True]["episode_id"])  # noqa: E712
        m_ids = set(dfs_ramp_eval[metodo][dfs_ramp_eval[metodo]["induced_detected"] == True]["episode_id"])  # noqa: E712
        exclusivas = m_ids - p_ids
        diag = diagnostico_solapamiento(conjuntos["ataques_ramp"], exclusivas, idx_ramp, n_train, n_grid, e_full,
                                         info["stat_full"], s["h"], metodo, u_q, s["L"], fin_grid)
        filas_solap.append({"metodo": metodo, "n_detecciones_exclusivas": diag["n_episodios"],
                             "pct_sobrevive_downsample_6x": diag["pct_sobrevive_downsample"]})
        if "tabla_resumen" in diag:
            _guardar_tabla(diag["tabla_resumen"], f"14_solapamiento_{metodo}")
    tabla_solapamiento = _guardar_tabla(pd.DataFrame(filas_solap), "14_influencia_solapamiento_resumen")
    print(tabla_solapamiento.to_string(index=False))

    print("\n[24] Evaluando criterios de exito (fijados antes de ver estos resultados)...")
    criterios_todos = []
    for metodo in METODOS:
        ai_pct = tabla_ai[tabla_ai["metodo"] == metodo]["pct_recuperado"].iloc[0]
        p_ramp_global = tabla_ramp_global[tabla_ramp_global["metodo"] == "P"]["induced_DR_pct"].iloc[0]
        m_ramp_global = tabla_ramp_global[tabla_ramp_global["metodo"] == metodo]["induced_DR_pct"].iloc[0]
        p_ramp_e = tabla_ramp_global[tabla_ramp_global["metodo"] == "P"]["energy_weighted_dr_pct"].iloc[0]
        m_ramp_e = tabla_ramp_global[tabla_ramp_global["metodo"] == metodo]["energy_weighted_dr_pct"].iloc[0]
        b_pct = tabla_b[tabla_b["metodo"] == metodo]["pct_recuperado"].iloc[0]
        fa_eval = tabla_fa[tabla_fa["metodo"] == metodo]["eventos_dia_evaluacion"].iloc[0]
        retraso = tabla_ramp_global[tabla_ramp_global["metodo"] == metodo]["retraso_mediano_min"].iloc[0]
        retention_control = comp_const[metodo]["retention_rate_pct"]
        no_det_p = df_p_ramp_full[(df_p_ramp_full["induced_detected"] == False) & (df_p_ramp_full["episode_id"].isin(conjuntos["R_eval"]))]  # noqa: E712
        rec_no_det = dfs_ramp_eval[metodo][dfs_ramp_eval[metodo]["episode_id"].isin(set(no_det_p["episode_id"]))]
        pct_recupera_no_det = 100 * rec_no_det["induced_detected"].mean() if len(rec_no_det) else float("nan")
        sobrevive = tabla_solapamiento[tabla_solapamiento["metodo"] == metodo]["pct_sobrevive_downsample_6x"].iloc[0]
        mejora_multi = (tabla_ramp_global[tabla_ramp_global["metodo"] == metodo]["induced_DR_pct"].iloc[0] >
                         tabla_ramp_global[tabla_ramp_global["metodo"] == "P"]["induced_DR_pct"].iloc[0])
        dur_con_mejora = 0
        for dur, g in pd.DataFrame(filas_dur).groupby("duration_min"):
            dp = g[g["metodo"] == "P"]["induced_DR_pct"]; dm = g[g["metodo"] == metodo]["induced_DR_pct"]
            if len(dp) and len(dm) and dm.iloc[0] > dp.iloc[0]:
                dur_con_mejora += 1
        tabla_crit = evaluar_criterios(metodo, pct_recupera_no_det, ai_pct, m_ramp_global - p_ramp_global,
                                        m_ramp_e - p_ramp_e, b_pct, fa_eval, retraso, retention_control,
                                        not np.isnan(sobrevive) and sobrevive >= 50.0, dur_con_mejora >= 2)
        tabla_crit.insert(0, "metodo", metodo)
        criterios_todos.append(tabla_crit)
        print(f"  {metodo}: {tabla_crit.attrs['n_cumplidos']}/10 criterios -> mayoria={tabla_crit.attrs['mayoria']}")
    tabla_criterios = pd.concat(criterios_todos, ignore_index=True)
    _guardar_tabla(tabla_criterios, "15_criterios_exito")

    print("\nGenerando figuras...")
    generar_figuras(ctx, div, u_qs, tabla_calib, seleccion, dfs_ramp_eval, dfs_const_eval, tabla_ramp_global,
                     pd.DataFrame(filas_dur), tabla_ai, tabla_b, comp_ramp, tabla_fa, pareada_ramp,
                     conjuntos, meta_ramp)

    assert np.array_equal(serie_train_antes, ctx["scores_train"])
    assert np.array_equal(serie_val_antes, ctx["particion_val_raw"])
    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "ctx": ctx, "div": div, "conjuntos": conjuntos, "u_qs": u_qs, "tabla_calib": tabla_calib,
        "trayectorias": trayectorias, "seleccion": seleccion, "dfs_ramp_eval": dfs_ramp_eval,
        "dfs_const_eval": dfs_const_eval, "tabla_ramp_global": tabla_ramp_global, "tabla_ai": tabla_ai,
        "tabla_b": tabla_b, "tabla_c": tabla_c, "tabla_fa": tabla_fa, "tabla_comp": tabla_comp,
        "pareada": pareada, "tabla_solapamiento": tabla_solapamiento, "tabla_criterios": tabla_criterios,
    }


def generar_figuras(ctx, div, u_qs, tabla_calib, seleccion, dfs_ramp_eval, dfs_const_eval, tabla_ramp_global,
                     tabla_ramp_dur, tabla_ai, tabla_b, comp_ramp, tabla_fa, pareada_ramp, conjuntos, meta_ramp):
    # 1/2) construccion de SM y EWMA (ejemplo sintetico ilustrativo)
    rng = np.random.default_rng(0)
    e_demo = np.clip(rng.normal(0.3, 1, 40), 0, None)
    e_demo[15:25] += np.linspace(0.5, 3, 10)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(estadistico_sm(e_demo, 6), color=AZUL); axes[0].set_title("Suma movil (L=6h)"); axes[0].set_ylabel("SM_t")
    axes[1].plot(estadistico_ewma(e_demo, alpha_de_L(6)), color=NARANJA); axes[1].set_title("EWMA (L=6h)"); axes[1].set_ylabel("EWMA_t")
    axes[1].set_xlabel("ventana"); fig.tight_layout(); _guardar_fig(fig, "01_02_construccion_sm_ewma")

    # 3) rendimiento por memoria
    fig, ax = plt.subplots(figsize=(7, 5))
    for metodo in METODOS:
        sub = tabla_calib[(tabla_calib["metodo"] == metodo) & (tabla_calib["q"] == seleccion[metodo]["q"])].sort_values("L")
        ax.plot(sub["L"], sub["duracion_max_evento_min"], marker="o", color=COLOR_METODO[metodo], label=metodo)
    ax.set_xlabel("L (horas)"); ax.set_ylabel("duracion max. evento limpio (min, desarrollo)"); ax.legend(fontsize=8)
    ax.set_title("Rendimiento por memoria")
    fig.tight_layout(); _guardar_fig(fig, "03_rendimiento_por_memoria")

    # 4) rendimiento por cuantil
    fig, ax = plt.subplots(figsize=(7, 5))
    for metodo in METODOS:
        sub = tabla_calib[(tabla_calib["metodo"] == metodo) & (tabla_calib["L"] == seleccion[metodo]["L"])].sort_values("q")
        ax.plot(sub["q"], sub["h"], marker="o", color=COLOR_METODO[metodo], label=metodo)
    ax.set_xlabel("cuantil (P)"); ax.set_ylabel("h calibrado"); ax.legend(fontsize=8)
    ax.set_title("h calibrado por cuantil secundario")
    fig.tight_layout(); _guardar_fig(fig, "04_rendimiento_por_cuantil")

    # 5) DR de ramp por metodo
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_ramp_global["metodo"], tabla_ramp_global["induced_DR_pct"], color=[COLOR_METODO[m] for m in tabla_ramp_global["metodo"]])
    ax.set_ylabel("induced_DR (%)"); ax.set_title("DR de ramp_lineal por metodo (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "05_dr_ramp_por_metodo")

    # 6) DR por duracion
    fig, ax = plt.subplots(figsize=(7, 5))
    for metodo in ["P"] + METODOS:
        sub = tabla_ramp_dur[tabla_ramp_dur["metodo"] == metodo].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", color=COLOR_METODO[metodo], label=metodo)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)"); ax.legend(fontsize=8)
    ax.set_title("DR de ramp por duracion y metodo")
    fig.tight_layout(); _guardar_fig(fig, "06_dr_por_duracion")

    # 7) DR energetico
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_ramp_global["metodo"], tabla_ramp_global["energy_weighted_dr_pct"], color=[COLOR_METODO[m] for m in tabla_ramp_global["metodo"]])
    ax.set_ylabel("DR ponderado por energia (%)"); ax.set_title("DR energetico de ramp (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "07_dr_energetico")

    # 8) retraso
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_ramp_global["metodo"], tabla_ramp_global["retraso_mediano_min"], color=[COLOR_METODO[m] for m in tabla_ramp_global["metodo"]])
    ax.set_ylabel("retraso mediano (min)"); ax.set_title("Retraso de deteccion de ramp")
    fig.tight_layout(); _guardar_fig(fig, "08_retraso")

    # 9) recuperacion de rampas no detectadas por P
    fig, ax = plt.subplots(figsize=(6, 5))
    no_det_ids = set(dfs_ramp_eval["P"][dfs_ramp_eval["P"]["induced_detected"] == False]["episode_id"])  # noqa: E712
    pcts = [100 * dfs_ramp_eval[m][dfs_ramp_eval[m]["episode_id"].isin(no_det_ids)]["induced_detected"].mean() for m in METODOS]
    ax.bar(METODOS, pcts, color=[COLOR_METODO[m] for m in METODOS])
    ax.set_ylabel("% recuperado de rampas no detectadas por P"); ax.set_title("Recuperacion exclusiva")
    fig.tight_layout(); _guardar_fig(fig, "09_recuperacion_no_detectadas")

    # 10) recuperacion de alto impacto
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_ai["metodo"], tabla_ai["pct_recuperado"], color=[COLOR_METODO.get(m, GRIS) for m in tabla_ai["metodo"]])
    ax.set_ylabel("% recuperado"); ax.set_title("Recuperacion de rampas de alto impacto")
    fig.tight_layout(); _guardar_fig(fig, "10_recuperacion_alto_impacto")

    # 11) recuperacion Grupo B
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_b["metodo"], tabla_b["pct_recuperado"], color=[COLOR_METODO.get(m, GRIS) for m in tabla_b["metodo"]])
    ax.set_ylabel("% del Grupo B recuperado"); ax.set_title("Recuperacion del Grupo B (reduccion_constante)")
    fig.tight_layout(); _guardar_fig(fig, "11_recuperacion_grupo_b")

    # 12) complementariedad
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metodo in zip(axes, METODOS):
        d = comp_ramp[metodo]
        ax.bar(["ambos", "solo P", f"solo {metodo}", "ninguno"],
               [d["detectado_ambos"], d["solo_puntual"], d["solo_acumulador"], d["detectado_ninguno"]],
               color=[VERDE, GRIS, COLOR_METODO[metodo], ROJO])
        ax.set_title(f"P vs {metodo} (ramp)\nretention={d['retention_rate_pct']:.1f}% excl={d['exclusive_recovery_rate_pct']:.1f}%", fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "12_complementariedad")

    # 13/14) DR y retraso vs falsas alarmas (usando los 3 objetivos de calibracion disponibles
    #        indirectamente via las 18 configuraciones exploradas, como proxy de sensibilidad)
    fig, ax = plt.subplots(figsize=(7, 5))
    for metodo in METODOS:
        sub = tabla_calib[tabla_calib["metodo"] == metodo]
        ax.scatter(sub["eventos_dia_desarrollo"], sub["duracion_max_evento_min"], color=COLOR_METODO[metodo], alpha=0.6, label=metodo)
    ax.set_xlabel("falsos eventos/dia (desarrollo)"); ax.set_ylabel("duracion max evento (min)"); ax.legend(fontsize=8)
    ax.set_title("Sensibilidad: duracion de evento vs falsas alarmas (18 configs)")
    fig.tight_layout(); _guardar_fig(fig, "13_14_sensibilidad_configs")

    # 19) evolucion temporal en un caso ramp (score, exceso, estadistico)
    ep_ids = list(dfs_ramp_eval["SM"][dfs_ramp_eval["SM"]["induced_detected"] == True]["episode_id"])  # noqa: E712
    if ep_ids:
        ep0 = ep_ids[0]
        a0 = conjuntos["ataques_ramp"][ep0]
        n_grid = ctx["grid_60"]["X"].shape[0]
        ks = sorted(_ventanas_solapadas(a0["offset_global"], a0["duracion"], STRIDE, n_grid))
        idx_ramp_scores = _indexar_scores_atacados(pd.read_csv(TABLAS_DIR / "trayectorias_ventanas_ramp.csv"))
        s_att = np.array([idx_ramp_scores[ep0][k] for k in ks])
        u_q = u_qs[seleccion["SM"]["q"]]
        e_att = np.maximum(0, s_att - u_q)
        sm_att = pd.Series(e_att).rolling(window=seleccion["SM"]["L"], min_periods=1).sum().to_numpy()
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(s_att, color=ROJO, marker="o", markersize=3); axes[0].axhline(u_q, color=GRIS_OSCURO, linestyle="--"); axes[0].set_ylabel("relative_kw")
        axes[1].plot(e_att, color=NARANJA, marker="o", markersize=3); axes[1].set_ylabel("exceso e_t")
        axes[2].plot(sm_att, color=AZUL, marker="o", markersize=3); axes[2].axhline(seleccion["SM"]["h"], color=GRIS_OSCURO, linestyle="--"); axes[2].set_ylabel("SM_t")
        axes[2].set_xlabel("ventana del episodio"); fig.suptitle(f"Evolucion temporal -- {ep0}")
        fig.tight_layout(); _guardar_fig(fig, "19_evolucion_temporal_caso")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
