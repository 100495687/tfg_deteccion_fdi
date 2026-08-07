"""Introduccion y analisis de ataques `ramp_lineal` (reduccion gradual del consumo reportado).

Analiza si el detector puntual congelado (360/60/relative_kw, umbral existente) responde a
ataques que empiezan siendo pequenos y aumentan linealmente hasta `rho_final`, comparandolos
pareadamente con reducciones constantes de (a) la misma severidad final y (b) exactamente la
misma energia ocultada. Es puramente diagnostico: no implementa suma movil, EWMA, `k de n`,
CUSUM ni predictor causal.

No reentrena nada, no cambia ventana/stride/relative_kw/umbral, no modifica los ataques
existentes (la familia se define aqui, sin tocar attacks.py). Reutiliza sin modificar
base_relative (cargar_ataques_y_modelo_360, calcular_relative_kw), varias funciones de
experimento_stride (_ventanas_solapadas, _ventana_atacada, calibrar_umbral_por_eventos,
contar_eventos, mcnemar_p, bootstrap_pareado, construir_rejilla),
analisis_detectabilidad._energia_antes_despues, episodes.wilson_ci y normalization.aplicar_zscore.

Ejecutable de forma aislada:
    python -m src.experimento_ramp
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
from src.episodes import wilson_ci
from src.experimento_stride import (
    _ventana_atacada,
    _ventanas_solapadas,
    bootstrap_pareado,
    calibrar_umbral_por_eventos,
    construir_rejilla,
    mcnemar_p,
)
from src.normalization import aplicar_zscore

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "experimento_ramp"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "experimento_ramp"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
COLOR_GRUPO = {"A": ROJO, "B": NARANJA, "C": "#e8c547", "D": TEAL, "E": VERDE}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDE = 60
SCORE = "relative_kw"
EPS_ENERGIA = 1e-9
RHO_FINALES = [0.70, 0.50, 0.30]
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
MIN_VENTANAS_TENDENCIA = 3  # minimo de ventanas evaluables para pendiente/correlacion fiable
FAMILIA = "ramp_lineal"


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 2/4) Definicion exacta de ramp_lineal + seleccion de episodios (reutiliza timestamps de
#      reduccion_constante; el comparador A ES literalmente ese episodio de reduccion_constante)
# ==========================================================================================

def _rho_lineal(duracion: int, rho_final: float) -> np.ndarray:
    j = np.arange(duracion)
    progress = (j + 1) / duracion
    return 1.0 - (1.0 - rho_final) * progress


def construir_episodios_ramp(master_csv: pd.DataFrame, particion_val_raw: np.ndarray, inicio_particion) -> tuple[pd.DataFrame, dict]:
    reduccion = master_csv[master_csv["family"] == "reduccion_constante"]
    filas, ataques = [], {}
    for rho_final in RHO_FINALES:
        param_str = f"rho={rho_final:.1f}"
        sub = reduccion[reduccion["parameter"] == param_str].sort_values(["duration_min", "repetition_idx"])
        for fila in sub.itertuples():
            offset_global = int((fila.attack_start - inicio_particion) / pd.Timedelta(minutes=1))
            dur = int(fila.duration_min)
            clean = particion_val_raw[offset_global:offset_global + dur].astype(np.float64)

            rho_t = _rho_lineal(dur, rho_final)
            ramp = clean * rho_t
            rho_energy_equivalent = float(ramp.sum() / clean.sum()) if clean.sum() > EPS_ENERGIA else float("nan")
            equivalent = clean * rho_energy_equivalent
            endpoint = clean * rho_final  # == la senal atacada del propio episodio de reduccion_constante

            episode_id = f"ramp_lineal__{fila.duration_min:04d}_{rho_final:.2f}_{fila.repetition_idx:02d}"
            ataques[episode_id] = {
                "offset_global": offset_global, "duracion": dur, "attack_start": fila.attack_start, "attack_end": fila.attack_end,
                "y_tramo_ramp": ramp, "y_tramo_endpoint": endpoint, "y_tramo_equivalent": equivalent,
                "matched_constant_episode_id": fila.episode_id,
            }

            clean_energy = float(clean.sum() / 60.0)
            ramp_energy = float(ramp.sum() / 60.0)
            hidden_ramp = clean_energy - ramp_energy
            hidden_endpoint = clean_energy - float(endpoint.sum() / 60.0)
            hidden_equiv = clean_energy - float(equivalent.sum() / 60.0)
            reduccion_ramp = clean - ramp

            filas.append({
                "episode_id": episode_id, "matched_constant_episode_id": fila.episode_id,
                "family": FAMILIA, "attack_start": fila.attack_start, "attack_end": fila.attack_end,
                "duration_min": dur, "repetition_idx": int(fila.repetition_idx), "rho_final": rho_final,
                "initial_reduction_pct": 100.0 * (1.0 - rho_t[0]), "final_reduction_pct": 100.0 * (1.0 - rho_final),
                "mean_rho": float(rho_t.mean()), "rho_energy_equivalent": rho_energy_equivalent,
                "clean_energy_kwh": clean_energy, "ramp_energy_kwh": ramp_energy,
                "hidden_energy_ramp_kwh": hidden_ramp, "hidden_energy_ramp_pct": 100 * hidden_ramp / clean_energy if clean_energy > EPS_ENERGIA else float("nan"),
                "hidden_energy_constant_endpoint_kwh": hidden_endpoint, "hidden_energy_constant_equivalent_kwh": hidden_equiv,
                "mean_reduction_kw": float(reduccion_ramp.mean()), "max_reduction_kw": float(reduccion_ramp.max()),
            })
    tabla = pd.DataFrame(filas)
    return tabla, ataques


# ==========================================================================================
# Puntuacion: 1 llamada batched al modelo para cada rama (ramp, endpoint, equivalent)
# ==========================================================================================

def puntuar_tres_ramas(ataques_ramp: dict, particion_val_raw: np.ndarray, grid_60: dict, modelo, params_norm: dict,
                        umbral: float, scores_limpios_60: np.ndarray) -> pd.DataFrame:
    n_ventanas_grid = grid_60["X"].shape[0]
    filas_meta, v_ramp, v_endpoint, v_equiv = [], [], [], []
    for episode_id, a in ataques_ramp.items():
        ks = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_ventanas_grid))
        for k in ks:
            w_start = k * STRIDE
            v_ramp.append(_ventana_atacada(particion_val_raw, w_start, a["offset_global"], a["duracion"], a["y_tramo_ramp"]))
            v_endpoint.append(_ventana_atacada(particion_val_raw, w_start, a["offset_global"], a["duracion"], a["y_tramo_endpoint"]))
            v_equiv.append(_ventana_atacada(particion_val_raw, w_start, a["offset_global"], a["duracion"], a["y_tramo_equivalent"]))
            filas_meta.append({"episode_id": episode_id, "k": k, "window_start": grid_60["inicio"][k], "window_end": grid_60["fin"][k]})

    meta = pd.DataFrame(filas_meta)
    for nombre, ventanas in (("ramp_score", v_ramp), ("endpoint_constant_score", v_endpoint), ("energy_equivalent_constant_score", v_equiv)):
        X_norm = aplicar_zscore(np.stack(ventanas).astype(np.float32), params_norm)
        X_hat = modelo.reconstruir(X_norm)
        meta[nombre] = calcular_relative_kw(X_norm, X_hat, params_norm)

    meta["clean_score"] = scores_limpios_60[meta["k"].to_numpy()]
    meta["threshold"] = umbral
    meta["ramp_score_delta"] = meta["ramp_score"] - meta["clean_score"]
    meta["endpoint_score_delta"] = meta["endpoint_constant_score"] - meta["clean_score"]
    meta["equivalent_score_delta"] = meta["energy_equivalent_constant_score"] - meta["clean_score"]
    meta["ramp_alarm"] = meta["ramp_score"] > umbral
    meta["clean_is_alarm"] = meta["clean_score"] > umbral
    return meta


# ==========================================================================================
# 6) Agregacion episodio-nivel para las 3 ramas (reutiliza ad._energia_antes_despues literal)
# ==========================================================================================

def agregar_episodio_rama(tabla_ventanas: pd.DataFrame, ataques_ramp: dict, master_csv_ramp: pd.DataFrame,
                           particion_val_raw: np.ndarray, umbral: float, columna_score: str, columna_y_tramo: str,
                           prefijo: str) -> pd.DataFrame:
    t = tabla_ventanas.copy()
    t["window_end"] = pd.to_datetime(t["window_end"])
    t["attacked_is_alarm"] = t[columna_score] > umbral
    t["attack_induced_alarm"] = t["attacked_is_alarm"] & ~t["clean_is_alarm"]
    t["score_delta"] = t[columna_score] - t["clean_score"]

    agg = t.groupby("episode_id").agg(
        max_attacked_score=(columna_score, "max"), max_clean_score=("clean_score", "max"),
        max_score_delta=("score_delta", "max"), induced_detected=("attack_induced_alarm", "any"),
        n_affected_windows=("k", "size"),
    ).reset_index()
    induced_ts = t[t["attack_induced_alarm"]].groupby("episode_id")["window_end"].min().rename("first_alarm_timestamp")
    agg = agg.merge(induced_ts, on="episode_id", how="left")

    resultado = agg.merge(master_csv_ramp[["episode_id", "attack_start", "attack_end", "duration_min"]], on="episode_id", how="left")

    sin_induced = ~resultado["induced_detected"]
    resultado["detection_delay_min"] = (resultado["first_alarm_timestamp"] - resultado["attack_start"]) / pd.Timedelta(minutes=1)
    resultado.loc[sin_induced, "detection_delay_min"] = np.nan
    durante = (resultado["first_alarm_timestamp"] <= resultado["attack_end"]).astype(object)
    despues = (resultado["first_alarm_timestamp"] > resultado["attack_end"]).astype(object)
    durante[sin_induced] = np.nan; despues[sin_induced] = np.nan
    resultado["detected_during_attack"] = durante; resultado["detected_after_attack"] = despues
    resultado["threshold"] = umbral
    resultado["distance_to_threshold"] = umbral - resultado["max_attacked_score"]
    resultado["relative_distance_to_threshold"] = resultado["distance_to_threshold"] / umbral

    ataques_rama = {ep: {"offset_global": a["offset_global"], "duracion": a["duracion"], "y_tramo": a[columna_y_tramo]}
                     for ep, a in ataques_ramp.items()}
    filas_hidden = []
    for ep, a in ataques_rama.items():
        clean = particion_val_raw[a["offset_global"]:a["offset_global"] + a["duracion"]].astype(np.float64)
        hidden = float((clean - a["y_tramo"]).sum() / 60.0)
        filas_hidden.append({"episode_id": ep, "hidden_energy_kwh": hidden})
    resultado = resultado.merge(pd.DataFrame(filas_hidden), on="episode_id")

    filas_energia = [ad._energia_antes_despues(row, ataques_rama, particion_val_raw) for row in
                      resultado.rename(columns={"first_alarm_timestamp": "first_induced_alarm_timestamp"}).itertuples()]
    resultado = pd.concat([resultado.reset_index(drop=True), pd.DataFrame(filas_energia)], axis=1)
    resultado.columns = [f"{prefijo}_{c}" if c != "episode_id" else c for c in resultado.columns]
    return resultado


# ==========================================================================================
# 8) Persistencia y forma temporal del score durante el episodio ramp
# ==========================================================================================

def _racha_maxima(bool_array: np.ndarray) -> int:
    mejor, actual = 0, 0
    for b in bool_array:
        actual = actual + 1 if b else 0
        mejor = max(mejor, actual)
    return mejor


def calcular_persistencia(tabla_ventanas: pd.DataFrame, ataques_ramp: dict, percentiles_limpios: dict) -> pd.DataFrame:
    filas = []
    for episode_id, g in tabla_ventanas.groupby("episode_id"):
        g = g.sort_values("k").reset_index(drop=True)
        n = len(g)
        delta = g["ramp_score_delta"].to_numpy()
        score = g["ramp_score"].to_numpy()
        pos_positiva = delta > 0

        fila = {
            "episode_id": episode_id, "n_ventanas_afectadas": n,
            "pct_ventanas_delta_positivo": 100 * pos_positiva.mean(), "pct_ventanas_delta_no_positivo": 100 * (~pos_positiva).mean(),
            "media_score_delta": float(delta.mean()), "mediana_score_delta": float(np.median(delta)),
            "suma_excesos_positivos": float(delta[delta > 0].sum()), "max_score_delta": float(delta.max()),
            "racha_max_delta_positivo": _racha_maxima(pos_positiva),
        }
        for etiqueta, p in percentiles_limpios.items():
            fila[f"racha_max_sobre_{etiqueta}"] = _racha_maxima(score > p)

        if n >= MIN_VENTANAS_TENDENCIA:
            t_idx = np.arange(n)
            pendiente_score = float(np.polyfit(t_idx, score, 1)[0])
            pendiente_delta = float(np.polyfit(t_idx, delta, 1)[0])
            spearman_tiempo, _ = sstats.spearmanr(t_idx, score)
            a = ataques_ramp[episode_id]
            progreso = np.array([(k - min(g["k"])) for k in g["k"]], dtype=float)
            progreso = progreso / max(progreso.max(), 1)
            spearman_progreso, _ = sstats.spearmanr(progreso, score)
        else:
            pendiente_score = pendiente_delta = spearman_tiempo = spearman_progreso = float("nan")
        fila["pendiente_score"] = pendiente_score; fila["pendiente_score_delta"] = pendiente_delta
        fila["spearman_tiempo_score"] = spearman_tiempo; fila["spearman_progreso_score"] = spearman_progreso

        idx_max = int(np.argmax(score))
        fila["timestamp_score_maximo"] = g.iloc[idx_max]["window_end"]
        fila["posicion_relativa_score_maximo"] = idx_max / max(n - 1, 1)

        t0 = pd.to_datetime(g.iloc[0]["window_end"])
        for etiqueta, p in {**percentiles_limpios, "umbral_puntual": g["threshold"].iloc[0]}.items():
            idx_cruce = np.where(score > p)[0]
            fila[f"tiempo_hasta_{etiqueta}_min"] = (pd.to_datetime(g.iloc[idx_cruce[0]]["window_end"]) - t0).total_seconds() / 60.0 if len(idx_cruce) else float("nan")
        filas.append(fila)
    return pd.DataFrame(filas)


# ==========================================================================================
# 9) Evidencia de senal temporal aprovechable (solo entre las rampas NO detectadas)
# ==========================================================================================

def evidencia_senal_temporal(tabla_persistencia: pd.DataFrame, resultado_ramp: pd.DataFrame, percentiles_limpios: dict) -> pd.DataFrame:
    no_det = resultado_ramp[resultado_ramp["ramp_induced_detected"] == False]  # noqa: E712
    p = tabla_persistencia[tabla_persistencia["episode_id"].isin(no_det["episode_id"])]
    n = len(p)
    p90 = percentiles_limpios["p90"]
    umbral_val = no_det["ramp_threshold"].iloc[0] if len(no_det) else float("nan")
    sobre_p90_sin_cruzar = p[(p["racha_max_sobre_p90"] > 0)]

    filas = [
        {"criterio": ">=2 ventanas consecutivas delta>0", "pct": 100 * (p["racha_max_delta_positivo"] >= 2).mean()},
        {"criterio": ">=3 ventanas consecutivas delta>0", "pct": 100 * (p["racha_max_delta_positivo"] >= 3).mean()},
        {"criterio": ">=4 ventanas consecutivas delta>0", "pct": 100 * (p["racha_max_delta_positivo"] >= 4).mean()},
        {"criterio": ">50% de ventanas con delta positivo", "pct": 100 * (p["pct_ventanas_delta_positivo"] > 50).mean()},
        {"criterio": ">75% de ventanas con delta positivo", "pct": 100 * (p["pct_ventanas_delta_positivo"] > 75).mean()},
        {"criterio": "pendiente del score positiva", "pct": 100 * (p["pendiente_score"] > 0).mean()},
        {"criterio": "correlacion de Spearman (tiempo) positiva", "pct": 100 * (p["spearman_tiempo_score"] > 0).mean()},
        {"criterio": "racha sobre P75 limpio >= 2 ventanas", "pct": 100 * (p["racha_max_sobre_p75"] >= 2).mean()},
        {"criterio": "sobre P90 limpio sin cruzar el umbral puntual", "pct": 100 * len(sobre_p90_sin_cruzar) / n if n else float("nan")},
    ]
    tabla = pd.DataFrame(filas)
    tabla.attrs["n_no_detectados"] = n
    return tabla


# ==========================================================================================
# 13) Clasificacion temporal de rampas NO detectadas (grupos A-D) + detectadas (E)
# ==========================================================================================

DESCRIPCIONES_GRUPO_RAMP = {
    "E": "Detectado: existe alarma inducida atribuible al ramp.",
    "A": "Sin respuesta: el score maximo nunca supera al limpio (max_score_delta <= 0).",
    "B": "Respuesta aislada: hay algun delta positivo, pero la racha maxima es de 1 sola ventana.",
    "C": "Respuesta persistente: racha maxima >= 2 ventanas, sin tendencia creciente clara.",
    "D": "Tendencia creciente: >=3 ventanas evaluables, pendiente y correlacion positivas, y el score maximo cae en la 2a mitad del episodio.",
}


def clasificar_temporal(resultado_ramp: pd.DataFrame, tabla_persistencia: pd.DataFrame) -> pd.DataFrame:
    df = resultado_ramp.merge(tabla_persistencia, on="episode_id")
    grupo = np.empty(len(df), dtype=object)
    detectado = df["ramp_induced_detected"].to_numpy(dtype=bool)
    grupo[detectado] = "E"

    no_det = ~detectado
    sin_respuesta = no_det & (df["max_score_delta"].to_numpy() <= 0)
    grupo[sin_respuesta] = "A"

    resto = no_det & ~sin_respuesta
    tendencia = resto & (df["n_ventanas_afectadas"].to_numpy() >= MIN_VENTANAS_TENDENCIA) & \
        (df["pendiente_score"].to_numpy() > 0) & (df["spearman_tiempo_score"].to_numpy() > 0) & \
        (df["posicion_relativa_score_maximo"].to_numpy() >= 0.5)
    grupo[resto & tendencia] = "D"

    resto2 = resto & ~tendencia
    persistente = resto2 & (df["racha_max_delta_positivo"].to_numpy() >= 2)
    grupo[resto2 & persistente] = "C"
    grupo[resto2 & ~persistente] = "B"

    assert (grupo != None).all() and set(np.unique(grupo)) <= set("ABCDE")  # noqa: E711
    df["grupo_temporal"] = grupo
    df["grupo_temporal_descripcion"] = df["grupo_temporal"].map(DESCRIPCIONES_GRUPO_RAMP)
    return df


# ==========================================================================================
# 11) Comparaciones pareadas (McNemar + bootstrap + Wilcoxon + tamano del efecto)
# ==========================================================================================

def _rank_biserial(diffs: np.ndarray) -> float:
    d = diffs[diffs != 0]
    if len(d) == 0:
        return 0.0
    return float(((d > 0).sum() - (d < 0).sum()) / len(d))


def comparacion_pareada_ramp(df: pd.DataFrame, sufijo_a: str, sufijo_b: str, nombre_a: str, nombre_b: str) -> pd.DataFrame:
    subconjuntos = {
        "global": df, "30_120min": df[df["duration_min"].isin([30, 60, 120])],
        "240_360min": df[df["duration_min"].isin([240, 360])], "720_1440min": df[df["duration_min"].isin([720, 1440])],
    }
    for dur in DURACIONES:
        subconjuntos[f"{dur}min"] = df[df["duration_min"] == dur]
    for rho in RHO_FINALES:
        subconjuntos[f"rho_final_{rho}"] = df[df["rho_final"] == rho]

    filas = []
    for nombre_sub, sub in subconjuntos.items():
        if len(sub) == 0:
            continue
        det_a = sub[f"{sufijo_a}_induced_detected"].to_numpy(dtype=bool)
        det_b = sub[f"{sufijo_b}_induced_detected"].to_numpy(dtype=bool)
        solo_a, solo_b = int((det_a & ~det_b).sum()), int((~det_a & det_b).sum())
        p_mcnemar = mcnemar_p(solo_a, solo_b)
        ci_dr_low, ci_dr_high = bootstrap_pareado(det_a.astype(float), det_b.astype(float))

        hidden = sub["hidden_energy_ramp_kwh"].to_numpy()
        hidden_a = np.where(det_a, hidden, 0.0); hidden_b = np.where(det_b, hidden, 0.0)
        total = hidden.sum()
        ci_e_low, ci_e_high = bootstrap_pareado(hidden_a, hidden_b) if total > EPS_ENERGIA else (float("nan"), float("nan"))

        score_a = sub[f"{sufijo_a}_max_attacked_score"].to_numpy(); score_b = sub[f"{sufijo_b}_max_attacked_score"].to_numpy()
        try:
            w_stat, p_w_score = sstats.wilcoxon(score_a, score_b)
        except ValueError:
            w_stat, p_w_score = float("nan"), float("nan")
        efecto_score = _rank_biserial(score_b - score_a)

        ambos_det = det_a & det_b
        if ambos_det.sum() >= 5:
            r_a = sub.loc[ambos_det, f"{sufijo_a}_detection_delay_min"].to_numpy()
            r_b = sub.loc[ambos_det, f"{sufijo_b}_detection_delay_min"].to_numpy()
            try:
                _, p_w_retraso = sstats.wilcoxon(r_a, r_b)
            except ValueError:
                p_w_retraso = float("nan")
            efecto_retraso = _rank_biserial(r_b - r_a)
        else:
            p_w_retraso, efecto_retraso = float("nan"), float("nan")

        filas.append({
            "comparacion": f"{nombre_a}_vs_{nombre_b}", "subconjunto": nombre_sub, "n_episodios": len(sub),
            "DR_a_pct": 100 * det_a.mean(), "DR_b_pct": 100 * det_b.mean(), "diferencia_DR_pp": 100 * (det_a.mean() - det_b.mean()),
            "mcnemar_p": p_mcnemar, "bootstrap_dr_ci95_low": ci_dr_low, "bootstrap_dr_ci95_high": ci_dr_high,
            "energy_weighted_dr_a_pct": 100 * hidden_a.sum() / total if total > EPS_ENERGIA else float("nan"),
            "energy_weighted_dr_b_pct": 100 * hidden_b.sum() / total if total > EPS_ENERGIA else float("nan"),
            "bootstrap_energia_ci95_low": ci_e_low, "bootstrap_energia_ci95_high": ci_e_high,
            "wilcoxon_p_score_maximo": p_w_score, "efecto_rank_biserial_score": efecto_score,
            "wilcoxon_p_retraso": p_w_retraso, "efecto_rank_biserial_retraso": efecto_retraso, "n_detectados_ambos": int(ambos_det.sum()),
            "significativo_mcnemar_p<0.05": p_mcnemar < 0.05,
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# 12) Metricas principales (global / por duracion / por rho_final)
# ==========================================================================================

def resumen_metricas_ramp(grupo: pd.DataFrame) -> dict:
    n = len(grupo)
    n_ind = int(grupo["ramp_induced_detected"].sum())
    ci = wilson_ci(n_ind, n) if n else (float("nan"), float("nan"))
    con_ind = grupo[grupo["ramp_induced_detected"] == True]  # noqa: E712
    total_hidden = grupo["hidden_energy_ramp_kwh"].sum()
    detected_hidden = grupo.loc[grupo["ramp_induced_detected"], "hidden_energy_ramp_kwh"].sum()
    after_total = grupo["ramp_hidden_energy_after_detection_kwh"].sum()
    grupos_temp = grupo["grupo_temporal"].value_counts()
    return {
        "n_episodios": n, "n_detectados": n_ind, "induced_DR_pct": 100 * n_ind / n if n else float("nan"),
        "induced_DR_ci95_low": ci[0], "induced_DR_ci95_high": ci[1],  # wilson_ci ya devuelve porcentaje
        "energy_weighted_dr_pct": 100 * detected_hidden / total_hidden if total_hidden > EPS_ENERGIA else float("nan"),
        "timely_energy_detection_rate_pct": 100 * after_total / total_hidden if total_hidden > EPS_ENERGIA else float("nan"),
        "retraso_mediano_min": float(con_ind["ramp_detection_delay_min"].median()) if len(con_ind) else float("nan"),
        "retraso_p75_min": float(con_ind["ramp_detection_delay_min"].quantile(0.75)) if len(con_ind) else float("nan"),
        "retraso_p90_min": float(con_ind["ramp_detection_delay_min"].quantile(0.90)) if len(con_ind) else float("nan"),
        "pct_detectado_durante_ataque": 100 * (con_ind["ramp_detected_during_attack"] == True).mean() if len(con_ind) else float("nan"),  # noqa: E712
        "pct_detectado_despues": 100 * (con_ind["ramp_detected_after_attack"] == True).mean() if len(con_ind) else float("nan"),  # noqa: E712
        "hidden_energy_before_detection_total_kwh": float(grupo["ramp_hidden_energy_before_detection_kwh"].sum()),
        "pct_recuperable_medio": float(grupo["ramp_recoverable_hidden_energy_pct"].mean()),
        "pct_grupo_A_sin_respuesta": 100 * grupos_temp.get("A", 0) / n if n else float("nan"),
        "pct_grupo_C_persistente": 100 * grupos_temp.get("C", 0) / n if n else float("nan"),
        "pct_grupo_D_tendencia": 100 * grupos_temp.get("D", 0) / n if n else float("nan"),
    }


def tablas_metricas_ramp(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tablas = {"global": pd.DataFrame([resumen_metricas_ramp(df)])}
    filas = []
    for dur, g in df.groupby("duration_min"):
        fila = {"duration_min": dur}; fila.update(resumen_metricas_ramp(g)); filas.append(fila)
    tablas["por_duracion"] = pd.DataFrame(filas)
    filas = []
    for rho, g in df.groupby("rho_final"):
        fila = {"rho_final": rho}; fila.update(resumen_metricas_ramp(g)); filas.append(fila)
    tablas["por_severidad"] = pd.DataFrame(filas)
    return tablas


# ==========================================================================================
# 14) Alto impacto entre las rampas no detectadas
# ==========================================================================================

def analizar_alto_impacto(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    p75 = float(df["hidden_energy_ramp_kwh"].quantile(0.75))
    no_det = df[df["ramp_induced_detected"] == False]  # noqa: E712
    alto_impacto = no_det[no_det["hidden_energy_ramp_kwh"] >= p75].sort_values("hidden_energy_ramp_kwh", ascending=False)
    return alto_impacto, p75


# ==========================================================================================
# 15) Casos visuales
# ==========================================================================================

def _elegir_representativos(sub: pd.DataFrame, columna: str, n: int = 3) -> list[str]:
    if len(sub) == 0:
        return []
    valores = sub[columna].to_numpy()
    elegidos, usados = [], set()
    for p in np.linspace(25, 75, n) if n > 1 else [50]:
        objetivo = np.percentile(valores, p)
        orden = np.argsort(np.abs(valores - objetivo))
        for idx in orden:
            ep = sub.iloc[idx]["episode_id"]
            if ep not in usados:
                elegidos.append(ep); usados.add(ep)
                break
    return elegidos


def figura_caso_ramp(episode_id: str, ataques_ramp: dict, particion_val_raw: np.ndarray, tabla_ventanas: pd.DataFrame,
                      umbral: float, inicio_particion, resultado_ramp: pd.DataFrame, nombre_archivo: str, etiqueta: str) -> None:
    a = ataques_ramp[episode_id]
    offset, dur = a["offset_global"], a["duracion"]
    margen = max(60, dur // 4)
    ini, fin = max(0, offset - margen), offset + dur + margen
    x_limpio = particion_val_raw[ini:fin].astype(np.float64)
    tramo_limpio = x_limpio[offset - ini:offset - ini + dur]
    x_ramp = x_limpio.copy(); x_ramp[offset - ini:offset - ini + dur] = a["y_tramo_ramp"]
    t = [inicio_particion + pd.Timedelta(minutes=m) for m in range(ini, fin)]
    rho_t = a["y_tramo_ramp"] / np.maximum(tramo_limpio, 1e-6)
    energia_acum = np.cumsum(tramo_limpio - a["y_tramo_ramp"]) / 60.0

    vt = tabla_ventanas[tabla_ventanas["episode_id"] == episode_id].sort_values("window_end")
    t_score = pd.to_datetime(vt["window_end"])
    fila = resultado_ramp[resultado_ramp["episode_id"] == episode_id].iloc[0]
    t_ataque = [inicio_particion + pd.Timedelta(minutes=offset + m) for m in range(dur)]

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=False)
    axes[0].plot(t, x_limpio, color=GRIS, linewidth=1.1, label="limpio")
    axes[0].plot(t, x_ramp, color=ROJO, linewidth=1.2, alpha=0.85, label="ramp")
    axes[0].axvspan(inicio_particion + pd.Timedelta(minutes=offset), inicio_particion + pd.Timedelta(minutes=offset + dur), color=VERDE, alpha=0.08)
    if fila["ramp_induced_detected"]:
        axes[0].axvline(fila["ramp_first_alarm_timestamp"], color=MORADO, linestyle=":", linewidth=1.6, label="1a alarma")
    axes[0].set_ylabel("kW"); axes[0].legend(fontsize=7.5)
    axes[0].set_title(f"{episode_id} | grupo={etiqueta} | dur={dur}min", fontsize=9.5)

    axes[1].plot(t_ataque, rho_t, color=NARANJA, linewidth=1.3)
    axes[1].set_ylabel("rho_t")

    axes[2].plot(t_ataque, energia_acum, color=NARANJA, linewidth=1.3)
    axes[2].set_ylabel("energia ocultada acum. (kWh)")

    axes[3].plot(t_score, vt["clean_score"], color=GRIS, marker="o", markersize=3, label="score limpio")
    axes[3].plot(t_score, vt["ramp_score"], color=ROJO, marker="o", markersize=3, label="score ramp")
    axes[3].axhline(umbral, color=GRIS_OSCURO, linestyle="--", linewidth=1, label="umbral")
    axes[3].set_ylabel(SCORE); axes[3].legend(fontsize=7.5)

    axes[4].bar(t_score, vt["ramp_score_delta"], color=AZUL, width=0.03)
    axes[4].axhline(0, color=GRIS_OSCURO, linewidth=0.8)
    axes[4].set_ylabel("score_delta")
    fig.tight_layout(); _guardar_fig(fig, nombre_archivo)


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("Cargando modelo, ataques existentes y rejilla val (checkpoint existente, sin reentrenar)...")
    config, datos, modelo, ataques_originales, master_csv = cargar_ataques_y_modelo_360()
    params_norm = datos["params_norm"]
    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    serie_antes = particion_val_raw.copy()
    n_dias = len(particion_val_raw) / 1440.0
    grid_60 = construir_rejilla(particion_val, STRIDE)
    inicio_particion = particion_val.index[0]

    X_norm_clean = aplicar_zscore(grid_60["X"], params_norm)
    X_hat_clean = modelo.reconstruir(X_norm_clean)
    scores_limpios_60 = calcular_relative_kw(X_norm_clean, X_hat_clean, params_norm)
    umbral, tasa_eventos, _ = calibrar_umbral_por_eventos(scores_limpios_60, n_dias, 0.06)
    print(f"Umbral puntual (0.06 eventos/dia): {umbral:.5f} | eventos/dia real={tasa_eventos:.4f}")

    print("\n[2/4] Construyendo episodios ramp_lineal (mismos timestamps que reduccion_constante)...")
    tabla_ramp, ataques_ramp = construir_episodios_ramp(master_csv, particion_val_raw, inicio_particion)
    print(f"OK -- {len(tabla_ramp)} episodios ramp (esperados 1050)")
    assert len(tabla_ramp) == 1050, f"se esperaban 1050 episodios ramp, se obtuvieron {len(tabla_ramp)}"

    print("\nPuntuando las 3 ramas (ramp, constante-endpoint, constante-equivalente)...")
    tabla_ventanas = puntuar_tres_ramas(ataques_ramp, particion_val_raw, grid_60, modelo, params_norm, umbral, scores_limpios_60)
    _guardar_tabla(tabla_ventanas, "trayectorias_score_ramp")
    print(f"OK -- {len(tabla_ventanas)} filas ventana-episodio")

    print("\nAgregando resultados por episodio (3 ramas)...")
    r_ramp = agregar_episodio_rama(tabla_ventanas, ataques_ramp, tabla_ramp, particion_val_raw, umbral, "ramp_score", "y_tramo_ramp", "ramp")
    r_endpoint = agregar_episodio_rama(tabla_ventanas, ataques_ramp, tabla_ramp, particion_val_raw, umbral, "endpoint_constant_score", "y_tramo_endpoint", "endpoint")
    r_equiv = agregar_episodio_rama(tabla_ventanas, ataques_ramp, tabla_ramp, particion_val_raw, umbral, "energy_equivalent_constant_score", "y_tramo_equivalent", "equivalent")

    resultado = tabla_ramp.merge(r_ramp, on="episode_id").merge(r_endpoint, on="episode_id").merge(r_equiv, on="episode_id")

    # verificacion seccion 5: energia del ramp ~= energia del comparador de energia equivalente
    dif_energia = (resultado["hidden_energy_ramp_kwh"] - resultado["hidden_energy_constant_equivalent_kwh"]).abs()
    assert (dif_energia < 1e-6).all(), "el comparador de energia equivalente no reproduce la energia del ramp"
    print(f"[verificacion] energia ramp == energia comparador equivalente (max dif={dif_energia.max():.2e} kWh)")

    print("\n[8] Calculando persistencia y forma temporal del score...")
    percentiles_limpios = {"p75": float(np.percentile(scores_limpios_60, 75)), "p80": float(np.percentile(scores_limpios_60, 80)),
                            "p90": float(np.percentile(scores_limpios_60, 90)), "p95": float(np.percentile(scores_limpios_60, 95))}
    tabla_persistencia = calcular_persistencia(tabla_ventanas, ataques_ramp, percentiles_limpios)
    _guardar_tabla(tabla_persistencia, "08_persistencia")

    print("\n[13] Clasificando temporalmente los episodios ramp...")
    resultado = clasificar_temporal(resultado, tabla_persistencia)
    _guardar_tabla(resultado, "episodios_ramp")
    print(resultado["grupo_temporal"].value_counts().to_string())

    print("\n[9] Evidencia de senal temporal aprovechable (entre no detectados)...")
    tabla_evidencia = evidencia_senal_temporal(tabla_persistencia, resultado, percentiles_limpios)
    _guardar_tabla(tabla_evidencia, "09_evidencia_senal_temporal")
    print(tabla_evidencia.to_string(index=False))

    print("\n[12] Metricas principales...")
    tablas_metricas = tablas_metricas_ramp(resultado)
    for nombre, t in tablas_metricas.items():
        _guardar_tabla(t, f"03_metricas_{nombre}")
    print(tablas_metricas["global"].to_string(index=False))

    print("\n[10/11] Comparaciones pareadas: ramp vs endpoint, ramp vs equivalent...")
    pareada_endpoint = comparacion_pareada_ramp(resultado, "ramp", "endpoint", "ramp", "endpoint")
    pareada_equiv = comparacion_pareada_ramp(resultado, "ramp", "equivalent", "ramp", "equivalent")
    pareada = pd.concat([pareada_endpoint, pareada_equiv], ignore_index=True)
    _guardar_tabla(pareada, "11_pruebas_pareadas")
    print(pareada[pareada["subconjunto"] == "global"].to_string(index=False))

    print("\n[14] Rampas no detectadas de alto impacto...")
    alto_impacto, p75_energia = analizar_alto_impacto(resultado)
    _guardar_tabla(alto_impacto, "ramp_no_detectados_alto_impacto")
    print(f"  p75 hidden_energy_ramp_kwh={p75_energia:.4f} | no detectados de alto impacto={len(alto_impacto)}")

    print("\n[15] Seleccionando y graficando casos representativos...")
    detectados = resultado[resultado["ramp_induced_detected"] == True]  # noqa: E712
    casos = {
        "detectados": _elegir_representativos(detectados, "ramp_detection_delay_min", 3),
        "sin_respuesta_A": _elegir_representativos(resultado[resultado["grupo_temporal"] == "A"], "hidden_energy_ramp_kwh", 3),
        "aislada_B": _elegir_representativos(resultado[resultado["grupo_temporal"] == "B"], "max_score_delta", 3),
        "persistente_C": _elegir_representativos(resultado[resultado["grupo_temporal"] == "C"], "racha_max_delta_positivo", 3),
        "tendencia_D": _elegir_representativos(resultado[resultado["grupo_temporal"] == "D"], "pendiente_score", 3),
        "alto_impacto": alto_impacto.nlargest(3, "hidden_energy_ramp_kwh")["episode_id"].tolist(),
    }
    tabla_casos_filas = []
    for etiqueta, ids in casos.items():
        for i, ep_id in enumerate(ids):
            nombre_archivo = f"16_caso_{etiqueta}_{i + 1}"
            figura_caso_ramp(ep_id, ataques_ramp, particion_val_raw, tabla_ventanas, umbral, inicio_particion, resultado, nombre_archivo, etiqueta)
            fila = resultado[resultado["episode_id"] == ep_id].iloc[0].to_dict()
            fila["seleccion"] = etiqueta; tabla_casos_filas.append(fila)
    _guardar_tabla(pd.DataFrame(tabla_casos_filas), "12_casos_seleccionados")

    print("\nGenerando figuras restantes...")
    generar_figuras(resultado, tabla_ventanas, tabla_persistencia, tablas_metricas, pareada, umbral, ataques_ramp,
                     particion_val_raw, inicio_particion)

    assert np.array_equal(serie_antes, particion_val_raw)
    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "config": config, "datos": datos, "modelo": modelo, "ataques_ramp": ataques_ramp, "tabla_ramp": tabla_ramp,
        "resultado": resultado, "tabla_ventanas": tabla_ventanas, "tabla_persistencia": tabla_persistencia,
        "tabla_evidencia": tabla_evidencia, "tablas_metricas": tablas_metricas, "pareada": pareada,
        "alto_impacto": alto_impacto, "p75_energia": p75_energia, "umbral": umbral, "grid_60": grid_60,
        "scores_limpios_60": scores_limpios_60, "percentiles_limpios": percentiles_limpios,
    }


def generar_figuras(resultado, tabla_ventanas, tabla_persistencia, tablas_metricas, pareada, umbral, ataques_ramp,
                     particion_val_raw, inicio_particion) -> None:
    # 1) ejemplo de construccion de un ataque ramp
    ep0 = resultado[(resultado["rho_final"] == 0.50) & (resultado["duration_min"] == 360)]
    if len(ep0) == 0:
        ep0 = resultado[resultado["duration_min"] == 360]
    ep0_id = ep0.iloc[0]["episode_id"]
    a0 = ataques_ramp[ep0_id]
    rho_t = _rho_lineal(a0["duracion"], ep0.iloc[0]["rho_final"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    j = np.arange(a0["duracion"])
    ax1.plot(j, particion_val_raw[a0["offset_global"]:a0["offset_global"] + a0["duracion"]], color=GRIS, label="limpio")
    ax1.plot(j, a0["y_tramo_ramp"], color=ROJO, label="ramp")
    ax1.set_ylabel("kW"); ax1.legend(fontsize=8); ax1.set_title(f"Construccion de un ataque ramp_lineal ({ep0_id})")
    ax2.plot(j, rho_t, color=AZUL); ax2.set_ylabel("rho_t"); ax2.set_xlabel("minuto del ataque")
    fig.tight_layout(); _guardar_fig(fig, "01_ejemplo_construccion_ramp")

    # 2) DR por duracion
    fig, ax = plt.subplots(figsize=(7, 5))
    t = tablas_metricas["por_duracion"].sort_values("duration_min")
    ax.plot(t["duration_min"], t["induced_DR_pct"], marker="o", color=ROJO)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)"); ax.set_title("DR de ramp_lineal por duracion")
    fig.tight_layout(); _guardar_fig(fig, "02_dr_por_duracion")

    # 3) DR por severidad final
    fig, ax = plt.subplots(figsize=(6, 5))
    t = tablas_metricas["por_severidad"].sort_values("rho_final", ascending=False)
    ax.bar(t["rho_final"].astype(str), t["induced_DR_pct"], color=NARANJA)
    ax.set_xlabel("rho_final"); ax.set_ylabel("induced_DR (%)"); ax.set_title("DR de ramp_lineal por severidad final")
    fig.tight_layout(); _guardar_fig(fig, "03_dr_por_severidad")

    # 4) DR ramp vs comparadores
    fig, ax = plt.subplots(figsize=(6, 5))
    dr_ramp = 100 * resultado["ramp_induced_detected"].mean()
    dr_endpoint = 100 * resultado["endpoint_induced_detected"].mean()
    dr_equiv = 100 * resultado["equivalent_induced_detected"].mean()
    ax.bar(["ramp", "constante\n(misma severidad)", "constante\n(misma energia)"], [dr_ramp, dr_endpoint, dr_equiv],
           color=[ROJO, AZUL, VERDE])
    ax.set_ylabel("induced_DR (%)"); ax.set_title("DR: ramp vs comparadores")
    fig.tight_layout(); _guardar_fig(fig, "04_dr_ramp_vs_comparadores")

    # 5) DR ponderado por energia
    fig, ax = plt.subplots(figsize=(6, 5))
    def _e_dr(pref):
        tot = resultado["hidden_energy_ramp_kwh"].sum()
        det = resultado.loc[resultado[f"{pref}_induced_detected"], "hidden_energy_ramp_kwh"].sum()
        return 100 * det / tot if tot > EPS_ENERGIA else float("nan")
    ax.bar(["ramp", "endpoint", "equivalent"], [_e_dr("ramp"), _e_dr("endpoint"), _e_dr("equivalent")], color=[ROJO, AZUL, VERDE])
    ax.set_ylabel("DR ponderado por energia (%)"); ax.set_title("DR energetico: ramp vs comparadores")
    fig.tight_layout(); _guardar_fig(fig, "05_dr_ponderado_energia")

    # 6) retraso por duracion
    fig, ax = plt.subplots(figsize=(7, 5))
    con_ind = resultado[resultado["ramp_induced_detected"] == True]  # noqa: E712
    med = con_ind.groupby("duration_min")["ramp_detection_delay_min"].median()
    ax.plot(med.index, med.values, marker="o", color=ROJO)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("retraso mediano (min)"); ax.set_title("Retraso de ramp_lineal por duracion")
    fig.tight_layout(); _guardar_fig(fig, "06_retraso_por_duracion")

    # 7) trayectoria media del score alineada al inicio
    fig, ax = plt.subplots(figsize=(8, 5))
    tv = tabla_ventanas.copy()
    tv["orden"] = tv.groupby("episode_id")["k"].rank().astype(int) - 1
    med_score = tv.groupby("orden")[["clean_score", "ramp_score"]].median()
    ax.plot(med_score.index, med_score["clean_score"], color=GRIS, label="limpio")
    ax.plot(med_score.index, med_score["ramp_score"], color=ROJO, label="ramp")
    ax.axhline(umbral, color=GRIS_OSCURO, linestyle="--", linewidth=1, label="umbral")
    ax.set_xlabel("ventana afectada (orden)"); ax.set_ylabel("score mediano"); ax.legend(fontsize=8)
    ax.set_title("Trayectoria mediana del score, alineada al inicio del ataque")
    fig.tight_layout(); _guardar_fig(fig, "07_trayectoria_media_score")

    # 8) trayectoria media del score_delta
    fig, ax = plt.subplots(figsize=(8, 5))
    med_delta = tv.groupby("orden")["ramp_score_delta"].median()
    ax.plot(med_delta.index, med_delta.values, color=AZUL, marker="o", markersize=3)
    ax.axhline(0, color=GRIS_OSCURO, linewidth=0.8)
    ax.set_xlabel("ventana afectada (orden)"); ax.set_ylabel("score_delta mediano")
    ax.set_title("Trayectoria mediana de score_delta")
    fig.tight_layout(); _guardar_fig(fig, "08_trayectoria_media_delta")

    # 9) score frente a progreso de la rampa
    fig, ax = plt.subplots(figsize=(7, 5.5))
    n_por_ep = tv.groupby("episode_id")["orden"].transform("max")
    tv["progreso"] = tv["orden"] / n_por_ep.replace(0, 1)
    ax.scatter(tv["progreso"], tv["ramp_score"], s=4, alpha=0.15, color=ROJO)
    prom = tv.groupby(pd.cut(tv["progreso"], 20))["ramp_score"].median()
    ax.plot([iv.mid for iv in prom.index], prom.values, color=GRIS_OSCURO, linewidth=2)
    ax.set_xlabel("progreso de la rampa (0-1)"); ax.set_ylabel("ramp_score"); ax.set_title("Score frente a progreso de la rampa")
    fig.tight_layout(); _guardar_fig(fig, "09_score_vs_progreso")

    # 10) rachas positivas por duracion
    fig, ax = plt.subplots(figsize=(7, 5))
    tp = resultado  # `resultado` ya incluye las columnas de persistencia (fusionadas en clasificar_temporal)
    med_racha = tp.groupby("duration_min")["racha_max_delta_positivo"].median()
    ax.plot(med_racha.index, med_racha.values, marker="o", color=TEAL)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("racha mediana (ventanas)"); ax.set_title("Racha maxima de score_delta>0, por duracion")
    fig.tight_layout(); _guardar_fig(fig, "10_rachas_por_duracion")

    # 11) pendiente por severidad
    fig, ax = plt.subplots(figsize=(6, 5))
    med_pend = tp.groupby("rho_final")["pendiente_score"].median()
    ax.bar(med_pend.index.astype(str), med_pend.values, color=MORADO)
    ax.set_xlabel("rho_final"); ax.set_ylabel("pendiente mediana del score"); ax.set_title("Pendiente del score por severidad")
    fig.tight_layout(); _guardar_fig(fig, "11_pendiente_por_severidad")

    # 12/13) grupos temporales por duracion / severidad
    for col, nombre in (("duration_min", "12_grupos_por_duracion"), ("rho_final", "13_grupos_por_severidad")):
        tabla_gd = resultado.groupby([col, "grupo_temporal"]).size().unstack(fill_value=0).reindex(columns=list("ABCDE"), fill_value=0)
        tabla_gd_pct = tabla_gd.div(tabla_gd.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(tabla_gd_pct))
        for g in "ABCDE":
            ax.bar(tabla_gd_pct.index.astype(str), tabla_gd_pct[g], bottom=bottom, color=COLOR_GRUPO[g], label=g)
            bottom += tabla_gd_pct[g].to_numpy()
        ax.set_xlabel(col); ax.set_ylabel("% episodios"); ax.legend(fontsize=8); ax.set_title(f"Grupos temporales por {col}")
        fig.tight_layout(); _guardar_fig(fig, nombre)

    # 14) energia oculta vs score maximo
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colores = np.where(resultado["ramp_induced_detected"], VERDE, ROJO)
    ax.scatter(resultado["hidden_energy_ramp_kwh"], resultado["max_score_delta"], c=colores, s=10, alpha=0.4)
    ax.axhline(0, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_xlabel("energia oculta (kWh)"); ax.set_ylabel("max_score_delta"); ax.set_title("Energia oculta vs score maximo (verde=detectado)")
    fig.tight_layout(); _guardar_fig(fig, "14_energia_vs_score_maximo")

    # 15) energia oculta vs persistencia
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(tp["hidden_energy_ramp_kwh"], tp["racha_max_delta_positivo"], c=np.where(tp["ramp_induced_detected"], VERDE, ROJO), s=10, alpha=0.4)
    ax.set_xlabel("energia oculta (kWh)"); ax.set_ylabel("racha maxima delta>0"); ax.set_title("Energia oculta vs persistencia")
    fig.tight_layout(); _guardar_fig(fig, "15_energia_vs_persistencia")

    # 17) comparaciones de igual energia: DR por duracion, ramp vs equivalent
    fig, ax = plt.subplots(figsize=(7, 5))
    dr_r = resultado.groupby("duration_min")["ramp_induced_detected"].mean() * 100
    dr_e = resultado.groupby("duration_min")["equivalent_induced_detected"].mean() * 100
    ax.plot(dr_r.index, dr_r.values, marker="o", color=ROJO, label="ramp")
    ax.plot(dr_e.index, dr_e.values, marker="o", color=VERDE, label="constante (misma energia)")
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)"); ax.legend(fontsize=8)
    ax.set_title("Ramp vs constante de igual energia, por duracion")
    fig.tight_layout(); _guardar_fig(fig, "17_comparacion_igual_energia")

    # 18) tabla visual de conclusiones
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    glob = pareada[pareada["subconjunto"] == "global"][["comparacion", "DR_a_pct", "DR_b_pct", "diferencia_DR_pp", "mcnemar_p"]].round(3)
    tabla_mpl = ax.table(cellText=glob.values, colLabels=glob.columns, loc="center", cellLoc="center")
    tabla_mpl.auto_set_font_size(False); tabla_mpl.set_fontsize(9); tabla_mpl.scale(1, 1.8)
    ax.set_title("Conclusiones: ramp vs comparadores (global)", fontsize=11)
    fig.tight_layout(); _guardar_fig(fig, "18_tabla_conclusiones")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
