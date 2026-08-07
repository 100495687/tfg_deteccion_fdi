"""Experimento 4: comparar formas de agregar el residuo de reconstruccion (anomaly score)
usando el mismo TCN-AE de ventana=360/stride=360 -- sin reentrenar nada, sin tocar ventana,
stride, ataques ni particiones.

Los 3 experimentos anteriores (stride, tamano de ventana) empeoraron el DR al recalibrar el
umbral de forma justa. El cuello de botella probable esta en como se agrega el residuo
punto a punto en un unico numero por ventana, no en la rejilla -- este experimento ataca
justo eso.

Residuo puntual comun a todos los scores:
    signed_deficit_t   = reconstruction_t - observed_t      (espacio normalizado, z-score)
    positive_deficit_t = max(signed_deficit_t, 0)

Todos se derivan de la MISMA reconstruccion (una llamada a `modelo.reconstruir` para toda
la rejilla limpia, y otra para todas las ventanas atacadas) -- nunca se repite la inferencia.

Scores:
    mean       = positive_deficit_t.mean()                              (baseline)
    relative   = mean( positive_deficit_kW_t / max(|reconstruction_kW_t|, epsilon) )
    top1/5/10  = mean de los k% valores mas altos de positive_deficit_t
    hybrid_a   = mean + a * top5, a en {0.1, 0.25, 0.5, 1.0}

Nota sobre `relative`: la serie de entrada esta en z-score (media≈0, cruza por cero), asi
que dividir por |reconstruction_t| en ese espacio no tiene sentido fisico y es inestable
(el denominador se acerca a 0 justo en las horas de consumo bajo, que es donde mas interesa
que el score sea fiable). Por eso se calcula desnormalizado, en kW, con epsilon=0.1 kW como
suelo de seguridad -- el consumo minimo observado en el hogar (~0.076 kW) esta justo por
debajo, asi que 0.1 kW evita explosiones sin descartar informacion real.

Ejecutable de forma aislada:
    python -m src.experimento_scores
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.anomaly_score import calcular_scores
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import wilson_ci
from src.experimento_stride import (
    bootstrap_pareado,
    calibrar_umbral_por_eventos,
    cargar_datos_y_modelo,
    contar_eventos,
    mcnemar_p,
    reconstruir_ataques_exactos,
)
from src.experimento_ventana120 import _ventana_atacada, _ventanas_disjuntas_afectadas
from src.normalization import aplicar_zscore, revertir_zscore

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables"
EXP_TABLAS_DIR = TABLAS_DIR / "experimento_scores"
EXP_FIGURAS_DIR = BASE_DIR / "results" / "figures" / "experimento_scores"

AZUL = "#2a78d6"
NARANJA = "#eb6834"
VERDE = "#2f9e44"
GRIS = "#8a8a86"
ROJO = "#c0392b"
MORADO = "#7d3c98"
TEAL = "#17a398"
MARRON = "#8a5a3c"
ROSA = "#c2548c"
GRIS_OSCURO = "#3d3d3d"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
EPSILON_KW = 0.1
ALPHAS_HIBRIDO = (0.1, 0.25, 0.5, 1.0)
NOMBRES_SCORE = ["mean", "relative", "top1", "top5", "top10"] + [f"hybrid_{a}" for a in ALPHAS_HIBRIDO]
BASELINE = "mean"
COLOR_SCORE = {
    "mean": GRIS_OSCURO, "relative": VERDE, "top1": "#bcd7f0", "top5": AZUL, "top10": "#0d3d73",
    "hybrid_0.1": "#f4b183", "hybrid_0.25": NARANJA, "hybrid_0.5": ROJO, "hybrid_1.0": MORADO,
}
OBJETIVOS_EVENTOS_DIA = {"principal_0.06": 0.06, "sec_0.05": 0.05, "sec_0.10": 0.10, "sec_0.25": 0.25}
OBJETIVO_PRINCIPAL = "principal_0.06"
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
DURACIONES_CORTAS = [30, 60, 120]


def _guardar_fig(fig, nombre):
    EXP_FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP_FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    EXP_TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP_TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ----------------------------------------------------------------------------------------
# 1) Residuo comun -> todos los scores, a partir de UNA reconstruccion ya calculada
# ----------------------------------------------------------------------------------------

def _top_k_muestras(pct: float, n: int) -> int:
    return max(1, round(pct / 100 * n))


def calcular_todos_los_scores(X_norm: np.ndarray, X_hat_norm: np.ndarray, params_norm: dict,
                               epsilon_kw: float = EPSILON_KW) -> dict[str, np.ndarray]:
    """A partir de UNA reconstruccion ya calculada (X_hat_norm), deriva los 9 scores. No
    vuelve a llamar a `modelo.reconstruir` en ningun caso."""
    signed_deficit = X_hat_norm - X_norm
    positive_deficit = np.maximum(signed_deficit, 0.0)
    n = X_norm.shape[1]

    scores: dict[str, np.ndarray] = {"mean": positive_deficit.mean(axis=1)}

    # relativo: en kW (desnormalizado), nunca en z-score -- ver docstring del modulo
    X_kw = revertir_zscore(X_norm, params_norm)
    X_hat_kw = revertir_zscore(X_hat_norm, params_norm)
    positive_deficit_kw = np.maximum(X_hat_kw - X_kw, 0.0)
    denominador_kw = np.maximum(np.abs(X_hat_kw), epsilon_kw)
    scores["relative"] = (positive_deficit_kw / denominador_kw).mean(axis=1)

    ordenado = np.sort(positive_deficit, axis=1)  # ascendente
    for pct, nombre in ((1, "top1"), (5, "top5"), (10, "top10")):
        k = _top_k_muestras(pct, n)
        scores[nombre] = ordenado[:, -k:].mean(axis=1)

    for alpha in ALPHAS_HIBRIDO:
        scores[f"hybrid_{alpha}"] = scores["mean"] + alpha * scores["top5"]

    assert not np.any(np.isnan(scores["relative"])), "NaN en score relativo"
    assert not np.any(np.isinf(scores["relative"])), "Inf en score relativo"
    return scores


# ----------------------------------------------------------------------------------------
# 2) Puntuacion: UNA reconstruccion para toda la rejilla limpia, UNA para todas las
#    ventanas atacadas de todos los episodios -- todos los scores salen de ahi.
# ----------------------------------------------------------------------------------------

def puntuar_todos_los_scores(ataques: dict, particion_val_raw: np.ndarray, grid: dict, modelo,
                              params_norm: dict) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    n_ventanas_grid = grid["X"].shape[0]

    X_clean_norm = aplicar_zscore(grid["X"], params_norm)
    X_hat_clean = modelo.reconstruir(X_clean_norm)                      # UNA llamada, rejilla completa
    scores_limpios = calcular_todos_los_scores(X_clean_norm, X_hat_clean, params_norm)

    filas_meta, ventanas_atacadas = [], []
    for episode_id, ataque in ataques.items():
        ks = _ventanas_disjuntas_afectadas(ataque["offset_global"], ataque["duracion"], TAMANO_VENTANA, n_ventanas_grid)
        assert len(ks) >= 1
        for k in ks:
            w_start = k * TAMANO_VENTANA
            va = _ventana_atacada(particion_val_raw, w_start, ataque["offset_global"], ataque["duracion"],
                                   ataque["y_tramo"], TAMANO_VENTANA)
            ventanas_atacadas.append(va)
            filas_meta.append({"episode_id": episode_id, "k": k, "window_start": grid["inicio"][k], "window_end": grid["fin"][k]})

    X_atacadas_norm = aplicar_zscore(np.stack(ventanas_atacadas).astype(np.float32), params_norm)
    X_hat_atacadas = modelo.reconstruir(X_atacadas_norm)                # UNA llamada, TODAS las ventanas atacadas
    scores_atacados = calcular_todos_los_scores(X_atacadas_norm, X_hat_atacadas, params_norm)

    meta = pd.DataFrame(filas_meta)
    ks_arr = meta["k"].to_numpy()
    tablas_por_score = {}
    for nombre in NOMBRES_SCORE:
        t = meta.copy()
        t["clean_score"] = scores_limpios[nombre][ks_arr]
        t["attacked_score"] = scores_atacados[nombre]
        tablas_por_score[nombre] = t
    return tablas_por_score, scores_limpios


# ----------------------------------------------------------------------------------------
# 3) Agregacion por umbral (por episodio) -- mismo esquema que experimentos anteriores
# ----------------------------------------------------------------------------------------

def agregar_por_umbral(tabla_ventanas: pd.DataFrame, master_csv: pd.DataFrame, umbral: float,
                        nombre_score: str) -> pd.DataFrame:
    t = tabla_ventanas.copy()
    t["window_end"] = pd.to_datetime(t["window_end"])
    t["clean_is_alarm"] = t["clean_score"] > umbral
    t["attacked_is_alarm"] = t["attacked_score"] > umbral
    t["attack_induced_alarm"] = t["attacked_is_alarm"] & ~t["clean_is_alarm"]
    t["score_delta"] = t["attacked_score"] - t["clean_score"]

    agg = t.groupby("episode_id").agg(
        n_affected_windows=("k", "size"),
        raw_detected=("attacked_is_alarm", "any"),
        induced_detected=("attack_induced_alarm", "any"),
        max_attacked_score=("attacked_score", "max"),
        max_clean_score=("clean_score", "max"),
        max_score_delta=("score_delta", "max"),
    ).reset_index()

    raw_ts = t[t["attacked_is_alarm"]].groupby("episode_id")["window_end"].min().rename("first_raw_alarm_timestamp")
    induced_ts = t[t["attack_induced_alarm"]].groupby("episode_id")["window_end"].min().rename("first_induced_alarm_timestamp")
    agg = agg.merge(raw_ts, on="episode_id", how="left").merge(induced_ts, on="episode_id", how="left")

    resultado = agg.merge(
        master_csv[["episode_id", "family", "parameter", "duration_min", "attack_start", "attack_end"]],
        on="episode_id", how="left",
    )
    resultado["raw_detection_delay_min"] = (resultado["first_raw_alarm_timestamp"] - resultado["attack_start"]) / pd.Timedelta(minutes=1)
    resultado["induced_detection_delay_min"] = (resultado["first_induced_alarm_timestamp"] - resultado["attack_start"]) / pd.Timedelta(minutes=1)

    durante = (resultado["first_induced_alarm_timestamp"] <= resultado["attack_end"]).astype(object)
    despues = (resultado["first_induced_alarm_timestamp"] > resultado["attack_end"]).astype(object)
    sin_induced = ~resultado["induced_detected"]
    durante[sin_induced] = np.nan
    despues[sin_induced] = np.nan
    resultado["induced_detected_during_attack"] = durante
    resultado["induced_detected_after_attack"] = despues
    resultado.loc[sin_induced, "induced_detection_delay_min"] = np.nan
    resultado.loc[~resultado["raw_detected"], "raw_detection_delay_min"] = np.nan

    resultado["score_type"] = nombre_score
    resultado["threshold"] = umbral
    return resultado


# ----------------------------------------------------------------------------------------
# 4) Duracion de eventos falsos (para la seccion 5 del encargo)
# ----------------------------------------------------------------------------------------

def duraciones_eventos_min(es_alarma: np.ndarray, tamano_ventana: int) -> np.ndarray:
    duraciones, contador = [], 0
    for a in es_alarma:
        if a:
            contador += 1
        else:
            if contador > 0:
                duraciones.append(contador * tamano_ventana)
            contador = 0
    if contador > 0:
        duraciones.append(contador * tamano_ventana)
    return np.array(duraciones) if duraciones else np.array([0.0])


# ----------------------------------------------------------------------------------------
# 5) Resumen por grupo (episodios + ventanas)
# ----------------------------------------------------------------------------------------

def _resumen(grupo: pd.DataFrame, ventanas_grupo: pd.DataFrame | None = None) -> dict:
    n = len(grupo)
    n_induced = int(grupo["induced_detected"].sum())
    ci = wilson_ci(n_induced, n)
    con_induced = grupo[grupo["induced_detected"] == True]  # noqa: E712
    resultado = {
        "n_episodios": n,
        "induced_DR_pct": 100 * n_induced / n if n else float("nan"),
        "induced_DR_ci95_low": ci[0], "induced_DR_ci95_high": ci[1],
        "retraso_mediano_induced_min": con_induced["induced_detection_delay_min"].median() if len(con_induced) else float("nan"),
        "pct_detectado_durante_ataque": 100 * (con_induced["induced_detected_during_attack"] == True).mean() if len(con_induced) else float("nan"),  # noqa: E712
    }
    if ventanas_grupo is not None and len(ventanas_grupo):
        delta = ventanas_grupo["score_delta"] if "score_delta" in ventanas_grupo.columns else (
            ventanas_grupo["attacked_score"] - ventanas_grupo["clean_score"])
        resultado["score_delta_medio"] = delta.mean()
        resultado["score_delta_mediano"] = delta.median()
        resultado["pct_ventanas_delta_positivo"] = 100 * (delta > 0).mean()
        resultado["pct_ventanas_delta_negativo"] = 100 * (delta < 0).mean()
    return resultado


# ----------------------------------------------------------------------------------------
# Orquestacion principal
# ----------------------------------------------------------------------------------------

def main() -> dict:
    config, datos, modelo = cargar_datos_y_modelo()  # checkpoint EXISTENTE, no reentrena
    tipo_score_config = config["score"]["tipo"]
    params_norm = datos["params_norm"]

    print("Reconstruyendo ataques exactos de episodes_master_val.csv...")
    ataques, master_csv = reconstruir_ataques_exactos(config, datos)
    print(f"{len(ataques)} episodios validados.\n")

    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    n_dias = len(particion_val_raw) / 1440.0
    grid = datos["ventanas"]["val"]

    print("Calculando reconstrucciones (2 llamadas al modelo en total) y derivando los 9 scores...")
    tablas_por_score, scores_limpios = puntuar_todos_los_scores(ataques, particion_val_raw, grid, modelo, params_norm)
    print(f"{len(tablas_por_score['mean'])} filas ventana-episodio por score.\n")

    # --- baseline de sanidad: el score 'deficit' original (mean SIN clip) para contraste ---
    scores_deficit_original = calcular_scores(modelo, grid_norm := aplicar_zscore(grid["X"], params_norm), "deficit")
    print(f"[sanity] score 'mean' (clip>=0): media={scores_limpios['mean'].mean():.5f}, "
          f"score 'deficit' original (con signo, sin clip): media={scores_deficit_original.mean():.5f}\n")

    filas_umbrales, filas_duracion_eventos = [], []
    resultados_por_score_objetivo = []

    for nombre_score in NOMBRES_SCORE:
        print(f"=== score: {nombre_score} ===")
        for objetivo_nombre, objetivo_valor in OBJETIVOS_EVENTOS_DIA.items():
            umbral, tasa_eventos, tasa_ventanas = calibrar_umbral_por_eventos(scores_limpios[nombre_score], n_dias, objetivo_valor)
            es_alarma = scores_limpios[nombre_score] > umbral
            n_eventos = contar_eventos(es_alarma)
            duraciones = duraciones_eventos_min(es_alarma, TAMANO_VENTANA)

            filas_umbrales.append({
                "score_type": nombre_score, "objetivo_nombre": objetivo_nombre, "objetivo_valor": objetivo_valor,
                "threshold": umbral, "tasa_eventos_dia_real": tasa_eventos, "tasa_ventanas_dia_real": tasa_ventanas,
                "n_eventos_falsos": n_eventos, "duracion_media_evento_min": float(duraciones.mean()),
                "duracion_maxima_evento_min": float(duraciones.max()),
            })
            if objetivo_nombre == OBJETIVO_PRINCIPAL:
                print(f"    {objetivo_nombre}: umbral={umbral:.5f}, eventos/dia={tasa_eventos:.4f}, "
                      f"n_eventos={n_eventos}, dur_media={duraciones.mean():.0f}min, dur_max={duraciones.max():.0f}min")

            resultado_ep = agregar_por_umbral(tablas_por_score[nombre_score], master_csv, umbral, nombre_score)
            resultado_ep["objetivo_nombre"] = objetivo_nombre
            resultado_ep["objetivo_valor"] = objetivo_valor
            resultados_por_score_objetivo.append(resultado_ep)
        print()

    umbrales_tabla = _guardar_tabla(pd.DataFrame(filas_umbrales), "01_umbrales_por_score_objetivo")
    resultados_todo = pd.concat(resultados_por_score_objetivo, ignore_index=True)
    _guardar_tabla(resultados_todo, "01_episode_detection_results_scores")

    print("Construyendo agregados...")
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL].copy()

    filas = []
    for score_type, grupo in principal.groupby("score_type"):
        vg = tablas_por_score[score_type]
        vg = vg.assign(score_delta=vg["attacked_score"] - vg["clean_score"])
        fila = {"score_type": score_type}
        fila.update(_resumen(grupo, vg))
        u = umbrales_tabla[(umbrales_tabla["score_type"] == score_type) & (umbrales_tabla["objetivo_nombre"] == OBJETIVO_PRINCIPAL)].iloc[0]
        fila["false_alarm_events_per_day"] = u["tasa_eventos_dia_real"]
        fila["false_alarm_windows_per_day"] = u["tasa_ventanas_dia_real"]
        filas.append(fila)
    tabla_por_score = _guardar_tabla(pd.DataFrame(filas).sort_values("induced_DR_pct", ascending=False), "02_por_score")
    print(tabla_por_score.to_string(index=False))

    filas = []
    for (score_type, dur), grupo in principal.groupby(["score_type", "duration_min"]):
        vg = tablas_por_score[score_type]
        vg = vg[vg["episode_id"].isin(grupo["episode_id"])].assign(score_delta=lambda d: d["attacked_score"] - d["clean_score"])
        fila = {"score_type": score_type, "duration_min": dur}
        fila.update(_resumen(grupo, vg))
        filas.append(fila)
    tabla_score_duracion = _guardar_tabla(pd.DataFrame(filas), "02_por_score_duracion")

    filas = []
    for (score_type, fam), grupo in principal.groupby(["score_type", "family"]):
        vg = tablas_por_score[score_type]
        vg = vg[vg["episode_id"].isin(grupo["episode_id"])].assign(score_delta=lambda d: d["attacked_score"] - d["clean_score"])
        fila = {"score_type": score_type, "family": fam}
        fila.update(_resumen(grupo, vg))
        filas.append(fila)
    tabla_score_familia = _guardar_tabla(pd.DataFrame(filas), "02_por_score_familia")

    filas = []
    for (score_type, fam, param), grupo in principal.groupby(["score_type", "family", "parameter"]):
        fila = {"score_type": score_type, "family": fam, "parameter": param}
        fila.update(_resumen(grupo))
        filas.append(fila)
    _guardar_tabla(pd.DataFrame(filas), "02_por_score_familia_parametro")

    filas = []
    for score_type, grupo in principal.groupby("score_type"):
        fila = {"score_type": score_type, "familias": "todas"}
        fila.update(_resumen(grupo))
        filas.append(fila)
        sin_picos = grupo[grupo["family"] != "recorte_picos"]
        fila2 = {"score_type": score_type, "familias": "sin_recorte_picos"}
        fila2.update(_resumen(sin_picos))
        filas.append(fila2)
    tabla_con_sin_picos = _guardar_tabla(pd.DataFrame(filas), "02_global_con_sin_recorte_picos")
    print("\nGlobal con vs sin recorte_picos:")
    print(tabla_con_sin_picos.to_string(index=False))

    filas = []
    for (score_type, objetivo), grupo in resultados_todo.groupby(["score_type", "objetivo_nombre"]):
        sin_picos = grupo[grupo["family"] != "recorte_picos"]
        fila = {"score_type": score_type, "objetivo_nombre": objetivo}
        fila.update(_resumen(sin_picos))
        u = umbrales_tabla[(umbrales_tabla["score_type"] == score_type) & (umbrales_tabla["objetivo_nombre"] == objetivo)].iloc[0]
        fila["false_alarm_events_per_day"] = u["tasa_eventos_dia_real"]
        filas.append(fila)
    tabla_objetivo = _guardar_tabla(pd.DataFrame(filas), "02_por_objetivo_fa")

    print("\nComparacion pareada (cada score vs 'mean')...")
    pareada = comparacion_pareada_vs_baseline(resultados_todo)
    _guardar_tabla(pareada, "03_comparacion_pareada")
    print(pareada.to_string(index=False))

    print("\nEstabilidad de scores limpios...")
    estabilidad = analizar_estabilidad(scores_limpios, grid, umbrales_tabla)
    _guardar_tabla(estabilidad, "04_estabilidad_scores_limpios")
    print(estabilidad.to_string(index=False))

    print("\nGenerando figuras...")
    generar_figuras(tabla_score_duracion, tabla_score_familia, tabla_objetivo, tablas_por_score, scores_limpios,
                     grid, pareada, ataques, particion_val_raw, particion_val.index[0], resultados_todo)

    print(f"\nTablas en {EXP_TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {EXP_FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "config": config, "datos": datos, "modelo": modelo, "ataques": ataques, "master_csv": master_csv,
        "tablas_por_score": tablas_por_score, "scores_limpios": scores_limpios, "umbrales_tabla": umbrales_tabla,
        "resultados_todo": resultados_todo, "tabla_por_score": tabla_por_score,
        "tabla_score_duracion": tabla_score_duracion, "tabla_score_familia": tabla_score_familia,
        "tabla_con_sin_picos": tabla_con_sin_picos, "pareada": pareada, "estabilidad": estabilidad,
        "n_dias": n_dias, "grid": grid,
    }


# ----------------------------------------------------------------------------------------
# 6) Comparacion pareada vs baseline
# ----------------------------------------------------------------------------------------

def comparacion_pareada_vs_baseline(resultados_todo: pd.DataFrame) -> pd.DataFrame:
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    pivote = principal.pivot(index="episode_id", columns="score_type", values="induced_detected")
    for c in pivote.columns:
        pivote[c] = pivote[c].astype(bool)
    meta = principal.drop_duplicates("episode_id").set_index("episode_id")[["family", "duration_min"]]
    pivote = pivote.join(meta)

    subconjuntos = {
        "todos": pivote,
        "30min": pivote[pivote["duration_min"] == 30],
        "60min": pivote[pivote["duration_min"] == 60],
        "120min": pivote[pivote["duration_min"] == 120],
        "240min_o_mas": pivote[pivote["duration_min"] >= 240],
        "sin_recorte_picos": pivote[pivote["family"] != "recorte_picos"],
    }
    for fam in pivote["family"].unique():
        subconjuntos[f"familia_{fam}"] = pivote[pivote["family"] == fam]

    filas = []
    for score_type in NOMBRES_SCORE:
        if score_type == BASELINE:
            continue
        for nombre_sub, sub in subconjuntos.items():
            det_score, det_base = sub[score_type].to_numpy(), sub[BASELINE].to_numpy()
            solo_score = int((det_score & ~det_base).sum())
            solo_base = int((~det_score & det_base).sum())
            ambos = int((det_score & det_base).sum())
            ninguno = int((~det_score & ~det_base).sum())
            p = mcnemar_p(solo_score, solo_base)
            ci_low, ci_high = bootstrap_pareado(det_score.astype(float), det_base.astype(float))
            filas.append({
                "score_type": score_type, "subconjunto": nombre_sub, "n_episodios": len(sub),
                "detectado_solo_con_score": solo_score, "detectado_solo_con_mean": solo_base,
                "detectado_con_ambos": ambos, "detectado_con_ninguno": ninguno,
                "DR_score_pct": 100 * det_score.mean() if len(sub) else float("nan"),
                "DR_mean_pct": 100 * det_base.mean() if len(sub) else float("nan"),
                "mcnemar_p_valor": p,
                "bootstrap_ci95_diferencia_(score-mean)_low": ci_low, "bootstrap_ci95_diferencia_(score-mean)_high": ci_high,
                "diferencia_significativa_p<0.05": p < 0.05,
            })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------
# 7) Estabilidad de scores limpios
# ----------------------------------------------------------------------------------------

def _skewness(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean((x - m) ** 3) / s ** 3)


def analizar_estabilidad(scores_limpios: dict[str, np.ndarray], grid: dict, umbrales_tabla: pd.DataFrame) -> pd.DataFrame:
    horas = pd.to_datetime(grid["inicio"]).hour
    consumo_medio = grid["X"].mean(axis=1)
    terciles = pd.qcut(consumo_medio, 3, labels=["bajo", "medio", "alto"])

    filas = []
    for nombre in NOMBRES_SCORE:
        s = scores_limpios[nombre]
        std_hora = pd.Series(s).groupby(horas).std()
        std_tercil = pd.Series(s).groupby(terciles, observed=True).std()

        u_principal = umbrales_tabla[(umbrales_tabla["score_type"] == nombre) & (umbrales_tabla["objetivo_nombre"] == OBJETIVO_PRINCIPAL)].iloc[0]
        es_alarma = s > u_principal["threshold"]
        duraciones = duraciones_eventos_min(es_alarma, TAMANO_VENTANA)

        filas.append({
            "score_type": nombre, "media": s.mean(), "std": s.std(), "mediana": np.median(s),
            "p90": np.percentile(s, 90), "p95": np.percentile(s, 95), "p99": np.percentile(s, 99),
            "p99_9": np.percentile(s, 99.9), "asimetria": _skewness(s),
            "std_por_hora_media": float(std_hora.mean()), "std_por_hora_max": float(std_hora.max()),
            "std_tercil_bajo": float(std_tercil.get("bajo", float("nan"))),
            "std_tercil_medio": float(std_tercil.get("medio", float("nan"))),
            "std_tercil_alto": float(std_tercil.get("alto", float("nan"))),
            "duracion_media_racha_alarma_min": float(duraciones.mean()),
            "duracion_max_racha_alarma_min": float(duraciones.max()),
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------
# 8) Figuras
# ----------------------------------------------------------------------------------------

def generar_figuras(tabla_score_duracion, tabla_score_familia, tabla_objetivo, tablas_por_score, scores_limpios,
                     grid, pareada, ataques, particion_val_raw, inicio_particion, resultados_todo):
    scores_principales = ["mean", "relative", "top1", "top5", "top10", "hybrid_0.25"]

    # 1) induced DR por duracion y score
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in scores_principales:
        sub = tabla_score_duracion[tabla_score_duracion["score_type"] == s].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", markersize=4, linewidth=1.8,
                color=COLOR_SCORE[s], label=s)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)")
    ax.set_title("induced_DR por duracion y score (0.06 eventos/dia)")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout(); _guardar_fig(fig, "01_induced_dr_por_duracion_score")

    # 2) induced DR por familia y score
    familias = sorted(tabla_score_familia["family"].unique())
    x = np.arange(len(familias)); ancho = 0.8 / len(scores_principales)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, s in enumerate(scores_principales):
        sub = tabla_score_familia[tabla_score_familia["score_type"] == s].set_index("family").reindex(familias)
        ax.bar(x + (i - len(scores_principales) / 2) * ancho, sub["induced_DR_pct"], width=ancho, color=COLOR_SCORE[s], label=s)
    ax.set_xticks(x); ax.set_xticklabels(familias, rotation=15, ha="right")
    ax.set_ylabel("induced_DR (%)")
    ax.set_title("induced_DR por familia y score")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout(); _guardar_fig(fig, "02_induced_dr_por_familia_score")

    # 3) curva DR vs falsos eventos/dia
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for s in scores_principales:
        sub = tabla_objetivo[tabla_objetivo["score_type"] == s].sort_values("false_alarm_events_per_day")
        ax.plot(sub["false_alarm_events_per_day"], sub["induced_DR_pct"], marker="o", markersize=5, linewidth=1.8,
                color=COLOR_SCORE[s], label=s)
    ax.axvline(0.06, color=GRIS, linestyle="--", linewidth=1)
    ax.set_xlabel("falsos eventos / dia"); ax.set_ylabel("induced_DR (%) -- sin recorte_picos")
    ax.set_title("induced_DR vs falsos eventos/dia, por score")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "03_induced_dr_vs_falsos_eventos_dia")

    # 4) score_delta por duracion y score
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in scores_principales:
        sub = tabla_score_duracion[tabla_score_duracion["score_type"] == s].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["score_delta_mediano"], marker="o", markersize=4, linewidth=1.8,
                color=COLOR_SCORE[s], label=s)
    ax.axhline(0, color=GRIS, linestyle="--", linewidth=1)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("score_delta mediano")
    ax.set_title("score_delta mediano por duracion y score")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout(); _guardar_fig(fig, "04_score_delta_por_duracion_score")

    # 5) distribucion de scores limpios (normalizados a [0,1] de su propio rango para comparar formas)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in scores_principales:
        v = scores_limpios[s]
        ax.hist(v, bins=60, color=COLOR_SCORE[s], alpha=0.35, density=True, label=s, histtype="step", linewidth=1.6)
    ax.set_xlabel("score limpio"); ax.set_ylabel("densidad")
    ax.set_title("Distribucion de scores limpios (val)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "05_distribucion_scores_limpios")

    # 6) matriz de mejora vs mean, por familia y duracion (para 'top5' como representante)
    for s_comparar in ("top5", "relative"):
        # reconstruir induced_DR por familia x duracion para mean y s_comparar
        principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
        filas = []
        for st in ("mean", s_comparar):
            for (fam, dur), grupo in principal[principal["score_type"] == st].groupby(["family", "duration_min"]):
                n = len(grupo)
                filas.append({"score_type": st, "family": fam, "duration_min": dur,
                               "induced_DR_pct": 100 * grupo["induced_detected"].sum() / n if n else float("nan")})
        tabla_fd = pd.DataFrame(filas)
        pv_mean = tabla_fd[tabla_fd["score_type"] == "mean"].pivot(index="family", columns="duration_min", values="induced_DR_pct")
        pv_score = tabla_fd[tabla_fd["score_type"] == s_comparar].pivot(index="family", columns="duration_min", values="induced_DR_pct")
        mejora = (pv_score - pv_mean).reindex(columns=DURACIONES)
        datos_m = mejora.values.astype(float)
        limite = np.nanmax(np.abs(datos_m)) if np.isfinite(datos_m).any() else 1
        fig, ax = plt.subplots(figsize=(8.5, 5))
        im = ax.imshow(datos_m, cmap="RdBu_r", aspect="auto", vmin=-limite, vmax=limite)
        ax.set_xticks(range(len(mejora.columns))); ax.set_xticklabels(mejora.columns)
        ax.set_yticks(range(len(mejora.index))); ax.set_yticklabels(mejora.index, fontsize=8)
        ax.set_xlabel("duracion (min)")
        for i in range(datos_m.shape[0]):
            for j in range(datos_m.shape[1]):
                v = datos_m[i, j]
                if np.isnan(v):
                    continue
                color = "white" if abs(v) > 0.55 * limite else "#333"
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7.5, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("mejora induced_DR (puntos %)")
        ax.set_title(f"Mejora de '{s_comparar}' vs 'mean' (induced_DR, puntos %)")
        fig.tight_layout(); _guardar_fig(fig, f"06_matriz_mejora_{s_comparar}_vs_mean")

    # 7) detecciones ganadas/perdidas vs mean, todos los scores, subconjunto 'todos'
    sub = pareada[pareada["subconjunto"] == "todos"].sort_values("score_type")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["detectado_solo_con_score"], width=0.4, color=VERDE, label="ganadas (solo score)")
    ax.bar(x + 0.2, -sub["detectado_solo_con_mean"], width=0.4, color=ROJO, label="perdidas (solo mean)")
    ax.axhline(0, color=GRIS, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(sub["score_type"], rotation=30, ha="right")
    ax.set_ylabel("nº de episodios")
    ax.set_title("Episodios ganados/perdidos frente a 'mean' (todos los episodios)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "07_detecciones_ganadas_perdidas_vs_mean")

    # 8) estabilidad por hora del dia
    horas = pd.to_datetime(grid["inicio"]).hour
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in scores_principales:
        std_hora = pd.Series(scores_limpios[s]).groupby(horas).std()
        ax.plot(std_hora.index, std_hora.values, marker="o", markersize=3, linewidth=1.5, color=COLOR_SCORE[s], label=s)
    ax.set_xlabel("hora del dia"); ax.set_ylabel("std del score limpio")
    ax.set_title("Estabilidad del score limpio por hora del dia")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout(); _guardar_fig(fig, "08_estabilidad_por_hora")

    # 9) recorte_picos por score
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    picos = principal[principal["family"] == "recorte_picos"]
    filas = []
    for st, grupo in picos.groupby("score_type"):
        n = len(grupo)
        filas.append({"score_type": st, "induced_DR_pct": 100 * grupo["induced_detected"].sum() / n if n else float("nan")})
    tabla_picos = pd.DataFrame(filas).set_index("score_type").reindex(NOMBRES_SCORE).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(tabla_picos["score_type"], tabla_picos["induced_DR_pct"], color=[COLOR_SCORE[s] for s in tabla_picos["score_type"]])
    ax.set_xticklabels(tabla_picos["score_type"], rotation=30, ha="right")
    ax.set_ylabel("induced_DR (%)")
    ax.set_title("recorte_picos: induced_DR por score")
    fig.tight_layout(); _guardar_fig(fig, "09_recorte_picos_por_score")
    _guardar_tabla(tabla_picos, "09_recorte_picos_por_score")

    # 10) ejemplos de ataques cortos detectados por top5 pero no por mean
    principal_top5 = principal[principal["score_type"] == "top5"].set_index("episode_id")["induced_detected"]
    principal_mean = principal[principal["score_type"] == "mean"].set_index("episode_id")["induced_detected"]
    cortos = set(principal[(principal["score_type"] == "mean") & (principal["duration_min"] <= 120)]["episode_id"])
    ids_ganados = [
        ep for ep in cortos
        if bool(principal_top5.get(ep, False)) and not bool(principal_mean.get(ep, False))
    ][:3]
    if ids_ganados:
        fig, axes = plt.subplots(len(ids_ganados), 1, figsize=(9, 3 * len(ids_ganados)))
        if len(ids_ganados) == 1:
            axes = [axes]
        margen = 45
        for ax, ep_id in zip(axes, ids_ganados):
            a = ataques[ep_id]
            ini = max(0, a["offset_global"] - margen)
            fin = a["offset_global"] + a["duracion"] + margen
            x_limpio = particion_val_raw[ini:fin]
            x_atacado = x_limpio.copy()
            x_atacado[a["offset_global"] - ini: a["offset_global"] - ini + a["duracion"]] = a["y_tramo"]
            t = [inicio_particion + pd.Timedelta(minutes=m) for m in range(ini, fin)]
            ax.plot(t, x_limpio, color=GRIS, linewidth=1.1, label="limpio")
            ax.plot(t, x_atacado, color=ROJO, linewidth=1.1, alpha=0.8, label="reportado")
            ax.axvspan(inicio_particion + pd.Timedelta(minutes=a["offset_global"]),
                       inicio_particion + pd.Timedelta(minutes=a["offset_global"] + a["duracion"]),
                       color=VERDE, alpha=0.08)
            ax.set_title(f"{ep_id} | {a['family']} ({a['parameter']}) | {a['duracion']} min -- top5 SI, mean NO")
            ax.legend(fontsize=7)
        fig.tight_layout(); _guardar_fig(fig, "10_ejemplos_top5_no_mean")
    else:
        print("[aviso] no se encontraron episodios cortos ganados por top5 y no por mean para la figura 10")


if __name__ == "__main__":
    main()
