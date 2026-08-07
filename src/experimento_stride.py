"""Experimento 2: ¿reducir el stride de evaluacion mejora la deteccion de ataques cortos y
moderados sin aumentar las falsas alarmas?

Cambia unicamente la frecuencia de generacion de ventanas sobre val (ventana=360min,
checkpoint y score `deficit` fijos, sin reentrenar). Los episodios son exactamente los de
results/tables/episodes_master_val.csv: no se sortea ninguna posicion nueva, se reconstruye
el tramo atacado exacto reproduciendo la misma llamada determinista a
`episodes._generar_episodios` (mismo seed, misma secuencia de RNG) y se valida linea a
linea contra ese CSV.

Cada stride tiene su propio umbral, calibrado sobre los scores limpios de su rejilla para
acercarse a una tasa de falsos eventos/dia objetivo (0.06 principal + 0.05/0.10/0.25
secundarios), en vez de reutilizar el percentil fijo del pipeline original. Un "evento" es
una racha de ventanas consecutivas que alarman sin ninguna ventana sana entre medias; con
stride<360 esto ya cubre el solape temporal directo entre ventanas vecinas.

Ejecutable de forma aislada:
    python -m src.experimento_stride
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.anomaly_score import calcular_scores
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import _generar_episodios, wilson_ci
from src.normalization import aplicar_zscore
from src.pipeline import obtener_modelo, preparar_datos
from src.windowing import cargar_config, ventanear

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables"
EXP_TABLAS_DIR = TABLAS_DIR / "experimento_stride"
EXP_FIGURAS_DIR = BASE_DIR / "results" / "figures" / "experimento_stride"

AZUL = "#2a78d6"
AZUL_CLARO = "#a9c8ec"
AZUL_OSCURO = "#154a86"
NARANJA = "#eb6834"
VERDE = "#2f9e44"
GRIS = "#8a8a86"
ROJO = "#c0392b"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDES = [360, 60, 30]
COLOR_STRIDE = {360: AZUL_CLARO, 60: AZUL, 30: AZUL_OSCURO}
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
# 0) Datos base (checkpoint ventana=360 existente, config sin cambios)
# ----------------------------------------------------------------------------------------

def cargar_datos_y_modelo():
    config = cargar_config()
    datos = preparar_datos(config)                       # ventana=360, stride config estandar (360/360 val/test)
    modelo = obtener_modelo(config, datos["ventanas_norm"])  # checkpoint EXISTENTE, no reentrena
    return config, datos, modelo


def reconstruir_ataques_exactos(config: dict, datos: dict) -> tuple[dict, pd.DataFrame]:
    """Reproduce (misma llamada determinista que genero episodes_master_val.csv) el tramo
    atacado EXACTO de cada episodio, y lo valida linea a linea contra ese CSV ya guardado."""
    seed = config["seed"]
    episodios = _generar_episodios(config, datos, "val", seed, modos=("continuous_timeline",), n_posiciones_override=50)

    particion = datos["particiones"]["val"]
    inicio_particion = particion.index[0]

    ataques = {}
    for ep in episodios:
        offset_global = int((ep["attack_start"] - inicio_particion) / pd.Timedelta(minutes=1))
        ks = sorted(ep["_ventanas_idx"])
        segmento = np.concatenate([ep["_ventanas_atacadas"][k] for k in ks])
        rel = offset_global - ks[0] * TAMANO_VENTANA
        y_tramo = segmento[rel:rel + ep["duration_min"]]
        ataques[ep["episode_id"]] = {
            "offset_global": offset_global, "duracion": ep["duration_min"], "y_tramo": y_tramo,
            "family": ep["family"], "parameter": ep["parameter"],
            "attack_start": ep["attack_start"], "attack_end": ep["attack_end"],
        }

    master_csv = pd.read_csv(TABLAS_DIR / "episodes_master_val.csv", parse_dates=["attack_start", "attack_end"])
    assert set(ataques.keys()) == set(master_csv["episode_id"]), \
        "los episodios reconstruidos no coinciden con episodes_master_val.csv"
    for fila in master_csv.itertuples():
        a = ataques[fila.episode_id]
        assert a["attack_start"] == fila.attack_start
        assert a["attack_end"] == fila.attack_end
        assert a["family"] == fila.family
        assert a["parameter"] == fila.parameter
        assert a["duracion"] == fila.duration_min

    return ataques, master_csv


# ----------------------------------------------------------------------------------------
# 1) Rejillas de evaluacion (windowing.ventanear reutilizado tal cual, sin tocarlo)
# ----------------------------------------------------------------------------------------

def construir_rejilla(particion_val: pd.DataFrame, stride: int) -> dict:
    return ventanear(particion_val, TAMANO_VENTANA, stride)


def _ventanas_solapadas(offset_global: int, duracion: int, stride: int, n_ventanas_grid: int) -> list[int]:
    a_end = offset_global + duracion
    k_min = max(0, (offset_global - TAMANO_VENTANA) // stride)
    k_max = min(n_ventanas_grid - 1, a_end // stride)
    resultado = []
    for k in range(int(k_min), int(k_max) + 1):
        w_start = k * stride
        w_end = w_start + TAMANO_VENTANA
        if w_start < a_end and w_end > offset_global:
            resultado.append(k)
    return resultado


def _ventana_atacada(valores_val: np.ndarray, w_start: int, offset_global: int, duracion: int, y_tramo: np.ndarray) -> np.ndarray:
    ventana = valores_val[w_start:w_start + TAMANO_VENTANA].copy()
    a_end = offset_global + duracion
    w_end = w_start + TAMANO_VENTANA
    solape_ini = max(offset_global, w_start)
    solape_fin = min(a_end, w_end)
    if solape_ini < solape_fin:
        ventana[solape_ini - w_start:solape_fin - w_start] = y_tramo[solape_ini - offset_global:solape_fin - offset_global]
    return ventana


# ----------------------------------------------------------------------------------------
# 2) Falsos eventos/dia y calibracion de umbral por stride
# ----------------------------------------------------------------------------------------

def contar_eventos(es_alarma: np.ndarray) -> int:
    """Cuenta rachas de True consecutivas en la rejilla ordenada por tiempo: dos ventanas
    alarmantes son el mismo evento si son adyacentes en la rejilla (con stride<360 eso ya
    implica solape temporal) o si todas las ventanas intermedias tambien alarman."""
    if len(es_alarma) == 0:
        return 0
    cambios = np.diff(es_alarma.astype(int))
    return int((cambios == 1).sum()) + (1 if es_alarma[0] else 0)


def calibrar_umbral_por_eventos(scores_limpios: np.ndarray, n_dias: float, objetivo: float,
                                 n_candidatos: int = 4000) -> tuple[float, float, float]:
    """Barre percentiles finos de scores_limpios y se queda con el umbral cuya tasa de
    eventos/dia resultante esta mas cerca del objetivo. Devuelve (umbral, eventos/dia real,
    ventanas_falsas/dia real)."""
    percentiles = np.linspace(70, 100, n_candidatos)
    candidatos = np.unique(np.percentile(scores_limpios, percentiles))
    mejor_umbral, mejor_dif, mejor_tasa = float(candidatos[-1]), np.inf, 0.0
    for u in candidatos:
        es_alarma = scores_limpios > u
        tasa = contar_eventos(es_alarma) / n_dias
        dif = abs(tasa - objetivo)
        if dif < mejor_dif:
            mejor_umbral, mejor_dif, mejor_tasa = float(u), dif, tasa
    tasa_ventanas = float((scores_limpios > mejor_umbral).sum()) / n_dias
    return mejor_umbral, mejor_tasa, tasa_ventanas


# ----------------------------------------------------------------------------------------
# 3) Puntuacion (una vez por stride, reutilizada para los 4 objetivos de FA)
# ----------------------------------------------------------------------------------------

def puntuar_stride(ataques: dict, particion_val_raw: np.ndarray, grid: dict, modelo, params_norm: dict,
                    tipo_score: str, stride: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Devuelve (tabla ventana-nivel SIN umbral aplicado, scores limpios de toda la rejilla).
    Una unica llamada batched al modelo para TODAS las ventanas atacadas de este stride."""
    n_ventanas_grid = grid["X"].shape[0]
    scores_limpios_grid = calcular_scores(modelo, aplicar_zscore(grid["X"], params_norm), tipo_score)

    filas_meta = []
    ventanas_atacadas = []
    for episode_id, ataque in ataques.items():
        ks = _ventanas_solapadas(ataque["offset_global"], ataque["duracion"], stride, n_ventanas_grid)
        assert len(ks) >= 1, f"episodio {episode_id} sin ninguna ventana afectada en stride={stride}"
        for k in ks:
            w_start = k * stride
            ventana_atacada = _ventana_atacada(particion_val_raw, w_start, ataque["offset_global"],
                                                ataque["duracion"], ataque["y_tramo"])
            assert len(ventana_atacada) == TAMANO_VENTANA
            ventanas_atacadas.append(ventana_atacada)
            filas_meta.append({
                "episode_id": episode_id, "k": k,
                "window_start": grid["inicio"][k], "window_end": grid["fin"][k],
                "clean_score": float(scores_limpios_grid[k]),
            })

    X_atacadas = np.stack(ventanas_atacadas).astype(np.float32)
    X_norm = aplicar_zscore(X_atacadas, params_norm)
    attacked_scores = calcular_scores(modelo, X_norm, tipo_score)

    tabla = pd.DataFrame(filas_meta)
    tabla["attacked_score"] = attacked_scores
    return tabla, scores_limpios_grid


def agregar_por_umbral(tabla_ventanas_stride: pd.DataFrame, master_csv: pd.DataFrame, umbral: float,
                        stride: int) -> pd.DataFrame:
    """A partir de la tabla ventana-nivel (clean_score/attacked_score, sin umbral) de UN
    stride ya calculada, aplica un umbral concreto y agrega a nivel de episodio. No vuelve a
    tocar el modelo."""
    t = tabla_ventanas_stride.copy()
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
    resultado["raw_detection_delay_min"] = (
        (resultado["first_raw_alarm_timestamp"] - resultado["attack_start"]) / pd.Timedelta(minutes=1)
    )
    resultado["induced_detection_delay_min"] = (
        (resultado["first_induced_alarm_timestamp"] - resultado["attack_start"]) / pd.Timedelta(minutes=1)
    )
    durante = (resultado["first_induced_alarm_timestamp"] <= resultado["attack_end"]).astype(object)
    despues = (resultado["first_induced_alarm_timestamp"] > resultado["attack_end"]).astype(object)
    sin_induced = ~resultado["induced_detected"]
    durante[sin_induced] = np.nan
    despues[sin_induced] = np.nan
    resultado["induced_detected_during_attack"] = durante
    resultado["induced_detected_after_attack"] = despues
    resultado.loc[sin_induced, "induced_detection_delay_min"] = np.nan
    resultado.loc[~resultado["raw_detected"], "raw_detection_delay_min"] = np.nan

    resultado["stride_min"] = stride
    resultado["threshold"] = umbral
    return resultado[[
        "episode_id", "stride_min", "threshold", "family", "parameter", "duration_min",
        "n_affected_windows", "raw_detected", "induced_detected",
        "first_raw_alarm_timestamp", "first_induced_alarm_timestamp",
        "raw_detection_delay_min", "induced_detection_delay_min",
        "induced_detected_during_attack", "induced_detected_after_attack",
        "max_attacked_score", "max_clean_score", "max_score_delta",
    ]]


# ----------------------------------------------------------------------------------------
# 4) Agregados (seccion 7 del encargo)
# ----------------------------------------------------------------------------------------

def _resumen(grupo: pd.DataFrame) -> dict:
    n = len(grupo)
    n_induced = int(grupo["induced_detected"].sum())
    ci = wilson_ci(n_induced, n)
    con_induced = grupo[grupo["induced_detected"] == True]  # noqa: E712
    return {
        "n_episodios": n,
        "induced_DR_pct": 100 * n_induced / n if n else float("nan"),
        "induced_DR_ci95_low": ci[0], "induced_DR_ci95_high": ci[1],
        "retraso_mediano_induced_min": con_induced["induced_detection_delay_min"].median() if len(con_induced) else float("nan"),
        "pct_detectado_durante_ataque": 100 * (con_induced["induced_detected_during_attack"] == True).mean() if len(con_induced) else float("nan"),  # noqa: E712
        "ventanas_afectadas_media": grupo["n_affected_windows"].mean(),
    }


def construir_agregados(resultados_todo: pd.DataFrame, umbrales_tabla: pd.DataFrame) -> dict:
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL].copy()
    tablas = {}

    filas = []
    for stride, grupo in principal.groupby("stride_min"):
        fila = {"stride_min": stride}
        fila.update(_resumen(grupo))
        u = umbrales_tabla[(umbrales_tabla["stride_min"] == stride) & (umbrales_tabla["objetivo_nombre"] == OBJETIVO_PRINCIPAL)].iloc[0]
        fila["false_alarm_events_per_day"] = u["tasa_eventos_dia_real"]
        fila["false_alarm_windows_per_day"] = u["tasa_ventanas_dia_real"]
        filas.append(fila)
    tablas["por_stride"] = pd.DataFrame(filas)

    filas = []
    for (stride, dur), grupo in principal.groupby(["stride_min", "duration_min"]):
        fila = {"stride_min": stride, "duration_min": dur}
        fila.update(_resumen(grupo))
        filas.append(fila)
    tablas["por_stride_duracion"] = pd.DataFrame(filas)

    filas = []
    for (stride, fam), grupo in principal.groupby(["stride_min", "family"]):
        fila = {"stride_min": stride, "family": fam}
        fila.update(_resumen(grupo))
        filas.append(fila)
    tablas["por_stride_familia"] = pd.DataFrame(filas)

    filas = []
    for (stride, fam, param), grupo in principal.groupby(["stride_min", "family", "parameter"]):
        fila = {"stride_min": stride, "family": fam, "parameter": param}
        fila.update(_resumen(grupo))
        filas.append(fila)
    tablas["por_stride_familia_parametro"] = pd.DataFrame(filas)

    # global con y sin recorte_picos
    filas = []
    for stride, grupo in principal.groupby("stride_min"):
        fila = {"stride_min": stride, "familias": "todas"}
        fila.update(_resumen(grupo))
        filas.append(fila)
        sin_picos = grupo[grupo["family"] != "recorte_picos"]
        fila2 = {"stride_min": stride, "familias": "sin_recorte_picos"}
        fila2.update(_resumen(sin_picos))
        filas.append(fila2)
    tablas["global_con_sin_recorte_picos"] = pd.DataFrame(filas)

    # por objetivo de FA (todas las combinaciones, agregado global sin recorte_picos)
    filas = []
    for (stride, objetivo), grupo in resultados_todo.groupby(["stride_min", "objetivo_nombre"]):
        sin_picos = grupo[grupo["family"] != "recorte_picos"]
        fila = {"stride_min": stride, "objetivo_nombre": objetivo}
        fila.update(_resumen(sin_picos))
        u = umbrales_tabla[(umbrales_tabla["stride_min"] == stride) & (umbrales_tabla["objetivo_nombre"] == objetivo)].iloc[0]
        fila["false_alarm_events_per_day"] = u["tasa_eventos_dia_real"]
        fila["false_alarm_windows_per_day"] = u["tasa_ventanas_dia_real"]
        filas.append(fila)
    tablas["por_objetivo_fa"] = pd.DataFrame(filas)

    return tablas


# ----------------------------------------------------------------------------------------
# 5) Comparacion pareada (seccion 9)
# ----------------------------------------------------------------------------------------

def mcnemar_p(b: int, c: int) -> float:
    """McNemar con correccion de continuidad, p-valor via aproximacion normal (chi2, 1 gl).
    Sin dependencia de scipy: sqrt(chi2_1gl) ~ |Z| normal estandar."""
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    z = math.sqrt(stat)
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def bootstrap_pareado(det_a: np.ndarray, det_b: np.ndarray, n_boot: int = 3000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(det_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = det_a[idx].mean() - det_b[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def comparacion_pareada(resultados_todo: pd.DataFrame) -> pd.DataFrame:
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    pivote = principal.pivot(index="episode_id", columns="stride_min", values="induced_detected").astype(bool)

    filas = []
    for a, b in [(360, 60), (60, 30), (360, 30)]:
        det_a, det_b = pivote[a].to_numpy(), pivote[b].to_numpy()
        solo_a = int((det_a & ~det_b).sum())
        solo_b = int((~det_a & det_b).sum())
        ambos = int((det_a & det_b).sum())
        ninguno = int((~det_a & ~det_b).sum())
        p = mcnemar_p(solo_a, solo_b)
        ci_low, ci_high = bootstrap_pareado(det_a.astype(float), det_b.astype(float))
        filas.append({
            "stride_a": a, "stride_b": b, "n_episodios": len(pivote),
            f"detectado_solo_con_{a}": solo_a, f"detectado_solo_con_{b}": solo_b,
            "detectado_con_ambos": ambos, "detectado_con_ninguno": ninguno,
            "DR_a_pct": 100 * det_a.mean(), "DR_b_pct": 100 * det_b.mean(),
            "mcnemar_p_valor": p, "bootstrap_ci95_diferencia_low": ci_low, "bootstrap_ci95_diferencia_high": ci_high,
            "diferencia_significativa_p<0.05": p < 0.05,
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------
# 6) Figuras
# ----------------------------------------------------------------------------------------

def fig_dr_por_duracion_stride(tabla_std: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for stride in STRIDES:
        sub = tabla_std[tabla_std["stride_min"] == stride].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", markersize=4, linewidth=1.8,
                color=COLOR_STRIDE[stride], label=f"stride={stride} min")
    ax.set_xlabel("duracion del ataque (min)"); ax.set_ylabel("induced_DR (%)")
    ax.set_title("induced_DR por duracion, para cada stride (umbral: 0.06 eventos/dia)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_dr_por_familia_stride(tabla_sf: pd.DataFrame, nombre: str) -> None:
    familias = sorted(tabla_sf["family"].unique())
    x = np.arange(len(familias)); ancho = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, stride in enumerate(STRIDES):
        sub = tabla_sf[tabla_sf["stride_min"] == stride].set_index("family").reindex(familias)
        ax.bar(x + (i - 1) * ancho, sub["induced_DR_pct"], width=ancho, color=COLOR_STRIDE[stride], label=f"stride={stride}")
    ax.set_xticks(x); ax.set_xticklabels(familias, rotation=15, ha="right")
    ax.set_ylabel("induced_DR (%)")
    ax.set_title("induced_DR por familia, para cada stride")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_retraso_por_duracion_stride(tabla_std: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for stride in STRIDES:
        sub = tabla_std[tabla_std["stride_min"] == stride].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["retraso_mediano_induced_min"], marker="o", markersize=4, linewidth=1.8,
                color=COLOR_STRIDE[stride], label=f"stride={stride} min")
    ax.set_xlabel("duracion del ataque (min)"); ax.set_ylabel("retraso mediano (min)")
    ax.set_title("Retraso mediano de deteccion inducida, por duracion y stride")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_dr_vs_fa(tabla_objetivo: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for stride in STRIDES:
        sub = tabla_objetivo[tabla_objetivo["stride_min"] == stride].sort_values("false_alarm_events_per_day")
        ax.plot(sub["false_alarm_events_per_day"], sub["induced_DR_pct"], marker="o", markersize=5, linewidth=1.8,
                color=COLOR_STRIDE[stride], label=f"stride={stride} min")
    ax.axvline(0.06, color=GRIS, linestyle="--", linewidth=1, label="objetivo principal (0.06/dia)")
    ax.set_xlabel("falsos eventos / dia (real, tras calibrar)"); ax.set_ylabel("induced_DR (%) -- sin recorte_picos")
    ax.set_title("induced_DR vs tasa de falsos eventos/dia")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_ventanas_vs_eventos(umbrales_tabla: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for stride in STRIDES:
        sub = umbrales_tabla[umbrales_tabla["stride_min"] == stride].sort_values("tasa_eventos_dia_real")
        ax.plot(sub["tasa_eventos_dia_real"], sub["tasa_ventanas_dia_real"], marker="o", markersize=6, linewidth=1.6,
                color=COLOR_STRIDE[stride], label=f"stride={stride} min")
    ax.set_xlabel("false_alarm_events_per_day"); ax.set_ylabel("false_alarm_windows_per_day")
    ax.set_title("Ventanas falsas/dia vs eventos falsos/dia (agrupacion de rachas)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_ventanas_afectadas(tabla_std: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for stride in STRIDES:
        sub = tabla_std[tabla_std["stride_min"] == stride].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["ventanas_afectadas_media"], marker="o", markersize=4, linewidth=1.8,
                color=COLOR_STRIDE[stride], label=f"stride={stride} min")
    ax.set_xlabel("duracion del ataque (min)"); ax.set_ylabel("nº medio de ventanas afectadas")
    ax.set_title("Oportunidades de deteccion (ventanas afectadas) por duracion y stride")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_matriz_mejora(tabla_sfd: pd.DataFrame, stride_a: int, stride_b: int, nombre: str) -> None:
    pivote_a = tabla_sfd[tabla_sfd["stride_min"] == stride_a].pivot(index="family", columns="duration_min", values="induced_DR_pct")
    pivote_b = tabla_sfd[tabla_sfd["stride_min"] == stride_b].pivot(index="family", columns="duration_min", values="induced_DR_pct")
    mejora = (pivote_b - pivote_a).reindex(columns=DURACIONES)

    datos = mejora.values.astype(float)
    limite = np.nanmax(np.abs(datos)) if np.isfinite(datos).any() else 1
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(datos, cmap="RdBu_r", aspect="auto", vmin=-limite, vmax=limite)
    ax.set_xticks(range(len(mejora.columns))); ax.set_xticklabels(mejora.columns)
    ax.set_yticks(range(len(mejora.index))); ax.set_yticklabels(mejora.index, fontsize=8)
    ax.set_xlabel("duracion (min)")
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            v = datos[i, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v) > 0.55 * limite else "#333"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=7.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("mejora induced_DR (puntos %)")
    ax.set_title(f"Mejora de stride {stride_a} -> {stride_b} (induced_DR, puntos porcentuales)")
    fig.tight_layout(); _guardar_fig(fig, nombre)


# ----------------------------------------------------------------------------------------
# Orquestacion
# ----------------------------------------------------------------------------------------

def main() -> dict:
    config, datos, modelo = cargar_datos_y_modelo()
    tipo_score = config["score"]["tipo"]
    params_norm = datos["params_norm"]

    print("Reconstruyendo ataques exactos de episodes_master_val.csv (misma llamada determinista)...")
    ataques, master_csv = reconstruir_ataques_exactos(config, datos)
    print(f"{len(ataques)} episodios validados linea a linea contra el CSV original.\n")

    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    n_dias = len(particion_val_raw) / 1440.0

    filas_umbrales = []
    resultados_por_stride_objetivo = []
    tablas_ventanas_por_stride = {}

    for stride in STRIDES:
        print(f"=== stride={stride} min ===")
        grid = construir_rejilla(particion_val, stride)
        print(f"  rejilla: {grid['X'].shape[0]} ventanas de {grid['X'].shape[1]} muestras "
              f"(gap={stride} min, descartadas={grid['n_muestras_descartadas']})")

        tabla_ventanas, scores_limpios_grid = puntuar_stride(ataques, particion_val_raw, grid, modelo,
                                                               params_norm, tipo_score, stride)
        tablas_ventanas_por_stride[stride] = tabla_ventanas
        print(f"  {len(tabla_ventanas)} filas ventana-episodio puntuadas (clean+attacked)")

        for objetivo_nombre, objetivo_valor in OBJETIVOS_EVENTOS_DIA.items():
            umbral, tasa_eventos, tasa_ventanas = calibrar_umbral_por_eventos(scores_limpios_grid, n_dias, objetivo_valor)
            filas_umbrales.append({
                "stride_min": stride, "objetivo_nombre": objetivo_nombre, "objetivo_valor": objetivo_valor,
                "threshold": umbral, "tasa_eventos_dia_real": tasa_eventos, "tasa_ventanas_dia_real": tasa_ventanas,
            })
            print(f"    objetivo {objetivo_nombre} ({objetivo_valor}/dia) -> umbral={umbral:.5f}, "
                  f"eventos/dia real={tasa_eventos:.4f}, ventanas/dia real={tasa_ventanas:.3f}")

            resultado_ep = agregar_por_umbral(tabla_ventanas, master_csv, umbral, stride)
            resultado_ep["objetivo_nombre"] = objetivo_nombre
            resultado_ep["objetivo_valor"] = objetivo_valor
            resultados_por_stride_objetivo.append(resultado_ep)
        print()

    umbrales_tabla = _guardar_tabla(pd.DataFrame(filas_umbrales), "00_umbrales_por_stride_objetivo")
    resultados_todo = pd.concat(resultados_por_stride_objetivo, ignore_index=True)
    _guardar_tabla(resultados_todo, "01_episode_detection_results_stride")

    print("Construyendo agregados...")
    agregados = construir_agregados(resultados_todo, umbrales_tabla)
    for nombre, tabla in agregados.items():
        _guardar_tabla(tabla, f"02_{nombre}")

    print("\nResumen por stride (objetivo principal 0.06 eventos/dia, TODAS las familias):")
    print(agregados["por_stride"].to_string(index=False))

    print("\nGlobal con vs sin recorte_picos:")
    print(agregados["global_con_sin_recorte_picos"].to_string(index=False))

    print("\nComparacion pareada (McNemar + bootstrap)...")
    pareada = comparacion_pareada(resultados_todo)
    _guardar_tabla(pareada, "03_comparacion_pareada")
    print(pareada.to_string(index=False))

    print("\nGenerando figuras...")
    fig_dr_por_duracion_stride(agregados["por_stride_duracion"], "01_induced_dr_por_duracion_stride")
    fig_dr_por_familia_stride(agregados["por_stride_familia"], "02_induced_dr_por_familia_stride")
    fig_retraso_por_duracion_stride(agregados["por_stride_duracion"], "03_retraso_mediano_por_duracion_stride")
    fig_dr_vs_fa(agregados["por_objetivo_fa"], "04_induced_dr_vs_falsos_eventos_dia")
    fig_ventanas_vs_eventos(umbrales_tabla, "05_ventanas_vs_eventos_falsos_dia")
    fig_ventanas_afectadas(agregados["por_stride_duracion"], "06_ventanas_afectadas_por_duracion_stride")

    # matrices de mejora familia x duracion
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    filas = []
    for (stride, fam, dur), grupo in principal.groupby(["stride_min", "family", "duration_min"]):
        n = len(grupo)
        filas.append({"stride_min": stride, "family": fam, "duration_min": dur,
                       "induced_DR_pct": 100 * grupo["induced_detected"].sum() / n if n else float("nan")})
    tabla_sfd_real = pd.DataFrame(filas)
    _guardar_tabla(tabla_sfd_real, "02_por_stride_familia_duracion")
    fig_matriz_mejora(tabla_sfd_real, 360, 60, "07_matriz_mejora_360_a_60")
    fig_matriz_mejora(tabla_sfd_real, 60, 30, "08_matriz_mejora_60_a_30")

    print(f"\nTablas en {EXP_TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {EXP_FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "ataques": ataques, "master_csv": master_csv, "umbrales_tabla": umbrales_tabla,
        "resultados_todo": resultados_todo, "agregados": agregados, "pareada": pareada,
        "tablas_ventanas_por_stride": tablas_ventanas_por_stride,
        "particion_val_raw_snapshot": particion_val_raw, "datos": datos, "config": config, "modelo": modelo,
        "tipo_score": tipo_score, "n_dias": n_dias,
    }


if __name__ == "__main__":
    main()
