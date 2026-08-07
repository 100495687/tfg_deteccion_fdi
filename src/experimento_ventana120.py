"""Experimento 3: ¿reducir el tamano de ventana de 360 a 120 minutos mejora la deteccion de
ataques cortos y moderados sin aumentar las falsas alarmas?

Compara dos configuraciones, ambas con stride = tamano de ventana (rejilla disjunta, sin
solape -- el experimento 2 ya probo el solape del stride y no ayudo):

  A) ventana=360, checkpoint TCN-AE existente (models/tcn_ae_ventana360.pt), sin reentrenar.
  B) ventana=120, TCN-AE nuevo con la misma arquitectura/hiperparametros/optimizador/loss/
     seed que el de 360 (solo cambia la longitud de secuencia), entrenado aqui.

Reutiliza la maquinaria ya construida en experimento_stride.py (reconstruccion determinista
de episodios, conteo de eventos falsos, calibracion de umbral, agregacion, McNemar,
bootstrap pareado) sin modificarla. Las funciones que dependian de TAMANO_VENTANA=360 fijo
no sirven tal cual para ventana=120, asi que se reimplementan aqui parametrizadas por
`tamano_ventana` explicito (`_ventanas_disjuntas_afectadas` / `_ventana_atacada`).

Igual que en el experimento 2, los episodios son exactamente los de episodes_master_val.csv:
se reconstruyen via replay determinista de `episodes._generar_episodios` y se validan linea
a linea contra ese CSV.

Ejecutable de forma aislada (entrena el modelo de 120 si no existe su checkpoint):
    python -m src.experimento_ventana120
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.anomaly_score import calcular_scores
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import wilson_ci
from src.experimento_stride import (
    agregar_por_umbral,
    bootstrap_pareado,
    calibrar_umbral_por_eventos,
    cargar_datos_y_modelo,
    contar_eventos,
    mcnemar_p,
    reconstruir_ataques_exactos,
)
from src.models.tcn_ae import TCNAEModelo
from src.normalization import aplicar_zscore
from src.pipeline import preparar_datos
from src.windowing import cargar_config

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables"
EXP_TABLAS_DIR = TABLAS_DIR / "experimento_ventana120"
EXP_FIGURAS_DIR = BASE_DIR / "results" / "figures" / "experimento_ventana120"
MODELOS_DIR = BASE_DIR / "models"

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

TAMANOS = [360, 120]
COLOR_TAMANO = {360: AZUL_CLARO, 120: AZUL_OSCURO}
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
# 1) Config y datos de la ventana=120 (sin tocar configs/base.yaml: copia en memoria)
# ----------------------------------------------------------------------------------------

def construir_config_120(config_base: dict) -> dict:
    config = copy.deepcopy(config_base)
    config["windowing"]["tamano_ventana_min"] = 120
    config["windowing"]["stride_eval_min"] = 120  # explicito: stride == ventana en val/test (sin solape)
    config["modelo"]["entrenamiento"]["epochs_max"] = 300
    return config


# ----------------------------------------------------------------------------------------
# 2) Entrenamiento (o carga) del TCN-AE de ventana=120
# ----------------------------------------------------------------------------------------

def entrenar_o_cargar_120(config_120: dict, datos_120: dict) -> tuple[TCNAEModelo, dict]:
    ruta_ckpt = MODELOS_DIR / f"{config_120['modelo']['nombre']}_ventana120.pt"
    ruta_historial = EXP_TABLAS_DIR / "03_entrenamiento_historial_120.csv"
    ruta_meta = EXP_TABLAS_DIR / "03_entrenamiento_meta_120.csv"

    modelo = TCNAEModelo(config_120, seed=config_120["seed"])

    if ruta_ckpt.exists() and ruta_historial.exists() and ruta_meta.exists():
        modelo.net.load_state_dict(torch.load(ruta_ckpt, map_location="cpu"))
        modelo.net.eval()
        historial_df = pd.read_csv(ruta_historial)
        meta = pd.read_csv(ruta_meta).iloc[0].to_dict()
        print(f"Checkpoint de ventana=120 ya existente ({ruta_ckpt.name}) -- reutilizado, NO se reentrena.")
        return modelo, {"historial": historial_df, **meta}

    print("Entrenando TCN-AE de ventana=120 desde cero (no habia checkpoint/historial previos)...")
    t0 = time.time()
    historial = modelo.fit(datos_120["ventanas_norm"]["train"], datos_120["ventanas_norm"]["val"])
    duracion_s = time.time() - t0

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.net.state_dict(), ruta_ckpt)

    EXP_TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    historial_df = pd.DataFrame({
        "epoca": range(1, len(historial["train_loss"]) + 1),
        "train_loss": historial["train_loss"], "val_loss": historial["val_loss"],
    })
    historial_df.to_csv(ruta_historial, index=False)

    mejor_epoca = int(np.argmin(historial["val_loss"])) + 1
    meta = {
        "epocas_entrenadas": historial["epocas_entrenadas"], "mejor_epoca": mejor_epoca,
        "mejor_val_loss": historial["mejor_val_loss"], "duracion_entrenamiento_s": duracion_s,
        "checkpoint": str(ruta_ckpt),
    }
    pd.DataFrame([meta]).to_csv(ruta_meta, index=False)
    print(f"Entrenamiento terminado: {historial['epocas_entrenadas']} epocas, mejor epoca={mejor_epoca}, "
          f"mejor_val_loss={historial['mejor_val_loss']:.5f}, tiempo={duracion_s:.1f}s")

    return modelo, {"historial": historial_df, **meta}


# ----------------------------------------------------------------------------------------
# 3) Documentacion de arquitectura
# ----------------------------------------------------------------------------------------

def documentar_arquitectura(modelo_360: TCNAEModelo, modelo_120: TCNAEModelo, config_base: dict) -> pd.DataFrame:
    arch = config_base["modelo"]["tcn_ae"]
    filas = []
    for ventana, modelo in ((360, modelo_360), (120, modelo_120)):
        n_params = sum(p.numel() for p in modelo.net.parameters())
        bottleneck_t = ventana // arch["pool_size"]
        filas.append({
            "ventana_min": ventana, "canales": str(arch["canales"]), "kernel_size": arch["kernel_size"],
            "dilataciones": str(arch["dilataciones"]), "latent_channels": arch["latent_channels"],
            "pool_size": arch["pool_size"], "bottleneck_shape": f"({arch['latent_channels']}, {bottleneck_t})",
            "bottleneck_valores_totales": arch["latent_channels"] * bottleneck_t,
            "factor_compresion_temporal": arch["pool_size"], "n_parametros": n_params,
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------
# 4) Rejilla disjunta (stride == ventana) y puntuacion, parametrizadas por tamano_ventana
# ----------------------------------------------------------------------------------------

def _ventanas_disjuntas_afectadas(offset_global: int, duracion: int, tamano_ventana: int, n_ventanas_grid: int) -> list[int]:
    k_start = offset_global // tamano_ventana
    k_end = (offset_global + duracion - 1) // tamano_ventana
    return list(range(max(0, k_start), min(n_ventanas_grid - 1, k_end) + 1))


def _ventana_atacada(valores: np.ndarray, w_start: int, offset_global: int, duracion: int,
                      y_tramo: np.ndarray, tamano_ventana: int) -> np.ndarray:
    ventana = valores[w_start:w_start + tamano_ventana].copy()
    a_end = offset_global + duracion
    w_end = w_start + tamano_ventana
    s_ini, s_fin = max(offset_global, w_start), min(a_end, w_end)
    if s_ini < s_fin:
        ventana[s_ini - w_start:s_fin - w_start] = y_tramo[s_ini - offset_global:s_fin - offset_global]
    return ventana


def puntuar_configuracion(ataques: dict, particion_val_raw: np.ndarray, grid: dict, modelo, params_norm: dict,
                           tipo_score: str, tamano_ventana: int) -> tuple[pd.DataFrame, np.ndarray]:
    n_ventanas_grid = grid["X"].shape[0]
    scores_limpios_grid = calcular_scores(modelo, aplicar_zscore(grid["X"], params_norm), tipo_score)

    filas_meta, ventanas_atacadas = [], []
    for episode_id, ataque in ataques.items():
        ks = _ventanas_disjuntas_afectadas(ataque["offset_global"], ataque["duracion"], tamano_ventana, n_ventanas_grid)
        assert len(ks) >= 1, f"episodio {episode_id} sin ventana afectada (ventana={tamano_ventana})"
        for k in ks:
            w_start = k * tamano_ventana
            va = _ventana_atacada(particion_val_raw, w_start, ataque["offset_global"], ataque["duracion"],
                                   ataque["y_tramo"], tamano_ventana)
            assert len(va) == tamano_ventana
            ventanas_atacadas.append(va)
            filas_meta.append({"episode_id": episode_id, "k": k, "window_start": grid["inicio"][k],
                                "window_end": grid["fin"][k], "clean_score": float(scores_limpios_grid[k])})

    X_atacadas = np.stack(ventanas_atacadas).astype(np.float32)
    attacked_scores = calcular_scores(modelo, aplicar_zscore(X_atacadas, params_norm), tipo_score)
    tabla = pd.DataFrame(filas_meta)
    tabla["attacked_score"] = attacked_scores
    return tabla, scores_limpios_grid


# ----------------------------------------------------------------------------------------
# 5) Agregados
# ----------------------------------------------------------------------------------------

def _resumen(grupo: pd.DataFrame, tabla_ventanas: pd.DataFrame | None = None) -> dict:
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
        "ventanas_afectadas_media": grupo["n_affected_windows"].mean(),
    }
    if tabla_ventanas is not None and len(tabla_ventanas):
        sub = tabla_ventanas[tabla_ventanas["episode_id"].isin(grupo["episode_id"])]
        delta = sub["attacked_score"] - sub["clean_score"]
        resultado["score_delta_medio"] = delta.mean()
        resultado["score_delta_mediano"] = delta.median()
        resultado["pct_ventanas_delta_positivo"] = 100 * (delta > 0).mean()
    return resultado


def construir_agregados(resultados_todo: pd.DataFrame, umbrales_tabla: pd.DataFrame,
                         tablas_ventanas: dict[int, pd.DataFrame]) -> dict:
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL].copy()
    tablas = {}

    filas = []
    for ventana, grupo in principal.groupby("ventana_min"):
        fila = {"ventana_min": ventana}
        fila.update(_resumen(grupo, tablas_ventanas[ventana]))
        u = umbrales_tabla[(umbrales_tabla["ventana_min"] == ventana) & (umbrales_tabla["objetivo_nombre"] == OBJETIVO_PRINCIPAL)].iloc[0]
        fila["false_alarm_events_per_day"] = u["tasa_eventos_dia_real"]
        fila["false_alarm_windows_per_day"] = u["tasa_ventanas_dia_real"]
        filas.append(fila)
    tablas["por_ventana"] = pd.DataFrame(filas)

    filas = []
    for (ventana, dur), grupo in principal.groupby(["ventana_min", "duration_min"]):
        fila = {"ventana_min": ventana, "duration_min": dur}
        fila.update(_resumen(grupo, tablas_ventanas[ventana]))
        filas.append(fila)
    tablas["por_ventana_duracion"] = pd.DataFrame(filas)

    filas = []
    for (ventana, fam), grupo in principal.groupby(["ventana_min", "family"]):
        fila = {"ventana_min": ventana, "family": fam}
        fila.update(_resumen(grupo, tablas_ventanas[ventana]))
        filas.append(fila)
    tablas["por_ventana_familia"] = pd.DataFrame(filas)

    filas = []
    for (ventana, fam, param), grupo in principal.groupby(["ventana_min", "family", "parameter"]):
        fila = {"ventana_min": ventana, "family": fam, "parameter": param}
        fila.update(_resumen(grupo, tablas_ventanas[ventana]))
        filas.append(fila)
    tablas["por_ventana_familia_parametro"] = pd.DataFrame(filas)

    filas = []
    for ventana, grupo in principal.groupby("ventana_min"):
        fila = {"ventana_min": ventana, "familias": "todas"}
        fila.update(_resumen(grupo, tablas_ventanas[ventana]))
        filas.append(fila)
        sin_picos = grupo[grupo["family"] != "recorte_picos"]
        fila2 = {"ventana_min": ventana, "familias": "sin_recorte_picos"}
        fila2.update(_resumen(sin_picos, tablas_ventanas[ventana]))
        filas.append(fila2)
    tablas["global_con_sin_recorte_picos"] = pd.DataFrame(filas)

    filas = []
    for (ventana, objetivo), grupo in resultados_todo.groupby(["ventana_min", "objetivo_nombre"]):
        sin_picos = grupo[grupo["family"] != "recorte_picos"]
        fila = {"ventana_min": ventana, "objetivo_nombre": objetivo}
        fila.update(_resumen(sin_picos))
        u = umbrales_tabla[(umbrales_tabla["ventana_min"] == ventana) & (umbrales_tabla["objetivo_nombre"] == objetivo)].iloc[0]
        fila["false_alarm_events_per_day"] = u["tasa_eventos_dia_real"]
        fila["false_alarm_windows_per_day"] = u["tasa_ventanas_dia_real"]
        filas.append(fila)
    tablas["por_objetivo_fa"] = pd.DataFrame(filas)

    return tablas


# ----------------------------------------------------------------------------------------
# 6) Comparacion pareada (120 vs 360), varios subconjuntos
# ----------------------------------------------------------------------------------------

def comparacion_pareada_120_vs_360(resultados_todo: pd.DataFrame) -> pd.DataFrame:
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    pivote = principal.pivot(index="episode_id", columns="ventana_min", values="induced_detected").astype(bool)
    familias = principal.drop_duplicates("episode_id").set_index("episode_id")[["family", "duration_min"]]
    pivote = pivote.join(familias)

    subconjuntos = {
        "todos": pivote,
        "30min": pivote[pivote["duration_min"] == 30],
        "60min": pivote[pivote["duration_min"] == 60],
        "120min": pivote[pivote["duration_min"] == 120],
        "240min_o_mas": pivote[pivote["duration_min"] >= 240],
        "sin_recorte_picos": pivote[pivote["family"] != "recorte_picos"],
    }

    filas = []
    for nombre_sub, sub in subconjuntos.items():
        det_120, det_360 = sub[120].to_numpy(), sub[360].to_numpy()
        solo_120 = int((det_120 & ~det_360).sum())
        solo_360 = int((~det_120 & det_360).sum())
        ambos = int((det_120 & det_360).sum())
        ninguno = int((~det_120 & ~det_360).sum())
        p = mcnemar_p(solo_120, solo_360)
        ci_low, ci_high = bootstrap_pareado(det_120.astype(float), det_360.astype(float))
        filas.append({
            "subconjunto": nombre_sub, "n_episodios": len(sub),
            "detectado_solo_con_120": solo_120, "detectado_solo_con_360": solo_360,
            "detectado_con_ambos": ambos, "detectado_con_ninguno": ninguno,
            "DR_120_pct": 100 * det_120.mean() if len(sub) else float("nan"),
            "DR_360_pct": 100 * det_360.mean() if len(sub) else float("nan"),
            "mcnemar_p_valor": p,
            "bootstrap_ci95_diferencia_(120-360)_low": ci_low, "bootstrap_ci95_diferencia_(120-360)_high": ci_high,
            "diferencia_significativa_p<0.05": p < 0.05,
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------
# 7) Calidad de reconstruccion limpia
# ----------------------------------------------------------------------------------------

def calidad_reconstruccion_limpia(datos_360, modelo_360, datos_120, modelo_120, tipo_score) -> pd.DataFrame:
    filas = []
    for ventana, datos, modelo in ((360, datos_360, modelo_360), (120, datos_120, modelo_120)):
        X = datos["ventanas_norm"]["val"]
        X_hat = modelo.reconstruir(X)
        mae = float(np.mean(np.abs(X - X_hat)))
        mse = float(np.mean((X - X_hat) ** 2))
        scores = calcular_scores(modelo, X, tipo_score)
        media, std = float(scores.mean()), float(scores.std())
        extremos = np.abs(scores - media) > 3 * std

        horas = pd.to_datetime(datos["ventanas"]["val"]["inicio"]).hour
        std_por_hora = pd.Series(scores).groupby(horas).std()

        consumo_medio = datos["ventanas"]["val"]["X"].mean(axis=1)
        terciles = pd.qcut(consumo_medio, 3, labels=["bajo", "medio", "alto"])
        std_por_tercil = pd.Series(scores).groupby(terciles, observed=True).std()

        filas.append({
            "ventana_min": ventana, "mae": mae, "mse": mse, "score_medio": media, "score_std": std,
            "pct_extremos_3sigma": 100 * float(extremos.mean()),
            "std_score_por_hora_media": float(std_por_hora.mean()), "std_score_por_hora_max": float(std_por_hora.max()),
            "std_score_tercil_bajo": float(std_por_tercil.get("bajo", float("nan"))),
            "std_score_tercil_medio": float(std_por_tercil.get("medio", float("nan"))),
            "std_score_tercil_alto": float(std_por_tercil.get("alto", float("nan"))),
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------
# 8) Figuras
# ----------------------------------------------------------------------------------------

def fig_dr_por_duracion(tabla_vd: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ventana in TAMANOS:
        sub = tabla_vd[tabla_vd["ventana_min"] == ventana].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", markersize=4, linewidth=1.8,
                color=COLOR_TAMANO[ventana], label=f"ventana={ventana} min")
    ax.set_xlabel("duracion del ataque (min)"); ax.set_ylabel("induced_DR (%)")
    ax.set_title("induced_DR por duracion, ventana=360 vs 120 (0.06 eventos/dia)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_dr_por_familia(tabla_vf: pd.DataFrame, nombre: str) -> None:
    familias = sorted(tabla_vf["family"].unique())
    x = np.arange(len(familias)); ancho = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, ventana in enumerate(TAMANOS):
        sub = tabla_vf[tabla_vf["ventana_min"] == ventana].set_index("family").reindex(familias)
        ax.bar(x + (i - 0.5) * ancho, sub["induced_DR_pct"], width=ancho, color=COLOR_TAMANO[ventana], label=f"ventana={ventana}")
    ax.set_xticks(x); ax.set_xticklabels(familias, rotation=15, ha="right")
    ax.set_ylabel("induced_DR (%)")
    ax.set_title("induced_DR por familia, ventana=360 vs 120")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_score_delta_por_duracion(tabla_vd: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ventana in TAMANOS:
        sub = tabla_vd[tabla_vd["ventana_min"] == ventana].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["score_delta_mediano"], marker="o", markersize=4, linewidth=1.8,
                color=COLOR_TAMANO[ventana], label=f"ventana={ventana} min")
    ax.axhline(0, color=GRIS, linestyle="--", linewidth=1)
    ax.set_xlabel("duracion del ataque (min)"); ax.set_ylabel("score_delta mediano")
    ax.set_title("score_delta mediano por duracion, ventana=360 vs 120")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_retraso_por_duracion(tabla_vd: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ventana in TAMANOS:
        sub = tabla_vd[tabla_vd["ventana_min"] == ventana].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["retraso_mediano_induced_min"], marker="o", markersize=4, linewidth=1.8,
                color=COLOR_TAMANO[ventana], label=f"ventana={ventana} min")
    ax.set_xlabel("duracion del ataque (min)"); ax.set_ylabel("retraso mediano (min)")
    ax.set_title("Retraso mediano de deteccion inducida, ventana=360 vs 120")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_dr_vs_fa(tabla_objetivo: pd.DataFrame, nombre: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for ventana in TAMANOS:
        sub = tabla_objetivo[tabla_objetivo["ventana_min"] == ventana].sort_values("false_alarm_events_per_day")
        ax.plot(sub["false_alarm_events_per_day"], sub["induced_DR_pct"], marker="o", markersize=5, linewidth=1.8,
                color=COLOR_TAMANO[ventana], label=f"ventana={ventana} min")
    ax.axvline(0.06, color=GRIS, linestyle="--", linewidth=1, label="objetivo principal (0.06/dia)")
    ax.set_xlabel("falsos eventos / dia (real)"); ax.set_ylabel("induced_DR (%) -- sin recorte_picos")
    ax.set_title("induced_DR vs falsos eventos/dia, ventana=360 vs 120")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_distribucion_scores_limpios(datos_360, modelo_360, datos_120, modelo_120, tipo_score, nombre: str) -> None:
    s360 = calcular_scores(modelo_360, datos_360["ventanas_norm"]["val"], tipo_score)
    s120 = calcular_scores(modelo_120, datos_120["ventanas_norm"]["val"], tipo_score)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.histogram_bin_edges(np.concatenate([s360, s120]), bins=70)
    ax.hist(s360, bins=bins, color=COLOR_TAMANO[360], alpha=0.65, density=True, label=f"ventana=360 (n={len(s360)})")
    ax.hist(s120, bins=bins, color=COLOR_TAMANO[120], alpha=0.65, density=True, label=f"ventana=120 (n={len(s120)})")
    ax.set_xlabel("score limpio (deficit)"); ax.set_ylabel("densidad")
    ax.set_title("Distribucion de scores limpios (val), ventana=360 vs 120")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_matriz_mejora(tabla_vfd: pd.DataFrame, nombre: str) -> None:
    p360 = tabla_vfd[tabla_vfd["ventana_min"] == 360].pivot(index="family", columns="duration_min", values="induced_DR_pct")
    p120 = tabla_vfd[tabla_vfd["ventana_min"] == 120].pivot(index="family", columns="duration_min", values="induced_DR_pct")
    mejora = (p120 - p360).reindex(columns=DURACIONES)
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
    ax.set_title("Mejora de ventana 360 -> 120 (induced_DR, puntos porcentuales)")
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_pareado_duraciones_cortas(pareada: pd.DataFrame, nombre: str) -> None:
    subset = pareada[pareada["subconjunto"].isin(["30min", "60min", "120min"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(subset)); ancho = 0.35
    ax.bar(x - ancho / 2, subset["DR_360_pct"], width=ancho, color=COLOR_TAMANO[360], label="ventana=360")
    ax.bar(x + ancho / 2, subset["DR_120_pct"], width=ancho, color=COLOR_TAMANO[120], label="ventana=120")
    for i, (_, fila) in enumerate(subset.iterrows()):
        marca = "*" if fila["diferencia_significativa_p<0.05"] else "n.s."
        ax.text(i, max(fila["DR_360_pct"], fila["DR_120_pct"]) + 1, marca, ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(subset["subconjunto"])
    ax.set_ylabel("induced_DR (%)")
    ax.set_title("Comparacion pareada 360 vs 120 (30/60/120 min) -- * = p<0.05 (McNemar)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_ejemplos_reconstruccion_limpia(datos_360, modelo_360, datos_120, modelo_120, nombre: str, n_ejemplos: int = 3) -> None:
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(n_ejemplos, 2, figsize=(11, 2.6 * n_ejemplos))
    for col, (ventana, datos, modelo) in enumerate(((360, datos_360, modelo_360), (120, datos_120, modelo_120))):
        idxs = rng.choice(datos["ventanas_norm"]["val"].shape[0], n_ejemplos, replace=False)
        for fila, idx in enumerate(idxs):
            X = datos["ventanas_norm"]["val"][idx:idx + 1]
            X_hat = modelo.reconstruir(X)
            ax = axes[fila, col]
            ax.plot(X[0], color=GRIS, linewidth=1, label="real")
            ax.plot(X_hat[0], color=COLOR_TAMANO[ventana], linewidth=1.3, label="reconstruido")
            if fila == 0:
                ax.set_title(f"ventana={ventana}")
                ax.legend(fontsize=7)
    fig.tight_layout(); _guardar_fig(fig, nombre)


def fig_ejemplos_120_no_360(pareada_todos: dict, ataques: dict, particion_val_raw: np.ndarray,
                             inicio_particion: pd.Timestamp, nombre: str, n_ejemplos: int = 3) -> None:
    ids = pareada_todos["ids_solo_120"][:n_ejemplos]
    if not ids:
        return
    fig, axes = plt.subplots(len(ids), 1, figsize=(9, 3 * len(ids)))
    if len(ids) == 1:
        axes = [axes]
    margen = 60
    for ax, ep_id in zip(axes, ids):
        a = ataques[ep_id]
        ini = max(0, a["offset_global"] - margen)
        fin = a["offset_global"] + a["duracion"] + margen
        x_limpio = particion_val_raw[ini:fin]
        x_atacado = x_limpio.copy()
        x_atacado[a["offset_global"] - ini: a["offset_global"] - ini + a["duracion"]] = a["y_tramo"]
        t = [inicio_particion + pd.Timedelta(minutes=m) for m in range(ini, fin)]
        ax.plot(t, x_limpio, color=GRIS, linewidth=1.1, label="limpio")
        ax.plot(t, x_atacado, color=ROJO, linewidth=1.1, alpha=0.8, label="reportado (atacado)")
        ax.axvspan(inicio_particion + pd.Timedelta(minutes=a["offset_global"]),
                   inicio_particion + pd.Timedelta(minutes=a["offset_global"] + a["duracion"]),
                   color=VERDE, alpha=0.08)
        ax.set_title(f"{ep_id} | {a['family']} ({a['parameter']}) | duracion={a['duracion']} min "
                     f"-- detectado por 120, NO por 360")
        ax.legend(fontsize=7)
    fig.tight_layout(); _guardar_fig(fig, nombre)


# ----------------------------------------------------------------------------------------
# Orquestacion
# ----------------------------------------------------------------------------------------

def main() -> dict:
    config_360, datos_360, modelo_360 = cargar_datos_y_modelo()   # checkpoint EXISTENTE, no reentrena
    tipo_score = config_360["score"]["tipo"]

    print("Reconstruyendo ataques exactos de episodes_master_val.csv...")
    ataques, master_csv = reconstruir_ataques_exactos(config_360, datos_360)
    print(f"{len(ataques)} episodios validados.\n")

    config_120 = construir_config_120(config_360)
    datos_120 = preparar_datos(config_120)
    modelo_120, meta_120 = entrenar_o_cargar_120(config_120, datos_120)
    print()

    arch_tabla = documentar_arquitectura(modelo_360, modelo_120, config_360)
    _guardar_tabla(arch_tabla, "01_arquitectura")
    print("Arquitectura (360 vs 120):")
    print(arch_tabla.to_string(index=False))
    print()

    particion_val = datos_360["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    inicio_particion = particion_val.index[0]
    n_dias = len(particion_val_raw) / 1440.0

    datos_por_ventana = {360: datos_360, 120: datos_120}
    modelos_por_ventana = {360: modelo_360, 120: modelo_120}

    filas_umbrales = []
    resultados_por_ventana_objetivo = []
    tablas_ventanas_por_config = {}

    for ventana in TAMANOS:
        print(f"=== ventana={ventana} min ===")
        datos_v = datos_por_ventana[ventana]
        modelo_v = modelos_por_ventana[ventana]
        grid = datos_v["ventanas"]["val"]
        print(f"  rejilla: {grid['X'].shape[0]} ventanas de {grid['X'].shape[1]} muestras")

        tabla_ventanas, scores_limpios_grid = puntuar_configuracion(
            ataques, particion_val_raw, grid, modelo_v, datos_v["params_norm"], tipo_score, ventana
        )
        tablas_ventanas_por_config[ventana] = tabla_ventanas
        print(f"  {len(tabla_ventanas)} filas ventana-episodio puntuadas")

        for objetivo_nombre, objetivo_valor in OBJETIVOS_EVENTOS_DIA.items():
            umbral, tasa_eventos, tasa_ventanas = calibrar_umbral_por_eventos(scores_limpios_grid, n_dias, objetivo_valor)
            filas_umbrales.append({
                "ventana_min": ventana, "objetivo_nombre": objetivo_nombre, "objetivo_valor": objetivo_valor,
                "threshold": umbral, "tasa_eventos_dia_real": tasa_eventos, "tasa_ventanas_dia_real": tasa_ventanas,
            })
            print(f"    objetivo {objetivo_nombre} -> umbral={umbral:.5f}, eventos/dia={tasa_eventos:.4f}, "
                  f"ventanas/dia={tasa_ventanas:.3f}")

            resultado_ep = agregar_por_umbral(tabla_ventanas, master_csv, umbral, ventana)
            resultado_ep = resultado_ep.rename(columns={"stride_min": "ventana_min"})
            resultado_ep["objetivo_nombre"] = objetivo_nombre
            resultado_ep["objetivo_valor"] = objetivo_valor
            resultados_por_ventana_objetivo.append(resultado_ep)
        print()

    umbrales_tabla = _guardar_tabla(pd.DataFrame(filas_umbrales), "02_umbrales_por_ventana_objetivo")
    resultados_todo = pd.concat(resultados_por_ventana_objetivo, ignore_index=True)
    _guardar_tabla(resultados_todo, "02_episode_detection_results_ventana")

    print("Construyendo agregados...")
    agregados = construir_agregados(resultados_todo, umbrales_tabla, tablas_ventanas_por_config)
    for nombre, tabla in agregados.items():
        _guardar_tabla(tabla, f"04_{nombre}")

    print("\nResumen por ventana (objetivo principal 0.06 eventos/dia, TODAS las familias):")
    print(agregados["por_ventana"].to_string(index=False))
    print("\nGlobal con vs sin recorte_picos:")
    print(agregados["global_con_sin_recorte_picos"].to_string(index=False))

    print("\nComparacion pareada 120 vs 360...")
    pareada = comparacion_pareada_120_vs_360(resultados_todo)
    _guardar_tabla(pareada, "05_comparacion_pareada")
    print(pareada.to_string(index=False))

    print("\nCalidad de reconstruccion limpia...")
    calidad = calidad_reconstruccion_limpia(datos_360, modelo_360, datos_120, modelo_120, tipo_score)
    _guardar_tabla(calidad, "06_calidad_reconstruccion_limpia")
    print(calidad.to_string(index=False))

    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    pivote_ids = principal.pivot(index="episode_id", columns="ventana_min", values="induced_detected").astype(bool)
    ids_solo_120 = pivote_ids[pivote_ids[120] & ~pivote_ids[360]].index.tolist()

    print("\nGenerando figuras...")
    fig_dr_por_duracion(agregados["por_ventana_duracion"], "01_induced_dr_por_duracion_ventana")
    fig_dr_por_familia(agregados["por_ventana_familia"], "02_induced_dr_por_familia_ventana")
    fig_score_delta_por_duracion(agregados["por_ventana_duracion"], "03_score_delta_por_duracion_ventana")
    fig_retraso_por_duracion(agregados["por_ventana_duracion"], "04_retraso_mediano_por_duracion_ventana")
    fig_dr_vs_fa(agregados["por_objetivo_fa"], "05_induced_dr_vs_falsos_eventos_dia")
    fig_distribucion_scores_limpios(datos_360, modelo_360, datos_120, modelo_120, tipo_score, "06_distribucion_scores_limpios")

    filas_vfd = []
    for (ventana, fam, dur), grupo in principal.groupby(["ventana_min", "family", "duration_min"]):
        n = len(grupo)
        filas_vfd.append({"ventana_min": ventana, "family": fam, "duration_min": dur,
                           "induced_DR_pct": 100 * grupo["induced_detected"].sum() / n if n else float("nan")})
    tabla_vfd = pd.DataFrame(filas_vfd)
    _guardar_tabla(tabla_vfd, "04_por_ventana_familia_duracion")
    fig_matriz_mejora(tabla_vfd, "07_matriz_mejora_360_a_120")

    fig_pareado_duraciones_cortas(pareada, "08_comparacion_pareada_duraciones_cortas")
    fig_ejemplos_reconstruccion_limpia(datos_360, modelo_360, datos_120, modelo_120, "09_ejemplos_reconstruccion_limpia")
    fig_ejemplos_120_no_360({"ids_solo_120": ids_solo_120}, ataques, particion_val_raw, inicio_particion,
                             "10_ejemplos_detectados_120_no_360")

    print(f"\nTablas en {EXP_TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {EXP_FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "config_360": config_360, "config_120": config_120, "datos_360": datos_360, "datos_120": datos_120,
        "modelo_360": modelo_360, "modelo_120": modelo_120, "meta_120": meta_120,
        "ataques": ataques, "master_csv": master_csv, "umbrales_tabla": umbrales_tabla,
        "resultados_todo": resultados_todo, "agregados": agregados, "pareada": pareada, "calidad": calidad,
        "tablas_ventanas_por_config": tablas_ventanas_por_config, "n_dias": n_dias, "tipo_score": tipo_score,
    }


if __name__ == "__main__":
    main()
