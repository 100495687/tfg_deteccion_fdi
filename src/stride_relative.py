"""FASE B -- repetir el experimento de stride (360/60/30) usando UNICAMENTE `relative_kw`
(el score declarado en la Fase A). Ventana=360 fija, mismo checkpoint existente. Reutiliza
`cargar_ataques_y_modelo_360` y `calcular_relative_kw` de base_relative.py (mismos episode_id
y misma senal atacada que la Fase A), y `_ventanas_solapadas`/`contar_eventos`/
`calibrar_umbral_por_eventos`/`mcnemar_p`/`bootstrap_pareado` de experimento_stride.py sin
modificarlos.

Ejecutable de forma aislada:
    python -m src.stride_relative
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.base_relative import calcular_relative_kw, cargar_ataques_y_modelo_360
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import wilson_ci
from src.experimento_stride import (
    _ventana_atacada,
    _ventanas_solapadas,
    bootstrap_pareado,
    calibrar_umbral_por_eventos,
    contar_eventos,
    construir_rejilla,
    mcnemar_p,
)
from src.normalization import aplicar_zscore

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables"
EXP_TABLAS_DIR = TABLAS_DIR / "stride_relative"
EXP_FIGURAS_DIR = BASE_DIR / "results" / "figures" / "stride_relative"

AZUL_CLARO, AZUL, AZUL_OSCURO, GRIS = "#a9c8ec", "#2a78d6", "#154a86", "#8a8a86"
COLOR_STRIDE = {360: AZUL_CLARO, 60: AZUL, 30: AZUL_OSCURO}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

STRIDES = [360, 60, 30]
OBJETIVOS_EVENTOS_DIA = {"principal_0.06": 0.06, "sec_0.05": 0.05, "sec_0.10": 0.10, "sec_0.25": 0.25}
OBJETIVO_PRINCIPAL = "principal_0.06"
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
SCORE = "relative_kw"


def _guardar_fig(fig, nombre):
    EXP_FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP_FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    EXP_TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP_TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


def puntuar_stride_relative(ataques: dict, particion_val_raw: np.ndarray, grid: dict, modelo,
                             params_norm: dict, stride: int) -> tuple[pd.DataFrame, np.ndarray]:
    n_ventanas_grid = grid["X"].shape[0]
    X_clean_norm = aplicar_zscore(grid["X"], params_norm)
    X_hat_clean = modelo.reconstruir(X_clean_norm)
    scores_limpios = calcular_relative_kw(X_clean_norm, X_hat_clean, params_norm)

    filas_meta, ventanas_atacadas = [], []
    for episode_id, ataque in ataques.items():
        ks = _ventanas_solapadas(ataque["offset_global"], ataque["duracion"], stride, n_ventanas_grid)
        for k in ks:
            w_start = k * stride
            va = _ventana_atacada(particion_val_raw, w_start, ataque["offset_global"], ataque["duracion"], ataque["y_tramo"])
            ventanas_atacadas.append(va)
            filas_meta.append({"episode_id": episode_id, "k": k, "window_start": grid["inicio"][k], "window_end": grid["fin"][k]})

    X_atacadas_norm = aplicar_zscore(np.stack(ventanas_atacadas).astype(np.float32), params_norm)
    X_hat_atacadas = modelo.reconstruir(X_atacadas_norm)
    scores_atacados = calcular_relative_kw(X_atacadas_norm, X_hat_atacadas, params_norm)

    tabla = pd.DataFrame(filas_meta)
    tabla["clean_score"] = scores_limpios[tabla["k"].to_numpy()]
    tabla["attacked_score"] = scores_atacados
    return tabla, scores_limpios


def agregar_por_umbral(tabla_ventanas: pd.DataFrame, master_csv: pd.DataFrame, umbral: float, stride: int) -> pd.DataFrame:
    t = tabla_ventanas.copy()
    t["window_end"] = pd.to_datetime(t["window_end"])
    t["clean_is_alarm"] = t["clean_score"] > umbral
    t["attacked_is_alarm"] = t["attacked_score"] > umbral
    t["attack_induced_alarm"] = t["attacked_is_alarm"] & ~t["clean_is_alarm"]

    agg = t.groupby("episode_id").agg(
        n_affected_windows=("k", "size"), raw_detected=("attacked_is_alarm", "any"),
        induced_detected=("attack_induced_alarm", "any"),
    ).reset_index()
    induced_ts = t[t["attack_induced_alarm"]].groupby("episode_id")["window_end"].min().rename("first_induced_alarm_timestamp")
    agg = agg.merge(induced_ts, on="episode_id", how="left")

    resultado = agg.merge(master_csv[["episode_id", "family", "parameter", "duration_min", "attack_start", "attack_end"]],
                           on="episode_id", how="left")
    resultado["induced_detection_delay_min"] = (resultado["first_induced_alarm_timestamp"] - resultado["attack_start"]) / pd.Timedelta(minutes=1)
    durante = (resultado["first_induced_alarm_timestamp"] <= resultado["attack_end"]).astype(object)
    sin_induced = ~resultado["induced_detected"]
    durante[sin_induced] = np.nan
    resultado["induced_detected_during_attack"] = durante
    resultado.loc[sin_induced, "induced_detection_delay_min"] = np.nan
    resultado["stride_min"] = stride
    resultado["threshold"] = umbral
    return resultado


def _resumen(grupo: pd.DataFrame) -> dict:
    n = len(grupo)
    n_induced = int(grupo["induced_detected"].sum())
    ci = wilson_ci(n_induced, n)
    con_induced = grupo[grupo["induced_detected"] == True]  # noqa: E712
    return {
        "n_episodios": n, "induced_DR_pct": 100 * n_induced / n if n else float("nan"),
        "induced_DR_ci95_low": ci[0], "induced_DR_ci95_high": ci[1],
        "retraso_mediano_induced_min": con_induced["induced_detection_delay_min"].median() if len(con_induced) else float("nan"),
        "pct_detectado_durante_ataque": 100 * (con_induced["induced_detected_during_attack"] == True).mean() if len(con_induced) else float("nan"),  # noqa: E712
        "ventanas_afectadas_media": grupo["n_affected_windows"].mean(),
    }


def comparacion_pareada(resultados_todo: pd.DataFrame) -> pd.DataFrame:
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    pivote = principal.pivot(index="episode_id", columns="stride_min", values="induced_detected").astype(bool)
    meta = principal.drop_duplicates("episode_id").set_index("episode_id")[["family", "duration_min"]]
    pivote = pivote.join(meta)

    subconjuntos = {"todos": pivote, "sin_recorte_picos": pivote[pivote["family"] != "recorte_picos"],
                     "30min": pivote[pivote["duration_min"] == 30], "60min": pivote[pivote["duration_min"] == 60],
                     "120min": pivote[pivote["duration_min"] == 120], "240min_o_mas": pivote[pivote["duration_min"] >= 240]}

    filas = []
    for a, b in [(360, 60), (60, 30), (360, 30)]:
        for nombre_sub, sub in subconjuntos.items():
            det_a, det_b = sub[a].to_numpy(), sub[b].to_numpy()
            solo_a, solo_b = int((det_a & ~det_b).sum()), int((~det_a & det_b).sum())
            p = mcnemar_p(solo_a, solo_b)
            ci_low, ci_high = bootstrap_pareado(det_a.astype(float), det_b.astype(float))
            filas.append({
                "stride_a": a, "stride_b": b, "subconjunto": nombre_sub, "n_episodios": len(sub),
                f"detectado_solo_con_{a}": solo_a, f"detectado_solo_con_{b}": solo_b,
                "detectado_con_ambos": int((det_a & det_b).sum()), "detectado_con_ninguno": int((~det_a & ~det_b).sum()),
                "DR_a_pct": 100 * det_a.mean() if len(sub) else float("nan"), "DR_b_pct": 100 * det_b.mean() if len(sub) else float("nan"),
                "mcnemar_p_valor": p, "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
                "diferencia_significativa_p<0.05": p < 0.05,
            })
    return pd.DataFrame(filas)


def main() -> dict:
    config, datos, modelo, ataques, master_csv = cargar_ataques_y_modelo_360()
    params_norm = datos["params_norm"]
    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    n_dias = len(particion_val_raw) / 1440.0

    print(f"FASE B -- {len(ataques)} episodios, ventana=360, score={SCORE}, strides={STRIDES}\n")

    filas_umbrales, resultados_por_stride_objetivo = [], []
    for stride in STRIDES:
        print(f"=== stride={stride} min ===")
        grid = construir_rejilla(particion_val, stride)
        tabla, scores_limpios = puntuar_stride_relative(ataques, particion_val_raw, grid, modelo, params_norm, stride)
        for objetivo_nombre, objetivo_valor in OBJETIVOS_EVENTOS_DIA.items():
            u, tasa_eventos, tasa_ventanas = calibrar_umbral_por_eventos(scores_limpios, n_dias, objetivo_valor)
            filas_umbrales.append({"stride_min": stride, "objetivo_nombre": objetivo_nombre, "objetivo_valor": objetivo_valor,
                                    "threshold": u, "tasa_eventos_dia_real": tasa_eventos, "tasa_ventanas_dia_real": tasa_ventanas})
            if objetivo_nombre == OBJETIVO_PRINCIPAL:
                print(f"  {objetivo_nombre}: umbral={u:.5f}, eventos/dia={tasa_eventos:.4f}, ventanas/dia={tasa_ventanas:.3f}")
            r = agregar_por_umbral(tabla, master_csv, u, stride)
            r["objetivo_nombre"], r["objetivo_valor"] = objetivo_nombre, objetivo_valor
            resultados_por_stride_objetivo.append(r)
        print()

    umbrales_tabla = _guardar_tabla(pd.DataFrame(filas_umbrales), "01_umbrales")
    resultados_todo = pd.concat(resultados_por_stride_objetivo, ignore_index=True)
    _guardar_tabla(resultados_todo, "01_episode_detection_results")

    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    filas = []
    for stride, grupo in principal.groupby("stride_min"):
        for etiqueta, g in (("todas", grupo), ("sin_recorte_picos", grupo[grupo["family"] != "recorte_picos"])):
            fila = {"stride_min": stride, "familias": etiqueta}
            fila.update(_resumen(g)); filas.append(fila)
    tabla_global = _guardar_tabla(pd.DataFrame(filas), "02_global_con_sin_recorte_picos")
    print("Global por stride:")
    print(tabla_global.to_string(index=False))

    filas = []
    for (stride, dur), grupo in principal.groupby(["stride_min", "duration_min"]):
        fila = {"stride_min": stride, "duration_min": dur}
        fila.update(_resumen(grupo)); filas.append(fila)
    tabla_duracion = _guardar_tabla(pd.DataFrame(filas), "02_por_duracion")
    print("\nPor duracion:")
    print(tabla_duracion.to_string(index=False))

    filas = []
    for (stride, fam), grupo in principal.groupby(["stride_min", "family"]):
        fila = {"stride_min": stride, "family": fam}
        fila.update(_resumen(grupo)); filas.append(fila)
    tabla_familia = _guardar_tabla(pd.DataFrame(filas), "02_por_familia")

    pareada = comparacion_pareada(resultados_todo)
    _guardar_tabla(pareada, "03_comparacion_pareada")
    print("\nComparacion pareada (subconjunto 'todos' y 'sin_recorte_picos'):")
    print(pareada[pareada["subconjunto"].isin(["todos", "sin_recorte_picos"])].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for stride in STRIDES:
        sub = tabla_duracion[tabla_duracion["stride_min"] == stride].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", color=COLOR_STRIDE[stride], linewidth=1.8, label=f"stride={stride}")
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)")
    ax.set_title(f"Fase B: induced_DR por duracion y stride ({SCORE}, 0.06 eventos/dia)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "01_induced_dr_por_duracion_stride")

    fig, ax = plt.subplots(figsize=(8, 5))
    for stride in STRIDES:
        sub = tabla_duracion[tabla_duracion["stride_min"] == stride].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["retraso_mediano_induced_min"], marker="o", color=COLOR_STRIDE[stride], linewidth=1.8, label=f"stride={stride}")
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("retraso mediano (min)")
    ax.set_title(f"Fase B: retraso mediano por duracion y stride ({SCORE})")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "02_retraso_por_duracion_stride")

    print(f"\nTablas en {EXP_TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {EXP_FIGURAS_DIR.relative_to(BASE_DIR)}/")
    return {"umbrales_tabla": umbrales_tabla, "resultados_todo": resultados_todo, "tabla_global": tabla_global,
            "tabla_duracion": tabla_duracion, "tabla_familia": tabla_familia, "pareada": pareada, "n_dias": n_dias}


if __name__ == "__main__":
    main()
