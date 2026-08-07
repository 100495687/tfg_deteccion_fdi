"""FASE C -- repetir el experimento de tamano de ventana (360 vs 120) usando UNICAMENTE
`relative_kw`. Reutiliza los checkpoints EXISTENTES de ambas ventanas (no reentrena ninguno:
el de 360 vino del pipeline original, el de 120 del experimento de ventana anterior).

La senal atacada se construye UNA sola vez por episodio (via
`base_relative.cargar_ataques_y_modelo_360`, que a su vez reutiliza
`episodes._generar_episodios` de forma determinista) y se reutiliza sin cambios para
ventanear con ambos tamanos -- asi `reduccion_variable` usa exactamente el mismo vector
aleatorio en las dos configuraciones.

Ejecutable de forma aislada:
    python -m src.ventana_relative
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.base_relative import calcular_relative_kw, cargar_ataques_y_modelo_360
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import wilson_ci
from src.experimento_scores import duraciones_eventos_min
from src.experimento_stride import bootstrap_pareado, calibrar_umbral_por_eventos, contar_eventos, mcnemar_p
from src.experimento_ventana120 import (
    _ventana_atacada,
    _ventanas_disjuntas_afectadas,
    construir_config_120,
    entrenar_o_cargar_120,
)
from src.normalization import aplicar_zscore
from src.pipeline import preparar_datos

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables"
EXP_TABLAS_DIR = TABLAS_DIR / "ventana_relative"
EXP_FIGURAS_DIR = BASE_DIR / "results" / "figures" / "ventana_relative"

AZUL_CLARO, AZUL_OSCURO, GRIS = "#a9c8ec", "#154a86", "#8a8a86"
COLOR_VENTANA = {360: AZUL_CLARO, 120: AZUL_OSCURO}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANOS = [360, 120]
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


def puntuar_ventana_relative(ataques: dict, particion_val_raw: np.ndarray, grid: dict, modelo,
                              params_norm: dict, tamano_ventana: int) -> tuple[pd.DataFrame, np.ndarray]:
    n_ventanas_grid = grid["X"].shape[0]
    X_clean_norm = aplicar_zscore(grid["X"], params_norm)
    X_hat_clean = modelo.reconstruir(X_clean_norm)
    scores_limpios = calcular_relative_kw(X_clean_norm, X_hat_clean, params_norm)

    filas_meta, ventanas_atacadas = [], []
    for episode_id, ataque in ataques.items():
        ks = _ventanas_disjuntas_afectadas(ataque["offset_global"], ataque["duracion"], tamano_ventana, n_ventanas_grid)
        for k in ks:
            w_start = k * tamano_ventana
            va = _ventana_atacada(particion_val_raw, w_start, ataque["offset_global"], ataque["duracion"],
                                   ataque["y_tramo"], tamano_ventana)
            ventanas_atacadas.append(va)
            filas_meta.append({"episode_id": episode_id, "k": k, "window_start": grid["inicio"][k], "window_end": grid["fin"][k]})

    X_atacadas_norm = aplicar_zscore(np.stack(ventanas_atacadas).astype(np.float32), params_norm)
    X_hat_atacadas = modelo.reconstruir(X_atacadas_norm)
    scores_atacados = calcular_relative_kw(X_atacadas_norm, X_hat_atacadas, params_norm)

    tabla = pd.DataFrame(filas_meta)
    tabla["clean_score"] = scores_limpios[tabla["k"].to_numpy()]
    tabla["attacked_score"] = scores_atacados
    return tabla, scores_limpios


def agregar_por_umbral(tabla_ventanas: pd.DataFrame, master_csv: pd.DataFrame, umbral: float, ventana: int) -> pd.DataFrame:
    t = tabla_ventanas.copy()
    t["window_end"] = pd.to_datetime(t["window_end"])
    t["clean_is_alarm"] = t["clean_score"] > umbral
    t["attacked_is_alarm"] = t["attacked_score"] > umbral
    t["attack_induced_alarm"] = t["attacked_is_alarm"] & ~t["clean_is_alarm"]
    t["score_delta"] = t["attacked_score"] - t["clean_score"]

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
    resultado["ventana_min"] = ventana
    resultado["threshold"] = umbral
    return resultado


def _resumen(grupo: pd.DataFrame, ventanas_grupo: pd.DataFrame | None = None) -> dict:
    n = len(grupo)
    n_induced = int(grupo["induced_detected"].sum())
    ci = wilson_ci(n_induced, n)
    con_induced = grupo[grupo["induced_detected"] == True]  # noqa: E712
    resultado = {
        "n_episodios": n, "induced_DR_pct": 100 * n_induced / n if n else float("nan"),
        "induced_DR_ci95_low": ci[0], "induced_DR_ci95_high": ci[1],
        "retraso_mediano_induced_min": con_induced["induced_detection_delay_min"].median() if len(con_induced) else float("nan"),
        "pct_detectado_durante_ataque": 100 * (con_induced["induced_detected_during_attack"] == True).mean() if len(con_induced) else float("nan"),  # noqa: E712
        "ventanas_afectadas_media": grupo["n_affected_windows"].mean(),
    }
    if ventanas_grupo is not None and len(ventanas_grupo):
        delta = ventanas_grupo["score_delta"]
        resultado["score_delta_medio"] = delta.mean()
        resultado["score_delta_mediano"] = delta.median()
    return resultado


def comparacion_pareada(resultados_todo: pd.DataFrame) -> pd.DataFrame:
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    pivote = principal.pivot(index="episode_id", columns="ventana_min", values="induced_detected").astype(bool)
    meta = principal.drop_duplicates("episode_id").set_index("episode_id")[["family", "duration_min"]]
    pivote = pivote.join(meta)

    subconjuntos = {
        "todos": pivote, "sin_recorte_picos": pivote[pivote["family"] != "recorte_picos"],
        "30min": pivote[pivote["duration_min"] == 30], "60min": pivote[pivote["duration_min"] == 60],
        "120min": pivote[pivote["duration_min"] == 120], "240min_o_mas": pivote[pivote["duration_min"] >= 240],
    }
    filas = []
    for nombre_sub, sub in subconjuntos.items():
        det_120, det_360 = sub[120].to_numpy(), sub[360].to_numpy()
        solo_120, solo_360 = int((det_120 & ~det_360).sum()), int((~det_120 & det_360).sum())
        p = mcnemar_p(solo_120, solo_360)
        ci_low, ci_high = bootstrap_pareado(det_120.astype(float), det_360.astype(float))
        filas.append({
            "subconjunto": nombre_sub, "n_episodios": len(sub),
            "detectado_solo_con_120": solo_120, "detectado_solo_con_360": solo_360,
            "detectado_con_ambos": int((det_120 & det_360).sum()), "detectado_con_ninguno": int((~det_120 & ~det_360).sum()),
            "DR_120_pct": 100 * det_120.mean() if len(sub) else float("nan"), "DR_360_pct": 100 * det_360.mean() if len(sub) else float("nan"),
            "mcnemar_p_valor": p, "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
            "diferencia_significativa_p<0.05": p < 0.05,
        })
    return pd.DataFrame(filas)


def _skewness(x: np.ndarray) -> float:
    m, s = x.mean(), x.std()
    return float(np.mean((x - m) ** 3) / s ** 3) if s else 0.0


def main() -> dict:
    config_360, datos_360, modelo_360, ataques, master_csv = cargar_ataques_y_modelo_360()
    print(f"FASE C -- {len(ataques)} episodios, score={SCORE}, ventanas={TAMANOS}\n")

    config_120 = construir_config_120(config_360)
    datos_120 = preparar_datos(config_120)
    ruta_ckpt_120 = BASE_DIR / "models" / "tcn_ae_ventana120.pt"
    if not ruta_ckpt_120.exists():
        raise RuntimeError(
            "No existe el checkpoint de ventana=120 (models/tcn_ae_ventana120.pt). Segun el "
            "encargo, si hiciera falta reentrenar hay que detenerse y documentarlo ANTES de "
            "hacerlo -- no se reentrena automaticamente aqui."
        )
    modelo_120, meta_120 = entrenar_o_cargar_120(config_120, datos_120)  # ya existe -> solo carga
    print(f"Checkpoint de 120 reutilizado sin reentrenar ({ruta_ckpt_120.name}).\n")

    particion_val_raw = datos_360["particiones"]["val"][COLUMNA_OBJETIVO].to_numpy().copy()
    n_dias = len(particion_val_raw) / 1440.0

    datos_por_ventana = {360: datos_360, 120: datos_120}
    modelos_por_ventana = {360: modelo_360, 120: modelo_120}

    filas_umbrales, resultados_por_ventana_objetivo, tablas_por_ventana = [], [], {}
    for ventana in TAMANOS:
        print(f"=== ventana={ventana} min ===")
        grid = datos_por_ventana[ventana]["ventanas"]["val"]
        params_norm = datos_por_ventana[ventana]["params_norm"]
        tabla, scores_limpios = puntuar_ventana_relative(ataques, particion_val_raw, grid, modelos_por_ventana[ventana],
                                                           params_norm, ventana)
        tablas_por_ventana[ventana] = tabla
        for objetivo_nombre, objetivo_valor in OBJETIVOS_EVENTOS_DIA.items():
            u, tasa_eventos, tasa_ventanas = calibrar_umbral_por_eventos(scores_limpios, n_dias, objetivo_valor)
            filas_umbrales.append({"ventana_min": ventana, "objetivo_nombre": objetivo_nombre, "objetivo_valor": objetivo_valor,
                                    "threshold": u, "tasa_eventos_dia_real": tasa_eventos, "tasa_ventanas_dia_real": tasa_ventanas})
            if objetivo_nombre == OBJETIVO_PRINCIPAL:
                print(f"  {objetivo_nombre}: umbral={u:.5f}, eventos/dia={tasa_eventos:.4f}, ventanas/dia={tasa_ventanas:.3f}")
            r = agregar_por_umbral(tabla, master_csv, u, ventana)
            r["objetivo_nombre"], r["objetivo_valor"] = objetivo_nombre, objetivo_valor
            resultados_por_ventana_objetivo.append(r)
        print()

    umbrales_tabla = _guardar_tabla(pd.DataFrame(filas_umbrales), "01_umbrales")
    resultados_todo = pd.concat(resultados_por_ventana_objetivo, ignore_index=True)
    _guardar_tabla(resultados_todo, "01_episode_detection_results")

    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    filas = []
    for ventana, grupo in principal.groupby("ventana_min"):
        vg = tablas_por_ventana[ventana].assign(score_delta=lambda d: d["attacked_score"] - d["clean_score"])
        for etiqueta, g in (("todas", grupo), ("sin_recorte_picos", grupo[grupo["family"] != "recorte_picos"])):
            fila = {"ventana_min": ventana, "familias": etiqueta}
            fila.update(_resumen(g, vg[vg["episode_id"].isin(g["episode_id"])])); filas.append(fila)
    tabla_global = _guardar_tabla(pd.DataFrame(filas), "02_global_con_sin_recorte_picos")
    print("Global por ventana:")
    print(tabla_global.to_string(index=False))

    filas = []
    for (ventana, dur), grupo in principal.groupby(["ventana_min", "duration_min"]):
        vg = tablas_por_ventana[ventana]
        vg = vg[vg["episode_id"].isin(grupo["episode_id"])].assign(score_delta=lambda d: d["attacked_score"] - d["clean_score"])
        fila = {"ventana_min": ventana, "duration_min": dur}
        fila.update(_resumen(grupo, vg)); filas.append(fila)
    tabla_duracion = _guardar_tabla(pd.DataFrame(filas), "02_por_duracion")
    print("\nPor duracion:")
    print(tabla_duracion.to_string(index=False))

    filas = []
    for (ventana, fam), grupo in principal.groupby(["ventana_min", "family"]):
        fila = {"ventana_min": ventana, "family": fam}
        fila.update(_resumen(grupo)); filas.append(fila)
    tabla_familia = _guardar_tabla(pd.DataFrame(filas), "02_por_familia")

    pareada = comparacion_pareada(resultados_todo)
    _guardar_tabla(pareada, "03_comparacion_pareada")
    print("\nComparacion pareada 120 vs 360:")
    print(pareada.to_string(index=False))

    # estabilidad
    filas = []
    for ventana in TAMANOS:
        grid = datos_por_ventana[ventana]["ventanas"]["val"]
        params_norm = datos_por_ventana[ventana]["params_norm"]
        X_norm = aplicar_zscore(grid["X"], params_norm)
        X_hat = modelos_por_ventana[ventana].reconstruir(X_norm)
        s = calcular_relative_kw(X_norm, X_hat, params_norm)
        u_principal = umbrales_tabla[(umbrales_tabla["ventana_min"] == ventana) & (umbrales_tabla["objetivo_nombre"] == OBJETIVO_PRINCIPAL)].iloc[0]
        duraciones = duraciones_eventos_min(s > u_principal["threshold"], ventana)
        filas.append({"ventana_min": ventana, "media": s.mean(), "std": s.std(), "mediana": np.median(s),
                       "p95": np.percentile(s, 95), "p99": np.percentile(s, 99), "asimetria": _skewness(s),
                       "duracion_media_racha_alarma_min": float(duraciones.mean())})
    estabilidad = _guardar_tabla(pd.DataFrame(filas), "04_estabilidad")
    print("\nEstabilidad limpia (relative_kw):")
    print(estabilidad.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for ventana in TAMANOS:
        sub = tabla_duracion[tabla_duracion["ventana_min"] == ventana].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", color=COLOR_VENTANA[ventana], linewidth=1.8, label=f"ventana={ventana}")
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)")
    ax.set_title(f"Fase C: induced_DR por duracion y ventana ({SCORE}, 0.06 eventos/dia)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "01_induced_dr_por_duracion_ventana")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ventana in TAMANOS:
        grid = datos_por_ventana[ventana]["ventanas"]["val"]
        params_norm = datos_por_ventana[ventana]["params_norm"]
        X_norm = aplicar_zscore(grid["X"], params_norm)
        X_hat = modelos_por_ventana[ventana].reconstruir(X_norm)
        s = calcular_relative_kw(X_norm, X_hat, params_norm)
        ax.hist(s, bins=60, color=COLOR_VENTANA[ventana], alpha=0.5, density=True, label=f"ventana={ventana}")
    ax.set_xlabel("relative_kw limpio"); ax.set_ylabel("densidad")
    ax.set_title("Fase C: distribucion de relative_kw limpio")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "02_distribucion_scores_limpios")

    print(f"\nTablas en {EXP_TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {EXP_FIGURAS_DIR.relative_to(BASE_DIR)}/")
    return {"umbrales_tabla": umbrales_tabla, "resultados_todo": resultados_todo, "tabla_global": tabla_global,
            "tabla_duracion": tabla_duracion, "tabla_familia": tabla_familia, "pareada": pareada,
            "estabilidad": estabilidad, "n_dias": n_dias}


if __name__ == "__main__":
    main()
