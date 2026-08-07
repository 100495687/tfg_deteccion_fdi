"""FASE A -- confirmacion formal del nuevo score principal.

Compara, bajo condiciones identicas (ventana=360, stride=360, checkpoint existente, mismos
episodios de episodes_master_val.csv, continuous_timeline):

  - signed_historical: la definicion exacta usada en los experimentos anteriores
    (anomaly_score.calcular_scores(..., "deficit"), con signo, sin recortar) -- verificado
    por assert, no solo documentado.
  - relative_kw: el score relativo en kW validado en el experimento de scores, reutilizado
    tal cual desde experimento_scores.calcular_todos_los_scores sin duplicar la formula.

Ambos se derivan de las mismas reconstrucciones (una llamada al modelo para toda la
rejilla limpia, otra para todas las ventanas atacadas).

Expone `cargar_ataques_y_modelo_360()` y `calcular_relative_kw(...)`, reutilizados por las
Fases B (stride_relative.py) y C (ventana_relative.py) para garantizar los mismos episode_id
y la misma senal atacada en las tres fases.

Ejecutable de forma aislada:
    python -m src.base_relative
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.anomaly_score import calcular_scores
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import wilson_ci
from src.experimento_scores import calcular_todos_los_scores, duraciones_eventos_min
from src.experimento_stride import (
    bootstrap_pareado,
    calibrar_umbral_por_eventos,
    cargar_datos_y_modelo,
    contar_eventos,
    mcnemar_p,
    reconstruir_ataques_exactos,
)
from src.experimento_ventana120 import _ventana_atacada, _ventanas_disjuntas_afectadas
from src.normalization import aplicar_zscore

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables"
EXP_TABLAS_DIR = TABLAS_DIR / "base_relative"
EXP_FIGURAS_DIR = BASE_DIR / "results" / "figures" / "base_relative"

GRIS = "#8a8a86"
AZUL = "#2a78d6"
VERDE = "#2f9e44"
GRIS_OSCURO = "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
OBJETIVOS_EVENTOS_DIA = {"principal_0.06": 0.06, "sec_0.05": 0.05, "sec_0.10": 0.10, "sec_0.25": 0.25}
OBJETIVO_PRINCIPAL = "principal_0.06"
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
NOMBRES_SCORE = ["signed_historical", "relative_kw"]
COLOR_SCORE = {"signed_historical": GRIS_OSCURO, "relative_kw": VERDE}


def _guardar_fig(fig, nombre):
    EXP_FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EXP_FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    EXP_TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP_TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ----------------------------------------------------------------------------------------
# Setup canonico compartido por las Fases A, B y C
# ----------------------------------------------------------------------------------------

def cargar_ataques_y_modelo_360():
    config, datos, modelo = cargar_datos_y_modelo()           # checkpoint EXISTENTE, no reentrena
    ataques, master_csv = reconstruir_ataques_exactos(config, datos)   # episodios EXACTOS, validados
    return config, datos, modelo, ataques, master_csv


def calcular_relative_kw(X_norm: np.ndarray, X_hat_norm: np.ndarray, params_norm: dict) -> np.ndarray:
    """Reutiliza la formula de 'relative' del experimento de scores (kW, epsilon=0.1) sin
    duplicarla. No opera nunca sobre valores normalizados."""
    return calcular_todos_los_scores(X_norm, X_hat_norm, params_norm)["relative"]


def calcular_scores_fase_a(X_norm: np.ndarray, X_hat_norm: np.ndarray, params_norm: dict) -> dict[str, np.ndarray]:
    signed_historical = (X_hat_norm - X_norm).mean(axis=1)
    relative_kw = calcular_relative_kw(X_norm, X_hat_norm, params_norm)
    return {"signed_historical": signed_historical, "relative_kw": relative_kw}


# ----------------------------------------------------------------------------------------
# Puntuacion (2 llamadas al modelo en total: rejilla limpia + todas las ventanas atacadas)
# ----------------------------------------------------------------------------------------

def puntuar_dos_scores(ataques: dict, particion_val_raw: np.ndarray, grid: dict, modelo,
                        params_norm: dict) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    n_ventanas_grid = grid["X"].shape[0]
    X_clean_norm = aplicar_zscore(grid["X"], params_norm)
    X_hat_clean = modelo.reconstruir(X_clean_norm)
    scores_limpios = calcular_scores_fase_a(X_clean_norm, X_hat_clean, params_norm)

    filas_meta, ventanas_atacadas = [], []
    for episode_id, ataque in ataques.items():
        ks = _ventanas_disjuntas_afectadas(ataque["offset_global"], ataque["duracion"], TAMANO_VENTANA, n_ventanas_grid)
        for k in ks:
            w_start = k * TAMANO_VENTANA
            va = _ventana_atacada(particion_val_raw, w_start, ataque["offset_global"], ataque["duracion"],
                                   ataque["y_tramo"], TAMANO_VENTANA)
            ventanas_atacadas.append(va)
            filas_meta.append({"episode_id": episode_id, "k": k, "window_start": grid["inicio"][k], "window_end": grid["fin"][k]})

    X_atacadas_norm = aplicar_zscore(np.stack(ventanas_atacadas).astype(np.float32), params_norm)
    X_hat_atacadas = modelo.reconstruir(X_atacadas_norm)
    scores_atacados = calcular_scores_fase_a(X_atacadas_norm, X_hat_atacadas, params_norm)

    meta = pd.DataFrame(filas_meta)
    ks_arr = meta["k"].to_numpy()
    tablas = {}
    for nombre in NOMBRES_SCORE:
        t = meta.copy()
        t["clean_score"] = scores_limpios[nombre][ks_arr]
        t["attacked_score"] = scores_atacados[nombre]
        tablas[nombre] = t
    return tablas, scores_limpios


# ----------------------------------------------------------------------------------------
# Agregacion por umbral -- episodio
# ----------------------------------------------------------------------------------------

def agregar_por_umbral(tabla_ventanas: pd.DataFrame, master_csv: pd.DataFrame, umbral: float, nombre_score: str) -> pd.DataFrame:
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
    resultado["score_type"] = nombre_score
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
    }


def comparacion_pareada(resultados_todo: pd.DataFrame) -> pd.DataFrame:
    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    pivote = principal.pivot(index="episode_id", columns="score_type", values="induced_detected").astype(bool)
    meta = principal.drop_duplicates("episode_id").set_index("episode_id")[["family", "duration_min"]]
    pivote = pivote.join(meta)

    subconjuntos = {"todos": pivote, "sin_recorte_picos": pivote[pivote["family"] != "recorte_picos"]}
    for dur in DURACIONES:
        subconjuntos[f"{dur}min"] = pivote[pivote["duration_min"] == dur]

    filas = []
    for nombre_sub, sub in subconjuntos.items():
        det_rel, det_hist = sub["relative_kw"].to_numpy(), sub["signed_historical"].to_numpy()
        solo_rel = int((det_rel & ~det_hist).sum())
        solo_hist = int((~det_rel & det_hist).sum())
        p = mcnemar_p(solo_rel, solo_hist)
        ci_low, ci_high = bootstrap_pareado(det_rel.astype(float), det_hist.astype(float))
        filas.append({
            "subconjunto": nombre_sub, "n_episodios": len(sub),
            "detectado_solo_relative": solo_rel, "detectado_solo_historico": solo_hist,
            "detectado_con_ambos": int((det_rel & det_hist).sum()), "detectado_con_ninguno": int((~det_rel & ~det_hist).sum()),
            "DR_relative_pct": 100 * det_rel.mean() if len(sub) else float("nan"),
            "DR_historico_pct": 100 * det_hist.mean() if len(sub) else float("nan"),
            "mcnemar_p_valor": p, "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
            "diferencia_significativa_p<0.05": p < 0.05,
        })
    return pd.DataFrame(filas)


def _skewness(x: np.ndarray) -> float:
    m, s = x.mean(), x.std()
    return float(np.mean((x - m) ** 3) / s ** 3) if s else 0.0


def analizar_estabilidad(scores_limpios: dict[str, np.ndarray], grid: dict, umbrales_tabla: pd.DataFrame) -> pd.DataFrame:
    horas = pd.to_datetime(grid["inicio"]).hour
    consumo_medio = grid["X"].mean(axis=1)
    terciles = pd.qcut(consumo_medio, 3, labels=["bajo", "medio", "alto"])
    filas = []
    for nombre in NOMBRES_SCORE:
        s = scores_limpios[nombre]
        std_hora = pd.Series(s).groupby(horas).std()
        std_tercil = pd.Series(s).groupby(terciles, observed=True).std()
        u = umbrales_tabla[(umbrales_tabla["score_type"] == nombre) & (umbrales_tabla["objetivo_nombre"] == OBJETIVO_PRINCIPAL)].iloc[0]
        duraciones = duraciones_eventos_min(s > u["threshold"], TAMANO_VENTANA)
        filas.append({
            "score_type": nombre, "media": s.mean(), "std": s.std(), "mediana": np.median(s),
            "p90": np.percentile(s, 90), "p95": np.percentile(s, 95), "p99": np.percentile(s, 99),
            "asimetria": _skewness(s), "std_por_hora_media": float(std_hora.mean()),
            "std_tercil_bajo": float(std_tercil.get("bajo", np.nan)), "std_tercil_alto": float(std_tercil.get("alto", np.nan)),
            "duracion_media_racha_alarma_min": float(duraciones.mean()),
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------------------
# Orquestacion
# ----------------------------------------------------------------------------------------

def main() -> dict:
    config, datos, modelo, ataques, master_csv = cargar_ataques_y_modelo_360()
    params_norm = datos["params_norm"]
    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    n_dias = len(particion_val_raw) / 1440.0
    grid = datos["ventanas"]["val"]

    print(f"FASE A -- {len(ataques)} episodios, ventana=360/stride=360, checkpoint existente.\n")
    tablas, scores_limpios = puntuar_dos_scores(ataques, particion_val_raw, grid, modelo, params_norm)

    # verificacion explicita: signed_historical == calcular_scores(modelo, X, "deficit")
    deficit_original = calcular_scores(modelo, aplicar_zscore(grid["X"], params_norm), "deficit")
    coincide = np.allclose(scores_limpios["signed_historical"], deficit_original, atol=1e-9)
    print(f"[verificacion] signed_historical == anomaly_score.calcular_scores(...,'deficit'): {coincide}\n")
    assert coincide, "signed_historical NO reproduce exactamente el score historico 'deficit'"

    filas_umbrales = []
    resultados_por_score_objetivo = []
    for nombre in NOMBRES_SCORE:
        for objetivo_nombre, objetivo_valor in OBJETIVOS_EVENTOS_DIA.items():
            u, tasa_eventos, tasa_ventanas = calibrar_umbral_por_eventos(scores_limpios[nombre], n_dias, objetivo_valor)
            filas_umbrales.append({"score_type": nombre, "objetivo_nombre": objetivo_nombre, "objetivo_valor": objetivo_valor,
                                    "threshold": u, "tasa_eventos_dia_real": tasa_eventos, "tasa_ventanas_dia_real": tasa_ventanas})
            if objetivo_nombre == OBJETIVO_PRINCIPAL:
                print(f"  {nombre}: umbral={u:.5f}, eventos/dia={tasa_eventos:.4f}, ventanas/dia={tasa_ventanas:.3f}")
            r = agregar_por_umbral(tablas[nombre], master_csv, u, nombre)
            r["objetivo_nombre"], r["objetivo_valor"] = objetivo_nombre, objetivo_valor
            resultados_por_score_objetivo.append(r)

    umbrales_tabla = _guardar_tabla(pd.DataFrame(filas_umbrales), "01_umbrales")
    resultados_todo = pd.concat(resultados_por_score_objetivo, ignore_index=True)
    _guardar_tabla(resultados_todo, "01_episode_detection_results")

    principal = resultados_todo[resultados_todo["objetivo_nombre"] == OBJETIVO_PRINCIPAL]
    filas = []
    for nombre, grupo in principal.groupby("score_type"):
        fila = {"score_type": nombre, "familias": "todas"}
        fila.update(_resumen(grupo)); filas.append(fila)
        sin_picos = grupo[grupo["family"] != "recorte_picos"]
        fila2 = {"score_type": nombre, "familias": "sin_recorte_picos"}
        fila2.update(_resumen(sin_picos)); filas.append(fila2)
    tabla_global = _guardar_tabla(pd.DataFrame(filas), "02_global_con_sin_recorte_picos")
    print("\nGlobal (con/sin recorte_picos):")
    print(tabla_global.to_string(index=False))

    filas = []
    for (nombre, dur), grupo in principal.groupby(["score_type", "duration_min"]):
        fila = {"score_type": nombre, "duration_min": dur}
        fila.update(_resumen(grupo)); filas.append(fila)
    tabla_duracion = _guardar_tabla(pd.DataFrame(filas), "02_por_duracion")
    print("\nPor duracion:")
    print(tabla_duracion.to_string(index=False))

    filas = []
    for (nombre, fam), grupo in principal.groupby(["score_type", "family"]):
        fila = {"score_type": nombre, "family": fam}
        fila.update(_resumen(grupo)); filas.append(fila)
    tabla_familia = _guardar_tabla(pd.DataFrame(filas), "02_por_familia")
    print("\nPor familia:")
    print(tabla_familia.to_string(index=False))

    pareada = comparacion_pareada(resultados_todo)
    _guardar_tabla(pareada, "03_comparacion_pareada")
    print("\nComparacion pareada relative_kw vs signed_historical:")
    print(pareada.to_string(index=False))

    estabilidad = analizar_estabilidad(scores_limpios, grid, umbrales_tabla)
    _guardar_tabla(estabilidad, "04_estabilidad")
    print("\nEstabilidad limpia:")
    print(estabilidad.to_string(index=False))

    # figuras
    fig, ax = plt.subplots(figsize=(8, 5))
    for nombre in NOMBRES_SCORE:
        sub = tabla_duracion[tabla_duracion["score_type"] == nombre].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", color=COLOR_SCORE[nombre], linewidth=1.8, label=nombre)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)")
    ax.set_title("Fase A: relative_kw vs signed_historical, por duracion (0.06 eventos/dia)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "01_induced_dr_por_duracion")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nombre in NOMBRES_SCORE:
        ax.hist(scores_limpios[nombre], bins=60, color=COLOR_SCORE[nombre], alpha=0.5, density=True, label=nombre)
    ax.set_xlabel("score limpio"); ax.set_ylabel("densidad"); ax.set_title("Distribucion de scores limpios (val)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "02_distribucion_scores_limpios")

    # ---- decision formal ----
    dr_rel = tabla_global[(tabla_global["score_type"] == "relative_kw") & (tabla_global["familias"] == "sin_recorte_picos")]["induced_DR_pct"].iloc[0]
    dr_hist = tabla_global[(tabla_global["score_type"] == "signed_historical") & (tabla_global["familias"] == "sin_recorte_picos")]["induced_DR_pct"].iloc[0]
    p_global = pareada[pareada["subconjunto"] == "sin_recorte_picos"]["mcnemar_p_valor"].iloc[0]
    peores = pareada[(pareada["DR_relative_pct"] < pareada["DR_historico_pct"]) & (pareada["diferencia_significativa_p<0.05"])]

    declarar_relative = bool(dr_rel > dr_hist and p_global < 0.05 and len(peores) == 0)
    print(f"\n¿relative_kw supera significativamente y sin empeorar ningun subconjunto? {declarar_relative}")
    print(f"  DR sin_recorte_picos: relative={dr_rel:.2f}% vs historico={dr_hist:.2f}% (p={p_global:.2e})")
    if len(peores):
        print(f"  subconjuntos donde relative_kw es SIGNIFICATIVAMENTE peor: {peores['subconjunto'].tolist()}")

    if declarar_relative:
        config_declaracion = {
            "score_base": "relative_kw",
            "descripcion": "Declarado formalmente en la Fase A del experimento de consolidacion.",
            "condiciones_de_la_comparacion": {
                "window_size_min": 360, "stride_min": 360, "checkpoint": "models/tcn_ae_ventana360.pt",
                "objetivo_falsos_eventos_dia": 0.06,
            },
            "resultado": {
                "induced_DR_sin_recorte_picos_relative_kw_pct": round(float(dr_rel), 2),
                "induced_DR_sin_recorte_picos_signed_historical_pct": round(float(dr_hist), 2),
                "mcnemar_p_valor_sin_recorte_picos": float(p_global),
            },
            "formula": {
                "relative_kw": "mean( max(reconstruccion_kw - observado_kw, 0) / max(|reconstruccion_kw|, 0.1) )",
                "epsilon_kw": 0.1,
            },
            "nota": "Esta decision NO fija todavia stride ni ventana -- eso se decide en las Fases B y C.",
        }
        ruta_yaml = TABLAS_DIR / "base_relative_config.yaml"
        with open(ruta_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_declaracion, f, allow_unicode=True, sort_keys=False)
        print(f"\nscore_base = relative_kw declarado formalmente en {ruta_yaml.relative_to(BASE_DIR)}")
    else:
        print("\nNO se declara relative_kw como score_base (no cumple el criterio) -- revisar manualmente.")

    print(f"\nTablas en {EXP_TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {EXP_FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "config": config, "datos": datos, "modelo": modelo, "ataques": ataques, "master_csv": master_csv,
        "tablas": tablas, "scores_limpios": scores_limpios, "umbrales_tabla": umbrales_tabla,
        "resultados_todo": resultados_todo, "tabla_global": tabla_global, "tabla_duracion": tabla_duracion,
        "tabla_familia": tabla_familia, "pareada": pareada, "estabilidad": estabilidad,
        "declarar_relative": declarar_relative, "n_dias": n_dias,
    }


if __name__ == "__main__":
    main()
