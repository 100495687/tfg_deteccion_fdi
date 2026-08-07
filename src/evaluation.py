"""Evaluacion por-ataque. Para cada familia+parametro+duracion se generan episodios
independientes: una ventana de test al azar, con el ataque inyectado en un tramo interno.
El TCN-AE no tiene memoria entre ventanas, asi que no hace falta que los episodios no se
solapen en el tiempo (a diferencia de detectores con estado de fases previas del TFG).

`generar_episodios_scores` calcula el score de los 588 episodios una vez -- es la parte cara.
`agregar_dr` aplica un umbral ya calculado y agrega a DR%, sin recalcular nada, así que se
pueden barrer muchos umbrales barato (ver `barrer_umbrales_dr_fa`).

FA = % de las 346 ventanas de test limpias (sin ataques) que superan el umbral.

Se puede correr suelto:
    python -m src.evaluation
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.anomaly_score import calcular_scores
from src.attacks import bypass_residual, bypass_total, reduccion_constante, reduccion_variable, recorte_picos
from src.data_loading import COLUMNA_OBJETIVO
from src.normalization import aplicar_zscore
from src.pipeline import obtener_modelo, preparar_datos
from src.thresholding import calcular_umbral
from src.windowing import cargar_config

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables"
FIGURAS_DIR = BASE_DIR / "results" / "figures"

GRIS = "#8a8a86"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})


def _combos_ataque(config: dict) -> list[dict]:
    """Aplana configs/base.yaml:ataques a una lista de combos (familia + sus parametros).
    14 combos con la config base: 7 rho de reduccion_constante, 1 reduccion_variable,
    1 bypass_total, 2 rho de bypass_residual, 3 cuantiles de recorte_picos."""
    cfg = config["ataques"]
    combos = []
    for rho in cfg["reduccion_constante"]["rho"]:
        combos.append({"familia": "reduccion_constante", "parametro": f"rho={rho}", "rho": rho})
    combos.append({
        "familia": "reduccion_variable",
        "parametro": f"rho_min={cfg['reduccion_variable']['rho_min']},rho_max={cfg['reduccion_variable']['rho_max']}",
        "rho_min": cfg["reduccion_variable"]["rho_min"], "rho_max": cfg["reduccion_variable"]["rho_max"],
    })
    combos.append({"familia": "bypass_total", "parametro": "rho=0.0"})
    for rho in cfg["bypass_residual"]["rho"]:
        combos.append({"familia": "bypass_residual", "parametro": f"rho={rho}", "rho": rho})
    for cuantil in cfg["recorte_picos"]["cuantil_umbral"]:
        combos.append({"familia": "recorte_picos", "parametro": f"cuantil={cuantil}", "cuantil": cuantil})
    return combos


def _aplicar_ataque(combo: dict, x: np.ndarray, umbral_picos_kw: dict, rng: np.random.Generator) -> np.ndarray:
    familia = combo["familia"]
    if familia == "reduccion_constante":
        return reduccion_constante(x, combo["rho"])
    if familia == "reduccion_variable":
        return reduccion_variable(x, combo["rho_min"], combo["rho_max"], rng)
    if familia == "bypass_total":
        return bypass_total(x)
    if familia == "bypass_residual":
        return bypass_residual(x, combo["rho"])
    if familia == "recorte_picos":
        return recorte_picos(x, umbral_picos_kw[combo["cuantil"]])
    raise ValueError(f"familia no reconocida: {familia}")


def generar_episodios_scores(config: dict, datos: dict, modelo, tipo_score: str, seed: int) -> pd.DataFrame:
    """Genera los episodios y calcula su score, SIN aplicar ningun umbral. Una fila por
    episodio individual (no agregado): familia, parametro, duracion_min, posicion_idx, score."""
    ventanas_test_raw = datos["ventanas"]["test"]["X"]   # (346, tamano_ventana), kW crudos, sin atacar
    tamano_ventana = config["windowing"]["tamano_ventana_min"]
    params_norm = datos["params_norm"]
    rng = np.random.default_rng(seed)

    valores_train = datos["particiones"]["train"][COLUMNA_OBJETIVO].to_numpy()
    umbral_picos_kw = {
        c: float(np.quantile(valores_train, c)) for c in config["ataques"]["recorte_picos"]["cuantil_umbral"]
    }

    duraciones = [d for d in config["ataques"]["duraciones_min"] if d <= tamano_ventana]
    if len(duraciones) < len(config["ataques"]["duraciones_min"]):
        print(f"[aviso] se descartan duraciones > tamano_ventana ({tamano_ventana} min)")

    combos = _combos_ataque(config)
    n_posiciones = config["ataques"]["n_posiciones_por_duracion"]

    filas = []
    for combo in combos:
        for duracion in duraciones:
            for posicion_idx in range(n_posiciones):
                idx_ventana = rng.integers(0, ventanas_test_raw.shape[0])
                offset = rng.integers(0, tamano_ventana - duracion + 1)

                ventana_atacada = ventanas_test_raw[idx_ventana].copy()
                x_tramo = ventana_atacada[offset:offset + duracion]
                ventana_atacada[offset:offset + duracion] = _aplicar_ataque(combo, x_tramo, umbral_picos_kw, rng)

                X_norm = aplicar_zscore(ventana_atacada[None, :], params_norm)
                score = calcular_scores(modelo, X_norm, tipo_score)[0]

                filas.append({
                    "familia": combo["familia"], "parametro": combo["parametro"],
                    "duracion_min": duracion, "posicion_idx": posicion_idx, "score": score,
                })

    return pd.DataFrame(filas)


def agregar_dr(df_episodios: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """DR% por (familia, parametro, duracion) al umbral dado. Barato: no reconstruye nada,
    solo compara scores ya calculados."""
    df = df_episodios.copy()
    df["detectado"] = df["score"] > umbral
    return (
        df.groupby(["familia", "parametro", "duracion_min"], as_index=False)
        .agg(n_episodios=("detectado", "size"), n_detectados=("detectado", "sum"))
        .assign(DR_pct=lambda d: 100 * d["n_detectados"] / d["n_episodios"])
    )


def calcular_fa(scores_test_limpio: np.ndarray, umbral: float) -> float:
    return float(100 * (scores_test_limpio > umbral).mean())


def barrer_umbrales_dr_fa(scores_train: np.ndarray, scores_test_limpio: np.ndarray,
                           df_episodios: pd.DataFrame, percentiles: list[float]) -> pd.DataFrame:
    """Prueba varios percentiles de scores_train como umbral candidato: FA global y DR global
    (todos los episodios juntos, sin desglosar por familia) para cada uno -- util para elegir
    un punto de operacion en vez de quedarse solo con percentil 99 / 3 sigma."""
    filas = []
    for p in percentiles:
        umbral = float(np.percentile(scores_train, p))
        fa = calcular_fa(scores_test_limpio, umbral)
        dr = float(100 * (df_episodios["score"] > umbral).mean())
        filas.append({"percentil": p, "umbral": umbral, "FA_pct": fa, "DR_pct": dr})
    return pd.DataFrame(filas)


def graficar_dr_vs_rho(tabla: pd.DataFrame, ruta_sin_extension: Path) -> None:
    sub = tabla[tabla["familia"] == "reduccion_constante"].copy()
    sub["rho"] = sub["parametro"].str.extract(r"rho=([\d.]+)").astype(float)
    duraciones = sorted(sub["duracion_min"].unique())
    colores = plt.cm.Blues(np.linspace(0.35, 0.95, len(duraciones)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for color, duracion in zip(colores, duraciones):
        s = sub[sub["duracion_min"] == duracion].sort_values("rho")
        ax.plot(s["rho"], s["DR_pct"], marker="o", markersize=4, linewidth=1.8, color=color,
                label=f"{duracion} min")

    ax.set_xlabel("ρ (fraccion de consumo reportado)")
    ax.set_ylabel("DR (%)")
    ax.set_ylim(-3, 103)
    ax.legend(title="duracion", fontsize=8, title_fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(f"{ruta_sin_extension}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{ruta_sin_extension}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    config = cargar_config()
    datos = preparar_datos(config)
    modelo = obtener_modelo(config, datos["ventanas_norm"])

    tipo_score = config["score"]["tipo"]
    scores_train = calcular_scores(modelo, datos["ventanas_norm"]["train"], tipo_score)
    scores_test_limpio = calcular_scores(modelo, datos["ventanas_norm"]["test"], tipo_score)

    estrategia = config["umbral"]["estrategia"]
    umbral = calcular_umbral(scores_train, estrategia, config["umbral"].get("percentil", 99))
    fa = calcular_fa(scores_test_limpio, umbral)

    print(f"Umbral ({estrategia}): {umbral:.4f}, FA sobre test limpio: {fa:.2f}%\n")

    df_episodios = generar_episodios_scores(config, datos, modelo, tipo_score, seed=config["seed"])
    tabla = agregar_dr(df_episodios, umbral)
    tabla["FA_pct"] = fa

    print("DR media por familia:")
    print(tabla.groupby("familia")["DR_pct"].mean().round(1))

    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_tabla = TABLAS_DIR / "tabla_dr_fa_por_ataque.csv"
    tabla.to_csv(ruta_tabla, index=False)
    print(f"\nTabla guardada en {ruta_tabla}")

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_fig = FIGURAS_DIR / "curva_dr_vs_rho"
    graficar_dr_vs_rho(tabla, ruta_fig)
    print(f"Figura guardada en {ruta_fig}.png / .pdf")

    assert (tabla["DR_pct"] >= 0).all() and (tabla["DR_pct"] <= 100).all()
    print(f"\n{len(tabla)} filas (familia x parametro x duracion), "
          f"{len(df_episodios)} episodios evaluados en total.")
