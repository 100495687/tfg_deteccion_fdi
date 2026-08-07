"""Umbral de deteccion, fijado solo con los scores de train (consumo limpio, sin ataques).
Dos estrategias (config: umbral.estrategia): percentil configurable, o media + 3 sigma.

Tambien dibuja un histograma train vs val con las dos marcadas, para decidir a ojo cual usar
antes de tocar test/ataques.

Se puede correr suelto:
    python -m src.thresholding
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURAS_DIR = BASE_DIR / "results" / "figures"

AZUL = "#2a78d6"
NARANJA = "#eb6834"
VERDE = "#2f9e44"
GRIS = "#8a8a86"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})


def calcular_umbral(scores_train: np.ndarray, estrategia: str, percentil: float = 99) -> float:
    if estrategia == "percentil":
        return float(np.percentile(scores_train, percentil))
    if estrategia == "media_3sigma":
        return float(scores_train.mean() + 3 * scores_train.std())
    raise ValueError(f"estrategia de umbral no reconocida: {estrategia}")


def graficar_histograma(scores_train: np.ndarray, scores_val: np.ndarray, umbrales: dict,
                         ruta_sin_extension: Path, tipo_score: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.histogram_bin_edges(np.concatenate([scores_train, scores_val]), bins=60)
    ax.hist(scores_train, bins=bins, color=AZUL, alpha=0.55, label=f"train (n={len(scores_train)})")
    ax.hist(scores_val, bins=bins, color=NARANJA, alpha=0.55, label=f"val (n={len(scores_val)})")

    colores_umbral = {"percentil": VERDE, "media_3sigma": GRIS}
    for nombre, valor in umbrales.items():
        ax.axvline(valor, color=colores_umbral.get(nombre, "black"), linestyle="--", linewidth=1.6,
                   label=f"{nombre} = {valor:.4f}")

    ax.set_xlabel(f"error de reconstruccion ({tipo_score}, z-score)")
    ax.set_ylabel("nº de ventanas")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{ruta_sin_extension}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{ruta_sin_extension}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from src.anomaly_score import calcular_scores
    from src.pipeline import obtener_modelo, preparar_datos
    from src.windowing import cargar_config

    config = cargar_config()
    datos = preparar_datos(config)
    modelo = obtener_modelo(config, datos["ventanas_norm"])

    tipo_score = config["score"]["tipo"]
    scores_train = calcular_scores(modelo, datos["ventanas_norm"]["train"], tipo_score)
    scores_val = calcular_scores(modelo, datos["ventanas_norm"]["val"], tipo_score)

    umbrales = {
        "percentil": calcular_umbral(scores_train, "percentil", config["umbral"]["percentil"]),
        "media_3sigma": calcular_umbral(scores_train, "media_3sigma"),
    }
    print(f"Umbrales calculados solo con scores de train ({tipo_score}):")
    for nombre, valor in umbrales.items():
        fa_train = (scores_train > valor).mean() * 100
        fa_val = (scores_val > valor).mean() * 100
        print(f"  {nombre}: umbral={valor:.4f}, falsos positivos train={fa_train:.2f}%, "
              f"val (solo diagnostico)={fa_val:.2f}%")

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_fig = FIGURAS_DIR / "hist_scores_train_val"
    graficar_histograma(scores_train, scores_val, umbrales, ruta_fig, tipo_score)
    print(f"\nHistograma guardado en {ruta_fig}.png / .pdf")

    assert all(v > 0 for v in umbrales.values())
