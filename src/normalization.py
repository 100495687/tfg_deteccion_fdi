"""Normalización z-score de las ventanas del dataset.

La media y la desviación estándar se calculan únicamente con los datos de
entrenamiento. Para el ajuste se utiliza la serie original, antes de construir
las ventanas, de manera que el solapamiento de train no dé más peso a unos
tramos que a otros.

Los mismos parámetros se aplican después a las ventanas de train, validación
y test. Los datos de validación y test no intervienen en el ajuste.

Ejecutable:  python -m src.normalization
"""

from __future__ import annotations

import numpy as np

from src.data_loading import COLUMNA_OBJETIVO, cargar_y_limpiar
from src.splitting import split_temporal
from src.windowing import cargar_config, construir_ventanas_por_particion


def ajustar_zscore(valores_train: np.ndarray) -> dict:
    """Calcula la media y la desviación estándar de la serie de entrenamiento.

    Se espera recibir la serie antes del ventaneo para evitar que el
    solapamiento de las ventanas afecte a estos valores.
    """
    media = float(np.mean(valores_train))
    std = float(np.std(valores_train))
    if std == 0:
        raise ValueError("desviacion estandar del train es 0: no se puede normalizar")
    return {"media": media, "std": std}


def aplicar_zscore(X: np.ndarray, params: dict) -> np.ndarray:
    return (X - params["media"]) / params["std"]


def revertir_zscore(X_norm: np.ndarray, params: dict) -> np.ndarray:
    return X_norm * params["std"] + params["media"]


if __name__ == "__main__":
    config = cargar_config()
    df = cargar_y_limpiar()
    particiones = split_temporal(df)
    ventanas = construir_ventanas_por_particion(particiones, config)

    params = ajustar_zscore(particiones["train"][COLUMNA_OBJETIVO].to_numpy())
    print(f"Normalizador (ajustado SOLO en train): media={params['media']:.4f} kW, "
          f"std={params['std']:.4f} kW\n")

    ventanas_norm = {}
    for nombre, v in ventanas.items():
        X_norm = aplicar_zscore(v["X"], params)
        ventanas_norm[nombre] = X_norm
        print(f"{nombre}: media={X_norm.mean():+.4f}, std={X_norm.std():.4f}, "
              f"min={X_norm.min():+.4f}, max={X_norm.max():+.4f}")
        assert np.isfinite(X_norm).all(), f"valores no finitos tras normalizar {nombre}"

    # La serie original de train debe quedar centrada en 0 y con desviación 1.
    # En las ventanas de train el resultado puede variar ligeramente por el solapamiento.
    train_crudo_norm = aplicar_zscore(particiones["train"][COLUMNA_OBJETIVO].to_numpy(), params)
    assert abs(train_crudo_norm.mean()) < 1e-6
    assert abs(train_crudo_norm.std() - 1.0) < 1e-6

    # Al deshacer la normalización se deben recuperar los valores originales
    recuperado = revertir_zscore(ventanas_norm["val"], params)
    assert np.allclose(recuperado, ventanas["val"]["X"], atol=1e-4)

    print("\nTodo correcto: z-score ajustado solo en train, aplicado de forma consistente, round-trip ok.")