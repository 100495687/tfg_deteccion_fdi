"""Construcción de ventanas temporales a partir de cada partición.

La serie ya llega separada en train, validación y test, por lo que ninguna
ventana puede contener datos de dos particiones distintas.

El tamaño de las ventanas y el desplazamiento entre ellas se definen en
configs/base.yaml. En entrenamiento se utiliza solapamiento para disponer de
más muestras. En validación y test se utilizan ventanas independientes para
evitar evaluar varias veces prácticamente el mismo tramo de datos.

Ejecutable:  python -m src.windowing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from src.data_loading import COLUMNA_OBJETIVO, cargar_y_limpiar
from src.splitting import split_temporal

CONFIG_PATH_DEFAULT = "configs/base.yaml"


def cargar_config(ruta: str = CONFIG_PATH_DEFAULT) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_stride(config: dict, particion: str) -> int:
    """Obtiene el desplazamiento entre ventanas para cada partición.

    En train se calcula como una fracción del tamaño de ventana. En validación
    y test se utiliza el valor indicado en la configuración o, si no se ha
    definido, el propio tamaño de la ventana.
    """
    tamano_ventana = config["windowing"]["tamano_ventana_min"]
    if particion == "train":
        stride = max(1, round(tamano_ventana * config["windowing"]["stride_train_fraccion"]))
        return stride
    stride_eval = config["windowing"].get("stride_eval_min")
    return stride_eval if stride_eval is not None else tamano_ventana


def ventanear(df_particion: pd.DataFrame, tamano_ventana: int, stride: int) -> dict:
    """Divide una partición en ventanas de tamaño fijo.

    La cola final se descarta cuando no contiene suficientes muestras para
    formar una ventana completa. Además de los valores de consumo, se guardan
    las fechas de inicio y fin, la proporción de datos imputados y el número
    de muestras descartadas.
    """
    if tamano_ventana <= 0 or stride <= 0:
        raise ValueError("tamano_ventana y stride deben ser positivos")

    valores = df_particion[COLUMNA_OBJETIVO].to_numpy(dtype=np.float32)
    imputado = df_particion["imputado"].to_numpy()
    timestamps = df_particion.index.to_numpy()

    n_muestras = len(valores)
    inicios = list(range(0, n_muestras - tamano_ventana + 1, stride))
    n_descartadas = n_muestras - (inicios[-1] + tamano_ventana if inicios else 0)

    if not inicios:
        return {
            "X": np.empty((0, tamano_ventana), dtype=np.float32),
            "inicio": np.empty(0, dtype="datetime64[ns]"),
            "fin": np.empty(0, dtype="datetime64[ns]"),
            "frac_imputado": np.empty(0, dtype=np.float64),
            "n_muestras_descartadas": n_muestras,
        }

    X = np.stack([valores[i:i + tamano_ventana] for i in inicios])
    frac_imputado = np.stack([imputado[i:i + tamano_ventana].mean() for i in inicios])

    return {
        "X": X,
        "inicio": timestamps[inicios],
        "fin": timestamps[[i + tamano_ventana - 1 for i in inicios]],
        "frac_imputado": frac_imputado,
        "n_muestras_descartadas": n_descartadas,
    }


def construir_ventanas_por_particion(particiones: dict, config: dict) -> dict:
    """Construye las ventanas de train, validación y test con su stride correspondiente."""
    tamano_ventana = config["windowing"]["tamano_ventana_min"]
    return {
        nombre: ventanear(df, tamano_ventana, resolver_stride(config, nombre))
        for nombre, df in particiones.items()
    }


if __name__ == "__main__":
    config = cargar_config()
    df = cargar_y_limpiar()
    particiones = split_temporal(df)
    ventanas = construir_ventanas_por_particion(particiones, config)

    tamano_ventana = config["windowing"]["tamano_ventana_min"]
    print(f"Tamano de ventana: {tamano_ventana} min\n")

    for nombre, v in ventanas.items():
        stride = resolver_stride(config, nombre)
        n = v["X"].shape[0]
        print(f"{nombre}: stride={stride} min, {n} ventanas, shape={v['X'].shape}, "
              f"frac_imputado media={v['frac_imputado'].mean() if n else float('nan'):.4f}, "
              f"descartadas al final={v['n_muestras_descartadas']} muestras")

        assert v["X"].shape == (n, tamano_ventana)
        assert not np.isnan(v["X"]).any(), f"NaN en ventanas de {nombre}"
        if n > 1:
            gaps = np.diff(v["inicio"]).astype("timedelta64[m]").astype(int)
            assert (gaps == stride).all(), f"stride inconsistente en {nombre}"

    # En validación y test, cada ventana debe comenzar justo después de la anterior
    for nombre in ("val", "test"):
        v = ventanas[nombre]
        if v["X"].shape[0] > 1:
            huecos = (v["inicio"][1:] - v["fin"][:-1]).astype("timedelta64[m]").astype(int)
            assert (huecos == 1).all(), f"ventanas de {nombre} no son disjuntas"

    # Comprueba que el ventaneo también funciona con otros tamaños habituales
    for alt_min in (360, 720):
        alt = ventanear(particiones["val"], alt_min, alt_min)
        print(f"val con ventana {alt_min} min -> {alt['X'].shape[0]} ventanas")
        assert alt["X"].shape[1] == alt_min

    print("\nVentaneo consistente: train con solape, val y test disjuntas.")
