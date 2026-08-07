"""carga+split+ventaneo+normalizacion, y entrenamiento/carga cacheada del modelo. 
Evita duplicar este boilerplate en cada modulo ejecutable de forma aislada.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.data_loading import COLUMNA_OBJETIVO, cargar_y_limpiar
from src.models.tcn_ae import TCNAEModelo
from src.normalization import ajustar_zscore, aplicar_zscore
from src.splitting import split_temporal
from src.windowing import construir_ventanas_por_particion

BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "models"


def preparar_datos(config: dict) -> dict:
    df = cargar_y_limpiar()
    particiones = split_temporal(df)
    ventanas = construir_ventanas_por_particion(particiones, config)

    params_norm = ajustar_zscore(particiones["train"][COLUMNA_OBJETIVO].to_numpy())
    ventanas_norm = {nombre: aplicar_zscore(v["X"], params_norm) for nombre, v in ventanas.items()}

    return {
        "particiones": particiones,
        "ventanas": ventanas,
        "ventanas_norm": ventanas_norm,
        "params_norm": params_norm,
    }


def _ruta_checkpoint(config: dict) -> Path:
    tam = config["windowing"]["tamano_ventana_min"]
    return MODELOS_DIR / f"{config['modelo']['nombre']}_ventana{tam}.pt"


def obtener_modelo(config: dict, ventanas_norm: dict, forzar_reentrenar: bool = False) -> TCNAEModelo:
    """Carga el checkpoint si existe; si no (o si `forzar_reentrenar`), entrena con
    train/val y lo guarda para las siguientes ejecuciones."""
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = _ruta_checkpoint(config)
    modelo = TCNAEModelo(config, seed=config["seed"])

    if ruta.exists() and not forzar_reentrenar:
        modelo.net.load_state_dict(torch.load(ruta, map_location="cpu"))
        modelo.net.eval()
        return modelo

    modelo.fit(ventanas_norm["train"], ventanas_norm["val"])
    torch.save(modelo.net.state_dict(), ruta)
    return modelo
