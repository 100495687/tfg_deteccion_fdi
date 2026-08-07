"""Partición cronológica del dataset en train, validación y test.

Los cortes se calculan desde la primera fecha real de la serie (el dataset no trae
fechas de corte propias): los dos primeros años para entrenamiento, el tercero para
validación y el resto para test.
Train debe quedarse solo con consumo limpio - los ataques (attacks.py)
se inyectan unicamente sobre test.

Ejecutable:  python -m src.splitting
"""
from __future__ import annotations

import pandas as pd

from src.data_loading import cargar_y_limpiar


def calcular_cortes(inicio: pd.Timestamp, fin: pd.Timestamp) -> dict:
    return {
        "inicio": inicio,
        "corte_train": inicio + pd.Timedelta(days=365 * 2),   # Dos primeros años para entrenamiento
        "corte_val": inicio + pd.Timedelta(days=365 * 3),     # Tercer año para validación
        "fin": fin,                                            # Periodo restante para test
    }


def particionar(df: pd.DataFrame, cortes: dict) -> dict:
    train = df.loc[(df.index >= cortes["inicio"]) & (df.index < cortes["corte_train"])]
    val = df.loc[(df.index >= cortes["corte_train"]) & (df.index < cortes["corte_val"])]
    test = df.loc[(df.index >= cortes["corte_val"]) & (df.index <= cortes["fin"])]

    assert train.index.max() < val.index.min(), "train/val se solapan"
    assert val.index.max() < test.index.min(), "val/test se solapan"

    return {"train": train, "val": val, "test": test}


def split_temporal(df: pd.DataFrame) -> dict:
    cortes = calcular_cortes(df.index.min(), df.index.max())
    return particionar(df, cortes)


if __name__ == "__main__":
    df = cargar_y_limpiar()
    partes = split_temporal(df)

    for nombre, parte in partes.items():
        dias = (parte.index.max() - parte.index.min()).days if len(parte) else 0
        pct_imputado = parte["imputado"].mean() * 100 if len(parte) else float("nan")
        print(f"{nombre}: {len(parte)} filas, de {parte.index.min()} a {parte.index.max()} "
              f"(~{dias} dias), {pct_imputado:.2f}% imputado")

    assert len(partes["train"]) + len(partes["val"]) + len(partes["test"]) <= len(df)
    assert partes["train"]["imputado"].notna().all()

    print("\nParticiones sin solape, todo correcto.")