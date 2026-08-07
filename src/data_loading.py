"""Carga y preparación del dataset IHEPC (UCI id=235)

Para este proyecto solo utilizamos la variable Global_active_power, que está
registrada cada minuto. El dataset contiene algunos valores ausentes, alrededor
del 1.25 %, que se rellenan mediante interpolación lineal
Ejecutable:  python -m src.data_loading"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "data" / "raw" / "data_raw.csv"
REPO_ROOT_CACHE = BASE_DIR.parent / "data_raw.csv"  # CSV generado previamente por exploracion.ipynb
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "serie_limpia.parquet"

COLUMNA_OBJETIVO = "Global_active_power"
REQUIRED_COLUMNS = {"Date", "Time", COLUMNA_OBJETIVO}


def cargar_dataset_crudo() -> tuple[pd.DataFrame, str]:
    """Carga el dataset desde la primera fuente disponible.

    Primero busca el archivo en data/raw/. Si no está ahí, utiliza el CSV
    generado durante la exploración inicial. Como última opción, descarga
    el dataset desde UCI.

    El archivo se guarda en data/raw/ para reutilizarlo en futuras ejecuciones.
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH, low_memory=False), "cache_local"

    if REPO_ROOT_CACHE.exists():
        df = pd.read_csv(REPO_ROOT_CACHE, low_memory=False)
        df.to_csv(CACHE_PATH, index=False)
        return df, "cache_raiz_repo"

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=235)
    df = dataset.data.features.copy()
    df.to_csv(CACHE_PATH, index=False)
    return df, "ucimlrepo_235"


def construir_serie_objetivo(df: pd.DataFrame) -> pd.Series:
    """Prepara la serie temporal de Global_active_power.

    Comprueba que estén disponibles las columnas necesarias, crea el índice
    temporal, ordena los registros y elimina posibles fechas duplicadas.
    Los minutos sin observación se mantienen como valores ausentes.
    """
    faltantes = REQUIRED_COLUMNS - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(faltantes)}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df[COLUMNA_OBJETIVO] = pd.to_numeric(df[COLUMNA_OBJETIVO], errors="coerce")  # Convierte "?" en NaN

    df = (
        df.dropna(subset=["datetime"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .set_index("datetime")
    )

    rango_completo = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    serie = df[COLUMNA_OBJETIVO].reindex(rango_completo)
    serie.index.name = "datetime"
    return serie


def interpolar_faltantes(serie: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Rellena los valores ausentes mediante interpolación lineal.

    También conserva una máscara que indica qué posiciones fueron imputadas,
    para poder identificarlas posteriormente.
    """
    mascara_imputado = serie.isna()
    serie_limpia = serie.interpolate(method="linear", limit_direction="both")
    return serie_limpia, mascara_imputado


def cargar_y_limpiar() -> pd.DataFrame:
    """Ejecuta el proceso completo de carga y limpieza.

    El resultado contiene la serie Global_active_power y una columna booleana
    que indica qué valores fueron imputados.
    """
    df_crudo, fuente = cargar_dataset_crudo()
    serie = construir_serie_objetivo(df_crudo)
    serie_limpia, mascara_imputado = interpolar_faltantes(serie)

    resultado = pd.DataFrame({
        COLUMNA_OBJETIVO: serie_limpia,
        "imputado": mascara_imputado,
    })
    resultado.attrs["fuente_datos"] = fuente
    resultado.attrs["pct_imputado"] = float(mascara_imputado.mean() * 100)
    return resultado


def guardar_procesado(df: pd.DataFrame) -> Path:
    """Guarda el dataset limpio en formato Parquet."""
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH)
    return PROCESSED_PATH


if __name__ == "__main__":
    df = cargar_y_limpiar()

    print(f"Fuente de los datos: {df.attrs['fuente_datos']}")
    print(f"Filas totales: {len(df)}")
    print(f"Rango: {df.index.min()} -> {df.index.max()}")
    print(f"Porcentaje imputado: {df.attrs['pct_imputado']:.3f}%")
    print(f"NaN que quedan: {df[COLUMNA_OBJETIVO].isna().sum()}")
    print(df[COLUMNA_OBJETIVO].describe())

    assert df[COLUMNA_OBJETIVO].isna().sum() == 0, "quedan NaN tras la interpolacion"
    assert df.index.is_monotonic_increasing, "el indice no esta ordenado"
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all(), "faltan minutos en la rejilla"
    assert 0.5 < df.attrs["pct_imputado"] < 3.0, "porcentaje imputado fuera del rango esperado (~1.25%)"

    ruta = guardar_procesado(df)
    print(f"\nGuardado en {ruta}")