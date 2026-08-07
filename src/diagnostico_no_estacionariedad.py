"""Diagnostico y correccion de la no estacionariedad del score `relative_kw`.

El experimento CUSUM anterior mostro que, incluso sobre datos limpios, el acumulador
sostiene excursiones de dias/semanas cuando `relative_kw` se estandariza con una unica
mediana/MAD global (ajustada sobre todo un anyo). Este experimento solo trabaja sobre datos
limpios: mide esa no estacionariedad, compara 8 formas de construir `center_t`/`scale_t`
(todas producen z_t = (score_t - center_t) / max(scale_t, epsilon)) y usa CUSUM unicamente
como herramienta diagnostica (nunca se evaluan ataques ni DR aqui).

No reentrena nada, no cambia ventana/stride/relative_kw/ataques/episodios. Reutiliza sin
modificar experimento_stride.{cargar_datos_y_modelo, construir_rejilla},
base_relative.calcular_relative_kw y experimento_cusum.{ajustar_estandarizacion, estandarizar,
simular_cusum, calibrar_h, _excursiones} -- el metodo G de este experimento usa literalmente
la misma funcion que el experimento CUSUM anterior, solo que ajustada sobre train en vez de val.

Ejecutable de forma aislada:
    python -m src.diagnostico_no_estacionariedad
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.base_relative import calcular_relative_kw
from src.data_loading import COLUMNA_OBJETIVO
from src.experimento_cusum import ajustar_estandarizacion, calibrar_h, estandarizar, simular_cusum
from src.experimento_cusum import _excursiones as excursiones_cusum
from src.experimento_stride import cargar_datos_y_modelo, construir_rejilla
from src.normalization import aplicar_zscore

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "diagnostico_no_estacionariedad"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "diagnostico_no_estacionariedad"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDE = 60           # 1 puntuacion por hora
EPS_Z = 1e-6
GAP_HORAS = 24         # hueco de seguridad de las referencias moviles
MIN_OBS_HOW = 30       # minimo de observaciones para el bin hora-de-la-semana (Metodo H)
MIN_OBS_HOD = 30       # minimo para el fallback hora-del-dia
MIN_OBS_CH = 5         # minimo de semanas historicas para el metodo contextual (CH)
SEMANA_H = 168
METODOS = ["G", "H", "R4", "R8", "R12", "CH8", "CH12", "HR"]
COLOR_METODO = {
    "G": GRIS_OSCURO, "H": NARANJA, "R4": "#8fb8e8", "R8": AZUL, "R12": AZUL_OSCURO,
    "CH8": "#c9a3e0", "CH12": MORADO, "HR": VERDE,
}
LAGS_HORAS = [1, 6, 12, 24, 48, 168, 336]

MESES_ES = {1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun",
            7: "jul", 8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic"}


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


def _skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m, s = x.mean(), x.std()
    return float(np.mean((x - m) ** 3) / s ** 3) if s > 0 else 0.0


def _estacion(mes: int) -> str:
    return {12: "invierno", 1: "invierno", 2: "invierno", 3: "primavera", 4: "primavera", 5: "primavera",
            6: "verano", 7: "verano", 8: "verano", 9: "otonio", 10: "otonio", 11: "otonio"}[mes]


def _autocorrelacion(x: np.ndarray, lags: list[int]) -> dict[int, float]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    resultado = {}
    for lag in lags:
        if lag >= n:
            resultado[lag] = float("nan")
            continue
        a, b = x[:-lag], x[lag:]
        if a.std() == 0 or b.std() == 0:
            resultado[lag] = float("nan")
        else:
            resultado[lag] = float(np.corrcoef(a, b)[0, 1])
    return resultado


def _duracion_rachas(activo: np.ndarray, resolucion_min: int = 60) -> np.ndarray:
    duraciones, contador = [], 0
    for a in activo:
        if a:
            contador += 1
        else:
            if contador > 0:
                duraciones.append(contador * resolucion_min)
            contador = 0
    if contador > 0:
        duraciones.append(contador * resolucion_min)
    return np.array(duraciones) if duraciones else np.array([0.0])


# ==========================================================================================
# 1) Secuencia continua limpia de relative_kw (train + val), 1 puntuacion/hora
# ==========================================================================================

def construir_secuencia_continua() -> dict:
    config, datos, modelo = cargar_datos_y_modelo()
    params_norm = datos["params_norm"]
    filas_por_particion = {}
    for nombre in ("train", "val"):
        particion = datos["particiones"][nombre]
        grid = construir_rejilla(particion, STRIDE)
        X_norm = aplicar_zscore(grid["X"], params_norm)
        X_hat = modelo.reconstruir(X_norm)
        rel = calcular_relative_kw(X_norm, X_hat, params_norm)
        filas_por_particion[nombre] = {
            "timestamps": pd.to_datetime(grid["fin"]), "relative_kw": rel,
            "consumo_medio_kw": grid["X"].mean(axis=1), "grid": grid,
        }

    tabla = pd.concat([
        pd.DataFrame({"particion": nombre, "timestamp": d["timestamps"], "relative_kw": d["relative_kw"],
                       "consumo_medio_kw": d["consumo_medio_kw"]})
        for nombre, d in filas_por_particion.items()
    ], ignore_index=True)
    _guardar_tabla(tabla, "00_secuencia_relative_kw_train_val")
    return {"config": config, "datos": datos, "modelo": modelo, "params_norm": params_norm,
            "por_particion": filas_por_particion, "tabla": tabla}


# ==========================================================================================
# 2) Separacion temporal: train limpio | primera mitad val | segunda mitad val
# ==========================================================================================

def dividir_val(serie_val: dict) -> dict:
    n = len(serie_val["relative_kw"])
    corte = n // 2
    primera = {k: (v[:corte] if not isinstance(v, dict) else v) for k, v in serie_val.items() if k != "grid"}
    segunda = {k: (v[corte:] if not isinstance(v, dict) else v) for k, v in serie_val.items() if k != "grid"}
    return {"primera_mitad": primera, "segunda_mitad": segunda, "corte_idx": corte}


# ==========================================================================================
# 3) Diagnostico inicial de relative_kw (sin normalizar)
# ==========================================================================================

def _estadisticos_grupo(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": len(x), "media": float(x.mean()), "mediana": float(np.median(x)), "std": float(x.std()),
        "mad": float(np.median(np.abs(x - np.median(x)))),
        "p5": float(np.percentile(x, 5)), "p25": float(np.percentile(x, 25)), "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
        "asimetria": _skewness(x),
    }


def diagnostico_inicial(tabla: pd.DataFrame) -> dict:
    t = tabla.copy()
    t["mes"] = t["timestamp"].dt.month
    t["estacion"] = t["mes"].map(_estacion)
    t["hora_dia"] = t["timestamp"].dt.hour
    t["dia_semana"] = t["timestamp"].dt.dayofweek
    t["hora_semana"] = t["dia_semana"] * 24 + t["hora_dia"]
    dias_totales = (t["timestamp"] - t["timestamp"].min()).dt.total_seconds() / 86400.0
    t["trimestre_consecutivo"] = (dias_totales // 91.25).astype(int)
    t["tercil_consumo"] = pd.qcut(t["consumo_medio_kw"], 3, labels=["bajo", "medio", "alto"])

    tablas = {}
    for nombre_grupo, col in (("mes", "mes"), ("estacion", "estacion"), ("hora_dia", "hora_dia"),
                               ("dia_semana", "dia_semana"), ("hora_semana", "hora_semana"),
                               ("trimestre_consecutivo", "trimestre_consecutivo"), ("tercil_consumo", "tercil_consumo")):
        filas = []
        for val_, g in t.groupby(col, observed=True):
            fila = {col: val_}
            fila.update(_estadisticos_grupo(g["relative_kw"].to_numpy()))
            filas.append(fila)
        tablas[f"por_{nombre_grupo}"] = pd.DataFrame(filas)

    medianas_mensuales = tablas["por_mes"].set_index("mes")["mediana"]
    rango_medianas_mensuales = float(medianas_mensuales.max() - medianas_mensuales.min())
    escalas_mensuales = tablas["por_mes"].set_index("mes")["mad"] * 1.4826
    cv_escala_mensual = float(escalas_mensuales.std() / escalas_mensuales.mean()) if escalas_mensuales.mean() else float("nan")

    val = t[t["particion"] == "val"].reset_index(drop=True)
    corte = len(val) // 2
    diff_mitades = float(val["relative_kw"].iloc[corte:].median() - val["relative_kw"].iloc[:corte].median())

    acf_por_particion = {}
    for nombre in ("train", "val"):
        x = t[t["particion"] == nombre]["relative_kw"].to_numpy()
        acf_por_particion[nombre] = _autocorrelacion(x, LAGS_HORAS)
    tabla_acf = pd.DataFrame([{"particion": p, "lag_horas": lag, "autocorrelacion": v}
                               for p, d in acf_por_particion.items() for lag, v in d.items()])

    p90, p95, p99 = np.percentile(t["relative_kw"], [90, 95, 99])
    filas_rachas = []
    for nombre in ("train", "val"):
        x = t[t["particion"] == nombre]["relative_kw"].to_numpy()
        for etiqueta, umbral in (("p90", p90), ("p95", p95), ("p99", p99)):
            dur = _duracion_rachas(x > umbral, STRIDE)
            filas_rachas.append({"particion": nombre, "percentil_global": etiqueta, "umbral": umbral,
                                  "n_rachas": len(dur), "duracion_media_min": float(dur.mean()),
                                  "duracion_max_min": float(dur.max())})
    tabla_rachas = pd.DataFrame(filas_rachas)

    resumen = pd.DataFrame([{
        "rango_medianas_mensuales": rango_medianas_mensuales, "cv_escala_mensual": cv_escala_mensual,
        "diff_mediana_1a_vs_2a_mitad_val": diff_mitades,
        "acf_val_lag24h": acf_por_particion["val"][24], "acf_val_lag168h": acf_por_particion["val"][168],
        "acf_val_lag336h": acf_por_particion["val"][336],
    }])

    for nombre, tb in tablas.items():
        _guardar_tabla(tb, f"01_diagnostico_{nombre}")
    _guardar_tabla(tabla_acf, "01_autocorrelacion_original")
    _guardar_tabla(tabla_rachas, "01_rachas_percentiles_globales")
    _guardar_tabla(resumen, "01_resumen_no_estacionariedad")

    return {"tablas_grupo": tablas, "acf": tabla_acf, "rachas": tabla_rachas, "resumen": resumen, "t": t}


# ==========================================================================================
# 4) Las 8 transformaciones (todas producen z_t = (score_t - center_t)/max(scale_t, eps))
# ==========================================================================================

def _ajustar_perfil(scores: np.ndarray, etiquetas: np.ndarray, n_bins: int, min_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.full(n_bins, np.nan); scale = np.full(n_bins, np.nan); n_obs = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        x = scores[etiquetas == b]
        n_obs[b] = len(x)
        if len(x) >= min_obs:
            med = np.median(x)
            center[b] = med
            scale[b] = 1.4826 * np.median(np.abs(x - med))
    return center, scale, n_obs


def metodo_G(scores_train: np.ndarray, scores_objetivo: np.ndarray) -> pd.DataFrame:
    """Referencia global, ajustada sobre TRAIN limpio -- misma formula que
    experimento_cusum.ajustar_estandarizacion/estandarizar, aplicada aqui con datos de train
    en vez de val (documentado en el modulo)."""
    params = ajustar_estandarizacion(scores_train)
    z = estandarizar(scores_objetivo, params)
    n = len(scores_objetivo)
    return pd.DataFrame({
        "z": z, "center": params["median_clean"], "scale": params["denom_efectivo"],
        "used_fallback": False, "fallback_type": "ninguno",
        "reference_start": 0, "reference_end": len(scores_train) - 1, "n_reference_samples": len(scores_train),
    })


def metodo_H(scores_train: np.ndarray, how_train: np.ndarray, hod_train: np.ndarray,
             scores_objetivo: np.ndarray, how_objetivo: np.ndarray, hod_objetivo: np.ndarray) -> pd.DataFrame:
    """Perfil estatico por hora de la semana (168 bins), ajustado sobre train. Fallback:
    (1) perfil por hora del dia (24 bins), (2) referencia global -- ambos tambien ajustados
    sobre train, si el bin de 168 no alcanza MIN_OBS_HOW observaciones."""
    center_how, scale_how, n_how = _ajustar_perfil(scores_train, how_train, SEMANA_H, MIN_OBS_HOW)
    center_hod, scale_hod, n_hod = _ajustar_perfil(scores_train, hod_train, 24, MIN_OBS_HOD)
    params_glob = ajustar_estandarizacion(scores_train)

    n = len(scores_objetivo)
    z = np.empty(n); center = np.empty(n); scale = np.empty(n)
    fallback = np.zeros(n, dtype=bool); tipo = np.array(["ninguno"] * n, dtype=object)
    for i in range(n):
        h = how_objetivo[i]
        if n_how[h] >= MIN_OBS_HOW:
            c, s = center_how[h], scale_how[h]
        else:
            fallback[i] = True
            hd = hod_objetivo[i]
            if n_hod[hd] >= MIN_OBS_HOD:
                c, s = center_hod[hd], scale_hod[hd]
                tipo[i] = "hora_del_dia"
            else:
                c, s = params_glob["median_clean"], params_glob["denom_efectivo"]
                tipo[i] = "global"
        center[i] = c; scale[i] = max(s, EPS_Z)
        z[i] = (scores_objetivo[i] - c) / max(s, EPS_Z)
    return pd.DataFrame({"z": z, "center": center, "scale": scale, "used_fallback": fallback, "fallback_type": tipo,
                          "reference_start": 0, "reference_end": len(scores_train) - 1,
                          "n_reference_samples": n_how[how_objetivo]})


def _referencia_movil(scores_full: np.ndarray, idx_objetivo: np.ndarray, semanas: int, gap_horas: int = GAP_HORAS,
                       min_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mediana/MAD causales de una ventana movil de `semanas*168` horas, terminando `gap_horas`
    antes de cada indice objetivo. Vectorizado con sliding_window_view sobre el tramo minimo
    necesario de `scores_full` (nunca toca indices >= idx_objetivo - gap_horas + 1)."""
    W = semanas * SEMANA_H
    n = len(idx_objetivo)
    center = np.full(n, np.nan); scale = np.full(n, np.nan); n_ref = np.zeros(n, dtype=int)
    ref_start = np.full(n, -1, dtype=np.int64); ref_end = np.full(n, -1, dtype=np.int64)
    fin_ref = idx_objetivo - gap_horas  # ultimo indice usable (inclusive)
    ini_ref = fin_ref - W + 1

    validos = ini_ref >= 0
    suficientes = validos.copy()
    if validos.any():
        lo_bloque, hi_bloque = int(ini_ref[validos].min()), int(fin_ref[validos].max())
        bloque = scores_full[lo_bloque:hi_bloque + 1]
        ventanas = np.lib.stride_tricks.sliding_window_view(bloque, W)  # shape (len(bloque)-W+1, W)
        offsets = (fin_ref[validos] - (lo_bloque + W - 1)).astype(int)  # indice de fila en `ventanas`
        sub = ventanas[offsets]  # (n_validos, W)
        med = np.median(sub, axis=1)
        mad = np.median(np.abs(sub - med[:, None]), axis=1)
        center[validos] = med
        scale[validos] = 1.4826 * mad
        n_ref[validos] = W
        ref_start[validos] = ini_ref[validos]
        ref_end[validos] = fin_ref[validos]
    # exige al menos min_frac*W de historial real disponible (aqui, o toda la ventana o nada,
    # porque train por si solo ya cubre 2 anyos >> 12 semanas: se documenta el criterio igualmente)
    suficientes = validos
    return center, scale, n_ref, ref_start, ref_end, suficientes


def metodo_R(scores_full: np.ndarray, idx_objetivo: np.ndarray, semanas: int) -> pd.DataFrame:
    center, scale, n_ref, ref_start, ref_end, suficientes = _referencia_movil(scores_full, idx_objetivo, semanas)
    z = np.where(suficientes, (scores_full[idx_objetivo] - center) / np.maximum(scale, EPS_Z), np.nan)
    return pd.DataFrame({"z": z, "center": center, "scale": scale, "used_fallback": ~suficientes,
                          "fallback_type": np.where(suficientes, "ninguno", "sin_historial_suficiente"),
                          "reference_start": ref_start, "reference_end": ref_end, "n_reference_samples": n_ref})


def metodo_CH(scores_full: np.ndarray, idx_objetivo: np.ndarray, semanas: int,
              center_how_fallback: np.ndarray, scale_how_fallback: np.ndarray, n_how_fallback: np.ndarray,
              how_objetivo: np.ndarray, params_glob_fallback: dict, ts_full: pd.DatetimeIndex) -> pd.DataFrame:
    """Para cada indice objetivo, usa las hasta `semanas` observaciones anteriores CON LA MISMA
    hora de la semana. Los candidatos se buscan por ARITMETICA DE TIMESTAMPS (t - 168*m horas) y
    se resuelven a un indice mediante un diccionario timestamp->posicion, NUNCA por resta directa
    de indices: train y val se ventanean por separado (ver construir_rejilla), asi que hay un
    hueco real de ~6h justo en el empalme train/val que rompe la periodicidad de 168 en la
    aritmetica de indices para los primeros timestamps de val cuya ventana de busqueda llega a
    tocar train. Con la resolucion por timestamp ese hueco simplemente se traduce en "no hay
    observacion exacta a esa hora" (se descarta ese candidato), nunca en una hora incorrecta.
    El hueco de seguridad de 24h se cumple automaticamente: 168 > 24. Fallback (1) Metodo H
    estatico, (2) global."""
    ts_a_idx = {ts: i for i, ts in enumerate(ts_full)}
    n = len(idx_objetivo)
    z = np.empty(n); center = np.empty(n); scale = np.empty(n); n_ref = np.zeros(n, dtype=int)
    ref_start = np.full(n, -1, dtype=np.int64); ref_end = np.full(n, -1, dtype=np.int64)
    fallback = np.zeros(n, dtype=bool); tipo = np.array(["ninguno"] * n, dtype=object)
    for i in range(n):
        idx_i = idx_objetivo[i]
        ts_i = ts_full[idx_i]
        candidatos_ts = [ts_i - pd.Timedelta(hours=SEMANA_H * m) for m in range(1, semanas + 1)]
        candidatos = np.array(sorted(ts_a_idx[ts] for ts in candidatos_ts if ts in ts_a_idx), dtype=np.int64)
        if len(candidatos) >= MIN_OBS_CH:
            muestras = scores_full[candidatos]
            med = np.median(muestras)
            c, s = med, 1.4826 * np.median(np.abs(muestras - med))
            n_ref[i] = len(candidatos)
            ref_start[i], ref_end[i] = int(candidatos.min()), int(candidatos.max())
        else:
            fallback[i] = True
            h = how_objetivo[i]
            if n_how_fallback[h] >= MIN_OBS_HOW:
                c, s = center_how_fallback[h], scale_how_fallback[h]
                tipo[i] = "hora_semana_estatica"
            else:
                c, s = params_glob_fallback["median_clean"], params_glob_fallback["denom_efectivo"]
                tipo[i] = "global"
        center[i] = c; scale[i] = max(s, EPS_Z)
        z[i] = (scores_full[idx_i] - c) / max(s, EPS_Z)
    return pd.DataFrame({"z": z, "center": center, "scale": scale, "used_fallback": fallback, "fallback_type": tipo,
                          "reference_start": ref_start, "reference_end": ref_end, "n_reference_samples": n_ref})


def metodo_HR(scores_full: np.ndarray, idx_objetivo: np.ndarray, center_how: np.ndarray, how_full: np.ndarray,
              semanas: int = 8) -> pd.DataFrame:
    """residual_t = score_t - center_how[hora_semana(t)]; despues aplica una referencia movil
    causal (misma funcion que Metodo R) de `semanas` semanas SOBRE EL RESIDUO."""
    residual_full = scores_full - center_how[how_full]
    center_r, scale_r, n_ref, ref_start, ref_end, suficientes = _referencia_movil(residual_full, idx_objetivo, semanas)
    residual_obj = residual_full[idx_objetivo]
    z = np.where(suficientes, (residual_obj - center_r) / np.maximum(scale_r, EPS_Z), np.nan)
    return pd.DataFrame({"z": z, "center": center_r, "scale": scale_r, "used_fallback": ~suficientes,
                          "fallback_type": np.where(suficientes, "ninguno", "sin_historial_suficiente"),
                          "reference_start": ref_start, "reference_end": ref_end, "n_reference_samples": n_ref})


def calcular_todos_los_metodos(scores_train: np.ndarray, scores_val: np.ndarray, ts_train: pd.DatetimeIndex,
                                ts_val: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    how_train = (pd.DatetimeIndex(ts_train).dayofweek * 24 + pd.DatetimeIndex(ts_train).hour).to_numpy()
    hod_train = pd.DatetimeIndex(ts_train).hour.to_numpy()
    how_val = (pd.DatetimeIndex(ts_val).dayofweek * 24 + pd.DatetimeIndex(ts_val).hour).to_numpy()
    hod_val = pd.DatetimeIndex(ts_val).hour.to_numpy()

    scores_full = np.concatenate([scores_train, scores_val])
    how_full = np.concatenate([how_train, how_val])
    ts_full = pd.DatetimeIndex(list(ts_train) + list(ts_val))
    idx_val_en_full = len(scores_train) + np.arange(len(scores_val))

    resultados = {}
    resultados["G"] = metodo_G(scores_train, scores_val)
    resultados["H"] = metodo_H(scores_train, how_train, hod_train, scores_val, how_val, hod_val)
    for semanas, nombre in ((4, "R4"), (8, "R8"), (12, "R12")):
        resultados[nombre] = metodo_R(scores_full, idx_val_en_full, semanas)

    center_how_train, scale_how_train, n_how_train = _ajustar_perfil(scores_train, how_train, SEMANA_H, MIN_OBS_HOW)
    params_glob_train = ajustar_estandarizacion(scores_train)
    for semanas, nombre in ((8, "CH8"), (12, "CH12")):
        resultados[nombre] = metodo_CH(scores_full, idx_val_en_full, semanas, center_how_train, scale_how_train,
                                        n_how_train, how_val, params_glob_train, ts_full)

    resultados["HR"] = metodo_HR(scores_full, idx_val_en_full, center_how_train, how_full, semanas=8)

    for nombre, df in resultados.items():
        df.insert(0, "timestamp", ts_val)
        df.insert(0, "metodo", nombre)
    return resultados


# ==========================================================================================
# 6) Estabilidad de cada transformacion (sobre la SEGUNDA mitad de val)
# ==========================================================================================

def analizar_estabilidad(z_segunda: np.ndarray, ts_segunda: pd.DatetimeIndex, consumo_segunda: np.ndarray,
                          fallback_segunda: np.ndarray) -> dict:
    valido = ~np.isnan(z_segunda)
    z = z_segunda[valido]
    horas_dia = pd.DatetimeIndex(ts_segunda).hour.to_numpy()[valido]
    horas_semana = (pd.DatetimeIndex(ts_segunda).dayofweek.to_numpy() * 24 + horas_dia)[valido]
    meses = pd.DatetimeIndex(ts_segunda).month.to_numpy()[valido]
    terciles = pd.qcut(consumo_segunda[valido], 3, labels=["bajo", "medio", "alto"])

    resumen = {
        "media": float(z.mean()), "mediana": float(np.median(z)), "std": float(z.std()),
        "mad": float(np.median(np.abs(z - np.median(z)))),
        "p1": float(np.percentile(z, 1)), "p5": float(np.percentile(z, 5)), "p25": float(np.percentile(z, 25)),
        "p50": float(np.percentile(z, 50)), "p75": float(np.percentile(z, 75)), "p95": float(np.percentile(z, 95)),
        "p99": float(np.percentile(z, 99)), "asimetria": _skewness(z),
        "pct_positivos": 100 * float((z > 0).mean()),
        "pct_fallback": 100 * float(fallback_segunda.mean()),
    }
    for lag in [1, 6, 12, 24, 48, 168]:
        resumen[f"acf_lag{lag}h"] = _autocorrelacion(z, [lag])[lag]

    medianas_mes = pd.Series(z).groupby(meses).median()
    resumen["rango_medianas_mensuales"] = float(medianas_mes.max() - medianas_mes.min()) if len(medianas_mes) > 1 else 0.0
    escalas_mes = pd.Series(z).groupby(meses).std()
    resumen["cv_escala_mensual"] = float(escalas_mes.std() / escalas_mes.mean()) if escalas_mes.mean() else float("nan")
    resumen["std_por_hora_dia"] = float(pd.Series(z).groupby(horas_dia).std().mean())
    resumen["std_por_hora_semana"] = float(pd.Series(z).groupby(horas_semana).std().mean())
    resumen["std_por_tercil_consumo"] = float(pd.Series(z).groupby(terciles, observed=True).std().mean())
    return resumen


# ==========================================================================================
# 7) Diagnostico CUSUM sobre datos limpios (calibrar en 1a mitad, evaluar sin recalibrar en 2a)
# ==========================================================================================

def diagnostico_cusum(z_primera: np.ndarray, z_segunda: np.ndarray, n_dias_primera: float, n_dias_segunda: float,
                       k_param: float) -> dict:
    z1 = z_primera[~np.isnan(z_primera)]
    z2 = z_segunda[~np.isnan(z_segunda)]
    h, tasa_calib, C_calib, alarm_calib = calibrar_h(z1, k_param, 0.06, n_dias_primera)
    C_eval, alarm_eval = simular_cusum(z2, k_param, h)
    tasa_eval = alarm_eval.sum() / n_dias_segunda
    excursiones = excursiones_cusum(C_eval, STRIDE)
    ts_alarma = np.where(alarm_eval)[0]
    tiempo_medio_min = float(np.diff(ts_alarma).mean() * STRIDE) if len(ts_alarma) > 1 else float("nan")
    return {
        "h": h, "falsos_eventos_dia_calibracion": tasa_calib, "falsos_eventos_dia_evaluacion": tasa_eval,
        "dif_abs_vs_0.06": abs(tasa_eval - 0.06), "falsas_ventanas_dia_evaluacion": float(alarm_eval.sum()) / n_dias_segunda,
        "duracion_media_excursion_min": float(excursiones.mean()), "duracion_mediana_excursion_min": float(np.median(excursiones)),
        "duracion_p95_excursion_min": float(np.percentile(excursiones, 95)), "duracion_max_excursion_min": float(excursiones.max()),
        "max_C_alcanzado": float(C_eval.max()), "tiempo_medio_entre_alarmas_min": tiempo_medio_min,
        "n_eventos_evaluacion": int(alarm_eval.sum()),
    }


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("[1] Construyendo la secuencia continua limpia relative_kw (train + val, 1 score/hora)...")
    base = construir_secuencia_continua()
    tabla = base["tabla"]
    d_train = base["por_particion"]["train"]
    d_val = base["por_particion"]["val"]
    serie_antes_val = d_val["grid"]["X"].copy()

    print(f"  train: {len(d_train['relative_kw'])} scores, {d_train['timestamps'][0]} -> {d_train['timestamps'][-1]}")
    print(f"  val:   {len(d_val['relative_kw'])} scores, {d_val['timestamps'][0]} -> {d_val['timestamps'][-1]}")

    print("\n[2] Dividiendo val en primera/segunda mitad temporal...")
    div = dividir_val(d_val)
    corte = div["corte_idx"]
    ts_val = pd.DatetimeIndex(d_val["timestamps"])
    print(f"  primera mitad: {ts_val[0]} -> {ts_val[corte - 1]} ({corte} scores)")
    print(f"  segunda mitad: {ts_val[corte]} -> {ts_val[-1]} ({len(ts_val) - corte} scores)")
    n_dias_primera = corte * STRIDE / 1440.0
    n_dias_segunda = (len(ts_val) - corte) * STRIDE / 1440.0
    _guardar_tabla(pd.DataFrame([{
        "primera_mitad_inicio": ts_val[0], "primera_mitad_fin": ts_val[corte - 1], "n_primera": corte, "n_dias_primera": n_dias_primera,
        "segunda_mitad_inicio": ts_val[corte], "segunda_mitad_fin": ts_val[-1], "n_segunda": len(ts_val) - corte, "n_dias_segunda": n_dias_segunda,
    }]), "00_particion_temporal_val")

    print("\n[3] Diagnostico inicial de relative_kw (sin normalizar)...")
    diag = diagnostico_inicial(tabla)
    print(diag["resumen"].to_string(index=False))

    print("\n[4] Calculando las 8 transformaciones sobre val (ajustadas con train + historial causal)...")
    metodos = calcular_todos_los_metodos(d_train["relative_kw"], d_val["relative_kw"], d_train["timestamps"], d_val["timestamps"])
    tabla_metodos = pd.concat(metodos.values(), ignore_index=True)
    _guardar_tabla(tabla_metodos, "02_transformaciones_val")
    for nombre, df in metodos.items():
        n_fallback = int(df["used_fallback"].sum())
        n_nan = int(df["z"].isna().sum())
        print(f"  {nombre}: fallback={n_fallback}/{len(df)} ({100 * n_fallback / len(df):.2f}%) | sin_dato(NaN)={n_nan}")

    print("\n[6] Estabilidad de cada metodo sobre la SEGUNDA mitad de val...")
    filas_estabilidad = []
    for nombre, df in metodos.items():
        z2 = df["z"].to_numpy()[corte:]
        ts2 = ts_val[corte:]
        cons2 = d_val["consumo_medio_kw"][corte:]
        fb2 = df["used_fallback"].to_numpy()[corte:]
        fila = {"metodo": nombre}
        fila.update(analizar_estabilidad(z2, ts2, cons2, fb2))
        filas_estabilidad.append(fila)
    tabla_estabilidad = _guardar_tabla(pd.DataFrame(filas_estabilidad), "03_estabilidad_segunda_mitad")
    print(tabla_estabilidad[["metodo", "media", "std", "asimetria", "pct_positivos", "acf_lag168h",
                              "rango_medianas_mensuales", "pct_fallback"]].to_string(index=False))

    print("\n[7] Diagnostico CUSUM limpio (calibrado en 1a mitad, evaluado sin recalibrar en 2a)...")
    filas_cusum = []
    for nombre, df in metodos.items():
        z = df["z"].to_numpy()
        z1, z2 = z[:corte], z[corte:]
        for k_param in (0.50, 1.00):
            fila = {"metodo": nombre, "k": k_param}
            fila.update(diagnostico_cusum(z1, z2, n_dias_primera, n_dias_segunda, k_param))
            filas_cusum.append(fila)
    tabla_cusum = _guardar_tabla(pd.DataFrame(filas_cusum), "04_diagnostico_cusum")
    print(tabla_cusum[["metodo", "k", "h", "falsos_eventos_dia_calibracion", "falsos_eventos_dia_evaluacion",
                        "duracion_max_excursion_min", "duracion_p95_excursion_min"]].to_string(index=False))

    # ---- [8] tabla de decision ----
    print("\n[8] Construyendo la tabla de decision...")
    cusum_k1 = tabla_cusum[tabla_cusum["k"] == 1.00].set_index("metodo")
    est_idx = tabla_estabilidad.set_index("metodo")
    causalidad = {"G": "no (global, estatica)", "H": "no (estatica)", "R4": "si", "R8": "si", "R12": "si",
                  "CH8": "si", "CH12": "si", "HR": "si (estatica + movil)"}
    historia = {"G": "2 anyos (train)", "H": "2 anyos (train)", "R4": "4 semanas", "R8": "8 semanas", "R12": "12 semanas",
                "CH8": "8 semanas (misma hora-semana)", "CH12": "12 semanas (misma hora-semana)", "HR": "8 semanas (residuo)"}
    complejidad = {"G": "muy baja", "H": "baja", "R4": "media", "R8": "media", "R12": "media",
                   "CH8": "media-alta", "CH12": "media-alta", "HR": "alta"}
    filas_decision = []
    for nombre in METODOS:
        filas_decision.append({
            "metodo": nombre, "causal": causalidad[nombre], "longitud_historia": historia[nombre],
            "hueco_seguridad_h": GAP_HORAS if nombre not in ("G", "H") else "n/a",
            "pct_fallback_2a_mitad": est_idx.loc[nombre, "pct_fallback"],
            "rango_medianas_mensuales_2a_mitad": est_idx.loc[nombre, "rango_medianas_mensuales"],
            "acf_lag168h_2a_mitad": est_idx.loc[nombre, "acf_lag168h"],
            "h_k1": cusum_k1.loc[nombre, "h"],
            "falsos_eventos_dia_calibracion_k1": cusum_k1.loc[nombre, "falsos_eventos_dia_calibracion"],
            "falsos_eventos_dia_evaluacion_k1": cusum_k1.loc[nombre, "falsos_eventos_dia_evaluacion"],
            "duracion_max_excursion_min_k1": cusum_k1.loc[nombre, "duracion_max_excursion_min"],
            "duracion_p95_excursion_min_k1": cusum_k1.loc[nombre, "duracion_p95_excursion_min"],
            "complejidad_operativa": complejidad[nombre],
        })
    tabla_decision = _guardar_tabla(pd.DataFrame(filas_decision), "05_tabla_decision")
    print(tabla_decision.to_string(index=False))

    candidatos = tabla_decision[
        (tabla_decision["falsos_eventos_dia_evaluacion_k1"] - 0.06).abs() < 0.03
    ].copy()
    if len(candidatos):
        candidatos["score_seleccion"] = (
            -candidatos["duracion_max_excursion_min_k1"] / candidatos["duracion_max_excursion_min_k1"].max()
            - candidatos["pct_fallback_2a_mitad"] / 100.0
            - candidatos["acf_lag168h_2a_mitad"].abs()
        )
        metodo_recomendado = candidatos.sort_values("score_seleccion", ascending=False).iloc[0]["metodo"]
    else:
        metodo_recomendado = tabla_decision.sort_values("duracion_max_excursion_min_k1").iloc[0]["metodo"]
    print(f"\nMetodo recomendado (excursiones cortas + baja deriva mensual/ACF + fallback bajo + generaliza fuera de muestra): {metodo_recomendado}")

    # ---- [9] comprobacion secundaria sobre secuencias atacadas (SOLO NaN/causalidad, SIN DR) ----
    print("\n[9] Comprobacion secundaria sobre secuencias atacadas (solo NaN / trazabilidad, sin DR)...")
    from src.base_relative import cargar_ataques_y_modelo_360
    from src.stride_relative import puntuar_stride_relative
    _, datos_360, modelo_360, ataques, _ = cargar_ataques_y_modelo_360()
    particion_val_raw = datos_360["particiones"]["val"][COLUMNA_OBJETIVO].to_numpy().copy()
    grid_val_60 = construir_rejilla(datos_360["particiones"]["val"], STRIDE)
    tabla_ventanas_atacadas, _ = puntuar_stride_relative(
        dict(list(ataques.items())[:20]), particion_val_raw, grid_val_60, modelo_360, datos_360["params_norm"], STRIDE)
    params_glob_check = ajustar_estandarizacion(d_train["relative_kw"])
    z_atacado_check = estandarizar(tabla_ventanas_atacadas["attacked_score"].to_numpy(), params_glob_check)
    assert not np.any(np.isnan(z_atacado_check)) and not np.any(np.isinf(z_atacado_check))
    assert np.array_equal(particion_val_raw, datos_360["particiones"]["val"][COLUMNA_OBJETIVO].to_numpy())
    print(f"  OK -- {len(z_atacado_check)} ventanas atacadas (20 episodios de muestra) transformadas con el "
          f"Metodo G sin NaN/Inf y sin alterar la senal atacada ni usar informacion futura")

    # ---- [10] figuras ----
    print("\nGenerando figuras...")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for nombre, d in base["por_particion"].items():
        color = GRIS if nombre == "train" else AZUL
        ax.plot(d["timestamps"], d["relative_kw"], color=color, linewidth=0.4, alpha=0.7, label=nombre)
    ax.set_ylabel("relative_kw"); ax.set_title("Serie temporal completa de relative_kw limpio (train + val)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "01_serie_temporal_relative_kw")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    tm = diag["tablas_grupo"]["por_mes"].sort_values("mes")
    ax1.plot(tm["mes"], tm["mediana"], marker="o", color=AZUL)
    ax1.set_xlabel("mes"); ax1.set_ylabel("mediana relative_kw"); ax1.set_title("Mediana mensual")
    ax2.plot(tm["mes"], tm["mad"] * 1.4826, marker="o", color=NARANJA)
    ax2.set_xlabel("mes"); ax2.set_ylabel("escala (1.4826*MAD)"); ax2.set_title("Escala mensual")
    fig.tight_layout(); _guardar_fig(fig, "02_mediana_mad_por_mes")

    fig, ax = plt.subplots(figsize=(8, 5))
    th = diag["tablas_grupo"]["por_hora_dia"].sort_values("hora_dia")
    ax.plot(th["hora_dia"], th["mediana"], marker="o", color=AZUL, label="mediana")
    ax.fill_between(th["hora_dia"], th["p25"], th["p75"], color=AZUL, alpha=0.2, label="p25-p75")
    ax.set_xlabel("hora del dia"); ax.set_ylabel("relative_kw"); ax.set_title("Distribucion por hora del dia")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "03_distribucion_por_hora_dia")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ths = diag["tablas_grupo"]["por_hora_semana"].sort_values("hora_semana")
    ax.plot(ths["hora_semana"], ths["mediana"], color=AZUL, linewidth=1.2)
    for d in range(8):
        ax.axvline(d * 24, color=GRIS, linewidth=0.5, linestyle=":")
    ax.set_xlabel("hora de la semana (0-167)"); ax.set_ylabel("mediana relative_kw"); ax.set_title("Perfil por hora de la semana (train)")
    fig.tight_layout(); _guardar_fig(fig, "04_perfil_hora_semana")

    fig, ax = plt.subplots(figsize=(8, 5))
    for particion, color in (("train", GRIS), ("val", AZUL)):
        sub = diag["acf"][diag["acf"]["particion"] == particion].sort_values("lag_horas")
        ax.plot(sub["lag_horas"], sub["autocorrelacion"], marker="o", color=color, label=particion)
    ax.axhline(0, color=GRIS_OSCURO, linewidth=0.8)
    ax.set_xlabel("lag (horas)"); ax.set_ylabel("autocorrelacion"); ax.set_title("Autocorrelacion de relative_kw original")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "05_autocorrelacion_original")

    fig, ax = plt.subplots(figsize=(12, 5))
    for nombre in METODOS:
        z = metodos[nombre]["z"].to_numpy()
        ax.plot(ts_val, z, linewidth=0.4, alpha=0.6, color=COLOR_METODO[nombre], label=nombre)
    ax.axvline(ts_val[corte], color=ROJO, linestyle="--", linewidth=1, label="corte 1a/2a mitad")
    ax.set_ylabel("z_t"); ax.set_title("Comparacion temporal de z_t, todos los metodos")
    ax.legend(fontsize=7, ncols=4)
    fig.tight_layout(); _guardar_fig(fig, "06_comparacion_temporal_z_todos_metodos")

    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre in METODOS:
        z = metodos[nombre]["z"].to_numpy()
        meses = ts_val.month.to_numpy()
        med_mes = pd.Series(z).groupby(meses).median()
        ax.plot(med_mes.index, med_mes.values, marker="o", markersize=3, color=COLOR_METODO[nombre], label=nombre)
    ax.axhline(0, color=GRIS_OSCURO, linewidth=0.8)
    ax.set_xlabel("mes"); ax.set_ylabel("mediana z_t"); ax.set_title("Mediana mensual de z_t por metodo")
    ax.legend(fontsize=7, ncols=4)
    fig.tight_layout(); _guardar_fig(fig, "07_mediana_mensual_z_por_metodo")

    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre in METODOS:
        z = metodos[nombre]["z"].to_numpy()
        meses = ts_val.month.to_numpy()
        std_mes = pd.Series(z).groupby(meses).std()
        ax.plot(std_mes.index, std_mes.values, marker="o", markersize=3, color=COLOR_METODO[nombre], label=nombre)
    ax.set_xlabel("mes"); ax.set_ylabel("std z_t"); ax.set_title("Escala mensual de z_t por metodo")
    ax.legend(fontsize=7, ncols=4)
    fig.tight_layout(); _guardar_fig(fig, "08_escala_mensual_z_por_metodo")

    fig, ax = plt.subplots(figsize=(8, 5))
    for nombre in METODOS:
        sub = tabla_estabilidad[tabla_estabilidad["metodo"] == nombre].iloc[0]
        acfs = [sub[f"acf_lag{lag}h"] for lag in [1, 6, 12, 24, 48, 168]]
        ax.plot([1, 6, 12, 24, 48, 168], acfs, marker="o", color=COLOR_METODO[nombre], label=nombre)
    ax.axhline(0, color=GRIS_OSCURO, linewidth=0.8)
    ax.set_xlabel("lag (horas)"); ax.set_ylabel("autocorrelacion de z_t"); ax.set_title("Autocorrelacion de z_t por metodo (2a mitad val)")
    ax.legend(fontsize=7, ncols=4)
    fig.tight_layout(); _guardar_fig(fig, "09_autocorrelacion_z_por_metodo")

    fig, ax = plt.subplots(figsize=(8, 5))
    sub_k1 = tabla_cusum[tabla_cusum["k"] == 1.00].set_index("metodo").reindex(METODOS)
    ax.bar(METODOS, sub_k1["duracion_max_excursion_min"] / 60.0, color=[COLOR_METODO[m] for m in METODOS])
    ax.set_ylabel("duracion maxima de excursion (horas)"); ax.set_title("Duracion de excursiones CUSUM limpias (k=1.00, 2a mitad)")
    ax.set_yscale("log")
    fig.tight_layout(); _guardar_fig(fig, "10_duracion_excursiones_cusum")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(METODOS)); ancho = 0.35
    ax.bar(x - ancho / 2, sub_k1["falsos_eventos_dia_calibracion"], width=ancho, color=GRIS, label="calibracion (1a mitad)")
    ax.bar(x + ancho / 2, sub_k1["falsos_eventos_dia_evaluacion"], width=ancho, color=ROJO, label="evaluacion (2a mitad, sin recalibrar)")
    ax.axhline(0.06, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(METODOS, rotation=20)
    ax.set_ylabel("falsos eventos/dia"); ax.set_title("Falsos eventos/dia: calibracion vs evaluacion fuera de muestra (k=1.00)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "11_falsos_eventos_calibracion_evaluacion")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(METODOS, est_idx.reindex(METODOS)["pct_fallback"], color=[COLOR_METODO[m] for m in METODOS])
    ax.set_ylabel("% de timestamps con fallback"); ax.set_title("Uso de fallback por metodo (2a mitad val)")
    fig.tight_layout(); _guardar_fig(fig, "12_pct_fallback_por_metodo")

    print("  ejemplos representativos de evolucion de CUSUM limpio (metodo recomendado vs G)...")
    z_rec = metodos[metodo_recomendado]["z"].to_numpy()
    z_g = metodos["G"].to_numpy() if False else metodos["G"]["z"].to_numpy()
    h_rec = cusum_k1.loc[metodo_recomendado, "h"]
    h_g = cusum_k1.loc["G", "h"]
    C_rec, _ = simular_cusum(np.nan_to_num(z_rec, nan=0.0), 1.00, h_rec)
    C_g, _ = simular_cusum(z_g, 1.00, h_g)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(ts_val, C_g, color=ROJO, linewidth=0.6)
    axes[0].axhline(h_g, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    axes[0].set_ylabel("C_t"); axes[0].set_title(f"CUSUM limpio, Metodo G (baseline), k=1.00, h={h_g:.1f}")
    axes[1].plot(ts_val, C_rec, color=VERDE, linewidth=0.6)
    axes[1].axhline(h_rec, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    axes[1].set_ylabel("C_t"); axes[1].set_title(f"CUSUM limpio, Metodo {metodo_recomendado} (recomendado), k=1.00, h={h_rec:.1f}")
    fig.tight_layout(); _guardar_fig(fig, "13_evolucion_cusum_limpio_ejemplos")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    filas_tabla = tabla_decision[["metodo", "causal", "longitud_historia", "pct_fallback_2a_mitad",
                                   "duracion_max_excursion_min_k1", "falsos_eventos_dia_evaluacion_k1",
                                   "complejidad_operativa"]].round(2)
    tabla_mpl = ax.table(cellText=filas_tabla.values, colLabels=filas_tabla.columns, loc="center", cellLoc="center")
    tabla_mpl.auto_set_font_size(False); tabla_mpl.set_fontsize(8); tabla_mpl.scale(1, 1.6)
    for j, col in enumerate(filas_tabla.columns):
        tabla_mpl[0, j].set_facecolor(AZUL_CLARO)
    for i, met in enumerate(filas_tabla["metodo"], start=1):
        if met == metodo_recomendado:
            for j in range(len(filas_tabla.columns)):
                tabla_mpl[i, j].set_facecolor("#d9f0d9")
    ax.set_title(f"Tabla de decision -- recomendado: {metodo_recomendado}", fontsize=11)
    fig.tight_layout(); _guardar_fig(fig, "14_tabla_decision_visual")

    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")
    assert np.array_equal(serie_antes_val, d_val["grid"]["X"]), "la serie limpia original no debe modificarse"

    return {
        "base": base, "div": div, "diag": diag, "metodos": metodos, "tabla_estabilidad": tabla_estabilidad,
        "tabla_cusum": tabla_cusum, "tabla_decision": tabla_decision, "metodo_recomendado": metodo_recomendado,
        "corte_idx": corte, "n_dias_primera": n_dias_primera, "n_dias_segunda": n_dias_segunda,
    }


if __name__ == "__main__":
    main()
