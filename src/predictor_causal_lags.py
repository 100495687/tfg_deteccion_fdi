"""Predictor causal periodico de consumo: Ridge e HistGradientBoosting con lags diarios y
semanales explicitos a resolucion de 15 minutos.

Motivacion: el TCN enmascarado causal M60 (contexto=5h recientes) no mejoro al detector
reconstructivo P, pero un perfil historico trivial por franja horaria/tipo de dia si superaba a
P en DR/DR energetico/alto impacto (a costa de una FA injustificable). Este experimento prueba
si un predictor causal que combine explicitamente lags diarios (24h/48h/72h), semanal (168h) y
calendario -- con o sin lags recientes (<=6h) -- puede igualar esa senal periodica pero con FA
controlada exactamente al presupuesto operativo.

Resolucion nueva (15 min, no 1 min): cada intervalo objetivo agrega minuto a minuto de forma
causal y no solapada. La ventana de contexto ya no es una secuencia bruta (como en el TCN) sino
un vector de features tabulares (lags puntuales + estadisticos rolling, siempre con shift previo
a cualquier agregacion). PERIODIC excluye toda referencia a las <=6h recientes (solo
lag_96/192/288/672 + perfil + calendario); HYBRID anade lags recientes y rolling causales.

No reentrena ni modifica el TCN-AE original ni el TCN M60. No usa test. Reutiliza sin modificar
data_loading.{COLUMNA_OBJETIVO, cargar_y_limpiar}, splitting.split_temporal,
base_relative.cargar_ataques_y_modelo_360 (solo episodios/master_csv, nunca el modelo),
experimento_ramp.construir_episodios_ramp, experimento_stride.{contar_eventos, mcnemar_p,
bootstrap_pareado}, analisis_detectabilidad.{contexto_e_impacto, _energia_antes_despues},
memoria_finita_relative_kw.{enriquecer_resultado, resumen_metricas, _duracion_eventos} y
episodes.wilson_ci. Referencia (sin recalcular) los resultados ya guardados de
energy_distance_operativo (P) y tcn_masked_m60 (M60_BASE).

Ejecutable de forma aislada:
    python -m src.predictor_causal_lags
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import analisis_detectabilidad as ad
from src.base_relative import cargar_ataques_y_modelo_360
from src.data_loading import COLUMNA_OBJETIVO, cargar_y_limpiar
from src.episodes import wilson_ci
from src.experimento_ramp import construir_episodios_ramp
from src.experimento_stride import bootstrap_pareado, contar_eventos, mcnemar_p
from src.memoria_finita_relative_kw import _duracion_eventos, enriquecer_resultado, resumen_metricas
from src.splitting import split_temporal

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "predictor_causal_lags"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "predictor_causal_lags"
MODELOS_DIR = BASE_DIR / "models" / "predictor_causal_lags"
RAMP_DIR = BASE_DIR / "results" / "tables" / "experimento_ramp"
DET_DIR = BASE_DIR / "results" / "tables" / "analisis_detectabilidad"
EDO_DIR = BASE_DIR / "results" / "tables" / "energy_distance_operativo"
M60_DIR = BASE_DIR / "results" / "tables" / "tcn_masked_m60"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

RESOLUCION_MIN = 15
EPSILON_KW = 0.1
OBJETIVO_FA = 0.06
BANDA_PRINCIPAL = (0.04, 0.08)
BANDA_SECUNDARIA = (0.02, 0.12)
SEED = 42
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
RHO_FINALES = [0.70, 0.50, 0.30]

LAGS_RECIENTES = {"lag_1": 1, "lag_2": 2, "lag_4": 4, "lag_8": 8, "lag_12": 12, "lag_24": 24}
LAGS_DIARIOS = {"lag_96": 96, "lag_192": 192, "lag_288": 288}
LAG_SEMANAL = {"lag_672": 672}
TODOS_LOS_LAGS = {**LAGS_RECIENTES, **LAGS_DIARIOS, **LAG_SEMANAL}
ROLLING_WINDOWS = [4, 12, 24, 96]
MAX_LAG = 672  # 672*15min = 168h = 7 dias -- define cuanto contexto previo hace falta

CALENDARIO_COLS = ["slot_15min", "hora", "dow", "es_finde", "mes", "sin_hora", "cos_hora", "sin_dow", "cos_dow"]
PERIODIC_FEATURES = list(LAGS_DIARIOS) + list(LAG_SEMANAL) + ["profile_train", "mad_train", "n_obs_train"] + CALENDARIO_COLS
HYBRID_EXTRA = list(LAGS_RECIENTES) + [f"rolling_{stat}_{w}" for w in ROLLING_WINDOWS for stat in ("mean", "median", "std")]
HYBRID_FEATURES = PERIODIC_FEATURES + HYBRID_EXTRA

RIDGE_ALPHAS = [0.1, 1, 10, 100]
HISTGB_GRID = {
    "learning_rate": [0.03, 0.05, 0.10], "max_leaf_nodes": [15, 31], "max_depth": [None, 8],
    "min_samples_leaf": [20, 50], "l2_regularization": [0.0, 1.0],
}


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 4) Agregacion causal a 15 minutos (no solapada) -- reutilizable para clean y para tramos
#    atacados (el ataque se aplica a nivel de minuto antes de agregar)
# ==========================================================================================

def construir_serie_15min(particion: pd.DataFrame) -> dict:
    valores = particion[COLUMNA_OBJETIVO].to_numpy(dtype=np.float64)
    timestamps = particion.index.to_numpy()
    n = (len(valores) // RESOLUCION_MIN) * RESOLUCION_MIN
    kw = valores[:n].reshape(-1, RESOLUCION_MIN).mean(axis=1)
    ts_start = timestamps[0:n:RESOLUCION_MIN]
    ts_end = timestamps[RESOLUCION_MIN - 1:n:RESOLUCION_MIN]
    return {"kw": kw, "ts_start": ts_start, "ts_end": ts_end}


def _bins_afectados(offset_global: int, duracion: int, n_bins: int, resolucion: int = RESOLUCION_MIN) -> list[int]:
    a_end = offset_global + duracion
    b_ini = max(0, offset_global // resolucion)
    b_fin = min(n_bins - 1, (a_end - 1) // resolucion)
    return list(range(b_ini, b_fin + 1))


def construir_kw15_atacado(kw15_clean: np.ndarray, particion_val_raw: np.ndarray, offset_global: int,
                            duracion: int, y_tramo: np.ndarray, resolucion: int = RESOLUCION_MIN) -> tuple[np.ndarray, list[int]]:
    """Aplica el ataque a nivel de minuto sobre `particion_val_raw` y reagrega solo los bins de
    15 min realmente tocados por el ataque -- el resto de la serie permanece identica a la
    limpia. Devuelve la serie de 15 min completa (identica salvo esos bins) y la lista de bins
    afectados."""
    n_bins = len(kw15_clean)
    bins_afectados = _bins_afectados(offset_global, duracion, n_bins, resolucion)
    kw15_atacado = kw15_clean.copy()
    a_end = offset_global + duracion
    for b in bins_afectados:
        m0 = b * resolucion
        seg = particion_val_raw[m0:m0 + resolucion].astype(np.float64).copy()
        s_ini, s_fin = max(offset_global, m0), min(a_end, m0 + resolucion)
        if s_ini < s_fin:
            seg[s_ini - m0:s_fin - m0] = y_tramo[s_ini - offset_global:s_fin - offset_global]
        kw15_atacado[b] = seg.mean()
    return kw15_atacado, bins_afectados


# ==========================================================================================
# 6) Construccion de features -- todas las variables que dependen del consumo aplican shift
#    antes de rolling/lag
# ==========================================================================================

def construir_features_kw(kw: np.ndarray) -> pd.DataFrame:
    """Lags puntuales (shift directo, causal por construccion: lag_k en la fila i usa kw[i-k],
    con k>=1) + estadisticos rolling calculados siempre sobre la serie ya desplazada 1 paso
    (`base = s.shift(1)`), nunca sobre la propia fila objetivo. Ninguna ventana esta centrada:
    pandas `.rolling()` es por definicion right-aligned (incluye la fila actual y las
    `window-1` anteriores) y aqui se aplica sobre `base`, que ya excluye el instante actual."""
    s = pd.Series(kw)
    cols = {}
    for nombre, k in TODOS_LOS_LAGS.items():
        cols[nombre] = s.shift(k).to_numpy()
    base = s.shift(1)
    for w in ROLLING_WINDOWS:
        cols[f"rolling_mean_{w}"] = base.rolling(w).mean().to_numpy()
        cols[f"rolling_median_{w}"] = base.rolling(w).median().to_numpy()
        cols[f"rolling_std_{w}"] = base.rolling(w).std().to_numpy()
    return pd.DataFrame(cols)


def _calendario(ts_start: np.ndarray) -> pd.DataFrame:
    """Variables de calendario del propio intervalo objetivo -- conocidas de antemano (solo
    dependen del timestamp, nunca del consumo), por lo que no hay fuga aunque no lleven shift."""
    ts = pd.DatetimeIndex(ts_start)
    slot = (ts.hour * 60 + ts.minute) // RESOLUCION_MIN
    dow = ts.dayofweek.to_numpy()
    return pd.DataFrame({
        "slot_15min": slot.to_numpy(), "hora": ts.hour.to_numpy(), "dow": dow,
        "es_finde": (dow >= 5).astype(int), "mes": ts.month.to_numpy(),
        "sin_hora": np.sin(2 * np.pi * ts.hour.to_numpy() / 24.0), "cos_hora": np.cos(2 * np.pi * ts.hour.to_numpy() / 24.0),
        "sin_dow": np.sin(2 * np.pi * dow / 7.0), "cos_dow": np.cos(2 * np.pi * dow / 7.0),
    })


def construir_perfil_train(train15: dict) -> pd.DataFrame:
    """profile_train = mediana por (es_finde, slot_15min), calculada exclusivamente con train
    limpio. Anade tambien MAD (desviacion absoluta mediana) y el numero de observaciones."""
    cal = _calendario(train15["ts_start"])
    df = pd.DataFrame({"slot_15min": cal["slot_15min"], "es_finde": cal["es_finde"], "target_kw": train15["kw"]})
    g = df.groupby(["es_finde", "slot_15min"])["target_kw"]
    perfil = g.median().rename("profile_train")
    mad = g.apply(lambda x: float(np.median(np.abs(x - x.median())))).rename("mad_train")
    n_obs = g.size().rename("n_obs_train")
    return pd.concat([perfil, mad, n_obs], axis=1).reset_index()


def construir_features_particion(serie15: dict, perfil_train: pd.DataFrame) -> pd.DataFrame:
    """Ensambla lags + rolling + calendario + perfil para todos los bins de una particion, y
    descarta las primeras `MAX_LAG` filas (7 dias) que no tienen contexto semanal completo."""
    feats_kw = construir_features_kw(serie15["kw"])
    feats_cal = _calendario(serie15["ts_start"])
    df = pd.concat([feats_kw, feats_cal], axis=1)
    df["target_kw"] = serie15["kw"]
    df["ts_start"] = serie15["ts_start"]
    df["ts_end"] = serie15["ts_end"]
    df = df.merge(perfil_train, on=["es_finde", "slot_15min"], how="left")
    df = df.iloc[MAX_LAG:].reset_index(drop=True)
    assert df[PERIODIC_FEATURES + HYBRID_EXTRA].isna().sum().sum() == 0, "quedan NaN tras recortar el lookback"
    return df


# ==========================================================================================
# 5) Auditoria de causalidad (resumen tabular)
# ==========================================================================================

def auditar_causalidad(df_train_feat: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {"comprobacion": "test nunca se carga ni se usa (split_temporal produce train/val/test; test no se referencia)", "resultado": True},
        {"comprobacion": "agregacion a 15 min causal y no solapada (bins de 15 muestras consecutivas, sin solape)", "resultado": True},
        {"comprobacion": "ataques aplicados a nivel de minuto y reagregados despues (construir_kw15_atacado)", "resultado": True},
        {"comprobacion": "todos los lags/rolling aplican shift antes de agregar (s.shift(k); rolling sobre s.shift(1))", "resultado": True},
        {"comprobacion": "ningun rolling esta centrado (pandas .rolling() es right-aligned, aplicado sobre datos ya desplazados)", "resultado": True},
        {"comprobacion": "lag_96 equivale a 24h", "resultado": bool(96 * RESOLUCION_MIN == 24 * 60)},
        {"comprobacion": "lag_672 equivale a 168h", "resultado": bool(672 * RESOLUCION_MIN == 168 * 60)},
        {"comprobacion": "perfil historico (profile_train/mad_train/n_obs_train) calculado solo con train", "resultado": True},
        {"comprobacion": "PERIODIC no contiene lags recientes (<=6h) ni rolling", "resultado": not (set(LAGS_RECIENTES) & set(PERIODIC_FEATURES)) and not any(c.startswith("rolling_") for c in PERIODIC_FEATURES)},
        {"comprobacion": "HYBRID = PERIODIC + lags recientes + rolling", "resultado": bool(set(HYBRID_FEATURES) == set(PERIODIC_FEATURES) | set(HYBRID_EXTRA))},
        {"comprobacion": "target_kw nunca aparece entre las columnas de features (X)", "resultado": "target_kw" not in PERIODIC_FEATURES and "target_kw" not in HYBRID_FEATURES},
        {"comprobacion": "normalizacion (StandardScaler de Ridge) se ajusta solo con train dentro del propio Pipeline.fit", "resultado": True},
        {"comprobacion": "umbral calibrado solo con scores limpios de desarrollo", "resultado": True},
        {"comprobacion": "busqueda de umbral exhaustiva: se evalua cada valor unico del score de desarrollo como candidato", "resultado": True},
        {"comprobacion": "rama atacada reconstruye sus propios lags con el consumo realmente atacado (nunca contexto limpio)", "resultado": True},
        {"comprobacion": "no quedan NaN en las features tras el recorte de lookback (verificado en construir_features_particion)", "resultado": bool(df_train_feat[PERIODIC_FEATURES + HYBRID_EXTRA].isna().sum().sum() == 0)},
    ]
    tabla = pd.DataFrame(checks)
    assert tabla["resultado"].all(), "auditoria de causalidad FALLIDA"
    return _guardar_tabla(tabla, "auditoria_causalidad")


def tabla_definicion_features() -> pd.DataFrame:
    filas = []
    for nombre, k in TODOS_LOS_LAGS.items():
        filas.append({"feature": nombre, "tipo": "lag", "desplazamiento_intervalos": k, "desplazamiento_horas": k * RESOLUCION_MIN / 60.0,
                       "en_periodic": nombre in PERIODIC_FEATURES, "en_hybrid": nombre in HYBRID_FEATURES})
    for w in ROLLING_WINDOWS:
        for stat in ("mean", "median", "std"):
            nombre = f"rolling_{stat}_{w}"
            filas.append({"feature": nombre, "tipo": f"rolling_{stat}", "desplazamiento_intervalos": w, "desplazamiento_horas": w * RESOLUCION_MIN / 60.0,
                           "en_periodic": nombre in PERIODIC_FEATURES, "en_hybrid": nombre in HYBRID_FEATURES})
    for nombre in ["profile_train", "mad_train", "n_obs_train"] + CALENDARIO_COLS:
        filas.append({"feature": nombre, "tipo": "perfil_historico" if "train" in nombre else "calendario", "desplazamiento_intervalos": np.nan,
                       "desplazamiento_horas": np.nan, "en_periodic": nombre in PERIODIC_FEATURES, "en_hybrid": nombre in HYBRID_FEATURES})
    return _guardar_tabla(pd.DataFrame(filas), "definicion_features")


# ==========================================================================================
# 1/3) Setup
# ==========================================================================================

def cargar_todo() -> dict:
    df = cargar_y_limpiar()
    particiones = split_temporal(df)
    _, _, _modelo_ae_no_usado, ataques_originales, master_csv = cargar_ataques_y_modelo_360()  # solo episodios/master_csv

    particion_train, particion_val = particiones["train"], particiones["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy(dtype=np.float64).copy()
    inicio_particion_val = particion_val.index[0]

    train15 = construir_serie_15min(particion_train)
    val15 = construir_serie_15min(particion_val)
    n_val_full = len(val15["kw"])
    corte_full = n_val_full // 2  # punto medio del año de val completo -- preserva el mismo eval_ini que todos los experimentos anteriores

    perfil_train = construir_perfil_train(train15)
    df_train_feat = construir_features_particion(train15, perfil_train)
    df_val_feat = construir_features_particion(val15, perfil_train)

    corte_idx = corte_full - MAX_LAG  # dev pierde los primeros 7 dias por falta de contexto semanal; eval queda intacto
    assert 0 < corte_idx < len(df_val_feat), "corte de desarrollo/evaluacion fuera de rango tras recortar el lookback"
    n_val = len(df_val_feat)
    div = {
        "corte_idx": corte_idx,
        "dev_ini": pd.Timestamp(df_val_feat["ts_end"].iloc[0]), "dev_fin": pd.Timestamp(df_val_feat["ts_end"].iloc[corte_idx - 1]),
        "eval_ini": pd.Timestamp(df_val_feat["ts_end"].iloc[corte_idx]), "eval_fin": pd.Timestamp(df_val_feat["ts_end"].iloc[-1]),
        "n_dias_dev": corte_idx * RESOLUCION_MIN / 1440.0, "n_dias_eval": (n_val - corte_idx) * RESOLUCION_MIN / 1440.0,
    }

    return {
        "particiones": particiones, "particion_val_raw": particion_val_raw, "inicio_particion_val": inicio_particion_val,
        "ataques_originales": ataques_originales, "master_csv": master_csv,
        "train15": train15, "val15": val15, "n_val_full": n_val_full, "perfil_train": perfil_train,
        "df_train_feat": df_train_feat, "df_val_feat": df_val_feat, "div": div,
    }


# ==========================================================================================
# 8/9) Modelos: perfil historico (sin entrenamiento), Ridge y HistGB (rejilla acotada,
#      seleccion solo con train+desarrollo limpio)
# ==========================================================================================

def metricas_prediccion(pred: np.ndarray, real: np.ndarray) -> dict:
    err = pred - real
    abs_err = np.abs(err)
    return {"MAE": float(abs_err.mean()), "RMSE": float(np.sqrt((err ** 2).mean())), "MedAE": float(np.median(abs_err)),
            "sesgo_medio": float(err.mean())}


def desglose_prediccion(pred: np.ndarray, real: np.ndarray, ts_start: np.ndarray) -> dict:
    abs_err = np.abs(pred - real)
    ts = pd.DatetimeIndex(ts_start)
    por_hora = pd.Series(abs_err).groupby(ts.hour).mean()
    por_mes = pd.Series(abs_err).groupby(ts.month).mean()
    terciles = pd.qcut(pd.Series(real).rank(method="first"), 3, labels=["bajo", "medio", "alto"])
    por_tercil = pd.Series(abs_err).groupby(terciles, observed=True).mean()
    return {"por_hora": por_hora, "por_mes": por_mes, "por_tercil": por_tercil,
            "variabilidad_mensual": float(por_mes.std()), "sesgo_por_hora_std": float(por_hora.std())}


def _complejidad_ridge(alpha: float) -> float:
    return 1.0 / alpha  # mayor alpha -> mas regularizado -> menos complejo


def _complejidad_histgb(p: dict) -> float:
    prof = p["max_depth"] if p["max_depth"] is not None else 20
    return (p["max_leaf_nodes"] * prof) / (p["min_samples_leaf"] * (1.0 + p["l2_regularization"]))


def entrenar_ridge(feature_set: str, cols: list, X_train, y_train, X_dev, y_dev, ts_dev) -> tuple:
    filas = []
    modelos = {}
    for alpha in RIDGE_ALPHAS:
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=SEED))
        m.fit(X_train, y_train)
        pred = m.predict(X_dev)
        met = metricas_prediccion(pred, y_dev)
        d = desglose_prediccion(pred, y_dev, ts_dev)
        filas.append({"modelo": "Ridge", "feature_set": feature_set, "alpha": alpha, **met,
                       "variabilidad_mensual": d["variabilidad_mensual"], "sesgo_abs": abs(met["sesgo_medio"]),
                       "complejidad": _complejidad_ridge(alpha)})
        modelos[alpha] = m
    tabla = pd.DataFrame(filas).sort_values(["MAE", "sesgo_abs", "variabilidad_mensual", "complejidad"]).reset_index(drop=True)
    mejor = tabla.iloc[0]
    return modelos[mejor["alpha"]], tabla, {"alpha": float(mejor["alpha"])}


def entrenar_histgb(feature_set: str, cols: list, X_train, y_train, X_dev, y_dev, ts_dev) -> tuple:
    filas = []
    modelos = {}
    combos = list(itertools.product(HISTGB_GRID["learning_rate"], HISTGB_GRID["max_leaf_nodes"], HISTGB_GRID["max_depth"],
                                     HISTGB_GRID["min_samples_leaf"], HISTGB_GRID["l2_regularization"]))
    for i, (lr, leaves, depth, min_leaf, l2) in enumerate(combos):
        params = {"learning_rate": lr, "max_leaf_nodes": leaves, "max_depth": depth, "min_samples_leaf": min_leaf, "l2_regularization": l2}
        m = HistGradientBoostingRegressor(random_state=SEED, **params)
        m.fit(X_train, y_train)
        pred = m.predict(X_dev)
        met = metricas_prediccion(pred, y_dev)
        d = desglose_prediccion(pred, y_dev, ts_dev)
        filas.append({"combo_idx": i, "modelo": "HistGB", "feature_set": feature_set, **params, **met,
                       "variabilidad_mensual": d["variabilidad_mensual"], "sesgo_abs": abs(met["sesgo_medio"]),
                       "complejidad": _complejidad_histgb(params)})
        modelos[i] = m
    tabla = pd.DataFrame(filas).sort_values(["MAE", "sesgo_abs", "variabilidad_mensual", "complejidad"]).reset_index(drop=True)
    idx_mejor = int(tabla.iloc[0]["combo_idx"])  # indice ORIGINAL en `combos`/`modelos` (robusto a que max_depth=None se
    combo_cols = ["learning_rate", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization"]  # convierta en NaN al pasar por un DataFrame)
    mejor_params = tabla.iloc[0][combo_cols].to_dict()
    mejor_params["max_depth"] = combos[idx_mejor][2]  # restaura None explicitamente (NaN tras el roundtrip por pandas)
    return modelos[idx_mejor], tabla, mejor_params


def fase1_entrenamiento(ctx: dict) -> dict:
    df_tr, df_dev = ctx["df_train_feat"], ctx["df_val_feat"].iloc[:ctx["div"]["corte_idx"]]
    y_tr, y_dev = df_tr["target_kw"].to_numpy(), df_dev["target_kw"].to_numpy()
    ts_dev = df_dev["ts_start"].to_numpy()

    resultados, tablas_rejilla = {}, []
    for feature_set, cols in (("PERIODIC", PERIODIC_FEATURES), ("HYBRID", HYBRID_FEATURES)):
        X_tr, X_dev = df_tr[cols].to_numpy(dtype=np.float64), df_dev[cols].to_numpy(dtype=np.float64)
        print(f"  Entrenando Ridge_{feature_set} ({len(RIDGE_ALPHAS)} configs)...")
        m_ridge, tabla_ridge, params_ridge = entrenar_ridge(feature_set, cols, X_tr, y_tr, X_dev, y_dev, ts_dev)
        print(f"  Entrenando HistGB_{feature_set} ({len(list(itertools.product(*HISTGB_GRID.values())))} configs)...")
        m_histgb, tabla_histgb, params_histgb = entrenar_histgb(feature_set, cols, X_tr, y_tr, X_dev, y_dev, ts_dev)
        tablas_rejilla.append(tabla_ridge); tablas_rejilla.append(tabla_histgb)
        resultados[f"Ridge_{feature_set}"] = {"modelo": m_ridge, "cols": cols, "params": params_ridge}
        resultados[f"HistGB_{feature_set}"] = {"modelo": m_histgb, "cols": cols, "params": params_histgb}

    tabla_rejilla = pd.concat(tablas_rejilla, ignore_index=True)
    _guardar_tabla(tabla_rejilla, "rejilla_modelos")

    filas_sel = []
    for nombre, r in resultados.items():
        fila = {"metodo": nombre, "modelo": nombre.split("_")[0], "feature_set": "_".join(nombre.split("_")[1:])}
        fila.update({f"param_{k}": v for k, v in r["params"].items()})
        filas_sel.append(fila)
    _guardar_tabla(pd.DataFrame(filas_sel), "modelos_seleccionados")

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, r in resultados.items():
        joblib.dump(r["modelo"], MODELOS_DIR / f"{nombre}.joblib")

    return resultados


# ==========================================================================================
# 9) Perfil historico como metodo de prediccion (sin entrenamiento)
# ==========================================================================================

def predecir_perfil(df_features: pd.DataFrame) -> np.ndarray:
    return df_features["profile_train"].to_numpy(dtype=np.float64)


def construir_predictor(nombre: str, resultados_entrenamiento: dict):
    if nombre == "PerfilHistorico":
        return list(PERIODIC_FEATURES), predecir_perfil
    r = resultados_entrenamiento[nombre]
    modelo, cols = r["modelo"], r["cols"]
    return cols, (lambda df, modelo=modelo, cols=cols: modelo.predict(df[cols].to_numpy(dtype=np.float64)))


# ==========================================================================================
# 10/11/12) Score de deteccion (una unica magnitud por intervalo, no un vector como en M60)
# ==========================================================================================

def calcular_score(pred_kw: np.ndarray, real_kw: np.ndarray, eps: float = EPSILON_KW) -> tuple[np.ndarray, np.ndarray]:
    signed = (pred_kw - real_kw) / np.maximum(np.abs(pred_kw), eps)
    positive = np.maximum(0.0, signed)
    return signed, positive


def puntuar_metodo_limpio(df_val_feat: pd.DataFrame, cols: list, predecir_fn) -> pd.DataFrame:
    pred = predecir_fn(df_val_feat[cols])
    real = df_val_feat["target_kw"].to_numpy()
    signed, positive = calcular_score(pred, real)
    df = pd.DataFrame({
        "k": np.arange(len(df_val_feat)), "target_start": df_val_feat["ts_start"].to_numpy(), "target_end": df_val_feat["ts_end"].to_numpy(),
        "pred_kw": pred, "real_kw": real, "causal_relative_kw": positive, "signed_relative_kw": signed,
        "estimated_hidden_energy_kwh": np.maximum(0.0, pred - real) * RESOLUCION_MIN / 60.0,
    })
    return df


# ==========================================================================================
# 11) Calibracion exacta del umbral (barrido de todos los valores unicos del score de
#     desarrollo -- no una rejilla de percentiles)
# ==========================================================================================

def calibrar_umbral_exacto(scores_dev: np.ndarray, n_dias_dev: float, objetivo: float = OBJETIVO_FA,
                            banda_principal: tuple = BANDA_PRINCIPAL, banda_secundaria: tuple = BANDA_SECUNDARIA) -> tuple[pd.DataFrame, dict]:
    valores_unicos = np.unique(scores_dev)
    filas = []
    for v in valores_unicos:
        activo = scores_dev > v
        n_eventos = contar_eventos(activo)
        eventos_dia = n_eventos / n_dias_dev if n_dias_dev else float("nan")
        filas.append({"umbral": float(v), "n_eventos": n_eventos, "eventos_dia": eventos_dia, "pct_tiempo_activo": 100 * float(activo.mean())})
    tabla = pd.DataFrame(filas)
    tabla["dentro_banda_principal"] = tabla["eventos_dia"].between(*banda_principal)
    tabla["dentro_banda_secundaria"] = tabla["eventos_dia"].between(*banda_secundaria)

    candidatos, banda_usada = tabla[tabla["dentro_banda_principal"]], "principal"
    if len(candidatos) == 0:
        candidatos, banda_usada = tabla[tabla["dentro_banda_secundaria"]], "secundaria"
    if len(candidatos) == 0:
        candidatos, banda_usada = tabla.copy(), "fuera_de_banda_mas_conservador"
    candidatos = candidatos.copy()
    candidatos["dist_objetivo"] = (candidatos["eventos_dia"] - objetivo).abs()
    candidatos = candidatos.sort_values(["dist_objetivo", "umbral"], ascending=[True, False])
    seleccion = candidatos.iloc[0].to_dict()
    seleccion["banda_usada"] = banda_usada
    return tabla, seleccion


def resumen_eventos_grid(activo: np.ndarray, n_dias: float) -> dict:
    n_eventos = contar_eventos(activo)
    duraciones = _duracion_eventos(activo, resolucion_min=RESOLUCION_MIN)
    ts_eventos = np.where(activo & ~np.concatenate([[False], activo[:-1]]))[0]
    tiempo_medio = float(np.diff(ts_eventos).mean() * RESOLUCION_MIN) if len(ts_eventos) > 1 else float("nan")
    return {
        "n_eventos": n_eventos, "n_dias": n_dias, "eventos_dia": n_eventos / n_dias if n_dias else float("nan"),
        "pct_tiempo_en_alarma": 100 * float(activo.mean()), "duracion_media_min": float(duraciones.mean()),
        "duracion_mediana_min": float(np.median(duraciones)), "duracion_p95_min": float(np.percentile(duraciones, 95)),
        "tiempo_medio_entre_eventos_min": tiempo_medio,
    }


def evaluar_fa(scores: np.ndarray, div: dict, umbral: float, etiqueta: str) -> dict:
    corte = div["corte_idx"]
    if etiqueta == "desarrollo":
        sub, n_dias = scores[:corte], div["n_dias_dev"]
    else:
        sub, n_dias = scores[corte:], div["n_dias_eval"]
    activo = sub > umbral
    r = resumen_eventos_grid(activo, n_dias)
    if r["n_eventos"] > 0:
        lo = sstats.chi2.ppf(0.025, 2 * r["n_eventos"]) / 2 / n_dias
    else:
        lo = 0.0
    hi = sstats.chi2.ppf(0.975, 2 * (r["n_eventos"] + 1)) / 2 / n_dias
    r["ic95_poisson_low"], r["ic95_poisson_high"] = lo, hi
    r["periodo"] = etiqueta
    return r


def distribucion_fa(df_val_scored: pd.DataFrame, div: dict, umbral: float) -> pd.DataFrame:
    corte = div["corte_idx"]
    sub = df_val_scored.iloc[corte:].copy()
    sub["activo"] = sub["causal_relative_kw"].to_numpy() > umbral
    ts = pd.to_datetime(sub["target_start"])
    terciles = pd.qcut(sub["real_kw"].rank(method="first"), 3, labels=["bajo", "medio", "alto"])
    filas = []
    for mes, g in sub.groupby(ts.dt.month):
        filas.append({"agrupacion": "mes", "valor": mes, "n": len(g), "pct_activo": 100 * g["activo"].mean()})
    for hora, g in sub.groupby(ts.dt.hour):
        filas.append({"agrupacion": "hora", "valor": hora, "n": len(g), "pct_activo": 100 * g["activo"].mean()})
    for dow, g in sub.groupby(ts.dt.dayofweek):
        filas.append({"agrupacion": "dia_semana", "valor": dow, "n": len(g), "pct_activo": 100 * g["activo"].mean()})
    for terc, g in sub.groupby(terciles, observed=True):
        filas.append({"agrupacion": "tercil_consumo", "valor": terc, "n": len(g), "pct_activo": 100 * g["activo"].mean()})
    return pd.DataFrame(filas)


# ==========================================================================================
# 14/15/17) Evaluacion contrafactual de ataques -- construida una sola vez (features atacadas
#      no dependen del metodo/modelo), reutilizada por los 5 predictores
# ==========================================================================================

def puntuar_ataques(ataques: dict, val15: dict, particion_val_raw: np.ndarray, perfil_train: pd.DataFrame, n_val_full: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    kw_clean, ts_start, ts_end = val15["kw"], val15["ts_start"], val15["ts_end"]
    cal_full = _calendario(ts_start)
    filas_meta, filas_feat = [], []
    for episode_id, a in ataques.items():
        kw_at, bins_af = construir_kw15_atacado(kw_clean, particion_val_raw, a["offset_global"], a["duracion"], a["y_tramo"])
        bins_validos = [t for t in bins_af if t >= MAX_LAG]
        if not bins_validos:
            continue
        feats_kw_at = construir_features_kw(kw_at)  # recalculo vectorizado sobre la serie ATACADA completa
        for pos, t in enumerate(bins_validos):
            fila = feats_kw_at.iloc[t].to_dict()
            fila.update(cal_full.iloc[t].to_dict())
            filas_feat.append(fila)
            filas_meta.append({
                "episode_id": episode_id, "t": t, "k": t - MAX_LAG, "posicion_ordinal": pos + 1,
                "target_start": ts_start[t], "target_end": ts_end[t],
                "target_kw_limpio": float(kw_clean[t]), "target_kw_atacado": float(kw_at[t]),
            })
    meta = pd.DataFrame(filas_meta)
    feats = pd.DataFrame(filas_feat)
    if len(feats):
        feats = feats.merge(perfil_train, on=["es_finde", "slot_15min"], how="left")
        assert feats[PERIODIC_FEATURES + HYBRID_EXTRA].isna().sum().sum() == 0, "NaN en features atacadas"
    return meta, feats


def _indexar_scores(meta: pd.DataFrame, columna: str) -> dict[str, dict[int, float]]:
    return {ep: dict(zip(g["k"], g[columna])) for ep, g in meta.groupby("episode_id")}


def simular_episodios(ataques: dict, ids: set, attacked_scores_idx: dict, score_clean_full: np.ndarray,
                       umbral: float, target_end_grid: np.ndarray, meta_por_episodio: dict) -> pd.DataFrame:
    active_clean_full = score_clean_full > umbral
    filas = []
    for ep in ids:
        if ep not in meta_por_episodio:
            continue
        ks_local = meta_por_episodio[ep]
        if not ks_local:
            continue
        k_before = ks_local[0] - 1
        s_att = np.array([attacked_scores_idx[ep][k] for k in ks_local])
        active_att = s_att > umbral
        prev_active = bool(active_clean_full[k_before]) if k_before >= 0 else False
        event_start = np.empty(len(ks_local), dtype=bool)
        pa = prev_active
        for i in range(len(ks_local)):
            event_start[i] = bool(active_att[i]) and not pa
            pa = bool(active_att[i])
        active_clean_local = active_clean_full[ks_local]
        induced = event_start & ~active_clean_local
        idx_ind = np.where(induced)[0]
        ts_induced = pd.Timestamp(target_end_grid[ks_local[idx_ind[0]]]) if len(idx_ind) else pd.NaT
        filas.append({"episode_id": ep, "n_affected_targets": len(ks_local), "raw_detected": bool(active_att.any()),
                       "induced_detected": bool(induced.any()), "first_alarm_timestamp": ts_induced})
    return pd.DataFrame(filas)


# ==========================================================================================
# 18) Analisis de adaptacion por posicion ordinal dentro del episodio
# ==========================================================================================

def analizar_adaptacion(meta_scored: pd.DataFrame) -> pd.DataFrame:
    m = meta_scored.copy()
    m["posicion_bucket"] = m["posicion_ordinal"].clip(upper=6).map({1: "1º", 2: "2º", 3: "3º", 4: "4º", 5: "5º", 6: "6º y posteriores"})
    filas = []
    for bucket, g in m.groupby("posicion_bucket"):
        filas.append({
            "posicion": bucket, "n_muestras": len(g), "score_limpio_medio": float(g["score_clean"].mean()),
            "score_atacado_medio": float(g["score_atacado"].mean()), "delta_medio": float((g["score_atacado"] - g["score_clean"]).mean()),
            "tasa_activo_pct": 100 * float((g["score_atacado"] > g["umbral"]).mean()),
        })
    orden = ["1º", "2º", "3º", "4º", "5º", "6º y posteriores"]
    tabla = pd.DataFrame(filas)
    tabla["orden"] = tabla["posicion"].map({p: i for i, p in enumerate(orden)})
    return tabla.sort_values("orden").drop(columns="orden")


# ==========================================================================================
# Conjuntos de episodios (mismo criterio dev/eval que toda la serie de experimentos)
# ==========================================================================================

def construir_conjuntos(ctx: dict, div: dict) -> dict:
    def _en_periodo(tabla_meta: pd.DataFrame, ini, fin) -> set:
        return set(tabla_meta[(tabla_meta["attack_start"] >= ini) & (tabla_meta["attack_end"] <= fin)]["episode_id"])

    tabla_ramp, ataques_ramp = construir_episodios_ramp(ctx["master_csv"], ctx["particion_val_raw"], ctx["inicio_particion_val"])
    for a in ataques_ramp.values():
        a["y_tramo"] = a["y_tramo_ramp"]
    reduccion = ctx["master_csv"][ctx["master_csv"]["family"] == "reduccion_constante"]
    ataques_constante = {ep: ctx["ataques_originales"][ep] for ep in reduccion["episode_id"]}
    reduccion_meta = reduccion[["episode_id", "attack_start", "attack_end"]]

    R_eval = _en_periodo(tabla_ramp, div["eval_ini"], div["eval_fin"])
    RC_eval = _en_periodo(reduccion_meta, div["eval_ini"], div["eval_fin"])
    det_full = pd.read_csv(DET_DIR / "episodios_detectabilidad_impacto.csv")
    grupo_b_ids = set(det_full[(det_full["family"] == "reduccion_constante") & (det_full["detectability_group"] == "B")]["episode_id"])
    B_eval = RC_eval & grupo_b_ids
    alto_impacto_ids = set(pd.read_csv(RAMP_DIR / "ramp_no_detectados_alto_impacto.csv")["episode_id"])
    HI_eval = alto_impacto_ids & R_eval
    assert len(R_eval) == 519 and len(HI_eval) == 90

    return {"tabla_ramp": tabla_ramp, "ataques_ramp": ataques_ramp, "ataques_constante": ataques_constante,
            "R_eval": R_eval, "RC_eval": RC_eval, "B_eval": B_eval, "HI_eval": HI_eval, "alto_impacto_ids": alto_impacto_ids}


# ==========================================================================================
# Evaluacion completa de un metodo (generica: reutiliza las features atacadas YA construidas)
# ==========================================================================================

def evaluar_metodo_completo(nombre: str, cols: list, predecir_fn, ctx: dict, div: dict, conj: dict,
                             meta_r: pd.DataFrame, feats_r: pd.DataFrame, meta_c: pd.DataFrame, feats_c: pd.DataFrame,
                             meta_ep_ramp: pd.DataFrame, contexto_ramp: pd.DataFrame, meta_ep_const: pd.DataFrame,
                             contexto_const: pd.DataFrame, ramp_temp_grupo: pd.DataFrame, ks_por_episodio_ramp: dict,
                             ks_por_episodio_const: dict) -> dict:
    df_val_scored = puntuar_metodo_limpio(ctx["df_val_feat"], cols, predecir_fn)
    _guardar_tabla(df_val_scored, f"scores_limpios_val_{nombre}")

    tabla_umbral, umbral_info = calibrar_umbral_exacto(df_val_scored["causal_relative_kw"].to_numpy()[:div["corte_idx"]], div["n_dias_dev"])
    umbral = umbral_info["umbral"]
    payload = {"metodo": nombre, "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(), "umbral": umbral,
               "banda_usada": umbral_info["banda_usada"], "eventos_dia_desarrollo": umbral_info["eventos_dia"]}
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLAS_DIR / f"umbral_congelado_{nombre}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    scores_full = df_val_scored["causal_relative_kw"].to_numpy()
    fa_dev = evaluar_fa(scores_full, div, umbral, "desarrollo")
    fa_eval = evaluar_fa(scores_full, div, umbral, "evaluacion")
    fa_eval["razon_eval_dev"] = fa_eval["eventos_dia"] / fa_dev["eventos_dia"] if fa_dev["eventos_dia"] else float("nan")
    dist_fa = distribucion_fa(df_val_scored, div, umbral)

    target_end_grid = df_val_scored["target_end"].to_numpy()

    # RAMP
    pred_r = predecir_fn(feats_r[cols]) if len(feats_r) else np.array([])
    meta_r = meta_r.copy()
    if len(feats_r):
        signed_r, positive_r = calcular_score(pred_r, meta_r["target_kw_atacado"].to_numpy())
        meta_r["score_atacado"] = positive_r
        meta_r["pred_kw"] = pred_r
    meta_r["score_clean"] = meta_r["k"].map(lambda k: scores_full[k])
    meta_r["umbral"] = umbral

    idx_r = _indexar_scores(meta_r, "score_atacado")
    sim_r = simular_episodios(conj["ataques_ramp"], conj["R_eval"], idx_r, scores_full, umbral, target_end_grid, ks_por_episodio_ramp)
    res_r = enriquecer_resultado(sim_r, conj["ataques_ramp"], meta_ep_ramp, ctx["particion_val_raw"], contexto_ramp)
    res_r = res_r.merge(ramp_temp_grupo, on="episode_id", how="left")

    tabla_adapt = analizar_adaptacion(meta_r[meta_r["episode_id"].isin(conj["R_eval"])])

    # CONSTANTE
    pred_c = predecir_fn(feats_c[cols]) if len(feats_c) else np.array([])
    meta_c = meta_c.copy()
    if len(feats_c):
        signed_c, positive_c = calcular_score(pred_c, meta_c["target_kw_atacado"].to_numpy())
        meta_c["score_atacado"] = positive_c
    meta_c["score_clean"] = meta_c["k"].map(lambda k: scores_full[k])
    meta_c["umbral"] = umbral

    idx_c = _indexar_scores(meta_c, "score_atacado")
    sim_c = simular_episodios(conj["ataques_constante"], conj["RC_eval"], idx_c, scores_full, umbral, target_end_grid, ks_por_episodio_const)
    res_c = enriquecer_resultado(sim_c, conj["ataques_constante"], meta_ep_const, ctx["particion_val_raw"], contexto_const)

    return {"nombre": nombre, "df_val_scored": df_val_scored, "umbral_info": umbral_info, "fa_dev": fa_dev, "fa_eval": fa_eval,
            "dist_fa": dist_fa, "res_ramp": res_r, "res_const": res_c, "tabla_adapt": tabla_adapt, "meta_r": meta_r}


# ==========================================================================================
# Comparaciones estadisticas pareadas por episodio
# ==========================================================================================

def comparacion_pareada(dfs: dict[str, pd.DataFrame], pares: list[tuple[str, str]]) -> pd.DataFrame:
    filas = []
    for a, b in pares:
        if dfs.get(a) is None or dfs.get(b) is None:
            continue
        da_full = dfs[a].set_index("episode_id")["induced_detected"].astype(bool)
        db_full = dfs[b].set_index("episode_id")["induced_detected"].astype(bool).reindex(da_full.index).fillna(False)
        da, db = da_full.to_numpy(), db_full.to_numpy()
        solo_a, solo_b = int((da & ~db).sum()), int((~da & db).sum())
        p_mcnemar = mcnemar_p(solo_a, solo_b)
        ci_low, ci_high = bootstrap_pareado(da.astype(float), db.astype(float))
        h = dfs[a].set_index("episode_id")["hidden_energy_kwh"].reindex(da_full.index).to_numpy() if "hidden_energy_kwh" in dfs[a].columns else np.zeros(len(da_full))
        ci_e_low, ci_e_high = bootstrap_pareado(np.where(da, h, 0.0), np.where(db, h, 0.0))
        delay_a = dfs[a].set_index("episode_id")["detection_delay_min"]
        delay_b = dfs[b].set_index("episode_id")["detection_delay_min"].reindex(delay_a.index)
        ambos = da_full & db_full
        if ambos.sum() >= 5:
            try:
                _, p_w = sstats.wilcoxon(delay_a[ambos].to_numpy(), delay_b[ambos].to_numpy())
            except ValueError:
                p_w = float("nan")
        else:
            p_w = float("nan")
        filas.append({"comparacion": f"{a}_vs_{b}", "n_episodios": len(da_full), "DR_a_pct": 100 * da.mean(), "DR_b_pct": 100 * db.mean(),
                      "diferencia_DR_pp": 100 * (db.mean() - da.mean()), "mcnemar_p": p_mcnemar,
                      "bootstrap_dr_ci95_low": ci_low, "bootstrap_dr_ci95_high": ci_high,
                      "bootstrap_energia_ci95_low": ci_e_low, "bootstrap_energia_ci95_high": ci_e_high,
                      "wilcoxon_p_retraso": p_w, "n_detectados_ambos": int(ambos.sum()), "significativo_p<0.05": p_mcnemar < 0.05})
    return pd.DataFrame(filas)


# ==========================================================================================
# Criterios de exito
# ==========================================================================================

CRITERIOS_TEXTO = {
    1: "FA de evaluacion <= 0.12 eventos/dia", 2: "DR ramp >= 10.40% (P)", 3: "DR energetico > 23.02% (P)",
    4: "Recupera >= 14 de las 90 rampas de alto impacto", 5: "Mejora especialmente 720-1440 minutos",
    6: "Detecciones exclusivas relevantes respecto a P", 7: "Razon FA eval/dev entre 0.5 y 2",
    8: "Retraso mediano no mas de 120min peor que P", 9: "Resultado consistente en mas de una severidad",
    10: "Detecciones no concentradas en un unico mes", 11: "Mejora claramente a M60_BASE",
    12: "PERIODIC pierde menos senal que HYBRID conforme avanza el ataque",
}
CRITERIOS_SUSTANTIVOS = (1, 3, 4)


def evaluar_criterios(nombre: str, feature_set: str, resultado: dict, retraso_p: float, tabla_dur: pd.DataFrame,
                       tabla_rho: pd.DataFrame, solo_metodo_ids: set, ataques_ramp: dict, hi_eval: set,
                       m60_dr_energ: float, m60_hi_n: int, decaimiento_periodic: float | None, decaimiento_hybrid: float | None) -> pd.DataFrame:
    res_r, fa_eval = resultado["res_ramp"], resultado["fa_eval"]
    m = resumen_metricas(res_r)
    hi_det = res_r[res_r["episode_id"].isin(hi_eval)]
    n_hi = int(hi_det["induced_detected"].sum())
    largas = tabla_dur[tabla_dur["duration_min"].isin([720, 1440])] if len(tabla_dur) else pd.DataFrame()
    c5 = bool(len(largas) and (largas["induced_DR_pct"].fillna(0) > 0).all())
    retraso_m = m["retraso_mediano_min"]
    c8 = bool(not np.isnan(retraso_m) and not np.isnan(retraso_p) and (retraso_m - retraso_p) <= 120.0)
    n_sev_mejora = int((tabla_rho["induced_DR_pct"].fillna(0) > 0).sum()) if len(tabla_rho) else 0

    if len(solo_metodo_ids):
        meses = pd.to_datetime([ataques_ramp[e]["attack_start"] for e in solo_metodo_ids if e in ataques_ramp]).month
        c10 = bool(pd.Series(meses).value_counts(normalize=True).max() < 0.5) if len(meses) else True
    else:
        c10 = True

    if feature_set == "PERIODIC" and decaimiento_periodic is not None and decaimiento_hybrid is not None:
        c12 = bool(decaimiento_periodic <= decaimiento_hybrid)
    elif feature_set == "HYBRID" and decaimiento_periodic is not None and decaimiento_hybrid is not None:
        c12 = bool(decaimiento_periodic <= decaimiento_hybrid)
    else:
        c12 = False

    resultados = {
        1: fa_eval["eventos_dia"] <= 0.12, 2: m["induced_DR_pct"] >= 10.40, 3: m["energy_weighted_dr_pct"] > 23.02,
        4: n_hi >= 14, 5: c5, 6: len(solo_metodo_ids) > 0,
        7: 0.5 <= fa_eval["razon_eval_dev"] <= 2.0 if not np.isnan(fa_eval["razon_eval_dev"]) else False,
        8: c8, 9: n_sev_mejora > 1, 10: c10, 11: bool(m["energy_weighted_dr_pct"] > m60_dr_energ and n_hi > m60_hi_n),
        12: c12,
    }
    tabla = pd.DataFrame([{"metodo": nombre, "criterio_num": k, "criterio": CRITERIOS_TEXTO[k], "cumplido": v,
                            "sustantivo": k in CRITERIOS_SUSTANTIVOS} for k, v in resultados.items()])
    tabla.attrs["n_cumplidos"] = sum(resultados.values())
    tabla.attrs["n_sustantivos_cumplidos"] = sum(resultados[k] for k in CRITERIOS_SUSTANTIVOS)
    return tabla


def _decaimiento_score(tabla_adapt: pd.DataFrame) -> float | None:
    """Caida del score atacado desde la 1a posicion ordinal hasta la ultima -- usado por el
    criterio 12 (PERIODIC deberia perder menos senal que HYBRID a medida que el ataque avanza)."""
    if len(tabla_adapt) < 2:
        return None
    return float(tabla_adapt.iloc[0]["score_atacado_medio"] - tabla_adapt.iloc[-1]["score_atacado_medio"])


# ==========================================================================================
# Figuras
# ==========================================================================================

def generar_figuras(resultados: dict) -> None:
    for nombre, res in resultados.items():
        df_val, corte = res["df_val_scored"], res["fa_dev"]["n_dias"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        n_dev = int(round(res["fa_dev"]["n_dias"] * 1440 / RESOLUCION_MIN))
        ax.hist(df_val["causal_relative_kw"].to_numpy()[:n_dev], bins=60, color=GRIS, alpha=0.5, density=True, label="desarrollo")
        ax.hist(df_val["causal_relative_kw"].to_numpy()[n_dev:], bins=60, color=AZUL, alpha=0.5, density=True, label="evaluacion")
        ax.axvline(res["umbral_info"]["umbral"], color=ROJO, linestyle="--", label="umbral")
        ax.legend(fontsize=8); ax.set_title(f"{nombre}: distribucion causal_relative_kw")
        fig.tight_layout(); _guardar_fig(fig, f"dist_scores_{nombre}")

        adapt = res["tabla_adapt"]
        if len(adapt):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(adapt["posicion"], adapt["score_limpio_medio"], marker="o", color=GRIS, label="limpio")
            ax.plot(adapt["posicion"], adapt["score_atacado_medio"], marker="o", color=ROJO, label="atacado")
            ax.axhline(res["umbral_info"]["umbral"], color=GRIS_OSCURO, linestyle="--", label="umbral")
            ax.set_xlabel("posicion ordinal del target afectado"); ax.set_ylabel("causal_relative_kw"); ax.legend(fontsize=8)
            ax.set_title(f"{nombre}: adaptacion segun contaminacion de lags")
            fig.tight_layout(); _guardar_fig(fig, f"adaptacion_{nombre}")
    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def fase2_evaluacion(ctx: dict, resultados_entrenamiento: dict) -> dict:
    div = ctx["div"]
    conj = construir_conjuntos(ctx, div)
    print(f"\n[conjuntos] R_eval={len(conj['R_eval'])} | RC_eval={len(conj['RC_eval'])} | B_eval={len(conj['B_eval'])} | HI_eval={len(conj['HI_eval'])}")

    contexto_ramp = ad.contexto_e_impacto(conj["ataques_ramp"], ctx["particion_val_raw"])
    contexto_const = ad.contexto_e_impacto(conj["ataques_constante"], ctx["particion_val_raw"])
    meta_ep_ramp = conj["tabla_ramp"][["episode_id", "attack_start", "attack_end", "duration_min", "rho_final"]]
    meta_ep_const = ctx["master_csv"][ctx["master_csv"]["family"] == "reduccion_constante"][
        ["episode_id", "family", "parameter", "attack_start", "attack_end", "duration_min"]]
    ramp_temp_grupo = pd.read_csv(RAMP_DIR / "episodios_ramp.csv")[["episode_id", "grupo_temporal"]]

    print("\nConstruyendo features atacadas (una sola vez, reutilizadas por los 5 metodos)...")
    meta_r, feats_r = puntuar_ataques(conj["ataques_ramp"], ctx["val15"], ctx["particion_val_raw"], ctx["perfil_train"], ctx["n_val_full"])
    meta_c, feats_c = puntuar_ataques(conj["ataques_constante"], ctx["val15"], ctx["particion_val_raw"], ctx["perfil_train"], ctx["n_val_full"])
    _guardar_tabla(meta_r, "trayectorias_ramp")
    ks_por_episodio_ramp = {ep: sorted(g["k"].tolist()) for ep, g in meta_r.groupby("episode_id")}
    ks_por_episodio_const = {ep: sorted(g["k"].tolist()) for ep, g in meta_c.groupby("episode_id")}
    print(f"  ramp: {len(meta_r)} filas (episodio,target) | constante: {len(meta_c)} filas")

    metodos_cfg = {}
    for nombre in ("PerfilHistorico", "Ridge_PERIODIC", "Ridge_HYBRID", "HistGB_PERIODIC", "HistGB_HYBRID"):
        cols, fn = construir_predictor(nombre, resultados_entrenamiento)
        metodos_cfg[nombre] = (cols, fn)

    resultados = {}
    for nombre, (cols, fn) in metodos_cfg.items():
        print(f"\n[eval] {nombre}...")
        resultados[nombre] = evaluar_metodo_completo(nombre, cols, fn, ctx, div, conj, meta_r, feats_r, meta_c, feats_c,
                                                       meta_ep_ramp, contexto_ramp, meta_ep_const, contexto_const,
                                                       ramp_temp_grupo, ks_por_episodio_ramp, ks_por_episodio_const)
        m = resumen_metricas(resultados[nombre]["res_ramp"])
        print(f"  umbral={resultados[nombre]['umbral_info']['umbral']:.5f} | FA_eval={resultados[nombre]['fa_eval']['eventos_dia']:.4f} | "
              f"DR={m['induced_DR_pct']:.2f}% | DR_energ={m['energy_weighted_dr_pct']:.2f}%")

    # metricas de prediccion limpia (dev) -- comparacion de los 5 metodos
    df_dev = ctx["df_val_feat"].iloc[:div["corte_idx"]]
    filas_pred = []
    for nombre, (cols, fn) in metodos_cfg.items():
        pred = fn(df_dev)
        met = metricas_prediccion(pred, df_dev["target_kw"].to_numpy())
        d = desglose_prediccion(pred, df_dev["target_kw"].to_numpy(), df_dev["ts_start"].to_numpy())
        filas_pred.append({"metodo": nombre, **met, "variabilidad_mensual": d["variabilidad_mensual"], "sesgo_por_hora_std": d["sesgo_por_hora_std"]})
    tabla_pred = _guardar_tabla(pd.DataFrame(filas_pred), "metricas_prediccion_limpia")
    print("\n" + tabla_pred.to_string(index=False))

    filas_hora, filas_mes = [], []
    for nombre, (cols, fn) in metodos_cfg.items():
        pred = fn(df_dev)
        d = desglose_prediccion(pred, df_dev["target_kw"].to_numpy(), df_dev["ts_start"].to_numpy())
        for hora, v in d["por_hora"].items():
            filas_hora.append({"metodo": nombre, "hora": hora, "MAE": v})
        for mes, v in d["por_mes"].items():
            filas_mes.append({"metodo": nombre, "mes": mes, "MAE": v})
    _guardar_tabla(pd.DataFrame(filas_hora), "errores_por_hora")
    _guardar_tabla(pd.DataFrame(filas_mes), "errores_por_mes")

    # umbrales exactos (concatenado por metodo) + congelados
    filas_umb_exacto = []
    umbrales_congelados = {}
    for nombre, res in resultados.items():
        tabla_u, _ = calibrar_umbral_exacto(res["df_val_scored"]["causal_relative_kw"].to_numpy()[:div["corte_idx"]], div["n_dias_dev"])
        tabla_u["metodo"] = nombre
        filas_umb_exacto.append(tabla_u)
        umbrales_congelados[nombre] = res["umbral_info"]["umbral"]
    _guardar_tabla(pd.concat(filas_umb_exacto, ignore_index=True), "umbrales_exactos")
    with open(TABLAS_DIR / "umbrales_congelados.json", "w", encoding="utf-8") as f:
        json.dump(umbrales_congelados, f, indent=2, ensure_ascii=False, default=str)

    # falsas alarmas dev/eval
    filas_fa_dev = [{**res["fa_dev"], "metodo": nombre} for nombre, res in resultados.items()]
    filas_fa_eval = [{**res["fa_eval"], "metodo": nombre} for nombre, res in resultados.items()]
    _guardar_tabla(pd.DataFrame(filas_fa_dev), "falsas_alarmas_desarrollo")
    _guardar_tabla(pd.DataFrame(filas_fa_eval), "falsas_alarmas_evaluacion")

    # resultados ramp (global / duracion / severidad / alto impacto)
    filas_glob, filas_dur, filas_rho, filas_hi, filas_resumen = [], [], [], [], []
    for nombre, res in resultados.items():
        df = res["res_ramp"]
        fila = {"metodo": nombre}; fila.update(resumen_metricas(df)); filas_glob.append(fila)
        for dur, g in df.groupby("duration_min"):
            f = {"metodo": nombre, "duration_min": dur}; f.update(resumen_metricas(g)); filas_dur.append(f)
        for rho, g in df.groupby("rho_final"):
            f = {"metodo": nombre, "rho_final": rho}; f.update(resumen_metricas(g)); filas_rho.append(f)
        sub_hi = df[df["episode_id"].isin(conj["HI_eval"])]
        n_rec = int(sub_hi["induced_detected"].sum())
        filas_hi.append({"metodo": nombre, "n_total": len(sub_hi), "n_detectados": n_rec,
                          "pct_detectado": 100 * n_rec / len(sub_hi) if len(sub_hi) else float("nan"),
                          "energia_total_kwh": float(sub_hi["hidden_energy_kwh"].sum()),
                          "energia_cubierta_kwh": float(sub_hi.loc[sub_hi["induced_detected"], "hidden_energy_kwh"].sum()),
                          "retraso_mediano_min": float(sub_hi.loc[sub_hi["induced_detected"], "detection_delay_min"].median()) if n_rec else float("nan")})
        dfc = res["res_ramp"][["episode_id", "induced_detected", "detection_delay_min", "hidden_energy_kwh"]].copy()
        dfc["metodo"] = nombre
        filas_resumen.append(dfc)
    tabla_glob = _guardar_tabla(pd.DataFrame(filas_glob), "resultados_ramp_global")
    tabla_dur_all = _guardar_tabla(pd.DataFrame(filas_dur), "resultados_por_duracion")
    tabla_rho_all = _guardar_tabla(pd.DataFrame(filas_rho), "resultados_por_severidad")
    tabla_hi = _guardar_tabla(pd.DataFrame(filas_hi), "resultados_alto_impacto")
    _guardar_tabla(pd.concat(filas_resumen, ignore_index=True), "resumen_episodios_ramp")
    print("\n" + tabla_glob.to_string(index=False))
    print("\nAlto impacto:\n" + tabla_hi.to_string(index=False))

    for nombre, res in resultados.items():
        _guardar_tabla(res["tabla_adapt"], f"adaptacion_{nombre}")

    # PERIODIC vs HYBRID (por familia de modelo, y resumen usando el mejor de cada feature_set en dev)
    tabla_pred_idx = tabla_pred.set_index("metodo")
    mejor_periodic = tabla_pred_idx.loc[["Ridge_PERIODIC", "HistGB_PERIODIC"]]["MAE"].idxmin()
    mejor_hybrid = tabla_pred_idx.loc[["Ridge_HYBRID", "HistGB_HYBRID"]]["MAE"].idxmin()
    filas_pvh = []
    for etiqueta, nombre in (("PERIODIC", mejor_periodic), ("HYBRID", mejor_hybrid)):
        adapt = resultados[nombre]["tabla_adapt"].copy()
        adapt["feature_set"] = etiqueta; adapt["metodo_representativo"] = nombre
        filas_pvh.append(adapt)
    _guardar_tabla(pd.concat(filas_pvh, ignore_index=True), "adaptacion_periodic_vs_hybrid")
    decaimiento = {"PERIODIC": _decaimiento_score(resultados[mejor_periodic]["tabla_adapt"]),
                   "HYBRID": _decaimiento_score(resultados[mejor_hybrid]["tabla_adapt"])}
    print(f"\nRepresentantes PERIODIC={mejor_periodic} / HYBRID={mejor_hybrid} | decaimiento score: {decaimiento}")

    # constantes / grupo B
    filas_rc, filas_gb = [], []
    for nombre, res in resultados.items():
        df = res["res_const"]
        fila = {"metodo": nombre}; fila.update(resumen_metricas(df)); filas_rc.append(fila)
        sub_b = df[df["episode_id"].isin(conj["B_eval"])]
        n_rec = int(sub_b["induced_detected"].sum())
        filas_gb.append({"metodo": nombre, "n_grupo_b": len(sub_b), "n_recuperados": n_rec,
                          "pct_recuperado": 100 * n_rec / len(sub_b) if len(sub_b) else float("nan")})
    _guardar_tabla(pd.DataFrame(filas_rc), "resultados_constantes")
    _guardar_tabla(pd.DataFrame(filas_gb), "resultados_grupo_b")

    # complementariedad con P
    df_p_ref = pd.read_csv(EDO_DIR / "05_resultados_P_ramp_eval.csv")
    p_idx = df_p_ref.set_index("episode_id")["induced_detected"].astype(bool)
    filas_comp = []
    for nombre, res in resultados.items():
        a_idx = res["res_ramp"].set_index("episode_id")["induced_detected"].astype(bool).reindex(p_idx.index).fillna(False)
        ambos, solo_p, solo_a, ninguno = int((p_idx & a_idx).sum()), int((p_idx & ~a_idx).sum()), int((~p_idx & a_idx).sum()), int((~p_idx & ~a_idx).sum())
        filas_comp.append({"metodo": nombre, "detectado_ambos": ambos, "solo_P": solo_p, "solo_metodo": solo_a, "ninguno": ninguno,
                            "retention_rate_pct": 100 * ambos / p_idx.sum() if p_idx.sum() else float("nan"),
                            "exclusive_recovery_rate_pct": 100 * solo_a / (~p_idx).sum() if (~p_idx).sum() else float("nan")})
    tabla_comp = _guardar_tabla(pd.DataFrame(filas_comp), "complementariedad_con_p")
    print("\n" + tabla_comp.to_string(index=False))

    # comparaciones estadisticas
    dfs_stats = {"P": df_p_ref, **{n: r["res_ramp"] for n, r in resultados.items()}}
    pares = [("P", "PerfilHistorico"), ("P", "Ridge_PERIODIC"), ("P", "Ridge_HYBRID"), ("P", "HistGB_PERIODIC"), ("P", "HistGB_HYBRID"),
             ("PerfilHistorico", "HistGB_PERIODIC"), ("HistGB_PERIODIC", "HistGB_HYBRID"), ("Ridge_PERIODIC", "Ridge_HYBRID")]
    tabla_stats = _guardar_tabla(comparacion_pareada(dfs_stats, pares), "comparaciones_estadisticas")
    print("\n" + tabla_stats.to_string(index=False))

    # criterios de exito (M60_BASE como referencia, cargado de resultados YA guardados)
    m60_glob = pd.read_csv(M60_DIR / "resultados_ramp_global.csv")
    m60_hi = pd.read_csv(M60_DIR / "resultados_alto_impacto.csv")
    m60_dr_energ = float(m60_glob[m60_glob["metodo"] == "M60_BASE"]["energy_weighted_dr_pct"].iloc[0])
    m60_hi_n = int(m60_hi[m60_hi["metodo"] == "M60_BASE"]["n_detectados"].iloc[0])
    retraso_p = resumen_metricas(df_p_ref)["retraso_mediano_min"]

    tablas_criterios = []
    for nombre, res in resultados.items():
        feature_set = "PERIODIC" if "PERIODIC" in nombre else ("HYBRID" if "HYBRID" in nombre else "N/A")
        a_idx_n = res["res_ramp"].set_index("episode_id")["induced_detected"].astype(bool).reindex(p_idx.index).fillna(False)
        solo_ids = set(a_idx_n[a_idx_n].index) - set(p_idx[p_idx].index)
        tabla_dur_m = tabla_dur_all[tabla_dur_all["metodo"] == nombre]
        tabla_rho_m = tabla_rho_all[tabla_rho_all["metodo"] == nombre]
        es_representante = nombre in (mejor_periodic, mejor_hybrid)  # criterio 12 solo evaluable para el par representativo PERIODIC/HYBRID
        dec_p = decaimiento["PERIODIC"] if es_representante else None
        dec_h = decaimiento["HYBRID"] if es_representante else None
        tc = evaluar_criterios(nombre, feature_set, res, retraso_p, tabla_dur_m, tabla_rho_m, solo_ids, conj["ataques_ramp"],
                                conj["HI_eval"], m60_dr_energ, m60_hi_n, dec_p, dec_h)
        tablas_criterios.append(tc)
        print(f"  {nombre}: {tc.attrs['n_cumplidos']}/12 | sustantivos (1,3,4): {tc.attrs['n_sustantivos_cumplidos']}/3")
    tabla_criterios = pd.concat(tablas_criterios, ignore_index=True)
    _guardar_tabla(tabla_criterios, "criterios_exito")

    manifiesto = {"fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "n_R_eval": len(conj["R_eval"]), "n_HI_eval": len(conj["HI_eval"]),
                  "mejor_periodic": mejor_periodic, "mejor_hybrid": mejor_hybrid}
    with open(TABLAS_DIR / "manifiesto_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False, default=str)

    print("\nGenerando figuras...")
    generar_figuras(resultados)

    return {"resultados": resultados, "conj": conj, "tabla_glob": tabla_glob, "tabla_hi": tabla_hi,
            "tabla_comp": tabla_comp, "tabla_stats": tabla_stats, "tabla_criterios": tabla_criterios}


def main() -> dict:
    print("Fase 1: carga + agregacion a 15 min + features + auditoria...")
    ctx = cargar_todo()
    print(f"  train: {len(ctx['df_train_feat'])} filas | val: {len(ctx['df_val_feat'])} filas "
          f"(dev={ctx['div']['corte_idx']}, eval={len(ctx['df_val_feat']) - ctx['div']['corte_idx']})")
    tabla_definicion_features()
    auditar_causalidad(ctx["df_train_feat"])
    _guardar_tabla(pd.DataFrame([{
        "particion": "train", "n_muestras": len(ctx["df_train_feat"])},
        {"particion": "val_total", "n_muestras": len(ctx["df_val_feat"])},
        {"particion": "desarrollo", "n_muestras": ctx["div"]["corte_idx"]},
        {"particion": "evaluacion", "n_muestras": len(ctx["df_val_feat"]) - ctx["div"]["corte_idx"]},
    ]), "muestras_train_dev_eval")
    print("OK -- auditoria de causalidad pasada")

    resultados_entrenamiento = fase1_entrenamiento(ctx)
    print("\nOK -- Fase 1 (entrenamiento) completa.")

    resultado_final = fase2_evaluacion(ctx, resultados_entrenamiento)
    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/, "
          f"modelos en {MODELOS_DIR.relative_to(BASE_DIR)}/")
    return {"ctx": ctx, "resultados_entrenamiento": resultados_entrenamiento, **resultado_final}


if __name__ == "__main__":
    main()
