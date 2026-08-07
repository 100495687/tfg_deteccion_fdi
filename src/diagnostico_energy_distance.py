"""Diagnostico inicial: ¿contiene el vector completo de 360 residuos de reconstruccion mas
informacion sobre `ramp_lineal` que su media (`relative_kw`)?

Hipotesis: una rampa puede modificar la forma de la distribucion interna de los residuos de
una ventana (dispersion, asimetria, cola positiva) aunque su media (`relative_kw`) siga
dentro del rango normal. Se compara, ventana a ventana, la distancia de energia
(`scipy.stats.energy_distance`) entre los 360 residuos de cada ventana y una referencia limpia
fija (train), tanto para el residuo firmado como para su parte positiva.

Es puramente diagnostico: no calibra ningun umbral operativo, no declara alarmas y no se
convierte en un detector. `delta_ED` y `delta_relative_kw` son contrafactuales y no se usan
nunca como entrada de ningun calculo salvo el propio diagnostico.

No reentrena nada, no cambia ventana/stride/relative_kw/checkpoint, no genera ataques nuevos
(reutiliza literalmente `experimento_ramp.construir_episodios_ramp`, deterministico). Reutiliza
sin modificar base_relative.{cargar_ataques_y_modelo_360, calcular_relative_kw},
experimento_ramp.{construir_episodios_ramp, _racha_maxima, RHO_FINALES, COLOR_GRUPO},
experimento_scores.EPSILON_KW, experimento_stride.{_ventana_atacada, _ventanas_solapadas,
construir_rejilla} y normalization.{aplicar_zscore, revertir_zscore}.

Ejecutable de forma aislada:
    python -m src.diagnostico_energy_distance
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.metrics import average_precision_score, roc_auc_score

from src.base_relative import cargar_ataques_y_modelo_360, calcular_relative_kw
from src.data_loading import COLUMNA_OBJETIVO
from src.experimento_ramp import COLOR_GRUPO, RHO_FINALES, _racha_maxima, construir_episodios_ramp
from src.experimento_scores import EPSILON_KW
from src.experimento_stride import _ventana_atacada, _ventanas_solapadas, construir_rejilla
from src.normalization import aplicar_zscore, revertir_zscore

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "diagnostico_energy_distance"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "diagnostico_energy_distance"
RAMP_DIR = BASE_DIR / "results" / "tables" / "experimento_ramp"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDE = 60
SEMILLA = 42
MAX_RESIDUOS_REF = 50_000
PASO_SUBMUESTREO_REF = 6  # 1 de cada 6 ventanas horarias de train -> separadas 6h == TAMANO_VENTANA
PERCENTILES_REF = (1, 5, 25, 50, 75, 90, 95, 99)
LAGS_HORAS = (1, 6, 12, 24, 168)
DURACION_BUCKETS = {"30_120min": (30, 120), "240_360min": (240, 360), "720_1440min": (720, 1440)}


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 1/3) Setup + residuos por minuto (formula EXACTA de relative_kw, sin colapsar a la media)
# ==========================================================================================

def cargar_todo() -> dict:
    config, datos, modelo, ataques_originales, master_csv = cargar_ataques_y_modelo_360()
    params_norm = datos["params_norm"]
    particion_train = datos["particiones"]["train"]
    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    inicio_particion = particion_val.index[0]
    grid_60 = construir_rejilla(particion_val, STRIDE)
    return {
        "config": config, "datos": datos, "modelo": modelo, "ataques_originales": ataques_originales,
        "master_csv": master_csv, "params_norm": params_norm, "particion_train": particion_train,
        "particion_val": particion_val, "particion_val_raw": particion_val_raw,
        "inicio_particion": inicio_particion, "grid_60": grid_60,
    }


def calcular_residuos_por_minuto(X_norm: np.ndarray, X_hat_norm: np.ndarray, params_norm: dict,
                                  epsilon_kw: float = EPSILON_KW) -> tuple[np.ndarray, np.ndarray]:
    """signed_relative_residual = (reconstruccion_kw - observado_kw) / max(|reconstruccion_kw|, eps).
    positive_relative_residual = max(0, signed). Ambos desnormalizados a kW ANTES de dividir --
    NUNCA se opera sobre valores normalizados. mean(positive, axis=1) reproduce relative_kw."""
    X_kw = revertir_zscore(X_norm, params_norm)
    X_hat_kw = revertir_zscore(X_hat_norm, params_norm)
    denominador = np.maximum(np.abs(X_hat_kw), epsilon_kw)
    signed = (X_hat_kw - X_kw) / denominador
    positive = np.maximum(0.0, signed)
    return signed, positive


def _skew_filas(X: np.ndarray) -> np.ndarray:
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, np.mean((X - m) ** 3, axis=1) / np.where(s > 0, s, 1.0) ** 3, 0.0)


# ==========================================================================================
# 4) Referencia limpia global (SOLO train, ventanas separadas 6h entre si, sin solape)
# ==========================================================================================

def construir_referencia_limpia(particion_train: pd.DataFrame, modelo, params_norm: dict, seed: int = SEMILLA) -> dict:
    grid_train_60 = construir_rejilla(particion_train, STRIDE)  # ventanas horarias de train (360/60)
    idx_ref = np.arange(0, grid_train_60["X"].shape[0], PASO_SUBMUESTREO_REF)
    X_ref = grid_train_60["X"][idx_ref]
    inicio_ref, fin_ref = grid_train_60["inicio"][idx_ref], grid_train_60["fin"][idx_ref]

    X_ref_norm = aplicar_zscore(X_ref, params_norm)
    X_hat_ref = modelo.reconstruir(X_ref_norm)
    signed_ref_win, positive_ref_win = calcular_residuos_por_minuto(X_ref_norm, X_hat_ref, params_norm)

    signed_flat = signed_ref_win.reshape(-1)
    positive_flat = positive_ref_win.reshape(-1)
    n_original = int(signed_flat.shape[0])

    rng = np.random.default_rng(seed)
    if n_original > MAX_RESIDUOS_REF:
        idx_muestra = np.sort(rng.choice(n_original, size=MAX_RESIDUOS_REF, replace=False))
        reference_signed, reference_positive = signed_flat[idx_muestra], positive_flat[idx_muestra]
    else:
        reference_signed, reference_positive = signed_flat, positive_flat

    return {
        "n_ventanas": int(X_ref.shape[0]), "n_residuos_originales": n_original,
        "n_residuos_usados": int(len(reference_signed)),
        "reference_signed": reference_signed, "reference_positive": reference_positive,
        "inicio_ref": inicio_ref, "fin_ref": fin_ref,
    }


def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def estadisticos_distribucion(x: np.ndarray) -> dict:
    pcts = np.percentile(x, PERCENTILES_REF)
    d = {"media": float(x.mean()), "mediana": float(np.median(x)), "desviacion": float(x.std()), "mad": _mad(x)}
    for p, v in zip(PERCENTILES_REF, pcts):
        d[f"p{p}"] = float(v)
    m, s = x.mean(), x.std()
    d["asimetria"] = float(np.mean((x - m) ** 3) / s ** 3) if s else 0.0
    d["minimo"], d["maximo"] = float(x.min()), float(x.max())
    return d


def tabla_referencia(referencia: dict) -> pd.DataFrame:
    filas = []
    for nombre, arr in (("signed", referencia["reference_signed"]), ("positive", referencia["reference_positive"])):
        fila = {"tipo_residuo": nombre, "n_ventanas_utilizadas": referencia["n_ventanas"],
                "n_residuos_originales": referencia["n_residuos_originales"], "n_residuos_usados": referencia["n_residuos_usados"]}
        fila.update(estadisticos_distribucion(arr))
        filas.append(fila)
    return pd.DataFrame(filas)


# ==========================================================================================
# 5) Energy distance (scipy.stats.energy_distance, sin reimplementar). Convencion: para
#    u_values~X, v_values~Y, energy_distance(u,v) = sqrt(2*E|X-Y| - E|X-X'| - E|Y-Y'|), con
#    X,X' iid de u y Y,Y' iid de v (estimador empirico via las muestras dadas). Es simetrica,
#    no negativa, y 0 solo si las dos muestras provienen de la misma distribucion.
# ==========================================================================================

def energy_distance_lote(residuos_ventanas: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    return np.array([sstats.energy_distance(residuos_ventanas[i], referencia) for i in range(residuos_ventanas.shape[0])])


# ==========================================================================================
# 6) Secuencia limpia de validacion (ED + estadisticos por ventana, SIN calibrar umbral)
# ==========================================================================================

def construir_secuencia_val(ctx: dict, referencia: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    grid_60 = ctx["grid_60"]
    X_norm = aplicar_zscore(grid_60["X"], ctx["params_norm"])
    X_hat = ctx["modelo"].reconstruir(X_norm)
    signed, positive = calcular_residuos_por_minuto(X_norm, X_hat, ctx["params_norm"])

    relative_kw = positive.mean(axis=1)
    relative_kw_ref = calcular_relative_kw(X_norm, X_hat, ctx["params_norm"])
    assert np.allclose(relative_kw, relative_kw_ref, atol=1e-9), "relative_kw no coincide con mean(positive_residuals)"

    ed_signed = energy_distance_lote(signed, referencia["reference_signed"])
    ed_positive = energy_distance_lote(positive, referencia["reference_positive"])

    df = pd.DataFrame({
        "k": np.arange(grid_60["X"].shape[0]), "timestamp": grid_60["fin"],
        "relative_kw": relative_kw, "ED_signed": ed_signed, "ED_positive": ed_positive,
        "media_residuo_firmado": signed.mean(axis=1), "mediana_residuo_firmado": np.median(signed, axis=1),
        "desviacion_residuo_firmado": signed.std(axis=1), "asimetria_residuo_firmado": _skew_filas(signed),
        "p90_residuo_firmado": np.percentile(signed, 90, axis=1), "p99_residuo_firmado": np.percentile(signed, 99, axis=1),
        "proporcion_residuos_positivos": (signed > 0).mean(axis=1), "consumo_medio_kw": grid_60["X"].mean(axis=1),
    })
    return df, signed, positive


def analizar_estabilidad_temporal(df_val: pd.DataFrame) -> dict:
    ts = pd.to_datetime(df_val["timestamp"])
    df = df_val.copy()
    df["mes"] = ts.dt.month
    df["hora"] = ts.dt.hour

    mensual = df.groupby("mes")[["ED_signed", "ED_positive"]].agg(["mean", "std", "median"])
    mensual.columns = ["_".join(c) for c in mensual.columns]
    mensual = mensual.reset_index()

    horario = df.groupby("hora")[["ED_signed", "ED_positive"]].agg(["mean", "std", "median"])
    horario.columns = ["_".join(c) for c in horario.columns]
    horario = horario.reset_index()

    filas_autocorr = []
    for lag_h in LAGS_HORAS:
        filas_autocorr.append({
            "lag_horas": lag_h, "autocorr_ED_signed": float(df["ED_signed"].autocorr(lag=lag_h)),
            "autocorr_ED_positive": float(df["ED_positive"].autocorr(lag=lag_h)),
        })
    autocorr = pd.DataFrame(filas_autocorr)

    correlaciones = pd.DataFrame([{
        "pearson_ED_signed_vs_relative_kw": float(sstats.pearsonr(df["ED_signed"], df["relative_kw"])[0]),
        "spearman_ED_signed_vs_relative_kw": float(sstats.spearmanr(df["ED_signed"], df["relative_kw"])[0]),
        "pearson_ED_positive_vs_relative_kw": float(sstats.pearsonr(df["ED_positive"], df["relative_kw"])[0]),
        "spearman_ED_positive_vs_relative_kw": float(sstats.spearmanr(df["ED_positive"], df["relative_kw"])[0]),
        "pearson_ED_signed_vs_consumo": float(sstats.pearsonr(df["ED_signed"], df["consumo_medio_kw"])[0]),
        "pearson_ED_positive_vs_consumo": float(sstats.pearsonr(df["ED_positive"], df["consumo_medio_kw"])[0]),
    }])
    return {"mensual": mensual, "horario": horario, "autocorrelacion": autocorr, "correlaciones": correlaciones}


# ==========================================================================================
# 7) Evaluacion contrafactual sobre ramp_lineal (rama limpia indexada de la secuencia val,
#    rama atacada puntuada una unica vez, batched, para TODAS las ventanas de los 1050 episodios)
# ==========================================================================================

def construir_ataques_ramp(ctx: dict) -> tuple[pd.DataFrame, dict]:
    tabla_ramp, ataques_ramp = construir_episodios_ramp(ctx["master_csv"], ctx["particion_val_raw"], ctx["inicio_particion"])
    for a in ataques_ramp.values():
        a["y_tramo"] = a["y_tramo_ramp"]  # alias para reutilizar _ventana_atacada sin modificarlo
    return tabla_ramp, ataques_ramp


def puntuar_ventanas_ramp(ataques_ramp: dict, particion_val_raw: np.ndarray, grid_60: dict, modelo, params_norm: dict) -> pd.DataFrame:
    n_grid = grid_60["X"].shape[0]
    filas_meta, ventanas = [], []
    for episode_id, a in ataques_ramp.items():
        ks = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_grid))
        for k in ks:
            w_start = k * STRIDE
            ventanas.append(_ventana_atacada(particion_val_raw, w_start, a["offset_global"], a["duracion"], a["y_tramo"]))
            filas_meta.append({"episode_id": episode_id, "k": k, "window_end": grid_60["fin"][k]})

    meta = pd.DataFrame(filas_meta)
    X_norm = aplicar_zscore(np.stack(ventanas).astype(np.float32), params_norm)
    X_hat = modelo.reconstruir(X_norm)
    signed, positive = calcular_residuos_por_minuto(X_norm, X_hat, params_norm)

    meta["relative_kw_ramp"] = positive.mean(axis=1)
    meta["relative_kw_ramp_check"] = calcular_relative_kw(X_norm, X_hat, params_norm)
    assert np.allclose(meta["relative_kw_ramp"], meta["relative_kw_ramp_check"], atol=1e-9)
    meta = meta.drop(columns=["relative_kw_ramp_check"])
    return meta, signed, positive


def enlazar_con_rama_limpia(meta: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
    """Une (por k) la rama limpia YA calculada en `construir_secuencia_val` -- nunca se
    recalcula el modelo para la rama limpia una segunda vez."""
    limpio = df_val.set_index("k")[["relative_kw", "ED_signed", "ED_positive"]].rename(
        columns={"relative_kw": "relative_kw_clean", "ED_signed": "ED_signed_clean", "ED_positive": "ED_positive_clean"})
    t = meta.join(limpio, on="k")
    t["delta_relative_kw"] = t["relative_kw_ramp"] - t["relative_kw_clean"]
    t["delta_ED_signed"] = t["ED_signed_ramp"] - t["ED_signed_clean"]
    t["delta_ED_positive"] = t["ED_positive_ramp"] - t["ED_positive_clean"]
    return t


# ==========================================================================================
# 8) Resumen por episodio
# ==========================================================================================

def resumen_por_episodio(meta_completo: pd.DataFrame, tabla_ramp_meta: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for ep, g in meta_completo.groupby("episode_id"):
        g = g.sort_values("k").reset_index(drop=True)
        n = len(g)
        delta_signed = g["delta_ED_signed"].to_numpy()
        delta_positive = g["delta_ED_positive"].to_numpy()
        idx_max_ed = int(np.argmax(g["ED_signed_ramp"].to_numpy()))
        filas.append({
            "episode_id": ep,
            "max_relative_kw_clean": float(g["relative_kw_clean"].max()), "max_relative_kw_ramp": float(g["relative_kw_ramp"].max()),
            "max_ED_signed_clean": float(g["ED_signed_clean"].max()), "max_ED_signed_ramp": float(g["ED_signed_ramp"].max()),
            "max_ED_positive_clean": float(g["ED_positive_clean"].max()), "max_ED_positive_ramp": float(g["ED_positive_ramp"].max()),
            "max_delta_relative_kw": float(g["delta_relative_kw"].max()),
            "max_delta_ED_signed": float(delta_signed.max()), "max_delta_ED_positive": float(delta_positive.max()),
            "mean_delta_ED_signed": float(delta_signed.mean()), "mean_delta_ED_positive": float(delta_positive.mean()),
            "pct_ventanas_delta_ED_signed_positivo": 100 * float((delta_signed > 0).mean()),
            "pct_ventanas_delta_ED_positive_positivo": 100 * float((delta_positive > 0).mean()),
            "racha_max_delta_ED_signed_positivo": _racha_maxima(delta_signed > 0),
            "racha_max_delta_ED_positive_positivo": _racha_maxima(delta_positive > 0),
            "timestamp_maximo_ED": g.iloc[idx_max_ed]["window_end"], "posicion_relativa_maximo_ED": idx_max_ed / max(n - 1, 1),
            "n_ventanas_afectadas": n,
        })
    resumen = pd.DataFrame(filas)

    meta_join = tabla_ramp_meta[["episode_id", "hidden_energy_ramp_kwh", "hidden_energy_ramp_pct", "duration_min",
                                  "rho_final", "grupo_temporal", "ramp_induced_detected"]].copy()
    p75_energia = float(meta_join["hidden_energy_ramp_kwh"].quantile(0.75))
    meta_join["alto_impacto"] = meta_join["hidden_energy_ramp_kwh"] >= p75_energia
    meta_join = meta_join.rename(columns={"hidden_energy_ramp_kwh": "hidden_energy_kwh", "hidden_energy_ramp_pct": "hidden_energy_pct"})
    resumen = resumen.merge(meta_join, on="episode_id")
    resumen.attrs["p75_hidden_energy_kwh"] = p75_energia
    return resumen


# ==========================================================================================
# 9) Comparaciones principales (mediana, IQR, Mann-Whitney/Wilcoxon, Cliff's delta,
#    bootstrap CI95%, ROC-AUC/PR-AUC como diagnostico de separabilidad -- NO un detector)
# ==========================================================================================

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u, _ = sstats.mannwhitneyu(x, y, alternative="two-sided")
    return float(2 * u / (len(x) * len(y)) - 1)


def bootstrap_ci_diferencia(x: np.ndarray, y: np.ndarray, pareado: bool, n_boot: int = 3000, seed: int = SEMILLA) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if pareado:
        d = y - x
        n = len(d)
        if n == 0:
            return float("nan"), float("nan")
        boot = np.array([np.median(d[rng.integers(0, n, n)]) for _ in range(n_boot)])
    else:
        nx, ny = len(x), len(y)
        if nx == 0 or ny == 0:
            return float("nan"), float("nan")
        boot = np.array([np.median(y[rng.integers(0, ny, ny)]) - np.median(x[rng.integers(0, nx, nx)]) for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def comparar_grupos(nombre: str, x: np.ndarray, y: np.ndarray, pareado: bool) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if pareado:
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
    resultado = {
        "comparacion": nombre, "pareado": pareado, "n_x": len(x), "n_y": len(y),
        "mediana_x": float(np.median(x)) if len(x) else float("nan"), "mediana_y": float(np.median(y)) if len(y) else float("nan"),
        "iqr_x": float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else float("nan"),
        "iqr_y": float(np.percentile(y, 75) - np.percentile(y, 25)) if len(y) else float("nan"),
    }
    if len(x) == 0 or len(y) == 0:
        resultado.update({"tipo_prueba": "sin_datos", "estadistico": float("nan"), "p_valor": float("nan"),
                           "cliffs_delta": float("nan"), "bootstrap_diff_mediana_ci95_low": float("nan"),
                           "bootstrap_diff_mediana_ci95_high": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan")})
        return resultado

    if pareado:
        try:
            stat, p = sstats.wilcoxon(x, y)
        except ValueError:
            stat, p = float("nan"), float("nan")
        resultado["tipo_prueba"] = "wilcoxon"
    else:
        try:
            stat, p = sstats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), float("nan")
        resultado["tipo_prueba"] = "mann_whitney"
    resultado["estadistico"], resultado["p_valor"] = float(stat), float(p)
    resultado["cliffs_delta"] = cliffs_delta(x, y)
    ci_low, ci_high = bootstrap_ci_diferencia(x, y, pareado)
    resultado["bootstrap_diff_mediana_ci95_low"], resultado["bootstrap_diff_mediana_ci95_high"] = ci_low, ci_high

    if not pareado:
        etiquetas = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
        valores = np.concatenate([x, y])
        try:
            resultado["roc_auc"] = float(roc_auc_score(etiquetas, valores))
            resultado["pr_auc"] = float(average_precision_score(etiquetas, valores))
        except ValueError:
            resultado["roc_auc"], resultado["pr_auc"] = float("nan"), float("nan")
    else:
        resultado["roc_auc"], resultado["pr_auc"] = float("nan"), float("nan")
    return resultado


VARIABLES_COMPARACION = {
    "relative_kw": "relative_kw", "ED_signed": "ED_signed", "ED_positive": "ED_positive",
}


def construir_comparaciones(df_val: pd.DataFrame, meta_completo: pd.DataFrame, resumen_ep: pd.DataFrame) -> pd.DataFrame:
    """AUC/Mann-Whitney son diagnostico de separabilidad UNICAMENTE -- ninguna comparacion aqui
    fija ni sugiere un umbral operativo."""
    filas = []
    for nombre_var, col_base in VARIABLES_COMPARACION.items():
        clean_val_pop = df_val[col_base].to_numpy()
        ramp_win_pop = meta_completo[f"{col_base}_ramp"].to_numpy()
        max_clean_col, max_ramp_col = f"max_{nombre_var}_clean", f"max_{nombre_var}_ramp"

        r = comparar_grupos("1_limpias_val_vs_ramp_ventanas", clean_val_pop, ramp_win_pop, pareado=False)
        r["variable"] = nombre_var; filas.append(r)

        r = comparar_grupos("2_rama_limpia_vs_atacada_episodio", resumen_ep[max_clean_col].to_numpy(),
                             resumen_ep[max_ramp_col].to_numpy(), pareado=True)
        r["variable"] = nombre_var; filas.append(r)

        det = resumen_ep.loc[resumen_ep["ramp_induced_detected"], max_ramp_col].to_numpy()
        no_det = resumen_ep.loc[~resumen_ep["ramp_induced_detected"], max_ramp_col].to_numpy()
        r = comparar_grupos("3_detectadas_vs_no_detectadas", no_det, det, pareado=False)
        r["variable"] = nombre_var; filas.append(r)

        no_det_ai = resumen_ep.loc[(~resumen_ep["ramp_induced_detected"]) & (resumen_ep["alto_impacto"]), max_ramp_col].to_numpy()
        r = comparar_grupos("4_no_detectadas_alto_impacto_vs_limpias", clean_val_pop, no_det_ai, pareado=False)
        r["variable"] = nombre_var; filas.append(r)

        for grupo, num in (("C", "5"), ("D", "6")):
            sub = resumen_ep.loc[resumen_ep["grupo_temporal"] == grupo, max_ramp_col].to_numpy()
            r = comparar_grupos(f"{num}_grupo_{grupo}_persistente_o_tendencia_vs_limpias", clean_val_pop, sub, pareado=False)
            r["variable"] = nombre_var; filas.append(r)

        for i, (etiqueta, (dmin, dmax)) in enumerate(DURACION_BUCKETS.items(), start=7):
            sub = resumen_ep[resumen_ep["duration_min"].between(dmin, dmax)]
            r = comparar_grupos(f"{i}_duracion_{etiqueta}_limpia_vs_atacada", sub[max_clean_col].to_numpy(),
                                 sub[max_ramp_col].to_numpy(), pareado=True)
            r["variable"] = nombre_var; filas.append(r)

        for rho in RHO_FINALES:
            sub = resumen_ep[np.isclose(resumen_ep["rho_final"], rho)]
            r = comparar_grupos(f"10_rho_final_{rho:.2f}_limpia_vs_atacada", sub[max_clean_col].to_numpy(),
                                 sub[max_ramp_col].to_numpy(), pareado=True)
            r["variable"] = nombre_var; filas.append(r)
    return pd.DataFrame(filas)


# ==========================================================================================
# 10) Comparacion con relative_kw: correlaciones + casos donde ED aporta informacion distinta
# ==========================================================================================

def comparar_con_relative_kw(resumen_ep: pd.DataFrame) -> dict:
    delta_rel, delta_ed_s, delta_ed_p = resumen_ep["max_delta_relative_kw"], resumen_ep["max_delta_ED_signed"], resumen_ep["max_delta_ED_positive"]

    umbral_apenas_rel = delta_rel.abs().quantile(0.25)
    umbral_aumenta_ed = delta_ed_s.quantile(0.75)
    umbral_apenas_ed = delta_ed_s.abs().quantile(0.25)
    umbral_aumenta_rel = delta_rel.quantile(0.75)

    casos = pd.DataFrame([{
        "definicion": "apenas_cambia = |delta| <= P25(|delta|); aumenta = delta >= P75(delta)",
        "relative_apenas_cambia_ED_signed_aumenta": int(((delta_rel.abs() <= umbral_apenas_rel) & (delta_ed_s >= umbral_aumenta_ed)).sum()),
        "ED_signed_apenas_cambia_relative_aumenta": int(((delta_ed_s.abs() <= umbral_apenas_ed) & (delta_rel >= umbral_aumenta_rel)).sum()),
        "ambos_aumentan": int(((delta_rel > 0) & (delta_ed_s > 0)).sum()),
        "ninguno_responde": int(((delta_rel <= 0) & (delta_ed_s <= 0)).sum()),
        "n_episodios": len(resumen_ep),
    }])
    return {"casos": casos}


def tabla_ed_aporta_sobre_relative(resumen_ep: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
    p90_signed = float(np.percentile(df_val["ED_signed"], 90))
    p90_positive = float(np.percentile(df_val["ED_positive"], 90))
    cond = (~resumen_ep["ramp_induced_detected"]) & (resumen_ep["alto_impacto"]) & \
           ((resumen_ep["max_ED_signed_ramp"] > p90_signed) | (resumen_ep["max_ED_positive_ramp"] > p90_positive))
    tabla = resumen_ep[cond].copy()
    tabla.attrs["p90_ED_signed_limpio"] = p90_signed
    tabla.attrs["p90_ED_positive_limpio"] = p90_positive
    return tabla


def resumen_descriptivo_grupo(resumen_ep: pd.DataFrame, columna_grupo: str, p90_signed: float, p90_positive: float) -> pd.DataFrame:
    filas = []
    for valor, g in resumen_ep.groupby(columna_grupo):
        filas.append({
            columna_grupo: valor, "n_episodios": len(g),
            "mediana_max_relative_kw_ramp": float(g["max_relative_kw_ramp"].median()),
            "mediana_max_ED_signed_ramp": float(g["max_ED_signed_ramp"].median()),
            "mediana_max_ED_positive_ramp": float(g["max_ED_positive_ramp"].median()),
            "iqr_max_ED_signed_ramp": float(g["max_ED_signed_ramp"].quantile(0.75) - g["max_ED_signed_ramp"].quantile(0.25)),
            "iqr_max_ED_positive_ramp": float(g["max_ED_positive_ramp"].quantile(0.75) - g["max_ED_positive_ramp"].quantile(0.25)),
            "pct_ED_signed_sobre_p90_limpio": 100 * float((g["max_ED_signed_ramp"] > p90_signed).mean()),
            "pct_ED_positive_sobre_p90_limpio": 100 * float((g["max_ED_positive_ramp"] > p90_positive).mean()),
            "pct_detectado_por_P": 100 * float(g["ramp_induced_detected"].mean()),
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# 11) Casos visuales obligatorios
# ==========================================================================================

def figura_caso_visual(episode_id: str, ataques_ramp: dict, particion_val_raw: np.ndarray, meta_completo: pd.DataFrame,
                        signed_ramp: np.ndarray, positive_ramp: np.ndarray, signed_val: np.ndarray,
                        referencia: dict, inicio_particion, prefijo: str, etiqueta: str) -> None:
    a = ataques_ramp[episode_id]
    offset, dur = a["offset_global"], a["duracion"]
    margen = max(60, dur // 4)
    ini, fin = max(0, offset - margen), offset + dur + margen
    x_limpio = particion_val_raw[ini:fin].astype(np.float64)
    x_ramp = x_limpio.copy()
    x_ramp[offset - ini:offset - ini + dur] = a["y_tramo_ramp"]
    t = [inicio_particion + pd.Timedelta(minutes=m) for m in range(ini, fin)]

    sub = meta_completo[meta_completo["episode_id"] == episode_id].sort_values("k")
    idx_pico_label = sub["ED_signed_ramp"].idxmax()  # etiqueta de fila == posicion en signed_ramp/positive_ramp
    residuo_atacado_pico = signed_ramp[idx_pico_label]
    k_pico = int(sub.loc[idx_pico_label, "k"])
    residuo_limpio_pico = signed_val[k_pico]
    g = sub.reset_index(drop=True)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    axes[0, 0].plot(t, x_limpio, color=GRIS, linewidth=1.1, label="limpio")
    axes[0, 0].plot(t, x_ramp, color=ROJO, linewidth=1.2, alpha=0.85, label="ramp")
    axes[0, 0].axvspan(inicio_particion + pd.Timedelta(minutes=offset), inicio_particion + pd.Timedelta(minutes=offset + dur), color=VERDE, alpha=0.08)
    axes[0, 0].set_ylabel("kW"); axes[0, 0].legend(fontsize=7.5); axes[0, 0].set_title(f"{episode_id} | {etiqueta}", fontsize=9.5)

    axes[0, 1].plot(np.arange(360), residuo_limpio_pico, color=GRIS, linewidth=1.0, label="limpio (ventana pico)")
    axes[0, 1].plot(np.arange(360), residuo_atacado_pico, color=ROJO, linewidth=1.0, alpha=0.85, label="ramp (ventana pico)")
    axes[0, 1].axhline(0, color=GRIS_OSCURO, linewidth=0.7)
    axes[0, 1].set_ylabel("residuo firmado"); axes[0, 1].set_xlabel("minuto dentro de la ventana"); axes[0, 1].legend(fontsize=7.5)

    axes[1, 0].hist(residuo_limpio_pico, bins=40, color=GRIS, alpha=0.5, density=True, label="limpio")
    axes[1, 0].hist(residuo_atacado_pico, bins=40, color=ROJO, alpha=0.5, density=True, label="ramp")
    axes[1, 0].set_xlabel("residuo firmado"); axes[1, 0].set_ylabel("densidad"); axes[1, 0].legend(fontsize=7.5)
    axes[1, 0].set_title("Histograma (ventana de ED_signed maximo)")

    for arr, color, nombre in ((residuo_limpio_pico, GRIS, "limpio"), (residuo_atacado_pico, ROJO, "ramp")):
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        axes[1, 1].plot(xs, ys, color=color, linewidth=1.3, label=nombre)
    axes[1, 1].set_xlabel("residuo firmado"); axes[1, 1].set_ylabel("ECDF"); axes[1, 1].legend(fontsize=7.5)
    axes[1, 1].set_title("ECDF (ventana de ED_signed maximo)")

    t_score = pd.to_datetime(g["window_end"])
    axes[2, 0].plot(t_score, g["relative_kw_clean"], color=GRIS, marker="o", markersize=3, label="relative_kw limpio")
    axes[2, 0].plot(t_score, g["relative_kw_ramp"], color=ROJO, marker="o", markersize=3, label="relative_kw ramp")
    axes[2, 0].set_ylabel("relative_kw"); axes[2, 0].legend(fontsize=7.5)

    axes[2, 1].plot(t_score, g["ED_signed_clean"], color=AZUL_CLARO, marker="o", markersize=3, label="ED_signed limpio")
    axes[2, 1].plot(t_score, g["ED_signed_ramp"], color=AZUL, marker="o", markersize=3, label="ED_signed ramp")
    axes[2, 1].plot(t_score, g["ED_positive_clean"], color="#f0b27a", marker="s", markersize=3, label="ED_positive limpio")
    axes[2, 1].plot(t_score, g["ED_positive_ramp"], color=NARANJA, marker="s", markersize=3, label="ED_positive ramp")
    axes[2, 1].set_ylabel("energy distance"); axes[2, 1].legend(fontsize=6.5)

    fig.tight_layout(); _guardar_fig(fig, f"{prefijo}_caso_{etiqueta}")


def figura_ejemplo_simple(episode_id: str, meta_completo: pd.DataFrame, nombre_archivo: str, titulo: str) -> None:
    g = meta_completo[meta_completo["episode_id"] == episode_id].sort_values("k")
    t_score = pd.to_datetime(g["window_end"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_score, g["relative_kw_clean"], color=GRIS, marker="o", markersize=3, label="relative_kw limpio")
    ax.plot(t_score, g["relative_kw_ramp"], color=ROJO, marker="o", markersize=3, label="relative_kw ramp")
    ax2 = ax.twinx()
    ax2.plot(t_score, g["ED_signed_ramp"], color=AZUL, marker="s", markersize=3, linestyle="--", label="ED_signed ramp")
    ax.set_ylabel("relative_kw"); ax2.set_ylabel("ED_signed")
    lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5)
    ax.set_title(f"{episode_id} | {titulo}", fontsize=9.5)
    fig.tight_layout(); _guardar_fig(fig, nombre_archivo)


# ==========================================================================================
# 12) Figuras (minimo 14)
# ==========================================================================================

def generar_figuras(referencia: dict, df_val: pd.DataFrame, meta_completo: pd.DataFrame, resumen_ep: pd.DataFrame) -> None:
    # 1/2) distribuciones de referencia
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(referencia["reference_signed"], bins=80, color=GRIS_OSCURO, alpha=0.8)
    ax.set_xlabel("residuo firmado"); ax.set_ylabel("frecuencia"); ax.set_title("Referencia limpia: residuos firmados (train)")
    fig.tight_layout(); _guardar_fig(fig, "01_referencia_residuos_firmados")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(referencia["reference_positive"], bins=80, color=VERDE, alpha=0.8)
    ax.set_xlabel("deficit relativo positivo"); ax.set_ylabel("frecuencia"); ax.set_title("Referencia limpia: deficit positivo (train)")
    fig.tight_layout(); _guardar_fig(fig, "02_referencia_deficit_positivo")

    # 3) distribucion de ED limpio en val
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(df_val["ED_signed"], bins=60, color=AZUL, alpha=0.8); axes[0].set_title("ED_signed limpio (val)")
    axes[1].hist(df_val["ED_positive"], bins=60, color=NARANJA, alpha=0.8); axes[1].set_title("ED_positive limpio (val)")
    fig.tight_layout(); _guardar_fig(fig, "03_distribucion_ED_limpio_val")

    # 4) ED limpio a lo largo del tiempo
    fig, ax = plt.subplots(figsize=(10, 4.5))
    t = pd.to_datetime(df_val["timestamp"])
    ax.plot(t, df_val["ED_signed"], color=AZUL, linewidth=0.5, alpha=0.7, label="ED_signed")
    ax.plot(t, df_val["ED_positive"], color=NARANJA, linewidth=0.5, alpha=0.7, label="ED_positive")
    ax.set_ylabel("energy distance"); ax.legend(fontsize=8); ax.set_title("ED limpio a lo largo de val")
    fig.tight_layout(); _guardar_fig(fig, "04_ED_limpio_temporal")

    # 5) ED vs relative_kw
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(df_val["relative_kw"], df_val["ED_signed"], s=3, alpha=0.15, color=AZUL); axes[0].set_xlabel("relative_kw"); axes[0].set_ylabel("ED_signed")
    axes[1].scatter(df_val["relative_kw"], df_val["ED_positive"], s=3, alpha=0.15, color=NARANJA); axes[1].set_xlabel("relative_kw"); axes[1].set_ylabel("ED_positive")
    fig.tight_layout(); _guardar_fig(fig, "05_ED_vs_relative_kw")

    # 6) ED limpio vs ramp
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(df_val["ED_signed"], bins=60, color=GRIS, alpha=0.5, density=True, label="limpio")
    axes[0].hist(meta_completo["ED_signed_ramp"], bins=60, color=ROJO, alpha=0.5, density=True, label="ramp")
    axes[0].set_title("ED_signed: limpio vs ramp"); axes[0].legend(fontsize=8)
    axes[1].hist(df_val["ED_positive"], bins=60, color=GRIS, alpha=0.5, density=True, label="limpio")
    axes[1].hist(meta_completo["ED_positive_ramp"], bins=60, color=ROJO, alpha=0.5, density=True, label="ramp")
    axes[1].set_title("ED_positive: limpio vs ramp"); axes[1].legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "06_ED_limpio_vs_ramp")

    # 7) delta ED por duracion
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    resumen_ep.boxplot(column="max_delta_ED_signed", by="duration_min", ax=axes[0]); axes[0].set_title("delta ED_signed por duracion")
    resumen_ep.boxplot(column="max_delta_ED_positive", by="duration_min", ax=axes[1]); axes[1].set_title("delta ED_positive por duracion")
    fig.suptitle(""); fig.tight_layout(); _guardar_fig(fig, "07_delta_ED_por_duracion")

    # 8) delta ED por rho_final
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    resumen_ep.boxplot(column="max_delta_ED_signed", by="rho_final", ax=axes[0]); axes[0].set_title("delta ED_signed por rho_final")
    resumen_ep.boxplot(column="max_delta_ED_positive", by="rho_final", ax=axes[1]); axes[1].set_title("delta ED_positive por rho_final")
    fig.suptitle(""); fig.tight_layout(); _guardar_fig(fig, "08_delta_ED_por_severidad")

    # 9) ED en rampas de alto impacto
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(resumen_ep["hidden_energy_kwh"], resumen_ep["max_ED_signed_ramp"],
               c=np.where(resumen_ep["alto_impacto"], ROJO, GRIS), s=10, alpha=0.4)
    ax.set_xlabel("energia oculta (kWh)"); ax.set_ylabel("max ED_signed (ramp)"); ax.set_title("ED_signed vs energia oculta (rojo=alto impacto)")
    fig.tight_layout(); _guardar_fig(fig, "09_ED_alto_impacto")

    # 10) ED por grupos temporales A-E
    fig, ax = plt.subplots(figsize=(7, 5))
    datos_grupo = [resumen_ep.loc[resumen_ep["grupo_temporal"] == g, "max_ED_signed_ramp"].to_numpy() for g in "ABCDE"]
    bp = ax.boxplot(datos_grupo, tick_labels=list("ABCDE"), patch_artist=True)
    for patch, g in zip(bp["boxes"], "ABCDE"):
        patch.set_facecolor(COLOR_GRUPO[g])
    ax.set_ylabel("max ED_signed (ramp)"); ax.set_title("ED_signed por grupo temporal (A-E)")
    fig.tight_layout(); _guardar_fig(fig, "10_ED_por_grupo_temporal")

    print(f"  figuras 1-10 guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("[1/3] Cargando modelo, ataques existentes y rejilla val (checkpoint existente, sin reentrenar)...")
    ctx = cargar_todo()
    serie_train_antes = ctx["particion_train"][COLUMNA_OBJETIVO].to_numpy().copy()
    serie_val_antes = ctx["particion_val_raw"].copy()

    print("\n[4] Construyendo referencia limpia (SOLO train, ventanas separadas 6h)...")
    referencia = construir_referencia_limpia(ctx["particion_train"], ctx["modelo"], ctx["params_norm"])
    print(f"  ventanas usadas={referencia['n_ventanas']} | residuos originales={referencia['n_residuos_originales']} | "
          f"residuos usados={referencia['n_residuos_usados']}")
    tabla_ref = _guardar_tabla(tabla_referencia(referencia), "referencia_residuos")
    print(tabla_ref.to_string(index=False))

    print("\n[6] Construyendo secuencia limpia de val (ED_signed/ED_positive + estadisticos por ventana)...")
    df_val, signed_val, positive_val = construir_secuencia_val(ctx, referencia)
    _guardar_tabla(df_val, "scores_limpios_val")
    print(f"  {len(df_val)} ventanas limpias de val puntuadas")

    print("\n[6] Analizando estabilidad temporal de ED limpio (mensual/horaria/autocorrelacion/correlaciones)...")
    estabilidad = analizar_estabilidad_temporal(df_val)
    _guardar_tabla(estabilidad["mensual"], "estabilidad_ED_mensual")
    _guardar_tabla(estabilidad["horario"], "estabilidad_ED_horaria")
    _guardar_tabla(estabilidad["autocorrelacion"], "estabilidad_ED_autocorrelacion")
    _guardar_tabla(estabilidad["correlaciones"], "correlaciones_ED_relative_kw")
    print(estabilidad["autocorrelacion"].to_string(index=False))
    print(estabilidad["correlaciones"].to_string(index=False))

    print("\n[7] Construyendo episodios ramp_lineal (reutiliza experimento_ramp, sin generar nuevos)...")
    tabla_ramp_fresca, ataques_ramp = construir_ataques_ramp(ctx)
    assert len(tabla_ramp_fresca) == 1050

    tabla_ramp_meta = pd.read_csv(RAMP_DIR / "episodios_ramp.csv")
    assert set(tabla_ramp_meta["episode_id"]) == set(tabla_ramp_fresca["episode_id"])
    dif_energia = (tabla_ramp_meta.set_index("episode_id")["hidden_energy_ramp_kwh"] -
                   tabla_ramp_fresca.set_index("episode_id")["hidden_energy_ramp_kwh"]).abs()
    assert (dif_energia < 1e-6).all(), "la energia oculta recalculada no coincide con experimento_ramp"
    print(f"  OK -- 1050 episodios ramp coinciden con experimento_ramp (dif energia max={dif_energia.max():.2e} kWh)")

    print("\nPuntuando ventanas atacadas de los 1050 episodios ramp (1 llamada batched al modelo)...")
    meta_ramp, signed_ramp, positive_ramp = puntuar_ventanas_ramp(ataques_ramp, ctx["particion_val_raw"], ctx["grid_60"], ctx["modelo"], ctx["params_norm"])
    print(f"  {len(meta_ramp)} ventanas atacadas puntuadas")

    print("\nCalculando ED_signed/ED_positive de las ventanas atacadas (scipy.stats.energy_distance)...")
    meta_ramp["ED_signed_ramp"] = energy_distance_lote(signed_ramp, referencia["reference_signed"])
    meta_ramp["ED_positive_ramp"] = energy_distance_lote(positive_ramp, referencia["reference_positive"])
    meta_completo = enlazar_con_rama_limpia(meta_ramp, df_val)
    _guardar_tabla(meta_completo, "scores_ventanas_ramp")

    print("\n[8] Calculando resumen por episodio...")
    resumen_ep = resumen_por_episodio(meta_completo, tabla_ramp_meta)
    _guardar_tabla(resumen_ep, "resumen_episodios_energy_distance")
    print(f"  p75 hidden_energy_ramp_kwh (alto impacto) = {resumen_ep.attrs['p75_hidden_energy_kwh']:.4f} kWh")

    print("\n[9] Construyendo las 10 comparaciones principales (x3 variables)...")
    pruebas = construir_comparaciones(df_val, meta_completo, resumen_ep)
    _guardar_tabla(pruebas, "pruebas_estadisticas")
    print(pruebas[pruebas["variable"] == "ED_signed"][["comparacion", "n_x", "n_y", "p_valor", "cliffs_delta", "roc_auc"]].to_string(index=False))

    print("\n[10] Comparando con relative_kw (correlaciones + casos)...")
    comp_rel = comparar_con_relative_kw(resumen_ep)
    _guardar_tabla(comp_rel["casos"], "comparacion_scores")
    print(comp_rel["casos"].to_string(index=False))

    tabla_aporta = tabla_ed_aporta_sobre_relative(resumen_ep, df_val)
    _guardar_tabla(tabla_aporta, "casos_ed_aporta_sobre_relative_kw")
    print(f"  episodios donde ED aporta sobre relative_kw (no detectados, alto impacto, ED>P90 limpio): {len(tabla_aporta)}")

    print("\n[por duracion/severidad/grupo] Tablas descriptivas...")
    p90_signed, p90_positive = float(np.percentile(df_val["ED_signed"], 90)), float(np.percentile(df_val["ED_positive"], 90))
    _guardar_tabla(resumen_descriptivo_grupo(resumen_ep, "duration_min", p90_signed, p90_positive), "resultados_por_duracion")
    _guardar_tabla(resumen_descriptivo_grupo(resumen_ep, "rho_final", p90_signed, p90_positive), "resultados_por_severidad")
    _guardar_tabla(resumen_descriptivo_grupo(resumen_ep, "grupo_temporal", p90_signed, p90_positive), "resultados_por_grupo_temporal")

    print("\nGuardando rampas de alto impacto (energy distance)...")
    _guardar_tabla(resumen_ep[resumen_ep["alto_impacto"]].sort_values("hidden_energy_kwh", ascending=False), "ramp_alto_impacto_energy_distance")

    print("\n[11] Generando los dos casos visuales obligatorios...")
    caso_largo = "ramp_lineal__1440_0.30_37"   # 1440min, rho_final=0.30, relative_kw sube pero no cruza el umbral
    caso_corto = "ramp_lineal__0120_0.50_08"   # 120min, rho_final=0.50, scores limpio/atacado casi identicos
    figura_caso_visual(caso_largo, ataques_ramp, ctx["particion_val_raw"], meta_completo, signed_ramp, positive_ramp,
                        signed_val, referencia, ctx["inicio_particion"], "11", "largo_alto_impacto")
    figura_caso_visual(caso_corto, ataques_ramp, ctx["particion_val_raw"], meta_completo, signed_ramp, positive_ramp,
                        signed_val, referencia, ctx["inicio_particion"], "12", "corto_indetectable")

    print("\n[13/14] Ejemplos donde ED responde y relative_kw no, y donde ninguno responde...")
    if len(tabla_aporta):
        ep_ed_responde = tabla_aporta.sort_values("max_delta_ED_signed", ascending=False).iloc[0]["episode_id"]
        figura_ejemplo_simple(ep_ed_responde, meta_completo, "13_ejemplo_ED_responde_relative_no", "ED responde, relative_kw no")
    ninguno = resumen_ep[(resumen_ep["max_delta_relative_kw"] <= 0) & (resumen_ep["max_delta_ED_signed"] <= 0)]
    if len(ninguno):
        ep_ninguno = ninguno.sort_values("hidden_energy_kwh", ascending=False).iloc[0]["episode_id"]
        figura_ejemplo_simple(ep_ninguno, meta_completo, "14_ejemplo_ninguno_responde", "ninguno responde")

    print("\n[12] Generando figuras 1-10...")
    generar_figuras(referencia, df_val, meta_completo, resumen_ep)

    assert np.array_equal(serie_train_antes, ctx["particion_train"][COLUMNA_OBJETIVO].to_numpy())
    assert np.array_equal(serie_val_antes, ctx["particion_val_raw"])
    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "ctx": ctx, "referencia": referencia, "df_val": df_val, "signed_val": signed_val, "positive_val": positive_val,
        "tabla_ramp_meta": tabla_ramp_meta, "ataques_ramp": ataques_ramp, "meta_completo": meta_completo,
        "signed_ramp": signed_ramp, "positive_ramp": positive_ramp, "resumen_ep": resumen_ep, "pruebas": pruebas,
        "comp_rel": comp_rel, "tabla_aporta": tabla_aporta,
    }


if __name__ == "__main__":
    main()
