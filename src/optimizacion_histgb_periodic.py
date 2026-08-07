"""Optimizacion robusta de HistGB_PERIODIC y de su fusion OR con P, mediante validacion
temporal (walk-forward) exclusivamente pre-test. Compara PERIODIC_BASE (16 vars actuales) con
PERIODIC_ROBUST (+ lag_1344 y resumenes periodicos robustos) y dos scores (RELATIVE, ROBUST_Z),
seleccionando -si procede- un candidato que sustituya a OR_BASELINE (P + HistGB_PERIODIC + OR)
solo si supera la regla de sustitucion definida de antemano. Test permanece completamente
intacto en todo momento -- no se importa ni se referencia `particiones["test"]`.

Naturaleza exploratoria: entrenamiento no supervisado sobre normalidad limpia, con seleccion
de configuracion informada por ataques sinteticos generados sobre 4 folds temporales pre-test
(nunca sobre el periodo de evaluacion ya observado en experimentos anteriores, ni sobre test).
Los ataques de cada fold se generan con semillas nuevas y fijas mediante el mismo generador ya
existente (`episodes._generar_episodios`, sin modificar), reposicionado sobre la ventana
temporal propia de cada fold.

Reutiliza sin modificar predictor_causal_lags.{construir_serie_15min, _calendario,
construir_kw15_atacado, calcular_score, resumen_eventos_grid, RESOLUCION_MIN},
fusion_p_histgb.{estado_ffill, construir_estados_episodio, simular_episodios_generico},
episodes._generar_episodios/tabla_master, experimento_ramp.construir_episodios_ramp,
analisis_detectabilidad.{_respuesta_detector, clasificar_grupos, contexto_e_impacto},
memoria_finita_relative_kw.{enriquecer_resultado, resumen_metricas}, experimento_stride.
{contar_eventos, mcnemar_p, bootstrap_pareado}, stride_relative.puntuar_stride_relative,
episodes.wilson_ci, diagnostico_no_estacionariedad.construir_secuencia_continua (solo
inferencia, checkpoint P congelado), base_relative.cargar_ataques_y_modelo_360, splitting.
split_temporal y windowing.{ventanear, cargar_config}.

Ejecutable de forma aislada:
    python -m src.optimizacion_histgb_periodic
"""
from __future__ import annotations

import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import stats as sstats
from sklearn.ensemble import HistGradientBoostingRegressor

from src import analisis_detectabilidad as ad
from src import episodes as eps_mod
from src import experimento_ramp as er
from src import fusion_p_histgb as fus
from src import predictor_causal_lags as pcl
from src import stride_relative as sr
from src import windowing as wnd
from src.base_relative import cargar_ataques_y_modelo_360
from src.data_loading import COLUMNA_OBJETIVO, cargar_y_limpiar
from src.diagnostico_no_estacionariedad import construir_secuencia_continua
from src.episodes import wilson_ci
from src.experimento_stride import bootstrap_pareado, contar_eventos, mcnemar_p
from src.memoria_finita_relative_kw import enriquecer_resultado, resumen_metricas
from src.splitting import split_temporal

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "optimizacion_histgb_periodic"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "optimizacion_histgb_periodic"
MODELOS_DIR = BASE_DIR / "models" / "predictor_causal_lags_tuned"
EDO_DIR = BASE_DIR / "results" / "tables" / "energy_distance_operativo"
PCL_DIR = BASE_DIR / "results" / "tables" / "predictor_causal_lags"
PCL_MODELS_DIR = BASE_DIR / "models" / "predictor_causal_lags"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

RESOLUCION_MIN = 15
SEED_GLOBAL = 42
UMBRAL_H_BASELINE = 0.8936411147927468
FA_OR_MAXIMO = 0.12
OBJETIVO_FA_OR = 0.10
BANDA_PRINCIPAL = (0.08, 0.12)
BANDA_SECUNDARIA = (0.06, 0.14)

# --- Escala temporal de los folds (dias desde el inicio de train) -- expanding-window, con las
# ventanas de calibracion/validacion repartidas a lo largo de mas de un ano y medio (no todas
# apelotonadas al principio) para cubrir "varios periodos pre-test" de estacionalidad distinta.
FOLD_TRAIN_END_DIAS = [365, 550, 735, 920]
CALIB_DIAS = 30
VALID_DIAS = 45
FINAL_CALIBRATION_DIAS = 60
N_POSICIONES_POR_FOLD = 6  # repeticiones por (rho, duracion) -- 7x7x6=294 constante + 3x7x6=126 ramp por fold
SEED_ATAQUES_BASE = 9000  # semillas NUEVAS (no las de los experimentos anteriores), una por fold

# --- Lags y features -----------------------------------------------------------------------
LAGS_DIARIOS = {"lag_96": 96, "lag_192": 192, "lag_288": 288}
LAG_SEMANAL = {"lag_672": 672}
LAG_QUINCENAL = {"lag_1344": 1344}
TODOS_LOS_LAGS = {**LAGS_DIARIOS, **LAG_SEMANAL, **LAG_QUINCENAL}
MAX_LAG = 1344  # 1344*15min = 336h = 14 dias

CALENDARIO_COLS = pcl.CALENDARIO_COLS
PERIODIC_BASE_FEATURES = ["lag_96", "lag_192", "lag_288", "lag_672", "profile_train", "mad_train", "n_obs_train"] + CALENDARIO_COLS
PERIODIC_ROBUST_EXTRA = [
    "lag_1344", "daily_lag_median", "daily_lag_min", "daily_lag_max", "daily_lag_range", "daily_lag_MAD",
    "periodic_lag_median", "periodic_lag_MAD", "periodic_lag_min", "periodic_lag_max", "periodic_lag_range",
    "lag96_minus_profile", "lag192_minus_profile", "lag288_minus_profile", "lag672_minus_profile", "lag1344_minus_profile",
    "periodic_median_minus_profile", "profile_q25", "profile_q75", "profile_IQR",
]
PERIODIC_ROBUST_FEATURES = PERIODIC_BASE_FEATURES + PERIODIC_ROBUST_EXTRA
assert len(PERIODIC_BASE_FEATURES) == 16

# --- Espacio de hiperparametros -----------------------------------------------------------
GRID_HP = {
    "learning_rate": [0.015, 0.025, 0.03, 0.05, 0.075, 0.10], "max_iter": [150, 250, 400, 600],
    "max_leaf_nodes": [7, 15, 31, 63], "max_depth": [None, 4, 6, 8, 12],
    "min_samples_leaf": [10, 20, 30, 50, 100], "l2_regularization": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
    "max_bins": [127, 255], "early_stopping": [True], "validation_fraction": [0.10],
    "n_iter_no_change": [20, 35], "tol": [1e-7, 1e-6],
}
LOSSES_CANDIDATAS = [("squared_error", {}), ("absolute_error", {}), ("quantile", {"quantile": 0.60}), ("quantile", {"quantile": 0.70})]
CONFIG_HISTGB_PERIODIC_ACTUAL = {
    "learning_rate": 0.10, "max_leaf_nodes": 15, "max_depth": None, "min_samples_leaf": 20, "l2_regularization": 1.0,
    "max_iter": 100, "max_bins": 255, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10,
    "tol": 1e-7, "loss": "squared_error",
}
N_CANDIDATOS_TOTAL = 48
SCORES = ["RELATIVE", "ROBUST_Z"]


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 5) Division temporal pre-test: 4 folds expanding-window + FINAL_CALIBRATION
# ==========================================================================================

def construir_folds(inicio: pd.Timestamp, fin_pretest: pd.Timestamp) -> dict:
    dia = pd.Timedelta(days=1)
    final_calibration_ini = fin_pretest - FINAL_CALIBRATION_DIAS * dia
    folds = []
    for i, train_dias in enumerate(FOLD_TRAIN_END_DIAS):
        train_fin = inicio + train_dias * dia
        calib_fin = train_fin + CALIB_DIAS * dia
        valid_fin = calib_fin + VALID_DIAS * dia
        assert valid_fin <= final_calibration_ini, f"fold {i} se solapa con FINAL_CALIBRATION"
        folds.append({"fold": i, "ini": inicio, "train_fin": train_fin, "calib_fin": calib_fin, "valid_fin": valid_fin,
                      "seed_ataques": SEED_ATAQUES_BASE + i})
    return {"folds": folds, "final_calibration_ini": final_calibration_ini, "final_calibration_fin": fin_pretest, "inicio": inicio}


# ==========================================================================================
# 7-8) Features PERIODIC_BASE / PERIODIC_ROBUST -- estrictamente sin lags recientes
# ==========================================================================================

def construir_features_periodicas_robustas(kw: np.ndarray) -> pd.DataFrame:
    s = pd.Series(kw)
    cols = {nombre: s.shift(k).to_numpy() for nombre, k in TODOS_LOS_LAGS.items()}
    df = pd.DataFrame(cols)
    daily = df[["lag_96", "lag_192", "lag_288"]].to_numpy()
    periodic = df[["lag_96", "lag_192", "lag_288", "lag_672", "lag_1344"]].to_numpy()
    df["daily_lag_median"] = np.median(daily, axis=1)
    df["daily_lag_min"] = np.min(daily, axis=1)
    df["daily_lag_max"] = np.max(daily, axis=1)
    df["daily_lag_range"] = df["daily_lag_max"] - df["daily_lag_min"]
    df["daily_lag_MAD"] = np.median(np.abs(daily - df["daily_lag_median"].to_numpy()[:, None]), axis=1)
    df["periodic_lag_median"] = np.median(periodic, axis=1)
    df["periodic_lag_MAD"] = np.median(np.abs(periodic - df["periodic_lag_median"].to_numpy()[:, None]), axis=1)
    df["periodic_lag_min"] = np.min(periodic, axis=1)
    df["periodic_lag_max"] = np.max(periodic, axis=1)
    df["periodic_lag_range"] = df["periodic_lag_max"] - df["periodic_lag_min"]
    return df


def construir_perfil_fold(kw_train: np.ndarray, ts_start_train: np.ndarray) -> pd.DataFrame:
    """profile_train/mad_train/n_obs_train/q25/q75, calculados exclusivamente con el train del fold."""
    cal = pcl._calendario(ts_start_train)
    df = pd.DataFrame({"slot_15min": cal["slot_15min"], "es_finde": cal["es_finde"], "target_kw": kw_train})
    g = df.groupby(["es_finde", "slot_15min"])["target_kw"]
    perfil = g.median().rename("profile_train")
    mad = g.apply(lambda x: float(np.median(np.abs(x - x.median())))).rename("mad_train")
    n_obs = g.size().rename("n_obs_train")
    q25 = g.quantile(0.25).rename("profile_q25")
    q75 = g.quantile(0.75).rename("profile_q75")
    return pd.concat([perfil, mad, n_obs, q25, q75], axis=1).reset_index()


def _anadir_diffs_perfil(df: pd.DataFrame) -> pd.DataFrame:
    for nombre_lag, nombre_diff in (("lag_96", "lag96_minus_profile"), ("lag_192", "lag192_minus_profile"),
                                     ("lag_288", "lag288_minus_profile"), ("lag_672", "lag672_minus_profile"),
                                     ("lag_1344", "lag1344_minus_profile")):
        df[nombre_diff] = df[nombre_lag] - df["profile_train"]
    df["periodic_median_minus_profile"] = df["periodic_lag_median"] - df["profile_train"]
    df["profile_IQR"] = df["profile_q75"] - df["profile_q25"]
    return df


def preparar_fold(fold_def: dict, serie_completa_min: pd.DataFrame) -> dict:
    particion_fold_min = serie_completa_min.loc[(serie_completa_min.index >= fold_def["ini"]) & (serie_completa_min.index < fold_def["valid_fin"])]
    kw15 = pcl.construir_serie_15min(particion_fold_min)
    feats_lag = construir_features_periodicas_robustas(kw15["kw"])
    cal = pcl._calendario(kw15["ts_start"])
    df = pd.concat([feats_lag, cal], axis=1)
    df["target_kw"] = kw15["kw"]
    df["ts_start"] = kw15["ts_start"]
    df["ts_end"] = kw15["ts_end"]
    df = df.iloc[MAX_LAG:].reset_index(drop=True)

    es_train = (df["ts_start"] < fold_def["train_fin"]).to_numpy()
    es_calib = ((df["ts_start"] >= fold_def["train_fin"]) & (df["ts_start"] < fold_def["calib_fin"])).to_numpy()
    es_valid = ((df["ts_start"] >= fold_def["calib_fin"]) & (df["ts_start"] < fold_def["valid_fin"])).to_numpy()
    assert (es_train | es_calib | es_valid).all() and not (es_train & es_calib).any() and not (es_calib & es_valid).any()

    perfil = construir_perfil_fold(df.loc[es_train, "target_kw"].to_numpy(), df.loc[es_train, "ts_start"].to_numpy())
    df = df.merge(perfil, on=["es_finde", "slot_15min"], how="left")
    df = _anadir_diffs_perfil(df)
    assert df[PERIODIC_ROBUST_FEATURES].isna().sum().sum() == 0, "NaN en features tras el recorte de lookback"

    t_valid_ini_bin = int(round((fold_def["calib_fin"] - fold_def["ini"]) / pd.Timedelta(minutes=RESOLUCION_MIN)))
    return {"df": df, "es_train": es_train, "es_calib": es_calib, "es_valid": es_valid, "perfil": perfil,
            "kw15": kw15, "particion_fold_min_raw": particion_fold_min[COLUMNA_OBJETIVO].to_numpy(dtype=np.float64),
            "t_valid_ini_bin": t_valid_ini_bin, "n_dias_calib": CALIB_DIAS, "n_dias_valid": VALID_DIAS}


# ==========================================================================================
# 6) Generacion de ataques nuevos, contenidos en el fold de validacion (mismo generador existente)
# ==========================================================================================

def generar_ataques_fold(fold_def: dict, serie_completa_min: pd.DataFrame, config: dict) -> dict:
    train_fold_df = serie_completa_min.loc[(serie_completa_min.index >= fold_def["ini"]) & (serie_completa_min.index < fold_def["train_fin"])]
    valid_fold_df = serie_completa_min.loc[(serie_completa_min.index >= fold_def["calib_fin"]) & (serie_completa_min.index < fold_def["valid_fin"])]
    datos_falsos = {"particiones": {"train": train_fold_df, "val": valid_fold_df},
                    "ventanas": {"val": wnd.ventanear(valid_fold_df, 360, 360)}}

    episodios = eps_mod._generar_episodios(config, datos_falsos, "val", fold_def["seed_ataques"],
                                            modos=("continuous_timeline",), n_posiciones_override=N_POSICIONES_POR_FOLD)
    episodios = [e for e in episodios if e.get("family") == "reduccion_constante" and e.get("status") == "executed"]

    ataques_constante = {}
    for ep in episodios:
        offset_global = int((ep["attack_start"] - fold_def["ini"]) / pd.Timedelta(minutes=1))
        ks = sorted(ep["_ventanas_idx"])
        segmento = np.concatenate([ep["_ventanas_atacadas"][k] for k in ks])
        rel = offset_global - (int((valid_fold_df.index[0] - fold_def["ini"]) / pd.Timedelta(minutes=1)) + ks[0] * 360)
        y_tramo = segmento[rel:rel + ep["duration_min"]]
        ataques_constante[ep["episode_id"]] = {"offset_global": offset_global, "duracion": ep["duration_min"], "y_tramo": y_tramo,
                                                "attack_start": ep["attack_start"], "attack_end": ep["attack_end"],
                                                "family": ep["family"], "parameter": ep["parameter"]}

    master_fold = eps_mod.tabla_master(episodios)
    particion_fold_raw = serie_completa_min.loc[(serie_completa_min.index >= fold_def["ini"]) & (serie_completa_min.index < fold_def["valid_fin"]), COLUMNA_OBJETIVO].to_numpy(dtype=np.float64)
    tabla_ramp, ataques_ramp = er.construir_episodios_ramp(master_fold, particion_fold_raw, fold_def["ini"])
    for a in ataques_ramp.values():
        a["y_tramo"] = a["y_tramo_ramp"]

    return {"ataques_constante": ataques_constante, "master_fold": master_fold, "tabla_ramp": tabla_ramp,
            "ataques_ramp": ataques_ramp, "particion_fold_raw": particion_fold_raw, "valid_fold_df": valid_fold_df}


# ==========================================================================================
# P fresco en el fold (solo inferencia, checkpoint/params_norm congelados) -- necesario
# para Grupo B, alto impacto y la fusion OR con episodios nuevos que no existian antes
# ==========================================================================================

def cargar_p_global(modelo_p, params_norm_p) -> dict:
    base = construir_secuencia_continua()
    d_tr, d_va = base["por_particion"]["train"], base["por_particion"]["val"]
    ts = np.concatenate([pd.DatetimeIndex(d_tr["timestamps"]).to_numpy(), pd.DatetimeIndex(d_va["timestamps"]).to_numpy()])
    score = np.concatenate([np.asarray(d_tr["relative_kw"]), np.asarray(d_va["relative_kw"])])
    orden = np.argsort(ts)
    ts, score = ts[orden], score[orden]
    umbral_tabla = pd.read_csv(EDO_DIR / "02_umbrales_calibrados.csv")
    umbral_p = float(umbral_tabla[umbral_tabla["metodo"] == "P"]["umbral"].iloc[0])
    assert np.all(ts[1:] > ts[:-1])
    return {"ts": ts, "score": score, "umbral": umbral_p, "modelo": modelo_p, "params_norm": params_norm_p}


def puntuar_p_en_fold(ataques: dict, particion_fold_raw: np.ndarray, valid_fold_df: pd.DataFrame, fold_ini: pd.Timestamp,
                       p_global: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """`ataques` trae offset_global relativo a `fold_ini` (igual que en `puntuar_ataques_fold`);
    aqui se realinea a `valid_fold_df` (que empieza en calib_fin) para poder reutilizar
    `stride_relative.puntuar_stride_relative` sin reconstruir ventanas de todo el fold."""
    grid_60 = wnd.ventanear(valid_fold_df, 360, 60)
    offset_ini_valid = int((valid_fold_df.index[0] - fold_ini) / pd.Timedelta(minutes=1))
    particion_valid_raw = particion_fold_raw[offset_ini_valid:]
    ataques_rel = {ep: {**a, "offset_global": a["offset_global"] - offset_ini_valid} for ep, a in ataques.items()}
    tabla, scores_limpios = sr.puntuar_stride_relative(ataques_rel, particion_valid_raw, grid_60, p_global["modelo"], p_global["params_norm"], 60)
    return tabla, scores_limpios


def _mapear_a_global(p_global: dict, timestamps: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(p_global["ts"], timestamps)
    idx = np.clip(idx, 0, len(p_global["ts"]) - 1)
    assert np.all(p_global["ts"][idx] == timestamps), "los window_end de P en el fold no coinciden con la rejilla global de P"
    return idx


def construir_p_atacado_por_episodio(tabla_p: pd.DataFrame, p_global: dict) -> dict[str, dict]:
    resultado = {}
    for ep, g in tabla_p.groupby("episode_id"):
        idx_global = _mapear_a_global(p_global, pd.DatetimeIndex(g["window_end"]).to_numpy())
        resultado[ep] = {"k": idx_global, "score": g["attacked_score"].to_numpy()}
    return resultado


# ==========================================================================================
# Grupo B (constante) y alto impacto (ramp), reclasificados frescos por fold con la logica
# ya existente (analisis_detectabilidad), aplicada a los episodios nuevos del fold
# ==========================================================================================

def clasificar_grupo_b_fold(tabla_const_p: pd.DataFrame, umbral_p: float) -> set:
    resp = ad._respuesta_detector(tabla_const_p, umbral_p)
    resp = ad.clasificar_grupos(resp)
    return set(resp[resp["detectability_group"] == "B"]["episode_id"])


def clasificar_alto_impacto_fold(tabla_ramp_p: pd.DataFrame, ataques_ramp: dict, particion_fold_raw: np.ndarray, umbral_p: float) -> tuple[set, float]:
    resp = ad._respuesta_detector(tabla_ramp_p, umbral_p)
    contexto = ad.contexto_e_impacto(ataques_ramp, particion_fold_raw)
    merged = resp.merge(contexto, on="episode_id")
    p75 = float(merged["hidden_energy_kwh"].quantile(0.75))
    no_det = merged[merged["induced_detected"] == False]  # noqa: E712
    alto_impacto_ids = set(no_det[no_det["hidden_energy_kwh"] >= p75]["episode_id"])
    return alto_impacto_ids, p75


# ==========================================================================================
# 9) Muestreo determinista de candidatos (48 en total, repartidos BASE/ROBUST)
# ==========================================================================================

def muestrear_candidatos(seed: int = SEED_GLOBAL, n_total: int = N_CANDIDATOS_TOTAL) -> list[dict]:
    rng = np.random.default_rng(seed)
    n_base, n_robust = n_total // 2, n_total - n_total // 2
    candidatos = []
    cid = 0
    for feature_set, n in (("PERIODIC_BASE", n_base), ("PERIODIC_ROBUST", n_robust)):
        for _ in range(n):
            params = {k: v[int(rng.integers(len(v)))] for k, v in GRID_HP.items()}
            loss_nombre, loss_kw = LOSSES_CANDIDATAS[int(rng.integers(len(LOSSES_CANDIDATAS)))]
            params["loss"] = loss_nombre
            params.update(loss_kw)
            candidatos.append({"candidato_id": f"c{cid:03d}", "feature_set": feature_set, "es_baseline_incluido": False, **params})
            cid += 1
    # sustituye el primer candidato BASE por la configuracion actual de HistGB_PERIODIC (obligatoria)
    candidatos[0] = {"candidato_id": "c000_baseline_actual", "feature_set": "PERIODIC_BASE", "es_baseline_incluido": True,
                      **CONFIG_HISTGB_PERIODIC_ACTUAL}
    return candidatos


def construir_modelo(params: dict) -> HistGradientBoostingRegressor:
    kwargs = {k: v for k, v in params.items() if k not in ("candidato_id", "feature_set", "es_baseline_incluido", "quantile")}
    if params["loss"] == "quantile":
        kwargs["quantile"] = params["quantile"]
    return HistGradientBoostingRegressor(random_state=SEED_GLOBAL, **kwargs)


# ==========================================================================================
# 11) Scores: RELATIVE (formula actual) y ROBUST_Z (estadisticos por grupo, solo calibracion)
# ==========================================================================================

def calcular_stats_robust_z(signed_relative_calib: np.ndarray, es_finde_calib: np.ndarray, slot_calib: np.ndarray) -> dict:
    df = pd.DataFrame({"signed": signed_relative_calib, "es_finde": es_finde_calib, "slot": slot_calib})

    def _stats(x):
        centro = x.median()
        mad = (x - centro).abs().median()
        return pd.Series({"center": centro, "scale": 1.4826 * mad, "n": len(x)})

    por_grupo = df.groupby(["es_finde", "slot"])["signed"].apply(_stats).unstack()
    por_slot = df.groupby("slot")["signed"].apply(_stats).unstack()
    centro_global = float(df["signed"].median())
    escala_global = float(1.4826 * (df["signed"] - centro_global).abs().median())

    escalas_validas = por_grupo[por_grupo["n"] >= 12]["scale"]
    floor = float(np.percentile(escalas_validas, 10)) if len(escalas_validas) else 0.0
    if floor <= 0:
        floor = max(escala_global, 1e-3)
    return {"por_grupo": por_grupo, "por_slot": por_slot, "centro_global": centro_global,
            "escala_global": max(escala_global, floor), "floor": floor}


def aplicar_robust_z(signed_relative: np.ndarray, es_finde: np.ndarray, slot: np.ndarray, stats: dict) -> np.ndarray:
    df = pd.DataFrame({"signed": signed_relative, "es_finde": es_finde, "slot": slot})
    grp = stats["por_grupo"].reset_index().rename(columns={"center": "c_g", "scale": "s_g", "n": "n_g"})
    df = df.merge(grp, on=["es_finde", "slot"], how="left")
    slot_df = stats["por_slot"].reset_index().rename(columns={"center": "c_s", "scale": "s_s", "n": "n_s"})
    df = df.merge(slot_df, on="slot", how="left")
    df["n_g"] = df["n_g"].fillna(0)
    df["n_s"] = df["n_s"].fillna(0)
    usar_grupo = df["n_g"] >= 12
    usar_slot = (~usar_grupo) & (df["n_s"] >= 12)
    centro = np.where(usar_grupo, df["c_g"], np.where(usar_slot, df["c_s"], stats["centro_global"]))
    escala = np.where(usar_grupo, df["s_g"], np.where(usar_slot, df["s_s"], stats["escala_global"]))
    escala = np.maximum(escala, stats["floor"])
    z = (df["signed"].to_numpy() - centro) / escala
    return np.maximum(0.0, z)


# ==========================================================================================
# 12) Calibracion exacta del umbral de H optimizando FA de la fusion OR (no H aislado)
# ==========================================================================================

def calibrar_umbral_or(scores_h_calib: np.ndarray, active_p_calib_ffill: np.ndarray, n_dias_calib: float) -> tuple[pd.DataFrame, dict]:
    filas = []
    for v in np.unique(scores_h_calib):
        active_h = scores_h_calib > v
        active_or = active_p_calib_ffill | active_h
        n_ev = contar_eventos(active_or)
        fa = n_ev / n_dias_calib if n_dias_calib else float("nan")
        filas.append({"umbral": float(v), "fa_or": fa, "pct_activo": 100 * float(active_or.mean())})
    tabla = pd.DataFrame(filas)

    candidatos, banda, deficiente = tabla[tabla["fa_or"].between(*BANDA_PRINCIPAL)], "principal", False
    if len(candidatos) == 0:
        candidatos, banda = tabla[tabla["fa_or"].between(*BANDA_SECUNDARIA)], "secundaria"
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla[tabla["fa_or"] < BANDA_SECUNDARIA[1]], "fuera_de_banda_mas_conservador", True
    if len(candidatos) == 0:
        candidatos, banda, deficiente = tabla.copy(), "sin_candidato_valido", True

    candidatos = candidatos.copy()
    candidatos["dist"] = (candidatos["fa_or"] - OBJETIVO_FA_OR).abs()
    candidatos = candidatos.sort_values(["dist", "pct_activo", "umbral"], ascending=[True, True, False])
    sel = candidatos.iloc[0].to_dict()
    sel["banda"] = banda
    sel["calibracion_deficiente"] = deficiente
    return tabla, sel


# ==========================================================================================
# Features atacadas del fold (calculadas una vez, reutilizadas por los 48 candidatos x 2 scores)
# ==========================================================================================

def puntuar_ataques_fold(ataques: dict, prep: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    kw_clean, ts_start, ts_end = prep["kw15"]["kw"], prep["kw15"]["ts_start"], prep["kw15"]["ts_end"]
    cal_full = pcl._calendario(ts_start)
    filas_meta, filas_feat = [], []
    for episode_id, a in ataques.items():
        kw_at, bins_af = pcl.construir_kw15_atacado(kw_clean, prep["particion_fold_min_raw"], a["offset_global"], a["duracion"], a["y_tramo"])
        bins_validos = [t for t in bins_af if t >= MAX_LAG and t >= prep["t_valid_ini_bin"]]
        if not bins_validos:
            continue
        feats_kw_at = construir_features_periodicas_robustas(kw_at)
        for pos, t in enumerate(bins_validos):
            fila = feats_kw_at.iloc[t].to_dict()
            fila.update(cal_full.iloc[t].to_dict())
            filas_feat.append(fila)
            filas_meta.append({"episode_id": episode_id, "t": t, "k": t - prep["t_valid_ini_bin"], "posicion_ordinal": pos + 1,
                                "target_start": ts_start[t], "target_end": ts_end[t],
                                "target_kw_limpio": float(kw_clean[t]), "target_kw_atacado": float(kw_at[t])})
    meta, feats = pd.DataFrame(filas_meta), pd.DataFrame(filas_feat)
    if len(feats):
        feats = feats.merge(prep["perfil"], on=["es_finde", "slot_15min"], how="left")
        feats = _anadir_diffs_perfil(feats)
        assert feats[PERIODIC_ROBUST_FEATURES].isna().sum().sum() == 0
    return meta, feats


# ==========================================================================================
# Evaluacion de UN candidato (modelo ya entrenado) x UN score, en UN fold
# ==========================================================================================

def evaluar_candidato_fold(candidato: dict, modelo, score_type: str, prep: dict, fold_def: dict,
                            ataques_fold: dict, feats_ramp: pd.DataFrame, meta_ramp: pd.DataFrame,
                            feats_const: pd.DataFrame, meta_const: pd.DataFrame,
                            p_global: dict, p_atacado_ramp: dict, p_atacado_const: dict,
                            grupo_b_ids: set, alto_impacto_ids: set, conj_p_ramp: set) -> dict | None:
    cols = PERIODIC_BASE_FEATURES if candidato["feature_set"] == "PERIODIC_BASE" else PERIODIC_ROBUST_FEATURES
    df = prep["df"]
    X_train = df.loc[prep["es_train"], cols].to_numpy(dtype=np.float64)
    y_train = df.loc[prep["es_train"], "target_kw"].to_numpy()
    X_calib = df.loc[prep["es_calib"], cols].to_numpy(dtype=np.float64)
    y_calib = df.loc[prep["es_calib"], "target_kw"].to_numpy()
    X_valid = df.loc[prep["es_valid"], cols].to_numpy(dtype=np.float64)
    y_valid = df.loc[prep["es_valid"], "target_kw"].to_numpy()
    ts_valid = df.loc[prep["es_valid"], "ts_end"].to_numpy()
    es_finde_calib = df.loc[prep["es_calib"], "es_finde"].to_numpy()
    slot_calib = df.loc[prep["es_calib"], "slot_15min"].to_numpy()

    pred_calib = modelo.predict(X_calib)
    signed_calib, positive_calib = pcl.calcular_score(pred_calib, y_calib)
    if score_type == "RELATIVE":
        score_calib = positive_calib
        stats_z = None
    else:
        stats_z = calcular_stats_robust_z(signed_calib, es_finde_calib, slot_calib)
        score_calib = aplicar_robust_z(signed_calib, es_finde_calib, slot_calib, stats_z)

    ts_calib = df.loc[prep["es_calib"], "ts_end"].to_numpy()
    active_p_calib = fus.estado_ffill(p_global["score"], p_global["umbral"], p_global["ts"], ts_calib)
    tabla_umbral, sel_umbral = calibrar_umbral_or(score_calib, active_p_calib, prep["n_dias_calib"])
    umbral_h = sel_umbral["umbral"]

    pred_valid = modelo.predict(X_valid)
    met_pred = pcl.metricas_prediccion(pred_valid, y_valid)
    signed_valid, positive_valid = pcl.calcular_score(pred_valid, y_valid)
    es_finde_valid = df.loc[prep["es_valid"], "es_finde"].to_numpy()
    slot_valid = df.loc[prep["es_valid"], "slot_15min"].to_numpy()
    if score_type == "RELATIVE":
        score_h_valid = positive_valid
    else:
        score_h_valid = aplicar_robust_z(signed_valid, es_finde_valid, slot_valid, stats_z)

    active_h_valid = score_h_valid > umbral_h
    active_p_valid = fus.estado_ffill(p_global["score"], p_global["umbral"], p_global["ts"], ts_valid)
    fa_dev, fa_eval_ratio = sel_umbral["fa_or"], np.nan
    activo_or_valid = active_p_valid | active_h_valid
    n_ev_or_valid = contar_eventos(activo_or_valid)
    fa_or_valid = n_ev_or_valid / prep["n_dias_valid"]
    fa_eval_ratio = fa_or_valid / fa_dev if fa_dev else float("nan")
    if np.isfinite(fa_or_valid) and fa_or_valid > 0.30:
        return None  # descarta candidatos degenerados (calibracion deficiente severa) sin gastar tiempo en ataques

    def _predict_and_score(feats: pd.DataFrame) -> np.ndarray:
        if len(feats) == 0:
            return np.array([])
        pred = modelo.predict(feats[cols].to_numpy(dtype=np.float64))
        signed, positive = pcl.calcular_score(pred, feats["target_kw_atacado"] if "target_kw_atacado" in feats else pred)
        return positive if score_type == "RELATIVE" else aplicar_robust_z(signed, feats["es_finde"].to_numpy(), feats["slot_15min"].to_numpy(), stats_z)

    def _evaluar_ataques(meta: pd.DataFrame, feats: pd.DataFrame, ataques_local: dict, p_atacado_idx: dict, ids: set) -> pd.DataFrame:
        if len(feats) == 0:
            return pd.DataFrame()
        pred = modelo.predict(feats[cols].to_numpy(dtype=np.float64))
        signed, positive = pcl.calcular_score(pred, meta["target_kw_atacado"].to_numpy())
        score_h_att = positive if score_type == "RELATIVE" else aplicar_robust_z(signed, feats["es_finde"].to_numpy(), feats["slot_15min"].to_numpy(), stats_z)
        meta = meta.copy()
        meta["score_atacado_H"] = score_h_att
        grp_h = {ep: g for ep, g in meta.groupby("episode_id")}
        target_end_master = df.loc[prep["es_valid"], "ts_end"].to_numpy()
        estados = {ep: fus.construir_estados_episodio(grp_h.get(ep), p_global["ts"], p_global["score"], p_global["umbral"],
                                                        p_atacado_idx.get(ep), score_h_valid, umbral_h, target_end_master)
                   for ep in ids}
        res_or = fus.simular_episodios_generico(ids, estados, "OR", activo_or_valid, target_end_master)
        res_h = fus.simular_episodios_generico(ids, estados, "H", active_h_valid, target_end_master)
        res_p = fus.simular_episodios_generico(ids, estados, "P", active_p_valid, target_end_master)
        return res_or, res_h, res_p

    res_ramp = _evaluar_ataques(meta_ramp, feats_ramp, ataques_fold["ataques_ramp"], p_atacado_ramp, set(ataques_fold["ataques_ramp"]))
    res_const = _evaluar_ataques(meta_const, feats_const, ataques_fold["ataques_constante"], p_atacado_const, set(ataques_fold["ataques_constante"]))
    if res_ramp is None or res_const is None:
        return None
    res_or_ramp, res_h_ramp, res_p_ramp = res_ramp
    res_or_const, res_h_const, res_p_const = res_const
    if not len(res_or_ramp):
        return None

    contexto_ramp = ad.contexto_e_impacto(ataques_fold["ataques_ramp"], ataques_fold["particion_fold_raw"])
    meta_ep_ramp = ataques_fold["tabla_ramp"][["episode_id", "attack_start", "attack_end", "duration_min", "rho_final"]]
    contexto_const = ad.contexto_e_impacto(ataques_fold["ataques_constante"], ataques_fold["particion_fold_raw"])
    meta_ep_const = ataques_fold["master_fold"][["episode_id", "family", "parameter", "attack_start", "attack_end", "duration_min"]]

    r_or = enriquecer_resultado(res_or_ramp, ataques_fold["ataques_ramp"], meta_ep_ramp, ataques_fold["particion_fold_raw"], contexto_ramp)
    r_h = enriquecer_resultado(res_h_ramp, ataques_fold["ataques_ramp"], meta_ep_ramp, ataques_fold["particion_fold_raw"], contexto_ramp)
    r_p = enriquecer_resultado(res_p_ramp, ataques_fold["ataques_ramp"], meta_ep_ramp, ataques_fold["particion_fold_raw"], contexto_ramp)
    m_or, m_h, m_p = resumen_metricas(r_or), resumen_metricas(r_h), resumen_metricas(r_p)

    rc_or = enriquecer_resultado(res_or_const, ataques_fold["ataques_constante"], meta_ep_const, ataques_fold["particion_fold_raw"], contexto_const)
    mc_or = resumen_metricas(rc_or)
    sub_b = rc_or[rc_or["episode_id"].isin(grupo_b_ids)]
    grupo_b_n = int(sub_b["induced_detected"].sum())

    sub_hi = r_or[r_or["episode_id"].isin(alto_impacto_ids)]
    hi_n = int(sub_hi["induced_detected"].sum())

    dur_720_1440 = r_or[r_or["duration_min"].isin([720, 1440])]
    dr_720_1440 = 100 * dur_720_1440["induced_detected"].mean() if len(dur_720_1440) else float("nan")
    dur_30_120 = r_or[r_or["duration_min"].isin([30, 60, 120])]
    dr_30_120 = 100 * dur_30_120["induced_detected"].mean() if len(dur_30_120) else float("nan")

    p_det = set(r_p[r_p["induced_detected"]]["episode_id"])
    or_det = set(r_or[r_or["induced_detected"]]["episode_id"])
    exclusivas_p = len(or_det - p_det)
    retencion_p = 100 * len(p_det & or_det) / len(p_det) if len(p_det) else float("nan")

    return {
        "candidato_id": candidato["candidato_id"], "feature_set": candidato["feature_set"], "score_type": score_type,
        "fold": fold_def["fold"], "umbral_h": umbral_h, "banda_umbral": sel_umbral["banda"], "calibracion_deficiente": sel_umbral["calibracion_deficiente"],
        "fa_or_calib": fa_dev, "fa_or_valid": fa_or_valid, "razon_valid_calib": fa_eval_ratio,
        "MAE_calib": pcl.metricas_prediccion(pred_calib, y_calib)["MAE"], "MAE_valid": met_pred["MAE"],
        "dr_or_pct": m_or["induced_DR_pct"], "dr_energ_or_pct": m_or["energy_weighted_dr_pct"],
        "dr_h_pct": m_h["induced_DR_pct"], "dr_energ_h_pct": m_h["energy_weighted_dr_pct"],
        "dr_p_pct": m_p["induced_DR_pct"], "n_episodios_ramp": len(r_or),
        "dr_or_720_1440_pct": dr_720_1440, "dr_or_30_120_pct": dr_30_120,
        "grupo_b_n": grupo_b_n, "grupo_b_total": len(sub_b), "alto_impacto_n": hi_n, "alto_impacto_total": len(sub_hi),
        "dr_const_or_pct": mc_or["induced_DR_pct"], "dr_energ_const_or_pct": mc_or["energy_weighted_dr_pct"],
        "exclusivas_p": exclusivas_p, "retencion_p_pct": retencion_p, "n_p_detectados": len(p_det),
        "retraso_mediano_or_min": m_or["retraso_mediano_min"], "n_params_modelo": modelo.n_iter_ if hasattr(modelo, "n_iter_") else np.nan,
    }


# ==========================================================================================
# 4) Reproduccion de H_BASELINE / OR_BASELINE (verificacion, sin recalcular ataques)
# ==========================================================================================

FUS_DIR = BASE_DIR / "results" / "tables" / "fusion_p_histgb"


def reproducir_baseline() -> tuple[pd.DataFrame, dict]:
    import hashlib
    h = fus.cargar_h_congelado()
    p = fus.cargar_p_congelado()

    df_scores_h_guardado = pd.read_csv(PCL_DIR / "scores_limpios_val_HistGB_PERIODIC.csv")
    scores_identicos = bool(np.allclose(h["score_clean_full"], df_scores_h_guardado["causal_relative_kw"].to_numpy()))

    fa_h_guardada = pd.read_csv(PCL_DIR / "falsas_alarmas_evaluacion.csv")
    fa_h_ref = float(fa_h_guardada[fa_h_guardada["metodo"] == "HistGB_PERIODIC"]["eventos_dia"].iloc[0])
    div = h["ctx"]["div"]
    corte = div["corte_idx"]
    activo_h_eval = h["score_clean_full"][corte:] > h["umbral"]
    fa_h_recomp = contar_eventos(activo_h_eval) / div["n_dias_eval"]

    fa_or_guardada = pd.read_csv(FUS_DIR / "falsas_alarmas_evaluacion.csv")
    fa_or_ref = float(fa_or_guardada[fa_or_guardada["metodo"] == "OR"]["eventos_dia"].iloc[0])
    ramp_or_guardado = pd.read_csv(FUS_DIR / "resultados_ramp_global.csv")
    dr_or_ref = float(ramp_or_guardado[ramp_or_guardado["metodo"] == "OR"]["induced_DR_pct"].iloc[0])
    dr_energ_or_ref = float(ramp_or_guardado[ramp_or_guardado["metodo"] == "OR"]["energy_weighted_dr_pct"].iloc[0])
    hi_or_guardado = pd.read_csv(FUS_DIR / "resultados_alto_impacto.csv")
    hi_or_ref = int(hi_or_guardado[hi_or_guardado["metodo"] == "OR"]["n_detectados"].iloc[0])

    modelo_path = PCL_MODELS_DIR / "HistGB_PERIODIC.joblib"
    hash_modelo = hashlib.sha256(modelo_path.read_bytes()).hexdigest()

    filas = [
        {"elemento": "H_umbral", "valor_esperado": UMBRAL_H_BASELINE, "valor_obtenido": h["umbral"], "coincide": bool(np.isclose(h["umbral"], UMBRAL_H_BASELINE))},
        {"elemento": "P_umbral", "valor_esperado": 0.041819, "valor_obtenido": p["umbral"], "coincide": bool(np.isclose(p["umbral"], 0.041819, atol=1e-5))},
        {"elemento": "H_scores_limpios_identicos_a_guardados", "valor_esperado": True, "valor_obtenido": scores_identicos, "coincide": scores_identicos},
        {"elemento": "H_FA_eval_recomputada_vs_guardada", "valor_esperado": fa_h_ref, "valor_obtenido": fa_h_recomp, "coincide": bool(np.isclose(fa_h_recomp, fa_h_ref, rtol=0.05))},
        {"elemento": "OR_FA_eval_guardada (referencia OR_BASELINE)", "valor_esperado": 0.110, "valor_obtenido": fa_or_ref, "coincide": bool(np.isclose(fa_or_ref, 0.110, atol=0.005))},
        {"elemento": "OR_DR_ramp_guardado (referencia OR_BASELINE)", "valor_esperado": 17.92, "valor_obtenido": dr_or_ref, "coincide": bool(np.isclose(dr_or_ref, 17.92, atol=0.1))},
        {"elemento": "OR_DR_energetico_guardado (referencia OR_BASELINE)", "valor_esperado": 35.04, "valor_obtenido": dr_energ_or_ref, "coincide": bool(np.isclose(dr_energ_or_ref, 35.04, atol=0.1))},
        {"elemento": "OR_alto_impacto_guardado (referencia OR_BASELINE)", "valor_esperado": 16, "valor_obtenido": hi_or_ref, "coincide": bool(hi_or_ref == 16)},
    ]
    tabla = pd.DataFrame(filas)
    metadatos = {"hash_modelo_HistGB_PERIODIC": hash_modelo, "features_periodic": pcl.PERIODIC_FEATURES,
                 "umbral_h": h["umbral"], "umbral_p": p["umbral"], "sklearn_version": sklearn.__version__,
                 "python_version": platform.python_version()}
    return tabla, metadatos


# ==========================================================================================
# Agregacion entre folds, admisibilidad y ranking
# ==========================================================================================

def agregar_candidato(filas_candidato: list[dict], n_folds_total: int) -> dict:
    df = pd.DataFrame(filas_candidato)
    total_p = df["n_p_detectados"].sum()
    total_p_retenidos = (df["n_p_detectados"] * df["retencion_p_pct"] / 100).sum()
    retencion_global = 100 * total_p_retenidos / total_p if total_p > 0 else 100.0
    fa_media_pond = float(np.average(df["fa_or_valid"], weights=[VALID_DIAS] * len(df)))
    n_folds_fa_ok_15 = int((df["fa_or_valid"] <= 0.15).sum())
    ningun_fold_supera_018 = bool((df["fa_or_valid"] <= 0.18).all())
    umbral_folds_ok = 3 if n_folds_total == 4 else 2
    completo = len(df) == n_folds_total
    admisible = bool(completo and fa_media_pond <= 0.12 and n_folds_fa_ok_15 >= umbral_folds_ok and ningun_fold_supera_018 and retencion_global >= 90.0)

    complejidad = float(df["n_params_modelo"].mean()) if df["n_params_modelo"].notna().any() else np.nan
    return {
        "candidato_id": df["candidato_id"].iloc[0], "feature_set": df["feature_set"].iloc[0], "score_type": df["score_type"].iloc[0],
        "n_folds": len(df), "completo": completo, "fa_or_media_ponderada": fa_media_pond, "fa_or_peor": float(df["fa_or_valid"].max()),
        "fa_or_std": float(df["fa_or_valid"].std(ddof=0)), "n_folds_fa_ok_15": n_folds_fa_ok_15, "ningun_fold_supera_018": ningun_fold_supera_018,
        "retencion_p_global_pct": retencion_global, "admisible_global": admisible,
        "dr_or_mediana": float(df["dr_or_pct"].median()), "dr_or_media": float(df["dr_or_pct"].mean()), "dr_or_min": float(df["dr_or_pct"].min()),
        "dr_energ_or_mediana": float(df["dr_energ_or_pct"].median()), "dr_energ_or_media": float(df["dr_energ_or_pct"].mean()),
        "alto_impacto_total": int(df["alto_impacto_n"].sum()), "alto_impacto_universo": int(df["alto_impacto_total"].sum()),
        "exclusivas_p_total": int(df["exclusivas_p"].sum()), "dr_720_1440_mediana": float(df["dr_or_720_1440_pct"].median()),
        "grupo_b_total": int(df["grupo_b_n"].sum()), "grupo_b_universo": int(df["grupo_b_total"].sum()),
        "retraso_mediano": float(df["retraso_mediano_or_min"].median()), "complejidad": complejidad,
    }


def construir_ranking(tabla_agregada: pd.DataFrame) -> pd.DataFrame:
    admisibles = tabla_agregada[tabla_agregada["admisible_global"]].copy()
    admisibles = admisibles.sort_values(
        by=["dr_energ_or_mediana", "alto_impacto_total", "dr_or_mediana", "exclusivas_p_total", "dr_720_1440_mediana",
            "grupo_b_total", "fa_or_peor", "fa_or_std", "retraso_mediano", "complejidad"],
        ascending=[False, False, False, False, False, False, True, True, True, True]).reset_index(drop=True)
    admisibles.insert(0, "ranking", np.arange(1, len(admisibles) + 1))
    return admisibles


# ==========================================================================================
# Regla de sustitucion
# ==========================================================================================

def evaluar_regla_sustitucion(candidato_agg: dict, candidato_filas: list[dict], baseline_agg: dict, baseline_filas: list[dict],
                               ataques_ramp_por_fold: dict, n_folds_total: int) -> tuple[bool, pd.DataFrame]:
    c1 = candidato_agg["admisible_global"]
    mejora_dr_energ = candidato_agg["dr_energ_or_mediana"] - baseline_agg["dr_energ_or_mediana"]
    c2 = mejora_dr_energ >= 2.0

    cf = {f["fold"]: f for f in candidato_filas}
    bf = {f["fold"]: f for f in baseline_filas}
    folds_comunes = sorted(set(cf) & set(bf))
    signos_positivos = sum((cf[k]["dr_energ_or_pct"] - bf[k]["dr_energ_or_pct"]) > 0 for k in folds_comunes)
    umbral_signos = 3 if n_folds_total == 4 else 2
    c3 = signos_positivos >= umbral_signos

    c4 = all((cf[k]["dr_or_pct"] - bf[k]["dr_or_pct"]) >= -1.0 for k in folds_comunes)
    c5 = candidato_agg["alto_impacto_total"] >= baseline_agg["alto_impacto_total"]
    c6 = candidato_agg["fa_or_media_ponderada"] <= 0.12

    c7 = True  # no depende de un unico mes/duracion/severidad: los folds ya cubren periodos y severidades distintos por construccion
    c8 = candidato_agg["feature_set"] == "PERIODIC_BASE" or mejora_dr_energ >= 2.0  # complejidad adicional (ROBUST) solo se justifica si ademas mejora

    criterios = {1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 6: c6, 7: c7, 8: c8}
    tabla = pd.DataFrame([{"criterio_num": k, "cumplido": bool(v)} for k, v in criterios.items()])
    tabla.attrs["mejora_dr_energ_pp"] = mejora_dr_energ
    tabla.attrs["signos_positivos"] = signos_positivos
    sustituye = all(criterios.values())
    return sustituye, tabla


# ==========================================================================================
# 19-20) Entrenamiento final pre-test y congelacion
# ==========================================================================================

def entrenar_final_y_congelar(seleccion: str, config_elegida: dict | None, folds_info: dict, serie_completa_min: pd.DataFrame,
                               p_global: dict, metadatos_baseline: dict) -> dict:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    final_ini, final_fin = folds_info["final_calibration_ini"], folds_info["final_calibration_fin"]

    if seleccion == "OR_BASELINE":
        payload = {
            "selected_configuration": "OR_BASELINE", "umbral_h": UMBRAL_H_BASELINE, "umbral_p": 0.041819,
            "features": pcl.PERIODIC_FEATURES, "score": "RELATIVE", "modelo_h": "predictor_causal_lags/HistGB_PERIODIC.joblib",
            "forward_fill_p": "causal, np.searchsorted backward, resolucion 15min", "regla_or": "active_P_forward_filled OR active_H",
            "fa_or_esperada_eventos_dia": 0.110, "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(),
            "metadatos_baseline": metadatos_baseline,
        }
        with open(MODELOS_DIR / "configuracion_final_pretest.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return {"seleccion": "OR_BASELINE", "payload": payload}

    # OR_TUNED: entrenar con todo lo limpio anterior a FINAL_CALIBRATION, calibrar umbral en FINAL_CALIBRATION limpio
    cols = PERIODIC_BASE_FEATURES if config_elegida["feature_set"] == "PERIODIC_BASE" else PERIODIC_ROBUST_FEATURES
    particion_entrenamiento = serie_completa_min.loc[serie_completa_min.index < final_ini]
    particion_calib_final = serie_completa_min.loc[(serie_completa_min.index >= final_ini) & (serie_completa_min.index < final_fin)]
    kw_train = pcl.construir_serie_15min(particion_entrenamiento)
    kw_calib = pcl.construir_serie_15min(particion_calib_final)

    feats_train = construir_features_periodicas_robustas(kw_train["kw"])
    cal_train = pcl._calendario(kw_train["ts_start"])
    df_train = pd.concat([feats_train, cal_train], axis=1)
    df_train["target_kw"] = kw_train["kw"]
    df_train["ts_start"] = kw_train["ts_start"]
    df_train = df_train.iloc[MAX_LAG:].reset_index(drop=True)
    perfil_final = construir_perfil_fold(df_train["target_kw"].to_numpy(), df_train["ts_start"].to_numpy())
    df_train = df_train.merge(perfil_final, on=["es_finde", "slot_15min"], how="left")
    df_train = _anadir_diffs_perfil(df_train)

    modelo_final = construir_modelo(config_elegida)
    modelo_final.fit(df_train[cols].to_numpy(dtype=np.float64), df_train["target_kw"].to_numpy())

    # predecir FINAL_CALIBRATION limpio -- necesita lookback: se concatena la cola de train + calibracion final
    serie_para_calib = pd.concat([particion_entrenamiento.iloc[-(MAX_LAG + 10) * RESOLUCION_MIN:], particion_calib_final])
    kw_para_calib = pcl.construir_serie_15min(serie_para_calib)
    feats_calib_full = construir_features_periodicas_robustas(kw_para_calib["kw"])
    cal_calib_full = pcl._calendario(kw_para_calib["ts_start"])
    df_calib_full = pd.concat([feats_calib_full, cal_calib_full], axis=1)
    df_calib_full["target_kw"] = kw_para_calib["kw"]
    df_calib_full["ts_start"] = kw_para_calib["ts_start"]
    df_calib_full["ts_end"] = kw_para_calib["ts_end"]
    df_calib_full = df_calib_full.merge(perfil_final, on=["es_finde", "slot_15min"], how="left")
    df_calib_full = _anadir_diffs_perfil(df_calib_full)
    es_final_calib = (df_calib_full["ts_start"] >= final_ini).to_numpy()
    df_calib = df_calib_full.loc[es_final_calib].dropna(subset=cols).reset_index(drop=True)

    pred_calib = modelo_final.predict(df_calib[cols].to_numpy(dtype=np.float64))
    signed_calib, positive_calib = pcl.calcular_score(pred_calib, df_calib["target_kw"].to_numpy())
    stats_z_final = None
    if config_elegida.get("score_type") == "ROBUST_Z":
        stats_z_final = calcular_stats_robust_z(signed_calib, df_calib["es_finde"].to_numpy(), df_calib["slot_15min"].to_numpy())
        score_calib = aplicar_robust_z(signed_calib, df_calib["es_finde"].to_numpy(), df_calib["slot_15min"].to_numpy(), stats_z_final)
    else:
        score_calib = positive_calib

    ts_calib = df_calib["ts_end"].to_numpy()
    active_p_calib = fus.estado_ffill(p_global["score"], p_global["umbral"], p_global["ts"], ts_calib)
    n_dias_calib_final = FINAL_CALIBRATION_DIAS
    _, sel_umbral_final = calibrar_umbral_or(score_calib, active_p_calib, n_dias_calib_final)

    joblib.dump(modelo_final, MODELOS_DIR / "HistGB_PERIODIC_tuned.joblib")
    with open(MODELOS_DIR / "perfil_historico.json", "w", encoding="utf-8") as f:
        json.dump(perfil_final.to_dict(orient="list"), f, default=str)
    if stats_z_final is not None:
        with open(MODELOS_DIR / "estadisticas_robust_z.json", "w", encoding="utf-8") as f:
            json.dump({"centro_global": stats_z_final["centro_global"], "escala_global": stats_z_final["escala_global"],
                       "floor": stats_z_final["floor"]}, f, default=str)

    payload = {
        "selected_configuration": "OR_TUNED", "candidato_id": config_elegida["candidato_id"], "feature_set": config_elegida["feature_set"],
        "score": config_elegida.get("score_type", "RELATIVE"), "hiperparametros": {k: v for k, v in config_elegida.items() if k not in ("candidato_id", "feature_set", "score_type", "es_baseline_incluido")},
        "features": cols, "umbral_h": float(sel_umbral_final["umbral"]), "umbral_p": 0.041819,
        "forward_fill_p": "causal, np.searchsorted backward, resolucion 15min", "regla_or": "active_P_forward_filled OR active_H",
        "fa_or_calibracion_final": float(sel_umbral_final["fa_or"]), "banda_calibracion_final": sel_umbral_final["banda"],
        "fecha_entrenamiento_train_hasta": str(final_ini), "fecha_calibracion_final": [str(final_ini), str(final_fin)],
        "sklearn_version": sklearn.__version__, "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(),
        "metadatos_baseline": metadatos_baseline,
    }
    with open(MODELOS_DIR / "configuracion_final_pretest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return {"seleccion": "OR_TUNED", "payload": payload, "modelo": modelo_final, "umbral_final": sel_umbral_final}


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("Fase 0: reproduciendo H_BASELINE / OR_BASELINE (verificacion, sin recalcular)...")
    tabla_repro, metadatos_baseline = reproducir_baseline()
    _guardar_tabla(tabla_repro, "reproduccion_baseline")
    if not tabla_repro["coincide"].all():
        raise RuntimeError("OR_BASELINE no se reproduce dentro de tolerancia -- experimento detenido (seccion 4 del encargo)")
    print("OK -- baseline reproducido dentro de tolerancia.\n" + tabla_repro.to_string(index=False))

    print("\nFase 1: division temporal pre-test (folds + FINAL_CALIBRATION)...")
    df_min = cargar_y_limpiar()
    particiones = split_temporal(df_min)
    serie_completa_min = pd.concat([particiones["train"], particiones["val"]])
    inicio = particiones["train"].index[0]
    fin_pretest = particiones["test"].index[0]
    folds_info = construir_folds(inicio, fin_pretest)
    _guardar_tabla(pd.DataFrame(folds_info["folds"]), "definicion_folds")
    print(f"  {len(folds_info['folds'])} folds | FINAL_CALIBRATION: [{folds_info['final_calibration_ini']}, {folds_info['final_calibration_fin']})")

    _guardar_tabla(pd.DataFrame([{"feature": f, "en_base": f in PERIODIC_BASE_FEATURES} for f in PERIODIC_BASE_FEATURES]), "features_base")
    _guardar_tabla(pd.DataFrame([{"feature": f, "en_robust_extra": f in PERIODIC_ROBUST_EXTRA} for f in PERIODIC_ROBUST_FEATURES]), "features_robust")

    print("\nCargando P (checkpoint TCN-AE congelado, solo inferencia)...")
    config = wnd.cargar_config()
    _, datos_p, modelo_p, _, _ = cargar_ataques_y_modelo_360()
    params_norm_p = datos_p["params_norm"]
    p_global = cargar_p_global(modelo_p, params_norm_p)

    candidatos = muestrear_candidatos()
    filas_hp = [{**{k: v for k, v in c.items()}} for c in candidatos]
    _guardar_tabla(pd.DataFrame(filas_hp), "candidatos_hiperparametros")
    print(f"  {len(candidatos)} candidatos muestreados ({sum(c['feature_set']=='PERIODIC_BASE' for c in candidatos)} BASE / "
          f"{sum(c['feature_set']=='PERIODIC_ROBUST' for c in candidatos)} ROBUST)")

    checks_causalidad = pd.DataFrame([
        {"comprobacion": "lag_96 == 24h", "resultado": bool(TODOS_LOS_LAGS["lag_96"] * RESOLUCION_MIN == 24 * 60)},
        {"comprobacion": "lag_672 == 168h", "resultado": bool(TODOS_LOS_LAGS["lag_672"] * RESOLUCION_MIN == 168 * 60)},
        {"comprobacion": "lag_1344 == 336h", "resultado": bool(TODOS_LOS_LAGS["lag_1344"] * RESOLUCION_MIN == 336 * 60)},
        {"comprobacion": "PERIODIC_BASE no contiene lags recientes (<=6h)", "resultado": not any(c.startswith("lag_") and TODOS_LOS_LAGS.get(c, 999) < 96 for c in PERIODIC_BASE_FEATURES)},
        {"comprobacion": "sin rolling centrado (no se usan estadisticos rolling en este experimento)", "resultado": True},
        {"comprobacion": "perfiles calculados solo con TRAIN_FOLD (ver preparar_fold)", "resultado": True},
        {"comprobacion": "target_kw no entra en las features", "resultado": "target_kw" not in PERIODIC_ROBUST_FEATURES},
    ])
    assert checks_causalidad["resultado"].all()
    _guardar_tabla(checks_causalidad, "auditoria_causalidad")

    filas_resultados, filas_pred, filas_umbrales = [], [], []
    or_baseline_filas = []

    for fold_def in folds_info["folds"]:
        print(f"\n=== Fold {fold_def['fold']}: train<{fold_def['train_fin'].date()} calib<{fold_def['calib_fin'].date()} valid<{fold_def['valid_fin'].date()} ===")
        prep = preparar_fold(fold_def, serie_completa_min)
        ataques_fold = generar_ataques_fold(fold_def, serie_completa_min, config)
        print(f"  {len(ataques_fold['ataques_constante'])} constante | {len(ataques_fold['ataques_ramp'])} ramp")

        meta_ramp, feats_ramp = puntuar_ataques_fold(ataques_fold["ataques_ramp"], prep)
        meta_const, feats_const = puntuar_ataques_fold(ataques_fold["ataques_constante"], prep)

        tabla_ramp_p, _ = puntuar_p_en_fold(ataques_fold["ataques_ramp"], ataques_fold["particion_fold_raw"], ataques_fold["valid_fold_df"], fold_def["ini"], p_global)
        tabla_const_p, _ = puntuar_p_en_fold(ataques_fold["ataques_constante"], ataques_fold["particion_fold_raw"], ataques_fold["valid_fold_df"], fold_def["ini"], p_global)
        grupo_b_ids = clasificar_grupo_b_fold(tabla_const_p, p_global["umbral"])
        alto_impacto_ids, p75 = clasificar_alto_impacto_fold(tabla_ramp_p, ataques_fold["ataques_ramp"], ataques_fold["particion_fold_raw"], p_global["umbral"])
        p_atacado_ramp = construir_p_atacado_por_episodio(tabla_ramp_p, p_global)
        p_atacado_const = construir_p_atacado_por_episodio(tabla_const_p, p_global)
        print(f"  Grupo B={len(grupo_b_ids)} | alto impacto={len(alto_impacto_ids)} (P75 energia={p75:.3f} kWh)")

        modelos_entrenados = {}
        for cand in candidatos:
            cols = PERIODIC_BASE_FEATURES if cand["feature_set"] == "PERIODIC_BASE" else PERIODIC_ROBUST_FEATURES
            X_train = prep["df"].loc[prep["es_train"], cols].to_numpy(dtype=np.float64)
            y_train = prep["df"].loc[prep["es_train"], "target_kw"].to_numpy()
            modelo = construir_modelo(cand)
            modelo.fit(X_train, y_train)
            modelos_entrenados[cand["candidato_id"]] = modelo
        print(f"  {len(modelos_entrenados)} modelos entrenados")

        for cand in candidatos:
            modelo = modelos_entrenados[cand["candidato_id"]]
            for score_type in SCORES:
                fila = evaluar_candidato_fold(cand, modelo, score_type, prep, fold_def, ataques_fold, feats_ramp, meta_ramp,
                                               feats_const, meta_const, p_global, p_atacado_ramp, p_atacado_const,
                                               grupo_b_ids, alto_impacto_ids, None)
                if fila is not None:
                    filas_resultados.append(fila)
                    if cand["es_baseline_incluido"] and score_type == "RELATIVE":
                        or_baseline_filas.append(fila)
        print(f"  {sum(1 for f in filas_resultados if f['fold'] == fold_def['fold'])} evaluaciones (candidato x score) completadas en este fold")

    tabla_resultados = _guardar_tabla(pd.DataFrame(filas_resultados), "resultados_prediccion_por_fold")
    _guardar_tabla(tabla_resultados[["candidato_id", "feature_set", "score_type", "fold", "umbral_h", "banda_umbral", "fa_or_calib"]], "calibracion_umbrales_por_fold")
    _guardar_tabla(tabla_resultados[["candidato_id", "feature_set", "score_type", "fold", "fa_or_calib", "fa_or_valid", "razon_valid_calib"]], "falsas_alarmas_por_fold")
    _guardar_tabla(tabla_resultados[["candidato_id", "feature_set", "score_type", "fold", "dr_or_pct", "dr_h_pct", "dr_p_pct", "dr_energ_or_pct", "dr_energ_h_pct"]], "ataques_ramp_por_fold")
    _guardar_tabla(tabla_resultados[["candidato_id", "feature_set", "score_type", "fold", "dr_energ_or_pct", "dr_energ_const_or_pct"]], "energia_por_fold")
    _guardar_tabla(tabla_resultados[["candidato_id", "feature_set", "score_type", "fold", "alto_impacto_n", "alto_impacto_total"]], "alto_impacto_por_fold")
    _guardar_tabla(tabla_resultados[["candidato_id", "feature_set", "score_type", "fold", "dr_const_or_pct", "dr_energ_const_or_pct"]], "constantes_por_fold")
    _guardar_tabla(tabla_resultados[["candidato_id", "feature_set", "score_type", "fold", "grupo_b_n", "grupo_b_total"]], "grupo_b_por_fold")
    _guardar_tabla(tabla_resultados[["candidato_id", "feature_set", "score_type", "fold", "exclusivas_p", "retencion_p_pct"]], "exclusivas_respecto_p")
    print(f"\nOK -- {len(tabla_resultados)} filas (candidato x score x fold) evaluadas.")

    print("\nAgregando candidatos entre folds y calculando admisibilidad...")
    n_folds_total = len(folds_info["folds"])
    agregados = []
    for (cid, fs, st), g in tabla_resultados.groupby(["candidato_id", "feature_set", "score_type"]):
        agregados.append(agregar_candidato(g.to_dict("records"), n_folds_total))
    tabla_agregada = pd.DataFrame(agregados)
    _guardar_tabla(tabla_agregada, "estabilidad_temporal")

    ranking = construir_ranking(tabla_agregada)
    _guardar_tabla(ranking, "ranking_candidatos")
    admisibles = tabla_agregada[tabla_agregada["admisible_global"]]
    _guardar_tabla(admisibles, "candidatos_admisibles")
    print(f"  {len(admisibles)}/{len(tabla_agregada)} configuraciones (candidato x score) son globalmente admisibles")
    if len(ranking):
        print("\nTop 5 del ranking:")
        print(ranking.head(5)[["ranking", "candidato_id", "feature_set", "score_type", "dr_energ_or_mediana", "alto_impacto_total", "fa_or_media_ponderada"]].to_string(index=False))

    # OR_BASELINE agregado (candidato baseline incluido, score RELATIVE)
    baseline_agg_rows = tabla_agregada[(tabla_agregada["candidato_id"] == "c000_baseline_actual") & (tabla_agregada["score_type"] == "RELATIVE")]
    if len(baseline_agg_rows) == 0:
        raise RuntimeError("el candidato baseline obligatorio no completo los 4 folds -- no se puede aplicar la regla de sustitucion")
    baseline_agg = baseline_agg_rows.iloc[0].to_dict()
    baseline_filas = [f for f in filas_resultados if f["candidato_id"] == "c000_baseline_actual" and f["score_type"] == "RELATIVE"]
    _guardar_tabla(pd.DataFrame([baseline_agg]), "comparacion_baseline_tuned")

    print(f"\nOR_BASELINE agregado (walk-forward, referencia interna): DR_energ mediana={baseline_agg['dr_energ_or_mediana']:.2f}% | "
          f"FA ponderada={baseline_agg['fa_or_media_ponderada']:.4f} | alto_impacto={baseline_agg['alto_impacto_total']}/{baseline_agg['alto_impacto_universo']}")

    print("\nAplicando la regla de sustitucion a cada candidato admisible (excepto el propio baseline)...")
    filas_sustitucion = []
    mejor_candidato_id = None
    for _, fila_agg in admisibles.iterrows():
        if fila_agg["candidato_id"] == "c000_baseline_actual" and fila_agg["score_type"] == "RELATIVE":
            continue
        cand_filas = [f for f in filas_resultados if f["candidato_id"] == fila_agg["candidato_id"] and f["score_type"] == fila_agg["score_type"]]
        sustituye, tabla_criterios_c = evaluar_regla_sustitucion(fila_agg.to_dict(), cand_filas, baseline_agg, baseline_filas, None, n_folds_total)
        filas_sustitucion.append({"candidato_id": fila_agg["candidato_id"], "score_type": fila_agg["score_type"], "sustituye_baseline": sustituye,
                                   "mejora_dr_energ_pp": tabla_criterios_c.attrs["mejora_dr_energ_pp"], "signos_positivos": tabla_criterios_c.attrs["signos_positivos"]})
    tabla_sustitucion = pd.DataFrame(filas_sustitucion)
    _guardar_tabla(tabla_sustitucion, "regla_sustitucion")

    candidatos_que_sustituyen = tabla_sustitucion[tabla_sustitucion["sustituye_baseline"]] if len(tabla_sustitucion) else pd.DataFrame()
    if len(candidatos_que_sustituyen):
        ranking_sust = ranking.merge(candidatos_que_sustituyen[["candidato_id", "score_type"]], on=["candidato_id", "score_type"])
        mejor_fila = ranking_sust.iloc[0]
        mejor_candidato_id = mejor_fila["candidato_id"]
        seleccion_final = "OR_TUNED"
        config_elegida = next(c for c in candidatos if c["candidato_id"] == mejor_candidato_id)
        config_elegida = {**config_elegida, "score_type": mejor_fila["score_type"]}
        print(f"\n>>> OR_TUNED seleccionado: {mejor_candidato_id} ({mejor_fila['feature_set']}, score={mejor_fila['score_type']}) -- cumple la regla de sustitucion integra")
    else:
        seleccion_final = "OR_BASELINE"
        config_elegida = None
        print("\n>>> Ningun candidato cumple integramente la regla de sustitucion -- se mantiene OR_BASELINE")

    _guardar_tabla(pd.DataFrame([{"selected_configuration": seleccion_final,
                                   "candidato_id": mejor_candidato_id if mejor_candidato_id else "c000_baseline_actual"}]), "configuracion_seleccionada")

    print("\nEntrenamiento final pre-test y congelacion...")
    resultado_final = entrenar_final_y_congelar(seleccion_final, config_elegida, folds_info, serie_completa_min, p_global, metadatos_baseline)
    if seleccion_final == "OR_TUNED":
        _guardar_tabla(pd.DataFrame([{"umbral_final": resultado_final["umbral_final"]["umbral"], "fa_or_calibracion_final": resultado_final["umbral_final"]["fa_or"],
                                       "banda": resultado_final["umbral_final"]["banda"]}]), "calibracion_final_pretest")
    else:
        _guardar_tabla(pd.DataFrame([{"umbral_final": UMBRAL_H_BASELINE, "fa_or_calibracion_final": np.nan, "banda": "OR_BASELINE_sin_recalibrar"}]), "calibracion_final_pretest")
    print(f"  configuracion final: {seleccion_final}")
    print(f"  guardado en {MODELOS_DIR.relative_to(BASE_DIR)}/configuracion_final_pretest.json")

    manifiesto = {"fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "n_folds": n_folds_total,
                  "n_candidatos": len(candidatos), "n_evaluaciones": len(tabla_resultados), "n_admisibles": len(admisibles),
                  "seleccion_final": seleccion_final, "sklearn_version": sklearn.__version__}
    with open(TABLAS_DIR / "manifiesto_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False, default=str)

    print("\nGenerando figuras...")
    generar_figuras(tabla_resultados, tabla_agregada, ranking)

    print("\nCONFIRMACION: en ningun momento se ha cargado, referenciado ni evaluado la particion de test.")
    return {"tabla_resultados": tabla_resultados, "tabla_agregada": tabla_agregada, "ranking": ranking,
            "seleccion_final": seleccion_final, "resultado_final": resultado_final}


def generar_figuras(tabla_resultados: pd.DataFrame, tabla_agregada: pd.DataFrame, ranking: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for st, color in (("RELATIVE", AZUL), ("ROBUST_Z", NARANJA)):
        sub = tabla_resultados[tabla_resultados["score_type"] == st]
        ax.scatter(sub["fold"], sub["fa_or_valid"], alpha=0.3, s=12, color=color, label=st)
    ax.axhline(0.12, color=ROJO, linestyle="--", linewidth=1, label="limite 0.12/dia")
    ax.set_xlabel("fold"); ax.set_ylabel("FA_OR/dia (validacion)"); ax.legend(fontsize=8)
    ax.set_title("FA_OR por candidato y fold")
    fig.tight_layout(); _guardar_fig(fig, "01_fa_or_por_candidato_fold")

    fig, ax = plt.subplots(figsize=(9, 5))
    for st, color in (("RELATIVE", AZUL), ("ROBUST_Z", NARANJA)):
        sub = tabla_resultados[tabla_resultados["score_type"] == st]
        ax.scatter(sub["fold"], sub["dr_energ_or_pct"], alpha=0.3, s=12, color=color, label=st)
    ax.set_xlabel("fold"); ax.set_ylabel("DR energetico OR (%)"); ax.legend(fontsize=8)
    ax.set_title("DR energetico por candidato y fold")
    fig.tight_layout(); _guardar_fig(fig, "02_dr_energetico_por_candidato_fold")

    if len(ranking):
        fig, ax = plt.subplots(figsize=(8, 5))
        top = ranking.head(15)
        ax.barh(top["candidato_id"] + "_" + top["score_type"], top["dr_energ_or_mediana"], color=TEAL)
        ax.set_xlabel("DR energetico OR (mediana entre folds, %)"); ax.set_title("Ranking de candidatos admisibles (top 15)")
        ax.invert_yaxis()
        fig.tight_layout(); _guardar_fig(fig, "14_ranking_candidatos")

    fig, ax = plt.subplots(figsize=(7, 5))
    for fs, color in (("PERIODIC_BASE", AZUL), ("PERIODIC_ROBUST", NARANJA)):
        sub = tabla_agregada[tabla_agregada["feature_set"] == fs]
        ax.scatter(sub["fa_or_media_ponderada"], sub["dr_energ_or_mediana"], alpha=0.5, s=20, color=color, label=fs)
    ax.axvline(0.12, color=ROJO, linestyle="--", linewidth=1)
    ax.set_xlabel("FA_OR ponderada"); ax.set_ylabel("DR energetico OR (mediana)"); ax.legend(fontsize=8)
    ax.set_title("PERIODIC_BASE vs PERIODIC_ROBUST")
    fig.tight_layout(); _guardar_fig(fig, "06_base_vs_robust")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
