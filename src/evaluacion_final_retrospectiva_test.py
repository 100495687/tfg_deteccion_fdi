"""Evaluacion final confirmatoria retrospectiva de la arquitectura congelada
OR_PMULTI_HMULTIWINDOW (P_MULTI_SEASON OR HistGB_PERIODIC/H_MULTIWINDOW_180) sobre test
limpio y los 1680 episodios de ataque congelados en manifests/test_attacks_v2/.

evaluation_status = RETROSPECTIVE_CONFIRMATORY_EVALUATION (no PRISTINE_UNSEEN_TEST, ver
docs/auditoria_uso_historico_test.md). Esta es la unica ejecucion: no se modifica nada de
la arquitectura, del manifiesto ni de los criterios despues de observar resultados.

Reutiliza literalmente, sin modificar: optimizacion_histgb_periodic.{puntuar_p_en_fold,
construir_p_atacado_por_episodio, PERIODIC_BASE_FEATURES, reproducir_baseline},
congelacion_final_or_pretest.predecir_pool_limpio, predictor_causal_lags.{puntuar_ataques,
construir_serie_15min, calcular_score, MAX_LAG}, fusion_p_histgb.{construir_estados_episodio,
simular_episodios_generico, estado_ffill}, analisis_detectabilidad.contexto_e_impacto,
memoria_finita_relative_kw.{enriquecer_resultado, resumen_metricas}, robustez_temporal_p.
calcular_saturacion, experimento_ramp._rho_lineal, attacks.{reduccion_constante,
reduccion_variable, bypass_total, bypass_residual}, experimento_stride.{contar_eventos,
mcnemar_p, bootstrap_pareado}, episodes.wilson_ci.

Ejecutable de forma aislada:
    python -m src.evaluacion_final_retrospectiva_test
"""
from __future__ import annotations

import hashlib
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

from src import analisis_detectabilidad as ad
from src import base_relative as br
from src import congelacion_final_or_pretest as cfg
from src import fusion_p_histgb as fus
from src import memoria_finita_relative_kw as mem
from src import normalization as norm
from src import optimizacion_histgb_periodic as opt
from src import predictor_causal_lags as pcl
from src import robustez_temporal_p as rtp
from src import windowing as wnd
from src.attacks import bypass_residual, bypass_total, reduccion_constante, reduccion_variable
from src.data_loading import COLUMNA_OBJETIVO, cargar_y_limpiar
from src.episodes import wilson_ci
from src.experimento_ramp import _rho_lineal
from src.experimento_stride import bootstrap_pareado, contar_eventos, mcnemar_p
from src.splitting import split_temporal

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_OR_DIR = BASE_DIR / "models" / "final_or_pretest"
MANIFEST_V2_DIR = BASE_DIR / "manifests" / "test_attacks_v2"
TABLAS_DIR = BASE_DIR / "results" / "tables" / "final_retrospective_test"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "final_retrospective_test"
MODELOS_DIR = BASE_DIR / "models" / "final_retrospective_test"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

THRESHOLD_P = 0.049748
THRESHOLD_H = 0.883628
RESOLUCION_H_MIN = 15
COLS_H = opt.PERIODIC_BASE_FEATURES
FAMILIAS = ["constant_reduction", "variable_reduction", "linear_ramp", "residual_bypass", "total_bypass"]
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
METODOS = ["P", "H", "OR"]
METHOD_COLS = ["n_affected_targets", "raw_detected", "induced_detected", "first_alarm_timestamp", "detection_delay_min",
               "detected_during_attack", "detected_after_attack", "hidden_energy_before_detection_kwh",
               "hidden_energy_after_detection_kwh", "hidden_energy_before_detection_pct", "recoverable_hidden_energy_pct"]


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ==========================================================================================
# 5) Verificacion previa obligatoria (no se accede a ningun valor numerico de test aqui)
# ==========================================================================================

def verificar_inputs_congelados() -> dict:
    checks = {}

    with open(FINAL_OR_DIR / "hashes_finales.json", encoding="utf-8") as f:
        hashes_modelo_guardados = json.load(f)
    hashes_modelo_actuales = {n: _sha256(FINAL_OR_DIR / n) for n in hashes_modelo_guardados}
    checks["hashes_modelo_coinciden"] = hashes_modelo_actuales == hashes_modelo_guardados

    with open(MANIFEST_V2_DIR / "hashes_attack_manifest_test_v2.json", encoding="utf-8") as f:
        hashes_manifiesto_guardados = json.load(f)
    hashes_manifiesto_actuales = {n: _sha256(MANIFEST_V2_DIR / n) for n in hashes_manifiesto_guardados}
    checks["hashes_manifiesto_coinciden"] = hashes_manifiesto_actuales == hashes_manifiesto_guardados

    with open(FINAL_OR_DIR / "configuracion_final_or_pretest.json", encoding="utf-8") as f:
        config_modelo = json.load(f)
    checks["status_modelo_ready"] = config_modelo["status"] == "READY_FOR_TEST"
    checks["selected_configuration"] = config_modelo["selected_configuration"] == "OR_PMULTI_HMULTIWINDOW"
    checks["threshold_P"] = abs(config_modelo["P"]["threshold"] - THRESHOLD_P) < 1e-6
    checks["threshold_H"] = abs(config_modelo["H"]["threshold"] - THRESHOLD_H) < 1e-6
    checks["regla_or"] = config_modelo["fusion"]["rule"] == "OR"
    checks["features_h_orden"] = config_modelo["H"]["features_orden"] == COLS_H

    with open(MANIFEST_V2_DIR / "manifest_freeze_record_v2.json", encoding="utf-8") as f:
        freeze_v2 = json.load(f)
    checks["status_manifiesto_frozen"] = freeze_v2["status"] == "MANIFEST_V2_FROZEN_READY_FOR_RETROSPECTIVE_TEST"
    checks["n_base_segments"] = freeze_v2["n_base_segments"] == 140
    checks["n_attack_episodes"] = freeze_v2["n_attack_episodes"] == 1680
    checks["v1_preservado"] = bool(freeze_v2["version_1_permanece_preservada"] and freeze_v2["v1_hashes_verificados_intactos"])

    try:
        m = joblib.load(FINAL_OR_DIR / "histgb_periodic_final.joblib")
        checks["modelo_h_carga"] = True
        checks["modelo_h_n_features"] = int(m.n_features_in_) == len(COLS_H)
    except Exception:
        checks["modelo_h_carga"] = False
        checks["modelo_h_n_features"] = False

    checks["todo_ok"] = bool(all(checks.values()))
    return checks


# ==========================================================================================
# 7.1) P limpio sobre una particion arbitraria (aqui: test) -- checkpoint/params_norm congelados
# ==========================================================================================

def puntuar_p_limpio(particion_df: pd.DataFrame, modelo_p, params_norm_p: dict) -> tuple[np.ndarray, np.ndarray]:
    grid = wnd.ventanear(particion_df, 360, 60)
    X_norm = norm.aplicar_zscore(grid["X"], params_norm_p)
    X_hat = modelo_p.reconstruir(X_norm)
    score = br.calcular_relative_kw(X_norm, X_hat, params_norm_p)
    ts = pd.to_datetime(grid["fin"]).to_numpy()
    return score, ts


# ==========================================================================================
# 9) Construccion de los 1680 ataques desde el manifiesto v2 congelado (no se regeneran
#    timestamps/semillas/parametros -- se leen literalmente del CSV)
# ==========================================================================================

def construir_ataques_desde_manifiesto(tabla_ep: pd.DataFrame, particion_test_raw: np.ndarray, test_ini: pd.Timestamp) -> dict:
    ataques = {}
    for row in tabla_ep.itertuples():
        offset_global = int((pd.Timestamp(row.start_timestamp) - test_ini) / pd.Timedelta(minutes=1))
        dur = int(row.duration_minutes)
        x_tramo = particion_test_raw[offset_global:offset_global + dur].astype(np.float64)
        fam = row.attack_family
        if fam == "constant_reduction":
            y = reduccion_constante(x_tramo, float(row.rho))
        elif fam == "variable_reduction":
            rng_ep = np.random.default_rng(int(row.episode_seed))
            y = reduccion_variable(x_tramo, float(row.rho_min), float(row.rho_max), rng_ep)
        elif fam == "linear_ramp":
            rho_t = _rho_lineal(dur, float(row.rho_final))
            y = rho_t * x_tramo
        elif fam == "residual_bypass":
            y = bypass_residual(x_tramo, float(row.rho_residual))
        elif fam == "total_bypass":
            y = bypass_total(x_tramo)
        else:
            raise ValueError(f"familia desconocida: {fam}")
        assert (y >= -1e-9).all() and (y <= x_tramo + 1e-9).all(), f"0<=y<=x violado en {row.episode_id}"
        ataques[row.episode_id] = {
            "offset_global": offset_global, "duracion": dur, "y_tramo": y,
            "attack_start": pd.Timestamp(row.start_timestamp), "attack_end": pd.Timestamp(row.end_timestamp),
            "family": fam, "parameter": row.severity_label, "base_segment_id": row.base_segment_id,
        }
    return ataques


# ==========================================================================================
# Orquestacion de la puntuacion (limpia + atacada, P y H) -- fase 3 del main()
# ==========================================================================================

def ejecutar_puntuacion(particion_test: pd.DataFrame, serie_pretest: pd.DataFrame, tabla_ep: pd.DataFrame,
                         modelo_p, params_norm_p, modelo_h_final, perfil_final) -> dict:
    test_ini = particion_test.index[0]

    p_score_clean, p_ts_clean = puntuar_p_limpio(particion_test, modelo_p, params_norm_p)
    positive_h_clean, ts_h_clean, df_h_clean = cfg.predecir_pool_limpio(modelo_h_final, perfil_final, serie_pretest, particion_test, test_ini)

    val15_test = pcl.construir_serie_15min(particion_test)
    particion_test_min_raw = particion_test[COLUMNA_OBJETIVO].to_numpy(dtype=np.float64)
    # val15_test (sin contexto previo externo) y ts_h_clean (con contexto previo real via
    # predecir_pool_limpio) son la misma rejilla densa de 15 min sobre test, bin a bin:
    assert len(val15_test["ts_end"]) == len(ts_h_clean)
    assert np.array_equal(pd.DatetimeIndex(val15_test["ts_end"]), pd.DatetimeIndex(ts_h_clean)), \
        "val15_test y ts_h_clean deben ser exactamente la misma rejilla densa de 15 min sobre test"

    active_p_clean_ffill = fus.estado_ffill(p_score_clean, THRESHOLD_P, p_ts_clean, ts_h_clean)
    active_h_clean = positive_h_clean > THRESHOLD_H
    active_or_clean = active_p_clean_ffill | active_h_clean

    p_global_test = {"ts": p_ts_clean, "score": p_score_clean, "umbral": THRESHOLD_P, "modelo": modelo_p, "params_norm": params_norm_p}
    ataques_dict = construir_ataques_desde_manifiesto(tabla_ep, particion_test_min_raw, test_ini)

    meta_h, feats_h = pcl.puntuar_ataques(ataques_dict, val15_test, particion_test_min_raw, perfil_final, len(val15_test["kw"]))
    if len(feats_h):
        pred_h_at = modelo_h_final.predict(feats_h[COLS_H].to_numpy(dtype=np.float64))
        _, score_h_at = pcl.calcular_score(pred_h_at, meta_h["target_kw_atacado"].to_numpy())
        meta_h = meta_h.copy()
        meta_h["score_atacado_H"] = score_h_at
        # pcl.puntuar_ataques calcula k=t-MAX_LAG asumiendo un target_end_master recortado
        # (sin contexto previo externo). Aqui ts_h_clean es la rejilla densa de test (con
        # contexto previo real, verificado identica a val15_test["ts_end"] bin a bin arriba),
        # por lo que el indice correcto para alinear con ts_h_clean es "t" directamente, no
        # "k". Se verifica explicitamente antes de sobrescribir.
        assert np.array_equal(meta_h["target_end"].to_numpy(), ts_h_clean[meta_h["t"].to_numpy()]), \
            "meta_h['t'] no indexa correctamente ts_h_clean -- desalineamiento en la puntuacion atacada de H"
        meta_h["k"] = meta_h["t"]

    tabla_p_atacada, _ = opt.puntuar_p_en_fold(ataques_dict, particion_test_min_raw, particion_test, test_ini, p_global_test)
    p_atacado_por_episodio = opt.construir_p_atacado_por_episodio(tabla_p_atacada, p_global_test)

    return {
        "p_score_clean": p_score_clean, "p_ts_clean": p_ts_clean, "positive_h_clean": positive_h_clean, "ts_h_clean": ts_h_clean,
        "active_p_clean_ffill": active_p_clean_ffill, "active_h_clean": active_h_clean, "active_or_clean": active_or_clean,
        "ataques_dict": ataques_dict, "meta_h": meta_h, "p_atacado_por_episodio": p_atacado_por_episodio,
        "particion_test_min_raw": particion_test_min_raw, "test_ini": test_ini,
    }


# ==========================================================================================
# 7.4/8) Metricas de test limpio (eventos + criterios operativos)
# ==========================================================================================

def metricas_limpias(nombre: str, activo: np.ndarray, ts: np.ndarray, n_dias_test: float) -> dict:
    sat = rtp.calcular_saturacion(activo, ts, RESOLUCION_H_MIN)
    fa = contar_eventos(activo) / n_dias_test
    admisible = bool(fa <= 0.15 and sat["active_fraction"] <= 0.10 and sat["longest_clean_event_hours"] <= 72.0
                     and sat["seven_day_max_active_fraction"] <= 0.30 and sat["consecutive_days_over_50"] < 3)
    return {"configuracion": nombre, "fa_dia": fa, **sat, "admisible": admisible, "clean_operational_failure": not admisible}


def eventos_detalle(activo: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    ts_idx = pd.DatetimeIndex(ts)
    event_start = activo & ~np.r_[False, activo[:-1]]
    event_end = activo & ~np.r_[activo[1:], False]
    starts, ends = ts_idx[event_start], ts_idx[event_end]
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    dur_h = (ends - starts).total_seconds() / 3600.0 + (RESOLUCION_H_MIN / 60.0)
    gap_h = (starts[1:] - ends[:-1]).total_seconds() / 3600.0 if n > 1 else np.array([])
    return pd.DataFrame({"start": starts, "end": ends, "duration_hours": dur_h}), gap_h


# ==========================================================================================
# 13-15) Estados por episodio (P/H/OR/AND), horizonte afectado y deteccion inducida --
#        reutiliza literalmente fusion_p_histgb.construir_estados_episodio/simular_episodios_generico
# ==========================================================================================

def construir_estados_por_episodio(r: dict) -> dict:
    ids = set(r["ataques_dict"].keys())
    meta_h = r["meta_h"]
    grp_h = {ep: g for ep, g in meta_h.groupby("episode_id")} if len(meta_h) else {}
    estados = {}
    for ep in ids:
        estados[ep] = fus.construir_estados_episodio(
            grp_h.get(ep), r["p_ts_clean"], r["p_score_clean"], THRESHOLD_P, r["p_atacado_por_episodio"].get(ep),
            r["positive_h_clean"], THRESHOLD_H, r["ts_h_clean"])
    return estados


def evaluar_detecciones(r: dict, estados: dict, tabla_ep: pd.DataFrame, particion_test_min_raw: np.ndarray) -> pd.DataFrame:
    ids = set(r["ataques_dict"].keys())
    target_end_master = r["ts_h_clean"]
    activo_clean = {"P": r["active_p_clean_ffill"], "H": r["active_h_clean"], "OR": r["active_or_clean"]}

    master_meta = tabla_ep[["episode_id", "base_segment_id", "attack_family", "attack_subtype", "severity_label",
                            "duration_minutes", "start_timestamp", "end_timestamp", "episode_seed"]].copy()
    master_meta["attack_start"] = pd.to_datetime(master_meta.pop("start_timestamp"))
    master_meta["attack_end"] = pd.to_datetime(master_meta.pop("end_timestamp"))

    contexto = ad.contexto_e_impacto(r["ataques_dict"], particion_test_min_raw)

    base = master_meta.merge(contexto, on="episode_id", how="left")

    filas_horizonte = []
    for ep in ids:
        ks_local = estados[ep]["ks_local"]
        if len(ks_local):
            first_aff, last_aff = target_end_master[ks_local[0]], target_end_master[ks_local[-1]]
        else:
            first_aff = last_aff = pd.NaT
        filas_horizonte.append({"episode_id": ep, "first_affected_decision": first_aff, "last_affected_decision": last_aff,
                                "valid_detection_horizon_start": first_aff, "valid_detection_horizon_end": last_aff})
    base = base.merge(pd.DataFrame(filas_horizonte), on="episode_id", how="left")

    for metodo in METODOS:
        res_raw = fus.simular_episodios_generico(ids, estados, metodo, activo_clean[metodo], target_end_master)
        res = mem.enriquecer_resultado(res_raw, r["ataques_dict"], master_meta[["episode_id", "attack_start", "attack_end"]], particion_test_min_raw, contexto)
        sub = res[["episode_id"] + METHOD_COLS].rename(columns={c: f"{c}_{metodo}" for c in METHOD_COLS})
        base = base.merge(sub, on="episode_id", how="left")
        base[f"detected_{metodo}"] = base[f"induced_detected_{metodo}"]
        base[f"detection_timestamp_{metodo}"] = base[f"first_alarm_timestamp_{metodo}"]
        base[f"delay_{metodo}"] = base[f"detection_delay_min_{metodo}"]
        base[f"blocked_by_clean_alarm_{metodo}"] = base[f"raw_detected_{metodo}"] & ~base[f"induced_detected_{metodo}"]
        base[f"clean_preexisting_alarm_{metodo}"] = base[f"blocked_by_clean_alarm_{metodo}"]
        base[f"detected_during_attack_{metodo}"] = base[f"detected_during_attack_{metodo}"].astype("boolean")
        base[f"detected_after_attack_but_within_horizon_{metodo}"] = base[f"induced_detected_{metodo}"] & (base[f"detected_after_attack_{metodo}"] == True)  # noqa: E712
        base[f"timely_energy_{metodo}"] = base[f"hidden_energy_after_detection_kwh_{metodo}"]

    base["clean_energy_kWh"] = base["clean_energy_kwh"]
    base["hidden_energy_kWh"] = base["hidden_energy_kwh"]
    base["hidden_fraction_real"] = base["hidden_energy_pct"] / 100.0
    assert (base["hidden_energy_kWh"] >= -1e-9).all(), "hidden_energy negativa detectada"

    base["P_only"] = base["induced_detected_P"] & ~base["induced_detected_H"]
    base["H_only"] = base["induced_detected_H"] & ~base["induced_detected_P"]
    base["both"] = base["induced_detected_P"] & base["induced_detected_H"]
    base["theoretical_union"] = base["induced_detected_P"] | base["induced_detected_H"]
    base["operational_OR"] = base["induced_detected_OR"]
    base["union_matches_or"] = base["theoretical_union"] == base["operational_OR"]

    calendario = tabla_ep[["episode_id", "calendar_quarter", "day_type", "daypart", "start_minute_mod_15", "start_minute_mod_60"]]
    base = base.merge(calendario, on="episode_id", how="left")
    base = base.merge(tabla_ep[["episode_id", "duration_minutes"]].rename(columns={"duration_minutes": "duration_minutes_chk"}), on="episode_id", how="left")

    assert base["episode_id"].is_unique and len(base) == 1680
    return base


# ==========================================================================================
# 17-18) Metricas por metodo (DR, DR energetico, timely, retraso...) y desgloses
# ==========================================================================================

def _dr_metrics(sub: pd.DataFrame, metodo: str) -> dict:
    n = len(sub)
    n_det = int(sub[f"induced_detected_{metodo}"].sum())
    dr = 100 * n_det / n if n else float("nan")
    ci = wilson_ci(n_det, n) if n else (float("nan"), float("nan"))
    hidden_total = float(sub["hidden_energy_kWh"].sum())
    hidden_det = float(sub.loc[sub[f"induced_detected_{metodo}"], "hidden_energy_kWh"].sum())
    dr_energ = 100 * hidden_det / hidden_total if hidden_total > 1e-9 else float("nan")
    after_total = float(sub[f"hidden_energy_after_detection_kwh_{metodo}"].sum())
    timely = 100 * after_total / hidden_total if hidden_total > 1e-9 else float("nan")
    con_ind = sub[sub[f"induced_detected_{metodo}"]]
    delays = con_ind[f"delay_{metodo}"].dropna()
    pct_durante = 100 * (con_ind[f"detected_during_attack_{metodo}"] == True).mean() if len(con_ind) else float("nan")  # noqa: E712
    pct_despues_horiz = 100 * sub[f"detected_after_attack_but_within_horizon_{metodo}"].mean() if n else float("nan")
    bloqueados = int(sub[f"blocked_by_clean_alarm_{metodo}"].sum())
    hidden_before_total = float(sub[f"hidden_energy_before_detection_kwh_{metodo}"].sum())
    return {
        "metodo": metodo, "n_episodios": n, "n_detectados": n_det, "dr_pct": dr, "dr_ci95_low": ci[0], "dr_ci95_high": ci[1],
        "dr_energ_pct": dr_energ, "timely_energy_dr_pct": timely,
        "retraso_mediano_min": float(delays.median()) if len(delays) else float("nan"),
        "retraso_p25_min": float(delays.quantile(0.25)) if len(delays) else float("nan"),
        "retraso_p75_min": float(delays.quantile(0.75)) if len(delays) else float("nan"),
        "retraso_p90_min": float(delays.quantile(0.90)) if len(delays) else float("nan"),
        "pct_detectado_durante_ataque": pct_durante, "pct_detectado_despues_dentro_horizonte": pct_despues_horiz,
        "bloqueados_por_alarma_limpia": bloqueados, "hidden_energy_before_alarm_total_kwh": hidden_before_total,
    }


def resumen_por_metodo(base: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_dr_metrics(base, m) for m in METODOS])


def desglose_por(base: pd.DataFrame, columna: str) -> pd.DataFrame:
    filas = []
    for val, sub in base.groupby(columna):
        for m in METODOS:
            filas.append({columna: val, **_dr_metrics(sub, m)})
    return pd.DataFrame(filas)


# ==========================================================================================
# 20) Comparaciones emparejadas por base_segment_id
# ==========================================================================================

def bootstrap_agrupado(vals_a: np.ndarray, vals_b: np.ndarray, grupos: np.ndarray, n_boot: int = 3000, seed: int = 42) -> tuple[float, float, float]:
    grupos_unicos = np.unique(grupos)
    idx_por_grupo = {g: np.where(grupos == g)[0] for g in grupos_unicos}
    n_g = len(grupos_unicos)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample_g = rng.choice(grupos_unicos, n_g, replace=True)
        idx = np.concatenate([idx_por_grupo[g] for g in sample_g])
        diffs[i] = vals_b[idx].mean() - vals_a[idx].mean()
    return float(vals_b.mean() - vals_a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def comparar_dos_series(nombre: str, grupos: np.ndarray, det_a: np.ndarray, det_b: np.ndarray,
                         energ_a: np.ndarray, energ_b: np.ndarray, delay_a: pd.Series, delay_b: pd.Series) -> dict:
    da, db = np.asarray(det_a, dtype=bool), np.asarray(det_b, dtype=bool)
    solo_a, solo_b = int((da & ~db).sum()), int((~da & db).sum())
    p_mc = mcnemar_p(solo_a, solo_b)
    diff_dr, cilo_dr, cihi_dr = bootstrap_agrupado(da.astype(float), db.astype(float), grupos)
    diff_e, cilo_e, cihi_e = bootstrap_agrupado(np.asarray(energ_a, dtype=float), np.asarray(energ_b, dtype=float), grupos)
    ambos = da & db
    da_delay, db_delay = pd.Series(np.asarray(delay_a)), pd.Series(np.asarray(delay_b))
    mask_ambos = pd.Series(ambos)
    common = da_delay[mask_ambos].dropna().index.intersection(db_delay[mask_ambos].dropna().index)
    if len(common) >= 5:
        try:
            _, p_w = sstats.wilcoxon(da_delay.loc[common], db_delay.loc[common])
        except ValueError:
            p_w = float("nan")
    else:
        p_w = float("nan")
    n = len(da)
    ci_a, ci_b = wilson_ci(int(da.sum()), n), wilson_ci(int(db.sum()), n)
    return {
        "comparacion": nombre, "n_episodios": n, "DR_a_pct": 100 * da.mean(), "DR_b_pct": 100 * db.mean(),
        "dr_a_ci95_low": ci_a[0], "dr_a_ci95_high": ci_a[1], "dr_b_ci95_low": ci_b[0], "dr_b_ci95_high": ci_b[1],
        "diferencia_dr_pp": diff_dr * 100, "bootstrap_dr_ci95_low_pp": cilo_dr * 100, "bootstrap_dr_ci95_high_pp": cihi_dr * 100,
        "mcnemar_p": p_mc, "significativo_p<0.05": bool(p_mc < 0.05),
        "diferencia_energia_kwh": diff_e, "bootstrap_energia_ci95_low": cilo_e, "bootstrap_energia_ci95_high": cihi_e,
        "wilcoxon_p_retraso": p_w, "n_detectados_ambos": int(ambos.sum()),
    }


def comparacion_familia(base: pd.DataFrame, nombre: str, mask_a: pd.Series, mask_b: pd.Series, metodo: str = "OR") -> dict:
    sub_a = base[mask_a].set_index("base_segment_id")
    sub_b = base[mask_b].set_index("base_segment_id")
    ids_comunes = sub_a.index.intersection(sub_b.index)
    sub_a, sub_b = sub_a.loc[ids_comunes], sub_b.loc[ids_comunes]
    da = sub_a[f"induced_detected_{metodo}"].to_numpy(dtype=bool)
    db = sub_b[f"induced_detected_{metodo}"].to_numpy(dtype=bool)
    ea = sub_a["hidden_energy_kWh"].to_numpy() * da
    eb = sub_b["hidden_energy_kWh"].to_numpy() * db
    r = comparar_dos_series(nombre, ids_comunes.to_numpy(), da, db, ea, eb, sub_a[f"delay_{metodo}"], sub_b[f"delay_{metodo}"])
    return r


def comparacion_metodos(base: pd.DataFrame, metodo_a: str, metodo_b: str) -> dict:
    da = base[f"induced_detected_{metodo_a}"].to_numpy(dtype=bool)
    db = base[f"induced_detected_{metodo_b}"].to_numpy(dtype=bool)
    ea = base["hidden_energy_kWh"].to_numpy() * da
    eb = base["hidden_energy_kWh"].to_numpy() * db
    return comparar_dos_series(f"{metodo_a}_vs_{metodo_b}", base["base_segment_id"].to_numpy(), da, db, ea, eb,
                                base[f"delay_{metodo_a}"], base[f"delay_{metodo_b}"])


def construir_comparaciones_emparejadas(base: pd.DataFrame) -> pd.DataFrame:
    filas = []
    filas.append(comparacion_familia(base, "constant_MEDIUM_vs_variable_MEDIUM",
                                      (base["attack_family"] == "constant_reduction") & (base["severity_label"] == "MEDIUM"),
                                      (base["attack_family"] == "variable_reduction") & (base["severity_label"] == "MEDIUM")))
    filas.append(comparacion_familia(base, "constant_MEDIUM_vs_ramp_MEDIUM_mismo_rho",
                                      (base["attack_family"] == "constant_reduction") & (base["severity_label"] == "MEDIUM"),
                                      (base["attack_family"] == "linear_ramp") & (base["severity_label"] == "MEDIUM")))
    filas.append(comparacion_familia(base, "bypass_residual_RHO010_vs_RHO005",
                                      (base["attack_family"] == "residual_bypass") & (base["severity_label"] == "RHO010"),
                                      (base["attack_family"] == "residual_bypass") & (base["severity_label"] == "RHO005")))
    filas.append(comparacion_familia(base, "bypass_residual_RHO010_vs_bypass_total",
                                      (base["attack_family"] == "residual_bypass") & (base["severity_label"] == "RHO010"),
                                      base["attack_family"] == "total_bypass"))
    filas.append(comparacion_metodos(base, "P", "H"))
    filas.append(comparacion_metodos(base, "P", "OR"))
    filas.append(comparacion_metodos(base, "H", "OR"))
    return pd.DataFrame(filas)


# ==========================================================================================
# 21) Comparacion descriptiva walk-forward vs test (no se usa para recalibrar nada)
# ==========================================================================================

def construir_walkforward_vs_test(resumen_metodo_test: pd.DataFrame, metricas_limpias_test: pd.DataFrame) -> pd.DataFrame:
    ruta_wf = BASE_DIR / "results" / "tables" / "auditoria_final_p_vs_or" / "tabla_principal.csv"
    wf = pd.read_csv(ruta_wf)
    wf_map = {"P": "P_MULTI_SEASON_ONLY", "H": "H_MULTIWINDOW_180_ONLY", "OR": "OR_PMULTI_HMULTIWINDOW"}
    filas = []
    for metodo, cfg_name in wf_map.items():
        row_wf = wf[wf["configuracion"] == cfg_name].iloc[0]
        row_test_atk = resumen_metodo_test[resumen_metodo_test["metodo"] == metodo].iloc[0]
        row_test_clean = metricas_limpias_test[metricas_limpias_test["configuracion"] == metodo].iloc[0]
        filas.append({
            "metodo": metodo,
            "fa_walkforward": row_wf["fa_ponderada"], "fa_test": row_test_clean["fa_dia"],
            "active_fraction_walkforward": row_wf["active_fraction_ponderada"], "active_fraction_test": row_test_clean["active_fraction"],
            "dr_walkforward_pct": row_wf["dr_ramp_pct"], "dr_test_pct": row_test_atk["dr_pct"],
            "dr_energ_walkforward_pct": row_wf["dr_energ_pct"], "dr_energ_test_pct": row_test_atk["dr_energ_pct"],
            "retraso_walkforward_min": row_wf["retraso_mediano_min"], "retraso_test_min": row_test_atk["retraso_mediano_min"],
            "dr_const_walkforward_pct": row_wf["dr_const_pct"], "dr_energ_const_walkforward_pct": row_wf["dr_energ_const_pct"],
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# 22) Clasificacion final descriptiva (no autoriza modificar el sistema)
# ==========================================================================================

def clasificar_resultado(metricas_limpias_test: pd.DataFrame, resumen_metodo_test: pd.DataFrame, base: pd.DataFrame) -> dict:
    or_clean = metricas_limpias_test[metricas_limpias_test["configuracion"] == "OR"].iloc[0]
    r_p = resumen_metodo_test[resumen_metodo_test["metodo"] == "P"].iloc[0]
    r_h = resumen_metodo_test[resumen_metodo_test["metodo"] == "H"].iloc[0]
    r_or = resumen_metodo_test[resumen_metodo_test["metodo"] == "OR"].iloc[0]

    admisible = bool(or_clean["admisible"])
    mejora_p = bool(r_or["dr_energ_pct"] > r_p["dr_energ_pct"])
    mejora_h = bool(r_or["dr_energ_pct"] > r_h["dr_energ_pct"])
    ret_p = 100 * base.loc[base["induced_detected_P"], "induced_detected_OR"].mean() if base["induced_detected_P"].any() else 100.0
    ret_h = 100 * base.loc[base["induced_detected_H"], "induced_detected_OR"].mean() if base["induced_detected_H"].any() else 100.0
    retiene_ambas = bool(ret_p >= 90.0 and ret_h >= 90.0)
    dr_energ_por_familia = base.groupby("attack_family").apply(lambda s: _dr_metrics(s, "OR")["dr_energ_pct"], include_groups=False)
    n_familias_con_senal = int((dr_energ_por_familia.fillna(0) > 5.0).sum())
    senal_varias_familias = bool(n_familias_con_senal >= 3)
    sin_saturacion = bool(not or_clean["falla_saturacion"])

    detalle = {"admisible_operativamente": admisible, "mejora_a_P_en_dr_energ": mejora_p, "mejora_a_H_en_dr_energ": mejora_h,
               "retencion_P_por_OR_pct": ret_p, "retencion_H_por_OR_pct": ret_h, "retiene_ambas_ramas_90pct": retiene_ambas,
               "n_familias_con_senal_dr_energ_mayor_5pct": n_familias_con_senal, "senal_en_varias_familias": senal_varias_familias,
               "sin_saturacion": sin_saturacion, "dr_energ_pct_P": r_p["dr_energ_pct"], "dr_energ_pct_H": r_h["dr_energ_pct"],
               "dr_energ_pct_OR": r_or["dr_energ_pct"], "dr_walkforward_referencia_pct": None}

    if admisible and mejora_p and mejora_h and retiene_ambas and senal_varias_familias and sin_saturacion:
        clasificacion = "CONFIRMATORY_SUPPORT"
    elif admisible and sin_saturacion:
        clasificacion = "PARTIAL_CONFIRMATORY_SUPPORT"
    else:
        clasificacion = "NOT_CONFIRMED_ON_RETROSPECTIVE_TEST"
    detalle["clasificacion"] = clasificacion
    return detalle


# ==========================================================================================
# Orquestacion principal
# ==========================================================================================

def main() -> dict:
    print("Fase 1: verificacion previa obligatoria (sin acceder a ningun valor numerico de test)...")
    verif = verificar_inputs_congelados()
    for k, v in verif.items():
        print(f"  {k}: {v}")
    _guardar_tabla(pd.DataFrame([verif]), "verification_frozen_inputs")
    if not verif["todo_ok"]:
        raise RuntimeError(f"verificacion previa FALLIDA -- evaluacion detenida. Checks: {verif}")
    print("OK -- todos los inputs congelados verificados.")

    print("\nFase 2: cargando test (PRIMER acceso numerico a la particion test)...")
    test_final_evaluation_started_at = datetime.now(timezone.utc).isoformat()
    df_min = cargar_y_limpiar()
    particiones = split_temporal(df_min)
    particion_test = particiones["test"]
    serie_pretest = pd.concat([particiones["train"], particiones["val"]])
    test_ini, test_fin = particion_test.index[0], particion_test.index[-1]
    n_dias_test = float((test_fin - test_ini) / pd.Timedelta(days=1))
    print(f"  test_final_evaluation_started_at = {test_final_evaluation_started_at}")
    print(f"  test: {test_ini} -> {test_fin} ({n_dias_test:.1f} dias)")

    print("\nFase 3: cargando artefactos congelados (checkpoint P, modelo H final, perfil H final)...")
    from src.base_relative import cargar_ataques_y_modelo_360
    _, datos_p, modelo_p, _, _ = cargar_ataques_y_modelo_360()
    params_norm_p = datos_p["params_norm"]
    modelo_h_final = joblib.load(FINAL_OR_DIR / "histgb_periodic_final.joblib")
    perfil_final = joblib.load(FINAL_OR_DIR / "profile_train_final.joblib")

    tabla_base_v2 = pd.read_csv(MANIFEST_V2_DIR / "base_segments_test_v2.csv")
    tabla_ep = pd.read_csv(MANIFEST_V2_DIR / "attack_manifest_test_v2.csv")
    assert len(tabla_base_v2) == 140 and len(tabla_ep) == 1680

    print("\nFase 4: puntuando test limpio y los 1680 episodios atacados (P y H)...")
    r = ejecutar_puntuacion(particion_test, serie_pretest, tabla_ep, modelo_p, params_norm_p, modelo_h_final, perfil_final)
    print("  OK -- puntuacion limpia y atacada completada.")

    print("\nFase 5: metricas de test limpio (eventos, saturacion, criterios operativos)...")
    activo_clean = {"P": r["active_p_clean_ffill"], "H": r["active_h_clean"], "OR": r["active_or_clean"]}
    metricas_limpias_test = pd.DataFrame([metricas_limpias(m, activo_clean[m], r["ts_h_clean"], n_dias_test) for m in METODOS])
    _guardar_tabla(metricas_limpias_test, "clean_test_metrics")
    _guardar_tabla(metricas_limpias_test[["configuracion", "fa_dia", "active_fraction", "longest_clean_event_hours",
                                          "seven_day_max_active_fraction", "consecutive_days_over_50", "admisible", "clean_operational_failure"]],
                   "operational_criteria_test")
    for m in METODOS:
        ev_df, _ = eventos_detalle(activo_clean[m], r["ts_h_clean"])
        _guardar_tabla(ev_df, f"clean_events_{m}")
    print(metricas_limpias_test.to_string(index=False))

    print("\nFase 6: falsas alarmas mensuales...")
    filas_fa_mes = []
    ts_idx = pd.DatetimeIndex(r["ts_h_clean"])
    for m in METODOS:
        event_start = activo_clean[m] & ~np.r_[False, activo_clean[m][:-1]]
        for periodo, idx in pd.Series(np.arange(len(ts_idx))).groupby(ts_idx.to_period("M")):
            filas_fa_mes.append({"metodo": m, "mes": str(periodo), "n_eventos_iniciados": int(event_start[idx].sum()),
                                 "active_fraction_mes": float(activo_clean[m][idx].mean())})
    _guardar_tabla(pd.DataFrame(filas_fa_mes), "test_monthly_false_alarms")

    print("\nFase 7: estados por episodio y evaluacion de detecciones (1680 episodios)...")
    estados = construir_estados_por_episodio(r)
    base = evaluar_detecciones(r, estados, tabla_ep, r["particion_test_min_raw"])
    _guardar_tabla(base, "episode_results_1680")
    print(f"  DR: P={base['induced_detected_P'].mean()*100:.1f}%  H={base['induced_detected_H'].mean()*100:.1f}%  OR={base['induced_detected_OR'].mean()*100:.1f}%")
    n_mismatch = int((~base["union_matches_or"]).sum())
    print(f"  episodios con theoretical_union != operational_OR: {n_mismatch} (investigado, ver limitaciones del informe)")

    _guardar_tabla(pd.DataFrame([{
        "score_p_clean_min": float(r["p_score_clean"].min()), "score_p_clean_max": float(r["p_score_clean"].max()),
        "score_p_clean_mean": float(r["p_score_clean"].mean()), "score_p_clean_median": float(np.median(r["p_score_clean"])),
        "score_h_clean_min": float(r["positive_h_clean"].min()), "score_h_clean_max": float(r["positive_h_clean"].max()),
        "score_h_clean_mean": float(r["positive_h_clean"].mean()), "score_h_clean_median": float(np.median(r["positive_h_clean"])),
        "n_decisiones_p": len(r["p_score_clean"]), "n_decisiones_h": len(r["positive_h_clean"]),
    }]), "episode_scores_summary")

    print("\nFase 8: desgloses (familia/duracion/severidad/calendario)...")
    _guardar_tabla(desglose_por(base, "attack_family"), "detection_by_family")
    _guardar_tabla(desglose_por(base, "duration_minutes"), "detection_by_duration")
    _guardar_tabla(desglose_por(base, "severity_label"), "detection_by_severity")
    _guardar_tabla(desglose_por(base, "attack_family")[["attack_family", "metodo", "dr_energ_pct", "timely_energy_dr_pct", "hidden_energy_before_alarm_total_kwh"]], "energy_metrics_by_family")
    _guardar_tabla(desglose_por(base, "duration_minutes")[["duration_minutes", "metodo", "dr_energ_pct", "timely_energy_dr_pct", "hidden_energy_before_alarm_total_kwh"]], "energy_metrics_by_duration")
    delay_global = resumen_por_metodo(base)[["metodo", "retraso_mediano_min", "retraso_p25_min", "retraso_p75_min", "retraso_p90_min"]]
    delay_familia = desglose_por(base, "attack_family")[["attack_family", "metodo", "retraso_mediano_min", "retraso_p25_min", "retraso_p75_min", "retraso_p90_min"]]
    _guardar_tabla(delay_global.assign(nivel="global").merge(delay_familia.assign(nivel="familia"), how="outer"), "delay_metrics")

    for fam in FAMILIAS:
        _guardar_tabla(base[base["attack_family"] == fam].copy(), f"{fam}_results")

    _guardar_tabla(desglose_por(base, "calendar_quarter"), "results_by_quarter")
    _guardar_tabla(desglose_por(base, "day_type"), "results_by_day_type")
    _guardar_tabla(desglose_por(base, "daypart"), "results_by_daypart")
    _guardar_tabla(desglose_por(base, "start_minute_mod_15"), "results_by_mod15")

    print("\nFase 9: complementariedad P/H y union teorica vs OR...")
    complementariedad = pd.DataFrame([{
        "n_P_only": int(base["P_only"].sum()), "n_H_only": int(base["H_only"].sum()), "n_both": int(base["both"].sum()),
        "n_theoretical_union": int(base["theoretical_union"].sum()), "n_operational_OR": int(base["operational_OR"].sum()),
        "n_mismatches_union_vs_or": n_mismatch,
        "retencion_P_por_OR_pct": 100 * base.loc[base["induced_detected_P"], "induced_detected_OR"].mean() if base["induced_detected_P"].any() else 100.0,
        "retencion_H_por_OR_pct": 100 * base.loc[base["induced_detected_H"], "induced_detected_OR"].mean() if base["induced_detected_H"].any() else 100.0,
    }])
    _guardar_tabla(complementariedad, "complementarity_P_H")
    _guardar_tabla(base[["episode_id", "base_segment_id", "attack_family", "severity_label", "induced_detected_P", "induced_detected_H",
                        "induced_detected_OR", "theoretical_union", "operational_OR", "union_matches_or"]], "theoretical_union_vs_OR")

    print("\nFase 10: comparaciones emparejadas por base_segment_id...")
    tabla_comp = construir_comparaciones_emparejadas(base)
    _guardar_tabla(tabla_comp, "statistical_comparisons")
    _guardar_tabla(tabla_comp[tabla_comp["comparacion"].isin([
        "constant_MEDIUM_vs_variable_MEDIUM", "constant_MEDIUM_vs_ramp_MEDIUM_mismo_rho",
        "bypass_residual_RHO010_vs_RHO005", "bypass_residual_RHO010_vs_bypass_total"])], "paired_family_comparisons")

    print("\nFase 11: comparacion descriptiva walk-forward vs test...")
    resumen_metodo_test = resumen_por_metodo(base)
    tabla_wf_vs_test = construir_walkforward_vs_test(resumen_metodo_test, metricas_limpias_test)
    _guardar_tabla(tabla_wf_vs_test, "walkforward_vs_test")
    print(tabla_wf_vs_test.to_string(index=False))

    print("\nFase 12: clasificacion final descriptiva...")
    clasif = clasificar_resultado(metricas_limpias_test, resumen_metodo_test, base)
    _guardar_tabla(pd.DataFrame([clasif]), "final_interpretation")
    print(f"  CLASIFICACION: {clasif['clasificacion']}")

    print("\nFase 13: generando figuras...")
    generar_figuras(base, metricas_limpias_test, activo_clean, r, tabla_comp)

    print("\nFase 14: congelando resultados finales (hashes + manifiesto de ejecucion)...")
    _, metadatos_previos = opt.reproducir_baseline()
    payload_final = congelar_resultados(verif, clasif, test_final_evaluation_started_at, metadatos_previos, n_mismatch)

    print(f"\n>>> CLASIFICACION FINAL: {clasif['clasificacion']}")
    print("\nCONFIRMACION: no se ha modificado ningun modelo, umbral, manifiesto ni criterio "
          "despues de observar estos resultados.")

    return {"verif": verif, "test_final_evaluation_started_at": test_final_evaluation_started_at,
            "particion_test": particion_test, "n_dias_test": n_dias_test, "tabla_base_v2": tabla_base_v2,
            "tabla_ep": tabla_ep, "puntuacion": r, "base": base, "metricas_limpias_test": metricas_limpias_test,
            "resumen_metodo_test": resumen_metodo_test, "tabla_wf_vs_test": tabla_wf_vs_test, "clasif": clasif,
            "payload_final": payload_final}


# ==========================================================================================
# Figuras (20 minimas)
# ==========================================================================================

def generar_figuras(base: pd.DataFrame, metricas_limpias_test: pd.DataFrame, activo_clean: dict, r: dict, tabla_comp: pd.DataFrame) -> None:
    ts_idx = pd.DatetimeIndex(r["ts_h_clean"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for ax, m, color in zip(axes, METODOS, [AZUL, NARANJA, VERDE]):
        ax.plot(ts_idx, activo_clean[m].astype(int), color=color, linewidth=0.6)
        ax.set_ylabel(m)
    axes[-1].set_xlabel("tiempo"); fig.suptitle("Eventos limpios P/H/OR sobre el tiempo (test)")
    fig.tight_layout(); _guardar_fig(fig, "01_eventos_limpios_tiempo")

    fig, ax = plt.subplots(figsize=(9, 5))
    for m, color in zip(METODOS, [AZUL, NARANJA, VERDE]):
        event_start = activo_clean[m] & ~np.r_[False, activo_clean[m][:-1]]
        por_mes = pd.Series(event_start.astype(int), index=ts_idx).groupby(ts_idx.to_period("M")).sum()
        ax.plot(por_mes.index.astype(str), por_mes.values, marker="o", label=m, color=color)
    ax.set_ylabel("eventos iniciados/mes"); ax.legend(); plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout(); _guardar_fig(fig, "02_fa_mensual")

    fig, ax = plt.subplots(figsize=(9, 5))
    for m, color in zip(METODOS, [AZUL, NARANJA, VERDE]):
        por_mes = pd.Series(activo_clean[m].astype(float), index=ts_idx).groupby(ts_idx.to_period("M")).mean()
        ax.plot(por_mes.index.astype(str), por_mes.values, marker="o", label=m, color=color)
    ax.set_ylabel("active_fraction mensual"); ax.legend(); plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout(); _guardar_fig(fig, "03_active_fraction_mensual")

    def _bar_por(col, nombre, titulo, ylabel="dr_pct"):
        d = desglose_por(base, col)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(d[col].nunique()); w = 0.25
        for i, m in enumerate(METODOS):
            sub = d[d["metodo"] == m].sort_values(col)
            ax.bar(x + i * w, sub[ylabel], width=w, label=m)
        ax.set_xticks(x + w); ax.set_xticklabels(sorted(d[col].unique(), key=str), rotation=45, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(titulo); ax.legend()
        fig.tight_layout(); _guardar_fig(fig, nombre)

    _bar_por("attack_family", "04_dr_por_familia", "DR por familia")
    _bar_por("attack_family", "05_dr_energetico_por_familia", "DR energetico por familia", "dr_energ_pct")
    _bar_por("duration_minutes", "06_dr_por_duracion", "DR por duracion")
    _bar_por("duration_minutes", "07_dr_energetico_por_duracion", "DR energetico por duracion", "dr_energ_pct")
    _bar_por("severity_label", "08_dr_por_severidad", "DR por severidad")
    _bar_por("attack_family", "09_retraso_por_familia", "Retraso mediano por familia", "retraso_mediano_min")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(METODOS, [base.loc[base[f"induced_detected_{m}"], "hidden_energy_before_detection_kwh_" + m].sum() for m in METODOS], color=[AZUL, NARANJA, VERDE])
    ax.set_ylabel("energia oculta antes de alarma (kWh, sumada)"); ax.set_title("Energia oculta antes de la alarma")
    fig.tight_layout(); _guardar_fig(fig, "10_energia_antes_de_alarma")

    resumen = resumen_por_metodo(base)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(resumen["metodo"], resumen["dr_energ_pct"], color=[AZUL, NARANJA, VERDE])
    ax.set_ylabel("DR energetico (%)"); ax.set_title("P vs H vs OR (DR energetico global)")
    fig.tight_layout(); _guardar_fig(fig, "11_p_vs_h_vs_or")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["P_only", "H_only", "both"], [base["P_only"].sum(), base["H_only"].sum(), base["both"].sum()], color=[AZUL, NARANJA, VERDE])
    ax.set_ylabel("n episodios"); ax.set_title("Complementariedad P/H")
    fig.tight_layout(); _guardar_fig(fig, "12_complementariedad_p_h")

    for nombre_comp, fname in [("constant_MEDIUM_vs_variable_MEDIUM", "13_constante_vs_variable"),
                                ("constant_MEDIUM_vs_ramp_MEDIUM_mismo_rho", "14_constante_vs_rampa"),
                                ("bypass_residual_RHO010_vs_RHO005", "15_bypass_residual_vs_total")]:
        fila = tabla_comp[tabla_comp["comparacion"] == nombre_comp]
        if len(fila):
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.bar(["a", "b"], [fila["DR_a_pct"].iloc[0], fila["DR_b_pct"].iloc[0]], color=[AZUL, NARANJA])
            ax.set_ylabel("DR (%)"); ax.set_title(nombre_comp)
            fig.tight_layout(); _guardar_fig(fig, fname)

    _bar_por("calendar_quarter", "16_resultados_por_trimestre", "DR por trimestre")
    _bar_por("daypart", "17_resultados_por_franja", "DR por franja horaria")

    fig, ax = plt.subplots(figsize=(7, 5))
    wf = pd.read_csv(BASE_DIR / "results" / "tables" / "auditoria_final_p_vs_or" / "tabla_principal.csv")
    wf_map = {"P": "P_MULTI_SEASON_ONLY", "H": "H_MULTIWINDOW_180_ONLY", "OR": "OR_PMULTI_HMULTIWINDOW"}
    x = np.arange(3); w = 0.35
    dr_wf = [wf[wf["configuracion"] == wf_map[m]]["dr_ramp_pct"].iloc[0] for m in METODOS]
    dr_test = [resumen[resumen["metodo"] == m]["dr_pct"].iloc[0] for m in METODOS]
    ax.bar(x - w / 2, dr_wf, width=w, label="walk-forward")
    ax.bar(x + w / 2, dr_test, width=w, label="test")
    ax.set_xticks(x); ax.set_xticklabels(METODOS); ax.set_ylabel("DR (%)"); ax.legend(); ax.set_title("Walk-forward vs test")
    fig.tight_layout(); _guardar_fig(fig, "18_walkforward_vs_test")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(metricas_limpias_test["configuracion"], metricas_limpias_test["admisible"].astype(int), color=VERDE)
    ax.set_ylabel("admisible (1=si)"); ax.set_title("Criterios operativos limpios")
    fig.tight_layout(); _guardar_fig(fig, "19_criterios_operativos")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["DR", "DR_energ"], [resumen[resumen["metodo"] == "OR"]["dr_pct"].iloc[0], resumen[resumen["metodo"] == "OR"]["dr_energ_pct"].iloc[0]], color=TEAL)
    ax.set_ylabel("%"); ax.set_title("Resumen final OR")
    fig.tight_layout(); _guardar_fig(fig, "20_resumen_final")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


# ==========================================================================================
# Congelacion de resultados: hashes + manifiesto de ejecucion + informe JSON
# ==========================================================================================

def congelar_resultados(verif: dict, clasif: dict, test_final_evaluation_started_at: str, metadatos_previos: dict, n_mismatch: int) -> dict:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "evaluation_status": "RETROSPECTIVE_CONFIRMATORY_EVALUATION", "selected_configuration": "OR_PMULTI_HMULTIWINDOW",
        "test_final_evaluation_started_at": test_final_evaluation_started_at,
        "verificacion_inputs_congelados": verif, "clasificacion_final": clasif,
        "n_episodios_theoretical_union_no_coincide_con_or": n_mismatch,
        "modificaciones_tras_observar_resultados": False, "metadatos_previos": metadatos_previos,
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(TABLAS_DIR / "final_retrospective_test_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    archivos_tabla = sorted(p.name for p in TABLAS_DIR.glob("*.csv"))
    hashes = {n: _sha256(TABLAS_DIR / n) for n in archivos_tabla}
    with open(TABLAS_DIR / "hashes_final_test_results.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False, sort_keys=True)

    manifiesto = {"fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "n_archivos_tabla": len(archivos_tabla),
                 "clasificacion_final": clasif["clasificacion"], "test_final_evaluation_started_at": test_final_evaluation_started_at,
                 "sklearn_version": sklearn.__version__, "python_version": platform.python_version(), "modificado_tras_resultados": False}
    with open(TABLAS_DIR / "execution_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)

    return payload


if __name__ == "__main__":
    main()
