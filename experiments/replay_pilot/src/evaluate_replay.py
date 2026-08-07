"""Inyeccion + evaluacion causal de los episodios replay congelados.

Reutiliza literalmente, sin modificar:
  - congelacion_final_or_pretest.predecir_pool_limpio (H limpio, checkpoint congelado)
  - evaluacion_final_retrospectiva_test.puntuar_p_limpio (P limpio, checkpoint congelado)
  - predictor_causal_lags.puntuar_ataques (features/score H atacado, causal)
  - optimizacion_histgb_periodic.puntuar_p_en_fold / construir_p_atacado_por_episodio (P atacado)
  - fusion_p_histgb.estado_ffill (propagacion causal de P)
  - analisis_detectabilidad.contexto_e_impacto (energia limpia/atacada por episodio)
  - memoria_finita_relative_kw.enriquecer_resultado (retraso, energia antes/despues)

No llama fit() en ningun momento de este modulo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import analisis_detectabilidad as ad
from src import congelacion_final_or_pretest as cfg
from src import fusion_p_histgb as fus
from src import memoria_finita_relative_kw as mem
from src import optimizacion_histgb_periodic as opt
from src import predictor_causal_lags as pcl
from src.data_loading import COLUMNA_OBJETIVO

from experiments.replay_pilot.src.build_replay_manifest import (
    EXP_DIR, MANIFESTS_DIR, TABLES_DIR, REPORTS_DIR, seleccionar_periodo,
)
from experiments.replay_pilot.src.inject_replay import construir_ataques_replay, verificar_copia_exacta

THRESHOLD_P = 0.049748
THRESHOLD_H = 0.883628
METODOS = ["P", "H", "OR"]
COLS_H = opt.PERIODIC_BASE_FEATURES
RESOLUCION_H_MIN = 15
BOUNDARY_MAX_MIN = 60
BOUNDARY_MIN_MIN = 30
POST_ATTACK_MIN = 60
PRE_BUFFER_MIN = 60
ENERGY_TOL_ABS = 0.01
ENERGY_TOL_REL = 0.01


def _boundary_width_minutes(duration_min: int) -> int:
    w = min(BOUNDARY_MAX_MIN, duration_min / 4)
    w = round(w / 15) * 15
    return max(BOUNDARY_MIN_MIN, int(w))


# ==========================================================================================
# Puntuacion limpia y atacada de todo el pool (una unica vez)
# ==========================================================================================

def puntuar_pool(periodo: dict, ataques: dict, modelo_p, params_norm_p, modelo_h_final, perfil_final) -> dict:
    particion_entrenamiento = periodo["particion_entrenamiento"]
    particion_final_pool = periodo["particion_final_pool"]
    final_cal_pool_ini = periodo["final_cal_pool_ini"]

    positive_h_clean, ts_h_clean, _ = cfg.predecir_pool_limpio(
        modelo_h_final, perfil_final, particion_entrenamiento, particion_final_pool, final_cal_pool_ini)

    from src import windowing as wnd
    from src import normalization as norm
    from src import base_relative as br
    grid = wnd.ventanear(particion_final_pool, 360, 60)
    X_norm = norm.aplicar_zscore(grid["X"], params_norm_p)
    X_hat = modelo_p.reconstruir(X_norm)
    p_score_clean = br.calcular_relative_kw(X_norm, X_hat, params_norm_p)
    p_ts_clean = pd.to_datetime(grid["fin"]).to_numpy()

    val15_pool = pcl.construir_serie_15min(particion_final_pool)
    assert len(val15_pool["ts_end"]) == len(ts_h_clean)
    assert np.array_equal(pd.DatetimeIndex(val15_pool["ts_end"]), pd.DatetimeIndex(ts_h_clean)), \
        "val15_pool y ts_h_clean deben ser la misma rejilla densa de 15 min sobre el pool"

    particion_pool_min_raw = particion_final_pool[COLUMNA_OBJETIVO].to_numpy(dtype=np.float64)
    meta_h, feats_h = pcl.puntuar_ataques(ataques, val15_pool, particion_pool_min_raw, perfil_final, len(val15_pool["kw"]))
    if len(feats_h):
        pred_h_at = modelo_h_final.predict(feats_h[COLS_H].to_numpy(dtype=np.float64))
        _, score_h_at = pcl.calcular_score(pred_h_at, meta_h["target_kw_atacado"].to_numpy())
        meta_h = meta_h.copy()
        meta_h["score_atacado_H"] = score_h_at
        assert np.array_equal(meta_h["target_end"].to_numpy(), ts_h_clean[meta_h["t"].to_numpy()]), \
            "meta_h['t'] no indexa correctamente ts_h_clean -- desalineamiento en la puntuacion atacada de H"
        meta_h["k"] = meta_h["t"]

    p_global_pool = {"ts": p_ts_clean, "score": p_score_clean, "umbral": THRESHOLD_P, "modelo": modelo_p, "params_norm": params_norm_p}
    tabla_p_atacada, _ = opt.puntuar_p_en_fold(ataques, particion_pool_min_raw, particion_final_pool, final_cal_pool_ini, p_global_pool)
    p_atacado_por_episodio = opt.construir_p_atacado_por_episodio(tabla_p_atacada, p_global_pool)

    activo_p_clean_full = fus.estado_ffill(p_score_clean, THRESHOLD_P, p_ts_clean, ts_h_clean)
    activo_h_clean_full = positive_h_clean > THRESHOLD_H

    return {
        "p_score_clean": p_score_clean, "p_ts_clean": p_ts_clean, "positive_h_clean": positive_h_clean,
        "ts_h_clean": ts_h_clean, "meta_h": meta_h, "p_atacado_por_episodio": p_atacado_por_episodio,
        "particion_pool_min_raw": particion_pool_min_raw,
        "activo_clean_full_P": activo_p_clean_full, "activo_clean_full_H": activo_h_clean_full,
        "activo_clean_full_OR": activo_p_clean_full | activo_h_clean_full,
    }


# ==========================================================================================
# Ventana explicita por episodio (buffer previo + ataque + posterior), zonas y series
# limpia/atacada P/H/OR alineadas sobre la rejilla de 15 min de H
# ==========================================================================================

def construir_ventana_episodio(a: dict, r: dict) -> dict:
    ts_h = r["ts_h_clean"]
    attack_start, attack_end = a["attack_start"], a["attack_end"]
    ventana_ini = attack_start - pd.Timedelta(minutes=PRE_BUFFER_MIN)
    ventana_fin = attack_end + pd.Timedelta(minutes=POST_ATTACK_MIN)

    idx_ini = int(np.searchsorted(ts_h, np.datetime64(ventana_ini), side="left"))
    idx_fin = int(np.searchsorted(ts_h, np.datetime64(ventana_fin), side="right"))
    ks = np.arange(idx_ini, idx_fin)
    ts_ventana = ts_h[ks]

    # H: limpio y atacado (donde no hay override atacado -> se usa el limpio, exactamente
    # igual que fusion_p_histgb.construir_estados_episodio)
    score_h_clean_ventana = r["positive_h_clean"][ks]
    meta_h = r["meta_h"]
    ep_meta = meta_h[meta_h["episode_id"] == a["episode_id"]] if len(meta_h) else meta_h
    score_h_atacado_ventana = score_h_clean_ventana.copy()
    if len(ep_meta):
        overrides = ep_meta.set_index("k")["score_atacado_H"]
        for k_local, k_global in enumerate(ks):
            if k_global in overrides.index:
                score_h_atacado_ventana[k_local] = overrides.loc[k_global]

    # P: limpio (ffill) y atacado (overlay + ffill), reutilizando fus.estado_ffill literalmente
    p_ts, p_score_clean_full = r["p_ts_clean"], r["p_score_clean"]
    activo_p_clean_ventana = fus.estado_ffill(p_score_clean_full, THRESHOLD_P, p_ts, ts_ventana)
    p_atacado_ep = r["p_atacado_por_episodio"].get(a["episode_id"])
    if p_atacado_ep is not None and len(p_atacado_ep["k"]):
        p_score_full_ep = p_score_clean_full.copy()
        p_score_full_ep[p_atacado_ep["k"]] = p_atacado_ep["score"]
    else:
        p_score_full_ep = p_score_clean_full
    activo_p_atacado_ventana = fus.estado_ffill(p_score_full_ep, THRESHOLD_P, p_ts, ts_ventana)

    activo_h_clean_ventana = score_h_clean_ventana > THRESHOLD_H
    activo_h_atacado_ventana = score_h_atacado_ventana > THRESHOLD_H

    activo_clean = {"P": activo_p_clean_ventana, "H": activo_h_clean_ventana, "OR": activo_p_clean_ventana | activo_h_clean_ventana}
    activo_attack = {"P": activo_p_atacado_ventana, "H": activo_h_atacado_ventana, "OR": activo_p_atacado_ventana | activo_h_atacado_ventana}
    scores_clean = {"P": None, "H": score_h_clean_ventana}
    scores_attack = {"P": None, "H": score_h_atacado_ventana}

    boundary_w = _boundary_width_minutes(a["duracion"])
    zona = np.full(len(ts_ventana), "pre_buffer", dtype=object)
    onset_end = attack_start + pd.Timedelta(minutes=boundary_w)
    end_start = attack_end - pd.Timedelta(minutes=boundary_w)
    zona[(ts_ventana >= np.datetime64(attack_start)) & (ts_ventana < np.datetime64(onset_end))] = "onset"
    zona[(ts_ventana >= np.datetime64(onset_end)) & (ts_ventana < np.datetime64(end_start))] = "interior"
    zona[(ts_ventana >= np.datetime64(end_start)) & (ts_ventana < np.datetime64(attack_end))] = "end"
    zona[(ts_ventana >= np.datetime64(attack_end)) & (ts_ventana < np.datetime64(ventana_fin))] = "post"

    return {
        "ks": ks, "ts": ts_ventana, "zona": zona, "boundary_width_min": boundary_w,
        "activo_clean": activo_clean, "activo_attack": activo_attack,
        "scores_clean": scores_clean, "scores_attack": scores_attack,
    }


# ==========================================================================================
# Deteccion estandar (reutiliza fus.construir_estados_episodio/simular_episodios_generico
# literalmente) + deteccion inducida/zonas/energia/preexistencia por episodio
# ==========================================================================================

def deteccion_estandar_episodio(a: dict, r: dict) -> dict:
    meta_h = r["meta_h"]
    grp = meta_h[meta_h["episode_id"] == a["episode_id"]] if len(meta_h) else meta_h
    p_atacado_ep = r["p_atacado_por_episodio"].get(a["episode_id"])
    estados = fus.construir_estados_episodio(
        grp if len(grp) else None, r["p_ts_clean"], r["p_score_clean"], THRESHOLD_P, p_atacado_ep,
        r["positive_h_clean"], THRESHOLD_H, r["ts_h_clean"])
    out = {}
    for metodo in METODOS:
        ids = {a["episode_id"]}
        estados_dict = {a["episode_id"]: estados}
        res = fus.simular_episodios_generico(ids, estados_dict, metodo, r[f"activo_clean_full_{metodo}"], r["ts_h_clean"])
        if len(res):
            out[f"detected_standard_{metodo}"] = bool(res.iloc[0]["induced_detected"])
        else:
            out[f"detected_standard_{metodo}"] = False
    return out


def analizar_episodio(a: dict, r: dict) -> dict:
    ventana = construir_ventana_episodio(a, r)
    zona = ventana["zona"]
    activo_clean, activo_attack = ventana["activo_clean"], ventana["activo_attack"]

    zona_ataque_mask = np.isin(zona, ["onset", "interior", "end"])
    resultado = {"episode_id": a["episode_id"]}

    ts_all = ventana["ts"]
    primera_deteccion_ts, primera_deteccion_zona = pd.NaT, "none"
    orden_zonas = ["onset", "interior", "end", "post"]

    for metodo in METODOS:
        inducido = activo_attack[metodo] & ~activo_clean[metodo]
        resultado[f"detected_onset_zone_{metodo}"] = bool(inducido[zona == "onset"].any())
        resultado[f"detected_interior_{metodo}"] = bool(inducido[zona == "interior"].any())
        resultado[f"detected_end_zone_{metodo}"] = bool(inducido[zona == "end"].any())
        resultado[f"detected_post_attack_{metodo}"] = bool(inducido[zona == "post"].any())
        resultado[f"detected_induced_{metodo}"] = bool(inducido[zona_ataque_mask | (zona == "post")].any())

        idx_ind = np.where(inducido)[0]
        if len(idx_ind):
            t0 = ts_all[idx_ind[0]]
            if pd.isna(primera_deteccion_ts) or t0 < primera_deteccion_ts:
                primera_deteccion_ts = t0
                for z in orden_zonas:
                    if zona[idx_ind[0]] == z:
                        primera_deteccion_zona = z
                        break

        # alerta preexistente: activo en clean inmediatamente antes del onset
        idx_pre = np.where(zona == "pre_buffer")[0]
        resultado[f"preexisting_alert_{metodo}"] = bool(activo_clean[metodo][idx_pre].any()) if len(idx_pre) else False

    resultado["first_detection_timestamp"] = primera_deteccion_ts
    resultado["first_detection_zone"] = primera_deteccion_zona
    resultado["delay_minutes"] = (
        float((primera_deteccion_ts - a["attack_start"]) / pd.Timedelta(minutes=1)) if pd.notna(primera_deteccion_ts) else np.nan)

    for metodo in METODOS:
        resultado[f"boundary_only_detection_{metodo}"] = bool(
            (resultado[f"detected_onset_zone_{metodo}"] or resultado[f"detected_end_zone_{metodo}"])
            and not resultado[f"detected_interior_{metodo}"])
        resultado[f"post_only_detection_{metodo}"] = bool(
            not (resultado[f"detected_onset_zone_{metodo}"] or resultado[f"detected_interior_{metodo}"] or resultado[f"detected_end_zone_{metodo}"])
            and resultado[f"detected_post_attack_{metodo}"])

    resultado["detected_by_P"] = resultado["detected_induced_P"]
    resultado["detected_by_H"] = resultado["detected_induced_H"]
    resultado["detected_by_OR"] = resultado["detected_induced_OR"]
    resultado["P_only"] = resultado["detected_by_P"] and not resultado["detected_by_H"]
    resultado["H_only"] = resultado["detected_by_H"] and not resultado["detected_by_P"]
    resultado["both"] = resultado["detected_by_P"] and resultado["detected_by_H"]

    h_clean_scores, h_attack_scores = ventana["scores_clean"]["H"], ventana["scores_attack"]["H"]
    resultado["max_score_H_clean"] = float(np.max(h_clean_scores)) if len(h_clean_scores) else np.nan
    resultado["max_score_H_attack"] = float(np.max(h_attack_scores)) if len(h_attack_scores) else np.nan
    resultado["max_delta_score_H"] = float(np.max(h_attack_scores - h_clean_scores)) if len(h_clean_scores) else np.nan

    resultado["boundary_width_minutes"] = ventana["boundary_width_min"]
    resultado["preexisting_alert"] = resultado["preexisting_alert_OR"]
    return resultado


def energia_y_similitud_episodio(a: dict, r: dict, serie_completa_pretest: pd.DataFrame) -> dict:
    valores = serie_completa_pretest[COLUMNA_OBJETIVO]
    dest_ts = pd.date_range(a["attack_start"], periods=a["duracion"], freq="1min")
    clean_dest = valores.reindex(dest_ts).to_numpy(dtype=np.float64)
    replay_vals = a["y_tramo"]

    e_clean = float(clean_dest.sum() / 60.0)
    e_replay = float(replay_vals.sum() / 60.0)
    hidden = e_clean - e_replay
    tol = max(ENERGY_TOL_ABS, ENERGY_TOL_REL * e_clean)
    if hidden > tol:
        categoria = "underreporting"
    elif hidden < -tol:
        categoria = "overreporting"
    else:
        categoria = "near_neutral"

    diff = clean_dest - replay_vals
    corr = float(np.corrcoef(clean_dest, replay_vals)[0, 1]) if np.std(clean_dest) > 0 and np.std(replay_vals) > 0 else np.nan

    last_clean_before = float(valores.loc[:a["attack_start"] - pd.Timedelta(minutes=1)].iloc[-1])
    first_replay = float(replay_vals[0])
    last_replay = float(replay_vals[-1])
    first_clean_after = float(valores.loc[a["attack_end"]:].iloc[0])

    onset_jump = first_replay - last_clean_before
    end_jump = first_clean_after - last_replay

    return {
        "episode_id": a["episode_id"], "clean_energy_kwh": e_clean, "replay_energy_kwh": e_replay,
        "hidden_energy_kwh": hidden, "hidden_energy_fraction": hidden / e_clean if e_clean > 1e-9 else np.nan,
        "economic_category": categoria, "tolerance_used": tol,
        "source_destination_mae": float(np.mean(np.abs(diff))), "source_destination_rmse": float(np.sqrt(np.mean(diff ** 2))),
        "source_destination_correlation": corr, "diff_mean": float(np.mean(diff)), "diff_median": float(np.median(diff)),
        "diff_p95": float(np.percentile(np.abs(diff), 95)),
        "onset_jump_kw": onset_jump, "end_jump_kw": end_jump,
        "onset_jump_relative": onset_jump / max(abs(last_clean_before), 1e-9),
        "end_jump_relative": end_jump / max(abs(last_replay), 1e-9),
    }


# ==========================================================================================
# Orquestacion principal
# ==========================================================================================

def cargar_modelos_congelados():
    from src.base_relative import cargar_ataques_y_modelo_360
    _, datos_p, modelo_p, _, _ = cargar_ataques_y_modelo_360()
    params_norm_p = datos_p["params_norm"]
    final_or_dir = EXP_DIR.parents[1] / "models" / "final_or_pretest"
    modelo_h_final = joblib.load(final_or_dir / "histgb_periodic_final.joblib")
    perfil_final = joblib.load(final_or_dir / "profile_train_final.joblib")
    return modelo_p, params_norm_p, modelo_h_final, perfil_final


def ejecutar_evaluacion(overwrite: bool = False) -> dict:
    out_path = TABLES_DIR / "replay_episode_results.csv"
    if out_path.exists() and not overwrite:
        raise SystemExit(f"{out_path} ya existe. Usa --overwrite para reemplazarlo.")

    manifest_path = MANIFESTS_DIR / "replay_pilot_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"FALTA {manifest_path}. Ejecuta antes build_replay_manifest.py --build-manifest.")
    tabla_manifest = pd.read_csv(manifest_path, parse_dates=["destination_start", "destination_end", "source_start", "source_end"])
    manifest_hash_congelado = tabla_manifest["manifest_hash"].iloc[0]

    t0 = time.time()
    print("[evaluate] Verificando decision previa y preparando periodo...")
    periodo = seleccionar_periodo()

    print("[evaluate] Cargando checkpoints congelados (solo inferencia, sin fit)...")
    modelo_p, params_norm_p, modelo_h_final, perfil_final = cargar_modelos_congelados()

    print("[evaluate] Construyendo diccionarios de ataque replay (copia exacta)...")
    ataques = construir_ataques_replay(tabla_manifest, periodo["serie_completa_pretest"], periodo["final_cal_pool_ini"])
    tabla_copia_exacta = verificar_copia_exacta(ataques, periodo["serie_completa_pretest"])
    assert (tabla_copia_exacta["n_exact_copy_mismatches"] == 0).all(), "copia no exacta detectada en al menos un episodio"

    print(f"[evaluate] Puntuando pool limpio y {len(ataques)} episodios atacados (P y H, UNA sola vez)...")
    r = puntuar_pool(periodo, ataques, modelo_p, params_norm_p, modelo_h_final, perfil_final)
    print(f"[evaluate]   OK en {time.time()-t0:.1f}s")

    filas_deteccion, filas_energia = [], []
    for episode_id, a in ataques.items():
        a = {**a, "episode_id": episode_id}
        std = deteccion_estandar_episodio(a, r)
        zon = analizar_episodio(a, r)
        fila = {**std, **zon}
        filas_deteccion.append(fila)
        filas_energia.append(energia_y_similitud_episodio(a, r, periodo["serie_completa_pretest"]))

    tabla_deteccion = pd.DataFrame(filas_deteccion)
    tabla_energia = pd.DataFrame(filas_energia)

    resultados = tabla_manifest.merge(tabla_deteccion, on="episode_id").merge(tabla_energia, on="episode_id")
    resultados["manifest_hash_used"] = manifest_hash_congelado
    resultados["valid_episode"] = True
    resultados["warnings"] = ""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(out_path, index=False)
    tabla_copia_exacta.to_csv(TABLES_DIR / "exact_copy_verification.csv", index=False)
    print(f"[evaluate] Resultados guardados en {out_path} ({len(resultados)} episodios)")

    print("[evaluate] Calculando resumenes, zonas, energia, complementariedad...")
    from experiments.replay_pilot.src import generate_replay_report as rep
    resumen = rep.construir_resumenes(resultados)
    rep.guardar_resumenes(resumen)
    print("[evaluate] Generando figuras (globales + muestra de episodios)...")
    rep.generar_figuras_globales(resultados, resumen)
    rep.generar_figuras_episodios(resultados, r, ataques, n_muestra=10)

    manifest_run = {
        "status": "REPLAY_PILOT_EVALUATED",
        "n_episodes": len(resultados), "manifest_hash_used": manifest_hash_congelado,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.time() - t0,
        "thresholds": {"P": THRESHOLD_P, "H": THRESHOLD_H},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "replay_pilot_results.json", "w", encoding="utf-8") as f:
        json.dump(manifest_run, f, indent=2, ensure_ascii=False, default=str)

    print(f"[evaluate] Completado en {time.time()-t0:.1f}s")
    return {"resultados": resultados, "resumen": resumen}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.parse_args()
    ejecutar_evaluacion(overwrite=("--overwrite" in sys.argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
