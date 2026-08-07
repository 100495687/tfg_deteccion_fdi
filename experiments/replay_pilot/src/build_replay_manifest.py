"""Construccion y congelacion del manifiesto de episodios replay (piloto).

Selecciona 2 desplazamientos (24h, 7d) x 3 duraciones (120/360/1440 min) x hasta 10
destinos emparejados por duracion = hasta 60 episodios, exclusivamente sobre un periodo
pre-test (FINAL_CAL_POOL_180), sin usar en ningun momento scores, energia ni resultados
de deteccion para seleccionar destinos (solo timestamp/calidad/calendario).

No entrena, no recalibra, no modifica artefactos congelados. Reutiliza literalmente
congelacion_final_or_pretest.verificar_decision_previa/preparar_periodos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import congelacion_final_or_pretest as cfg
from src import optimizacion_histgb_periodic as opt
from src import predictor_causal_lags as pcl

EXP_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXP_DIR / "config" / "replay_pilot.yaml"
MANIFESTS_DIR = EXP_DIR / "manifests"
TABLES_DIR = EXP_DIR / "tables"
REPORTS_DIR = EXP_DIR / "reports"
REPO_DIR = EXP_DIR.parents[1]

RANDOM_SEED = 20260801
DURATIONS_MIN = [120, 360, 1440]
SHIFTS = {"DAILY": 1440, "WEEKLY": 10080}  # minutos
N_DEST_PER_DURATION = 10
POST_CONTEXT_MIN = 60  # "al menos una hora posterior" (criterio 13)
H_LAG_CONTEXT_MIN = pcl.MAX_LAG * 15  # 672*15 = 10080 min = 7 dias (criterio 12)
P_CONTEXT_MIN = 360  # ventana de P (criterio 11)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
    except Exception:
        return "unknown"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================================================================
# Seccion 8: inventario de artefactos de entrada
# ==========================================================================================

def build_input_inventory() -> pd.DataFrame:
    rows = []

    def add(name, path, typ, split, mutable, selected, reason, contains_test):
        p = REPO_DIR / path if not str(path).startswith("experiments") else EXP_DIR.parent.parent / path
        p = Path(path)
        if not p.is_absolute():
            p = REPO_DIR / path
        exists = p.exists() and p.is_file()
        rows.append({
            "artifact_name": name, "path": str(path), "type": typ, "split": split, "fold": "n/a",
            "date_start": "", "date_end": "", "n_rows": "",
            "hash": _sha256_file(p) if exists else "MISSING",
            "selected": selected, "reason": reason, "contains_test_information": contains_test,
            "mutable": mutable, "validation_status": "OK" if exists else "MISSING",
        })

    add("frozen_h_model", "models/final_or_pretest/histgb_periodic_final.joblib", "frozen_model_checkpoint",
        "final_pretest_freeze", False, True, "Unico modelo H ya entrenado y congelado disponible; se reutiliza via predict() puro", False)
    add("frozen_h_profile", "models/final_or_pretest/profile_train_final.joblib", "frozen_artifact",
        "final_pretest_freeze", False, True, "Perfil historico congelado usado por H para las features PERIODIC", False)
    add("threshold_p_final", "models/final_or_pretest/threshold_p_final.json", "frozen_threshold",
        "final_pretest_freeze", False, True, "Umbral P congelado = 0.049748", False)
    add("threshold_h_final", "models/final_or_pretest/threshold_h_final.json", "frozen_threshold",
        "final_pretest_freeze", False, True, "Umbral H congelado = 0.883628", False)
    add("decision_config", "models/auditoria_final_p_vs_or/configuracion_final_decision_pretest.json",
        "frozen_config", "pretest", False, True, "Decision de arquitectura ya cerrada; se verifica sin recomparar", False)
    add("edo_umbrales_p", "results/tables/energy_distance_operativo/02_umbrales_calibrados.csv",
        "frozen_threshold_table", "pretest", False, True, "Umbral P global (P_MULTI_SEASON) leido por optimizacion_histgb_periodic.cargar_p_global", False)
    add("congelacion_config", "models/final_or_pretest/configuracion_final_or_pretest.json", "frozen_config",
        "final_pretest_freeze", False, False, "Referencia informativa, no se reutiliza directamente en este piloto", False)
    add("test_attacks_v2", "manifests/test_attacks_v2/attack_manifest_test_v2.csv", "attack_manifest", "test",
        False, False, "Manifiesto del TEST FINAL -- PROHIBIDO en este piloto (solo pre-test)", True)
    add("threshold_tradeoff_selected", "experiments/threshold_tradeoff/tables/selected_operating_points.csv",
        "other_experiment_output", "n/a", False, False, "Experimento anterior independiente; no se modifica ni se reutiliza aqui", False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "input_artifacts_inventory.csv", index=False)
    return df


# ==========================================================================================
# Seccion 5: seleccion del periodo pre-test
# ==========================================================================================

def seleccionar_periodo() -> dict:
    print("[manifest] Verificando decision de arquitectura ya cerrada (sin recomparar)...")
    cfg.verificar_decision_previa()
    print("[manifest] OK.")

    periodos = cfg.preparar_periodos()
    particion_entrenamiento = periodos["particion_entrenamiento"]
    particion_final_pool = periodos["particion_final_pool"]
    final_cal_pool_ini = periodos["final_cal_pool_ini"]
    fin_pretest = periodos["fin_pretest"]

    justificacion = {
        "periodo_elegido": "FINAL_CAL_POOL_180",
        "rango": [str(final_cal_pool_ini), str(fin_pretest)],
        "duracion_dias": float((fin_pretest - final_cal_pool_ini) / pd.Timedelta(days=1)),
        "razones": [
            "Es el UNICO periodo pre-test con un modelo H ya entrenado y congelado en disco "
            "(models/final_or_pretest/histgb_periodic_final.joblib) -- los 4 folds walk-forward "
            "de seleccion de arquitectura no tienen H persistido por fold (requeriria fit()).",
            "P es un checkpoint global ya congelado, valido para cualquier periodo pre-test por igual.",
            "Continuidad temporal: 180 dias consecutivos sin huecos de particion.",
            f"Compatible con los lags de H (7 dias = {H_LAG_CONTEXT_MIN} min): FINAL_TRAIN precede "
            "directamente al pool con ~2.5 anios de datos limpios, suficiente para el contexto de "
            "origen incluso en destinos cercanos al inicio del pool.",
            "Facilidad tecnica: reutiliza literalmente congelacion_final_or_pretest.predecir_pool_limpio "
            "y evaluacion_final_retrospectiva_test.puntuar_p_limpio sin modificarlas.",
        ],
        "no_se_uso_como_criterio": "cantidad de detecciones esperada en el periodo (no se consulto)",
    }
    print(f"[manifest] Periodo seleccionado: FINAL_CAL_POOL_180 [{final_cal_pool_ini} -> {fin_pretest}]")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "period_selection_justification.json", "w", encoding="utf-8") as f:
        json.dump(justificacion, f, indent=2, ensure_ascii=False, default=str)

    return {
        "particion_entrenamiento": particion_entrenamiento, "particion_final_pool": particion_final_pool,
        "final_cal_pool_ini": final_cal_pool_ini, "fin_pretest": fin_pretest,
        "serie_completa_pretest": pd.concat([particion_entrenamiento, particion_final_pool]),
        "justificacion": justificacion,
    }


# ==========================================================================================
# Seccion 11: calendario (trimestre / franja horaria / laborable-finde)
# ==========================================================================================

def _daypart(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 0 <= h < 6:
        return "madrugada"
    if 6 <= h < 12:
        return "mañana"
    if 12 <= h < 18:
        return "tarde"
    return "noche"


def _quarter(ts: pd.Timestamp) -> int:
    return ts.quarter


def _is_weekend(ts: pd.Timestamp) -> bool:
    return ts.dayofweek >= 5


# ==========================================================================================
# Seccion 12: validez de un destino candidato (para ambos desplazamientos)
# ==========================================================================================

def evaluar_candidato(dest_start: pd.Timestamp, duration_min: int, serie: pd.DataFrame,
                       pool_ini: pd.Timestamp, pool_fin: pd.Timestamp) -> dict:
    dest_end = dest_start + pd.Timedelta(minutes=duration_min)  # exclusivo
    razones = []

    # 1. destino completo dentro del periodo pre-test elegido
    if not (dest_start >= pool_ini and dest_end <= pool_fin):
        return {"valid": False, "reason": "destino_fuera_del_pool"}

    # 12/13: contexto causal para P (360 min) y H (7 dias) antes del destino, y 60 min despues.
    # critico: el contexto de H se exige relativo al inicio del pool, no al inicio de toda la
    # serie pre-test -- predictor_causal_lags.puntuar_ataques descarta internamente (t < MAX_LAG)
    # cualquier bin cuyo indice en la rejilla densa del pool sea anterior a 7 dias desde el
    # inicio del pool, aunque exista historia limpia mas alla en FINAL_TRAIN. Un destino dentro
    # de los primeros 7 dias del pool nunca produce filas en meta_h (bug detectado empiricamente
    # en el piloto v1, ver manifest_freeze_record_v2 -- corregido aqui, no relajado).
    if dest_start < pool_ini + pd.Timedelta(minutes=H_LAG_CONTEXT_MIN):
        return {"valid": False, "reason": "destino_dentro_del_calentamiento_de_7_dias_del_pool_para_H"}
    if dest_start - pd.Timedelta(minutes=max(P_CONTEXT_MIN, H_LAG_CONTEXT_MIN)) < serie.index.min():
        return {"valid": False, "reason": "contexto_causal_insuficiente_antes_del_destino"}
    if dest_end + pd.Timedelta(minutes=POST_CONTEXT_MIN) > serie.index.max() + pd.Timedelta(minutes=1):
        return {"valid": False, "reason": "sin_una_hora_de_recuperacion_tras_el_destino"}

    origenes = {}
    for shift_name, shift_min in SHIFTS.items():
        o_start = dest_start - pd.Timedelta(minutes=shift_min)
        o_end = o_start + pd.Timedelta(minutes=duration_min)
        # 16/17: origen estrictamente anterior al destino, sin solape
        if not (o_end <= dest_start):
            return {"valid": False, "reason": f"origen_{shift_name}_no_precede_al_destino"}
        # 2/3: origen debe existir completamente dentro de la serie pre-test disponible
        if o_start < serie.index.min() or o_end > serie.index.max() + pd.Timedelta(minutes=1):
            return {"valid": False, "reason": f"origen_{shift_name}_fuera_de_rango_pretest"}
        origenes[shift_name] = (o_start, o_end)

    dest_slice = serie.loc[dest_start:dest_end - pd.Timedelta(minutes=1)]
    # 4/5: misma cantidad de muestras y todos los timestamps esperados
    if len(dest_slice) != duration_min:
        return {"valid": False, "reason": "destino_con_muestras_faltantes"}
    # 6/8/9: datos validos, sin NaN, sin valores no finitos
    if dest_slice["Global_active_power"].isna().any() or not np.isfinite(dest_slice["Global_active_power"]).all():
        return {"valid": False, "reason": "destino_con_nan_o_no_finito"}
    # calidad: no atravesar tramos imputados (criterio de calidad, no de energia/score)
    if dest_slice["imputado"].any():
        return {"valid": False, "reason": "destino_contiene_valores_imputados"}

    for shift_name, (o_start, o_end) in origenes.items():
        o_slice = serie.loc[o_start:o_end - pd.Timedelta(minutes=1)]
        if len(o_slice) != duration_min:
            return {"valid": False, "reason": f"origen_{shift_name}_con_muestras_faltantes"}
        if o_slice["Global_active_power"].isna().any() or not np.isfinite(o_slice["Global_active_power"]).all():
            return {"valid": False, "reason": f"origen_{shift_name}_con_nan_o_no_finito"}
        if o_slice["imputado"].any():
            return {"valid": False, "reason": f"origen_{shift_name}_contiene_valores_imputados"}

    return {
        "valid": True, "reason": "ok", "dest_start": dest_start, "dest_end": dest_end,
        "origin_daily_start": origenes["DAILY"][0], "origin_daily_end": origenes["DAILY"][1],
        "origin_weekly_start": origenes["WEEKLY"][0], "origin_weekly_end": origenes["WEEKLY"][1],
        "quarter": _quarter(dest_start), "daypart": _daypart(dest_start), "is_weekend": _is_weekend(dest_start),
    }


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def _todas_las_ventanas(cand: dict) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(cand["dest_start"], cand["dest_end"]), (cand["origin_daily_start"], cand["origin_daily_end"]),
            (cand["origin_weekly_start"], cand["origin_weekly_end"])]


def _choca_con_seleccionados(cand: dict, seleccionados: list[dict]) -> bool:
    ventanas_cand = _todas_las_ventanas(cand)
    for sel in seleccionados:
        for s1, e1 in ventanas_cand:
            for s2, e2 in _todas_las_ventanas(sel):
                if _overlaps(s1, e1, s2, e2):
                    return True
    return False


# ==========================================================================================
# Seccion 9-11: generacion de candidatos y seleccion determinista estratificada
# ==========================================================================================

def generar_candidatos_duracion(duration_min: int, periodo: dict, rng: np.random.Generator,
                                 n_muestra: int = 4000) -> tuple[pd.DataFrame, pd.DataFrame]:
    serie = periodo["serie_completa_pretest"]
    pool_ini, pool_fin = periodo["final_cal_pool_ini"], periodo["fin_pretest"]

    max_start_min = int((pool_fin - pool_ini) / pd.Timedelta(minutes=1)) - duration_min
    offsets = rng.choice(max_start_min, size=min(n_muestra, max_start_min), replace=False)
    offsets = np.sort(offsets)

    validos, rechazados = [], []
    for off in offsets:
        dest_start = pool_ini + pd.Timedelta(minutes=int(off))
        res = evaluar_candidato(dest_start, duration_min, serie, pool_ini, pool_fin)
        if res["valid"]:
            res["duration_minutes"] = duration_min
            validos.append(res)
        else:
            rechazados.append({"duration_minutes": duration_min, "dest_start_candidate": str(dest_start), "rejection_reason": res["reason"]})
    return pd.DataFrame(validos), pd.DataFrame(rechazados)


def _objetivo_trimestre(quarters_presentes: list[int], n: int) -> dict:
    base = n // len(quarters_presentes)
    resto = n - base * len(quarters_presentes)
    obj = {q: base for q in quarters_presentes}
    for i, q in enumerate(sorted(quarters_presentes)):
        if i < resto:
            obj[q] += 1
    return obj


def _objetivo_daypart(n: int, rotacion: int) -> dict:
    dayparts = ["madrugada", "mañana", "tarde", "noche"]
    orden_rotado = dayparts[rotacion % 4:] + dayparts[:rotacion % 4]
    objetivo_base = [3, 3, 2, 2]
    return {orden_rotado[i]: objetivo_base[i] for i in range(4)}


def seleccionar_destinos_duracion(candidatos: pd.DataFrame, duration_min: int, n_deseado: int,
                                   rng: np.random.Generator, rotacion_daypart: int,
                                   ya_seleccionados_global: list[dict] | None = None) -> tuple[list[dict], dict]:
    """`ya_seleccionados_global` acumula los destinos ya elegidos en duraciones anteriores del
    mismo manifiesto (criterio 14/15: los destinos -- y sus origenes -- no deben
    solaparse entre si ni entre duraciones distintas, no solo dentro de la misma duracion)."""
    if len(candidatos) == 0:
        return [], {"duration_minutes": duration_min, "n_candidatos_validos": 0, "n_seleccionados": 0,
                     "motivo_insuficiente": "no hay candidatos validos para esta duracion"}

    quarters_presentes = sorted(candidatos["quarter"].unique().tolist())
    obj_q = _objetivo_trimestre(quarters_presentes, n_deseado)
    obj_dp = _objetivo_daypart(n_deseado, rotacion_daypart)
    obj_weekday, obj_weekend = 6, 4
    if n_deseado != 10:
        obj_weekday = round(n_deseado * 0.6)
        obj_weekend = n_deseado - obj_weekday

    orden = rng.permutation(len(candidatos))
    pool = [candidatos.iloc[i].to_dict() for i in orden]

    seleccionados: list[dict] = []
    bloqueados_por_otras_duraciones = list(ya_seleccionados_global or [])
    cnt_q: dict = {q: 0 for q in quarters_presentes}
    cnt_dp: dict = {dp: 0 for dp in obj_dp}
    cnt_wd, cnt_we = 0, 0

    while len(seleccionados) < n_deseado and pool:
        mejor_idx, mejor_puntuacion = None, None
        for i, cand in enumerate(pool):
            if _choca_con_seleccionados(cand, seleccionados) or _choca_con_seleccionados(cand, bloqueados_por_otras_duraciones):
                continue
            deficit_q = max(0, obj_q.get(cand["quarter"], 0) - cnt_q.get(cand["quarter"], 0))
            deficit_dp = max(0, obj_dp.get(cand["daypart"], 0) - cnt_dp.get(cand["daypart"], 0))
            deficit_wd = max(0, (obj_weekend - cnt_we) if cand["is_weekend"] else (obj_weekday - cnt_wd))
            puntuacion = (deficit_q, deficit_dp, deficit_wd)
            if mejor_puntuacion is None or puntuacion > mejor_puntuacion:
                mejor_puntuacion, mejor_idx = puntuacion, i
        if mejor_idx is None:
            break
        elegido = pool.pop(mejor_idx)
        seleccionados.append(elegido)
        cnt_q[elegido["quarter"]] = cnt_q.get(elegido["quarter"], 0) + 1
        cnt_dp[elegido["daypart"]] = cnt_dp.get(elegido["daypart"], 0) + 1
        if elegido["is_weekend"]:
            cnt_we += 1
        else:
            cnt_wd += 1

    auditoria = {
        "duration_minutes": duration_min, "n_candidatos_validos": len(candidatos), "n_seleccionados": len(seleccionados),
        "objetivo_trimestre": obj_q, "conteo_trimestre": cnt_q, "objetivo_daypart": obj_dp, "conteo_daypart": cnt_dp,
        "objetivo_weekday": obj_weekday, "objetivo_weekend": obj_weekend, "conteo_weekday": cnt_wd, "conteo_weekend": cnt_we,
        "insuficiente": len(seleccionados) < n_deseado,
    }
    return seleccionados, auditoria


# ==========================================================================================
# Seccion 13: construccion y congelacion del manifiesto
# ==========================================================================================

def construir_manifiesto(seleccionados_por_duracion: dict, periodo: dict) -> pd.DataFrame:
    filas = []
    for duration_min, seleccionados in seleccionados_por_duracion.items():
        for i, dest in enumerate(seleccionados):
            paired_id = f"D{duration_min}_{i:03d}"
            for shift_name, shift_min in SHIFTS.items():
                o_start = dest[f"origin_{shift_name.lower()}_start"]
                o_end = dest[f"origin_{shift_name.lower()}_end"]
                filas.append({
                    "episode_id": f"{paired_id}_{shift_name}", "paired_destination_id": paired_id,
                    "shift_type": shift_name, "shift_minutes": shift_min, "duration_minutes": duration_min,
                    "destination_start": dest["dest_start"], "destination_end": dest["dest_end"],
                    "source_start": o_start, "source_end": o_end,
                    "quarter": dest["quarter"], "daypart": dest["daypart"], "is_weekend": bool(dest["is_weekend"]),
                    "fold": "FINAL_CAL_POOL_180", "split": "pretest", "seed": RANDOM_SEED,
                    "source_valid": True, "destination_valid": True, "p_context_valid": True, "h_context_valid": True,
                    "post_context_valid": True, "selected": True, "rejection_reason": "",
                })
    return pd.DataFrame(filas)


def congelar_manifiesto(tabla: pd.DataFrame, auditorias: list[dict], periodo: dict, version: int = 1) -> dict:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    tabla = tabla.copy()
    tabla["manifest_version"] = version
    csv_bytes = tabla.drop(columns=["manifest_hash"], errors="ignore").to_csv(index=False).encode("utf-8")
    manifest_hash = _sha256_bytes(csv_bytes)
    tabla["manifest_hash"] = manifest_hash
    tabla.to_csv(MANIFESTS_DIR / "replay_pilot_manifest.csv", index=False)

    freeze = {
        "status": "MANIFEST_FROZEN_READY_FOR_INJECTION" if len(tabla) > 0 else "MANIFEST_EMPTY",
        "manifest_version": version, "n_episodes": int(len(tabla)),
        "n_paired_destinations": int(tabla["paired_destination_id"].nunique()) if len(tabla) else 0,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "random_seed": RANDOM_SEED,
        "manifest_hash_sha256": manifest_hash,
        "period_used": "FINAL_CAL_POOL_180",
        "period_range": [str(periodo["final_cal_pool_ini"]), str(periodo["fin_pretest"])],
        "stratification_audit": auditorias,
        "selection_criteria_used": ["timestamp", "data_quality_mask(imputado)", "quarter", "daypart", "weekday_weekend",
                                     "non_overlap", "causal_context_availability"],
        "selection_criteria_forbidden_and_confirmed_not_used": ["score_P", "score_H", "clean_alert", "detection_rate",
                                                                  "energy_difference", "source_destination_similarity",
                                                                  "boundary_jump", "mean_consumption", "max_consumption",
                                                                  "economic_family_result", "replay_result"],
    }
    with open(REPORTS_DIR / "replay_pilot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(freeze, f, indent=2, ensure_ascii=False, default=str)
    return freeze


# ==========================================================================================
# Orquestacion / CLI
# ==========================================================================================

def run_inventory_only():
    df = build_input_inventory()
    print(f"Inventario escrito ({len(df)} artefactos). Faltantes: {(df['validation_status']=='MISSING').sum()}")
    faltantes = df[df["validation_status"] == "MISSING"]
    if len(faltantes):
        for _, r in faltantes.iterrows():
            print(f"  FALTA: {r['artifact_name']} -> {r['path']}")


MANIFEST_VERSION = 3  # v1: bug de contexto de 7 dias de H (ver manifests/v1_superseded/correction_note.json)
                       # v2: bug de solape cruzado entre duraciones (ver manifests/v2_superseded/correction_note.json)
                       # ambas versiones anteriores quedan preservadas sin sobrescribir


def run_build_manifest(overwrite: bool = False):
    manifest_path = MANIFESTS_DIR / "replay_pilot_manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise SystemExit(f"{manifest_path} ya existe. Usa --overwrite para regenerarlo.")

    build_input_inventory()
    periodo = seleccionar_periodo()
    rng_global = np.random.default_rng(RANDOM_SEED)

    seleccionados_por_duracion = {}
    auditorias = []
    todos_candidatos, todos_rechazados = [], []
    seleccionados_global: list[dict] = []  # acumulado entre duraciones -- evita solapes cruzados (criterio 14/15)
    for i, duration_min in enumerate(DURATIONS_MIN):
        seed_local = np.random.default_rng(RANDOM_SEED + duration_min)
        candidatos, rechazados = generar_candidatos_duracion(duration_min, periodo, seed_local)
        candidatos.insert(0, "duration_minutes_group", duration_min)
        todos_candidatos.append(candidatos)
        todos_rechazados.append(rechazados)
        print(f"[manifest] Duracion {duration_min} min: {len(candidatos)} candidatos validos de {len(candidatos)+len(rechazados)} evaluados")

        seleccionados, auditoria = seleccionar_destinos_duracion(
            candidatos, duration_min, N_DEST_PER_DURATION, rng_global, rotacion_daypart=i,
            ya_seleccionados_global=seleccionados_global)
        seleccionados_por_duracion[duration_min] = seleccionados
        seleccionados_global.extend(seleccionados)
        auditorias.append(auditoria)
        if auditoria["insuficiente"]:
            print(f"[manifest] AVISO: duracion {duration_min} min -- solo {len(seleccionados)}/{N_DEST_PER_DURATION} destinos validos disponibles.")
            print(f"           Motivo: candidatos validos totales = {auditoria['n_candidatos_validos']}. No se relajan criterios silenciosamente.")
        else:
            print(f"[manifest] Duracion {duration_min} min: {len(seleccionados)}/{N_DEST_PER_DURATION} destinos seleccionados "
                  f"(trimestre {auditoria['conteo_trimestre']}, daypart {auditoria['conteo_daypart']}, "
                  f"laborable={auditoria['conteo_weekday']} finde={auditoria['conteo_weekend']})")

    pd.concat(todos_candidatos, ignore_index=True).to_csv(MANIFESTS_DIR / "candidate_destinations.csv", index=False)
    pd.concat(todos_rechazados, ignore_index=True).to_csv(MANIFESTS_DIR / "rejected_candidates.csv", index=False)

    tabla = construir_manifiesto(seleccionados_por_duracion, periodo)
    freeze = congelar_manifiesto(tabla, auditorias, periodo, version=MANIFEST_VERSION)
    print(f"[manifest] Manifiesto congelado: {freeze['n_episodes']} episodios, status={freeze['status']}")
    return freeze


def run_validate_manifest():
    manifest_path = MANIFESTS_DIR / "replay_pilot_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"No existe {manifest_path}. Ejecuta antes --build-manifest.")
    tabla = pd.read_csv(manifest_path, parse_dates=["destination_start", "destination_end", "source_start", "source_end"])

    checks = {
        "n_episodios_maximo_60": len(tabla) <= 60,
        "solo_dos_desplazamientos": set(tabla["shift_type"].unique()) <= {"DAILY", "WEEKLY"},
        "solo_tres_duraciones": set(tabla["duration_minutes"].unique()) <= set(DURATIONS_MIN),
        "origen_precede_destino": (tabla["source_end"] <= tabla["destination_start"]).all(),
        "duracion_origen_igual_destino": ((tabla["destination_end"] - tabla["destination_start"]) ==
                                           (tabla["source_end"] - tabla["source_start"])).all(),
        "hash_presente": tabla["manifest_hash"].notna().all(),
        "episode_id_unico": tabla["episode_id"].is_unique,
    }
    print("Validacion del manifiesto:")
    for k, v in checks.items():
        print(f"  {k}: {'OK' if v else 'FALLA'}")
    if not all(checks.values()):
        raise SystemExit("Validacion del manifiesto FALLIDA.")
    print(f"Manifiesto valido: {len(tabla)} episodios, {tabla['paired_destination_id'].nunique()} destinos emparejados.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--validate-manifest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.inventory_only:
        run_inventory_only()
    elif args.build_manifest:
        run_build_manifest(overwrite=args.overwrite)
    elif args.validate_manifest:
        run_validate_manifest()
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
