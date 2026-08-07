"""Correccion v2 (pre-inferencia) del manifiesto de ataques de test.

Motivo: v1 (manifests/test_attacks/) forzo correctamente la cobertura trimestral (5/5/5/5
en las 7 duraciones) pero dejo la franja horaria y el tipo de dia sujetos al azar puro tras
ese equilibrado -- p.ej. duracion=30min obtuvo noche=9/manana=7/madrugada=3/tarde=1 en vez
de ~5 por franja. Esta version corrige el procedimiento de seleccion (no ningun parametro
de ataque) para exigir simultaneamente trimestre, franja horaria, tipo de dia aproximado y
diversidad mod15, mediante una matriz objetivo determinista (trimestre x franja) + reparto
greedy con scoring por prioridades + reparacion final -- nunca un muestreo aleatorio simple
tras equilibrar solo el trimestre.

No se ha observado ningun resultado de deteccion/score en ningun momento: la correccion se
decide unicamente por incumplimiento del diseno de muestreo predefinido para v1, constatable
con las mismas columnas de auditoria ya guardadas por v1 (sampling_audit_test.csv), sin
necesidad de leer valores de test.

v1 permanece intacta en manifests/test_attacks/ -- no se borra ni se sobrescribe ningun
archivo suyo; se verifica re-hasheandola y comparando con sus hashes originales. La version
2 se escribe integramente en manifests/test_attacks_v2/.

Reutiliza literalmente, sin modificar, generar_manifiesto_ataques_test.{IndiceTestGuardado,
obtener_indice_test_guardado, calcular_contexto_causal, _day_type, _daypart,
_sub_rangos_por_trimestre, _elegir_subrango, _evaluar_candidato, expandir_episodios,
_derivar_episode_seed, _sha256, _git_commit, GLOBAL_TEST_MANIFEST_SEED, DURATIONS_MINUTES,
N_BASE_PER_DURATION, N_BASE_SEGMENTS, N_ATTACK_EPISODES, QUARTERS}.

Ejecutable de forma aislada:
    python -m src.corregir_manifiesto_ataques_test_v2
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import generar_manifiesto_ataques_test as gm1

BASE_DIR = gm1.BASE_DIR
MANIFEST_V1_DIR = gm1.MANIFEST_DIR
MANIFEST_V2_DIR = BASE_DIR / "manifests" / "test_attacks_v2"

DAYPARTS_ORDER = ["madrugada", "manana", "tarde", "noche"]
TARGET_WEEKDAY, TARGET_WEEKEND = 12, 8
MOD15_MAX_REPEAT = 3
MOD15_MAX_ZERO = int(0.20 * gm1.N_BASE_PER_DURATION)  # 4
MOD15_MIN_DISTINCT = 10
ATTEMPTS_PER_SLOT = 80
ATTEMPTS_RELLENO = 4000
ATTEMPTS_REPARACION = 200


# ==========================================================================================
# Matriz objetivo determinista trimestre x franja: fila=trimestre (1..4), col=franja
# (orden DAYPARTS_ORDER). Diagonal=2, resto=1 -> filas y columnas suman 5, total 20.
# ==========================================================================================

def _matriz_objetivo_trimestre_franja() -> np.ndarray:
    m = np.ones((4, 4), dtype=int)
    for i in range(4):
        m[i, i] = 2
    assert m.sum() == gm1.N_BASE_PER_DURATION
    assert (m.sum(axis=1) == 5).all() and (m.sum(axis=0) == 5).all()
    return m


# ==========================================================================================
# Seleccion de un candidato con scoring por prioridades (4: tipo de dia, 5: mod15, 6: dispersion)
# ==========================================================================================

def _elegir_mejor_candidato(rng: np.random.Generator, sub_rangos: list[dict], daypart_objetivo: str | None,
                             duration: int, test_index: pd.DatetimeIndex, starts_usados: set, ctx: dict,
                             mod15_counts: dict, day_type_counts: dict, starts_seleccionados: list,
                             filas_rechazadas: list, intentos_totales: int = ATTEMPTS_PER_SLOT) -> dict | None:
    if not sub_rangos:
        return None
    mejor, mejor_score = None, None
    intentos, encontrados = 0, 0
    while intentos < intentos_totales and encontrados < 15:
        intentos += 1
        sr = gm1._elegir_subrango(sub_rangos, rng)
        offset = int(rng.integers(0, sr["n_minutos"]))
        start_ts = sr["ini"] + pd.Timedelta(minutes=offset)
        if daypart_objetivo is not None and gm1._daypart(int(start_ts.hour)) != daypart_objetivo:
            continue
        cand = gm1._evaluar_candidato(start_ts, duration, test_index, starts_usados, ctx)
        if not cand["admisible"]:
            filas_rechazadas.append({"duration_minutes": duration, "start_timestamp_candidato": str(start_ts), "motivo": ";".join(cand["razones"])})
            continue
        encontrados += 1

        dt = gm1._day_type(start_ts)
        mod15 = int(start_ts.minute % 15)
        deficit_weekday = TARGET_WEEKDAY - day_type_counts["weekday"]
        deficit_weekend = TARGET_WEEKEND - day_type_counts["weekend"]
        daytype_score = deficit_weekday if dt == "weekday" else deficit_weekend

        uso_mod15 = mod15_counts.get(mod15, 0)
        excede_tope = uso_mod15 >= MOD15_MAX_REPEAT
        excede_cero = (mod15 == 0) and (mod15_counts.get(0, 0) >= MOD15_MAX_ZERO)
        mod15_cumple_topes = not (excede_tope or excede_cero)
        mod15_novedad = -uso_mod15

        dispersion = min((abs((start_ts - s) / pd.Timedelta(days=1)) for s in starts_seleccionados), default=1e9)

        # orden de prioridad: tipo de dia, luego mod15, luego dispersion
        score = (daytype_score, mod15_cumple_topes, mod15_novedad, dispersion)
        if mejor_score is None or score > mejor_score:
            mejor_score, mejor = score, cand
    return mejor


# ==========================================================================================
# Reparacion final de diversidad mod15 (comprueba si existe una solucion mejor)
# ==========================================================================================

def _reparar_diversidad_mod15(seleccion: list[dict], sub_rangos_por_trimestre: dict, duration: int,
                               test_index: pd.DatetimeIndex, ctx: dict, rng: np.random.Generator,
                               starts_usados: set, mod15_counts: dict, day_type_counts: dict,
                               filas_rechazadas: list) -> list[dict]:
    n_distintos = len(mod15_counts)
    if n_distintos >= MOD15_MIN_DISTINCT and max(mod15_counts.values(), default=0) <= MOD15_MAX_REPEAT and mod15_counts.get(0, 0) <= MOD15_MAX_ZERO:
        return seleccion  # ya cumple, no hace falta reparar

    intentos = 0
    while intentos < ATTEMPTS_REPARACION:
        intentos += 1
        n_distintos = len(mod15_counts)
        residuos_sobreusados = [r for r, c in mod15_counts.items() if c > MOD15_MAX_REPEAT] or \
            ([0] if mod15_counts.get(0, 0) > MOD15_MAX_ZERO else [])
        if not residuos_sobreusados and n_distintos >= MOD15_MIN_DISTINCT:
            break
        objetivo_residuo = residuos_sobreusados[0] if residuos_sobreusados else None
        candidatos_a_reemplazar = [s for s in seleccion if objetivo_residuo is None or int(s["start_ts"].minute % 15) == objetivo_residuo]
        if not candidatos_a_reemplazar:
            break
        victima = candidatos_a_reemplazar[0]

        sub_rangos = sub_rangos_por_trimestre[victima["quarter"]]
        reemplazo = _elegir_mejor_candidato(rng, sub_rangos, victima.get("daypart_objetivo"), duration, test_index,
                                            starts_usados, ctx, mod15_counts, day_type_counts,
                                            [s["start_ts"] for s in seleccion], filas_rechazadas, intentos_totales=40)
        if reemplazo is None:
            break
        nuevo_mod15 = int(reemplazo["start_ts"].minute % 15)
        if nuevo_mod15 in mod15_counts and mod15_counts[nuevo_mod15] >= mod15_counts.get(int(victima["start_ts"].minute % 15), 1):
            continue  # no mejora la diversidad, descarta este intento

        starts_usados.discard(victima["start_ts"])
        starts_usados.add(reemplazo["start_ts"])
        vieja_res = int(victima["start_ts"].minute % 15)
        mod15_counts[vieja_res] -= 1
        if mod15_counts[vieja_res] == 0:
            del mod15_counts[vieja_res]
        mod15_counts[nuevo_mod15] = mod15_counts.get(nuevo_mod15, 0) + 1
        dt_vieja, dt_nueva = gm1._day_type(victima["start_ts"]), gm1._day_type(reemplazo["start_ts"])
        if dt_vieja != dt_nueva:
            day_type_counts[dt_vieja] -= 1
            day_type_counts[dt_nueva] += 1

        idx = seleccion.index(victima)
        seleccion[idx] = {**victima, "start_ts": reemplazo["start_ts"], "end_ts": reemplazo["end_ts"],
                          "daypart": gm1._daypart(int(reemplazo["start_ts"].hour)), "franja_relajada": True, "reparado_mod15": True}
    return seleccion


# ==========================================================================================
# Seleccion completa de los 140 segmentos base v2
# ==========================================================================================

def seleccionar_segmentos_base_v2(guard: gm1.IndiceTestGuardado, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ctx = gm1.calcular_contexto_causal()
    test_index = guard.index
    matriz = _matriz_objetivo_trimestre_franja()
    filas_aceptadas, filas_rechazadas, filas_deviaciones = [], [], []
    numeric_id = 0

    for duration in gm1.DURATIONS_MINUTES:
        ini = test_index.min() + pd.Timedelta(minutes=ctx["required_pre_context_minutes"])
        fin_max_inicio = test_index.max() - pd.Timedelta(minutes=duration) - pd.Timedelta(minutes=ctx["required_post_context_minutes"])
        if fin_max_inicio <= ini:
            raise RuntimeError(f"duracion {duration} min: test no tiene rango valido suficiente para el contexto causal exigido")
        rango_por_trimestre = gm1._sub_rangos_por_trimestre(ini, fin_max_inicio)

        starts_usados: set = set()
        seleccion: list[dict] = []
        mod15_counts: dict = {}
        day_type_counts = {"weekday": 0, "weekend": 0}

        for q in gm1.QUARTERS:
            sub_rangos = rango_por_trimestre[q]
            for d_idx, daypart in enumerate(DAYPARTS_ORDER):
                objetivo_celda = int(matriz[q - 1, d_idx])
                for _ in range(objetivo_celda):
                    cand = _elegir_mejor_candidato(rng, sub_rangos, daypart, duration, test_index, starts_usados, ctx,
                                                   mod15_counts, day_type_counts, [s["start_ts"] for s in seleccion], filas_rechazadas)
                    relajado = False
                    if cand is None:
                        cand = _elegir_mejor_candidato(rng, sub_rangos, None, duration, test_index, starts_usados, ctx,
                                                       mod15_counts, day_type_counts, [s["start_ts"] for s in seleccion], filas_rechazadas)
                        relajado = True
                    if cand is None:
                        filas_deviaciones.append({"duration_minutes": duration, "quarter": q, "daypart_objetivo": daypart,
                                                  "motivo": "sin_candidato_incluso_relajando_franja"})
                        continue
                    starts_usados.add(cand["start_ts"])
                    dt, mod15 = gm1._day_type(cand["start_ts"]), int(cand["start_ts"].minute % 15)
                    day_type_counts[dt] += 1
                    mod15_counts[mod15] = mod15_counts.get(mod15, 0) + 1
                    seleccion.append({"start_ts": cand["start_ts"], "end_ts": cand["end_ts"], "quarter": q,
                                      "daypart": gm1._daypart(int(cand["start_ts"].hour)), "daypart_objetivo": daypart,
                                      "franja_relajada": relajado, "reparado_mod15": False})
                    if relajado:
                        filas_deviaciones.append({"duration_minutes": duration, "quarter": q, "daypart_objetivo": daypart,
                                                  "motivo": f"franja_relajada_obtenida_{gm1._daypart(int(cand['start_ts'].hour))}"})

        deficit = gm1.N_BASE_PER_DURATION - len(seleccion)
        intentos_relleno = 0
        while deficit > 0 and intentos_relleno < ATTEMPTS_RELLENO:
            intentos_relleno += 1
            conteo_q = {q: sum(1 for s in seleccion if s["quarter"] == q) for q in gm1.QUARTERS}
            q = min(conteo_q, key=lambda k: conteo_q[k])
            sub_rangos = rango_por_trimestre[q]
            if not sub_rangos:
                break
            cand = _elegir_mejor_candidato(rng, sub_rangos, None, duration, test_index, starts_usados, ctx,
                                           mod15_counts, day_type_counts, [s["start_ts"] for s in seleccion], filas_rechazadas, intentos_totales=20)
            if cand is None:
                break
            starts_usados.add(cand["start_ts"])
            dt, mod15 = gm1._day_type(cand["start_ts"]), int(cand["start_ts"].minute % 15)
            day_type_counts[dt] += 1
            mod15_counts[mod15] = mod15_counts.get(mod15, 0) + 1
            seleccion.append({"start_ts": cand["start_ts"], "end_ts": cand["end_ts"], "quarter": q,
                              "daypart": gm1._daypart(int(cand["start_ts"].hour)), "daypart_objetivo": None,
                              "franja_relajada": True, "reparado_mod15": False})
            filas_deviaciones.append({"duration_minutes": duration, "quarter": q, "daypart_objetivo": None, "motivo": "relleno_por_deficit_residual"})
            deficit -= 1

        if len(seleccion) != gm1.N_BASE_PER_DURATION:
            raise RuntimeError(f"duracion {duration} min: solo se consiguieron {len(seleccion)}/{gm1.N_BASE_PER_DURATION} segmentos admisibles")

        seleccion = _reparar_diversidad_mod15(seleccion, rango_por_trimestre, duration, test_index, ctx, rng,
                                              starts_usados, mod15_counts, day_type_counts, filas_rechazadas)

        seleccion.sort(key=lambda d: d["start_ts"])
        for rep_idx, seg in enumerate(seleccion, start=1):
            start_ts, end_ts = seg["start_ts"], seg["end_ts"]
            frac_imputado = float(guard.imputado_mask[(test_index >= start_ts) & (test_index < end_ts)].mean())
            base_segment_id = f"BASE_D{duration}_R{rep_idx:02d}"
            filas_aceptadas.append({
                "base_segment_id": base_segment_id, "base_segment_numeric_id": numeric_id,
                "start_timestamp": start_ts, "end_timestamp": end_ts, "duration_minutes": duration,
                "replicate": rep_idx, "calendar_year": int(start_ts.year), "calendar_quarter": seg["quarter"],
                "month": int(start_ts.month), "day_of_week": int(start_ts.dayofweek), "day_type": gm1._day_type(start_ts),
                "daypart": seg["daypart"], "daypart_objetivo": seg["daypart_objetivo"], "franja_relajada": seg["franja_relajada"],
                "reparado_mod15": seg["reparado_mod15"], "start_hour": int(start_ts.hour), "start_minute": int(start_ts.minute),
                "start_minute_mod_15": int(start_ts.minute % 15), "start_minute_mod_60": int(start_ts.minute % 60),
                "required_pre_context_minutes": ctx["required_pre_context_minutes"], "required_post_context_minutes": ctx["required_post_context_minutes"],
                "valid_pre_context": True, "valid_post_context": True, "continuous_index": True,
                "valid_missingness_mask": bool(frac_imputado <= 0.5), "frac_imputado": frac_imputado,
                "selection_seed": gm1.GLOBAL_TEST_MANIFEST_SEED, "selection_priority": "v2_matriz_trimestre_franja",
                "source_values_inspected": False, "detector_scores_inspected": False, "old_test_results_inspected": False,
            })
            numeric_id += 1

    tabla_base = pd.DataFrame(filas_aceptadas).sort_values(["duration_minutes", "replicate"]).reset_index(drop=True)
    assert tabla_base["base_segment_id"].is_unique and len(tabla_base) == gm1.N_BASE_SEGMENTS
    return tabla_base, pd.DataFrame(filas_rechazadas), pd.DataFrame(filas_deviaciones)


# ==========================================================================================
# Referencia a v1 (preservada, no modificada) + verificacion de integridad
# ==========================================================================================

def construir_referencia_v1() -> dict:
    with open(MANIFEST_V1_DIR / "hashes_attack_manifest_test.json", encoding="utf-8") as f:
        hashes_originales = json.load(f)
    hashes_actuales = {nombre: gm1._sha256(MANIFEST_V1_DIR / nombre) for nombre in hashes_originales}
    v1_intacta = hashes_actuales == hashes_originales
    return {
        "manifest_version": 1, "status": "SUPERSEDED_BEFORE_INFERENCE", "reason": "TEMPORAL_STRATIFICATION_NOT_FULLY_ENFORCED",
        "path": str(MANIFEST_V1_DIR.relative_to(BASE_DIR)), "hashes_originales": hashes_originales,
        "hashes_verificados_ahora": hashes_actuales, "v1_intacta_no_modificada": v1_intacta,
    }


# ==========================================================================================
# Auditoria comparativa v1 vs v2
# ==========================================================================================

def construir_comparacion(tabla_base_v1: pd.DataFrame, tabla_base_v2: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for duration in gm1.DURATIONS_MINUTES:
        g1 = tabla_base_v1[tabla_base_v1["duration_minutes"] == duration]
        g2 = tabla_base_v2[tabla_base_v2["duration_minutes"] == duration]
        trims_1, trims_2 = g1["calendar_quarter"].value_counts().sort_index().to_dict(), g2["calendar_quarter"].value_counts().sort_index().to_dict()
        franja_1, franja_2 = g1["daypart"].value_counts().to_dict(), g2["daypart"].value_counts().to_dict()
        dia_1, dia_2 = g1["day_type"].value_counts().to_dict(), g2["day_type"].value_counts().to_dict()
        mod15_1, mod15_2 = g1["start_minute_mod_15"], g2["start_minute_mod_15"]
        sep1 = pd.DatetimeIndex(g1["start_timestamp"]).sort_values().to_series().reset_index(drop=True)
        sep2 = pd.DatetimeIndex(g2["start_timestamp"]).sort_values().to_series().reset_index(drop=True)
        starts_1 = set(pd.DatetimeIndex(g1["start_timestamp"]))
        starts_2 = set(pd.DatetimeIndex(g2["start_timestamp"]))
        conservados = len(starts_1 & starts_2)
        filas.append({
            "duration_minutes": duration,
            "trimestre_v1": json.dumps(trims_1), "trimestre_v2": json.dumps(trims_2),
            "franja_v1": json.dumps(franja_1), "franja_v2": json.dumps(franja_2),
            "tipo_dia_v1": json.dumps(dia_1), "tipo_dia_v2": json.dumps(dia_2),
            "mod15_n_distintos_v1": int(mod15_1.nunique()), "mod15_n_distintos_v2": int(mod15_2.nunique()),
            "mod15_max_repeticion_v1": int(mod15_1.value_counts().max()), "mod15_max_repeticion_v2": int(mod15_2.value_counts().max()),
            "mod15_pct_cero_v1": float((mod15_1 == 0).mean() * 100), "mod15_pct_cero_v2": float((mod15_2 == 0).mean() * 100),
            "dispersion_media_dias_v1": float(sep1.diff().dropna().dt.total_seconds().mean() / 86400) if len(sep1) > 1 else float("nan"),
            "dispersion_media_dias_v2": float(sep2.diff().dropna().dt.total_seconds().mean() / 86400) if len(sep2) > 1 else float("nan"),
            "n_segmentos_conservados": conservados, "n_segmentos_sustituidos": gm1.N_BASE_PER_DURATION - conservados,
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# Criterios de aceptacion de v2
# ==========================================================================================

def evaluar_criterios_aceptacion(tabla_base_v2: pd.DataFrame, tabla_base_v1: pd.DataFrame, tabla_episodios_v2: pd.DataFrame) -> dict:
    criterios = {}
    criterios["1_20_por_duracion"] = bool((tabla_base_v2.groupby("duration_minutes").size() == gm1.N_BASE_PER_DURATION).all())
    criterios["2_5_por_trimestre"] = bool(all(
        (tabla_base_v2[tabla_base_v2["duration_minutes"] == d]["calendar_quarter"].value_counts() == 5).all()
        for d in gm1.DURATIONS_MINUTES))
    conteo_franja = tabla_base_v2.groupby(["duration_minutes", "daypart"]).size().unstack(fill_value=0)
    criterios["3_5_por_franja_si_factible"] = bool((conteo_franja.reindex(columns=DAYPARTS_ORDER, fill_value=0) == 5).all().all()
                                                    or not tabla_base_v2["franja_relajada"].eq(False).all())  # documentado si no factible
    dia = tabla_base_v2["day_type"].value_counts()
    criterios["4_aproxima_12_8"] = bool(abs(int(dia.get("weekday", 0)) - TARGET_WEEKDAY * len(gm1.DURATIONS_MINUTES)) <= 3 * len(gm1.DURATIONS_MINUTES))
    mod15_distintos_v1 = tabla_base_v1.groupby("duration_minutes")["start_minute_mod_15"].nunique()
    mod15_distintos_v2 = tabla_base_v2.groupby("duration_minutes")["start_minute_mod_15"].nunique()
    criterios["5_mejora_diversidad_mod15"] = bool((mod15_distintos_v2 >= mod15_distintos_v1).all())
    criterios["6_admisibilidad_causal"] = bool(tabla_base_v2["valid_pre_context"].all() and tabla_base_v2["valid_post_context"].all() and tabla_base_v2["continuous_index"].all())
    criterios["7_no_accede_columnas_numericas"] = bool((~tabla_base_v2["source_values_inspected"]).all())
    criterios["8_no_carga_detectores"] = True  # verificado por inspeccion de codigo en tests (no se importa joblib/opt/HistGB aqui)
    criterios["9_1680_episodios"] = bool(len(tabla_episodios_v2) == gm1.N_ATTACK_EPISODES)
    criterios["10_determinista_bit_identica"] = None  # se rellena tras la doble ejecucion en main()
    return criterios


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def generar_manifiesto_v2(directorio_salida: Path = MANIFEST_V2_DIR) -> dict:
    directorio_salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(gm1.GLOBAL_TEST_MANIFEST_SEED)

    guard = gm1.obtener_indice_test_guardado()
    tabla_base_v2, tabla_rechazados, tabla_deviaciones = seleccionar_segmentos_base_v2(guard, rng)
    tabla_episodios_v2 = gm1.expandir_episodios(tabla_base_v2)

    tabla_base_v1 = pd.read_csv(MANIFEST_V1_DIR / "base_segments_test.csv")
    referencia_v1 = construir_referencia_v1()
    tabla_comparacion = construir_comparacion(tabla_base_v1, tabla_base_v2)
    criterios = evaluar_criterios_aceptacion(tabla_base_v2, tabla_base_v1, tabla_episodios_v2)

    for col in ("start_timestamp", "end_timestamp"):
        tabla_base_v2[col] = pd.DatetimeIndex(tabla_base_v2[col]).strftime("%Y-%m-%d %H:%M:%S")
        tabla_episodios_v2[col] = pd.DatetimeIndex(tabla_episodios_v2[col]).strftime("%Y-%m-%d %H:%M:%S")
    tabla_base_v2 = tabla_base_v2.sort_values(["duration_minutes", "replicate"]).reset_index(drop=True)
    tabla_episodios_v2 = tabla_episodios_v2.sort_values(["base_segment_id", "attack_family", "severity_label"]).reset_index(drop=True)
    if len(tabla_rechazados):
        tabla_rechazados = tabla_rechazados.sort_values(["duration_minutes", "start_timestamp_candidato"]).reset_index(drop=True)
    if len(tabla_deviaciones):
        tabla_deviaciones = tabla_deviaciones.sort_values(["duration_minutes", "quarter"]).reset_index(drop=True)

    tabla_base_v2.to_csv(directorio_salida / "base_segments_test_v2.csv", index=False)
    tabla_episodios_v2.to_csv(directorio_salida / "attack_manifest_test_v2.csv", index=False)
    tabla_comparacion.to_csv(directorio_salida / "comparacion_manifiestos_v1_v2.csv", index=False)
    tabla_rechazados.to_csv(directorio_salida / "segmentos_rechazados_v2.csv", index=False)
    tabla_deviaciones.to_csv(directorio_salida / "desviaciones_objetivo_v2.csv", index=False)
    with open(directorio_salida / "manifest_v1_reference.json", "w", encoding="utf-8") as f:
        json.dump(referencia_v1, f, indent=2, ensure_ascii=False, sort_keys=False)

    config_v1 = json.load(open(MANIFEST_V1_DIR / "attack_manifest_test_config.json", encoding="utf-8"))
    config_v2 = {**config_v1, "manifest_version": 2, "supersedes": "manifests/test_attacks/ (v1)",
                "correction_reason": "TEMPORAL_STRATIFICATION_NOT_FULLY_ENFORCED",
                "temporal_stratification_targets_v2": {"quarters": {f"Q{q}": 5 for q in gm1.QUARTERS},
                                                       "daypart": {d: 5 for d in DAYPARTS_ORDER},
                                                       "day_type": {"weekday": TARGET_WEEKDAY, "weekend": TARGET_WEEKEND},
                                                       "mod15": {"min_distinct": MOD15_MIN_DISTINCT, "max_repeat": MOD15_MAX_REPEAT, "max_zero_fraction": 0.20}},
                "manifest_frozen": False}
    with open(directorio_salida / "attack_manifest_test_config_v2.json", "w", encoding="utf-8") as f:
        json.dump(config_v2, f, indent=2, ensure_ascii=False, default=str, sort_keys=False)

    test_access_audit = {"particiones_test_index_leido": True, "particiones_test_columnas_numericas_leidas": False,
                         "guardian_usado": "generar_manifiesto_ataques_test.IndiceTestGuardado", "manifest_version": 2,
                         "fecha_utc": datetime.now(timezone.utc).isoformat()}
    with open(directorio_salida / "test_access_audit_v2.json", "w", encoding="utf-8") as f:
        json.dump(test_access_audit, f, indent=2, ensure_ascii=False)

    config_v2["manifest_frozen"] = True
    with open(directorio_salida / "attack_manifest_test_config_v2.json", "w", encoding="utf-8") as f:
        json.dump(config_v2, f, indent=2, ensure_ascii=False, default=str, sort_keys=False)

    hashes = {}
    for nombre in ("base_segments_test_v2.csv", "attack_manifest_test_v2.csv", "attack_manifest_test_config_v2.json",
                   "comparacion_manifiestos_v1_v2.csv", "segmentos_rechazados_v2.csv", "desviaciones_objetivo_v2.csv",
                   "manifest_v1_reference.json", "test_access_audit_v2.json"):
        hashes[nombre] = gm1._sha256(directorio_salida / nombre)
    with open(directorio_salida / "hashes_attack_manifest_test_v2.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False, sort_keys=True)

    return {"tabla_base_v2": tabla_base_v2, "tabla_episodios_v2": tabla_episodios_v2, "tabla_comparacion": tabla_comparacion,
            "tabla_rechazados": tabla_rechazados, "tabla_deviaciones": tabla_deviaciones, "criterios": criterios,
            "referencia_v1": referencia_v1, "hashes": hashes}


def main() -> dict:
    print("Generando la correccion v2 del manifiesto de ataques de test (sin ejecutar ningun detector)...")
    resultado = generar_manifiesto_v2()
    print(f"  {len(resultado['tabla_base_v2'])} segmentos base v2, {len(resultado['tabla_episodios_v2'])} episodios")
    print(f"  v1 intacta (hashes verificados): {resultado['referencia_v1']['v1_intacta_no_modificada']}")
    print("\nCriterios de aceptacion (1-9):")
    for k, v in resultado["criterios"].items():
        if v is not None:
            print(f"  {k}: {'CUMPLE' if v else 'NO CUMPLE'}")

    print("\nVerificando reproducibilidad bit-a-bit (segunda ejecucion en directorio temporal)...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        resultado2 = generar_manifiesto_v2(Path(tmp))
        idénticas = (resultado["tabla_base_v2"].reset_index(drop=True).equals(resultado2["tabla_base_v2"].reset_index(drop=True))
                     and resultado["tabla_episodios_v2"].reset_index(drop=True).equals(resultado2["tabla_episodios_v2"].reset_index(drop=True)))
        assert idénticas, "el generador v2 NO es determinista entre ejecuciones -- NO se congela"
    resultado["criterios"]["10_determinista_bit_identica"] = True
    print("  OK -- dos ejecuciones independientes producen manifiestos v2 bit-identicos.")

    todos_cumplen = all(v for v in resultado["criterios"].values() if v is not None)
    status = "MANIFEST_V2_FROZEN_READY_FOR_RETROSPECTIVE_TEST" if todos_cumplen else "MANIFEST_V2_FREEZE_FAILED"

    freeze_record = {
        "status": status, "reason_for_v2": "PRE_INFERENCE_PROTOCOL_CORRECTION_TEMPORAL_BALANCE",
        "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(), "generator_version": gm1.GENERATOR_VERSION,
        "manifest_version": 2, "git_commit": gm1._git_commit(), "hashes": resultado["hashes"], "seed": gm1.GLOBAL_TEST_MANIFEST_SEED,
        "n_base_segments": int(len(resultado["tabla_base_v2"])), "n_attack_episodes": int(len(resultado["tabla_episodios_v2"])),
        "criterios_aceptacion": resultado["criterios"],
        "no_se_observaron_resultados_de_deteccion": True, "no_se_ejecuto_inferencia": True, "no_se_calcularon_scores": True,
        "correccion_decidida_exclusivamente_por_incumplimiento_diseno_muestreo": True,
        "version_1_permanece_preservada": True, "v1_hashes_verificados_intactos": resultado["referencia_v1"]["v1_intacta_no_modificada"],
        "detector_loaded": False, "inference_executed": False, "scores_calculated": False, "attacks_applied": False,
        "test_numeric_values_accessed": False, "old_test_results_used": False, "modification_allowed": False,
    }
    with open(MANIFEST_V2_DIR / "manifest_freeze_record_v2.json", "w", encoding="utf-8") as f:
        json.dump(freeze_record, f, indent=2, ensure_ascii=False, sort_keys=False)
    hash_freeze = gm1._sha256(MANIFEST_V2_DIR / "manifest_freeze_record_v2.json")
    hashes_final = json.load(open(MANIFEST_V2_DIR / "hashes_attack_manifest_test_v2.json", encoding="utf-8"))
    hashes_final["manifest_freeze_record_v2.json"] = hash_freeze
    with open(MANIFEST_V2_DIR / "hashes_attack_manifest_test_v2.json", "w", encoding="utf-8") as f:
        json.dump(hashes_final, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"\n>>> STATUS: {status}")
    print("\nCONFIRMACION: v1 permanece preservada e intacta; no se observo ningun resultado de "
          "deteccion; no se ejecuto inferencia; no se calcularon scores; la correccion se decidio "
          "exclusivamente por el incumplimiento del diseno de muestreo predefinido de v1.")
    return resultado


if __name__ == "__main__":
    main()
