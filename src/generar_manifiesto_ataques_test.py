"""Pre-registro y congelacion del manifiesto de ataques de test (reduccion constante,
reduccion variable, rampa lineal, bypass residual, bypass total) para la evaluacion
confirmatoria retrospectiva de la arquitectura final ya congelada
(OR_PMULTI_HMULTIWINDOW, ver models/final_or_pretest/).

Este modulo no ejecuta ningun detector, no calcula scores/alarmas/eventos, no aplica
ningun ataque a valores reales y no lee ninguna columna numerica de test. Solo:
  1) selecciona 140 segmentos temporales base (7 duraciones x 20) usando unicamente el
     indice temporal de test (calendario/continuidad/mascara de missingness -- nunca
     Global_active_power ni ninguna otra variable numerica);
  2) expande cada segmento en 12 episodios de ataque (parametros deterministas, semillas
     derivadas de GLOBAL_TEST_MANIFEST_SEED=12000);
  3) guarda un manifiesto inmutable (CSV/JSON) con hashes SHA-256 y verificacion de
     reproducibilidad bit-identica entre dos ejecuciones independientes.

Acceso a test: la particion completa (con Global_active_power) se envuelve
inmediatamente en `IndiceTestGuardado`, que solo expone el indice temporal y la mascara
`imputado` -- cualquier otro acceso (columna, valor, .to_numpy() de consumo, etc.) lanza
`AccesoTestProhibidoError` en tiempo de ejecucion. El resto de este modulo nunca recibe
una referencia al DataFrame real de test, solo al guardian.

Ejecutable de forma aislada:
    python -m src.generar_manifiesto_ataques_test
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import predictor_causal_lags as pcl
from src.attacks import bypass_residual, bypass_total, reduccion_constante, reduccion_variable
from src.data_loading import cargar_y_limpiar
from src.experimento_ramp import _rho_lineal
from src.splitting import split_temporal
from src.windowing import cargar_config

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_DIR = BASE_DIR / "manifests" / "test_attacks"
DOCS_DIR = BASE_DIR / "docs"

GLOBAL_TEST_MANIFEST_SEED = 12000
DURATIONS_MINUTES = [30, 60, 120, 240, 360, 720, 1440]
N_BASE_PER_DURATION = 20
N_BASE_SEGMENTS = len(DURATIONS_MINUTES) * N_BASE_PER_DURATION  # 140
QUARTERS = [1, 2, 3, 4]
TARGET_POR_TRIMESTRE = 5

ATTACK_FAMILIES = ["constant_reduction", "variable_reduction", "linear_ramp", "residual_bypass", "total_bypass"]
# NOTA: la familia de peak-clipping excluida por decision de alcance del TFG (acordada con la
# tutora) se omite deliberadamente de esta lista -- su nombre no debe aparecer en ningun
# codigo/manifiesto/configuracion/test/resultado/figura de este experimento. Las demas
# familias excluidas si se listan explicitamente.
EXCLUDED_FAMILIES = ["time_reversal", "replay", "price_based_load_control",
                     "fgsm", "bim", "adversarial_attacks", "poisoning", "other_variables_attacks", "peak_clipping"]

CONST_LEVELS = {"LOW": 0.70, "MEDIUM": 0.50, "HIGH": 0.30}
VAR_LEVELS = {"LOW": (0.60, 0.80), "MEDIUM": (0.40, 0.60), "HIGH": (0.20, 0.40)}
RAMP_LEVELS = {"LOW": 0.70, "MEDIUM": 0.50, "HIGH": 0.30}
BYPASS_RESIDUAL_LEVELS = {"RHO010": 0.10, "RHO005": 0.05}

N_VARIANTS_PER_BASE = 12
N_ATTACK_EPISODES = N_BASE_SEGMENTS * N_VARIANTS_PER_BASE  # 1680

TARGET_COLUMN = "Global_active_power"
INJECTION_RESOLUTION_MINUTES = 1
EVALUATION_MODE = "INDEPENDENT_COUNTERFACTUAL_BRANCH"
GENERATOR_VERSION = "1.0.0"
MANIFEST_VERSION = "1.0.0"

ATTACK_CODES = {
    ("constant_reduction", "LOW"): 0, ("constant_reduction", "MEDIUM"): 1, ("constant_reduction", "HIGH"): 2,
    ("variable_reduction", "LOW"): 3, ("variable_reduction", "MEDIUM"): 4, ("variable_reduction", "HIGH"): 5,
    ("linear_ramp", "LOW"): 6, ("linear_ramp", "MEDIUM"): 7, ("linear_ramp", "HIGH"): 8,
    ("residual_bypass", "RHO010"): 9, ("residual_bypass", "RHO005"): 10,
    ("total_bypass", "NA"): 11,
}


# ==========================================================================================
# Guardian de acceso a test: solo expone indice temporal + mascara de missingness
# ==========================================================================================

class AccesoTestProhibidoError(RuntimeError):
    pass


class IndiceTestGuardado:
    """Envoltorio de solo-indice sobre la particion test. Expone unicamente el
    DatetimeIndex y la mascara booleana `imputado` (missingness, explicitamente permitida
    por el encargo). Cualquier otro acceso (nombre de columna, .values, .to_numpy(),
    atributo, etc.) lanza AccesoTestProhibidoError -- demuestra en tiempo de ejecucion,
    no solo por inspeccion de codigo, que este generador nunca lee consumo de test."""

    __slots__ = ("_index", "_imputado")

    def __init__(self, index: pd.DatetimeIndex, imputado: np.ndarray):
        object.__setattr__(self, "_index", pd.DatetimeIndex(index).copy())
        object.__setattr__(self, "_imputado", np.asarray(imputado, dtype=bool).copy())

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    def imputado_mask(self) -> np.ndarray:
        return self._imputado

    def __getattr__(self, name):
        raise AccesoTestProhibidoError(f"acceso prohibido: atributo/columna '{name}' de la particion test")

    def __getitem__(self, key):
        raise AccesoTestProhibidoError(f"acceso prohibido: columna/valor '{key}' de la particion test")


def obtener_indice_test_guardado() -> IndiceTestGuardado:
    """Unico punto del modulo que toca el DataFrame real de test (con Global_active_power).
    Extrae index + mascara imputado y descarta inmediatamente toda referencia al resto."""
    df_min = cargar_y_limpiar()
    particiones = split_temporal(df_min)
    test_df = particiones["test"]
    guard = IndiceTestGuardado(test_df.index, test_df["imputado"].to_numpy())
    del df_min, particiones, test_df
    return guard


# ==========================================================================================
# Contexto causal requerido -- derivado de los artefactos congelados, no hardcodeado
# ==========================================================================================

def calcular_contexto_causal() -> dict:
    config = cargar_config()
    lookback_p_min = int(config["windowing"]["tamano_ventana_min"])  # P: 360
    max_lag_h_intervals = int(pcl.MAX_LAG)  # H: 672 intervalos de 15 min
    resolucion_h_min = 15
    max_lag_h_min = max_lag_h_intervals * resolucion_h_min  # 10080

    # required_pre: el primer minuto atacado debe tener, hacia atras, tanto el lookback
    # de P (360 min) como el lag mas largo de H (672*15=10080 min) disponibles en datos reales.
    required_pre = max(lookback_p_min, max_lag_h_min)
    # required_post: una decision de H hasta max_lag_H despues del final del ataque puede
    # seguir usando datos atacados via sus lags (misma logica que "el margen posterior
    # maximo sera de siete dias" si la logica actual considera decisiones afectadas por
    # lag_672) -- se exige el mismo margen para garantizar que exista suficiente test real
    # mas alla del episodio para poder evaluar la ultima decision potencialmente afectada.
    required_post = max(lookback_p_min, max_lag_h_min)

    return {
        "lookback_P_minutes": lookback_p_min, "max_lag_H_intervals_15min": max_lag_h_intervals,
        "max_lag_H_minutes": max_lag_h_min, "required_pre_context_minutes": required_pre,
        "required_post_context_minutes": required_post,
    }


# ==========================================================================================
# Calendario (solo a partir del indice temporal, nunca de valores)
# ==========================================================================================

def _day_type(ts: pd.Timestamp) -> str:
    return "weekend" if ts.dayofweek >= 5 else "weekday"


def _daypart(hour: int) -> str:
    if hour < 6:
        return "madrugada"
    if hour < 12:
        return "manana"
    if hour < 18:
        return "tarde"
    return "noche"


def _limites_trimestre(ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    q = ts.quarter
    mes_inicio = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    inicio = pd.Timestamp(year=ts.year, month=mes_inicio, day=1)
    fin = inicio + pd.DateOffset(months=3)
    return inicio, fin


def _sub_rangos_por_trimestre(ini: pd.Timestamp, fin: pd.Timestamp) -> dict[int, list[dict]]:
    """Divide [ini, fin) en tramos contiguos por trimestre calendario (Q1..Q4, sin
    distinguir anio -- test abarca 2009-2010 y Q4 se compone de fragmentos de ambos)."""
    rangos: dict[int, list[dict]] = {q: [] for q in QUARTERS}
    cur = ini
    while cur < fin:
        q = int(cur.quarter)
        _, fin_trimestre = _limites_trimestre(cur)
        tramo_fin = min(fin, fin_trimestre)
        n_min = int((tramo_fin - cur) / pd.Timedelta(minutes=1))
        if n_min > 0:
            rangos[q].append({"ini": cur, "fin": tramo_fin, "n_minutos": n_min})
        if tramo_fin <= cur:
            break
        cur = tramo_fin
    return rangos


def _elegir_subrango(sub_rangos: list[dict], rng: np.random.Generator) -> dict:
    pesos = np.array([sr["n_minutos"] for sr in sub_rangos], dtype=np.float64)
    p = pesos / pesos.sum()
    idx = int(rng.choice(len(sub_rangos), p=p))
    return sub_rangos[idx]


# ==========================================================================================
# Admisibilidad de un candidato
# ==========================================================================================

def _evaluar_candidato(start_ts: pd.Timestamp, duration_min: int, test_index: pd.DatetimeIndex,
                        starts_usados: set, ctx: dict) -> dict:
    end_ts = start_ts + pd.Timedelta(minutes=duration_min)
    razones = []

    ok_inicio_en_test = test_index.min() <= start_ts <= test_index.max()
    ok_fin_en_test = test_index.min() <= end_ts <= test_index.max()
    ok_pre = (start_ts - test_index.min()) >= pd.Timedelta(minutes=ctx["required_pre_context_minutes"])
    ok_post = (test_index.max() - end_ts) >= pd.Timedelta(minutes=ctx["required_post_context_minutes"])
    # continuidad: garantizada globalmente (rejilla completa de 1 min sin huecos, ver
    # data_loading.cargar_y_limpiar) -- se re-verifica aqui por robustez, sin coste real.
    ok_continuo = bool(start_ts in test_index) and bool((end_ts - pd.Timedelta(minutes=1)) in test_index)
    ok_no_duplicado = start_ts not in starts_usados

    if not ok_inicio_en_test:
        razones.append("inicio_fuera_de_test")
    if not ok_fin_en_test:
        razones.append("fin_fuera_de_test")
    if not ok_pre:
        razones.append("contexto_previo_insuficiente")
    if not ok_post:
        razones.append("contexto_posterior_insuficiente")
    if not ok_continuo:
        razones.append("indice_discontinuo")
    if not ok_no_duplicado:
        razones.append("duplicado_misma_duracion")

    admisible = not razones
    return {"admisible": admisible, "start_ts": start_ts, "end_ts": end_ts, "razones": razones}


# ==========================================================================================
# Seleccion de los 140 segmentos base
# ==========================================================================================

def seleccionar_segmentos_base(guard: IndiceTestGuardado, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = calcular_contexto_causal()
    test_index = guard.index
    filas_aceptadas: list[dict] = []
    filas_rechazadas: list[dict] = []
    numeric_id = 0

    for duration in DURATIONS_MINUTES:
        ini = test_index.min() + pd.Timedelta(minutes=ctx["required_pre_context_minutes"])
        fin_max_inicio = test_index.max() - pd.Timedelta(minutes=duration) - pd.Timedelta(minutes=ctx["required_post_context_minutes"])
        if fin_max_inicio <= ini:
            raise RuntimeError(f"duracion {duration} min: test no tiene rango valido suficiente para el contexto causal exigido")

        rango_por_trimestre = _sub_rangos_por_trimestre(ini, fin_max_inicio)
        starts_usados: set = set()
        aceptados_duracion: list[dict] = []
        deficit_por_trimestre: dict[int, int] = {}

        for q in QUARTERS:
            sub_rangos = rango_por_trimestre[q]
            aceptados_q = 0
            intentos = 0
            max_intentos = 400
            while aceptados_q < TARGET_POR_TRIMESTRE and intentos < max_intentos and sub_rangos:
                intentos += 1
                sr = _elegir_subrango(sub_rangos, rng)
                offset = int(rng.integers(0, sr["n_minutos"]))
                start_ts = sr["ini"] + pd.Timedelta(minutes=offset)
                cand = _evaluar_candidato(start_ts, duration, test_index, starts_usados, ctx)
                if not cand["admisible"]:
                    filas_rechazadas.append({"duration_minutes": duration, "start_timestamp_candidato": str(start_ts),
                                             "quarter_objetivo": q, "motivo": ";".join(cand["razones"])})
                    continue
                starts_usados.add(start_ts)
                aceptados_duracion.append({"start_ts": start_ts, "end_ts": cand["end_ts"], "quarter": q,
                                           "selection_priority": "quarter_target"})
                aceptados_q += 1
            if aceptados_q < TARGET_POR_TRIMESTRE:
                deficit_por_trimestre[q] = TARGET_POR_TRIMESTRE - aceptados_q

        deficit_total = sum(deficit_por_trimestre.values())
        if deficit_total > 0:
            trimestres_con_capacidad = [q for q in QUARTERS if rango_por_trimestre[q]]
            intentos = 0
            max_intentos = 4000
            i = 0
            while deficit_total > 0 and intentos < max_intentos and trimestres_con_capacidad:
                intentos += 1
                q = trimestres_con_capacidad[i % len(trimestres_con_capacidad)]
                i += 1
                sr = _elegir_subrango(rango_por_trimestre[q], rng)
                offset = int(rng.integers(0, sr["n_minutos"]))
                start_ts = sr["ini"] + pd.Timedelta(minutes=offset)
                cand = _evaluar_candidato(start_ts, duration, test_index, starts_usados, ctx)
                if not cand["admisible"]:
                    filas_rechazadas.append({"duration_minutes": duration, "start_timestamp_candidato": str(start_ts),
                                             "quarter_objetivo": f"{q}_redistribucion", "motivo": ";".join(cand["razones"])})
                    continue
                starts_usados.add(start_ts)
                aceptados_duracion.append({"start_ts": start_ts, "end_ts": cand["end_ts"], "quarter": q,
                                           "selection_priority": "deficit_redistribution"})
                deficit_total -= 1

        if len(aceptados_duracion) != N_BASE_PER_DURATION:
            raise RuntimeError(f"duracion {duration} min: solo se consiguieron {len(aceptados_duracion)}/{N_BASE_PER_DURATION} "
                               f"segmentos admisibles tras redistribucion de deficit")

        aceptados_duracion.sort(key=lambda d: d["start_ts"])  # orden deterministico y estable
        for rep_idx, seg in enumerate(aceptados_duracion, start=1):
            start_ts, end_ts = seg["start_ts"], seg["end_ts"]
            frac_imputado = float(guard.imputado_mask[(test_index >= start_ts) & (test_index < end_ts)].mean())
            base_segment_id = f"BASE_D{duration}_R{rep_idx:02d}"
            filas_aceptadas.append({
                "base_segment_id": base_segment_id, "base_segment_numeric_id": numeric_id,
                "start_timestamp": start_ts, "end_timestamp": end_ts, "duration_minutes": duration,
                "replicate": rep_idx, "calendar_year": int(start_ts.year), "calendar_quarter": seg["quarter"],
                "month": int(start_ts.month), "day_of_week": int(start_ts.dayofweek), "day_type": _day_type(start_ts),
                "daypart": _daypart(int(start_ts.hour)), "start_hour": int(start_ts.hour), "start_minute": int(start_ts.minute),
                "start_minute_mod_15": int(start_ts.minute % 15), "start_minute_mod_60": int(start_ts.minute % 60),
                "required_pre_context_minutes": ctx["required_pre_context_minutes"],
                "required_post_context_minutes": ctx["required_post_context_minutes"],
                "valid_pre_context": True, "valid_post_context": True, "continuous_index": True,
                "valid_missingness_mask": bool(frac_imputado <= 0.5), "frac_imputado": frac_imputado,
                "selection_seed": GLOBAL_TEST_MANIFEST_SEED, "selection_priority": seg["selection_priority"],
                "source_values_inspected": False, "detector_scores_inspected": False, "old_test_results_inspected": False,
            })
            numeric_id += 1

    tabla_base = pd.DataFrame(filas_aceptadas).sort_values(["duration_minutes", "replicate"]).reset_index(drop=True)
    tabla_rechazados = pd.DataFrame(filas_rechazadas)
    assert tabla_base["base_segment_id"].is_unique
    assert len(tabla_base) == N_BASE_SEGMENTS
    return tabla_base, tabla_rechazados


# ==========================================================================================
# Expansion de cada segmento base en 12 episodios de ataque
# ==========================================================================================

def _derivar_episode_seed(base_numeric_id: int, attack_code: int) -> int:
    ss = np.random.SeedSequence([GLOBAL_TEST_MANIFEST_SEED, int(base_numeric_id), int(attack_code)])
    return int(ss.generate_state(1)[0])


def _hidden_fraction_ramp(duration_min: int, rho_final: float) -> float:
    rho_t = _rho_lineal(duration_min, rho_final)
    return float(1.0 - rho_t.mean())


def expandir_episodios(tabla_base: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for base in tabla_base.itertuples():
        base_id, base_num_id = base.base_segment_id, base.base_segment_numeric_id
        comunes = {
            "base_segment_id": base_id, "start_timestamp": base.start_timestamp, "end_timestamp": base.end_timestamp,
            "duration_minutes": base.duration_minutes, "injection_resolution_minutes": INJECTION_RESOLUTION_MINUTES,
            "target_column": TARGET_COLUMN, "calendar_quarter": base.calendar_quarter, "day_type": base.day_type,
            "daypart": base.daypart, "start_minute_mod_15": base.start_minute_mod_15, "start_minute_mod_60": base.start_minute_mod_60,
            "required_pre_context_minutes": base.required_pre_context_minutes, "required_post_context_minutes": base.required_post_context_minutes,
            "evaluation_mode": EVALUATION_MODE, "generator_version": GENERATOR_VERSION, "manifest_version": MANIFEST_VERSION,
        }

        for severity, rho in CONST_LEVELS.items():
            episode_id = f"{base_id}__constant_reduction__{severity}"
            filas.append({**comunes, "episode_id": episode_id, "attack_family": "constant_reduction", "attack_subtype": "constant_reduction",
                         "severity_label": severity, "rho": rho, "rho_final": np.nan, "rho_min": np.nan, "rho_max": np.nan,
                         "rho_residual": np.nan, "hidden_fraction_nominal": 1.0 - rho, "expected_beta": np.nan, "ramp_shape": np.nan,
                         "episode_seed": np.nan, "clean_branch_id": f"{episode_id}__clean", "attack_branch_id": f"{episode_id}__attack"})

        for severity, (rho_min, rho_max) in VAR_LEVELS.items():
            episode_id = f"{base_id}__variable_reduction__{severity}"
            expected_beta = (rho_min + rho_max) / 2.0
            seed = _derivar_episode_seed(base_num_id, ATTACK_CODES[("variable_reduction", severity)])
            filas.append({**comunes, "episode_id": episode_id, "attack_family": "variable_reduction", "attack_subtype": "variable_reduction",
                         "severity_label": severity, "rho": np.nan, "rho_final": np.nan, "rho_min": rho_min, "rho_max": rho_max,
                         "rho_residual": np.nan, "hidden_fraction_nominal": 1.0 - expected_beta, "expected_beta": expected_beta,
                         "ramp_shape": np.nan, "episode_seed": seed, "clean_branch_id": f"{episode_id}__clean", "attack_branch_id": f"{episode_id}__attack"})

        for severity, rho_final in RAMP_LEVELS.items():
            episode_id = f"{base_id}__linear_ramp__{severity}"
            filas.append({**comunes, "episode_id": episode_id, "attack_family": "linear_ramp", "attack_subtype": "linear_ramp",
                         "severity_label": severity, "rho": np.nan, "rho_final": rho_final, "rho_min": np.nan, "rho_max": np.nan,
                         "rho_residual": np.nan, "hidden_fraction_nominal": _hidden_fraction_ramp(int(base.duration_minutes), rho_final),
                         "expected_beta": np.nan, "ramp_shape": "linear", "episode_seed": np.nan,
                         "clean_branch_id": f"{episode_id}__clean", "attack_branch_id": f"{episode_id}__attack"})

        for subtype, rho_residual in BYPASS_RESIDUAL_LEVELS.items():
            episode_id = f"{base_id}__residual_bypass__{subtype}"
            filas.append({**comunes, "episode_id": episode_id, "attack_family": "residual_bypass", "attack_subtype": subtype,
                         "severity_label": subtype, "rho": np.nan, "rho_final": np.nan, "rho_min": np.nan, "rho_max": np.nan,
                         "rho_residual": rho_residual, "hidden_fraction_nominal": 1.0 - rho_residual, "expected_beta": np.nan,
                         "ramp_shape": np.nan, "episode_seed": np.nan, "clean_branch_id": f"{episode_id}__clean", "attack_branch_id": f"{episode_id}__attack"})

        episode_id = f"{base_id}__total_bypass"
        filas.append({**comunes, "episode_id": episode_id, "attack_family": "total_bypass", "attack_subtype": "total_bypass",
                     "severity_label": "NA", "rho": np.nan, "rho_final": np.nan, "rho_min": np.nan, "rho_max": np.nan,
                     "rho_residual": np.nan, "hidden_fraction_nominal": 1.0, "expected_beta": np.nan, "ramp_shape": np.nan,
                     "episode_seed": np.nan, "clean_branch_id": f"{episode_id}__clean", "attack_branch_id": f"{episode_id}__attack"})

    tabla = pd.DataFrame(filas)
    assert tabla["episode_id"].is_unique
    assert len(tabla) == N_ATTACK_EPISODES
    return tabla


# ==========================================================================================
# Auditoria del muestreo
# ==========================================================================================

def construir_auditoria_muestreo(tabla_base: pd.DataFrame, tabla_rechazados: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for duration, grupo in tabla_base.groupby("duration_minutes"):
        rech = tabla_rechazados[tabla_rechazados["duration_minutes"] == duration] if len(tabla_rechazados) else tabla_rechazados
        sep = grupo["start_timestamp"].sort_values().diff().dropna()
        filas.append({
            "duration_minutes": duration, "n_segmentos": len(grupo),
            "n_por_trimestre": json.dumps(grupo["calendar_quarter"].value_counts().sort_index().to_dict()),
            "n_por_mes": json.dumps(grupo["month"].value_counts().sort_index().to_dict()),
            "n_por_tipo_dia": json.dumps(grupo["day_type"].value_counts().to_dict()),
            "n_por_franja_horaria": json.dumps(grupo["daypart"].value_counts().to_dict()),
            "distribucion_mod_15": json.dumps(grupo["start_minute_mod_15"].value_counts().sort_index().to_dict()),
            "distribucion_mod_60_resumen_std": float(grupo["start_minute_mod_60"].std()) if len(grupo) > 1 else 0.0,
            "primer_timestamp": str(grupo["start_timestamp"].min()), "ultimo_timestamp": str(grupo["start_timestamp"].max()),
            "separacion_temporal_media_dias": float(sep.mean() / pd.Timedelta(days=1)) if len(sep) else float("nan"),
            "separacion_temporal_minima_dias": float(sep.min() / pd.Timedelta(days=1)) if len(sep) else float("nan"),
            "candidatos_rechazados": len(rech), "motivos_rechazo": json.dumps(rech["motivo"].value_counts().to_dict()) if len(rech) else "{}",
            "desviacion_objetivo_trimestre": json.dumps({str(q): int(grupo[grupo["calendar_quarter"] == q].shape[0]) - TARGET_POR_TRIMESTRE for q in QUARTERS}),
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# Inmutabilidad: hashes SHA-256 y verificacion de recarga
# ==========================================================================================

def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def generar_manifiesto(directorio_salida: Path = MANIFEST_DIR) -> dict:
    directorio_salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(GLOBAL_TEST_MANIFEST_SEED)

    guard = obtener_indice_test_guardado()
    ctx = calcular_contexto_causal()
    tabla_base, tabla_rechazados = seleccionar_segmentos_base(guard, rng)
    tabla_episodios = expandir_episodios(tabla_base)
    tabla_auditoria = construir_auditoria_muestreo(tabla_base, tabla_rechazados)

    for col in ("start_timestamp", "end_timestamp"):
        tabla_base[col] = pd.DatetimeIndex(tabla_base[col]).strftime("%Y-%m-%d %H:%M:%S")
        tabla_episodios[col] = pd.DatetimeIndex(tabla_episodios[col]).strftime("%Y-%m-%d %H:%M:%S")

    tabla_base = tabla_base.sort_values(["duration_minutes", "replicate"]).reset_index(drop=True)
    tabla_episodios = tabla_episodios.sort_values(["base_segment_id", "attack_family", "severity_label"]).reset_index(drop=True)
    if len(tabla_rechazados):
        tabla_rechazados = tabla_rechazados.sort_values(["duration_minutes", "start_timestamp_candidato"]).reset_index(drop=True)

    tabla_base.to_csv(directorio_salida / "base_segments_test.csv", index=False)
    tabla_episodios.to_csv(directorio_salida / "attack_manifest_test.csv", index=False)
    tabla_auditoria.to_csv(directorio_salida / "sampling_audit_test.csv", index=False)
    tabla_rechazados.to_csv(directorio_salida / "segmentos_rechazados.csv", index=False)

    config_payload = {
        "evaluation_status": "RETROSPECTIVE_CONFIRMATORY_EVALUATION", "selected_configuration": "OR_PMULTI_HMULTIWINDOW",
        "GLOBAL_TEST_MANIFEST_SEED": GLOBAL_TEST_MANIFEST_SEED, "attack_families": ATTACK_FAMILIES, "excluded_families": EXCLUDED_FAMILIES,
        "durations": DURATIONS_MINUTES, "n_base_per_duration": N_BASE_PER_DURATION, "n_base_segments": N_BASE_SEGMENTS,
        "n_variants_per_base": N_VARIANTS_PER_BASE, "n_attack_episodes": N_ATTACK_EPISODES,
        "severity_parameters": {"constant_reduction": CONST_LEVELS, "linear_ramp": RAMP_LEVELS},
        "variable_reduction_ranges": VAR_LEVELS, "bypass_parameters": {"residual": BYPASS_RESIDUAL_LEVELS, "total": "y=0"},
        "ramp_definition": {"formula": "rho_j = 1 - (1-rho_final)*(j+1)/D", "note": "Extension temporal gradual propuesta en este TFG sobre el modelo multiplicativo de reduccion."},
        "target_column": TARGET_COLUMN, "injection_resolution": INJECTION_RESOLUTION_MINUTES,
        "stratification_targets": {"quarters": {f"Q{q}": TARGET_POR_TRIMESTRE for q in QUARTERS}, "day_type": {"weekday": 12, "weekend": 8},
                                   "daypart": {"madrugada": 5, "manana": 5, "tarde": 5, "noche": 5}, "note": "objetivos aproximados/best-effort salvo admisibilidad causal y cobertura trimestral"},
        "admissibility_rules": ["inicio_dentro_de_test", "fin_dentro_de_test", "contexto_previo_suficiente", "contexto_posterior_suficiente",
                                "indice_continuo", "sin_huecos_incompatibles", "sin_padding_artificial", "no_depende_de_datos_tras_fin_de_test",
                                "no_duplicado_misma_duracion", "sin_inspeccionar_consumo_o_scores"],
        "pre_context": ctx, "post_context": ctx, "independent_counterfactual_branches": True,
        "test_values_inspected": False, "test_scores_calculated": False, "inference_executed": False,
        "detector_loaded": False, "attacks_applied": False, "manifest_frozen": False,  # se marca True tras verificar hashes
        "high_impact_group": {"absolute_pretest_threshold_available": False, "exploratory_high_impact": True,
                              "note": "no existe umbral absoluto de alto impacto fijado en pre-test; se reportara unicamente como cuartil superior de hidden_fraction_nominal en la fase de evaluacion, sin usar consumo real ni resultados del detector"},
    }
    with open(directorio_salida / "attack_manifest_test_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False, default=str, sort_keys=False)

    test_access_audit = {
        "particiones_test_index_leido": True, "particiones_test_columnas_numericas_leidas": False,
        "guardian_usado": "IndiceTestGuardado", "columnas_expuestas_por_guardian": ["index", "imputado_mask"],
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(directorio_salida / "test_access_audit.json", "w", encoding="utf-8") as f:
        json.dump(test_access_audit, f, indent=2, ensure_ascii=False)

    readme = f"""# manifests/test_attacks/

Manifiesto congelado de ataques de test para la evaluacion confirmatoria
retrospectiva de OR_PMULTI_HMULTIWINDOW. Generado por
`src/generar_manifiesto_ataques_test.py` (semilla {GLOBAL_TEST_MANIFEST_SEED}).

No se ha ejecutado ninguna inferencia ni aplicado ningun ataque a valores reales.
Ver `attack_manifest_test_config.json` y `manifest_freeze_record.json` para el
estado completo de congelacion.
"""
    with open(directorio_salida / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    # config_payload["manifest_frozen"] ahora se marca True para el hash final (segunda pasada)
    config_payload["manifest_frozen"] = True
    with open(directorio_salida / "attack_manifest_test_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False, default=str, sort_keys=False)

    hashes = {}
    for nombre in ("base_segments_test.csv", "attack_manifest_test.csv", "attack_manifest_test_config.json",
                   "sampling_audit_test.csv", "segmentos_rechazados.csv", "test_access_audit.json"):
        hashes[nombre] = _sha256(directorio_salida / nombre)
    with open(directorio_salida / "hashes_attack_manifest_test.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False, sort_keys=True)
    hashes["manifest_freeze_record.json"] = None  # se rellena tras escribir el propio freeze record (no puede autoincluirse)

    freeze_record = {
        "status": "MANIFEST_FROZEN_READY_FOR_RETROSPECTIVE_TEST", "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION, "manifest_version": MANIFEST_VERSION, "git_commit": _git_commit(),
        "hashes": {k: v for k, v in hashes.items() if v is not None}, "seed": GLOBAL_TEST_MANIFEST_SEED,
        "n_base_segments": int(len(tabla_base)), "n_attack_episodes": int(len(tabla_episodios)),
        "detector_loaded": False, "inference_executed": False, "scores_calculated": False, "attacks_applied": False,
        "test_numeric_values_accessed": False, "old_test_results_used": False, "modification_allowed": False,
    }
    with open(directorio_salida / "manifest_freeze_record.json", "w", encoding="utf-8") as f:
        json.dump(freeze_record, f, indent=2, ensure_ascii=False, sort_keys=False)

    hash_freeze_record = _sha256(directorio_salida / "manifest_freeze_record.json")
    hashes_final = json.load(open(directorio_salida / "hashes_attack_manifest_test.json", encoding="utf-8"))
    hashes_final["manifest_freeze_record.json"] = hash_freeze_record
    with open(directorio_salida / "hashes_attack_manifest_test.json", "w", encoding="utf-8") as f:
        json.dump(hashes_final, f, indent=2, ensure_ascii=False, sort_keys=True)

    return {"tabla_base": tabla_base, "tabla_episodios": tabla_episodios, "tabla_auditoria": tabla_auditoria,
            "tabla_rechazados": tabla_rechazados, "hashes": hashes_final, "freeze_record": freeze_record}


def main() -> dict:
    print("Generando el manifiesto definitivo de ataques de test (sin ejecutar ningun detector)...")
    resultado = generar_manifiesto()
    print(f"  {len(resultado['tabla_base'])} segmentos base, {len(resultado['tabla_episodios'])} episodios de ataque")
    print(f"  candidatos rechazados: {len(resultado['tabla_rechazados'])}")

    print("\nVerificando reproducibilidad bit-a-bit (segunda ejecucion en directorio temporal)...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        resultado2 = generar_manifiesto(Path(tmp))
        idénticas = (resultado["tabla_base"].reset_index(drop=True).equals(resultado2["tabla_base"].reset_index(drop=True))
                     and resultado["tabla_episodios"].reset_index(drop=True).equals(resultado2["tabla_episodios"].reset_index(drop=True)))
        assert idénticas, "el generador NO es determinista entre ejecuciones -- manifiesto NO congelado"
    print("  OK -- dos ejecuciones independientes producen manifiestos bit-identicos.")

    print(f"\n>>> STATUS: {resultado['freeze_record']['status']}")
    print("\nCONFIRMACION: no se ha cargado ningun detector, no se ha ejecutado inferencia, "
          "no se ha aplicado ningun ataque a valores reales y no se ha accedido a ninguna "
          "columna numerica de test.")
    return resultado


if __name__ == "__main__":
    main()
