"""Evaluacion sobre serie continua (`continuous_timeline`) frente al evaluador original
(`legacy_aligned_window`); dejamos los dos disponibles para comparar. El modo legacy solo
puntua UNA ventana por episodio, obligada a contener el ataque completo (nunca cruza un
limite de ventana, y descarta duraciones mayores que la ventana sin dejar rastro).
`continuous_timeline` lo corrige: coloca el ataque en cualquier punto de la particion,
deja que cruce limites de ventana, y puntua todas las ventanas que solapan el ataque.

Contrafactual: para cada ventana afectada tambien puntuamos la MISMA ventana sin atacar
(nunca reconstruida). Asi distinguimos una alarma provocada por el ataque
(`attack_induced_alarm`) de una que ya saltaba en limpio (`preexisting_alarm`).
`raw_detected` cuenta cualquier ventana atacada por encima del umbral; `induced_detected`
solo cuenta si esa ventana no alarmaba ya en limpio.

Reutiliza tal cual evaluation.py, attacks.py, anomaly_score.py, thresholding.py, tcn_ae.py
y windowing.py (mismo modelo, mismo umbral, mismos ataques); los combos de ataque y la
funcion que los aplica se importan de evaluation.py para no duplicar logica.

`_generar_episodios` es el unico generador determinista (mismo seed, misma secuencia).
`tabla_master` lo usa para construir episodes_master.csv. `puntuar_episodios` reutiliza
esa misma lista en memoria (no relee el CSV, porque reduccion_variable tiene aleatoriedad
interna) para que otro modelo o score pueda reusar los mismos episodios sin volver a
sortear nada.

Ejecutable de forma aislada (experimento contrafactual sobre val, 50 repeticiones):
    python -m src.episodes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly_score import calcular_scores
from src.data_loading import COLUMNA_OBJETIVO
from src.evaluation import _aplicar_ataque, _combos_ataque
from src.normalization import aplicar_zscore
from src.thresholding import calcular_umbral

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables"

MODOS = ("legacy_aligned_window", "continuous_timeline")

# Intervalo semiabierto [attack_start, attack_end) para contar/inyectar muestras (attack_end
# es la ultima marca temporal modificada, inclusive, igual que `fin` en windowing.ventanear).
# Para clasificar una alarma como "durante" o "despues" del ataque se usa <= attack_end: una
# alarma disponible EXACTAMENTE al terminar la ultima ventana atacada cuenta como deteccion
# al finalizar el ataque, no como deteccion posterior.


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Intervalo de Wilson (aproximacion normal a la proporcion binomial, sin correccion de
    continuidad) para una tasa de deteccion. z=1.959963984540054 -> 95% de confianza.
    Devuelve (limite_inferior_pct, limite_superior_pct)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    centro = phat + z**2 / (2 * n)
    margen = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (100 * (centro - margen) / denom, 100 * (centro + margen) / denom)


def _generar_episodios(config: dict, datos: dict, split: str, seed: int,
                        modos: tuple[str, ...] = MODOS, n_posiciones_override: int | None = None) -> list[dict]:
    """Unico generador determinista de episodios (no usa el modelo). `n_posiciones_override`,
    si se da, sustituye a config["ataques"]["n_posiciones_por_duracion"] solo para esta
    llamada, sin tocar configs/base.yaml.

    Cada episodio trae, ademas de las columnas persistibles, claves internas
    (`_ventanas_idx`, `_ventanas_atacadas`) con las ventanas ya atacadas, calculadas aqui
    una unica vez para que reduccion_variable (el unico ataque con aleatoriedad interna) no
    se regenere distinto segun quien lo consuma despues.
    """
    particion = datos["particiones"][split]
    ventanas = datos["ventanas"][split]
    valores_particion = particion[COLUMNA_OBJETIVO].to_numpy()
    n_ventanas_grid = ventanas["X"].shape[0]
    tamano_ventana = ventanas["X"].shape[1]
    rango_valido = n_ventanas_grid * tamano_ventana  # cobertura real de la rejilla (descarta la misma cola que ventanear())

    valores_train = datos["particiones"]["train"][COLUMNA_OBJETIVO].to_numpy()
    umbral_picos_kw = {
        c: float(np.quantile(valores_train, c)) for c in config["ataques"]["recorte_picos"]["cuantil_umbral"]
    }

    combos = _combos_ataque(config)
    duraciones = config["ataques"]["duraciones_min"]
    n_posiciones = n_posiciones_override if n_posiciones_override is not None else config["ataques"]["n_posiciones_por_duracion"]

    episodios = []
    contador = {modo: 0 for modo in modos}

    for modo in modos:
        rng = np.random.default_rng(seed)  # stream propio por modo: mismo seed, distinto procedimiento de muestreo
        for combo in combos:
            for duracion in duraciones:
                for repeticion_idx in range(n_posiciones):
                    base = {
                        "split": split, "evaluation_mode": modo,
                        "family": combo["familia"], "parameter": combo["parametro"],
                        "duration_min": duracion, "repetition_idx": repeticion_idx, "seed": seed,
                        "requested_samples": duracion,
                    }

                    if modo == "legacy_aligned_window":
                        if duracion > tamano_ventana:
                            base.update({
                                "status": "not_executed", "motivo": "duration_exceeds_window",
                                "attack_start": pd.NaT, "attack_end": pd.NaT, "modified_samples": 0,
                                "episode_id": None,
                            })
                            episodios.append(base)
                            continue

                        idx_ventana = int(rng.integers(0, n_ventanas_grid))
                        offset = int(rng.integers(0, tamano_ventana - duracion + 1))
                        attack_start_ts = pd.Timestamp(ventanas["inicio"][idx_ventana]) + pd.Timedelta(minutes=offset)
                        x_tramo = ventanas["X"][idx_ventana][offset:offset + duracion]
                        y_tramo = _aplicar_ataque(combo, x_tramo, umbral_picos_kw, rng)

                        ventana_atacada = ventanas["X"][idx_ventana].copy()
                        ventana_atacada[offset:offset + duracion] = y_tramo
                        ventanas_idx = [idx_ventana]
                        ventanas_atacadas = {idx_ventana: ventana_atacada}

                    else:  # continuous_timeline
                        if duracion > rango_valido:
                            base.update({
                                "status": "not_executed", "motivo": "duration_exceeds_partition",
                                "attack_start": pd.NaT, "attack_end": pd.NaT, "modified_samples": 0,
                                "episode_id": None,
                            })
                            episodios.append(base)
                            continue

                        attack_start_offset_global = int(rng.integers(0, rango_valido - duracion + 1))
                        attack_start_ts = pd.Timestamp(particion.index[attack_start_offset_global])
                        x_tramo = valores_particion[attack_start_offset_global:attack_start_offset_global + duracion]
                        y_tramo = _aplicar_ataque(combo, x_tramo, umbral_picos_kw, rng)  # UNA sola vez, tramo completo

                        serie_atacada = valores_particion.copy()
                        serie_atacada[attack_start_offset_global:attack_start_offset_global + duracion] = y_tramo

                        attack_end_offset_global = attack_start_offset_global + duracion - 1
                        primer_k = attack_start_offset_global // tamano_ventana
                        ultimo_k = attack_end_offset_global // tamano_ventana
                        ventanas_idx = list(range(primer_k, ultimo_k + 1))
                        ventanas_atacadas = {
                            k: serie_atacada[k * tamano_ventana:(k + 1) * tamano_ventana].copy()
                            for k in ventanas_idx
                        }

                    attack_end_ts = attack_start_ts + pd.Timedelta(minutes=duracion - 1)
                    episode_id = f"{modo}__{contador[modo]:05d}"
                    contador[modo] += 1

                    base.update({
                        "episode_id": episode_id, "status": "executed", "motivo": None,
                        "attack_start": attack_start_ts, "attack_end": attack_end_ts,
                        "modified_samples": duracion,  # la asignacion por slice siempre toca exactamente `duracion` posiciones
                        "_ventanas_idx": ventanas_idx, "_ventanas_atacadas": ventanas_atacadas,
                    })
                    episodios.append(base)

    return episodios


_COLUMNAS_MASTER = [
    "episode_id", "split", "evaluation_mode", "family", "parameter", "duration_min",
    "repetition_idx", "attack_start", "attack_end", "requested_samples", "modified_samples",
    "seed", "status", "motivo",
]


def tabla_master(episodios: list[dict]) -> pd.DataFrame:
    """Vuelca los episodios a las columnas persistibles (sin las ventanas ya atacadas,
    que son detalle interno de puntuacion). No toca el modelo."""
    return pd.DataFrame(episodios)[_COLUMNAS_MASTER]


def puntuar_episodios(episodios: list[dict], datos: dict, split: str, modelo, tipo_score: str,
                       umbral: float, detector_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Puntua episodios ya generados. Para cada ventana afectada calcula el score de la
    version atacada y de la misma ventana limpia (nunca reconstruida). Los scores limpios
    se calculan una sola vez para toda la rejilla, para no repetir trabajo cuando varios
    episodios reusan la misma ventana. Devuelve (episode_detection_results, episode_window_scores).
    """
    ventanas = datos["ventanas"][split]
    ventanas_limpias_raw = ventanas["X"]  # nunca se muta
    params_norm = datos["params_norm"]
    tamano_ventana = ventanas_limpias_raw.shape[1]

    # scores de TODA la rejilla limpia, de una vez -- evita recalcular el mismo score limpio
    # cada vez que dos episodios distintos reusan la misma ventana (el muestreo es con reemplazo)
    scores_limpios_grid = calcular_scores(modelo, datos["ventanas_norm"][split], tipo_score)

    filas_resultado = []
    filas_ventanas = []

    for ep in episodios:
        identificacion = {c: ep.get(c) for c in _COLUMNAS_MASTER}

        if ep["status"] == "not_executed":
            filas_resultado.append({
                **identificacion, "detector_name": detector_name, "score_type": tipo_score,
                "n_affected_windows": 0, "max_score": np.nan, "detected": np.nan,
                "first_alarm_timestamp": pd.NaT, "detection_delay_min": np.nan,
                "first_affected_window_start": pd.NaT, "last_affected_window_end": pd.NaT,
                "raw_detected": np.nan, "induced_detected": np.nan,
                "n_raw_alarm_windows": 0, "n_induced_alarm_windows": 0, "n_preexisting_alarm_windows": 0,
                "first_raw_alarm_timestamp": pd.NaT, "first_induced_alarm_timestamp": pd.NaT,
                "raw_detection_delay_min": np.nan, "induced_detection_delay_min": np.nan,
                "raw_detected_during_attack": np.nan, "induced_detected_during_attack": np.nan,
                "raw_detected_after_attack": np.nan, "induced_detected_after_attack": np.nan,
            })
            continue

        attacked_scores, clean_scores, alarmas_raw, alarmas_induced, alarmas_preexist = {}, {}, {}, {}, {}

        for k in ep["_ventanas_idx"]:
            ventana_atacada = ep["_ventanas_atacadas"][k]
            assert len(ventana_atacada) == tamano_ventana, "ventana puntuada con tamano distinto al configurado"

            X_norm_atacada = aplicar_zscore(ventana_atacada[None, :], params_norm)
            attacked_score = float(calcular_scores(modelo, X_norm_atacada, tipo_score)[0])
            clean_score = float(scores_limpios_grid[k])
            score_delta = attacked_score - clean_score

            clean_is_alarm = clean_score > umbral
            attacked_is_alarm = attacked_score > umbral
            attack_induced_alarm = attacked_is_alarm and not clean_is_alarm
            preexisting_alarm = attacked_is_alarm and clean_is_alarm
            alarm_suppressed_by_attack = clean_is_alarm and not attacked_is_alarm

            attacked_scores[k] = attacked_score
            clean_scores[k] = clean_score
            alarmas_raw[k] = attacked_is_alarm
            alarmas_induced[k] = attack_induced_alarm
            alarmas_preexist[k] = preexisting_alarm

            inicio_k = pd.Timestamp(ventanas["inicio"][k])
            fin_k = pd.Timestamp(ventanas["fin"][k])
            solape_inicio = max(inicio_k, ep["attack_start"])
            solape_fin = min(fin_k, ep["attack_end"])
            overlap_minutes = int((solape_fin - solape_inicio) / pd.Timedelta(minutes=1)) + 1

            filas_ventanas.append({
                "episode_id": ep["episode_id"], "window_start": inicio_k, "window_end": fin_k,
                "threshold": umbral,
                "clean_score": clean_score, "attacked_score": attacked_score, "score_delta": score_delta,
                "clean_is_alarm": clean_is_alarm, "attacked_is_alarm": attacked_is_alarm,
                "attack_induced_alarm": attack_induced_alarm, "preexisting_alarm": preexisting_alarm,
                "alarm_suppressed_by_attack": alarm_suppressed_by_attack,
                "overlap_minutes": overlap_minutes, "attack_fraction_in_window": overlap_minutes / tamano_ventana,
            })

        ks_ordenados = sorted(attacked_scores)  # orden temporal (k creciente == tiempo creciente)
        max_score = max(attacked_scores.values())
        raw_detected = any(alarmas_raw.values())
        induced_detected = any(alarmas_induced.values())
        n_raw_alarm_windows = sum(alarmas_raw.values())
        n_induced_alarm_windows = sum(alarmas_induced.values())
        n_preexisting_alarm_windows = sum(alarmas_preexist.values())

        def _primera_alarma(mapa_alarmas):
            k_alarma = next((k for k in ks_ordenados if mapa_alarmas[k]), None)
            if k_alarma is None:
                return pd.NaT, np.nan, np.nan, np.nan
            ts = pd.Timestamp(ventanas["fin"][k_alarma])  # el score solo esta disponible al terminar la ventana
            delay = (ts - ep["attack_start"]) / pd.Timedelta(minutes=1)
            durante = ts <= ep["attack_end"]
            return ts, delay, durante, not durante

        first_raw_alarm_timestamp, raw_detection_delay_min, raw_detected_during_attack, raw_detected_after_attack = \
            _primera_alarma(alarmas_raw)
        first_induced_alarm_timestamp, induced_detection_delay_min, induced_detected_during_attack, induced_detected_after_attack = \
            _primera_alarma(alarmas_induced)

        filas_resultado.append({
            **identificacion, "detector_name": detector_name, "score_type": tipo_score,
            "n_affected_windows": len(ks_ordenados), "max_score": max_score, "detected": raw_detected,
            "first_alarm_timestamp": first_raw_alarm_timestamp, "detection_delay_min": raw_detection_delay_min,
            "first_affected_window_start": pd.Timestamp(ventanas["inicio"][ks_ordenados[0]]),
            "last_affected_window_end": pd.Timestamp(ventanas["fin"][ks_ordenados[-1]]),
            "raw_detected": raw_detected, "induced_detected": induced_detected,
            "n_raw_alarm_windows": n_raw_alarm_windows, "n_induced_alarm_windows": n_induced_alarm_windows,
            "n_preexisting_alarm_windows": n_preexisting_alarm_windows,
            "first_raw_alarm_timestamp": first_raw_alarm_timestamp,
            "first_induced_alarm_timestamp": first_induced_alarm_timestamp,
            "raw_detection_delay_min": raw_detection_delay_min, "induced_detection_delay_min": induced_detection_delay_min,
            "raw_detected_during_attack": raw_detected_during_attack, "induced_detected_during_attack": induced_detected_during_attack,
            "raw_detected_after_attack": raw_detected_after_attack, "induced_detected_after_attack": induced_detected_after_attack,
        })

    return pd.DataFrame(filas_resultado), pd.DataFrame(filas_ventanas)


def resumen_comparativo(resultados: pd.DataFrame) -> pd.DataFrame:
    """DR global (raw), por duracion y por familia, ventanas afectadas medias y retraso, por
    modo -- resumen ligero para el experimento legacy-vs-continuous."""
    filas = []
    for modo, grupo in resultados.groupby("evaluation_mode"):
        ejecutados = grupo[grupo["status"] == "executed"]
        detectados = ejecutados["detected"].astype(bool)
        retrasos = ejecutados.loc[ejecutados["detected"].astype(bool), "detection_delay_min"]
        filas.append({
            "evaluation_mode": modo,
            "n_episodios_configurados": len(grupo),
            "n_episodios_ejecutados": len(ejecutados),
            "n_not_executed": (grupo["status"] == "not_executed").sum(),
            "DR_global_pct": 100 * detectados.mean() if len(ejecutados) else float("nan"),
            "ventanas_afectadas_media": ejecutados["n_affected_windows"].mean() if len(ejecutados) else float("nan"),
            "retraso_medio_min": retrasos.mean() if len(retrasos) else float("nan"),
            "retraso_mediano_min": retrasos.median() if len(retrasos) else float("nan"),
        })
    return pd.DataFrame(filas)


def _fila_metrica(grupo: pd.DataFrame, ws_grupo: pd.DataFrame) -> dict:
    grupo = grupo.copy()
    grupo["raw_detected"] = grupo["raw_detected"].astype(bool)
    grupo["induced_detected"] = grupo["induced_detected"].astype(bool)
    n = len(grupo)

    n_raw = int(grupo["raw_detected"].sum())
    n_induced = int(grupo["induced_detected"].sum())
    raw_ci = wilson_ci(n_raw, n)
    induced_ci = wilson_ci(n_induced, n)

    con_alarma_raw = grupo[grupo["raw_detected"]]
    pct_solo_preexistente = (
        100 * (~con_alarma_raw["induced_detected"]).mean() if len(con_alarma_raw) else float("nan")
    )

    return {
        "n_episodios": n,
        "raw_DR_pct": 100 * n_raw / n if n else float("nan"),
        "raw_DR_ci95_low": raw_ci[0], "raw_DR_ci95_high": raw_ci[1],
        "induced_DR_pct": 100 * n_induced / n if n else float("nan"),
        "induced_DR_ci95_low": induced_ci[0], "induced_DR_ci95_high": induced_ci[1],
        # ojo: raw_detected_during_attack/after_attack son NaN (no aplicable) cuando no hubo
        # alarma -- .astype(bool) convertiria ese NaN en True (bool(nan) es True en Python),
        # asi que se compara explicitamente contra True para no contaminar el porcentaje.
        "DR_durante_ataque_pct": 100 * (grupo["raw_detected_during_attack"] == True).mean() if n else float("nan"),  # noqa: E712
        "DR_despues_ataque_pct": 100 * (grupo["raw_detected_after_attack"] == True).mean() if n else float("nan"),  # noqa: E712
        "pct_alarma_solo_preexistente": pct_solo_preexistente,
        "retraso_mediano_raw_min": grupo.loc[grupo["raw_detected"], "raw_detection_delay_min"].median(),
        "retraso_mediano_induced_min": grupo.loc[grupo["induced_detected"], "induced_detection_delay_min"].median(),
        "ventanas_afectadas_media": grupo["n_affected_windows"].mean(),
        "score_delta_medio": ws_grupo["score_delta"].mean() if len(ws_grupo) else float("nan"),
        "score_delta_mediano": ws_grupo["score_delta"].median() if len(ws_grupo) else float("nan"),
    }


def resumen_contrafactual(master: pd.DataFrame, resultados: pd.DataFrame, ventanas_scores: pd.DataFrame) -> pd.DataFrame:
    """Tabla agregada (continuous_detection_summary.csv): raw_DR / induced_DR con IC 95%
    (Wilson), por familia+parametro+duracion, por duracion, y global. score_delta se
    agrega a nivel de ventana, no de episodio."""
    ejecutados = resultados[resultados["status"] == "executed"].copy()
    ws = ventanas_scores.merge(master[["episode_id", "family", "parameter", "duration_min"]], on="episode_id", how="left")

    filas = []

    for (familia, parametro, duracion), grupo in ejecutados.groupby(["family", "parameter", "duration_min"]):
        ws_grupo = ws[(ws["family"] == familia) & (ws["parameter"] == parametro) & (ws["duration_min"] == duracion)]
        fila = {"nivel": "familia_parametro_duracion", "family": familia, "parameter": parametro, "duration_min": duracion}
        fila.update(_fila_metrica(grupo, ws_grupo))
        filas.append(fila)

    for duracion, grupo in ejecutados.groupby("duration_min"):
        ws_grupo = ws[ws["duration_min"] == duracion]
        fila = {"nivel": "duracion", "family": "ALL", "parameter": "ALL", "duration_min": duracion}
        fila.update(_fila_metrica(grupo, ws_grupo))
        filas.append(fila)

    fila_global = {"nivel": "global", "family": "ALL", "parameter": "ALL", "duration_min": "ALL"}
    fila_global.update(_fila_metrica(ejecutados, ws))
    filas.append(fila_global)

    return pd.DataFrame(filas)


def experimento_legacy_vs_continuous(config: dict, datos: dict, modelo, split: str = "test") -> dict:
    """Experimento de la fase anterior (legacy_aligned_window vs continuous_timeline, sin
    contrafactual, split=test). No se ejecuta desde __main__ en esta fase; se deja
    importable para no perder la funcionalidad."""
    tipo_score = config["score"]["tipo"]
    scores_train = calcular_scores(modelo, datos["ventanas_norm"]["train"], tipo_score)
    umbral = calcular_umbral(scores_train, config["umbral"]["estrategia"], config["umbral"].get("percentil", 99))
    detector_name = config["modelo"]["nombre"]

    episodios = _generar_episodios(config, datos, split, config["seed"])
    master = tabla_master(episodios)
    resultados, ventanas_scores = puntuar_episodios(episodios, datos, split, modelo, tipo_score, umbral, detector_name)
    resumen = resumen_comparativo(resultados)
    return {"master": master, "resultados": resultados, "ventanas_scores": ventanas_scores, "resumen": resumen}


def experimento_contrafactual(config: dict, datos: dict, modelo, split: str = "val", n_posiciones: int = 50,
                               sufijo_salida: str = "_val") -> dict:
    """Experimento contrafactual (limpio vs atacado) sobre `split` (por defecto val, para no
    contaminar test), con `n_posiciones` repeticiones por combinacion (default 50, solo
    para esta llamada) y unicamente en modo continuous_timeline. Reutiliza el checkpoint,
    el umbral y el score ya configurados, sin recalcular nada."""
    tipo_score = config["score"]["tipo"]
    scores_train = calcular_scores(modelo, datos["ventanas_norm"]["train"], tipo_score)
    umbral = calcular_umbral(scores_train, config["umbral"]["estrategia"], config["umbral"].get("percentil", 99))
    detector_name = config["modelo"]["nombre"]
    seed = config["seed"]

    episodios = _generar_episodios(config, datos, split, seed, modos=("continuous_timeline",),
                                    n_posiciones_override=n_posiciones)
    master = tabla_master(episodios)
    resultados, ventanas_scores = puntuar_episodios(episodios, datos, split, modelo, tipo_score, umbral, detector_name)
    resumen = resumen_contrafactual(master, resultados, ventanas_scores)

    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(TABLAS_DIR / f"episodes_master{sufijo_salida}.csv", index=False)
    resultados.to_csv(TABLAS_DIR / f"episode_detection_results{sufijo_salida}.csv", index=False)
    ventanas_scores.to_csv(TABLAS_DIR / f"episode_window_scores{sufijo_salida}.csv", index=False)
    resumen.to_csv(TABLAS_DIR / "continuous_detection_summary.csv", index=False)

    return {"master": master, "resultados": resultados, "ventanas_scores": ventanas_scores, "resumen": resumen,
            "umbral": umbral, "tipo_score": tipo_score, "detector_name": detector_name}


if __name__ == "__main__":
    from src.pipeline import obtener_modelo, preparar_datos
    from src.windowing import cargar_config

    config = cargar_config()
    datos = preparar_datos(config)
    modelo = obtener_modelo(config, datos["ventanas_norm"])  # checkpoint existente, NO reentrena

    print("Experimento contrafactual: split=val, evaluation_mode=continuous_timeline, "
          "n_posiciones=50 (override solo de esta ejecucion)\n")
    salida = experimento_contrafactual(config, datos, modelo, split="val", n_posiciones=50)

    print(f"Umbral reutilizado (sin recalcular): {salida['umbral']:.4f}, score={salida['tipo_score']}, "
          f"detector={salida['detector_name']}\n")
    print(f"episodes_master_val.csv: {len(salida['master'])} filas")
    print(f"episode_detection_results_val.csv: {len(salida['resultados'])} filas")
    print(f"episode_window_scores_val.csv: {len(salida['ventanas_scores'])} filas\n")

    resumen_global = salida["resumen"][salida["resumen"]["nivel"] == "global"]
    print("Resumen global:")
    print(resumen_global.to_string(index=False))

    print("\nResumen por duracion:")
    print(salida["resumen"][salida["resumen"]["nivel"] == "duracion"].to_string(index=False))

    print(f"\ncontinuous_detection_summary.csv guardada "
          f"({len(salida['resumen'])} filas: {(salida['resumen']['nivel']=='familia_parametro_duracion').sum()} "
          f"por familia/parametro/duracion + {(salida['resumen']['nivel']=='duracion').sum()} por duracion + 1 global).")
