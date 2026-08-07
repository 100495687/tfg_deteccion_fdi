"""Evaluacion final de CUSUM con normalizacion movil causal -- cierra la linea de acumulacion
temporal iniciada en experimento_cusum.py y diagnostico_no_estacionariedad.py.

Compara, sobre la segunda mitad de val (nunca usada para calibrar nada), 4 configuraciones:
  P  -- detector puntual consolidado (360/60/relative_kw).
  G  -- CUSUM con referencia global (mediana/MAD ajustadas en train, igual que en
        diagnostico_no_estacionariedad.metodo_G).
  R4 -- CUSUM con referencia movil causal de 4 semanas (hueco de 24h).
  R8 -- CUSUM con referencia movil causal de 8 semanas (hueco de 24h).

Para cada metodo y cada k en {0.75, 1.00, 1.25, 1.50}, h se calibra exclusivamente sobre la
primera mitad de val ("desarrollo") y se congela; la segunda mitad ("evaluacion") solo se usa
para medir, nunca para recalibrar. De los 4 k probados por metodo (y, para R4/R8, de las 2
variantes de suelo de escala) se selecciona una unica configuracion final por metodo usando
solo datos limpios de desarrollo (nunca DR, energia oculta ni etiquetas de ataque) -- esa es la
que se evalua sobre los episodios de ataque.

Pieza central: la rama atacada de R4/R8 construye su referencia movil con el historial
realmente observable en esa rama (ver `_referencia_atacada_contaminada`) -- nunca reutiliza
center_t/scale_t de la rama limpia. En la inmensa mayoria de ventanas la referencia atacada
coincide exactamente con la limpia porque el hueco de 24h y la duracion maxima de ataque
(1440 min) hacen que solo las ventanas mas tardias de los ataques mas largos puedan llegar a
"ver" sus propias muestras atacadas en la referencia; se detecta esa contaminacion
explicitamente y se recalcula la mediana/MAD solo en esos casos.

No reentrena nada, no cambia ventana/stride/relative_kw/ataques/episodios. Reutiliza sin
modificar base_relative.{cargar_ataques_y_modelo_360, calcular_relative_kw},
experimento_stride.{_ventanas_solapadas, calibrar_umbral_por_eventos, contar_eventos,
mcnemar_p, bootstrap_pareado, construir_rejilla}, experimento_cusum.{ajustar_estandarizacion,
estandarizar, simular_cusum, calibrar_h, _excursiones}, diagnostico_no_estacionariedad.
{construir_secuencia_continua, _referencia_movil}, stride_relative.puntuar_stride_relative y
episodes.wilson_ci.

Ejecutable de forma aislada:
    python -m src.cusum_final_normalizacion_movil
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

from src.base_relative import cargar_ataques_y_modelo_360, calcular_relative_kw
from src.data_loading import COLUMNA_OBJETIVO
from src.diagnostico_no_estacionariedad import GAP_HORAS, SEMANA_H, _referencia_movil, construir_secuencia_continua
from src.episodes import wilson_ci
from src.experimento_cusum import _excursiones as excursiones_cusum
from src.experimento_cusum import ajustar_estandarizacion, calibrar_h, estandarizar, simular_cusum
from src.experimento_stride import (
    _ventanas_solapadas,
    bootstrap_pareado,
    calibrar_umbral_por_eventos,
    construir_rejilla,
    contar_eventos,
    mcnemar_p,
)
from src.normalization import aplicar_zscore
from src.stride_relative import puntuar_stride_relative

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "cusum_final_normalizacion_movil"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "cusum_final_normalizacion_movil"
DET_DIR = BASE_DIR / "results" / "tables" / "analisis_detectabilidad"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDE = 60
SCORE = "relative_kw"
EPS_Z = 1e-6
K_VALUES = [0.75, 1.00, 1.25, 1.50]
OBJETIVO_FA = 0.06
SUELO_FRACCION = 0.25
METODOS_CUSUM = ["G", "R4", "R8"]
COLOR_METODO = {"P": GRIS_OSCURO, "G": NARANJA, "R4": AZUL, "R8": AZUL_OSCURO}


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


# ==========================================================================================
# 0/1/2) Setup: secuencia continua, rejilla val, episodios y division desarrollo/evaluacion
# ==========================================================================================

def cargar_todo() -> dict:
    config, datos, modelo, ataques, master_csv = cargar_ataques_y_modelo_360()
    params_norm = datos["params_norm"]
    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    n_dias_val = len(particion_val_raw) / 1440.0

    base = construir_secuencia_continua()  # reutiliza el checkpoint existente, no reentrena
    d_train, d_val = base["por_particion"]["train"], base["por_particion"]["val"]
    scores_train, scores_val = d_train["relative_kw"], d_val["relative_kw"]
    ts_train, ts_val = pd.DatetimeIndex(d_train["timestamps"]), pd.DatetimeIndex(d_val["timestamps"])
    scores_full = np.concatenate([scores_train, scores_val])
    ts_full = pd.DatetimeIndex(list(ts_train) + list(ts_val))
    n_train = len(scores_train)
    idx_val_en_full = n_train + np.arange(len(scores_val))

    grid_60 = construir_rejilla(particion_val, STRIDE)
    assert np.allclose(scores_val, calcular_relative_kw(
        aplicar_zscore(grid_60["X"], params_norm),
        modelo.reconstruir(aplicar_zscore(grid_60["X"], params_norm)), params_norm), atol=1e-6)
    tabla_ventanas, _ = puntuar_stride_relative(ataques, particion_val_raw, grid_60, modelo, params_norm, STRIDE)

    return {
        "config": config, "datos": datos, "modelo": modelo, "params_norm": params_norm, "ataques": ataques,
        "master_csv": master_csv, "particion_val_raw": particion_val_raw, "n_dias_val": n_dias_val,
        "scores_train": scores_train, "scores_val": scores_val, "ts_train": ts_train, "ts_val": ts_val,
        "scores_full": scores_full, "ts_full": ts_full, "n_train": n_train, "idx_val_en_full": idx_val_en_full,
        "grid_60": grid_60, "tabla_ventanas": tabla_ventanas,
    }


def dividir_desarrollo_evaluacion(ctx: dict) -> dict:
    n_val = len(ctx["scores_val"])
    corte = n_val // 2
    ts_val = ctx["ts_val"]
    dev_ini, dev_fin = ts_val[0], ts_val[corte - 1]
    eval_ini, eval_fin = ts_val[corte], ts_val[-1]
    n_dias_dev = corte * STRIDE / 1440.0
    n_dias_eval = (n_val - corte) * STRIDE / 1440.0

    master_csv = ctx["master_csv"]
    dentro = (master_csv["attack_start"] >= eval_ini) & (master_csv["attack_end"] <= eval_fin)
    incluidos = master_csv[dentro].copy()
    excluidos = master_csv[~dentro].copy()
    excluidos["motivo_exclusion"] = np.where(
        excluidos["attack_start"] < eval_ini, "attack_start anterior al inicio de evaluacion",
        "attack_end posterior al final de evaluacion")

    tabla_seleccion = pd.DataFrame([{
        "desarrollo_inicio": dev_ini, "desarrollo_fin": dev_fin, "n_ventanas_desarrollo": corte, "n_dias_desarrollo": n_dias_dev,
        "evaluacion_inicio": eval_ini, "evaluacion_fin": eval_fin, "n_ventanas_evaluacion": n_val - corte, "n_dias_evaluacion": n_dias_eval,
        "n_episodios_totales": len(master_csv), "n_episodios_incluidos": len(incluidos), "n_episodios_excluidos": len(excluidos),
    }])
    _guardar_tabla(tabla_seleccion, "00_particion_temporal")
    _guardar_tabla(incluidos, "00_episodios_incluidos")
    _guardar_tabla(excluidos[["episode_id", "family", "parameter", "duration_min", "attack_start", "attack_end", "motivo_exclusion"]],
                    "00_episodios_excluidos")
    for nombre_grupo, col in (("familia", "family"), ("duracion", "duration_min"), ("severidad", "parameter")):
        _guardar_tabla(incluidos.groupby(col).size().rename("n_episodios").reset_index(), f"00_distribucion_incluidos_por_{nombre_grupo}")

    return {"corte_idx": corte, "dev_ini": dev_ini, "dev_fin": dev_fin, "eval_ini": eval_ini, "eval_fin": eval_fin,
            "n_dias_dev": n_dias_dev, "n_dias_eval": n_dias_eval, "incluidos": incluidos, "excluidos": excluidos,
            "episode_ids_incluidos": set(incluidos["episode_id"])}


# ==========================================================================================
# 3/4) z_t de cada metodo sobre TODO val (dev+eval), sin tocar ataques
# ==========================================================================================

def calcular_z_metodos(ctx: dict) -> dict:
    scores_full, idx_val = ctx["scores_full"], ctx["idx_val_en_full"]
    params_g = ajustar_estandarizacion(ctx["scores_train"])  # G: ajustado SOLO en train
    z_g = estandarizar(ctx["scores_val"], params_g)
    scale_global = params_g["denom_efectivo"]

    resultados = {"G": {"z": z_g, "params": params_g, "scale_global": scale_global}}
    for semanas, nombre in ((4, "R4"), (8, "R8")):
        center_r, scale_r, n_ref, ref_start, ref_end, suficientes = _referencia_movil(scores_full, idx_val, semanas)
        z_original = np.where(suficientes, (ctx["scores_val"] - center_r) / np.maximum(scale_r, EPS_Z), np.nan)
        suelo = SUELO_FRACCION * scale_global
        scale_r_piso = np.maximum(scale_r, suelo)
        z_piso = np.where(suficientes, (ctx["scores_val"] - center_r) / np.maximum(scale_r_piso, EPS_Z), np.nan)
        resultados[nombre] = {
            "center_r": center_r, "scale_r": scale_r, "ref_start": ref_start, "ref_end": ref_end,
            "suficientes": suficientes, "semanas": semanas, "z_original": z_original, "z_piso": z_piso,
            "scale_r_piso": scale_r_piso,
        }
    return resultados


# ==========================================================================================
# 5) Diagnostico de scale_t (R4/R8) sobre datos limpios de DESARROLLO
# ==========================================================================================

def diagnostico_scale(z_metodos: dict, div: dict, ts_val: pd.DatetimeIndex, scale_global: float) -> dict:
    corte = div["corte_idx"]
    tablas, extremos = {}, {}
    for nombre in ("R4", "R8"):
        info = z_metodos[nombre]
        scale_dev = info["scale_r"][:corte]
        z_dev = info["z_original"][:corte]
        pct_25 = 100 * float((scale_dev < 0.25 * scale_global).mean())
        pct_50 = 100 * float((scale_dev < 0.50 * scale_global).mean())
        resumen = {
            "metodo": nombre, "min": float(np.min(scale_dev)), "media": float(np.mean(scale_dev)),
            "mediana": float(np.median(scale_dev)), "std": float(np.std(scale_dev)),
            "p1": float(np.percentile(scale_dev, 1)), "p5": float(np.percentile(scale_dev, 5)),
            "p25": float(np.percentile(scale_dev, 25)), "p75": float(np.percentile(scale_dev, 75)),
            "p95": float(np.percentile(scale_dev, 95)), "p99": float(np.percentile(scale_dev, 99)),
            "pct_bajo_25pct_global": pct_25, "pct_bajo_50pct_global": pct_50, "scale_global": scale_global,
        }
        tablas[nombre] = resumen
        orden = np.argsort(scale_dev)[:20]
        extremos[nombre] = pd.DataFrame({
            "timestamp": ts_val[:corte][orden], "scale_t": scale_dev[orden], "z_t": z_dev[orden],
        }).sort_values("scale_t")
    tabla_resumen = _guardar_tabla(pd.DataFrame(tablas.values()), "01_diagnostico_scale_desarrollo")
    for nombre, tb in extremos.items():
        _guardar_tabla(tb, f"01_escalas_minimas_{nombre}")
    return {"resumen": tabla_resumen, "extremos": extremos}


def seleccionar_variante_escala(diag_scale: dict) -> dict:
    decision = {}
    for _, fila in diag_scale["resumen"].iterrows():
        nombre = fila["metodo"]
        if fila["pct_bajo_25pct_global"] < 0.01:
            decision[nombre] = {"variante": "original", "motivo": "el suelo del 25% no se activa practicamente nunca "
                                 f"({fila['pct_bajo_25pct_global']:.3f}% de las ventanas de desarrollo) -> equivalente al original"}
        else:
            decision[nombre] = {"variante": "piso", "motivo": f"el {fila['pct_bajo_25pct_global']:.2f}% de las escalas de "
                                 "desarrollo caen por debajo del 25% de la escala global -> se aplica el suelo robusto"}
    return decision


# ==========================================================================================
# 6/11) Calibracion de h por (metodo, variante, k), SOLO en desarrollo
# ==========================================================================================

def calibrar_todas_las_configuraciones(z_metodos: dict, div: dict) -> pd.DataFrame:
    corte = div["corte_idx"]
    n_dias_dev = div["n_dias_dev"]
    filas = []
    configs = {"G": z_metodos["G"]["z"]}
    for nombre in ("R4", "R8"):
        configs[f"{nombre}_original"] = z_metodos[nombre]["z_original"]
        configs[f"{nombre}_piso"] = z_metodos[nombre]["z_piso"]

    for nombre_config, z in configs.items():
        z_dev = z[:corte]
        z_dev_valido = z_dev[~np.isnan(z_dev)]
        for k in K_VALUES:
            h, tasa_dev, C_dev, alarm_dev = calibrar_h(z_dev_valido, k, OBJETIVO_FA, n_dias_dev)
            excursiones = excursiones_cusum(C_dev, STRIDE)
            filas.append({
                "config": nombre_config, "k": k, "h": h, "n_alarmas_desarrollo": int(alarm_dev.sum()),
                "n_dias_desarrollo": n_dias_dev, "eventos_dia_desarrollo": tasa_dev,
                "excursion_media_min": float(excursiones.mean()), "excursion_mediana_min": float(np.median(excursiones)),
                "excursion_p95_min": float(np.percentile(excursiones, 95)), "excursion_max_min": float(excursiones.max()),
                "max_C_desarrollo": float(C_dev.max()),
            })
    tabla = _guardar_tabla(pd.DataFrame(filas), "02_calibracion_desarrollo_todas_configs")
    return tabla


def seleccionar_configuraciones_finales(tabla_calib: pd.DataFrame, variantes: dict) -> dict:
    """Unicamente con datos de desarrollo: para cada metodo, la variante de escala ya decidida
    en `seleccionar_variante_escala`, y dentro de ella el k con menor excursion p95 (desempate
    por excursion maxima, despues por cercania de h a un valor no patologico)."""
    seleccion = {}
    sub_g = tabla_calib[tabla_calib["config"] == "G"].sort_values(["excursion_p95_min", "excursion_max_min"])
    seleccion["G"] = {"config": "G", "k": float(sub_g.iloc[0]["k"]), "h": float(sub_g.iloc[0]["h"]), "variante": "n/a"}
    for nombre in ("R4", "R8"):
        variante = variantes[nombre]["variante"]
        sub = tabla_calib[tabla_calib["config"] == f"{nombre}_{variante}"].sort_values(["excursion_p95_min", "excursion_max_min"])
        seleccion[nombre] = {"config": f"{nombre}_{variante}", "k": float(sub.iloc[0]["k"]), "h": float(sub.iloc[0]["h"]), "variante": variante}
    tabla_sel = pd.DataFrame(seleccion.values())
    _guardar_tabla(tabla_sel, "03_configuraciones_finales_congeladas")
    return seleccion


# ==========================================================================================
# 7/8) Referencia observable en la rama atacada (R4/R8) -- pieza central del experimento
# ==========================================================================================

def _referencia_atacada_contaminada(scores_full: np.ndarray, ref_start: int, ref_end: int, W: int,
                                     ks_globales_episodio: set[int], scores_atacados_globales: dict[int, float],
                                     suelo: float | None) -> tuple[float, float, int]:
    """Mediana/MAD de la ventana de referencia [ref_start, ref_end], sustituyendo por su valor
    atacado cualquier posicion que pertenezca a las propias ventanas afectadas del episodio.
    Si ninguna posicion esta contaminada, el resultado es matematicamente identico al de la
    referencia limpia (se puede verificar: el parche no tiene efecto fuera de las posiciones
    contaminadas)."""
    tramo = scores_full[ref_start:ref_end + 1].copy()
    n_contaminadas = 0
    for k_glob in ks_globales_episodio:
        if ref_start <= k_glob <= ref_end:
            tramo[k_glob - ref_start] = scores_atacados_globales[k_glob]
            n_contaminadas += 1
    med = float(np.median(tramo))
    mad = float(np.median(np.abs(tramo - med)))
    scale = 1.4826 * mad
    if suelo is not None:
        scale = max(scale, suelo)
    return med, scale, n_contaminadas


def construir_z_atacado_episodios(ctx: dict, episode_ids: set[str], nombre_metodo: str, semanas: int,
                                   center_r: np.ndarray, scale_r: np.ndarray, ref_start_arr: np.ndarray,
                                   ref_end_arr: np.ndarray, suelo: float | None) -> dict[str, dict]:
    """Para cada episodio incluido, calcula z_atacado en cada ventana de su horizonte
    (primera ventana afectada -> last_affected_window_end), usando el historial realmente
    observable en la rama atacada (nunca center/scale de la rama limpia salvo que, por
    construccion, coincidan exactamente al no haber contaminacion -- ver docstring de
    `_referencia_atacada_contaminada`)."""
    n_train, n_ventanas_grid = ctx["n_train"], ctx["grid_60"]["X"].shape[0]
    W = semanas * SEMANA_H
    tabla_ventanas = ctx["tabla_ventanas"]
    atacados_por_episodio = {ep: dict(zip(g["k"], g["attacked_score"])) for ep, g in
                              tabla_ventanas.groupby("episode_id") if ep in episode_ids}

    resultado = {}
    for episode_id, a in ctx["ataques"].items():
        if episode_id not in episode_ids:
            continue
        ks_local = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_ventanas_grid))
        ks_global = [n_train + k for k in ks_local]
        ks_global_set = set(ks_global)
        atacados_globales = {n_train + k: v for k, v in atacados_por_episodio[episode_id].items()}

        z_att, centros, escalas, n_contam_lista = [], [], [], []
        for k_glob in ks_global:
            k_idx_val = k_glob - n_train
            ref_s, ref_e = int(ref_start_arr[k_idx_val]), int(ref_end_arr[k_idx_val])
            solapa = any(ref_s <= kg <= ref_e for kg in ks_global_set)
            if not solapa:
                c, s = float(center_r[k_idx_val]), float(scale_r[k_idx_val])
                if suelo is not None:
                    s = max(s, suelo)
                n_contam = 0
            else:
                c, s, n_contam = _referencia_atacada_contaminada(
                    ctx["scores_full"], ref_s, ref_e, W, ks_global_set, atacados_globales, suelo)
            centros.append(c); escalas.append(max(s, EPS_Z)); n_contam_lista.append(n_contam)
            z_att.append((atacados_globales[k_glob] - c) / max(s, EPS_Z))

        resultado[episode_id] = {
            "ks_local": ks_local, "z_atacado": np.array(z_att), "center_atacado": np.array(centros),
            "scale_atacado": np.array(escalas), "n_contaminadas": np.array(n_contam_lista),
            "center_limpio": center_r[[k - n_train for k in ks_global]], "scale_limpio": scale_r[[k - n_train for k in ks_global]],
        }
    return resultado


# ==========================================================================================
# 9/13/14) Simulacion CUSUM por episodio (clean trajectory continua + rama atacada)
# ==========================================================================================

def simular_metodo_episodios(ctx: dict, episode_ids: set[str], z_val: np.ndarray, k_param: float, h: float,
                              z_atacado_por_episodio: dict | None = None) -> pd.DataFrame:
    """z_val: serie z limpia continua de todo val (dev+eval), ya con NaN descartado por
    construccion salvo insuficiencia de historial (no ocurre en este dataset). Si
    `z_atacado_por_episodio` es None, se asume G (referencia estatica: z_atacado se calcula
    directamente con los mismos center/scale globales, sin contaminacion posible)."""
    n_train, n_ventanas_grid = ctx["n_train"], ctx["grid_60"]["X"].shape[0]
    C_clean_val, alarm_clean_val = simular_cusum(np.nan_to_num(z_val, nan=0.0), k_param, h)
    fin_val = pd.to_datetime(ctx["grid_60"]["fin"])

    filas = []
    for episode_id, a in ctx["ataques"].items():
        if episode_id not in episode_ids:
            continue
        ks_local = sorted(_ventanas_solapadas(a["offset_global"], a["duracion"], STRIDE, n_ventanas_grid))
        k_before = ks_local[0] - 1
        c_prev = float(C_clean_val[k_before]) if k_before >= 0 else 0.0

        if z_atacado_por_episodio is not None:
            z_att = z_atacado_por_episodio[episode_id]["z_atacado"]
        else:  # metodo G: referencia estatica, sin adaptacion posible
            tabla_ep = ctx["tabla_ventanas"]
            tabla_ep = tabla_ep[(tabla_ep["episode_id"] == episode_id)].set_index("k")
            scores_att = tabla_ep.loc[ks_local, "attacked_score"].to_numpy()
            z_att = estandarizar(scores_att, ctx["_params_g"])

        C_att, alarm_att = [], []
        for zt in z_att:
            c_t = max(0.0, c_prev + zt - k_param)
            es_alarma = c_t > h
            C_att.append(c_t); alarm_att.append(es_alarma)
            c_prev = 0.0 if es_alarma else c_t
        C_att = np.array(C_att); alarm_att = np.array(alarm_att, dtype=bool)
        alarm_lim = alarm_clean_val[ks_local]
        induced = alarm_att & ~alarm_lim
        idx_ind = np.where(induced)[0]
        ts_induced = fin_val[ks_local[idx_ind[0]]] if len(idx_ind) else pd.NaT

        filas.append({
            "episode_id": episode_id, "n_affected_windows": len(ks_local),
            "max_C_attacked": float(C_att.max()), "max_C_clean": float(C_clean_val[ks_local].max()),
            "raw_detected": bool(alarm_att.any()), "induced_detected": bool(induced.any()),
            "first_induced_alarm_timestamp": ts_induced, "preexisting_alarm": bool((alarm_att & alarm_lim).any()),
            "suppressed_alarm": bool((alarm_lim & ~alarm_att).any()),
        })
    return pd.DataFrame(filas), C_clean_val, alarm_clean_val


def enriquecer_resultado(sim: pd.DataFrame, ataques: dict, master_csv: pd.DataFrame, particion_val_raw: np.ndarray,
                          contexto_impacto: pd.DataFrame) -> pd.DataFrame:
    from src.analisis_detectabilidad import _energia_antes_despues
    meta_cols = ["episode_id", "family", "parameter", "duration_min", "repetition_idx", "attack_start", "attack_end"]
    df = master_csv[meta_cols].merge(sim, on="episode_id").merge(contexto_impacto, on="episode_id")
    sin_induced = ~df["induced_detected"]
    df["induced_detection_delay_min"] = (df["first_induced_alarm_timestamp"] - df["attack_start"]) / pd.Timedelta(minutes=1)
    df.loc[sin_induced, "induced_detection_delay_min"] = np.nan
    durante = (df["first_induced_alarm_timestamp"] <= df["attack_end"]).astype(object)
    despues = (df["first_induced_alarm_timestamp"] > df["attack_end"]).astype(object)
    durante[sin_induced] = np.nan; despues[sin_induced] = np.nan
    df["detected_during_attack"] = durante; df["detected_after_attack"] = despues
    filas_energia = [_energia_antes_despues(row, ataques, particion_val_raw) for row in df.itertuples()]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(filas_energia)], axis=1)
    return df


# ==========================================================================================
# 15) Metricas principales (reutilizable por metodo/subconjunto)
# ==========================================================================================

def resumen_metricas(grupo: pd.DataFrame) -> dict:
    n = len(grupo)
    n_ind = int(grupo["induced_detected"].sum())
    ci = wilson_ci(n_ind, n) if n else (float("nan"), float("nan"))
    con_ind = grupo[grupo["induced_detected"] == True]  # noqa: E712
    total_hidden = grupo["hidden_energy_kwh"].sum()
    detected_hidden = grupo.loc[grupo["induced_detected"], "hidden_energy_kwh"].sum()
    after_total = grupo["hidden_energy_after_detection_kwh"].sum()
    delays = con_ind["induced_detection_delay_min"]
    return {
        "n_episodios": n, "n_detectados": n_ind, "induced_DR_pct": 100 * n_ind / n if n else float("nan"),
        "induced_DR_ci95_low": 100 * ci[0], "induced_DR_ci95_high": 100 * ci[1],
        "energy_weighted_dr_pct": 100 * detected_hidden / total_hidden if total_hidden > 1e-9 else float("nan"),
        "timely_energy_detection_rate_pct": 100 * after_total / total_hidden if total_hidden > 1e-9 else float("nan"),
        "retraso_mediano_min": float(delays.median()) if len(delays) else float("nan"),
        "retraso_p75_min": float(delays.quantile(0.75)) if len(delays) else float("nan"),
        "retraso_p90_min": float(delays.quantile(0.90)) if len(delays) else float("nan"),
        "pct_detectado_durante_ataque": 100 * (con_ind["detected_during_attack"] == True).mean() if len(con_ind) else float("nan"),  # noqa: E712
        "hidden_energy_before_detection_total_kwh": float(grupo["hidden_energy_before_detection_kwh"].sum()),
        "pct_recuperable_medio": float(grupo["recoverable_hidden_energy_pct"].mean()),
    }


def tablas_por_subconjunto(df_todos: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    tablas = {}
    filas_glob = []
    for metodo, df in df_todos.items():
        for etiqueta, sub in (("todas", df), ("sin_recorte_picos", df[df["family"] != "recorte_picos"])):
            fila = {"metodo": metodo, "familias": etiqueta}
            fila.update(resumen_metricas(sub)); filas_glob.append(fila)
    tablas["global"] = pd.DataFrame(filas_glob)

    for nombre_grupo, col in (("familia", "family"), ("duracion", "duration_min"), ("severidad", "parameter")):
        filas = []
        for metodo, df in df_todos.items():
            for val_, sub in df.groupby(col):
                fila = {"metodo": metodo, col: val_}
                fila.update(resumen_metricas(sub)); filas.append(fila)
        tablas[f"por_{nombre_grupo}"] = pd.DataFrame(filas)

    tramos = {"30_60_120min": lambda d: d["duration_min"].isin([30, 60, 120]),
              "240_360min": lambda d: d["duration_min"].isin([240, 360]),
              "720_1440min": lambda d: d["duration_min"].isin([720, 1440])}
    filas = []
    for metodo, df in df_todos.items():
        for nombre_tramo, filtro in tramos.items():
            sub = df[filtro(df)]
            fila = {"metodo": metodo, "tramo_duracion": nombre_tramo}
            fila.update(resumen_metricas(sub)); filas.append(fila)
    tablas["por_tramo_duracion"] = pd.DataFrame(filas)
    return tablas


def falsas_alarmas_evaluacion(z_val: np.ndarray, k_param: float, h: float, corte: int, n_dias_eval: float) -> dict:
    z_eval = np.nan_to_num(z_val[corte:], nan=0.0)
    C_eval, alarm_eval = simular_cusum(z_eval, k_param, h)
    n_alarmas = int(alarm_eval.sum())
    tasa = n_alarmas / n_dias_eval
    # IC de Poisson (Wilson-esque exacto via chi2 no disponible sin scipy.stats.chi2 -- se usa
    # la aproximacion de scipy.stats.poisson por bisection del IC exacto de Garwood)
    lo, hi = sstats.chi2.ppf(0.025, 2 * n_alarmas) / 2 if n_alarmas > 0 else 0.0, sstats.chi2.ppf(0.975, 2 * (n_alarmas + 1)) / 2
    excursiones = excursiones_cusum(C_eval, STRIDE)
    ts_alarma = np.where(alarm_eval)[0]
    tiempo_medio = float(np.diff(ts_alarma).mean() * STRIDE) if len(ts_alarma) > 1 else float("nan")
    return {
        "n_alarmas_evaluacion": n_alarmas, "n_dias_evaluacion": n_dias_eval, "eventos_dia_evaluacion": tasa,
        "ic95_poisson_low": lo / n_dias_eval, "ic95_poisson_high": hi / n_dias_eval,
        "ventanas_falsas_dia_evaluacion": float(alarm_eval.sum()) / n_dias_eval,
        "tiempo_medio_entre_alarmas_min": tiempo_medio,
        "excursion_media_min": float(excursiones.mean()), "excursion_mediana_min": float(np.median(excursiones)),
        "excursion_p95_min": float(np.percentile(excursiones, 95)), "excursion_max_min": float(excursiones.max()),
    }


# ==========================================================================================
# 18/19) Complementariedad y comparaciones pareadas
# ==========================================================================================

def tabla_complementariedad(df_p: pd.DataFrame, df_cusum: pd.DataFrame) -> dict:
    p = df_p.set_index("episode_id")["induced_detected"].astype(bool)
    c = df_cusum.set_index("episode_id")["induced_detected"].astype(bool)
    c = c.reindex(p.index)
    ambos = int((p & c).sum()); solo_p = int((p & ~c).sum()); solo_c = int((~p & c).sum()); ninguno = int((~p & ~c).sum())
    retention = 100 * ambos / p.sum() if p.sum() else float("nan")
    exclusive_recovery = 100 * solo_c / (~p).sum() if (~p).sum() else float("nan")
    return {"detectado_ambos": ambos, "solo_puntual": solo_p, "solo_cusum": solo_c, "detectado_ninguno": ninguno,
            "retention_rate_pct": retention, "exclusive_recovery_rate_pct": exclusive_recovery}


def comparacion_pareada_generica(dfs: dict[str, pd.DataFrame], grupo_b_ids: set, alto_impacto_ids: set) -> pd.DataFrame:
    detecciones = {m: df.set_index("episode_id")["induced_detected"].astype(bool) for m, df in dfs.items()}
    meta = dfs["P"].set_index("episode_id")[["family", "duration_min"]]
    pares = [("P", "G"), ("P", "R4"), ("P", "R8"), ("G", "R4"), ("R4", "R8")]
    filas = []
    for a, b in pares:
        det_a, det_b = detecciones[a].reindex(meta.index), detecciones[b].reindex(meta.index)
        comb = pd.DataFrame({"a": det_a, "b": det_b}).join(meta)
        comb["grupo_b"] = comb.index.isin(grupo_b_ids)
        comb["alto_impacto"] = comb.index.isin(alto_impacto_ids)
        subconjuntos = {"global": comb, "sin_recorte_picos": comb[comb["family"] != "recorte_picos"],
                        "grupo_b": comb[comb["grupo_b"]], "alto_impacto": comb[comb["alto_impacto"]],
                        "duraciones_largas_720_1440": comb[comb["duration_min"].isin([720, 1440])]}
        for fam in comb["family"].unique():
            sub_fam = comb[comb["family"] == fam]
            if len(sub_fam) >= 30:
                subconjuntos[f"familia_{fam}"] = sub_fam
        for nombre_sub, sub in subconjuntos.items():
            if len(sub) == 0:
                continue
            da, db = sub["a"].to_numpy(), sub["b"].to_numpy()
            solo_a, solo_b = int((da & ~db).sum()), int((~da & db).sum())
            p_mcnemar = mcnemar_p(solo_a, solo_b)
            ci_low, ci_high = bootstrap_pareado(da.astype(float), db.astype(float))
            filas.append({
                "metodo_a": a, "metodo_b": b, "subconjunto": nombre_sub, "n_episodios": len(sub),
                "DR_a_pct": 100 * da.mean(), "DR_b_pct": 100 * db.mean(), "diferencia_pp": 100 * (db.mean() - da.mean()),
                "solo_a": solo_a, "solo_b": solo_b, "ambos": int((da & db).sum()), "ninguno": int((~da & ~db).sum()),
                "mcnemar_p": p_mcnemar, "bootstrap_ci95_diferencia_low": ci_low, "bootstrap_ci95_diferencia_high": ci_high,
                "significativo_p<0.05": p_mcnemar < 0.05,
                "relevante_practicamente_(>=3pp)": abs(100 * (db.mean() - da.mean())) >= 3.0,
            })
    return pd.DataFrame(filas)


# ==========================================================================================
# 21) Criterios de exito (fijados ANTES de observar resultados de ataques)
# ==========================================================================================

CRITERIOS_EXITO_TEXTO = {
    1: "Recupera >= 10% del Grupo B no detectado por el puntual",
    2: "Recupera >= 10% de los ataques de alto impacto no detectados",
    3: "Mejora >= 3pp el DR ponderado por energia frente al puntual (o aporta detecciones exclusivas equivalentes)",
    4: "Mantiene una tasa de falsas alarmas operativamente compatible con 0.06 eventos/dia",
    5: "Retraso mediano <= ~180 min",
    6: "Conserva una proporcion elevada de los ataques severos ya detectados por el puntual (retention_rate alta)",
    7: "La mejora aparece en mas de una familia/subconjunto, no depende de pocos episodios",
    8: "La comparacion estadistica y la magnitud practica son coherentes",
}


def evaluar_criterios_exito(nombre_cusum: str, gb_pct: float, ai_pct: float, dif_energia_pp: float,
                             fa_eval: float, retraso_mediano: float, retention_rate: float,
                             pareada: pd.DataFrame) -> pd.DataFrame:
    familias_con_mejora = pareada[(pareada["metodo_a"] == "P") & (pareada["metodo_b"] == nombre_cusum) &
                                   (pareada["subconjunto"].str.startswith("familia_")) &
                                   (pareada["diferencia_pp"] > 0)]
    coherencia = pareada[(pareada["metodo_a"] == "P") & (pareada["metodo_b"] == nombre_cusum) & (pareada["subconjunto"] == "sin_recorte_picos")]
    coherente = bool(len(coherencia) and (coherencia.iloc[0]["significativo_p<0.05"] == coherencia.iloc[0]["relevante_practicamente_(>=3pp)"]))

    resultados = {
        1: gb_pct >= 10.0, 2: ai_pct >= 10.0, 3: dif_energia_pp >= 3.0,
        4: abs(fa_eval - 0.06) <= 0.06, 5: (retraso_mediano <= 180.0) if not np.isnan(retraso_mediano) else False,
        6: retention_rate >= 80.0, 7: len(familias_con_mejora) >= 2, 8: coherente,
    }
    filas = [{"criterio_num": k, "criterio": CRITERIOS_EXITO_TEXTO[k], "cumplido": v} for k, v in resultados.items()]
    tabla = pd.DataFrame(filas)
    n_cumplidos = sum(resultados.values())
    tabla.attrs["n_cumplidos"] = n_cumplidos
    tabla.attrs["mayoria"] = n_cumplidos >= 5
    return tabla


# ==========================================================================================
# Figuras
# ==========================================================================================

def _z_final(z_metodos: dict, nombre: str, variante: str) -> np.ndarray:
    if nombre == "G":
        return z_metodos["G"]["z"]
    return z_metodos[nombre]["z_original"] if variante == "original" else z_metodos[nombre]["z_piso"]


def generar_figuras(z_metodos, div, ctx, diag_scale, tabla_calib, seleccion, tablas_metricas, tabla_fa,
                     tabla_gb, tabla_ai, comp_dict, pareada, casos_ts, df_todos) -> None:
    ts_val = ctx["ts_val"]; corte = div["corte_idx"]

    # 1) distribucion de scale_t para R4 y R8
    fig, ax = plt.subplots(figsize=(8, 5))
    for nombre, color in (("R4", AZUL), ("R8", AZUL_OSCURO)):
        ax.hist(z_metodos[nombre]["scale_r"][:corte], bins=60, color=color, alpha=0.5, density=True, label=nombre)
    ax.axvline(z_metodos["G"]["scale_global"], color=ROJO, linestyle="--", label="scale_global (G)")
    ax.axvline(0.25 * z_metodos["G"]["scale_global"], color=GRIS_OSCURO, linestyle=":", label="25% scale_global")
    ax.set_xlabel("scale_t"); ax.set_ylabel("densidad"); ax.set_title("Distribucion de scale_t (desarrollo)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "01_distribucion_scale_t")

    # 2) scale_t vs |z_t| extremos
    fig, ax = plt.subplots(figsize=(8, 5))
    for nombre, color in (("R4", AZUL), ("R8", AZUL_OSCURO)):
        s, z = z_metodos[nombre]["scale_r"][:corte], z_metodos[nombre]["z_original"][:corte]
        ax.scatter(s, np.abs(z), s=6, alpha=0.3, color=color, label=nombre)
    ax.set_xlabel("scale_t"); ax.set_ylabel("|z_t|"); ax.set_title("scale_t frente a valores extremos de z_t (desarrollo)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "02_scale_vs_z_extremos")

    # 3) falsas alarmas desarrollo vs evaluacion
    fig, ax = plt.subplots(figsize=(8, 5))
    metodos_fa = list(tabla_fa["metodo"])
    x = np.arange(len(metodos_fa)); ancho = 0.35
    ax.bar(x - ancho / 2, tabla_fa["eventos_dia_desarrollo"], width=ancho, color=GRIS, label="desarrollo")
    ax.bar(x + ancho / 2, tabla_fa["eventos_dia_evaluacion"], width=ancho, color=ROJO, label="evaluacion")
    ax.axhline(0.06, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(metodos_fa)
    ax.set_ylabel("falsos eventos/dia"); ax.set_title("Falsas alarmas: desarrollo vs evaluacion (h congelado)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "03_falsas_alarmas_desarrollo_evaluacion")

    # 4) IC de las tasas
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(tabla_fa))
    ax.errorbar(tabla_fa["eventos_dia_evaluacion"], y,
                xerr=[tabla_fa["eventos_dia_evaluacion"] - tabla_fa["ic95_poisson_low"],
                      tabla_fa["ic95_poisson_high"] - tabla_fa["eventos_dia_evaluacion"]],
                fmt="o", color=AZUL, capsize=4)
    ax.axvline(0.06, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(metodos_fa)
    ax.set_xlabel("falsos eventos/dia (evaluacion)"); ax.set_title("IC95% de Poisson de la tasa de falsas alarmas en evaluacion")
    fig.tight_layout(); _guardar_fig(fig, "04_ic_tasas_falsas_alarmas")

    # 5) excursiones limpias por metodo (desarrollo vs evaluacion) -- solo metodos CUSUM (P no acumula)
    metodos_cusum_fa = [m for m in metodos_fa if m in seleccion]
    xc = np.arange(len(metodos_cusum_fa))
    fig, ax = plt.subplots(figsize=(8, 5))
    tabla_fa_cusum = tabla_fa[tabla_fa["metodo"].isin(metodos_cusum_fa)].set_index("metodo").reindex(metodos_cusum_fa)
    excursion_dev = np.array([
        tabla_calib[(tabla_calib["config"] == seleccion[m]["config"]) & (tabla_calib["k"] == seleccion[m]["k"])]["excursion_max_min"].iloc[0]
        for m in metodos_cusum_fa
    ])
    ax.bar(xc - ancho / 2, excursion_dev, width=ancho, color=GRIS, label="max desarrollo")
    ax.bar(xc + ancho / 2, tabla_fa_cusum["excursion_max_min"], width=ancho, color=ROJO, label="max evaluacion")
    ax.set_xticks(xc); ax.set_xticklabels(metodos_cusum_fa); ax.set_yscale("log")
    ax.set_ylabel("excursion maxima (min, log)"); ax.set_title("Excursiones limpias por metodo")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "05_excursiones_limpias_por_metodo")

    # 6) DR por duracion
    fig, ax = plt.subplots(figsize=(8, 5))
    tb = tablas_metricas["por_duracion"]
    for metodo in ["P", "G", "R4", "R8"]:
        sub = tb[tb["metodo"] == metodo].sort_values("duration_min")
        ax.plot(sub["duration_min"], sub["induced_DR_pct"], marker="o", color=COLOR_METODO[metodo], label=metodo)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("induced_DR (%)"); ax.set_title("DR por duracion (evaluacion)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "06_dr_por_duracion")

    # 7) DR por familia
    fig, ax = plt.subplots(figsize=(9, 5))
    tb = tablas_metricas["por_familia"]
    familias = sorted(tb["family"].unique()); xf = np.arange(len(familias)); anchof = 0.2
    for i, metodo in enumerate(["P", "G", "R4", "R8"]):
        sub = tb[tb["metodo"] == metodo].set_index("family").reindex(familias)
        ax.bar(xf + (i - 1.5) * anchof, sub["induced_DR_pct"], width=anchof, color=COLOR_METODO[metodo], label=metodo)
    ax.set_xticks(xf); ax.set_xticklabels(familias, rotation=15, ha="right")
    ax.set_ylabel("induced_DR (%)"); ax.set_title("DR por familia (evaluacion)")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "07_dr_por_familia")

    # 8) DR ponderado por energia
    fig, ax = plt.subplots(figsize=(7, 5))
    tbg = tablas_metricas["global"]
    sub = tbg[tbg["familias"] == "sin_recorte_picos"].set_index("metodo").reindex(["P", "G", "R4", "R8"])
    ax.bar(sub.index, sub["energy_weighted_dr_pct"], color=[COLOR_METODO[m] for m in sub.index])
    ax.set_ylabel("DR ponderado por energia (%) -- sin recorte_picos"); ax.set_title("Cobertura energetica por metodo (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "08_dr_ponderado_energia")

    # 9) recuperacion grupo B
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(tabla_gb["metodo"], tabla_gb["pct_recuperado"], color=[COLOR_METODO.get(m, GRIS) for m in tabla_gb["metodo"]])
    ax.set_ylabel("% del Grupo B recuperado"); ax.set_title("Recuperacion del Grupo B (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "09_recuperacion_grupo_b")

    # 10) recuperacion alto impacto
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(tabla_ai["metodo"], tabla_ai["pct_recuperado"], color=[COLOR_METODO.get(m, GRIS) for m in tabla_ai["metodo"]])
    ax.set_ylabel("% de alto impacto recuperado"); ax.set_title("Recuperacion de ataques de alto impacto (evaluacion)")
    fig.tight_layout(); _guardar_fig(fig, "10_recuperacion_alto_impacto")

    # 11) complementariedad
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metodo in zip(axes, ["R4", "R8"]):
        d = comp_dict[metodo]
        ax.bar(["ambos", "solo puntual", f"solo {metodo}", "ninguno"],
               [d["detectado_ambos"], d["solo_puntual"], d["solo_cusum"], d["detectado_ninguno"]],
               color=[VERDE, GRIS, COLOR_METODO[metodo], ROJO])
        ax.set_title(f"Complementariedad puntual vs {metodo}\nretention={d['retention_rate_pct']:.1f}% excl.recovery={d['exclusive_recovery_rate_pct']:.1f}%", fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "11_complementariedad")

    # 12/13) DR y retraso vs falsos eventos/dia (curvas de sensibilidad, solo desarrollo)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for nombre_config in tabla_calib["config"].unique():
        sub = tabla_calib[tabla_calib["config"] == nombre_config].sort_values("eventos_dia_desarrollo")
        color = COLOR_METODO.get(nombre_config.split("_")[0], GRIS)
        ax1.plot(sub["eventos_dia_desarrollo"], sub["excursion_max_min"], marker="o", markersize=3, alpha=0.6, color=color)
    ax1.set_xlabel("falsos eventos/dia (desarrollo)"); ax1.set_ylabel("excursion maxima (min)"); ax1.set_yscale("log")
    ax1.set_title("Excursion vs falsos eventos/dia (desarrollo)")
    ax2.axis("off")
    fig.tight_layout(); _guardar_fig(fig, "12_13_curvas_sensibilidad_desarrollo")

    # 14/15) ejemplos temporales
    for etiqueta, ids in casos_ts.items():
        if not ids:
            continue
        fig, axes = plt.subplots(len(ids), 1, figsize=(10, 3 * len(ids)))
        if len(ids) == 1:
            axes = [axes]
        for ax, ep_id in zip(axes, ids):
            a = ctx["ataques"][ep_id]
            ini, fin = max(0, a["offset_global"] - 60), a["offset_global"] + a["duracion"] + 60
            x = ctx["particion_val_raw"][ini:fin]
            x_att = x.copy(); x_att[a["offset_global"] - ini:a["offset_global"] - ini + a["duracion"]] = a["y_tramo"]
            t = [ctx["datos"]["particiones"]["val"].index[0] + pd.Timedelta(minutes=m) for m in range(ini, fin)]
            ax.plot(t, x, color=GRIS, linewidth=1); ax.plot(t, x_att, color=ROJO, linewidth=1, alpha=0.8)
            ax.set_title(f"{ep_id} | {a['family']} | {a['duracion']}min", fontsize=8)
        fig.tight_layout(); _guardar_fig(fig, f"14_15_ejemplos_{etiqueta}")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("[1] Cargando episodios, secuencia continua y rejilla val (checkpoint existente)...")
    ctx = cargar_todo()
    serie_train_antes = ctx["scores_train"].copy(); serie_val_antes = ctx["particion_val_raw"].copy()

    print("[2] Dividiendo val en desarrollo/evaluacion y filtrando episodios contenidos en evaluacion...")
    div = dividir_desarrollo_evaluacion(ctx)
    print(f"  desarrollo: {div['dev_ini']} -> {div['dev_fin']} ({div['n_dias_dev']:.1f} dias)")
    print(f"  evaluacion: {div['eval_ini']} -> {div['eval_fin']} ({div['n_dias_eval']:.1f} dias)")
    print(f"  episodios incluidos: {len(div['incluidos'])} / {len(ctx['master_csv'])}")

    print("\n[3/4] Calculando z_t de G, R4 (2 variantes) y R8 (2 variantes) sobre todo val...")
    z_metodos = calcular_z_metodos(ctx)
    ctx["_params_g"] = z_metodos["G"]["params"]

    print("\n[5] Diagnostico de scale_t (R4/R8) en desarrollo...")
    diag_scale = diagnostico_scale(z_metodos, div, ctx["ts_val"], z_metodos["G"]["scale_global"])
    print(diag_scale["resumen"].to_string(index=False))

    print("\n[6] Seleccion de variante de escala (solo datos limpios de desarrollo)...")
    variantes = seleccionar_variante_escala(diag_scale)
    for nombre, d in variantes.items():
        print(f"  {nombre}: {d['variante']} -- {d['motivo']}")

    print("\n[11] Calibrando h para todas las configuraciones (metodo x variante x k) en desarrollo...")
    tabla_calib = calibrar_todas_las_configuraciones(z_metodos, div)

    print("\n[12] Seleccionando la configuracion final por metodo (solo datos limpios de desarrollo)...")
    seleccion = seleccionar_configuraciones_finales(tabla_calib, variantes)
    for nombre, d in seleccion.items():
        print(f"  {nombre}: config={d['config']} k={d['k']} h={d['h']:.3f}")

    print("\n[9,13,14] Simulando episodios (puntual + G + R4 + R8) sobre el subconjunto de evaluacion...")
    from src.analisis_detectabilidad import clasificar_grupos as ad_clasificar_grupos
    from src.analisis_detectabilidad import construir_tabla_episodios as ad_construir_tabla_episodios
    from src.analisis_detectabilidad import contexto_e_impacto as ad_contexto_e_impacto

    contexto_impacto = ad_contexto_e_impacto(ctx["ataques"], ctx["particion_val_raw"])
    ids_eval = div["episode_ids_incluidos"]

    umbral_p, tasa_p_dev_full, _ = calibrar_umbral_por_eventos(ctx["scores_val"], ctx["n_dias_val"], OBJETIVO_FA)
    df_p_full = ad_clasificar_grupos(ad_construir_tabla_episodios(
        ctx["tabla_ventanas"], ctx["ataques"], ctx["master_csv"], ctx["particion_val_raw"], contexto_impacto, umbral_p))
    df_p = df_p_full[df_p_full["episode_id"].isin(ids_eval)].reset_index(drop=True)

    sim_g, C_clean_g, alarm_clean_g = simular_metodo_episodios(ctx, ids_eval, z_metodos["G"]["z"],
                                                                 seleccion["G"]["k"], seleccion["G"]["h"], None)
    df_g = enriquecer_resultado(sim_g, ctx["ataques"], ctx["master_csv"], ctx["particion_val_raw"], contexto_impacto)

    z_atacados = {}
    df_cusum_r = {}
    for nombre in ("R4", "R8"):
        variante = seleccion[nombre]["variante"]
        suelo = SUELO_FRACCION * z_metodos["G"]["scale_global"] if variante == "piso" else None
        z_atacados[nombre] = construir_z_atacado_episodios(
            ctx, ids_eval, nombre, z_metodos[nombre]["semanas"], z_metodos[nombre]["center_r"], z_metodos[nombre]["scale_r"],
            z_metodos[nombre]["ref_start"], z_metodos[nombre]["ref_end"], suelo)
        z_val_final = _z_final(z_metodos, nombre, variante)
        sim_r, _, _ = simular_metodo_episodios(ctx, ids_eval, z_val_final, seleccion[nombre]["k"], seleccion[nombre]["h"], z_atacados[nombre])
        df_cusum_r[nombre] = enriquecer_resultado(sim_r, ctx["ataques"], ctx["master_csv"], ctx["particion_val_raw"], contexto_impacto)

    df_todos = {"P": df_p, "G": df_g, "R4": df_cusum_r["R4"], "R8": df_cusum_r["R8"]}
    for m, df in df_todos.items():
        _guardar_tabla(df, f"05_episode_detection_results_{m}")
    n_contaminadas_total = {nombre: int(sum(v["n_contaminadas"].sum() for v in z_atacados[nombre].values())) for nombre in ("R4", "R8")}
    print(f"  ventanas contaminadas por el propio ataque en su referencia movil: {n_contaminadas_total}")

    print("\n[15] Construyendo tablas de metricas principales...")
    tablas_metricas = tablas_por_subconjunto(df_todos)
    for nombre, t in tablas_metricas.items():
        _guardar_tabla(t, f"06_metricas_{nombre}")
    print(tablas_metricas["global"][tablas_metricas["global"]["familias"] == "sin_recorte_picos"].to_string(index=False))

    print("\n[7/20A] Falsas alarmas: desarrollo vs evaluacion (h congelado, comparacion principal)...")
    filas_fa = []
    for nombre in ("G", "R4", "R8"):
        variante = seleccion[nombre]["variante"]
        z_val_final = _z_final(z_metodos, nombre, variante)
        r = falsas_alarmas_evaluacion(z_val_final, seleccion[nombre]["k"], seleccion[nombre]["h"], div["corte_idx"], div["n_dias_eval"])
        fila = {"metodo": nombre, "k": seleccion[nombre]["k"], "h": seleccion[nombre]["h"]}
        u_dev = tabla_calib[(tabla_calib["config"] == seleccion[nombre]["config"]) & (tabla_calib["k"] == seleccion[nombre]["k"])].iloc[0]
        fila["eventos_dia_desarrollo"] = u_dev["eventos_dia_desarrollo"]
        fila.update(r); filas_fa.append(fila)
    es_alarma_p_eval = ctx["scores_val"][div["corte_idx"]:] > umbral_p
    n_eventos_p_eval = contar_eventos(es_alarma_p_eval)
    filas_fa.append({"metodo": "P", "k": float("nan"), "h": umbral_p, "eventos_dia_desarrollo": tasa_p_dev_full,
                      "n_alarmas_evaluacion": n_eventos_p_eval, "n_dias_evaluacion": div["n_dias_eval"],
                      "eventos_dia_evaluacion": n_eventos_p_eval / div["n_dias_eval"],
                      "ic95_poisson_low": sstats.chi2.ppf(0.025, 2 * n_eventos_p_eval) / 2 / div["n_dias_eval"] if n_eventos_p_eval else 0.0,
                      "ic95_poisson_high": sstats.chi2.ppf(0.975, 2 * (n_eventos_p_eval + 1)) / 2 / div["n_dias_eval"],
                      "ventanas_falsas_dia_evaluacion": float(es_alarma_p_eval.sum()) / div["n_dias_eval"],
                      "tiempo_medio_entre_alarmas_min": float("nan"), "excursion_media_min": float("nan"),
                      "excursion_mediana_min": float("nan"), "excursion_p95_min": float("nan"), "excursion_max_min": float("nan")})
    tabla_fa = pd.DataFrame(filas_fa)
    tabla_fa = pd.concat([tabla_fa[tabla_fa["metodo"] == "P"], tabla_fa[tabla_fa["metodo"] != "P"]], ignore_index=True)
    _guardar_tabla(tabla_fa, "07_falsas_alarmas_desarrollo_evaluacion")
    print(tabla_fa[["metodo", "k", "h", "eventos_dia_desarrollo", "eventos_dia_evaluacion", "excursion_max_min"]].to_string(index=False))

    print("\n[16] Grupo B (episodios de evaluacion)...")
    det_full = pd.read_csv(DET_DIR / "episodios_detectabilidad_impacto.csv")
    grupo_b_ids_eval = set(det_full[(det_full["detectability_group"] == "B") & (det_full["episode_id"].isin(ids_eval))]["episode_id"])
    filas_gb = []
    for m, df in df_todos.items():
        sub = df[df["episode_id"].isin(grupo_b_ids_eval)]
        n_rec = int(sub["induced_detected"].sum())
        filas_gb.append({"metodo": m, "n_grupo_b": len(sub), "n_recuperados": n_rec,
                          "pct_recuperado": 100 * n_rec / len(sub) if len(sub) else float("nan"),
                          "energia_recuperada_kwh": float(sub.loc[sub["induced_detected"], "hidden_energy_kwh"].sum()),
                          "retraso_mediano_min": float(sub.loc[sub["induced_detected"], "induced_detection_delay_min"].median()) if n_rec else float("nan")})
    tabla_gb = _guardar_tabla(pd.DataFrame(filas_gb), "08_grupo_b_evaluacion")
    print(tabla_gb.to_string(index=False))
    print(f"  n episodios Grupo B en evaluacion: {len(grupo_b_ids_eval)}")

    print("\n[17] Ataques no detectados de alto impacto (umbral p75 heredado del analisis global)...")
    alto_impacto_full = pd.read_csv(DET_DIR / "09_ataques_no_detectados_alto_impacto.csv")
    alto_impacto_ids_eval = set(alto_impacto_full[alto_impacto_full["episode_id"].isin(ids_eval)]["episode_id"])
    filas_ai = []
    for m, df in df_todos.items():
        sub = df[df["episode_id"].isin(alto_impacto_ids_eval)]
        n_rec = int(sub["induced_detected"].sum())
        rc_720_1440 = sub[(sub["family"] == "reduccion_constante") & (sub["duration_min"].isin([720, 1440]))]
        filas_ai.append({"metodo": m, "n_alto_impacto": len(sub), "n_recuperados": n_rec,
                          "pct_recuperado": 100 * n_rec / len(sub) if len(sub) else float("nan"),
                          "energia_recuperada_kwh": float(sub.loc[sub["induced_detected"], "hidden_energy_kwh"].sum()),
                          "n_reduccion_constante_720_1440": len(rc_720_1440),
                          "n_reduccion_constante_720_1440_recuperados": int(rc_720_1440["induced_detected"].sum()),
                          "retraso_mediano_min": float(sub.loc[sub["induced_detected"], "induced_detection_delay_min"].median()) if n_rec else float("nan"),
                          "hidden_energy_before_detection_media_kwh": float(sub["hidden_energy_before_detection_kwh"].mean())})
    tabla_ai = _guardar_tabla(pd.DataFrame(filas_ai), "alto_impacto_recuperado_final_cusum")
    print(tabla_ai.to_string(index=False))
    print(f"  n episodios alto impacto en evaluacion: {len(alto_impacto_ids_eval)}")

    print("\n[18] Complementariedad con el detector puntual...")
    comp_dict = {m: tabla_complementariedad(df_p, df_todos[m]) for m in ("G", "R4", "R8")}
    tabla_comp = pd.DataFrame(comp_dict).T.reset_index().rename(columns={"index": "metodo"})
    _guardar_tabla(tabla_comp, "09_complementariedad")
    print(tabla_comp.to_string(index=False))

    print("\n[19] Comparaciones pareadas (McNemar + bootstrap)...")
    pareada = comparacion_pareada_generica(df_todos, grupo_b_ids_eval, alto_impacto_ids_eval)
    _guardar_tabla(pareada, "10_comparacion_pareada")
    print(pareada[pareada["subconjunto"].isin(["global", "sin_recorte_picos", "grupo_b", "alto_impacto"])].to_string(index=False))

    print("\n[21] Evaluando criterios de exito (fijados antes de ver estos resultados)...")
    criterios_todos = []
    for nombre in ("R4", "R8"):
        gb_pct = tabla_gb[tabla_gb["metodo"] == nombre]["pct_recuperado"].iloc[0]
        ai_pct = tabla_ai[tabla_ai["metodo"] == nombre]["pct_recuperado"].iloc[0]
        e_p = tablas_metricas["global"][(tablas_metricas["global"]["metodo"] == "P") & (tablas_metricas["global"]["familias"] == "sin_recorte_picos")]["energy_weighted_dr_pct"].iloc[0]
        e_c = tablas_metricas["global"][(tablas_metricas["global"]["metodo"] == nombre) & (tablas_metricas["global"]["familias"] == "sin_recorte_picos")]["energy_weighted_dr_pct"].iloc[0]
        fa_eval = tabla_fa[tabla_fa["metodo"] == nombre]["eventos_dia_evaluacion"].iloc[0]
        retraso = tablas_metricas["global"][(tablas_metricas["global"]["metodo"] == nombre) & (tablas_metricas["global"]["familias"] == "sin_recorte_picos")]["retraso_mediano_min"].iloc[0]
        retention = comp_dict[nombre]["retention_rate_pct"]
        tabla_crit = evaluar_criterios_exito(nombre, gb_pct, ai_pct, e_c - e_p, fa_eval, retraso, retention, pareada)
        tabla_crit.insert(0, "metodo", nombre)
        criterios_todos.append(tabla_crit)
        print(f"  {nombre}: {tabla_crit.attrs['n_cumplidos']}/8 criterios cumplidos -> mayoria={tabla_crit.attrs['mayoria']}")
    tabla_criterios = pd.concat(criterios_todos, ignore_index=True)
    _guardar_tabla(tabla_criterios, "11_criterios_exito")

    print("\nGenerando figuras...")
    casos_ts = {"solo_R4": [], "solo_puntual": [], "adaptacion": []}
    p_idx = df_p.set_index("episode_id")["induced_detected"]
    r4_idx = df_todos["R4"].set_index("episode_id")["induced_detected"]
    solo_r4_ids = [e for e in r4_idx.index if r4_idx[e] and not p_idx.get(e, False)]
    solo_p_ids = [e for e in p_idx.index if p_idx[e] and not r4_idx.get(e, False)]
    casos_ts["solo_R4"] = solo_r4_ids[:3]
    casos_ts["solo_puntual"] = solo_p_ids[:3]
    contam_ids = [ep for ep, v in z_atacados["R4"].items() if v["n_contaminadas"].sum() > 0]
    casos_ts["adaptacion"] = contam_ids[:3]
    generar_figuras(z_metodos, div, ctx, diag_scale, tabla_calib, seleccion, tablas_metricas, tabla_fa,
                     tabla_gb, tabla_ai, comp_dict, pareada, casos_ts, df_todos)

    assert np.array_equal(serie_train_antes, ctx["scores_train"])
    assert np.array_equal(serie_val_antes, ctx["particion_val_raw"])

    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")
    return {
        "ctx": ctx, "div": div, "z_metodos": z_metodos, "diag_scale": diag_scale, "variantes": variantes,
        "tabla_calib": tabla_calib, "seleccion": seleccion, "df_todos": df_todos, "z_atacados": z_atacados,
        "tablas_metricas": tablas_metricas, "tabla_fa": tabla_fa, "tabla_gb": tabla_gb, "tabla_ai": tabla_ai,
        "comp_dict": comp_dict, "pareada": pareada, "tabla_criterios": tabla_criterios,
        "grupo_b_ids_eval": grupo_b_ids_eval, "alto_impacto_ids_eval": alto_impacto_ids_eval,
    }


if __name__ == "__main__":
    main()
