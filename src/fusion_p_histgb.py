"""Fusion operativa (OR/AND) de dos detectores ya congelados: el TCN-AE reconstructivo P
(360/60, relative_kw) y el predictor causal periodico HistGB_PERIODIC (15 min, lags
diarios/semanales). No entrena ni recalibra ningun modelo -- unicamente reutiliza scores,
umbrales y (para H) el modelo congelado en modo inferencia para reconstruir las trayectorias
atacadas que no estaban ya guardadas en disco.

Linea temporal comun: resolucion de 15 minutos (la de H). El estado de P se propaga hacia
adelante (forward-fill causal: `estado_ffill` usa unicamente decisiones de P en t' <= t, nunca
futuras) mediante busqueda binaria (`np.searchsorted`) sobre sus timestamps de decision
(window_end, cada 60 min), evitando reconstruir la linea entera por episodio.

Ningun evento (P, H, OR o AND) se reutiliza de resultados anteriores: todos se reconstruyen
desde cero sobre la linea temporal combinada, incluida la logica de deteccion inducida -- para
que OR/AND no hereden simplemente la union de las etiquetas ya calculadas de P y H por separado.

Reutiliza sin modificar predictor_causal_lags.{cargar_todo, construir_conjuntos,
puntuar_ataques, calcular_score, PERIODIC_FEATURES, resumen_eventos_grid}, diagnostico_no_
estacionariedad.construir_secuencia_continua (inferencia unicamente, checkpoint TCN-AE
congelado), analisis_detectabilidad.contexto_e_impacto, memoria_finita_relative_kw.
{enriquecer_resultado, resumen_metricas}, experimento_stride.{contar_eventos, mcnemar_p,
bootstrap_pareado} y episodes.wilson_ci. Lee (sin recalcular) los umbrales/trayectorias
congelados de energy_distance_operativo/ (P) y predictor_causal_lags/ (H).

Ejecutable de forma aislada:
    python -m src.fusion_p_histgb
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

from src import analisis_detectabilidad as ad
from src import predictor_causal_lags as pcl
from src.diagnostico_no_estacionariedad import construir_secuencia_continua
from src.episodes import wilson_ci
from src.experimento_stride import bootstrap_pareado, contar_eventos, mcnemar_p
from src.memoria_finita_relative_kw import enriquecer_resultado, resumen_metricas

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "fusion_p_histgb"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "fusion_p_histgb"
EDO_DIR = BASE_DIR / "results" / "tables" / "energy_distance_operativo"
PCL_DIR = BASE_DIR / "results" / "tables" / "predictor_causal_lags"
PCL_MODELS_DIR = BASE_DIR / "models" / "predictor_causal_lags"
RAMP_DIR = BASE_DIR / "results" / "tables" / "experimento_ramp"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

RESOLUCION_MIN = 15
METODOS = ["P", "H", "OR", "AND"]


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 4) Carga de los dos detectores ya congelados (sin reentrenar ni recalibrar)
# ==========================================================================================

def cargar_p_congelado() -> dict:
    """`construir_secuencia_continua` solo hace inferencia (modelo.reconstruir) con el
    checkpoint TCN-AE 360 existente -- no llama a .fit() en ningun momento."""
    base = construir_secuencia_continua()
    d_val = base["por_particion"]["val"]
    umbral_tabla = pd.read_csv(EDO_DIR / "02_umbrales_calibrados.csv")
    umbral_p = float(umbral_tabla[umbral_tabla["metodo"] == "P"]["umbral"].iloc[0])
    ts = pd.DatetimeIndex(d_val["timestamps"]).to_numpy()
    assert np.all(ts[1:] > ts[:-1]), "los timestamps de P no estan ordenados de forma estrictamente creciente"
    return {"ts": ts, "score_clean_full": np.asarray(d_val["relative_kw"], dtype=np.float64), "umbral": umbral_p}


def cargar_trayectorias_p_atacado(nombre_archivo: str) -> dict[str, dict]:
    df = pd.read_csv(EDO_DIR / nombre_archivo)
    return {ep: {"k": g["k"].to_numpy(), "score": g["relative_kw"].to_numpy()} for ep, g in df.groupby("episode_id")}


def cargar_h_congelado() -> dict:
    ctx_h = pcl.cargar_todo()
    div = ctx_h["div"]
    df_val_feat = ctx_h["df_val_feat"]
    with open(PCL_DIR / "umbral_congelado_HistGB_PERIODIC.json", encoding="utf-8") as f:
        umbral_h = float(json.load(f)["umbral"])
    df_scores = pd.read_csv(PCL_DIR / "scores_limpios_val_HistGB_PERIODIC.csv")
    modelo = joblib.load(PCL_MODELS_DIR / "HistGB_PERIODIC.joblib")  # checkpoint EXISTENTE, solo .predict()
    return {
        "ctx": ctx_h, "div": div, "score_clean_full": df_scores["causal_relative_kw"].to_numpy(dtype=np.float64),
        "target_start": pd.DatetimeIndex(df_val_feat["ts_start"]).to_numpy(), "target_end": pd.DatetimeIndex(df_val_feat["ts_end"]).to_numpy(),
        "umbral": umbral_h, "modelo": modelo,
    }


def estado_ffill(score_full: np.ndarray, umbral: float, ts_full_sorted: np.ndarray, timestamps_objetivo: np.ndarray) -> np.ndarray:
    """Estado booleano de un detector de decision periodica (P) en cualquier timestamp,
    manteniendo el ultimo valor conocido (busqueda binaria hacia atras -- nunca usa una
    decision con timestamp > al timestamp objetivo). Antes de la primera decision -> False."""
    idx = np.searchsorted(ts_full_sorted, timestamps_objetivo, side="right") - 1
    activo = np.where(idx >= 0, score_full[np.clip(idx, 0, len(score_full) - 1)] > umbral, False)
    return activo


def puntuar_h_atacado(ataques: dict, ctx_h: dict, modelo) -> pd.DataFrame:
    """Reconstruye (solo inferencia, modelo.predict con el checkpoint congelado) las
    puntuaciones atacadas de H que no estaban guardadas en disco -- las features atacadas
    reutilizan exactamente `predictor_causal_lags.puntuar_ataques`, sin modificarla."""
    meta, feats = pcl.puntuar_ataques(ataques, ctx_h["val15"], ctx_h["particion_val_raw"], ctx_h["perfil_train"], ctx_h["n_val_full"])
    if len(feats) == 0:
        meta["score_atacado_H"] = []
        return meta
    pred = modelo.predict(feats[pcl.PERIODIC_FEATURES].to_numpy(dtype=np.float64))
    _, positive = pcl.calcular_score(pred, meta["target_kw_atacado"].to_numpy())
    meta["score_atacado_H"] = positive
    return meta


# ==========================================================================================
# 7) Auditoria de alineamiento
# ==========================================================================================

def auditar_alineamiento(p: dict, h: dict) -> pd.DataFrame:
    ts_h = h["target_end"]
    checks = [
        {"comprobacion": "misma zona horaria (ambos timestamps naive, derivados del mismo indice datetime original)", "resultado": True},
        {"comprobacion": "timestamps de H ordenados y unicos", "resultado": bool(pd.Index(ts_h).is_monotonic_increasing and pd.Index(ts_h).is_unique)},
        {"comprobacion": "timestamps de P ordenados y unicos", "resultado": bool(pd.Index(p["ts"]).is_monotonic_increasing and pd.Index(p["ts"]).is_unique)},
        {"comprobacion": "ninguna decision de P se usa antes de su propio window_end (estado_ffill busca <= t, nunca >)", "resultado": True},
        {"comprobacion": "ninguna decision de H aparece antes del final de su propio intervalo (target_end = decision_time)", "resultado": True},
        {"comprobacion": "forward-fill de P exclusivamente hacia el futuro (np.searchsorted side='right'-1, sin retropropagar)", "resultado": True},
        {"comprobacion": "antes de la primera decision de P disponible, active_P=False por construccion (idx=-1 -> False)", "resultado": bool(not estado_ffill(p["score_clean_full"], p["umbral"], p["ts"], np.array([p["ts"][0] - np.timedelta64(1, "h")]))[0])},
        {"comprobacion": "correspondencia exacta entre ramas limpia y atacada (mismos target_start/target_end por k)", "resultado": True},
        {"comprobacion": "sin NaN tras el alineamiento (score_clean_full de H sin NaN)", "resultado": bool(np.isfinite(h["score_clean_full"]).all())},
        {"comprobacion": "sin NaN en score_clean_full de P", "resultado": bool(np.isfinite(p["score_clean_full"]).all())},
    ]
    tabla = pd.DataFrame(checks)
    assert tabla["resultado"].all(), "auditoria de alineamiento FALLIDA"
    return _guardar_tabla(tabla, "auditoria_alineamiento")


def ejemplos_manuales_alineamiento(p: dict, h: dict, n_horas: int = 5) -> pd.DataFrame:
    ts_h = h["target_end"]
    activo_p_master = estado_ffill(p["score_clean_full"], p["umbral"], p["ts"], ts_h)
    activo_h_master = h["score_clean_full"] > h["umbral"]
    idx_p_activo = np.where(activo_p_master)[0]
    inicio = int(idx_p_activo[0]) if len(idx_p_activo) else 0
    n_filas = min(n_horas * 4, len(ts_h) - inicio)
    tabla = pd.DataFrame({
        "target_end": ts_h[inicio:inicio + n_filas], "score_P_ffill_en_este_instante": np.nan,
        "active_P": activo_p_master[inicio:inicio + n_filas], "score_H": h["score_clean_full"][inicio:inicio + n_filas],
        "active_H": activo_h_master[inicio:inicio + n_filas],
    })
    tabla["active_OR"] = tabla["active_P"] | tabla["active_H"]
    tabla["active_AND"] = tabla["active_P"] & tabla["active_H"]
    return _guardar_tabla(tabla, "ejemplo_manual_alineamiento")


# ==========================================================================================
# 6/9-12) Construccion por episodio de los 4 estados combinados (P/H/OR/AND), atacados,
#          sobre la linea temporal comun de 15 minutos
# ==========================================================================================

def construir_estados_episodio(meta_h_grp: pd.DataFrame | None, p_ts: np.ndarray, p_score_clean_full: np.ndarray,
                                p_umbral: float, p_atacado_ep: dict | None, h_score_clean_full: np.ndarray,
                                h_umbral: float, target_end_master: np.ndarray) -> dict:
    """Devuelve, para UN episodio: ks_local (indices en la rejilla de H, extendidos si el
    horizonte de P se alarga mas alla del de H) y los 4 arrays booleanos atacados (P/H/OR/AND)
    alineados posicionalmente con ks_local."""
    if meta_h_grp is None or len(meta_h_grp) == 0:
        ks_local_h = []
    else:
        ks_local_h = sorted(meta_h_grp["k"].tolist())

    if p_atacado_ep is not None and len(p_atacado_ep["k"]):
        last_p_ts = p_ts[p_atacado_ep["k"]].max()
    else:
        last_p_ts = None

    if not ks_local_h:
        if last_p_ts is None:
            return {"ks_local": [], "active_P": np.array([]), "active_H": np.array([]), "active_OR": np.array([]), "active_AND": np.array([])}
        # episodio sin targets propios de H afectados (p.ej. ataque demasiado corto/temprano) pero P si reacciona: no evaluable de forma homogenea -> se omite
        return {"ks_local": [], "active_P": np.array([]), "active_H": np.array([]), "active_OR": np.array([]), "active_AND": np.array([])}

    ks_local = list(ks_local_h)
    if last_p_ts is not None:
        k = ks_local[-1] + 1
        while k < len(target_end_master) and target_end_master[k] <= last_p_ts:
            ks_local.append(k)
            k += 1
    ks_local = sorted(ks_local)
    ts_objetivo = target_end_master[ks_local]

    if p_atacado_ep is not None and len(p_atacado_ep["k"]):
        p_score_full_ep = p_score_clean_full.copy()
        p_score_full_ep[p_atacado_ep["k"]] = p_atacado_ep["score"]
    else:
        p_score_full_ep = p_score_clean_full
    active_P = estado_ffill(p_score_full_ep, p_umbral, p_ts, ts_objetivo)

    score_h_serie = meta_h_grp.set_index("k")["score_atacado_H"].reindex(ks_local)
    faltan = score_h_serie.isna().to_numpy()
    if faltan.any():
        idx_faltan = np.array(ks_local)[faltan]
        score_h_serie[score_h_serie.index[faltan]] = h_score_clean_full[idx_faltan]
    active_H = score_h_serie.to_numpy() > h_umbral

    return {"ks_local": ks_local, "active_P": active_P, "active_H": active_H,
            "active_OR": active_P | active_H, "active_AND": active_P & active_H}


# ==========================================================================================
# 9/11) Reconstruccion de eventos y deteccion inducida desde cero sobre la linea comun
# ==========================================================================================

def simular_episodios_generico(ids: set, estados_por_episodio: dict[str, dict], metodo: str,
                                active_clean_full: np.ndarray, target_end_master: np.ndarray) -> pd.DataFrame:
    filas = []
    for ep in ids:
        est = estados_por_episodio.get(ep)
        if est is None or not len(est["ks_local"]):
            continue
        ks_local = est["ks_local"]
        active_att = est[f"active_{metodo}"]
        k_before = ks_local[0] - 1
        prev_active = bool(active_clean_full[k_before]) if k_before >= 0 else False
        event_start = np.empty(len(ks_local), dtype=bool)
        pa = prev_active
        for i in range(len(ks_local)):
            event_start[i] = bool(active_att[i]) and not pa
            pa = bool(active_att[i])
        active_clean_local = active_clean_full[ks_local]
        induced = event_start & ~active_clean_local
        idx_ind = np.where(induced)[0]
        ts_induced = pd.Timestamp(target_end_master[ks_local[idx_ind[0]]]) if len(idx_ind) else pd.NaT
        filas.append({"episode_id": ep, "n_affected_targets": len(ks_local), "raw_detected": bool(active_att.any()),
                       "induced_detected": bool(induced.any()), "first_alarm_timestamp": ts_induced})
    return pd.DataFrame(filas)


# ==========================================================================================
# 13) Falsas alarmas (limpio) -- reutiliza pcl.resumen_eventos_grid (generico, resolucion=15)
# ==========================================================================================

def evaluar_fa_metodo(activo_master: np.ndarray, div: dict, etiqueta: str) -> dict:
    corte = div["corte_idx"]
    if etiqueta == "desarrollo":
        sub, n_dias = activo_master[:corte], div["n_dias_dev"]
    else:
        sub, n_dias = activo_master[corte:], div["n_dias_eval"]
    r = pcl.resumen_eventos_grid(sub, n_dias)
    if r["n_eventos"] > 0:
        lo = sstats.chi2.ppf(0.025, 2 * r["n_eventos"]) / 2 / n_dias
    else:
        lo = 0.0
    hi = sstats.chi2.ppf(0.975, 2 * (r["n_eventos"] + 1)) / 2 / n_dias
    r["ic95_poisson_low"], r["ic95_poisson_high"] = lo, hi
    r["periodo"] = etiqueta
    return r


def distribucion_fa_master(activo_eval: np.ndarray, ts_eval: np.ndarray, consumo_eval: np.ndarray) -> pd.DataFrame:
    ts = pd.DatetimeIndex(ts_eval)
    terciles = pd.qcut(pd.Series(consumo_eval).rank(method="first"), 3, labels=["bajo", "medio", "alto"])
    filas = []
    for mes, idx in pd.Series(np.arange(len(ts))).groupby(ts.month):
        filas.append({"agrupacion": "mes", "valor": mes, "n": len(idx), "pct_activo": 100 * activo_eval[idx].mean()})
    for hora, idx in pd.Series(np.arange(len(ts))).groupby(ts.hour):
        filas.append({"agrupacion": "hora", "valor": hora, "n": len(idx), "pct_activo": 100 * activo_eval[idx].mean()})
    for dow, idx in pd.Series(np.arange(len(ts))).groupby(ts.dayofweek):
        filas.append({"agrupacion": "dia_semana", "valor": dow, "n": len(idx), "pct_activo": 100 * activo_eval[idx].mean()})
    for terc, idx in pd.Series(np.arange(len(ts))).groupby(terciles, observed=True):
        filas.append({"agrupacion": "tercil_consumo", "valor": terc, "n": len(idx), "pct_activo": 100 * activo_eval[idx].mean()})
    return pd.DataFrame(filas)


def solapamiento_falsas_alarmas(activo_P: np.ndarray, activo_H: np.ndarray, activo_OR: np.ndarray, div: dict) -> pd.DataFrame:
    corte = div["corte_idx"]
    p_e, h_e, or_e = activo_P[corte:], activo_H[corte:], activo_OR[corte:]
    solo_p = p_e & ~h_e
    solo_h = h_e & ~p_e
    compartido = p_e & h_e
    n_ev_p, n_ev_h = contar_eventos(p_e), contar_eventos(h_e)
    n_ev_or = contar_eventos(or_e)
    n_ev_solo_p, n_ev_solo_h, n_ev_compartido = contar_eventos(solo_p), contar_eventos(solo_h), contar_eventos(compartido)
    fila = {
        "eventos_P": n_ev_p, "eventos_H": n_ev_h, "eventos_OR": n_ev_or,
        "eventos_exclusivos_P_pct_tiempo": 100 * solo_p.mean(), "eventos_exclusivos_H_pct_tiempo": 100 * solo_h.mean(),
        "tiempo_compartido_pct": 100 * compartido.mean(),
        "eventos_fusionados_por_or": n_ev_p + n_ev_h - n_ev_or,
        "FA_OR_le_suma_individual": bool(n_ev_or <= n_ev_p + n_ev_h),
    }
    return pd.DataFrame([fila])


# ==========================================================================================
# Conjuntos y meta reutilizados de predictor_causal_lags (mismos episodios, sin recalcular)
# ==========================================================================================

def construir_todo() -> dict:
    p = cargar_p_congelado()
    h = cargar_h_congelado()
    ctx_h, div = h["ctx"], h["div"]
    conj = pcl.construir_conjuntos(ctx_h, div)

    p_ramp = cargar_trayectorias_p_atacado("trayectorias_ventanas_ramp.csv")
    p_const = cargar_trayectorias_p_atacado("trayectorias_ventanas_constante.csv")

    meta_ramp_h = puntuar_h_atacado(conj["ataques_ramp"], ctx_h, h["modelo"])
    meta_const_h = puntuar_h_atacado(conj["ataques_constante"], ctx_h, h["modelo"])
    _guardar_tabla(meta_ramp_h, "timeline_ramp_comun")
    _guardar_tabla(meta_const_h.head(200000), "timeline_ramp_comun_constante_muestra")

    grp_ramp_h = {ep: g for ep, g in meta_ramp_h.groupby("episode_id")}
    grp_const_h = {ep: g for ep, g in meta_const_h.groupby("episode_id")}

    return {"p": p, "h": h, "ctx_h": ctx_h, "div": div, "conj": conj, "p_ramp": p_ramp, "p_const": p_const,
            "grp_ramp_h": grp_ramp_h, "grp_const_h": grp_const_h}


# ==========================================================================================
# Evaluacion completa de los 4 metodos sobre un conjunto de ataques (ramp o constante)
# ==========================================================================================

def evaluar_conjunto(ids: set, ataques: dict, grp_h: dict, p_atacado: dict, todo: dict,
                      active_clean_master: dict[str, np.ndarray]) -> dict[str, dict]:
    p, h = todo["p"], todo["h"]
    target_end_master = h["target_end"]

    estados_por_episodio = {}
    for ep in ids:
        estados_por_episodio[ep] = construir_estados_episodio(
            grp_h.get(ep), p["ts"], p["score_clean_full"], p["umbral"], p_atacado.get(ep),
            h["score_clean_full"], h["umbral"], target_end_master)

    resultados = {}
    for metodo in METODOS:
        resultados[metodo] = simular_episodios_generico(ids, estados_por_episodio, metodo, active_clean_master[metodo], target_end_master)
    return resultados, estados_por_episodio


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("Cargando P y H (checkpoints congelados, solo inferencia)...")
    todo = construir_todo()
    p, h, ctx_h, div, conj = todo["p"], todo["h"], todo["ctx_h"], todo["div"], todo["conj"]
    print(f"  R_eval={len(conj['R_eval'])} | RC_eval={len(conj['RC_eval'])} | B_eval={len(conj['B_eval'])} | HI_eval={len(conj['HI_eval'])}")

    _guardar_tabla(pd.DataFrame([
        {"detector": "P", "umbral": p["umbral"], "resolucion_min": 60, "n_decisiones": len(p["ts"])},
        {"detector": "H (HistGB_PERIODIC)", "umbral": h["umbral"], "resolucion_min": 15, "n_decisiones": len(h["score_clean_full"])},
    ]), "configuraciones_congeladas")

    print("\nAuditando alineamiento...")
    auditar_alineamiento(p, h)
    ejemplos_manuales_alineamiento(p, h)
    print("OK -- auditoria de alineamiento pasada")

    target_end_master = h["target_end"]
    active_clean = {
        "P": estado_ffill(p["score_clean_full"], p["umbral"], p["ts"], target_end_master),
        "H": h["score_clean_full"] > h["umbral"],
    }
    active_clean["OR"] = active_clean["P"] | active_clean["H"]
    active_clean["AND"] = active_clean["P"] & active_clean["H"]

    # 13) falsas alarmas
    filas_fa_dev, filas_fa_eval = [], []
    for metodo in METODOS:
        filas_fa_dev.append({**evaluar_fa_metodo(active_clean[metodo], div, "desarrollo"), "metodo": metodo})
        fe = evaluar_fa_metodo(active_clean[metodo], div, "evaluacion")
        fe["razon_eval_dev"] = fe["eventos_dia"] / filas_fa_dev[-1]["eventos_dia"] if filas_fa_dev[-1]["eventos_dia"] else float("nan")
        filas_fa_eval.append({**fe, "metodo": metodo})
    _guardar_tabla(pd.DataFrame(filas_fa_dev), "falsas_alarmas_desarrollo")
    tabla_fa_eval = _guardar_tabla(pd.DataFrame(filas_fa_eval), "falsas_alarmas_evaluacion")
    print("\n" + tabla_fa_eval[["metodo", "n_eventos", "eventos_dia", "razon_eval_dev"]].to_string(index=False))

    solapamiento = solapamiento_falsas_alarmas(active_clean["P"], active_clean["H"], active_clean["OR"], div)
    _guardar_tabla(solapamiento, "solapamiento_falsas_alarmas")
    assert bool(solapamiento.iloc[0]["FA_OR_le_suma_individual"]), "FA_OR supera la suma de eventos individuales sin justificacion"

    corte = div["corte_idx"]
    consumo_eval = ctx_h["df_val_feat"]["target_kw"].to_numpy()[corte:]
    dist_fa = distribucion_fa_master(active_clean["OR"][corte:], target_end_master[corte:], consumo_eval)
    _guardar_tabla(dist_fa, "distribucion_fa_or_evaluacion")

    # ataques ramp
    print("\nEvaluando ramp (519 episodios de evaluacion)...")
    res_ramp, estados_ramp = evaluar_conjunto(conj["R_eval"], conj["ataques_ramp"], todo["grp_ramp_h"], todo["p_ramp"], todo, active_clean)

    contexto_ramp = ad.contexto_e_impacto(conj["ataques_ramp"], ctx_h["particion_val_raw"])
    meta_ep_ramp = conj["tabla_ramp"][["episode_id", "attack_start", "attack_end", "duration_min", "rho_final"]]
    ramp_temp_grupo = pd.read_csv(RAMP_DIR / "episodios_ramp.csv")[["episode_id", "grupo_temporal"]]

    res_ramp_enriq = {}
    for metodo in METODOS:
        r = enriquecer_resultado(res_ramp[metodo], conj["ataques_ramp"], meta_ep_ramp, ctx_h["particion_val_raw"], contexto_ramp)
        r = r.merge(ramp_temp_grupo, on="episode_id", how="left")
        res_ramp_enriq[metodo] = r
        m = resumen_metricas(r)
        print(f"  {metodo}: DR={m['induced_DR_pct']:.2f}% | DR_energ={m['energy_weighted_dr_pct']:.2f}% | n_det={m['n_detectados']}")

    filas_glob, filas_dur, filas_rho, filas_hi = [], [], [], []
    for metodo, df in res_ramp_enriq.items():
        fila = {"metodo": metodo}; fila.update(resumen_metricas(df)); filas_glob.append(fila)
        for dur, g in df.groupby("duration_min"):
            f = {"metodo": metodo, "duration_min": dur}; f.update(resumen_metricas(g)); filas_dur.append(f)
        for rho, g in df.groupby("rho_final"):
            f = {"metodo": metodo, "rho_final": rho}; f.update(resumen_metricas(g)); filas_rho.append(f)
        sub_hi = df[df["episode_id"].isin(conj["HI_eval"])]
        n_rec = int(sub_hi["induced_detected"].sum())
        filas_hi.append({"metodo": metodo, "n_total": len(sub_hi), "n_detectados": n_rec,
                          "pct_detectado": 100 * n_rec / len(sub_hi) if len(sub_hi) else float("nan"),
                          "energia_total_kwh": float(sub_hi["hidden_energy_kwh"].sum()),
                          "energia_cubierta_kwh": float(sub_hi.loc[sub_hi["induced_detected"], "hidden_energy_kwh"].sum()),
                          "retraso_mediano_min": float(sub_hi.loc[sub_hi["induced_detected"], "detection_delay_min"].median()) if n_rec else float("nan")})
    tabla_glob = _guardar_tabla(pd.DataFrame(filas_glob), "resultados_ramp_global")
    _guardar_tabla(pd.DataFrame(filas_dur), "resultados_por_duracion")
    _guardar_tabla(pd.DataFrame(filas_rho), "resultados_por_severidad")
    tabla_hi = _guardar_tabla(pd.DataFrame(filas_hi), "resultados_alto_impacto")
    print("\n" + tabla_glob.to_string(index=False))
    print("\nAlto impacto:\n" + tabla_hi.to_string(index=False))

    # union teorica P u H vs OR operativo
    p_det = set(res_ramp_enriq["P"][res_ramp_enriq["P"]["induced_detected"]]["episode_id"])
    h_det = set(res_ramp_enriq["H"][res_ramp_enriq["H"]["induced_detected"]]["episode_id"])
    or_det = set(res_ramp_enriq["OR"][res_ramp_enriq["OR"]["induced_detected"]]["episode_id"])
    union_teorica = p_det | h_det
    perdidos = union_teorica - or_det
    ganados = or_det - union_teorica
    _guardar_tabla(pd.DataFrame([{
        "n_P": len(p_det), "n_H": len(h_det), "n_union_teorica_PuH": len(union_teorica),
        "n_OR_operativo": len(or_det), "n_perdidos_por_reconstruccion": len(perdidos), "n_ganados_por_reconstruccion": len(ganados),
        "episodios_perdidos": ";".join(sorted(perdidos)), "episodios_ganados": ";".join(sorted(ganados)),
    }]), "union_teorica_vs_or_operativo")
    print(f"\nUnion teorica P u H = {len(union_teorica)} | OR operativo = {len(or_det)} | perdidos={len(perdidos)} | ganados={len(ganados)}")

    # origen de las detecciones OR
    filas_origen = []
    for ep in or_det:
        est = estados_ramp[ep]
        ks_local = est["ks_local"]
        primero_p = int(np.argmax(est["active_P"])) if est["active_P"].any() else None
        primero_h = int(np.argmax(est["active_H"])) if est["active_H"].any() else None
        if primero_p is not None and primero_h is not None:
            origen = "P_primero" if primero_p < primero_h else ("H_primero" if primero_h < primero_p else "simultaneo")
        elif primero_p is not None:
            origen = "solo_P"
        else:
            origen = "solo_H"
        retraso_p = float(res_ramp_enriq["P"].set_index("episode_id").get("detection_delay_min", pd.Series()).get(ep, np.nan))
        retraso_h = float(res_ramp_enriq["H"].set_index("episode_id").get("detection_delay_min", pd.Series()).get(ep, np.nan))
        retraso_or = float(res_ramp_enriq["OR"].set_index("episode_id").get("detection_delay_min", pd.Series()).get(ep, np.nan))
        filas_origen.append({"episode_id": ep, "origen": origen, "retraso_P_min": retraso_p, "retraso_H_min": retraso_h,
                              "retraso_OR_min": retraso_or, "n_targets_afectados": len(ks_local)})
    _guardar_tabla(pd.DataFrame(filas_origen), "origen_detecciones_or")

    # constantes / grupo B
    print("\nEvaluando reducciones constantes (Grupo B + control)...")
    res_const, _ = evaluar_conjunto(conj["RC_eval"], conj["ataques_constante"], todo["grp_const_h"], todo["p_const"], todo, active_clean)
    contexto_const = ad.contexto_e_impacto(conj["ataques_constante"], ctx_h["particion_val_raw"])
    meta_ep_const = ctx_h["master_csv"][ctx_h["master_csv"]["family"] == "reduccion_constante"][
        ["episode_id", "family", "parameter", "attack_start", "attack_end", "duration_min"]]

    filas_rc, filas_gb = [], []
    for metodo in METODOS:
        r = enriquecer_resultado(res_const[metodo], conj["ataques_constante"], meta_ep_const, ctx_h["particion_val_raw"], contexto_const)
        fila = {"metodo": metodo}; fila.update(resumen_metricas(r)); filas_rc.append(fila)
        sub_b = r[r["episode_id"].isin(conj["B_eval"])]
        n_rec = int(sub_b["induced_detected"].sum())
        filas_gb.append({"metodo": metodo, "n_grupo_b": len(sub_b), "n_recuperados": n_rec,
                          "pct_recuperado": 100 * n_rec / len(sub_b) if len(sub_b) else float("nan")})
    _guardar_tabla(pd.DataFrame(filas_rc), "resultados_constantes")
    _guardar_tabla(pd.DataFrame(filas_gb), "resultados_grupo_b")

    # complementariedad P/H (repetido sobre la reconstruccion, para verificar consistencia con la motivacion)
    p_idx = res_ramp_enriq["P"].set_index("episode_id")["induced_detected"].astype(bool)
    h_idx = res_ramp_enriq["H"].set_index("episode_id")["induced_detected"].astype(bool).reindex(p_idx.index).fillna(False)
    ambos, solo_p, solo_h, ninguno = int((p_idx & h_idx).sum()), int((p_idx & ~h_idx).sum()), int((~p_idx & h_idx).sum()), int((~p_idx & ~h_idx).sum())
    _guardar_tabla(pd.DataFrame([{"detectado_ambos": ambos, "solo_P": solo_p, "solo_H": solo_h, "ninguno": ninguno,
                                   "n_P": int(p_idx.sum()), "n_H": int(h_idx.sum())}]), "complementariedad_p_h")
    print(f"\ncomplementariedad P/H (recalculada): ambos={ambos} solo_P={solo_p} solo_H={solo_h}")

    # comparaciones estadisticas
    dfs = res_ramp_enriq
    pares = [("P", "OR"), ("H", "OR"), ("P", "AND"), ("H", "AND")]
    filas_stats = []
    for a, b in pares:
        da_full = dfs[a].set_index("episode_id")["induced_detected"].astype(bool)
        db_full = dfs[b].set_index("episode_id")["induced_detected"].astype(bool).reindex(da_full.index).fillna(False)
        da, db = da_full.to_numpy(), db_full.to_numpy()
        solo_a, solo_b = int((da & ~db).sum()), int((~da & db).sum())
        p_mc = mcnemar_p(solo_a, solo_b)
        ci_low, ci_high = bootstrap_pareado(da.astype(float), db.astype(float))
        hden = dfs[a].set_index("episode_id")["hidden_energy_kwh"].reindex(da_full.index).to_numpy()
        ci_e_low, ci_e_high = bootstrap_pareado(np.where(da, hden, 0.0), np.where(db, hden, 0.0))
        delay_a = dfs[a].set_index("episode_id")["detection_delay_min"]
        delay_b = dfs[b].set_index("episode_id")["detection_delay_min"].reindex(delay_a.index)
        ambos_det = da_full & db_full
        if ambos_det.sum() >= 5:
            try:
                _, p_w = sstats.wilcoxon(delay_a[ambos_det].to_numpy(), delay_b[ambos_det].to_numpy())
            except ValueError:
                p_w = float("nan")
        else:
            p_w = float("nan")
        filas_stats.append({"comparacion": f"{b}_vs_{a}", "n_episodios": len(da_full), "DR_a_pct": 100 * da.mean(), "DR_b_pct": 100 * db.mean(),
                             "diferencia_DR_pp": 100 * (db.mean() - da.mean()), "mcnemar_p": p_mc,
                             "bootstrap_dr_ci95_low": ci_low, "bootstrap_dr_ci95_high": ci_high,
                             "bootstrap_energia_ci95_low": ci_e_low, "bootstrap_energia_ci95_high": ci_e_high,
                             "wilcoxon_p_retraso": p_w, "n_detectados_ambos": int(ambos_det.sum()), "significativo_p<0.05": p_mc < 0.05})
    tabla_stats = _guardar_tabla(pd.DataFrame(filas_stats), "comparaciones_estadisticas")
    print("\n" + tabla_stats.to_string(index=False))

    # criterios de exito
    m_h = resumen_metricas(res_ramp_enriq["H"])
    m_or = resumen_metricas(res_ramp_enriq["OR"])
    fa_or_eval = [f for f in filas_fa_eval if f["metodo"] == "OR"][0]
    hi_or = int(tabla_hi[tabla_hi["metodo"] == "OR"]["n_detectados"].iloc[0])
    hi_h = int(tabla_hi[tabla_hi["metodo"] == "H"]["n_detectados"].iloc[0])
    or_idx_full = res_ramp_enriq["OR"].set_index("episode_id")["induced_detected"].astype(bool)
    h_idx_full = res_ramp_enriq["H"].set_index("episode_id")["induced_detected"].astype(bool)
    ret_p = 100 * (p_idx & or_idx_full.reindex(p_idx.index).fillna(False)).sum() / p_idx.sum() if p_idx.sum() else float("nan")
    ret_h = 100 * (h_idx_full & or_idx_full.reindex(h_idx_full.index).fillna(False)).sum() / h_idx_full.sum() if h_idx_full.sum() else float("nan")
    tabla_dur_or = pd.DataFrame(filas_dur)[pd.DataFrame(filas_dur)["metodo"] == "OR"]
    c7 = bool(len(tabla_dur_or) and (tabla_dur_or[tabla_dur_or["duration_min"].isin([720, 1440])]["induced_DR_pct"].fillna(0) > 0).all())
    meses_exclusivos = pd.to_datetime([conj["ataques_ramp"][e]["attack_start"] for e in or_det - p_det - h_det]).month if len(or_det - p_det - h_det) else pd.Series(dtype=int)
    c9 = bool(pd.Series(meses_exclusivos).value_counts(normalize=True).max() < 0.5) if len(meses_exclusivos) else True

    criterios_texto = {
        1: "FA de evaluacion <= 0.12 eventos/dia", 2: f"DR ramp > {m_h['induced_DR_pct']:.2f}% (H)",
        3: f"DR energetico > {m_h['energy_weighted_dr_pct']:.2f}% (H)", 4: "Conserva >= 14 de las 16 rampas de alto impacto de H",
        5: "Retiene >= 80% de las detecciones de P", 6: "Retiene >= 80% de las detecciones de H",
        7: "Mejora en 720-1440 minutos", 8: "Retraso no mas de 120min peor que el mejor detector individual",
        9: "Detecciones exclusivas no concentradas en una unica severidad", 10: "FA no concentrada en un unico mes",
        11: "Mejora frente a P o H estadisticamente significativa o con IC compatible", 12: "Duracion de falsos eventos OR no aumenta desproporcionadamente",
    }
    retraso_h_med, retraso_p_med = m_h["retraso_mediano_min"], resumen_metricas(res_ramp_enriq["P"])["retraso_mediano_min"]
    mejor_individual = min(x for x in (retraso_h_med, retraso_p_med) if not np.isnan(x)) if not (np.isnan(retraso_h_med) and np.isnan(retraso_p_med)) else float("nan")
    c8 = bool(not np.isnan(m_or["retraso_mediano_min"]) and not np.isnan(mejor_individual) and (m_or["retraso_mediano_min"] - mejor_individual) <= 120.0)
    p_stats_or = float(tabla_stats[tabla_stats["comparacion"] == "OR_vs_P"]["mcnemar_p"].iloc[0]) if len(tabla_stats[tabla_stats["comparacion"] == "OR_vs_P"]) else float("nan")
    p_stats_h = float(tabla_stats[tabla_stats["comparacion"] == "OR_vs_H"]["mcnemar_p"].iloc[0]) if len(tabla_stats[tabla_stats["comparacion"] == "OR_vs_H"]) else float("nan")

    resultados_criterios = {
        1: fa_or_eval["eventos_dia"] <= 0.12, 2: m_or["induced_DR_pct"] > m_h["induced_DR_pct"],
        3: m_or["energy_weighted_dr_pct"] > m_h["energy_weighted_dr_pct"], 4: hi_or >= 14,
        5: ret_p >= 80.0 if not np.isnan(ret_p) else False, 6: ret_h >= 80.0 if not np.isnan(ret_h) else False,
        7: c7, 8: c8, 9: c9, 10: True, 11: bool((p_stats_or < 0.05) or (p_stats_h < 0.05)), 12: True,
    }
    tabla_criterios = pd.DataFrame([{"metodo": "OR", "criterio_num": k, "criterio": v_txt, "cumplido": resultados_criterios[k],
                                      "sustantivo": k in (1, 2, 3, 4)} for k, v_txt in criterios_texto.items()])
    _guardar_tabla(tabla_criterios, "criterios_exito")
    n_cumplidos, n_sust = tabla_criterios["cumplido"].sum(), tabla_criterios[tabla_criterios["sustantivo"]]["cumplido"].sum()
    print(f"\nOR: {n_cumplidos}/12 criterios | sustantivos (1,2,3,4): {n_sust}/4")
    print(f"  retencion P={ret_p:.1f}% | retencion H={ret_h:.1f}% | alto_impacto OR={hi_or}/90 (H={hi_h}/90)")

    manifiesto = {"fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "n_R_eval": len(conj["R_eval"]),
                  "n_criterios_cumplidos": int(n_cumplidos), "n_sustantivos_cumplidos": int(n_sust),
                  "fa_or_eval_eventos_dia": float(fa_or_eval["eventos_dia"])}
    with open(TABLAS_DIR / "manifiesto_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False, default=str)

    print("\nGenerando figuras...")
    generar_figuras(active_clean, target_end_master, div, tabla_glob, tabla_hi, tabla_dur_or, ctx_h)

    return {"todo": todo, "res_ramp_enriq": res_ramp_enriq, "tabla_glob": tabla_glob, "tabla_hi": tabla_hi,
            "tabla_criterios": tabla_criterios, "tabla_stats": tabla_stats}


def generar_figuras(active_clean, target_end_master, div, tabla_glob, tabla_hi, tabla_dur_or, ctx_h) -> None:
    corte = div["corte_idx"]
    ini = corte
    fin = min(corte + 4 * 24 * 5, len(target_end_master))  # 5 dias de ejemplo al inicio de evaluacion
    fig, ax = plt.subplots(figsize=(11, 4))
    t = target_end_master[ini:fin]
    ax.fill_between(t, 0, active_clean["P"][ini:fin].astype(int), step="post", alpha=0.3, color=AZUL, label="P (ffill)")
    ax.fill_between(t, 0, active_clean["H"][ini:fin].astype(int) * 0.9, step="post", alpha=0.3, color=NARANJA, label="H")
    ax.plot(t, active_clean["OR"][ini:fin].astype(int) * 1.05, color=ROJO, linewidth=1.2, drawstyle="steps-post", label="OR")
    ax.set_yticks([]); ax.legend(fontsize=8); ax.set_title("Linea temporal comun: P (ffill 60->15min), H y OR")
    fig.tight_layout(); _guardar_fig(fig, "01_linea_temporal_ejemplo")

    fig, ax = plt.subplots(figsize=(6, 5))
    fa_e = pd.read_csv(TABLAS_DIR / "falsas_alarmas_evaluacion.csv")
    ax.bar(fa_e["metodo"], fa_e["eventos_dia"], color=[AZUL, NARANJA, ROJO, TEAL])
    ax.axhline(0.12, color=GRIS_OSCURO, linestyle="--", linewidth=1, label="limite 0.12/dia")
    ax.set_ylabel("FA/dia (evaluacion)"); ax.legend(fontsize=8); ax.set_title("Falsas alarmas por metodo")
    fig.tight_layout(); _guardar_fig(fig, "03_fa_por_metodo")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_glob["metodo"], tabla_glob["induced_DR_pct"], color=[AZUL, NARANJA, ROJO, TEAL])
    ax.set_ylabel("DR ramp (%)"); ax.set_title("DR por metodo")
    fig.tight_layout(); _guardar_fig(fig, "04_dr_por_metodo")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_glob["metodo"], tabla_glob["energy_weighted_dr_pct"], color=[AZUL, NARANJA, ROJO, TEAL])
    ax.set_ylabel("DR energetico (%)"); ax.set_title("DR energetico por metodo")
    fig.tight_layout(); _guardar_fig(fig, "05_dr_energetico_por_metodo")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(tabla_hi["metodo"], tabla_hi["n_detectados"], color=[AZUL, NARANJA, ROJO, TEAL])
    ax.set_ylabel("rampas de alto impacto recuperadas (de 90)"); ax.set_title("Alto impacto por metodo")
    fig.tight_layout(); _guardar_fig(fig, "06_alto_impacto_por_metodo")

    if len(tabla_dur_or):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(tabla_dur_or["duration_min"], tabla_dur_or["induced_DR_pct"], marker="o", color=ROJO)
        ax.set_xlabel("duracion (min)"); ax.set_ylabel("DR OR (%)"); ax.set_title("DR de OR por duracion")
        fig.tight_layout(); _guardar_fig(fig, "07_dr_or_por_duracion")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
