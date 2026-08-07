"""Analisis de detectabilidad e impacto energetico de los ataques no detectados.

Es unicamente un analisis sobre la configuracion base ya congelada: ventana=360, stride=60,
score=relative_kw, checkpoint tcn_ae_ventana360.pt existente, umbral calibrado a 0.06 falsos
eventos/dia, evaluacion continuous_timeline, split=val.

No modifica el modelo, el score, el umbral, las funciones de ataque ni la config base.
Reutiliza sin modificar: base_relative.cargar_ataques_y_modelo_360 (episodios exactos de
episodes_master_val.csv), stride_relative.puntuar_stride_relative (score relative_kw en la
rejilla de stride=60), y varias funciones de experimento_stride (construir_rejilla,
calibrar_umbral_por_eventos, _ventanas_solapadas, _ventana_atacada, contar_eventos).

Ejecutable de forma aislada:
    python -m src.analisis_detectabilidad
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from src.base_relative import calcular_relative_kw, cargar_ataques_y_modelo_360
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import wilson_ci
from src.experimento_stride import (
    _ventana_atacada,
    _ventanas_solapadas,
    calibrar_umbral_por_eventos,
    construir_rejilla,
)
from src.normalization import aplicar_zscore, revertir_zscore
from src.stride_relative import puntuar_stride_relative

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "analisis_detectabilidad"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "analisis_detectabilidad"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
COLOR_GRUPO = {"A": ROJO, "B": NARANJA, "C": "#e8c547", "D": VERDE}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

TAMANO_VENTANA = 360
STRIDE = 60
SCORE = "relative_kw"
OBJETIVO_PRINCIPAL_VALOR = 0.06
OBJETIVOS_SENSIBILIDAD = {"sec_0.05": 0.05, "principal_0.06": 0.06, "sec_0.10": 0.10, "sec_0.25": 0.25}
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
EPS_ENERGIA = 1e-9


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 0) Setup: episodios exactos + rejilla/score de la configuracion base (360/60/relative_kw)
# ==========================================================================================

def cargar_todo():
    config, datos, modelo, ataques, master_csv = cargar_ataques_y_modelo_360()
    params_norm = datos["params_norm"]
    particion_val = datos["particiones"]["val"]
    particion_val_raw = particion_val[COLUMNA_OBJETIVO].to_numpy().copy()
    n_dias = len(particion_val_raw) / 1440.0
    grid_60 = construir_rejilla(particion_val, STRIDE)
    tabla_ventanas, scores_limpios_60 = puntuar_stride_relative(
        ataques, particion_val_raw, grid_60, modelo, params_norm, STRIDE)
    return {
        "config": config, "datos": datos, "modelo": modelo, "ataques": ataques, "master_csv": master_csv,
        "params_norm": params_norm, "particion_val": particion_val, "particion_val_raw": particion_val_raw,
        "n_dias": n_dias, "grid_60": grid_60, "tabla_ventanas": tabla_ventanas,
        "scores_limpios_60": scores_limpios_60,
    }


# ==========================================================================================
# 3) Tabla maestra enriquecida -- contexto limpio + impacto del ataque (secciones 3, sin la
#    parte de respuesta del detector, que depende del umbral -> ver construir_tabla_episodios)
# ==========================================================================================

_RE_NUM = re.compile(r"-?\d+\.?\d*")


def _severidad_numerica(parameter: str) -> float:
    m = _RE_NUM.search(str(parameter))
    return float(m.group()) if m else float("nan")


def contexto_e_impacto(ataques: dict, particion_val_raw: np.ndarray) -> pd.DataFrame:
    """Contexto de la senal limpia e impacto energetico del ataque, calculado UNICAMENTE
    sobre [attack_start, attack_end) -- una fila por episodio."""
    filas = []
    for episode_id, a in ataques.items():
        offset, dur = a["offset_global"], a["duracion"]
        clean = particion_val_raw[offset:offset + dur].astype(np.float64)
        attacked = np.asarray(a["y_tramo"], dtype=np.float64)
        reduccion = clean - attacked  # deductivo: siempre >= 0 (ver attacks.py)

        clean_energy_kwh = float(clean.sum() / 60.0)
        attacked_energy_kwh = float(attacked.sum() / 60.0)
        hidden_energy_kwh = float(reduccion.sum() / 60.0)
        hidden_energy_pct = 100.0 * hidden_energy_kwh / clean_energy_kwh if clean_energy_kwh > EPS_ENERGIA else float("nan")
        modified_samples = int((np.abs(reduccion) > 1e-9).sum())

        filas.append({
            "episode_id": episode_id,
            "clean_mean_kw": float(clean.mean()), "clean_median_kw": float(np.median(clean)),
            "clean_max_kw": float(clean.max()), "clean_std_kw": float(clean.std()),
            "clean_energy_kwh": clean_energy_kwh,
            "attacked_energy_kwh": attacked_energy_kwh,
            "hidden_energy_kwh": hidden_energy_kwh,
            "hidden_energy_pct": hidden_energy_pct,
            "mean_reduction_kw": float(reduccion.mean()), "max_reduction_kw": float(reduccion.max()),
            "modified_samples": modified_samples,
            "modified_fraction_of_episode": modified_samples / dur,
        })
    return pd.DataFrame(filas)


# ==========================================================================================
# 3 (cont.) + 4) Respuesta del detector a un umbral dado, y energia ocultada antes/despues
#    de la primera alarma inducida. Reutilizable para el umbral principal (0.06) y para el
#    barrido de sensibilidad de la seccion 12.
# ==========================================================================================

def _respuesta_detector(tabla_ventanas: pd.DataFrame, umbral: float) -> pd.DataFrame:
    t = tabla_ventanas.copy()
    t["window_end"] = pd.to_datetime(t["window_end"])
    t["clean_is_alarm"] = t["clean_score"] > umbral
    t["attacked_is_alarm"] = t["attacked_score"] > umbral
    t["attack_induced_alarm"] = t["attacked_is_alarm"] & ~t["clean_is_alarm"]
    t["score_delta"] = t["attacked_score"] - t["clean_score"]

    agg = t.groupby("episode_id").agg(
        n_affected_windows=("k", "size"),
        max_clean_score=("clean_score", "max"),
        max_attacked_score=("attacked_score", "max"),
        max_score_delta=("score_delta", "max"),
        median_score_delta=("score_delta", "median"),
        mean_score_delta=("score_delta", "mean"),
        induced_detected=("attack_induced_alarm", "any"),
    ).reset_index()
    pct_pos = t.groupby("episode_id")["score_delta"].apply(lambda s: 100.0 * (s > 0).mean())
    pct_neg = t.groupby("episode_id")["score_delta"].apply(lambda s: 100.0 * (s < 0).mean())
    agg = agg.merge(pct_pos.rename("pct_windows_delta_positive"), on="episode_id")
    agg = agg.merge(pct_neg.rename("pct_windows_delta_negative"), on="episode_id")

    induced_ts = t[t["attack_induced_alarm"]].groupby("episode_id")["window_end"].min().rename("first_induced_alarm_timestamp")
    agg = agg.merge(induced_ts, on="episode_id", how="left")

    agg["threshold"] = umbral
    agg["distance_to_threshold"] = umbral - agg["max_attacked_score"]
    agg["relative_distance_to_threshold"] = agg["distance_to_threshold"] / umbral
    return agg


def _energia_antes_despues(fila_meta_row, ataques: dict, particion_val_raw: np.ndarray) -> dict:
    """Convencion (seccion 4 del encargo): instante de alarma = fin de la primera ventana con
    alarma inducida. Si no hay deteccion, o la alarma llega despues de attack_end, TODA la
    energia se cuenta como oculta ANTES de la deteccion y recoverable_hidden_energy_pct = 0."""
    a = ataques[fila_meta_row.episode_id]
    offset, dur = a["offset_global"], a["duracion"]
    clean = particion_val_raw[offset:offset + dur].astype(np.float64)
    attacked = np.asarray(a["y_tramo"], dtype=np.float64)
    hidden_total = fila_meta_row.hidden_energy_kwh

    alarm_ts = fila_meta_row.first_induced_alarm_timestamp
    if not fila_meta_row.induced_detected or pd.isna(alarm_ts) or alarm_ts >= fila_meta_row.attack_end:
        hidden_before, hidden_after = hidden_total, 0.0
    else:
        cutoff = int(round((alarm_ts - fila_meta_row.attack_start) / pd.Timedelta(minutes=1)))
        cutoff = max(0, min(dur, cutoff))
        hidden_before = float((clean[:cutoff] - attacked[:cutoff]).sum() / 60.0)
        hidden_after = hidden_total - hidden_before

    hidden_before_pct = 100.0 * hidden_before / hidden_total if hidden_total > EPS_ENERGIA else float("nan")
    recoverable_pct = 100.0 * hidden_after / hidden_total if hidden_total > EPS_ENERGIA else 0.0
    return {
        "hidden_energy_before_detection_kwh": hidden_before,
        "hidden_energy_after_detection_kwh": hidden_after,
        "hidden_energy_before_detection_pct": hidden_before_pct,
        "recoverable_hidden_energy_pct": max(0.0, recoverable_pct),
    }


def construir_tabla_episodios(tabla_ventanas: pd.DataFrame, ataques: dict, master_csv: pd.DataFrame,
                               particion_val_raw: np.ndarray, contexto_impacto: pd.DataFrame,
                               umbral: float) -> pd.DataFrame:
    """Una fila por episodio con TODAS las columnas de las secciones 3 y 4 del encargo."""
    respuesta = _respuesta_detector(tabla_ventanas, umbral)

    meta_cols = ["episode_id", "family", "parameter", "duration_min", "repetition_idx",
                 "attack_start", "attack_end", "modified_samples"]
    meta = master_csv[meta_cols].rename(columns={"modified_samples": "modified_samples_master"})

    df = meta.merge(contexto_impacto, on="episode_id").merge(respuesta, on="episode_id")

    df["induced_detection_delay_min"] = (df["first_induced_alarm_timestamp"] - df["attack_start"]) / pd.Timedelta(minutes=1)
    sin_induced = ~df["induced_detected"]
    df.loc[sin_induced, "induced_detection_delay_min"] = np.nan
    durante = (df["first_induced_alarm_timestamp"] <= df["attack_end"]).astype(object)
    despues = (df["first_induced_alarm_timestamp"] > df["attack_end"]).astype(object)
    durante[sin_induced] = np.nan
    despues[sin_induced] = np.nan
    df["detected_during_attack"] = durante
    df["detected_after_attack"] = despues

    filas_energia = [_energia_antes_despues(row, ataques, particion_val_raw) for row in df.itertuples()]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(filas_energia)], axis=1)

    df["parameter_value"] = df["parameter"].apply(_severidad_numerica)
    return df


# ==========================================================================================
# 5) Clasificacion de detectabilidad (4 grupos mutuamente excluyentes)
# ==========================================================================================

DESCRIPCIONES_GRUPO = {
    "A": "No responde: el ataque no incrementa el score maximo respecto a la version limpia de la misma ventana.",
    "B": "Responde pero queda lejos: el score sube, pero se queda a mas del 10% del umbral.",
    "C": "Cerca del umbral: el score sube y queda a <=10% del umbral (incluye el caso, minoritario, en que "
         "la ventana ya alarmaba en limpio y por tanto la deteccion no puede atribuirse al ataque).",
    "D": "Detectado: existe al menos una alarma atribuible al ataque (attack-induced).",
}


def clasificar_grupos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    induced = df["induced_detected"].to_numpy(dtype=bool)
    delta = df["max_score_delta"].to_numpy(dtype=float)
    rel_dist = df["relative_distance_to_threshold"].to_numpy(dtype=float)

    grupo = np.empty(len(df), dtype=object)
    grupo[induced] = "D"
    no_det = ~induced
    grupo[no_det & (delta <= 0)] = "A"
    responde = no_det & (delta > 0)
    grupo[responde & (rel_dist > 0.10)] = "B"
    # <=0.10 incluye distancias negativas (ventana ya alarmaba en limpio -> no se atribuye al ataque,
    # pero el detector "ya esta" en zona de alarma): se trata como Grupo C, no como caso sin clasificar.
    grupo[responde & (rel_dist <= 0.10)] = "C"

    assert (grupo != None).all() and set(np.unique(grupo)) <= set("ABCD")  # noqa: E711
    df["detectability_group"] = grupo
    df["detectability_group_description"] = df["detectability_group"].map(DESCRIPCIONES_GRUPO)
    return df


# ==========================================================================================
# 6) DR ponderado por energia
# ==========================================================================================

def resumen_dr_energia(df: pd.DataFrame) -> dict:
    n = len(df)
    total_hidden = df["hidden_energy_kwh"].sum()
    detected_hidden = df.loc[df["induced_detected"], "hidden_energy_kwh"].sum()
    after_total = df["hidden_energy_after_detection_kwh"].sum()
    return {
        "n_episodios": n,
        "n_detectados": int(df["induced_detected"].sum()),
        "episode_dr_pct": 100 * df["induced_detected"].mean() if n else float("nan"),
        "hidden_energy_total_kwh": float(total_hidden),
        "hidden_energy_detected_kwh": float(detected_hidden),
        "energy_weighted_dr_pct": 100 * detected_hidden / total_hidden if total_hidden > EPS_ENERGIA else float("nan"),
        "timely_energy_detection_rate_pct": 100 * after_total / total_hidden if total_hidden > EPS_ENERGIA else float("nan"),
    }


def tablas_dr_energia(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tablas = {}
    filas = []
    for etiqueta, sub in (("todas", df), ("sin_recorte_picos", df[df["family"] != "recorte_picos"])):
        fila = {"familias": etiqueta}
        fila.update(resumen_dr_energia(sub))
        filas.append(fila)
    tablas["global"] = pd.DataFrame(filas)

    for nombre_grupo, col in (("familia", "family"), ("duracion", "duration_min"), ("parametro", "parameter")):
        filas = []
        for val, sub in df.groupby(col):
            fila = {col: val}
            fila.update(resumen_dr_energia(sub))
            filas.append(fila)
        tablas[f"por_{nombre_grupo}"] = pd.DataFrame(filas)
    return tablas


# ==========================================================================================
# 7) Detectados frente a no detectados: descriptivos + Mann-Whitney + Spearman + Cliff's delta
# ==========================================================================================

VARIABLES_COMPARAR = [
    "duration_min", "parameter_value", "clean_mean_kw", "clean_energy_kwh",
    "hidden_energy_kwh", "hidden_energy_pct", "n_affected_windows",
    "max_score_delta", "distance_to_threshold",
]


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """delta = P(x>y) - P(x<y). O(n log n) via busqueda ordenada (sin scipy: no existe alli)."""
    x_sorted = np.sort(x)
    n1, n2 = len(x_sorted), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    menor = np.searchsorted(x_sorted, y, side="left")       # #x < y_i
    menor_o_igual = np.searchsorted(x_sorted, y, side="right")  # #x <= y_i
    mayor = n1 - menor_o_igual                                # #x > y_i
    return float((mayor.sum() - menor.sum()) / (n1 * n2))


def _interpretar_cliffs(d: float) -> str:
    ad = abs(d)
    if np.isnan(ad):
        return "n/a"
    if ad < 0.147:
        return "insignificante"
    if ad < 0.33:
        return "pequeno"
    if ad < 0.474:
        return "mediano"
    return "grande"


def _resumen_grupo(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, media=np.nan, mediana=np.nan, std=np.nan, q1=np.nan, q3=np.nan, ci_low=np.nan, ci_high=np.nan)
    media = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    return dict(n=n, media=media, mediana=float(np.median(x)), std=std,
                q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)),
                ci_low=media - 1.96 * se, ci_high=media + 1.96 * se)


def comparar_detectados_no_detectados(df: pd.DataFrame) -> pd.DataFrame:
    det = df[df["induced_detected"] == True]  # noqa: E712
    nodet = df[df["induced_detected"] == False]  # noqa: E712
    filas = []
    for var in VARIABLES_COMPARAR:
        xd = det[var].to_numpy(dtype=float)
        xn = nodet[var].to_numpy(dtype=float)
        xd, xn = xd[~np.isnan(xd)], xn[~np.isnan(xn)]
        rd, rn = _resumen_grupo(xd), _resumen_grupo(xn)
        if len(xd) > 0 and len(xn) > 0:
            u_stat, p_mw = sstats.mannwhitneyu(xd, xn, alternative="two-sided")
            delta = _cliffs_delta(xd, xn)
        else:
            u_stat, p_mw, delta = float("nan"), float("nan"), float("nan")
        filas.append({
            "variable": var,
            "n_detectados": rd["n"], "media_detectados": rd["media"], "mediana_detectados": rd["mediana"],
            "std_detectados": rd["std"], "q1_detectados": rd["q1"], "q3_detectados": rd["q3"],
            "ci95_low_detectados": rd["ci_low"], "ci95_high_detectados": rd["ci_high"],
            "n_no_detectados": rn["n"], "media_no_detectados": rn["media"], "mediana_no_detectados": rn["mediana"],
            "std_no_detectados": rn["std"], "q1_no_detectados": rn["q1"], "q3_no_detectados": rn["q3"],
            "ci95_low_no_detectados": rn["ci_low"], "ci95_high_no_detectados": rn["ci_high"],
            "mannwhitney_u": float(u_stat), "mannwhitney_p": float(p_mw),
            "cliffs_delta": delta, "cliffs_delta_interpretacion": _interpretar_cliffs(delta),
            "significativo_p<0.05": bool(p_mw < 0.05) if not np.isnan(p_mw) else False,
        })
    return pd.DataFrame(filas)


def correlaciones_spearman(df: pd.DataFrame) -> pd.DataFrame:
    y = df["induced_detected"].astype(int).to_numpy()
    filas = []
    for var in VARIABLES_COMPARAR:
        x = df[var].to_numpy(dtype=float)
        valido = ~np.isnan(x)
        if valido.sum() < 3:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = sstats.spearmanr(x[valido], y[valido])
        filas.append({"variable": var, "spearman_rho": float(rho), "spearman_p": float(p),
                       "significativo_p<0.05": bool(p < 0.05) if not np.isnan(p) else False})
    return pd.DataFrame(filas)


# ==========================================================================================
# 8) Factores asociados a la deteccion: correlaciones (arriba) + 2 regresiones logisticas
#    (factores fisicos/del ataque; respuesta interna del detector) + arbol pequeno explicativo
# ==========================================================================================

VARS_FISICAS = ["duration_min", "hidden_energy_kwh", "hidden_energy_pct", "clean_mean_kw", "clean_max_kw"]
VARS_INTERNAS = ["max_score_delta", "distance_to_threshold", "n_affected_windows"]


def _logit_interpretable(df: pd.DataFrame, variables_continuas: list[str], incluir_familia: bool) -> dict:
    d = df.copy()
    y = d["induced_detected"].astype(int).to_numpy()
    X_cont = d[variables_continuas].fillna(d[variables_continuas].median())
    scaler = StandardScaler()
    X_cont_z = pd.DataFrame(scaler.fit_transform(X_cont), columns=variables_continuas, index=d.index)

    if incluir_familia:
        dummies = pd.get_dummies(d["family"], prefix="family", drop_first=True).astype(float)
        X = pd.concat([X_cont_z, dummies], axis=1)
    else:
        X = X_cont_z

    modelo = LogisticRegression(max_iter=3000)
    modelo.fit(X, y)
    coefs = pd.DataFrame({
        "variable": X.columns, "coeficiente": modelo.coef_[0], "odds_ratio": np.exp(modelo.coef_[0]),
    }).sort_values("coeficiente", key=np.abs, ascending=False)

    corr = X_cont.corr()
    colineales = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr.columns)
                  for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) > 0.7]
    return {"coeficientes": coefs, "accuracy_train": float(modelo.score(X, y)),
            "colinealidades_|r|>0.7": colineales, "n": len(d)}


def analizar_factores(df: pd.DataFrame) -> dict:
    """2 regresiones logisticas SEPARADAS (fisicas y respuesta interna, nunca mezcladas), mas
    un arbol de decision pequeno (max_depth=3) puramente explicativo -- NO es una nueva
    solucion de deteccion. No se combinan max_attacked_score y distance_to_threshold (son
    redundantes: distance_to_threshold = threshold - max_attacked_score)."""
    fisico = _logit_interpretable(df, VARS_FISICAS, incluir_familia=True)
    interno = _logit_interpretable(df, VARS_INTERNAS, incluir_familia=False)

    d = df.copy()
    y = d["induced_detected"].astype(int).to_numpy()
    X_arbol = d[VARS_FISICAS].fillna(d[VARS_FISICAS].median())
    arbol = DecisionTreeClassifier(max_depth=3, min_samples_leaf=80, random_state=42)
    arbol.fit(X_arbol, y)
    texto_arbol = export_text(arbol, feature_names=list(X_arbol.columns))

    return {"logit_fisico": fisico, "logit_interno": interno, "arbol": arbol,
            "arbol_texto": texto_arbol, "arbol_variables": list(X_arbol.columns)}


# ==========================================================================================
# 9) Estudio de los episodios no detectados
# ==========================================================================================

def estudio_no_detectados(df: pd.DataFrame) -> dict:
    no_det = df[df["induced_detected"] == False].copy()  # noqa: E712
    n = len(no_det)

    resumen_grupos = no_det["detectability_group"].value_counts().rename_axis("grupo").reset_index(name="n_episodios")
    resumen_grupos["pct_de_no_detectados"] = 100 * resumen_grupos["n_episodios"] / n
    resumen_grupos = resumen_grupos.set_index("grupo").reindex(["A", "B", "C"]).fillna(0).reset_index()

    energia_por_grupo = no_det.groupby("detectability_group")["hidden_energy_kwh"].agg(
        energia_oculta_total_kwh="sum", energia_oculta_media_kwh="mean", n="count").reindex(["A", "B", "C"]).reset_index()

    def _moda(s):
        return s.value_counts().idxmax() if len(s) else None

    familia_por_grupo = no_det.groupby("detectability_group")["family"].agg(_moda)
    duracion_por_grupo = no_det.groupby("detectability_group")["duration_min"].agg(_moda)
    parametro_por_grupo = no_det.groupby("detectability_group")["parameter"].agg(_moda)

    p75_global = float(df["hidden_energy_kwh"].quantile(0.75))
    alto_impacto = no_det[no_det["hidden_energy_kwh"] >= p75_global].sort_values("hidden_energy_kwh", ascending=False)

    return {
        "n_no_detectados": n, "resumen_grupos": resumen_grupos, "energia_por_grupo": energia_por_grupo,
        "familia_por_grupo": familia_por_grupo, "duracion_por_grupo": duracion_por_grupo,
        "parametro_por_grupo": parametro_por_grupo, "p75_global_hidden_energy_kwh": p75_global,
        "alto_impacto": alto_impacto,
    }


# ==========================================================================================
# 10) Casos concretos (criterios reproducibles) + figuras individuales
# ==========================================================================================

def _elegir_representativos(sub: pd.DataFrame, columna: str, n: int = 3) -> list[str]:
    if len(sub) == 0:
        return []
    valores = sub[columna].to_numpy()
    elegidos, usados = [], set()
    for p in np.linspace(25, 75, n) if n > 1 else [50]:
        objetivo = np.percentile(valores, p)
        orden = np.argsort(np.abs(valores - objetivo))
        for idx in orden:
            ep = sub.iloc[idx]["episode_id"]
            if ep not in usados:
                elegidos.append(ep); usados.add(ep)
                break
    return elegidos


def seleccionar_casos(df: pd.DataFrame, alto_impacto: pd.DataFrame) -> dict[str, list[str]]:
    detectados = df[df["induced_detected"] == True]  # noqa: E712
    casos = {
        "detectados_correctamente": _elegir_representativos(detectados, "induced_detection_delay_min", 3),
        "no_detectados_sin_respuesta_A": _elegir_representativos(df[df["detectability_group"] == "A"], "hidden_energy_kwh", 3),
        "no_detectados_lejos_umbral_B": _elegir_representativos(df[df["detectability_group"] == "B"], "relative_distance_to_threshold", 3),
        "no_detectados_cerca_umbral_C": _elegir_representativos(df[df["detectability_group"] == "C"], "relative_distance_to_threshold", 3),
        "no_detectados_alto_impacto": alto_impacto.nlargest(3, "hidden_energy_kwh")["episode_id"].tolist(),
    }
    return casos


def figura_caso(episode_id: str, df: pd.DataFrame, ataques: dict, particion_val_raw: np.ndarray,
                 grid_60: dict, modelo, params_norm: dict, umbral: float, inicio_particion,
                 tabla_ventanas: pd.DataFrame, nombre_archivo: str, etiqueta_grupo: str) -> None:
    fila = df[df["episode_id"] == episode_id].iloc[0]
    a = ataques[episode_id]
    offset, dur = a["offset_global"], a["duracion"]
    margen = max(60, dur // 4)
    ini, fin = max(0, offset - margen), offset + dur + margen

    x_limpio = particion_val_raw[ini:fin].astype(np.float64)
    x_atacado = x_limpio.copy()
    x_atacado[offset - ini: offset - ini + dur] = a["y_tramo"]
    t = [inicio_particion + pd.Timedelta(minutes=m) for m in range(ini, fin)]

    # ventana representativa (mayor solape con el intervalo atacado) para la curva de reconstruccion
    ks = _ventanas_solapadas(offset, dur, STRIDE, grid_60["X"].shape[0])
    k_repr = max(ks, key=lambda k: min(offset + dur, k * STRIDE + TAMANO_VENTANA) - max(offset, k * STRIDE))
    w_start = k_repr * STRIDE
    ventana_atacada_repr = _ventana_atacada(particion_val_raw, w_start, offset, dur, a["y_tramo"])
    X_norm = aplicar_zscore(ventana_atacada_repr[None, :].astype(np.float32), params_norm)
    X_hat_kw = revertir_zscore(modelo.reconstruir(X_norm), params_norm)[0]
    t_recon = [inicio_particion + pd.Timedelta(minutes=w_start + i) for i in range(TAMANO_VENTANA)]

    vt = tabla_ventanas[tabla_ventanas["episode_id"] == episode_id].sort_values("window_end")
    t_score = pd.to_datetime(vt["window_end"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), height_ratios=[2, 1], sharex=False)
    ax1.plot(t, x_limpio, color=GRIS, linewidth=1.1, label="limpio")
    ax1.plot(t, x_atacado, color=ROJO, linewidth=1.2, alpha=0.85, label="reportado (atacado)")
    ax1.plot(t_recon, X_hat_kw, color=AZUL, linewidth=1.1, linestyle="--",
              label=f"reconstruccion (ventana representativa, k={k_repr})")
    ax1.axvspan(inicio_particion + pd.Timedelta(minutes=offset), inicio_particion + pd.Timedelta(minutes=offset + dur),
                color=VERDE, alpha=0.08, label="intervalo atacado")
    if fila["induced_detected"]:
        ax1.axvline(fila["first_induced_alarm_timestamp"], color=MORADO, linestyle=":", linewidth=1.6, label="1ª alarma inducida")
    ax1.set_ylabel("kW"); ax1.legend(fontsize=7.5, ncols=2)
    ax1.set_title(f"{episode_id} | {a['family']} ({a['parameter']}) | {dur} min | grupo={etiqueta_grupo} | "
                  f"energia oculta={fila['hidden_energy_kwh']:.3f} kWh ({fila['hidden_energy_pct']:.1f}%)", fontsize=9.5)

    ax2.plot(t_score, vt["clean_score"], color=GRIS, marker="o", markersize=3, linewidth=1.1, label="score limpio")
    ax2.plot(t_score, vt["attacked_score"], color=ROJO, marker="o", markersize=3, linewidth=1.1, label="score atacado")
    ax2.axhline(umbral, color=GRIS_OSCURO, linestyle="--", linewidth=1, label="umbral")
    ax2.set_ylabel(SCORE); ax2.legend(fontsize=7.5)
    fig.tight_layout()
    _guardar_fig(fig, nombre_archivo)


# ==========================================================================================
# 11) Matrices duracion x severidad (fila = family|parameter, columna = duration_min)
# ==========================================================================================

def matrices_duracion_severidad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fila_severidad"] = df["family"] + " | " + df["parameter"]
    filas = []
    for (fila_sev, dur), g in df.groupby(["fila_severidad", "duration_min"]):
        n = len(g)
        no_det = g[g["induced_detected"] == False]  # noqa: E712
        total_hidden = g["hidden_energy_kwh"].sum()
        detected_hidden = g.loc[g["induced_detected"], "hidden_energy_kwh"].sum()
        filas.append({
            "fila_severidad": fila_sev, "duration_min": dur, "n_episodios": n,
            "induced_DR_pct": 100 * g["induced_detected"].sum() / n if n else float("nan"),
            "energy_weighted_dr_pct": 100 * detected_hidden / total_hidden if total_hidden > EPS_ENERGIA else float("nan"),
            "mediana_hidden_energy_kwh": float(g["hidden_energy_kwh"].median()),
            "pct_cerca_umbral_C": 100 * (no_det["detectability_group"] == "C").sum() / n if n else float("nan"),
            "pct_sin_respuesta_A": 100 * (no_det["detectability_group"] == "A").sum() / n if n else float("nan"),
        })
    return pd.DataFrame(filas)


def _fig_heatmap(matriz: pd.DataFrame, columna_valor: str, titulo: str, nombre: str, cmap="Blues") -> None:
    piv = matriz.pivot(index="fila_severidad", columns="duration_min", values=columna_valor).reindex(columns=DURACIONES)
    datos = piv.values.astype(float)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(piv) + 2))
    im = ax.imshow(datos, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7.5)
    ax.set_xlabel("duracion (min)")
    vmax = np.nanmax(datos) if np.isfinite(datos).any() else 1
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            v = datos[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.6 * vmax else "#222"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=color)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout(); _guardar_fig(fig, nombre)


# ==========================================================================================
# 12) Sensibilidad al umbral (0.05 / 0.06 / 0.10 / 0.25 falsos eventos/dia)
# ==========================================================================================

def sensibilidad_umbral(tabla_ventanas: pd.DataFrame, ataques: dict, master_csv: pd.DataFrame,
                         particion_val_raw: np.ndarray, contexto_impacto: pd.DataFrame,
                         scores_limpios_60: np.ndarray, n_dias: float) -> tuple[pd.DataFrame, dict]:
    filas = []
    tablas_por_objetivo = {}
    for nombre_obj, valor_obj in OBJETIVOS_SENSIBILIDAD.items():
        umbral, tasa_eventos, tasa_ventanas = calibrar_umbral_por_eventos(scores_limpios_60, n_dias, valor_obj)
        tabla = construir_tabla_episodios(tabla_ventanas, ataques, master_csv, particion_val_raw, contexto_impacto, umbral)
        tabla = clasificar_grupos(tabla)
        tablas_por_objetivo[nombre_obj] = tabla

        resumen = resumen_dr_energia(tabla)
        con_ind = tabla[tabla["induced_detected"] == True]  # noqa: E712
        filas.append({
            "objetivo_nombre": nombre_obj, "objetivo_falsos_eventos_dia": valor_obj, "threshold": umbral,
            "tasa_eventos_dia_real": tasa_eventos, "tasa_ventanas_dia_real": tasa_ventanas,
            **resumen,
            "retraso_mediano_min": float(con_ind["induced_detection_delay_min"].median()) if len(con_ind) else float("nan"),
            "hidden_energy_before_detection_total_kwh": float(tabla["hidden_energy_before_detection_kwh"].sum()),
            "pct_cerca_umbral_C": 100 * (tabla["detectability_group"] == "C").sum() / len(tabla),
        })

    tabla_sens = pd.DataFrame(filas).sort_values("objetivo_falsos_eventos_dia").reset_index(drop=True)
    base = tabla_sens[tabla_sens["objetivo_nombre"] == "principal_0.06"].iloc[0]
    tabla_sens["episodios_adicionales_vs_0.06"] = tabla_sens["n_detectados"] - base["n_detectados"]
    tabla_sens["energia_adicional_vs_0.06_kwh"] = tabla_sens["hidden_energy_detected_kwh"] - base["hidden_energy_detected_kwh"]
    return tabla_sens, tablas_por_objetivo


# ==========================================================================================
# Orquestacion
# ==========================================================================================

def main() -> dict:
    print("Cargando episodios exactos y puntuando la rejilla stride=60 con relative_kw...")
    ctx = cargar_todo()
    ataques, master_csv = ctx["ataques"], ctx["master_csv"]
    particion_val_raw = ctx["particion_val_raw"]
    tabla_ventanas, scores_limpios_60 = ctx["tabla_ventanas"], ctx["scores_limpios_60"]
    n_dias = ctx["n_dias"]
    inicio_particion = ctx["particion_val"].index[0]

    umbral, tasa_eventos, tasa_ventanas = calibrar_umbral_por_eventos(scores_limpios_60, n_dias, OBJETIVO_PRINCIPAL_VALOR)
    print(f"Umbral principal (0.06 eventos/dia): {umbral:.5f}, eventos/dia real={tasa_eventos:.4f}, ventanas/dia={tasa_ventanas:.3f}")

    print("\n[3] Construyendo tabla enriquecida de episodios...")
    contexto_impacto = contexto_e_impacto(ataques, particion_val_raw)
    df = construir_tabla_episodios(tabla_ventanas, ataques, master_csv, particion_val_raw, contexto_impacto, umbral)
    df = clasificar_grupos(df)
    _guardar_tabla(df, "episodios_detectabilidad_impacto")
    print(f"{len(df)} episodios, {df['induced_detected'].sum()} detectados "
          f"({100 * df['induced_detected'].mean():.2f}% episode_DR)")

    print("\n[6] DR ponderado por energia...")
    tablas_energia = tablas_dr_energia(df)
    for nombre, t in tablas_energia.items():
        _guardar_tabla(t, f"06_dr_energia_{nombre}")
    print(tablas_energia["global"].to_string(index=False))

    print("\n[7] Detectados vs no detectados (Mann-Whitney, Spearman, Cliff's delta)...")
    comparacion = comparar_detectados_no_detectados(df)
    _guardar_tabla(comparacion, "08_detectados_vs_no_detectados")
    correlaciones = correlaciones_spearman(df)
    _guardar_tabla(correlaciones, "10a_correlaciones_spearman")
    print(comparacion[["variable", "mediana_detectados", "mediana_no_detectados", "cliffs_delta",
                        "cliffs_delta_interpretacion", "mannwhitney_p"]].to_string(index=False))

    print("\n[8] Factores asociados a la deteccion (2 regresiones logisticas + arbol)...")
    factores = analizar_factores(df)
    _guardar_tabla(factores["logit_fisico"]["coeficientes"], "10b_logit_factores_fisicos")
    _guardar_tabla(factores["logit_interno"]["coeficientes"], "10c_logit_respuesta_interna")
    (TABLAS_DIR).mkdir(parents=True, exist_ok=True)
    with open(TABLAS_DIR / "10d_arbol_explicativo.txt", "w", encoding="utf-8") as f:
        f.write(factores["arbol_texto"])
    print(f"  logit fisico: accuracy_train={factores['logit_fisico']['accuracy_train']:.3f}, "
          f"colinealidades={factores['logit_fisico']['colinealidades_|r|>0.7']}")
    print(f"  logit interno: accuracy_train={factores['logit_interno']['accuracy_train']:.3f}")

    print("\n[5] Grupos de detectabilidad...")
    resumen_grupos_global = df["detectability_group"].value_counts().rename_axis("grupo").reset_index(name="n_episodios")
    resumen_grupos_global["pct"] = 100 * resumen_grupos_global["n_episodios"] / len(df)
    _guardar_tabla(resumen_grupos_global, "05_grupos_detectabilidad")
    print(resumen_grupos_global.to_string(index=False))

    print("\n[9] Estudio de episodios no detectados...")
    est = estudio_no_detectados(df)
    _guardar_tabla(est["resumen_grupos"], "09a_no_detectados_resumen_grupos")
    _guardar_tabla(est["energia_por_grupo"], "09b_no_detectados_energia_por_grupo")
    _guardar_tabla(est["alto_impacto"], "09_ataques_no_detectados_alto_impacto")
    print(f"  p75 global hidden_energy_kwh={est['p75_global_hidden_energy_kwh']:.4f}, "
          f"no detectados de alto impacto={len(est['alto_impacto'])}")

    print("\n[11] Matrices duracion x severidad...")
    matriz = matrices_duracion_severidad(df)
    _guardar_tabla(matriz, "11_matriz_duracion_severidad")
    _fig_heatmap(matriz, "induced_DR_pct", "induced_DR (%) por severidad x duracion", "08_heatmap_dr_duracion_severidad", cmap="Blues")
    _fig_heatmap(matriz, "energy_weighted_dr_pct", "DR ponderado por energia (%) por severidad x duracion",
                 "09_heatmap_energia_duracion_severidad", cmap="Blues")

    print("\n[12] Sensibilidad al umbral (0.05/0.06/0.10/0.25 eventos/dia)...")
    tabla_sens, _ = sensibilidad_umbral(tabla_ventanas, ataques, master_csv, particion_val_raw,
                                         contexto_impacto, scores_limpios_60, n_dias)
    _guardar_tabla(tabla_sens, "11b_sensibilidad_umbral")
    print(tabla_sens[["objetivo_nombre", "threshold", "tasa_eventos_dia_real", "episode_dr_pct",
                       "energy_weighted_dr_pct", "retraso_mediano_min"]].to_string(index=False))

    print("\n[10] Seleccionando y graficando casos concretos...")
    casos = seleccionar_casos(df, est["alto_impacto"])
    filas_casos = []
    for etiqueta, ids in casos.items():
        for i, ep_id in enumerate(ids):
            nombre_archivo = f"12_caso_{etiqueta}_{i+1}"
            figura_caso(ep_id, df, ataques, particion_val_raw, ctx["grid_60"], ctx["modelo"], ctx["params_norm"],
                        umbral, inicio_particion, tabla_ventanas, nombre_archivo, etiqueta)
            fila = df[df["episode_id"] == ep_id].iloc[0].to_dict()
            fila["seleccion"] = etiqueta
            fila["figura"] = nombre_archivo
            filas_casos.append(fila)
    tabla_casos = pd.DataFrame(filas_casos)
    _guardar_tabla(tabla_casos, "12_casos_seleccionados")
    print(f"  {len(tabla_casos)} casos, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")

    # ---- figuras restantes de la seccion 13 ----
    print("\nGenerando figuras restantes...")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    filas_fam = tablas_energia["por_familia"]
    ax.scatter(filas_fam["episode_dr_pct"], filas_fam["energy_weighted_dr_pct"], color=AZUL, s=60, zorder=3)
    for _, r in filas_fam.iterrows():
        ax.annotate(r["family"], (r["episode_dr_pct"], r["energy_weighted_dr_pct"]), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points")
    lim = [0, max(filas_fam[["episode_dr_pct", "energy_weighted_dr_pct"]].max()) * 1.1]
    ax.plot(lim, lim, color=GRIS, linestyle="--", linewidth=1, label="y=x")
    ax.set_xlabel("DR por episodios (%)"); ax.set_ylabel("DR ponderado por energia (%)")
    ax.set_title("DR por episodios vs. DR ponderado por energia, por familia")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "01_dr_episodios_vs_energia")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    total = df["hidden_energy_kwh"].sum()
    detectada = df.loc[df["induced_detected"], "hidden_energy_kwh"].sum()
    ax.bar(["energia oculta total"], [detectada], color=VERDE, label="en episodios detectados")
    ax.bar(["energia oculta total"], [total - detectada], bottom=[detectada], color=ROJO, label="en episodios no detectados")
    ax.set_ylabel("kWh"); ax.set_title("Energia ocultada: detectada vs no detectada")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "02_energia_detectada_no_detectada")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(df.loc[df["induced_detected"], "hidden_energy_kwh"], bins=40, color=VERDE, alpha=0.55, label="detectados", density=True)
    ax.hist(df.loc[~df["induced_detected"], "hidden_energy_kwh"], bins=40, color=ROJO, alpha=0.55, label="no detectados", density=True)
    ax.set_xlabel("energia oculta (kWh)"); ax.set_ylabel("densidad")
    ax.set_title("Distribucion de energia ocultada, detectados vs no detectados")
    ax.legend(fontsize=8)
    fig.tight_layout(); _guardar_fig(fig, "03_distribucion_energia_oculta")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colores = np.where(df["induced_detected"], VERDE, ROJO)
    ax.scatter(df["hidden_energy_kwh"], df["max_score_delta"], c=colores, s=10, alpha=0.4)
    ax.axhline(0, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_xlabel("energia oculta (kWh)"); ax.set_ylabel("max_score_delta")
    ax.set_title("max_score_delta vs energia oculta (verde=detectado, rojo=no detectado)")
    fig.tight_layout(); _guardar_fig(fig, "04_delta_vs_energia")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(df["hidden_energy_kwh"], df["distance_to_threshold"], c=colores, s=10, alpha=0.4)
    ax.axhline(0, color=GRIS_OSCURO, linestyle="--", linewidth=1)
    ax.set_xlabel("energia oculta (kWh)"); ax.set_ylabel("distance_to_threshold")
    ax.set_title("Distancia al umbral vs energia oculta (verde=detectado, rojo=no detectado)")
    fig.tight_layout(); _guardar_fig(fig, "05_distancia_umbral_vs_energia")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    tabla_gd = df.groupby(["duration_min", "detectability_group"]).size().unstack(fill_value=0)
    tabla_gd = tabla_gd.reindex(columns=list("ABCD"), fill_value=0)
    tabla_gd_pct = tabla_gd.div(tabla_gd.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(tabla_gd_pct))
    for g in "ABCD":
        ax.bar(tabla_gd_pct.index.astype(str), tabla_gd_pct[g], bottom=bottom, color=COLOR_GRUPO[g], label=g)
        bottom += tabla_gd_pct[g].to_numpy()
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("% episodios"); ax.set_title("Grupos de detectabilidad por duracion")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "06_grupos_por_duracion")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    tabla_gf = df.groupby(["family", "detectability_group"]).size().unstack(fill_value=0)
    tabla_gf = tabla_gf.reindex(columns=list("ABCD"), fill_value=0)
    tabla_gf_pct = tabla_gf.div(tabla_gf.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(tabla_gf_pct))
    for g in "ABCD":
        ax.bar(tabla_gf_pct.index.astype(str), tabla_gf_pct[g], bottom=bottom, color=COLOR_GRUPO[g], label=g)
        bottom += tabla_gf_pct[g].to_numpy()
    ax.set_xticklabels(tabla_gf_pct.index, rotation=15, ha="right")
    ax.set_ylabel("% episodios"); ax.set_title("Grupos de detectabilidad por familia")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "07_grupos_por_familia")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(tabla_sens["tasa_eventos_dia_real"], tabla_sens["episode_dr_pct"], marker="o", color=AZUL, label="DR por episodios")
    ax.plot(tabla_sens["tasa_eventos_dia_real"], tabla_sens["energy_weighted_dr_pct"], marker="o", color=NARANJA, label="DR ponderado por energia")
    ax.axvline(0.06, color=GRIS, linestyle="--", linewidth=1)
    ax.set_xlabel("falsos eventos / dia"); ax.set_ylabel("DR (%)")
    ax.set_title("Sensibilidad al umbral: DR por episodios vs DR ponderado por energia")
    ax.legend(fontsize=9)
    fig.tight_layout(); _guardar_fig(fig, "10_dr_vs_falsos_eventos_dia")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    con_ind = df[df["induced_detected"] == True]  # noqa: E712
    med_antes = con_ind.groupby("duration_min")["hidden_energy_before_detection_kwh"].median()
    ax.plot(med_antes.index, med_antes.values, marker="o", color=ROJO, linewidth=1.8)
    ax.set_xlabel("duracion (min)"); ax.set_ylabel("energia oculta ANTES de la alarma (kWh, mediana)")
    ax.set_title("Energia ocultada antes de la primera alarma inducida, por duracion")
    fig.tight_layout(); _guardar_fig(fig, "11_energia_antes_alarma_por_duracion")

    print(f"\nTablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/")

    return {
        "ctx": ctx, "umbral": umbral, "df": df, "tablas_energia": tablas_energia, "comparacion": comparacion,
        "correlaciones": correlaciones, "factores": factores, "estudio_no_detectados": est,
        "matriz": matriz, "tabla_sens": tabla_sens, "casos": casos,
    }


if __name__ == "__main__":
    main()
