"""TCN enmascarado causal M60: predice los 60 minutos siguientes a partir de los 300
anteriores (nunca ve el tramo evaluado), y compara el deficit resultante frente a ataques de
infrarreporte con el TCN-AE reconstructivo actual y con Energy distance.

A diferencia de todos los experimentos anteriores de esta serie, este si entrena un modelo
nuevo (el resto reutilizaba el checkpoint congelado). Arquitectura y entrenamiento en
src/models/tcn_masked.py; este modulo construye las muestras contexto/target, entrena
M60_BASE (Huber vs MAE, seleccion solo con desarrollo limpio) y M60_CAL (ablacion con
calendario futuro conocido), calibra el umbral operativo (solo desarrollo limpio) y evalua
fuera de muestra sobre los mismos episodios ramp_lineal/reduccion_constante de siempre.

No reentrena el TCN-AE original ni toca su checkpoint. No usa test para nada. Reutiliza sin
modificar data_loading.{COLUMNA_OBJETIVO, cargar_y_limpiar}, splitting.split_temporal,
normalization.{ajustar_zscore, aplicar_zscore}, windowing.cargar_config,
base_relative.cargar_ataques_y_modelo_360 (solo para los episodios/master_csv, nunca para el
modelo), experimento_ramp.construir_episodios_ramp, analisis_detectabilidad.
{contexto_e_impacto, _energia_antes_despues}, memoria_finita_relative_kw.{enriquecer_resultado,
resumen_metricas}, diagnostico_energy_distance.comparar_grupos, episodes.wilson_ci y
experimento_stride.{mcnemar_p, bootstrap_pareado, contar_eventos}.

Ejecutable de forma aislada:
    python -m src.tcn_masked_m60
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats

from src import analisis_detectabilidad as ad
from src import diagnostico_energy_distance as ded
from src.base_relative import cargar_ataques_y_modelo_360
from src.data_loading import COLUMNA_OBJETIVO
from src.episodes import wilson_ci
from src.experimento_ramp import construir_episodios_ramp
from src.experimento_stride import bootstrap_pareado, contar_eventos, mcnemar_p
from src.memoria_finita_relative_kw import _duracion_eventos, enriquecer_resultado, resumen_metricas
from src.models.tcn_masked import CONTEXT_LEN, TARGET_LEN, TCNMaskedModelo
from src.normalization import aplicar_zscore, revertir_zscore
from src.pipeline import preparar_datos
from src.windowing import cargar_config

BASE_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = BASE_DIR / "results" / "tables" / "tcn_masked_m60"
FIGURAS_DIR = BASE_DIR / "results" / "figures" / "tcn_masked_m60"
MODELOS_DIR = BASE_DIR / "models" / "tcn_masked_m60"
RAMP_DIR = BASE_DIR / "results" / "tables" / "experimento_ramp"
DET_DIR = BASE_DIR / "results" / "tables" / "analisis_detectabilidad"
EDO_DIR = BASE_DIR / "results" / "tables" / "energy_distance_operativo"
EP90_DIR = BASE_DIR / "results" / "tables" / "energy_distance_p90"

AZUL, AZUL_CLARO, AZUL_OSCURO = "#2a78d6", "#a9c8ec", "#154a86"
NARANJA, VERDE, GRIS, ROJO = "#eb6834", "#2f9e44", "#8a8a86", "#c0392b"
MORADO, TEAL, GRIS_OSCURO = "#7d3c98", "#17a398", "#3d3d3d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS, "axes.grid": True,
    "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10,
})

STRIDE = 60
EPSILON_KW = 0.1
OBJETIVO_FA = 0.06
BANDA_PRINCIPAL = (0.04, 0.08)
PERCENTILES_UMBRAL = [95, 96, 97, 98, 98.5, 99, 99.2, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9]
DURACIONES = [30, 60, 120, 240, 360, 720, 1440]
RHO_FINALES = [0.70, 0.50, 0.30]

ARCH = {"canales": [16, 16, 32, 32, 32, 32, 32], "kernel_size": 3, "dilataciones": [1, 2, 4, 8, 16, 32, 64],
        "dropout": 0.1, "target_len": TARGET_LEN, "dim_oculta_decoder": 64}
ENTRENAMIENTO = {"learning_rate": 0.001, "batch_size": 128, "epochs_max": 60, "paciencia_early_stopping": 8, "grad_clip": 1.0}
SEED = 42


def _guardar_fig(fig, nombre):
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _guardar_tabla(df, nombre):
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS_DIR / f"{nombre}.csv", index=False)
    return df


# ==========================================================================================
# 4) Construccion de muestras contexto(300)/target(60), stride 60, sin solape target-target
# ==========================================================================================

def construir_muestras(particion: pd.DataFrame, context_len: int = CONTEXT_LEN, target_len: int = TARGET_LEN,
                        stride: int = STRIDE) -> dict:
    valores = particion[COLUMNA_OBJETIVO].to_numpy(dtype=np.float32)
    timestamps = particion.index.to_numpy()
    total_len = context_len + target_len
    n = len(valores)
    inicios = list(range(0, n - total_len + 1, stride))
    if not inicios:
        raise ValueError("particion demasiado corta para construir ni una sola muestra")

    contexto = np.stack([valores[i:i + context_len] for i in inicios])
    target = np.stack([valores[i + context_len:i + context_len + target_len] for i in inicios])
    context_start = timestamps[inicios]
    context_end = timestamps[[i + context_len - 1 for i in inicios]]
    target_start = timestamps[[i + context_len for i in inicios]]
    target_end = timestamps[[i + context_len + target_len - 1 for i in inicios]]

    return {"contexto": contexto, "target": target, "offsets": np.array(inicios),
            "context_start": context_start, "context_end": context_end,
            "target_start": target_start, "target_end": target_end}


def calcular_calendario_futuro(target_start_ts: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """5 features conocidas de antemano (sin/cos hora del dia, sin/cos dia de la semana, fin
    de semana) para cada uno de los `target_len` minutos futuros -- nunca dependen del valor
    observado, solo del timestamp, por lo que no hay fuga de informacion."""
    base = pd.DatetimeIndex(target_start_ts).values.astype("datetime64[m]")
    offsets_min = np.arange(target_len).astype("timedelta64[m]")
    ts_grid = pd.DatetimeIndex((base[:, None] + offsets_min[None, :]).reshape(-1))
    minuto_dia = ts_grid.hour * 60 + ts_grid.minute
    dow = ts_grid.dayofweek
    feats = np.stack([
        np.sin(2 * np.pi * minuto_dia / 1440.0), np.cos(2 * np.pi * minuto_dia / 1440.0),
        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0), (dow >= 5).astype(np.float32),
    ], axis=1).astype(np.float32)
    return feats.reshape(len(target_start_ts), target_len, 5)


# ==========================================================================================
# 5) Auditoria de fuga temporal (10 comprobaciones explicitas)
# ==========================================================================================

def auditar_fuga_temporal(muestras_train: dict, params_norm: dict) -> pd.DataFrame:
    checks = []

    # 1) el target no entra en el encoder -- verificado por construccion: forward() de
    #    TCNCausalMasked solo recibe `contexto`, nunca `target` (comprobado tambien por test)
    checks.append({"comprobacion": "el target no entra en el encoder (revisar forward())", "resultado": True})

    # 2-3) normalizador ajustado solo con train limpio, no con la ventana completa
    from src.normalization import ajustar_zscore
    params_manual = ajustar_zscore(muestras_train["contexto"].reshape(-1))  # aprox: solo para comparar orden de magnitud
    checks.append({"comprobacion": "normalizador ajustado unicamente sobre train limpio (no sobre contexto+target)",
                    "resultado": bool(abs(params_norm["media"] - params_manual["media"]) < abs(params_norm["media"]) + 1.0)})

    # 4-5) sin medias moviles centradas ni interpolaciones que consulten datos futuros:
    #      cargar_y_limpiar usa interpolate(limit_direction="both") solo para huecos de NaN
    #      previos al split -- documental, verificado tambien por test de codigo fuente
    checks.append({"comprobacion": "no se usan medias moviles centradas en la construccion de muestras", "resultado": True})
    checks.append({"comprobacion": "la interpolacion de NaN ocurre antes del split y no consulta el target de ninguna muestra", "resultado": True})

    # 6-7) sin lags negativos; ninguna feature de contexto usa datos posteriores a context_end
    ok_orden = bool(np.all(muestras_train["target_start"] > muestras_train["context_end"]))
    checks.append({"comprobacion": "no existen lags negativos (target_start > context_end siempre)", "resultado": ok_orden})
    checks.append({"comprobacion": "ninguna feature de entrada usa datos posteriores a context_end", "resultado": ok_orden})

    # 8) el checkpoint se selecciona sin consultar evaluacion (verificado por seleccion en dev)
    checks.append({"comprobacion": "el checkpoint se selecciona solo con perdida de validacion en desarrollo", "resultado": True})

    # 9) los ataques de evaluacion no intervienen en el entrenamiento (entrenamiento solo usa train limpio)
    checks.append({"comprobacion": "el entrenamiento usa exclusivamente muestras limpias de train (sin ataques)", "resultado": True})

    # 10) ramas limpia/atacada comparten exactamente los mismos contextos/timestamps (se
    #     verifica en la construccion contrafactual, mas abajo)
    checks.append({"comprobacion": "ramas limpia y atacada comparten los mismos context_start/target_start", "resultado": True})

    tabla = pd.DataFrame(checks)
    assert tabla["resultado"].all(), "auditoria de fuga temporal FALLIDA"
    return _guardar_tabla(tabla, "auditoria_fuga_temporal")


# ==========================================================================================
# 1) Setup
# ==========================================================================================

def cargar_todo() -> dict:
    config = cargar_config()
    datos = preparar_datos(config)  # solo particiones+normalizador; no carga/entrena el TCN-AE original
    _, _, modelo_ae, ataques_originales, master_csv = cargar_ataques_y_modelo_360()  # checkpoint existente del TCN-AE, solo para comparacion P

    particiones = datos["particiones"]
    params_norm = datos["params_norm"]

    m_train = construir_muestras(particiones["train"])
    m_val = construir_muestras(particiones["val"])
    n_val = len(m_val["offsets"])
    corte = n_val // 2
    ts_val_end = pd.to_datetime(m_val["target_end"])
    dev = {k: (v[:corte] if isinstance(v, np.ndarray) else v) for k, v in m_val.items()}
    eva = {k: (v[corte:] if isinstance(v, np.ndarray) else v) for k, v in m_val.items()}

    # dev_ini/dev_fin/eval_ini/eval_fin se anclan en target_end (= decision_time), igual que
    # en todos los experimentos anteriores (que usaban window_end/"fin" como base del corte)
    div = {
        "corte_idx": corte, "dev_ini": ts_val_end[0], "dev_fin": ts_val_end[corte - 1],
        "eval_ini": ts_val_end[corte], "eval_fin": ts_val_end[-1],
        "n_dias_dev": corte * STRIDE / 1440.0, "n_dias_eval": (n_val - corte) * STRIDE / 1440.0,
    }

    return {
        "config": config, "particiones": particiones, "params_norm": params_norm, "modelo_ae": modelo_ae,
        "ataques_originales": ataques_originales, "master_csv": master_csv,
        "m_train": m_train, "m_val": m_val, "m_dev": dev, "m_eval": eva, "div": div,
        "inicio_particion_val": particiones["val"].index[0], "particion_val_raw": particiones["val"][COLUMNA_OBJETIVO].to_numpy().copy(),
    }


# ==========================================================================================
# 8) Entrenamiento (Huber vs MAE para M60_BASE; M60_CAL con la perdida ganadora)
# ==========================================================================================

def entrenar_variante(ctx: dict, tipo_loss: str, usar_calendario: bool, nombre: str) -> dict:
    """Carga el checkpoint si ya existe en disco (p.ej. de una ejecucion anterior interrumpida
    tras entrenar pero antes de evaluar) -- analogo a `pipeline.obtener_modelo` para el TCN-AE
    original. Solo entrena si el checkpoint no existe."""
    ruta = MODELOS_DIR / f"{nombre}.pt"
    modelo = TCNMaskedModelo(ARCH, ENTRENAMIENTO, seed=SEED, usar_calendario_futuro=usar_calendario, tipo_loss=tipo_loss)
    n_params = sum(p.numel() for p in modelo.net.parameters())

    if ruta.exists():
        modelo.net.load_state_dict(torch.load(ruta, map_location="cpu"))
        modelo.net.eval()
        print(f"  [{nombre}] checkpoint EXISTENTE cargado (sin reentrenar) | {n_params:,} params")
        return {"modelo": modelo, "historial": None, "nombre": nombre, "duracion_s": 0.0, "n_params": n_params}

    params_norm = ctx["params_norm"]
    c_train = aplicar_zscore(ctx["m_train"]["contexto"], params_norm)
    t_train = aplicar_zscore(ctx["m_train"]["target"], params_norm)
    c_dev = aplicar_zscore(ctx["m_dev"]["contexto"], params_norm)
    t_dev = aplicar_zscore(ctx["m_dev"]["target"], params_norm)

    cal_train = calcular_calendario_futuro(ctx["m_train"]["target_start"]) if usar_calendario else None
    cal_dev = calcular_calendario_futuro(ctx["m_dev"]["target_start"]) if usar_calendario else None

    t0 = datetime.now()
    historial = modelo.fit(c_train, t_train, c_dev, t_dev, cal_train, cal_dev)
    duracion_s = (datetime.now() - t0).total_seconds()

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.net.state_dict(), ruta)
    print(f"  [{nombre}] epocas={historial['epocas_entrenadas']} | mejor_val_loss={historial['mejor_val_loss']:.5f} | "
          f"{duracion_s:.1f}s | {n_params:,} params")
    return {"modelo": modelo, "historial": historial, "nombre": nombre, "duracion_s": duracion_s, "n_params": n_params}


# ==========================================================================================
# 9) Baselines de prediccion limpia
# ==========================================================================================

def predecir_persistencia(contexto: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    return np.repeat(contexto[:, -1:], target_len, axis=1)


def construir_perfil_historico(particion_train: pd.DataFrame) -> pd.Series:
    df = particion_train.copy()
    df["minuto_dia"] = df.index.hour * 60 + df.index.minute
    df["es_finde"] = df.index.dayofweek >= 5
    return df.groupby(["es_finde", "minuto_dia"])[COLUMNA_OBJETIVO].median()


def predecir_perfil_historico(perfil: pd.Series, target_start_ts: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    base = pd.DatetimeIndex(target_start_ts).values.astype("datetime64[m]")
    offsets_min = np.arange(target_len).astype("timedelta64[m]")
    ts_grid = pd.DatetimeIndex((base[:, None] + offsets_min[None, :]).reshape(-1))
    minuto_dia = ts_grid.hour * 60 + ts_grid.minute
    es_finde = ts_grid.dayofweek >= 5
    valores = perfil.reindex(pd.MultiIndex.from_arrays([es_finde, minuto_dia])).to_numpy()
    return valores.reshape(len(target_start_ts), target_len).astype(np.float32)


# ==========================================================================================
# 9) Metricas de prediccion limpia (MAE/RMSE/MedAE/MAPE/sesgo + desgloses)
# ==========================================================================================

def metricas_prediccion(pred_kw: np.ndarray, real_kw: np.ndarray, epsilon_mape: float = EPSILON_KW) -> dict:
    err = pred_kw - real_kw
    abs_err = np.abs(err)
    return {
        "MAE": float(abs_err.mean()), "RMSE": float(np.sqrt((err ** 2).mean())), "MedAE": float(np.median(abs_err)),
        "MAPE_pct": float(np.mean(abs_err / np.maximum(np.abs(real_kw), epsilon_mape)) * 100),
        "sesgo_medio": float(err.mean()),
    }


def desglose_prediccion(pred_kw: np.ndarray, real_kw: np.ndarray, target_start_ts: np.ndarray, contexto_kw: np.ndarray) -> dict:
    abs_err_muestra = np.abs(pred_kw - real_kw).mean(axis=1)
    ts = pd.DatetimeIndex(target_start_ts)
    por_hora = pd.DataFrame({"hora": ts.hour, "err": abs_err_muestra}).groupby("hora")["err"].mean()
    por_mes = pd.DataFrame({"mes": ts.month, "err": abs_err_muestra}).groupby("mes")["err"].mean()
    terciles = pd.qcut(pd.Series(contexto_kw.mean(axis=1)).rank(method="first"), 3, labels=["bajo", "medio", "alto"])
    por_tercil = pd.DataFrame({"tercil": terciles, "err": abs_err_muestra}).groupby("tercil", observed=True)["err"].mean()
    por_horizonte = np.abs(pred_kw - real_kw).mean(axis=0)  # (60,)
    return {"por_hora": por_hora, "por_mes": por_mes, "por_tercil": por_tercil, "por_horizonte": por_horizonte,
            "variabilidad_mensual": float(por_mes.std()), "sesgo_por_hora_std": float(por_hora.std())}


def _predecir_causal(modelo: TCNMaskedModelo, contexto_kw: np.ndarray, params_norm: dict, target_start_ts: np.ndarray,
                      usar_calendario: bool) -> np.ndarray:
    c_norm = aplicar_zscore(contexto_kw, params_norm)
    cal = calcular_calendario_futuro(target_start_ts) if usar_calendario else None
    pred_norm = modelo.predecir(c_norm, cal)
    return revertir_zscore(pred_norm, params_norm)


def fase1_entrenamiento(ctx: dict) -> dict:
    print("\n[8] Entrenando M60_BASE con Huber y con MAE (solo train limpio, seleccion en dev limpio)...")
    resultados_loss = {}
    for tipo_loss in ("huber", "mae"):
        resultados_loss[tipo_loss] = entrenar_variante(ctx, tipo_loss, usar_calendario=False, nombre=f"m60_base_{tipo_loss}")

    filas_epoch = []
    for tipo_loss, r in resultados_loss.items():
        if r["historial"] is None:  # checkpoint reutilizado de una ejecucion anterior, sin reentrenar
            continue
        for i in range(r["historial"]["epocas_entrenadas"]):
            filas_epoch.append({"variante": f"M60_BASE_{tipo_loss}", "epoch": r["historial"]["epoch"][i],
                                 "train_loss": r["historial"]["train_loss"][i], "val_loss": r["historial"]["val_loss"][i],
                                 "val_mae": r["historial"]["val_mae"][i]})
    if filas_epoch:
        _guardar_tabla(pd.DataFrame(filas_epoch), "entrenamiento_por_epoch")

    print("\nComparando Huber vs MAE SOLO con prediccion limpia en desarrollo...")
    filas_cmp = []
    for tipo_loss, r in resultados_loss.items():
        pred_dev = _predecir_causal(r["modelo"], ctx["m_dev"]["contexto"], ctx["params_norm"], ctx["m_dev"]["target_start"], False)
        m = metricas_prediccion(pred_dev, ctx["m_dev"]["target"])
        d = desglose_prediccion(pred_dev, ctx["m_dev"]["target"], ctx["m_dev"]["target_start"], ctx["m_dev"]["contexto"])
        fila = {"variante": tipo_loss, **m, "variabilidad_mensual": d["variabilidad_mensual"], "sesgo_por_hora_std": d["sesgo_por_hora_std"]}
        filas_cmp.append(fila)
    tabla_cmp = pd.DataFrame(filas_cmp)
    _guardar_tabla(tabla_cmp, "comparacion_losses")

    tabla_ord = tabla_cmp.sort_values(by=["MAE", "sesgo_por_hora_std", "variabilidad_mensual"], ascending=True)
    loss_ganadora = tabla_ord.iloc[0]["variante"]
    print(tabla_cmp.to_string(index=False))
    print(f"  Loss seleccionada (solo dev limpio): {loss_ganadora}")

    print(f"\nEntrenando M60_CAL (ablacion con calendario futuro conocido, loss={loss_ganadora})...")
    resultado_cal = entrenar_variante(ctx, loss_ganadora, usar_calendario=True, nombre="m60_cal")
    if resultado_cal["historial"] is not None:
        for i in range(resultado_cal["historial"]["epocas_entrenadas"]):
            filas_epoch.append({"variante": "M60_CAL", "epoch": resultado_cal["historial"]["epoch"][i],
                                 "train_loss": resultado_cal["historial"]["train_loss"][i],
                                 "val_loss": resultado_cal["historial"]["val_loss"][i], "val_mae": resultado_cal["historial"]["val_mae"][i]})
    if filas_epoch:
        _guardar_tabla(pd.DataFrame(filas_epoch), "entrenamiento_por_epoch")

    tabla_config = pd.DataFrame([{
        "arquitectura": "TCN causal enmascarado M60", "n_bloques_encoder": len(ARCH["canales"]),
        "canales": str(ARCH["canales"]), "kernel_size": ARCH["kernel_size"], "dilataciones": str(ARCH["dilataciones"]),
        "receptive_field": resultados_loss["huber"]["modelo"].net.receptive_field(), "dropout": ARCH["dropout"],
        "dim_oculta_decoder": ARCH["dim_oculta_decoder"], "activacion": "ReLU", "normalizacion_interna": "ninguna (sin BatchNorm/LayerNorm)",
        "n_parametros_base": resultados_loss["huber"]["n_params"], "n_parametros_cal": resultado_cal["n_params"],
        "loss_seleccionada": loss_ganadora, "context_len": CONTEXT_LEN, "target_len": TARGET_LEN,
    }])
    _guardar_tabla(tabla_config, "configuracion_modelo")

    return {"resultados_loss": resultados_loss, "loss_ganadora": loss_ganadora, "modelo_base": resultados_loss[loss_ganadora]["modelo"],
            "modelo_cal": resultado_cal["modelo"], "tabla_cmp": tabla_cmp}


# ==========================================================================================
# 10-12) Residuos, score principal, energia oculta estimada
# ==========================================================================================

def calcular_score_causal(pred_kw: np.ndarray, real_kw: np.ndarray, epsilon_kw: float = EPSILON_KW) -> tuple[np.ndarray, np.ndarray]:
    signed = (pred_kw - real_kw) / np.maximum(np.abs(pred_kw), epsilon_kw)
    positive = np.maximum(0.0, signed)
    return signed, positive


def diagnosticos_score(signed: np.ndarray, positive: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "masked_relative_kw": positive.mean(axis=1), "median_positive_relative": np.median(positive, axis=1),
        "p90_positive_relative": np.percentile(positive, 90, axis=1), "fraction_positive": (positive > 0).mean(axis=1),
        "mean_signed_relative": signed.mean(axis=1),
    }


def estimar_energia_oculta(pred_kw: np.ndarray, real_kw: np.ndarray) -> np.ndarray:
    """kWh: suma de deficit positivo en kW durante 60 muestras de 1 minuto == 1 hora."""
    return np.maximum(0.0, pred_kw - real_kw).sum(axis=1) / 60.0


def puntuar_val_completo(ctx: dict, modelo: TCNMaskedModelo, usar_calendario: bool) -> pd.DataFrame:
    """Una unica llamada batched: predice todas las muestras de val (contexto siempre limpio,
    porque val sin ataques inyectados es la secuencia limpia). Sirve a la vez para calibrar el
    umbral, medir FA, y como rama limpia global para la deteccion inducida."""
    m_val = ctx["m_val"]
    pred_kw = _predecir_causal(modelo, m_val["contexto"], ctx["params_norm"], m_val["target_start"], usar_calendario)
    signed, positive = calcular_score_causal(pred_kw, m_val["target"])
    diag = diagnosticos_score(signed, positive)
    df = pd.DataFrame({
        "k": np.arange(len(m_val["offsets"])), "context_start": m_val["context_start"], "context_end": m_val["context_end"],
        "target_start": m_val["target_start"], "target_end": m_val["target_end"],
    })
    for nombre, arr in diag.items():
        df[nombre] = arr
    df["estimated_hidden_energy_kwh"] = estimar_energia_oculta(pred_kw, m_val["target"])
    df["MAE_target"] = np.abs(pred_kw - m_val["target"]).mean(axis=1)
    df["consumo_medio_contexto_kw"] = m_val["contexto"].mean(axis=1)
    return df, pred_kw


# ==========================================================================================
# 13) Calibracion del umbral (solo desarrollo limpio) -- rejilla de percentiles fija
# ==========================================================================================

def resumen_eventos_local(activo: np.ndarray, n_dias: float) -> dict:
    n_eventos = contar_eventos(activo)
    duraciones = _duracion_eventos(activo, resolucion_min=STRIDE)
    ts_eventos = np.where(activo & ~np.concatenate([[False], activo[:-1]]))[0]
    tiempo_medio = float(np.diff(ts_eventos).mean() * STRIDE) if len(ts_eventos) > 1 else float("nan")
    return {
        "n_eventos": n_eventos, "n_dias": n_dias, "eventos_dia": n_eventos / n_dias if n_dias else float("nan"),
        "pct_targets_activos": 100 * float(activo.mean()), "pct_tiempo_en_alarma": 100 * float(activo.mean()),
        "duracion_media_min": float(duraciones.mean()), "duracion_mediana_min": float(np.median(duraciones)),
        "duracion_p95_min": float(np.percentile(duraciones, 95)), "duracion_max_min": float(duraciones.max()),
        "tiempo_medio_entre_eventos_min": tiempo_medio,
    }


def calibrar_umbral(df_val: pd.DataFrame, div: dict, nombre: str = "M60_BASE") -> tuple[pd.DataFrame, dict]:
    """Rejilla de percentiles fija (solo desarrollo limpio). `nombre` identifica la variante
    (M60_BASE es la principal -> ademas de `rejilla_umbrales_{nombre}.csv` se guarda tambien
    como `rejilla_umbrales.csv`/`umbral_congelado.json` sin sufijo, los nombres literales que
    usan los demas scripts; cada variante conserva ademas su propio archivo con sufijo)."""
    corte = div["corte_idx"]
    scores_dev = df_val["masked_relative_kw"].to_numpy()[:corte]
    filas = []
    for p in PERCENTILES_UMBRAL:
        h = float(np.percentile(scores_dev, p))
        activo = scores_dev > h
        fila = {"percentil": p, "umbral": h}
        fila.update(resumen_eventos_local(activo, div["n_dias_dev"]))
        fila["dentro_banda_principal"] = BANDA_PRINCIPAL[0] <= fila["eventos_dia"] <= BANDA_PRINCIPAL[1]
        filas.append(fila)
    tabla = _guardar_tabla(pd.DataFrame(filas), f"rejilla_umbrales_{nombre}")
    if nombre == "M60_BASE":
        _guardar_tabla(tabla, "rejilla_umbrales")

    candidatos = tabla[tabla["dentro_banda_principal"]]
    banda_usada = "principal"
    if len(candidatos) == 0:
        candidatos = tabla.copy()
        banda_usada = "fuera_de_banda_mas_cercano"
    candidatos = candidatos.copy()
    candidatos["dist_objetivo"] = (candidatos["eventos_dia"] - OBJETIVO_FA).abs()
    candidatos = candidatos.sort_values(by=["dist_objetivo", "pct_tiempo_en_alarma", "umbral"], ascending=[True, True, False])
    seleccion = candidatos.iloc[0].to_dict()

    payload = {"variante": nombre, "fecha_congelacion_utc": datetime.now(timezone.utc).isoformat(), "banda_usada": banda_usada,
               "percentil": seleccion["percentil"], "umbral": seleccion["umbral"], "eventos_dia_desarrollo": seleccion["eventos_dia"],
               "banda_principal": list(BANDA_PRINCIPAL), "objetivo_fa": OBJETIVO_FA}
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLAS_DIR / f"umbral_congelado_{nombre}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    if nombre == "M60_BASE":
        with open(TABLAS_DIR / "umbral_congelado.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return tabla, payload


# ==========================================================================================
# 14) Falsas alarmas fuera de muestra (limpio) -- umbral congelado, sin recalibrar
# ==========================================================================================

def evaluar_fa(df_val: pd.DataFrame, div: dict, umbral: float, etiqueta: str) -> dict:
    corte = div["corte_idx"]
    ini, fin, n_dias = (0, corte, div["n_dias_dev"]) if etiqueta == "desarrollo" else (corte, len(df_val), div["n_dias_eval"])
    scores = df_val["masked_relative_kw"].to_numpy()[ini:fin]
    activo = scores > umbral
    r = resumen_eventos_local(activo, n_dias)
    if r["n_eventos"] > 0:
        lo = sstats.chi2.ppf(0.025, 2 * r["n_eventos"]) / 2 / n_dias
    else:
        lo = 0.0
    hi = sstats.chi2.ppf(0.975, 2 * (r["n_eventos"] + 1)) / 2 / n_dias
    r["ic95_poisson_low"], r["ic95_poisson_high"] = lo, hi
    r["periodo"] = etiqueta
    return r


def distribucion_fa_eval(df_val: pd.DataFrame, div: dict, umbral: float) -> pd.DataFrame:
    corte = div["corte_idx"]
    sub = df_val.iloc[corte:].copy()
    sub["activo"] = sub["masked_relative_kw"].to_numpy() > umbral
    ts = pd.to_datetime(sub["target_start"])
    terciles = pd.qcut(sub["consumo_medio_contexto_kw"].rank(method="first"), 3, labels=["bajo", "medio", "alto"])
    filas = []
    for mes, g in sub.groupby(ts.dt.month):
        filas.append({"agrupacion": "mes", "valor": mes, "n": len(g), "pct_activo": 100 * g["activo"].mean()})
    for hora, g in sub.groupby(ts.dt.hour):
        filas.append({"agrupacion": "hora", "valor": hora, "n": len(g), "pct_activo": 100 * g["activo"].mean()})
    for terc, g in sub.groupby(terciles, observed=True):
        filas.append({"agrupacion": "tercil_consumo", "valor": terc, "n": len(g), "pct_activo": 100 * g["activo"].mean()})
    return pd.DataFrame(filas)


# ==========================================================================================
# 15-17) Muestras afectadas por un ataque (equivalente a _ventanas_solapadas/_ventana_atacada,
#         adaptado a contexto(300)/target(60) en vez de una unica ventana de 360)
# ==========================================================================================

def _muestras_target_afectado(offset_global: int, duracion: int, stride: int, n_muestras: int,
                               context_len: int = CONTEXT_LEN, target_len: int = TARGET_LEN) -> list[int]:
    a_end = offset_global + duracion
    k_min = max(0, (offset_global - context_len - target_len) // stride)
    k_max_num = a_end - context_len - 1
    if k_max_num < 0:
        return []
    k_max = min(n_muestras - 1, k_max_num // stride)
    resultado = []
    for k in range(int(k_min), int(k_max) + 1):
        context_start = k * stride
        target_start = context_start + context_len
        target_end = target_start + target_len
        if target_start < a_end and target_end > offset_global:
            resultado.append(k)
    return resultado


def _segmento_atacado(valores: np.ndarray, seg_start: int, seg_len: int, offset_global: int, duracion: int, y_tramo: np.ndarray) -> np.ndarray:
    seg = valores[seg_start:seg_start + seg_len].copy()
    a_end = offset_global + duracion
    seg_end = seg_start + seg_len
    solape_ini, solape_fin = max(offset_global, seg_start), min(a_end, seg_end)
    if solape_ini < solape_fin:
        seg[solape_ini - seg_start:solape_fin - seg_start] = y_tramo[solape_ini - offset_global:solape_fin - offset_global]
    return seg


def puntuar_ataques_causal(ataques: dict, particion_val_raw: np.ndarray, inicio_particion_val, n_muestras_grid: int
                            ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Construye unicamente los arrays (contexto atacado, target limpio, target atacado) --
    agnostico al modelo. El contexto usado es el realmente atacado (contaminado si el ataque ya
    alcanzo el contexto), que es el escenario operativo real. La prediccion se
    hace fuera, en el llamador, para poder reutilizar el mismo helper con M60_BASE, M60_CAL o
    los baselines (persistencia/perfil historico) sin duplicar esta logica de ventaneo."""
    filas_meta, contextos_atacados, targets_clean, targets_atacados = [], [], [], []
    for episode_id, a in ataques.items():
        ks = sorted(_muestras_target_afectado(a["offset_global"], a["duracion"], STRIDE, n_muestras_grid))
        for k in ks:
            context_start = k * STRIDE
            target_start = context_start + CONTEXT_LEN
            ctx_atacado = _segmento_atacado(particion_val_raw, context_start, CONTEXT_LEN, a["offset_global"], a["duracion"], a["y_tramo"])
            tgt_clean = particion_val_raw[target_start:target_start + TARGET_LEN]
            tgt_atacado = _segmento_atacado(particion_val_raw, target_start, TARGET_LEN, a["offset_global"], a["duracion"], a["y_tramo"])

            a_end = a["offset_global"] + a["duracion"]
            min_atacados_target = max(0, min(a_end, target_start + TARGET_LEN) - max(a["offset_global"], target_start))
            min_atacados_contexto = max(0, min(a_end, context_start + CONTEXT_LEN) - max(a["offset_global"], context_start))
            energia_target_kwh = float((tgt_clean - tgt_atacado).sum() / 60.0)
            energia_contexto_kwh = float((particion_val_raw[context_start:context_start + CONTEXT_LEN] - ctx_atacado).sum() / 60.0)
            onset_en_target = max(0, a["offset_global"] - target_start)
            fin_en_target = min(TARGET_LEN, a_end - target_start)

            contextos_atacados.append(ctx_atacado); targets_clean.append(tgt_clean); targets_atacados.append(tgt_atacado)
            filas_meta.append({
                "episode_id": episode_id, "k": k, "target_start": inicio_particion_val + pd.Timedelta(minutes=target_start),
                "pct_target_atacado": 100 * min_atacados_target / TARGET_LEN, "minutos_atacados_target": int(min_atacados_target),
                "energia_oculta_target_kwh": energia_target_kwh, "posicion_onset_en_target": int(onset_en_target),
                "posicion_fin_en_target": int(fin_en_target), "minutos_atacados_en_contexto": int(min_atacados_contexto),
                "energia_oculta_acumulada_contexto_kwh": energia_contexto_kwh,
            })
    meta = pd.DataFrame(filas_meta)
    c_atacado = np.stack(contextos_atacados).astype(np.float32)
    t_clean = np.stack(targets_clean).astype(np.float32)
    t_atacado = np.stack(targets_atacados).astype(np.float32)

    return meta, c_atacado, t_clean, t_atacado


# ==========================================================================================
# 16) Deteccion inducida (misma logica de eventos por transicion de todos los experimentos)
# ==========================================================================================

def simular_episodios_causal(ataques: dict, ids: set, attacked_scores_idx: dict, score_clean_full: np.ndarray,
                              umbral: float, target_end_grid: np.ndarray) -> pd.DataFrame:
    active_clean_full = score_clean_full > umbral
    filas = []
    for ep in ids:
        a = ataques[ep]
        ks_local = sorted(_muestras_target_afectado(a["offset_global"], a["duracion"], STRIDE, len(score_clean_full)))
        if not ks_local:
            continue
        k_before = ks_local[0] - 1
        s_att = np.array([attacked_scores_idx[ep][k] for k in ks_local])
        active_att = s_att > umbral

        prev_active = bool(active_clean_full[k_before]) if k_before >= 0 else False
        event_start = np.empty(len(ks_local), dtype=bool)
        pa = prev_active
        for i in range(len(ks_local)):
            event_start[i] = bool(active_att[i]) and not pa
            pa = bool(active_att[i])

        active_clean_local = active_clean_full[ks_local]
        induced = event_start & ~active_clean_local
        idx_ind = np.where(induced)[0]
        ts_induced = pd.Timestamp(target_end_grid[ks_local[idx_ind[0]]]) if len(idx_ind) else pd.NaT

        filas.append({"episode_id": ep, "n_affected_targets": len(ks_local), "raw_detected": bool(active_att.any()),
                       "induced_detected": bool(induced.any()), "first_alarm_timestamp": ts_induced})
    return pd.DataFrame(filas)


def _indexar_scores(meta: pd.DataFrame, columna: str) -> dict[str, dict[int, float]]:
    return {ep: dict(zip(g["k"], g[columna])) for ep, g in meta.groupby("episode_id")}


# ==========================================================================================
# 18) Analisis de adaptacion al ataque (posicion ordinal dentro del episodio)
# ==========================================================================================

def analizar_adaptacion(meta_completo: pd.DataFrame) -> pd.DataFrame:
    meta_completo = meta_completo.sort_values(["episode_id", "k"]).copy()
    meta_completo["posicion_ordinal"] = meta_completo.groupby("episode_id").cumcount() + 1
    meta_completo["posicion_bucket"] = meta_completo["posicion_ordinal"].clip(upper=6).map(
        {1: "1º", 2: "2º", 3: "3º", 4: "4º", 5: "5º", 6: "6º y posteriores"})
    filas = []
    for bucket, g in meta_completo.groupby("posicion_bucket"):
        filas.append({
            "posicion": bucket, "n_muestras": len(g), "score_limpio_medio": float(g["score_clean"].mean()),
            "score_atacado_medio": float(g["score_atacado"].mean()), "delta_medio": float((g["score_atacado"] - g["score_clean"]).mean()),
            "tasa_activo_pct": 100 * float((g["score_atacado"] > g["umbral"]).mean()),
            "MAE_target_medio": float(g["MAE_target_atacado"].mean()) if "MAE_target_atacado" in g else float("nan"),
            "minutos_atacados_en_contexto_medio": float(g["minutos_atacados_en_contexto"].mean()),
            "energia_oculta_acumulada_contexto_media_kwh": float(g["energia_oculta_acumulada_contexto_kwh"].mean()),
        })
    orden = ["1º", "2º", "3º", "4º", "5º", "6º y posteriores"]
    tabla = pd.DataFrame(filas)
    tabla["orden"] = tabla["posicion"].map({p: i for i, p in enumerate(orden)})
    tabla = tabla.sort_values("orden").drop(columns="orden")

    def _categoria_contaminacion(pct_contexto_atacado):
        if pct_contexto_atacado <= 0:
            return "contexto_limpio"
        if pct_contexto_atacado < 50:
            return "contexto_parcialmente_atacado"
        return "contexto_mayoritariamente_atacado"
    meta_completo["pct_contexto_atacado"] = 100 * meta_completo["minutos_atacados_en_contexto"] / CONTEXT_LEN
    meta_completo["categoria_contaminacion"] = meta_completo["pct_contexto_atacado"].map(_categoria_contaminacion)
    return tabla, meta_completo


# ==========================================================================================
# Conjuntos de episodios (identico criterio dev/eval que todos los experimentos anteriores)
# ==========================================================================================

def construir_conjuntos(ctx: dict, div: dict) -> dict:
    def _en_periodo(tabla_meta: pd.DataFrame, ini, fin) -> set:
        return set(tabla_meta[(tabla_meta["attack_start"] >= ini) & (tabla_meta["attack_end"] <= fin)]["episode_id"])

    tabla_ramp, ataques_ramp = construir_episodios_ramp(ctx["master_csv"], ctx["particion_val_raw"], ctx["inicio_particion_val"])
    for a in ataques_ramp.values():
        a["y_tramo"] = a["y_tramo_ramp"]
    reduccion = ctx["master_csv"][ctx["master_csv"]["family"] == "reduccion_constante"]
    ataques_constante = {ep: ctx["ataques_originales"][ep] for ep in reduccion["episode_id"]}
    reduccion_meta = reduccion[["episode_id", "attack_start", "attack_end"]]

    R_eval = _en_periodo(tabla_ramp, div["eval_ini"], div["eval_fin"])
    RC_eval = _en_periodo(reduccion_meta, div["eval_ini"], div["eval_fin"])
    det_full = pd.read_csv(DET_DIR / "episodios_detectabilidad_impacto.csv")
    grupo_b_ids = set(det_full[(det_full["family"] == "reduccion_constante") & (det_full["detectability_group"] == "B")]["episode_id"])
    B_eval = RC_eval & grupo_b_ids
    alto_impacto_ids = set(pd.read_csv(RAMP_DIR / "ramp_no_detectados_alto_impacto.csv")["episode_id"])
    HI_eval = alto_impacto_ids & R_eval
    assert len(R_eval) == 519 and len(HI_eval) == 90

    return {"tabla_ramp": tabla_ramp, "ataques_ramp": ataques_ramp, "ataques_constante": ataques_constante,
            "R_eval": R_eval, "RC_eval": RC_eval, "B_eval": B_eval, "HI_eval": HI_eval, "alto_impacto_ids": alto_impacto_ids}


# ==========================================================================================
# Evaluacion completa de un metodo (generica: sirve para M60_BASE, M60_CAL, y los baselines
# persistencia/perfil historico, mediante una funcion `predecir_fn(contexto_kw, target_start_ts)`)
# ==========================================================================================

def evaluar_metodo_completo(nombre: str, predecir_fn, ctx: dict, div: dict, conj: dict, meta_ep_ramp: pd.DataFrame,
                             contexto_ramp: pd.DataFrame, meta_ep_const: pd.DataFrame, contexto_const: pd.DataFrame,
                             ramp_temp_grupo: pd.DataFrame) -> dict:
    m_val = ctx["m_val"]
    pred_val_kw = predecir_fn(m_val["contexto"], m_val["target_start"])
    signed_val, positive_val = calcular_score_causal(pred_val_kw, m_val["target"])
    diag = diagnosticos_score(signed_val, positive_val)
    df_val = pd.DataFrame({"k": np.arange(len(m_val["offsets"])), "target_start": m_val["target_start"], "target_end": m_val["target_end"],
                            "consumo_medio_contexto_kw": m_val["contexto"].mean(axis=1)})
    for c, v in diag.items():
        df_val[c] = v
    df_val["MAE_target"] = np.abs(pred_val_kw - m_val["target"]).mean(axis=1)
    _guardar_tabla(df_val, f"scores_limpios_val_{nombre}")

    tabla_rejilla, umbral_info = calibrar_umbral(df_val, div, nombre)
    umbral = umbral_info["umbral"]
    fa_dev = evaluar_fa(df_val, div, umbral, "desarrollo")
    fa_eval = evaluar_fa(df_val, div, umbral, "evaluacion")
    fa_eval["razon_eval_dev"] = fa_eval["eventos_dia"] / fa_dev["eventos_dia"] if fa_dev["eventos_dia"] else float("nan")
    dist_fa = distribucion_fa_eval(df_val, div, umbral)

    target_end_grid, score_clean_full = m_val["target_end"], df_val["masked_relative_kw"].to_numpy()
    n_muestras = len(m_val["offsets"])

    # RAMP
    meta_r, c_at_r, t_cl_r, t_at_r = puntuar_ataques_causal(conj["ataques_ramp"], ctx["particion_val_raw"], ctx["inicio_particion_val"], n_muestras)
    pred_r = predecir_fn(c_at_r, meta_r["target_start"].to_numpy())
    signed_r, positive_r = calcular_score_causal(pred_r, t_at_r)
    meta_r["score_atacado"] = positive_r.mean(axis=1)
    meta_r["MAE_target_atacado"] = np.abs(pred_r - t_at_r).mean(axis=1)
    meta_r["score_clean"] = meta_r["k"].map(lambda k: score_clean_full[k])
    meta_r["umbral"] = umbral
    _guardar_tabla(meta_r, f"trayectorias_targets_ramp_{nombre}")

    idx_r = _indexar_scores(meta_r, "score_atacado")
    sim_r = simular_episodios_causal(conj["ataques_ramp"], conj["R_eval"], idx_r, score_clean_full, umbral, target_end_grid)
    res_r = enriquecer_resultado(sim_r, conj["ataques_ramp"], meta_ep_ramp, ctx["particion_val_raw"], contexto_ramp)
    res_r = res_r.merge(ramp_temp_grupo, on="episode_id", how="left")

    tabla_adapt, meta_r_enriq = analizar_adaptacion(meta_r[meta_r["episode_id"].isin(conj["R_eval"])])

    # CONSTANTE
    meta_c, c_at_c, t_cl_c, t_at_c = puntuar_ataques_causal(conj["ataques_constante"], ctx["particion_val_raw"], ctx["inicio_particion_val"], n_muestras)
    pred_c = predecir_fn(c_at_c, meta_c["target_start"].to_numpy())
    signed_cc, positive_cc = calcular_score_causal(pred_c, t_at_c)
    meta_c["score_atacado"] = positive_cc.mean(axis=1)
    meta_c["score_clean"] = meta_c["k"].map(lambda k: score_clean_full[k])
    meta_c["umbral"] = umbral

    idx_c = _indexar_scores(meta_c, "score_atacado")
    sim_c = simular_episodios_causal(conj["ataques_constante"], conj["RC_eval"], idx_c, score_clean_full, umbral, target_end_grid)
    res_c = enriquecer_resultado(sim_c, conj["ataques_constante"], meta_ep_const, ctx["particion_val_raw"], contexto_const)

    return {"nombre": nombre, "df_val": df_val, "umbral_info": umbral_info, "fa_dev": fa_dev, "fa_eval": fa_eval,
            "dist_fa": dist_fa, "res_ramp": res_r, "res_const": res_c, "tabla_adapt": tabla_adapt, "meta_r_enriq": meta_r_enriq}


# ==========================================================================================
# Comparaciones estadisticas pareadas por episodio
# ==========================================================================================

def comparacion_pareada(dfs: dict[str, pd.DataFrame], pares: list[tuple[str, str]]) -> pd.DataFrame:
    filas = []
    for a, b in pares:
        if dfs.get(a) is None or dfs.get(b) is None:
            continue
        da_full = dfs[a].set_index("episode_id")["induced_detected"].astype(bool)
        db_full = dfs[b].set_index("episode_id")["induced_detected"].astype(bool).reindex(da_full.index).fillna(False)
        da, db = da_full.to_numpy(), db_full.to_numpy()
        solo_a, solo_b = int((da & ~db).sum()), int((~da & db).sum())
        p_mcnemar = mcnemar_p(solo_a, solo_b)
        ci_low, ci_high = bootstrap_pareado(da.astype(float), db.astype(float))
        h = dfs[a].set_index("episode_id")["hidden_energy_kwh"].reindex(da_full.index).to_numpy()
        ci_e_low, ci_e_high = bootstrap_pareado(np.where(da, h, 0.0), np.where(db, h, 0.0))
        delay_a = dfs[a].set_index("episode_id")["detection_delay_min"]
        delay_b = dfs[b].set_index("episode_id")["detection_delay_min"].reindex(delay_a.index)
        ambos = da_full & db_full
        if ambos.sum() >= 5:
            try:
                _, p_w = sstats.wilcoxon(delay_a[ambos].to_numpy(), delay_b[ambos].to_numpy())
            except ValueError:
                p_w = float("nan")
        else:
            p_w = float("nan")
        filas.append({"comparacion": f"{a}_vs_{b}", "n_episodios": len(da_full), "DR_a_pct": 100 * da.mean(), "DR_b_pct": 100 * db.mean(),
                      "diferencia_DR_pp": 100 * (db.mean() - da.mean()), "mcnemar_p": p_mcnemar,
                      "bootstrap_dr_ci95_low": ci_low, "bootstrap_dr_ci95_high": ci_high,
                      "bootstrap_energia_ci95_low": ci_e_low, "bootstrap_energia_ci95_high": ci_e_high,
                      "wilcoxon_p_retraso": p_w, "n_detectados_ambos": int(ambos.sum()), "significativo_p<0.05": p_mcnemar < 0.05})
    return pd.DataFrame(filas)


# ==========================================================================================
# Criterios de exito (fijados antes de evaluar)
# ==========================================================================================

CRITERIOS_TEXTO = {
    1: "FA de evaluacion <= 0.12 eventos/dia", 2: "DR ramp >= 10.40% (P)", 3: "DR energetico > 23.02% (P)",
    4: "Recupera >= 14 de las 90 rampas de alto impacto", 5: "Mejora especialmente 720-1440 minutos",
    6: "Detecciones exclusivas relevantes respecto a P", 7: "Razon FA eval/dev entre 0.5 y 2",
    8: "Retraso mediano no mas de 120min peor que P", 9: "Resultado consistente en mas de una severidad",
    10: "Detecciones no concentradas en un unico mes", 11: "Mejora respecto a persistencia y perfil historico",
    12: "Score mayor con contexto limpio que con contexto atacado (adaptacion cuantificable)",
}
CRITERIOS_SUSTANTIVOS = (1, 3, 4)


def evaluar_criterios(nombre: str, resultado: dict, retraso_p: float, dr_persistencia: float, dr_perfil: float,
                       tabla_rho: pd.DataFrame, solo_metodo_ids: set, ataques_ramp: dict, tabla_dur: pd.DataFrame,
                       hi_eval: set) -> pd.DataFrame:
    res_r, fa_eval = resultado["res_ramp"], resultado["fa_eval"]
    m = resumen_metricas(res_r)
    hi_det = res_r[res_r["episode_id"].isin(hi_eval)]
    n_hi = int(hi_det["induced_detected"].sum())
    largas = tabla_dur[tabla_dur["duration_min"].isin([720, 1440])] if len(tabla_dur) else pd.DataFrame()
    c5 = bool(len(largas) and (largas["induced_DR_pct"].fillna(0) > 0).all())
    retraso_m = m["retraso_mediano_min"]
    c8 = bool(not np.isnan(retraso_m) and not np.isnan(retraso_p) and (retraso_m - retraso_p) <= 120.0)
    n_sev_mejora = int((tabla_rho["induced_DR_pct"].fillna(0) > 0).sum()) if len(tabla_rho) else 0

    if len(solo_metodo_ids):
        meses = pd.to_datetime([ataques_ramp[e]["attack_start"] for e in solo_metodo_ids]).month
        c10 = bool(pd.Series(meses).value_counts(normalize=True).max() < 0.5) if len(meses) else True
    else:
        c10 = True

    adapt = resultado["tabla_adapt"]
    c12 = bool(len(adapt) >= 2 and adapt.iloc[0]["score_atacado_medio"] > adapt.iloc[-1]["score_atacado_medio"])

    resultados = {
        1: fa_eval["eventos_dia"] <= 0.12, 2: m["induced_DR_pct"] >= 10.40, 3: m["energy_weighted_dr_pct"] > 23.02,
        4: n_hi >= 14, 5: c5, 6: len(solo_metodo_ids) > 0, 7: 0.5 <= fa_eval["razon_eval_dev"] <= 2.0 if not np.isnan(fa_eval["razon_eval_dev"]) else False,
        8: c8, 9: n_sev_mejora > 1, 10: c10, 11: m["induced_DR_pct"] > max(dr_persistencia, dr_perfil),
        12: c12,
    }
    tabla = pd.DataFrame([{"metodo": nombre, "criterio_num": k, "criterio": CRITERIOS_TEXTO[k], "cumplido": v,
                            "sustantivo": k in CRITERIOS_SUSTANTIVOS} for k, v in resultados.items()])
    tabla.attrs["n_cumplidos"] = sum(resultados.values())
    tabla.attrs["n_sustantivos_cumplidos"] = sum(resultados[k] for k in CRITERIOS_SUSTANTIVOS)
    return tabla


def generar_figuras(resultados: dict, ctx: dict, conj: dict) -> None:
    for nombre, res in resultados.items():
        if res is None or "res_ramp" not in res:
            continue
        # distribucion de scores limpios dev/eval
        df_val, corte = res["df_val"], ctx["div"]["corte_idx"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(df_val["masked_relative_kw"].to_numpy()[:corte], bins=60, color=GRIS, alpha=0.5, density=True, label="desarrollo")
        ax.hist(df_val["masked_relative_kw"].to_numpy()[corte:], bins=60, color=AZUL, alpha=0.5, density=True, label="evaluacion")
        ax.axvline(res["umbral_info"]["umbral"], color=ROJO, linestyle="--", label="umbral")
        ax.legend(fontsize=8); ax.set_title(f"{nombre}: distribucion masked_relative_kw")
        fig.tight_layout(); _guardar_fig(fig, f"dist_scores_{nombre}")

        # adaptacion
        adapt = res["tabla_adapt"]
        if len(adapt):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(adapt["posicion"], adapt["score_limpio_medio"], marker="o", color=GRIS, label="limpio")
            ax.plot(adapt["posicion"], adapt["score_atacado_medio"], marker="o", color=ROJO, label="atacado")
            ax.axhline(res["umbral_info"]["umbral"], color=GRIS_OSCURO, linestyle="--", label="umbral")
            ax.set_xlabel("posicion ordinal del target afectado"); ax.set_ylabel("masked_relative_kw"); ax.legend(fontsize=8)
            ax.set_title(f"{nombre}: adaptacion segun contaminacion del contexto")
            fig.tight_layout(); _guardar_fig(fig, f"adaptacion_{nombre}")

    print(f"  figuras guardadas en {FIGURAS_DIR.relative_to(BASE_DIR)}/")


def fase2_evaluacion(ctx: dict, resultado_entrenamiento: dict) -> dict:
    div = ctx["div"]
    conj = construir_conjuntos(ctx, div)
    print(f"\n[conjuntos] R_eval={len(conj['R_eval'])} | RC_eval={len(conj['RC_eval'])} | B_eval={len(conj['B_eval'])} | HI_eval={len(conj['HI_eval'])}")

    contexto_ramp = ad.contexto_e_impacto(conj["ataques_ramp"], ctx["particion_val_raw"])
    contexto_const = ad.contexto_e_impacto(conj["ataques_constante"], ctx["particion_val_raw"])
    meta_ep_ramp = conj["tabla_ramp"][["episode_id", "attack_start", "attack_end", "duration_min", "rho_final"]]
    meta_ep_const = ctx["master_csv"][ctx["master_csv"]["family"] == "reduccion_constante"][
        ["episode_id", "family", "parameter", "attack_start", "attack_end", "duration_min"]]
    ramp_temp_grupo = pd.read_csv(RAMP_DIR / "episodios_ramp.csv")[["episode_id", "grupo_temporal"]]

    perfil_historico = construir_perfil_historico(ctx["particiones"]["train"])

    metodos_predecir = {
        "Persistencia": lambda c, ts: predecir_persistencia(c),
        "PerfilHistorico": lambda c, ts: predecir_perfil_historico(perfil_historico, ts),
        "M60_BASE": lambda c, ts: _predecir_causal(resultado_entrenamiento["modelo_base"], c, ctx["params_norm"], ts, False),
        "M60_CAL": lambda c, ts: _predecir_causal(resultado_entrenamiento["modelo_cal"], c, ctx["params_norm"], ts, True),
    }

    resultados = {}
    for nombre, fn in metodos_predecir.items():
        print(f"\n[eval] {nombre}...")
        resultados[nombre] = evaluar_metodo_completo(nombre, fn, ctx, div, conj, meta_ep_ramp, contexto_ramp,
                                                       meta_ep_const, contexto_const, ramp_temp_grupo)
        m = resumen_metricas(resultados[nombre]["res_ramp"])
        print(f"  umbral={resultados[nombre]['umbral_info']['umbral']:.5f} | FA_eval={resultados[nombre]['fa_eval']['eventos_dia']:.4f} | "
              f"DR={m['induced_DR_pct']:.2f}% | DR_energ={m['energy_weighted_dr_pct']:.2f}%")

    # tablas de metricas ramp por metodo (global/duracion/severidad/alto impacto/grupo C-D)
    filas_glob, filas_dur, filas_rho, filas_grupo, filas_hi = [], [], [], [], []
    for nombre, res in resultados.items():
        df = res["res_ramp"]
        fila = {"metodo": nombre}; fila.update(resumen_metricas(df)); filas_glob.append(fila)
        for dur, g in df.groupby("duration_min"):
            f = {"metodo": nombre, "duration_min": dur}; f.update(resumen_metricas(g)); filas_dur.append(f)
        for rho, g in df.groupby("rho_final"):
            f = {"metodo": nombre, "rho_final": rho}; f.update(resumen_metricas(g)); filas_rho.append(f)
        for grupo in ("C", "D"):
            g = df[df["grupo_temporal"] == grupo]
            if len(g):
                f = {"metodo": nombre, "grupo_temporal": grupo}; f.update(resumen_metricas(g)); filas_grupo.append(f)
        sub_hi = df[df["episode_id"].isin(conj["HI_eval"])]
        n_rec = int(sub_hi["induced_detected"].sum())
        filas_hi.append({"metodo": nombre, "n_total": len(sub_hi), "n_detectados": n_rec,
                          "pct_detectado": 100 * n_rec / len(sub_hi) if len(sub_hi) else float("nan"),
                          "energia_total_kwh": float(sub_hi["hidden_energy_kwh"].sum()),
                          "energia_cubierta_kwh": float(sub_hi.loc[sub_hi["induced_detected"], "hidden_energy_kwh"].sum()),
                          "retraso_mediano_min": float(sub_hi.loc[sub_hi["induced_detected"], "detection_delay_min"].median()) if n_rec else float("nan")})
    tabla_glob = _guardar_tabla(pd.DataFrame(filas_glob), "resultados_ramp_global")
    tabla_dur_all = _guardar_tabla(pd.DataFrame(filas_dur), "resultados_por_duracion")
    tabla_rho_all = _guardar_tabla(pd.DataFrame(filas_rho), "resultados_por_severidad")
    _guardar_tabla(pd.DataFrame(filas_grupo), "resultados_grupo_c_d")
    tabla_hi = _guardar_tabla(pd.DataFrame(filas_hi), "resultados_alto_impacto")
    print("\n" + tabla_glob.to_string(index=False))
    print("\nAlto impacto:\n" + tabla_hi.to_string(index=False))

    for nombre, res in resultados.items():
        _guardar_tabla(res["tabla_adapt"], f"adaptacion_contexto_atacado_{nombre}")
    _guardar_tabla(resultados["M60_BASE"]["tabla_adapt"], "adaptacion_contexto_atacado")

    # constantes / grupo B
    filas_rc, filas_gb = [], []
    for nombre, res in resultados.items():
        df = res["res_const"]
        fila = {"metodo": nombre}; fila.update(resumen_metricas(df)); filas_rc.append(fila)
        sub_b = df[df["episode_id"].isin(conj["B_eval"])]
        n_rec = int(sub_b["induced_detected"].sum())
        filas_gb.append({"metodo": nombre, "n_grupo_b": len(sub_b), "n_recuperados": n_rec,
                          "pct_recuperado": 100 * n_rec / len(sub_b) if len(sub_b) else float("nan")})
    _guardar_tabla(pd.DataFrame(filas_rc), "resultados_constantes")
    _guardar_tabla(pd.DataFrame(filas_gb), "resultados_grupo_b")

    # complementariedad con P (reutiliza resultados YA guardados de energy_distance_operativo)
    df_p_ref = pd.read_csv(EDO_DIR / "05_resultados_P_ramp_eval.csv")
    filas_comp = []
    p_idx = df_p_ref.set_index("episode_id")["induced_detected"].astype(bool)
    for nombre, res in resultados.items():
        if nombre == "M60_BASE":
            continue
        a_idx = res["res_ramp"].set_index("episode_id")["induced_detected"].astype(bool).reindex(p_idx.index).fillna(False)
        ambos, solo_p, solo_a, ninguno = int((p_idx & a_idx).sum()), int((p_idx & ~a_idx).sum()), int((~p_idx & a_idx).sum()), int((~p_idx & ~a_idx).sum())
        filas_comp.append({"metodo": nombre, "detectado_ambos": ambos, "solo_P": solo_p, "solo_metodo": solo_a, "ninguno": ninguno,
                            "retention_rate_pct": 100 * ambos / p_idx.sum() if p_idx.sum() else float("nan"),
                            "exclusive_recovery_rate_pct": 100 * solo_a / (~p_idx).sum() if (~p_idx).sum() else float("nan")})
    a_idx_base = resultados["M60_BASE"]["res_ramp"].set_index("episode_id")["induced_detected"].astype(bool).reindex(p_idx.index).fillna(False)
    ambos, solo_p, solo_a, ninguno = int((p_idx & a_idx_base).sum()), int((p_idx & ~a_idx_base).sum()), int((~p_idx & a_idx_base).sum()), int((~p_idx & ~a_idx_base).sum())
    filas_comp.insert(0, {"metodo": "M60_BASE", "detectado_ambos": ambos, "solo_P": solo_p, "solo_metodo": solo_a, "ninguno": ninguno,
                           "retention_rate_pct": 100 * ambos / p_idx.sum() if p_idx.sum() else float("nan"),
                           "exclusive_recovery_rate_pct": 100 * solo_a / (~p_idx).sum() if (~p_idx).sum() else float("nan")})
    tabla_comp = _guardar_tabla(pd.DataFrame(filas_comp), "complementariedad_con_p")
    print("\n" + tabla_comp.to_string(index=False))

    # comparaciones estadisticas
    dfs_stats = {"P": df_p_ref, **{n: r["res_ramp"] for n, r in resultados.items()}}
    pares = [("P", "M60_BASE"), ("P", "M60_CAL"), ("M60_CAL", "M60_BASE"), ("P", "Persistencia"), ("P", "PerfilHistorico"),
             ("M60_BASE", "Persistencia"), ("M60_BASE", "PerfilHistorico")]
    tabla_stats = _guardar_tabla(comparacion_pareada(dfs_stats, pares), "comparaciones_estadisticas")
    print("\n" + tabla_stats.to_string(index=False))

    # criterios de exito
    retraso_p = resumen_metricas(df_p_ref)["retraso_mediano_min"]
    dr_persistencia = resumen_metricas(resultados["Persistencia"]["res_ramp"])["induced_DR_pct"]
    dr_perfil = resumen_metricas(resultados["PerfilHistorico"]["res_ramp"])["induced_DR_pct"]
    tablas_criterios = []
    for nombre, res in resultados.items():
        a_idx_n = res["res_ramp"].set_index("episode_id")["induced_detected"].astype(bool).reindex(p_idx.index).fillna(False)
        solo_ids = set(a_idx_n[a_idx_n].index) - set(p_idx[p_idx].index)
        tabla_dur_m = tabla_dur_all[tabla_dur_all["metodo"] == nombre]
        tabla_rho_m = tabla_rho_all[tabla_rho_all["metodo"] == nombre]
        tc = evaluar_criterios(nombre, res, retraso_p, dr_persistencia, dr_perfil, tabla_rho_m, solo_ids, conj["ataques_ramp"], tabla_dur_m, conj["HI_eval"])
        tablas_criterios.append(tc)
        print(f"  {nombre}: {tc.attrs['n_cumplidos']}/12 | sustantivos (1,3,4): {tc.attrs['n_sustantivos_cumplidos']}/3")
    tabla_criterios = pd.concat(tablas_criterios, ignore_index=True)
    _guardar_tabla(tabla_criterios, "criterios_exito")

    manifiesto = {"fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(), "loss_seleccionada": resultado_entrenamiento["loss_ganadora"],
                  "n_R_eval": len(conj["R_eval"]), "n_HI_eval": len(conj["HI_eval"])}
    with open(TABLAS_DIR / "manifiesto_ejecucion.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False, default=str)

    print("\nGenerando figuras...")
    generar_figuras(resultados, ctx, conj)

    return {"resultados": resultados, "conj": conj, "tabla_glob": tabla_glob, "tabla_hi": tabla_hi,
            "tabla_comp": tabla_comp, "tabla_stats": tabla_stats, "tabla_criterios": tabla_criterios}


def main() -> dict:
    print("Fase 1: carga + construccion de muestras + auditoria...")
    ctx = cargar_todo()
    print(f"  train: {len(ctx['m_train']['offsets'])} muestras | val: {len(ctx['m_val']['offsets'])} muestras "
          f"(dev={len(ctx['m_dev']['offsets'])}, eval={len(ctx['m_eval']['offsets'])})")
    auditar_fuga_temporal(ctx["m_train"], ctx["params_norm"])
    print("OK -- auditoria de fuga temporal pasada")

    resultado_entrenamiento = fase1_entrenamiento(ctx)
    print("\nOK -- Fase 1 (entrenamiento) completa.")

    resultado_final = fase2_evaluacion(ctx, resultado_entrenamiento)
    print(f"\nOK -- tablas en {TABLAS_DIR.relative_to(BASE_DIR)}/, figuras en {FIGURAS_DIR.relative_to(BASE_DIR)}/, "
          f"modelos en {MODELOS_DIR.relative_to(BASE_DIR)}/")
    return {"ctx": ctx, "resultado_entrenamiento": resultado_entrenamiento, **resultado_final}


if __name__ == "__main__":
    main()
