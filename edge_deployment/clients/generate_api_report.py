"""Genera las 14 figuras desde tablas ya persistidas -- nunca vuelve a llamar
al motor ni a la API.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
FIGURES_DIR = EDGE_DIR / "results" / "figures"

AZUL, NARANJA, ROJO, VERDE, GRIS, MORADO = "#2a78d6", "#eb6834", "#c0392b", "#2f9e44", "#8a8a86", "#7d3c98"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS,
                      "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "font.size": 10})


def _save(fig, nombre):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"api_{nombre}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_01_latency_direct_vs_testclient_vs_uvicorn():
    eq_path = TABLES_DIR / "api_engine_equivalence.csv"
    uv_path = TABLES_DIR / "api_uvicorn_latency.csv"
    if not eq_path.exists():
        return
    eq = pd.read_csv(eq_path)
    series = {"motor directo": eq["direct_round_trip_ms"].dropna(), "API (TestClient)": eq["api_round_trip_ms"].dropna()}
    if uv_path.exists():
        uv = pd.read_csv(uv_path)
        series["API (Uvicorn real)"] = uv["round_trip_ms"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(series.values(), tick_labels=series.keys(), showfliers=False)
    ax.set_ylabel("ms"); ax.set_title("Latencia: motor directo vs. TestClient vs. Uvicorn real")
    fig.tight_layout()
    _save(fig, "01_latencia_directo_vs_testclient_vs_uvicorn")


def fig_02_overhead_by_request_type():
    path = TABLES_DIR / "api_testclient_latency.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["pydantic_routing_overhead_ms"], bins=40, color=MORADO, alpha=0.8)
    ax.set_xlabel("overhead API - motor (ms)"); ax.set_ylabel("n lecturas")
    ax.set_title("Overhead de Pydantic/rutas/serializacion por peticion")
    _save(fig, "02_overhead_por_peticion")


def fig_03_p50_p95_p99_by_endpoint():
    path = TABLES_DIR / "api_testclient_latency.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    percentiles = df["testclient_total_ms"].quantile([0.5, 0.95, 0.99])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["p50", "p95", "p99"], percentiles.values, color=[AZUL, NARANJA, ROJO])
    ax.set_ylabel("ms"); ax.set_title("Latencia /readings (TestClient): p50/p95/p99")
    _save(fig, "03_p50_p95_p99")


def fig_04_latency_no_inference_vs_h_vs_ph():
    path = TABLES_DIR / "api_testclient_latency.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    grupos = {
        "sin modelo": df.loc[~df["p_evaluated"].astype(bool) & ~df["h_evaluated"].astype(bool), "testclient_total_ms"],
        "H": df.loc[df["h_evaluated"].astype(bool) & ~df["p_evaluated"].astype(bool), "testclient_total_ms"],
        "P y H": df.loc[df["p_evaluated"].astype(bool) & df["h_evaluated"].astype(bool), "testclient_total_ms"],
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([g.dropna() for g in grupos.values()], tick_labels=grupos.keys(), showfliers=False)
    ax.set_ylabel("ms"); ax.set_title("Latencia segun rama evaluada (TestClient)")
    _save(fig, "04_latencia_por_rama")


def fig_05_timeline_or_direct_vs_api():
    path = TABLES_DIR / "api_engine_equivalence.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, parse_dates=["timestamp"])
    ini = df["timestamp"].min()
    sub = df[(df["timestamp"] >= ini) & (df["timestamp"] < ini + pd.Timedelta(days=5))]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(sub["timestamp"], 0, sub["direct_alert_or"].astype(bool).astype(int), step="post", alpha=0.35, color=AZUL, label="directo")
    ax.plot(sub["timestamp"], sub["api_alert_or"].astype(bool).astype(int) * 1.05, color=ROJO, lw=1.2, drawstyle="steps-post", label="API")
    ax.set_yticks([0, 1]); ax.set_title("Timeline OR: motor directo vs. API (primeros 5 dias)")
    ax.legend(fontsize=8)
    _save(fig, "05_timeline_or")


def _score_diff_fig(df, col_direct, col_api, titulo, nombre):
    diff = (df[col_api].astype(float) - df[col_direct].astype(float)).dropna()
    fig, ax = plt.subplots(figsize=(10, 3))
    if (diff.abs() < 1e-12).all():
        ax.axhline(0, color=VERDE, lw=1.5)
        ax.set_ylim(-1e-6, 1e-6)
        ax.text(0.5, 0.5, "diferencia = 0 en todos los puntos (bit-identico)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=VERDE)
    else:
        ax.plot(diff.to_numpy(), color=ROJO, lw=0.8)
    ax.set_title(titulo)
    _save(fig, nombre)


def fig_06_07_score_diffs():
    path = TABLES_DIR / "api_engine_equivalence.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    _score_diff_fig(df, "direct_score_h", "api_score_h", "Diferencia score H: API - directo", "06_score_h_diff")
    _score_diff_fig(df, "direct_score_p", "api_score_p", "Diferencia score P: API - directo", "07_score_p_diff")


def fig_08_memory_before_after_models():
    path = TABLES_DIR / "api_memory_usage.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[df["stage"].isin(["before_models_loaded", "after_models_loaded", "after_bootstrap"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(sub["stage"], sub["rss_mb"], color=[GRIS, AZUL, NARANJA])
    ax.set_ylabel("RSS (MB)"); ax.set_title("Memoria del proceso: antes/despues de cargar modelos")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "08_memoria_antes_despues_modelos")


def fig_09_memory_over_days():
    path = TABLES_DIR / "api_memory_usage.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[df["stage"].str.startswith("after_") & df["stage"].str.endswith("days")]
    if not len(sub):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["stage"], sub["rss_mb"], marker="o", color=AZUL)
    ax.set_ylabel("RSS (MB)"); ax.set_title("Memoria del proceso durante el streaming simulado")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "09_memoria_durante_streaming")


def fig_10_buffer_sizes():
    path = TABLES_DIR / "api_memory_usage.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[df["stage"].str.startswith("after_") & df["stage"].str.endswith("days")]
    if not len(sub) or "buffer_1min_size" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["stage"], sub["buffer_1min_size"], marker="o", color=AZUL, label="buffer 1 min (P)")
    ax.plot(sub["stage"], sub["history_15min_size"], marker="o", color=NARANJA, label="historial 15 min (H)")
    ax.set_ylabel("numero de elementos"); ax.set_title("Tamano de los buffers durante el streaming")
    ax.tick_params(axis="x", rotation=20); ax.legend(fontsize=8)
    _save(fig, "10_tamano_buffers")


def fig_11_throughput():
    path = TABLES_DIR / "api_throughput.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["throughput"], df["throughput_readings_per_second"], color=VERDE)
    ax.set_ylabel("lecturas/segundo"); ax.set_title("Throughput (Uvicorn real, acelerado)")
    _save(fig, "11_throughput")


def fig_12_http_status_distribution():
    path = TABLES_DIR / "api_error_cases.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    conteo = df["status_code"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(conteo.index.astype(str), conteo.values, color=MORADO)
    ax.set_xlabel("codigo HTTP"); ax.set_ylabel("n casos"); ax.set_title("Distribucion de codigos HTTP (casos de error)")
    _save(fig, "12_distribucion_codigos_http")


def fig_13_error_cases_result():
    path = TABLES_DIR / "api_error_cases.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(11, 6))
    colores = df["passed"].map({True: VERDE, False: ROJO})
    ax.barh(df["case"], [1] * len(df), color=colores)
    ax.set_xticks([]); ax.set_title("Resultado de los 24 casos de error (verde=paso)")
    fig.tight_layout()
    _save(fig, "13_casos_de_error")


def fig_14_concurrency_results():
    path = TABLES_DIR / "api_concurrency_cases.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 4))
    colores = df["passed"].map({True: VERDE, False: ROJO})
    ax.barh(df["case"], [1] * len(df), color=colores)
    ax.set_xticks([]); ax.set_title("Resultado de las pruebas de concurrencia (verde=paso)")
    fig.tight_layout()
    _save(fig, "14_concurrencia")


def generar_todas_las_figuras():
    fig_01_latency_direct_vs_testclient_vs_uvicorn()
    fig_02_overhead_by_request_type()
    fig_03_p50_p95_p99_by_endpoint()
    fig_04_latency_no_inference_vs_h_vs_ph()
    fig_05_timeline_or_direct_vs_api()
    fig_06_07_score_diffs()
    fig_08_memory_before_after_models()
    fig_09_memory_over_days()
    fig_10_buffer_sizes()
    fig_11_throughput()
    fig_12_http_status_distribution()
    fig_13_error_cases_result()
    fig_14_concurrency_results()


if __name__ == "__main__":
    generar_todas_las_figuras()
    print(f"Figuras guardadas en {FIGURES_DIR}")
