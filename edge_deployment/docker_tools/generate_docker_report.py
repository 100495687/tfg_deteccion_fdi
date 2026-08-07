"""Fase 3: figuras del informe final. Cada funcion se salta con un aviso (no
lanza excepcion) si el CSV/JSON del que depende todavia no existe -- permite generar figuras
parciales mientras otros experimentos siguen en marcha.
"""
from __future__ import annotations

import json
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


def _save(fig, nombre: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{nombre}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}.png")


def _try(fn):
    try:
        fn()
    except FileNotFoundError as e:
        print(f"  [omitida] {fn.__name__}: {e}")
    except Exception as e:
        print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")


def fig_01_image_size_baseline_vs_optimized() -> None:
    df = pd.read_csv(TABLES_DIR / "docker_image_metrics.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    mb = df["size_bytes_true_docker_save"] / 1e6
    ax.bar(df["image"], mb, color=["#888888", "#2a7f3f"])
    for i, v in enumerate(mb):
        ax.text(i, v + 10, f"{v:.0f} MB", ha="center")
    ax.set_ylabel("Tamano real de imagen (MB, docker save)")
    ax.set_title("Tamano de imagen: baseline vs optimized")
    _save(fig, "docker_fig_01_image_size")


def fig_02_time_to_ready() -> None:
    df = pd.read_csv(TABLES_DIR / "docker_startup_metrics.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df[df.image == img]["time_to_ready_s"].dropna().values for img in df["image"].unique()]
    ax.boxplot(data, tick_labels=df["image"].unique())
    ax.set_ylabel("Tiempo hasta /ready (s)")
    ax.set_title(f"Tiempo de arranque hasta ready ({len(data[0])} arranques/imagen)")
    _save(fig, "docker_fig_02_time_to_ready")


def fig_03_latency_by_environment() -> None:
    rows = []
    smoke = json.loads((REPORTS_DIR / "container_equivalence_summary_smoke.json").read_text(encoding="utf-8"))
    lat_path = TABLES_DIR / "docker_latency.csv"
    if lat_path.exists():
        dfl = pd.read_csv(lat_path)
        rows.append(("C_docker_optimized\n(latency_benchmark)", dfl["round_trip_ms"].median()))
    mm_path = TABLES_DIR / "docker_memory_limit_matrix.csv"
    if mm_path.exists():
        dfm = pd.read_csv(mm_path)
        best = dfm.sort_values("memory_limit_mb", ascending=False).iloc[0]
        if pd.notna(best.get("latency_p50_ms")):
            rows.append((f"optimized {int(best['memory_limit_mb'])}MB\n(sin restriccion practica)", best["latency_p50_ms"]))
    if not rows:
        raise FileNotFoundError("no hay datos de latencia todavia")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, vals = zip(*rows)
    ax.bar(labels, vals, color="#3070b3")
    ax.set_ylabel("Latencia mediana round-trip (ms)")
    ax.set_title("Latencia por entorno")
    plt.xticks(rotation=15, ha="right")
    _save(fig, "docker_fig_03_latency_by_environment")


def fig_04_ram_by_memory_limit() -> None:
    df = pd.read_csv(TABLES_DIR / "docker_memory_limit_matrix.csv").sort_values("memory_limit_mb")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["memory_limit_mb"], df["memory_limit_mb"], "--", color="gray", label="limite asignado")
    ax.plot(df["memory_limit_mb"], df["peak_memory_mb_docker_stats"], "o-", color="#c0392b", label="pico RAM observado")
    ax.set_xlabel("Limite de memoria del contenedor (MB)")
    ax.set_ylabel("MB")
    ax.set_title("RAM observada vs limite asignado (1 CPU)")
    ax.legend()
    _save(fig, "docker_fig_04_ram_by_memory_limit")


def fig_05_success_failure_by_memory() -> None:
    df = pd.read_csv(TABLES_DIR / "docker_memory_limit_matrix.csv").sort_values("memory_limit_mb")
    fig, ax = plt.subplots(figsize=(8, 4))
    stages = ["startup_success", "ready_success", "bootstrap_success", "streaming_success", "equivalence_passed"]
    # NaN (etapa nunca alcanzada, p.ej. equivalence_passed cuando el contenedor ni siquiera
    # llego a ready) debe leerse como fallo/no-aplicable, no como exito -- bool(NaN) es True
    # en pandas/numpy, asi que hay que rellenar antes de castear (bug real, corregido tras
    # inspeccionar visualmente esta figura con los 8 limites de memoria reales).
    mat = df[stages].fillna(False).astype(bool).astype(int).to_numpy().T
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{m}MB" for m in df["memory_limit_mb"]], rotation=45, ha="right")
    ax.set_title("Exito/fallo por etapa segun limite de memoria (verde=OK, rojo=fallo)")
    _save(fig, "docker_fig_05_success_failure_by_memory")


def fig_06_latency_vs_cpu() -> None:
    df = pd.read_csv(TABLES_DIR / "docker_cpu_limit_matrix.csv").sort_values("cpu_limit")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["cpu_limit"], df["latency_p50_ms"], "o-", label="p50")
    ax.plot(df["cpu_limit"], df["latency_p95_ms"], "s-", label="p95")
    ax.plot(df["cpu_limit"], df["latency_p99_ms"], "^-", label="p99")
    ax.set_xlabel("CPUs asignadas")
    ax.set_ylabel("Latencia (ms)")
    ax.set_title("Latencia vs limite de CPU")
    ax.legend()
    _save(fig, "docker_fig_06_latency_vs_cpu")


def fig_07_throughput_vs_cpu() -> None:
    df = pd.read_csv(TABLES_DIR / "docker_cpu_limit_matrix.csv").sort_values("cpu_limit")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["cpu_limit"].astype(str), df["throughput_rps"], color="#27ae60")
    ax.set_xlabel("CPUs asignadas")
    ax.set_ylabel("Throughput (lecturas/s)")
    ax.set_title("Throughput vs limite de CPU")
    _save(fig, "docker_fig_07_throughput_vs_cpu")


def fig_08_memory_evolution() -> None:
    p = TABLES_DIR / "docker_soak_checkpoints.csv"
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["day"], df["mem_mb_recent_mean"], "o-", color="#8e44ad")
    ax.set_xlabel("Dia simulado")
    ax.set_ylabel("RAM media reciente (MB)")
    ax.set_title("Evolucion de memoria durante la prueba prolongada")
    _save(fig, "docker_fig_08_memory_evolution")


def fig_09_equivalence_local_vs_docker() -> None:
    d = json.loads((REPORTS_DIR / "container_equivalence_summary_smoke.json").read_text(encoding="utf-8"))
    envs = list(d["per_environment"].keys())
    fields = list(next(iter(d["per_environment"].values()))["match_pct"].keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fields))
    width = 0.8 / len(envs)
    for i, env in enumerate(envs):
        vals = [d["per_environment"][env]["match_pct"][f] for f in fields]
        ax.bar(x + i * width, vals, width, label=env)
    ax.set_xticks(x + width * (len(envs) - 1) / 2)
    ax.set_xticklabels(fields, rotation=45, ha="right")
    ax.set_ylabel("% de coincidencia")
    ax.set_ylim(0, 105)
    ax.set_title("Equivalencia local vs Docker, por campo (smoke test 3 dias)")
    ax.legend()
    _save(fig, "docker_fig_09_equivalence_local_vs_docker")


def main() -> None:
    print("[docker-figures] generando figuras disponibles...")
    for fn in [fig_01_image_size_baseline_vs_optimized, fig_02_time_to_ready, fig_03_latency_by_environment,
               fig_04_ram_by_memory_limit, fig_05_success_failure_by_memory, fig_06_latency_vs_cpu,
               fig_07_throughput_vs_cpu, fig_08_memory_evolution, fig_09_equivalence_local_vs_docker]:
        _try(fn)


if __name__ == "__main__":
    main()
