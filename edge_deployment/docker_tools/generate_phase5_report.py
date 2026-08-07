"""Fase 5, seccion 22: figuras del informe final. Mismo patron que
`docker_tools/generate_docker_report.py` (Fase 3): cada funcion se salta con un aviso (no
lanza excepcion) si el CSV/JSON del que depende todavia no existe.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
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


def _box(ax, xy, w, h, text, color="#3070b3", fontsize=9, textcolor="white"):
    ax.add_patch(mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", linewidth=1,
                                          edgecolor="#333333", facecolor=color))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=textcolor, wrap=True)


def _arrow(ax, xy1, xy2):
    ax.annotate("", xy=xy2, xytext=xy1, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))


def fig_01_layered_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _box(ax, (0.5, 8), 9, 1.2, "Cliente / contador (firma HMAC-SHA256, session_id, sequence_number)", "#555555")
    _box(ax, (0.5, 6.2), 4.2, 1.2, "POST /readings\n(Fase 1-4, sin cambios)", "#888888")
    _box(ax, (5.3, 6.2), 4.2, 1.2, "POST /secure-readings\n(Fase 5, opcional)", "#2a7f3f")
    _box(ax, (5.3, 4.4), 4.2, 1.2, "AntiReplayGuard\nHMAC + secuencia + frescura", "#c0392b")
    _box(ax, (0.5, 2.6), 9, 1.2, "DetectorEngine.ingest()  (P + H + OR -- sin modificar, Fase 1)", "#3070b3")
    _box(ax, (0.5, 0.5), 9, 1.2, "P (TCN-AE) / H (HistGB) / OR  -- deteccion de contenido anomalo", "#1a1a1a")
    _arrow(ax, (2.6, 8), (2.6, 7.4))
    _arrow(ax, (7.4, 8), (7.4, 7.4))
    _arrow(ax, (7.4, 6.2), (7.4, 5.6))
    _arrow(ax, (2.6, 6.2), (2.6, 3.8))
    _arrow(ax, (7.4, 4.4), (5, 3.8))
    _arrow(ax, (5, 2.6), (5, 1.7))
    ax.set_title("Arquitectura por capas: deteccion de contenido (P/H/OR) + capa anti-replay opcional (Fase 5)")
    _save(fig, "phase5_fig_01_layered_architecture")


def fig_02_phase4_vs_phase5_extension() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    _box(ax, (0.5, 5.5), 9, 1.8, "fdia-edge:phase4-autonomous\n(congelada, inmutable, GATE 0)", "#888888", fontsize=11)
    _box(ax, (0.5, 2.8), 4, 1.8, "edge_deployment/\napi/ + core/\n(SIN modificar)", "#3070b3")
    _box(ax, (5.2, 2.8), 4.3, 1.8, "edge_deployment/security/\n+ secure_routes.py\n(NUEVO, Fase 5)", "#2a7f3f")
    _box(ax, (0.5, 0.3), 9, 1.8, "fdia-edge:phase5-antireplay\n(extension aditiva -- eliminar security/ vuelve a Fase 4)", "#1a1a1a", fontsize=11)
    _arrow(ax, (2.5, 5.5), (2.5, 4.6))
    _arrow(ax, (5, 3.7), (5.2, 3.7))
    _arrow(ax, (2.5, 2.8), (2.5, 2.1))
    _arrow(ax, (7.3, 2.8), (7.3, 2.1))
    ax.set_title("Fase 4 (baseline congelada) + extension aditiva = Fase 5")
    _save(fig, "phase5_fig_02_phase4_baseline_vs_phase5_extension")


def fig_03_legitimate_vs_replay_flow() -> None:
    d = json.loads((REPORTS_DIR / "phase5_gate5_summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(TABLES_DIR / "anti_replay_cases.csv")
    b = df[df.case == "B_exact_replay"].iloc[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    categorias = ["Primer mensaje\n(legitimo)", "Replay exacto\n(mismo mensaje)"]
    valores = [1 if b.get("first_status") == 200 else 0, 1 if b.get("replay_status") == 200 else 0]
    colores = ["#2a7f3f" if v else "#c0392b" for v in valores]
    ax.bar(categorias, [1, 1], color=colores)
    for i, (cat, ok) in enumerate(zip(categorias, [b.get("first_status") == 200, b.get("replay_status") != 200])):
        ax.text(i, 0.5, "ACEPTADO" if (i == 0) else "RECHAZADO", ha="center", va="center", color="white", fontweight="bold")
    ax.set_yticks([])
    ax.set_title("Flujo legitimo aceptado frente a replay exacto rechazado (caso B)")
    _save(fig, "phase5_fig_03_legitimate_vs_replay")


def fig_04_rejection_rate_by_attack() -> None:
    df = pd.read_csv(TABLES_DIR / "anti_replay_cases.csv")
    df["ok"] = df["criteria_met"].astype(bool)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#2a7f3f" if v else "#c0392b" for v in df["ok"]]
    ax.bar(df["case"], df["ok"].astype(int) * 100, color=colors)
    ax.set_ylabel("% criterio cumplido")
    ax.set_ylim(0, 110)
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Resultado por caso de ataque (verde = criterio de la seccion 16 cumplido)")
    _save(fig, "phase5_fig_04_rejection_rate_by_attack")


def fig_05_latency_plain_vs_secure() -> None:
    df = pd.read_csv(TABLES_DIR / "anti_replay_latency.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [df[df.target == t]["round_trip_ms"].dropna().values for t in df["target"].unique()]
    ax.boxplot(data, tick_labels=df["target"].unique(), showfliers=False)
    ax.set_ylabel("Latencia round-trip (ms)")
    ax.set_title("Latencia: /readings (Fase 4/5) vs /secure-readings (Fase 5)")
    plt.xticks(rotation=15, ha="right")
    _save(fig, "phase5_fig_05_latency_plain_vs_secure")


def fig_06_overhead_breakdown() -> None:
    d = json.loads((REPORTS_DIR / "anti_replay_microbenchmark.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    steps = ["canonicalization", "hmac_compute", "hmac_verify", "sequence_and_freshness_check"]
    labels = ["Canonicalizacion", "HMAC (firmar)", "HMAC (verificar)", "Secuencia+frescura"]
    valores = [d[s]["median_ms"] for s in steps]
    ax.bar(labels, valores, color="#c0392b")
    for i, v in enumerate(valores):
        ax.text(i, v, f"{v:.4f}ms", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mediana (ms)")
    ax.set_title("Desglose del overhead de seguridad por paso (microbenchmark en proceso)")
    _save(fig, "phase5_fig_06_overhead_breakdown")


def fig_07_soak_memory_evolution() -> None:
    df = pd.read_csv(TABLES_DIR / "anti_replay_soak_checkpoints.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["day"], df["mem_mb_recent_mean"], "o-", color="#8e44ad")
    ax.set_xlabel("Dia simulado")
    ax.set_ylabel("RAM media reciente (MB)")
    ax.set_title("Evolucion de memoria durante la prueba prolongada (trafico legitimo firmado)")
    _save(fig, "phase5_fig_07_soak_memory_evolution")


def fig_08_detector_state_invariance() -> None:
    d = json.loads((REPORTS_DIR / "phase5_gate5_summary.json").read_text(encoding="utf-8"))
    invariant = d["invariant_results"]
    fig, ax = plt.subplots(figsize=(6, 4))
    casos = [r["case"] for r in invariant]
    ok = [1 if r["state_unchanged"] else 0 for r in invariant]
    colors = ["#2a7f3f" if v else "#c0392b" for v in ok]
    ax.bar(casos, [1] * len(casos), color=colors)
    for i, v in enumerate(ok):
        ax.text(i, 0.5, "SIN CAMBIOS" if v else "CAMBIO DETECTADO", ha="center", va="center", color="white", fontsize=8)
    ax.set_yticks([])
    plt.xticks(rotation=20, ha="right")
    ax.set_title("Invariancia del estado del motor tras mensajes rechazados (seccion 17)")
    _save(fig, "phase5_fig_08_detector_state_invariance")


def main() -> None:
    print("[phase5-figures] generando figuras disponibles...")
    for fn in [fig_01_layered_architecture, fig_02_phase4_vs_phase5_extension, fig_03_legitimate_vs_replay_flow,
               fig_04_rejection_rate_by_attack, fig_05_latency_plain_vs_secure, fig_06_overhead_breakdown,
               fig_07_soak_memory_evolution, fig_08_detector_state_invariance]:
        _try(fn)


if __name__ == "__main__":
    main()
