"""Resumenes, tablas de desglose, figuras e informe final del piloto de replay.

Las figuras se generan siempre a partir de tablas ya persistidas (replay_episode_results.csv),
nunca recalculando scores ni volviendo a puntuar episodios.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.replay_pilot.src.build_replay_manifest import EXP_DIR, TABLES_DIR, REPORTS_DIR

FIGURES_DIR = EXP_DIR / "figures"
METODOS = ["P", "H", "OR"]


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z ** 2 / n
    centro = phat + z ** 2 / (2 * n)
    margen = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return (100 * (centro - margen) / denom, 100 * (centro + margen) / denom)


def _dr(sub: pd.DataFrame, col: str) -> dict:
    n = len(sub)
    k = int(sub[col].sum())
    lo, hi = _wilson_ci(k, n)
    return {"n": n, "n_detected": k, "dr_pct": 100 * k / n if n else np.nan, "dr_ci95_low": lo, "dr_ci95_high": hi}


# ==========================================================================================
# Resumenes principales
# ==========================================================================================

def construir_resumenes(base: pd.DataFrame) -> dict:
    out = {}

    filas_global = []
    for metodo in METODOS:
        row = {"metodo": metodo}
        row.update({f"standard_{k}": v for k, v in _dr(base, f"detected_standard_{metodo}").items()})
        row.update({f"induced_{k}": v for k, v in _dr(base, f"detected_induced_{metodo}").items()})
        row.update({f"interior_{k}": v for k, v in _dr(base, f"detected_interior_{metodo}").items()})
        row["pct_boundary_only"] = 100 * base[f"boundary_only_detection_{metodo}"].mean()
        row["pct_post_only"] = 100 * base[f"post_only_detection_{metodo}"].mean()
        con_ind = base[base[f"detected_induced_{metodo}"]]
        row["retraso_mediano_min"] = float(con_ind["delay_minutes"].median()) if len(con_ind) else np.nan
        row["retraso_p90_min"] = float(con_ind["delay_minutes"].quantile(0.90)) if len(con_ind) else np.nan
        filas_global.append(row)
    out["global"] = pd.DataFrame(filas_global)

    energia_oculta = base[base["hidden_energy_kwh"] > 0]
    filas_energ = []
    for metodo in METODOS:
        tot = energia_oculta["hidden_energy_kwh"].sum()
        det = energia_oculta.loc[energia_oculta[f"detected_induced_{metodo}"], "hidden_energy_kwh"].sum()
        filas_energ.append({"metodo": metodo, "n_underreporting_episodes": len(energia_oculta),
                             "hidden_energy_total_kwh": float(tot), "hidden_energy_detected_kwh": float(det),
                             "energy_DR_pct": 100 * det / tot if tot > 1e-9 else np.nan})
    out["energy_dr"] = pd.DataFrame(filas_energ)

    out["by_shift"] = pd.concat([
        base[base["shift_type"] == s].assign(shift_type=s).groupby("shift_type").apply(
            lambda g: pd.Series({f"dr_induced_OR_pct": 100 * g["detected_induced_OR"].mean(),
                                  "dr_interior_OR_pct": 100 * g["detected_interior_OR"].mean(),
                                  "retraso_mediano_min": g.loc[g["detected_induced_OR"], "delay_minutes"].median(),
                                  "n": len(g)}), include_groups=False)
        for s in base["shift_type"].unique()
    ]).reset_index()

    out["by_duration"] = base.groupby("duration_minutes").apply(
        lambda g: pd.Series({"dr_induced_OR_pct": 100 * g["detected_induced_OR"].mean(),
                              "dr_interior_OR_pct": 100 * g["detected_interior_OR"].mean(), "n": len(g)}),
        include_groups=False).reset_index()

    out["by_economic_category"] = base.groupby("economic_category").apply(
        lambda g: pd.Series({"dr_induced_OR_pct": 100 * g["detected_induced_OR"].mean(), "n": len(g)}),
        include_groups=False).reset_index()

    filas_zona = []
    for zona_col in ["detected_onset_zone", "detected_interior", "detected_end_zone", "detected_post_attack"]:
        for metodo in METODOS:
            col = f"{zona_col}_{metodo}"
            filas_zona.append({"zona": zona_col.replace("detected_", "").replace("_zone", ""), "metodo": metodo,
                                "pct": 100 * base[col].mean()})
    out["by_zone"] = pd.DataFrame(filas_zona)

    filas_rama = []
    for metodo in ["P", "H", "OR"]:
        filas_rama.append({"metodo": metodo, "n_detected_induced": int(base[f"detected_induced_{metodo}"].sum())})
    filas_rama.append({"metodo": "P_only", "n_detected_induced": int(base["P_only"].sum())})
    filas_rama.append({"metodo": "H_only", "n_detected_induced": int(base["H_only"].sum())})
    filas_rama.append({"metodo": "both", "n_detected_induced": int(base["both"].sum())})
    out["by_branch"] = pd.DataFrame(filas_rama)

    filas_par = []
    for pid, g in base.groupby("paired_destination_id"):
        daily = g[g["shift_type"] == "DAILY"]
        weekly = g[g["shift_type"] == "WEEKLY"]
        if len(daily) and len(weekly):
            d, w = daily.iloc[0], weekly.iloc[0]
            filas_par.append({
                "paired_destination_id": pid, "daily_detected_induced_OR": bool(d["detected_induced_OR"]),
                "weekly_detected_induced_OR": bool(w["detected_induced_OR"]),
                "diff_max_delta_score_H": float(d["max_delta_score_H"] - w["max_delta_score_H"]),
                "diff_delay_minutes": (d["delay_minutes"] - w["delay_minutes"]) if pd.notna(d["delay_minutes"]) and pd.notna(w["delay_minutes"]) else np.nan,
                "diff_hidden_energy_kwh": float(d["hidden_energy_kwh"] - w["hidden_energy_kwh"]),
            })
    out["paired_comparison"] = pd.DataFrame(filas_par)

    out["complementarity"] = pd.DataFrame([{
        "n_P_only": int(base["P_only"].sum()), "n_H_only": int(base["H_only"].sum()), "n_both": int(base["both"].sum()),
        "n_none": int((~base["detected_induced_OR"]).sum()),
        "retencion_P_por_OR_pct": 100 * base.loc[base["detected_induced_P"], "detected_induced_OR"].mean() if base["detected_induced_P"].any() else np.nan,
        "retencion_H_por_OR_pct": 100 * base.loc[base["detected_induced_H"], "detected_induced_OR"].mean() if base["detected_induced_H"].any() else np.nan,
    }])

    out["boundary_detection_analysis"] = base[[
        "episode_id", "paired_destination_id", "shift_type", "duration_minutes",
        "onset_jump_kw", "end_jump_kw", "onset_jump_relative", "end_jump_relative",
        "detected_onset_zone_OR", "detected_interior_OR", "detected_end_zone_OR", "detected_post_attack_OR",
        "boundary_only_detection_OR", "post_only_detection_OR", "max_delta_score_H",
    ]].copy()

    out["source_destination_similarity"] = base[[
        "episode_id", "paired_destination_id", "shift_type", "duration_minutes",
        "source_destination_mae", "source_destination_rmse", "source_destination_correlation",
        "diff_mean", "diff_median", "diff_p95", "hidden_energy_kwh", "detected_induced_OR",
    ]].copy()

    out["p_h_complementarity"] = out["complementarity"]

    return out


def guardar_resumenes(resumen: dict) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    resumen["global"].to_csv(TABLES_DIR / "replay_summary.csv", index=False)
    resumen["by_shift"].to_csv(TABLES_DIR / "results_by_shift.csv", index=False)
    resumen["by_duration"].to_csv(TABLES_DIR / "results_by_duration.csv", index=False)
    resumen["by_economic_category"].to_csv(TABLES_DIR / "results_by_energy_category.csv", index=False)
    resumen["by_zone"].to_csv(TABLES_DIR / "replay_zone_results.csv", index=False)
    resumen["boundary_detection_analysis"].to_csv(TABLES_DIR / "boundary_detection_analysis.csv", index=False)
    resumen["source_destination_similarity"].to_csv(TABLES_DIR / "source_destination_similarity.csv", index=False)
    resumen["p_h_complementarity"].to_csv(TABLES_DIR / "p_h_complementarity.csv", index=False)
    resumen["by_branch"].to_csv(TABLES_DIR / "results_by_branch.csv", index=False)
    resumen["paired_comparison"].to_csv(TABLES_DIR / "paired_comparison_daily_vs_weekly.csv", index=False)
    resumen["energy_dr"].to_csv(TABLES_DIR / "energy_dr.csv", index=False)


# ==========================================================================================
# Figuras globales -- generadas desde replay_episode_results.csv, no recalculan
# ==========================================================================================

def generar_figuras_globales(base: pd.DataFrame, resumen: dict) -> None:
    (FIGURES_DIR / "summary").mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "summary"

    fig, ax = plt.subplots(figsize=(8, 5))
    g = resumen["global"].set_index("metodo")
    x = np.arange(len(g))
    ax.bar(x - 0.25, g["standard_dr_pct"], width=0.25, label="estandar")
    ax.bar(x, g["induced_dr_pct"], width=0.25, label="inducida")
    ax.bar(x + 0.25, g["interior_dr_pct"], width=0.25, label="interior")
    ax.set_xticks(x); ax.set_xticklabels(g.index); ax.set_ylabel("DR (%)")
    ax.set_title("DR estandar, inducida e interior por rama"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "01_dr_standard_induced_interior.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    piv = base.groupby("shift_type")["detected_induced_OR"].mean() * 100
    ax.bar(piv.index, piv.values, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("DR inducido OR (%)"); ax.set_title("Replay diario frente a semanal")
    fig.tight_layout(); fig.savefig(out / "02_daily_vs_weekly.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    piv = base.groupby("duration_minutes")["detected_induced_OR"].mean() * 100
    ax.plot(piv.index.astype(str), piv.values, marker="o")
    ax.set_ylabel("DR inducido OR (%)"); ax.set_title("DR por duracion")
    fig.tight_layout(); fig.savefig(out / "03_dr_by_duration.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    piv = base.groupby("economic_category")["detected_induced_OR"].mean() * 100
    ax.bar(piv.index, piv.values)
    ax.set_ylabel("DR inducido OR (%)"); ax.set_title("DR por categoria economica")
    fig.tight_layout(); fig.savefig(out / "04_dr_by_energy_category.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    zona_a_columna = {"onset": "detected_onset_zone_OR", "interior": "detected_interior_OR",
                       "end": "detected_end_zone_OR", "post_attack": "detected_post_attack_OR"}
    zonas = list(zona_a_columna.keys())
    vals = [100 * base[zona_a_columna[z]].mean() for z in zonas]
    ax.bar(zonas, vals)
    ax.set_ylabel("% episodios con deteccion inducida en la zona"); ax.set_title("Deteccion por zona (OR)")
    fig.tight_layout(); fig.savefig(out / "05_detection_by_zone.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    pct_boundary = 100 * base["boundary_only_detection_OR"].mean()
    ax.bar(["boundary_only"], [pct_boundary])
    ax.set_ylabel("%"); ax.set_title("Porcentaje boundary_only (OR)")
    fig.tight_layout(); fig.savefig(out / "06_pct_boundary_only.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sub = base[base["detected_induced_OR"]]
    if len(sub):
        sub.boxplot(column="delay_minutes", by="shift_type", ax=ax)
    ax.set_title("Retraso por desplazamiento"); plt.suptitle("")
    fig.tight_layout(); fig.savefig(out / "07_delay_by_shift.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(base["hidden_energy_kwh"], base["max_delta_score_H"], c=base["detected_induced_OR"].map({True: "tab:green", False: "tab:red"}))
    ax.set_xlabel("energia oculta (kWh)"); ax.set_ylabel("max delta score H")
    ax.set_title("Energia ocultada frente a detectabilidad")
    fig.tight_layout(); fig.savefig(out / "08_energy_vs_detection.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(base["source_destination_correlation"], base["max_delta_score_H"], c=base["detected_induced_OR"].map({True: "tab:green", False: "tab:red"}))
    ax.set_xlabel("correlacion origen-destino"); ax.set_ylabel("max delta score H")
    ax.set_title("Similitud origen-destino frente a score")
    fig.tight_layout(); fig.savefig(out / "09_similarity_vs_score.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(base["onset_jump_kw"], base["max_delta_score_H"])
    axes[0].set_xlabel("salto inicial (kW)"); axes[0].set_ylabel("max delta score H"); axes[0].set_title("Salto inicial")
    axes[1].scatter(base["end_jump_kw"], base["max_delta_score_H"])
    axes[1].set_xlabel("salto final (kW)"); axes[1].set_title("Salto final")
    fig.tight_layout(); fig.savefig(out / "10_11_jumps_vs_detection.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    piv = pd.DataFrame({m: 100 * base[f"detected_induced_{m}"].astype(float) for m in METODOS}).mean()
    ax.bar(piv.index, piv.values)
    ax.set_ylabel("DR inducido (%)"); ax.set_title("P frente a H frente a OR")
    fig.tight_layout(); fig.savefig(out / "12_p_vs_h_vs_or.png", dpi=120); plt.close(fig)

    comp = resumen["complementarity"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["P_only", "H_only", "both", "none"], [comp["n_P_only"], comp["n_H_only"], comp["n_both"], comp["n_none"]])
    ax.set_ylabel("n episodios"); ax.set_title("Complementariedad P/H")
    fig.tight_layout(); fig.savefig(out / "13_complementarity.png", dpi=120); plt.close(fig)

    if len(resumen["paired_comparison"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        pc = resumen["paired_comparison"]
        ax.scatter(pc["daily_detected_induced_OR"].astype(int) + np.random.default_rng(0).uniform(-0.05, 0.05, len(pc)),
                   pc["weekly_detected_induced_OR"].astype(int) + np.random.default_rng(1).uniform(-0.05, 0.05, len(pc)))
        ax.set_xlabel("daily detectado"); ax.set_ylabel("weekly detectado"); ax.set_title("Comparacion emparejada diaria-semanal")
        fig.tight_layout(); fig.savefig(out / "14_paired_daily_weekly.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    piv = base.pivot_table(index="duration_minutes", columns="shift_type", values="detected_induced_OR", aggfunc="mean") * 100
    im = ax.imshow(piv.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    fig.colorbar(im, ax=ax, label="DR inducido OR (%)")
    ax.set_title("Matriz duracion x desplazamiento")
    fig.tight_layout(); fig.savefig(out / "15_duration_shift_matrix.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(base["max_delta_score_H"].dropna(), bins=20)
    ax.set_xlabel("max delta score H"); ax.set_title("Distribucion del maximo delta de score H")
    fig.tight_layout(); fig.savefig(out / "16_max_delta_score_h.png", dpi=120); plt.close(fig)

    umbral_h = 0.883628
    margen = umbral_h - base["max_score_H_attack"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(min(base["max_score_H_clean"].min(), base["max_score_H_attack"].min()),
                        max(base["max_score_H_clean"].max(), base["max_score_H_attack"].max()), 25)
    axes[0].hist(base["max_score_H_clean"], bins=bins, alpha=0.55, label="max score H limpio", color="tab:blue")
    axes[0].hist(base["max_score_H_attack"], bins=bins, alpha=0.55, label="max score H atacado", color="tab:orange")
    axes[0].axvline(umbral_h, color="red", linestyle="--", linewidth=1.5, label=f"umbral H = {umbral_h}")
    axes[0].set_xlabel("score H (max en la ventana del episodio)"); axes[0].set_ylabel("n episodios")
    axes[0].set_title("Distribucion de max score H: limpio vs. atacado"); axes[0].legend(fontsize=8)

    axes[1].hist(margen, bins=20, color="tab:green", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="umbral (margen=0)")
    axes[1].set_xlabel("margen = umbral H - max score H atacado"); axes[1].set_ylabel("n episodios")
    axes[1].set_title("Distancia al umbral (atacado)\n(margen<=0 -> hubiera cruzado el umbral)")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "17_score_h_threshold_margin.png", dpi=120); plt.close(fig)


def generar_figuras_episodios(base: pd.DataFrame, r: dict, ataques: dict, n_muestra: int = 6) -> None:
    (FIGURES_DIR / "episodes").mkdir(parents=True, exist_ok=True)
    from experiments.replay_pilot.src.evaluate_replay import construir_ventana_episodio
    muestra = base.sample(min(n_muestra, len(base)), random_state=0)
    for _, row in muestra.iterrows():
        ep_id = row["episode_id"]
        a = {**ataques[ep_id], "episode_id": ep_id}
        ventana = construir_ventana_episodio(a, r)
        ts = ventana["ts"]
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        h_clean, h_att = ventana["scores_clean"]["H"], ventana["scores_attack"]["H"]
        axes[0].plot(ts, h_clean, label="score H limpio", alpha=0.7)
        axes[0].plot(ts, h_att, label="score H atacado", alpha=0.7)
        axes[0].axhline(0.883628, color="red", linestyle="--", label="umbral H")
        axes[0].axvspan(a["attack_start"], a["attack_end"], color="gray", alpha=0.15)
        axes[0].legend(fontsize=7); axes[0].set_title(f"{ep_id} ({row['economic_category']}, detectado_OR={row['detected_induced_OR']})")
        axes[1].plot(ts, ventana["activo_attack"]["OR"].astype(int), label="active_OR atacado", drawstyle="steps-post")
        axes[1].plot(ts, ventana["activo_clean"]["OR"].astype(int), label="active_OR limpio", drawstyle="steps-post", alpha=0.6)
        axes[1].axvspan(a["attack_start"], a["attack_end"], color="gray", alpha=0.15)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        tag = "interior" if row["detected_interior_OR"] else ("boundary" if row["boundary_only_detection_OR"] else "none")
        fname = f"{ep_id}_{row['duration_minutes']}min_{row['shift_type'].lower()}_{row['economic_category']}_{tag}.png"
        fig.savefig(FIGURES_DIR / "episodes" / fname, dpi=110)
        plt.close(fig)


def main():
    base = pd.read_csv(TABLES_DIR / "replay_episode_results.csv", parse_dates=[
        "destination_start", "destination_end", "source_start", "source_end", "first_detection_timestamp"])
    resumen = construir_resumenes(base)
    guardar_resumenes(resumen)
    generar_figuras_globales(base, resumen)
    print("Reporte e informe generados desde tablas persistidas.")


if __name__ == "__main__":
    sys.exit(main() or 0)
