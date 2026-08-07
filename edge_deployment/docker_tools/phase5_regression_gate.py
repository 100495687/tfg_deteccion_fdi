"""Fase 5, GATE 3: regresion de Fase 4. Compara `/readings` entre `fdia-edge:phase4-autonomous`
(congelada en GATE 0, nunca reconstruida) y `fdia-edge:phase5-antireplay` (nueva). Ambas se
comparan contra la MISMA referencia (`DirectEngineClient`, motor autonomo sin HTTP, identico
codigo de `/readings` en ambas imagenes) sobre el MISMO periodo pre-test real -- por
transitividad, si las dos coinciden al 100% con la referencia, coinciden al 100% entre si, sin
necesidad de una comparacion par a par que duplique el trafico HTTP.

Si aparece una sola diferencia: seccion 3 exige detener Fase 5, no ejecutar ataques, identificar
y revertir el cambio, conservar Fase 4 intacta -- este script lo hace explicito en su salida
(`gate_passed`), pero la decision de detenerse la toma quien lo ejecuta (documentado en el
informe final).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from edge_deployment.docker_tools import container_control as cc

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

PORT_A = 19100  # fdia-edge:phase4-autonomous
PORT_B = 19101  # fdia-edge:phase5-antireplay


def _run_against_image(image: str, port: int, days: int, out_suffix: str, label: str) -> dict:
    from edge_deployment.clients.api_stream_client import run_equivalence

    name = cc.new_run_id(label)
    print(f"[gate3] arrancando {image} ({name}, puerto {port})...")
    cc.start_container(image, name, port, mount_legacy_data=False)
    try:
        ok, dt = cc.wait_ready(port, timeout=90)
        if not ok:
            raise RuntimeError(f"{image} no alcanzo ready=true. Logs:\n{cc.container_logs(name, 80)}")
        result = run_equivalence(max_stream_days=days, base_url=f"http://127.0.0.1:{port}",
                                  overwrite=True, out_suffix=out_suffix)
    finally:
        cc.stop_and_remove(name)
    return result


def run_gate3(days: int = 3) -> dict:
    r_a = _run_against_image("fdia-edge:phase4-autonomous", PORT_A, days, "_phase5gate_A_phase4", "gate3_phase4")
    r_b = _run_against_image("fdia-edge:phase5-antireplay", PORT_B, days, "_phase5gate_B_phase5", "gate3_phase5")

    df_a = r_a["df"].copy(); df_a.insert(0, "image", "A_phase4-autonomous")
    df_b = r_b["df"].copy(); df_b.insert(0, "image", "B_phase5-antireplay")
    combined = pd.concat([df_a, df_b], ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES_DIR / "phase4_vs_phase5_plain_equivalence.csv"
    combined.to_csv(out_csv, index=False)

    campos = ["accepted", "engine_status", "p_ready", "h_ready", "p_evaluated", "h_evaluated",
              "alert_p", "alert_p_ffill", "alert_h", "alert_or", "detection_source",
              "buffer_1min_size", "history_15min_size", "rejection_reason"]
    match_a = {c: r_a["resumen"]["match_pct"][c] for c in campos}
    match_b = {c: r_b["resumen"]["match_pct"][c] for c in campos}
    gate_passed = all(v == 100.0 for v in match_a.values()) and all(v == 100.0 for v in match_b.values())

    summary = {
        "days": days, "gate": "GATE_3_phase4_vs_phase5_regression",
        "A_phase4_autonomous": {"n_readings": r_a["resumen"]["n_readings"], "match_pct": match_a,
                                 "score_p_max_abs_diff": r_a["resumen"]["score_p_max_abs_diff"],
                                 "score_h_max_abs_diff": r_a["resumen"]["score_h_max_abs_diff"]},
        "B_phase5_antireplay": {"n_readings": r_b["resumen"]["n_readings"], "match_pct": match_b,
                                 "score_p_max_abs_diff": r_b["resumen"]["score_p_max_abs_diff"],
                                 "score_h_max_abs_diff": r_b["resumen"]["score_h_max_abs_diff"]},
        "gate_passed": gate_passed,
        "action_if_failed": "detener Fase 5, no ejecutar ataques ni benchmarks, identificar y revertir el cambio, "
                             "conservar fdia-edge:phase4-autonomous intacta",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "phase5_gate3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"[gate3] -> {out_csv}")
    print(f"[gate3] gate_passed={gate_passed}")
    print(json.dumps({"A_phase4": match_a, "B_phase5": match_b}, indent=2))
    return summary


if __name__ == "__main__":
    run_gate3()
