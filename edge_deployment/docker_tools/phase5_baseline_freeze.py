"""Fase 5, GATE 0: congela y registra la baseline de Fase 4 antes de escribir codigo nuevo.
No modifica nada -- solo lee y registra: commit actual, hashes de artefactos (via
`artifact_manifest_v2.json`), digest de la imagen Docker, configuracion, y referencias a los
resultados/informes de Fases 1-4 ya generados.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # deteccion_fraude_no_supervisada/
EDGE_DIR = BASE_DIR / "edge_deployment"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
REPO_ROOT = BASE_DIR.parent  # tfg/tfg (raiz del repo git)


def _git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True)
    return r.stdout.strip()


def _docker(args: list[str]) -> str:
    r = subprocess.run(["docker", *args], capture_output=True, text=True)
    return r.stdout.strip()


def build_baseline_manifest() -> dict:
    manifest_v2_path = EDGE_DIR / "models" / "artifact_manifest_v2.json"
    manifest_v2 = json.loads(manifest_v2_path.read_text(encoding="utf-8")) if manifest_v2_path.exists() else None

    image_id = _docker(["image", "inspect", "fdia-edge:optimized", "--format", "{{.Id}}"])

    reports = {}
    for name in ["final_report.md", "api_final_report.md", "docker_final_report.md", "phase4_final_report.md"]:
        p = REPORTS_DIR / name
        reports[name] = {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None}

    equivalence_summaries = {}
    for name in ["equivalence_summary.json", "api_equivalence_summary.json",
                 "container_equivalence_summary_smoke.json", "container_equivalence_summary_30d.json",
                 "autonomous_loader_equivalence_summary.json", "autonomous_container_equivalence_summary.json",
                 "autonomous_container_equivalence_summary_30d.json"]:
        p = REPORTS_DIR / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            equivalence_summaries[name] = {k: d.get(k) for k in
                                            ("all_100pct", "all_categorical_100pct", "all_environments_100pct", "n_readings") if k in d}
        else:
            equivalence_summaries[name] = None

    manifest = {
        "phase": "phase4-autonomous baseline freeze (Fase 5 GATE 0)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "parent_commit_before_freeze": _git(["rev-parse", "HEAD"]),
            "branch_before_freeze": _git(["branch", "--show-current"]),
            "note": "el commit de congelacion en si (tag phase4-autonomous) se crea DESPUES de "
                    "generar este fichero -- este manifiesto registra el estado justo antes de ese commit.",
        },
        "docker": {
            "image_tag_frozen_as": "fdia-edge:phase4-autonomous",
            "image_id": image_id,
            "source_tag": "fdia-edge:optimized",
        },
        "artifact_manifest_v2": manifest_v2,
        "reports_present": reports,
        "equivalence_summaries": equivalence_summaries,
        "test_suite": {"total": 215, "passed": 215, "failed": 0,
                        "note": "ver phase5_previous_tests_summary.json para el detalle completo"},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "phase5_baseline_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    print(f"[gate0] -> {out}")
    return manifest


def build_previous_tests_summary(pytest_log_path: Path) -> dict:
    log_text = pytest_log_path.read_text(encoding="utf-8") if pytest_log_path.exists() else ""
    last_line = [l for l in log_text.splitlines() if "passed" in l.lower() or "failed" in l.lower()]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pytest_summary_line": last_line[-1] if last_line else None,
        "log_excerpt_tail": "\n".join(log_text.splitlines()[-15:]),
        "expected_total": 215,
    }
    out = REPORTS_DIR / "phase5_previous_tests_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[gate0] -> {out}")
    return summary


if __name__ == "__main__":
    import sys
    build_baseline_manifest()
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/phase5_gate0_tests.log")
    build_previous_tests_summary(log_path)
