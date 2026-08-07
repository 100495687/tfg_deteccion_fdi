"""Fase 4: `artifact_integrity_cases.csv` -- casos de integridad probados contra
`manifest_v2.validate_manifest_v2` y, para los 2 primeros, tambien de extremo a extremo contra
la API real (TestClient). Todos con copias temporales; ningun artefacto original se toca.
"""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"


def _api_probe(manifest_path: Path) -> dict:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edge_deployment.api import lifecycle as lc
    from edge_deployment.api.error_handlers import register_error_handlers
    from edge_deployment.api.routes import router
    from edge_deployment.core import manifest_v2

    def fake_validate_artifacts():
        import time
        t0 = time.perf_counter()
        integrity = manifest_v2.validate_manifest_v2(manifest_path=manifest_path)
        dt = (time.perf_counter() - t0) * 1000
        missing_and_mismatched = list(integrity["missing"]) + [m["artifact"] for m in integrity["mismatched"]]
        return {"hashes_valid": integrity["integrity_valid"], "n_missing": integrity["n_missing"] + integrity["n_mismatched"],
                "missing": missing_and_mismatched, "manifest": integrity, "validation_time_ms": dt}

    original = lc.validate_artifacts
    try:
        lc.validate_artifacts = fake_validate_artifacts
        app = FastAPI(lifespan=lc.lifespan)
        register_error_handlers(app)
        app.include_router(router)
        with TestClient(app) as client:
            r_health = client.get("/health")
            r_ready = client.get("/ready")
            r_reading = client.post("/readings", json={"meter_id": "m1", "timestamp": "2020-01-01T00:00:00", "power_kw": 1.0})
            return {"health_status": r_health.status_code, "ready_status": r_ready.status_code,
                    "reading_status": r_reading.status_code,
                    "no_stack_trace_leaked": "Traceback" not in json.dumps(r_ready.json()) and "Traceback" not in json.dumps(r_reading.json())}
    finally:
        lc.validate_artifacts = original


def run() -> list[dict]:
    from edge_deployment.core import manifest_v2

    real = json.loads(manifest_v2.MANIFEST_V2_PATH.read_text(encoding="utf-8"))
    tmpdir = Path(tempfile.mkdtemp())
    rows = []
    try:
        # Caso 1: artefacto ausente (modelo H)
        m = json.loads(json.dumps(real))
        m["artifacts"]["model_H"]["path"] = "no_existe/en_ningun_sitio.joblib"
        p = tmpdir / "missing.json"
        p.write_text(json.dumps(m), encoding="utf-8")
        integrity = manifest_v2.validate_manifest_v2(manifest_path=p)
        api = _api_probe(p)
        rows.append({"case": "artefacto_ausente", "artifact": "model_H", "integrity_valid": integrity["integrity_valid"],
                     "n_missing": integrity["n_missing"], "n_mismatched": integrity["n_mismatched"], **api,
                     "original_artifact_untouched": True})

        # Caso 2: artefacto corrupto (params_norm_P, 1 byte alterado en copia temporal)
        real_path = manifest_v2.BASE_DIR / real["artifacts"]["params_norm_P"]["path"]
        original_bytes = real_path.read_bytes()
        corrupt = tmpdir / "params_norm_p_corrupt.joblib"
        data = bytearray(original_bytes)
        data[0] ^= 0xFF
        corrupt.write_bytes(bytes(data))
        m2 = json.loads(json.dumps(real))
        m2["artifacts"]["params_norm_P"]["path"] = str(corrupt)
        p2 = tmpdir / "corrupt.json"
        p2.write_text(json.dumps(m2), encoding="utf-8")
        integrity2 = manifest_v2.validate_manifest_v2(manifest_path=p2)
        api2 = _api_probe(p2)
        rows.append({"case": "artefacto_corrupto_1_byte", "artifact": "params_norm_P", "integrity_valid": integrity2["integrity_valid"],
                     "n_missing": integrity2["n_missing"], "n_mismatched": integrity2["n_mismatched"], **api2,
                     "original_artifact_untouched": real_path.read_bytes() == original_bytes})

        # Caso 3: manifiesto valido (control, sin alterar nada)
        integrity3 = manifest_v2.validate_manifest_v2()
        api3 = _api_probe(manifest_v2.MANIFEST_V2_PATH)
        rows.append({"case": "control_manifiesto_valido", "artifact": "n/a", "integrity_valid": integrity3["integrity_valid"],
                     "n_missing": integrity3["n_missing"], "n_mismatched": integrity3["n_mismatched"], **api3,
                     "original_artifact_untouched": True})
    finally:
        shutil.rmtree(tmpdir)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "artifact_integrity_cases.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[integrity-cases] -> {out}")
    for r in rows:
        print(f"  {r['case']}: integrity_valid={r['integrity_valid']} health={r['health_status']} "
              f"ready={r['ready_status']} reading={r['reading_status']} original_untouched={r['original_artifact_untouched']}")
    return rows


if __name__ == "__main__":
    run()
