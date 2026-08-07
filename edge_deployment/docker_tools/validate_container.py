"""Fase 3: validacion funcional del contenedor. Para el caso de hash incorrecto,
la salvaguarda del usuario prohibe modificar ningun artefacto congelado original -- se
construye una imagen de prueba desechable (`fdia-edge:broken-hash-test`) a partir de una
copia temporal de docker_context/ con un solo byte alterado en un artefacto, nunca se toca
edge_deployment/docker_context/ ni los originales en models/. La imagen desechable se borra al
terminar.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from edge_deployment.docker_tools import container_control as cc

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
CONTEXT_DIR = EDGE_DIR / "docker_context"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"
BROKEN_CONTEXT_DIR = EDGE_DIR / "_tmp_broken_docker_context"
BROKEN_TAG = "fdia-edge:broken-hash-test"


def _docker(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _build_broken_context(mode: str) -> None:
    """Copia completa de docker_context/ (nunca los originales de models/) en un directorio
    temporal. `mode="missing"` borra un artefacto requerido -- este es el fallo que
    `model_loader.build_input_artifacts_inventory` (Fase 1, sin modificar) detecta realmente:
    `hashes_valid` en ese codigo comprueba presencia de cada artefacto requerido, no una
    comparacion criptografica contra un hash de referencia congelado (el manifiesto solo
    registra el sha256, no lo valida contra nada). `mode="corrupt"` altera un byte del joblib
    de H para documentar, honestamente, que ese escenario (fichero presente pero corrupto) no
    lo detecta la logica actual -- no se parchea `model_loader.py` para cubrirlo (fuera de
    alcance de Fase 3, ver salvaguarda de no tocar core/api)."""
    if BROKEN_CONTEXT_DIR.exists():
        shutil.rmtree(BROKEN_CONTEXT_DIR)
    shutil.copytree(CONTEXT_DIR, BROKEN_CONTEXT_DIR)
    target = BROKEN_CONTEXT_DIR / "models" / "final_or_pretest" / "histgb_periodic_final.joblib"
    if mode == "missing":
        target.unlink()
    elif mode == "corrupt":
        data = bytearray(target.read_bytes())
        data[0] ^= 0xFF
        target.write_bytes(bytes(data))
    else:
        raise ValueError(mode)


def _build_broken_image() -> None:
    r = _docker(["build", "-f", str(EDGE_DIR / "Dockerfile"), "-t", BROKEN_TAG, str(BROKEN_CONTEXT_DIR)])
    if r.returncode != 0:
        raise RuntimeError(f"build de imagen de prueba (hash roto) fallo: {r.stdout[-3000:]} {r.stderr[-3000:]}")


def test_hash_failure_yields_503(mode: str = "missing") -> dict:
    print(f"[validate] construyendo contexto e imagen desechables (mode={mode})...")
    _build_broken_context(mode)
    _build_broken_image()
    name = cc.new_run_id(f"hashfail_{mode}")
    port = 18199
    result = {"scenario": f"artefacto_{mode}"}
    try:
        cc.start_container(BROKEN_TAG, name, port)
        ok_health, t_health = cc.wait_health(port, timeout=60)
        time.sleep(2)  # tiempo de sobra para que /ready refleje el fallo si health ya respondio
        try:
            r_ready = httpx.get(f"http://127.0.0.1:{port}/ready", timeout=5.0)
            ready_status, ready_body = r_ready.status_code, r_ready.json()
        except Exception as e:
            ready_status, ready_body = None, {"connect_error": str(e)}
        try:
            r_reading = httpx.post(f"http://127.0.0.1:{port}/readings", timeout=5.0,
                                    json={"meter_id": "m_hashfail_test", "timestamp": "2020-01-01T00:00:00", "power_kw": 1.0})
            reading_status, reading_body = r_reading.status_code, r_reading.json()
        except Exception as e:
            reading_status, reading_body = None, {"connect_error": str(e)}
        logs_tail = cc.container_logs(name, tail=40)
        result.update({
            "health_ok": ok_health, "time_to_health_s": round(t_health, 2) if ok_health else None,
            "ready_status_code": ready_status, "ready_body": ready_body,
            "reading_status_code": reading_status, "reading_body": reading_body,
            "criteria_met": bool(ok_health and ready_status == 503 and reading_status in (503, 404)),
            "logs_tail": logs_tail[-1500:],
        })
    finally:
        cc.stop_and_remove(name)
        _docker(["rmi", "-f", BROKEN_TAG])
        if BROKEN_CONTEXT_DIR.exists():
            shutil.rmtree(BROKEN_CONTEXT_DIR)
    return result


def test_non_root_and_single_worker(image_tag: str = "fdia-edge:optimized") -> dict:
    name = cc.new_run_id("permcheck")
    port = 18198
    result = {"scenario": "usuario_no_root_y_worker_unico", "image": image_tag}
    try:
        cc.start_container(image_tag, name, port)
        cc.wait_ready(port, timeout=60)
        r_id = _docker(["exec", name, "id", "-u"])
        r_name = _docker(["exec", name, "id", "-un"])
        r_cmdline = _docker(["exec", name, "sh", "-c",
                              "tr '\\0' ' ' < /proc/1/cmdline"])
        r_write_models = _docker(["exec", name, "sh", "-c", "touch /app/models/.write_test 2>&1; echo EXIT:$?"])
        uid = r_id.stdout.strip()
        result.update({
            "uid": uid, "username": r_name.stdout.strip(), "is_root": uid == "0",
            "pid1_cmdline": r_cmdline.stdout.strip(),
            "workers_1_confirmed": "--workers 1" in r_cmdline.stdout,
            "models_dir_write_blocked": "Permission denied" in r_write_models.stdout,
            "criteria_met": (uid != "0" and "--workers 1" in r_cmdline.stdout and "Permission denied" in r_write_models.stdout),
        })
    finally:
        cc.stop_and_remove(name)
    return result


def run_all() -> dict:
    r1 = test_hash_failure_yields_503(mode="missing")
    print(f"[validate] artefacto ausente -> health={r1['health_ok']} ready={r1['ready_status_code']} "
          f"reading={r1['reading_status_code']} criteria_met={r1['criteria_met']}")
    r1b = test_hash_failure_yields_503(mode="corrupt")
    print(f"[validate] artefacto presente pero corrupto (exploratorio, limitacion conocida) -> "
          f"health={r1b['health_ok']} ready={r1b['ready_status_code']} reading={r1b['reading_status_code']}")
    r2 = test_non_root_and_single_worker("fdia-edge:optimized")
    print(f"[validate] optimized: uid={r2['uid']} workers1={r2['workers_1_confirmed']} "
          f"models_ro={r2['models_dir_write_blocked']} criteria_met={r2['criteria_met']}")
    r3 = test_non_root_and_single_worker("fdia-edge:baseline")
    print(f"[validate] baseline: uid={r3['uid']} (se espera root=True, no aplica no-root a baseline) "
          f"workers1={r3['workers_1_confirmed']}")
    summary = {"missing_artifact_test": r1, "corrupt_artifact_exploratory_test": r1b,
               "optimized_permissions_test": r2, "baseline_permissions_test": r3}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "docker_startup_validation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


if __name__ == "__main__":
    run_all()
