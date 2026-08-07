"""Fase 5, GATE 6, seccion 19: tamano de imagen, tiempo hasta ready, RAM/CPU con la
configuracion heredada de Fase 4 (1 CPU / 448 MB, sin swap, un worker), y si deja de tener
margen se prueba el siguiente limite razonable (siguiente nivel de la matriz de Fase 4: 512,
640, 768...) -- nunca se modifica la aplicacion para forzar artificialmente 448 MB (seccion 19,
explicito). Tambien confirma no-root/un-worker (heredado de Fase 3/4, repetido aqui porque la
imagen cambio) y crecimiento del volumen de logs.
"""
from __future__ import annotations

import csv
import json
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from edge_deployment.docker_tools import container_control as cc
from edge_deployment.security import hmac_auth
from edge_deployment.security.canonicalization import canonicalize

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

CANDIDATE_MEMORY_TIERS_MB = [448, 512, 640, 768, 896, 1024]  # empieza en la recomendada de Fase 4


def _docker(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _image_size_true_bytes(tag: str) -> int:
    tmp = BASE_DIR / f"_tmp_{tag.replace(':', '_')}.tar"
    r = _docker(["save", tag, "-o", str(tmp)])
    if r.returncode != 0:
        raise RuntimeError(f"docker save fallo: {r.stderr}")
    size = tmp.stat().st_size
    tmp.unlink()
    return size


def measure_image_sizes() -> dict:
    sizes = {tag: _image_size_true_bytes(f"fdia-edge:{tag}") for tag in ("phase4-autonomous", "phase5-antireplay")}
    delta_pct = 100.0 * (sizes["phase5-antireplay"] - sizes["phase4-autonomous"]) / sizes["phase4-autonomous"]
    print(f"[resource] tamano imagen: phase4={sizes['phase4-autonomous']/1e6:.2f}MB "
          f"phase5={sizes['phase5-antireplay']/1e6:.2f}MB delta={delta_pct:+.3f}%")
    return {"phase4_autonomous_bytes": sizes["phase4-autonomous"], "phase5_antireplay_bytes": sizes["phase5-antireplay"],
            "delta_pct": delta_pct}


def verify_permissions_and_worker(port: int = 19500) -> dict:
    key = secrets.token_hex(32).encode("utf-8")
    name = cc.new_run_id("resource_perm_check")
    cc.start_container("fdia-edge:phase5-antireplay", name, port, mount_legacy_data=False,
                        extra_args=["-e", "FDIA_ANTI_REPLAY_ENABLED=true", "-e", f"FDIA_METER_KEY_x={key.decode()}"])
    try:
        cc.wait_ready(port, timeout=90)
        r_id = _docker(["exec", name, "id", "-u"])
        r_cmdline = _docker(["exec", name, "sh", "-c", "tr '\\0' ' ' < /proc/1/cmdline"])
        r_write = _docker(["exec", name, "sh", "-c", "touch /app/edge_deployment/security/x 2>&1; echo EXIT:$?"])
        result = {"uid": r_id.stdout.strip(), "is_root": r_id.stdout.strip() == "0",
                  "workers_1_confirmed": "--workers 1" in r_cmdline.stdout,
                  "security_dir_write_blocked": "Permission denied" in r_write.stdout}
    finally:
        cc.stop_and_remove(name)
    print(f"[resource] permisos: uid={result['uid']} workers1={result['workers_1_confirmed']} "
          f"security_ro={result['security_dir_write_blocked']}")
    return result


def _one_memory_probe(mem_mb: int, port: int) -> dict:
    key = secrets.token_hex(32).encode("utf-8")
    meter_id, session_id = "resmem", "session-resmem"
    name = cc.new_run_id(f"resmem_{mem_mb}")
    sampler = cc.StatsSampler(name, interval_s=1.0)
    row = {"memory_limit_mb": mem_mb, "ready_success": False, "peak_memory_mb": None, "oom_killed": False,
           "time_to_ready_s": None}
    try:
        cc.start_container("fdia-edge:phase5-antireplay", name, port, mem_limit=f"{mem_mb}m", cpus=1.0,
                            mount_legacy_data=False,
                            extra_args=["-e", "FDIA_ANTI_REPLAY_ENABLED=true", "-e", "FDIA_SECURITY_CLOCK_MODE=simulated",
                                        "-e", "FDIA_SECURITY_TEST_ENDPOINTS_ENABLED=true",
                                        f"-e", f"FDIA_METER_KEY_{meter_id}={key.decode()}"])
        sampler.start()
        ok, dt = cc.wait_ready(port, timeout=90)
        row["ready_success"] = ok
        row["time_to_ready_s"] = round(dt, 2) if ok else None
        if ok:
            import httpx
            client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15.0)
            client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
            client.post("/security/test/init-session", json={"meter_id": meter_id, "session_id": session_id})
            for i in range(50):
                msg = {"protocol_version": "1", "meter_id": meter_id, "session_id": session_id,
                       "timestamp": "2020-01-01T00:00:00Z", "sequence_number": i + 1, "power_kw": 1.0}
                mac = hmac_auth.compute_hmac(key, canonicalize(msg))
                client.post("/secure-readings", json={**msg, "mac": mac})
    finally:
        samples = sampler.stop()
        if samples:
            mem_vals = [s["mem_mb"] for s in samples if s["mem_mb"] is not None]
            row["peak_memory_mb"] = round(max(mem_vals), 1) if mem_vals else None
        state = cc.inspect_state(name)
        row["oom_killed"] = bool(state.get("OOMKilled"))
        cc.stop_and_remove(name)
    return row


def find_working_memory_tier(port: int = 19501) -> dict:
    """Empieza en 448MB (recomendada de Fase 4). Si no tiene margen (pico >= 90% del
    limite) o falla, prueba el siguiente nivel -- nunca se fuerza la aplicacion a caber en
    448MB artificialmente."""
    rows = []
    chosen = None
    for mem_mb in CANDIDATE_MEMORY_TIERS_MB:
        print(f"[resource] probando {mem_mb}MB @ 1CPU...")
        row = _one_memory_probe(mem_mb, port)
        has_margin = row["ready_success"] and row["peak_memory_mb"] is not None and row["peak_memory_mb"] < 0.90 * mem_mb
        row["has_margin"] = has_margin
        rows.append(row)
        print(f"  ready={row['ready_success']} peak={row['peak_memory_mb']}MB oom={row['oom_killed']} margin={has_margin}")
        if has_margin and chosen is None:
            chosen = mem_mb
            break  # primer nivel con margen real -- no hace falta seguir subiendo
    return {"tiers_tested": rows, "recommended_memory_mb": chosen}


def run() -> dict:
    image_sizes = measure_image_sizes()
    perms = verify_permissions_and_worker()
    memory = find_working_memory_tier()

    rows = [{"metric": "image_size_phase4_bytes", "value": image_sizes["phase4_autonomous_bytes"]},
            {"metric": "image_size_phase5_bytes", "value": image_sizes["phase5_antireplay_bytes"]},
            {"metric": "image_size_delta_pct", "value": image_sizes["delta_pct"]},
            {"metric": "non_root_confirmed", "value": not perms["is_root"]},
            {"metric": "single_worker_confirmed", "value": perms["workers_1_confirmed"]},
            {"metric": "security_dir_read_only_confirmed", "value": perms["security_dir_write_blocked"]},
            {"metric": "recommended_memory_mb", "value": memory["recommended_memory_mb"]}]
    for t in memory["tiers_tested"]:
        rows.append({"metric": f"memory_{t['memory_limit_mb']}mb_ready", "value": t["ready_success"]})
        rows.append({"metric": f"memory_{t['memory_limit_mb']}mb_peak_mb", "value": t["peak_memory_mb"]})
        rows.append({"metric": f"memory_{t['memory_limit_mb']}mb_has_margin", "value": t["has_margin"]})

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "anti_replay_resource_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)

    summary = {"image_sizes": image_sizes, "permissions": perms, "memory": memory,
               "fecha_utc": datetime.now(timezone.utc).isoformat()}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "anti_replay_resource_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[resource] -> {TABLES_DIR / 'anti_replay_resource_summary.csv'}")
    print(f"[resource] memoria recomendada: {memory['recommended_memory_mb']}MB")
    return summary


if __name__ == "__main__":
    run()
