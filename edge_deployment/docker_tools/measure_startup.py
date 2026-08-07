"""Fase 3: mide, con
varios arranques (contenedor nuevo cada vez, salvaguarda de no reutilizar contenedores
calentados), el tiempo hasta /health, tiempo hasta /ready, RAM y CPU durante el arranque, para
ambas imagenes (baseline y optimized). El tamano de imagen se mide con `docker save` (tamano
real de capas) en vez de `docker images`/`docker system df`, que en este entorno (BuildKit +
containerd store con atestaciones de proveniencia) sobre-reporta el tamano real por un factor
de ~3.5-4.6x -- discrepancia verificada exportando el tar y comparando bytes, ver
docker_final_report.md, nota metodologica.
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import httpx

from edge_deployment.docker_tools import container_control as cc

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"

IMAGES = {"baseline": "fdia-edge:baseline", "optimized": "fdia-edge:optimized"}
N_STARTUP_RUNS = 5
BASE_PORT = 18100


def _docker(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _image_size_true_bytes(tag: str) -> int:
    """`docker save` exporta las capas reales (single-platform) -- coincide con
    `docker image inspect --format '{{.Size}}'`, a diferencia de `docker images`."""
    tmp = BASE_DIR / f"_tmp_{tag.replace(':', '_')}.tar"
    r = _docker(["save", tag, "-o", str(tmp)])
    if r.returncode != 0:
        raise RuntimeError(f"docker save fallo: {r.stderr}")
    size = tmp.stat().st_size
    tmp.unlink()
    return size


def _image_size_reported_bytes(tag: str) -> int:
    r = _docker(["image", "inspect", tag, "--format", "{{json .Size}}"])
    return int(json.loads(r.stdout.strip()))


def _n_layers(tag: str) -> int:
    r = _docker(["history", tag, "-q"])
    return len([l for l in r.stdout.splitlines() if l.strip()])


def measure_image_metrics(build_times_s: dict[str, float]) -> list[dict]:
    context_manifest = json.loads((EDGE_DIR / "docker_context" / "context_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for tag_name, image in IMAGES.items():
        rows.append({
            "image": tag_name, "docker_tag": image,
            "size_bytes_true_docker_save": _image_size_true_bytes(image),
            "size_bytes_reported_docker_images": _image_size_reported_bytes(image),
            "n_layers": _n_layers(image),
            "build_context_n_files": context_manifest["n_files"],
            "build_context_size_bytes": context_manifest["total_size_bytes"],
            "build_time_seconds": build_times_s.get(tag_name),
        })
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "docker_image_metrics.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[image-metrics] -> {out}")
    for r in rows:
        print(f"  {r['image']}: true={r['size_bytes_true_docker_save']/1e6:.1f}MB "
              f"reported={r['size_bytes_reported_docker_images']/1e6:.1f}MB layers={r['n_layers']} "
              f"build={r['build_time_seconds']:.1f}s")
    return rows


def _wait_both_from_start(host_port: int, timeout: float = 120.0) -> tuple[bool, float | None, bool, float | None]:
    """Sondea /health y /ready en el mismo bucle, ambos cronometrados desde el mismo t0 (el
    arranque del contenedor) -- necesario porque Starlette bloquea todas las rutas (incluida
    /health) hasta que el lifespan llega al `yield`, que en el camino de exito ocurre despues
    de cargar P+H (ver lifecycle.py, sin modificar): /health y /ready quedan disponibles casi
    en el mismo instante, y medirlas de forma secuencial (primero /health hasta que responde,
    luego /ready desde un cronometro nuevo) infravalora sistematicamente el tiempo hasta
    /ready (queda como "tiempo extra tras /health" en vez de "tiempo desde el arranque")."""
    base = f"http://127.0.0.1:{host_port}"
    t0 = time.perf_counter()
    t_health, t_ready = None, None
    ok_health, ok_ready = False, False
    while time.perf_counter() - t0 < timeout and not (ok_health and ok_ready):
        now = time.perf_counter() - t0
        if not ok_health:
            try:
                r = httpx.get(f"{base}/health", timeout=2.0)
                if r.status_code == 200:
                    ok_health, t_health = True, now
            except Exception:
                pass
        if not ok_ready:
            try:
                r = httpx.get(f"{base}/ready", timeout=2.0)
                if r.status_code == 200 and r.json().get("ready"):
                    ok_ready, t_ready = True, now
            except Exception:
                pass
        if not (ok_health and ok_ready):
            time.sleep(0.1)
    return ok_health, t_health, ok_ready, t_ready


def _one_startup_run(image_tag: str, image_name: str, run_idx: int, host_port: int) -> dict:
    name = cc.new_run_id(f"startup_{image_name}_{run_idx}")
    sampler = cc.StatsSampler(name, interval_s=0.5)
    t0 = time.perf_counter()
    cc.start_container(image_tag, name, host_port)
    sampler.start()
    ok_health, t_health, ok_ready, t_ready = _wait_both_from_start(host_port, timeout=120)
    # una muestra extra justo en el instante "ready" para "RAM al quedar ready"
    time.sleep(0.5)
    samples = sampler.stop()
    ram_at_ready_mb = samples[-1]["mem_mb"] if samples else None
    cpu_during_startup = [s["cpu_pct"] for s in samples if s["cpu_pct"] is not None]
    rss_mb = None
    try:
        r = httpx.get(f"http://127.0.0.1:{host_port}/metrics", timeout=5.0)
        if r.status_code == 200:
            rss_mb = r.json().get("approximate_memory_mb")
    except Exception:
        pass
    state = cc.inspect_state(name)
    logs_tail = cc.container_logs(name, tail=50)
    cc.stop_and_remove(name)
    return {
        "image": image_name, "run_idx": run_idx, "run_id": name,
        "time_to_health_s": round(t_health, 3) if ok_health else None,
        "time_to_ready_s": round(t_ready, 3) if ok_ready else None,
        "health_ok": ok_health, "ready_ok": ok_ready,
        "ram_at_ready_mb_docker_stats": round(ram_at_ready_mb, 1) if ram_at_ready_mb else None,
        "process_rss_mb_at_ready_via_metrics": rss_mb,
        "cpu_pct_mean_during_startup": round(sum(cpu_during_startup) / len(cpu_during_startup), 2) if cpu_during_startup else None,
        "cpu_pct_max_during_startup": round(max(cpu_during_startup), 2) if cpu_during_startup else None,
        "n_stats_samples": len(samples),
        "container_status": state.get("Status"), "exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled"),
        "elapsed_total_s": round(time.perf_counter() - t0, 3),
        "logs_tail_if_not_ready": logs_tail[-1500:] if not ok_ready else "",
    }


def measure_startup(n_runs: int = N_STARTUP_RUNS) -> list[dict]:
    rows = []
    port = BASE_PORT
    for image_name, image_tag in IMAGES.items():
        for i in range(n_runs):
            print(f"[startup] {image_name} run {i+1}/{n_runs} (port {port})...")
            row = _one_startup_run(image_tag, image_name, i, port)
            print(f"  health={row['time_to_health_s']}s ready={row['time_to_ready_s']}s "
                  f"ram_docker_stats={row['ram_at_ready_mb_docker_stats']}MB rss={row['process_rss_mb_at_ready_via_metrics']}MB "
                  f"cpu_mean={row['cpu_pct_mean_during_startup']}%")
            rows.append(row)
            port += 1
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "docker_startup_metrics.csv"
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[startup] -> {out}")
    return rows


if __name__ == "__main__":
    import sys
    build_times = {"baseline": 149.485, "optimized": 136.157}  # medidos en esta sesion (ver mensajes de build)
    measure_image_metrics(build_times)
    measure_startup(n_runs=int(sys.argv[1]) if len(sys.argv) > 1 else N_STARTUP_RUNS)
