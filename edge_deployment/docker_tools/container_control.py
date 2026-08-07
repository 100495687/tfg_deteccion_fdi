"""Fase 3: control de contenedores compartido por todos los experimentos.
Cada experimento arranca siempre un contenedor nuevo con un run_id unico (salvaguarda del
usuario: "cada configuracion debe ejecutarse en un contenedor nuevo, con run_id unico y
estado limpio. No reutilices contenedores calentados"). Invocado siempre como subproceso de
Python (nunca desde git-bash directamente) para evitar el problema conocido de MSYS
reescribiendo argumentos "/app/..." como rutas de Windows en las llamadas a `docker` desde
Git Bash -- ver docker_final_report.md, nota metodologica.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # deteccion_fraude_no_supervisada/
DATA_RAW_HOST = BASE_DIR / "data" / "raw" / "data_raw.csv"
EPISODES_HOST = BASE_DIR / "results" / "tables" / "episodes_master_val.csv"

CONTAINER_PORT = 8000


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _docker(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kwargs)


def start_container(image: str, name: str, host_port: int, mem_limit: str | None = None,
                     cpus: float | None = None, extra_args: list[str] | None = None,
                     mount_legacy_data: bool = False) -> str:
    """Arranca un contenedor nuevo (nombre unico obligatorio). Fase 4: por defecto no monta
    ningun CSV -- el autonomous_loader (produccion, por defecto en DetectorEngine()) no los
    necesita. `mount_legacy_data=True` monta los 2 ficheros de solo lectura documentados en
    docker_context/RUNTIME_MOUNTS.md (nota historica) -- solo para probar explicitamente el
    legacy_loader dentro de un contenedor, no es el camino de produccion. Devuelve el
    container id."""
    args = ["run", "-d", "--name", name, "-p", f"{host_port}:{CONTAINER_PORT}"]
    if mount_legacy_data:
        if not DATA_RAW_HOST.exists() or not EPISODES_HOST.exists():
            raise FileNotFoundError(f"faltan los ficheros de montaje: {DATA_RAW_HOST} / {EPISODES_HOST}")
        args += ["-v", f"{DATA_RAW_HOST}:/app/data/raw/data_raw.csv:ro",
                 "-v", f"{EPISODES_HOST}:/app/results/tables/episodes_master_val.csv:ro"]
    if mem_limit:
        args += ["--memory", mem_limit, "--memory-swap", mem_limit]  # sin swap: OOM real y medible
    if cpus:
        args += ["--cpus", str(cpus)]
    if extra_args:
        args += extra_args
    args.append(image)
    r = _docker(args)
    if r.returncode != 0:
        raise RuntimeError(f"docker run fallo para {name}: {r.stderr}")
    return r.stdout.strip()


def stop_and_remove(name: str, timeout: int = 10) -> None:
    _docker(["stop", "-t", str(timeout), name])
    _docker(["rm", "-f", name])


def inspect_state(name: str) -> dict:
    r = _docker(["inspect", name, "--format", "{{json .State}}"])
    if r.returncode != 0:
        return {"error": r.stderr}
    return json.loads(r.stdout)


def container_logs(name: str, tail: int = 200) -> str:
    r = _docker(["logs", "--tail", str(tail), name])
    return (r.stdout or "") + (r.stderr or "")


def _wait_http(url_path: str, host_port: int, timeout: float, expect_field: str | None = None) -> tuple[bool, float]:
    import httpx
    t0 = time.perf_counter()
    base = f"http://127.0.0.1:{host_port}"
    while time.perf_counter() - t0 < timeout:
        try:
            r = httpx.get(f"{base}{url_path}", timeout=2.0)
            if r.status_code == 200 and (expect_field is None or r.json().get(expect_field)):
                return True, time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(0.25)
    return False, time.perf_counter() - t0


def wait_health(host_port: int, timeout: float = 120.0) -> tuple[bool, float]:
    return _wait_http("/health", host_port, timeout, expect_field=None)


def wait_ready(host_port: int, timeout: float = 120.0) -> tuple[bool, float]:
    return _wait_http("/ready", host_port, timeout, expect_field="ready")


@dataclass
class StatsSampler:
    name: str
    interval_s: float = 1.0
    samples: list = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _parse_mem(self, mem_usage: str) -> float | None:
        # "123.4MiB / 512MiB" -> 123.4 (en MB, convertido desde MiB/GiB/KiB con base 1e6/1e3 decimal
        # para que sea comparable con RSS medido via psutil, que tambien reportamos en MB decimales)
        try:
            used = mem_usage.split(" / ")[0].strip()
            for suf, mult in (("GiB", 1024**3 / 1e6), ("MiB", 1024**2 / 1e6), ("KiB", 1024 / 1e6), ("B", 1 / 1e6)):
                if used.endswith(suf):
                    return float(used[: -len(suf)]) * mult
        except Exception:
            return None
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            r = _docker(["stats", self.name, "--no-stream", "--format", "{{json .}}"])
            t = time.time()
            if r.returncode == 0 and r.stdout.strip():
                try:
                    d = json.loads(r.stdout.strip().splitlines()[-1])
                    cpu_pct = float(d.get("CPUPerc", "0%").rstrip("%"))
                    mem_mb = self._parse_mem(d.get("MemUsage", ""))
                    self.samples.append({"t_wall": t, "cpu_pct": cpu_pct, "mem_mb": mem_mb,
                                          "pids": d.get("PIDs")})
                except Exception:
                    pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples
