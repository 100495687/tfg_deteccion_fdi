"""Fase 3: comparacion final entre entornos. Reutiliza medidas ya calculadas (no
vuelve a arrancar nada): api_uvicorn_benchmark.json de Fase 2 (API local, Uvicorn nativo, sin
Docker) + docker_latency_benchmark.json/docker_resource_summary.csv (Docker sin restricciones
"referencia/cloud emulado") + docker_memory_limit_matrix.csv (para los niveles "edge moderado",
"edge limitado", "minima estable" y "primera fallida").

Definiciones (se citan literalmente en el informe):
  - referencia/cloud emulado = el mismo contenedor optimized, sin --memory/--cpus (recursos
    del host Docker Desktop/WSL2 disponibles).
  - edge emulado = el mismo contenedor con --memory/--cpus limitados. No se
    afirma que esto reproduzca una Raspberry Pi real -- ver docker_final_report.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
TABLES_DIR = EDGE_DIR / "results" / "tables"
REPORTS_DIR = EDGE_DIR / "results" / "reports"


def _row_from_phase2_local_api() -> dict:
    p = REPORTS_DIR / "api_uvicorn_benchmark.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "environment": "A_api_local_sin_docker", "description": "Fase 2: Uvicorn nativo, sin contenedor",
        "memory_limit_mb": None, "cpu_limit": None,
        "latency_p50_ms": d["round_trip"]["median_ms"], "latency_p95_ms": d["round_trip"]["p95_ms"],
        "latency_p99_ms": d["round_trip"]["p99_ms"], "throughput_rps": d["throughput_readings_per_second"],
        "ram_mb": 831.6,  # Fase 2, api_final_report.md (memoria estable documentada)
        "startup_time_s": None, "equivalence_all_100pct": True, "stability": "n/a (no es Docker)",
        "source": "results/reports/api_uvicorn_benchmark.json (Fase 2)",
    }


def _row_docker_unrestricted() -> dict:
    lat = json.loads((REPORTS_DIR / "docker_latency_benchmark.json").read_text(encoding="utf-8"))
    startup = pd.read_csv(TABLES_DIR / "docker_startup_metrics.csv")
    startup_opt = startup[startup.image == "optimized"]
    res = None
    res_path = TABLES_DIR / "docker_resource_summary.csv"
    ram_max = None
    if res_path.exists():
        dfres = pd.read_csv(res_path)
        steady = dfres[dfres.phase.str.contains("estable", na=False)]
        ram_max = float(steady["mem_mb_max"].max()) if len(steady) else float(dfres["mem_mb_max"].max())
    smoke = json.loads((REPORTS_DIR / "container_equivalence_summary_smoke.json").read_text(encoding="utf-8"))
    return {
        "environment": "B_docker_sin_restricciones", "description": "fdia-edge:optimized, sin --memory/--cpus (referencia/cloud emulado)",
        "memory_limit_mb": None, "cpu_limit": None,
        "latency_p50_ms": lat["overall_round_trip"]["median_ms"], "latency_p95_ms": lat["overall_round_trip"]["p95_ms"],
        "latency_p99_ms": lat["overall_round_trip"]["p99_ms"], "throughput_rps": lat["throughput_readings_per_second"],
        "ram_mb": ram_max if ram_max is not None else startup_opt["ram_at_ready_mb_docker_stats"].mean(),
        "startup_time_s": startup_opt["time_to_ready_s"].mean(),
        "equivalence_all_100pct": smoke["per_environment"]["C_docker_optimized"]["all_alert_fields_100pct"],
        "stability": "estable (sin restriccion)",
        "source": "docker_latency_benchmark.json + docker_resource_summary.csv + docker_startup_metrics.csv",
    }


def _rows_from_memory_matrix() -> list[dict]:
    df = pd.read_csv(TABLES_DIR / "docker_memory_limit_matrix.csv").sort_values("memory_limit_mb", ascending=False)
    tiers = json.loads((REPORTS_DIR / "docker_memory_limit_tiers.json").read_text(encoding="utf-8"))
    filas = []
    funcionales = df[df["ready_success"] & df["bootstrap_success"] & df["streaming_success"] & df["equivalence_passed"]]
    if len(funcionales) >= 2:
        moderado = funcionales.iloc[len(funcionales) // 3]
        filas.append(_row_from_matrix_row(moderado, "C_docker_edge_moderado", "nivel intermedio de la matriz de memoria (funcional, con margen)"))
    estable_mb = tiers.get("minimo_estable_mb")
    if estable_mb is not None:
        row = df[df.memory_limit_mb == estable_mb].iloc[0]
        filas.append(_row_from_matrix_row(row, "D_config_minima_estable", f"minimo_estable_mb={estable_mb} (docker_memory_limit_tiers.json)"))
    limitados = funcionales[funcionales.memory_limit_mb < (estable_mb or 1e9)] if estable_mb else funcionales
    if len(limitados):
        limitado = limitados.sort_values("memory_limit_mb").iloc[0]
        if limitado["memory_limit_mb"] != estable_mb:
            filas.append(_row_from_matrix_row(limitado, "E_docker_edge_limitado", "funcional pero por debajo de la config. minima estable -- no recomendada"))
    fallidas = df[~(df["ready_success"] & df["bootstrap_success"] & df["streaming_success"] & df["equivalence_passed"])].sort_values("memory_limit_mb", ascending=False)
    if len(fallidas):
        primera_fallida = fallidas.iloc[0]
        filas.append(_row_from_matrix_row(primera_fallida, "F_primera_config_fallida", f"primer limite (de mayor a menor) que falla: {primera_fallida['failure_stage']}/{primera_fallida['failure_reason']}"))
    return filas


def _row_from_matrix_row(row: pd.Series, env_name: str, description: str) -> dict:
    return {
        "environment": env_name, "description": description,
        "memory_limit_mb": row["memory_limit_mb"], "cpu_limit": row.get("cpu_limit"),
        "latency_p50_ms": row.get("latency_p50_ms"), "latency_p95_ms": row.get("latency_p95_ms"),
        "latency_p99_ms": row.get("latency_p99_ms"), "throughput_rps": row.get("throughput_rps"),
        "ram_mb": row.get("peak_memory_mb_docker_stats"), "startup_time_s": row.get("time_to_ready_s"),
        "equivalence_all_100pct": row.get("equivalence_passed"),
        "stability": "estable" if bool(row.get("ready_success")) and bool(row.get("streaming_success")) and bool(row.get("equivalence_passed")) else "fallo",
        "source": "docker_memory_limit_matrix.csv",
    }


def build() -> pd.DataFrame:
    filas = [_row_from_phase2_local_api(), _row_docker_unrestricted(), *_rows_from_memory_matrix()]
    df = pd.DataFrame(filas)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "edge_cloud_comparison.csv"
    df.to_csv(out, index=False)
    print(f"[final-comparison] -> {out}")
    print(df[["environment", "memory_limit_mb", "latency_p50_ms", "throughput_rps", "ram_mb", "equivalence_all_100pct", "stability"]].to_string(index=False))
    return df


if __name__ == "__main__":
    build()
