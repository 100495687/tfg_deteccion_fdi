"""Fase 3, seccion 1: ensambla un contexto Docker MINIMO en edge_deployment/docker_context/.

No usa el repositorio completo como contexto de build (evitaria excluir nada de forma
fiable). En su lugar, esta lista explicita decide -- archivo a archivo -- que entra en la
imagen. Todo lo demas (data/, datasets, notebooks, experiments/, results/, refit/, tests,
figuras, logs, .git, entornos virtuales, cachés, checkpoints no usados) queda fuera por
omision, no por filtrado.

Rastreo de imports real (no supuesto) que justifica el conjunto elegido: se importo
`edge_deployment.core.model_loader.load_frozen_artifacts()` mas los modulos de `core/` en un
proceso limpio y se inspecciono `sys.modules` -- ver docker_final_report.md, seccion de
contexto minimo, para el listado completo. Hallazgo relevante: `base_relative.py` (via
`experimento_stride.cargar_datos_y_modelo`) importa TODO `src/` transitivamente porque varios
modulos de analisis importan `matplotlib.pyplot`/`scipy.stats` a nivel de modulo aunque el
motor online nunca dibuja ni usa esos submodulos en tiempo de ejecucion -- por eso
`requirements-edge.txt` incluye matplotlib/scipy pese a que el servicio no genera figuras.

`core/period_selection.py` se excluye deliberadamente: solo lo usan los simuladores/clientes
de equivalencia (herramientas de host, fuera de la imagen), nunca el arranque de la API ni el
procesamiento de una lectura -- confirmado por grep (ningun archivo de `api/` lo importa).

Fase 4 (autonomia de artefactos): `data/raw/data_raw.csv` y
`results/tables/episodes_master_val.csv` YA NO son necesarios en ningun momento del ciclo de
vida del contenedor de produccion. Hasta Fase 3, el motor los necesitaba UNA vez, en el
arranque, para reconstruir `params_norm` del checkpoint P (via
`base_relative.cargar_ataques_y_modelo_360`). Fase 4 congela ese `params_norm` en
`edge_deployment/models/params_norm_p.joblib` (ver `core/freeze_params_norm.py`) y el
`autonomous_loader` (`core/model_loader.py::load_frozen_artifacts_autonomous`, usado por
defecto por `DetectorEngine()`) lo carga directamente -- el contenedor ya no monta ningun CSV.
La ruta legacy (`load_frozen_artifacts`, que SI abre esos CSV) se conserva solo para el
experimento de equivalencia legacy-vs-autonomo (ver `docker_tools/`), nunca se usa en el
Dockerfile de produccion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # deteccion_fraude_no_supervisada/
EDGE_DIR = BASE_DIR / "edge_deployment"
CONTEXT_DIR = EDGE_DIR / "docker_context"
MANIFEST_PATH = CONTEXT_DIR / "context_manifest.json"

# Paquetes de terceros REALMENTE importados al cargar los artefactos congelados + core/,
# trazados empiricamente con sys.modules (no copiados de requirements.txt a ciegas). Se fija
# la version instalada en ESTE entorno (el mismo usado para validar Fase 1/2) para que
# `requirements-edge.txt` sea reproducible, salvaguarda (1) del encargo: no se actualiza nada
# automaticamente para la imagen optimized.
RUNTIME_PACKAGES = [
    ("torch", "el checkpoint P (TCN-AE) es un modelo PyTorch"),
    ("numpy", "operaciones numericas en P y H"),
    ("pandas", "series temporales, ventanas 15 min, indices datetime"),
    ("scikit-learn", "modelo H (HistGradientBoostingRegressor) y utilidades (Ridge, StandardScaler)"),
    ("scipy", "importado a nivel de modulo por base_relative.py/fusion_p_histgb.py/predictor_causal_lags.py (scipy.stats) -- no se usa en el camino de inferencia online pero el import es incondicional"),
    ("matplotlib", "importado a nivel de modulo por base_relative.py/experimento_stride.py/evaluacion_final_retrospectiva_test.py/fusion_p_histgb.py/optimizacion_histgb_periodic.py -- el servicio nunca dibuja, pero el import es incondicional"),
    ("joblib", "carga de histgb_periodic_final.joblib y profile_train_final.joblib"),
    ("pyyaml", "windowing.cargar_config lee configs/base.yaml"),
    ("pyarrow", "importado transitivamente por pandas en este entorno"),
    ("fastapi", "microservicio HTTP (Fase 2)"),
    ("pydantic", "validacion de payloads (dependencia de fastapi, se fija explicitamente)"),
    ("uvicorn", "servidor ASGI, --workers 1"),
    ("starlette", "dependencia de fastapi, TestClient/routing (se fija explicitamente)"),
    ("psutil", "edge_deployment/api/routes.py lo usa en /metrics (memory_info().rss) -- SI se instala en la imagen"),
]
# httpx: NO se instala en la imagen (ningun archivo de api/ lo importa, confirmado por grep);
# solo lo usan los clientes/benchmarks de equivalencia ejecutados en el host.

# (origen relativo a BASE_DIR, destino relativo a CONTEXT_DIR). Directorios completos se
# expanden mas abajo con reglas explicitas (solo *.py o solo los ficheros nombrados).
SRC_DIR = BASE_DIR / "src"
INCLUDE_SINGLE_FILES = [
    (BASE_DIR / "configs" / "base.yaml", "configs/base.yaml"),
    (BASE_DIR / "models" / "tcn_ae_ventana360.pt", "models/tcn_ae_ventana360.pt"),
    (BASE_DIR / "models" / "final_or_pretest" / "histgb_periodic_final.joblib", "models/final_or_pretest/histgb_periodic_final.joblib"),
    (BASE_DIR / "models" / "final_or_pretest" / "profile_train_final.joblib", "models/final_or_pretest/profile_train_final.joblib"),
    (BASE_DIR / "models" / "final_or_pretest" / "features_h_final.json", "models/final_or_pretest/features_h_final.json"),
    (BASE_DIR / "models" / "final_or_pretest" / "hyperparameters_h_final.json", "models/final_or_pretest/hyperparameters_h_final.json"),
    (BASE_DIR / "models" / "final_or_pretest" / "threshold_h_final.json", "models/final_or_pretest/threshold_h_final.json"),
    (BASE_DIR / "models" / "final_or_pretest" / "threshold_p_final.json", "models/final_or_pretest/threshold_p_final.json"),
    (BASE_DIR / "models" / "auditoria_final_p_vs_or" / "configuracion_final_decision_pretest.json", "models/auditoria_final_p_vs_or/configuracion_final_decision_pretest.json"),
    (EDGE_DIR / "__init__.py", "edge_deployment/__init__.py"),
    (EDGE_DIR / "config" / "edge_engine.yaml", "edge_deployment/config/edge_engine.yaml"),
    (EDGE_DIR / "config" / "api.yaml", "edge_deployment/config/api.yaml"),
    (EDGE_DIR / "docker_entrypoint.sh", "docker_entrypoint.sh"),
    # Fase 4: artefacto autonomo (params_norm_p) + manifiesto v2 (referencia congelada de
    # hashes que el autonomous_loader compara en cada arranque). Sin estos dos, el contenedor
    # no puede validar integridad ni cargar P sin CSV -- son ahora artefactos de PRODUCCION,
    # no solo de trazabilidad.
    (EDGE_DIR / "models" / "params_norm_p.joblib", "edge_deployment/models/params_norm_p.joblib"),
    (EDGE_DIR / "models" / "params_norm_p_metadata.json", "edge_deployment/models/params_norm_p_metadata.json"),
    (EDGE_DIR / "models" / "artifact_manifest_v2.json", "edge_deployment/models/artifact_manifest_v2.json"),
]
CORE_EXCLUDE = {
    "period_selection.py",  # solo usado por simuladores/clientes de host (seccion 4), nunca en tiempo de arranque/peticion -- confirmado por grep en api/*.py
    "freeze_params_norm.py",  # Fase 4: herramienta de congelacion MANUAL (necesita data_raw.csv) -- nunca se ejecuta en produccion/Docker
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tracked(src: Path, dest_rel: str, manifest_rows: list) -> None:
    dest = CONTEXT_DIR / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    manifest_rows.append({
        "dest_path": dest_rel, "source_path": str(src.relative_to(BASE_DIR)).replace("\\", "/"),
        "size_bytes": dest.stat().st_size, "sha256": _sha256(dest),
    })


def _resolved_version(pkg: str) -> str | None:
    try:
        return pkg_version(pkg)
    except PackageNotFoundError:
        return None


def build_context(clean: bool = True) -> dict:
    if clean and CONTEXT_DIR.exists():
        shutil.rmtree(CONTEXT_DIR)
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    for src, dest_rel in INCLUDE_SINGLE_FILES:
        if not src.exists():
            raise FileNotFoundError(f"artefacto requerido para el contexto Docker no encontrado: {src}")
        _copy_tracked(src, dest_rel, manifest_rows)

    for py in sorted(SRC_DIR.glob("*.py")):
        _copy_tracked(py, f"src/{py.name}", manifest_rows)
    models_pkg_dir = SRC_DIR / "models"
    for py in sorted(models_pkg_dir.glob("*.py")):
        _copy_tracked(py, f"src/models/{py.name}", manifest_rows)

    for py in sorted((EDGE_DIR / "api").glob("*.py")):
        _copy_tracked(py, f"edge_deployment/api/{py.name}", manifest_rows)
    for py in sorted((EDGE_DIR / "core").glob("*.py")):
        if py.name in CORE_EXCLUDE:
            continue
        _copy_tracked(py, f"edge_deployment/core/{py.name}", manifest_rows)
    # Fase 5: capa anti-replay OPCIONAL -- solo modulos stdlib (hmac/hashlib/json/dataclasses),
    # sin dependencias nuevas. Eliminar este bloque + edge_deployment/security/*.py de la
    # imagen basta para volver a la imagen de Fase 4 (ver seccion 5 del encargo).
    for py in sorted((EDGE_DIR / "security").glob("*.py")):
        _copy_tracked(py, f"edge_deployment/security/{py.name}", manifest_rows)

    req_src = EDGE_DIR / "requirements-edge.txt"
    _copy_tracked(req_src, "requirements-edge.txt", manifest_rows)
    dockerignore_src = EDGE_DIR / ".dockerignore"
    if dockerignore_src.exists():
        _copy_tracked(dockerignore_src, ".dockerignore", manifest_rows)

    runtime_mounts_doc = CONTEXT_DIR / "RUNTIME_MOUNTS.md"
    runtime_mounts_doc.write_text(
        "# Montajes de solo lectura en `docker run`\n\n"
        "**Fase 4 (autonomia de artefactos): ya no hace falta montar ningun dataset.** El "
        "`autonomous_loader` (por defecto en `DetectorEngine()`) carga `params_norm_p.joblib` "
        "(congelado, horneado en la imagen) y nunca abre `data/raw/data_raw.csv` ni "
        "`results/tables/episodes_master_val.csv`. El contenedor alcanza `/ready` con:\n\n"
        "```\n"
        "docker run --rm -p 8000:8000 fdia-edge:optimized\n"
        "```\n\n"
        "sin ningun `-v` de datos experimentales -- ver `phase4_final_report.md`.\n\n"
        "## Nota historica (Fase 3, ya no aplica al camino de produccion)\n\n"
        "Hasta Fase 3, el `legacy_loader` (que reconstruye `params_norm` desde los CSV en cada "
        "arranque, conservado solo para el experimento de equivalencia legacy-vs-autonomo, "
        "nunca usado en el Dockerfile de produccion) necesitaba montar:\n\n"
        "- `data/raw/data_raw.csv`  ->  `/app/data/raw/data_raw.csv`\n"
        "- `results/tables/episodes_master_val.csv`  ->  `/app/results/tables/episodes_master_val.csv`\n",
        encoding="utf-8",
    )

    version_lock = {pkg: _resolved_version(pkg) for pkg, _reason in RUNTIME_PACKAGES}

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version_host": sys.version,
        "n_files": len(manifest_rows),
        "total_size_bytes": sum(r["size_bytes"] for r in manifest_rows),
        "files": sorted(manifest_rows, key=lambda r: r["dest_path"]),
        "runtime_packages": [{"package": p, "reason": r, "version_locked": version_lock.get(p)} for p, r in RUNTIME_PACKAGES],
        "excluded_by_design": ["data/", "results/ (salvo el bootstrap del contexto: no aplica)", "notebooks/*.ipynb",
                                "experiments/", "refit/", "edge_deployment/tests/", "edge_deployment/results/",
                                "edge_deployment/benchmarks/", "edge_deployment/clients/", "edge_deployment/simulators/",
                                ".git/", "__pycache__/", "*.pyc", "entornos virtuales (.venv/venv/)",
                                "checkpoints no usados: tcn_ae_ventana120.pt/720.pt/1440.pt, models/predictor_causal_lags*, "
                                "models/calibracion_temporal_h, models/robustez_temporal_p, models/tcn_masked_m60, models/final_retrospective_test"],
        "excluded_but_required_at_runtime_via_readonly_mount": [],
        "note_fase4_autonomia": "Hasta Fase 3 esta lista contenia data/raw/data_raw.csv y "
                                 "results/tables/episodes_master_val.csv (necesarios para el "
                                 "legacy_loader). Fase 4 congela params_norm_p.joblib y el "
                                 "autonomous_loader (por defecto en produccion) no abre ningun "
                                 "CSV -- el contenedor ya no necesita montar ningun dataset.",
        "core_period_selection_excluded_reason": "solo usado por simuladores/clientes de equivalencia (host), nunca en el arranque ni en el procesamiento de una lectura -- confirmado por grep en edge_deployment/api/*.py",
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    manifest = build_context(clean=not args.no_clean)
    print(f"[docker-context] {manifest['n_files']} archivos, "
          f"{manifest['total_size_bytes'] / 1e6:.2f} MB -> {CONTEXT_DIR}")
    print(f"[docker-context] manifiesto -> {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
