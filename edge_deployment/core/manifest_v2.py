"""Fase 4: manifiesto de artefactos v2 -- a diferencia de `model_loader.py` (Fase 1,
sin modificar aqui), que solo registra hashes SHA-256 sin compararlos contra nada, este modulo
implementa la comparacion real `hash_calculado == hash_esperado` que Fase 3 documento como
pendiente. No sustituye a `model_loader.py` (que se sigue usando para la ruta legacy de
equivalencia) -- es un modulo nuevo, separado, para el camino de produccion.

`artifact_manifest_v2.json` se genera una vez (`build_artifact_manifest_v2()`, ejecucion manual,
igual que `freeze_params_norm.py`) y se versiona junto al codigo como la referencia congelada.
En cada arranque de la API, `validate_manifest_v2()` recalcula los hashes actuales y los compara
contra esa referencia -- si no coinciden (fichero ausente, corrupto, o distinto), el resultado
marca ese artefacto como invalido, sin lanzar excepcion y sin desertizar/cargar nada todavia
(el hash se compara antes de intentar `joblib.load`/`torch.load`, evitando excepciones de
deserializacion sobre datos corruptos).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # deteccion_fraude_no_supervisada/
EDGE_DIR = BASE_DIR / "edge_deployment"
MODELS_DIR = BASE_DIR / "models"
FINAL_OR_DIR = MODELS_DIR / "final_or_pretest"
CONFIGS_DIR = BASE_DIR / "configs"

MANIFEST_V2_PATH = EDGE_DIR / "models" / "artifact_manifest_v2.json"


class ArtifactIntegrityError(RuntimeError):
    """Al menos un artefacto falta o su SHA-256 no coincide con `artifact_manifest_v2.json`.
    Se lanza antes de deserializar ningun artefacto (solo se calculan hashes de bytes crudos),
    para que un fichero corrupto nunca llegue a `joblib.load`/`torch.load`."""

    def __init__(self, integrity_result: dict) -> None:
        self.integrity_result = integrity_result
        super().__init__(
            f"Fallo de integridad de artefactos: {integrity_result['n_missing']} ausentes "
            f"{integrity_result['missing']}, {integrity_result['n_mismatched']} con hash "
            f"distinto al esperado {[m['artifact'] for m in integrity_result['mismatched']]}. "
            f"No se ha cargado ni deserializado ningun artefacto."
        )

# nombre_logico -> (ruta relativa a BASE_DIR, requerido)
ARTIFACT_PATHS: dict[str, Path] = {
    "model_P_checkpoint": MODELS_DIR / "tcn_ae_ventana360.pt",
    "model_H": FINAL_OR_DIR / "histgb_periodic_final.joblib",
    "profile_H": FINAL_OR_DIR / "profile_train_final.joblib",
    "params_norm_P": EDGE_DIR / "models" / "params_norm_p.joblib",
    "threshold_P_json": FINAL_OR_DIR / "threshold_p_final.json",
    "threshold_H_json": FINAL_OR_DIR / "threshold_h_final.json",
    "features_H_json": FINAL_OR_DIR / "features_h_final.json",
    "base_config_yaml": CONFIGS_DIR / "base.yaml",
}
ALL_REQUIRED = set(ARTIFACT_PATHS)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_artifact_manifest_v2() -> dict:
    """Genera/actualiza la referencia congelada. Ejecucion manual unicamente (igual que
    freeze_params_norm.py) -- nunca se llama desde la API ni desde el arranque del
    contenedor. Recongelar el manifiesto tras cambiar deliberadamente un artefacto es una
    accion explicita del desarrollador, no automatica."""
    with open(FINAL_OR_DIR / "threshold_h_final.json", encoding="utf-8") as f:
        threshold_h = float(json.load(f)["umbral_h"])
    with open(FINAL_OR_DIR / "threshold_p_final.json", encoding="utf-8") as f:
        threshold_p = float(json.load(f)["umbral_p"])

    artifacts = {}
    for name, path in ARTIFACT_PATHS.items():
        sha = _sha256(path)
        if sha is None:
            raise FileNotFoundError(f"artefacto requerido no encontrado al congelar el manifiesto v2: {path}")
        artifacts[name] = {"path": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
                            "sha256": sha, "required": True}

    manifest = {
        "manifest_version": 2,
        "architecture": "P_OR_H",
        "threshold_P": threshold_p,
        "threshold_H": threshold_h,
        "artifacts": artifacts,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "edge_deployment/core/manifest_v2.py::build_artifact_manifest_v2 (ejecucion manual)",
        "note": "referencia congelada -- la API compara hash_calculado==hash_esperado contra "
                "este fichero en cada arranque (validate_manifest_v2), nunca lo regenera sola.",
    }
    MANIFEST_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def validate_manifest_v2(manifest_path: Path = MANIFEST_V2_PATH) -> dict:
    """Comparacion real de integridad: hash_calculado == hash_esperado, para cada artefacto
    del manifiesto congelado. No deserializa ningun artefacto -- solo lee bytes y calcula
    SHA-256, por eso un artefacto corrupto (bytes alterados que romperian `joblib.load`) se
    detecta antes de intentar cargarlo, nunca lanza una excepcion de deserializacion."""
    if not manifest_path.exists():
        return {"integrity_valid": False, "n_checked": 0, "n_missing": 0, "n_mismatched": 0,
                "missing": [], "mismatched": [], "error": f"manifiesto v2 no encontrado: {manifest_path}"}

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    missing, mismatched, ok = [], [], []
    for name, entry in manifest["artifacts"].items():
        path = BASE_DIR / entry["path"]
        actual = _sha256(path)
        if actual is None:
            missing.append(name)
        elif actual != entry["sha256"]:
            mismatched.append({"artifact": name, "expected_sha256": entry["sha256"], "actual_sha256": actual})
        else:
            ok.append(name)

    return {
        "integrity_valid": len(missing) == 0 and len(mismatched) == 0,
        "n_checked": len(manifest["artifacts"]), "n_ok": len(ok), "n_missing": len(missing),
        "n_mismatched": len(mismatched), "missing": missing, "mismatched": mismatched,
        "manifest_version": manifest.get("manifest_version"),
        "manifest_generated_utc": manifest.get("generated_utc"),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="(re)genera el manifiesto v2 (accion manual explicita)")
    parser.add_argument("--validate", action="store_true", help="valida el manifiesto v2 existente")
    args = parser.parse_args()

    if args.build:
        m = build_artifact_manifest_v2()
        print(f"[manifest-v2] generado -> {MANIFEST_V2_PATH}")
        print(json.dumps({k: v for k, v in m.items() if k != "artifacts"}, indent=2, ensure_ascii=False))
        for name, e in m["artifacts"].items():
            print(f"  {name}: {e['sha256'][:16]}... ({e['path']})")
        return 0
    if args.validate:
        r = validate_manifest_v2()
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0 if r["integrity_valid"] else 1
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
