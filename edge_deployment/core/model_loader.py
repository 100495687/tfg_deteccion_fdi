"""Inventario, localizacion y carga de los artefactos congelados que el motor online
necesita. No copia artefactos grandes dentro de edge_deployment/:
apunta a las rutas originales de solo lectura (models/, configs/). Nunca llama a `fit`,
nunca recalibra, nunca modifica un checkpoint.

Reutiliza sin modificar:
  - src.base_relative.cargar_ataques_y_modelo_360 (checkpoint P + normalizacion, solo
    inferencia -- exactamente la misma via ya usada por experiments/replay_pilot)
  - src.optimizacion_histgb_periodic.PERIODIC_BASE_FEATURES (verificacion del orden de
    features de H)

Si falta un artefacto necesario, `_require` detiene la ejecucion explicando que falta,
donde se esperaba y por que hace falta -- nunca reconstruye ni reentrena automaticamente.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]  # deteccion_fraude_no_supervisada/
EDGE_DIR = BASE_DIR / "edge_deployment"
MODELS_DIR = BASE_DIR / "models"
FINAL_OR_DIR = MODELS_DIR / "final_or_pretest"
CONFIGS_DIR = BASE_DIR / "configs"
DATA_RAW = BASE_DIR / "data" / "raw" / "data_raw.csv"

P_CHECKPOINT_PATH = MODELS_DIR / "tcn_ae_ventana360.pt"
P_CONFIG_PATH = CONFIGS_DIR / "base.yaml"
H_MODEL_PATH = FINAL_OR_DIR / "histgb_periodic_final.joblib"
H_PROFILE_PATH = FINAL_OR_DIR / "profile_train_final.joblib"
H_FEATURES_PATH = FINAL_OR_DIR / "features_h_final.json"
H_HYPERPARAMS_PATH = FINAL_OR_DIR / "hyperparameters_h_final.json"
H_THRESHOLD_PATH = FINAL_OR_DIR / "threshold_h_final.json"
P_THRESHOLD_PATH = FINAL_OR_DIR / "threshold_p_final.json"
DECISION_CONFIG_PATH = MODELS_DIR / "auditoria_final_p_vs_or" / "configuracion_final_decision_pretest.json"
PARAMS_NORM_P_PATH = EDGE_DIR / "models" / "params_norm_p.joblib"  # Fase 4: artefacto congelado (freeze_params_norm.py)

INVENTORY_PATH = EDGE_DIR / "results" / "tables" / "input_artifacts_inventory.csv"
MANIFEST_PATH = EDGE_DIR / "models" / "artifact_manifest.json"


class MissingArtifactError(RuntimeError):
    pass


def _require(path: Path, purpose: str, needed_by: str) -> None:
    if not path.exists():
        raise MissingArtifactError(
            f"Artefacto necesario no encontrado.\n"
            f"  artefacto faltante : {path.name}\n"
            f"  ruta esperada      : {path}\n"
            f"  funcion que lo necesita : {needed_by}\n"
            f"  razon              : {purpose}\n"
            f"No se reconstruye ni reentrena automaticamente ningun artefacto faltante. "
            f"Ejecuta primero el modulo que lo genera (ver src/congelacion_final_or_pretest.py)."
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ==========================================================================================
# Inventario programatico de artefactos
# ==========================================================================================

def _fila_artefacto(name: str, path: Path, tipo: str, purpose: str, model_branch: str, split: str,
                     required: bool, mutable: bool, contains_test: bool, selected: bool, reason: str) -> dict:
    found = path.exists() if path is not None else False
    return {
        "artifact_name": name, "path": str(path.relative_to(BASE_DIR)) if path else "", "type": tipo,
        "purpose": purpose, "model_branch": model_branch, "split": split,
        "date_start": "", "date_end": "",
        "size_bytes": path.stat().st_size if found and path.is_file() else 0,
        "sha256": _sha256(path) if found and path.is_file() else "",
        "required": required, "found": found, "mutable": mutable,
        "contains_test_information": contains_test, "selected": selected, "selection_reason": reason,
        "validation_status": "OK" if (found or not required) else "MISSING",
    }


def build_input_artifacts_inventory() -> pd.DataFrame:
    filas = [
        _fila_artefacto("tcn_ae_checkpoint_360", P_CHECKPOINT_PATH, "model_checkpoint",
                         "checkpoint TCN-AE congelado (P, ventana=360/stride=60, relative_kw)",
                         "P", "train", True, False, False, True,
                         "checkpoint congelado usado por toda la arquitectura P_OR_H ya seleccionada"),
        _fila_artefacto("base_config_yaml", P_CONFIG_PATH, "config",
                         "arquitectura TCN-AE (canales/kernel/dilataciones/pool_size) necesaria para reconstruir la clase antes de cargar el state_dict",
                         "P", "n/a", True, False, False, True, "config congelada, no se modifica"),
        _fila_artefacto("histgb_periodic_final", H_MODEL_PATH, "model_checkpoint",
                         "modelo HistGB_PERIODIC final (H), entrenado exclusivamente sobre FINAL_TRAIN",
                         "H", "final_train", True, False, False, True, "checkpoint final congelado por congelacion_final_or_pretest.py"),
        _fila_artefacto("profile_train_final", H_PROFILE_PATH, "profile_table",
                         "perfil historico (mediana/MAD/n_obs por es_finde+slot_15min), calculado solo con FINAL_TRAIN",
                         "H", "final_train", True, False, False, True, "perfil congelado, necesario para las features periodic de H"),
        _fila_artefacto("features_h_final", H_FEATURES_PATH, "config",
                         "orden exacto de columnas de features de H", "H", "n/a", True, False, False, True,
                         "debe coincidir con optimizacion_histgb_periodic.PERIODIC_BASE_FEATURES"),
        _fila_artefacto("hyperparameters_h_final", H_HYPERPARAMS_PATH, "config",
                         "hiperparametros de HistGB_PERIODIC (solo informativo, el modelo ya esta entrenado)",
                         "H", "n/a", False, False, False, False, "informativo, no se usa para reinstanciar el modelo"),
        _fila_artefacto("threshold_h_final", H_THRESHOLD_PATH, "threshold",
                         "umbral congelado de H (H_MULTIWINDOW_180)", "H", "n/a", True, False, False, True,
                         "umbral operativo congelado, no se recalibra"),
        _fila_artefacto("threshold_p_final", P_THRESHOLD_PATH, "threshold",
                         "umbral congelado de P (P_MULTI_SEASON)", "P", "n/a", True, False, False, True,
                         "umbral operativo congelado, no se recalibra"),
        _fila_artefacto("decision_config_pretest", DECISION_CONFIG_PATH, "config",
                         "decision metodologica ya cerrada (OR_PMULTI_HMULTIWINDOW), solo para verificar consistencia",
                         "OR", "n/a", True, False, False, True, "verificacion de que la arquitectura congelada sigue siendo la vigente"),
        _fila_artefacto("data_raw_csv", DATA_RAW, "raw_data",
                         "serie completa 1-min (para reconstruir periodo pre-test, bootstrap y streaming)",
                         "n/a", "train+val", True, False, False, True,
                         "fuente unica de datos; el motor nunca lee test salvo el timestamp de arranque (igual que el pipeline offline)"),
    ]
    funciones = [
        ("windowing.ventanear", "src/windowing.py", "P: construccion de ventanas 360/stride 60"),
        ("normalization.aplicar_zscore", "src/normalization.py", "P: normalizacion z-score (ajustada solo en train)"),
        ("base_relative.calcular_relative_kw", "src/base_relative.py", "P: formula del score relative_kw"),
        ("evaluacion_final_retrospectiva_test.puntuar_p_limpio", "src/evaluacion_final_retrospectiva_test.py", "P: envoltura ventana+normalizacion+reconstruccion+score, reutilizada literalmente"),
        ("predictor_causal_lags.construir_serie_15min", "src/predictor_causal_lags.py", "H: agregacion causal a 15 min"),
        ("predictor_causal_lags.construir_features_kw", "src/predictor_causal_lags.py", "H: lags/rolling causales"),
        ("predictor_causal_lags._calendario", "src/predictor_causal_lags.py", "H: variables de calendario"),
        ("predictor_causal_lags.calcular_score", "src/predictor_causal_lags.py", "H: formula de causal_relative_kw"),
        ("congelacion_final_or_pretest.predecir_pool_limpio", "src/congelacion_final_or_pretest.py", "H: referencia offline (batch) sobre el pool"),
        ("congelacion_final_or_pretest.preparar_periodos", "src/congelacion_final_or_pretest.py", "seleccion de FINAL_TRAIN/FINAL_CAL_POOL_180"),
        ("fusion_p_histgb.estado_ffill", "src/fusion_p_histgb.py", "propagacion causal forward-fill de P"),
        ("fusion_p_histgb.construir_estados_episodio", "src/fusion_p_histgb.py", "referencia de fusion OR (solo para tests de equivalencia conceptual)"),
    ]
    for nombre, ruta, proposito in funciones:
        filas.append({
            "artifact_name": nombre, "path": ruta, "type": "function", "purpose": proposito,
            "model_branch": "n/a", "split": "n/a", "date_start": "", "date_end": "", "size_bytes": 0, "sha256": "",
            "required": True, "found": (BASE_DIR / ruta).exists(), "mutable": False,
            "contains_test_information": False, "selected": True, "selection_reason": "reutilizada literalmente, sin modificar",
            "validation_status": "OK" if (BASE_DIR / ruta).exists() else "MISSING",
        })
    df = pd.DataFrame(filas)
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INVENTORY_PATH, index=False)
    return df


# ==========================================================================================
# Manifiesto de modelos -- apunta a las rutas originales, no copia nada todavia
# ==========================================================================================

def build_artifact_manifest() -> dict:
    for path, purpose, needed_by in [
        (P_CHECKPOINT_PATH, "checkpoint P", "p_online"), (H_MODEL_PATH, "modelo H", "h_online"),
        (H_PROFILE_PATH, "perfil de H", "h_online"), (H_THRESHOLD_PATH, "umbral H", "fusion_online"),
        (P_THRESHOLD_PATH, "umbral P", "fusion_online"), (H_FEATURES_PATH, "orden de features H", "h_online"),
    ]:
        _require(path, purpose, needed_by)

    with open(H_THRESHOLD_PATH, encoding="utf-8") as f:
        threshold_h = float(json.load(f)["umbral_h"])
    with open(P_THRESHOLD_PATH, encoding="utf-8") as f:
        threshold_p = float(json.load(f)["umbral_p"])

    manifiesto = {
        "architecture": "P_OR_H",
        "model_P_path": str(P_CHECKPOINT_PATH.relative_to(BASE_DIR)),
        "model_H_path": str(H_MODEL_PATH.relative_to(BASE_DIR)),
        "profiles_path": str(H_PROFILE_PATH.relative_to(BASE_DIR)),
        "config_paths": {
            "p_architecture": str(P_CONFIG_PATH.relative_to(BASE_DIR)),
            "h_features_order": str(H_FEATURES_PATH.relative_to(BASE_DIR)),
            "h_hyperparameters": str(H_HYPERPARAMS_PATH.relative_to(BASE_DIR)),
        },
        "threshold_P": threshold_p, "threshold_H": threshold_h,
        "hashes": {
            "model_P": _sha256(P_CHECKPOINT_PATH), "model_H": _sha256(H_MODEL_PATH),
            "profile_H": _sha256(H_PROFILE_PATH), "threshold_H_json": _sha256(H_THRESHOLD_PATH),
            "threshold_P_json": _sha256(P_THRESHOLD_PATH),
        },
        "configuration_hashes": {"base_yaml": _sha256(P_CONFIG_PATH) if P_CONFIG_PATH.exists() else None},
        "training_enabled": False, "calibration_enabled": False,
        "artifact_version": 1,
        "reference_commit_or_code": "congelacion_final_or_pretest.py (models/final_or_pretest/)",
        "artifacts_copied_into_edge_deployment": False,
        "note": "Fase de equivalencia online/offline: los artefactos se referencian por ruta relativa de "
                "solo lectura. La copia fisica para la imagen Docker se hara en una fase posterior.",
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)
    return manifiesto


# ==========================================================================================
# Carga en memoria (solo inferencia)
# ==========================================================================================
# Fase 4: dos caminos de carga, misma forma de salida (mismo dict de 7 claves).
#
#   - `load_frozen_artifacts` ("legacy_loader"): reutiliza literalmente
#     `base_relative.cargar_ataques_y_modelo_360` (Fase 1). Reconstruye params_norm de P leyendo
#     data_raw.csv + episodes_master_val.csv en cada llamada. Se mantiene sin modificar, solo
#     para el experimento de equivalencia legacy-vs-autonomo -- ya no es el camino
#     de produccion.
#   - `load_frozen_artifacts_autonomous` ("autonomous_loader"): carga params_norm_p desde el
#     artefacto congelado (`freeze_params_norm.py`), nunca abre data_raw.csv ni
#     episodes_master_val.csv, y valida SHA-256 real (hash_calculado==hash_esperado via
#     `manifest_v2.validate_manifest_v2`) antes de deserializar nada. Es el camino de
#     produccion (API/Docker).
#
# H se carga igual en los dos caminos (`_load_h_artifacts`, sin duplicar).

def _load_h_artifacts() -> dict:
    for path, purpose in [(H_MODEL_PATH, "modelo H"), (H_PROFILE_PATH, "perfil de H"),
                           (H_THRESHOLD_PATH, "umbral H"), (P_THRESHOLD_PATH, "umbral P"),
                           (H_FEATURES_PATH, "orden de features H")]:
        _require(path, purpose, "_load_h_artifacts")

    from src import optimizacion_histgb_periodic as opt

    modelo_h = joblib.load(H_MODEL_PATH)
    perfil_h = joblib.load(H_PROFILE_PATH)

    with open(H_THRESHOLD_PATH, encoding="utf-8") as f:
        threshold_h = float(json.load(f)["umbral_h"])
    with open(P_THRESHOLD_PATH, encoding="utf-8") as f:
        threshold_p = float(json.load(f)["umbral_p"])
    with open(H_FEATURES_PATH, encoding="utf-8") as f:
        features_h_orden = json.load(f)["features_orden"]

    if features_h_orden != opt.PERIODIC_BASE_FEATURES:
        raise RuntimeError("el orden de features de H congelado no coincide con opt.PERIODIC_BASE_FEATURES -- "
                            "posible desincronizacion entre el artefacto congelado y el codigo actual")
    if int(modelo_h.n_features_in_) != len(features_h_orden):
        raise RuntimeError("el numero de features del modelo H cargado no coincide con features_h_orden")

    return {"modelo_h": modelo_h, "perfil_h": perfil_h, "threshold_h": threshold_h,
            "threshold_p": threshold_p, "features_h_orden": features_h_orden}


def load_frozen_artifacts() -> dict:
    """legacy_loader (Fase 1, sin modificar): reconstruye params_norm de P leyendo
    data_raw.csv + episodes_master_val.csv via base_relative.cargar_ataques_y_modelo_360.
    Mantenido solo para el experimento de equivalencia de Fase 4 -- no es el
    camino de produccion desde Fase 4 (ver load_frozen_artifacts_autonomous)."""
    _require(P_CHECKPOINT_PATH, "checkpoint P", "load_frozen_artifacts")
    h = _load_h_artifacts()

    from src import base_relative as br

    _, datos_p, modelo_p, _, _ = br.cargar_ataques_y_modelo_360()  # checkpoint existente, solo inferencia
    params_norm_p = datos_p["params_norm"]

    return {"modelo_p": modelo_p, "params_norm_p": params_norm_p, "threshold_p": h["threshold_p"],
            "modelo_h": h["modelo_h"], "perfil_h": h["perfil_h"], "threshold_h": h["threshold_h"],
            "features_h_orden": h["features_h_orden"]}


def load_frozen_artifacts_autonomous() -> dict:
    """autonomous_loader (Fase 4): camino de produccion. Nunca abre data_raw.csv ni
    episodes_master_val.csv -- carga params_norm_p desde el artefacto congelado
    (edge_deployment/models/params_norm_p.joblib, ver freeze_params_norm.py) y reconstruye el
    modelo P via `pipeline.obtener_modelo(config, ventanas_norm={})`: como el checkpoint ya
    existe en disco, esa funcion (Fase 1, sin modificar) toma la rama `load_state_dict` y
    nunca evalua `ventanas_norm` -- pasar un dict vacio es seguro y no reimplementa la
    reconstruccion del modelo. Antes de cargar/deserializar nada, valida SHA-256 real de
    todos los artefactos contra `artifact_manifest_v2.json` (manifest_v2.validate_manifest_v2)
    -- un artefacto ausente o con el hash distinto lanza `ArtifactIntegrityError` antes de
    llamar a `joblib.load`/`torch.load`, nunca despues."""
    from edge_deployment.core import manifest_v2

    integrity = manifest_v2.validate_manifest_v2()
    if not integrity["integrity_valid"]:
        raise manifest_v2.ArtifactIntegrityError(integrity)

    from src import windowing as wnd
    from src.pipeline import obtener_modelo

    config = wnd.cargar_config()
    modelo_p = obtener_modelo(config, ventanas_norm={})  # checkpoint existente -> nunca entra en .fit()
    params_norm_p = joblib.load(PARAMS_NORM_P_PATH)

    h = _load_h_artifacts()
    return {"modelo_p": modelo_p, "params_norm_p": params_norm_p, "threshold_p": h["threshold_p"],
            "modelo_h": h["modelo_h"], "perfil_h": h["perfil_h"], "threshold_h": h["threshold_h"],
            "features_h_orden": h["features_h_orden"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--validate-artifacts", action="store_true")
    args = parser.parse_args()

    if args.inventory_only:
        df = build_input_artifacts_inventory()
        print(f"Inventario: {len(df)} artefactos -> {INVENTORY_PATH}")
        faltantes = df[~df["found"] & df["required"]]
        if len(faltantes):
            print(f"FALTAN {len(faltantes)} artefactos requeridos:\n{faltantes[['artifact_name', 'path']].to_string(index=False)}")
            return 1
        return 0

    if args.validate_artifacts:
        try:
            build_input_artifacts_inventory()
            manifiesto = build_artifact_manifest()
            load_frozen_artifacts()
        except MissingArtifactError as e:
            print(str(e))
            return 1
        print(f"OK -- artefactos validados y cargados. Manifiesto en {MANIFEST_PATH}")
        print(json.dumps({k: v for k, v in manifiesto.items() if k not in ("hashes", "configuration_hashes")}, indent=2, ensure_ascii=False))
        return 0

    build_input_artifacts_inventory()
    build_artifact_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
