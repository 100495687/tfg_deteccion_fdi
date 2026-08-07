"""Fase 4: congela `params_norm` de P en un artefacto de inferencia autonomo.

Inventario (inspeccion del codigo, no supuesto):
  - Funcion que construye params_norm: `src.normalization.ajustar_zscore(valores_train)`
    -> `{"media": float(np.mean(...)), "std": float(np.std(...))}`. Dos escalares Python
    `float` (float64 de numpy convertido a float nativo), sin arrays.
  - Quien la llama: `src.pipeline.preparar_datos(config)` hace
    `ajustar_zscore(particiones["train"][COLUMNA_OBJETIVO].to_numpy())` -- la media/std del
    tren (particion "train" de `src.splitting.split_temporal`), nunca de val/test.
  - `src.base_relative.cargar_ataques_y_modelo_360()` (la funcion que
    `core.model_loader.load_frozen_artifacts()` ya usa, sin modificar) llama a
    `preparar_datos` (que internamente llama a `src.data_loading.cargar_y_limpiar()`, que lee
    `data/raw/data_raw.csv`) para obtener `datos_p["params_norm"]`.
  - `episodes_master_val.csv` no participa en el calculo de params_norm: solo lo usa
    `reconstruir_ataques_exactos()` (dentro de la misma funcion) para reconstruir `ataques`/
    `master_csv`, que `model_loader.load_frozen_artifacts()` ya descarta
    (`_, datos_p, modelo_p, _, _ = br.cargar_ataques_y_modelo_360()`). Es decir: el camino
    actual paga el coste de leer y validar `episodes_master_val.csv` sin necesitarlo para
    params_norm -- un desperdicio que el `autonomous_loader` tambien elimina.
  - Determinismo: `ajustar_zscore` es `np.mean`/`np.std` sobre una particion fija y
    determinista de un CSV congelado -- sin aleatoriedad, sin `.fit()`, sin recalibracion.
    Confirmado empiricamente en este script comparando dos ejecuciones independientes de la
    ruta legacy.

Este script se ejecuta una sola vez, manualmente (no forma parte de la API ni de los tests
automaticos), y llama literalmente a la ruta legacy ya validada -- nunca recalcula nada nuevo,
nunca ajusta un modelo.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]  # deteccion_fraude_no_supervisada/
EDGE_DIR = BASE_DIR / "edge_deployment"
MODELS_DIR = BASE_DIR / "models"
FINAL_OR_DIR = MODELS_DIR / "final_or_pretest"
DATA_RAW = BASE_DIR / "data" / "raw" / "data_raw.csv"
EPISODES_MASTER_VAL = BASE_DIR / "results" / "tables" / "episodes_master_val.csv"
P_CHECKPOINT_PATH = MODELS_DIR / "tcn_ae_ventana360.pt"
P_THRESHOLD_PATH = FINAL_OR_DIR / "threshold_p_final.json"

OUT_ARTIFACT = EDGE_DIR / "models" / "params_norm_p.joblib"
OUT_METADATA = EDGE_DIR / "models" / "params_norm_p_metadata.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_params_norm_via_legacy_path() -> dict:
    """Llama literalmente a la ruta legacy ya validada (Fase 1/2, sin modificar) -- no
    reimplementa ajustar_zscore ni preparar_datos."""
    from src import base_relative as br

    _, datos_p, _modelo_p, _ataques, _master_csv = br.cargar_ataques_y_modelo_360()
    return datos_p["params_norm"]


def _confirm_determinism(p1: dict, p2: dict) -> bool:
    return p1["media"] == p2["media"] and p1["std"] == p2["std"]


def main() -> int:
    for p in (DATA_RAW, EPISODES_MASTER_VAL, P_CHECKPOINT_PATH, P_THRESHOLD_PATH):
        if not p.exists():
            print(f"FALTA {p} -- no se puede ejecutar la ruta legacy de congelacion")
            return 1

    print("[freeze] ejecutando la ruta legacy (1a vez)...")
    params_norm_1 = _load_params_norm_via_legacy_path()
    print("[freeze] ejecutando la ruta legacy (2a vez, para confirmar determinismo)...")
    params_norm_2 = _load_params_norm_via_legacy_path()
    determinista = _confirm_determinism(params_norm_1, params_norm_2)
    if not determinista:
        print(f"ERROR: params_norm NO es deterministico entre 2 llamadas: {params_norm_1} vs {params_norm_2}")
        return 1
    print(f"[freeze] params_norm = {params_norm_1} (determinista, confirmado 2/2)")

    for k, v in params_norm_1.items():
        if not isinstance(v, float):
            print(f"ERROR: campo {k} no es float nativo (es {type(v)}) -- no se congela sin revisar")
            return 1

    OUT_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(params_norm_1, OUT_ARTIFACT)
    print(f"[freeze] artefacto -> {OUT_ARTIFACT}")

    with open(P_THRESHOLD_PATH, encoding="utf-8") as f:
        threshold_p = float(json.load(f)["umbral_p"])

    metadata = {
        "format": "joblib (dict de 2 floats nativos de Python: 'media', 'std')",
        "version": 1,
        "keys": sorted(params_norm_1.keys()),
        "shapes": {k: [] for k in params_norm_1},  # escalares -- shape vacia
        "dtypes": {k: "python float (float64)" for k in params_norm_1},
        "values": params_norm_1,
        "sha256_artifact": _sha256(OUT_ARTIFACT),
        "source_csv_hashes": {
            "data_raw_csv": _sha256(DATA_RAW),
            "episodes_master_val_csv": _sha256(EPISODES_MASTER_VAL),
        },
        "note_episodes_master_val": "episodes_master_val.csv NO participa en el calculo de "
                                     "params_norm (solo lo usa reconstruir_ataques_exactos, "
                                     "cuyo resultado se descarta) -- se registra su hash solo "
                                     "por trazabilidad de la ejecucion legacy completa que "
                                     "genero este artefacto, no porque params_norm dependa de el.",
        "provenance_function": "src.base_relative.cargar_ataques_y_modelo_360 -> "
                                "src.pipeline.preparar_datos -> "
                                "src.normalization.ajustar_zscore(particiones['train'][COLUMNA_OBJETIVO])",
        "p_checkpoint_path": str(P_CHECKPOINT_PATH.relative_to(BASE_DIR)),
        "p_checkpoint_sha256": _sha256(P_CHECKPOINT_PATH),
        "p_threshold": threshold_p,
        "p_threshold_path": str(P_THRESHOLD_PATH.relative_to(BASE_DIR)),
        "deterministic_confirmed_2_of_2_runs": determinista,
        "no_fit_no_recalibration": "confirmado: ajustar_zscore es np.mean/np.std sobre una "
                                    "particion fija; obtener_modelo() carga el checkpoint "
                                    "existente via load_state_dict, nunca llama a modelo.fit() "
                                    "porque el checkpoint ya existe en disco (ver pipeline.py)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "edge_deployment/core/freeze_params_norm.py (Fase 4, ejecucion manual unica)",
    }
    with open(OUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[freeze] metadatos -> {OUT_METADATA}")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("values",)}, indent=2, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
