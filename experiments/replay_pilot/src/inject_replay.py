"""Construccion de los diccionarios de ataque 'replay' a partir del manifiesto congelado.

Sigue exactamente el mismo formato de diccionario `ataques` que el resto del proyecto usa
para constant_reduction/variable_reduction/linear_ramp/residual_bypass/total_bypass
(offset_global, duracion, y_tramo, attack_start, attack_end, family, parameter), de forma
que las funciones ya existentes y sin modificar (`predictor_causal_lags.puntuar_ataques`,
`optimizacion_histgb_periodic.puntuar_p_en_fold`) puedan reutilizarse literalmente.

La sustitucion es una copia exacta de los valores de origen -- sin escalado, suavizado,
interpolacion, ni ajuste de ningun tipo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loading import COLUMNA_OBJETIVO


def construir_ataques_replay(tabla_manifest: pd.DataFrame, serie_completa_pretest: pd.DataFrame,
                              pool_ini: pd.Timestamp) -> dict:
    """Un episodio por fila del manifiesto. offset_global es relativo a `pool_ini` (para ser
    compatible con `opt.puntuar_p_en_fold(..., particion_final_pool, pool_ini, ...)` y con
    `particion_pool_min_raw` extraida de `particion_final_pool`)."""
    ataques = {}
    valores = serie_completa_pretest[COLUMNA_OBJETIVO]
    for row in tabla_manifest.itertuples():
        dest_start, dest_end = pd.Timestamp(row.destination_start), pd.Timestamp(row.destination_end)
        source_start, source_end = pd.Timestamp(row.source_start), pd.Timestamp(row.source_end)
        dur = int(row.duration_minutes)

        offset_global = int((dest_start - pool_ini) / pd.Timedelta(minutes=1))
        y_tramo = valores.loc[source_start:source_end - pd.Timedelta(minutes=1)].to_numpy(dtype=np.float64)
        assert len(y_tramo) == dur, f"{row.episode_id}: origen con {len(y_tramo)} muestras, esperadas {dur}"

        # Comprobacion de igualdad exacta: el valor que se va a insertar en el
        # destino debe coincidir bit a bit con el valor limpio en el origen.
        clean_origin_check = valores.loc[source_start:source_end - pd.Timedelta(minutes=1)].to_numpy(dtype=np.float64)
        assert np.array_equal(y_tramo, clean_origin_check), f"{row.episode_id}: la copia no es exacta"

        ataques[row.episode_id] = {
            "offset_global": offset_global, "duracion": dur, "y_tramo": y_tramo,
            "attack_start": dest_start, "attack_end": dest_end,
            "family": "replay", "parameter": row.shift_type,
            "paired_destination_id": row.paired_destination_id, "shift_type": row.shift_type,
            "shift_minutes": int(row.shift_minutes), "source_start": source_start, "source_end": source_end,
        }
    return ataques


def verificar_copia_exacta(ataques: dict, serie_completa_pretest: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye, por episodio, la comprobacion muestra a muestra clean_value/replay_value/
    attacked_value con sus timestamps de origen y destino."""
    valores = serie_completa_pretest[COLUMNA_OBJETIVO]
    filas = []
    for episode_id, a in ataques.items():
        dest_ts = pd.date_range(a["attack_start"], periods=a["duracion"], freq="1min")
        src_ts = pd.date_range(a["source_start"], periods=a["duracion"], freq="1min")
        clean_dest_values = valores.reindex(dest_ts).to_numpy(dtype=np.float64)
        n_mismatch = int((np.abs(a["y_tramo"] - valores.reindex(src_ts).to_numpy(dtype=np.float64)) > 0).sum())
        filas.append({
            "episode_id": episode_id, "n_samples": a["duracion"], "n_exact_copy_mismatches": n_mismatch,
            "first_source_value": float(a["y_tramo"][0]), "last_source_value": float(a["y_tramo"][-1]),
            "first_dest_clean_value": float(clean_dest_values[0]), "last_dest_clean_value": float(clean_dest_values[-1]),
        })
    return pd.DataFrame(filas)
