"""Estado por contador. Estructuras de tamano acotado (deque con maxlen):
nunca se conserva indefinidamente el historial completo de lecturas -- solo el contexto
maximo que P y H necesitan, mas un margen de seguridad.

Contexto minimo (`required_context`):
  - P: 360 minutos de lecturas de streaming (nunca del bootstrap -- ver period_selection.py
    para la justificacion de por que P no usa contexto previo al inicio del streaming).
  - H: 672 bins de 15 min (7 dias) + 1 bin objetivo = 673 bins como minimo. El bootstrap
    aporta exactamente 682 bins (672 + 10 de margen, identico a
    congelacion_final_or_pretest.predecir_pool_limpio) para no operar al limite exacto.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

MAX_LAG_H_BINS = 672          # pcl.MAX_LAG -- 7 dias a 15 min
H_BOOTSTRAP_MARGIN_BINS = 10  # identico margen a predecir_pool_limpio
H_HISTORY_MAXLEN = MAX_LAG_H_BINS + H_BOOTSTRAP_MARGIN_BINS + 20  # margen de seguridad adicional

P_WINDOW_MIN = 360
P_STRIDE_MIN = 60
BUFFER_1MIN_MAXLEN = P_WINDOW_MIN + 60  # margen de seguridad

RESOLUCION_H_MIN = 15


def required_context() -> dict:
    return {
        "p_context_minutes": P_WINDOW_MIN,
        "h_context_minutes": MAX_LAG_H_BINS * RESOLUCION_H_MIN,
        "h_context_bins": MAX_LAG_H_BINS,
        "global_context_minutes": MAX_LAG_H_BINS * RESOLUCION_H_MIN,
        "bootstrap_margin_bins": H_BOOTSTRAP_MARGIN_BINS,
        "reason_p": "ventana de reconstruccion del TCN-AE (windowing.ventanear, tamano_ventana_min=360); "
                    "P NO usa contexto anterior al inicio del streaming (ver period_selection.py)",
        "reason_h": f"lag_672 en predictor_causal_lags.TODOS_LOS_LAGS = 672 bins de 15 min = 7 dias "
                    f"(pcl.MAX_LAG); +{H_BOOTSTRAP_MARGIN_BINS} bins de margen, identico a "
                    f"congelacion_final_or_pretest.predecir_pool_limpio",
    }


@dataclass
class Bin15Min:
    ts_start: object
    ts_end: object
    kw: float


@dataclass
class DetectorState:
    meter_id: str

    bootstrapped: bool = False
    bootstrap_end_timestamp: object = None  # ultimo timestamp incluido en el bootstrap (< stream_start)

    last_accepted_timestamp: object = None
    last_accepted_value: float | None = None

    # buffer de 1 minuto (stream unicamente, nunca incluye lecturas de bootstrap) -- usado
    # tanto para las ventanas de P como para acumular el bucket de 15 min abierto
    buffer_1min: deque = field(default_factory=lambda: deque(maxlen=BUFFER_1MIN_MAXLEN))

    # bucket de 15 minutos actualmente abierto (lecturas de stream desde el ultimo cierre)
    open_bucket_values: list = field(default_factory=list)
    open_bucket_index: int = 0  # indice 0-based del bin actual dentro del streaming

    # agregados historicos de 15 min (bootstrap + streaming ya cerrados), acotado
    historial_15min: deque = field(default_factory=lambda: deque(maxlen=H_HISTORY_MAXLEN))

    stream_minutes_ingested: int = 0  # minutos de stream aceptados desde el inicio (excluye bootstrap)

    last_score_p: float | None = None
    last_score_h: float | None = None
    last_alert_p: bool | None = None          # alerta de la ultima decision de P (sin propagar)
    last_alert_h: bool | None = None
    p_last_evaluation_timestamp: object = None
    h_last_evaluation_timestamp: object = None

    n_accepted: int = 0
    n_rejected: int = 0
    n_gaps: int = 0
    n_duplicates_identical: int = 0
    n_duplicates_conflicting: int = 0
    n_out_of_order: int = 0

    current_15min_bucket_valid: bool = True  # False si el bucket abierto tiene un hueco dentro
    degraded: bool = False  # True tras detectar un hueco, hasta que P/H recuperen ventana completa

    recent_warnings: deque = field(default_factory=lambda: deque(maxlen=20))

    def has_complete_window(self, n_minutes: int) -> bool:
        """True si el buffer de 1 min tiene al menos `n_minutes` lecturas y las ultimas
        `n_minutes` son contiguas (sin huecos). Un simple contador de lecturas aceptadas no
        basta: tras un hueco, el contador sigue subiendo pero la ventana deja de ser valida
        hasta que se acumulan de nuevo `n_minutes` contiguos."""
        if len(self.buffer_1min) < n_minutes:
            return False
        cola = list(self.buffer_1min)[-n_minutes:]
        ts = [t for t, _ in cola]
        return all((ts[i] - ts[i - 1]) == pd.Timedelta(minutes=1) for i in range(1, len(ts)))

    def historial_contiguous_tail(self, n_bins: int) -> bool:
        """Equivalente de `has_complete_window` para el historial de 15 min: los ultimos
        `n_bins` deben ser bins completos (15 min cada uno) y consecutivos entre si."""
        if len(self.historial_15min) < n_bins:
            return False
        cola = list(self.historial_15min)[-n_bins:]
        for b in cola:
            if (b.ts_end - b.ts_start) != pd.Timedelta(minutes=RESOLUCION_H_MIN - 1):
                return False
        for i in range(1, len(cola)):
            if (cola[i].ts_start - cola[i - 1].ts_end) != pd.Timedelta(minutes=1):
                return False
        return True

    def p_ready(self) -> bool:
        return self.stream_minutes_ingested >= P_WINDOW_MIN and self.has_complete_window(P_WINDOW_MIN)

    def h_ready(self) -> bool:
        return self.historial_contiguous_tail(MAX_LAG_H_BINS + 1)

    def active_p_ffill(self) -> bool:
        """Estado de P propagado: el ultimo valor conocido, False antes de la primera decision
        (identico a fusion_p_histgb.estado_ffill con idx=-1 -> False)."""
        if self.last_alert_p is None:
            return False
        return bool(self.last_alert_p)
