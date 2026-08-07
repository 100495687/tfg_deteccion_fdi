"""Propagacion causal de P y fusion OR.

El estado propagado de P se reduce, en streaming, a mantener unicamente la ultima
decision conocida (score/alerta/timestamp): en cualquier instante t, "el estado de P en t"
es por definicion la decision de P mas reciente con `timestamp <= t` (identico a
`fusion_p_histgb.estado_ffill`, que hace exactamente esa busqueda mediante
`np.searchsorted(..., side='right')-1` sobre el historial completo). No hace falta
reconstruir el historial de P para consultarlo: como el motor solo pregunta por el
"ahora", basta O(1) en vez de la busqueda binaria offline -- ver
`edge_deployment/tests/test_causality.py::test_propagacion_equivale_a_estado_ffill`.
"""
from __future__ import annotations


def fusionar(active_p_ffill: bool, p_ready: bool, active_h: bool | None, h_ready: bool) -> dict:
    """Devuelve alert_or y detection_source. Si una rama no esta disponible todavia, no se
    trata su ausencia como "normal" (False): se marca explicitamente `partial_availability`
    y alert_or se calcula solo con lo disponible (mejor esfuerzo, nunca
    oculto)."""
    p_contribuye = bool(p_ready and active_p_ffill)
    h_contribuye = bool(h_ready and active_h)

    if p_ready and h_ready:
        alert_or = p_contribuye or h_contribuye
        if p_contribuye and h_contribuye:
            source = "both"
        elif p_contribuye:
            source = "P_only"
        elif h_contribuye:
            source = "H_only"
        else:
            source = "none"
    elif p_ready or h_ready:
        alert_or = p_contribuye or h_contribuye
        source = "partial_availability"
    else:
        alert_or = False
        source = "none"

    return {"alert_or": alert_or, "detection_source": source}
