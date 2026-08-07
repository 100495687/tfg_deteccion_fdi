"""Verifica el anclaje posicional (no de reloj) de las rejillas de P y H --
"no asumir ninguna convencion temporal sin comprobarla en el codigo"."""
from __future__ import annotations

import pandas as pd

from edge_deployment.core.detector_state import P_WINDOW_MIN


def test_bootstrap_end_immediately_precedes_stream_start(bootstrapped):
    boot_end = pd.Timestamp(bootstrapped["report"]["last_bootstrap_timestamp"])
    assert bootstrapped["stream_start"] == boot_end + pd.Timedelta(minutes=1)


def test_p_grid_anchored_to_stream_start_not_wallclock(engine, bootstrapped):
    """El primer timestamp de decision de P debe ser `stream_start + 359 minutos`,
    independientemente de si `stream_start` cae en una hora en punto."""
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    # stream_start no esta alineado a una hora en punto salvo coincidencia -- se comprueba
    # igualmente el anclaje relativo, no absoluto
    resp = None
    for i in range(P_WINDOW_MIN):
        resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
    assert pd.Timestamp(resp.p_last_evaluation_timestamp) == stream_start + pd.Timedelta(minutes=P_WINDOW_MIN - 1)


def test_h_bin_boundaries_positional_from_stream_start(engine, bootstrapped):
    mid, stream_start = bootstrapped["meter_id"], bootstrapped["stream_start"]
    resp = None
    for i in range(15):
        resp = engine.ingest(mid, stream_start + pd.Timedelta(minutes=i), 1.0)
    assert resp.h_evaluated
    assert pd.Timestamp(resp.h_last_evaluation_timestamp) == stream_start + pd.Timedelta(minutes=14)


def test_real_pretest_boundary_is_not_clock_aligned():
    """Evidencia empirica (period_selection.py sobre el dataset real): el limite
    bootstrap/streaming cae en 17:23/17:24, no en :00/:15/:30/:45 -- confirma que
    `construir_serie_15min`/`ventanear` indexan posicionalmente, nunca por reloj."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "results" / "reports" / "period_selection.json"
    if not path.exists():
        import pytest
        pytest.skip("period_selection.json aun no generado (ejecutar replay_clean_stream primero)")
    with open(path, encoding="utf-8") as f:
        reporte = json.load(f)
    fin_boot = pd.Timestamp(reporte["bootstrap_period"]["end"])
    ini_stream = pd.Timestamp(reporte["streaming_period"]["start"])
    assert ini_stream == fin_boot + pd.Timedelta(minutes=1)
    assert fin_boot.minute % 15 != 0 or fin_boot.minute not in (0, 15, 30, 45, 59)
