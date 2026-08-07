"""Motor de inferencia online, causal y con estado temporal. Interfaz separada
de cualquier framework web -- FastAPI, en una fase posterior, se limitara a validar el JSON
de entrada y llamar a `engine.ingest(...)`.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from edge_deployment.core import detector_state as ds
from edge_deployment.core import fusion_online
from edge_deployment.core import h_online
from edge_deployment.core import model_loader as ml
from edge_deployment.core import p_online
from edge_deployment.core import streaming_preprocessing as spp
from edge_deployment.core.detector_state import Bin15Min, DetectorState
from edge_deployment.core.schemas import IngestResponse, Reading
from edge_deployment.core.validation import validate_reading

BASE_DIR = Path(__file__).resolve().parents[2]
EDGE_DIR = BASE_DIR / "edge_deployment"
REPORTS_DIR = EDGE_DIR / "results" / "reports"


class DetectorEngine:
    def __init__(self, artifact_manifest: dict | None = None, config: dict | None = None,
                 use_legacy_loader: bool = False):
        """Fase 4: `artifact_manifest`, si se pasa, es el dict de artefactos ya cargado (mismas
        7 claves que devuelven `model_loader.load_frozen_artifacts[_autonomous]`) -- permite
        inyectarlo desde fuera (p.ej. el experimento de equivalencia) sin volver a
        cargar nada. Si no se pasa, `use_legacy_loader=False` (por defecto, produccion) usa
        `load_frozen_artifacts_autonomous` (autonomous_loader: sin CSV, con validacion real de
        SHA-256); `use_legacy_loader=True` usa `load_frozen_artifacts` (legacy_loader,
        reconstruye params_norm desde data_raw.csv/episodes_master_val.csv) -- solo para el
        experimento de equivalencia, nunca en la API/Docker de produccion."""
        if artifact_manifest is not None:
            artefactos = artifact_manifest
        elif use_legacy_loader:
            artefactos = ml.load_frozen_artifacts()
        else:
            artefactos = ml.load_frozen_artifacts_autonomous()
        self.modelo_p = artefactos["modelo_p"]
        self.params_norm_p = artefactos["params_norm_p"]
        self.threshold_p = artefactos["threshold_p"]
        self.modelo_h = artefactos["modelo_h"]
        self.perfil_h = artefactos["perfil_h"]
        self.threshold_h = artefactos["threshold_h"]
        self.features_h_orden = artefactos["features_h_orden"]
        self.config = config or {}
        self._states: dict[str, DetectorState] = {}

    # --------------------------------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------------------------------

    def bootstrap(self, meter_id: str, historical_readings: pd.DataFrame) -> dict:
        from src import predictor_causal_lags as pcl
        from src.data_loading import COLUMNA_OBJETIVO

        if len(historical_readings) == 0:
            raise ValueError("bootstrap: historical_readings esta vacio")
        if not historical_readings.index.is_monotonic_increasing:
            raise ValueError("bootstrap: las lecturas historicas deben estar en orden temporal estrictamente creciente")
        if historical_readings.index.has_duplicates:
            raise ValueError("bootstrap: se detectaron timestamps duplicados en el contexto historico")

        esperado = pd.date_range(historical_readings.index.min(), historical_readings.index.max(), freq="1min")
        missing_minutes = len(esperado) - len(historical_readings)
        if missing_minutes > 0:
            raise ValueError(f"bootstrap: el contexto historico tiene {missing_minutes} minutos faltantes -- "
                              f"no se rellena automaticamente, debe suministrarse continuo")

        invalid_values = int(historical_readings[COLUMNA_OBJETIVO].isna().sum())
        if invalid_values > 0:
            raise ValueError(f"bootstrap: {invalid_values} valores NaN en el contexto historico")

        state = DetectorState(meter_id=meter_id)
        kw15 = pcl.construir_serie_15min(historical_readings)
        for kw, ts_s, ts_e in zip(kw15["kw"], kw15["ts_start"], kw15["ts_end"]):
            state.historial_15min.append(Bin15Min(ts_start=pd.Timestamp(ts_s), ts_end=pd.Timestamp(ts_e), kw=float(kw)))

        state.bootstrapped = True
        state.bootstrap_end_timestamp = historical_readings.index.max()
        # el ultimo timestamp del bootstrap se registra como "ultimo aceptado" para poder
        # detectar un hueco entre el fin del bootstrap y la primera lectura de streaming;
        # no cuenta como lectura de streaming (no incrementa stream_minutes_ingested/n_accepted)
        state.last_accepted_timestamp = historical_readings.index.max()
        state.last_accepted_value = float(historical_readings[COLUMNA_OBJETIVO].iloc[-1])
        self._states[meter_id] = state

        required = ds.required_context()
        reporte = {
            "meter_id": meter_id,
            "first_bootstrap_timestamp": str(historical_readings.index.min()),
            "last_bootstrap_timestamp": str(historical_readings.index.max()),
            "first_stream_timestamp": None,
            "number_of_readings": int(len(historical_readings)),
            "expected_readings": int(len(esperado)),
            "missing_minutes": int(missing_minutes),
            "duplicates": 0,
            "invalid_values": invalid_values,
            "p_ready": state.p_ready(), "h_ready": state.h_ready(),
            "required_context": required, "available_context_bins_15min": len(state.historial_15min),
            "warnings": [] if state.h_ready() else ["contexto de bootstrap insuficiente para H (menos de 673 bins de 15 min)"],
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "bootstrap_report.json", "w", encoding="utf-8") as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
        return reporte

    # --------------------------------------------------------------------------------
    # Ingestion
    # --------------------------------------------------------------------------------

    def ingest(self, meter_id: str, timestamp, power_kw) -> IngestResponse:
        t0 = time.perf_counter()
        warnings: list[str] = []
        state = self._states.get(meter_id)
        if state is None:
            state = DetectorState(meter_id=meter_id)
            self._states[meter_id] = state

        reading = Reading(meter_id=meter_id, timestamp=timestamp, power_kw=power_kw)
        vres = validate_reading(reading, state.last_accepted_timestamp, state.last_accepted_value)
        ts_out = timestamp if (timestamp is None or pd.isna(timestamp)) else pd.Timestamp(timestamp)

        if not vres.accepted:
            state.n_rejected += 1
            if vres.classification == "duplicate_identical":
                state.n_duplicates_identical += 1
            elif vres.classification == "duplicate_conflicting":
                state.n_duplicates_conflicting += 1
            elif vres.classification == "out_of_order":
                state.n_out_of_order += 1
            dt_ms = (time.perf_counter() - t0) * 1000
            return IngestResponse(
                meter_id=meter_id, timestamp=ts_out, accepted=False, rejection_reason=vres.rejection_reason,
                engine_status="invalid_input", warming_up=not (state.p_ready() and state.h_ready()),
                p_ready=state.p_ready(), h_ready=state.h_ready(), p_evaluated=False, h_evaluated=False,
                score_p=state.last_score_p, score_h=state.last_score_h, alert_p=state.last_alert_p,
                alert_p_ffill=state.active_p_ffill(), alert_h=state.last_alert_h, alert_or=None,
                p_last_evaluation_timestamp=state.p_last_evaluation_timestamp,
                h_last_evaluation_timestamp=state.h_last_evaluation_timestamp,
                buffer_1min_size=len(state.buffer_1min), buffer_15min_size=len(state.historial_15min),
                processing_time_ms=dt_ms, warnings=warnings, detection_source=None,
            )

        info = spp.append_reading(state, pd.Timestamp(timestamp), float(power_kw))
        if info["gap_detected"]:
            warnings.append("hueco temporal detectado: bucket de 15min en curso descartado, ventanas afectadas no se evaluan")

        p_evaluated = False
        p_res = p_online.evaluar_p_si_corresponde(state, self.modelo_p, self.params_norm_p, self.threshold_p)
        if p_res is not None:
            state.last_score_p = p_res["score_p"]
            state.last_alert_p = p_res["active_p"]
            state.p_last_evaluation_timestamp = p_res["timestamp"]
            p_evaluated = True

        h_evaluated = False
        h_res = h_online.evaluar_h_si_corresponde(state, self.modelo_h, self.perfil_h, self.features_h_orden,
                                                    self.threshold_h, info["bin_closed"])
        if h_res is not None:
            state.last_score_h = h_res["score_h"]
            state.last_alert_h = h_res["active_h"]
            state.h_last_evaluation_timestamp = h_res["timestamp"]
            h_evaluated = True

        fusion = fusion_online.fusionar(state.active_p_ffill(), state.p_ready(), state.last_alert_h, state.h_ready())
        engine_status = self._compute_status(state, current_gap=info["gap_detected"])
        warnings.extend(list(state.recent_warnings)[-3:] if info["gap_detected"] else [])

        dt_ms = (time.perf_counter() - t0) * 1000
        return IngestResponse(
            meter_id=meter_id, timestamp=ts_out, accepted=True, rejection_reason=None,
            engine_status=engine_status, warming_up=(engine_status == "warming_up"),
            p_ready=state.p_ready(), h_ready=state.h_ready(), p_evaluated=p_evaluated, h_evaluated=h_evaluated,
            score_p=state.last_score_p, score_h=state.last_score_h, alert_p=state.last_alert_p,
            alert_p_ffill=state.active_p_ffill(), alert_h=state.last_alert_h, alert_or=fusion["alert_or"],
            p_last_evaluation_timestamp=state.p_last_evaluation_timestamp,
            h_last_evaluation_timestamp=state.h_last_evaluation_timestamp,
            buffer_1min_size=len(state.buffer_1min), buffer_15min_size=len(state.historial_15min),
            processing_time_ms=dt_ms, warnings=warnings, detection_source=fusion["detection_source"],
            aggregate_15min_kw=(h_res["observed_kw"] if h_res is not None else None),
        )

    # --------------------------------------------------------------------------------

    def _compute_status(self, state: DetectorState, current_gap: bool = False) -> str:
        if not state.bootstrapped and state.n_accepted == 0:
            return "empty"
        if current_gap:
            return "temporal_gap"
        if state.degraded:
            return "degraded"
        if state.p_ready() and state.h_ready():
            return "ready"
        return "warming_up"

    def get_status(self, meter_id: str) -> dict:
        """Nota (Fase 2, adaptacion menor documentada): se anadieron
        `last_accepted_timestamp`, `stream_anchor_timestamp`, `current_bucket_start`,
        `current_bucket_sample_count` y `last_alert_or` respecto a la version de Fase 1 --
        son lecturas directas de campos que `DetectorState` ya mantenia (o, para
        `last_alert_or`, una llamada a `fusion_online.fusionar`, la misma funcion pura que ya
        usa `ingest()`). No se anadio ni cambio ninguna logica de causalidad/scoring; los
        tests de equivalencia de Fase 1 (`edge_deployment/tests`) siguen pasando sin cambios."""
        state = self._states.get(meter_id)
        if state is None:
            return {"meter_id": meter_id, "state_exists": False, "engine_status": "empty", "p_ready": False, "h_ready": False,
                    "bootstrapped": False, "last_accepted_timestamp": None, "stream_anchor_timestamp": None,
                    "current_bucket_start": None, "current_bucket_sample_count": 0, "last_alert_or": None,
                    "n_accepted": 0, "n_rejected": 0, "n_gaps": 0, "n_duplicates_identical": 0,
                    "n_duplicates_conflicting": 0, "n_out_of_order": 0,
                    "buffer_1min_size": 0, "buffer_15min_size": 0}
        fusion = fusion_online.fusionar(state.active_p_ffill(), state.p_ready(), state.last_alert_h, state.h_ready())
        return {
            "meter_id": meter_id, "state_exists": True, "engine_status": self._compute_status(state),
            "p_ready": state.p_ready(), "h_ready": state.h_ready(), "bootstrapped": state.bootstrapped,
            "last_accepted_timestamp": state.last_accepted_timestamp,
            "stream_anchor_timestamp": (state.bootstrap_end_timestamp + pd.Timedelta(minutes=1)) if state.bootstrap_end_timestamp is not None else None,
            "current_bucket_start": state.open_bucket_values[0][0] if state.open_bucket_values else None,
            "current_bucket_sample_count": len(state.open_bucket_values),
            "score_p": state.last_score_p, "score_h": state.last_score_h,
            "alert_p_ffill": state.active_p_ffill(), "alert_h": state.last_alert_h, "last_alert_or": fusion["alert_or"],
            "p_last_evaluation_timestamp": state.p_last_evaluation_timestamp,
            "h_last_evaluation_timestamp": state.h_last_evaluation_timestamp,
            "n_accepted": state.n_accepted, "n_rejected": state.n_rejected, "n_gaps": state.n_gaps,
            "n_duplicates_identical": state.n_duplicates_identical, "n_duplicates_conflicting": state.n_duplicates_conflicting,
            "n_out_of_order": state.n_out_of_order,
            "buffer_1min_size": len(state.buffer_1min), "buffer_15min_size": len(state.historial_15min),
        }

    def n_active_meters(self) -> int:
        """Fase 2: lectura de solo diagnostico (usada por /metrics), sin logica nueva."""
        return len(self._states)

    def reset(self, meter_id: str) -> dict:
        """Vacia el estado del contador. Los modelos (self.modelo_p/self.modelo_h/etc.)
        permanecen cargados -- reset() nunca recarga ni modifica artefactos."""
        existed = meter_id in self._states
        self._states.pop(meter_id, None)
        return {"meter_id": meter_id, "reset": True, "existed": existed,
                "fecha_utc": datetime.now(timezone.utc).isoformat()}
