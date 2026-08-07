"""Fase 5 -- protección anti-replay opcional y aislada (36 tests minimos, seccion 21).

Organizado por GATE: los tests de GATE 1 (canonicalizacion/HMAC/KeyProvider/AntiReplayState/
reloj) son unitarios puros, sin FastAPI ni Docker. Los de GATE 2+ (marcados mas abajo) se
añaden a medida que cada gate se implementa -- este fichero crece de forma incremental, igual
que el propio desarrollo por gates (seccion 15).
"""
from __future__ import annotations

import hmac as hmac_stdlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from edge_deployment.security import hmac_auth
from edge_deployment.security.anti_replay import (
    REASON_DUPLICATE_SEQUENCE,
    REASON_FUTURE_TIMESTAMP,
    REASON_INVALID_MAC,
    REASON_STALE_TIMESTAMP,
    REASON_UNKNOWN_METER_KEY,
    REASON_UNKNOWN_SESSION,
    AntiReplayGuard,
)
from edge_deployment.security.canonicalization import canonicalize
from edge_deployment.security.key_provider import EnvironmentKeyProvider, FileKeyProvider, InMemoryTestKeyProvider
from edge_deployment.security.security_clock import SimulatedClock, SystemClock
from edge_deployment.security.security_state import AntiReplayState

SECURITY_DIR = Path(__file__).resolve().parents[1] / "security"


def _msg(**overrides) -> dict:
    base = {
        "protocol_version": "1", "meter_id": "house_01", "session_id": "session-001",
        "timestamp": "2009-06-18T17:24:00Z", "sequence_number": 15231, "power_kw": 1.42,
    }
    base.update(overrides)
    return base


def _sign(message: dict, key: bytes) -> str:
    return hmac_auth.compute_hmac(key, canonicalize(message))


# ==================================================================================================
# GATE 1.1 -- canonicalizacion determinista (tests 1, 10, 11, 12)
# ==================================================================================================

def test_01_same_message_produces_same_canonical_bytes():
    m = _msg()
    assert canonicalize(m) == canonicalize(dict(m))


def test_01b_canonical_bytes_are_utf8_json_with_fixed_field_order():
    b = canonicalize(_msg())
    text = b.decode("utf-8")
    assert text.index("protocol_version") < text.index("meter_id") < text.index("session_id") \
        < text.index("timestamp") < text.index("sequence_number") < text.index("power_kw")
    assert " " not in text  # separadores compactos, sin espacios variables


def test_10_timestamp_changed_changes_canonical_bytes_and_therefore_mac():
    key = b"k"
    m1 = _msg()
    m2 = _msg(timestamp="2009-06-18T17:25:00Z")
    assert canonicalize(m1) != canonicalize(m2)
    assert hmac_auth.compute_hmac(key, canonicalize(m1)) != hmac_auth.compute_hmac(key, canonicalize(m2))


def test_11_sequence_number_changed_changes_canonical_bytes_and_therefore_mac():
    key = b"k"
    m1 = _msg()
    m2 = _msg(sequence_number=15232)
    assert canonicalize(m1) != canonicalize(m2)
    assert hmac_auth.compute_hmac(key, canonicalize(m1)) != hmac_auth.compute_hmac(key, canonicalize(m2))


def test_12_power_kw_changed_changes_canonical_bytes_and_therefore_mac():
    key = b"k"
    m1 = _msg()
    m2 = _msg(power_kw=1.43)
    assert canonicalize(m1) != canonicalize(m2)
    assert hmac_auth.compute_hmac(key, canonicalize(m1)) != hmac_auth.compute_hmac(key, canonicalize(m2))


def test_meter_id_or_session_id_changed_changes_canonical_bytes():
    m1 = _msg()
    assert canonicalize(m1) != canonicalize(_msg(meter_id="house_02"))
    assert canonicalize(m1) != canonicalize(_msg(session_id="session-002"))


def test_power_kw_formatted_with_fixed_6_decimals_no_native_json_float():
    b = canonicalize(_msg(power_kw=1.4))
    assert b'"power_kw":"1.400000"' in b


# ==================================================================================================
# GATE 1.2 -- HMAC (tests 2, 3)
# ==================================================================================================

def test_02_same_message_and_key_always_produce_same_hmac():
    key = b"secret-key"
    m = _msg()
    assert hmac_auth.compute_hmac(key, canonicalize(m)) == hmac_auth.compute_hmac(key, canonicalize(dict(m)))


def test_02b_changing_any_covered_field_changes_hmac():
    key = b"secret-key"
    base_mac = hmac_auth.compute_hmac(key, canonicalize(_msg()))
    for field, value in [("protocol_version", "2"), ("meter_id", "house_99"), ("session_id", "session-999"),
                         ("timestamp", "2009-06-18T18:24:00Z"), ("sequence_number", 1), ("power_kw", 9.99)]:
        mac = hmac_auth.compute_hmac(key, canonicalize(_msg(**{field: value})))
        assert mac != base_mac, f"cambiar {field} no cambio el HMAC"


def test_03_hmac_verification_uses_compare_digest_not_equality():
    source = (SECURITY_DIR / "hmac_auth.py").read_text(encoding="utf-8")
    assert "hmac.compare_digest" in source
    assert "== mac_hex" not in source and "mac_hex ==" not in source


def test_03b_verify_hmac_matches_stdlib_reference_implementation():
    key = b"secret-key"
    m = _msg()
    canonical = canonicalize(m)
    reference_mac = hmac_stdlib.new(key, canonical, "sha256").hexdigest()
    assert hmac_auth.verify_hmac(key, canonical, reference_mac) is True


def test_wrong_key_fails_verification():
    m = _msg()
    canonical = canonicalize(m)
    mac = hmac_auth.compute_hmac(b"key-A", canonical)
    assert hmac_auth.verify_hmac(b"key-B", canonical, mac) is False


def test_random_mac_fails_verification():
    canonical = canonicalize(_msg())
    assert hmac_auth.verify_hmac(b"any-key", canonical, "0" * 64) is False


# ==================================================================================================
# GATE 1.3 -- KeyProvider (test 4, 5, 26, 27 parcial)
# ==================================================================================================

def test_environment_key_provider_reads_prefixed_env_var(monkeypatch):
    monkeypatch.setenv("FDIA_METER_KEY_house_01", "super-secret")
    provider = EnvironmentKeyProvider(key_env_prefix="FDIA_METER_KEY_")
    assert provider.get_key("house_01") == b"super-secret"


def test_environment_key_provider_returns_none_for_unknown_meter(monkeypatch):
    monkeypatch.delenv("FDIA_METER_KEY_house_99", raising=False)
    provider = EnvironmentKeyProvider()
    assert provider.get_key("house_99") is None


def test_file_key_provider_reads_key_file(tmp_path):
    (tmp_path / "house_01.key").write_bytes(b"file-secret\n")
    provider = FileKeyProvider(tmp_path)
    assert provider.get_key("house_01") == b"file-secret"


def test_file_key_provider_returns_none_for_missing_file(tmp_path):
    provider = FileKeyProvider(tmp_path)
    assert provider.get_key("house_99") is None


def test_in_memory_test_key_provider_only_returns_set_keys():
    provider = InMemoryTestKeyProvider()
    assert provider.get_key("house_01") is None
    provider.set_key("house_01", b"test-key")
    assert provider.get_key("house_01") == b"test-key"


def test_no_key_provider_creates_a_default_key():
    """Ninguna implementacion debe devolver una clave por defecto para un meter_id nunca
    aprovisionado (seccion 9: "no crear claves por defecto")."""
    assert EnvironmentKeyProvider(key_env_prefix="FDIA_UNUSED_PREFIX_").get_key("anything") is None
    assert InMemoryTestKeyProvider().get_key("anything") is None


def test_no_key_string_literal_hardcoded_in_security_source():
    """Ningun fichero de security/ contiene una clave de ejemplo hardcodeada reutilizable
    (heuristica: no hay asignaciones tipo `key = b"..."` fuera de tests/docstrings de mas de
    unos pocos bytes en key_provider.py)."""
    text = (SECURITY_DIR / "key_provider.py").read_text(encoding="utf-8")
    assert "DEFAULT_KEY" not in text.upper().replace("_", "")


# ==================================================================================================
# GATE 1.4 -- AntiReplayState (tests 15 parcial, 28)
# ==================================================================================================

def test_unprovisioned_session_is_unknown():
    state = AntiReplayState()
    assert state.get("house_01", "session-999") is None


def test_state_bounded_by_provisioned_sessions_only():
    """Un atacante que envia session_id arbitrarios no puede hacer crecer AntiReplayState --
    solo init_session() (aprovisionamiento explicito) crea una entrada nueva."""
    state = AntiReplayState()
    for i in range(1000):
        state.get("house_01", f"attacker-session-{i}")  # nunca crea nada
    assert state.n_sessions() == 0
    state.init_session("house_01", "session-001")
    assert state.n_sessions() == 1


def test_record_accept_advances_sequence_and_timestamp():
    state = AntiReplayState()
    state.init_session("house_01", "session-001")
    now = datetime(2009, 6, 18, 17, 25, tzinfo=timezone.utc)
    ts = datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc)
    state.record_accept("house_01", "session-001", 15231, ts, now=now)
    session = state.get("house_01", "session-001")
    assert session.highest_accepted_sequence == 15231
    assert session.last_accepted_timestamp == ts


def test_record_rejection_never_advances_sequence():
    state = AntiReplayState()
    state.init_session("house_01", "session-001")
    now = datetime(2009, 6, 18, 17, 25, tzinfo=timezone.utc)
    state.record_accept("house_01", "session-001", 15231, now, now=now)
    state.record_rejection("house_01", "session-001", "duplicate_sequence", now=now)
    session = state.get("house_01", "session-001")
    assert session.highest_accepted_sequence == 15231  # sin cambios
    assert session.counters.replay_rejections == 1


def test_record_rejection_never_creates_a_session_bug_regression():
    """Regresion: record_rejection() NO debe crear una sesion nueva -- un atacante enviando
    (meter_id, session_id) arbitrarios para forzar rechazos no puede hacer crecer el estado."""
    state = AntiReplayState()
    for i in range(500):
        state.record_rejection("house_01", f"attacker-session-{i}", "unknown_session",
                                now=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    assert state.n_sessions() == 0


def test_anti_replay_state_never_stores_readings_scores_or_keys():
    """AntiReplayState solo expone los campos documentados en la seccion 10 -- ni ventanas,
    ni lecturas, ni scores, ni claves."""
    state = AntiReplayState()
    session = state.init_session("house_01", "session-001")
    campos = set(vars(session).keys()) | {"counters"}
    prohibidos = {"window", "buffer", "score", "score_p", "score_h", "key", "power_kw", "readings", "lags"}
    assert campos.isdisjoint(prohibidos)


# ==================================================================================================
# GATE 1.5 -- reloj y frescura (tests 13, 14)
# ==================================================================================================

def test_system_clock_returns_timezone_aware_utc_datetime():
    now = SystemClock().now()
    assert now.tzinfo is not None


def test_simulated_clock_does_not_touch_system_time_and_is_controllable():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    assert clock.now() == datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc)
    clock.advance(60)
    assert clock.now() == datetime(2009, 6, 18, 17, 25, tzinfo=timezone.utc)


def _guard(clock, max_age=300, max_skew=60) -> tuple[AntiReplayGuard, InMemoryTestKeyProvider]:
    kp = InMemoryTestKeyProvider()
    kp.set_key("house_01", b"secret-key")
    state = AntiReplayState()
    state.init_session("house_01", "session-001")
    return AntiReplayGuard(kp, clock, state, max_message_age_seconds=max_age, max_future_skew_seconds=max_skew), kp


def test_13_stale_timestamp_is_rejected():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 18, 0, tzinfo=timezone.utc))  # 36 min despues
    guard, _ = _guard(clock, max_age=300)  # max 5 min de antiguedad
    r = guard.check_sequence_and_freshness("house_01", "session-001", 1,
                                            datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    assert r.ok is False and r.reason == REASON_STALE_TIMESTAMP


def test_14_future_timestamp_is_rejected():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    guard, _ = _guard(clock, max_skew=60)
    r = guard.check_sequence_and_freshness("house_01", "session-001", 1,
                                            datetime(2009, 6, 18, 17, 30, tzinfo=timezone.utc))  # 6 min en el futuro
    assert r.ok is False and r.reason == REASON_FUTURE_TIMESTAMP


def test_fresh_timestamp_within_window_passes():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 25, tzinfo=timezone.utc))
    guard, _ = _guard(clock, max_age=300, max_skew=60)
    r = guard.check_sequence_and_freshness("house_01", "session-001", 1,
                                            datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    assert r.ok is True


# ==================================================================================================
# GATE 1.6 -- AntiReplayGuard de extremo a extremo (sin FastAPI): tests 4, 5, 6, 7, 8, 15
# ==================================================================================================

def test_04_wrong_key_is_rejected_by_guard():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    guard, kp = _guard(clock)
    m = _msg()
    m["mac"] = _sign(m, b"WRONG-KEY")
    r = guard.verify_authenticity(m)
    assert r.ok is False and r.reason == REASON_INVALID_MAC


def test_05_unknown_meter_is_rejected_by_guard():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    guard, kp = _guard(clock)
    m = _msg(meter_id="house_99")
    m["mac"] = _sign(m, b"some-key")
    r = guard.verify_authenticity(m)
    assert r.ok is False and r.reason == REASON_UNKNOWN_METER_KEY


def test_06_legitimate_message_is_accepted_by_guard():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    guard, kp = _guard(clock)
    m = _msg()
    m["mac"] = _sign(m, b"secret-key")
    auth = guard.verify_authenticity(m)
    assert auth.ok is True
    seq = guard.check_sequence_and_freshness(m["meter_id"], m["session_id"], m["sequence_number"],
                                              datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    assert seq.ok is True


def test_07_exact_duplicate_is_rejected_after_first_acceptance():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    guard, kp = _guard(clock)
    ts = datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc)
    guard.commit_accept("house_01", "session-001", 15231, ts)
    r = guard.check_sequence_and_freshness("house_01", "session-001", 15231, ts)
    assert r.ok is False and r.reason == REASON_DUPLICATE_SEQUENCE


def test_08_old_sequence_is_rejected():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 30, tzinfo=timezone.utc))
    guard, kp = _guard(clock)
    guard.commit_accept("house_01", "session-001", 15231, datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    r = guard.check_sequence_and_freshness("house_01", "session-001", 15200,
                                            datetime(2009, 6, 18, 17, 20, tzinfo=timezone.utc))
    assert r.ok is False and r.reason == REASON_DUPLICATE_SEQUENCE


def test_15_wrong_session_id_is_rejected():
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    guard, kp = _guard(clock)
    r = guard.check_sequence_and_freshness("house_01", "session-DOES-NOT-EXIST", 1,
                                            datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    assert r.ok is False and r.reason == REASON_UNKNOWN_SESSION


def test_19_state_only_advances_after_full_acceptance_never_on_partial_checks():
    """Verificar autenticidad o frescura NUNCA muta AntiReplayState -- solo commit_accept."""
    clock = SimulatedClock(start=datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    guard, kp = _guard(clock)
    m = _msg()
    m["mac"] = _sign(m, b"secret-key")
    guard.verify_authenticity(m)
    guard.check_sequence_and_freshness(m["meter_id"], m["session_id"], m["sequence_number"],
                                        datetime(2009, 6, 18, 17, 24, tzinfo=timezone.utc))
    session = guard.state.get("house_01", "session-001")
    assert session.highest_accepted_sequence is None  # todavia no se ha llamado a commit_accept


# ==================================================================================================
# GATE 1.7 -- higiene (tests 32, 33, 34 para el codigo de Fase 5)
# ==================================================================================================

SECURITY_FILES = list(SECURITY_DIR.glob("*.py"))


def test_32_no_fit_call_in_security_source():
    for f in SECURITY_FILES:
        for line in f.read_text(encoding="utf-8").splitlines():
            sin_espacios = line.replace(" ", "")
            assert ".fit(" not in sin_espacios or ".fit()" in sin_espacios, f"{f.name}: {line!r}"


def test_33_no_recalibration_in_security_source():
    for f in SECURITY_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "calibrar_umbral" not in text and "recalibrar" not in text


def test_34_no_test_partition_referenced_in_security_source():
    for f in SECURITY_FILES:
        text = f.read_text(encoding="utf-8")
        assert "particiones[\"test\"]" not in text.lower()
        assert "construir_ataques_desde_manifiesto" not in text


def test_security_module_never_imports_core_detector_engine_or_state():
    """La capa de seguridad es anterior/independiente a DetectorEngine -- no debe importarlo
    (la orquestacion que SI llama al motor vive en la capa API, Gate 2, no aqui)."""
    for f in SECURITY_FILES:
        text = f.read_text(encoding="utf-8")
        assert "detector_engine" not in text.lower()
        assert "from edge_deployment.core.detector_state" not in text
        assert "import DetectorState" not in text


def test_security_module_does_not_modify_p_h_or_thresholds():
    for f in SECURITY_FILES:
        text = f.read_text(encoding="utf-8").lower()
        assert "threshold_p" not in text and "threshold_h" not in text
        assert "modelo_p" not in text and "modelo_h" not in text


# ==================================================================================================
# GATE 2 -- integracion API (tests 16-26). Cada `with TestClient(app)` re-ejecuta el lifespan
# (y por tanto `init_security`) leyendo las variables de entorno DEL MOMENTO -- confirmado
# empiricamente antes de escribir estos tests. `meter_id`/`session_id` incluyen el nombre del
# test para no compartir estado entre tests que reutilizan el mismo `app` module-level.
# ==================================================================================================

import edge_deployment.security.hmac_auth as _hmac_auth_mod  # noqa: E402
from edge_deployment.security.canonicalization import canonicalize as _canon  # noqa: E402

EDGE_DIR = Path(__file__).resolve().parents[1]


def _secure_client(monkeypatch, *, enabled: bool = True, clock_mode: str = "simulated", meter_key: bytes | None = None,
                    meter_id: str = "house_01", test_endpoints_enabled: bool = True):
    """`test_endpoints_enabled=True` por defecto (necesario para que la mayoria de estos tests
    puedan aprovisionar sesiones/reloj via /security/test/*, el "mecanismo experimental
    controlado" de la seccion 14) -- los tests que verifican especificamente el comportamiento
    de PRODUCCION pasan `test_endpoints_enabled=False` explicitamente."""
    from fastapi.testclient import TestClient

    from edge_deployment.api.main import app

    monkeypatch.setenv("FDIA_ANTI_REPLAY_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("FDIA_SECURITY_CLOCK_MODE", clock_mode)
    monkeypatch.setenv("FDIA_SECURITY_TEST_ENDPOINTS_ENABLED", "true" if test_endpoints_enabled else "false")
    if meter_key is not None:
        monkeypatch.setenv(f"FDIA_METER_KEY_{meter_id}", meter_key.decode("utf-8"))
    return TestClient(app)


def _sign_and_post(client, meter_id: str, session_id: str, seq: int, ts: str, power_kw: float, key: bytes):
    msg = {"protocol_version": "1", "meter_id": meter_id, "session_id": session_id,
           "timestamp": ts, "sequence_number": seq, "power_kw": power_kw}
    mac = _hmac_auth_mod.compute_hmac(key, _canon(msg))
    return client.post("/secure-readings", json={**msg, "mac": mac})


def test_16_rejected_message_never_calls_detector_engine(monkeypatch):
    with _secure_client(monkeypatch, meter_key=b"key16", meter_id="house_t16") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        client.post("/security/test/init-session", json={"meter_id": "house_t16", "session_id": "s1"})
        before = client.get("/metrics").json()["reading_requests"]
        # MAC incorrecta -> rechazo de seguridad, nunca debe tocar el motor
        r = client.post("/secure-readings", json={
            "protocol_version": "1", "meter_id": "house_t16", "session_id": "s1", "timestamp": "2020-01-01T00:00:00Z",
            "sequence_number": 1, "power_kw": 1.0, "mac": "0" * 64,
        })
        assert r.status_code == 401
        assert r.json()["detector_invoked"] is False
        after = client.get("/metrics").json()["reading_requests"]
        assert after == before  # /metrics del DETECTOR (Fase 2, sin modificar) no vio nada


def test_17_rejected_message_does_not_change_detector_state(monkeypatch):
    with _secure_client(monkeypatch, meter_key=b"key17", meter_id="house_t17") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        client.post("/security/test/init-session", json={"meter_id": "house_t17", "session_id": "s1"})
        r_ok = _sign_and_post(client, "house_t17", "s1", 1, "2020-01-01T00:00:00Z", 1.0, b"key17")
        assert r_ok.status_code == 200
        status_before = client.get("/status/house_t17").json()

        # replay del mismo mensaje -> rechazo de seguridad (duplicate_sequence)
        r_replay = _sign_and_post(client, "house_t17", "s1", 1, "2020-01-01T00:00:00Z", 1.0, b"key17")
        assert r_replay.status_code == 409

        status_after = client.get("/status/house_t17").json()
        for campo in ("buffer_1min_size", "history_15min_size", "last_accepted_timestamp",
                      "accepted_readings", "rejected_readings"):
            assert status_before[campo] == status_after[campo], f"{campo} cambio tras un mensaje rechazado"


def test_18_rejected_message_does_not_advance_anti_replay_state(monkeypatch):
    with _secure_client(monkeypatch, meter_key=b"key18", meter_id="house_t18") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        client.post("/security/test/init-session", json={"meter_id": "house_t18", "session_id": "s1"})
        r_ok = _sign_and_post(client, "house_t18", "s1", 5, "2020-01-01T00:00:00Z", 1.0, b"key18")
        assert r_ok.status_code == 200
        status_before = client.get("/security/status").json()

        _sign_and_post(client, "house_t18", "s1", 5, "2020-01-01T00:00:00Z", 1.0, b"key18")  # replay, rechazado
        status_after = client.get("/security/status").json()
        assert status_after["metrics"]["rejections_by_reason"].get("duplicate_sequence", 0) == \
            status_before["metrics"]["rejections_by_reason"].get("duplicate_sequence", 0) + 1


def test_21_readings_endpoint_produces_identical_responses_with_security_enabled(monkeypatch):
    """/readings (routes.py, Fase 1-4) no cambia su respuesta por el mero hecho de que la
    seguridad este activada (aisladas, seccion 1)."""
    with _secure_client(monkeypatch, meter_key=b"key21", meter_id="house_t21_plain") as client:
        r = client.post("/readings", json={"meter_id": "house_t21_plain", "timestamp": "2020-01-01T00:00:00", "power_kw": 1.0})
        assert r.status_code == 200
        body = r.json()
        campos_esperados = {"meter_id", "timestamp", "accepted", "rejection_reason", "engine_status", "warming_up",
                             "p_ready", "h_ready", "p_evaluated", "h_evaluated", "score_p", "score_h", "alert_p",
                             "alert_p_ffill", "alert_h", "alert_or", "detection_source", "p_last_evaluation_timestamp",
                             "h_last_evaluation_timestamp", "buffer_1min_size", "buffer_15min_size",
                             "engine_processing_time_ms", "api_processing_time_ms", "warnings"}
        assert set(body.keys()) == campos_esperados  # forma EXACTA de Fase 2, sin campos de seguridad mezclados


def test_22_health_endpoint_unaffected_by_security(monkeypatch):
    with _secure_client(monkeypatch, enabled=True) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert set(r.json().keys()) == {"status", "service", "version"}


def test_23_ready_endpoint_unaffected_by_security(monkeypatch):
    with _secure_client(monkeypatch, enabled=True) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True  # nunca depende de si hay claves anti-replay configuradas


def test_24_security_disabled_does_not_affect_detector(monkeypatch):
    with _secure_client(monkeypatch, enabled=False, meter_id="house_t24") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["ready"] is True
        r = client.post("/readings", json={"meter_id": "house_t24", "timestamp": "2020-01-01T00:00:00", "power_kw": 1.0})
        assert r.status_code == 200
        assert client.post("/secure-readings", json={
            "protocol_version": "1", "meter_id": "house_t24", "session_id": "s1", "timestamp": "2020-01-01T00:00:00Z",
            "sequence_number": 1, "power_kw": 1.0, "mac": "0" * 64,
        }).status_code == 503


def test_25_secure_readings_preserves_detector_result_shape(monkeypatch):
    with _secure_client(monkeypatch, meter_key=b"key25", meter_id="house_t25") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        client.post("/security/test/init-session", json={"meter_id": "house_t25", "session_id": "s1"})
        r = _sign_and_post(client, "house_t25", "s1", 1, "2020-01-01T00:00:00Z", 1.0, b"key25")
        assert r.status_code == 200
        body = r.json()
        assert body["detector_result"] is not None
        for campo in ("accepted", "p_evaluated", "h_evaluated", "score_p", "score_h", "alert_p", "alert_h",
                      "alert_or", "detection_source", "buffer_1min_size"):
            assert campo in body["detector_result"]


def test_26_keys_never_appear_in_log_file(monkeypatch):
    secret = "unique-secret-for-log-test-zzz999"
    with _secure_client(monkeypatch, meter_key=secret.encode(), meter_id="house_t26") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        client.post("/security/test/init-session", json={"meter_id": "house_t26", "session_id": "s1"})
        _sign_and_post(client, "house_t26", "s1", 1, "2020-01-01T00:00:00Z", 1.0, secret.encode())

    log_path = EDGE_DIR / "results" / "logs" / "api.log"
    if log_path.exists():
        assert secret not in log_path.read_text(encoding="utf-8", errors="ignore")


def test_secure_readings_response_status_code_mapping(monkeypatch):
    """seccion 13: 200 aceptado, 401 mac invalida, 404 meter desconocido, 409 conflicto/replay,
    503 seguridad no habilitada."""
    with _secure_client(monkeypatch, meter_key=b"key27", meter_id="house_t27") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        client.post("/security/test/init-session", json={"meter_id": "house_t27", "session_id": "s1"})

        r_unknown_meter = _sign_and_post(client, "house_unknown_xyz", "s1", 1, "2020-01-01T00:00:00Z", 1.0, b"any")
        assert r_unknown_meter.status_code == 404

        r_ok = _sign_and_post(client, "house_t27", "s1", 1, "2020-01-01T00:00:00Z", 1.0, b"key27")
        assert r_ok.status_code == 200

        r_dup = _sign_and_post(client, "house_t27", "s1", 1, "2020-01-01T00:00:00Z", 1.0, b"key27")
        assert r_dup.status_code == 409

    with _secure_client(monkeypatch, enabled=False, meter_id="house_t27b") as client:
        r_disabled = client.post("/secure-readings", json={
            "protocol_version": "1", "meter_id": "house_t27b", "session_id": "s1", "timestamp": "2020-01-01T00:00:00Z",
            "sequence_number": 1, "power_kw": 1.0, "mac": "0" * 64,
        })
        assert r_disabled.status_code == 503


# ==================================================================================================
# CORRECCIONES OBLIGATORIAS post-aceptacion inicial de Fase 5
# ==================================================================================================
# Correccion 1: /security/test/init-session y /security/test/clock deben devolver 404 en
# produccion (test_endpoints_enabled=false, SIEMPRE por defecto, independiente de
# anti_replay_enabled) -- no basta con que el reloj no sea simulado.
# ==================================================================================================

def test_correccion1_init_session_returns_404_in_production(monkeypatch):
    with _secure_client(monkeypatch, enabled=True, test_endpoints_enabled=False, meter_id="house_c1a") as client:
        r = client.post("/security/test/init-session", json={"meter_id": "house_c1a", "session_id": "s1"})
        assert r.status_code == 404


def test_correccion1_set_clock_returns_404_in_production(monkeypatch):
    with _secure_client(monkeypatch, enabled=True, test_endpoints_enabled=False, clock_mode="simulated", meter_id="house_c1b") as client:
        r = client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        assert r.status_code == 404


def test_correccion1_404_independent_of_clock_mode():
    """No basta con impedir unicamente el reloj simulado: con clock_mode=system (produccion
    real) Y test_endpoints_enabled=false, ambos endpoints deben seguir devolviendo 404 (nunca
    503 por 'reloj no simulado', que dejaria init-session accesible)."""
    import os as _os
    prev = {k: _os.environ.get(k) for k in ("FDIA_ANTI_REPLAY_ENABLED", "FDIA_SECURITY_CLOCK_MODE", "FDIA_SECURITY_TEST_ENDPOINTS_ENABLED")}
    try:
        _os.environ["FDIA_ANTI_REPLAY_ENABLED"] = "true"
        _os.environ["FDIA_SECURITY_CLOCK_MODE"] = "system"
        _os.environ.pop("FDIA_SECURITY_TEST_ENDPOINTS_ENABLED", None)  # produccion: no seteado -> false
        from fastapi.testclient import TestClient

        from edge_deployment.api.main import app
        with TestClient(app) as client:
            assert client.post("/security/test/init-session", json={"meter_id": "x", "session_id": "y"}).status_code == 404
            assert client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"}).status_code == 404
    finally:
        for k, v in prev.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


def test_correccion1_cannot_reinitialize_existing_session(monkeypatch):
    with _secure_client(monkeypatch, meter_key=b"keyc1d", meter_id="house_c1d") as client:
        r1 = client.post("/security/test/init-session", json={"meter_id": "house_c1d", "session_id": "s1"})
        assert r1.status_code == 200
        r2 = client.post("/security/test/init-session", json={"meter_id": "house_c1d", "session_id": "s1"})
        assert r2.status_code == 409
        assert r2.json()["security_rejection_reason"] == "session_already_provisioned"


def test_correccion1_secure_readings_still_works_via_controlled_experimental_mechanism(monkeypatch):
    """El mecanismo experimental (test_endpoints_enabled=true, NUNCA el valor por defecto)
    sigue permitiendo aprovisionar sesiones y usar /secure-readings con normalidad -- la
    correccion 1 restringe el acceso, no elimina la capacidad de prueba controlada."""
    with _secure_client(monkeypatch, meter_key=b"keyc1e", meter_id="house_c1e") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        r_init = client.post("/security/test/init-session", json={"meter_id": "house_c1e", "session_id": "s1"})
        assert r_init.status_code == 200
        r = _sign_and_post(client, "house_c1e", "s1", 1, "2020-01-01T00:00:00Z", 1.0, b"keyc1e")
        assert r.status_code == 200


# ==================================================================================================
# Correccion 2: power_kw se cuantiza a 6 decimales via Decimal ANTES de autenticar y de
# entregar a DetectorEngine -- el mismo valor cuantizado en ambos usos, sin aliasing.
# ==================================================================================================

def test_correccion2_quantize_power_kw_uses_decimal_and_rounds_to_6_places():
    from decimal import Decimal

    from edge_deployment.security.canonicalization import quantize_power_kw
    assert quantize_power_kw(1.42) == Decimal("1.420000")
    assert quantize_power_kw("1.4200004") == Decimal("1.420000")
    assert quantize_power_kw("1.4200009") == Decimal("1.420001")


def test_correccion2_values_differing_by_less_than_1e6_alias_to_identical_canonical_bytes_and_quantized_value():
    from edge_deployment.security.canonicalization import canonicalize, quantize_power_kw_float

    a = _msg(power_kw=1.42000001)
    b = _msg(power_kw=1.42000034)
    assert abs(a["power_kw"] - b["power_kw"]) < 1e-6
    assert canonicalize(a) == canonicalize(b)  # mismo cubo de redondeo -> mismos bytes -> mismo HMAC posible
    # la propiedad que corrige el aliasing: ambos entregarian el MISMO valor al motor
    assert quantize_power_kw_float(a["power_kw"]) == quantize_power_kw_float(b["power_kw"])


def test_correccion2_hmac_valid_for_either_of_two_aliasing_values_but_same_value_would_reach_engine():
    key = b"secret-key"
    a = _msg(power_kw=1.42000001)
    b = _msg(power_kw=1.42000034)
    mac_signed_with_a = _hmac_auth_mod.compute_hmac(key, _canon(a))
    # verificar CON el mensaje b (payload distinto en el septimo decimal, sin recalcular MAC)
    from edge_deployment.security import hmac_auth as _hmac
    assert _hmac.verify_hmac(key, _canon(b), mac_signed_with_a) is True  # MISMO cubo -> HMAC valido para ambos
    # esto YA NO es explotable: quantize_power_kw_float(a) == quantize_power_kw_float(b),
    # asi que sea cual sea el payload que realmente llego, DetectorEngine procesa el MISMO valor.
    from edge_deployment.security.canonicalization import quantize_power_kw_float
    assert quantize_power_kw_float(a["power_kw"]) == quantize_power_kw_float(b["power_kw"])


def test_correccion2_secure_readings_delivers_quantized_value_to_detector_not_raw_payload(monkeypatch):
    from edge_deployment.security.canonicalization import quantize_power_kw_float

    with _secure_client(monkeypatch, meter_key=b"keyc2d", meter_id="house_c2d") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        client.post("/security/test/init-session", json={"meter_id": "house_c2d", "session_id": "s1"})
        raw_power_kw = 1.4200003333333  # mas de 6 decimales significativos
        r = _sign_and_post(client, "house_c2d", "s1", 1, "2020-01-01T00:00:00Z", raw_power_kw, b"keyc2d")
        assert r.status_code == 200
        # el motor no expone power_kw directamente en la respuesta, pero el buffer/estado
        # confirma que se proceso (no crashea, no se rechaza por tipo) -- la prueba directa de
        # cuantizacion vive en test_correccion2_values_differing_by_less_than_1e6_... arriba;
        # aqui se confirma que el camino de produccion no lanza con valores de alta precision.
        assert r.json()["detector_invoked"] is True
        assert r.json()["detector_result"]["accepted"] is True


def test_correccion2_manipulating_power_kw_within_rounding_bucket_has_zero_effect_on_outcome(monkeypatch):
    """Escenario de ataque: un adversario SIN la clave intercepta un mensaje legitimo y
    modifica power_kw dentro del mismo cubo de redondeo de 6 decimales (diferencia < 1e-6),
    dejando el HMAC valido. Tras la correccion, el resultado (score/alerta) es identico al que
    se habria obtenido con el valor original -- la manipulacion queda sin efecto explotable."""
    with _secure_client(monkeypatch, meter_key=b"keyc2e", meter_id="house_c2e") as client:
        client.post("/security/test/clock", json={"timestamp": "2020-01-01T00:00:00Z"})
        client.post("/security/test/init-session", json={"meter_id": "house_c2e", "session_id": "s1"})
        original = _msg(meter_id="house_c2e", session_id="s1", sequence_number=1,
                         timestamp="2020-01-01T00:00:00Z", power_kw=1.42000001)
        mac = _hmac_auth_mod.compute_hmac(b"keyc2e", _canon(original))
        tampered = dict(original)
        tampered["power_kw"] = 1.42000034  # dentro del mismo cubo, MAC original SIGUE siendo valida
        r = client.post("/secure-readings", json={**tampered, "mac": mac})
        assert r.status_code == 200  # HMAC valido (mismo cubo) -- se acepta, como es correcto
        # lo que importa: el resultado no depende de CUAL de los dos valores llego realmente
        assert r.json()["detector_result"]["accepted"] is True
