"""Fase 5, seccion 8: serializacion canonica determinista del mensaje seguro -- es el
contenido EXACTO que cubre el HMAC (`hmac_auth.py`). Nunca pickle/joblib/binario dependiente
de Python (seccion 8, explicito).

Formato canonico:
  - JSON compacto (`separators=(",", ":")`, sin espacios variables), UTF-8, `ensure_ascii=True`.
  - Orden FIJO de campos (el orden de insercion del dict, nunca `sort_keys=True` -- el orden
    es parte del contrato, no una funcion de las claves): protocol_version, meter_id,
    session_id, timestamp, sequence_number, power_kw.
  - `timestamp` normalizado a UTC, formato `%Y-%m-%dT%H:%M:%SZ` (sin fraccion de segundo --
    la resolucion nativa de las lecturas es de 1 minuto en todo el proyecto).
  - `sequence_number` como entero JSON nativo (representacion exacta, sin ambiguedad).
  - `power_kw` como CADENA de texto con 6 decimales fijos, nunca como float nativo de JSON.
  - `protocol_version`/`meter_id`/`session_id` como cadenas de texto tal cual.

CORRECCION OBLIGATORIA (post-aceptacion inicial de Fase 5) -- politica explicita de
cuantizacion de `power_kw`, para eliminar un aliasing real entre "valor autenticado" y "valor
procesado":

  Antes: `_normalize_power_kw` formateaba `power_kw` a 6 decimales SOLO para los bytes
  canonicos (HMAC), pero `secure_routes.py` seguia entregando el valor CRUDO (sin cuantizar) a
  `DetectorEngine.ingest()`. Dos floats distintos que redondean al MISMO string de 6 decimales
  (p.ej. 1.4200004 y 1.4200009, diferencia < 1e-6) producian bytes canonicos identicos y, por
  tanto, el MISMO HMAC valido -- pero si un atacante (SIN conocer la clave) intercambiaba uno
  por otro en transito, el motor procesaba un valor sutilmente distinto del que el emisor
  legitimo penso que estaba autenticando. No hacia falta la clave para explotarlo: solo hacia
  falta que el valor alterado cayera en el mismo "cubo" de redondeo de 6 decimales.

  Ahora: `quantize_power_kw()` usa `decimal.Decimal` (nunca aritmetica binaria de coma
  flotante) para cuantizar a exactamente 6 decimales con redondeo `ROUND_HALF_EVEN`
  (determinista, sin sesgo). Ese MISMO valor cuantizado es el que:
    1. se usa para construir los bytes canonicos (HMAC), y
    2. se entrega a `DetectorEngine.ingest()` (ver `secure_routes.py`, que llama a
       `quantize_power_kw_float()` en vez de usar `payload.power_kw` crudo).
  Como ambos usos comparten la MISMA fuente cuantizada, dos valores que producen el mismo HMAC
  producen, por construccion, el mismo valor procesado -- el aliasing deja de ser explotable
  (no es que se detecte; es que ya no existe la discrepancia que lo hacia posible).

Vectores de prueba: mismo mensaje -> mismos bytes; mismo mensaje+clave -> mismo HMAC; cambiar
un campo -> HMAC distinto; dos power_kw con diferencia < 1e-6 -> mismo HMAC Y mismo valor
cuantizado entregado al motor (nunca el valor crudo de ninguno de los dos).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

CANONICAL_FIELD_ORDER = ("protocol_version", "meter_id", "session_id", "timestamp", "sequence_number", "power_kw")

POWER_KW_DECIMALS = 6
_POWER_KW_QUANTUM = Decimal("1").scaleb(-POWER_KW_DECIMALS)  # Decimal("0.000001")


class CanonicalizationError(ValueError):
    pass


def _normalize_timestamp(ts) -> str:
    if isinstance(ts, str):
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError as e:
            raise CanonicalizationError(f"timestamp no parseable: {ts!r}") from e
    elif isinstance(ts, datetime):
        dt = ts
    else:
        raise CanonicalizationError(f"tipo de timestamp no soportado: {type(ts)!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def quantize_power_kw(power_kw) -> Decimal:
    """Cuantiza a exactamente 6 decimales via Decimal (ROUND_HALF_EVEN, determinista). Se
    construye desde `str(power_kw)` (no desde el float directamente) para partir de la misma
    representacion decimal "corta" que ya usan `json`/`repr` de Python -- evita heredar el
    ruido binario interno de IEEE-754 (p.ej. `Decimal(1.42)` arrastra digitos espureos que
    `Decimal(str(1.42))` no)."""
    try:
        d = Decimal(str(power_kw))
    except (InvalidOperation, TypeError, ValueError) as e:
        raise CanonicalizationError(f"power_kw no es numerico: {power_kw!r}") from e
    return d.quantize(_POWER_KW_QUANTUM, rounding=ROUND_HALF_EVEN)


def quantize_power_kw_float(power_kw) -> float:
    """Valor cuantizado como float nativo -- este es el UNICO valor que
    `edge_deployment/api/secure_routes.py` debe entregar a `DetectorEngine.ingest()`, nunca el
    valor crudo del payload. Ver la nota de "correccion obligatoria" arriba."""
    return float(quantize_power_kw(power_kw))


def _normalize_power_kw(power_kw) -> str:
    return format(quantize_power_kw(power_kw), f"0.{POWER_KW_DECIMALS}f")


def canonicalize(message: dict) -> bytes:
    """Construye los bytes canonicos que se firman/verifican. Lanza `CanonicalizationError`
    si falta algun campo obligatorio o tiene un tipo no soportado -- nunca canonicaliza un
    mensaje incompleto en silencio."""
    faltantes = [f for f in CANONICAL_FIELD_ORDER if f not in message]
    if faltantes:
        raise CanonicalizationError(f"faltan campos obligatorios para canonicalizar: {faltantes}")

    try:
        sequence_number = int(message["sequence_number"])
    except (TypeError, ValueError) as e:
        raise CanonicalizationError(f"sequence_number no es entero: {message['sequence_number']!r}") from e

    ordered = {
        "protocol_version": str(message["protocol_version"]),
        "meter_id": str(message["meter_id"]),
        "session_id": str(message["session_id"]),
        "timestamp": _normalize_timestamp(message["timestamp"]),
        "sequence_number": sequence_number,
        "power_kw": _normalize_power_kw(message["power_kw"]),
    }
    return json.dumps(ordered, ensure_ascii=True, separators=(",", ":"), sort_keys=False).encode("utf-8")
