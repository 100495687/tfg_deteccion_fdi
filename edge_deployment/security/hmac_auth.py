"""Fase 5, seccion 9: HMAC-SHA-256 con la biblioteca estandar UNICAMENTE (`hmac`,
`hashlib.sha256`, `hmac.compare_digest`) -- nunca `==` para comparar MAC (vulnerable a ataques
de temporizacion). Ver `key_provider.py` para el origen de `meter_key` -- nunca hardcodeada,
nunca en logs, nunca en el repositorio ni en la imagen.
"""
from __future__ import annotations

import hashlib
import hmac


def compute_hmac(meter_key: bytes, canonical_message_bytes: bytes) -> str:
    """mac = HMAC-SHA256(meter_key, canonical_message_bytes) -- devuelve hexadecimal en
    minusculas (`hexdigest()`)."""
    return hmac.new(meter_key, canonical_message_bytes, hashlib.sha256).hexdigest()


def verify_hmac(meter_key: bytes, canonical_message_bytes: bytes, mac_hex: str) -> bool:
    """Comparacion en tiempo constante via `hmac.compare_digest` -- nunca `==`."""
    expected = compute_hmac(meter_key, canonical_message_bytes)
    try:
        return hmac.compare_digest(expected, mac_hex)
    except TypeError:
        # mac_hex con un tipo/codificacion inesperada (p.ej. no str) -- se trata como MAC
        # invalido, nunca como excepcion sin capturar hacia el llamador HTTP.
        return False
