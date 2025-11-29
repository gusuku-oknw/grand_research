"""Banded HMAC token helpers for private candidate pre-selection."""

from __future__ import annotations

import hashlib
import hmac
from typing import List


def split_bands(hash64: int, bands: int) -> List[int]:
    """Split a 64-bit integer into evenly sized bands."""
    if 64 % bands != 0:
        raise ValueError("Number of bands must divide 64.")
    band_bits = 64 // bands
    mask = (1 << band_bits) - 1
    out: List[int] = []
    for idx in range(bands):
        shift = 64 - (idx + 1) * band_bits
        out.append((hash64 >> shift) & mask)
    return out


def hmac_token(key: bytes, value_bytes: bytes, outlen: int) -> bytes:
    """Return a truncated HMAC-SHA256 token."""
    digest = hmac.new(key, value_bytes, hashlib.sha256).digest()
    return digest[:outlen]


__all__ = ["split_bands", "hmac_token"]
