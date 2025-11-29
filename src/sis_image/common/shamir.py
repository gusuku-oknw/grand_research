"""Shamir secret sharing utilities for GF(257) coefficients."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

FIELD_PRIME = 257


def _lagrange_basis_at_zero(xs: Sequence[int]) -> np.ndarray:
    """Compute Lagrange basis coefficients evaluated at zero."""
    xs_mod = [int(x) % FIELD_PRIME for x in xs]
    k = len(xs_mod)
    coeffs = np.zeros(k, dtype=np.int64)
    for j in range(k):
        numerator, denominator = 1, 1
        xj = xs_mod[j]
        for m, xm in enumerate(xs_mod):
            if m == j:
                continue
            numerator = (numerator * (-xm)) % FIELD_PRIME
            denominator = (denominator * (xj - xm)) % FIELD_PRIME
        inv = pow(denominator % FIELD_PRIME, FIELD_PRIME - 2, FIELD_PRIME)
        coeffs[j] = (numerator * inv) % FIELD_PRIME
    return coeffs.astype(np.uint16)


def shamir_share_bytes(
    secret_bytes: Iterable[int],
    k: int,
    n: int,
    rng: np.random.Generator | None = None,
) -> Dict[int, List[int]]:
    """Produce k-of-n Shamir shares for byte-sized secrets."""
    if not (2 <= k <= n):
        raise ValueError("Require 2 <= k <= n for Shamir sharing.")
    data = [int(b) % FIELD_PRIME for b in secret_bytes]
    length = len(data)
    if rng is None:
        rng = np.random.default_rng()
    coeffs = [
        [int(rng.integers(0, FIELD_PRIME)) for _ in range(k - 1)]
        for _ in range(length)
    ]
    shares: Dict[int, List[int]] = {x: [0] * length for x in range(1, n + 1)}
    for idx, secret in enumerate(data):
        for x in range(1, n + 1):
            y = secret
            power = 1
            for coeff in coeffs[idx]:
                power = (power * x) % FIELD_PRIME
                y = (y + coeff * power) % FIELD_PRIME
            shares[x][idx] = int(y)
    return shares


def shamir_recover_bytes(subshares: Dict[int, Iterable[int]]) -> List[int]:
    """Recover byte-sized secrets from a subset of shares."""
    if not subshares:
        return []
    xs = sorted(int(x) for x in subshares.keys())
    basis = _lagrange_basis_at_zero(xs).astype(np.uint32)
    rows = {x: list(subshares[x]) for x in xs}
    length = len(next(iter(rows.values())))
    recovered: List[int] = [0] * length
    for i in range(length):
        accum = 0
        for j, x in enumerate(xs):
            row = rows[x]
            if len(row) != length:
                raise ValueError("Share length mismatch during recovery.")
            accum = (accum + (row[i] % FIELD_PRIME) * basis[j]) % FIELD_PRIME
        if accum == FIELD_PRIME - 1:
            accum = 0
        recovered[i] = int(accum)
    return recovered


def shamir_share_flat_bytes(
    flat_u8: np.ndarray,
    k: int,
    n: int,
    rng: np.random.Generator | None = None,
) -> Dict[int, np.ndarray]:
    """Vectorised share generation for flattened uint8 arrays."""
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(flat_u8, dtype=np.uint8).reshape(-1)
    coeffs = [
        rng.integers(0, FIELD_PRIME, size=data.size, dtype=np.uint16)
        for _ in range(k - 1)
    ]
    shares: Dict[int, np.ndarray] = {}
    base = data.astype(np.uint16)
    for x in range(1, n + 1):
        y = base.astype(np.uint32)
        power = np.ones(data.size, dtype=np.uint32)
        for coeff in coeffs:
            power = (power * x) % FIELD_PRIME
            y = (y + coeff.astype(np.uint32) * power) % FIELD_PRIME
        shares[x] = y.astype(np.uint16)
    return shares


def shamir_recover_flat_bytes(subshares: Dict[int, np.ndarray]) -> np.ndarray:
    """Recover flattened uint8 arrays from Shamir shares."""
    if not subshares:
        raise ValueError("No shares provided for recovery.")
    xs = sorted(int(x) for x in subshares.keys())
    arrays = [np.asarray(subshares[x], dtype=np.uint16).reshape(-1) for x in xs]
    length = arrays[0].size
    if any(arr.size != length for arr in arrays):
        raise ValueError("All shares must have the same length.")
    basis = _lagrange_basis_at_zero(xs).astype(np.uint32)
    acc = np.zeros(length, dtype=np.uint32)
    for coeff, arr in zip(basis, arrays):
        acc = (acc + arr.astype(np.uint32) * coeff) % FIELD_PRIME
    acc = np.where(acc == FIELD_PRIME - 1, 0, acc)
    return acc.astype(np.uint8)


__all__ = [
    "FIELD_PRIME",
    "shamir_share_bytes",
    "shamir_recover_bytes",
    "shamir_share_flat_bytes",
    "shamir_recover_flat_bytes",
]
