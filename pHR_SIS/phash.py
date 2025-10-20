"""Perceptual hash helpers for SIS workflows."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from PIL import Image


def _dct_matrix(n: int) -> np.ndarray:
    """Return an orthonormal DCT-II matrix of size n."""
    k = np.arange(n)[:, None]
    grid = np.arange(n)[None, :]
    mat = np.cos(np.pi * (grid + 0.5) * k / n)
    mat[0, :] *= np.sqrt(1 / n)
    mat[1:, :] *= np.sqrt(2 / n)
    return mat


def _dct2(x: np.ndarray) -> np.ndarray:
    """Compute a 2D orthonormal DCT-II."""
    rows, cols = x.shape
    c_row = _dct_matrix(rows)
    c_col = _dct_matrix(cols)
    return c_row @ x @ c_col.T


def phash64(image_path: str, resize: int = 32, band: int = 8) -> int:
    """Compute a 64-bit perceptual hash for an image file."""
    img = Image.open(image_path).convert("L")
    img = img.resize((resize, resize), Image.BILINEAR)
    values = np.asarray(img, dtype=np.float32)
    dct = _dct2(values)
    low = dct[:band, :band]
    avg = float(low.mean())
    bits = (low.flatten() > avg).astype(np.uint8)
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def hamming64(a: int, b: int) -> int:
    """Return the Hamming distance between two 64 bit integers."""
    return (a ^ b).bit_count()


def hash64_to_bytes(h: int) -> List[int]:
    """Convert a 64-bit integer hash to eight GF(257)-compatible bytes."""
    return [(h >> (8 * (7 - i))) & 0xFF for i in range(8)]


def bytes_to_hash64(values: Iterable[int]) -> int:
    """Convert an iterable of eight bytes back into a 64-bit integer."""
    data = list(values)
    if len(data) != 8:
        raise ValueError("Expected 8 bytes for a 64-bit hash.")
    h = 0
    for b in data:
        h = (h << 8) | (b & 0xFF)
    return h


__all__ = [
    "phash64",
    "hamming64",
    "hash64_to_bytes",
    "bytes_to_hash64",
]
