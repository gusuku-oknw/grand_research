"""Perceptual hash fusion helpers inspired by saliency-aware texture descriptors.

The approach combines the global DCT-based pHash with a lightweight local
descriptor that weights block averages by local gradient (a simplified proxy
for the saliency-enhanced LBP fusion described in the referenced literature).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from ..common.phash import phash64


def _block_descriptor(arr: np.ndarray, grid: int) -> Tuple[int, float]:
    """Compute a binary descriptor per block and return bit and weighted value."""
    height, width = arr.shape
    block_h = max(height // grid, 1)
    block_w = max(width // grid, 1)
    bits = 0
    weighted_value = 0.0
    idx = 0
    global_avg = float(arr.mean())
    for row in range(grid):
        for col in range(grid):
            top = row * block_h
            left = col * block_w
            block = arr[top : top + block_h, left : left + block_w]
            if block.size == 0:
                continue
            block_mean = float(block.mean())
            gy, gx = np.gradient(block.astype(np.float32))
            saliency = float(np.mean(np.hypot(gx, gy)))
            weighted = block_mean + saliency * 0.5
            bit = 1 if weighted >= global_avg else 0
            bits = (bits << 1) | bit
            weighted_value += weighted
            idx += 1
            if idx >= 64:
                break
        if idx >= 64:
            break
    return bits << (64 - idx) if idx else 0, weighted_value


def fusion_hash64(image_path: str, resize: int = 32, band: int = 8, grid: int = 8) -> int:
    """Return a 64-bit fused hash combining DCT (pHash) and local texture bits.

    Inspired by the “Perceptual Image Hashing Fusing Zernike Moments and
    Saliency-Based Local Binary Patterns” idea that mixes a global perceptual
    fingerprint with a saliency-weighted local descriptor to improve robustness.
    """
    base_hash = phash64(image_path, resize=resize, band=band)
    img = Image.open(image_path).convert("L")
    img = img.resize((resize, resize), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    local_bits, _ = _block_descriptor(arr, grid=grid)
    return base_hash ^ local_bits
