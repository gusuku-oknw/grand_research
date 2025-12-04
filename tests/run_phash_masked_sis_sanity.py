"""
Lightweight sanity checks for phash_masked_sis without pytest.

Runs:
 1) pHash preservation between original and masked dummy image
 2) Two-level Shamir round-trip

Exit code 0 on success; raises on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from phash_masked_sis import (
    PHashConfig,
    compute_phash,
    make_phash_preserving_dummy,
    TwoLevelShamirScheme,
)


def make_noise_image(size: int = 64, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def check_phash_preservation() -> None:
    cfg = PHashConfig(hash_size=8, highfreq_factor=4)
    img = make_noise_image()
    dummy = make_phash_preserving_dummy(img, cfg, seed=123)
    h1 = compute_phash(img, cfg)
    h2 = compute_phash(dummy, cfg)
    if not np.array_equal(h1, h2):
        raise AssertionError("pHash bits are not preserved in dummy image")
    print("OK: pHash preserved between original and masked dummy.")
    out_dir = Path("output/figures/phash_masked_sis")
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / "original_noise.png")
    dummy.save(out_dir / "dummy_noise.png")


def check_twolevel_shamir() -> None:
    s1 = b"dummy_secret_low"
    s2 = b"real_secret_high"
    scheme = TwoLevelShamirScheme(n=5, k1=2, k2=4)
    shares = scheme.split(s1, s2)

    recovered_s1 = scheme.recover_level1(shares[:2])
    if recovered_s1 != s1:
        raise AssertionError("Level1 secret mismatch")

    recovered_s2 = scheme.recover_level2(shares[:4])
    if recovered_s2 != s2:
        raise AssertionError("Level2 secret mismatch")

    print("OK: Two-level Shamir round-trip for S1 and S2.")


def main() -> None:
    check_phash_preservation()
    check_twolevel_shamir()
    print("All phash_masked_sis sanity checks passed.")


if __name__ == "__main__":
    main()
