import numpy as np
from PIL import Image

from phash_masked_sis import (
    PHashConfig,
    compute_phash,
    make_phash_preserving_dummy,
    TwoLevelShamirScheme,
)


def _make_noise_image(size: int = 64, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def test_dummy_preserves_phash_bits():
    cfg = PHashConfig(hash_size=8, highfreq_factor=4)
    img = _make_noise_image()
    dummy = make_phash_preserving_dummy(img, cfg, seed=123)

    h1 = compute_phash(img, cfg)
    h2 = compute_phash(dummy, cfg)

    # すべてのビットが一致することを確認
    assert np.array_equal(h1, h2), "pHash bits should be preserved in dummy image"


def test_twolevel_shamir_roundtrip(tmp_path):
    # 2階層 Shamir で S1/S2 を round-trip
    s1 = b"dummy_secret_low"
    s2 = b"real_secret_high"
    scheme = TwoLevelShamirScheme(n=5, k1=2, k2=4)
    shares = scheme.split(s1, s2)

    recovered_s1 = scheme.recover_level1(shares[:2])
    assert recovered_s1 == s1

    recovered_s2 = scheme.recover_level2(shares[:4])
    assert recovered_s2 == s2
