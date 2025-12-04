from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .phash_core import PHashConfig, dct2, idct2, phash_core


def _build_lowfreq_from_bits(bits: np.ndarray, base_alpha: float = 10.0) -> np.ndarray:
    """
    pHash のビットパターン bits に対応する低周波ブロックを作る。
    平均値が 0 付近になるよう、1→+A, 0→-B を割り当てる。
    """
    bits = bits.astype(np.uint8)
    n1 = int(bits.sum())
    n_total = bits.size
    n0 = n_total - n1

    if n1 == 0:
        return -base_alpha * np.ones_like(bits, dtype=np.float64)
    if n0 == 0:
        return base_alpha * np.ones_like(bits, dtype=np.float64)

    B = base_alpha
    A = base_alpha * (n0 / n1)  # mean(lf)=0 となるよう調整
    lf = np.where(bits == 1, A, -B).astype(np.float64)
    return lf


def make_phash_preserving_dummy(
    img: Image.Image,
    cfg: PHashConfig = PHashConfig(),
    base_alpha: float = 10.0,
    highfreq_scale: float = 20.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    pHash は元画像と一致しつつ、人間にはノイズに見えるダミー画像を生成する。

    - 低周波 8x8 は pHash ビットパターンに対応する符号のみ持たせる
    - 高周波は疑似乱数ノイズで埋める
    - IDCT して 32x32 → 元サイズにアップサンプル
    """
    bits, _ = phash_core(img, cfg)
    lf_dummy = _build_lowfreq_from_bits(bits, base_alpha=base_alpha)

    size = cfg.hash_size * cfg.highfreq_factor
    C = np.zeros((size, size), dtype=np.float64)
    C[: cfg.hash_size, : cfg.hash_size] = lf_dummy

    rng = np.random.default_rng(seed)
    if size > cfg.hash_size:
        C[cfg.hash_size :, :] = rng.normal(0.0, highfreq_scale, size=(size - cfg.hash_size, size))
        C[:, cfg.hash_size :] = rng.normal(0.0, highfreq_scale, size=(size, size - cfg.hash_size))
        C[: cfg.hash_size, : cfg.hash_size] = lf_dummy  # 念のため再上書き

    spatial = idct2(C)
    spatial -= spatial.min()
    max_val = spatial.max()
    if max_val > 0:
        spatial = spatial / max_val * 255.0
    spatial = np.clip(spatial, 0, 255).astype(np.uint8)

    img_small = Image.fromarray(spatial, mode="L")
    img_out = img_small.resize(img.size, Image.BICUBIC)
    return img_out
