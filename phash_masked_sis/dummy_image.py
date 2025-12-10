from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .phash_core import PHashConfig, dct2, idct2, phash_core


def _build_lowfreq_from_bits(bits: np.ndarray, base_alpha: float = 24.0, margin: float = 8.0) -> np.ndarray:
    """
    pHash のビットパターン bits に対応する低周波ブロックを作る。
    平均値が 0 付近になるよう、1→+A, 0→-B を割り当て、さらに margin で符号を強調する。
    """
    bits = bits.astype(np.uint8)
    n1 = int(bits.sum())
    n_total = bits.size
    n0 = n_total - n1

    if n1 == 0:
        lf = -np.ones_like(bits, dtype=np.float64) * (base_alpha + margin)
        return lf - lf.mean()
    if n0 == 0:
        lf = np.ones_like(bits, dtype=np.float64) * (base_alpha + margin)
        return lf - lf.mean()

    B = base_alpha
    A = base_alpha * (n0 / n1)  # mean(lf)=0 となるよう調整
    lf = np.where(bits == 1, A + margin, -(B + margin)).astype(np.float64)
    lf -= lf.mean()
    return lf


def _reinforce_margin(lf: np.ndarray, bits: np.ndarray, margin: float) -> np.ndarray:
    """ビット符号を保ったまま、閾値ずれを防ぐために最低振幅 margin を強制する。"""
    lf = lf - lf.mean()
    signed = np.where(bits == 1, 1.0, -1.0)
    lf = signed * np.maximum(np.abs(lf), margin)
    lf -= lf.mean()
    return lf


def make_phash_preserving_dummy(
    img: Image.Image,
    cfg: PHashConfig = PHashConfig(),
    base_alpha: float = 28.0,
    highfreq_scale: float = 12.0,
    margin: float = 10.0,
    reinforce_steps: int = 2,
    seed: Optional[int] = None,
    max_trials: int = 20,
    return_debug: bool = False,
) -> Image.Image | tuple[Image.Image, dict[str, object]]:
    """
    pHash は元画像と一致しつつ、人間にはノイズに見えるダミー画像を生成する。

    - 低周波 8x8 は pHash ビットパターンに対応する符号のみ持たせる
    - 高周波は疑似乱数ノイズで埋める
    - IDCT して 32x32 → 元サイズにアップサンプル
    """
    bits, _ = phash_core(img, cfg)
    orig_hash = bits.flatten()
    rng = np.random.default_rng(seed)

    best_img = None
    best_dist = 65  # larger than max possible
    best_debug: dict[str, object] | None = None

    for trial in range(max_trials):
        # 低周波ブロックを構築し、少しずつ強調してビット反転を防ぐ
        lf_dummy = _build_lowfreq_from_bits(bits, base_alpha=base_alpha, margin=margin)
        for _ in range(reinforce_steps):
            lf_dummy = _reinforce_margin(lf_dummy, bits, margin=margin)

        size = cfg.hash_size * cfg.highfreq_factor
        C = np.zeros((size, size), dtype=np.float64)
        C[: cfg.hash_size, : cfg.hash_size] = lf_dummy

        # 高周波ノイズ
        if size > cfg.hash_size:
            C[cfg.hash_size :, :] = rng.normal(0.0, highfreq_scale, size=(size - cfg.hash_size, size))
            C[:, cfg.hash_size :] = rng.normal(0.0, highfreq_scale, size=(size, size - cfg.hash_size))
            C[: cfg.hash_size, : cfg.hash_size] = lf_dummy

        spatial = idct2(C)
        spatial -= spatial.min()
        max_val = spatial.max()
        if max_val > 0:
            spatial = spatial / max_val * 255.0
        spatial = np.clip(spatial, 0, 255).astype(np.uint8)

        img_small = Image.fromarray(spatial, mode="L")
        img_out = img_small.resize(img.size, Image.BICUBIC)

        # pHash 距離を評価し、最良のものを採用
        from phash_masked_sis import compute_phash  # 遅延インポートで循環回避

        dist = int(np.count_nonzero(compute_phash(img_out, cfg) ^ orig_hash))
        if dist < best_dist:
            best_dist = dist
            best_img = img_out
            if return_debug:
                best_debug = {
                    "bits": bits.copy(),
                    "lf_final": lf_dummy.copy(),
                    "spatial_small": spatial.copy(),
                    "dist": dist,
                }
        if dist == 0:
            break

    if best_img is None:
        best_img = img_out
    if return_debug:
        return best_img, best_debug or {}
    return best_img
