from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

# シンプルな DCT/IDCT のためのキャッシュ
_DCT_CACHE: dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _get_dct_mats(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    サイズ n の 1D 正規直交 DCT-II 変換行列とその転置（逆行列）をキャッシュして返す。
    """
    if n in _DCT_CACHE:
        return _DCT_CACHE[n]

    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    scale = np.sqrt(2.0 / n) * np.ones_like(k, dtype=np.float64)
    scale[0] = np.sqrt(1.0 / n)
    dct_mat = scale * np.cos(np.pi * (2 * x + 1) * k / (2.0 * n))
    inv_mat = dct_mat.T  # 正規直交なので転置が逆行列
    _DCT_CACHE[n] = (dct_mat, inv_mat)
    return dct_mat, inv_mat


def dct2(a: np.ndarray) -> np.ndarray:
    """2D DCT（type-II 相当）: D * A * D^T"""
    assert a.shape[0] == a.shape[1], "Only square matrices are supported"
    n = a.shape[0]
    D, _ = _get_dct_mats(n)
    return D @ a @ D.T


def idct2(a: np.ndarray) -> np.ndarray:
    """2D 逆 DCT: D^{-1} * A * (D^{-1})^T"""
    assert a.shape[0] == a.shape[1], "Only square matrices are supported"
    n = a.shape[0]
    _, D_inv = _get_dct_mats(n)
    return D_inv @ a @ D_inv.T


@dataclass
class PHashConfig:
    """
    pHash の設定値。
    hash_size: 低周波ブロックの一辺（8 -> 64bit）
    highfreq_factor: DCT を計算する際の縮小サイズ係数 (8 * 4 = 32 など)
    """

    hash_size: int = 8
    highfreq_factor: int = 4


def _preprocess_image(img: Image.Image, cfg: PHashConfig) -> np.ndarray:
    """画像をグレースケール+リサイズして DCT 入力にする。"""
    size = cfg.hash_size * cfg.highfreq_factor
    img_gray = img.convert("L").resize((size, size), Image.BICUBIC)
    return np.asarray(img_gray, dtype=np.float64)


def phash_core(img: Image.Image, cfg: PHashConfig = PHashConfig()) -> Tuple[np.ndarray, np.ndarray]:
    """
    pHash 計算のコア。DCT を取り、左上の低周波ブロックからビットを作る。
    戻り値: (bits, lowfreq) それぞれ (hash_size, hash_size) の配列。
    """
    arr = _preprocess_image(img, cfg)
    d = dct2(arr)
    lf = d[: cfg.hash_size, : cfg.hash_size]
    mean_val = lf.mean()
    bits = (lf > mean_val).astype(np.uint8)
    return bits, lf


def compute_phash(img: Image.Image, cfg: PHashConfig = PHashConfig()) -> np.ndarray:
    """画像から pHash（64bit 相当）を平坦な {0,1} 配列で返す。"""
    bits, _ = phash_core(img, cfg)
    return bits.flatten()
