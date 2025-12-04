"""
phash_masked_sis

pHash と整合する「マスク（不可視化）画像」を低層シークレットに、
本物画像を高層シークレットに置く 2 階層 Shamir SIS のミニパッケージ。

依存:
    - Pillow
    - numpy
"""

from .phash_core import PHashConfig, compute_phash, phash_core, dct2, idct2
from .dummy_image import make_phash_preserving_dummy
from .sis_twolevel import TwoLevelShamirScheme, TwoLevelShare

__all__ = [
    "PHashConfig",
    "compute_phash",
    "phash_core",
    "dct2",
    "idct2",
    "make_phash_preserving_dummy",
    "TwoLevelShamirScheme",
    "TwoLevelShare",
]
