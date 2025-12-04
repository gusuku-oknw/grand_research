from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Sequence, Tuple

import secrets
from PIL import Image

from .dummy_image import make_phash_preserving_dummy
from .phash_core import PHashConfig

# 大きめの素数（実験用）とチャンクサイズ
PRIME = 2**521 - 1
CHUNK_SIZE = PRIME.bit_length() // 8 - 1


# ---- Shamir（チャンク対応） ----
def _bytes_to_int(data: bytes) -> int:
    return int.from_bytes(data, "big")


def _int_to_bytes(x: int, length: int) -> bytes:
    return int.to_bytes(x, length, "big")


def _poly_eval(coeffs: List[int], x: int) -> int:
    res = 0
    for c in reversed(coeffs):
        res = (res * x + c) % PRIME
    return res


def _lagrange_interpolate(x: int, xs: List[int], ys: List[int]) -> int:
    k = len(xs)
    total = 0
    for i in range(k):
        xi, yi = xs[i], ys[i]
        num, den = 1, 1
        for j in range(k):
            if i == j:
                continue
            xj = xs[j]
            num = (num * (x - xj)) % PRIME
            den = (den * (xi - xj)) % PRIME
        inv_den = pow(den, PRIME - 2, PRIME)
        term = yi * num * inv_den
        total = (total + term) % PRIME
    return total


def shamir_split_secret(secret: bytes, n: int, k: int) -> List[Tuple[int, List[int], List[int]]]:
    """チャンク化したシークレットを n 分割（k閾値）。(x, y_list, len_list) のリストを返す。"""
    chunks = [secret[i : i + CHUNK_SIZE] for i in range(0, len(secret), CHUNK_SIZE)]
    polys: List[List[int]] = []
    for chunk in chunks:
        s_int = _bytes_to_int(chunk)
        assert s_int < PRIME, "Secret chunk too large for chosen PRIME"
        coeffs = [s_int] + [secrets.randbelow(PRIME) for _ in range(k - 1)]
        polys.append(coeffs)

    shares: List[Tuple[int, List[int], List[int]]] = []
    for x in range(1, n + 1):
        y_list: List[int] = []
        len_list: List[int] = []
        for chunk, coeffs in zip(chunks, polys):
            y = _poly_eval(coeffs, x)
            y_list.append(y)
            len_list.append(len(chunk))
        shares.append((x, y_list, len_list))
    return shares


def shamir_combine_secret(shares: List[Tuple[int, List[int], List[int]]]) -> bytes:
    """(x, y_list, len_list) の share からシークレット bytes を復元。"""
    num_chunks = len(shares[0][1])
    xs_all = [s[0] for s in shares]
    result = bytearray()
    for idx in range(num_chunks):
        ys = [s[1][idx] for s in shares]
        length = shares[0][2][idx]
        s_int = _lagrange_interpolate(0, xs_all, ys)
        result.extend(_int_to_bytes(s_int, length))
    return bytes(result)


# ---- マルチシークレット SIS ----
@dataclass
class MultiLevelImageShare:
    index: int
    y1: List[int]
    len1: List[int]
    y2: List[int]
    len2: List[int]

    def to_dict(self) -> dict:
        return {"index": self.index, "y1": self.y1, "len1": self.len1, "y2": self.y2, "len2": self.len2}

    @classmethod
    def from_dict(cls, d: dict) -> "MultiLevelImageShare":
        return cls(index=int(d["index"]), y1=list(d["y1"]), len1=list(d["len1"]), y2=list(d["y2"]), len2=list(d["len2"]))


class MultiSecretImageSIS:
    """
    k1 share で pHash 一致のマスク画像 J、k2 share で本物画像 I を復元する 2階層 SIS。
    k1 未満では何も復元せず、ノイズ扱いとする。
    """

    def __init__(self, n: int, k1: int, k2: int, cfg: PHashConfig | None = None) -> None:
        assert 1 < k1 <= k2 <= n
        self.n = n
        self.k1 = k1
        self.k2 = k2
        self.cfg = cfg or PHashConfig()
        self._base_size: Tuple[int, int] | None = None

    def split_from_image(self, image_path: Path, seed: int | None = None) -> List[MultiLevelImageShare]:
        img = Image.open(image_path)
        self._base_size = img.size
        dummy = make_phash_preserving_dummy(img, self.cfg, seed=seed)
        # 秘密を bytes 化
        s1 = self._encode_image(dummy)
        s2 = image_path.read_bytes()

        shares1 = shamir_split_secret(s1, self.n, self.k1)
        shares2 = shamir_split_secret(s2, self.n, self.k2)

        shares: List[MultiLevelImageShare] = []
        for (x1, y1, len1), (x2, y2, len2) in zip(shares1, shares2):
            assert x1 == x2
            shares.append(MultiLevelImageShare(index=x1, y1=y1, len1=len1, y2=y2, len2=len2))
        return shares

    def recover_dummy_image(self, shares: Sequence[MultiLevelImageShare]) -> Image.Image:
        if len(shares) < self.k1:
            raise ValueError("not enough shares for dummy reconstruction")
        subset = list(shares[: self.k1])
        s1_bytes = shamir_combine_secret([(s.index, s.y1, s.len1) for s in subset])
        return self._decode_image(s1_bytes)

    def recover_full_image(self, shares: Sequence[MultiLevelImageShare]) -> Image.Image:
        if len(shares) < self.k2:
            raise ValueError("not enough shares for full reconstruction")
        subset = list(shares[: self.k2])
        s2_bytes = shamir_combine_secret([(s.index, s.y2, s.len2) for s in subset])
        return Image.open(BytesIO(s2_bytes))

    def reconstruct_with_levels(self, shares: Sequence[MultiLevelImageShare]) -> Tuple[str, Image.Image]:
        """
        share 数に応じて (level, image) を返す:
          - level="noise" : k1 未満 → ノイズ画像
          - level="dummy" : k1 以上 k2 未満 → pHash 一致のマスク画像
          - level="full"  : k2 以上 → 本物画像
        """
        if len(shares) < self.k1:
            return "noise", self._noise_image()
        if len(shares) < self.k2:
            return "dummy", self.recover_dummy_image(shares)
        return "full", self.recover_full_image(shares)

    # ---- ユーティリティ ----
    def _encode_image(self, img: Image.Image) -> bytes:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _decode_image(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data))

    def _noise_image(self, size: Tuple[int, int] | None = None) -> Image.Image:
        target = size or self._base_size or (256, 256)
        arr = secrets.token_bytes(target[0] * target[1])
        return Image.frombytes("L", target, arr)

    # --- share の永続化 ---
    def save_shares(self, shares: Sequence[MultiLevelImageShare], out_dir: Path) -> None:
        """
        share を JSON 形式でディレクトリに書き出す。
        """
        import json

        out_dir.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(shares, start=1):
            with open(out_dir / f"share_{i}.json", "w", encoding="utf-8") as f:
                json.dump(s.to_dict(), f)

    def load_shares(self, in_dir: Path) -> List[MultiLevelImageShare]:
        import json

        shares: List[MultiLevelImageShare] = []
        for p in sorted(in_dir.glob("share_*.json")):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            shares.append(MultiLevelImageShare.from_dict(d))
        return shares
