from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import secrets

# 大きめの素数（実験用途）。必要に応じて変更する。
PRIME = 2**521 - 1


def _bytes_to_int(data: bytes) -> int:
    return int.from_bytes(data, "big")


def _int_to_bytes(x: int, length: int) -> bytes:
    return int.to_bytes(x, length, "big")


def _poly_eval(coeffs: List[int], x: int) -> int:
    """f(x) = a0 + a1 x + ... (mod PRIME) を評価。"""
    res = 0
    for c in reversed(coeffs):
        res = (res * x + c) % PRIME
    return res


def _lagrange_interpolate(x: int, xs: List[int], ys: List[int]) -> int:
    """Shamir の復元で使うラグランジュ補間 (mod PRIME)。"""
    assert len(xs) == len(ys)
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


def shamir_split(secret: bytes, n: int, k: int) -> List[Tuple[int, int, int]]:
    """1つのシークレットを Shamir で n 分割（k 閾値）。"""
    s_int = _bytes_to_int(secret)
    assert s_int < PRIME, "Secret too large for chosen PRIME"
    coeffs = [s_int] + [secrets.randbelow(PRIME) for _ in range(k - 1)]
    shares: List[Tuple[int, int, int]] = []
    for x in range(1, n + 1):
        y = _poly_eval(coeffs, x)
        shares.append((x, y, len(secret)))
    return shares


def shamir_combine(shares: List[Tuple[int, int, int]]) -> bytes:
    """Shamir の share (x, y, length) からシークレット bytes を復元。"""
    xs = [s[0] for s in shares]
    ys = [s[1] for s in shares]
    length = shares[0][2]
    s_int = _lagrange_interpolate(0, xs, ys)
    return _int_to_bytes(s_int, length)


@dataclass
class TwoLevelShare:
    """1 参加者あたりの 2階層 share。"""

    index: int
    y1: int
    len1: int
    y2: int
    len2: int


class TwoLevelShamirScheme:
    """
    S1（低閾値）と S2（高閾値）を別々の Shamir で分割し、同じ index を共有させる。
    - S1: k1 で復元（ダミー画像など）
    - S2: k2 で復元（本物画像など）
    """

    def __init__(self, n: int, k1: int, k2: int) -> None:
        assert 1 < k1 <= k2 <= n
        self.n = n
        self.k1 = k1
        self.k2 = k2

    def split(self, secret1: bytes, secret2: bytes) -> List[TwoLevelShare]:
        shares1 = shamir_split(secret1, self.n, self.k1)
        shares2 = shamir_split(secret2, self.n, self.k2)
        combined: List[TwoLevelShare] = []
        for (x1, y1, len1), (x2, y2, len2) in zip(shares1, shares2):
            assert x1 == x2
            combined.append(TwoLevelShare(index=x1, y1=y1, len1=len1, y2=y2, len2=len2))
        return combined

    def recover_level1(self, shares: List[TwoLevelShare]) -> bytes:
        """k1 以上の share から S1 を復元。"""
        assert len(shares) >= self.k1
        subset = shares[: self.k1]
        shamir_shares = [(s.index, s.y1, s.len1) for s in subset]
        return shamir_combine(shamir_shares)

    def recover_level2(self, shares: List[TwoLevelShare]) -> bytes:
        """k2 以上の share から S2 を復元。"""
        assert len(shares) >= self.k2
        subset = shares[: self.k2]
        shamir_shares = [(s.index, s.y2, s.len2) for s in subset]
        return shamir_combine(shamir_shares)
