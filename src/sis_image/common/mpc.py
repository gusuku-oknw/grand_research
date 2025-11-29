"""Simulated MPC utilities for SIS demos.

このモジュールはデモ用途のため、実際の安全計算を行わずに Shamir 共有から
平文を復元してハミング距離を算出します。MPC を導入する際のフック位置や
プロトコル設計を可視化することを目的としています。
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .phash import bytes_to_hash64, hamming64
from .shamir import shamir_recover_bytes


def recover_hash_from_shares(shares: Dict[int, Iterable[int]]) -> int:
    """Shamir 共有から 64bit ハッシュを復元するヘルパー."""
    recovered_bytes = shamir_recover_bytes({k: list(v) for k, v in shares.items()})
    return bytes_to_hash64(recovered_bytes)


def simulated_secure_hamming(
    query_shares: Dict[int, Iterable[int]],
    candidate_shares: Dict[int, Iterable[int]],
) -> int:
    """共有値からハミング距離を算出する疑似 MPC 処理."""
    # NOTE: 実際の MPC では平文を復元せずに計算する。
    # ここではデモ目的で復元後に距離を算出している。
    query_hash = recover_hash_from_shares(query_shares)
    candidate_hash = recover_hash_from_shares(candidate_shares)
    return hamming64(query_hash, candidate_hash)


__all__ = ["simulated_secure_hamming", "recover_hash_from_shares"]
