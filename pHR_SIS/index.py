"""Searchable SIS index built on Shamir sharing and banded HMAC tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .phash import phash64, hash64_to_bytes, bytes_to_hash64, hamming64
from .shamir import shamir_share_bytes, shamir_recover_bytes
from .tokens import hmac_token, split_bands


@dataclass
class StoredShare:
    image_id: str
    share_bytes: List[int]


class SearchableSISIndex:
    """Index that enables privacy-preserving nearest-neighbour queries over pHash values."""

    def __init__(
        self,
        k: int = 3,
        n: int = 5,
        bands: int = 8,
        token_len: int = 8,
        seed: int = 1234,
    ):
        if not (2 <= k <= n <= 20):
            raise ValueError("Require 2 <= k <= n <= 20.")
        if 64 % bands != 0:
            raise ValueError("Number of bands must divide 64.")
        self.k, self.n = k, n
        self.bands = bands
        self.token_len = token_len
        self.band_bits = 64 // bands
        self.band_bytes = max(1, (self.band_bits + 7) // 8)
        self.server_shares: Dict[int, Dict[str, StoredShare]] = {
            x: {} for x in range(1, n + 1)
        }
        self.server_band_buckets: Dict[int, List[Dict[bytes, Set[str]]]] = {
            x: [dict() for _ in range(bands)] for x in range(1, n + 1)
        }
        rng = np.random.default_rng(seed)
        self.hmac_keys: Dict[int, List[bytes]] = {
            x: [
                rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
                for _ in range(bands)
            ]
            for x in range(1, n + 1)
        }
        self._images: Dict[str, str] = {}

    def _index_band_tokens_for_image(self, image_id: str, phash64: int) -> None:
        band_values = split_bands(phash64, bands=self.bands)
        for server in range(1, self.n + 1):
            for band_idx, value in enumerate(band_values):
                key = self.hmac_keys[server][band_idx]
                token = hmac_token(
                    key,
                    value.to_bytes(self.band_bytes, "big"),
                    outlen=self.token_len,
                )
                bucket = self.server_band_buckets[server][band_idx].setdefault(
                    token, set()
                )
                bucket.add(image_id)

    def add_image(self, image_id: str, image_path: str) -> int:
        """Register an image and return its pHash."""
        phash = phash64(image_path)
        self._images[image_id] = image_path
        secret_bytes = hash64_to_bytes(phash)
        shares = shamir_share_bytes(secret_bytes, self.k, self.n)
        for server, share_bytes in shares.items():
            self.server_shares[server][image_id] = StoredShare(
                image_id=image_id, share_bytes=share_bytes
            )
        self._index_band_tokens_for_image(image_id, phash)
        return phash

    def list_servers(self) -> List[int]:
        """Return the configured server identifiers."""
        return list(sorted(self.server_shares.keys()))

    def _reconstruct_hash_from_servers(
        self,
        image_id: str,
        servers: Iterable[int],
    ) -> Optional[int]:
        xs = sorted(set(int(s) for s in servers))
        if len(xs) < self.k:
            return None
        subshares: Dict[int, List[int]] = {}
        for server in xs:
            stored = self.server_shares.get(server, {}).get(image_id)
            if stored is None:
                return None
            subshares[server] = stored.share_bytes
        recovered = shamir_recover_bytes(subshares)
        return bytes_to_hash64(recovered)

    def preselect_candidates(
        self,
        query_hash: int,
        servers_for_query: Iterable[int],
        min_band_votes: int = 3,
    ) -> List[Tuple[str, int]]:
        """Return candidate image IDs with vote counts using band token matches."""
        xs = sorted(set(int(s) for s in servers_for_query))
        band_values = split_bands(query_hash, bands=self.bands)
        votes: Dict[str, int] = {}
        for server in xs:
            for band_idx, value in enumerate(band_values):
                key = self.hmac_keys[server][band_idx]
                token = hmac_token(
                    key,
                    value.to_bytes(self.band_bytes, "big"),
                    outlen=self.token_len,
                )
                matches = self.server_band_buckets[server][band_idx].get(token, set())
                for image_id in matches:
                    votes[image_id] = votes.get(image_id, 0) + 1
        filtered = [(image_id, count) for image_id, count in votes.items() if count >= min_band_votes]
        filtered.sort(key=lambda item: (-item[1], item[0]))
        return filtered

    def rank_candidates(
        self,
        query_hash: int,
        servers_for_query: Iterable[int],
        candidates: Iterable[str],
        topk: int = 10,
        max_hamming: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Reconstruct candidate hashes and return the closest matches."""
        xs = sorted(set(int(s) for s in servers_for_query))
        if len(xs) < self.k:
            raise ValueError(f"Need at least k={self.k} servers to reconstruct.")
        results: List[Tuple[str, int]] = []
        for image_id in candidates:
            recovered = self._reconstruct_hash_from_servers(image_id, xs)
            if recovered is None:
                continue
            distance = hamming64(query_hash, recovered)
            if max_hamming is not None and distance > max_hamming:
                continue
            results.append((image_id, distance))
        results.sort(key=lambda item: (item[1], item[0]))
        return results[:topk]

    def query_selective(
        self,
        query_image_path: str,
        servers_for_query: Iterable[int],
        min_band_votes: int = 3,
        topk: int = 5,
        max_hamming: Optional[int] = 10,
    ) -> Dict[str, object]:
        """High-level API combining pre-selection and distance ranking."""
        query_hash = phash64(query_image_path)
        preselected = self.preselect_candidates(
            query_hash, servers_for_query, min_band_votes=min_band_votes
        )
        candidate_ids = [image_id for image_id, _ in preselected]
        ranked = self.rank_candidates(
            query_hash,
            servers_for_query,
            candidate_ids,
            topk=topk,
            max_hamming=max_hamming,
        )
        return {
            "query_phash": f"0x{query_hash:016x}",
            "preselected": preselected,
            "ranked": ranked,
            "servers": list(sorted(set(int(s) for s in servers_for_query))),
            "params": {
                "min_band_votes": min_band_votes,
                "topk": topk,
                "max_hamming": max_hamming,
            },
        }


__all__ = ["StoredShare", "SearchableSISIndex"]
