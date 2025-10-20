"""Composite workflows combining searchable SIS and image reconstruction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .image_store import ShamirImageStore
from .index import SearchableSISIndex


@dataclass
class ReconstructionResult:
    image_id: str
    path: str


class SearchableSISWithImageStore:
    """High-level helper tying together the SIS index and image storage."""

    def __init__(
        self,
        k: int = 3,
        n: int = 5,
        bands: int = 8,
        token_len: int = 8,
        seed: int = 2025,
        shares_dir: str = "img_shares",
        meta_dir: str = "img_meta",
    ):
        self.index = SearchableSISIndex(
            k=k, n=n, bands=bands, token_len=token_len, seed=seed
        )
        self.store = ShamirImageStore(k=k, n=n, shares_dir=shares_dir, meta_dir=meta_dir)

    def list_servers(self) -> List[int]:
        return self.index.list_servers()

    def add_image(self, image_id: str, image_path: str) -> int:
        """Register image shares for both hash search and full reconstruction."""
        phash = self.index.add_image(image_id, image_path)
        self.store.add_image(image_id, image_path)
        return phash

    def rank_candidates(
        self,
        query_hash: int,
        servers_for_query: Iterable[int],
        candidates: Iterable[str],
        topk: int,
        max_hamming: Optional[int],
    ) -> List[Tuple[str, int]]:
        return self.index.rank_candidates(
            query_hash,
            servers_for_query=servers_for_query,
            candidates=candidates,
            topk=topk,
            max_hamming=max_hamming,
        )

    def query_and_optionally_reconstruct(
        self,
        query_image_path: str,
        servers_for_query: Iterable[int],
        min_band_votes: int,
        topk: int,
        max_hamming: Optional[int],
        reconstruct_top: int,
        recon_dir: str,
    ) -> Dict[str, object]:
        """Execute a search and reconstruct top matches if requested."""
        result = self.index.query_selective(
            query_image_path,
            servers_for_query=servers_for_query,
            min_band_votes=min_band_votes,
            topk=topk,
            max_hamming=max_hamming,
        )
        os.makedirs(recon_dir, exist_ok=True)
        ranked: List[Tuple[str, int]] = result["ranked"]
        reconstructions: List[ReconstructionResult] = []
        for image_id, _ in ranked[:reconstruct_top]:
            out_path = os.path.join(recon_dir, f"reconstructed_{image_id}.png")
            ok = self.store.reconstruct(image_id, servers_for_query, out_path)
            if ok:
                reconstructions.append(ReconstructionResult(image_id=image_id, path=out_path))
        result["reconstructed"] = [(item.image_id, item.path) for item in reconstructions]
        return result


__all__ = ["SearchableSISWithImageStore", "ReconstructionResult"]
