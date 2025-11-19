"""Composite workflows combining searchable SIS and image reconstruction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .image_store import ShamirImageStore
from .index import SearchableSISIndex
from .fusion_index import FusionAwareSearchableSISIndex


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
        secure_distance: bool = False,
        share_strategy: str = "shamir",
        fusion_grid: int = 8,
        fusion_threshold: int | None = None,
    ):
        if share_strategy not in {"shamir", "phash-fusion"}:
            raise ValueError(f"Unsupported share_strategy='{share_strategy}'")
        self.share_strategy = share_strategy
        if share_strategy == "phash-fusion":
            self.index = FusionAwareSearchableSISIndex(
                k=k,
                n=n,
                bands=bands,
                token_len=token_len,
                seed=seed,
                fusion_grid=fusion_grid,
                fusion_threshold=fusion_threshold,
            )
        else:
            self.index = SearchableSISIndex(
                k=k, n=n, bands=bands, token_len=token_len, seed=seed
            )
        self.store = ShamirImageStore(k=k, n=n, shares_dir=shares_dir, meta_dir=meta_dir)
        self.secure_distance = secure_distance

    def list_servers(self) -> List[int]:
        return self.index.list_servers()

    def add_image(self, image_id: str, image_path: str, phash: int | None = None) -> int:
        """Register image shares for both hash search and full reconstruction."""
        if phash is None:
            phash = self.index.add_image(image_id, image_path)
        else:
            self.index.add_image_with_phash(image_id, image_path, phash)
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
        if self.secure_distance:
            return self.index.rank_candidates_secure(
                query_hash,
                servers_for_query=servers_for_query,
                candidates=candidates,
                topk=topk,
                max_hamming=max_hamming,
            )
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
        result.setdefault("share_mode", self.share_strategy)
        result.setdefault("fusion_mode", False)
        os.makedirs(recon_dir, exist_ok=True)
        query_hash = int(result["query_phash"], 16)
        pre_ids = [img_id for img_id, _ in result["preselected"]]
        if self.secure_distance:
            ranked = self.rank_candidates(
                query_hash,
                servers_for_query=servers_for_query,
                candidates=pre_ids,
                topk=topk,
                max_hamming=max_hamming,
            )
            result["ranked"] = ranked
        else:
            ranked = result["ranked"]
        result["mode"] = "mpc" if self.secure_distance else "standard"
        reconstructions: List[ReconstructionResult] = []
        reconstruction_errors: List[str] = []
        for image_id, _ in ranked[:reconstruct_top]:
            out_path = os.path.join(recon_dir, f"reconstructed_{image_id}.png")
            try:
                ok = self.store.reconstruct(image_id, servers_for_query, out_path)
            except ValueError as exc:
                ok = False
                reconstruction_errors.append(str(exc))
            if ok:
                reconstructions.append(ReconstructionResult(image_id=image_id, path=out_path))
        result["reconstructed"] = [(item.image_id, item.path) for item in reconstructions]
        result["reconstruction_errors"] = reconstruction_errors
        result["share_strategy"] = self.share_strategy
        result["insufficient_shares"] = len(set(int(s) for s in servers_for_query)) < self.index.k
        return result


__all__ = ["SearchableSISWithImageStore", "ReconstructionResult"]
