"""Fusion-aware SIS index that keeps a fused perceptual hash backup."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from .fusion import fusion_hash64
from .index import SearchableSISIndex
from .phash import hamming64, phash64


class FusionAwareSearchableSISIndex(SearchableSISIndex):
    """Index that supplements Shamir shares with a fusion hash fallback."""

    def __init__(
        self,
        k: int = 3,
        n: int = 5,
        bands: int = 8,
        token_len: int = 8,
        seed: int = 1234,
        fusion_grid: int = 8,
        fusion_threshold: int | None = None,
        key_store_path: str | None = None,
        key_env_var: str | None = None,
        key_encrypt_env_var: str | None = None,
    ) -> None:
        super().__init__(
            k=k,
            n=n,
            bands=bands,
            token_len=token_len,
            seed=seed,
            key_store_path=key_store_path,
            key_env_var=key_env_var,
            key_encrypt_env_var=key_encrypt_env_var,
        )
        self.fusion_grid = fusion_grid
        self.fusion_threshold = fusion_threshold if fusion_threshold is not None else k
        self._fusion_hashes: Dict[str, int] = {}

    def _add_image_with_phash(self, image_id: str, image_path: str, phash: int) -> int:
        phash_value = super()._add_image_with_phash(image_id, image_path, phash)
        fused = fusion_hash64(image_path, grid=self.fusion_grid)
        self._fusion_hashes[image_id] = fused
        return phash_value

    def rank_candidates_fusion(
        self,
        query_fusion_hash: int,
        candidates: Iterable[str],
        topk: int = 10,
        max_hamming: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        results: List[Tuple[str, int]] = []
        for image_id in candidates:
            fused = self._fusion_hashes.get(image_id)
            if fused is None:
                continue
            distance = hamming64(query_fusion_hash, fused)
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
        dummy_band_queries: int = 0,
        pad_band_queries: Optional[int] = None,
        fixed_band_queries: Optional[int] = None,
    ) -> Dict[str, object]:
        query_hash = phash64(query_image_path)
        query_fusion = fusion_hash64(query_image_path, grid=self.fusion_grid)
        preselected = self.preselect_candidates(
            query_hash,
            servers_for_query,
            min_band_votes=min_band_votes,
            dummy_band_queries=dummy_band_queries,
            pad_band_queries=pad_band_queries,
            fixed_band_queries=fixed_band_queries,
        )
        candidate_ids = [image_id for image_id, _ in preselected]
        xs = sorted(set(int(s) for s in servers_for_query))
        can_reconstruct = len(xs) >= self.k
        if can_reconstruct:
            ranked = super().rank_candidates(
                query_hash,
                servers_for_query=servers_for_query,
                candidates=candidate_ids,
                topk=topk,
                max_hamming=max_hamming,
            )
            share_mode = "shamir"
            fusion_mode = False
        else:
            ranked = self.rank_candidates_fusion(
                query_fusion,
                candidates=candidate_ids,
                topk=topk,
                max_hamming=max_hamming,
            )
            share_mode = "phash-fusion"
            fusion_mode = True
        return {
            "query_phash": f"0x{query_hash:016x}",
            "query_fusion_phash": f"0x{query_fusion:016x}",
            "preselected": preselected,
            "ranked": ranked,
            "servers": list(xs),
            "params": {
                "min_band_votes": min_band_votes,
                "topk": topk,
                "max_hamming": max_hamming,
            },
            "share_mode": share_mode,
            "fusion_mode": fusion_mode,
            "fusion_k": self.fusion_threshold,
        }
