"""Analysis helpers for MPC-aware SIS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from sis_image.dealer_based import SearchableSISIndex
from sis_image.common.phash import phash64


@dataclass
class SecureDistanceComparison:
    """Comparison between baseline and MPC rankings for a single query."""

    query_image: str
    query_hash: int
    servers: Tuple[int, ...]
    min_band_votes: int
    topk: int
    max_hamming: Optional[int]
    preselected: List[Tuple[str, int]]
    baseline_ranked: List[Tuple[str, int]]
    secure_ranked: List[Tuple[str, int]]

    def to_dict(self) -> dict:
        return {
            "query_image": self.query_image,
            "query_hash": f"0x{self.query_hash:016x}",
            "servers": list(self.servers),
            "min_band_votes": self.min_band_votes,
            "topk": self.topk,
            "max_hamming": self.max_hamming,
            "preselected": [
                {"image_id": image_id, "votes": votes} for image_id, votes in self.preselected
            ],
            "baseline_ranked": [
                {"image_id": image_id, "distance": distance}
                for image_id, distance in self.baseline_ranked
            ],
            "secure_ranked": [
                {"image_id": image_id, "distance": distance}
                for image_id, distance in self.secure_ranked
            ],
        }


def compare_secure_vs_baseline(
    index: SearchableSISIndex,
    query_image_path: Path | str,
    servers: Sequence[int],
    min_band_votes: int = 3,
    topk: int = 5,
    max_hamming: Optional[int] = None,
) -> SecureDistanceComparison:
    image_path = Path(query_image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Query image not found: {image_path}")
    if len(servers) < index.k:
        raise ValueError(f"Need at least k={index.k} servers to compare secure distance.")
    ordered_servers = tuple(int(s) for s in sorted(set(servers)))
    query_hash = phash64(str(image_path))
    preselected = index.preselect_candidates(
        query_hash,
        ordered_servers,
        min_band_votes=min_band_votes,
    )
    candidate_ids = [image_id for image_id, _ in preselected]
    baseline_ranked = index.rank_candidates(
        query_hash,
        ordered_servers,
        candidate_ids,
        topk=topk,
        max_hamming=max_hamming,
    )
    secure_ranked = index.rank_candidates_secure(
        query_hash,
        ordered_servers,
        candidate_ids,
        topk=topk,
        max_hamming=max_hamming,
    )
    return SecureDistanceComparison(
        query_image=image_path.name,
        query_hash=query_hash,
        servers=ordered_servers,
        min_band_votes=min_band_votes,
        topk=topk,
        max_hamming=max_hamming,
        preselected=preselected,
        baseline_ranked=baseline_ranked,
        secure_ranked=secure_ranked,
    )


__all__ = ["compare_secure_vs_baseline", "SecureDistanceComparison"]
