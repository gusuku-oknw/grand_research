"""Simulator helpers for dealer-free SIS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from sis_image.dealer_based import SearchableSISIndex


@dataclass
class RegistrationMetrics:
    """Byte-level accounting for registering an image."""

    image_id: str
    stage_a_baseline: int
    stage_a_distributed: int
    share_length: int
    phash: int


@dataclass
class QueryMetrics:
    """Simulated communication for a query across SIS stages."""

    image_id: str
    stage_b_baseline: int
    stage_b_distributed: int
    stage_c: int
    share_length: int


class DealerFreeSimulator:
    """Lightweight orchestration for dealer-free experimentation."""

    def __init__(
        self,
        k: int = 3,
        n: int = 5,
        bands: int = 8,
        token_len: int = 8,
        use_oprf: bool = False,
        distributed_contributors: int = 3,
        padding_tokens: int = 4,
    ) -> None:
        if not (2 <= k <= n <= 20):
            raise ValueError("Require 2 <= k <= n <= 20 for SIS experiments.")
        self.k = k
        self.n = n
        self.bands = bands
        self.token_len = token_len
        self.use_oprf = use_oprf
        self.distributed_contributors = distributed_contributors
        self.padding_tokens = padding_tokens
        self.index = SearchableSISIndex(
            k=k,
            n=n,
            bands=bands,
            token_len=token_len,
            use_oprf=use_oprf,
        )
        self._registrations: Dict[str, RegistrationMetrics] = {}

    def register_image(self, image_id: str, image_path: str) -> RegistrationMetrics:
        """Add an image to the index and return stage metrics."""
        phash = self.index.add_image(image_id, image_path)
        share_length = self._share_length_for(image_id)
        stage_a_baseline = self.n * share_length
        stage_a_distributed = stage_a_baseline + self._dkg_overhead(share_length)
        metrics = RegistrationMetrics(
            image_id=image_id,
            stage_a_baseline=stage_a_baseline,
            stage_a_distributed=stage_a_distributed,
            share_length=share_length,
            phash=phash,
        )
        self._registrations[image_id] = metrics
        return metrics

    def query_metrics(self, image_id: str) -> QueryMetrics:
        """Produce communication estimates for querying a given image."""
        registration = self._registrations.get(image_id)
        if registration is None:
            raise KeyError(f"Image {image_id} has not been registered yet.")
        stage_b_baseline = self.n * self.bands * self.token_len
        stage_b_distributed = stage_b_baseline + self.padding_tokens * self.token_len
        stage_c = self.k * registration.share_length
        return QueryMetrics(
            image_id=image_id,
            stage_b_baseline=stage_b_baseline,
            stage_b_distributed=stage_b_distributed,
            stage_c=stage_c,
            share_length=registration.share_length,
        )

    def _share_length_for(self, image_id: str) -> int:
        """Return the byte-length of the Shamir share stored per server."""
        server_share = self.index.server_shares[1].get(image_id)
        if server_share is None:
            raise KeyError(f"No share stored for image {image_id} on server 1.")
        return len(server_share.share_bytes)

    def _dkg_overhead(self, share_length: int) -> int:
        """Estimate bytes exchanged during a distributed key generation round."""
        if self.distributed_contributors <= 0:
            return 0
        per_contributor = self.k * share_length
        return self.distributed_contributors * per_contributor

    def registered_image_ids(self) -> Sequence[str]:
        """List the images that have already been registered."""
        return list(self._registrations.keys())
