"""Base class for the comparison modes described in docs/README.md."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sis_image.dealer_based import SearchableSISIndex


@dataclass
class ModeSummary:
    mode: str
    server_count: int
    preselected: int
    ranked: int
    dummy_queries: int
    pad_queries: int | None
    fixed_queries: int | None


class ModeRunner(ABC):
    """Common interface that all comparison modes implement."""

    name: str
    description: str

    def __init__(
        self,
        images: Sequence[Path],
        k: int = 3,
        n: int = 5,
        min_band_votes: int = 3,
        dummy_band_queries: int = 0,
        pad_band_queries: int | None = None,
        fixed_band_queries: int | None = None,
    ):
        if not images:
            raise ValueError("At least one image is required to run a mode.")
        if dummy_band_queries < 0:
            raise ValueError("dummy_band_queries must be non-negative.")
        self.images = images
        self.k = k
        self.n = n
        self.min_band_votes = min_band_votes
        self.dummy_band_queries = dummy_band_queries
        self.pad_band_queries = pad_band_queries
        self.fixed_band_queries = fixed_band_queries
        self.index = SearchableSISIndex(k=k, n=n)

    def run(self) -> ModeSummary:
        for idx, image_path in enumerate(self.images):
            self.index.add_image(f"{self.name}_{idx:04d}", str(image_path))
        query_path = str(self.images[0])
        servers = list(range(1, min(self.k, self.n) + 1))
        result = self.index.query_selective(
            query_path,
            servers_for_query=servers,
            min_band_votes=self.min_band_votes,
            dummy_band_queries=self.dummy_band_queries,
            pad_band_queries=self.pad_band_queries,
            fixed_band_queries=self.fixed_band_queries,
        )
        return ModeSummary(
            mode=self.name,
            server_count=len(servers),
            preselected=len(result.get("preselected", [])),
            ranked=len(result.get("ranked", [])),
            dummy_queries=self.dummy_band_queries,
            pad_queries=self.pad_band_queries,
            fixed_queries=self.fixed_band_queries,
        )
