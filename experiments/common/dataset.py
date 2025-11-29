"""Dataset utilities for COCO derivative experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Sample:
    image_id: str
    variant: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.image_id}__{self.variant}"


def load_derivative_mapping(mapping_json: Path) -> Dict[str, Dict[str, str]]:
    with mapping_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {img_id: {variant: str(path) for variant, path in variants.items()} for img_id, variants in raw.items()}


def build_samples(mapping: Dict[str, Dict[str, str]], include_variants: Iterable[str] | None = None) -> List[Sample]:
    samples: List[Sample] = []
    for image_id, variants in mapping.items():
        for variant, path in variants.items():
            if include_variants is not None and variant not in include_variants:
                continue
            samples.append(Sample(image_id=image_id, variant=variant, path=Path(path)))
    return samples


def build_positive_lookup(samples: List[Sample]) -> Dict[str, List[int]]:
    lookup: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        lookup.setdefault(sample.image_id, []).append(idx)
    return lookup


def list_variants(mapping: Dict[str, Dict[str, str]]) -> List[str]:
    variants = set()
    for variants_map in mapping.values():
        variants.update(variants_map.keys())
    return sorted(variants)
