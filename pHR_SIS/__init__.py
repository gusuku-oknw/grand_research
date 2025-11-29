"""Legacy wrapper exposing the sis_image.dealer_based package for backwards compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sis_image.dealer_based import (
    FusionAwareSearchableSISIndex,
    SearchableSISIndex,
    ShamirImageStore,
    SearchableSISWithImageStore,
    StoredShare,
    ImageMeta,
    build_parser,
    run_selective_demo,
    run_workflow_demo,
    run_image_store_demo,
    run_secure_demo,
    fusion_hash64,
)

__all__ = [
    "FusionAwareSearchableSISIndex",
    "SearchableSISIndex",
    "ShamirImageStore",
    "SearchableSISWithImageStore",
    "StoredShare",
    "ImageMeta",
    "build_parser",
    "run_selective_demo",
    "run_workflow_demo",
    "run_image_store_demo",
    "run_secure_demo",
    "fusion_hash64",
]
