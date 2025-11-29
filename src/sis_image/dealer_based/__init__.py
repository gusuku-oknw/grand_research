"""Dealer-based SIS implementations with workflow helpers."""

from .cli import build_parser, run_selective_demo, run_workflow_demo, run_image_store_demo, run_secure_demo
from .fusion import fusion_hash64
from .fusion_index import FusionAwareSearchableSISIndex
from .image_store import ImageMeta, ShamirImageStore
from .index import SearchableSISIndex, StoredShare
from .workflow import ReconstructionResult, SearchableSISWithImageStore

__all__ = [
    "build_parser",
    "run_selective_demo",
    "run_workflow_demo",
    "run_image_store_demo",
    "run_secure_demo",
    "fusion_hash64",
    "FusionAwareSearchableSISIndex",
    "ImageMeta",
    "ShamirImageStore",
    "SearchableSISIndex",
    "StoredShare",
    "ReconstructionResult",
    "SearchableSISWithImageStore",
]
