"""Secret Image Sharing (SIS) package with pHash search utilities."""

from .phash import (
    phash64,
    hamming64,
    hash64_to_bytes,
    bytes_to_hash64,
)
from .shamir import (
    shamir_share_bytes,
    shamir_recover_bytes,
    shamir_share_flat_bytes,
    shamir_recover_flat_bytes,
    FIELD_PRIME,
)
from .fusion import fusion_hash64
from .fusion_index import FusionAwareSearchableSISIndex
from .index import SearchableSISIndex, StoredShare
from .image_store import ShamirImageStore, ImageMeta
from .workflow import SearchableSISWithImageStore

__all__ = [
    "phash64",
    "hamming64",
    "hash64_to_bytes",
    "bytes_to_hash64",
    "shamir_share_bytes",
    "shamir_recover_bytes",
    "shamir_share_flat_bytes",
    "shamir_recover_flat_bytes",
    "FIELD_PRIME",
    "fusion_hash64",
    "FusionAwareSearchableSISIndex",
    "SearchableSISIndex",
    "StoredShare",
    "ShamirImageStore",
    "ImageMeta",
    "SearchableSISWithImageStore",
]
