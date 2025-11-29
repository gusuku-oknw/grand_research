"""Common helpers shared by both dealer-based and dealer-free SIS implementations."""

from .mpc import simulated_secure_hamming, recover_hash_from_shares
from .phash import phash64, hamming64, hash64_to_bytes, bytes_to_hash64
from .shamir import (
    FIELD_PRIME,
    shamir_share_bytes,
    shamir_recover_bytes,
    shamir_share_flat_bytes,
    shamir_recover_flat_bytes,
)
from .tokens import hmac_token, split_bands

__all__ = [
    "phash64",
    "hamming64",
    "hash64_to_bytes",
    "bytes_to_hash64",
    "FIELD_PRIME",
    "shamir_share_bytes",
    "shamir_recover_bytes",
    "shamir_share_flat_bytes",
    "shamir_recover_flat_bytes",
    "hmac_token",
    "split_bands",
    "simulated_secure_hamming",
    "recover_hash_from_shares",
]
