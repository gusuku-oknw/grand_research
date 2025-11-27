"""Searchable SIS index built on Shamir sharing and banded HMAC tokens."""

from __future__ import annotations

import json
import os
import base64
import secrets
from hashlib import sha256
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .phash import phash64, hash64_to_bytes, bytes_to_hash64, hamming64
from .shamir import shamir_share_bytes, shamir_recover_bytes
from .mpc import simulated_secure_hamming
from .tokens import hmac_token, split_bands
from .oprf import VOPRFHandler, load_or_create_oprf_keys


@dataclass
class StoredShare:
    image_id: str
    share_bytes: List[int]


class SearchableSISIndex:
    """Index that enables privacy-preserving nearest-neighbour queries over pHash values."""

    def __init__(
        self,
        k: int = 3,
        n: int = 5,
        bands: int = 8,
        token_len: int = 8,
        seed: int = 1234,
        key_store_path: Optional[str] = None,
        key_env_var: Optional[str] = None,
        key_encrypt_env_var: Optional[str] = None,
        use_oprf: bool = False,
        oprf_key_path: Optional[str] = None,
        oprf_key_env_var: Optional[str] = None,
        oprf_key_encrypt_env_var: Optional[str] = None,
    ):
        if not (2 <= k <= n <= 20):
            raise ValueError("Require 2 <= k <= n <= 20.")
        if 64 % bands != 0:
            raise ValueError("Number of bands must divide 64.")
        self.k, self.n = k, n
        self.bands = bands
        self.token_len = token_len
        self.band_bits = 64 // bands
        self.band_bytes = max(1, (self.band_bits + 7) // 8)
        self.server_shares: Dict[int, Dict[str, StoredShare]] = {
            x: {} for x in range(1, n + 1)
        }
        self.server_band_buckets: Dict[int, List[Dict[bytes, Set[str]]]] = {
            x: [dict() for _ in range(bands)] for x in range(1, n + 1)
        }
        self.seed = seed  # kept for API compatibility; not used for keying now
        self.hmac_keys: Dict[int, List[bytes]] = self._load_or_create_keys(
            key_store_path=key_store_path,
            key_env_var=key_env_var,
            key_encrypt_env_var=key_encrypt_env_var,
        )
        self.use_oprf = use_oprf
        self.oprf_handler: Optional[VOPRFHandler] = None
        if self.use_oprf:
            oprf_keys = load_or_create_oprf_keys(
                n=self.n,
                key_store_path=oprf_key_path,
                key_env_var=oprf_key_env_var,
                key_encrypt_env_var=oprf_key_encrypt_env_var,
            )
            self.oprf_handler = VOPRFHandler(server_keys=oprf_keys, token_len=token_len)
        self._images: Dict[str, str] = {}

    def _validate_loaded_keys(self, loaded: Dict[int, List[bytes]]) -> Dict[int, List[bytes]]:
        expected_servers = set(range(1, self.n + 1))
        if set(loaded.keys()) != expected_servers or any(
            len(bands) != self.bands for bands in loaded.values()
        ):
            raise ValueError(
                "Loaded HMAC keys are incompatible with current k/n/bands."
            )
        return loaded

    def _load_or_create_keys(
        self, key_store_path: Optional[str], key_env_var: Optional[str], key_encrypt_env_var: Optional[str]
    ) -> Dict[int, List[bytes]]:
        """Load band HMAC keys from env or disk, else create them with a CSPRNG."""
        enc_key = None
        if key_encrypt_env_var and key_encrypt_env_var in os.environ:
            raw = os.environ[key_encrypt_env_var]
            try:
                if all(c in "0123456789abcdefABCDEF" for c in raw) and len(raw) % 2 == 0:
                    enc_key = bytes.fromhex(raw)
                else:
                    enc_key = base64.b64decode(raw)
                if len(enc_key) not in (16, 24, 32):
                    raise ValueError("Encryption key must be 128/192/256-bit.")
            except Exception as exc:
                raise ValueError(f"Failed to decode encryption key from env {key_encrypt_env_var}: {exc}") from exc

        # 1) Environment override (base64 encoded JSON of {server: [hex,...]})
        if key_env_var and key_env_var in os.environ:
            raw = os.environ[key_env_var]
            try:
                decoded = base64.b64decode(raw)
                data = json.loads(decoded)
                loaded: Dict[int, List[bytes]] = {
                    int(server_str): [bytes.fromhex(h) for h in band_hex]
                    for server_str, band_hex in data.items()
                }
                return self._validate_loaded_keys(loaded)
            except Exception as exc:
                raise ValueError(
                    f"Failed to load HMAC keys from env var {key_env_var}: {exc}"
                ) from exc

        # 2) File-based load
        if key_store_path and os.path.exists(key_store_path):
            with open(key_store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("enc") == "aesgcm":
                if enc_key is None:
                    raise ValueError("Encrypted key file found but no encryption key provided.")
                nonce = base64.b64decode(data["nonce"])
                ct = base64.b64decode(data["data"])
                try:
                    plain = AESGCM(enc_key).decrypt(nonce, ct, None)
                    data = json.loads(plain.decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"Failed to decrypt HMAC key file: {exc}") from exc
            loaded = {
                int(server_str): [bytes.fromhex(h) for h in band_hex]
                for server_str, band_hex in data.items()
            }
            return self._validate_loaded_keys(loaded)

        # 3) Create new keys
        keys = {
            x: [secrets.token_bytes(32) for _ in range(self.bands)]
            for x in range(1, self.n + 1)
        }
        if key_store_path:
            os.makedirs(os.path.dirname(key_store_path) or ".", exist_ok=True)
            to_dump = {str(server): [k.hex() for k in band_keys] for server, band_keys in keys.items()}
            if enc_key is not None:
                nonce = secrets.token_bytes(12)
                ct = AESGCM(enc_key).encrypt(nonce, json.dumps(to_dump).encode("utf-8"), None)
                wrapped = {
                    "enc": "aesgcm",
                    "nonce": base64.b64encode(nonce).decode("ascii"),
                    "data": base64.b64encode(ct).decode("ascii"),
                    "alg": "AESGCM",
                    "hash": sha256(json.dumps(to_dump).encode("utf-8")).hexdigest(),
                }
                with open(key_store_path, "w", encoding="utf-8") as f:
                    json.dump(wrapped, f, indent=2)
            else:
                with open(key_store_path, "w", encoding="utf-8") as f:
                    json.dump(to_dump, f, indent=2)
        else:
            print(
                "[WARN] No key_store_path provided; HMAC keys live in-memory only and will be lost on restart."
            )
        return keys

    def _index_band_tokens_for_image(self, image_id: str, phash64: int) -> None:
        band_values = split_bands(phash64, bands=self.bands)
        for server in range(1, self.n + 1):
            for band_idx, value in enumerate(band_values):
                key = self.hmac_keys[server][band_idx]
                val_bytes = value.to_bytes(self.band_bytes, "big")
                if self.use_oprf and self.oprf_handler:
                    token = self.oprf_handler.evaluate(val_bytes, server=server)
                else:
                    token = hmac_token(key, val_bytes, outlen=self.token_len)
                bucket = self.server_band_buckets[server][band_idx].setdefault(
                    token, set()
                )
                bucket.add(image_id)

    def _add_image_with_phash(self, image_id: str, image_path: str, phash: int) -> int:
        self._images[image_id] = image_path
        secret_bytes = hash64_to_bytes(phash)
        shares = shamir_share_bytes(secret_bytes, self.k, self.n)
        for server, share_bytes in shares.items():
            self.server_shares[server][image_id] = StoredShare(
                image_id=image_id, share_bytes=share_bytes
            )
        self._index_band_tokens_for_image(image_id, phash)
        return phash

    def add_image(self, image_id: str, image_path: str) -> int:
        """Register an image and return its pHash."""
        phash = phash64(image_path)
        return self._add_image_with_phash(image_id, image_path, phash)

    def add_image_with_phash(self, image_id: str, image_path: str, phash: int) -> int:
        """Register an image using a precomputed pHash."""
        return self._add_image_with_phash(image_id, image_path, phash)

    def list_servers(self) -> List[int]:
        """Return the configured server identifiers."""
        return list(sorted(self.server_shares.keys()))

    def _reconstruct_hash_from_servers(
        self,
        image_id: str,
        servers: Iterable[int],
    ) -> Optional[int]:
        xs = sorted(set(int(s) for s in servers))
        if len(xs) < self.k:
            return None
        subshares: Dict[int, List[int]] = {}
        for server in xs:
            stored = self.server_shares.get(server, {}).get(image_id)
            if stored is None:
                return None
            subshares[server] = stored.share_bytes
        recovered = shamir_recover_bytes(subshares)
        return bytes_to_hash64(recovered)

    def get_shares(
        self,
        image_id: str,
        servers: Iterable[int],
    ) -> Optional[Dict[int, List[int]]]:
        shares: Dict[int, List[int]] = {}
        for server in servers:
            stored = self.server_shares.get(server, {}).get(image_id)
            if stored is None:
                return None
            shares[int(server)] = stored.share_bytes
        return shares

    def preselect_candidates(
        self,
        query_hash: int,
        servers_for_query: Iterable[int],
        min_band_votes: int = 3,
        dummy_band_queries: int = 0,
        pad_band_queries: Optional[int] = None,
        fixed_band_queries: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Return candidate image IDs with vote counts using band token matches."""
        xs = sorted(set(int(s) for s in servers_for_query))
        band_values = split_bands(query_hash, bands=self.bands)
        votes: Dict[str, int] = {}
        for server in xs:
            for band_idx, value in enumerate(band_values):
                key = self.hmac_keys[server][band_idx]
                val_bytes = value.to_bytes(self.band_bytes, "big")
                if self.use_oprf and self.oprf_handler:
                    token = self.oprf_handler.evaluate(val_bytes, server=server)
                else:
                    token = hmac_token(
                        key,
                        val_bytes,
                        outlen=self.token_len,
                    )
                matches = self.server_band_buckets[server][band_idx].get(token, set())
                for image_id in matches:
                    votes[image_id] = votes.get(image_id, 0) + 1
                # ダミー問い合わせでアクセスパターンの揺らぎを抑える（単なるパディング）
                dummy_total = dummy_band_queries
                if pad_band_queries is not None:
                    dummy_total = max(pad_band_queries - 1, 0)
                if fixed_band_queries is not None:
                    dummy_total = max(fixed_band_queries - 1, 0)
                for _ in range(dummy_total):
                    dummy_value = secrets.randbits(self.band_bits)
                    dummy_token = hmac_token(
                        key, dummy_value.to_bytes(self.band_bytes, "big"), outlen=self.token_len
                    )
                    _ = self.server_band_buckets[server][band_idx].get(dummy_token, set())
        filtered = [(image_id, count) for image_id, count in votes.items() if count >= min_band_votes]
        filtered.sort(key=lambda item: (-item[1], item[0]))
        return filtered

    def rank_candidates(
        self,
        query_hash: int,
        servers_for_query: Iterable[int],
        candidates: Iterable[str],
        topk: int = 10,
        max_hamming: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Reconstruct candidate hashes and return the closest matches."""
        xs = sorted(set(int(s) for s in servers_for_query))
        if len(xs) < self.k:
            raise ValueError(f"Need at least k={self.k} servers to reconstruct.")
        results: List[Tuple[str, int]] = []
        for image_id in candidates:
            recovered = self._reconstruct_hash_from_servers(image_id, xs)
            if recovered is None:
                continue
            distance = hamming64(query_hash, recovered)
            if max_hamming is not None and distance > max_hamming:
                continue
            results.append((image_id, distance))
        results.sort(key=lambda item: (item[1], item[0]))
        return results[:topk]

    def rank_candidates_secure(
        self,
        query_hash: int,
        servers_for_query: Iterable[int],
        candidates: Iterable[str],
        topk: int = 10,
        max_hamming: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Rank candidates using疑似MPCハミング距離."""
        xs = sorted(set(int(s) for s in servers_for_query))
        if len(xs) < self.k:
            raise ValueError(f"Need at least k={self.k} servers to reconstruct.")
        query_shares_all = shamir_share_bytes(hash64_to_bytes(query_hash), self.k, self.n)
        query_shares = {server: query_shares_all[server] for server in xs}
        results: List[Tuple[str, int]] = []
        for image_id in candidates:
            candidate_shares = self.get_shares(image_id, xs)
            if candidate_shares is None:
                continue
            distance = simulated_secure_hamming(query_shares, candidate_shares)
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
        """High-level API combining pre-selection and distance ranking."""
        query_hash = phash64(query_image_path)
        preselected = self.preselect_candidates(
            query_hash,
            servers_for_query,
            min_band_votes=min_band_votes,
            dummy_band_queries=dummy_band_queries,
            pad_band_queries=pad_band_queries,
            fixed_band_queries=fixed_band_queries,
        )
        candidate_ids = [image_id for image_id, _ in preselected]
        ranked = self.rank_candidates(
            query_hash,
            servers_for_query,
            candidate_ids,
            topk=topk,
            max_hamming=max_hamming,
        )
        return {
            "query_phash": f"0x{query_hash:016x}",
            "preselected": preselected,
            "ranked": ranked,
            "servers": list(sorted(set(int(s) for s in servers_for_query))),
            "params": {
                "min_band_votes": min_band_votes,
                "topk": topk,
                "max_hamming": max_hamming,
            },
        }


__all__ = ["StoredShare", "SearchableSISIndex"]
