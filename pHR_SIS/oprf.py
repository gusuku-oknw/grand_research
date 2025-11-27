"""Minimal VOPRF-style helper using Ristretto (via `oprf`/`oblivious`).

This implements a basic blind-evaluate-unblind flow:
  - Client blinds hash(value) with random scalar r.
  - Server multiplies blinded point by secret scalar k.
  - Client unblinds: (k * r * H(x)) / r = k * H(x), then hashes to token.

Security note: This is an in-process VOPRF; in a real deployment the client and
server would be separate principals. Keys are persisted and can be encrypted
with AES-GCM using an env-provided key.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import oprf as oprf_lib


@dataclass
class OPRFKey:
    server: int
    scalar: bytes  # 32-byte scalar


def _load_enc_key(env_var: Optional[str]) -> Optional[bytes]:
    if not env_var or env_var not in os.environ:
        return None
    raw = os.environ[env_var]
    key = (
        bytes.fromhex(raw)
        if all(c in "0123456789abcdefABCDEF" for c in raw) and len(raw) % 2 == 0
        else base64.b64decode(raw)
    )
    if len(key) not in (16, 24, 32):
        raise ValueError("Encryption key must be 128/192/256-bit.")
    return key


def load_or_create_oprf_keys(
    n: int,
    key_store_path: Optional[str],
    key_env_var: Optional[str],
    key_encrypt_env_var: Optional[str],
) -> Dict[int, bytes]:
    enc_key = _load_enc_key(key_encrypt_env_var)

    if key_env_var and key_env_var in os.environ:
        raw = base64.b64decode(os.environ[key_env_var])
        data = json.loads(raw.decode("utf-8"))
        keys = {int(k): bytes.fromhex(v) for k, v in data.items()}
        _validate_oprf_keys(keys, n)
        return keys

    if key_store_path and os.path.exists(key_store_path):
        with open(key_store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("enc") == "aesgcm":
            if enc_key is None:
                raise ValueError("Encrypted OPRF key file found but no encryption key provided.")
            nonce = base64.b64decode(data["nonce"])
            ct = base64.b64decode(data["data"])
            plain = AESGCM(enc_key).decrypt(nonce, ct, None)
            data = json.loads(plain.decode("utf-8"))
        keys = {int(k): bytes.fromhex(v) for k, v in data.items()}
        _validate_oprf_keys(keys, n)
        return keys

    keys = {server: _random_scalar() for server in range(1, n + 1)}
    if key_store_path:
        os.makedirs(os.path.dirname(key_store_path) or ".", exist_ok=True)
        to_dump = {str(k): v.hex() for k, v in keys.items()}
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
    return keys


def _random_scalar() -> bytes:
    # Use mask.random() to ensure valid non-zero scalar
    return bytes(oprf_lib.mask.random())


def _validate_oprf_keys(keys: Dict[int, bytes], n: int) -> None:
    expected = set(range(1, n + 1))
    if set(keys.keys()) != expected:
        raise ValueError("OPRF key file/env is incompatible with server count.")
    if any(len(v) != 32 for v in keys.values()):
        raise ValueError("OPRF keys must be 32-byte scalars.")


class VOPRFClient:
    def __init__(self):
        pass

    def blind(self, value: bytes) -> Tuple[bytes, oprf_lib.mask]:
        x = oprf_lib.data.hash(value)
        r = oprf_lib.mask.random()
        blinded = r * x
        return bytes(blinded), r

    def finalize(self, evaluated: bytes, r: oprf_lib.mask, outlen: int) -> bytes:
        eval_point = oprf_lib.data(evaluated)
        unblinded = r.unmask(eval_point)
        return sha256(bytes(unblinded)).digest()[:outlen]


class VOPRFServer:
    def __init__(self, key_scalar: bytes):
        self.key = oprf_lib.mask(key_scalar)

    def evaluate(self, blinded: bytes) -> bytes:
        point = oprf_lib.data(blinded)
        evaluated = self.key * point
        return bytes(evaluated)


class VOPRFHandler:
    """Helper to compute opaque tokens deterministically via VOPRF."""

    def __init__(self, server_keys: Dict[int, bytes], token_len: int):
        self.client = VOPRFClient()
        self.servers: Dict[int, VOPRFServer] = {
            server: VOPRFServer(key) for server, key in server_keys.items()
        }
        self.token_len = token_len

    def evaluate(self, value: bytes, server: int) -> bytes:
        blinded, r = self.client.blind(value)
        evaluated = self.servers[server].evaluate(blinded)
        return self.client.finalize(evaluated, r, outlen=self.token_len)


__all__ = [
    "VOPRFHandler",
    "load_or_create_oprf_keys",
]
