from __future__ import annotations

import base64
import os
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Dict

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image


@dataclass
class EncryptedImageRecord:
    image_id: str
    nonce: bytes
    ciphertext: bytes


class AESGCMStorage:
    def __init__(self, aesgcm: AESGCM):
        self._aesgcm = aesgcm

    def encrypt_image(self, image_id: str, raw_bytes: bytes) -> EncryptedImageRecord:
        nonce = os.urandom(12)
        aad = image_id.encode("utf-8")
        ciphertext = self._aesgcm.encrypt(nonce, raw_bytes, aad)
        return EncryptedImageRecord(image_id=image_id, nonce=nonce, ciphertext=ciphertext)

    def decrypt_image(self, record: EncryptedImageRecord) -> bytes:
        aad = record.image_id.encode("utf-8")
        return self._aesgcm.decrypt(record.nonce, record.ciphertext, aad)


def load_aesgcm_master() -> AESGCM:
    key_b64 = os.environ.get("PHASH_AES_MASTER_KEY")
    if key_b64 is None:
        warnings.warn("PHASH_AES_MASTER_KEY not set; using a temporary random key for AES-GCM mode.")
        key = AESGCM.generate_key(bit_length=256)
    else:
        key = base64.b64decode(key_b64)
        if len(key) not in (16, 24, 32):
            raise ValueError("PHASH_AES_MASTER_KEY must be 128/192/256-bit base64")
    return AESGCM(key)


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    grid = np.arange(n)[None, :]
    mat = np.cos(np.pi * (grid + 0.5) * k / n)
    mat[0, :] *= np.sqrt(1 / n)
    mat[1:, :] *= np.sqrt(2 / n)
    return mat


def _dct2(x: np.ndarray) -> np.ndarray:
    rows, cols = x.shape
    c_row = _dct_matrix(rows)
    c_col = _dct_matrix(cols)
    return c_row @ x @ c_col.T


def compute_phash_from_bytes(data: bytes, resize: int = 32, band: int = 8) -> int:
    with Image.open(BytesIO(data)) as img:
        img = img.convert("L")
        img = img.resize((resize, resize), Image.BILINEAR)
        values = np.asarray(img, dtype=np.float32)
        dct = _dct2(values)
        low = dct[:band, :band]
        avg = float(low.mean())
        bits = (low.flatten() > avg).astype(np.uint8)
        h = 0
        for bit in bits:
            h = (h << 1) | int(bit)
    return h


__all__ = [
    "AESGCMStorage",
    "EncryptedImageRecord",
    "compute_phash_from_bytes",
    "load_aesgcm_master",
]
