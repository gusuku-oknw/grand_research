"""Persistent Shamir image store for full-resolution reconstruction."""

from __future__ import annotations

import base64
import hmac
import os
import secrets
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from .shamir import shamir_share_flat_bytes, shamir_recover_flat_bytes


def _load_image_u8(path: Path) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Load an RGB image as uint8 array plus shape metadata."""
    img = Image.open(path).convert("RGB")
    array = np.asarray(img, dtype=np.uint8)
    return array, array.shape


def _save_image_u8(path: Path, array: np.ndarray) -> None:
    """Write an RGB uint8 array to disk."""
    Image.fromarray(array, mode="RGB").save(path)


@dataclass
class ImageMeta:
    image_id: str
    shape: Tuple[int, int, int]
    filename: str


class ShamirImageStore:
    """Store that shards data into per-server Shamir shares."""

    def __init__(
        self,
        k: int = 3,
        n: int = 5,
        shares_dir: str = "img_shares",
        meta_dir: str = "img_meta",
        mac_env_var: str = "PHR_SIS_MAC_KEY",
        mac_key_path: str | None = None,
    ):
        if not (2 <= k <= n <= 20):
            raise ValueError("Require 2 <= k <= n <= 20.")
        self.k, self.n = k, n
        self.shares_dir = Path(shares_dir)
        self.meta_dir = Path(meta_dir)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        for server in range(1, n + 1):
            self._server_dir(server).mkdir(parents=True, exist_ok=True)
        self.mac_key_path = Path(mac_key_path) if mac_key_path else self.meta_dir / "mac_key.bin"
        self.mac_key = self._load_or_create_mac_key(mac_env_var)

    def _server_dir(self, server: int) -> Path:
        return self.shares_dir / f"server_{server}"

    def _load_or_create_mac_key(self, mac_env_var: str) -> bytes:
        # 1) env var (hex or base64) takes precedence
        if mac_env_var in os.environ:
            raw = os.environ[mac_env_var]
            try:
                if all(c in "0123456789abcdefABCDEF" for c in raw) and len(raw) % 2 == 0:
                    key = bytes.fromhex(raw)
                else:
                    key = base64.b64decode(raw)
                if len(key) < 16:
                    raise ValueError("MAC key too short")
                return key
            except Exception as exc:
                raise ValueError(f"Failed to decode MAC key from env {mac_env_var}: {exc}") from exc
        # 2) file load
        if self.mac_key_path.exists():
            data = self.mac_key_path.read_bytes()
            if len(data) < 16:
                raise ValueError("MAC key file is too short or corrupted.")
            return data
        # 3) create new
        key = secrets.token_bytes(32)
        self.mac_key_path.write_bytes(key)
        return key

    def _mac(self, *parts: bytes) -> bytes:
        h = hmac.new(self.mac_key, digestmod=sha256)
        for part in parts:
            h.update(part)
        return h.digest()

    def add_image(
        self,
        image_id: str,
        image_path: str | Path,
        rng_seed: int | None = None,
        skip_if_exists: bool = True,
    ) -> ImageMeta:
        """Shard an image across servers and persist metadata."""
        meta_path = self.meta_dir / f"{image_id}.npz"
        share_paths = [self._server_dir(x) / f"{image_id}.npz" for x in range(1, self.n + 1)]
        if skip_if_exists and meta_path.exists() and all(path.exists() for path in share_paths):
            try:
                with np.load(meta_path) as meta_np:
                    shape = tuple(int(v) for v in meta_np["shape"])
                    filename = str(meta_np["filename"])
                # verify shares & meta integrity
                self._verify_meta(meta_path, image_id)
                for server in range(1, self.n + 1):
                    self._verify_share_file(server, image_id)
                return ImageMeta(image_id=image_id, shape=shape, filename=filename)
            except Exception:
                # integrity check failed; fall through to regenerate
                pass

        array, shape = _load_image_u8(Path(image_path))
        # Use a secrets-derived seed if none provided to avoid weak RNG
        if rng_seed is None:
            rng_seed = int.from_bytes(secrets.token_bytes(16), "big")
        rng = np.random.default_rng(rng_seed)
        shares = shamir_share_flat_bytes(array.reshape(-1), self.k, self.n, rng)
        for server, values in shares.items():
            out_path = self._server_dir(server) / f"{image_id}.npz"
            mac = self._mac(image_id.encode(), str(server).encode(), values.tobytes())
            np.savez_compressed(out_path, share=values, mac=mac)
        meta_mac = self._mac(
            image_id.encode(),
            array.shape[0].to_bytes(8, "big"),
            array.shape[1].to_bytes(8, "big"),
            array.shape[2].to_bytes(8, "big"),
            Path(image_path).name.encode(),
        )
        np.savez_compressed(
            meta_path,
            shape=np.array(shape, dtype=np.int32),
            filename=Path(image_path).name,
            mac=meta_mac,
        )
        return ImageMeta(image_id=image_id, shape=shape, filename=Path(image_path).name)

    def _verify_meta(self, meta_path: Path, image_id: str) -> None:
        with np.load(meta_path) as meta_np:
            shape = tuple(int(v) for v in meta_np["shape"])
            filename = str(meta_np["filename"])
            stored_mac = meta_np.get("mac")
        if stored_mac is None:
            raise ValueError(f"Missing MAC for meta: {meta_path}")
        expected = self._mac(
            image_id.encode(),
            shape[0].to_bytes(8, "big"),
            shape[1].to_bytes(8, "big"),
            shape[2].to_bytes(8, "big"),
            filename.encode(),
        )
        if not hmac.compare_digest(bytes(stored_mac), expected):
            raise ValueError(f"Meta MAC mismatch for {meta_path}")

    def _verify_share_file(self, server: int, image_id: str) -> np.ndarray:
        share_path = self._server_dir(server) / f"{image_id}.npz"
        if not share_path.exists():
            raise ValueError(f"Share missing: server {server}, image {image_id}")
        with np.load(share_path) as share_np:
            share = share_np["share"]
            stored_mac = share_np.get("mac")
        if stored_mac is None:
            raise ValueError(f"Missing MAC for share: {share_path}")
        expected = self._mac(image_id.encode(), str(server).encode(), share.tobytes())
        if not hmac.compare_digest(bytes(stored_mac), expected):
            raise ValueError(f"Share MAC mismatch for {share_path}")
        return share

    def reconstruct(
        self,
        image_id: str,
        servers: Iterable[int],
        out_path: str | Path,
    ) -> bool:
        """Reconstruct an image using shares from at least k servers."""
        server_ids = sorted(set(int(s) for s in servers))
        if len(server_ids) < self.k:
            raise ValueError(f"Need at least k={self.k} servers to reconstruct.")
        meta_path = self.meta_dir / f"{image_id}.npz"
        if not meta_path.exists():
            return False
        # verify integrity before reconstruct
        self._verify_meta(meta_path, image_id)
        with np.load(meta_path) as meta_np:
            shape = tuple(int(v) for v in meta_np["shape"])
        subshares: Dict[int, np.ndarray] = {}
        for server in server_ids:
            try:
                subshares[server] = self._verify_share_file(server, image_id)
            except ValueError:
                return False
        flat = shamir_recover_flat_bytes(subshares)
        array = flat.reshape(shape)
        _save_image_u8(Path(out_path), array)
        return True


__all__ = ["ImageMeta", "ShamirImageStore"]
