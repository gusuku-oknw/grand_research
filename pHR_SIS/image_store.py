"""Persistent Shamir image store for full-resolution reconstruction."""

from __future__ import annotations

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
    """Store that shards images into per-server Shamir shares."""

    def __init__(
        self,
        k: int = 3,
        n: int = 5,
        shares_dir: str = "img_shares",
        meta_dir: str = "img_meta",
    ):
        if not (2 <= k <= n <= 20):
            raise ValueError("Require 2 <= k <= n <= 20.")
        self.k, self.n = k, n
        self.shares_dir = Path(shares_dir)
        self.meta_dir = Path(meta_dir)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        for server in range(1, n + 1):
            self._server_dir(server).mkdir(parents=True, exist_ok=True)

    def _server_dir(self, server: int) -> Path:
        return self.shares_dir / f"server_{server}"

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
            with np.load(meta_path) as meta_np:
                shape = tuple(int(v) for v in meta_np["shape"])
                filename = str(meta_np["filename"])
            return ImageMeta(image_id=image_id, shape=shape, filename=filename)

        array, shape = _load_image_u8(Path(image_path))
        rng = np.random.default_rng(rng_seed)
        shares = shamir_share_flat_bytes(array.reshape(-1), self.k, self.n, rng)
        for server, values in shares.items():
            out_path = self._server_dir(server) / f"{image_id}.npz"
            np.savez_compressed(out_path, share=values)
        np.savez_compressed(
            meta_path,
            shape=np.array(shape, dtype=np.int32),
            filename=Path(image_path).name,
        )
        return ImageMeta(image_id=image_id, shape=shape, filename=Path(image_path).name)

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
        with np.load(meta_path) as meta_np:
            shape = tuple(int(v) for v in meta_np["shape"])
        subshares: Dict[int, np.ndarray] = {}
        for server in server_ids:
            share_path = self._server_dir(server) / f"{image_id}.npz"
            if not share_path.exists():
                return False
            with np.load(share_path) as share_np:
                subshares[server] = share_np["share"]
        flat = shamir_recover_flat_bytes(subshares)
        array = flat.reshape(shape)
        _save_image_u8(Path(out_path), array)
        return True


__all__ = ["ImageMeta", "ShamirImageStore"]
