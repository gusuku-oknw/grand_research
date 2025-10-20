"""Verification script comparing SIS workflows with/without simulated MPC."""

from __future__ import annotations

import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from PIL import Image

from pHR_SIS.image_store import ShamirImageStore
from pHR_SIS.workflow import SearchableSISWithImageStore


def _create_sample_images(path: Path, count: int = 3) -> List[Path]:
    path.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    rng = random.Random(2025)
    for idx in range(count):
        arr = [[(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(32)] for _ in range(32)]
        img = Image.new("RGB", (32, 32))
        for y in range(32):
            for x in range(32):
                img.putpixel((x, y), arr[y][x])
        file_path = path / f"sample_{idx:02d}.png"
        img.save(file_path)
        files.append(file_path)
    return files


def main() -> int:
    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        images_dir = base / "images"
        shares_dir = base / "img_shares"
        meta_dir = base / "img_meta"
        recon_dir = base / "recon_out"

        files = _create_sample_images(images_dir)

        store = ShamirImageStore(k=3, n=5, shares_dir=str(shares_dir), meta_dir=str(meta_dir))
        workflow_std = SearchableSISWithImageStore(
            k=3,
            n=5,
            bands=8,
            token_len=8,
            seed=2025,
            shares_dir=str(shares_dir),
            meta_dir=str(meta_dir),
            secure_distance=False,
        )
        workflow_mpc = SearchableSISWithImageStore(
            k=3,
            n=5,
            bands=8,
            token_len=8,
            seed=2025,
            shares_dir=str(shares_dir),
            meta_dir=str(meta_dir),
            secure_distance=True,
        )

        for idx, path in enumerate(files):
            image_id = f"img_{idx:04d}"
            store.add_image(image_id, str(path), rng_seed=2025, skip_if_exists=False)
            workflow_std.add_image(image_id, str(path))
            workflow_mpc.add_image(image_id, str(path))

        query = str(files[0])
        servers = workflow_std.list_servers()[:3]

        result_standard = workflow_std.query_and_optionally_reconstruct(
            query,
            servers_for_query=servers,
            min_band_votes=2,
            topk=3,
            max_hamming=10,
            reconstruct_top=1,
            recon_dir=str(recon_dir),
        )
        result_mpc = workflow_mpc.query_and_optionally_reconstruct(
            query,
            servers_for_query=servers,
            min_band_votes=2,
            topk=3,
            max_hamming=10,
            reconstruct_top=1,
            recon_dir=str(recon_dir),
        )

        if result_standard["ranked"] != result_mpc["ranked"]:
            raise SystemExit("MPC and standard rankings differ.")

        recon_target = "img_0000"
        recon_ok = store.reconstruct(recon_target, servers, str(recon_dir / "reconstructed_demo.png"))
        if not recon_ok:
            raise SystemExit("SIS-only reconstruction failed.")

        print("Verification complete.")
        print(f"Ranked results: {result_standard['ranked']}")
        print(f"Reconstruction output: {recon_dir / 'reconstructed_demo.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
