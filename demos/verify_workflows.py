"""Verification script comparing SIS workflows with/without simulated MPC."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from PIL import Image

from pHR_SIS.image_store import ShamirImageStore
from pHR_SIS.workflow import SearchableSISWithImageStore


def _create_sample_images(path: Path, count: int = 3) -> List[Path]:
    """Generate simple, visually distinct sample images for verification."""
    path.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    size = 64
    for idx in range(count):
        img = Image.new("RGB", (size, size))
        for y in range(size):
            for x in range(size):
                if idx == 0:
                    r = int(255 * x / (size - 1))
                    g = int(255 * y / (size - 1))
                    b = 128
                elif idx == 1:
                    r = 64 if (x // 8) % 2 == 0 else 192
                    g = 192 if (y // 8) % 2 == 0 else 64
                    b = 64
                else:
                    r = int(255 * (x + y) / (2 * (size - 1)))
                    g = int(255 * ((size - 1 - x) + y) / (2 * (size - 1)))
                    b = int(255 * (x % 16) / 15)
                img.putpixel((x, y), (r, g, b))
        file_path = path / f"sample_{idx:02d}.png"
        img.save(file_path)
        files.append(file_path)
    return files


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify SIS workflows (standard vs simulated MPC).")
    parser.add_argument("--images_dir", type=Path, default=Path("images"))
    parser.add_argument("--shares_dir", type=Path, default=Path("output/img_shares"))
    parser.add_argument("--meta_dir", type=Path, default=Path("output/img_meta"))
    parser.add_argument("--recon_dir", type=Path, default=Path("output/recon_out"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Regenerate images even if directory exists.")
    parser.add_argument("--scratch_dir", type=Path, default=Path("verify_artifacts"))
    args = parser.parse_args()

    recon_dir_root = args.recon_dir
    recon_dir_root.mkdir(parents=True, exist_ok=True)

    have_existing = (
        not args.force
        and args.images_dir.exists()
        and any(args.images_dir.glob("*.png"))
    )

    if have_existing:
        images_dir = args.images_dir
        shares_dir = args.shares_dir
        meta_dir = args.meta_dir
        files = sorted(images_dir.glob("*.png"))
        if len(files) < args.count:
            raise SystemExit(f"Not enough images in {images_dir}. Use --force to generate samples.")
        files = files[: args.count]
        scratch = args.scratch_dir
        recon_dir_tmp = scratch / "recon_out"
        recon_dir_tmp.mkdir(parents=True, exist_ok=True)
    else:
        scratch = args.scratch_dir
        if args.force and scratch.exists():
            shutil.rmtree(scratch)
        images_dir = scratch / "images"
        shares_dir = scratch / "img_shares"
        meta_dir = scratch / "img_meta"
        recon_dir_tmp = scratch / "recon_out"
        files = _create_sample_images(images_dir, count=args.count)

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
        store.add_image(image_id, str(path), rng_seed=2025, skip_if_exists=True)
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
        recon_dir=str(recon_dir_tmp),
    )
    result_mpc = workflow_mpc.query_and_optionally_reconstruct(
        query,
        servers_for_query=servers,
        min_band_votes=2,
        topk=3,
        max_hamming=10,
        reconstruct_top=1,
        recon_dir=str(recon_dir_tmp),
    )

    if result_standard["ranked"] != result_mpc["ranked"]:
        raise SystemExit("MPC and standard rankings differ.")

    recon_target = "img_0000"
    final_path = recon_dir_root / "reconstructed_demo.png"
    recon_ok = store.reconstruct(recon_target, servers, str(final_path))
    if not recon_ok:
        raise SystemExit("SIS-only reconstruction failed.")

    print("Verification complete.")
    print(f"Ranked results: {result_standard['ranked']}")
    print(f"Reconstruction output: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
