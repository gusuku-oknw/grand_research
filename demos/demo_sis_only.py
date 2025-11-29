"""Demo for pure Shamir Image Sharing without pHash search."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pHR_SIS.image_store import ShamirImageStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Share and reconstruct data with Shamir SIS.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--images_dir", type=Path, default=Path("data"))
    parser.add_argument("--shares_dir", type=Path, default=Path("output/img_shares"))
    parser.add_argument("--meta_dir", type=Path, default=Path("output/img_meta"))
    parser.add_argument("--recon_dir", type=Path, default=Path("output/recon_out"))
    parser.add_argument("--seed", type=int, default=2025)
    return parser


def ensure_images(path: Path) -> list[Path]:
    path.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in path.iterdir() if p.is_file())
    if not images:
        raise SystemExit(f"No data found in {path}. Add sample data and rerun.")
    return images


def main() -> int:
    args = build_parser().parse_args()
    images = ensure_images(args.images_dir)
    store = ShamirImageStore(
        k=args.k,
        n=args.n,
        shares_dir=str(args.shares_dir),
        meta_dir=str(args.meta_dir),
    )

    print(f"[CONFIG] k={args.k} n={args.n}")
    print(f"[PATHS] data={args.images_dir} shares={args.shares_dir} meta={args.meta_dir}")
    t0 = time.perf_counter()
    image_ids: list[str] = []
    for idx, path in enumerate(images):
        image_id = f"img_{idx:04d}"
        store.add_image(image_id, str(path), rng_seed=args.seed, skip_if_exists=False)
        image_ids.append(image_id)
        print(f"[SHARE] {image_id:<12} <- {path.name}")
    print(f"[DONE] shared {len(image_ids)} data in {time.perf_counter() - t0:.3f}s")

    args.recon_dir.mkdir(parents=True, exist_ok=True)
    target = image_ids[0]
    servers = list(range(1, args.k + 1))
    out_path = args.recon_dir / f"reconstructed_{target}.png"
    ok = store.reconstruct(target, servers, str(out_path))
    print(f"[RECONSTRUCT] {target} using servers={servers} -> {'OK' if ok else 'FAIL'} ({out_path if ok else 'n/a'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
