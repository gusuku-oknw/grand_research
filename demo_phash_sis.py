"""Unified demo for pHash + SIS workflows (standard or MPC simulated)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pHR_SIS.workflow import SearchableSISWithImageStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the searchable SIS demo with optional simulated MPC distance."
    )
    parser.add_argument("--mode", choices=["standard", "mpc"], default="standard")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--min_band_votes", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_hamming", type=int, default=10)
    parser.add_argument("--reconstruct_top", type=int, default=1)
    parser.add_argument("--images_dir", type=Path, default=Path("images"))
    parser.add_argument("--shares_dir", type=Path, default=Path("img_shares"))
    parser.add_argument("--meta_dir", type=Path, default=Path("img_meta"))
    parser.add_argument("--recon_dir", type=Path, default=Path("recon_out"))
    parser.add_argument("--query", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    return parser


def ensure_images(path: Path) -> list[Path]:
    path.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in path.iterdir() if p.is_file())
    if not images:
        raise SystemExit(f"No images found in {path}. Add sample images and rerun.")
    return images


def main() -> int:
    args = build_parser().parse_args()
    images = ensure_images(args.images_dir)
    workflow = SearchableSISWithImageStore(
        k=args.k,
        n=args.n,
        bands=args.bands,
        token_len=8,
        seed=args.seed,
        shares_dir=str(args.shares_dir),
        meta_dir=str(args.meta_dir),
        secure_distance=args.mode == "mpc",
    )

    print(f"[MODE] {args.mode}")
    print(f"[CONFIG] k={args.k} n={args.n} bands={args.bands} min_band_votes={args.min_band_votes} "
          f"topk={args.topk} max_hamming={args.max_hamming} reconstruct_top={args.reconstruct_top}")
    print(f"[PATHS] images={args.images_dir} shares={args.shares_dir} meta={args.meta_dir} recon={args.recon_dir}")

    t0 = time.perf_counter()
    for idx, path in enumerate(images):
        image_id = f"img_{idx:04d}"
        phash = workflow.add_image(image_id, str(path))
        print(f"[ADD] {image_id:<12} {path.name:<30} pHash=0x{phash:016x}")
    print(f"[DONE] registered {len(images)} images in {time.perf_counter() - t0:.3f}s")

    query_path = str(args.query) if args.query else str(images[0])
    servers = workflow.list_servers()[: args.k]
    print(f"[QUERY] file={Path(query_path).name} servers={servers}")
    result = workflow.query_and_optionally_reconstruct(
        query_path,
        servers_for_query=servers,
        min_band_votes=args.min_band_votes,
        topk=args.topk,
        max_hamming=args.max_hamming,
        reconstruct_top=args.reconstruct_top,
        recon_dir=str(args.recon_dir),
    )

    print(f"[P-HASH] {result['query_phash']}")
    print(f"[PRESELECT] {result['preselected'] or '(empty)'}")
    print(f"[RANKED/{result['mode']}] {result['ranked'] or '(empty)'}")
    if result["reconstructed"]:
        print("[RECONSTRUCTED]")
        for image_id, path in result["reconstructed"]:
            print(f"  - {image_id} -> {path}")
    else:
        print("[RECONSTRUCTED] (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
