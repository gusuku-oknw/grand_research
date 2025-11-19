"""Show how retrieved images evolve as `k` grows for Shamir vs. pHash-fusion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter


def blend_fusion_visual(candidate_img: Image.Image, distance: Optional[int], max_distance: int) -> Image.Image:
    resized = candidate_img.resize((120, 120))
    ratio = min(1.0, (distance if distance is not None else max_distance) / max(max_distance, 1))
    radius = 2 + ratio * 10
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=radius))
    clarity = 1.0 - ratio
    return Image.blend(blurred, resized, alpha=clarity)

from pHR_SIS.phash import phash64
from pHR_SIS.workflow import SearchableSISWithImageStore


def load_images_from_dir(path: Path) -> List[Path]:
    path.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in path.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"No images found under {path}.")
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize the recovered image per k step.")
    parser.add_argument("--images_dir", type=Path, default=Path("data"))
    parser.add_argument("--shares_dir", type=Path, default=Path("img_shares"))
    parser.add_argument("--meta_dir", type=Path, default=Path("img_meta"))
    parser.add_argument("--recon_dir", type=Path, default=Path("recon_out"))
    parser.add_argument("--output", type=Path, default=Path("figures/k_recovery_gallery.png"))
    parser.add_argument("--query", type=Path, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--token_len", type=int, default=8)
    parser.add_argument("--min_band_votes", type=int, default=3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_hamming", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--reconstruct_top", type=int, default=1)
    parser.add_argument("--secure", action="store_true")
    return parser


def run_query_strategy(
    strategy: str,
    args: argparse.Namespace,
    query_path: str,
    images: List[Path],
    max_servers: int,
) -> List[Dict[str, Optional[str]]]:
    workflow = SearchableSISWithImageStore(
        k=args.k,
        n=args.n,
        bands=args.bands,
        token_len=args.token_len,
        seed=args.seed,
        shares_dir=str(args.shares_dir),
        meta_dir=str(args.meta_dir),
        secure_distance=args.secure,
        share_strategy=strategy,
    )
    for idx, image_path in enumerate(images):
        workflow.add_image(f"img_{idx:04d}", str(image_path))

    workers = []
    all_servers = workflow.list_servers()
    query_hash = phash64(query_path)
    for count in range(1, min(max_servers, len(all_servers)) + 1):
        servers = all_servers[:count]
        try:
            result = workflow.query_and_optionally_reconstruct(
                query_path,
                servers_for_query=servers,
                min_band_votes=args.min_band_votes,
                topk=args.topk,
                max_hamming=args.max_hamming,
                reconstruct_top=args.reconstruct_top,
                recon_dir=str(args.recon_dir),
            )
        except ValueError as exc:
            preselected = workflow.index.preselect_candidates(
                query_hash, servers, min_band_votes=args.min_band_votes
            )
            candidate_id = preselected[0][0] if preselected else None
            workers.append(
                {
                    "strategy": strategy,
                    "k": count,
                    "image_id": candidate_id,
                    "image_path": workflow.index._images.get(candidate_id) if candidate_id else None,
                    "reconstructed": None,
                    "status": "awaiting k",
                    "distance": None,
                }
            )
            continue
        ranked = result.get("ranked", [])
        candidate_id = ranked[0][0] if ranked else None
        candidate_distance = ranked[0][1] if ranked else None
        recon_info = result.get("reconstructed", [])
        status = "fusion fallback" if result.get("fusion_mode") else "reconstructed"
        workers.append(
            {
                "strategy": strategy,
                "k": count,
                "image_id": candidate_id,
                "image_path": workflow.index._images.get(candidate_id) if candidate_id else None,
                "reconstructed": recon_info[0][1] if recon_info else None,
                "status": status,
                "distance": candidate_distance,
            }
        )
    return workers


def plot_gallery(
    records: Dict[str, List[Dict[str, Optional[str]]]],
    query_image: str,
    max_servers: int,
    output_path: Path,
    max_distance: int,
) -> None:
    strategies = list(records.keys())
    fig, axes = plt.subplots(
        nrows=len(strategies),
        ncols=max_servers,
        squeeze=False,
        figsize=(max_servers * 2.5, len(strategies) * 3),
    )
    query_img = Image.open(query_image).resize((120, 120))
    wait_img = Image.new("RGB", query_img.size, color=(20, 20, 20))
    wait_draw = ImageDraw.Draw(wait_img)
    wait_draw.text((10, 50), "Awaiting k", fill="white")
    for row_idx, strat in enumerate(strategies):
        for col_idx in range(max_servers):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            record = records[strat][col_idx] if col_idx < len(records[strat]) else None
            subtitle = f"k={col_idx+1}"
            if record and record["reconstructed"]:
                img_path = record["reconstructed"]
                img = Image.open(img_path).resize((120, 120))
                subtitle += " | recon"
            elif record and record["image_path"]:
                img_path = record["image_path"]
                img = Image.open(img_path).resize((120, 120))
                if record["status"] == "fusion fallback":
                    img = blend_fusion_visual(img, record["distance"], max_distance)
                    subtitle += " | fusion"
                else:
                    subtitle += " | candidate"
            else:
                img = wait_img.copy()
                subtitle += " | waiting"
            ax.imshow(img)
            ax.set_title(f"{strat} | {subtitle}", fontsize=8)
    fig.suptitle("Recovered / candidate images as servers increase", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved gallery to {output_path}")


def main() -> int:
    args = build_parser().parse_args()
    images = load_images_from_dir(args.images_dir)
    query_path = str(args.query) if args.query else str(images[0])
    max_servers = args.n
    records = {}
    for strategy in ["shamir", "phash-fusion"]:
        records[strategy] = run_query_strategy(
            strategy, args, query_path, images, max_servers
        )
    plot_gallery(records, query_path, max_servers, args.output, args.max_hamming)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
