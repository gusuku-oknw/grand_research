"""Visualize how available server count (k) impacts SIS ranking/reconstruction."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pHR_SIS.workflow import SearchableSISWithImageStore


def ensure_images(path: Path) -> List[Path]:
    path.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in path.iterdir() if p.is_file())
    if not images:
        raise SystemExit(f"No images found in {path}. Add files and rerun.")
    return images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot how different numbers of responding SIS servers affect ranking & reconstruction."
    )
    parser.add_argument("--images_dir", type=Path, default=Path("data"))
    parser.add_argument("--shares_dir", type=Path, default=Path("output/img_shares"))
    parser.add_argument("--meta_dir", type=Path, default=Path("output/img_meta"))
    parser.add_argument("--recon_dir", type=Path, default=Path("output/recon_out"))
    parser.add_argument("--output", type=Path, default=Path("k_sweep.png"))
    parser.add_argument("--mode", choices=["standard", "mpc"], default="standard")
    parser.add_argument("--share_strategy", choices=["shamir", "phash-fusion"], default="phash-fusion")
    parser.add_argument("--fusion_grid", type=int, default=8)
    parser.add_argument("--fusion_threshold", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--token_len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min_band_votes", type=int, default=3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_hamming", type=int, default=10)
    parser.add_argument("--reconstruct_top", type=int, default=0)
    parser.add_argument("--query", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between queries (seconds).")
    return parser


def summarize_records(records: List[dict]) -> None:
    print("k sweep summary:")
    for rec in records:
        mode = rec["share_mode"]
        dist = rec["top_distance"]
        recon = rec["recon_count"]
        note = "fusion fallback" if rec["fusion_mode"] else "shamir"
        print(f"  servers={rec['servers']:2d} => mode={mode:<12} {note:<15} top_dist={dist} recons={recon}")


def plot_records(records: List[dict], total_k: int, output_path: Path) -> None:
    counts = [rec["servers"] for rec in records]
    distances = [rec["top_distance"] for rec in records]
    recon_counts = [rec["recon_count"] for rec in records]
    share_modes = [rec["share_mode"] for rec in records]

    fig, (ax_dist, ax_recon) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": (2, 1)}
    )

    valid = [(c, d) for c, d in zip(counts, distances) if d is not None]
    if valid:
        ax_dist.plot([c for c, _ in valid], [d for _, d in valid], marker="o", label="Top Hamming distance")
    for c, mode, d in zip(counts, share_modes, distances):
        y = d if d is not None else 0
        color = "tab:blue" if mode == "shamir" else "tab:orange"
        ax_dist.scatter(c, y, color=color, s=120, edgecolor="k", linewidth=0.7, zorder=3)
    ax_dist.set_ylabel("Top Hamming distance")
    ax_dist.set_title("K-sweep: Ranking quality & fallback mode")
    ax_dist.axvline(total_k, color="gray", linestyle="--", label=f"k = {total_k}")
    ax_dist.set_xlim(0.5, max(counts) + 0.5)
    annotations = [
        ("shamir", "tab:blue"),
        ("phash-fusion (fallback)", "tab:orange"),
    ]
    custom_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="k",
            markersize=10,
            label=label,
        )
        for label, color in annotations
    ]
    base_handles, base_labels = ax_dist.get_legend_handles_labels()
    ax_dist.legend(
        handles=base_handles + custom_handles,
        labels=base_labels + [label for label, _ in annotations],
    )
    ax_dist.set_xticks(counts)

    ax_recon.bar(counts, recon_counts, color="tab:green", alpha=0.7)
    ax_recon.set_ylabel("Reconstructed hits")
    ax_recon.set_xlabel("Servers used for query")
    ax_recon.set_ylim(0, max(recon_counts, default=1) + 0.5)
    ax_recon.set_xticks(counts)
    for count, recon in zip(counts, recon_counts):
        ax_recon.text(count, recon + 0.05, str(recon), ha="center", va="bottom")

    annotations = [
        ("shamir", "tab:blue"),
        ("phash-fusion (fallback)", "tab:orange"),
    ]
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="k", markersize=10, label=label)
        for label, color in annotations
    ]
    ax_dist.legend(handles=legend_elements + ax_dist.get_legend_handles_labels()[0])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


def main() -> int:
    args = build_parser().parse_args()
    images = ensure_images(args.images_dir)
    workflow = SearchableSISWithImageStore(
        k=args.k,
        n=args.n,
        bands=args.bands,
        token_len=args.token_len,
        seed=args.seed,
        shares_dir=str(args.shares_dir),
        meta_dir=str(args.meta_dir),
        secure_distance=args.mode == "mpc",
        share_strategy=args.share_strategy,
        fusion_grid=args.fusion_grid,
        fusion_threshold=args.fusion_threshold,
    )

    recon_dir = args.recon_dir / "k_sweep"
    recon_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(images):
        workflow.add_image(f"img_{idx:04d}", str(path))

    query_path = str(args.query) if args.query else str(images[0])
    all_servers = workflow.list_servers()
    max_servers = min(len(all_servers), args.n)
    records: List[dict] = []

    for count in range(1, max_servers + 1):
        servers = all_servers[:count]
        print(f"[RUN] servers={servers} (k target: {args.k})")
        result = workflow.query_and_optionally_reconstruct(
            query_path,
            servers_for_query=servers,
            min_band_votes=args.min_band_votes,
            topk=args.topk,
            max_hamming=args.max_hamming,
            reconstruct_top=args.reconstruct_top,
            recon_dir=str(recon_dir),
        )
        ranked = result.get("ranked", [])
        top_distance = ranked[0][1] if ranked else None
        records.append(
            {
                "servers": count,
                "share_mode": result.get("share_mode", workflow.share_strategy),
                "fusion_mode": result.get("fusion_mode", False),
                "top_distance": top_distance,
                "recon_count": len(result.get("reconstructed", [])),
            }
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    summarize_records(records)
    plot_records(records, total_k=workflow.index.k, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
