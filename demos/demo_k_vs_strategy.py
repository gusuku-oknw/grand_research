"""Compare Shamir vs. pHash-fusion SIS as `k` increases and visualize the difference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pHR_SIS.workflow import SearchableSISWithImageStore


def ensure_images(path: Path) -> List[Path]:
    path.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in path.iterdir() if p.is_file())
    if not images:
        raise SystemExit(f"No images in {path}. Add files and rerun.")
    return images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot how Shamir vs fusion behaves when `k` grows."
    )
    parser.add_argument("--images_dir", type=Path, default=Path("data"))
    parser.add_argument("--shares_dir", type=Path, default=Path("output/img_shares"))
    parser.add_argument("--meta_dir", type=Path, default=Path("output/img_meta"))
    parser.add_argument("--recon_dir", type=Path, default=Path("output/recon_out"))
    parser.add_argument("--output", type=Path, default=Path("k_strategy_compare.png"))
    parser.add_argument("--mode", choices=["standard", "mpc"], default="standard")
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
    parser.add_argument("--reconstruct_top", type=int, default=1)
    parser.add_argument("--query", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser


def collect_records(
    workflow: SearchableSISWithImageStore,
    query_path: str,
    max_servers: int,
    args: argparse.Namespace,
) -> List[Dict]:
    records: List[Dict] = []
    recon_dir = args.recon_dir / f"k_strategy_{workflow.share_strategy}"
    recon_dir.mkdir(parents=True, exist_ok=True)
    for count in range(1, max_servers + 1):
        servers = workflow.list_servers()[:count]
        print(f"[{workflow.share_strategy}] servers={servers} (target k={args.k})")
        try:
            result = workflow.query_and_optionally_reconstruct(
                query_path,
                servers_for_query=servers,
                min_band_votes=args.min_band_votes,
                topk=args.topk,
                max_hamming=args.max_hamming,
                reconstruct_top=args.reconstruct_top,
                recon_dir=str(recon_dir),
            )
        except ValueError as exc:
            print(f"  -> skipped (need k={workflow.index.k}): {exc}")
            records.append(
                {
                    "servers": count,
                    "top_distance": None,
                    "recon_count": 0,
                    "share_mode": workflow.share_strategy,
                    "fusion_mode": workflow.share_strategy != "shamir",
                }
            )
            if args.sleep > 0:
                time.sleep(args.sleep)
            continue
        ranked = result.get("ranked", [])
        top_distance = ranked[0][1] if ranked else None
        records.append(
            {
                "servers": count,
                "top_distance": top_distance,
                "recon_count": len(result.get("reconstructed", [])),
                "share_mode": result.get("share_mode", workflow.share_strategy),
                "fusion_mode": result.get("fusion_mode", False),
            }
        )
        if args.sleep > 0:
            time.sleep(args.sleep)
    return records


def plot(records_by_strategy: Dict[str, List[Dict]], target_k: int, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    colors = {"shamir": "tab:blue", "phash-fusion": "tab:orange"}

    for strategy, records in records_by_strategy.items():
        servers = [r["servers"] for r in records]
        distances = [r["top_distance"] if r["top_distance"] is not None else float("nan") for r in records]
        recon_counts = [r["recon_count"] for r in records]
        axes[0].plot(
            servers, distances, label=strategy, color=colors[strategy], marker="o", linewidth=2
        )
        axes[1].plot(
            servers, recon_counts, label=strategy, color=colors[strategy], marker="s", linewidth=2
        )
        for r in records:
            if strategy == "phash-fusion" and r["share_mode"] != "phash-fusion":
                axes[0].annotate(
                    "shamir fallback",
                    (r["servers"], r["top_distance"] or 0),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color="gray",
                )

    axes[0].set_ylabel("Top Hamming distance")
    axes[0].set_title("Shamir vs pHash-fusion as available servers increase")
    axes[0].axvline(target_k, color="black", linestyle="--", label=f"k target = {target_k}")
    axes[0].grid(True, linestyle=":", alpha=0.7)


    axes[1].set_ylabel("Reconstructed images")
    axes[1].set_xlabel("Servers used for query")
    axes[1].grid(True, linestyle=":", alpha=0.7)
    axes[1].legend()

    fallback_patch = Patch(facecolor="tab:orange", edgecolor="k", label="fusion fallback zone")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(fallback_patch)
    labels.append("fusion fallback zone")
    axes[0].legend(handles=handles, labels=labels)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved comparison plot to {output}")


def main() -> int:
    args = build_parser().parse_args()
    images = ensure_images(args.images_dir)
    recon_path = str(args.recon_dir)
    query_path = str(args.query) if args.query else str(images[0])

    strategies = ["shamir", "phash-fusion"]
    records_by_strategy: Dict[str, List[Dict]] = {}
    max_servers = min(len(images), args.n)
    for strategy in strategies:
        workflow = SearchableSISWithImageStore(
            k=args.k,
            n=args.n,
            bands=args.bands,
            token_len=args.token_len,
            seed=args.seed,
            shares_dir=str(args.shares_dir),
            meta_dir=str(args.meta_dir),
            secure_distance=args.mode == "mpc",
            share_strategy=strategy,
            fusion_grid=args.fusion_grid,
            fusion_threshold=args.fusion_threshold,
        )
        for idx, path in enumerate(images):
            workflow.add_image(f"img_{idx:04d}", str(path))
        records = collect_records(workflow, query_path, max_servers, args)
        records_by_strategy[strategy] = records

    plot(records_by_strategy, target_k=args.k, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
