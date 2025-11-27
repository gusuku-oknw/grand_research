"""
Quick comparison of baseline vs VOPRF+padding on a small dataset.

Runs two configs on `data/tests` (by default) and plots add/query latency.
This is a lightweight proxy to visualize overhead—not a full COCO benchmark.

Usage:
    python scripts/compare_oprf_effect.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ensure repo root on path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pHR_SIS.workflow import SearchableSISWithImageStore  # noqa: E402


@dataclass
class RunConfig:
    name: str
    use_oprf: bool
    dummy_band_queries: int
    fixed_band_queries: Optional[int]
    shares_dir: Path
    meta_dir: Path
    recon_dir: Path


def run_once(images: List[Path], cfg: RunConfig, k: int, n: int, bands: int, min_band_votes: int,
             topk: int, max_hamming: Optional[int], reconstruct_top: int) -> Dict[str, float]:
    wf = SearchableSISWithImageStore(
        k=k,
        n=n,
        bands=bands,
        token_len=8,
        seed=2025,
        shares_dir=str(cfg.shares_dir),
        meta_dir=str(cfg.meta_dir),
        secure_distance=False,
        share_strategy="shamir",
        fusion_grid=8,
        fusion_threshold=None,
        dummy_band_queries=cfg.dummy_band_queries,
        fixed_band_queries=cfg.fixed_band_queries,
        use_oprf=cfg.use_oprf,
    )
    t_add = time.perf_counter()
    for idx, path in enumerate(images):
        wf.add_image(f"img_{idx:04d}", str(path))
    add_elapsed = time.perf_counter() - t_add

    qpath = str(images[0])
    t_query = time.perf_counter()
    wf.query_and_optionally_reconstruct(
        query_image_path=qpath,
        servers_for_query=wf.list_servers()[:k],
        min_band_votes=min_band_votes,
        topk=topk,
        max_hamming=max_hamming,
        reconstruct_top=reconstruct_top,
        recon_dir=str(cfg.recon_dir),
        dummy_band_queries=cfg.dummy_band_queries,
        fixed_band_queries=cfg.fixed_band_queries,
    )
    query_elapsed = time.perf_counter() - t_query

    return {"add_seconds": add_elapsed, "query_seconds": query_elapsed}


def main() -> None:
    images_dir = Path("data/tests")
    images = sorted(p for p in images_dir.iterdir() if p.is_file())
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    out_dir = Path("evaluation/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = Path("out")
    summary_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        RunConfig(
            name="baseline",
            use_oprf=False,
            dummy_band_queries=0,
            fixed_band_queries=None,
            shares_dir=Path("out/compare_baseline_shares"),
            meta_dir=Path("out/compare_baseline_meta"),
            recon_dir=Path("out/compare_baseline_recon"),
        ),
        RunConfig(
            name="voprf_padded",
            use_oprf=True,
            dummy_band_queries=1,
            fixed_band_queries=4,
            shares_dir=Path("out/compare_voprf_shares"),
            meta_dir=Path("out/compare_voprf_meta"),
            recon_dir=Path("out/compare_voprf_recon"),
        ),
    ]

    k, n, bands = 2, 3, 8
    min_band_votes, topk, max_hamming, reconstruct_top = 3, 10, 10, 1

    results: Dict[str, Dict[str, float]] = {}
    for cfg in configs:
        cfg.shares_dir.mkdir(parents=True, exist_ok=True)
        cfg.meta_dir.mkdir(parents=True, exist_ok=True)
        cfg.recon_dir.mkdir(parents=True, exist_ok=True)
        stats = run_once(
            images,
            cfg,
            k=k,
            n=n,
            bands=bands,
            min_band_votes=min_band_votes,
            topk=topk,
            max_hamming=max_hamming,
            reconstruct_top=reconstruct_top,
        )
        results[cfg.name] = stats

    # Save JSON
    summary_path = summary_dir / "compare_oprf_effect.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot
    labels = list(results.keys())
    add_vals = [results[k]["add_seconds"] for k in labels]
    query_vals = [results[k]["query_seconds"] for k in labels]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], add_vals, width, label="Add (s)")
    ax.bar([i + width / 2 for i in x], query_vals, width, label="Query (s)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Seconds (data/tests)")
    ax.set_title("Baseline vs VOPRF+Padding (Stage-A) on data/tests")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "oprf_padding_overhead.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison plot to {plot_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
