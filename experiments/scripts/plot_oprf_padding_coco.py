"""
Plot baseline vs VOPRF+padding latency/comm from two COCO metrics.csv files.

Usage:
  python experiments/scripts/plot_oprf_padding_coco.py \
    --baseline metrics_baseline.csv \
    --oprf metrics_voprf.csv \
    --output output/figures/oprf_padding_coco
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: mode, total_ms, stage1_ms, stage2_ms, stage1_bytes, stage2_bytes, ...
    return df


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "stage1_ms": df["stage1_ms"].mean(),
        "stage2_ms": df["stage2_ms"].mean(),
        "total_ms": df["total_ms"].mean(),
        "stage1_bytes": df["stage1_bytes"].mean(),
        "stage2_bytes": df["stage2_bytes"].mean(),
    }


def plot_latency(baseline: Dict[str, float], oprf: Dict[str, float], out: Path) -> None:
    labels = ["Stage-1", "Stage-2", "Total"]
    base_vals = [baseline["stage1_ms"], baseline["stage2_ms"], baseline["total_ms"]]
    oprf_vals = [oprf["stage1_ms"], oprf["stage2_ms"], oprf["total_ms"]]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], base_vals, w, label="Baseline")
    ax.bar([i + w / 2 for i in x], oprf_vals, w, label="VOPRF+Padding")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("ms (avg over queries)")
    ax.set_title("Latency (COCO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_bytes(baseline: Dict[str, float], oprf: Dict[str, float], out: Path) -> None:
    labels = ["Stage-1", "Stage-2"]
    base_vals = [baseline["stage1_bytes"], baseline["stage2_bytes"]]
    oprf_vals = [oprf["stage1_bytes"], oprf["stage2_bytes"]]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], base_vals, w, label="Baseline")
    ax.bar([i + w / 2 for i in x], oprf_vals, w, label="VOPRF+Padding")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bytes (avg over queries)")
    ax.set_title("Communication (COCO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True, help="metrics.csv for baseline run")
    ap.add_argument("--oprf", type=Path, required=True, help="metrics.csv for VOPRF+padding run")
    ap.add_argument("--output", type=Path, default=Path("output/figures/oprf_padding_coco"))
    args = ap.parse_args()

    base_df = load_metrics(args.baseline)
    oprf_df = load_metrics(args.oprf)

    base_stats = summarize(base_df)
    oprf_stats = summarize(oprf_df)

    args.output.mkdir(parents=True, exist_ok=True)
    plot_latency(base_stats, oprf_stats, args.output / "latency_compare.png")
    plot_bytes(base_stats, oprf_stats, args.output / "bytes_compare.png")
    print(f"Saved plots to {args.output}")


if __name__ == "__main__":
    main()
