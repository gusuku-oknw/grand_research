"""
Plot Recall@10 for a few important transforms across modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def load_filtered_metrics(metrics_csv: Path, transforms: Sequence[str], tau: int = 10) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    base_tau = df["tau"].min()
    filtered = df[
        (df["transform"].isin(transforms)) &
        (df["tau"] == base_tau)
    ]
    return filtered


def summarize(filtered: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        filtered.groupby(["mode", "transform"], as_index=False)
        .agg(recall_at_10_mean=("recall_at_10", "mean"))
    )
    return grouped


def plot_recall(summary: pd.DataFrame, transforms: Sequence[str], out_path: Path) -> None:
    modes = sorted(summary["mode"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, transform in enumerate(transforms):
        transform_data = summary[summary["transform"] == transform]
        recalls = []
        for mode in modes:
            row = transform_data[transform_data["mode"] == mode]
            recalls.append(row["recall_at_10_mean"].iloc[0] if not row.empty else 0.0)
        ax.plot(modes, recalls, marker="o", label=transform)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 for Key Transforms")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Transform")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Recall@10 across a selected set of transforms."
    )
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/figures/selected_recall.png"))
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=["original", "jpeg70", "crop5%", "rotate10%"],
        help="Transform names to include",
    )
    args = parser.parse_args()

    filtered = load_filtered_metrics(args.metrics_csv, args.transforms)
    summary = summarize(filtered)
    plot_recall(summary, args.transforms, args.output)
    print("Saved selected transforms recall plot to", args.output)


if __name__ == "__main__":
    main()
