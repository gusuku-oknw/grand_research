"""
Generate a log-scale breakdown of Stage-2 (or total) latency across modes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("mode", as_index=False).agg(
        stage1_ms=pd.NamedAgg(column="stage1_ms", aggfunc="mean"),
        stage2_ms=pd.NamedAgg(column="stage2_ms", aggfunc="mean"),
        total_ms=pd.NamedAgg(column="total_ms", aggfunc="mean"),
    )
    return grouped.sort_values("total_ms", ascending=False)


def plot_log_bar(summary: pd.DataFrame, out_path: Path) -> None:
    labels = summary["mode"].tolist()
    total = summary["total_ms"].tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, total)
    ax.set_yscale("log")
    ax.set_ylabel("Total Time (ms, log scale)")
    ax.set_title("Mode Latency (log scale)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height * 1.1, f"{height:.1f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot latency by mode on a log scale (optional exclusions)."
    )
    parser.add_argument("metrics_csv", type=Path, help="Path to metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("output/figures/log_latency"))
    parser.add_argument("--exclude", type=str, nargs="+", default=["aes_gcm"])
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    if args.exclude:
        df = df[~df["mode"].isin(args.exclude)]
    summary = summarize(df)
    plot_log_bar(summary, args.output_dir / "latency_log_scale.png")
    print("Saved log-scale latency figure to", args.output_dir / "latency_log_scale.png")


if __name__ == "__main__":
    main()
