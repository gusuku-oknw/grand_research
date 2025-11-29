"""
Plot key timing metrics with mean ± 95% CI using the summary CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_error_bars(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    x = df["mode"]
    means = df[f"{metric}_mean"]
    cis = df[f"{metric}_ci"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, means, yerr=cis, fmt="o", capsize=4, markersize=6, linestyle="None")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (ms)")
    ax.set_title(f"{metric.replace('_', ' ')} ±95% CI")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def get_summary(metrics_summary: Path) -> pd.DataFrame:
    return pd.read_csv(metrics_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics summary with CI.")
    parser.add_argument("summary_csv", type=Path, help="CSV produced by metrics_stats.py")
    parser.add_argument("--output-dir", type=Path, default=Path("output/figures/metrics_ci"))
    args = parser.parse_args()

    df = get_summary(args.summary_csv)
    plot_error_bars(df, "total_ms", args.output_dir / "total_ms_ci.png")
    plot_error_bars(df, "stage2_ms", args.output_dir / "stage2_ms_ci.png")
    plot_error_bars(df, "stage2_bytes", args.output_dir / "stage2_bytes_ci.png")
    print(f"Saved CI-plots to {args.output_dir}")


if __name__ == "__main__":
    main()
