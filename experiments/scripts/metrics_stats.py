"""
Compute mean, standard deviation, and approximate 95% confidence intervals for timing/bytes metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["stage1_ms", "stage2_ms", "total_ms", "stage1_bytes", "stage2_bytes"]
    summaries = []
    for mode, group in df.groupby("mode"):
        n = len(group)
        row = {"mode": mode, "n": n}
        for metric in metrics:
            mean = group[metric].mean()
            std = group[metric].std(ddof=1)
            ci = 1.96 * std / (n**0.5) if n > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std if pd.notna(std) else 0.0
            row[f"{metric}_ci"] = ci
        summaries.append(row)
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics with std/CI.")
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/results/metrics_summary.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    summary = summarize(df)
    summary.to_csv(args.output, index=False)
    print("Saved metrics summary with std/CI to", args.output)


if __name__ == "__main__":
    main()
