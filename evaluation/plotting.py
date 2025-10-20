"""
Matplotlib helpers for SIS + pHash experiment reports.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence
import json

import matplotlib.pyplot as plt
import numpy as np


NUMERIC_FIELDS = {
    "precision_at_1",
    "precision_at_5",
    "precision_at_10",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "map",
    "phash_ms",
    "stage_a_ms",
    "stage_b_ms",
    "stage_c_ms",
    "total_ms",
    "bytes_a",
    "bytes_b",
    "bytes_c",
}

INT_FIELDS = {
    "n_dataset",
    "n_candidates_a",
    "n_candidates_b",
    "n_candidates_c",
    "n_reconstructed",
    "tau",
}


@dataclass(frozen=True)
class MetricRecord:
    """Typed representation of a single CSV metric row."""

    dataset: str
    mode: str
    query_key: str
    query_variant: str
    transform: str
    tau: int
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    map: float
    phash_ms: float
    stage_a_ms: float
    stage_b_ms: float
    stage_c_ms: float
    total_ms: float
    n_dataset: int
    n_candidates_a: int
    n_candidates_b: int
    n_candidates_c: int
    n_reconstructed: int
    bytes_a: float
    bytes_b: float
    bytes_c: float

    @classmethod
    def from_dict(cls, row: Mapping[str, str]) -> "MetricRecord":
        data: Dict[str, object] = {}
        for key, value in row.items():
            if key in NUMERIC_FIELDS:
                data[key] = float(value)
            elif key in INT_FIELDS:
                data[key] = int(float(value))  # tolerate float-looking ints
            else:
                data[key] = value
        return cls(**data)  # type: ignore[arg-type]


def load_metrics_csv(path: Path) -> List[MetricRecord]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [MetricRecord.from_dict(row) for row in reader]


def group_by_mode(records: Sequence[MetricRecord]) -> Dict[str, List[MetricRecord]]:
    groups: Dict[str, List[MetricRecord]] = defaultdict(list)
    for record in records:
        groups[record.mode].append(record)
    return groups


def filter_base_tau(records: Sequence[MetricRecord]) -> List[MetricRecord]:
    if not records:
        return []
    min_tau = min(rec.tau for rec in records)
    return [rec for rec in records if rec.tau == min_tau]


def summarize_variants(groups: Mapping[str, Sequence[MetricRecord]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode, records in groups.items():
        filtered = filter_base_tau(records)
        if not filtered:
            continue
        variant_groups: Dict[str, List[MetricRecord]] = defaultdict(list)
        for rec in filtered:
            variant_groups[rec.query_variant].append(rec)
        if not variant_groups:
            continue
        mode_summary: Dict[str, Dict[str, float]] = {}
        for variant, recs in variant_groups.items():
            arr = np.array(
                [
                    [
                        rec.precision_at_1,
                        rec.precision_at_5,
                        rec.precision_at_10,
                        rec.recall_at_1,
                        rec.recall_at_5,
                        rec.recall_at_10,
                        rec.map,
                    ]
                    for rec in recs
                ],
                dtype=np.float64,
            )
            mode_summary[variant] = {
                "precision_at_1": float(arr[:, 0].mean()),
                "precision_at_5": float(arr[:, 1].mean()),
                "precision_at_10": float(arr[:, 2].mean()),
                "recall_at_1": float(arr[:, 3].mean()),
                "recall_at_5": float(arr[:, 4].mean()),
                "recall_at_10": float(arr[:, 5].mean()),
                "map": float(arr[:, 6].mean()),
            }
        if mode_summary:
            summary[mode] = mode_summary
    return summary


def summarize_stage_ratios(groups: Mapping[str, Sequence[MetricRecord]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for mode, records in groups.items():
        filtered = filter_base_tau(records)
        if not filtered:
            continue
        arr = np.array(
            [
                [
                    rec.n_candidates_a / rec.n_dataset if rec.n_dataset else 0.0,
                    rec.n_candidates_b / rec.n_dataset if rec.n_dataset else 0.0,
                    rec.n_candidates_c / rec.n_dataset if rec.n_dataset else 0.0,
                ]
                for rec in filtered
            ],
            dtype=np.float64,
        )
        if not len(arr):
            continue
        summary[mode] = {
            "stage_a_ratio": float(arr[:, 0].mean()),
            "stage_b_ratio": float(arr[:, 1].mean()),
            "stage_c_ratio": float(arr[:, 2].mean()),
        }
    return summary


def summarize_reconstruction_ratio(groups: Mapping[str, Sequence[MetricRecord]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for mode, records in groups.items():
        filtered = filter_base_tau(records)
        if not filtered:
            continue
        ratios = [
            (rec.n_reconstructed / rec.n_dataset) if rec.n_dataset else 0.0
            for rec in filtered
        ]
        if ratios:
            summary[mode] = float(np.mean(ratios))
    return summary


def average_precision_table(groups: Mapping[str, Sequence[MetricRecord]]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for mode, records in groups.items():
        if not records:
            continue
        arr = np.array(
            [
                [
                    rec.precision_at_1,
                    rec.precision_at_5,
                    rec.precision_at_10,
                    rec.recall_at_10,
                    rec.map,
                ]
                for rec in records
            ]
        )
        table[mode] = {
            "P@1": float(arr[:, 0].mean()),
            "P@5": float(arr[:, 1].mean()),
            "P@10": float(arr[:, 2].mean()),
            "R@10": float(arr[:, 3].mean()),
            "mAP": float(arr[:, 4].mean()),
        }
    return table


def plot_precision_bars(groups: Mapping[str, Sequence[MetricRecord]], output_path: Path) -> None:
    table = average_precision_table(groups)
    if not table:
        return
    modes = list(table.keys())
    metrics = ["P@1", "P@5", "P@10", "R@10", "mAP"]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, mode in enumerate(modes):
        values = [table[mode][metric] for metric in metrics]
        ax.bar(x + idx * width, values, width, label=mode)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision / Recall Summary")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_candidate_reduction(groups: Mapping[str, Sequence[MetricRecord]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["Stage-A", "Stage-B", "Stage-C"]
    for mode, records in groups.items():
        if not records:
            continue
        arr = np.array(
            [
                [
                    rec.n_candidates_a,
                    rec.n_candidates_b,
                    rec.n_candidates_c,
                ]
                for rec in records
            ]
        )
        mean_vals = arr.mean(axis=0)
        ax.plot(stages, mean_vals, marker="o", label=mode)
    ax.set_ylabel("Average Candidates")
    ax.set_title("Candidate Reduction per Stage")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_time_breakdown(groups: Mapping[str, Sequence[MetricRecord]], output_path: Path) -> None:
    modes: List[str] = []
    stacks: List[np.ndarray] = []
    for mode, records in groups.items():
        if not records:
            continue
        arr = np.array(
            [
                [
                    rec.phash_ms,
                    rec.stage_a_ms,
                    rec.stage_b_ms,
                    rec.stage_c_ms,
                ]
                for rec in records
            ]
        )
        modes.append(mode)
        stacks.append(arr.mean(axis=0))
    if not stacks:
        return
    stacks_arr = np.vstack(stacks)
    labels = ["pHash", "Stage-A", "Stage-B", "Stage-C"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(modes))
    for idx in range(stacks_arr.shape[1]):
        ax.bar(modes, stacks_arr[:, idx], bottom=bottoms, label=labels[idx])
        bottoms += stacks_arr[:, idx]
    ax.set_ylabel("Time (ms)")
    ax.set_title("Stage Timing Breakdown")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_bytes_breakdown(groups: Mapping[str, Sequence[MetricRecord]], output_path: Path) -> None:
    modes: List[str] = []
    stacks: List[np.ndarray] = []
    for mode, records in groups.items():
        if not records:
            continue
        arr = np.array(
            [
                [
                    rec.bytes_a,
                    rec.bytes_b,
                    rec.bytes_c,
                ]
                for rec in records
            ]
        )
        modes.append(mode)
        stacks.append(arr.mean(axis=0))
    if not stacks:
        return
    stacks_arr = np.vstack(stacks)
    labels = ["Stage-A", "Stage-B", "Stage-C"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(modes))
    for idx in range(stacks_arr.shape[1]):
        ax.bar(modes, stacks_arr[:, idx], bottom=bottoms, label=labels[idx])
        bottoms += stacks_arr[:, idx]
    ax.set_ylabel("Bytes per Query")
    ax.set_title("Average Communication Volume")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_latency_scatter(groups: Mapping[str, Sequence[MetricRecord]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, records in groups.items():
        if not records:
            continue
        precisions = [rec.precision_at_10 for rec in records]
        latencies = [rec.total_ms for rec in records]
        ax.scatter(latencies, precisions, label=mode, alpha=0.7)
    ax.set_xlabel("Total Time (ms)")
    ax.set_ylabel("Precision@10")
    ax.set_title("Precision vs Latency")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_variant_recall(groups: Mapping[str, Sequence[MetricRecord]], output_path: Path) -> None:
    summary = summarize_variants(groups)
    if not summary:
        return
    variant_names = sorted({variant for mode_summary in summary.values() for variant in mode_summary})
    modes_list = [mode for mode in summary if summary[mode]]
    if not variant_names or not modes_list:
        return
    width = 0.8 / len(modes_list)
    x = np.arange(len(variant_names))
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(variant_names)), 5))
    for idx, mode in enumerate(modes_list):
        values = [summary[mode].get(variant, {}).get("recall_at_10", np.nan) for variant in variant_names]
        ax.bar(x + idx * width, values, width, label=mode)
    ax.set_xticks(x + width * (len(modes_list) - 1) / 2)
    ax.set_xticklabels(variant_names, rotation=45, ha="right")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Recall@10 by Transform Variant")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_stage_ratio(groups: Mapping[str, Sequence[MetricRecord]], output_path: Path) -> None:
    summary = summarize_stage_ratios(groups)
    if not summary:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["Stage-A", "Stage-B", "Stage-C"]
    for mode, stats in summary.items():
        ax.plot(
            stages,
            [stats["stage_a_ratio"], stats["stage_b_ratio"], stats["stage_c_ratio"]],
            marker="o",
            label=mode,
        )
    ax.set_ylabel("Candidate Ratio vs Full Dataset")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Normalized Candidate Reduction per Stage")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_reconstruction_ratio(groups: Mapping[str, Sequence[MetricRecord]], output_path: Path) -> None:
    summary = summarize_reconstruction_ratio(groups)
    if not summary:
        return
    modes_list = list(summary.keys())
    values = [summary[mode] for mode in modes_list]
    if not modes_list:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(modes_list, values, color="steelblue")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Full Reconstruction")
    ax.set_ylabel("Reconstructed / Total")
    ax.set_ylim(0.0, max(1.05, max(values) * 1.05))
    ax.set_title("Selective Reconstruction Ratio")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_tau_sensitivity(summary_path: Path, output_path: Path) -> None:
    if not summary_path.exists():
        return
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not data:
        return
    first_mode = next(iter(data))
    tau_values = data[first_mode].get("tau_values")
    if not tau_values:
        return
    tau_array = np.array([float(v) for v in tau_values])
    fig, ax = plt.subplots(figsize=(6, 5))
    for mode, stats in data.items():
        mean_recall = stats.get("mean_recall")
        if not mean_recall:
            continue
        ax.plot(tau_array, mean_recall, marker="o", label=mode)
    ax.set_xlabel("Tau Threshold")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Recall vs Tau Sensitivity")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_plots(metrics_csv: Path, output_dir: Path) -> None:
    records = load_metrics_csv(metrics_csv)
    groups = group_by_mode(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_precision_bars(groups, output_dir / "precision_summary.png")
    plot_candidate_reduction(groups, output_dir / "candidate_reduction.png")
    plot_time_breakdown(groups, output_dir / "time_breakdown.png")
    plot_bytes_breakdown(groups, output_dir / "communication_breakdown.png")
    plot_latency_scatter(groups, output_dir / "precision_latency.png")
    plot_variant_recall(groups, output_dir / "variant_recall.png")
    plot_stage_ratio(groups, output_dir / "candidate_reduction_ratio.png")
    plot_reconstruction_ratio(groups, output_dir / "reconstruction_ratio.png")
    roc_summary_path = metrics_csv.parent / "roc_pr_summary.json"
    plot_tau_sensitivity(roc_summary_path, output_dir / "tau_sensitivity.png")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot SIS+pHash metrics using Matplotlib.")
    parser.add_argument("metrics_csv", type=Path, help="Path to metrics.csv generated by run_search_experiments.py")
    parser.add_argument("--output_dir", type=Path, default=Path("evaluation/figures"))
    args = parser.parse_args()

    generate_all_plots(args.metrics_csv, args.output_dir)
    print(f"Saved Matplotlib figures to {args.output_dir}")


if __name__ == "__main__":
    main()
