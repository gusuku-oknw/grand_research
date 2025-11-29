"""Experiment support for dealer-free SIS simulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .mpc import compare_secure_vs_baseline
from .simulator import DealerFreeSimulator


def _collect_images(directory: Path) -> Sequence[Path]:
    extensions = ("*.jpg", "*.jpeg", "*.png")
    paths: list[Path] = []
    for pattern in extensions:
        paths.extend(sorted(directory.glob(pattern)))
    return tuple(sorted(paths))


def _plot_results(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    image_ids = df["image_id"].tolist()
    x = range(len(image_ids))
    width = 0.35

    axes[0, 0].bar([i - width / 2 for i in x], df["stage_a_baseline"], width, label="Baseline")
    axes[0, 0].bar([i + width / 2 for i in x], df["stage_a_distributed"], width, label="Dealer-free")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(image_ids, rotation=45, ha="right")
    axes[0, 0].set_ylabel("Bytes")
    axes[0, 0].set_title("Stage-1 Registration Bytes")
    axes[0, 0].legend()

    stage_b_mean = df[["stage_b_baseline", "stage_b_distributed"]].mean()
    axes[0, 1].bar(["Baseline", "Dealer-free"], stage_b_mean, color=["#1f77b4", "#ff7f0e"])
    axes[0, 1].set_ylabel("Bytes")
    axes[0, 1].set_title("Stage-2 Query Tokens")

    axes[1, 0].plot(image_ids, df["stage_c"], marker="o")
    axes[1, 0].set_ylabel("Bytes")
    axes[1, 0].set_title("Stage-2 Reconstruction Bytes")
    axes[1, 0].tick_params(axis="x", labelrotation=45)

    ratio = df["stage_a_distributed"] / df["stage_a_baseline"]
    axes[1, 1].plot(image_ids, ratio, marker="s")
    axes[1, 1].axhline(1.0, linestyle="--", color="gray")
    axes[1, 1].set_ylabel("Multiplier")
    axes[1, 1].set_title("DKG Overhead Ratio")
    axes[1, 1].tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_dealer_free_experiment(
    images_dir: Path,
    output: Path,
    k: int = 3,
    n: int = 5,
    bands: int = 8,
    token_len: int = 8,
    contributors: int = 3,
    padding_tokens: int = 4,
    use_oprf: bool = False,
    mpc_query_image: Path | None = None,
    mpc_servers: Sequence[int] | None = None,
    mpc_topk: int = 5,
    mpc_min_band_votes: int = 3,
    mpc_max_hamming: int | None = None,
) -> tuple[Path, Path, Path | None]:
    image_paths = _collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}.")

    simulator = DealerFreeSimulator(
        k=k,
        n=n,
        bands=bands,
        token_len=token_len,
        use_oprf=use_oprf,
        distributed_contributors=contributors,
        padding_tokens=padding_tokens,
    )

    records: list[dict[str, object]] = []
    for image_path in image_paths:
        image_id = image_path.stem
        registration = simulator.register_image(image_id, str(image_path))
        query = simulator.query_metrics(image_id)
        records.append(
            {
                "image_id": registration.image_id,
                "stage_a_baseline": registration.stage_a_baseline,
                "stage_a_distributed": registration.stage_a_distributed,
                "stage_b_baseline": query.stage_b_baseline,
                "stage_b_distributed": query.stage_b_distributed,
                "stage_c": query.stage_c,
                "share_length": registration.share_length,
                "phash": registration.phash,
            }
        )

    df = pd.DataFrame(records)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "dealer_free_metrics.csv"
    df.to_csv(metrics_path, index=False)
    plot_path = output / "dealer_free_experiment.png"
    _plot_results(df, plot_path)

    mpc_comparison_path: Path | None = None
    if mpc_query_image:
        servers = tuple(
            sorted(set(mpc_servers)) if mpc_servers else range(1, simulator.k + 1)
        )
        comparison = compare_secure_vs_baseline(
            simulator.index,
            mpc_query_image,
            servers=servers,
            min_band_votes=mpc_min_band_votes,
            topk=mpc_topk,
            max_hamming=mpc_max_hamming,
        )
        mpc_comparison_path = output / "secure_distance_comparison.json"
        mpc_comparison_path.write_text(json.dumps(comparison.to_dict(), indent=2))

    return metrics_path, plot_path, mpc_comparison_path
