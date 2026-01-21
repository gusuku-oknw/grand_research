from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_eval(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
        }
    )


def plot_phash_distances(images: List[Dict], out_path: Path) -> None:
    names = [img["image"] for img in images]
    dummy = [img["phash_dist_dummy"] for img in images]
    full = [img["phash_dist_full"] for img in images]
    less = [img["phash_dist_less_than_k1"] for img in images]
    blur = [img["baseline_blur"]["phash_dist"] for img in images]
    noise = [img["baseline_noise"]["phash_dist"] for img in images]

    x = np.arange(len(names))
    w = 0.18
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 2 * w, dummy, width=w, label="Dummy (k1)", color="#4C78A8")
    ax.bar(x - w, full, width=w, label="Full (k2)", color="#72B7B2")
    ax.bar(x, less, width=w, label="<k1 noise", color="#E45756")
    ax.bar(x + w, blur, width=w, label="Blur baseline", color="#F58518")
    ax.bar(x + 2 * w, noise, width=w, label="Noise baseline", color="#B279A2")
    ax.set_ylabel("Hamming distance (64-bit pHash)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 64)
    ax.set_title("pHash distances vs. original (proposed vs baselines)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_psnr(images: List[Dict], out_path: Path) -> None:
    # Filter to top 2 images only as requested for clarity
    images = images[:2]
    
    names = [img["image"] for img in images]
    psnr_dummy = [img["psnr_dummy_vs_original"] for img in images]
    psnr_full = [img["psnr_full_vs_original"] for img in images]
    psnr_blur = [img["baseline_blur"]["psnr"] for img in images]
    psnr_noise = [img["baseline_noise"]["psnr"] for img in images]

    x = np.arange(len(names))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot bars
    ax.bar(x - 1.5 * w, psnr_dummy, width=w, label="Dummy (k1)", color="#F58518")
    
    # Full is theoretically lossless (PSNR=inf). Plot as a high bar (capped) to be visible.
    full_capped = [60.0 if math.isinf(v) else v for v in psnr_full]
    ax.bar(x - 0.5 * w, full_capped, width=w, label="Full (k2)", color="#54A24B")
    
    ax.bar(x + 0.5 * w, psnr_blur, width=w, label="Blur baseline", color="#4C78A8")
    ax.bar(x + 1.5 * w, psnr_noise, width=w, label="Noise baseline", color="#B279A2")
    
    ax.axhline(30, color="#666666", linestyle="--", linewidth=1, label="30 dB reference")
    
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(0, 70)  # Increased margin (data capped at 60)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha="center")
    ax.set_title("Reconstruction quality (proposed vs baselines)")
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_attack_distances(images: List[Dict], out_path: Path) -> None:
    # Aggregate per-attack across images for multiple variants
    def _attacks(img: Dict, key: str, fallback: str | None = None) -> Dict[str, int]:
        if key in img:
            return img[key]
        if fallback and fallback in img:
            return img[fallback]
        raise KeyError(key)

    try:
        attack_keys = sorted(images[0]["attack_phash_distances_dummy"].keys())
        series = {
            "dummy (k1)": [img["attack_phash_distances_dummy"] for img in images],
            "full (k2)": [img["attack_phash_distances_full"] for img in images],
            "<k1 noise": [img["attack_phash_distances_less"] for img in images],
            "blur baseline": [img["baseline_blur"]["attack_phash_distances"] for img in images],
            "noise baseline": [img["baseline_noise"]["attack_phash_distances"] for img in images],
        }
    except KeyError:
        # Backward compatibility: older JSON only has attack_phash_distances for dummy/baselines
        attack_keys = sorted(images[0]["attack_phash_distances"].keys())
        series = {
            "dummy (k1)": [img["attack_phash_distances"] for img in images],
            "blur baseline": [img["baseline_blur"]["attack_phash_distances"] for img in images],
            "noise baseline": [img["baseline_noise"]["attack_phash_distances"] for img in images],
        }

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(attack_keys))
    w = 0.15
    palette = ["#4C78A8", "#54A24B", "#E45756", "#F58518", "#B279A2"]

    for idx, (name, stats_list) in enumerate(series.items()):
        means = []
        stds = []
        for key in attack_keys:
            vals = [s[key] for s in stats_list]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        ax.bar(x + (idx - 2) * w, means, width=w, yerr=stds, capsize=3, label=name, color=palette[idx % len(palette)])

    ax.set_ylabel("Hamming distance (mean ± sd)")
    ax.set_xticks(x)
    ax.set_xticklabels(attack_keys, rotation=15, ha="right")
    ax.set_ylim(0, 64)
    ax.set_title("Robustness under attacks (pHash distance)")
    ax.legend(title="variant", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_timings(dct: Dict, shamir: Dict, images: List[Dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax_dct = axes[0, 0]
    ax_shamir = axes[0, 1]
    ax_split = axes[1, 0]
    ax_recover = axes[1, 1]

    # DCT micro
    ax_dct.bar(["forward", "inverse"], [dct["forward_ms_mean"], dct["inverse_ms_mean"]], color="#4C78A8")
    ax_dct.set_ylabel("ms")
    ax_dct.set_title(f"DCT ({dct['size']}×{dct['size']})\nmax error={dct['max_reconstruction_error']:.2e}")

    # Shamir micro
    ax_shamir.bar(["split", "recover"], [shamir["split_ms_mean"], shamir["recover_ms_mean"]], color="#72B7B2")
    ax_shamir.set_ylabel("ms")
    ax_shamir.set_title(f"Shamir micro (n={shamir['n']}, k={shamir['k']}, {shamir['secret_len_bytes']}B)")

    # Per-image split
    names = [img["image"] for img in images]
    x = np.arange(len(names))
    ax_split.bar(x, [img["split_ms"] for img in images], color="#F58518")
    ax_split.set_ylabel("ms")
    ax_split.set_xticks(x)
    ax_split.set_xticklabels(names, rotation=20, ha="right")
    ax_split.set_title("Split time per image")

    # Per-image recover
    ax_recover.bar(
        x - 0.15, [img["recover_dummy_ms"] for img in images], width=0.3, label="dummy (k1)", color="#E45756"
    )
    ax_recover.bar(
        x + 0.15, [img["recover_full_ms"] for img in images], width=0.3, label="full (k2)", color="#54A24B"
    )
    ax_recover.scatter(
        x, [img["single_layer_shamir"]["recover_ms"] for img in images], marker="x", color="#222222", label="single-layer Shamir"
    )
    ax_recover.set_ylabel("ms")
    ax_recover.set_xticks(x)
    ax_recover.set_xticklabels(names, rotation=20, ha="right")
    ax_recover.set_title("Recover time per image")
    ax_recover.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot phash_masked_sis evaluation results.")
    parser.add_argument("--eval_json", type=Path, default=Path("output/phash_masked_sis_eval/phash_masked_sis_eval.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/phash_masked_sis_eval/figures"))
    args = parser.parse_args()

    data = _load_eval(args.eval_json)
    images = data["images"]
    dct = data["dct"]
    shamir = data["shamir"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()

    plot_phash_distances(images, args.output_dir / "phash_distances.png")
    plot_psnr(images, args.output_dir / "psnr.png")
    plot_attack_distances(images, args.output_dir / "attack_phash_distances.png")
    plot_timings(dct, shamir, images, args.output_dir / "timings.png")

    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
