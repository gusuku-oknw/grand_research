
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_eval(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_histogram(images, out_path: Path):
    # Extract distances
    dist_dummy = [img["phash_dist_dummy"] for img in images]
    dist_noise = [img["phash_dist_less_than_k1"] for img in images]
    
    # Setup plot
    plt.rcParams.update({
        "figure.dpi": 200,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot histograms
    bins = np.arange(0, 66, 1) - 0.5 # Center bars on integers
    
    ax.hist(dist_noise, bins=bins, alpha=0.6, label="Noise (r < k1)", color="#E45756", density=True, edgecolor='white', linewidth=0.5)
    ax.hist(dist_dummy, bins=bins, alpha=0.7, label="Dummy (k1 <= r < k2)", color="#4C78A8", density=True, edgecolor='white', linewidth=0.5)
    
    # Add theoretical bin for random noise (mean=32)
    # x = np.arange(0, 65)
    # ax.plot(x, ...) # Optional: overlay binomial distribution
    
    ax.set_xlabel("Hamming Distance (bits)")
    ax.set_ylabel("Frequency (Density)")
    ax.set_title("Distribution of pHash Distances (vs Original)")
    ax.set_xlim(-1, 65)
    ax.legend()
    
    # Annotations
    mu_noise = np.mean(dist_noise)
    mu_dummy = np.mean(dist_dummy)
    ax.axvline(mu_noise, color="#E45756", linestyle="--", alpha=0.5)
    ax.axvline(mu_dummy, color="#4C78A8", linestyle="--", alpha=0.5)
    
    ax.text(mu_noise + 1, ax.get_ylim()[1]*0.9, f"Mean: {mu_noise:.1f}", color="#E45756", fontsize=9)
    # ax.text(mu_dummy + 1, ax.get_ylim()[1]*0.9, f"Mean: {mu_dummy:.1f}", color="#4C78A8", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved histogram to {out_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", type=Path, default=Path("output/phash_masked_sis_eval/phash_masked_sis_eval.json"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/phash_masked_sis/dist_phash_histogram.png"))
    args = parser.parse_args()
    
    if not args.eval_json.exists():
        print(f"Error: {args.eval_json} not found.")
        return

    data = load_eval(args.eval_json)
    images = data["images"]
    
    plot_histogram(images, args.output)

if __name__ == "__main__":
    main()
