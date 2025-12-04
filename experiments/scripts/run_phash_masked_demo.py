"""
Generate pHash-preserving masked image and two-level SIS shares, then dump stats.

Outputs under: output/figures/phash_masked_sis/<name>/
  - original_<name>.jpg
  - dummy_<name>.jpg
  - recovered_dummy_<name>.jpg
  - recovered_secret_<name>.jpg
  - grid.png (4枚並べ)
  - stats.png (MSE と pHash 距離の横棒グラフ)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from phash_masked_sis import (
    PHashConfig,
    compute_phash,
    MultiSecretImageSIS,
)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def make_noise_like(base_size: tuple[int, int], cfg: PHashConfig, seed: int | None = None) -> Image.Image:
    rng = np.random.default_rng(seed)
    size_small = cfg.hash_size * cfg.highfreq_factor
    arr = rng.integers(0, 256, size=(size_small, size_small), dtype=np.uint8)
    noise_small = Image.fromarray(arr, mode="L")
    return noise_small.resize(base_size, Image.BICUBIC)


def draw_bar_chart(labels, values, out_path: Path) -> None:
    max_val = max(values) if values else 1
    width, height = 900, 60 * len(labels) + 80
    margin_left, margin_right, margin_v = 220, 60, 40
    bar_height = 30
    spacing = 20
    scale = (width - margin_left - margin_right) / max_val if max_val else 1

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    y = margin_v
    for label, val in zip(labels, values):
        bar_len = val * scale
        draw.rectangle(
            [margin_left, y, margin_left + bar_len, y + bar_height],
            fill="#4a90e2",
            outline="black",
        )
        text = f"{val:.1f}"
        draw.text((margin_left + bar_len + 8, y + (bar_height - 10) / 2), text, fill="black", font=font)
        draw.text((10, y + (bar_height - 10) / 2), label, fill="black", font=font)
        y += bar_height + spacing

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phash_masked_sis demo on an image.")
    parser.add_argument("image", type=Path, help="Path to source image (e.g., data/00.jpg)")
    parser.add_argument("--name", type=str, default="demo", help="Name prefix for output files")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducibility")
    parser.add_argument("--n", type=int, default=5, help="Total number of shares")
    parser.add_argument("--k1", type=int, default=2, help="Threshold for dummy reconstruction")
    parser.add_argument("--k2", type=int, default=4, help="Threshold for full reconstruction")
    parser.add_argument("--save-shares", action="store_true", help="Persist shares as JSON in output dir")
    parser.add_argument("--load-shares", type=Path, default=None, help="Load shares from directory instead of splitting anew")
    args = parser.parse_args()

    img = Image.open(args.image)
    cfg = PHashConfig(hash_size=8, highfreq_factor=4)

    sis = MultiSecretImageSIS(n=args.n, k1=args.k1, k2=args.k2, cfg=cfg)
    out_dir = Path(f"output/figures/phash_masked_sis/{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.load_shares:
        shares = sis.load_shares(args.load_shares)
    else:
        shares = sis.split_from_image(args.image, seed=args.seed)
        if args.save_shares:
            sis.save_shares(shares, out_dir / "shares")

    orig_path = out_dir / f"original_{args.name}.jpg"
    img.save(orig_path)

    # 3段階の復元
    # ノイズは「元画像と pHash が十分離れるまで」生成し直す
    base_size = img.size
    noise_img = None
    if args.k1 > 1:
        for attempt in range(10):
            candidate = make_noise_like(base_size, cfg, seed=args.seed + attempt if args.seed is not None else None)
            dist = int(np.count_nonzero(compute_phash(candidate, cfg) ^ compute_phash(img, cfg)))
            if dist >= 24:  # ランダム画像なら平均32程度を期待
                noise_img = candidate
                break
        if noise_img is None:
            noise_img = candidate  # fallback
    else:
        noise_img = sis._noise_image()

    _, dummy_img = sis.reconstruct_with_levels(shares[: args.k1])
    _, full_img = sis.reconstruct_with_levels(shares[: args.k2])

    noise_path = out_dir / f"noise_{args.name}.jpg"
    dummy_path = out_dir / f"dummy_{args.name}.jpg"
    rec_dummy = out_dir / f"recovered_dummy_{args.name}.jpg"
    rec_secret = out_dir / f"recovered_secret_{args.name}.jpg"
    noise_img.save(noise_path)
    dummy_img.save(dummy_path)
    dummy_img.save(rec_dummy)
    full_img.save(rec_secret)

    # Stats
    a = load_gray(orig_path)
    b = load_gray(dummy_path)
    c = load_gray(rec_dummy)
    d = load_gray(rec_secret)
    h_rec = compute_phash(Image.open(rec_secret), cfg)
    h_orig = compute_phash(Image.open(orig_path), cfg)
    h_dummy = compute_phash(Image.open(dummy_path), cfg)
    h_noise = compute_phash(Image.open(noise_path), cfg)
    stats = {
        "MSE(orig,noise)": mse(a, load_gray(noise_path)),
        "MSE(orig,dummy)": mse(a, b),
        "MSE(dummy,recdummy)": mse(b, c),
        "MSE(orig,recsecret)": mse(a, d),
    }
    phash_stats = {
        "pHash(orig,noise)": int(np.count_nonzero(h_orig ^ h_noise)),
        "pHash(orig,dummy)": int(np.count_nonzero(h_orig ^ h_dummy)),
        "pHash(orig,recovered)": int(np.count_nonzero(h_orig ^ h_rec)),
    }

    # Grid
    imgs = [Image.open(p).convert("RGB") for p in (orig_path, noise_path, dummy_path, rec_secret)]
    max_h = max(im.height for im in imgs)
    resized = []
    for im in imgs:
        if im.height != max_h:
            new_w = int(im.width * max_h / im.height)
            im = im.resize((new_w, max_h), Image.BICUBIC)
        resized.append(im)
    pad = 30
    width = sum(im.width for im in resized)
    height = max_h + pad
    grid = Image.new("RGB", (width, height), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    labels = ["original", "noise(<k1)", "dummy(k1≤)", "full(k2≤)"]
    x = 0
    for im, label in zip(resized, labels):
        grid.paste(im, (x, pad))
        bbox = draw.textbbox((0, 0), label, font=font)
        lw, lh = bbox[2], bbox[3]
        draw.text((x + (im.width - lw) / 2, (pad - lh) / 2), label, fill="black", font=font)
        x += im.width
    grid.save(out_dir / "grid.png")

    # Bars
    draw_bar_chart(list(stats.keys()), list(stats.values()), out_dir / "stats_mse.png")
    draw_bar_chart(list(phash_stats.keys()), list(phash_stats.values()), out_dir / "stats_phash.png")

    print("pHash Hamming (orig,dummy):", phash_stats["pHash(orig,dummy)"])
    print("pHash Hamming (orig,recovered):", phash_stats["pHash(orig,recovered)"])
    print("Saved outputs to", out_dir)


if __name__ == "__main__":
    main()
