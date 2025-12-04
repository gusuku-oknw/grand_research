"""
Compute simple stats for phash_masked_sis outputs and render PNGs (no matplotlib dependency).

Expects:
  output/figures/phash_masked_sis/data00/original_00.jpg
  output/figures/phash_masked_sis/data00/dummy_00.jpg
  output/figures/phash_masked_sis/data00/recovered_dummy_00.jpg
  output/figures/phash_masked_sis/data00/recovered_secret_00.jpg

Outputs:
  output/figures/phash_masked_sis/data00/grid.png   (4枚並べ)
  output/figures/phash_masked_sis/data00/stats.png  (MSEとpHash距離の簡易バー)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from phash_masked_sis import PHashConfig, compute_phash


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def draw_bar_chart(labels, values, out_path: Path) -> None:
    # 簡易横棒グラフ（PIL のみ）
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
    base = Path("output/figures/phash_masked_sis/data00")
    orig = base / "original_00.jpg"
    dummy = base / "dummy_00.jpg"
    rec_dummy = base / "recovered_dummy_00.jpg"
    rec_secret = base / "recovered_secret_00.jpg"
    for p in (orig, dummy, rec_dummy, rec_secret):
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    a = load_gray(orig)
    b = load_gray(dummy)
    c = load_gray(rec_dummy)
    d = load_gray(rec_secret)

    cfg = PHashConfig()
    h_orig = compute_phash(Image.open(orig), cfg)
    h_dummy = compute_phash(Image.open(dummy), cfg)
    h_rec = compute_phash(Image.open(rec_secret), cfg)
    h_dist_orig_dummy = int(np.count_nonzero(h_orig ^ h_dummy))
    h_dist_orig_rec = int(np.count_nonzero(h_orig ^ h_rec))

    stats = {
        "MSE(orig,dummy)": mse(a, b),
        "MSE(dummy,recdummy)": mse(b, c),
        "MSE(orig,recsecret)": mse(a, d),
    }

    # 4枚並べる
    imgs = [Image.open(p).convert("RGB") for p in (orig, dummy, rec_dummy, rec_secret)]
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
    labels = ["original_00", "dummy_00", "recovered_dummy", "recovered_secret"]
    x = 0
    for im, label in zip(resized, labels):
        grid.paste(im, (x, pad))
        bbox = draw.textbbox((0, 0), label, font=font)
        lw, lh = bbox[2], bbox[3]
        draw.text((x + (im.width - lw) / 2, (pad - lh) / 2), label, fill="black", font=font)
        x += im.width
    grid.save(base / "grid.png")

    # 簡易バー図
    bar_labels = list(stats.keys()) + ["pHash(orig,dummy)", "pHash(orig,rec)"]
    bar_vals = list(stats.values()) + [h_dist_orig_dummy, h_dist_orig_rec]
    draw_bar_chart(bar_labels, bar_vals, base / "stats.png")

    print("Saved grid to", base / "grid.png")
    print("Saved stats to", base / "stats.png")
    print("pHash Hamming (orig,dummy):", h_dist_orig_dummy)
    print("pHash Hamming (orig,rec):", h_dist_orig_rec)
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    main()
