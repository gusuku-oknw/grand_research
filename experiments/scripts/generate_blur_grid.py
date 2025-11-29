"""
Generate a grid of blurred variants for a single image to compare blur strength.

Example:
    python experiments/scripts/generate_blur_grid.py \
        --image_path data/img02.jpg \
        --output output/blur_strength_grid.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageFilter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_low_res_blur(img: Image.Image, scale: float, blur_radius: float) -> Image.Image:
    """
    Downsample heavily then blur to reduce information.
    """
    w, h = img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    back = small.resize((w, h), Image.Resampling.BILINEAR)
    return back.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def build_grid(
    img: Image.Image,
    scales: Iterable[float],
    radii: Iterable[float],
) -> plt.Figure:
    scales_list: List[float] = list(scales)
    radii_list: List[float] = list(radii)

    fig, axes = plt.subplots(
        len(radii_list),
        len(scales_list),
        figsize=(3 * len(scales_list), 2.5 * len(radii_list)),
    )
    for r_idx, radius in enumerate(radii_list):
        for c_idx, scale in enumerate(scales_list):
            ax = axes[r_idx][c_idx] if len(radii_list) > 1 else axes[c_idx]
            blurred = make_low_res_blur(img, scale=scale, blur_radius=radius)
            ax.imshow(blurred)
            ax.axis("off")
            ax.set_title(f"scale={scale:.3f}\nblur={radius}", fontsize=8)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to source image")
    parser.add_argument(
        "--output",
        default="output/blur_strength_grid.png",
        help="Path to save the comparison figure",
    )
    parser.add_argument(
        "--scales",
        default="0.001,0.003,0.005,0.008,0.01,0.02,0.05,0.1",
        help="Comma-separated downscale factors",
    )
    parser.add_argument(
        "--radii",
        default="2,4,6,10,14,18",
        help="Comma-separated Gaussian blur radii",
    )
    parser.add_argument(
        "--max_display_width",
        type=int,
        default=1200,
        help="Resize source image to this width for visualization speed",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    radii = [float(r.strip()) for r in args.radii.split(",") if r.strip()]

    img = Image.open(image_path).convert("RGB")
    if args.max_display_width and img.width > args.max_display_width:
        ratio = args.max_display_width / float(img.width)
        new_h = int(img.height * ratio)
        img = img.resize((args.max_display_width, new_h), Image.Resampling.LANCZOS)

    fig = build_grid(img, scales=scales, radii=radii)

    out_path = Path(args.output)
    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved blur grid to {out_path}")


if __name__ == "__main__":
    main()
