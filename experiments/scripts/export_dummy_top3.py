"""Export the top-3 panels (target bits / reinforced low-freq / 32x32 spatial) from the dummy generator."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Allow running without installation
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phash_masked_sis import PHashConfig, idct2  # noqa: E402
from phash_masked_sis.dummy_image import make_phash_preserving_dummy  # noqa: E402


def _to_img(arr, *, mode: str = "L", size: Tuple[int, int] = (128, 128)) -> Image.Image:
    """Convert numpy array to PIL.Image with nearest resizing."""
    return Image.fromarray(arr.astype("uint8"), mode=mode).resize(size, Image.NEAREST)


def export_top3(image_path: Path, output_path: Path, cfg: PHashConfig, seed: int) -> Path:
    """Generate and export target bits / reinforced low-freq / 32x32 spatial panels."""
    img = Image.open(image_path).convert("RGB")
    _, debug = make_phash_preserving_dummy(img, cfg=cfg, seed=seed, return_debug=True)

    panels: List[Tuple[str, Image.Image]] = []
    if debug:
        bits = debug.get("bits")
        if bits is not None:
            panels.append(("target bits", _to_img(bits * 255)))
        lf = debug.get("lf_final")
        if lf is not None:
            size = cfg.hash_size * cfg.highfreq_factor
            coeffs = np.zeros((size, size), dtype=np.float64)
            coeffs[: cfg.hash_size, : cfg.hash_size] = lf
            spatial = idct2(coeffs)
            spatial -= spatial.min()
            if spatial.max() > 0:
                spatial = spatial / spatial.max() * 255.0
            panels.append(("low-freq spatial", _to_img(spatial)))
        spatial = debug.get("spatial_small")
        if spatial is not None:
            panels.append(("32x32 spatial", _to_img(spatial)))

    if not panels:
        raise RuntimeError("Debug info is missing; cannot export panels.")

    pad = 10
    text_pad = 14
    width = sum(im.width for _, im in panels) + pad * (len(panels) + 1)
    height = max(im.height for _, im in panels) + pad * 2 + text_pad
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = pad
    y = pad
    for title, im in panels:
        canvas.paste(im.convert("RGB"), (x, y))
        draw.text((x, y + im.height + 2), title, fill=(0, 0, 0))
        x += im.width + pad

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export dummy k1 generation top-3 panels.")
    p.add_argument("--image", type=Path, required=True, help="Input image path.")
    p.add_argument("--output", type=Path, default=Path("output/phash_masked_sis_eval/pipeline_top3.png"))
    p.add_argument("--hash_size", type=int, default=8)
    p.add_argument("--highfreq_factor", type=int, default=4)
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PHashConfig(hash_size=args.hash_size, highfreq_factor=args.highfreq_factor)
    out = export_top3(args.image, args.output, cfg=cfg, seed=args.seed)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
