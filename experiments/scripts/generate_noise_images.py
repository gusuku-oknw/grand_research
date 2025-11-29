"""Generate synthetic noise images for SIS visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make random noise images for SIS demos.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/noise"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--grayscale", action="store_true", help="Generate grayscale noise instead of RGB.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for idx in range(args.count):
        if args.grayscale:
            array = rng.integers(0, 256, size=args.size, dtype=np.uint8)
            mode = "L"
        else:
            array = rng.integers(0, 256, size=(*args.size, 3), dtype=np.uint8)
            mode = "RGB"
        output_file = output_dir / f"noise_{idx:03d}.png"
        Image.fromarray(array, mode=mode).save(output_file)
    print(f"Generated {args.count} noise images in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
