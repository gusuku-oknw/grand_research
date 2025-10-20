"""Prepare COCO val2017 derivatives for SIS+pHash experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


@dataclass
class TransformSpec:
    name: str
    params: Dict[str, Any]


DEFAULT_TRANSFORMS: List[TransformSpec] = [
    TransformSpec("jpeg95", {"kind": "jpeg", "quality": 95}),
    TransformSpec("jpeg85", {"kind": "jpeg", "quality": 85}),
    TransformSpec("jpeg75", {"kind": "jpeg", "quality": 75}),
    TransformSpec("jpeg60", {"kind": "jpeg", "quality": 60}),
    TransformSpec("gaussian_sigma2", {"kind": "gaussian_noise", "sigma": 2.0}),
    TransformSpec("gaussian_sigma5", {"kind": "gaussian_noise", "sigma": 5.0}),
    TransformSpec("gaussian_sigma10", {"kind": "gaussian_noise", "sigma": 10.0}),
    TransformSpec("rotate_plus5", {"kind": "rotate", "angle": 5}),
    TransformSpec("rotate_minus5", {"kind": "rotate", "angle": -5}),
    TransformSpec("scale_0.9", {"kind": "scale", "factor": 0.9}),
    TransformSpec("scale_1.1", {"kind": "scale", "factor": 1.1}),
    TransformSpec("crop_5pct", {"kind": "crop", "percent": 5}),
    TransformSpec("crop_10pct", {"kind": "crop", "percent": 10}),
    TransformSpec("color_shift", {"kind": "color", "factor": 1.1}),
]


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: Path, quality: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if quality is None:
        image.save(path)
    else:
        image.save(path, quality=quality)


def apply_transform(img: Image.Image, spec: TransformSpec, rng: np.random.Generator) -> Image.Image:
    kind = spec.params.get("kind")
    if kind == "jpeg":
        quality = spec.params["quality"]
        return jpeg_compress(img, quality)
    if kind == "gaussian_noise":
        sigma = spec.params["sigma"]
        return gaussian_noise(img, sigma, rng)
    if kind == "rotate":
        angle = spec.params["angle"]
        return img.rotate(angle, resample=Image.BICUBIC, expand=False)
    if kind == "scale":
        factor = spec.params["factor"]
        return scale_image(img, factor)
    if kind == "crop":
        percent = spec.params["percent"]
        return crop_border(img, percent, rng)
    if kind == "color":
        factor = spec.params["factor"]
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    raise ValueError(f"Unknown transform kind: {kind}")


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    from io import BytesIO

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def scale_image(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * factor), int(h * factor)
    resized = img.resize((nw, nh), resample=Image.BICUBIC)
    return resized.resize((w, h), resample=Image.BICUBIC)


def crop_border(img: Image.Image, percent: int, rng: np.random.Generator) -> Image.Image:
    if percent <= 0:
        return img
    w, h = img.size
    dw = int(w * percent / 100)
    dh = int(h * percent / 100)
    left = rng.integers(0, dw + 1)
    top = rng.integers(0, dh + 1)
    right = w - (dw - left)
    bottom = h - (dh - top)
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), resample=Image.BICUBIC)


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.jpg")):
        yield path


def prepare_derivatives(
    source_dir: Path,
    output_dir: Path,
    transforms: List[TransformSpec],
    max_images: int | None,
    seed: int,
) -> Dict[str, Dict[str, str]]:
    rng = np.random.default_rng(seed)
    mapping: Dict[str, Dict[str, str]] = {}
    for idx, img_path in enumerate(iter_image_paths(source_dir)):
        if max_images is not None and idx >= max_images:
            break
        image_id = img_path.stem
        base_img = load_image(img_path)
        out_base = output_dir / image_id / "original"
        save_image(base_img, out_base.with_suffix(".png"))
        mapping.setdefault(image_id, {})["original"] = str(out_base.with_suffix(".png"))
        for spec in transforms:
            out_path = output_dir / image_id / spec.name
            transformed = apply_transform(base_img, spec, rng)
            quality = spec.params.get("quality")
            save_image(transformed, out_path.with_suffix(".png"), quality=None if quality is None else quality)
            mapping[image_id][spec.name] = str(out_path.with_suffix(".png"))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate transformed variants of COCO val2017 images.")
    parser.add_argument("--coco_dir", type=Path, default=Path("data/coco/val2017"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/coco/val2017_derivatives"))
    parser.add_argument("--transforms", type=str, default="default")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mapping_json", type=Path, default=Path("data/coco/derivative_mapping.json"))
    args = parser.parse_args()

    if args.transforms == "default":
        transforms = DEFAULT_TRANSFORMS
    else:
        raise NotImplementedError("Custom transforms not yet implemented.")

    mapping = prepare_derivatives(
        source_dir=args.coco_dir,
        output_dir=args.output_dir,
        transforms=transforms,
        max_images=args.max_images,
        seed=args.seed,
    )

    args.mapping_json.parent.mkdir(parents=True, exist_ok=True)
    with args.mapping_json.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Generated derivatives for {len(mapping)} images.")


if __name__ == "__main__":
    main()
