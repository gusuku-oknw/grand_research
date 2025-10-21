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
    *,
    skip_existing: bool,
    existing_mapping: Dict[str, Dict[str, str]] | None = None,
) -> tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    mapping: Dict[str, Dict[str, str]] = dict(existing_mapping or {})
    processed_images = 0
    generated_variants = 0
    skipped_variants = 0

    for idx, img_path in enumerate(iter_image_paths(source_dir)):
        if max_images is not None and idx >= max_images:
            break
        image_id = img_path.stem
        image_mapping = mapping.setdefault(image_id, {})
        base_img: Image.Image | None = None

        def ensure_base() -> Image.Image:
            nonlocal base_img
            if base_img is None:
                base_img = load_image(img_path)
            return base_img

        processed_images += 1

        out_base = (output_dir / image_id / "original").with_suffix(".png")
        if skip_existing and out_base.exists():
            skipped_variants += 1
        else:
            img_obj = ensure_base()
            save_image(img_obj, out_base)
            generated_variants += 1
        image_mapping["original"] = str(out_base)

        for spec in transforms:
            out_path = (output_dir / image_id / spec.name).with_suffix(".png")
            if skip_existing and out_path.exists():
                skipped_variants += 1
            else:
                img_obj = ensure_base()
                transformed = apply_transform(img_obj, spec, rng)
                quality = spec.params.get("quality")
                save_image(transformed, out_path, quality=None if quality is None else quality)
                generated_variants += 1
            image_mapping[spec.name] = str(out_path)

    stats = {
        "processed_images": processed_images,
        "generated_variants": generated_variants,
        "skipped_variants": skipped_variants,
    }
    return mapping, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate transformed variants of COCO val2017 images.")
    parser.add_argument("--coco_dir", type=Path, default=Path("data/coco/val2017"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/coco/val2017_derivatives"))
    parser.add_argument("--transforms", type=str, default="default")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mapping_json", type=Path, default=Path("data/coco/derivative_mapping.json"))
    parser.add_argument("--force", action="store_true", help="Regenerate derivatives even if outputs exist.")
    args = parser.parse_args()

    if args.transforms == "default":
        transforms = DEFAULT_TRANSFORMS
    else:
        raise NotImplementedError("Custom transforms not yet implemented.")

    existing_mapping: Dict[str, Dict[str, str]] | None = None
    if not args.force and args.mapping_json.exists():
        with args.mapping_json.open("r", encoding="utf-8") as f:
            existing_mapping = json.load(f)

    mapping, stats = prepare_derivatives(
        source_dir=args.coco_dir,
        output_dir=args.output_dir,
        transforms=transforms,
        max_images=args.max_images,
        seed=args.seed,
        skip_existing=not args.force,
        existing_mapping=existing_mapping,
    )

    args.mapping_json.parent.mkdir(parents=True, exist_ok=True)
    with args.mapping_json.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(
        "Processed {processed} images "
        "(generated {generated} variants, reused {skipped} existing).".format(
            processed=stats["processed_images"],
            generated=stats["generated_variants"],
            skipped=stats["skipped_variants"],
        )
    )


if __name__ == "__main__":
    main()
