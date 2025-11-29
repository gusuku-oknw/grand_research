"""Prepare COCO val2017 derivatives for SIS+pHash experiments with configurable difficulty."""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
)

try:
    import pillow_avif  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pillow_avif = None  # type: ignore

RESAMPLE_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


@dataclass(frozen=True)
class TransformSpec:
    name: str
    params: Dict[str, Any]


def _spec(name: str, **params: Any) -> TransformSpec:
    return TransformSpec(name=name, params=params)


TRANSFORM_LIBRARY: Dict[str, TransformSpec] = {
    # Baseline quality tweaks
    "jpeg95": _spec("jpeg95", kind="jpeg", quality=95),
    "jpeg85": _spec("jpeg85", kind="jpeg", quality=85),
    "jpeg75": _spec("jpeg75", kind="jpeg", quality=75),
    "jpeg60": _spec("jpeg60", kind="jpeg", quality=60),
    "jpeg_q50_subs": _spec("jpeg_q50_subs", kind="jpeg", quality=50, subsampling="4:2:2"),
    "jpeg_q35_420": _spec("jpeg_q35_420", kind="jpeg", quality=35, subsampling="4:2:0"),
    "jpeg_q20_420": _spec("jpeg_q20_420", kind="jpeg", quality=20, subsampling="4:2:0"),
    # Codec round-trips
    "webp_q70": _spec("webp_q70", kind="codec_roundtrip", format="WEBP", quality=70),
    "webp_q30": _spec("webp_q30", kind="codec_roundtrip", format="WEBP", quality=30),
    "avif_cq40": _spec("avif_cq40", kind="codec_roundtrip", format="AVIF", quality=40),
    "avif_cq30": _spec("avif_cq30", kind="codec_roundtrip", format="AVIF", quality=30),
    # Geometric transforms
    "rotate_plus15_reflect": _spec("rotate_plus15_reflect", kind="rotate_pad", angle=15, pad_mode="reflect"),
    "rotate_minus15_reflect": _spec("rotate_minus15_reflect", kind="rotate_pad", angle=-15, pad_mode="reflect"),
    "rotate_plus30_black": _spec("rotate_plus30_black", kind="rotate_pad", angle=30, pad_mode="black"),
    "rotate_minus30_black": _spec("rotate_minus30_black", kind="rotate_pad", angle=-30, pad_mode="black"),
    "crop_balanced_20": _spec("crop_balanced_20", kind="balanced_crop", min_percent=20, max_percent=25),
    "crop_balanced_30": _spec("crop_balanced_30", kind="balanced_crop", min_percent=30, max_percent=35),
    "crop_balanced_40": _spec("crop_balanced_40", kind="balanced_crop", min_percent=35, max_percent=40),
    "perspective_trapezoid": _spec("perspective_trapezoid", kind="perspective", max_ratio=0.22),
    "resample_nearest_lanczos": _spec(
        "resample_nearest_lanczos",
        kind="resample_shuffle",
        down_factor=0.8,
        down_method="nearest",
        up_method="lanczos",
    ),
    "resample_bilinear_nearest": _spec(
        "resample_bilinear_nearest",
        kind="resample_shuffle",
        down_factor=0.7,
        down_method="bilinear",
        up_method="nearest",
    ),
    "resample_bicubic_lanczos": _spec(
        "resample_bicubic_lanczos",
        kind="resample_shuffle",
        down_factor=0.65,
        down_method="bicubic",
        up_method="lanczos",
    ),
    "downscale_half_jpeg35": _spec(
        "downscale_half_jpeg35",
        kind="composite",
        steps=[
            {"kind": "downscale_upscale", "factor": 0.5, "down_method": "bilinear", "up_method": "nearest"},
            {"kind": "jpeg", "quality": 35, "subsampling": "4:2:0"},
        ],
    ),
    # Photometric transforms
    "gamma_0_7": _spec("gamma_0_7", kind="gamma", gamma=0.7),
    "gamma_1_3": _spec("gamma_1_3", kind="gamma", gamma=1.3),
    "brightness_minus25": _spec("brightness_minus25", kind="brightness", factor=0.75),
    "brightness_plus25": _spec("brightness_plus25", kind="brightness", factor=1.25),
    "contrast_minus30": _spec("contrast_minus30", kind="contrast", factor=0.7),
    "contrast_plus30": _spec("contrast_plus30", kind="contrast", factor=1.3),
    "hue_shift_plus20": _spec("hue_shift_plus20", kind="hue_shift", delta_degrees=20),
    "hue_shift_minus20": _spec("hue_shift_minus20", kind="hue_shift", delta_degrees=-20),
    "saturation_plus40": _spec("saturation_plus40", kind="saturation", factor=1.4),
    "saturation_minus40": _spec("saturation_minus40", kind="saturation", factor=0.6),
    "grayscale": _spec("grayscale", kind="grayscale"),
    # Noise families
    "gaussian_sigma5": _spec("gaussian_sigma5", kind="gaussian_noise", sigma=5.0),
    "gaussian_sigma10": _spec("gaussian_sigma10", kind="gaussian_noise", sigma=10.0),
    "gaussian_sigma15": _spec("gaussian_sigma15", kind="gaussian_noise", sigma=15.0),
    "gaussian_sigma25": _spec("gaussian_sigma25", kind="gaussian_noise", sigma=25.0),
    "salt_pepper_5": _spec("salt_pepper_5", kind="salt_pepper_noise", amount=0.05),
    "salt_pepper_10": _spec("salt_pepper_10", kind="salt_pepper_noise", amount=0.1),
    "speckle_noise": _spec("speckle_noise", kind="speckle_noise", sigma=0.2),
    "poisson_noise": _spec("poisson_noise", kind="poisson_noise"),
    # Blur and optics
    "motion_blur": _spec("motion_blur", kind="motion_blur", kernel_min=7, kernel_max=15),
    "defocus_blur": _spec("defocus_blur", kind="defocus_blur", radius_min=3.0, radius_max=6.0),
    # Overlays / occlusions
    "watermark_logo": _spec("watermark_logo", kind="watermark", mode="logo"),
    "watermark_timestamp": _spec("watermark_timestamp", kind="watermark", mode="timestamp"),
    "occlusion_rectangle": _spec("occlusion_rectangle", kind="occlusion", mode="rectangle", ratio_min=0.1, ratio_max=0.2),
    "occlusion_stamp": _spec("occlusion_stamp", kind="occlusion", mode="stamp", ratio_min=0.12, ratio_max=0.18),
}

TRANSFORM_PROFILES: Dict[str, Sequence[str]] = {
    "low": (
        "jpeg95",
        "jpeg85",
        "jpeg75",
        "gaussian_sigma5",
        "rotate_plus15_reflect",
        "rotate_minus15_reflect",
        "crop_balanced_20",
        "brightness_plus25",
        "contrast_minus30",
        "gamma_0_7",
        "resample_nearest_lanczos",
    ),
    "medium": (
        "jpeg75",
        "jpeg60",
        "jpeg_q50_subs",
        "webp_q70",
        "rotate_plus30_black",
        "rotate_minus30_black",
        "crop_balanced_30",
        "perspective_trapezoid",
        "resample_bilinear_nearest",
        "gamma_0_7",
        "gamma_1_3",
        "brightness_minus25",
        "contrast_plus30",
        "gaussian_sigma10",
        "gaussian_sigma15",
        "salt_pepper_5",
        "motion_blur",
        "watermark_logo",
        "occlusion_rectangle",
    ),
    "high": (
        "jpeg_q35_420",
        "jpeg_q20_420",
        "webp_q30",
        "avif_cq40",
        "avif_cq30",
        "rotate_plus30_black",
        "rotate_minus30_black",
        "crop_balanced_40",
        "perspective_trapezoid",
        "resample_bicubic_lanczos",
        "downscale_half_jpeg35",
        "gamma_0_7",
        "gamma_1_3",
        "brightness_minus25",
        "brightness_plus25",
        "contrast_minus30",
        "contrast_plus30",
        "hue_shift_plus20",
        "hue_shift_minus20",
        "saturation_plus40",
        "saturation_minus40",
        "grayscale",
        "gaussian_sigma25",
        "salt_pepper_10",
        "speckle_noise",
        "poisson_noise",
        "motion_blur",
        "defocus_blur",
        "watermark_logo",
        "watermark_timestamp",
        "occlusion_rectangle",
        "occlusion_stamp",
    ),
}


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    w, h = img.size
    left = max((w - width) // 2, 0)
    top = max((h - height) // 2, 0)
    right = left + width
    bottom = top + height
    return img.crop((left, top, right, bottom))


def pad_image(img: Image.Image, pad: int, mode: str) -> Image.Image:
    if pad <= 0:
        return img
    if mode == "black":
        return ImageOps.expand(img, border=pad, fill=(0, 0, 0))
    if mode == "reflect":
        arr = np.asarray(img)
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        return Image.fromarray(padded)
    raise ValueError(f"Unsupported pad_mode: {mode}")


def rotate_with_padding(img: Image.Image, angle: float, pad_mode: str) -> Image.Image:
    w, h = img.size
    rad = math.radians(abs(angle))
    new_w = abs(w * math.cos(rad)) + abs(h * math.sin(rad))
    new_h = abs(w * math.sin(rad)) + abs(h * math.cos(rad))
    pad = int(math.ceil(max(new_w - w, new_h - h) / 2))
    padded = pad_image(img, pad, pad_mode)
    rotated = padded.rotate(angle, resample=Image.BICUBIC, expand=False)
    return center_crop(rotated, w, h)


def balanced_random_crop(img: Image.Image, min_percent: float, max_percent: float, rng: np.random.Generator) -> Image.Image:
    w, h = img.size
    percent = float(rng.uniform(min_percent, max_percent)) / 100.0
    percent = min(percent, 0.9)
    new_w = max(1, int(round(w * (1.0 - percent))))
    new_h = max(1, int(round(h * (1.0 - percent))))
    modes = np.array(["center", "edge", "corner"])
    mode = str(rng.choice(modes))
    if mode == "center":
        left = (w - new_w) // 2
        top = (h - new_h) // 2
    elif mode == "edge":
        edges = np.array(["top", "bottom", "left", "right"])
        edge = str(rng.choice(edges))
        if edge == "top":
            left = int(rng.integers(0, max(1, w - new_w + 1)))
            top = 0
        elif edge == "bottom":
            left = int(rng.integers(0, max(1, w - new_w + 1)))
            top = h - new_h
        elif edge == "left":
            left = 0
            top = int(rng.integers(0, max(1, h - new_h + 1)))
        else:
            left = w - new_w
            top = int(rng.integers(0, max(1, h - new_h + 1)))
    else:
        corners = ((0, 0), (w - new_w, 0), (0, h - new_h), (w - new_w, h - new_h))
        left, top = corners[int(rng.integers(0, len(corners)))]
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), resample=Image.BICUBIC)


def find_perspective_coeffs(src_points: Sequence[tuple[float, float]], dst_points: Sequence[tuple[float, float]]) -> List[float]:
    matrix: list[list[float]] = []
    target: list[float] = []
    for (x_src, y_src), (x_dst, y_dst) in zip(src_points, dst_points):
        matrix.append([x_src, y_src, 1, 0, 0, 0, -x_dst * x_src, -x_dst * y_src])
        matrix.append([0, 0, 0, x_src, y_src, 1, -y_dst * x_src, -y_dst * y_src])
        target.extend([x_dst, y_dst])
    a = np.asarray(matrix, dtype=np.float32)
    b = np.asarray(target, dtype=np.float32)
    coeffs = np.linalg.solve(a, b)
    return coeffs.tolist()


def random_perspective(img: Image.Image, max_ratio: float, rng: np.random.Generator) -> Image.Image:
    w, h = img.size
    src = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    delta = max_ratio * min(w, h)
    dst = []
    for x, y in src:
        shift_x = float(rng.uniform(-delta, delta))
        shift_y = float(rng.uniform(-delta, delta))
        dst.append((x + shift_x, y + shift_y))
    coeffs = find_perspective_coeffs(src, dst)
    transformed = img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)
    return transformed


def resample_shuffle(img: Image.Image, down_factor: float, down_method: str, up_method: str) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(round(w * down_factor)))
    new_h = max(1, int(round(h * down_factor)))
    down = img.resize((new_w, new_h), resample=RESAMPLE_MAP[down_method])
    return down.resize((w, h), resample=RESAMPLE_MAP[up_method])


def jpeg_compress(img: Image.Image, quality: int, subsampling: str | None = None) -> Image.Image:
    from io import BytesIO

    buffer = BytesIO()
    save_kwargs: Dict[str, Any] = {"format": "JPEG", "quality": quality}
    if subsampling is not None:
        save_kwargs["subsampling"] = subsampling
    img.save(buffer, **save_kwargs)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def codec_roundtrip(img: Image.Image, fmt: str, quality: int | None = None) -> Image.Image:
    from io import BytesIO

    buffer = BytesIO()
    save_kwargs: Dict[str, Any] = {"format": fmt}
    if quality is not None:
        save_kwargs["quality"] = quality
    img.save(buffer, **save_kwargs)
    buffer.seek(0)
    try:
        reopened = Image.open(buffer)
    except (ValueError, OSError) as exc:  # pragma: no cover - surface helpful hint
        raise RuntimeError(
            f"Failed to encode using format '{fmt}'. Install the required Pillow plugin (e.g. pillow-avif-plugin)."
        ) from exc
    return reopened.convert("RGB")


def gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def salt_pepper_noise(img: Image.Image, amount: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img)
    total = arr.shape[0] * arr.shape[1]
    num = int(total * amount)
    num_salt = num // 2
    num_pepper = num - num_salt
    arr = arr.copy()
    coords_salt = (
        rng.integers(0, arr.shape[0], num_salt),
        rng.integers(0, arr.shape[1], num_salt),
    )
    coords_pepper = (
        rng.integers(0, arr.shape[0], num_pepper),
        rng.integers(0, arr.shape[1], num_pepper),
    )
    arr[coords_salt] = 255
    arr[coords_pepper] = 0
    return Image.fromarray(arr)


def speckle_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + arr * noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def poisson_noise(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    noisy = rng.poisson(arr.clip(0, 255)).astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def shift_hue(img: Image.Image, delta_degrees: float) -> Image.Image:
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.asarray(h, dtype=np.uint16)
    delta = int((delta_degrees / 360.0) * 255) % 255
    h_arr = (h_arr + delta) % 255
    h_shifted = Image.fromarray(h_arr.astype(np.uint8), mode="L")
    return Image.merge("HSV", (h_shifted, s, v)).convert("RGB")


def adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def downscale_upscale(img: Image.Image, factor: float, down_method: str, up_method: str) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    down = img.resize((new_w, new_h), resample=RESAMPLE_MAP[down_method])
    return down.resize((w, h), resample=RESAMPLE_MAP[up_method])


def motion_blur(img: Image.Image, kernel_min: int, kernel_max: int, rng: np.random.Generator) -> Image.Image:
    radius = float(rng.uniform(kernel_min / 2.0, kernel_max / 2.0))
    angle = float(rng.uniform(0, 360))
    padded = pad_image(img, int(0.1 * max(img.size)), "reflect")
    rotated = padded.rotate(angle, resample=Image.BICUBIC, expand=True)
    blurred = rotated.filter(ImageFilter.BoxBlur(radius))
    restored = blurred.rotate(-angle, resample=Image.BICUBIC, expand=True)
    return center_crop(restored, *img.size)


def defocus_blur(img: Image.Image, radius_min: float, radius_max: float, rng: np.random.Generator) -> Image.Image:
    radius = float(rng.uniform(radius_min, radius_max))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_watermark(img: Image.Image, mode: str, rng: np.random.Generator) -> Image.Image:
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    alpha = int(rng.uniform(80, 140))
    margin = int(0.04 * min(w, h))
    if mode == "logo":
        text = str(rng.choice(["SHOP", "DEMO", "SECURE", "SALE"]))
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        box_w = max(int(0.2 * w), text_w + 20)
        box_h = max(int(0.1 * h), text_h + 12)
        x0 = w - box_w - margin
        y0 = h - box_h - margin
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(255, 255, 255, alpha))
        text_x = x0 + (box_w - text_w) // 2
        text_y = y0 + (box_h - text_h) // 2
        draw.text((text_x, text_y), text, fill=(0, 0, 0, 220), font=font)
    else:
        text = "2025-10-24 18:00"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        x0 = margin
        y0 = h - text_h - margin
        draw.rectangle([x0, y0 - 4, x0 + text_w + 8, y0 + text_h + 4], fill=(0, 0, 0, alpha))
        draw.text((x0 + 4, y0), text, fill=(255, 255, 255, 220), font=font)
    composite = Image.alpha_composite(img.convert("RGBA"), overlay)
    return composite.convert("RGB")


def add_occlusion(img: Image.Image, mode: str, ratio_min: float, ratio_max: float, rng: np.random.Generator) -> Image.Image:
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ratio = float(rng.uniform(ratio_min, ratio_max))
    occ_w = max(1, int(w * ratio))
    occ_h = max(1, int(h * ratio))
    x0 = int(rng.integers(0, max(1, w - occ_w)))
    y0 = int(rng.integers(0, max(1, h - occ_h)))
    if mode == "rectangle":
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        alpha = int(rng.uniform(140, 200))
        draw.rectangle([x0, y0, x0 + occ_w, y0 + occ_h], fill=(*color, alpha))
    else:
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        alpha = int(rng.uniform(160, 220))
        draw.ellipse([x0, y0, x0 + occ_w, y0 + occ_h], fill=(*color, alpha))
    composite = Image.alpha_composite(img.convert("RGBA"), overlay)
    return composite.convert("RGB")


def apply_transform(img: Image.Image, spec: TransformSpec, rng: np.random.Generator) -> Image.Image:
    params = spec.params
    kind = params.get("kind")
    if kind == "jpeg":
        return jpeg_compress(img, params["quality"], params.get("subsampling"))
    if kind == "codec_roundtrip":
        return codec_roundtrip(img, params["format"], params.get("quality"))
    if kind == "rotate_pad":
        return rotate_with_padding(img, params["angle"], params["pad_mode"])
    if kind == "balanced_crop":
        return balanced_random_crop(img, params["min_percent"], params["max_percent"], rng)
    if kind == "perspective":
        return random_perspective(img, params["max_ratio"], rng)
    if kind == "resample_shuffle":
        return resample_shuffle(img, params["down_factor"], params["down_method"], params["up_method"])
    if kind == "downscale_upscale":
        return downscale_upscale(img, params["factor"], params["down_method"], params["up_method"])
    if kind == "composite":
        result = img
        for step in params["steps"]:
            inner_spec = TransformSpec(name=spec.name, params=step)
            result = apply_transform(result, inner_spec, rng)
        return result
    if kind == "gamma":
        return apply_gamma(img, params["gamma"])
    if kind == "brightness":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(params["factor"])
    if kind == "contrast":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(params["factor"])
    if kind == "saturation":
        return adjust_saturation(img, params["factor"])
    if kind == "hue_shift":
        return shift_hue(img, params["delta_degrees"])
    if kind == "grayscale":
        return ImageOps.grayscale(img).convert("RGB")
    if kind == "gaussian_noise":
        return gaussian_noise(img, params["sigma"], rng)
    if kind == "salt_pepper_noise":
        return salt_pepper_noise(img, params["amount"], rng)
    if kind == "speckle_noise":
        return speckle_noise(img, params["sigma"], rng)
    if kind == "poisson_noise":
        return poisson_noise(img, rng)
    if kind == "motion_blur":
        return motion_blur(img, params["kernel_min"], params["kernel_max"], rng)
    if kind == "defocus_blur":
        return defocus_blur(img, params["radius_min"], params["radius_max"], rng)
    if kind == "watermark":
        return add_watermark(img, params["mode"], rng)
    if kind == "occlusion":
        return add_occlusion(img, params["mode"], params["ratio_min"], params["ratio_max"], rng)
    raise ValueError(f"Unknown transform kind: {kind}")


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.jpg")):
        yield path


def resolve_transforms(names: Sequence[str]) -> List[TransformSpec]:
    resolved: List[TransformSpec] = []
    for name in names:
        if name not in TRANSFORM_LIBRARY:
            raise KeyError(f"Transform '{name}' is not registered.")
        spec = TRANSFORM_LIBRARY[name]
        resolved.append(TransformSpec(name=spec.name, params=deepcopy(spec.params)))
    return resolved


def prepare_derivatives(
    source_dir: Path,
    output_dir: Path,
    transforms: List[TransformSpec],
    max_images: int | None,
    seed: int,
    *,
    skip_existing: bool,
    existing_mapping: Dict[str, Dict[str, str]] | None = None,
    show_progress: bool,
) -> tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    mapping: Dict[str, Dict[str, str]] = dict(existing_mapping or {})
    processed_images = 0
    generated_variants = 0
    skipped_variants = 0
    image_paths = list(iter_image_paths(source_dir))
    if max_images is not None:
        image_paths = image_paths[: max_images]
    total_images = len(image_paths)

    if total_images == 0:
        stats = {
            "processed_images": 0,
            "generated_variants": 0,
            "skipped_variants": 0,
        }
        return mapping, stats

    for idx, img_path in enumerate(image_paths):
        if show_progress:
            print(f"[{idx + 1}/{total_images}] {img_path.stem}", flush=True)
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
                save_image(transformed, out_path)
                generated_variants += 1
            image_mapping[spec.name] = str(out_path)

    stats = {
        "processed_images": processed_images,
        "generated_variants": generated_variants,
        "skipped_variants": skipped_variants,
    }
    return mapping, stats


def parse_transform_overrides(value: str) -> Sequence[str]:
    value = value.strip()
    if not value or value == "default":
        return ()
    if "," in value:
        return tuple(item.strip() for item in value.split(",") if item.strip())
    possible_path = Path(value)
    if possible_path.exists():
        with possible_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("Transform JSON must be a list of transform names.")
        return tuple(data)
    return (value,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate transformed variants of COCO val2017 images.")
    parser.add_argument("--coco_dir", type=Path, default=Path("data/coco/val2017"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/coco/val2017_derivatives"))
    parser.add_argument("--profile", choices=sorted(TRANSFORM_PROFILES.keys()), default="medium")
    parser.add_argument(
        "--variant_scope",
        choices=["all", "original_only"],
        default="all",
        help="Choose whether to generate all configured transforms or only the original images.",
    )
    parser.add_argument("--transforms", type=str, default="default", help="Override transforms (comma list or JSON file).")
    parser.add_argument("--include_transforms", nargs="*", default=None, help="Additional transform names to append.")
    parser.add_argument("--exclude_transforms", nargs="*", default=None, help="Transform names to remove from the final set.")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mapping_json", type=Path, default=Path("data/coco/derivative_mapping.json"))
    parser.add_argument("--force", action="store_true", help="Regenerate derivatives even if outputs exist.")
    parser.add_argument("--list_transforms", action="store_true", help="List available transform names and exit.")
    parser.add_argument("--no_progress", action="store_true", help="Suppress per-image progress output.")
    args = parser.parse_args()

    if args.list_transforms:
        for name in sorted(TRANSFORM_LIBRARY):
            print(name)
        return

    base_names = list(TRANSFORM_PROFILES[args.profile])
    overrides = parse_transform_overrides(args.transforms)
    if overrides:
        base_names = list(overrides)

    if args.include_transforms:
        if args.variant_scope == "original_only":
            print("Ignoring --include_transforms because --variant_scope=original_only.", file=sys.stderr)
        for name in args.include_transforms:
            if name not in base_names:
                base_names.append(name)

    if args.exclude_transforms:
        if args.variant_scope == "original_only":
            print("Ignoring --exclude_transforms because --variant_scope=original_only.", file=sys.stderr)
        exclude = set(args.exclude_transforms)
        base_names = [name for name in base_names if name not in exclude]

    if args.variant_scope == "original_only":
        base_names = []

    transforms = resolve_transforms(base_names)

    existing_mapping: Dict[str, Dict[str, str]] | None = None
    if not args.force and args.mapping_json.exists():
        with args.mapping_json.open("r", encoding="utf-8") as f:
            existing_mapping = json.load(f)

    show_progress = not args.no_progress

    mapping, stats = prepare_derivatives(
        source_dir=args.coco_dir,
        output_dir=args.output_dir,
        transforms=transforms,
        max_images=args.max_images,
        seed=args.seed,
        skip_existing=not args.force,
        existing_mapping=existing_mapping,
        show_progress=show_progress,
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
