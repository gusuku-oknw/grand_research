from __future__ import annotations

import argparse
import json
import math
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Iterable

# Allow running the script without installing as a package (editable install recommended).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageFilter

def load_derivative_mapping(mapping_json: Path) -> Dict[str, Dict[str, str]]:
    with mapping_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {img_id: {variant: str(path) for variant, path in variants.items()} for img_id, variants in raw.items()}

from phash_masked_sis import (
    MultiSecretImageSIS,
    PHashConfig,
    compute_phash,
    dct2,
    idct2,
)
from phash_masked_sis.multisecret_image import shamir_combine_secret, shamir_split_secret
from phash_masked_sis.sis_twolevel import shamir_combine, shamir_split


def _phash_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.astype(np.uint8) ^ b.astype(np.uint8)))


def _psnr(a: Image.Image, b: Image.Image) -> float:
    a_arr = np.asarray(a.convert("RGB"), dtype=np.float64)
    b_arr = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = np.mean((a_arr - b_arr) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _attack_variants(img: Image.Image, rng: np.random.Generator) -> Dict[str, Image.Image]:
    variants: Dict[str, Image.Image] = {}

    arr = np.asarray(img.convert("L"), dtype=np.float64)
    noisy = arr + rng.normal(0.0, 8.0, size=arr.shape)
    variants["gaussian_noise"] = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="L").convert(img.mode)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=30, subsampling=2)
    variants["jpeg_q30"] = Image.open(BytesIO(buf.getvalue())).convert(img.mode)

    variants["rotate_5deg"] = img.rotate(5, resample=Image.BICUBIC, fillcolor=0)
    variants["blur"] = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    scaled = img.resize((max(1, img.width - 8), max(1, img.height - 8)), Image.BICUBIC)
    variants["shrink"] = scaled.resize(img.size, Image.BICUBIC)

    return variants


def _blur_image(img: Image.Image, radius: float = 3.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _noise_image_like(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = rng.integers(0, 256, size=(img.height, img.width, len(img.getbands())), dtype=np.uint8)
    mode = img.mode
    if mode == "L":
        arr = arr[:, :, 0]
    return Image.fromarray(arr, mode=mode)


def benchmark_dct(cfg: PHashConfig, repeats: int, seed: int | None) -> Dict[str, float]:
    size = cfg.hash_size * cfg.highfreq_factor
    rng = np.random.default_rng(seed)
    forward_times: List[float] = []
    inverse_times: List[float] = []
    max_errors: List[float] = []

    for _ in range(repeats):
        arr = rng.normal(0.0, 1.0, size=(size, size))

        t0 = time.perf_counter()
        coeffs = dct2(arr)
        t1 = time.perf_counter()
        recon = idct2(coeffs)
        t2 = time.perf_counter()

        forward_times.append((t1 - t0) * 1000.0)
        inverse_times.append((t2 - t1) * 1000.0)
        max_errors.append(float(np.abs(arr - recon).max()))

    return {
        "size": size,
        "forward_ms_mean": float(np.mean(forward_times)),
        "inverse_ms_mean": float(np.mean(inverse_times)),
        "max_reconstruction_error": float(np.max(max_errors)),
    }


def benchmark_shamir(secret_len: int, n: int, k: int, repeats: int, seed: int | None) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    split_times: List[float] = []
    recover_times: List[float] = []

    for _ in range(repeats):
        secret = rng.bytes(secret_len)

        t0 = time.perf_counter()
        shares = shamir_split(secret, n=n, k=k)
        t1 = time.perf_counter()
        recovered = shamir_combine(shares[:k])
        t2 = time.perf_counter()

        assert recovered == secret, "Shamir reconstruction mismatch"
        split_times.append((t1 - t0) * 1000.0)
        recover_times.append((t2 - t1) * 1000.0)

    return {
        "secret_len_bytes": secret_len,
        "n": n,
        "k": k,
        "split_ms_mean": float(np.mean(split_times)),
        "recover_ms_mean": float(np.mean(recover_times)),
    }


def evaluate_image(
    image_path: Path,
    sis: MultiSecretImageSIS,
    cfg: PHashConfig,
    rng: np.random.Generator,
    fast_mode: bool = False,
) -> Dict[str, object]:
    img = Image.open(image_path).convert("RGB")
    split_start = time.perf_counter()
    shares = sis.split_from_image(image_path, seed=int(rng.integers(0, 1 << 31)))
    split_ms = (time.perf_counter() - split_start) * 1000.0

    recover_dummy_start = time.perf_counter()
    dummy_img = sis.recover_dummy_image(shares[: sis.k1])
    recover_dummy_ms = (time.perf_counter() - recover_dummy_start) * 1000.0

    recover_dummy_ms = (time.perf_counter() - recover_dummy_start) * 1000.0

    if not fast_mode:
        recover_full_start = time.perf_counter()
        full_img = sis.recover_full_image(shares[: sis.k2])
        recover_full_ms = (time.perf_counter() - recover_full_start) * 1000.0
    else:
        recover_full_ms = 0.0
        full_img = img  # Use original as placeholder to avoid errors downstream if referenced

    h_orig = compute_phash(img, cfg)
    h_dummy = compute_phash(dummy_img, cfg)
    h_full = compute_phash(full_img, cfg)

    attack_metrics_dummy: Dict[str, int] = {}
    for name, variant in _attack_variants(dummy_img, rng=rng).items():
        attack_metrics_dummy[name] = _phash_distance(h_orig, compute_phash(variant, cfg))

    attack_metrics_full: Dict[str, int] = {}
    for name, variant in _attack_variants(full_img, rng=rng).items():
        attack_metrics_full[name] = _phash_distance(h_orig, compute_phash(variant, cfg))

    less_than_k1_level, less_than_k1_image = sis.reconstruct_with_levels(shares[: max(1, sis.k1 - 1)])
    h_less = compute_phash(less_than_k1_image, cfg)
    attack_metrics_less: Dict[str, int] = {}
    for name, variant in _attack_variants(less_than_k1_image, rng=rng).items():
        attack_metrics_less[name] = _phash_distance(h_orig, compute_phash(variant, cfg))

    # Baseline: blurred dummy
    blur_img = _blur_image(img)
    h_blur = compute_phash(blur_img, cfg)
    blur_attacks = {
        name: _phash_distance(h_orig, compute_phash(v, cfg)) for name, v in _attack_variants(blur_img, rng).items()
    }

    # Baseline: random noise
    noise_img = _noise_image_like(img, rng)
    h_noise = compute_phash(noise_img, cfg)
    noise_attacks = {
        name: _phash_distance(h_orig, compute_phash(v, cfg)) for name, v in _attack_variants(noise_img, rng).items()
    }

    # Baseline: single-layer Shamir (full image only)
    if not fast_mode:
        s_bytes = image_path.read_bytes()
        t0 = time.perf_counter()
        shares_single = shamir_split_secret(s_bytes, n=sis.n, k=sis.k2)
        t1 = time.perf_counter()
        full_recovered = shamir_combine_secret(shares_single[: sis.k2])
        t2 = time.perf_counter()
        assert full_recovered == s_bytes

        single_split_ms = (t1 - t0) * 1000.0
        single_recover_ms = (t2 - t1) * 1000.0
        single_phash_dist = _phash_distance(h_orig, compute_phash(Image.open(BytesIO(full_recovered)).convert("RGB"), cfg))
    else:
        single_split_ms = 0.0
        single_recover_ms = 0.0
        single_phash_dist = 0

    return {
        "image": image_path.name,
        "phash_dist_dummy": _phash_distance(h_orig, h_dummy),
        "phash_dist_full": _phash_distance(h_orig, h_full),
        "phash_dist_less_than_k1": _phash_distance(h_orig, h_less),
        "psnr_dummy_vs_original": _psnr(dummy_img, img),
        "psnr_full_vs_original": _psnr(full_img, img),
        "attack_phash_distances_dummy": attack_metrics_dummy,
        "attack_phash_distances_full": attack_metrics_full,
        "attack_phash_distances_less": attack_metrics_less,
        "split_ms": split_ms,
        "recover_dummy_ms": recover_dummy_ms,
        "recover_full_ms": recover_full_ms,
        "less_than_k1_level": less_than_k1_level,
        "baseline_blur": {
            "phash_dist": _phash_distance(h_orig, h_blur),
            "psnr": _psnr(blur_img, img),
            "attack_phash_distances": blur_attacks,
        },
        "baseline_noise": {
            "phash_dist": _phash_distance(h_orig, h_noise),
            "psnr": _psnr(noise_img, img),
            "attack_phash_distances": noise_attacks,
        },
        "single_layer_shamir": {
            "split_ms": single_split_ms,
            "recover_ms": single_recover_ms,
            "phash_dist": single_phash_dist,
        },
    }


def gather_images(images_dir: Path, max_images: int) -> List[Path]:
    paths: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        # rglob でサブディレクトリも探索
        paths.extend(sorted(images_dir.rglob(ext)))
    return paths[:max_images]


def gather_images_from_mapping(mapping_json: Path, max_images: int, include_variants: Iterable[str] | None) -> List[Path]:
    mapping = load_derivative_mapping(mapping_json)
    paths: List[Path] = []
    for variants in mapping.values():
        for variant, p in variants.items():
            if include_variants is not None and variant not in include_variants:
                continue
            paths.append(Path(p))
    return paths[:max_images]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate phash_masked_sis performance and robustness.")
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=None,
        help="Directory containing input images. Optional if mapping_json is provided.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("output/phash_masked_sis_eval"))
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k1", type=int, default=2)
    parser.add_argument("--k2", type=int, default=4)
    parser.add_argument("--hash_size", type=int, default=8)
    parser.add_argument("--highfreq_factor", type=int, default=4)
    parser.add_argument("--shamir_secret_len", type=int, default=65536, help="Bytes for Shamir micro-benchmark.")
    parser.add_argument("--repeat", type=int, default=5, help="Repetitions for micro-benchmarks.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mapping_json", type=Path, default=None, help="Optional mapping JSON to select images.")
    parser.add_argument(
        "--include_variants",
        type=str,
        default=None,
        help="Comma-separated variant names when using mapping_json (e.g., original,jpeg75).",
    )
    parser.add_argument("--fast", action="store_true", help="Skip heavy reconstruction benchmarks.")
    args = parser.parse_args()

    cfg = PHashConfig(hash_size=args.hash_size, highfreq_factor=args.highfreq_factor)
    rng = np.random.default_rng(args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dct_stats = benchmark_dct(cfg, repeats=args.repeat, seed=args.seed)
    shamir_stats = benchmark_shamir(
        secret_len=args.shamir_secret_len, n=args.n, k=args.k2, repeats=args.repeat, seed=args.seed
    )

    sis = MultiSecretImageSIS(n=args.n, k1=args.k1, k2=args.k2, cfg=cfg)

    include_variants = None
    if args.include_variants:
        include_variants = [v.strip() for v in args.include_variants.split(",") if v.strip()]

    if args.mapping_json:
        images = gather_images_from_mapping(args.mapping_json, args.max_images, include_variants)
        if not images:
            raise SystemExit(f"No images found in mapping {args.mapping_json} (variants={include_variants}).")
    else:
        if args.images_dir is None:
            raise SystemExit("Either --images_dir or --mapping_json must be provided.")
        images = gather_images(args.images_dir, args.max_images)
        if not images:
            raise SystemExit(f"No images found under {args.images_dir}")

    image_results: List[Dict[str, object]] = []
    for img_path in images:
        print(f"[eval] {img_path.name}")
        image_results.append(evaluate_image(img_path, sis, cfg, rng, fast_mode=args.fast))

    results = {
        "config": {
            "n": args.n,
            "k1": args.k1,
            "k2": args.k2,
            "hash_size": args.hash_size,
            "highfreq_factor": args.highfreq_factor,
            "max_images": args.max_images,
            "seed": args.seed,
        },
        "dct": dct_stats,
        "shamir": shamir_stats,
        "images": image_results,
    }

    with open(out_dir / "phash_masked_sis_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== DCT ===")
    print(json.dumps(dct_stats, indent=2))
    print("=== Shamir ===")
    print(json.dumps(shamir_stats, indent=2))
    print(f"Wrote image-level metrics for {len(image_results)} images to {out_dir/'phash_masked_sis_eval.json'}")


if __name__ == "__main__":
    main()
