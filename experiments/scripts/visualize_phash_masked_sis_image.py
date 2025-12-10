from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

# Allow running without installation
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phash_masked_sis import MultiSecretImageSIS, PHashConfig, compute_phash  # noqa: E402
from phash_masked_sis.dummy_image import make_phash_preserving_dummy  # noqa: E402


def _attack_variants(img: Image.Image) -> Dict[str, Image.Image]:
    arr = np.asarray(img.convert("L"), dtype=np.float64)
    noisy = arr + np.random.normal(0.0, 8.0, size=arr.shape)
    variants: Dict[str, Image.Image] = {
        "gaussian_noise": Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="L").convert(img.mode)
    }
    buf = Image.new(img.mode, img.size)
    buf.paste(img)
    from io import BytesIO

    tmp = BytesIO()
    buf.save(tmp, format="JPEG", quality=30, subsampling=2)
    variants["jpeg_q30"] = Image.open(BytesIO(tmp.getvalue())).convert(img.mode)
    variants["rotate_5deg"] = img.rotate(5, resample=Image.BICUBIC, fillcolor=0)
    variants["blur"] = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return variants


def _noise_like(img: Image.Image) -> Image.Image:
    arr = np.random.randint(0, 256, size=(img.height, img.width, len(img.getbands())), dtype=np.uint8)
    if img.mode == "L":
        arr = arr[:, :, 0]
    return Image.fromarray(arr, mode=img.mode)


def _phash_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.astype(np.uint8) ^ b.astype(np.uint8)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize phash_masked_sis pipeline for a single image.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--output", type=Path, default=Path("output/phash_masked_sis_eval/pipeline.png"))
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k1", type=int, default=2)
    parser.add_argument("--k2", type=int, default=4)
    parser.add_argument("--hash_size", type=int, default=8)
    parser.add_argument("--highfreq_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    cfg = PHashConfig(hash_size=args.hash_size, highfreq_factor=args.highfreq_factor)
    sis = MultiSecretImageSIS(n=args.n, k1=args.k1, k2=args.k2, cfg=cfg)

    img = Image.open(args.image).convert("RGB")
    orig_hash = compute_phash(img, cfg)

    shares = sis.split_from_image(args.image, seed=args.seed)
    level_noise, img_noise = sis.reconstruct_with_levels(shares[: max(1, args.k1 - 1)])
    img_dummy_dbg, debug = make_phash_preserving_dummy(img, cfg=cfg, seed=args.seed, return_debug=True)
    img_dummy = img_dummy_dbg
    img_full = sis.recover_full_image(shares[: args.k2])

    blur_baseline = img.filter(ImageFilter.GaussianBlur(radius=3.0))
    noise_baseline = _noise_like(img)

    dummy_attacks = _attack_variants(img_dummy)

    steps = []
    if debug:
        bits = debug.get("bits")
        if bits is not None:
            bits_img = Image.fromarray((bits.astype(np.uint8) * 255), mode="L").resize((64, 64), Image.NEAREST)
            steps.append(("target bits", bits_img, None))
        lf = debug.get("lf_final")
        if lf is not None:
            lf_norm = lf - lf.min()
            if lf_norm.max() > 0:
                lf_norm = lf_norm / lf_norm.max() * 255.0
            lf_img = Image.fromarray(lf_norm.astype(np.uint8), mode="L").resize((64, 64), Image.BICUBIC)
            steps.append(("low-freq (reinforced)", lf_img, None))
        spatial = debug.get("spatial_small")
        if spatial is not None:
            steps.append(("32x32 spatial", Image.fromarray(spatial.astype(np.uint8), mode="L"), None))

    entries = [
        ("original", img, _phash_distance(orig_hash, orig_hash)),
        (f"<k1 ({level_noise})", img_noise, _phash_distance(orig_hash, compute_phash(img_noise, cfg))),
        (f"dummy k1", img_dummy, _phash_distance(orig_hash, compute_phash(img_dummy, cfg))),
        ("full k2", img_full, _phash_distance(orig_hash, compute_phash(img_full, cfg))),
        ("blur baseline", blur_baseline, _phash_distance(orig_hash, compute_phash(blur_baseline, cfg))),
        ("noise baseline", noise_baseline, _phash_distance(orig_hash, compute_phash(noise_baseline, cfg))),
    ]

    for name, variant in dummy_attacks.items():
        entries.append((f"dummy + {name}", variant, _phash_distance(orig_hash, compute_phash(variant, cfg))))

    all_entries = steps + entries

    cols = 3
    rows = int(np.ceil(len(all_entries) / cols))
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for ax, (title, im, dist) in zip(axes, all_entries):
        ax.imshow(im)
        if dist is None:
            ax.set_title(title)
        else:
            ax.set_title(f"{title}\npHash dist={dist}")
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(all_entries) :]:
        ax.axis("off")

    fig.suptitle("phash_masked_sis pipeline (single image)", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
