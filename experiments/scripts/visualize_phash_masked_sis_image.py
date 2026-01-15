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

from phash_masked_sis import MultiSecretImageSIS, PHashConfig, compute_phash, idct2  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Visualize phash_masked_sis pipeline for multiple images with multiple dummies.")
    parser.add_argument("--image", type=Path, required=True, nargs="+", help="Input image path(s).")
    parser.add_argument("--combine_output", type=Path, default=None, help="If set, combine all images into a single figure saved to this path.")
    parser.add_argument("--output_dir", type=Path, default=Path("output/phash_masked_sis_eval/pipeline_viz"), help="Directory to save output images (if not combined).")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k1", type=int, default=2)
    parser.add_argument("--k2", type=int, default=4)
    parser.add_argument("--hash_size", type=int, default=8)
    parser.add_argument("--highfreq_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_dummies", type=int, default=1, help="Number of dummy variants to generate per image.")
    args = parser.parse_args()

    cfg = PHashConfig(hash_size=args.hash_size, highfreq_factor=args.highfreq_factor)
    sis = MultiSecretImageSIS(n=args.n, k1=args.k1, k2=args.k2, cfg=cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows_entries = []
    
    # Common helper to resize for display (height=256, keep aspect)
    def resize_for_viz(im, target_h=256):
        w, h = im.size
        new_w = int(w * target_h / h)
        return im.resize((new_w, target_h), Image.BICUBIC)

    for img_idx, img_path in enumerate(args.image):
        print(f"Processing {img_path.name}...")
        img = Image.open(img_path).convert("RGB")
        orig_hash = compute_phash(img, cfg)

        # 1. SIS Split & Partial Reconstruction (<k1)
        shares = sis.split_from_image(img_path, seed=args.seed)
        level_noise, img_noise = sis.reconstruct_with_levels(shares[: max(1, args.k1 - 1)])
        
        # 2. Generate Multiple Dummies (k1)
        dummies = []
        for i in range(args.num_dummies):
            # Change seed for each dummy to get different visual noise
            d_img, _ = make_phash_preserving_dummy(img, cfg=cfg, seed=args.seed + i, return_debug=True)
            dummies.append(d_img)

        # 3. Full Reconstruction (k2)
        img_full = sis.recover_full_image(shares[: args.k2])

        # Prepare entries for plotting: [Original, Noise, Dummy 1, ..., Dummy N, Full]
        # Resize all to same height for consistent grid
        row_entries = [
            ("Original", resize_for_viz(img), _phash_distance(orig_hash, orig_hash)),
            (f"<k1 ({level_noise})", resize_for_viz(img_noise), _phash_distance(orig_hash, compute_phash(img_noise, cfg)))
        ]
        
        for i, d_img in enumerate(dummies):
            label = f"Dummy {i+1} (k1)" if len(dummies) > 1 else "Dummy (k1)"
            row_entries.append((label, resize_for_viz(d_img), _phash_distance(orig_hash, compute_phash(d_img, cfg))))
            
        row_entries.append(("Full (k2)", resize_for_viz(img_full), _phash_distance(orig_hash, compute_phash(img_full, cfg))))
        
        all_rows_entries.append((img_path.name, row_entries))
    
    if args.combine_output:
        # Generate single combined figure
        print(f"Generating combined figure with {len(all_rows_entries)} rows...")
        cols = len(all_rows_entries[0][1])
        rows = len(all_rows_entries)
        
        plt.rcParams.update({"figure.dpi": 200, "font.size": 8})
        # Determine figure width based on the aspect ratio of the first image's row
        # Roughly: (width of one image * cols) : (height of one image * rows)
        # We start with a base size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8), squeeze=False)
        
        for r, (row_name, row_entries) in enumerate(all_rows_entries):
            for c, (title, im, dist) in enumerate(row_entries):
                ax = axes[r][c]
                ax.imshow(im)
                ax.axis("off")
                
                # Title only on top row or specifically requested?
                # For combined figure, we might want column headers on top row
                # and maybe pHash dist below?
                if r == 0:
                    # Column header
                    clean_title = title.split("\n")[0] # remove pHash part if in title
                    ax.set_title(f"{clean_title}\n(dist={dist})", fontsize=10)
                else:
                    # Only dist for subsequent rows to save space
                    ax.set_title(f"dist={dist}", fontsize=8)
            
            # Add row label (Filename) to the left of the first column
            # axes[r][0].text(-0.25, 0.5, row_name, transform=axes[r][0].transAxes, 
            #                 rotation=90, va='center', ha='right', fontsize=9, fontweight='bold')

        fig.tight_layout()
        args.combine_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.combine_output, bbox_inches="tight")
        print(f"Saved combined figure: {args.combine_output}")
        plt.close(fig)

    else:
        # Save individual figures (legacy mode)
        for img_name, row_entries in all_rows_entries:
            plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
            cols = len(row_entries)
            rows = 1
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, 3.5))
            if cols == 1: axes = [axes]
            
            for ax, (title, im, dist) in zip(axes, row_entries):
                ax.imshow(im)
                if dist is None:
                    ax.set_title(title)
                else:
                    ax.set_title(f"{title}\npHash dist={dist}")
                ax.axis("off")

            fig.suptitle(f"SIS Pipeline: {img_name}", fontsize=12)
            fig.tight_layout()
            out_path = args.output_dir / f"{Path(img_name).stem}_pipeline.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
