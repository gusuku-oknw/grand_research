"""Evaluate plain vs pHash-preserving dummy (k1) search on a mapped dataset, optionally plotting summary graphs."""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
from PIL import Image

# Allow running without installation
import sys
import json
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Minimal dataset helpers (avoids importing experiments.common.__init__ which pulls matplotlib)
@dataclass(frozen=True)
class Sample:
    image_id: str
    variant: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.image_id}__{self.variant}"


def load_derivative_mapping(mapping_json: Path) -> Dict[str, Dict[str, str]]:
    with mapping_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    def _normalize(p: str) -> str:
        normalized = p.replace("\\", "/")
        if len(normalized) >= 2 and normalized[1] == ":":
            drive = normalized[0].lower()
            normalized = f"/mnt/{drive}{normalized[2:]}"
        return normalized

    return {
        img_id: {variant: _normalize(str(path)) for variant, path in variants.items()}
        for img_id, variants in raw.items()
    }


def build_samples(mapping: Dict[str, Dict[str, str]], include_variants: Iterable[str] | None = None) -> List[Sample]:
    samples: List[Sample] = []
    for image_id, variants in mapping.items():
        for variant, path in variants.items():
            if include_variants is not None and variant not in include_variants:
                continue
            samples.append(Sample(image_id=image_id, variant=variant, path=Path(path)))
    return samples


def build_positive_lookup(samples: List[Sample]) -> Dict[str, List[int]]:
    lookup: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        lookup.setdefault(sample.image_id, []).append(idx)
    return lookup
from phash_masked_sis import PHashConfig, compute_phash  # noqa: E402
from phash_masked_sis.dummy_image import make_phash_preserving_dummy  # noqa: E402


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.astype(np.uint8) ^ b.astype(np.uint8)))


def compute_phashes_with_dummy(
    samples: List[Sample],
    cfg: PHashConfig,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    plain: Dict[str, np.ndarray] = {}
    dummy: Dict[str, np.ndarray] = {}
    for s in samples:
        img = Image.open(s.path).convert("RGB")
        plain[s.key] = compute_phash(img, cfg=cfg)
        dummy_img, _ = make_phash_preserving_dummy(img, cfg=cfg, seed=seed, return_debug=True)
        dummy[s.key] = compute_phash(dummy_img, cfg=cfg)
    return plain, dummy


def rank(db_ph: Dict[str, np.ndarray], q_ph: np.ndarray) -> List[str]:
    items = [(k, hamming(v, q_ph)) for k, v in db_ph.items()]
    items.sort(key=lambda x: x[1])
    return [k for k, _ in items]


@dataclass
class Row:
    query_key: str
    mode: str
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    map: float
    mrr: float
    total_ms: float
    n_dataset: int


def precision_recall_at_k(ranked: List[int], positives: List[int], k_values=(1, 5, 10)) -> Tuple[List[float], List[float]]:
    pos_set = set(positives)
    p_vals, r_vals = [], []
    for k in k_values:
        topk = ranked[:k]
        hits = sum(1 for x in topk if x in pos_set)
        p = hits / k
        r = hits / len(pos_set) if pos_set else 0.0
        p_vals.append(p)
        r_vals.append(r)
    return p_vals, r_vals


def average_precision_from_ranking(ranked: List[int], positives: set[int]) -> float:
    if not positives:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for rank, idx in enumerate(ranked, start=1):
        if idx in positives:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(positives)


def reciprocal_rank(ranked: List[int], positives: set[int]) -> float:
    for rank, idx in enumerate(ranked, start=1):
        if idx in positives:
            return 1.0 / rank
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate plain vs dummy (k1) pHash search.")
    ap.add_argument("--mapping_json", type=Path, required=True, help="Derivative mapping JSON (from prepare_coco.py).")
    ap.add_argument("--max_queries", type=int, default=2000)
    ap.add_argument("--hash_size", type=int, default=8)
    ap.add_argument("--highfreq_factor", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--output_csv", type=Path, default=Path("output/results/masked_phash_eval.csv"))
    ap.add_argument(
        "--include_variants",
        type=str,
        default=None,
        help="Comma-separated variant names to include (e.g., original,jpeg75). Default: all variants.",
    )
    ap.add_argument(
        "--plot_dir",
        type=Path,
        default=None,
        help="Directory to save summary plots (requires matplotlib). Default: auto (same as output_csv stem + _figs). Use 'none' to disable.",
    )
    ap.add_argument(
        "--per_variant_plots",
        action="store_true",
        help="Loop over all variants in mapping_json and generate CSV/plots per variant (ignores include_variants if set).",
    )
    args = ap.parse_args()

    summary_rows: List[Dict[str, object]] = []

    def run_once(include_variants: Iterable[str] | None, output_csv: Path, plot_dir: Path | None, variant_name: str) -> None:
        all_samples = build_samples(mapping, include_variants=include_variants)
        samples = all_samples[: args.max_queries]
        positives_lookup = build_positive_lookup(samples)
        if not samples:
            print(f"No samples for variants={include_variants}; skipping.")
            return
        cfg = PHashConfig(hash_size=args.hash_size, highfreq_factor=args.highfreq_factor)
        db_ph, db_dummy = compute_phashes_with_dummy(samples, cfg, seed=args.seed)

        rows: List[Row] = []
        id_to_index = {s.key: idx for idx, s in enumerate(samples)}

        for s in samples:
            positives = positives_lookup.get(s.image_id, [])
            q_ph_plain = db_ph[s.key]
            q_ph_dummy = db_dummy[s.key]

            for mode, q_ph in [("plain", q_ph_plain), ("dummy_k1", q_ph_dummy)]:
                t0 = time.perf_counter()
                ranked_keys = rank(db_ph, q_ph)
                ms = (time.perf_counter() - t0) * 1000.0
                ranked_idx = [id_to_index[k] for k in ranked_keys]
                p_vals, r_vals = precision_recall_at_k(ranked_idx, positives)
                pos_set = set(positives)
                ap = average_precision_from_ranking(ranked_idx, pos_set)
                rr = reciprocal_rank(ranked_idx, pos_set)
                rows.append(
                    Row(
                        query_key=s.key,
                        mode=mode,
                        precision_at_1=p_vals[0],
                        precision_at_5=p_vals[1],
                        precision_at_10=p_vals[2],
                        recall_at_1=r_vals[0],
                        recall_at_5=r_vals[1],
                        recall_at_10=r_vals[2],
                        map=ap,
                        mrr=rr,
                        total_ms=ms,
                        n_dataset=len(samples),
                    )
                )

        def aggregate(rows: List[Row]) -> Dict[str, Dict[str, float]]:
            agg: Dict[str, Dict[str, List[float]]] = {}
            for r in rows:
                agg.setdefault(r.mode, {}).setdefault("p1", []).append(r.precision_at_1)
                agg[r.mode].setdefault("p5", []).append(r.precision_at_5)
                agg[r.mode].setdefault("p10", []).append(r.precision_at_10)
                agg[r.mode].setdefault("r1", []).append(r.recall_at_1)
                agg[r.mode].setdefault("r5", []).append(r.recall_at_5)
                agg[r.mode].setdefault("r10", []).append(r.recall_at_10)
                agg[r.mode].setdefault("map", []).append(r.map)
                agg[r.mode].setdefault("mrr", []).append(r.mrr)
                agg[r.mode].setdefault("ms", []).append(r.total_ms)
            out: Dict[str, Dict[str, float]] = {}
            for mode, vals in agg.items():
                out[mode] = {
                    "precision_at_1": float(np.mean(vals["p1"])),
                    "precision_at_5": float(np.mean(vals["p5"])),
                    "precision_at_10": float(np.mean(vals["p10"])),
                    "recall_at_1": float(np.mean(vals["r1"])),
                    "recall_at_5": float(np.mean(vals["r5"])),
                    "recall_at_10": float(np.mean(vals["r10"])),
                    "map": float(np.mean(vals["map"])),
                    "mrr": float(np.mean(vals["mrr"])),
                    "total_ms": float(np.mean(vals["ms"])),
                }
            return out

        summary = aggregate(rows)
        # record summary for global aggregation
        for mode, vals in summary.items():
            summary_rows.append(
                {
                    "variant": variant_name,
                    "mode": mode,
                    "precision_at_1": vals["precision_at_1"],
                    "precision_at_5": vals["precision_at_5"],
                    "precision_at_10": vals["precision_at_10"],
                    "recall_at_1": vals["recall_at_1"],
                    "recall_at_5": vals["recall_at_5"],
                    "recall_at_10": vals["recall_at_10"],
                    "map": vals["map"],
                    "mrr": vals["mrr"],
                    "total_ms": vals["total_ms"],
                }
            )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "query_key",
                    "mode",
                    "precision_at_1",
                    "precision_at_5",
                    "precision_at_10",
                    "recall_at_1",
                    "recall_at_5",
                    "recall_at_10",
                    "map",
                    "mrr",
                    "total_ms",
                    "n_dataset",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r.query_key,
                        r.mode,
                        r.precision_at_1,
                        r.precision_at_5,
                        r.precision_at_10,
                        r.recall_at_1,
                        r.recall_at_5,
                        r.recall_at_10,
                        r.map,
                        r.mrr,
                        r.total_ms,
                        r.n_dataset,
                    ]
                )
        print(f"saved: {output_csv}")

        if plot_dir:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise SystemExit("matplotlib is required for plotting. Install with `pip install matplotlib`.") from None

            plot_dir.mkdir(parents=True, exist_ok=True)
            modes = sorted(summary.keys())
            if not modes:
                print("No modes to plot.")
                return

            recall_fig, recall_ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(modes))
            width = 0.25
            r1 = [summary[m]["recall_at_1"] for m in modes]
            r5 = [summary[m]["recall_at_5"] for m in modes]
            r10 = [summary[m]["recall_at_10"] for m in modes]
            recall_ax.bar(x - width, r1, width, label="R@1")
            recall_ax.bar(x, r5, width, label="R@5")
            recall_ax.bar(x + width, r10, width, label="R@10")
            recall_ax.set_xticks(x)
            recall_ax.set_xticklabels(modes, rotation=20)
            recall_ax.set_ylim(0, 1.05)
            recall_ax.set_ylabel("Recall")
            recall_ax.set_title("Recall (mean)")
            recall_ax.legend()
            recall_fig.tight_layout()
            recall_out = plot_dir / "recall_summary.png"
            recall_fig.savefig(recall_out, bbox_inches="tight", dpi=150)
            plt.close(recall_fig)

            time_fig, time_ax = plt.subplots(figsize=(6, 4))
            ms_vals = [summary[m]["total_ms"] for m in modes]
            time_ax.bar(modes, ms_vals, color="#4c72b0")
            time_ax.set_ylabel("Time per query (ms)")
            time_ax.set_title("Latency (mean)")
            time_ax.set_ylim(0, max(ms_vals) * 1.2 if ms_vals else 1)
            time_fig.tight_layout()
            time_out = plot_dir / "time_summary.png"
            time_fig.savefig(time_out, bbox_inches="tight", dpi=150)
            plt.close(time_fig)

            print(f"saved plots: {recall_out}, {time_out}")

    mapping = load_derivative_mapping(args.mapping_json)

    if args.per_variant_plots:
        all_variants = sorted({v for variants in mapping.values() for v in variants.keys()})
        for variant in all_variants:
            suffix = variant.replace("/", "_")
            out_csv = args.output_csv.parent / f"{args.output_csv.stem}_{suffix}.csv"
            pd = None
            if not (isinstance(args.plot_dir, str) and args.plot_dir.lower() == "none"):
                base_dir = args.plot_dir if args.plot_dir else args.output_csv.parent / f"{args.output_csv.stem}_figs"
                pd = base_dir / variant
            print(f"[per-variant] {variant}")
            run_once(include_variants=[variant], output_csv=out_csv, plot_dir=pd, variant_name=variant)

        # combined summary plots across variants
        if summary_rows:
            summary_csv = args.output_csv.parent / f"{args.output_csv.stem}_summary.csv"
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            with summary_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "variant",
                        "mode",
                        "precision_at_1",
                        "precision_at_5",
                        "precision_at_10",
                        "recall_at_1",
                        "recall_at_5",
                        "recall_at_10",
                        "map",
                        "mrr",
                        "total_ms",
                    ]
                )
                for r in summary_rows:
                    writer.writerow(
                        [
                            r["variant"],
                            r["mode"],
                            r["precision_at_1"],
                            r["precision_at_5"],
                            r["precision_at_10"],
                            r["recall_at_1"],
                            r["recall_at_5"],
                            r["recall_at_10"],
                            r["map"],
                            r["mrr"],
                            r["total_ms"],
                        ]
                    )
            # plots
            plot_base = args.plot_dir if args.plot_dir else args.output_csv.parent / f"{args.output_csv.stem}_figs"
            if not (isinstance(args.plot_dir, str) and args.plot_dir.lower() == "none"):
                try:
                    import matplotlib.pyplot as plt
                except ImportError:
                    raise SystemExit("matplotlib is required for plotting. Install with `pip install matplotlib`.") from None
                plot_base.mkdir(parents=True, exist_ok=True)
                # Recall@10 per variant
                variants = sorted({r["variant"] for r in summary_rows})
                modes = sorted({r["mode"] for r in summary_rows})
                x = np.arange(len(variants))
                width = 0.8 / max(len(modes), 1)
                fig, ax = plt.subplots(figsize=(max(6, len(variants) * 0.6), 4))
                for i, mode in enumerate(modes):
                    vals = []
                    for v in variants:
                        matched = [r for r in summary_rows if r["variant"] == v and r["mode"] == mode]
                        vals.append(matched[0]["recall_at_10"] if matched else 0.0)
                    ax.bar(x + i * width, vals, width, label=mode)
                ax.set_xticks(x + width * (len(modes) - 1) / 2)
                ax.set_xticklabels(variants, rotation=45, ha="right")
                ax.set_ylabel("Recall@10")
                ax.set_ylim(0, 1.05)
                ax.set_title("Recall@10 per variant")
                ax.legend()
                fig.tight_layout()
                fig.savefig(plot_base / "recall_summary_all_variants.png", bbox_inches="tight", dpi=150)
                plt.close(fig)

                # Time per variant
                fig, ax = plt.subplots(figsize=(max(6, len(variants) * 0.6), 4))
                for i, mode in enumerate(modes):
                    vals = []
                    for v in variants:
                        matched = [r for r in summary_rows if r["variant"] == v and r["mode"] == mode]
                        vals.append(matched[0]["total_ms"] if matched else 0.0)
                    ax.bar(x + i * width, vals, width, label=mode)
                ax.set_xticks(x + width * (len(modes) - 1) / 2)
                ax.set_xticklabels(variants, rotation=45, ha="right")
                ax.set_ylabel("Time per query (ms)")
                ax.set_title("Latency per variant")
                ax.legend()
                fig.tight_layout()
                fig.savefig(plot_base / "time_summary_all_variants.png", bbox_inches="tight", dpi=150)
                plt.close(fig)

                print(f"saved combined plots under {plot_base}")
    else:
        # single run with optional include_variants
        include_variants = None
        if args.include_variants:
            if args.include_variants.strip().lower() == "all":
                include_variants = None
            else:
                include_variants = [v.strip() for v in args.include_variants.split(",") if v.strip()]
        plot_dir = args.plot_dir
        if isinstance(plot_dir, str) and plot_dir.lower() == "none":
            plot_dir = None
        if plot_dir is None and args.plot_dir is None:
            plot_dir = args.output_csv.parent / f"{args.output_csv.stem}_figs"
        run_once(include_variants=include_variants, output_csv=args.output_csv, plot_dir=plot_dir, variant_name=args.include_variants or "all")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
