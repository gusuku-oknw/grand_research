"""
Run a tiny searchable-SIS demo on tests/fixtures and save a JSON summary.

Usage (defaults are tuned for the small tests/fixtures set):
    python experiments/scripts/run_demo_tests_summary.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import sys

# Ensure repository root is on PYTHONPATH when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pHR_SIS.workflow import SearchableSISWithImageStore

DEMO_BASE_OUTPUT = Path("output/demo_tests")


def run_demo(
    images_dir: Path = Path("tests/fixtures"),
    shares_dir: Path = DEMO_BASE_OUTPUT / "shares",
    meta_dir: Path = DEMO_BASE_OUTPUT / "meta",
    recon_dir: Path = DEMO_BASE_OUTPUT / "recon",
    summary_path: Path = DEMO_BASE_OUTPUT / "summary.json",
    k: int = 2,
    n: int = 3,
    bands: int = 8,
    min_band_votes: int = 3,
    topk: int = 10,
    max_hamming: int | None = 10,
    reconstruct_top: int = 1,
    dummy_band_queries: int = 1,
    pad_band_queries: int | None = None,
    fixed_band_queries: int | None = 4,
) -> Dict[str, object]:
    images = sorted(p for p in images_dir.iterdir() if p.is_file())
    if not images:
        raise SystemExit(f"No images under {images_dir}")

    wf = SearchableSISWithImageStore(
        k=k,
        n=n,
        bands=bands,
        token_len=8,
        seed=2025,
        shares_dir=str(shares_dir),
        meta_dir=str(meta_dir),
        secure_distance=False,
        share_strategy="shamir",
        fusion_grid=8,
        fusion_threshold=None,
        dummy_band_queries=dummy_band_queries,
        pad_band_queries=pad_band_queries,
        fixed_band_queries=fixed_band_queries,
    )

    add_t0 = time.perf_counter()
    added: List[Dict[str, object]] = []
    for idx, path in enumerate(images):
        image_id = f"img_{idx:04d}"
        phash = wf.add_image(image_id, str(path))
        added.append({"image_id": image_id, "file": path.name, "phash": f"0x{phash:016x}"})
    add_elapsed = time.perf_counter() - add_t0

    query_path = str(images[0])
    query_t0 = time.perf_counter()
    result = wf.query_and_optionally_reconstruct(
        query_image_path=query_path,
        servers_for_query=wf.list_servers()[:k],
        min_band_votes=min_band_votes,
        topk=topk,
        max_hamming=max_hamming,
        reconstruct_top=reconstruct_top,
        recon_dir=str(recon_dir),
        dummy_band_queries=dummy_band_queries,
        pad_band_queries=pad_band_queries,
        fixed_band_queries=fixed_band_queries,
    )
    query_elapsed = time.perf_counter() - query_t0

    summary: Dict[str, object] = {
        "config": {
            "k": k,
            "n": n,
            "bands": bands,
            "min_band_votes": min_band_votes,
            "topk": topk,
            "max_hamming": max_hamming,
            "reconstruct_top": reconstruct_top,
            "dummy_band_queries": dummy_band_queries,
            "pad_band_queries": pad_band_queries,
            "fixed_band_queries": fixed_band_queries,
            "images_dir": str(images_dir),
            "shares_dir": str(shares_dir),
            "meta_dir": str(meta_dir),
            "recon_dir": str(recon_dir),
        },
        "added": added,
        "timing": {"add_seconds": add_elapsed, "query_seconds": query_elapsed},
        "result": {
            "query": Path(query_path).name,
            "preselected": result.get("preselected"),
            "ranked": result.get("ranked"),
            "reconstructed": result.get("reconstructed"),
            "reconstruction_errors": result.get("reconstruction_errors"),
            "insufficient_shares": result.get("insufficient_shares"),
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    summary = run_demo()
    print(f"Saved summary to {DEMO_BASE_OUTPUT / 'summary.json'}")
    print(json.dumps(summary, indent=2))
