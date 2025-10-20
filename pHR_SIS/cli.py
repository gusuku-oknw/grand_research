"""Command line entry points mirroring the original SIS demos."""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Iterable, List

from .image_store import ShamirImageStore
from .index import SearchableSISIndex
from .workflow import SearchableSISWithImageStore


def _ensure_images(images_dir: str) -> List[str]:
    os.makedirs(images_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(images_dir, "*")))
    if not images:
        raise SystemExit(f"Populate {images_dir} with images before running the demo.")
    return images


def _default_servers(index: SearchableSISIndex, k: int) -> List[int]:
    servers = index.list_servers()
    if len(servers) < k:
        raise SystemExit(f"Configured index exposes only {len(servers)} servers; need at least k={k}.")
    return servers[:k]


def run_selective_demo(args: argparse.Namespace) -> int:
    index = SearchableSISIndex(
        k=args.k,
        n=args.n,
        bands=args.bands,
        token_len=args.token_len,
        seed=args.seed,
    )
    db_images = _ensure_images(args.images_dir)
    print(f"[CONFIG] k={args.k}, n={args.n}, bands={args.bands}, "
          f"min_band_votes={args.min_band_votes}, topk={args.topk}, max_hamming={args.max_hamming}")
    print(f"[INFO]   images_dir={args.images_dir} (found {len(db_images)})")
    print("-" * 72)

    t0 = time.perf_counter()
    for i, path in enumerate(db_images):
        image_id = f"img_{i:04d}"
        phash = index.add_image(image_id, path)
        print(f"[ADD] {image_id:<12} file={os.path.basename(path):<36} pHash=0x{phash:016x}")
    dt = time.perf_counter() - t0
    print(f"[DONE] registered {len(db_images)} images in {dt:.3f}s")
    print("-" * 72)

    query_path = args.query or db_images[0]
    servers = _default_servers(index, args.k)
    print(f"[QUERY] query={os.path.basename(query_path)} servers={servers}")
    result = index.query_selective(
        query_path,
        servers_for_query=servers,
        min_band_votes=args.min_band_votes,
        topk=args.topk,
        max_hamming=args.max_hamming,
    )

    print(f"[P-HASH] query = {result['query_phash']}")
    print("[PRESELECT] (image_id, votes)")
    if result["preselected"]:
        for image_id, votes in result["preselected"]:
            print(f"  - {image_id:<12} votes={votes}")
    else:
        print("  (no candidates passed min_band_votes)")
    print("[RANK] (image_id, Hamming)")
    if result["ranked"]:
        for image_id, dist in result["ranked"]:
            print(f"  - {image_id:<12} Hamming={dist}")
    else:
        print("  (no matches within threshold)")
    print("-" * 72)

    query_hash = int(result["query_phash"], 16)
    all_ids = list(index._images.keys())
    t_full0 = time.perf_counter()
    _ = index.rank_candidates(query_hash, servers, all_ids, topk=args.topk, max_hamming=args.max_hamming)
    t_full = time.perf_counter() - t_full0
    t_sel0 = time.perf_counter()
    _ = index.rank_candidates(query_hash, servers, [img for img, _ in result["preselected"]],
                              topk=args.topk, max_hamming=args.max_hamming)
    t_sel = time.perf_counter() - t_sel0
    print(f"[BENCH] full-reconstruct ranking : {t_full:.3f}s ({len(all_ids)} candidates)")
    print(f"[BENCH] selective reconstructing: {t_sel:.3f}s ({len(result['preselected'])} candidates after preselect)")
    return 0


def run_workflow_demo(args: argparse.Namespace) -> int:
    workflow = SearchableSISWithImageStore(
        k=args.k,
        n=args.n,
        bands=args.bands,
        token_len=args.token_len,
        seed=args.seed,
        shares_dir=args.shares_dir,
        meta_dir=args.meta_dir,
    )
    db_images = _ensure_images(args.images_dir)
    print(f"[CONFIG] k={args.k}, n={args.n}, bands={args.bands}, "
          f"min_band_votes={args.min_band_votes}, topk={args.topk}, "
          f"max_hamming={args.max_hamming}, reconstruct_top={args.reconstruct_top}")
    print(f"[INFO]   images_dir={args.images_dir} (found {len(db_images)})")
    print("-" * 72)

    t0 = time.perf_counter()
    ids: List[str] = []
    for i, path in enumerate(db_images):
        image_id = f"img_{i:04d}"
        phash = workflow.add_image(image_id, path)
        ids.append(image_id)
        print(f"[ADD] {image_id:<12} file={os.path.basename(path):<36} pHash=0x{phash:016x}")
    print(f"[DONE] registered {len(ids)} images in {time.perf_counter() - t0:.3f}s")
    print("-" * 72)

    query_path = args.query or db_images[0]
    servers = _default_servers(workflow.index, args.k)
    print(f"[QUERY] query={os.path.basename(query_path)} servers={servers}")
    result = workflow.query_and_optionally_reconstruct(
        query_path,
        servers_for_query=servers,
        min_band_votes=args.min_band_votes,
        topk=args.topk,
        max_hamming=args.max_hamming,
        reconstruct_top=args.reconstruct_top,
        recon_dir=args.recon_dir,
    )

    print(f"[P-HASH] query = {result['query_phash']}")
    print("[PRESELECT] (image_id, votes)")
    if result["preselected"]:
        for image_id, votes in result["preselected"]:
            print(f"  - {image_id:<12} votes={votes}")
    else:
        print("  (no candidates passed min_band_votes)")
    print("[RANK] (image_id, Hamming)")
    if result["ranked"]:
        for image_id, distance in result["ranked"]:
            print(f"  - {image_id:<12} Hamming={distance}")
    else:
        print("  (no matches within threshold)")
    if result["reconstructed"]:
        print("[RECONSTRUCTED]")
        for image_id, path in result["reconstructed"]:
            print(f"  - {image_id:<12} -> {path}")
    else:
        print("[RECONSTRUCTED] (none)")
    return 0


def run_image_store_demo(args: argparse.Namespace) -> int:
    store = ShamirImageStore(k=args.k, n=args.n, shares_dir=args.shares_dir, meta_dir=args.meta_dir)
    db_images = _ensure_images(args.images_dir)
    print(f"[CONFIG] k={args.k}, n={args.n}")
    print(f"[INFO]   images_dir={args.images_dir} (found {len(db_images)})")
    print("-" * 72)

    t0 = time.perf_counter()
    ids: List[str] = []
    for i, path in enumerate(db_images):
        image_id = f"img_{i:04d}"
        meta = store.add_image(image_id, path, rng_seed=args.seed, skip_if_exists=False)
        ids.append(image_id)
        print(f"[ADD] {image_id:<12} <- {meta.filename:<32} shape={meta.shape}")
    print(f"[DONE] shared {len(ids)} images in {time.perf_counter() - t0:.3f}s")
    print("-" * 72)

    target = ids[0]
    servers = list(range(1, args.k + 1))
    out_path = os.path.join(args.recon_dir, f"reconstructed_{target}.png")
    os.makedirs(args.recon_dir, exist_ok=True)
    ok = store.reconstruct(target, servers, out_path)
    print(f"[RECONSTRUCT] {target} with servers={servers} -> {'OK' if ok else 'FAIL'} ({out_path if ok else '-'})")
    try:
        store.reconstruct(target, servers[:-1], out_path + ".tmp")
    except ValueError:
        print(f"[RECONSTRUCT] {target} with servers={servers[:-1]} -> FAIL (expected)")
    return 0


def run_secure_demo(args: argparse.Namespace) -> int:
    index = SearchableSISIndex(k=args.k, n=args.n, bands=args.bands, token_len=args.token_len, seed=args.seed)
    db_images = _ensure_images(args.images_dir)
    for i, path in enumerate(db_images):
        index.add_image(f"img_{i:04d}", path)

    servers_ok = _default_servers(index, args.k)
    query_path = db_images[0]
    print(f"[QUERY-A] query={os.path.basename(query_path)} servers={servers_ok}")
    hits = index.query_selective(query_path, servers_ok, min_band_votes=3, topk=args.topk, max_hamming=args.max_hamming)
    print(f"[RESULT-A] top matches: {hits['ranked']}")

    servers_ng = servers_ok[:-1]
    print(f"[QUERY-B] insufficient servers={servers_ng}")
    try:
        index.rank_candidates(0, servers_ng, [])
    except ValueError as exc:
        print(f"[RESULT-B] expected failure: {exc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sis-image", description="Secret Image Sharing demos.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--k", type=int, default=3)
        subparser.add_argument("--n", type=int, default=5)
        subparser.add_argument("--bands", type=int, default=8)
        subparser.add_argument("--token_len", type=int, default=8)
        subparser.add_argument("--seed", type=int, default=2025)
        subparser.add_argument("--images_dir", type=str, default="images")
        subparser.add_argument("--topk", type=int, default=10)
        subparser.add_argument("--max_hamming", type=int, default=10)

    selective = sub.add_parser("selective-demo", help="Run the selective reconstruction demo.")
    add_common(selective)
    selective.add_argument("--min_band_votes", type=int, default=3)
    selective.add_argument("--query", type=str, default=None)
    selective.set_defaults(func=run_selective_demo)

    workflow = sub.add_parser("search-demo", help="Run the combined search + image reconstruction demo.")
    add_common(workflow)
    workflow.add_argument("--min_band_votes", type=int, default=3)
    workflow.add_argument("--reconstruct_top", type=int, default=1)
    workflow.add_argument("--recon_dir", type=str, default="recon_out")
    workflow.add_argument("--shares_dir", type=str, default="img_shares")
    workflow.add_argument("--meta_dir", type=str, default="img_meta")
    workflow.add_argument("--query", type=str, default=None)
    workflow.set_defaults(func=run_workflow_demo)

    store = sub.add_parser("image-store-demo", help="Share and reconstruct raw images.")
    store.add_argument("--k", type=int, default=3)
    store.add_argument("--n", type=int, default=5)
    store.add_argument("--seed", type=int, default=2025)
    store.add_argument("--images_dir", type=str, default="images")
    store.add_argument("--shares_dir", type=str, default="img_shares")
    store.add_argument("--meta_dir", type=str, default="img_meta")
    store.add_argument("--recon_dir", type=str, default="recon_out")
    store.set_defaults(func=run_image_store_demo)

    secure = sub.add_parser("secure-demo", help="Exercise failure scenarios for the SIS search index.")
    add_common(secure)
    secure.set_defaults(func=run_secure_demo)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
