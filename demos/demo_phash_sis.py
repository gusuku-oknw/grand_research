"""Unified demo for pHash + SIS workflows (standard or MPC simulated)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from pHR_SIS.workflow import SearchableSISWithImageStore


def _resolve_query_servers(raw: str | None, available: List[int], default_count: int) -> List[int]:
    if raw:
        parts = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parts.append(int(part))
            except ValueError:
                raise SystemExit(f"Invalid server ID '{part}' in --query_servers.")
        selected = [s for s in parts if s in available]
        if not selected:
            raise SystemExit("No requested servers match the available cluster nodes.")
        return selected
    return available[:min(default_count, len(available))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the searchable SIS demo with optional simulated MPC distance."
    )
    parser.add_argument("--mode", choices=["standard", "mpc"], default="standard")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--min_band_votes", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_hamming", type=int, default=10)
    parser.add_argument("--reconstruct_top", type=int, default=1)
    parser.add_argument("--images_dir", type=Path, default=Path("data"))
    parser.add_argument("--shares_dir", type=Path, default=Path("output/img_shares"))
    parser.add_argument("--meta_dir", type=Path, default=Path("output/img_meta"))
    parser.add_argument("--recon_dir", type=Path, default=Path("output/recon_out"))
    parser.add_argument("--query", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--share_strategy", choices=["shamir", "phash-fusion"], default="shamir")
    parser.add_argument("--fusion_grid", type=int, default=8)
    parser.add_argument("--fusion_threshold", type=int, default=None)
    parser.add_argument(
        "--query_servers",
        type=str,
        default=None,
        help="Comma-separated server IDs to use for the query (can be < k to exercise fusion fallback).",
    )
    parser.add_argument(
        "--dummy_band_queries",
        type=int,
        default=0,
        help="Number of dummy band-token lookups per band/server to pad access patterns.",
    )
    parser.add_argument(
        "--pad_band_queries",
        type=int,
        default=None,
        help="Optional fixed number of total band lookups (real+dummy) per band/server; overrides dummy_band_queries.",
    )
    parser.add_argument(
        "--fixed_band_queries",
        type=int,
        default=None,
        help="Hard cap on total band lookups (real+dummy) per band/server; takes precedence over pad_band_queries if set.",
    )
    parser.add_argument(
        "--hmac_key_encrypt_env",
        type=str,
        default=None,
        help="Env var name holding AES-GCM key (hex or base64) to encrypt hmac_keys.json on disk.",
    )
    parser.add_argument(
        "--use_oprf",
        action="store_true",
        help="Use simulated OPRF for Stage-A token generation (demonstration; not real blinding).",
    )
    parser.add_argument(
        "--oprf_key_encrypt_env",
        type=str,
        default=None,
        help="Env var name holding AES-GCM key (hex or base64) to encrypt oprf_keys.json on disk.",
    )
    return parser


def ensure_images(path: Path) -> list[Path]:
    path.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in path.iterdir() if p.is_file())
    if not images:
        raise SystemExit(f"No data found in {path}. Add sample data and rerun.")
    return images


def main() -> int:
    args = build_parser().parse_args()
    images = ensure_images(args.images_dir)
    workflow = SearchableSISWithImageStore(
        k=args.k,
        n=args.n,
        bands=args.bands,
        token_len=8,
        seed=args.seed,
        shares_dir=str(args.shares_dir),
        meta_dir=str(args.meta_dir),
        secure_distance=args.mode == "mpc",
        share_strategy=args.share_strategy,
        fusion_grid=args.fusion_grid,
        fusion_threshold=args.fusion_threshold,
        dummy_band_queries=args.dummy_band_queries,
        pad_band_queries=args.pad_band_queries,
        fixed_band_queries=args.fixed_band_queries,
        hmac_key_encrypt_env_var=args.hmac_key_encrypt_env,
        use_oprf=args.use_oprf,
        oprf_key_encrypt_env_var=args.oprf_key_encrypt_env,
    )

    print(f"[MODE] {args.mode}")
    print(f"[CONFIG] k={args.k} n={args.n} bands={args.bands} min_band_votes={args.min_band_votes} "
          f"topk={args.topk} max_hamming={args.max_hamming} reconstruct_top={args.reconstruct_top}")
    print(f"[PATHS] data={args.images_dir} shares={args.shares_dir} meta={args.meta_dir} recon={args.recon_dir}")

    t0 = time.perf_counter()
    for idx, path in enumerate(images):
        image_id = f"img_{idx:04d}"
        phash = workflow.add_image(image_id, str(path))
        print(f"[ADD] {image_id:<12} {path.name:<30} pHash=0x{phash:016x}")
    print(f"[DONE] registered {len(images)} data in {time.perf_counter() - t0:.3f}s")

    query_path = str(args.query) if args.query else str(images[0])
    all_servers = workflow.list_servers()
    servers = _resolve_query_servers(args.query_servers, all_servers, args.k)
    server_note = " (insufficient shares)" if len(servers) < args.k else ""
    print(f"[QUERY] file={Path(query_path).name} servers={servers}{server_note}")
    result = workflow.query_and_optionally_reconstruct(
        query_path,
        servers_for_query=servers,
        min_band_votes=args.min_band_votes,
        topk=args.topk,
        max_hamming=args.max_hamming,
        reconstruct_top=args.reconstruct_top,
        recon_dir=str(args.recon_dir),
        dummy_band_queries=args.dummy_band_queries,
        pad_band_queries=args.pad_band_queries,
        fixed_band_queries=args.fixed_band_queries,
        use_oprf=args.use_oprf,
    )

    print(f"[P-HASH] {result['query_phash']}")
    if result.get("query_fusion_phash"):
        print(f"[FUSION] {result['query_fusion_phash']}")
    print(f"[PRESELECT] {result['preselected'] or '(empty)'}")
    print(f"[RANKED/{result['mode']}/{result.get('share_mode', args.share_strategy)}] {result['ranked'] or '(empty)'}")
    if result["reconstructed"]:
        print("[RECONSTRUCTED]")
        for image_id, path in result["reconstructed"]:
            print(f"  - {image_id} -> {path}")
    else:
        print("[RECONSTRUCTED] (none)")
    if result.get("reconstruction_errors"):
        print("[RECON ERRORS]")
        for err in result["reconstruction_errors"]:
            print(f"  - {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
