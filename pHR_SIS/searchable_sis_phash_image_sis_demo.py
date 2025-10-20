"""Wrapper script for the combined searchable SIS + image reconstruction demo."""

from __future__ import annotations

import argparse

try:
    from .cli import run_workflow_demo  # type: ignore
except ImportError:  # pragma: no cover
    from pHR_SIS.cli import run_workflow_demo  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Searchable SIS demo with optional reconstruction of top matches."
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--min_band_votes", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_hamming", type=int, default=10)
    parser.add_argument("--reconstruct_top", type=int, default=1)
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--shares_dir", type=str, default="img_shares")
    parser.add_argument("--meta_dir", type=str, default="img_meta")
    parser.add_argument("--recon_dir", type=str, default="recon_out")
    parser.add_argument("--query", type=str, default=None)
    parser.set_defaults(token_len=8, seed=2025)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_workflow_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
