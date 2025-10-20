"""Wrapper around pHR_SIS to demonstrate full-image secret sharing."""

from __future__ import annotations

import argparse

try:
    from .cli import run_image_store_demo  # type: ignore
except ImportError:  # pragma: no cover
    from pHR_SIS.cli import run_image_store_demo  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shamir (k, n) secret sharing demo for full RGB images."
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--shares_dir", type=str, default="img_shares")
    parser.add_argument("--meta_dir", type=str, default="img_meta")
    parser.add_argument("--recon_dir", type=str, default="recon_out")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_image_store_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
