"""Wrapper for exercising SIS search failure scenarios via the packaged CLI."""

from __future__ import annotations

import argparse

try:
    from .cli import run_secure_demo  # type: ignore
except ImportError:  # pragma: no cover
    from pHR_SIS.cli import run_secure_demo  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect resilience of the SIS search index under failure scenarios."
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--token_len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_hamming", type=int, default=10)
    parser.add_argument("--images_dir", type=str, default="images")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_secure_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
