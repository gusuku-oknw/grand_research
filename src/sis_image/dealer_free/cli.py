"""CLI entry point for dealer-free SIS experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_dealer_free_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Dealer-free SIS experiment runner.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("tests/fixtures"),
        help="Directory that contains sample images.",
    )
    parser.add_argument("--k", type=int, default=3, help="Shamir threshold k.")
    parser.add_argument("--n", type=int, default=5, help="Number of SIS servers.")
    parser.add_argument("--bands", type=int, default=8, help="Number of bands for tokens.")
    parser.add_argument("--token-len", type=int, default=8, help="Length of each band token.")
    parser.add_argument(
        "--contributors",
        type=int,
        default=3,
        help="Number of parties contributing randomness in DKG.",
    )
    parser.add_argument(
        "--padding-tokens",
        type=int,
        default=4,
        help="Extra dummy tokens per query for padding.",
    )
    parser.add_argument(
        "--use-oprf",
        action="store_true",
        help="Activate VOPRF tokens during experiments.",
    )
    parser.add_argument(
        "--mpc-query-image",
        type=Path,
        help="Optional image path to use for secure-distance comparison.",
    )
    parser.add_argument(
        "--mpc-servers",
        type=int,
        nargs="+",
        help="Server IDs to use for secure-distance comparisons (defaults to 1..k).",
    )
    parser.add_argument(
        "--mpc-topk",
        type=int,
        default=5,
        help="Number of candidates to compare in MPC analysis.",
    )
    parser.add_argument(
        "--mpc-min-band-votes",
        type=int,
        default=3,
        help="Minimum band votes required for preselection during MPC analysis.",
    )
    parser.add_argument(
        "--mpc-max-hamming",
        type=int,
        help="Maximum Hamming distance filter used during MPC analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figures/dealer_free"),
        help="Directory where metrics and plots will be stored.",
    )
    args = parser.parse_args()

    metrics_path, plot_path, mpc_path = run_dealer_free_experiment(
        images_dir=args.images_dir,
        output=args.output,
        k=args.k,
        n=args.n,
        bands=args.bands,
        token_len=args.token_len,
        contributors=args.contributors,
        padding_tokens=args.padding_tokens,
        use_oprf=args.use_oprf,
        mpc_query_image=args.mpc_query_image,
        mpc_servers=args.mpc_servers,
        mpc_topk=args.mpc_topk,
        mpc_min_band_votes=args.mpc_min_band_votes,
        mpc_max_hamming=args.mpc_max_hamming,
    )
    print(f"Saved metrics to {metrics_path} and figure to {plot_path}")
    if mpc_path:
        print(f"Saved secure-distance comparison to {mpc_path}")
