"""Orchestrate the mode comparison experiments described in docs/README.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[0]
if str(ROOT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parent))
SRC_ROOT = ROOT_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.modes.plain import PlainModeRunner
from experiments.modes.sis_client_dealer_free import SISClientDealerFreeModeRunner
from experiments.modes.sis_client_partial import SISClientPartialModeRunner
from experiments.modes.sis_mpc import SISMPCModeRunner
from experiments.modes.sis_server_naive import SISServerNaiveModeRunner

MODE_REGISTRY = {
    PlainModeRunner.name: PlainModeRunner,
    SISServerNaiveModeRunner.name: SISServerNaiveModeRunner,
    SISClientDealerFreeModeRunner.name: SISClientDealerFreeModeRunner,
    SISClientPartialModeRunner.name: SISClientPartialModeRunner,
    SISMPCModeRunner.name: SISMPCModeRunner,
}


def _collect_images(images_dir: Path) -> Sequence[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the comparison modes from docs/README.md."
    )
    parser.add_argument(
        "--architecture",
        choices=["dealer_based", "dealer_free"],
        default="dealer_based",
        help="Select the architecture under test.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODE_REGISTRY.keys()),
        default=list(MODE_REGISTRY.keys()),
        help="Modes to execute.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("tests/fixtures"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/experiments"),
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--min-band-votes", type=int, default=3)
    args = parser.parse_args()

    if args.architecture != "dealer_based":
        raise NotImplementedError("Dealer-free experiments are not yet supported here.")
    images = _collect_images(args.images_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for mode_name in args.modes:
        ModeClass = MODE_REGISTRY[mode_name]
        runner = ModeClass(
            images,
            k=args.k,
            n=args.n,
            min_band_votes=args.min_band_votes,
        )
        summary = runner.run()
        report[mode_name] = summary.__dict__
    summary_path = args.output / "mode_comparison.json"
    summary_path.write_text(json.dumps(report, indent=2))
    print(f"Saved mode comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
