"""Wrapper exposing the dealer-free helpers under the legacy package name."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sis_image.dealer_free import (
    DealerFreeSimulator,
    compare_secure_vs_baseline,
    run_cli,
    run_dealer_free_experiment,
    SecureDistanceComparison,
)

__all__ = [
    "DealerFreeSimulator",
    "compare_secure_vs_baseline",
    "run_cli",
    "run_dealer_free_experiment",
    "SecureDistanceComparison",
]
