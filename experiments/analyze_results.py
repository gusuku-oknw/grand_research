"""Summarize experiment outputs stored under output/experiments."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    summary_path = Path("output/experiments/mode_comparison.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    data = json.loads(summary_path.read_text())
    for mode, stats in data.items():
        print(f"{mode}: preselected={stats['preselected']} ranked={stats['ranked']}")


if __name__ == "__main__":
    main()
