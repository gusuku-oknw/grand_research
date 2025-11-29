"""Simulated FHE latency demo (bitwise + busy loop)."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Iterable, List


def phash_to_bits(phash: int, bits: int = 64) -> List[int]:
    return [(phash >> i) & 1 for i in range(bits)]


def simulate_fhe_distance(bits_a: Iterable[int], bits_b: Iterable[int]) -> int:
    total = 0
    for a, b in zip(bits_a, bits_b):
        total += a ^ b
        # Busy loop to exaggerate latency (simulate FHE cost)
        for _ in range(20000):
            pass
    return total


def write_metrics_csv(path: Path, avg_ms: float, trials: int) -> None:
    import csv

    fieldnames = ["mode", "dataset", "stage1_ms", "stage2_ms", "total_ms", "stage1_bytes", "stage2_bytes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "mode": "sis_fhe",
                "dataset": "fhe_demo",
                "stage1_ms": 0.0,
                "stage2_ms": avg_ms,
                "total_ms": avg_ms,
                "stage1_bytes": 0,
                "stage2_bytes": trials * 8,
            }
        )


def run_demo(iterations: int, seed: int, output_csv: Path | None = None) -> None:
    rng = random.Random(seed)
    results: List[float] = []
    for i in range(1, iterations + 1):
        p1 = rng.getrandbits(64)
        p2 = rng.getrandbits(64)
        bits1 = phash_to_bits(p1)
        bits2 = phash_to_bits(p2)
        start = time.perf_counter()
        fhe_distance = simulate_fhe_distance(bits1, bits2)
        elapsed = (time.perf_counter() - start) * 1000.0
        results.append(elapsed)
        plain_distance = sum(a != b for a, b in zip(bits1, bits2))
        print(f"Trial {i}: plain={plain_distance}, fhe={fhe_distance}, time={elapsed:.1f}ms")
    avg = sum(results) / len(results)
    print(f"\nAverage simulated FHE latency: {avg:.1f} ms")
    if output_csv:
        write_metrics_csv(output_csv, avg, iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated FHE latency demo.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()
    csv_path = Path(args.output_csv) if args.output_csv else None
    run_demo(iterations=args.iterations, seed=args.seed, output_csv=csv_path)


if __name__ == "__main__":
    main()
