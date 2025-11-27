"""
Visualize the three-stage SIS pipeline (Stage-A/B/C) in one figure.

Output: evaluation/figures/stage_pipeline.png (created if missing).

Usage:
    python scripts/plot_stage_pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def add_box(ax, xy, text, color="#2D8CFF", width=3.2, height=1.8):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.2,rounding_size=0.15",
        fc=color,
        ec="#0b335f",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        wrap=True,
    )


def add_arrow(ax, xy_from, xy_to, text=None):
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(arrowstyle="->", color="#333", linewidth=1.2),
    )
    if text:
        ax.text(
            (xy_from[0] + xy_to[0]) / 2,
            (xy_from[1] + xy_to[1]) / 2 + 0.15,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-1.5, 5)

    # Stage boxes
    add_box(ax, (0.5, 2.6), "Stage-A\nBand tokens via VOPRF\n+ fixed batch + dummies\n(Output: candidates)", width=3.4, height=2.0)
    add_box(ax, (4.0, 2.6), "Stage-B\nPartial pHash recon\nfor candidates\n(Output: filtered Top-L)", width=3.4, height=2.0)
    add_box(ax, (7.5, 2.6), "Stage-C\nHamming distance\nMPC / TEE / plain\n(Output: ranked Top-L)", width=3.4, height=2.0)
    add_box(ax, (7.5, 0.4), "Reconstruction\nk-of-n image restore\nTop-K only (access-controlled)", color="#0FB879", width=3.4, height=1.8)

    # Arrows between stages
    add_arrow(ax, (3.7, 3.6), (1.3, 3.6), "Candidates (votes ≥ τ)")
    add_arrow(ax, (7.2, 3.6), (4.2, 3.6), "Top-L")
    add_arrow(ax, (9.2, 1.6), (9.2, 2.5), "Top-K")

    # Inputs/notes
    ax.text(
        0.5,
        4.7,
        "Query → pHash → band split → VOPRF/HMAC tokens (fixed-length, padded batches)",
        fontsize=9,
        color="#111",
    )
    ax.text(
        0.5,
        -0.8,
        "Notes:\n"
        "• Stage-A: VOPRF hides band value; fixed-length batches + dummies soften access-pattern leakage\n"
        "• Stage-B: partial pHash recon only for candidates; skip when MPC/TEE to avoid extra leakage\n"
        "• Stage-C: Hamming distance via MPC/TEE/plain; only Top-L candidates enter\n"
        "• Reconstruction: only Top-K that meet k-of-n; output is access-controlled\n"
        "• Keys: HMAC/OPRF keys can be encrypted at rest; TEE/attestation gates key delivery",
        fontsize=8.5,
        color="#222",
    )

    out_dir = Path("evaluation/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage_pipeline.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved pipeline figure to {out_path}")


if __name__ == "__main__":
    main()
