# sis_modes/base.py
from __future__ import annotations

"""
Base interfaces and small utilities for modular SIS+pHash experiment modes.

This module intentionally contains **no mode-specific logic**.
It defines:
  - ModeRunner: abstract base class every mode implements
  - PhaseTimer: tiny context manager to time phases in milliseconds
  - ByteMeter: helper to accumulate per-phase byte counts safely
  - Small guard/helpers commonly needed by runners

Design notes
------------
* A "mode" implements one public method:
    run_query(query_key: str, query_hash: int, ctx: ModeContext) -> ModeResult
  and returns:
    - final_ranking_ids: the ranking *within the mode's world* (no plain tail mix-in)
    - ranked_pairs:     the [(id, distance)] actually evaluated in the final scoring step
    - stats:            PhaseStats with F0..F3 timings/bytes/counters

* Phase semantics (shared across all modes):
    F0: Feature       (pHash/secret features)           -> stats.f0_ms
    F1: Screen        (coarse preselect)                -> stats.f1_ms, stats.bytes_f1, stats.n_cand_f1
    F2: Score         (final scoring, may split early/final inside runner) 
                       -> stats.f2_ms, bytes_f2_early/bytes_f2_final, n_cand_f2, n_eval_final
    F3: Materialize   (reconstruction if any)           -> stats.f3_ms, stats.n_reconstructed

* Runners SHOULD:
    - not mutate ctx
    - only account bytes they actually transfer (or estimate consistently)
    - keep "ranked_pairs" to what they truly computed in F2-final
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Callable, Optional

from .types import ModeContext, ModeResult, PhaseStats


class ModeRunnerError(RuntimeError):
    """Errors raised by a specific mode implementation."""
    pass


class PhaseTimer:
    """
    Context manager that measures wall-clock duration (ms).
    Usage:
        with PhaseTimer() as t:
            ... do work ...
        elapsed_ms = t.ms
    """
    __slots__ = ("_start", "ms")

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.ms: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import time
        end = time.perf_counter()
        if self._start is None:
            self.ms = 0.0
        else:
            self.ms = (end - self._start) * 1000.0


class ByteMeter:
    """
    Minimal helper to track bytes per sub-phase safely.

    Example:
        bm = ByteMeter()
        bm.add_f1(tokens * servers)
        # ... later ...
        used = bm.f1  # int
    """
    __slots__ = ("f1", "f2_early", "f2_final", "f3")

    def __init__(self) -> None:
        self.f1: int = 0
        self.f2_early: int = 0
        self.f2_final: int = 0
        self.f3: int = 0

    def add_f1(self, v: int) -> None:
        self.f1 += max(int(v), 0)

    def add_f2_early(self, v: int) -> None:
        self.f2_early += max(int(v), 0)

    def add_f2_final(self, v: int) -> None:
        self.f2_final += max(int(v), 0)

    def add_f3(self, v: int) -> None:
        self.f3 += max(int(v), 0)


class ModeRunner(ABC):
    """
    Abstract base for all SIS+pHash experiment modes.

    Subclasses MUST implement:
        - name (str)
        - run_query(query_key, query_hash, ctx) -> ModeResult

    Optional hooks you MAY override:
        - preflight(ctx): validate dependent resources before first use
        - describe() -> str: one-line human-readable description
    """

    name: str

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("ModeRunner requires a non-empty name.")
        self.name = name
        self._preflight_done: bool = False

    # -------- Lifecycle hooks --------

    def preflight(self, ctx: ModeContext) -> None:
        """
        Validate mode prerequisites. Called (idempotently) before first query.
        Default checks:
          - plain mode: none
          - non-plain modes: workflow presence in ctx.workflows
        Override to add stricter guarantees (e.g., MPC backend reachability).
        """
        if self.name != "plain":
            if self.name not in ctx.workflows:
                raise ModeRunnerError(f"workflow for mode='{self.name}' not provided in ctx.workflows")
        self._preflight_done = True

    def _ensure_preflight(self, ctx: ModeContext) -> None:
        if not self._preflight_done:
            self.preflight(ctx)

    def describe(self) -> str:
        return self.name

    # -------- Public API --------

    @abstractmethod
    def run_query(self, query_key: str, query_hash: int, ctx: ModeContext) -> ModeResult:
        """
        Execute the mode for a single query.

        MUST:
          - call self._ensure_preflight(ctx) at the beginning (or rely on orchestrator calling preflight once)
          - return ModeResult with:
              final_ranking_ids : List[str]
              ranked_pairs      : List[Tuple[id, distance]]
              stats             : PhaseStats (F0..F3 filled as applicable)
        """
        raise NotImplementedError

    # -------- Convenience helpers for subclasses --------

    def new_stats(self) -> PhaseStats:
        """Return a zero-initialized PhaseStats."""
        return PhaseStats()

    def with_f0(self, stats: PhaseStats, f0_ms: float) -> PhaseStats:
        """Return a copy of stats with f0_ms set (non-destructive)."""
        return replace(stats, f0_ms=max(float(f0_ms), 0.0))

    @staticmethod
    def measure_ms(fn: Callable[[], None]) -> float:
        """Execute callable and return elapsed time in milliseconds."""
        with PhaseTimer() as t:
            fn()
        return t.ms

    @staticmethod
    def safe_topk(k: int, size: int) -> int:
        """Clamp top-k into [0, size]."""
        if k <= 0:
            return 0
        return min(k, max(size, 0))

    @staticmethod
    def ensure_nonempty(sequence, what: str = "sequence") -> None:
        if not sequence:
            raise ModeRunnerError(f"{what} is empty")

    @staticmethod
    def ensure_key_in(mapping, key, what: str = "mapping") -> None:
        if key not in mapping:
            raise ModeRunnerError(f"key '{key}' not found in {what}")
