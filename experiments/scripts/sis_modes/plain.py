from __future__ import annotations
import time
from typing import List, Tuple
from .base import ModeRunner
from .types import ModeContext, ModeResult, PhaseStats
from .sis_common import compute_plain_distances, build_ranking_list

class PlainRunner(ModeRunner):
    def __init__(self) -> None:
        super().__init__("plain")

    def run_query(self, query_key: str, query_hash: int, ctx: ModeContext) -> ModeResult:
        t0 = time.perf_counter()
        keys = [s.key for s in ctx.samples]
        dists = compute_plain_distances(query_hash, keys, ctx.phashes)
        sorted_items: List[Tuple[str, int]] = sorted(dists.items(), key=lambda kv: (kv[1], kv[0]))
        final_ids = [k for k, _ in sorted_items]
        f3 = (time.perf_counter() - t0) * 1000.0
        stats = PhaseStats(
            f0_ms=0.0, f1_ms=0.0, f2_ms=0.0, f3_ms=f3,
            n_cand_f1=len(final_ids), n_cand_f2=len(final_ids),
            n_eval_final=len(final_ids), n_reconstructed=len(final_ids),
        )
        return ModeResult(final_ranking_ids=final_ids, ranked_pairs=sorted_items, stats=stats)
