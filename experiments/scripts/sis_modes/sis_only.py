from __future__ import annotations
import time
from typing import List, Tuple
from .base import ModeRunner
from .types import ModeContext, ModeResult, PhaseStats
from .sis_common import build_ranking_list


class SISOnlyRunner(ModeRunner):
    name = "sis_only"

    def __init__(self) -> None:
        super().__init__(self.name)

    def run_query(self, query_key: str, query_hash: int, ctx: ModeContext) -> ModeResult:
        wf = ctx.workflows[self.name]
        idx = wf.index
        servers = ctx.servers
        pool = [s.key for s in ctx.samples]

        # F2: evaluate all candidates without any Stage-1 filtering
        t2 = time.perf_counter()
        ranked: List[Tuple[str, int]] = idx.rank_candidates(
            query_hash,
            servers_for_query=servers,
            candidates=pool,
            topk=len(pool),
            max_hamming=ctx.args.max_hamming,
        )
        f2 = (time.perf_counter() - t2) * 1000.0
        final_ids = build_ranking_list(ranked_subset=ranked, remaining_candidates=pool, plain_distances=None)
        bytes_final = len(servers) * 8 * len(ranked)

        stats = PhaseStats(
            f1_ms=0.0,
            f2_ms=f2,
            bytes_f1=0,
            bytes_f2_final=bytes_final,
            n_cand_f1=0,
            n_cand_f2=len(pool),
            n_eval_final=len(ranked),
            n_reconstructed=len(ranked),
        )
        return ModeResult(final_ranking_ids=final_ids, ranked_pairs=ranked, stats=stats)
