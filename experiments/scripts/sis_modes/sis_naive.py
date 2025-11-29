from __future__ import annotations
import time
from typing import List, Tuple
from .base import ModeRunner
from .types import ModeContext, ModeResult, PhaseStats

class NaiveRunner(ModeRunner):
    def __init__(self) -> None:
        super().__init__("sis_server_naive")

    def run_query(self, query_key: str, query_hash: int, ctx: ModeContext) -> ModeResult:
        t2 = time.perf_counter()
        wf = ctx.workflows[self.name]
        idx = wf.index
        servers = ctx.servers
        # 全件評価（重い）
        pool = [s.key for s in ctx.samples]
        ranked: List[Tuple[str, int]] = idx.rank_candidates(
            query_hash, servers_for_query=servers, candidates=pool, topk=len(pool), max_hamming=ctx.args.max_hamming
        )
        f2 = (time.perf_counter() - t2) * 1000.0
        final_ids = [i for i, _ in ranked] + [cid for cid in pool if cid not in {i for i,_ in ranked}]
        bytes_final = len(servers) * 8 * len(ranked)
        stats = PhaseStats(
            f2_ms=f2, n_cand_f1=len(pool), n_cand_f2=len(pool),
            n_eval_final=len(ranked), n_reconstructed=len(ranked),
            bytes_f2_final=bytes_final
        )
        return ModeResult(final_ranking_ids=final_ids, ranked_pairs=ranked, stats=stats)
