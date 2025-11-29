from __future__ import annotations
import time
from typing import List, Tuple
from .base import ModeRunner
from .types import ModeContext, ModeResult, PhaseStats
from .sis_common import stage_b_filter, build_ranking_list

class SelectiveRunner(ModeRunner):
    def __init__(self) -> None:
        super().__init__("sis_selective")

    @staticmethod
    def _stage_a_bytes(idx, args, servers) -> int:
        tokens_per_band = (
            args.fixed_band_queries
            if getattr(args, "fixed_band_queries", None) is not None
            else (args.pad_band_queries if getattr(args, "pad_band_queries", None) is not None else 1 + getattr(args, "dummy_band_queries", 0))
        )
        token_size = idx.token_len + (64 if getattr(idx, "use_oprf", False) else 0)  # 32B blinded + 32B evaluated (rough)
        return len(servers) * idx.bands * tokens_per_band * token_size

    def run_query(self, query_key: str, query_hash: int, ctx: ModeContext) -> ModeResult:
        wf = ctx.workflows[self.name]; idx = wf.index; servers = ctx.servers
        # F1: preselect
        t1 = time.perf_counter()
        cand_a: List[Tuple[str,int]] = idx.preselect_candidates(
            query_hash,
            servers,
            min_band_votes=ctx.args.min_band_votes,
            dummy_band_queries=getattr(ctx.args, "dummy_band_queries", 0),
            pad_band_queries=getattr(ctx.args, "pad_band_queries", None),
            fixed_band_queries=getattr(ctx.args, "fixed_band_queries", None),
        )
        f1 = (time.perf_counter() - t1) * 1000.0
        bytes_f1 = self._stage_a_bytes(idx, ctx.args, servers)
        # F2-early: Stage-B filter
        cand_ids = [c[0] for c in cand_a]
        cand_b, f2_early_ms, bytes_b = stage_b_filter(
            idx, query_hash, servers, cand_ids, ctx.args.max_hamming, ctx.args.stage_b_bytes, ctx.args.stage_b_margin
        )
        # F2-final: rank only topK pool
        pool_size = len(cand_b)
        topk_eval = min(ctx.args.topk, pool_size if pool_size>0 else 0)
        t2 = time.perf_counter()
        ranked: List[Tuple[str,int]] = idx.rank_candidates(
            query_hash, servers_for_query=servers, candidates=cand_b, topk=topk_eval, max_hamming=ctx.args.max_hamming
        )
        f2_final = (time.perf_counter() - t2) * 1000.0
        bytes_final = len(servers) * 8 * len(ranked)
        final_ids = build_ranking_list(ranked_subset=ranked, remaining_candidates=cand_b, plain_distances=None)
        stats = PhaseStats(
            f1_ms=f1, f2_ms=f2_early_ms + f2_final,
            bytes_f1=bytes_f1, bytes_f2_early=bytes_b, bytes_f2_final=bytes_final,
            n_cand_f1=len(cand_a), n_cand_f2=len(cand_b), n_eval_final=len(ranked),
            n_reconstructed=len(ranked)
        )
        return ModeResult(final_ranking_ids=final_ids, ranked_pairs=ranked, stats=stats)
