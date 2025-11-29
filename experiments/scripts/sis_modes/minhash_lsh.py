from __future__ import annotations

import time
from typing import Dict, List, Tuple

from datasketch import MinHash, MinHashLSH
from sis_image.common.phash import hamming64

from .base import ModeRunner
from .types import ModeContext, ModeResult, PhaseStats
from .sis_common import build_ranking_list


class MinHashLSHRunner(ModeRunner):
    name = "minhash_lsh"

    def __init__(self, num_perm: int = 128, threshold: float = 0.5) -> None:
        super().__init__(self.name)
        self.num_perm = num_perm
        self.threshold = threshold
        self._lsh: MinHashLSH | None = None
        self._mh_cache: Dict[str, MinHash] | None = None
        self._prepared = False

    @staticmethod
    def _phash_to_minhash(phash: int, num_perm: int) -> MinHash:
        mh = MinHash(num_perm=num_perm)
        for bit_idx in range(64):
            if (phash >> bit_idx) & 1:
                mh.update(f"{bit_idx}".encode("ascii"))
        return mh

    def preflight(self, ctx: ModeContext) -> None:
        if self._prepared:
            return
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        mh_cache: Dict[str, MinHash] = {}
        for sample in ctx.samples:
            mh = self._phash_to_minhash(ctx.phashes[sample.key], self.num_perm)
            mh_cache[sample.key] = mh
            lsh.insert(sample.key, mh)
        self._lsh = lsh
        self._mh_cache = mh_cache
        self._prepared = True
        self._preflight_done = True

    def run_query(self, query_key: str, query_hash: int, ctx: ModeContext) -> ModeResult:
        self._ensure_preflight(ctx)
        assert self._lsh is not None and self._mh_cache is not None
        query_mh = self._phash_to_minhash(query_hash, self.num_perm)
        t1 = time.perf_counter()
        candidates = list(self._lsh.query(query_mh))
        stage1_ms = (time.perf_counter() - t1) * 1000.0
        cand_set = candidates or list(ctx.phashes.keys())
        stage1_bytes = len(cand_set) * 16
        # Stage-2: full Hamming evaluation on candidate list
        t2 = time.perf_counter()
        ranked_pairs: List[Tuple[str, int]] = []
        for image_id in cand_set:
            dist = hamming64(query_hash, ctx.phashes[image_id])
            ranked_pairs.append((image_id, dist))
        ranked_pairs.sort(key=lambda pair: pair[1])
        stage2_ms = (time.perf_counter() - t2) * 1000.0
        stats = PhaseStats(
            f1_ms=stage1_ms,
            f2_ms=stage2_ms,
            bytes_f1=stage1_bytes,
            bytes_f2_final=len(cand_set) * 8,
            n_cand_f1=len(cand_set),
            n_cand_f2=len(cand_set),
            n_eval_final=len(ranked_pairs),
            n_reconstructed=len(ranked_pairs),
        )
        final_ids = build_ranking_list(
            ranked_subset=ranked_pairs,
            remaining_candidates=[cid for cid, _ in ranked_pairs],
            plain_distances=None,
        )
        return ModeResult(final_ranking_ids=final_ids, ranked_pairs=ranked_pairs, stats=stats)
