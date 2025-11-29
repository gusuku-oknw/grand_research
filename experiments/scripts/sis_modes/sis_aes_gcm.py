from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from sis_image.common.phash import hamming64

from .base import ModeRunner, ModeRunnerError
from .aes_crypto import AESGCMStorage, EncryptedImageRecord, compute_phash_from_bytes
from .types import ModeContext, ModeResult, PhaseStats
from .sis_common import build_ranking_list


class AESGCMRunner(ModeRunner):
    name = "aes_gcm"

    def __init__(self) -> None:
        super().__init__(self.name)

    def preflight(self, ctx: ModeContext) -> None:
        if ctx.aes_storage is None or not ctx.aes_records:
            raise ModeRunnerError("AES-GCM mode requires encrypted image records in context.")
        self._preflight_done = True

    def run_query(self, query_key: str, query_hash: int, ctx: ModeContext) -> ModeResult:
        self._ensure_preflight(ctx)
        records = ctx.aes_records  # type: ignore[assignment]
        storage = ctx.aes_storage  # type: ignore[assignment]
        stage2_candidates: List[Tuple[str, int]] = []
        bytes_transferred = 0
        t0 = time.perf_counter()
        for sample in ctx.samples:
            rec = records[sample.key]
            raw = storage.decrypt_image(rec)
            ph = compute_phash_from_bytes(raw)
            dist = hamming64(query_hash, ph)
            stage2_candidates.append((sample.key, dist))
            bytes_transferred += len(rec.ciphertext) + len(rec.nonce)
        f2 = (time.perf_counter() - t0) * 1000.0
        ranked = sorted(stage2_candidates, key=lambda item: item[1])
        final_ids = build_ranking_list(
            ranked_subset=ranked,
            remaining_candidates=[key for key, _ in stage2_candidates],
            plain_distances=None,
        )
        stats = PhaseStats(
            f1_ms=0.0,
            f2_ms=f2,
            bytes_f1=0,
            bytes_f2_final=bytes_transferred,
            n_cand_f1=0,
            n_cand_f2=len(stage2_candidates),
            n_eval_final=len(ranked),
            n_reconstructed=len(ranked),
        )
        return ModeResult(final_ranking_ids=final_ids, ranked_pairs=ranked, stats=stats)
