from __future__ import annotations
import time
from typing import Dict, List, Sequence, Tuple
from pHR_SIS.phash import bytes_to_hash64, hamming64
from pHR_SIS.shamir import shamir_recover_bytes
from pHR_SIS.index import SearchableSISIndex

def compute_plain_distances(query_hash: int, keys: Sequence[str], phashes: Dict[str, int]) -> Dict[str, int]:
    return {k: hamming64(query_hash, phashes[k]) for k in keys}

def stage_b_filter(
    index: SearchableSISIndex,
    query_hash: int,
    servers: Sequence[int],
    candidates: Sequence[str],
    max_hamming: int | None,
    bytes_per_candidate: int = 2,
    margin: int = 8,
) -> Tuple[List[str], float, int]:
    start = time.perf_counter()
    filtered: List[str] = []
    bytes_used = 0
    for image_id in candidates:
        shares = index.get_shares(image_id, servers)
        if shares is None:
            continue
        partial = {s: share[:bytes_per_candidate] for s, share in shares.items()}
        recovered = shamir_recover_bytes(partial)
        approx = bytes_to_hash64(recovered + [0] * (8 - len(recovered)))
        dist = hamming64(query_hash, approx)
        bytes_used += len(partial) * bytes_per_candidate
        if max_hamming is None or dist <= max_hamming + margin:
            filtered.append(image_id)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return filtered, elapsed_ms, bytes_used

def build_ranking_list(
    ranked_subset: List[Tuple[str, int]],
    remaining_candidates: Sequence[str],
    plain_distances: Dict[str, int] | None,
) -> List[str]:
    ranked_ids = [i for i, _ in ranked_subset]
    tail = [cid for cid in remaining_candidates if cid not in ranked_ids]
    if plain_distances is not None:
        # plainのみ距離で尻尾ソート
        tail.sort(key=lambda cid: plain_distances.get(cid, 999))
    return ranked_ids + tail
