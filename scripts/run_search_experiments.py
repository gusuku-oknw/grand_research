"""Run SIS+pHash search experiments on COCO derivatives (patched)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from evaluation.dataset import build_positive_lookup, build_samples, load_derivative_mapping, Sample
from evaluation.metrics import (
    RankingMetrics,
    average_precision_from_ranking,
    histogram,
    precision_recall_at_k,
    roc_pr_curves,
    area_under_curve,
)
from pHR_SIS.index import SearchableSISIndex
from pHR_SIS.phash import phash64, hash64_to_bytes, bytes_to_hash64, hamming64
from pHR_SIS.shamir import shamir_recover_bytes
from pHR_SIS.workflow import SearchableSISWithImageStore


@dataclass
class QueryLog:
    dataset: str
    mode: str
    query_key: str
    query_variant: str
    transform: str
    tau: int
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    map: float
    phash_ms: float
    stage_a_ms: float   # F1: Screen
    stage_b_ms: float   # F2: Score (前処理側)
    stage_c_ms: float   # F2: Score (最終評価側)
    total_ms: float
    n_dataset: int
    n_candidates_a: int
    n_candidates_b: int
    n_candidates_c: int
    n_reconstructed: int
    bytes_a: int
    bytes_b: int
    bytes_c: int


def _load_phash_cache(cache_path: Path) -> Dict[str, int]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {key: int(value, 16) for key, value in data.items()}


def _save_phash_cache(cache_path: Path, phashes: Dict[str, int]) -> None:
    serialisable = {key: f"{value:016x}" for key, value in phashes.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f)


def _compute_phash(sample: Sample) -> Tuple[str, int, float]:
    start = time.perf_counter()
    value = phash64(str(sample.path))
    elapsed = time.perf_counter() - start
    return sample.key, value, elapsed


def compute_phashes(
    samples: List[Sample],
    cache_path: Path,
    *,
    force_recompute: bool = False,
    progress_interval: int = 1000,
) -> Tuple[Dict[str, int], float]:
    total = len(samples)
    print(f"[pHash] Preparing hashes for {total} samples.")
    cached: Dict[str, int] = {}
    if cache_path.exists() and not force_recompute:
        try:
            cached = _load_phash_cache(cache_path)
            print(f"[pHash] Loaded cache with {len(cached)} entries from {cache_path}.")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[pHash] Failed to load cache ({exc}); recomputing all hashes.")
            cached = {}
    phashes = {sample.key: cached[sample.key] for sample in samples if sample.key in cached}
    missing_samples = [sample for sample in samples if sample.key not in phashes]
    total_missing = len(missing_samples)
    if total_missing == 0:
        print("[pHash] Cache covers all samples; skipping recomputation.")
        return phashes, 0.0

    print(f"[pHash] Computing {total_missing} missing hashes (using up to {max(1, (os.cpu_count() or 1))} threads).")
    start_all = time.perf_counter()
    elapsed_sum = 0.0
    processed = 0
    workers = max(1, (os.cpu_count() or 1))
    interval = max(1, progress_interval)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_sample = {executor.submit(_compute_phash, sample): sample for sample in missing_samples}
        for future in as_completed(future_to_sample):
            key, value, elapsed = future.result()
            phashes[key] = value
            elapsed_sum += elapsed
            processed += 1
            if processed % interval == 0 or processed == total_missing:
                percent = processed / total_missing * 100.0
                total_elapsed = time.perf_counter() - start_all
                print(f"[pHash] {processed}/{total_missing} computed ({percent:.1f}%) in {total_elapsed:.1f}s.")

    if cache_path:
        _save_phash_cache(cache_path, phashes)
        print(f"[pHash] Cache written with {len(phashes)} entries to {cache_path}.")

    avg_ms = (elapsed_sum / max(len(samples), 1)) * 1000.0
    return phashes, avg_ms


def stage_b_filter(
    index: SearchableSISIndex,
    query_hash: int,
    servers: Sequence[int],
    candidates: Sequence[str],
    max_hamming: int | None,
    bytes_per_candidate: int = 2,
    margin: int = 8,
) -> Tuple[List[str], float, int]:
    # 粗選別後の「近い候補」を部分復元バイトのみでさらに絞る
    start = time.perf_counter()
    filtered: List[str] = []
    bytes_used = 0
    for image_id in candidates:
        shares = index.get_shares(image_id, servers)
        if shares is None:
            continue
        partial_shares = {s: share[:bytes_per_candidate] for s, share in shares.items()}
        recovered = shamir_recover_bytes(partial_shares)
        partial_bytes = recovered + [0] * (8 - len(recovered))
        approx_hash = bytes_to_hash64(partial_bytes)
        dist = hamming64(query_hash, approx_hash)
        bytes_used += len(partial_shares) * bytes_per_candidate
        if max_hamming is None or dist <= max_hamming + margin:
            filtered.append(image_id)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return filtered, elapsed_ms, bytes_used


def build_ranking_list(
    ranked_subset: List[Tuple[str, int]],
    remaining_candidates: Sequence[str],
    plain_distances: Dict[str, int],
) -> List[str]:
    """
    ranked_subset: [(id, distance)] ・・・モードが実際に評価した上位群
    remaining_candidates: そのモードが「評価対象」とみなすIDプール（非評価分）
    plain_distances: plainのみ尻尾のソートに使用。非plainでは使わず順不同のまま後段に連結。
    """
    ranked_ids = [image_id for image_id, _ in ranked_subset]
    remaining = [cid for cid in remaining_candidates if cid not in ranked_ids]
    if plain_distances:  # plainのみ尻尾を距離順ソート
        remaining.sort(key=lambda cid: plain_distances.get(cid, 999))
    return ranked_ids + remaining


def compute_precision_metrics(
    ranking: List[str],
    id_to_index: Dict[str, int],
    positives: Sequence[int],
) -> RankingMetrics:
    ranked_indices = [id_to_index[rid] for rid in ranking]
    return precision_recall_at_k(ranked_indices, positives, k_values=[1, 5, 10])


def compute_map(ranking: List[str], id_to_index: Dict[str, int], positives: Sequence[int]) -> float:
    ranked_indices = [id_to_index[rid] for rid in ranking]
    return average_precision_from_ranking(ranked_indices, set(positives))


def compute_plain_distances(
    query_phash: int,
    all_samples: List[Sample],
    phashes: Dict[str, int],
) -> Dict[str, int]:
    distances: Dict[str, int] = {}
    for sample in all_samples:
        distances[sample.key] = hamming64(query_phash, phashes[sample.key])
    return distances


def ensure_workflow(
    mode: str,
    base_dir: Path,
    k: int,
    n: int,
    bands: int,
    secure: bool,
    seed: int,
    *,
    shares_dir: Path,
    meta_dir: Path,
) -> SearchableSISWithImageStore:
    shares_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    workflow = SearchableSISWithImageStore(
        k=k,
        n=n,
        bands=bands,
        token_len=8,
        seed=seed,
        shares_dir=str(shares_dir),
        meta_dir=str(meta_dir),
        secure_distance=secure,
    )
    return workflow


def aggregate_distance_histograms(
    mapping: Dict[str, Dict[str, str]],
    phashes: Dict[str, int],
) -> Dict[str, List[int]]:
    hist: Dict[str, List[int]] = defaultdict(list)
    for image_id, variants in mapping.items():
        if "original" not in variants:
            continue
        orig_key = f"{image_id}__original"
        if orig_key not in phashes:
            continue
        orig_hash = phashes[orig_key]
        for variant, _ in variants.items():
            if variant == "original":
                continue
            key = f"{image_id}__{variant}"
            if key not in phashes:
                continue
            dist = hamming64(orig_hash, phashes[key])
            hist[variant].append(dist)
    return hist


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SIS+pHash experiments and collect metrics (patched).")
    parser.add_argument("--mapping_json", type=Path, default=Path("data/coco/derivative_mapping.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    parser.add_argument("--work_dir", type=Path, default=Path("eval_artifacts"))
    parser.add_argument("--dataset_name", type=str, default="coco_val2017")
    parser.add_argument("--max_queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--min_band_votes", type=int, default=3)
    parser.add_argument("--max_hamming", type=int, default=10)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--tau_values", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--stage_b_bytes", type=int, default=2)
    parser.add_argument("--stage_b_margin", type=int, default=8)
    parser.add_argument("--sis_modes", type=str, nargs="+", default=["plain", "sis_naive", "sis_selective", "sis_staged", "sis_mpc"])
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if metrics already exist.")
    args = parser.parse_args()

    tau_values = sorted(set(args.tau_values)) or [args.max_hamming or 10]
    base_tau = tau_values[0]

    output_dir = args.output_dir
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists() and not args.force:
        print(f"Metrics already exist at {metrics_path}. Use --force to regenerate.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_derivative_mapping(args.mapping_json)
    samples = build_samples(mapping, include_variants=None)
    if args.max_queries and args.max_queries < len(samples):
        rng = np.random.default_rng(args.seed)
        query_indices = rng.choice(len(samples), size=args.max_queries, replace=False)
        query_samples = [samples[i] for i in query_indices]
    else:
        query_samples = samples

    positives_lookup = build_positive_lookup(samples)
    id_to_index = {sample.key: idx for idx, sample in enumerate(samples)}

    phash_cache_path = work_dir / "phash_cache.json"
    phashes, phash_ms_avg = compute_phashes(
        samples,
        phash_cache_path,
        force_recompute=args.force,
    )

    workflows: Dict[str, SearchableSISWithImageStore] = {}
    modes = args.modes
    shared_store_dir = work_dir / "shared_store"
    shared_shares_dir = shared_store_dir / "img_shares"
    shared_meta_dir = shared_store_dir / "img_meta"
    index_progress_interval = max(1, len(samples) // 20)
    for mode in modes:
        if mode == "plain":
            continue
        secure = mode == "sis_mpc"
        workflows[mode] = ensure_workflow(
            mode=mode,
            base_dir=work_dir / mode,
            k=args.k,
            n=args.n,
            bands=args.bands,
            secure=secure,
            seed=args.seed,
            shares_dir=shared_shares_dir,
            meta_dir=shared_meta_dir,
        )
        start_mode = time.perf_counter()
        print(f"[Index:{mode}] Adding {len(samples)} samples to index/store.")
        for idx, sample in enumerate(samples, 1):
            phash_value = phashes.get(sample.key)
            if phash_value is None:
                phash_value = phash64(str(sample.path))
                phashes[sample.key] = phash_value
            workflows[mode].add_image(sample.key, str(sample.path), phash=phash_value)
            if idx % index_progress_interval == 0 or idx == len(samples):
                elapsed = time.perf_counter() - start_mode
                print(
                    f"[Index:{mode}] {idx}/{len(samples)} ({idx / len(samples) * 100:.1f}%) processed in {elapsed:.1f}s."
                )

    plain_index = SearchableSISIndex(k=args.k, n=args.n, bands=args.bands, token_len=8, seed=args.seed)
    print(f"[Index:plain] Adding {len(samples)} samples to plain index.")
    start_plain = time.perf_counter()
    for idx, sample in enumerate(samples, 1):
        phash_value = phashes.get(sample.key)
        if phash_value is None:
            phash_value = phash64(str(sample.path))
            phashes[sample.key] = phash_value
        plain_index.add_image_with_phash(sample.key, str(sample.path), phash_value)
        if idx % index_progress_interval == 0 or idx == len(samples):
            elapsed = time.perf_counter() - start_plain
            print(
                f"[Index:plain] {idx}/{len(samples)} ({idx / len(samples) * 100:.1f}%) processed in {elapsed:.1f}s."
            )

    logs: List[QueryLog] = []
    roc_data: Dict[str, List[Tuple[List[float], List[float], List[float], List[float]]]] = defaultdict(list)
    distance_hist = aggregate_distance_histograms(mapping, phashes)
    candidate_reduction: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    time_stack: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)  # F0/F1/F2/F3
    byte_stack: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)  # F1/F2/F3 のみ
    variant_metrics_raw: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    reconstruction_ratios: Dict[str, List[float]] = defaultdict(list)
    stage_ratio: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    for query in query_samples:
        positives = positives_lookup[query.image_id]
        query_hash = phashes[query.key]
        plain_distances = compute_plain_distances(query_hash, samples, phashes)
        modes_to_run = modes
        servers = plain_index.list_servers()[: args.k]
        for mode in modes_to_run:
            stage_a_ms = stage_b_ms = stage_c_ms = 0.0
            bytes_a = bytes_b = bytes_c = 0
            candidates_a: List[Tuple[str, int]] = []
            candidates_b: List[str] = []
            final_ranking_ids: List[str] = []
            n_reconstructed = 0
            start_total = time.perf_counter()

            if mode == "plain":
                # F0: Feature (pHash)は事前計算の平均を使用
                # F1/F2/F3は定義上ほぼ0。ここではF3に合算（結果出力）
                sorted_items = sorted(plain_distances.items(), key=lambda kv: (kv[1], kv[0]))
                final_ranking_ids = [item[0] for item in sorted_items]
                n_dataset = len(sorted_items)
                stage_a_ms = stage_b_ms = 0.0
                stage_c_ms = (time.perf_counter() - start_total) * 1000.0
                bytes_a = bytes_b = bytes_c = 0
                n_candidates_a = n_candidates_b = n_candidates_c = len(sorted_items)
                n_reconstructed = len(sorted_items)  # baseline=1.0
            else:
                workflow = workflows[mode]
                index = workflow.index

                # ==== F1: Screen ====
                if mode == "sis_naive":
                    stage_a_ms = 0.0
                    bytes_a = 0
                    candidates_a = [(sample.key, 0) for sample in samples]
                    n_candidates_a = len(candidates_a)
                    candidates_b = [sample.key for sample in samples]
                    n_candidates_b = len(candidates_b)
                    stage_b_ms = 0.0
                    bytes_b = 0
                else:
                    stage_a_start = time.perf_counter()
                    candidates_a = index.preselect_candidates(query_hash, servers, min_band_votes=args.min_band_votes)
                    stage_a_ms = (time.perf_counter() - stage_a_start) * 1000.0
                    n_candidates_a = len(candidates_a)
                    # トークン送信の概算：サーバ×バンド×トークン長
                    bytes_a = len(servers) * index.bands * index.token_len

                    candidate_ids_a = [c[0] for c in candidates_a]
                    stage_b_candidates, stage_b_ms, bytes_b = stage_b_filter(
                        index=index,
                        query_hash=query_hash,
                        servers=servers,
                        candidates=candidate_ids_a,
                        max_hamming=args.max_hamming,
                        bytes_per_candidate=args.stage_b_bytes,
                        margin=args.stage_b_margin,
                    )
                    candidates_b = stage_b_candidates
                    n_candidates_b = len(candidates_b)

                # ==== F2: Score ====
                stage_c_start = time.perf_counter()
                # 評価対象トップK（staged/selectiveは限定、naiveは全件、mpcは候補全件を評価してもよいが計算量を考慮）
                if mode in ("sis_selective", "sis_staged"):
                    pool_size = len(candidates_b if mode != "sis_naive" else samples)
                    topk_eval = min(args.topk, pool_size)
                elif mode == "sis_naive":
                    topk_eval = len(samples)
                else:  # sis_mpc
                    topk_eval = len(candidates_b)

                eval_pool = (candidates_b if mode != "sis_naive" else [s.key for s in samples])

                if mode == "sis_mpc":
                    ranked = index.rank_candidates_secure(
                        query_hash,
                        servers_for_query=servers,
                        candidates=eval_pool,
                        topk=topk_eval,
                        max_hamming=args.max_hamming,
                    )
                else:
                    ranked = index.rank_candidates(
                        query_hash,
                        servers_for_query=servers,
                        candidates=eval_pool,
                        topk=topk_eval,
                        max_hamming=args.max_hamming,
                    )
                stage_c_ms = (time.perf_counter() - stage_c_start) * 1000.0

                # ==== ランキングをモードの世界観で構成 ====
                # 非plainは尻尾をplain距離でソートしない（未評価は順不同で後ろへ）
                tail_pool = eval_pool
                final_ranking_ids = build_ranking_list(
                    ranked_subset=ranked,
                    remaining_candidates=tail_pool,
                    plain_distances=(plain_distances if mode == "plain" else {}),
                )
                n_candidates_c = len(ranked)
                n_dataset = len(samples)

                # ==== 通信・復元のカウント ====
                if mode == "sis_naive":
                    n_candidates_a = n_dataset
                    n_candidates_b = n_dataset

                if mode == "sis_mpc":
                    bytes_c = 0  # 秘密演算は別カウンタを設けるまで0扱い
                    n_reconstructed = 0  # 復元しない
                else:
                    bytes_c = len(servers) * 8 * len(ranked)
                    n_reconstructed = len(ranked)  # 実際に評価・復元した件数（topk_eval相当）

            total_ms = (time.perf_counter() - start_total) * 1000.0
            metrics = compute_precision_metrics(final_ranking_ids, id_to_index, positives)
            map_value = compute_map(final_ranking_ids, id_to_index, positives)

            # ==== ROC/PR 用：モード固有の距離ベクトル ====
            # plain: 平文距離。非plain: 評価済みはStage-Cの距離、未評価は最大+1の距離として扱う
            if mode == "plain":
                mode_distances = [plain_distances[s.key] for s in samples]
            else:
                ranked_dists = dict(ranked)  # id -> distance
                mode_distances = [ranked_dists.get(s.key, 65) for s in samples]  # 未評価=65(>64)

            labels_vec = [1 if id_to_index[s.key] in positives else 0 for s in samples]
            fpr, tpr, prec, rec = roc_pr_curves(
                distances=mode_distances,
                labels=labels_vec,
                tau_values=tau_values,
            )
            roc_data[mode].append((fpr, tpr, prec, rec))

            # ==== ログ出力 ====
            for tau in tau_values:
                logs.append(
                    QueryLog(
                        dataset=args.dataset_name,
                        mode=mode,
                        query_key=query.key,
                        query_variant=query.variant,
                        transform=query.variant,
                        tau=tau,
                        precision_at_1=metrics.precision_at.get(1, 0.0),
                        precision_at_5=metrics.precision_at.get(5, 0.0),
                        precision_at_10=metrics.precision_at.get(10, 0.0),
                        recall_at_1=metrics.recall_at.get(1, 0.0),
                        recall_at_5=metrics.recall_at.get(5, 0.0),
                        recall_at_10=metrics.recall_at.get(10, 0.0),
                        map=map_value,
                        phash_ms=phash_ms_avg,     # F0
                        stage_a_ms=stage_a_ms,     # F1
                        stage_b_ms=stage_b_ms,     # F2(前段)
                        stage_c_ms=stage_c_ms,     # F2(最終)
                        total_ms=total_ms,
                        n_dataset=n_dataset,
                        n_candidates_a=n_candidates_a,
                        n_candidates_b=n_candidates_b,
                        n_candidates_c=n_candidates_c,
                        n_reconstructed=n_reconstructed,
                        bytes_a=bytes_a,           # F1
                        bytes_b=bytes_b,           # F2(前段)
                        bytes_c=bytes_c,           # F2(最終) ※MPCは0
                    )
                )
                if tau == base_tau and n_dataset > 0:
                    variant_metrics_raw[mode][query.variant].append(
                        {
                            "precision_at_1": metrics.precision_at.get(1, 0.0),
                            "precision_at_5": metrics.precision_at.get(5, 0.0),
                            "precision_at_10": metrics.precision_at.get(10, 0.0),
                            "recall_at_1": metrics.recall_at.get(1, 0.0),
                            "recall_at_5": metrics.recall_at.get(5, 0.0),
                            "recall_at_10": metrics.recall_at.get(10, 0.0),
                            "map": map_value,
                        }
                    )
                    reconstruction_ratios[mode].append(n_reconstructed / n_dataset)
                    stage_ratio[mode].append(
                        (
                            n_candidates_a / n_dataset,
                            n_candidates_b / n_dataset,
                            n_candidates_c / n_dataset if n_dataset else 0.0,
                        )
                    )
            candidate_reduction[mode].append((n_candidates_a, n_candidates_b, n_reconstructed))
            time_stack[mode].append((phash_ms_avg, stage_a_ms, stage_b_ms, stage_c_ms))
            byte_stack[mode].append((bytes_a, bytes_b, bytes_c))

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(logs[0]).keys()))
        writer.writeheader()
        for log in logs:
            writer.writerow(asdict(log))

    # ===== Summary text =====
    summary_lines = []
    for mode in modes:
        mode_logs = [log for log in logs if log.mode == mode]
        if not mode_logs:
            continue
        avg_precision1 = np.mean([log.precision_at_1 for log in mode_logs])
        avg_precision5 = np.mean([log.precision_at_5 for log in mode_logs])
        avg_precision10 = np.mean([log.precision_at_10 for log in mode_logs])
        avg_recall10 = np.mean([log.recall_at_10 for log in mode_logs])
        avg_map = np.mean([log.map for log in mode_logs])
        avg_total_ms = np.mean([log.total_ms for log in mode_logs])
        summary_lines.append(
            f"{mode}: P@1={avg_precision1:.3f}, P@5={avg_precision5:.3f}, "
            f"P@10={avg_precision10:.3f}, R@10={avg_recall10:.3f}, mAP={avg_map:.3f}, total_ms={avg_total_ms:.2f}"
        )
    summary_txt = output_dir / "summary.txt"
    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8")

    analysis_payload: Dict[str, object] = {}

    # ===== Variant recall summary & plot =====
    variant_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    if variant_metrics_raw:
        for mode, variant_map in variant_metrics_raw.items():
            mode_summary: Dict[str, Dict[str, float]] = {}
            for variant, entries in variant_map.items():
                if not entries:
                    continue
                arr = np.array(
                    [
                        [
                            entry["precision_at_1"],
                            entry["precision_at_5"],
                            entry["precision_at_10"],
                            entry["recall_at_1"],
                            entry["recall_at_5"],
                            entry["recall_at_10"],
                            entry["map"],
                        ]
                        for entry in entries
                    ],
                    dtype=np.float64,
                )
                mode_summary[variant] = {
                    "precision_at_1": float(arr[:, 0].mean()),
                    "precision_at_5": float(arr[:, 1].mean()),
                    "precision_at_10": float(arr[:, 2].mean()),
                    "recall_at_1": float(arr[:, 3].mean()),
                    "recall_at_5": float(arr[:, 4].mean()),
                    "recall_at_10": float(arr[:, 5].mean()),
                    "map": float(arr[:, 6].mean()),
                }
            if mode_summary:
                variant_summary[mode] = mode_summary
        if variant_summary:
            analysis_payload["variant_metrics"] = variant_summary
            variant_names = sorted({variant for mode_summary in variant_summary.values() for variant in mode_summary})
            modes_for_variants = [mode for mode in modes if mode in variant_summary and variant_summary[mode]]
            if variant_names and modes_for_variants:
                fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(variant_names)), 5))
                width = 0.8 / max(len(modes_for_variants), 1)
                x = np.arange(len(variant_names))
                for idx, mode in enumerate(modes_for_variants):
                    values = [
                        variant_summary[mode].get(variant, {}).get("recall_at_10", np.nan) for variant in variant_names
                    ]
                    ax.bar(x + idx * width, values, width, label=mode)
                ax.set_xticks(x + width * (len(modes_for_variants) - 1) / 2)
                ax.set_xticklabels(variant_names, rotation=45, ha="right")
                ax.set_ylabel("Recall@10")
                ax.set_ylim(0.0, 1.05)
                ax.set_title("Recall@10 by Transform Variant")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / "variant_recall.png")
                plt.close(fig)

    # ===== Candidate reduction ratio (F1/F2/F3) =====
    stage_ratio_summary: Dict[str, Dict[str, float]] = {}
    if stage_ratio:
        for mode, ratios in stage_ratio.items():
            if not ratios:
                continue
            arr = np.array(ratios, dtype=np.float64)
            stage_ratio_summary[mode] = {
                "stage_a_ratio": float(arr[:, 0].mean()),
                "stage_b_ratio": float(arr[:, 1].mean()),
                "stage_c_ratio": float(arr[:, 2].mean()),
            }
        if stage_ratio_summary:
            analysis_payload["stage_reduction_ratios"] = stage_ratio_summary
            fig, ax = plt.subplots(figsize=(8, 5))
            for mode in modes:
                stats = stage_ratio_summary.get(mode)
                if not stats:
                    continue
                ax.plot(
                    ["F1", "F2", "F3"],
                    [stats["stage_a_ratio"], stats["stage_b_ratio"], stats["stage_c_ratio"]],
                    marker="o",
                    label=mode,
                )
            ax.set_ylabel("Candidate Ratio vs Full Dataset")
            ax.set_ylim(0.0, 1.05)
            ax.set_title("Normalized Candidate Reduction by Phase")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "candidate_reduction_ratio.png")
            plt.close(fig)

    # ===== Reconstruction ratio =====
    reconstruction_summary: Dict[str, float] = {}
    if reconstruction_ratios:
        for mode, values in reconstruction_ratios.items():
            if not values:
                continue
            reconstruction_summary[mode] = float(np.mean(values))
        if reconstruction_summary:
            analysis_payload["reconstruction_ratio"] = reconstruction_summary
            modes_rr = [mode for mode in modes if mode in reconstruction_summary]
            values = [reconstruction_summary[mode] for mode in modes_rr]
            if modes_rr:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(modes_rr, values)
                ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Full Reconstruction")
                ax.set_ylabel("Reconstructed / Total")
                ax.set_ylim(0.0, max(1.05, max(values) * 1.05))
                ax.set_title("Selective Reconstruction Ratio")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / "reconstruction_ratio.png")
                plt.close(fig)

    if analysis_payload:
        analysis_path = output_dir / "analysis_extended.json"
        analysis_path.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")

    # ===== Candidate reduction plot (F1/F2/F3) =====
    fig, ax = plt.subplots(figsize=(8, 5))
    modes_for_plot = [mode for mode in modes if mode != "plain"]
    for mode in modes_for_plot:
        counts = candidate_reduction[mode]
        if not counts:
            continue
        arr = np.array(counts)
        mean_counts = arr.mean(axis=0)
        ax.plot(["F1", "F2", "F3"], mean_counts, marker="o", label=mode)
    ax.set_ylabel("Candidates")
    ax.set_title("Candidate Reduction by Phase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "candidate_reduction.png")
    plt.close(fig)

    # ===== Time stack bar (F0/F1/F2/F3) =====
    fig, ax = plt.subplots(figsize=(8, 5))
    modes_time = []
    stack_vals = []
    for mode in modes:
        if mode not in time_stack:
            continue
        arr = np.array(time_stack[mode])
        modes_time.append(mode)
        stack_vals.append(arr.mean(axis=0))
    if stack_vals:
        stack_vals = np.array(stack_vals)
        bottoms = np.zeros(len(modes_time))
        labels = ["F0: Feature", "F1: Screen", "F2: Score", "F3: Materialize"]
        for i in range(stack_vals.shape[1]):
            ax.bar(modes_time, stack_vals[:, i], bottom=bottoms, label=labels[i])
            bottoms += stack_vals[:, i]
        ax.set_ylabel("Time (ms)")
        ax.set_title("Phase Timing Breakdown")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "time_breakdown.png")
        plt.close(fig)

    # ===== Communication breakdown (F1/F2/F3) =====
    # byte_stack: (bytes_a, bytes_b, bytes_c)
    if byte_stack:
        fig, ax = plt.subplots(figsize=(8, 5))
        modes_comm = []
        comm_vals = []
        for mode in modes:
            if mode not in byte_stack:
                continue
            arr = np.array(byte_stack[mode])
            modes_comm.append(mode)
            comm_vals.append(arr.mean(axis=0))
        if comm_vals:
            comm_vals = np.array(comm_vals)
            bottoms = np.zeros(len(modes_comm))
            labels = ["F1: Screen", "F2: Score(early)", "F2: Score(final)"]
            for i in range(comm_vals.shape[1]):
                ax.bar(modes_comm, comm_vals[:, i], bottom=bottoms, label=labels[i])
                bottoms += comm_vals[:, i]
            ax.set_ylabel("Bytes per Query")
            ax.set_title("Phase Communication Breakdown")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "communication_breakdown.png")
            plt.close(fig)

    # ===== ROC / PR curves =====
    roc_pr_summary_data: Dict[str, Dict[str, List[float]]] = {}
    for mode in modes:
        records = roc_data.get(mode)
        if not records:
            continue
        mean_fpr = np.mean([np.array(r[0]) for r in records], axis=0)
        mean_tpr = np.mean([np.array(r[1]) for r in records], axis=0)
        mean_prec = np.mean([np.array(r[2]) for r in records], axis=0)
        mean_rec = np.mean([np.array(r[3]) for r in records], axis=0)
        roc_auc = area_under_curve(mean_fpr, mean_tpr)
        pr_auc = area_under_curve(mean_rec, mean_prec)

        roc_pr_summary_data[mode] = {
            "tau_values": [float(v) for v in tau_values],
            "mean_fpr": mean_fpr.tolist(),
            "mean_tpr": mean_tpr.tolist(),
            "mean_precision": mean_prec.tolist(),
            "mean_recall": mean_rec.tolist(),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        }

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mean_fpr, mean_tpr, marker="o")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curve - {mode} (AUC={roc_auc:.3f})")
        fig.tight_layout()
        fig.savefig(output_dir / f"roc_{mode}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mean_rec, mean_prec, marker="o")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve - {mode} (AUC={pr_auc:.3f})")
        fig.tight_layout()
        fig.savefig(output_dir / f"pr_{mode}.png")
        plt.close(fig)

    if roc_pr_summary_data:
        roc_summary_path = output_dir / "roc_pr_summary.json"
        roc_summary_path.write_text(json.dumps(roc_pr_summary_data, indent=2), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(6, 5))
        tau_array = np.array([float(v) for v in tau_values])
        for mode in modes:
            data = roc_pr_summary_data.get(mode)
            if not data:
                continue
            ax.plot(tau_array, data["mean_recall"], marker="o", label=mode)
        ax.set_xlabel("Tau Threshold")
        ax.set_ylabel("Recall")
        ax.set_title("Recall vs Tau Sensitivity")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "tau_sensitivity.png")
        plt.close(fig)

    # ===== Distance histograms =====
    bins = list(range(0, 65))
    for variant, distances in distance_hist.items():
        counts, edges = histogram(distances, bins=bins)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(edges[:-1], counts, width=1, align="edge")
        ax.set_xlim(0, 64)
        ax.set_xlabel("Hamming Distance")
        ax.set_ylabel("Count")
        ax.set_title(f"Distance Distribution - {variant}")
        fig.tight_layout()
        fig.savefig(output_dir / f"hist_{variant}.png")
        plt.close(fig)

    print("Experiment complete. Results saved to", output_dir)


if __name__ == "__main__":
    main()
