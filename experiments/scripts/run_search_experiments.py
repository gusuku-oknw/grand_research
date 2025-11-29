"""Run SIS+pHash experiments (modular orchestrator)."""
from __future__ import annotations
import argparse, csv, json, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from experiments.common.dataset import build_positive_lookup, build_samples, load_derivative_mapping, Sample
from experiments.common.metrics import (
    RankingMetrics,
    average_precision_from_ranking,
    precision_recall_at_k,
    roc_pr_curves,
    area_under_curve,
    histogram,
)
from sis_image.dealer_based import SearchableSISIndex, SearchableSISWithImageStore
from sis_image.common.phash import phash64, hamming64

from sis_modes.types import ModeContext, ModeResult, PhaseStats
from sis_modes.base import ModeRunner
from sis_modes.plain import PlainRunner
from sis_modes.sis_aes_gcm import AESGCMRunner
from sis_modes.sis_naive import NaiveRunner
from sis_modes.sis_only import SISOnlyRunner
from sis_modes.sis_selective import PartialRunner, SelectiveRunner
from sis_modes.sis_mpc import MPCRunner
from sis_modes.minhash_lsh import MinHashLSHRunner
from sis_modes.aes_crypto import AESGCMStorage, EncryptedImageRecord, load_aesgcm_master

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# クエリ単位で Stage 計測値や精度をまとめるデータクラス
@dataclass
class QueryLog:
    dataset: str; mode: str; query_key: str; query_variant: str; transform: str; tau: int
    precision_at_1: float; precision_at_5: float; precision_at_10: float
    recall_at_1: float; recall_at_5: float; recall_at_10: float; map: float
    phash_ms: float; stage1_ms: float; stage2_ms: float; total_ms: float
    n_dataset: int; stage1_candidates: int; stage2_candidates: int; stage2_ranked: int
    n_reconstructed: int; stage1_bytes: int; stage2_bytes: int

# 実行するモード名のリストから ModeRunner の辞書を生成
def build_runners(names: List[str]) -> Dict[str, ModeRunner]:
    alias = {
        "plain": PlainRunner,
        "sis_server_naive": NaiveRunner,
        "sis_client_dealer_free": SelectiveRunner,
        "sis_client_partial": PartialRunner,
        "sis_mpc": MPCRunner,
        "sis_only": SISOnlyRunner,
        "aes_gcm": AESGCMRunner,
        "minhash_lsh": MinHashLSHRunner,
    }
    return {n: alias[n]() for n in names}

# SIS モードごとに共有ストアを初期化（必要なフォルダも作成）
def ensure_workflow(
    mode: str,
    k: int,
    n: int,
    bands: int,
    seed: int,
    secure: bool,
    shares_dir: Path,
    meta_dir: Path,
    share_strategy: str,
    fusion_grid: int,
    fusion_threshold: int | None,
    use_oprf: bool,
    dummy_band_queries: int,
    pad_band_queries: int | None,
    fixed_band_queries: int | None,
    hmac_key_encrypt_env_var: str | None = None,
    oprf_key_encrypt_env_var: str | None = None,
) -> SearchableSISWithImageStore:
    shares_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return SearchableSISWithImageStore(
        k=k,
        n=n,
        bands=bands,
        token_len=8,
        seed=seed,
        shares_dir=str(shares_dir),
        meta_dir=str(meta_dir),
        secure_distance=secure,
        share_strategy=share_strategy,
        fusion_grid=fusion_grid,
        fusion_threshold=fusion_threshold,
        use_oprf=use_oprf,
        dummy_band_queries=dummy_band_queries,
        pad_band_queries=pad_band_queries,
        fixed_band_queries=fixed_band_queries,
        hmac_key_encrypt_env_var=hmac_key_encrypt_env_var,
        oprf_key_encrypt_env_var=oprf_key_encrypt_env_var,
    )

# すべてのサンプルについて pHash を計算し辞書と平均処理時間を返す
def compute_phashes(samples: List[Sample]) -> Tuple[Dict[str,int], float]:
    t0 = time.perf_counter(); ph = {}
    for s in samples:
        ph[s.key] = phash64(str(s.path))
    avg_ms = ((time.perf_counter() - t0) * 1000.0) / max(len(samples),1)
    return ph, avg_ms

# モードのランキング結果から Precision/Recall と AP を算出
def metrics_from_ranking(ranking_ids: List[str], id_to_index: Dict[str,int], positives: List[int]) -> Tuple[RankingMetrics, float]:
    ranked_indices = [id_to_index[r] for r in ranking_ids]
    metrics = precision_recall_at_k(ranked_indices, positives, k_values=[1,5,10])
    ap = average_precision_from_ranking(ranked_indices, set(positives))
    return metrics, ap


# tqdm が利用可能な場合だけ進捗バーに差し替えるヘルパ
def iter_progress(iterable, *, desc: str, total: int | None = None, leave: bool = False):
    """Wrap iterable with tqdm when available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping_json", type=Path, default=Path("data/coco/derivative_mapping.json"))
    ap.add_argument("--output_dir", type=Path, default=Path("reports"))
    ap.add_argument("--work_dir", type=Path, default=Path("eval_artifacts"))
    ap.add_argument("--dataset_name", type=str, default="coco_val2017")
    ap.add_argument("--max_queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--min_band_votes", type=int, default=3)
    ap.add_argument("--max_hamming", type=int, default=10)
    ap.add_argument("--share_strategy", choices=["shamir", "phash-fusion"], default="shamir")
    ap.add_argument("--fusion_grid", type=int, default=8)
    ap.add_argument("--fusion_threshold", type=int, default=None)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--tau_values", type=int, nargs="+", default=[6,8,10,12])
    ap.add_argument("--stage_b_bytes", type=int, default=2)
    ap.add_argument("--stage_b_margin", type=int, default=8)
    allowed_modes = [
        "plain",
        "sis_only",
        "sis_server_naive",
        "sis_client_dealer_free",
        "sis_client_partial",
        "sis_mpc",
        "aes_gcm",
        "minhash_lsh",
    ]
    ap.add_argument("--modes", type=str, nargs="+", choices=allowed_modes, default=allowed_modes)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--use_oprf", action="store_true", help="Use VOPRF for Stage-1 band tokens.")
    ap.add_argument("--dummy_band_queries", type=int, default=0)
    ap.add_argument("--pad_band_queries", type=int, default=None)
    ap.add_argument("--fixed_band_queries", type=int, default=None)
    ap.add_argument("--hmac_key_encrypt_env", type=str, default=None)
    ap.add_argument("--oprf_key_encrypt_env", type=str, default=None)
    args = ap.parse_args()

    output_dir: Path = args.output_dir; output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_derivative_mapping(args.mapping_json)
    samples = build_samples(mapping, include_variants=None)
    # クエリ数をサンプリングして高速化するオプション
    if args.max_queries and args.max_queries < len(samples):
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(samples), size=args.max_queries, replace=False)
        samples = [samples[i] for i in idxs]

    # Ground truth（同一画像ID）と pHash 辞書を構築
    positives_lookup = build_positive_lookup(samples)
    id_to_index = {s.key:i for i,s in enumerate(samples)}
    phashes, phash_ms_avg = compute_phashes(samples)
    aes_storage: AESGCMStorage | None = None
    aes_records: Dict[str, EncryptedImageRecord] | None = None
    if "aes_gcm" in args.modes:
        aes_storage = AESGCMStorage(load_aesgcm_master())
        aes_records = {}
        for sample in samples:
            raw = sample.path.read_bytes()
            aes_records[sample.key] = aes_storage.encrypt_image(sample.key, raw)

    # indices
    plain_index = SearchableSISIndex(
        k=args.k,
        n=args.n,
        bands=args.bands,
        token_len=8,
        seed=args.seed,
        use_oprf=args.use_oprf,
        key_encrypt_env_var=args.hmac_key_encrypt_env,
        oprf_key_encrypt_env_var=args.oprf_key_encrypt_env,
    )
    for s in iter_progress(samples, desc="Indexing (plain)", total=len(samples), leave=False):
        plain_index.add_image_with_phash(s.key, str(s.path), phashes[s.key])

    workflows: Dict[str, SearchableSISWithImageStore] = {}
    for m in iter_progress(args.modes, desc="Preparing workflows", total=len(args.modes), leave=False):
        if m in {"plain", "aes_gcm"}:
            continue
        wf = ensure_workflow(
            m,
            args.k,
            args.n,
            args.bands,
            args.seed,
            secure=(m == "sis_mpc"),
            shares_dir=args.work_dir / "shared_store" / "img_shares",
            meta_dir=args.work_dir / "shared_store" / "img_meta",
            share_strategy=args.share_strategy,
            fusion_grid=args.fusion_grid,
            fusion_threshold=args.fusion_threshold,
            use_oprf=args.use_oprf,
            dummy_band_queries=args.dummy_band_queries,
            pad_band_queries=args.pad_band_queries,
            fixed_band_queries=args.fixed_band_queries,
            hmac_key_encrypt_env_var=args.hmac_key_encrypt_env,
            oprf_key_encrypt_env_var=args.oprf_key_encrypt_env,
        )
        for s in iter_progress(samples, desc=f"Ingest {m}", total=len(samples), leave=False):
            wf.add_image(s.key, str(s.path), phash=phashes[s.key])
        workflows[m] = wf

    # クエリごとに問い合わせるサーバー（k 台）を決定
    servers = plain_index.list_servers()[: args.k]
    ctx = ModeContext(
        samples=samples,
        id_to_index=id_to_index,
        phashes=phashes,
        plain_index=plain_index,
        workflows=workflows,
        servers=servers,
        args=args,
        aes_storage=aes_storage,
        aes_records=aes_records,
    )

    runners = build_runners(args.modes)

    logs: List[QueryLog] = []
    # ROC / PR 曲線を後段で生成するための蓄積領域
    roc_data: Dict[str, List[Tuple[List[float], List[float], List[float], List[float]]]] = {m:[] for m in args.modes}

    for q in iter_progress(samples, desc="Running queries", total=len(samples)):
        positives = positives_lookup[q.image_id]
        qhash = phashes[q.key]
        for name, runner in runners.items():
            start_total = time.perf_counter()
            res: ModeResult = runner.run_query(q.key, qhash, ctx)
            total_ms = (time.perf_counter() - start_total) * 1000.0
            metrics, apv = metrics_from_ranking(res.final_ranking_ids, id_to_index, positives)
            # ROC/PR 用に候補距離を配列化（伸びしろ=65で代用）

            ranked_d = dict(res.ranked_pairs)
            mode_d = [ (ranked_d.get(s.key, 65)) for s in samples ]
            labels = [1 if id_to_index[s.key] in positives else 0 for s in samples]
            fpr,tpr,prec,rec = roc_pr_curves(mode_d, labels, tau_values=sorted(set(args.tau_values)) or [args.max_hamming or 10])
            roc_data[name].append((fpr,tpr,prec,rec))

            stage2_bytes = res.stats.bytes_f2_early + res.stats.bytes_f2_final
            for tau in (sorted(set(args.tau_values)) or [args.max_hamming or 10]):
                logs.append(QueryLog(
                    dataset=args.dataset_name, mode=name, query_key=q.key, query_variant=q.variant,
                    transform=q.variant, tau=tau, precision_at_1=metrics.precision_at.get(1,0.0),
                    precision_at_5=metrics.precision_at.get(5,0.0), precision_at_10=metrics.precision_at.get(10,0.0),
                    recall_at_1=metrics.recall_at.get(1,0.0), recall_at_5=metrics.recall_at.get(5,0.0),
                    recall_at_10=metrics.recall_at.get(10,0.0), map=apv,
                    phash_ms=phash_ms_avg, stage1_ms=res.stats.f1_ms, stage2_ms=res.stats.f2_ms,
                    total_ms=total_ms, n_dataset=len(samples),
                    stage1_candidates=res.stats.n_cand_f1, stage2_candidates=res.stats.n_cand_f2,
                    stage2_ranked=res.stats.n_eval_final, n_reconstructed=res.stats.n_reconstructed,
                    stage1_bytes=res.stats.bytes_f1, stage2_bytes=stage2_bytes
                ))

    # metrics.csv を書き出し（各クエリ x モードの行）
    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(asdict(logs[0]).keys()))
        writer.writeheader()
        for log in logs: writer.writerow(asdict(log))

    # 以降の図表生成は既存 plotting を流用（省略可）
    print("Done. Wrote:", metrics_csv)

if __name__ == "__main__":
    main()
