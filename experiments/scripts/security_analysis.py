"""Security evaluation utilities for SIS+pHash system (patched)."""

from __future__ import annotations

import argparse
import json
import sys
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np

from experiments.common.dataset import build_samples, load_derivative_mapping
from sis_image.dealer_based import SearchableSISIndex
from sis_image.common.phash import phash64, hamming64
from sis_image.common.shamir import shamir_recover_bytes
from sis_image.common.tokens import split_bands, hmac_token


def compute_share_entropy(index: SearchableSISIndex) -> Dict[int, Dict[int, float]]:
    """
    各サーバ・各バイト位置ごとの頻度分布からエントロピー（bit）を算出。
    GF(257) を想定し、ビン数は 257（0..256）。実データは通常 0..255 だが、257ビンでも安全。
    """
    per_server_counts: Dict[int, Dict[int, np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.zeros(257, dtype=np.int64))
    )
    for server, server_shares in index.server_shares.items():
        for share in server_shares.values():
            for pos, byte in enumerate(share.share_bytes):
                per_server_counts[server][pos][byte] += 1

    entropy: Dict[int, Dict[int, float]] = {}
    for server, pos_counts in per_server_counts.items():
        entropy[server] = {}
        for pos, hist in pos_counts.items():
            total = hist.sum()
            if total == 0:
                entropy[server][pos] = 0.0
                continue
            probs = hist / total
            probs = probs[probs > 0]
            entropy_value = float(-(probs * np.log2(probs)).sum())
            entropy[server][pos] = entropy_value
    return entropy


def estimate_mutual_information(
    entropy: Dict[int, Dict[int, float]],
    field_bits: float = math.log2(257.0),
) -> Dict[int, Dict[int, float]]:
    """
    MI ≈ H_max - H(obs)。GF(257)想定なら H_max ≈ log2(257) ≈ 8.01 を既定に。
    """
    max_entropy = float(field_bits)
    return {
        server: {pos: max(0.0, max_entropy - value) for pos, value in pos_map.items()}
        for server, pos_map in entropy.items()
    }


def summarize_nested_stats(nested: Dict[int, Dict[int, float]]) -> Dict[str, float]:
    values = [value for pos_map in nested.values() for value in pos_map.values()]
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def stage1_link_probability(
    index_a: SearchableSISIndex,
    index_b: SearchableSISIndex,
    phashes: Dict[str, int],
) -> float:
    """
    「同一画像・同一バンド」のトークンが一致する割合を推定。
    重要点：
      - サーバ集合は共通部分のみ
      - bands は min(index_a.bands, index_b.bands)
      - token_len は min で合わせて比較（長さが違うと常に不一致になりうる）
      - 画像キーも共通部分のみ
    """
    matches = 0
    total = 0

    servers_a = set(index_a.list_servers())
    servers_b = set(index_b.list_servers())
    servers = sorted(servers_a & servers_b)
    if not servers:
        return 0.0

    common_keys = set(index_a._images.keys()) & set(index_b._images.keys())
    if not common_keys:
        return 0.0

    bands_common = min(index_a.bands, index_b.bands)
    token_len_common = min(index_a.token_len, index_b.token_len)

    for image_id in common_keys:
        hash_val = phashes.get(image_id)
        if hash_val is None:
            continue
        bands_a = split_bands(hash_val, index_a.bands)
        bands_b = split_bands(hash_val, index_b.bands)

        for server in servers:
            # hmac_keys が両indexに存在するか確認
            if server not in index_a.hmac_keys or server not in index_b.hmac_keys:
                continue

            for band_idx in range(bands_common):
                try:
                    key_a = index_a.hmac_keys[server][band_idx]
                    key_b = index_b.hmac_keys[server][band_idx]
                except (IndexError, KeyError):
                    continue

                token_a = hmac_token(
                    key_a,
                    bands_a[band_idx].to_bytes(index_a.band_bytes, "big"),
                    token_len_common,
                )
                token_b = hmac_token(
                    key_b,
                    bands_b[band_idx].to_bytes(index_b.band_bytes, "big"),
                    token_len_common,
                )
                if token_a == token_b:
                    matches += 1
                total += 1

    return matches / total if total else 0.0


def leakage_attack_statistics(
    index: SearchableSISIndex,
    phashes: Dict[str, int],
    trials: int = 100,
) -> Tuple[float, float, int]:
    """
    k-1 台のサーバが漏洩したと仮定し、Shamir復元を（あえて）未満しきい値で試みる攻撃を模擬。
    - 成功率: 復元ハッシュ == 真値 となった割合（理論上 0 に近い）
    - 平均ハミング距離: 復元値(8Bにパディング) と真値 の距離
    - 有効試行回数: 実際に距離計算できた件数
    """
    rng = np.random.default_rng(2025)
    successes = 0
    distances: List[int] = []

    server_ids = index.list_servers()
    if len(server_ids) < index.k:
        return 0.0, 0.0, 0

    image_keys = list(index._images.keys())
    if not image_keys:
        return 0.0, 0.0, 0

    for _ in range(trials):
        image_id = rng.choice(image_keys)
        hash_true = phashes.get(image_id)
        if hash_true is None:
            continue

        shares = index.get_shares(image_id, server_ids)
        if not shares:
            continue

        # k-1 台だけを漏洩
        if index.k - 1 <= 0 or len(server_ids) < (index.k - 1):
            continue
        compromised = rng.choice(server_ids, size=index.k - 1, replace=False)

        subset = {int(server): shares[int(server)] for server in compromised if server in shares}
        if not subset:
            continue

        # しきい値未満での復元は通常失敗する。実装が例外を投げる可能性に対応。
        try:
            recovered = shamir_recover_bytes(subset)  # type: ignore[arg-type]
        except Exception:
            # 復元不能 → 計測不能としてスキップ
            continue

        # 8バイト（64bit）にパディング/切り詰め
        rec_bytes = bytes(recovered[:8] + [0] * max(0, 8 - len(recovered)))
        recovered_hash = int.from_bytes(rec_bytes, "big")

        if recovered_hash == hash_true:
            successes += 1
        distances.append(hamming64(hash_true, recovered_hash))

    success_rate = (successes / len(distances)) if distances else 0.0
    mean_distance = float(np.mean(distances)) if distances else 0.0
    return success_rate, mean_distance, len(distances)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run security analysis for SIS+pHash (patched).")
    parser.add_argument("--mapping_json", type=Path, default=Path("data/coco/derivative_mapping.json"))
    parser.add_argument("--max_images", type=int, default=1000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output_json", type=Path, default=Path("reports/security_summary.json"))
    parser.add_argument("--leakage_trials", type=int, default=200)
    parser.add_argument("--force", action="store_true", help="Regenerate summary even if output exists.")
    args = parser.parse_args()

    if args.output_json.exists() and not args.force:
        print(f"Security summary already exists at {args.output_json}. Use --force to regenerate.")
        return

    # データ読み込み
    mapping = load_derivative_mapping(args.mapping_json)
    samples = build_samples(mapping)
    if args.max_images and args.max_images < len(samples):
        samples = samples[: args.max_images]

    # 2つのエポックの index（鍵・シードの違いを模擬）
    index_a = SearchableSISIndex(k=args.k, n=args.n, bands=args.bands, token_len=8, seed=args.seed)
    index_b = SearchableSISIndex(k=args.k, n=args.n, bands=args.bands, token_len=8, seed=args.seed + 1)

    # pHash 準備 & インデックス投入
    phashes: Dict[str, int] = {}
    for sample in samples:
        ph_val = phash64(str(sample.path))
        phashes[sample.key] = ph_val
        index_a.add_image(sample.key, str(sample.path))
        index_b.add_image(sample.key, str(sample.path))

    # 1) エントロピー & MI
    entropy = compute_share_entropy(index_a)
    mutual_info = estimate_mutual_information(entropy, field_bits=math.log2(257.0))
    entropy_summary = summarize_nested_stats(entropy)
    mutual_summary = summarize_nested_stats(mutual_info)

    # 2) Stage-1トークンのリンク確率（同一鍵 vs 別鍵）
    link_same_epoch = stage1_link_probability(index_a, index_a, phashes)
    link_rotated_epoch = stage1_link_probability(index_a, index_b, phashes)
    link_drop_ratio = (link_rotated_epoch / link_same_epoch) if link_same_epoch > 0 else 0.0

    # 3) しきい値未満漏洩攻撃の統計
    leak_success, leak_mean_distance, leak_trials = leakage_attack_statistics(
        index_a, phashes, trials=args.leakage_trials
    )
    random_baseline = 1.0 / len(index_a._images) if index_a._images else 0.0

    # JSON化（数値キーは文字列化）
    entropy_json = {str(server): {str(pos): value for pos, value in pos_map.items()} for server, pos_map in entropy.items()}
    mutual_json = {str(server): {str(pos): value for pos, value in pos_map.items()} for server, pos_map in mutual_info.items()}

    summary = {
        "entropy_bits": entropy_json,
        "entropy_bits_summary": entropy_summary,
        "mutual_information_estimate": mutual_json,
        "mutual_information_summary": mutual_summary,
        "stage1_token_link_probability_same_epoch": link_same_epoch,
        "stage1_token_link_probability_rotated_epoch": link_rotated_epoch,
        "stage1_token_link_drop_ratio": link_drop_ratio,
        "leakage_top1_success_rate": leak_success,
        "leakage_random_baseline": random_baseline,
        "leakage_mean_hamming_distance": leak_mean_distance,
        "leakage_effective_trials": leak_trials,
        "parameters": {
            "k": args.k,
            "n": args.n,
            "bands": args.bands,
            "max_images": len(samples),
            "leakage_trials": args.leakage_trials,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Security summary written to", args.output_json)


if __name__ == "__main__":
    main()
