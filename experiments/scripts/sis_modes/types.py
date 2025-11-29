from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from pathlib import Path

from experiments.common.dataset import Sample
from sis_image.dealer_based import SearchableSISIndex
from sis_image.dealer_based import SearchableSISWithImageStore

from .aes_crypto import AESGCMStorage, EncryptedImageRecord

# F0-F3の共通ログ単位（ms/bytes/候補数/復元数）
@dataclass
class PhaseStats:
    f0_ms: float = 0.0
    f1_ms: float = 0.0
    f2_ms: float = 0.0  # 最終評価含む
    f3_ms: float = 0.0
    bytes_f1: int = 0
    bytes_f2_early: int = 0
    bytes_f2_final: int = 0
    n_cand_f1: int = 0
    n_cand_f2: int = 0
    n_eval_final: int = 0
    n_reconstructed: int = 0

# 各クエリの結果（メトリクス計算に必要な最小集合）
@dataclass
class ModeResult:
    final_ranking_ids: List[str]                 # 評価モードの世界観での最終ランキング（尻尾のplain混入なし）
    ranked_pairs: List[Tuple[str, int]]          # [(id, distance)] F2最終で実際に評価した分
    stats: PhaseStats

# オーケストレータから渡されるコンテキスト
@dataclass
class ModeContext:
    samples: List[Sample]
    id_to_index: Dict[str, int]
    phashes: Dict[str, int]
    plain_index: SearchableSISIndex
    workflows: Dict[str, SearchableSISWithImageStore]  # mode -> workflow（plain以外）
    servers: Sequence[int]
    args: object                                     # argparse.Namespace 相当
    aes_storage: AESGCMStorage | None
    aes_records: Dict[str, EncryptedImageRecord] | None
