# SIS Image Sharing Package（日本語版ハンドブック）

このリポジトリは、検索可能な秘密分散 (Searchable Image Sharing; SIS) のデモ実装を Python パッケージ `sis_image` として再編したものです。pHash（64bit パーセプトルハッシュ）と Shamir 秘密分散を組み合わせ、プライバシーを保ちながら類似画像検索と再構成を行います。

## ステージ別パイプライン概要

**処理の流れ**

1. **準備 / インデックス構築**
   1. 各画像の 64bit pHash を計算。
   2. ハッシュを `bands` 個に分割（例: 8bit × 8 bands）。
   3. 各 band について `HMAC(key_i, band_i)` を計算。
   4. 得られたトークンを複数サーバーの索引に登録（この時点では暗号化＋検索可能化だけで、Shamir シェアを復元する処理は発生しません）。
2. **Stage-A（トークン一致による前段選別）** – クエリも同じくバンド化＋HMAC を行い、サーバーから一致トークンを持つ候補 ID を取得。Shamir 再構成は一切行わず、トークン比較のみで候補を絞ります。
3. **Stage-B（部分的な SIS 復元）** – Stage-A の候補について、Shamir 分割されたシェアから数バイトだけ取得し「近似 pHash」を復元。ここで初めて SIS の復元を一部使います。
4. **Stage-C（完全復元 / MPC）** – 生き残った候補について、必要な `k` 個のシェアを集めて完全復元し、ランキングします。`sis_mpc` ではこの段階も暗号下で計算し、平文復元なしで順位付けが可能です。

| ステージ | 説明 | 主なモジュール | 主要メトリクス |
| :-- | :-- | :-- | :-- |
| **Stage-A** | pHash を複数バンドに分割し、各バンドを HMAC トークン化してサーバー側のバケット投票にかけることで、シェア復元前に 90% 以上の候補を除外します。 | `index.preselect_candidates` | 候補数 (`n_cand_f1`)、通信量 (`bytes_f1`) |
| **Stage-B** | サーバーごとに数バイトだけ部分シェアを取得し、おおよそのハミング距離で足切りする段階。各候補を落とすたびに時間と帯域を記録し、再復号コストを抑えます。 | `sis_common.stage_b_filter` | 時間、通信量、候補削減率 |
| **Stage-C** | 上位候補について完全なシェア復号とランキングを実行。Selective/MPC いずれのモードでも、最終的なハッシュ/画像再構成と順位付けを生成します。 | `index.rank_candidates` / `rank_candidates_secure` | 精度、再現率、レイテンシー |

#### モード別の Stage 利用

| モード | Stage-A | Stage-B | Stage-C |
| :--- | :------ | :------ | :------ |
| `plain` | ❌（全件 pHash 距離で走査） | ❌ | ✅ `compute_plain_distances` で距離ソートのみ |
| `sis_naive` | ❌（候補削減なし） | ❌ | ✅ `rank_candidates` で全候補を復号・評価 |
| `sis_selective` | ✅ HMAC 投票で候補削減 | ✅ `stage_b_filter` による部分シェア検査 | ✅ Top-K のみ復号・評価 |
| `sis_staged` | ✅（`sis_selective` と同一） | ✅ | ✅ |
| `sis_mpc` | ✅ HMAC 投票 | ❌（漏えい防止のためスキップ） | ✅ `rank_candidates_secure` でMPCランキング（再構成なし） |

## セットアップ

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

最小限の依存だけで動かしたい場合は `pip install -r requirements.txt` でも構いません。

## COCO 派生データの準備

```bash
python scripts/prepare_coco.py \
    --coco_dir data/coco2017/val2017 \
    --output_dir data/coco2017_derivatives/val2017 \
    --mapping_json data/coco2017_derivatives/derivative_mapping.json \
    --profile medium \
    --variant_scope all \
    --max_images 5000
```

- 既存の派生画像やマッピングは自動的に再利用されます。すべて作り直す場合は `--force` を指定してください。
- `--profile` で全体の強度を切り替え、`--variant_scope original_only` や `--include_transforms watermark_timestamp` で検証用サブセットだけを生成できます。
- AVIF 派生を使う場合は `pip install pillow-avif-plugin` が必要です。ログを減らしたい場合は `--no_progress` で per-image の進捗表示を抑制できます。

## 検索実験（Stage-A/B/C）

```bash
PYTHONPATH=. python scripts/run_search_experiments.py \
    --mapping_json data/coco2017_derivatives/derivative_mapping.json \
    --output_dir evaluation/results/coco_val2017_modular \
    --work_dir evaluation/artifacts/coco_val2017_modular \
    --modes plain sis_naive sis_selective sis_staged sis_mpc \
    --max_queries 500 \
    --bands 8 --k 3 --n 5 \
    --force
```

- 各モードの Stage-A/B/C で時間・通信量・精度が `metrics.csv` に記録されます。`pip install tqdm` 済みであれば `Indexing / Preparing workflows / Running queries` の進捗バーが表示されます。
- Colab など計算資源が限られる環境では `--max_images`, `--max_queries` を下げてください。

## 図表の生成

```bash
python -m evaluation.plotting \
    evaluation/results/coco_val2017_modular/metrics.csv \
    --output_dir evaluation/figures/coco_val2017_modular
```

候補削減グラフや通信量の内訳、Precision-Latency カーブなどが `evaluation/figures` 配下に出力されます。

## 補足

- 研究レポート用テンプレートは `reports/2025_selective_reconstruction_report_template.md` にまとまっています。
- セキュリティ強度検証や ROC/PR 曲線は `metrics.csv` と同じ成果物から再生成できます。
- 追加の CLI オプションは `python scripts/prepare_coco.py --help` と `python scripts/run_search_experiments.py --help` で確認してください。
