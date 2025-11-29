# SIS Image Sharing Package（日本語版ハンドブック）

このリポジトリは、検索可能な秘密分散 (Searchable Image Sharing; SIS) のデモ実装を Python パッケージ `sis_image` として再編したものです。pHash（64bit パーセプトルハッシュ）と Shamir 秘密分散を組み合わせ、プライバシーを保ちながら類似画像検索と再構成を行います。

## 設計思想とアーキテクチャ

本研究の核心は、**pHashで候補を絞り込み、MPC（マルチパーティ計算）を用いて秘密分散状態（SIS）のまま距離を計算する**ことで、計算効率と機密性を両立させる点にあります。

従来の「全画像を再構成 → 平文で類似検索」という素朴なアプローチは、計算量が膨大で、プライバシー漏洩のリスクを伴いました。本方式では、画像の原本やpHash（画像ハッシュ）を平文に戻すことなく、秘密分散とMPCの枠組みの中で処理を完結させます。

用語の整理： 暗号学的な「復号」とは異なり、SISではシェア（断片）を集めて元の情報を復元するため、本稿では「**再構成 (reconstruction)**」と呼びます。

### 2段階の検索パイプライン

検索は 2 段階に整理され、Stage-1 → Stage-2 の流れに統合することで構造をシンプルにしています。

- **Stage-1（索引による候補削減）**  
  クエリの pHash から HMAC トークンを生成し、各サーバーのインデックスと照合することで候補集合を絞り込みます。この段階で検索対象を数パーセントまで圧縮します。

- **Stage-2（再構成 + セキュア距離計算）**  
  Stage-1 で残った候補に対して、SIS シェアの部分的な再構成（`sis_selective` 等）や、再構成せずに MPC で距離を算出する処理を実行します。Stage-2 は従来の Stage-B 〜 Stage-C を合わせたもので、再構成と距離計算をまとめて扱います。

最終的な画像の**再構成**は検索とは切り離されており、実際に原本が必要な場合のみ K-of-N によってクライアント側で行われます。

### 高速かつ安全な理由

- **計算量の削減**: 全件比較 O(N) を、ごく少数の候補 O(|候補|) のみに限定することで、計算量を大幅に削減します。
- **平文非曝露**: 検索プロセス全体を通じて、画像の原本もpHashも平文に再構成されないため、高い機密性を維持します。
- **スケーラビリティ**: 計算負荷が最も高いMPC処理を、最終段階の少数候補に限定することで、システム全体のスケーラビリティを確保します。

### 留意点と今後の課題

- **アクセスパターンの漏洩**: どのHMACトークンが索引にヒットしたかという情報（アクセスパターン）が漏洩する可能性があります。ダミークエリの挿入やバッチ処理によるクエリの匿名化で緩和するアプローチが考えられます。
- **鍵管理**: HMAC鍵はバンドごとに分離し、定期的にローテーションさせることが推奨されます。将来的にはOPRF（Oblivious Pseudorandom Function）の導入が望まれます。
- **MPCの実装**: 現在の実装は、シミュレーションとして「再構成→距離計算」を行う擬似的なMPCです。実用上は、これを真のMPCプロトコルやTEE（Trusted Execution Environment）に置き換える必要があります。

### モード別の Stage 利用

| モード | Stage-1 | Stage-2 |
| :--- | :------ | :------ |
| `plain` | ❌（全件 pHash 距離で走査） | ✅ `compute_plain_distances` で距離ソートのみ |
| `sis_naive` | ❌（候補削減なし） | ✅ `rank_candidates` で全候補を復号・評価 |
| `sis_selective` | ✅ HMAC 投票で候補削減 | ✅ `stage_b_filter` + 再構成で Top-K だけ評価 |
| `sis_staged` | ✅ | ✅ |
| `sis_mpc` | ✅ HMAC 投票 | ✅ `rank_candidates_secure` で MPC ランキング（再構成なし） |

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
python experiments/scripts/prepare_coco.py \
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

## 検索実験（Stage-1 → Stage-2）

```bash
PYTHONPATH=. python experiments/scripts/run_search_experiments.py \
    --mapping_json data/coco2017_derivatives/derivative_mapping.json \
    --output_dir output/results/coco_val2017_modular \
    --work_dir output/artifacts/coco_val2017_modular \
    --modes plain sis_naive sis_selective sis_staged sis_mpc \
    --max_queries 500 \
    --bands 8 --k 3 --n 5 \
    --force
```

- 各モードの Stage-1 / Stage-2 で時間・通信量・精度が `metrics.csv` に記録されます。`pip install tqdm` 済みであれば `Indexing / Preparing workflows / Running queries` の進捗バーが表示されます。
- Colab など計算資源が限られる環境では `--max_images`, `--max_queries` を下げてください。

## 図表の生成

```bash
python -m experiments.common.plotting \
    output/results/coco_val2017_modular/metrics.csv \
    --output_dir output/figures/coco_val2017_modular
```

候補削減グラフや通信量の内訳、Precision-Latency カーブなどが `output/figures` 配下に出力されます。

## 補足

- 研究レポート用テンプレートは `docs/reports/2025_selective_reconstruction_report_template.md` にまとまっています。
- セキュリティ強度検証や ROC/PR 曲線は `metrics.csv` と同じ成果物から再生成できます。
- 追加の CLI オプションは `python experiments/scripts/prepare_coco.py --help` と `python experiments/scripts/run_search_experiments.py --help` で確認してください。

## ディレクトリ構成（整理後）

- **`src/sis_image/`**: SISのコアライブラリ。共通処理は `common/`、ディーラー有りの index/workflow/CLI は `dealer_based/`、ディーラーフリーのシミュレータ・MPC分分析は `dealer_free/` に分かれています。
- **`experiments/`**: データ準備・実験実行・分析をまとめたコード群。`prepare_data.py` で派生データを生成し、`run_experiments.py` が `experiments/modes/` にある各モードを動かし、`analyze_results.py` が出力を解釈します。
- **`output/`**: 全出力物を集約。`output/datasets/` に派生データ、`output/artifacts/` に作業中の共有ストア、`output/results/` に `metrics.csv` などの数値、`output/figures/` に Matplotlib図版、`output/reports/` にレポート書類を置きます。

このように `experiments/` が「何を計算するか」、`output/` が「何が生成されたか」を分担することで、コードと成果物の境界が明確になり、ドキュメントや再現性の要求にも対応しやすくなります。
