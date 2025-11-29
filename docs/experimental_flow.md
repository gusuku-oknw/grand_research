# 実験フローと意義

このプロジェクトでは、**Stage-1（索引による候補削減）→ Stage-2（再構成 or MPC による距離計算）**という２段階パイプラインを軸に、COCO 派生データを使った比較実験を行います。ここでは「Stage-1/Stage-2」の骨格はそのまま維持しつつ、モード名と dealer-free の視点を明示的に整理した上で、実行手順・理由・出力の集中を一枚にまとめます。

---

## 1. Why：この流れがなぜ正しいのか

| 課題 | 対策 | Stage | モード例 |
| --- | --- | --- | --- |
| 検索時に処理すべき候補が多すぎる | pHash をバンド分割し、HMAC トークンで絞り込む | Stage-1 | 全モード（`plain`, `sis_server_naive`, `sis_client_*`, `sis_mpc`） |
| pHash を再構成すると漏洩リスク | 再構成は Top-K に限定、あるいは MPC で距離のみ計算 | Stage-2 | `sis_server_naive`, `sis_client_dealer_free`, `sis_client_partial`, `sis_mpc` |
| ディーラーがすべて知る構成 | クライアント側で pHash を生成・SIS 分散し、サーバーはトークンのみ扱う | Stage-2 | `sis_client_dealer_free`, `sis_client_partial`, `sis_mpc` |
| 実験の再現性・成果物管理 | `experiments/` にコード、`output/` に artifacts/results/figures を整理 | 全体 | 全モード |

`sis_server_naive`/`sis_client_*` の違いは「誰が pHash を知るか」「再構成を行う範囲」を変えることで、ディーラーフリーの効果を測るための軸を提供します。 `sis_mpc` は Stage-2 で再構成を完全にスキップし、MPC のみで距離を計算するため、最も強いプライバシー保証を持ちます。

---

## 2. 着火前の準備

1. 仮想環境を作り、依存をインストール：  
   ```bash
   pip install -r requirements.txt
   ```
2. スクリプトを `src/` を先頭にした `PYTHONPATH=./src:.` で実行（`experiments/scripts/*` すべて）。  
3. COCO 画像・派生データを `S:/` に置く（WSL からは `/mnt/s/`）。例：`/mnt/s/coco2017/val2017`。

---

## 3. 実験フロー

### 3.1 派生データ生成

```bash
python experiments/scripts/prepare_coco.py \
  --coco_dir /mnt/s/coco2017/val2017 \
  --output_dir /mnt/s/coco2017_derivatives/val2017 \
  --mapping_json /mnt/s/coco2017_derivatives/val2017/mapping.json \
  --profile medium --variant_scope all --max_images 5000
```

- **意義**: `mapping.json` によって「元画像 ↔ 派生画像」が固定され、再現性のある精度/耐性評価が可能になる。  
- **なぜ `S:/` か**: COCO データ数百万枚級のため、WSL 環境で容量・I/O に余裕のある S ドライブを使うのが実用的。

### 3.2 モード比較実行

```bash
PYTHONPATH=./src:. python experiments/scripts/run_search_experiments.py \
  --mapping_json /mnt/s/coco2017_derivatives/val2017/mapping.json \
  --output_dir output/results/coco_val2017_modular \
  --work_dir output/artifacts/coco_val2017_modular \
  --modes plain sis_only sis_server_naive sis_client_dealer_free sis_client_partial sis_mpc \
  --max_queries 500 --bands 8 --k 3 --n 5 --force

> AES ファイルの全件復号ベースライン（M1）を回すには `--modes aes_gcm` を追加し、`PHASH_AES_MASTER_KEY` を環境変数で渡してから実行してください。`PHASH_AES_MASTER_KEY` は Base64 化した 256-bit キーです（dotenv や `python -c` で生成可能）。
```

- Stage-1 は全モード共通でインデックス照合。  
- Stage-2 はモードごとに再構成/距離処理を分けた上で同じ結果フォーマットにまとめる。  
- `sis_client_dealer_free` / `sis_client_partial` では再構成を分割し、`sis_mpc` では再構成を完全にスキップして MPC で距離だけ算出。これが dealer-free の意義。  
- 実行中の shared store は `output/artifacts/...` に置き、結果（CSV/JSON）は `output/results/...` に集約することで、「何を出したか」が一目で分かる。
- ステージ1の別軸としては `minhash_lsh` モード（`pip install datasketch`）を組み込み、LSH による候補削減 + Hamming 距離を既存 metrics に流し込めるので、Stage-1/Stage-2 の候補減少比と通信量差を MinHash で比較できます。

### 3.3 図版出力

```bash
PYTHONPATH=./src:. python -m experiments.common.plotting \
  output/results/coco_val2017_modular/metrics.csv \
  --output_dir output/figures/coco_val2017_modular
```

- `precision_summary`, `time_breakdown`, `communication_breakdown`, `candidate_reduction` など主要な指標を `output/figures/...` に出力。  
- Stage-1/Stage-2 という呼び方を図の凡例にも適用し、chapter-level report で Stage-1 の候補削減 vs Stage-2 の secure evaluation という対比が伝わるようにする。
- `experiments/scripts/plot_selected_variants_recall.py ... --short-labels` を使えば、x 軸に `DF`/`CP`/`SN`/`SO` の略称を表示して、モード名が長くても潰れない図が出来ます。
- `metrics_stats.py` を併用すれば、`total_ms` などの平均・標準偏差・95% CI を `output/results/metrics_summary.csv` に出力し、図の注釈やスライドに「ばらつき」を入れられます。
- その `metrics_summary.csv` を `experiments/scripts/plot_metrics_with_ci.py` でプロットすると、mean±CI の error bar 図（`output/figures/metrics_ci/*.png`）が得られますので、グラフでもばらつきが示せます。
- `python -m experiments.common.plotting ...` に `--exclude-mode aes_gcm` などを渡せば、全ての図から特定モードを除外して描けます。AES-GCM だけ別グラフ・他は別に調べたいときに活用してください。  
- `plot_time_log_scale.py` も併用して AES を除いた log スケール図を得ると比較が崩れずにすみます。
- 特に AES-GCM は他と桁違いに遅いので、`python experiments/scripts/plot_time_log_scale.py output/results/coco_val2017_modular/metrics.csv --output-dir output/figures/log_latency` のように `aes_gcm` を除外した log スケール図も生成して比較してください。
- FHE の遅延を示すには `python experiments/scripts/fhe_distance_demo.py` を走らせ、`pyfhel` を使って 64bit pHash の距離計算にかかる時間を記録して `output` にまとめるのも手です。  
- `PYTHONPATH=./src:. python experiments/scripts/fhe_distance_demo.py --iterations 3` で、FHE 相当の bitwise 演算 + ループを使って遅延を測定する実験が動きます。
- `--output-csv` オプションを指定すると `mode=sis_fhe` 形式の metrics.csv が生成されるので、既存のプロットと同様に `experiments/common/plotting` へ流せます。

### 3.4 dealer-free 特化

- dealer-free 実験は `python -m dealer_free_sis` で実行。`--mpc-query-image` オプションなどで Stage-2 の MPC 影響を JSON化して `output/figures/dealer_free/*` に書き出す。  
- `sis_client_dealer_free`/`sis_client_partial` の挙動は Stage-2 における再構成範囲を制御することで、どの程度 dealer-free に近づけるかを調べる手段になっている。

---

## 4. まとめ

このフローを“燃料供給→点火前チェック”として通し、`prepare_coco.py`→`run_search_experiments.py`→`experiments.common.plotting`→レポート作成という順に進めてください。Stage-1/Stage-2 の呼称、`sis_server_naive`～`sis_client_partial`～`sis_mpc` という dealer-free対策を含むモード体系、そして成果物の `output/` 集約によって、研究の意図と再現性が両立する構造になっています。次はこの流れをスクリプトやジョブ定義として自動化するか、COCO全体に拡張するステップに移ってください。
