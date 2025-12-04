# プライバシー保護型類似画像検索における比較モード総まとめ  
（pHash × SIS × Dealer-free × MPC）  
---

本資料は卒業研究向けに、  
**pHash を使わない方式 → pHash を使った既存方式 → 本研究の提案方式**  
を体系的に比較できるよう Markdown 形式で整理したものである。

医療画像や機密画像が「SISで分散保存されている」というユースケースを前提とする。

---

# 1. 比較対象モード一覧（全9モード）

## 表1. pHash なし → pHash ありの総合比較

| 番号 | モード名 | pHash | SIS | DF（Dealer-free） | MPC | 類似検索可否 | 概要 |
|------|----------|--------|------|-------------------|------|--------------|-------|
| **M0** | SIS-only（No-index） | × | ○ | × | × | × | 最悪ケース。検索速度が壊滅的。 |
| **M1** | AES暗号化画像 | × | × | - | × | × | 復号なしでは検索不可能。 |
| **M2** | FHE距離計算 | × | × | - | △ | △（極遅） | 理論上可能だが実用不可。 |
| **M3** | MinHash + LSH（平文） | × | × | - | × | ○ | 高速だがプライバシーゼロ。 |
| **P0** | plain-pHash | ○ | × | - | × | ○ | 速度・精度のベースライン。 |
| **S1** | sis_server_naive | ○ | ○ | × | × | ○ | 従来型SIS。ディーラー問題あり。 |
| **DF1** | sis_client_dealer_free | ○ | ○ | ○ | × | ○ | 本研究の中心方式その①。 |
| **DF2** | sis_client_partial | ○ | ○ | ○ | × | ○ | 高速化版（部分復元）。 |
| **MPC1** | sis_MPC_full | ○ | ○ | ○ | ○ | ○ | pHash 非復元で最強セキュリティ。 |

---

# 2. 各モードの概要と図解用説明文

卒論の図として利用できる「説明文」をすべて掲載。

---

## M0. SIS-only（No-index）
### 説明文
画像は SIS により分散保存されるが、pHash などの特徴が存在しないため、類似検索は全件復元もしくは全件MPCしか手段がない。したがって検索時間は実用不可能レベルとなる。
実験では `sis_only` モードを使い、Stage-1 をスキップして全件復元/比較を行う形でこの悲惨さを再現します。

---

## M1. AES暗号化画像
### 説明文
画像ファイルが AES 暗号化されて保存される。暗号文から類似検索を行うことはできず、復号が必要となるため、プライバシー要件を満たさない。
### 実装ノート
`run_search_experiments.py` の `aes_gcm` モードでは、`PHASH_AES_MASTER_KEY` を Base64 で環境変数に設定し、画像を AES-GCM で丸暗号化して一度記録したうえで、検索時に全件復号して pHash を計算します。  
```bash
PHASH_AES_MASTER_KEY=$(python - <<'PY'
import base64, os
print(base64.b64encode(os.urandom(32)).decode())
PY)
PYTHONPATH=./src:. python experiments/scripts/run_search_experiments.py --modes aes_gcm --max_queries 20
```
とすれば、暗号化されたデータセットに対して「インデックスなしはここまで遅い」ことが定量的に示せます。

### M2 実験ノート
`experiments/scripts/fhe_distance_demo.py` は本格的な Pyfhel FHE ではなく、FHE 並みに遅い bitwise 演算 + 消費ループで `64bit pHash` の距離計算にかかる時間を模擬します。実行例：
```bash
PYTHONPATH=./src:. python experiments/scripts/fhe_distance_demo.py --iterations 3
```
各試行で `plain` と `simulated FHE` の距離/時間を出力するので「FHE ではこのくらい遅い」という定性を示せます。  
`--output-csv output/results/fhe_demo/metrics.csv` を付けると `mode=sis_fhe` の metrics.csv が生成されるため、他モードと同じグラフに含められます。
## M2. Homomorphic Encryption（FHE）距離計算
### 説明文
画像を FHE 暗号化し暗号状態で距離計算を行う方式。理論的には類似検索が可能だが、医療画像の解像度では1件あたり数秒〜数十秒かかり、実運用は困難。

---

## M3. MinHash + LSH
### 説明文
画像を平文特徴に変換し、MinHash と LSH により近似検索を実施する。高速だが、特徴が平文であるためプライバシー保護要件を満たさない。

### 実装ノート
`datasketch` の `MinHash`/`MinHashLSH` を使って、pHash の 1 ビット集合を MinHash 化し LSH に登録した上で候補を Stage-1 で絞り込み、Stage-2 で pHash Hamming 距離を計算します。  
```bash
pip install datasketch
PYTHONPATH=./src:. python experiments/scripts/run_search_experiments.py --modes minhash_lsh --max_queries 200
```
このモードは LSH の候補数と Stage-2 の Hamming 時間を同じ metrics.csv に書き出すので、既存プロット (precision/time/bytes) に `minhash_lsh` を含めれば「MinHash も候補を絞っている」ことが可視化できます。

---

## P0. plain-pHash
### 説明文
サーバが pHash を平文で保持する方式。高速かつ高精度であり、速度・精度のベースラインとなるが、セキュリティが最も低い。

---

## S1. sis_server_naive（従来型SIS）
### 説明文
pHash をサーバが計算し SIS で分散する。サーバが pHash の平文を閲覧可能なため、ディーラーフリーではない。検索時には候補の pHash を復元する必要がある。

---

## DF1. sis_client_dealer_free（提案方式1）
### 説明文
クライアント側で pHash を計算し SIS で分散。サーバは pHash を一切閲覧できない。インデックスは HMAC により構築され、検索時は必要な候補だけ復元する server-side dealer-free 方式。

---

## DF2. sis_client_partial（提案方式2）
### 説明文
DF1 に加え、検索時に pHash の一部のみ復元して近似距離を計算し候補を絞り込む。通信量と計算量を削減しつつ高い精度を維持する高速化アプローチ。

---

## MPC1. sis_MPC_full（提案方式3）
### 説明文
pHash は一切復元されず、SIS シェアのまま MPC によって距離計算のみ行う方式。セキュリティが最も高いが、計算コストが最も大きい。

---

# 3. 共通処理フロー（全モード共通仕様）

比較実験の公平性を保つため、どの方式でも外部仕様は統一する。

---

## 3.1 Stage-1（インデックス検索）
- pHash を 8 バンドに分割  
- 各バンドの equality を鍵付き HMAC または平文で比較  
- 一致バンド数 ≥ τ のものを候補 C1 とする  
- 上位 N 件に絞り Stage-2 へ送出

---

## 3.2 Stage-2（距離計算）
- 入力：候補 C1、クエリ pHash  
- 出力：Hamming距離に基づく上位 K 件  
- 実装はモードにより差異：  
  - plain：平文距離  
  - SIS：復元後距離  
  - MPC：復元なしで Hamming 距離
- Stage-2 は従来 Stage-B/Stage-C に相当する再構成と距離評価をまとめたものであり、どのモードでも同じ出力フォーマットに整合することが評価の前提となる。
- Stage-2 だけでなく Stage-1 を持たないモード（`plain`, `aes_gcm`, `sis_fhe` など）については [`docs/stage_less_modes.md`](stage_less_modes.md) にまとめています。

---

# 4. 評価実験設計（卒論 4 章用）

---

## 4.1 評価目的
SISで分散保存された機密医療画像に対して、  
pHash × SIS による検索可能化がどの程度有効かを、  
精度・性能・通信量・セキュリティの観点から総合的に評価する。

---

## 4.2 評価対象モード（推奨 6 モード）

| 採用モード | 理由 |
|------------|------|
| **M0** SIS-only | pHashなしの悲惨さを示す |
| **M3** LSH | pHash以外の高速手法との比較 |
| **P0** plain | ベースライン |
| **S1** sis_server_naive | 従来SIS |
| **DF1** sis_client_dealer_free | 本研究の中心 |
| **MPC1** sis_MPC_full | 最強セキュリティ |

（必要であれば DF2 も追加）

---

## 4.3 データセット  
- COCO2017 val + 医療画像のサブセット  
- 変換（類似画像生成）  
  - jpeg85 / jpeg70  
  - crop5%  
  - rotate10°  
  - noise10%  

---

## 4.4 評価項目

### 精度
- Precision@1,5,10  
- Recall@10  
- **Recall@10（重要な変換のみ）**  
  `experiments/scripts/plot_selected_variants_recall.py output/results/coco_val2017_modular/metrics.csv` を実行すると `original`, `jpeg70`, `crop5%`, `rotate10%` などのキー変換だけで Recall を比較できます。  
  スライドに貼るときは `--short-labels` を付けると `DF/SN/CP/SO` などの略称が x 軸に入り、文字が潰れにくくなります。
- mAP

### 速度
- Stage1（インデックス検索時間）  
- Stage2（距離計算時間）  
- **標準偏差/信頼区間**  
  `experiments/scripts/metrics_stats.py` で `output/results/metrics_summary.csv` を出力し、`experiments/scripts/plot_metrics_with_ci.py output/results/metrics_summary.csv` を実行すると、`total_ms`/`stage2_ms`/`stage2_bytes` のmean±95%CI を error bar 付き PNG にできます。

### 通信量
- 登録時（シェア + トークンサイズ）  
- 検索時（復元 or MPC 通信量）

### セキュリティ
- pHash 平文露出の範囲  
- ディーラーフリー性  
- 許容リーク（バンド一致／距離のみ）

---

## 4.5 仮説

1. **M0**：壊滅的に遅い  
2. **M3**：高速だが安全性ゼロ  
3. **P0**：最高速・最高精度  
4. **S1**：pHash漏洩リスクあり  
5. **DF1**：精度は plain と同等、安全性は高い  
6. **MPC1**：最も安全だが最も重い  

---

# 5. 結論（卒論用まとめ）

本比較は、  
**「pHashを使わない場合 → pHash平文 → 従来SIS → 提案（DF/MPC）」**  
という階層的評価を可能にし、  
提案手法の必要性・有効性・安全性を総合的に示す構成となる。  

詳細な図版と各モードのカバレッジについては `docs/graph_mode_summary.md` に一覧化しています（図の出力先や対象モードも記載）。

補足パッケージ:
- `phash_masked_sis`: pHash と整合する「マスク画像」を低層シークレットに、本物画像を高層シークレットに置く 2 階層 SIS のミニパッケージ。

---

# 6. 今後追加可能な付録

- モード別の数理モデル  
- リークモデル比較表  
- 評価用グラフテンプレ  
- 卒論全体の章立てテンプレ  

必要に応じて提供可能。

---
