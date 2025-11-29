# Figure Output Reference / 図表リファレンス

This note explains every figure emitted by `experiments/scripts/run_search_experiments.py` and `python -m evaluation.plotting`. Keep it nearby when you review `metrics.csv` or compile reports.  
このドキュメントは `experiments/scripts/run_search_experiments.py` と `python -m evaluation.plotting` が生成する図表の内容と目的を整理したものです。`metrics.csv` の検証やレポート作成時に参照してください。

### Metric Glossary / 指標の用語集
- **P@k (Precision@k)**: 上位 k 件の候補のうち正解が含まれる割合。例として P@1 は「最上位が正解か」、P@10 は上位10件のうち正解率がどれだけあるかを示します。  
  Share of relevant items within the top-k ranked results; P@1 captures whether the very first item is relevant, while P@10 averages relevance across the top ten.
- **R@k (Recall@k)**: 上位 k 件までに正解が何件含まれているかを、正解の総数で割った値。正解を取り逃していないか、カバレッジを評価します。  
  Fraction of all relevant items that appear within the top-k results; measures coverage and missed positives.
- **mAP (mean Average Precision)**: 各クエリごとの Average Precision（正解を見つけるたびの精度を順位で平均した値）を、すべてのクエリで平均した指標。ランキング全体で早い段階に正解を提示できているほど高くなります。  
  Dataset-wide mean of per-query average precision; rewards rankers that surface relevant items near the top.

## precision_summary.png
- **Source / 出力元**: `python -m evaluation.plotting ...`
- **Contents / 内容**: Bar chart of mean P@1, P@5, P@10, R@10, and mAP for each retrieval mode.  
  各検索モードの平均 P@1・P@5・P@10・R@10・mAP を棒グラフで比較します。
- **Purpose / 目的**: Demonstrates that the Stage-A/B/C pipeline retains accuracy relative to plain pHash.  
  Stage-A/B/C を導入しても pHash 単体と同等の精度・再現率を維持できているか確認します。

## candidate_reduction.png
- **Source / 出力元**: `experiments/scripts/run_search_experiments.py` と `evaluation.plotting` の双方
- **Contents / 内容**: Line plot of average candidate counts after Stage-A, Stage-B, Stage-C (per mode).  
  各ステージ通過後の平均候補数をモード別に線グラフ化します。
- **Purpose / 目的**: Visualises how aggressively each stage prunes the search space.  
  段階的フィルタが候補集合をどの程度削減できているか示します。

## time_breakdown.png
- **Source / 出力元**: `experiments/scripts/run_search_experiments.py` と `evaluation.plotting`
- **Contents / 内容**: Stacked bar chart of mean per-query latency split into pHash, Stage-A, Stage-B, Stage-C.  
  クエリ処理時間を pHash 計算と各 Stage の内訳ごとに積み上げ棒グラフで表示します。
- **Purpose / 目的**: Highlights latency hotspots and verifies Stage-A/B/C is faster than full reconstruction.  
  遅延のボトルネックを特定し、段階的復元がフル復元より高速か確認します。

## communication_breakdown.png
- **Source / 出力元**: `python -m evaluation.plotting ...`
- **Contents / 内容**: Stacked bar chart of average bytes transferred during Stage-A/B/C per query.  
  クエリごとに各ステージで消費した通信バイト数を積み上げ棒グラフで表示します。
- **Purpose / 目的**: Quantifies bandwidth savings versus reconstructing every candidate.  
  全候補を復元する場合との通信量削減効果を数値化します。

## precision_latency.png
- **Source / 出力元**: `python -m evaluation.plotting ...`
- **Contents / 内容**: Scatter plot of total latency vs. Precision@10 for every query/mode pair.  
  モード別に総処理時間と Precision@10 を散布図で配置します。
- **Purpose / 目的**: Examines the time–accuracy trade-off and pinpoints Pareto-dominant modes.  
  時間と精度のトレードオフを評価し、有利なモードを特定します。

## variant_recall.png
- **Source / 出力元**: `experiments/scripts/run_search_experiments.py` と `evaluation.plotting`
- **Contents / 内容**: Grouped bar chart of Recall@10 by transform variant (JPEG quality, rotation, noise, etc.).  
  JPEG 圧縮率・回転・ノイズなどの派生画像ごとに Recall@10 を棒グラフで比較します。
- **Purpose / 目的**: Checks robustness of each mode against common perturbations.  
  各モードが変換耐性をどの程度維持しているか確認します。

## candidate_reduction_ratio.png
- **Source / 出力元**: `experiments/scripts/run_search_experiments.py` と `evaluation.plotting`
- **Contents / 内容**: Line plot of candidate ratios (Stage-A/B/C counts ÷ dataset size).  
  各ステージの候補数をデータセット総数で割り正規化した値を線グラフ化します。
- **Purpose / 目的**: Enables cross-dataset comparison of pruning efficiency.  
  データセット間で削減率を比較できる指標として利用します。

## reconstruction_ratio.png
- **Source / 出力元**: `experiments/scripts/run_search_experiments.py` と `evaluation.plotting`
- **Contents / 内容**: Bar chart of (reconstructed images ÷ total queries) per mode with a 1.0 baseline.  
  モード別に再構成画像数／総クエリ数を棒グラフで示し、常時復元 (=1.0) と比較します。
- **Purpose / 目的**: Measures how selective reconstruction reduces full image recovery workload.  
  選択的復元が復元コストをどれだけ削減するか評価します。

## roc_<mode>.png / pr_<mode>.png
- **Source / 出力元**: `experiments/scripts/run_search_experiments.py`（モードごとに生成）
- **Contents / 内容**: Mean ROC curves (TPR vs FPR) and precision–recall curves with AUC annotations.  
  平均 ROC 曲線および Precision-Recall 曲線をモードごとに描画し、AUC を併記します。
- **Purpose / 目的**: Evaluates discriminative quality of distance thresholds for every configuration.  
  距離閾値設定の識別性能を把握し、モード間の差異を比較します。

## tau_sensitivity.png
- **Source / 出力元**: `experiments/scripts/run_search_experiments.py` と `evaluation.plotting`（`roc_pr_summary.json` を入力）
- **Contents / 内容**: Line plot of mean recall versus tau threshold for each mode.  
  各モードの平均 Recall と tau 閾値の関係を線グラフで表示します。
- **Purpose / 目的**: Guides selection of secure-distance thresholds that maintain recall targets.  
  セキュア距離モードで目標再現率を保つ tau 設定を決める指針になります。

## hist_<variant>.png
- **Source / 出力元**: `experiments/scripts/run_search_experiments.py`（派生ごとに1枚）
- **Contents / 内容**: Histogram of Hamming distances between query and database samples for a variant.  
  クエリとデータベースのハミング距離分布を変換別にヒストグラムで示します。
- **Purpose / 目的**: Visualises distance distributions to justify threshold choices and detect overlap.  
  閾値設定の妥当性やクラス間の重なりを確認するための材料になります。
