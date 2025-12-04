# 図と対象モードの一覧

| 図 | 元データ | 対象モード | 備考 |
|----|----------|------------|------|
| `precision_summary.png` | `metrics.csv` | `plain`, `sis_only`, `sis_server_naive`, `sis_client_dealer_free`, `sis_client_partial`, `sis_mpc`, `aes_gcm`, `minhash_lsh` | Precision@k を比較。`--exclude-mode` で AES/GCM などを削る。 |
| `time_breakdown.png` | 同上 | 同上 | Stage-1/Stage-2 の処理時間。AES を含めれば重いベースラインが分かるが、`--exclude-mode aes_gcm` で拡大可。 |
| `communication_breakdown.png` | 同上 | 同上 | クエリごとのバイト数。 |
| `candidate_reduction.png` | 同上 | 同上 | Stage-1 と Stage-2 の候補数比較。 |
| `latency_log_scale.png` | 同上 | `plain`, `sis_only`, `sis_server_naive`, `sis_client_dealer_free`, `sis_client_partial`, `sis_mpc`, `minhash_lsh`（AES を除外） | `plot_time_log_scale.py` で出力するログスケール版。 |
| `selected_recall.png` | `metrics.csv`（`--exclude-mode aes_gcm`） | 上記モード | `{original, jpeg70, crop5%, rotate10%}` の Recall@10。`--short-labels` で DF/CP/SN/SO 表示可。 |
| `total_ms_ci.png` | `metrics_summary.csv` | 同じまとめ | `total_ms` の平均±95%CI。 |
| `stage2_ms_ci.png` | 同上 | 同上 | Stage-2 時間の CI。 |
| `stage2_bytes_ci.png` | 同上 | 同上 | Stage-2 通信 bytes の CI。 |

そのほか（precision_latency, candidate_reduction_ratio, reconstruction_ratio）も同じ `metrics.csv` を使って全モードを描けるので、必要に応じて `--exclude-mode` で AES-GCM などを外してください。
