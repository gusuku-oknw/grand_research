## セキュリティ関連の改善まとめ

このリポジトリに最近加えた主な改善点と、設定方法・制約を簡潔にまとめます。

### 1. HMAC 鍵管理
- **CSPRNG 生成 & 永続化**: バンドごとの HMAC 鍵を `secrets` で生成し、`meta_dir/hmac_keys.json` に保存（k/n/bands が変わると読み込みエラー）。
- **環境変数での供給**: Base64 化した JSON (`{server: [hex,...]}`) を環境変数に渡す `key_env_var` をサポートし、ファイル保存を回避可能。
- **注意**: 鍵は平文で保存されるため、KMS/秘密管理・ファイル暗号化・権限分離は別途必要。ローテーション手順も要検討。

### 2. Stage-1 アクセスパターン緩和
- **ダミー照会**: `dummy_band_queries` で各バンド/サーバごとの追加照会回数を挟み、トークンの照会頻度を均一化します。
- **固定回数パディング**: `pad_band_queries` を使うと、実トークン＋ダミーの合計を固定し、照会回数のバラつきを抑えます。
- **固定長バッチ**: `fixed_band_queries` でさらに厳密に本番クエリと完全同数のバッチを送信し、タイミングやサイズの差分を消します。
- **VOPRF (Ristretto) 対応**: `--use_oprf` で Stage-1 トークンを VOPRF で生成することで、バンド値を復号せずに候補一致を行えます。
- **鍵暗号化保存**: 上記設定を保存する HMAC/OPRF 鍵は AES-GCM で暗号化しておき、本番環境では KMS や env 変数から復号します。
- **限界**: VOPRF とパディングはアクセスパターンを緩和するだけで、実際の匿名化には TEE/TEE+シークレットチャネルが必要になる場合があります。

### 3. シェア不足の扱い
- fusion 以外のモードで `servers < k` の場合、距離計算/復元をスキップして明示メッセージを返し、例外クラッシュを防止。

### 4. 実行・確認用スクリプト
- `experiments/scripts/run_demo_tests_summary.py`: `tests/fixtures` を使った最小デモを実行し、結果を `output/demo_tests/summary.json` に保存（設定・pHash・所要時間・検索/復元結果を含む）。
- CLI (`demos/demo_phash_sis.py`) から `--dummy_band_queries`, `--pad_band_queries` でパディング設定を指定可能。鍵は `meta_dir/hmac_keys.json` または環境変数経由で供給。

### 5. テスト実行（tests/fixtures）
- コマンド: `python experiments/scripts/run_demo_tests_summary.py`（デフォルトで `fixed_band_queries=4`, `dummy_band_queries=1` を適用）
- 出力: `output/demo_tests/summary.json` に設定・pHash・所要時間・検索/復元結果を記録。`output/demo_tests/recon/` に復元画像を保存。

### 5. 未解決の課題（要追加設計）
- **アクセスパターン秘匿**: OPRF 化（VOPRF 等）、固定長バッチ/ダミーをプロトコル化、サーバ集合を固定化、TEE/SGX 内で Stage-1 実行。現状は固定長パディングのみ。
- **鍵保護/ローテーション**: KMS 連携、鍵ファイル暗号化と ACL、鍵バージョン管理と再インデックス手順の整備。現状は AES-GCM での保存と env 供給止まり。
- **改ざん検知の強化**: 現状は MAC 検証を追加済みだが、署名や監査ログ、ハッシュチェーン等でより強い検知を検討。

### 6. 今後の具体的改善案
- **Stage-1 の根本対策**: OPRF ベースのトークン生成、固定長バッチ照会（全クエリ同数・同順序・ダミー混入必須）、サーバ集合の固定化、TEE/SGX での照合。
- **鍵管理の実運用化**: KMS 取得レイヤーを追加し、鍵バージョンをメタに記録。ローテーション用の自動化スクリプトと手順書を整備。アクセスを監査ログに記録。
- **完全性/改ざん検知**: シェア/メタへの署名または MAC に加え、検証結果のログ化と、破損検出時のリカバリ手順を用意。

### 7. TEE (SGX/TrustZone) 導入に向けたメモ
- 目的: Stage-1（トークン生成・照合）を enclave 内で実行し、アクセスパターンや鍵をホスト OS から隔離する。
- 想定フロー: クライアントが固定長バッチでトークンを送付 → enclave 内で鍵を用いて照合 → 閾値判定結果のみをホストに返す。距離計算は従来どおり Stage-2 側の MPC/TEE に委譲。
- 移行手順（案）:
  1) SGX/TrustZone のサンプル enclave で AES/HMAC/VOPRF を動作確認する。
  2) バンド照合ロジックを enclave に移植し、gRPC/Unix ドメインソケットで呼び出す薄いプロキシを挟む。
  3) 固定長バッチをプロトコル仕様として固定し、ダミーを enclave 側で強制生成・消費する。
  4) リモートアテステーションで enclave が正当なバイナリで動作していることを確認し、鍵はアテステーション後にデリバリする。
- 本リポジトリでは TEE 実装は未提供。上記の方針に沿った PoC/テストは別途環境依存で実施してください。

### 8. Threat Model & Leakage (Draft)
- **Assets**: Query pHash bands, HMAC/OPRF keys, image shares, reconstruction outputs.
- **Adversary**: Honest-but-curious servers observing access patterns and stored tokens/shares; host OS compromise (no TEE); network observer.
- **Leakage points**:
  - Stage-1 access pattern (which band tokens hit): mitigated by VOPRF (hides band value) + fixed-length batches + dummies; still observable traffic size/shape if not padded uniformly.
  - Token/key exposure: mitigated by AES-GCM–encrypted key files or env/KMS; full protection requires TEE or strict ACL.
  - Stage-2 partial reconstruction: leaks pHash to the party doing reconstruction; skip in MPC mode to avoid.
  - Stage-2 distance: in MPC mode avoids plaintext pHash; in standard mode pHash is reconstructed.
  - Reconstruction outputs: only top results, k-of-n; still plaintext images—must be access-controlled.
- **Trust assumptions**:
  - Without TEE: servers can see traffic pattern; VOPRF hides values but not access pattern length if unpadded.
  - With TEE (future): enclave isolates keys/tokens; remote attestation required before key delivery.
- **Residual risks**: Traffic analysis if padding not enforced; key theft if key storage/ACL is weak; integrity loss if MAC/签名 not enforced everywhere; replay unless request IDs/timestamps are checked (not yet implemented).

### 9. Protocol Draft (Fixed-Length + VOPRF + Key Management)
- **Stage-1 (client/server split)**:
  1) Client computes pHash, splits into bands.
  2) For each band, client blinds with VOPRF and sends **fixed-length batch** of tokens (1 real + dummies) to a fixed server set.
  3) Server (or TEE) evaluates blinded tokens; client unblinds and filters candidates by vote threshold.
  4) Batch size, ordering, and server set are constant across queries to hide access pattern length.
- **Stage-2**: Optional partial pHash reconstruction for candidates only; in MPC mode skip to avoid leakage.
- **Stage-2**: Distance via MPC/TEE/plain (configurable). Reconstruction only for Top-K that satisfy k-of-n.
- **Key management**:
  - HMAC/OPRF keys fetched from KMS (preferred) or env; on-disk keys are AES-GCM encrypted with short-lived KMS token.
  - Keys carry a **version**; rotation requires reindex or dual-read period. Versions stored in metadata.
  - Remote attestation (TEE) gates key delivery; audit log records key access and rotation events.
- **Padding policy**:
  - `fixed_band_queries` dictates batch size per band/server; dummies are generated server-side if needed to enforce constant load.
  - All queries use the same server subset to avoid server-selection leakage.

### 10. TEE PoC Plan (Draft)
- **Scope**: Move Stage-1 token evaluation into an enclave; keep Stage-2 optional (MPC/TEE/plain); Stage-2 optional/skip for MPC.
- **Steps**:
  1) Build SGX/TrustZone enclave with AES/HMAC/VOPRF primitives; validate with unit tests inside enclave.
  2) Expose a minimal API (e.g., gRPC/UDS) for fixed-length token batches; enforce padding and dummy generation inside enclave.
  3) Implement remote attestation; only after attestation deliver HMAC/OPRF keys (from KMS) into enclave.
  4) Add audit logging for enclave requests and key loads; include request IDs/timestamps to mitigate replay.
  5) Benchmark latency/throughput with COCO derivatives; compare vs non-TEE baseline.
- **Open items**: Host<->enclave data marshalling, enclave memory limits for batch size, attestation flow per platform, integration with KMS.
