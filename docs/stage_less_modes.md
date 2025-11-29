# Stage-less Modes

この実験セットに含まれるモードのうち、Stage-1/Stage-2 の典型的な分割を持たない「Stage-less」モードについて整理します。

| モード | Stage-1 | Stage-2 | 説明 |
|--------|---------|---------|------|
| `plain` | データ登録時に pHash をそのまま保持（Stage-1 では簡略化） | 各クエリで全件距離計算 | ベースライン。全ての pHash を照合する最も速いモードだが、セキュリティは最も低い。 |
| `aes_gcm` | なし（画像は丸暗号化して保存） | 毎回全件復号 → pHash 計算 | AES-GCM による丸暗号化ベースライン。Stage-1 が存在しないため通信・計算が極めて重い。 |
| `sis_fhe`（`fhe_distance_demo.py`） | なし（pHash を暗号化したまま） | FHE 相当の bitwise 演算で距離を算出 | FHE の実運用の重さを強調する PoC。Stage-1/Stage-2 がないため、単独での遅延評価に使う。 |

これらのモードは `run_search_experiments.py --modes ...` から直接試せる `plain` を除けば、`aes_gcm` と `sis_fhe` が Stage-1/Stage-2 のフローに属さない設計です。スライド等で「Stage-1/Stage-2 を持たないモード群」として1枚にまとめる場合は、本ファイルの表を参考にしてください。
