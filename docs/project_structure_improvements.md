# プロジェクト構造の改善提案（研究ゴールに基づく再設計）

## 1. はじめに

現在のプロジェクト `README.md` を分析した結果、このプロジェクトが以下の2つの主要な目的を持っていることが確認できました。

1.  **`sis_image` という再利用可能なPythonライブラリの開発:** 知覚ハッシュ、秘密分散、検索可能インデックスなどのコア機能を提供する。
2.  **体系的な研究実験の実施と評価:** 複数の検索モード（`plain`, `sis_naive`, `sis_selective`, `sis_staged`, `sis_mpc`）を異なる条件下で実行し、性能やプライバシー特性を比較評価する。

この2つの目的を明確に分離し、`README.md` に記載されているワークフローを直感的に反映させるため、以下のディレクトリ構造を提案します。

## 2. 提案するディレクトリ構造

```
C:\Users\tmkjn\PycharmProjects\Grand_Research\
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── README.md  # プロジェクトへの簡単な導入と、各ディレクトリへの案内
│
├── src/
│   └── sis_image/      # インストール可能なPythonパッケージのソースコード
│       ├── __init__.py
│       ├── common/         # 共通コンポーネント (phash, utilsなど)
│       │
│       ├── dealer_based/   # 従来のディーラー有りアーキテクチャ
│       │   ├── shamir.py
│       │   └── workflow.py
│       │
│       └── dealer_free/    # ディーラーフリーアーキテクチャ
│           ├── secret_sharing.py
│           └── workflow.py
│
├── experiments/            # 研究実験に関するすべてのスクリプトと設定
│   ├── configs/            # 実験設定ファイル
│   ├── modes/              # 比較対象となる各検索モードの実装
│   │   ├── base_runner.py  # 全モード共通のインターフェース
│   │   ├── plain.py
│   │   ├── sis_naive.py
│   │   ├── sis_selective.py
│   │   ├── sis_staged.py
│   │   └── sis_mpc.py
│   │
│   ├── prepare_data.py     # データセット準備スクリプト
│   ├── run_experiments.py  # 実験実行スクリプト
│   └── analyze_results.py  # 結果のプロット・分析
│
├── demos/                  # READMEで紹介されているデモスクリプト
│   ├── demo_k_sweep.py
│   └── ...
│
├── data/                   # 実験の元となる入力データ
│   └── coco2017/
│
├── output/                 # すべての生成物を格納する統一ディレクトリ
│   ├── datasets/           # 準備された派生データセット
│   ├── artifacts/          # 実験の中間生成物
│   ├── results/            # 実験結果のメトリクス (metrics.csvなど)
│   ├── figures/            # 生成されたグラフ
│   └── reports/            # 生成されたレポート
│
└── docs/                   # 静的なドキュメント
    ├── README.md           # 詳細なプロジェクト説明 (現在のルートのREADME.mdを移動)
    └── ...
```

## 3. 各コンポーネントの詳細

### `src/sis_image` (コアライブラリ)

*   **目的:** 再利用可能なパッケージとして、プロジェクトのコア機能を提供します。
*   **構造:** 
    *   `dealer_based` と `dealer_free` のように、根本的なアーキテクチャの変種をサブパッケージとして明確に区別します。これにより、ライブラリ利用者はどちらの設計を使うかを選択できます。
    *   CLIの機能 (`sis-image` コマンド）もここに含めます。

### `experiments` (研究実験)

*   **目的:** `README.md` に記載されている比較実験のワークフローを体系的に管理します。
*   **構造:** 
    *   `experiments/modes/`: `README.md` で比較対象とされている **`plain`, `sis_naive`, `sis_selective`, `sis_staged`, `sis_mpc`** といった各検索戦略の実装をここに配置します。これらはすべて共通の `base_runner.py` インターフェースを実装し、交換可能にします。
    *   `run_experiments.py`: 実験のオーケストレーターです。以下のように、評価したい「アーキテクチャ」と「検索戦略（モード）」を引数で組み合わせて実行できる設計を想定しています。

      ```bash
      # 例: ディーラー有りモデルで、selectiveモードとmpcモードを実行
      python experiments/run_experiments.py --architecture dealer_based --modes sis_selective sis_mpc

      # 例: ディーラーフリーモデルで、mpcモードを実行
      python experiments/run_experiments.py --architecture dealer_free --modes sis_mpc
      ```

## 4. この構造の利点

1.  **関心の分離:** 「ライブラリ (`src`)」と「実験 (`experiments`)」が明確に分離され、それぞれ独立して開発・変更しやすくなります。
2.  **`README.md` との整合性:** `README.md` に記述されたパッケージ構造、比較モード、実験フローが、そのままディレクトリ構造に反映され、直感的になります。
3.  **拡張性:** 新しいアーキテクチャ（例: `trusted_hardware`）や新しい検索モードを、それぞれ `src/sis_image/` や `experiments/modes/` に追加するだけで、容易に実験の組み合わせを拡張できます。
4.  **管理の容易さ:** すべての出力が `output/` に集約されるため、成果物の管理が簡素化されます。