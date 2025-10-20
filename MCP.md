# MCP概要

## 背景とタイムライン
Model Context Protocol（MCP）は、Anthropicが2024年11月25日に公開したオープンスタンダードであり、AIアシスタントが外部データやツールへ統一的にアクセスできるよう設計されています。citeturn1news15
最新仕様は2025年6月18日にリリースされ、構造化されたツール出力、OAuthによる認証強化、対話型エリシテーションなどが追加されました。citeturn1search0
次期仕様は2025年11月11日にリリース候補版、11月25日に正式版が予定され、14日間の互換性検証期間が設定されています。citeturn1search0

## アーキテクチャと基本概念
MCPはJSON-RPC 2.0を基盤に、ホスト・クライアント・サーバーの3役が能力交渉、セッション管理、リソース・プロンプト・ツールの提供を行う対話プロトコルを定義します。citeturn0search18
設計思想はLanguage Server Protocolに着想を得ており、拡張可能なモジュール構成と必須のトランスポート要件を両立させています。citeturn0search18

## ガバナンスとロードマップ
2025年以降、MCPコミュニティは仕様変更を審査するSpecification Enhancement Proposal（SEP）とマイルストーン管理を正式化し、ステュワード役が仕様品質を監督する体制を整えました。citeturn1search0
リリース候補期間を制度化することで、SDKメンテナーやクライアント実装者が差分を事前検証できるよう配慮されています。citeturn1search0

## エコシステムと採用動向
Anthropicのデモでは、Git操作やリポジトリ内タスクがMCP経由で完結し、ReplitやCodeium、Sourcegraphなどが早期採用企業として名乗りを上げています。citeturn1news15
Microsoftは「AIアプリのUSB-C」としてMCPを位置づけ、Windows AI Foundryを通じてファイルシステムやWSLへの制御付きアクセスを提供する計画を発表しました。citeturn1news19
同社の戦略では、異なるAIエージェントの協調や長期メモリ共有を可能にする標準としてMCPが重視されています。citeturn1news23
企業向けにはWorkatoがMCPベースのマネージドプラットフォームを展開し、クラウド／ローカル双方のサーバーをセキュアにオーケストレーションできるようにしています。citeturn0search0turn0search3

## レジストリと開発ツール
2025年には公開MCPサーバーを収録するレジストリとAPIがプレビュー提供され、互換性チェックやクライアントからの検索が容易になりました。citeturn1search2
Workatoの実装事例では、カタログ機能やAPIM連携、主要クライアントとの互換テストを通じて導入が加速しています。citeturn0search3

## セキュリティと今後の課題
Windows側でも認可ダイアログやグループポリシー統合による安全策が紹介されていますが、静的資格情報への依存やツール偽装などのリスクが残されており、短命トークンや統一的ポリシー管理が推奨されています。citeturn1news23
研究コミュニティは、ツールポイズニングやラグプル攻撃を防ぐため、署名付きツール定義とポリシーベースアクセス制御を組み込むETDI拡張案を提示しています。citeturn1academia14

## 主要リソース
- 公式仕様とリリースロードマップ。citeturn1search0
- エコシステム採用事例（Anthropic・Microsoft）。citeturn1news15turn1news19turn1news23
- WorkatoのMCPドキュメントとプロダクトアップデート。citeturn0search0turn0search3
- MCPセキュリティに関するアカデミック提案。citeturn1academia14
