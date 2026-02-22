# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ACL 2024論文「EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities」の公式実装。LLM（GPT）を用いてマクロ経済活動をシミュレートするエージェントフレームワーク。Salesforceの[AI Economist Foundation](https://github.com/MaciejMacko/ai-economist)フレームワークを拡張している。

## シミュレーション実行コマンド

GPTベース（`simulate_utils.py`内の`openai.api_key`を事前に設定すること）:
```bash
python simulate.py --policy_model gpt --num_agents 100 --episode_length 240
```

ヒューリスティック（Composite）ベース（API不要）:
```bash
python simulate.py --policy_model complex --num_agents 100 --episode_length 240
```

主なオプション: `--dialog_len`（GPTの会話履歴長）、`--beta`/`--gamma`/`--h`（complexモデル用ハイパーパラメータ）、`--max_price_inflation`/`--max_wage_inflation`

## アーキテクチャ

### エントリーポイント
- **`simulate.py`**: メインのシミュレーション実行スクリプト。`fire.Fire(main)`でCLI引数を解析
- **`simulate_utils.py`**: OpenAI API呼び出し、プロンプト整形、コスト計算等のユーティリティ
- **`config.yaml`**: 環境・RL学習・エージェントポリシーの設定ファイル

### シミュレーションフロー
`main()` → 環境初期化(`foundation.make_env_instance`) → 月次ループ（`gpt_actions`または`complex_actions`でエージェント行動を生成 → `env.step(actions)` → 6ヶ月毎にチェックポイント保存）

### Foundation フレームワーク (`ai_economist/foundation/`)

コンポーネントベースの経済シミュレーション基盤。レジストリパターンでモジュールを動的に登録・構成する。

- **`base/`**: コア抽象クラス群
  - `base_env.py`: `BaseEnvironment` — OpenAI Gym風API（`reset()`, `step()`, `seed()`）
  - `base_agent.py`: `BaseAgent` — エージェントの状態・インベントリ・行動空間を管理
  - `base_component.py`: `BaseComponent` — 経済ダイナミクスの抽象基底。行動空間追加・状態更新・観測生成を担当
  - `world.py`: `World`/`Maps` — 空間構成、経済指標（物価・賃金・金利・インフレ率・失業率・GDP）を管理
  - `registrar.py`: `Registry` — `@component_registry.add`デコレータでコンポーネントを登録

- **`agents/`**: エージェント種別
  - `mobiles.py`: `BasicMobileAgent` — 個人経済アクター
  - `planners.py`: `BasicPlanner` — 社会計画者（税率等のマクロ政策を設定、インデックスは常に`"p"`）

- **`components/`**: 経済コンポーネント（本プロジェクトで使用するもの）
  - `simple_labor.py`: 労働時間選択（0-168時間）、スキル×労働時間で収入決定
  - `redistribution.py`: `PeriodicBracketTax` — 累進課税・再分配（US Federal 2018ベース）
  - `simple_consumption.py`: 消費率選択、物価のインフレ/デフレ
  - `simple_saving.py`: 貯蓄利息・金利ダイナミクス

- **`scenarios/one_step_economy.py`**: `OneStepEconomy` — 月次ステップの経済シナリオ。エージェントにname/age/city/job等の属性を付与

### 2つの政策モデル
1. **`gpt`**: LLMが経済状況を自然言語で受け取り、JSON（`{'work': 0-1, 'consumption': 0-1}`）で行動を出力。3ヶ月毎に振り返り（reflection）プロンプトも実行
2. **`complex`**: ヒューリスティック関数（`consumption_len`/`consumption_cats`/`work_income_wealth`）で行動を決定

### データ
- **`data/profiles.json`**: エージェントのプロフィール（名前・年齢・都市・職業・給与帯）
- シミュレーション出力は `data/{policy_model}-{params}/` に保存（pkl形式、6ヶ月毎のスナップショット）

## 重要な注意事項

- `simulate_utils.py`のGPTモデルは`gpt-3.5-turbo-0613`（現在アクセス不可）が指定されている。`gpt-4o-mini`への変更が推奨されているが、`gpt_error`が多発する場合はプロンプトのJSON出力指示部分の調整が必要
- OpenAI APIは旧形式（`openai.ChatCompletion.create`）を使用。新しいopenaiパッケージ（v1.0+）ではAPIが変更されているため注意
- `get_multiple_completion`は`multiprocessing.Pool`（15プロセス）で並列API呼び出しを行う
- GPTレスポンスの解析に`eval()`を使用している（セキュリティ上の注意点）
