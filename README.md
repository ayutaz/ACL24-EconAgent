# EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities
ACL 2024 論文の公式実装です。

本プロジェクトは、以下の論文で発表された経済シミュレーションフレームワーク [Foundation](https://github.com/MaciejMacko/ai-economist) をベースにしています:

Zheng, Stephan, et al. "The ai economist: Improving equality and productivity with ai-driven tax policies." arXiv preprint arXiv:2004.13332 (2020).

# 概要

EconAgent は、LLM（GPT）を経済エージェントの意思決定エンジンとして活用し、マクロ経済活動をシミュレートするフレームワークです。各エージェントは名前・年齢・職業などの個性を持ち、毎月の **労働** と **消費** を自律的に決定します。エージェント全体の行動が集積されることで、物価・賃金・金利・GDP・失業率といったマクロ経済指標が内生的に変動します。

## シミュレーションの仕組み

月次ステップで以下の経済サイクルが繰り返されます:

1. **労働**: エージェントが労働時間を選択し、スキルに応じた収入を得る。GDP・失業率が更新される
2. **消費**: エージェントが消費率を選択し、保有資産の一部を支出する。需給バランスに基づき物価・賃金が変動する
3. **貯蓄・金利**: 年末に貯蓄へ利息が付与され、テイラールールに基づいて金利が更新される
4. **課税・再分配**: 累進所得税（US Federal 2018ベース、7段階）を徴収し、全エージェントへ均等に再分配する
5. **職業提示**: 失業中のエージェントにスキルレベルに応じた職業がオファーされる

## 政策モデル（GPT）

- LLM が経済状況（物価・賃金・貯蓄・金利など）を自然言語プロンプトで受け取り、JSON（`{'work': 0-1, 'consumption': 0-1}`）で行動を出力する
- 各エージェントにプロフィール（名前・年齢・都市・職業・給与）が付与され、個性ある意思決定を行う
- **3ヶ月毎の振り返り（Reflection）**: 四半期の経済環境を回顧し、次の意思決定に活かす
- 対話履歴を保持し、過去の文脈を踏まえた判断が可能
- OpenAI API が必要（15プロセスで並列呼び出し、gpt-4.1-mini を使用）

## エージェントのプロフィール

`data/profiles.json` から最大200人分の属性が付与されます:
- **名前**: 200種の英語フルネーム
- **年齢**: 18〜59歳
- **都市**: 米国主要都市
- **職業**: スキルレベルに応じた10段階の給与帯（Intern〜Tech Company Founder）に対応する計100職種

## 追跡されるマクロ経済指標

| 指標 | 説明 |
|---|---|
| 物価 (Price) | 需給バランスで内生的に変動 |
| 賃金 (Wage) | 需給バランスで内生的に変動 |
| 金利 (Interest Rate) | テイラールールで決定 |
| 物価インフレ率 | 年次の物価変動率 |
| 賃金インフレ率 | 年次の賃金変動率 |
| 失業率 | 労働未選択エージェントの割合 |
| 名目 GDP / 実質 GDP | 年次で集計 |
| GDP 成長率 | 名目・実質それぞれ算出 |

# 環境構築
パッケージ管理には [uv](https://docs.astral.sh/uv/) を使用します。

```bash
uv sync                # 基本依存のインストール
uv sync --extra gpt    # openai も含めてインストール
```

# 実行方法

## API キーの設定

OpenAI API キーをプロジェクトルートの `.env` ファイルに記載するか、環境変数として設定してください（`python-dotenv` により `.env` が自動読み込みされます）:

```bash
# .env ファイルを作成（推奨）
echo 'OPENAI_API_KEY=your-api-key-here' > .env

# または環境変数を直接設定
export OPENAI_API_KEY="your-api-key-here"
```

## シミュレーション実行

100エージェント、240ヶ月でシミュレーションを実行する場合:

`uv run python simulate.py --num_agents 100 --episode_length 240`

RL ベースのアプローチ（**The ai economist**）については、提供されている学習コードに従い、学習済みモデルをシミュレーションに使用しています。詳細は論文の付録を参照してください。

## CLI オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--num_agents` | `100` | エージェント数 |
| `--episode_length` | `240` | シミュレーション期間（月数） |
| `--dialog_len` | `3` | GPT モードの対話履歴長 |
| `--max_price_inflation` | `0.1` | 物価の最大インフレ率 |
| `--max_wage_inflation` | `0.05` | 賃金の最大インフレ率 |

## 出力データ

シミュレーション結果は `data/` 配下に保存されます（6ヶ月毎のチェックポイント + 最終ステップ）。

| ファイル | 内容 |
|---|---|
| `actions_{step}.pkl` | 各エージェントの行動（work, consumption） |
| `obs_{step}.pkl` | 環境の観測データ |
| `env_{step}.pkl` | 環境オブジェクト全体 |
| `dense_log_{step}.pkl` | 全ステップの詳細ログ |
| `dialog_{step}.pkl` | 対話履歴 |
| `dialog4ref_{step}.pkl` | 振り返り用対話履歴 |
| `dialogs/{name}` | 各エージェントの全対話ログ |

## 結果分析

シミュレーション完了後、`analyze_results.py` でマクロ経済指標・資産分布・就業率等を集計できます:

```bash
uv run python analyze_results.py
```

分析内容: 物価/賃金/金利の推移、四半期別就業率・消費、年次失業率、資産分布（ジニ係数・五分位）、スキル-資産相関

# モデルについて
本シミュレーションは **gpt-4.1-mini** を使用します。openai Python SDK v1.0+ と JSON mode (`response_format: json_object`) に対応しています。

`gpt_error` が多発する場合は、プロンプトの JSON 出力指示部分を調整してください。
