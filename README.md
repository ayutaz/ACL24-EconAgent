# EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities
ACL 2024 論文の公式実装です。

本プロジェクトは、以下の論文で発表された経済シミュレーションフレームワーク [Foundation](https://github.com/MaciejMacko/ai-economist) をベースにしています:

Zheng, Stephan, et al. "The ai economist: Improving equality and productivity with ai-driven tax policies." arXiv preprint arXiv:2004.13332 (2020).

# 実行方法
GPT-3.5、100エージェント、240ヶ月でシミュレーションを実行する場合（simulate_utils.py に openai.api_key を設定してください）:

`python simulate.py --policy_model gpt --num_agents 100 --episode_length 240`

Composite、100エージェント、240ヶ月でシミュレーションを実行する場合:

`python simulate.py --policy_model complex --num_agents 100 --episode_length 240`

RL ベースのアプローチ（**The ai economist**）については、提供されている学習コードに従い、学習済みモデルをシミュレーションに使用しています。詳細は論文の付録を参照してください。

# 2024年8月16日の更新
シミュレーションは gpt-3.5-turbo-0613 でのみテストされていましたが、このモデルは現在利用できなくなり、gpt-4o-mini に置き換えられています。`gpt_error` が0より大幅に大きい場合（例: 10を超える場合）、GPT が不合理な意思決定を多数生成していることを意味しますので、プロンプトを適宜調整してください。特にフォーマット指示に関する部分を見直してください:

*"Please share your decisions in a JSON format. The format should have two keys: 'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your savings and income you intend to spend on essential goods)."*
