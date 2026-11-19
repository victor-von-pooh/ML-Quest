# ML-Quest

機械学習をもとにした課題解決のために, 問題を設定するライブラリ.

機械学習は課題を解く手段の 1 つにすぎない. 課題から始めて, 課題を分解し, どの部分を機械学習に任せると課題の解決につながるかを, 決まった順の問いで埋めて 1 つのファイル (`problem.yaml`) に残す. 答えどうしの食い違いを確かめ, 後ろの段階 (データの分析, モデルの作成, 学習・評価, 推論) が要る値をまとめ, [ML-Found](https://github.com/victor-von-pooh/ML-Found) のプロジェクトの骨組み (config と学習のスクリプト) を書き出す. 機械学習が要らないと分かれば, そこで終える.

画面は [ML-Pathfinder](https://github.com/victor-von-pooh/ML-Pathfinder). 設計の方針は [docs/DESIGN.md](docs/DESIGN.md) を参照.

## 導入

```bash
pip install "ml-quest @ git+https://github.com/victor-von-pooh/ML-Quest.git"
```

Python 3.10 以上. 依存は PyYAML と pandas だけ. 手元のデータから答えの候補を出す (`attach_data`, `inspect_data`) には, ML-EDA を入れる (`pip install "ml-quest[data] @ git+..."`).

## 使い方

```python
from ml_quest import (
    answer, check, describe, handoff, make_project, new_problem,
    questions, save_problem
)

# 問いの順に答える (questions で問いの並びと, 今要るかが分かる)
problem = new_problem("総菜の作る量")
problem = answer(problem, "issue.owner", "店長")
problem = answer(problem, "issue.pain", "総菜の廃棄が多く, 夕方には品切れも出る")
# ... (問いの全ては docs/DESIGN.md の「問題の設定の形」)

# 足りない答えと食い違い (error, warning, info)
for item in check(problem):
    print(item["level"], item["id"], item["message"])

# 課題から行動までの要約, 後ろの段階ごとの値
print("\n".join(describe(problem)))
stages = handoff(problem)   # data, model, training, inference と, 要る機能 (requires), 方策の守り (policy)

# 保存と, ML-Found のプロジェクトの骨組みの書き出し
save_problem(problem, "problem.yaml")
make_project(problem, "my_project")
```

手元のデータは, 課題, 指標, 分解, 決めること (1 から 4) に答えた後, 機械学習の受け持ち (5) で入れる. 写し先を渡すとデータを写し, 写した場所と元の場所と ML-EDA の読み方を持つ.

```python
from ml_quest import answer, attach_data, inspect_data

problem = attach_data(problem, "/data/sozai.csv", copy_to="/work/problem/data")   # 写して (写し先は絶対パス), 場所, 読み方, 列の名前
problem = answer(problem, "action.current", {"type": "lag", "periods": 7})   # 今のやり方 (前の週の同じ曜日と同じ)
found = inspect_data(problem)    # 概要, 答えの候補, 確かめ, 今のやり方の損 (指標の今の値の候補)
```

データの分析の段階とつなぐ部品 (全体のシステムで使う):

```python
from ml_quest import apply_recipe, recipe_from

start = recipe_from(problem)        # データの読み方と, ML-EDA のレシピの雛形 (目的の列, タスク, 除く列, 過去の値, 分割)
result = apply_recipe(problem, recipe)   # 目的の列とタスクを書き戻し, 列の使い方の食い違いと選ぶ問いを返す
```

要る機能 (強化学習, MuJoCo, ROS2, CUDA) は答えの途中でも出せる. 名前は ML-Found の登録簿 (`describe_requirements`) と同じなので, 画面は入れ方のカードを早めに出せる.

```python
from ml_quest import FEATURES, required_features

required_features(problem)   # 例: ["rl", "mujoco"]
```

見本の問題の設定 (架空の課題) は `example_problem` で作れる. 量を決める予測 (総菜の作る量), 対象を選ぶ分類 (解約の引き止め), 機械学習を使わない出口 (問い合わせの振り分け) の 3 つ.

```python
from ml_quest import EXAMPLES, example_problem

problem = example_problem("総菜の作る量")
```

書き出した骨組みの README に, 次にすること (データの分析で選ぶ値, モデルの出力の大きさ, 学習の起動, 推論の一式の登録) が書かれる. 学習のスクリプトは ML-EDA, ML-ModelControl, ML-Found, ML-Inference を使うので, 学習する環境にはそれらを入れる.

## 開発

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

| ディレクトリ | 内容 |
| --- | --- |
| `src/ml_quest/questions.py` | 問いの並び, タスクと指標の選択肢 |
| `src/ml_quest/problem.py` | 問題の設定の作成, 答え, 読み書き, 列の名前の読み込み |
| `src/ml_quest/check.py` | 確かめ |
| `src/ml_quest/handoff.py` | 受け渡しの値と要約の文 |
| `src/ml_quest/requirements.py` | 要る機能の名前 (ML-Found の登録簿の名前) と, 答えから要る機能 |
| `src/ml_quest/skeleton.py` | ML-Found のプロジェクトの骨組み |
| `src/ml_quest/examples.py` | 見本の問題の設定 |
| `src/ml_quest/data.py` | 手元のデータの読み込みと写し, 答えの候補と確かめ, 今のやり方の損 (ML-EDA) |
| `src/ml_quest/bridge.py` | データの分析の段階とつなぐ部品 (レシピの雛形と書き戻し) |
| `docs/DESIGN.md` | 設計の方針と段階 |

## ライセンス

MIT
