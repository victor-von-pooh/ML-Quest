# ML-Quest

「機械学習を全て GUI 上で行う」うちの「問題の設定」の段階のライブラリ. 課題を分解し, どの部分を機械学習に任せると課題が解けるかを決めて, 問題の設定のファイルに残す. 画面のアプリ [ML-Pathfinder](https://github.com/victor-von-pooh/ML-Pathfinder) が使う.

## できること

- 課題から始めて, 課題を分解し, どの部分を機械学習に任せると課題の解決につながるかを, 決まった順の問いで埋めて 1 つのファイル (`problem.yaml`) に残す. 機械学習が要らないと分かれば, そこで終える
- 答えどうしの食い違いと足りない答えを確かめる
- 後ろの段階 (データの分析, モデルの作成, 学習・評価, 推論) が要る値をまとめて返す
- 手元のデータから答えの候補と今のやり方の損を出す (ML-EDA を使う)
- [ML-Found](https://github.com/victor-von-pooh/ML-Found) のプロジェクトの骨組み (config と学習のスクリプト) を書き出す

## 導入

```bash
pip install "ml-quest @ git+https://github.com/victor-von-pooh/ML-Quest.git"
```

Python 3.10 以上. 依存は PyYAML と pandas だけ. 手元のデータから答えの候補を出す (`attach_data`, `inspect_data`) には, ML-EDA を入れる (`pip install "ml-quest[data] @ git+..."`).

## 使い方

```python
from ml_quest import (
    answer, check, describe, handoff, make_project, new_problem,
    questions, record_project, save_problem
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
stages = handoff(problem)   # part, data, model, training, inference と, 要る機能 (requires), 方策の守り (policy), プロジェクト (project)

# ML-Found のプロジェクトの骨組みの書き出しと, その場所の記録 (データセットとモデルの定義の置き場も) の保存
save_problem(problem, "/work/problem.yaml")
make_project(problem, "/work/my_project", origin="/work/problem.yaml")   # 写しに元の場所 (origin) を残す
problem = record_project(problem, "/work/my_project")
save_problem(problem, "/work/problem.yaml")
```

プロジェクトと実験に写した問題の設定は, 元の場所を持つ. 推論の段階は `problem_origin(load_problem(写し))` で元を見つけ, 写した後に直した損得を読む.

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

予測するで正解の種類が画像の領域か点群の受け持ちでは, `recipe_from` は ML-EDA のファイルの組の書き出し (`export_image_masks`, `export_capture_clouds`) の設定の下書き (入力と正解のディレクトリ, 分割, グループ, 正解の出どころ, 他の引数) を `export` で返し, `apply_recipe` はその設定を受け取って問題の設定に書き戻す. 他の引数 (`options`) には中身の手順 (`item_steps`. `{"image": [{"op": "resize", "width": 64, "height": 64}]}` など) も持てる. 手順を渡すと ML-EDA はデータセットを版 3 で書くので, 書き戻した設定から同じデータセットを作り直せる.

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

書き出した骨組みの README に, 次にすること (データの分析で選ぶ値, モデルの入力の形と出力の大きさ, 学習の起動, 推論の一式の登録) が書かれる. 学習のスクリプトは ML-EDA, ML-ModelControl, ML-Found, ML-Inference を使い, config の `requires` にその名前 (ML-Found の登録簿の `eda`, `modelcontrol`, `inference`) を書くので, ML-Found が学習の前に足りないものと入れ方を示す. 問題の設定に今のやり方をデータで書いてデータを入れていれば, 学習のスクリプトは検証用とテスト用で今のやり方と比べた評価の表 (`evaluations/baseline_valid.json` など) を書く.

骨組みを書き出せるのは, 列の予測 (分類, 回帰, 時系列の予測) と, 予測するで正解の種類が画像の領域か点群の受け持ち. 後の 2 つは ML-EDA のファイルの組のデータセット (版 2 か, 中身の手順をかけた版 3. `export_image_masks`, `export_capture_clouds`) で学習する骨組みで, 入れたデータセットをプロジェクトに写し, 仮のモデル (印 TODO) と, 損失と指標 (領域は交差エントロピーと IoU, 点群は Chamfer) を書く. config に ML-ModelControl の定義を書けば, 学習の後に ML-Inference の推論の一式も実験の中に作る. 書き出せない理由は `project_error(problem)` で分かる.

## 文書

- [docs/DESIGN.md](docs/DESIGN.md): 今の設計と決まり, 他のリポジトリとの約束 (問題の設定の形と版)
- [TODO.md](TODO.md): まだやっていないこと (優先度の順)
- [CODING_STYLE.md](CODING_STYLE.md): コードの書き方
- [CLAUDE.md](CLAUDE.md): このリポジトリで作業するときの方針

## ライセンス

MIT
