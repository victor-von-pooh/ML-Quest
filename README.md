# ML-Quest

「機械学習を全て GUI 上で行う」うちの「問題の設定」の段階のライブラリ. 課題を分解し, どの部分を機械学習に任せると課題が解けるかを決めて, 問題の設定のファイル (`problem.yaml`) に残す. 画面のアプリは [ML-Pathfinder](https://github.com/victor-von-pooh/ML-Pathfinder).

## できること

- **問い**: 課題から始めて, 課題の分解, 決めること, 機械学習の受け持ち, 損得と制約, 学習で見る指標, 効果の確かめ方を, 決まった順の問いで埋めて `problem.yaml` に残す. 機械学習が要らないと分かれば, そこで終える
- **確かめ**: 答えどうしの食い違いと足りない答えを `error`, `warning`, `info` で返す. 課題から行動までの要約の文も返す
- **受け渡し**: 後ろの段階 (データの分析, モデルの作成, 学習・評価, 推論) が要る値と, 要る機能 (強化学習, MuJoCo, ROS2, CUDA), 方策の守り, 書き出したプロジェクトの場所をまとめて返す
- **データからの候補**: 手元のデータを入れると, 答えの候補と今のやり方の損 (指標の今の値の候補) を出す (ML-EDA を使う)
- **損得の値の幅**: 損得と制約の値を幅の中で変えたとき, 決め方の値 (置く水準, 得になる確率の下限, 回すしきい値), 件数の上限, データの上の今のやり方の損がどう変わるかを返す
- **骨組み**: [ML-Found](https://github.com/victor-von-pooh/ML-Found) のプロジェクトの骨組み (config, 学習のスクリプト, 比べる相手との評価の表) を書き出す. 書き出せるのは, 列の予測 (表, 時系列, 画像, 文, 音声), 画像の領域と点群, 画像の生成, 表の生成 (合成のデータ), 続けて行動を選ぶ方策 (最良を選ぶ指標は報酬 (`reward`) が既定. 方策の頭は ML-ModelControl の定義で組み, 方策の一式に定義を写す. 守りと一緒に書き出す). 既製のモデルをそのまま使う画像の分類は, 学習のスクリプトの代わりに ML-Inference の既製のモデルの一式を作るスクリプトと config を書き出す
- **見本**: 架空の課題の問題の設定 (量を決める予測, 対象を選ぶ分類, 機械学習を使わない出口) を作る

## どのアプリから使われるか

| 相手 | 使うもの |
| --- | --- |
| [ML-Pathfinder](https://github.com/victor-von-pooh/ML-Pathfinder) (問題の設定) | 問い (`questions`, `answer`), 確かめ (`check`), 要約 (`describe`), データの候補 (`attach_data`, `inspect_data`), 損得の値の幅 (`decision_sensitivity`), 骨組みの書き出し (`make_project`, `record_project`), 見本 (`example_problem`) |
| ML-Fathom (データの分析) | データの分析の受け渡しの値 (`handoff`), レシピの下書きと書き戻し (`recipe_from`, `apply_recipe`) |
| ML-CreateArchitecture (モデルの作成) | モデルの作成の受け渡しの値 (出力の大きさ, 受け持ちに合う雛形 `example`. `model_example` でも求まる), プロジェクトのモデルの定義の置き場 |
| ML-Dashboard (学習・評価) | 学習・評価の受け渡しの値 (骨組みの外の学習でも使う監視する指標と向き), 要る機能, 書き出した骨組みのプロジェクト |
| ML-Provide (推論) | 損得と制約 (文は `decision_text`), 今のやり方, 元の問題の設定の場所 (`problem_origin`), 書き出したプロジェクトの記録 (`project_of`), 要る問いの答え (`active_value`) |

このライブラリが依存するのは, データの候補を出すときの ML-EDA (任意) だけ. 書き出した学習のスクリプトは, ML-EDA, ML-ModelControl, ML-Found, ML-Inference を使う.

## 導入

```bash
pip install "ml-quest @ git+https://github.com/victor-von-pooh/ML-Quest.git"
```

Python 3.10 以上. 依存は PyYAML と pandas だけ. 手元のデータから答えの候補を出す (`attach_data`, `inspect_data`) には, ML-EDA を入れる (`pip install "ml-quest[data] @ git+..."`. 画像と, WAVE の他の音声 (FLAC, MP3 など) を読む Pillow と soundfile も入る).

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
stages = handoff(problem)   # part, data, model (雛形の名前 example も), training (監視する指標 monitor と向き mode), inference と, 要る機能 (requires), 方策の守り (policy), プロジェクト (project)

# ML-Found のプロジェクトの骨組みの書き出しと, その場所の記録 (データセットとモデルの定義の置き場も) の保存
save_problem(problem, "/work/problem.yaml")
make_project(problem, "/work/my_project", origin="/work/problem.yaml")   # 写しに元の場所 (origin) を残す
problem = record_project(problem, "/work/my_project")
save_problem(problem, "/work/problem.yaml")
```

プロジェクトと実験に写した問題の設定は, 元の場所を持つ. 推論の段階は `problem_origin(load_problem(写し))` で元を見つけ, 写した後に直した損得を読む.

手元のデータは, 課題, 指標, 分解, 決めること (1 から 4) に答えた後, 機械学習の受け持ち (5) で入れる. 写し先を渡すとデータを写し, 写した場所と元の場所と ML-EDA の読み方を持つ.

```python
from ml_quest import COPY_NAME, answer, attach_data, inspect_data, inspect_key

problem = attach_data(problem, "/data/sozai.csv", copy_to="/work/problem/data")   # 写して (写し先は絶対パス), 場所, 読み方, 列の名前
problem = answer(problem, "action.current", {"type": "lag", "periods": 7})   # 今のやり方 (前の週の同じ曜日と同じ)
found = inspect_data(problem)    # 概要, 答えの候補, 確かめ, 今のやり方の損 (指標の今の値の候補)
found["key"] == inspect_key(problem)   # 覚えておいた結果が今の問題の設定とデータのものか (変われば False)
COPY_NAME.fullmatch("5a75d2e92f54")    # attach_data が作る写しのサブディレクトリの名前の形 (12 桁の 16 進)
```

損得の値に自信が無ければ, 幅を渡して結論の変わり方を見る (値は 1 つずつ変え, 他の値は今のまま). 下の例の `under_cost` と `over_cost` は量を決める型の問題の項目 (型ごとの項目は `SENSITIVITY_FIELDS`). 行動の件数と量そのもの (予測が要る) と, 対象を選ぶ型の得は出さない. 意思決定しない問題の設定, 損得と制約に直すところがある, 幅を渡せない項目や列で書いた項目, 幅の誤りでは `ValueError` になる (詳しくは [docs/DESIGN.md](docs/DESIGN.md) の 7 節の「損得の値の幅」):

```python
from ml_quest import SENSITIVITY_FIELDS, decision_sensitivity

result = decision_sensitivity(problem, {"under_cost": [100, 300], "over_cost": [100, 200]})
for field in result["fields"]:
    print(field["sentence"])   # 置く水準と, データの上の今のやり方の損の変わり方
SENSITIVITY_FIELDS["select"]   # 型ごとの幅を渡せる項目
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

見本の問題の設定 (架空の課題) は `example_problem` で作れる. 量を決める予測 (総菜の作る量), 対象を選ぶ分類 (解約の引き止め), 既製のモデルをそのまま使う画像の分類 (写真の仕分け. torchvision の ResNet18), 機械学習を使わない出口 (問い合わせの振り分け) の 4 つ.

```python
from ml_quest import EXAMPLES, example_problem

problem = example_problem("総菜の作る量")
```

書き出した骨組みの README に, 次にすること (データの分析で選ぶ値, モデルの入力の形と出力の大きさ, 学習の起動, 推論の一式の登録) が書かれる. 学習のスクリプトは ML-EDA, ML-ModelControl, ML-Found, ML-Inference を使い, config の `requires` にその名前 (ML-Found の登録簿の `eda`, `modelcontrol`, `inference`. 画像の生成で最良を選ぶ指標が FID なら `fid` も) を書くので, ML-Found が学習の前に足りないものと入れ方を示す. 問題の設定に今のやり方をデータで書いてデータを入れていれば, 学習のスクリプトは検証用とテスト用で今のやり方と比べた評価の表 (`evaluations/baseline_valid.json` など) を書く. 今のやり方をデータの上で作れない画像, 文, 音声, 画像の領域と点群では, 今のやり方での指標の値 (`ml.parts.0.evaluation.current`. 人の目視や今の仕組みでの値を, 分かれば答える. 数え方と単位はモデルの値と同じに) を, テスト用の分割のモデルの値と並べて `evaluations/baseline_test.json` に書く (別のデータで数えた値との比べで目安).

骨組みを書き出せるのは, 列の予測 (分類, 回帰, 時系列の予測. 表と時系列のほか, 画像, 文, 音声のデータの分類も. 画像, 文, 音声では README が 1 件の中身で学ぶ案内になる) と, 予測するで正解の種類が画像の領域か点群の受け持ち. 後の 2 つは ML-EDA のファイルの組のデータセット (版 2 か, 中身の手順をかけた版 3. `export_image_masks`, `export_capture_clouds`) で学習する骨組みで, 入れたデータセットをプロジェクトに写し, 仮のモデル (印 TODO) と, 損失と指標 (領域は交差エントロピーと IoU, 点群は Chamfer) を書く. config の `model.spec` は `models/model.yaml` で, そこに ML-ModelControl の定義を置けばそのモデルで学習し, 学習の後に ML-Inference の推論の一式も実験の中に作る (置くまでは仮のモデルで学習する). 学習の後に, 検証用の数件の入力, 予測, 正解を ML-Found の見本 (`samples/`) に書く. 作るで正解の種類と出力の形が画像の受け持ち (条件の無い画像の生成) も書き出せる. 画像のディレクトリから目的の列の無いレシピ (`task: none`) で書き出した ML-EDA のデータセットで, 拡散のモデル (雑音を加えた画像と時刻から, 加えた雑音を当てる) を ML-Found の重みの指数移動平均 (`trainer.ema`) 付きで学び, 損失と作った画像の見本で評価する (最良を選ぶ指標に FID を選ぶと, 検証で作った画像の FID も数えて選ぶ. ML-Found の追加の依存 `fid`). 定義を置けば, ML-Inference の雑音だけの一式 (件数と種で画像を作る) も作る. 作るで正解の種類と出力の形が表の受け持ち (今ある表に似た行を作る合成のデータ) も書き出せる. 表のファイルから目的の列の無いレシピ (`task: none`) で書き出した ML-EDA の表のデータセットを, 数の列は標準化, 種類の列はワンホットにして同じ拡散のモデルで学び, 作った表と本物の表の列ごとの分布の近さ (数の列は KS 統計量と Wasserstein 距離, 種類の列は全変動距離. 最良を選ぶ指標の `column_distance` は KS 統計量と全変動距離の列の平均) を検証用とテスト用の分割で数え, 作った表の見本 (CSV) を書く. 定義を置けば, 雑音だけの一式 (推論の一式の版 7. 実験の列の符号化 `table_columns.json` を写し, 件数と種で列の名前と値の表を返す. 版 7 の一式を作れる ML-Inference が要る) も作る. 予測の表の列と重なる名前の列 (`sample`, `status`, `prediction` など) があれば, 学習の前に `drop_columns` で落とすか名前を変えるよう案内して止まる. モデルの入力と出力の幅 (符号化した行の幅) は, 学習の前に `table_encoding(problem, データセットの場所)` で求められる (`width` と, 数と種類の列の数, 列ごとの符号化を返す. 表の生成でなければ `None`. ML-EDA が要る). 効果を見積もるで続けて行動を選ぶ方策の受け持ちも書き出せる. 答えた Gymnasium の環境の登録名 (無ければ書き出した自作の環境の雛形 `env.py`) で, ML-Found の強化学習 (`RolloutLoader` と, 番号で選ぶ行動は `ReinforceStep`, 連続の行動は `ActorCriticStep`) で方策を学び, 問題の設定の守りと一緒に方策の一式 (`policy/`) に書き出して, 守りを当てたまま環境で確かめる. 作り方が既製のモデルをそのまま (`pretrained`) の画像の分類 (予測するで正解の種類が表の列, 出力の形がクラス) は学習しない骨組みで, 問いで答えた出どころ (今は `torchvision`), モデルの名前, 重みの名前と hash (任意) を `configs/default/bundle.yaml` に書き, `python bundle.py` が ML-Inference の `make_ready_bundle` で推論の一式 (版 6) を作る (ML-Inference と `ml-modelcontrol[torchvision]` が要る. hash を決めていなければ, 読んだ重みの hash を表示する). 書き出せない理由は `project_error(problem)` で分かる.

確かめの直すところ (`error`) のうち骨組みを組めるもの (問題の設定の `check` の, 確かめ方や損得などの `error` と, 入れたデータの確かめ `inspect_data` の返す辞書の `findings` の `error`) が残っていると, `make_project` は書き出さない. 残ったまま書き出すなら「それでも書き出す」として `accept_errors=True` を渡す. そのときは README の最後に, 書き出した時点で残っていた直すところの一覧を書く. 残っているものは `open_errors(problem)` で分かる (覚えておいた `inspect_data` の結果を `inspected` に渡せば分析し直さない. 結果の印 `key` が今の問題の設定とデータのものでなければ求め直す). 損得と制約に直すところが残るときは, 意思決定の既定値を付けずに推論の一式を作る骨組みにする. 書き出した後のレシピの下書き (`recipe_from`) も止まらず, 残っている直すところを `notes` に出す. 推論の段階に渡す値 (`handoff`) は, 直すまで作れない. 目的の列が無いなど骨組みを組めない `error` は, 指定しても書き出さない. 入れたデータの確かめには ML-EDA が要り, 無ければ「データを読めません」が直すところに入る (`ml-quest[data]` で入れると消える).

## 文書

- [docs/DESIGN.md](docs/DESIGN.md): 今の設計と決まり, 他のリポジトリとの約束 (問題の設定の形と版)
- [TODO.md](TODO.md): まだやっていないこと (優先度の順)
- [CODING_STYLE.md](CODING_STYLE.md): コードの書き方
- [CLAUDE.md](CLAUDE.md): このリポジトリで作業するときの方針

## ライセンス

MIT
