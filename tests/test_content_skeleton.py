import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, attach_data, check, example_problem, make_project, open_errors,
    project_error, questions, recipe_from, record_project
)
from ml_quest.data import CONTENT_LOADERS, content_kind
from ml_quest.questions import REPORTED_BASELINE

# 画像と文の 2 クラスの合成のデータの, クラスの名前と当たりのクラス
CLASSES = {"image": ["circle", "square"], "text": ["good", "bad"]}

# 文のクラスごとの語 (クラスで語を分け, 学習できるように)
WORDS = {"good": ["great", "nice", "love", "happy"],
         "bad": ["awful", "hate", "poor", "sad"]}

# 1 クラスの件数と, 画像の大きさと, 文の長さ (語の番号の並びの長さ)
COUNT = 12
SIZE = 16
LENGTH = 16

# 学習と推論の一式まで通すのに要るもの (CI の train のジョブで入れる)
TRAINING = ("torch", "ml_found", "ml_modelcontrol", "ml_inference")


def write_images(root: Path) -> Path:
    """
    Pillow で丸と四角を描いた 2 クラスの画像のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        <クラス>/<画像>.png のディレクトリ
    """
    # 黒地に白の丸か四角 (位置を少しずつずらす)
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    path = root / "images"
    for name in CLASSES["image"]:
        (path / name).mkdir(parents=True)
        for index in range(COUNT):
            image = image_module.new("RGB", (SIZE, SIZE), "black")
            draw = draw_module.Draw(image)
            shift = index % 4
            box = (2 + shift, 2 + shift, 10 + shift, 10 + shift)
            if name == "circle":
                draw.ellipse(box, fill="white")
            else:
                draw.rectangle(box, fill="white")
            image.save(path / name / f"{index}.png")

    return path


def write_texts(root: Path) -> Path:
    """
    クラスごとに語を分けた, 2 クラスの短い文のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        <クラス>/<文>.txt のディレクトリ
    """
    # 共通の語とクラスの語 2 つの文
    path = root / "texts"
    for name in CLASSES["text"]:
        (path / name).mkdir(parents=True)
        words = WORDS[name]
        for index in range(COUNT):
            text = (f"this movie is {words[index % 4]} and "
                    f"{words[(index + 1) % 4]}")
            (path / name / f"{index}.txt").write_text(text, encoding="utf-8")

    return path


def content_problem(kind: str, root: Path) -> dict:
    """
    見本の課題から, 画像か文の分類の問題の設定を作り, データを入れる関数

    Parameters
    ----------
    kind: str
        データの種類 (image か text)
    root: Path
        データと写しを置く場所

    Returns
    ----------
    problem: dict
        データを入れた問題の設定 (比べられる指標の正解率で監視)
    """
    # ML-EDA の読み込みの列から, 目的の列と分かる列を答える
    eda = pytest.importorskip("ml_eda")
    source = write_images(root) if kind == "image" else write_texts(root)
    columns = [str(name) for name in eda.load_data(source).frame.columns]
    problem = example_problem("解約の引き止め")
    answers = [
        ("information.columns", columns),
        ("ml.parts.0.target.column", "label"),
        ("ml.parts.0.positive", CLASSES[kind][0]),
        ("information.known", [name for name in columns if name != "label"]),
        ("information.forbidden", []),
        ("ml.parts.0.evaluation.metric", "accuracy"),
    ]
    for key, value in answers:
        problem = answer(problem, key, value)
    problem = attach_data(problem, str(source), copy_to=str(root / "copies"))

    return problem


def export(problem: dict, project: Path, kind: str) -> dict:
    """
    レシピの雛形 (文は vocabulary を足す) で, 骨組みの場所にデータセットを
    書き出す関数

    Parameters
    ----------
    problem: dict
        データを入れた問題の設定
    project: Path
        書き出したプロジェクトのディレクトリ
    kind: str
        データの種類

    Returns
    ----------
    inputs: dict
        ML-EDA の model_inputs の結果
    """
    # record_project の場所 (train.py が読む場所) に書き出し
    eda = pytest.importorskip("ml_eda")
    start = recipe_from(problem)
    recipe = start["recipe"]
    assert recipe["target"] == "label"
    assert recipe["task"] == "classification"
    assert recipe["split"]["method"] == "stratified"
    if kind == "text":
        recipe["item_steps"] = [{"op": "vocabulary", "length": LENGTH}]
    load = start["load"]
    data = eda.load_data(load["path"], loader=load["loader"],
                         **(load["options"] or {}))
    place = Path(record_project(problem, project)["project"]["dataset"])
    eda.export_dataset(data, recipe, place)
    inputs = eda.model_inputs(place)

    return inputs


def vocabulary_size(dataset: Path) -> int:
    """
    データセットのレシピの vocabulary の手順から, 語彙の数を返す関数

    Parameters
    ----------
    dataset: Path
        ML-EDA のデータセットのディレクトリ

    Returns
    ----------
    size: int
        語彙の数 (埋め草と語彙に無い語を含む)
    """
    # 書き出したレシピの item_steps の vocabulary の state.vocabulary
    recipe = yaml.safe_load((dataset / "recipe.yaml")
                            .read_text(encoding="utf-8"))
    steps = [step for step in recipe["item_steps"]
             if step["op"] == "vocabulary"]
    size = len(steps[0]["state"]["vocabulary"])

    return size


def model_spec(kind: str, inputs: dict, vocabulary: int) -> dict:
    """
    データセットの入力の形とクラスの数に合わせた, ML-ModelControl の定義を
    作る関数

    Parameters
    ----------
    kind: str
        データの種類
    inputs: dict
        ML-EDA の model_inputs の結果
    vocabulary: int
        語彙の数 (画像では使わない)

    Returns
    ----------
    spec: dict
        画像は畳み込み, 文は埋め込みの平均の小さなモデルの定義
    """
    # 入力の形の 1 つ目 (件数) を 2 にした例
    shape = [2, *inputs["inputs"][0]["shape"][1:]]
    classes = inputs["outputs"]
    if kind == "image":
        spec = {
            "format_version": 1, "name": "image_classifier",
            "inputs": {"image": {"shape": shape}},
            "nodes": {
                "conv": {"unit": "nn.Conv2d", "inputs": "image",
                         "params": {"in_channels": shape[1],
                                    "out_channels": 8, "kernel_size": 3,
                                    "padding": 1}},
                "act": {"unit": "nn.ReLU", "inputs": "conv"},
                "pool": {"unit": "nn.AdaptiveAvgPool2d", "inputs": "act",
                         "params": {"output_size": 1}},
                "flat": {"unit": "nn.Flatten", "inputs": "pool"},
                "head": {"unit": "nn.Linear", "inputs": "flat",
                         "params": {"in_features": 8,
                                    "out_features": classes}},
            },
            "outputs": {"logits": "head"},
        }
        return spec
    spec = {
        "format_version": 1, "name": "text_classifier",
        "inputs": {"ids": {"shape": shape, "dtype": "int64"}},
        "nodes": {
            "emb": {"unit": "nn.Embedding", "inputs": "ids",
                    "params": {"num_embeddings": vocabulary,
                               "embedding_dim": 8}},
            "pool": {"unit": "op.Mean", "inputs": "emb",
                     "params": {"dim": 1}},
            "head": {"unit": "nn.Linear", "inputs": "pool",
                     "params": {"in_features": 8, "out_features": classes}},
        },
        "outputs": {"logits": "head"},
    }

    return spec


@pytest.mark.parametrize("kind", ["image", "text"])
def test_content_project(tmp_path: Path, kind: str) -> None:
    """
    画像と文の分類の問題の設定から, 中身で学ぶ案内の骨組みを書き出せ, 案内
    どおりにデータセットを書き出せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    kind: str
        データの種類

    Returns
    ----------
    None
    """
    # 誤りの無い問題の設定から書き出せる
    problem = content_problem(kind, tmp_path)
    assert [f for f in check(problem) if f["level"] == "error"] == []
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert "train.py" in written

    # README: 表の列を学習に使う列として並べず, 中身の手順と入力の形の案内
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "学習に使ってよい列" not in readme
    assert "学ぶのは 1 件の" in readme and "`item_steps`" in readme
    assert ("`resize`" in readme) == (kind == "image")
    assert ("`vocabulary`" in readme) == (kind == "text")
    assert ("`num_embeddings`" in readme) == (kind == "text")
    assert "学習に使う列の数 (`dataset.yaml`" not in readme
    shape = {"image": "`[None, チャンネル, 高さ, 幅]` の float32",
             "text": "`[None, 長さ]` の int64"}[kind]
    assert shape in readme
    assert "今のやり方とは比べない (入れたデータが 1 件の中身で学ぶ" in readme

    # train.py: 記録は 1 件の形, 比べない理由 (書き出し直しの案内は無し)
    script = (project / "train.py").read_text(encoding="utf-8")
    compile(script, "train.py", "exec")
    assert max(len(line) for line in script.splitlines()) <= 79
    assert "dataset.features" not in script
    assert "tuple(x_train.shape[1:])" in script
    assert "入れたデータが 1 件の中身で学ぶ画像か文" in script
    assert "答えて骨組みを書き出し直すと" not in script
    assert "evaluation.current" in script and "save_reported" not in script
    hint = "8 の「今のやり方 (人の目視や今の仕組み) で, 正解率は幾つですか (分かれば)」"
    assert hint in readme and "BASELINE_REASON = (" in script
    assert "最良を選ぶ指標を" not in readme
    assert "logger.info(f\"今のやり方とは比べません: {BASELINE_REASON}" in (
        script
    )
    assert "分からない ため" not in readme
    config = yaml.safe_load(
        (project / "configs/default/train.yaml").read_text(encoding="utf-8")
    )
    assert config["trainer"]["monitor"] == "valid/accuracy"

    # 雛形のレシピ (文は vocabulary を足す) で書き出せ, 入力の形が案内どおり
    inputs = export(problem, project, kind)
    assert inputs["outputs"] == 2
    expected = {"image": ([None, 3, SIZE, SIZE], "float32"),
                "text": ([None, LENGTH], "int64")}[kind]
    assert (inputs["inputs"][0]["shape"],
            inputs["inputs"][0]["dtype"]) == expected
    if kind == "text":
        # 埋め草と語彙に無い語と, 学習用の文の語 (共通の 4 語とクラスの語)
        size = vocabulary_size(project / "outputs" / "dataset")
        assert 2 + 4 < size <= 2 + 4 + sum(len(w) for w in WORDS.values())
        # README の案内どおり, model_inputs の語彙の大きさがレシピの語彙の数
        assert inputs["inputs"][0]["vocabulary"] == size


@pytest.mark.parametrize("kind", ["image", "text"])
def test_content_reported(tmp_path: Path, kind: str) -> None:
    """
    画像と文で今のやり方での指標の値を問い, 答えれば骨組みがテスト用の値と
    並べることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    kind: str
        データの種類

    Returns
    ----------
    None
    """
    # 画像と文で, 比べられる指標のときだけ問う (表の課題と loss は問わない)
    key = "ml.parts.0.evaluation.current"
    problem = content_problem(kind, tmp_path)
    applies = {item["id"]: item["applies"] for item in questions(problem)}
    assert applies[key] is True
    loss = answer(problem, "ml.parts.0.evaluation.metric", "loss")
    assert {item["id"]: item["applies"]
            for item in questions(loss)}[key] is False
    table = {item["id"]: item["applies"]
             for item in questions(example_problem("総菜の作る量"))}
    assert table[key] is False

    # 割合の指標は 0 から 1 で, 負と 1 を超える値は誤り
    for wrong in (85, -0.1):
        wrong_problem = answer(problem, key, wrong)
        found = check(wrong_problem)
        assert [item["id"] for item in found if item["level"] == "error"] == [
            key
        ]
        # 骨組みは組めるので, 「それでも書き出す」の対象
        assert project_error(wrong_problem) is None
        assert [item["id"] for item in open_errors(wrong_problem)] == [key]
    assert "正解率は 0 から 1" in check(answer(problem, key, 85))[0]["message"]
    problem = answer(problem, key, 0.75)
    assert [f for f in check(problem) if f["level"] == "error"] == []

    # 骨組み: テスト用の評価で並べ, README に目安と書く
    project = tmp_path / "project"
    make_project(problem, project)
    script = (project / "train.py").read_text(encoding="utf-8")
    compile(script, "train.py", "exec")
    assert max(len(line) for line in script.splitlines()) <= 79
    assert "BASELINE_VALUE = 0.75" in script
    call = ('\n\n    # 答えた今のやり方での指標の値と並べる\n'
            '    save_reported(experiment, "accuracy", accuracy.item())')
    assert call in script
    assert "テスト用の件のうち当たった割合" in script
    assert "今のやり方 (今のやり方)" not in script
    assert "テスト用の分割が無いので, 評価せず, 答えた値とも並べません" in script
    assert "save_evaluation" in script and "compare_baseline" not in script
    text = next(ast.literal_eval(node.value) for node in ast.parse(
        script
    ).body if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "REPORTED_TEXT")
    assert "差 (accuracy_diff) はモデル − 今のやり方" in text
    assert "大きいほど良い" in text and "今のやり方 (問い合わせ" in text
    assert "正解率 (テスト用の件のうち当たった割合" in text
    assert "METRIC_NAME = \"正解率\"" in script
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "モデルの正解率 (テスト用の件のうち当たった割合" in readme
    assert "((" not in readme and "値 (正解率 (" not in readme
    assert ("今のやり方での `accuracy` の値 (0.75. "
            "`ml.parts.0.evaluation.current`)") in readme
    assert "別のデータで数えた値との比べで目安です" in readme


def test_content_questions(tmp_path: Path) -> None:
    """
    画像と文では予測の時点で分かる列を問わず, 今のやり方での指標の値の問い
    の題が指標を名指すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表の列を学習に使わないので, 分かる列は要らない (答えなくても誤りなし)
    problem = answer(content_problem("image", tmp_path),
                     "information.known", None)
    items = {item["id"]: item for item in questions(problem)}
    assert items["information.known"]["applies"] is False
    assert [f for f in check(problem) if f["level"] == "error"] == []
    table = {item["id"]: item for item in questions(
        example_problem("総菜の作る量")
    )}
    assert table["information.known"]["applies"] is True

    # 題は最良を選ぶ指標の名前で
    title = items["ml.parts.0.evaluation.current"]["title"]
    assert "正解率は幾つですか" in title and "この指標" not in title


@pytest.mark.parametrize("value", [85, -1, float("nan"), float("inf")])
def test_content_reported_wrong(tmp_path: Path, value: float) -> None:
    """
    今のやり方での指標の値に誤りがあれば並べず, 理由をコメント, README,
    train.log に書くことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    value: float
        誤りのある値 (非数と無限大は手で書いたもの)

    Returns
    ----------
    None
    """
    # 骨組みを組める error なので, 「それでも書き出す」で書き出せる
    problem = content_problem("image", tmp_path)
    problem["ml"]["parts"][0]["evaluation"]["current"] = value
    assert project_error(problem) is None
    make_project(problem, tmp_path / "project", accept_errors=True)
    script = (tmp_path / "project" / "train.py").read_text(encoding="utf-8")
    compile(script, "train.py", "exec")
    assert "save_reported" not in script and "BASELINE_VALUE" not in script
    assert "BASELINE_REASON" in script
    readme = (tmp_path / "project" / "README.md").read_text("utf-8")
    assert "答えた今のやり方での指標の値は使えないので並べない" in readme


def test_content_predict_log(tmp_path: Path) -> None:
    """
    何を出すかが空なら, train.log の初めの行の「予測すること」の句を省く
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 何を出すかを消した骨組み
    problem = answer(content_problem("image", tmp_path), "ml.parts.0.what",
                     None)
    make_project(problem, tmp_path / "project", accept_errors=True)
    script = (tmp_path / "project" / "train.py").read_text(encoding="utf-8")
    assert "予測すること: {PREDICT}" not in script


@pytest.mark.parametrize("kind", ["image", "text"])
def test_content_training(tmp_path: Path, kind: str) -> None:
    """
    画像と文の分類の骨組みで案内どおりに学習し, 推論の一式の確かめが学習の
    記録と合うことを確かめる関数 (PyTorch と段階のライブラリが無ければ
    飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    kind: str
        データの種類

    Returns
    ----------
    None
    """
    # 書き出し, データセット, 形と語彙の数とクラスの数を合わせた定義
    for name in TRAINING:
        pytest.importorskip(name)
    problem = answer(content_problem(kind, tmp_path),
                     "ml.parts.0.evaluation.current", 0.5)
    project = tmp_path / "project"
    make_project(problem, project)
    inputs = export(problem, project, kind)
    vocabulary = (
        inputs["inputs"][0]["vocabulary"] if kind == "text" else 0
    )
    place = Path(record_project(problem, project)["project"]["model_spec"])
    place.write_text(
        yaml.safe_dump(model_spec(kind, inputs, vocabulary),
                       allow_unicode=True), encoding="utf-8"
    )

    # 2 エポックの学習 (別のプロセスで)
    done = subprocess.run(
        [sys.executable, "train.py", "trainer.epochs=2",
         "experiment.device=cpu"],
        cwd=project, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-2000:]

    # 実験の中の一式: 確かめ (検証用の正解率. ML-Inference の名前は acc) が
    # 学習の記録と合う
    runs = sorted((project / "outputs").glob("*/*/*/inference"))
    assert len(runs) == 1
    info = yaml.safe_load((runs[0] / "inference.yaml")
                          .read_text(encoding="utf-8"))
    assert info["check"]["metric"] == "acc"
    assert info["check"]["recorded"] is not None
    assert info["check"]["matches"] is True, info
    log = (runs[0].parent / "train.log").read_text(encoding="utf-8")
    assert "1 件の" in log and "合う: True" in log

    # 答えた今のやり方での指標の値と, テスト用の正解率を並べた表
    table = json.loads((runs[0].parent / "evaluations" / "baseline_test.json")
                       .read_text(encoding="utf-8"))
    assert table["split"] == "test" and "目安" in table["description"]
    row = table["rows"][0]
    assert row["accuracy_baseline"] == 0.5
    assert row["accuracy_diff"] == pytest.approx(row["accuracy"] - 0.5)
    assert "別のデータで数えた値との比べで目安です" in log


def test_content_loaders() -> None:
    """
    写して持つ画像と文の読み込みの名前が, ML-EDA の読み込みの登録簿の
    kind と合い, 表の中の文の列も文とすることを確かめる関数 (ML-EDA が
    無ければ登録簿との突き合わせだけ飛ばす)

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 読み込みの名前と表の中の文の列から決まる種類
    sources = [
        ({"loader": "image_folder"}, "image"),
        ({"loader": "text_lines"}, "text"),
        ({"loader": "csv"}, None),
        ({"loader": "csv", "options": {"text_column": "review"}}, "text"),
    ]
    for source, expected in sources:
        assert content_kind({"data": {"source": source}}) == expected
    assert content_kind({"data": {}}) is None

    # ML-EDA の登録簿で kind が画像か文の読み込みと, 写した対応が同じ
    eda = pytest.importorskip("ml_eda")
    kinds = {
        name: loader.kind for name, loader in eda.LOADERS.items()
        if getattr(loader, "kind", None) in ("image", "text")
    }
    assert kinds == CONTENT_LOADERS


def test_reported_loaders() -> None:
    """
    今のやり方での指標の値の問いが, 写して持つ画像と文の読み込みの名前と
    同じ名前で要るかを決めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 問いの条件の読み込みの名前 (any の 1 つ目)
    names = REPORTED_BASELINE[-1]["any"][0]["in"]
    assert sorted(names) == sorted(CONTENT_LOADERS)
