import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, attach_data, example_problem, inspect_data, make_project,
    project_error
)

# 正解の種類ごとの, 型に依らない答えと最小の続く答え
KIND_ANSWERS = {
    "mask": [("unit", "pair"), ("output", "region"),
             ("target.source", "process"),
             ("target.process.name", "今の塗り分け"),
             ("target.process.settings", "細かさ 1 画素"),
             ("method", "scratch"), ("target.kind", "mask"),
             ("evaluation.metric", "iou")],
    "point_cloud": [("unit", "bundle"), ("output", "shape"),
                    ("target.source", "files"), ("method", "scratch"),
                    ("target.kind", "point_cloud"),
                    ("evaluation.metric", "chamfer")],
}

# 学習まで通すのに要るもの (無ければ学習は飛ばす)
TRAINING = ("ml_found", "torch")


def kind_problem(kind: str) -> dict:
    """
    見本の課題から, 予測するで正解の種類がファイルの組の問題の設定を作る
    関数

    Parameters
    ----------
    kind: str
        正解の種類 (mask か point_cloud)

    Returns
    ----------
    problem: dict
        問題の設定
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = answer(problem, "ml.parts.0.role", "predict")
    for key, value in [("what", "画素ごとの領域"), ("item", "1 枚"),
                       ("baseline", "今のやり方"),
                       ("evaluation.link", "良くなると損が減る"),
                       *KIND_ANSWERS[kind]]:
        problem = answer(problem, f"ml.parts.0.{key}", value)

    return problem


def write_datasets(root: Path) -> dict:
    """
    ML-EDA の合成の見本から, 版 2 のデータセットを 2 つ書き出す関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    paths: dict
        正解の種類からデータセットの場所
    """
    # 画像と領域 (検証用とテスト用も), 撮影と点群
    pytest.importorskip("PIL")
    eda = pytest.importorskip("ml_eda")
    synthetic = pytest.importorskip("ml_eda.synthetic")
    split = {"method": "random", "valid": 0.25, "test": 0.25, "seed": 0}
    synthetic.write_image_masks(root / "raw_m", count=8, size=(16, 16))
    eda.export_image_masks(
        root / "raw_m" / "images", root / "raw_m" / "masks",
        root / "masks", split=split,
        classes={0: "background", 1: "circle", 2: "square"},
        origin={"type": "process", "name": "今の塗り分け",
                "settings": {"note": "細かさ 1 画素"}}
    )
    synthetic.write_shape_captures(root / "raw_c", count=6, views=3,
                                   size=(16, 16), points=64)
    eda.export_capture_clouds(
        root / "raw_c" / "captures", root / "raw_c" / "clouds",
        root / "clouds", split=split
    )
    paths = {"mask": root / "masks", "point_cloud": root / "clouds"}

    return paths


def train(project: Path) -> None:
    """
    書き出した学習のスクリプトを 2 エポックだけ動かす関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 別のプロセスで, 項目=値の上書き
    done = subprocess.run(
        [sys.executable, "train.py", "trainer.epochs=2",
         "experiment.device=cpu"],
        cwd=project, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize("kind", ["mask", "point_cloud"])
def test_items_project(tmp_path: Path, kind: str) -> None:
    """
    データを入れずに, ファイルの組の骨組みを書き出せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    kind: str
        正解の種類

    Returns
    ----------
    None
    """
    # 書き出せて, スクリプトは Python として読める
    problem = kind_problem(kind)
    assert project_error(problem) is None
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert "train.py" in written
    script = (project / "train.py").read_text(encoding="utf-8")
    compile(script, "train.py", "exec")
    assert "load_items" in script and "task_from_dataset" in script
    assert "PlaceholderModel" in script and "TODO" in script
    assert max(len(line) for line in script.splitlines()) <= 79

    # config: 監視は受け渡しから, 点群だけ検証の間隔. データセットは書き出す
    # 前の場所
    config = yaml.safe_load(
        (project / "configs/default/train.yaml").read_text(encoding="utf-8")
    )
    monitor = {"mask": ("valid/iou", "max"),
               "point_cloud": ("valid/chamfer", "min")}[kind]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        monitor
    )
    assert config["trainer"].get("valid_every_n_epochs") == (
        2 if kind == "point_cloud" else None
    )
    assert config["data"]["dataset"] == "outputs/dataset"
    assert config["model"]["spec"] is None
    assert config["experiment"]["name"] == f"{kind}_predict"

    # README: 書き出しの関数と, 今の処理の正解は超えられないこと
    readme = (project / "README.md").read_text(encoding="utf-8")
    function = {"mask": "export_image_masks",
                "point_cloud": "export_capture_clouds"}[kind]
    assert function in readme
    assert ("超えられない" in readme) == (kind == "mask")


def test_items_refused() -> None:
    """
    書き出せない受け持ちと指標の誤りが, make_project と同じ文になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 作るの点群, その他の指標, 文の正解
    generate = answer(kind_problem("point_cloud"), "ml.parts.0.role",
                      "generate")
    generate = answer(generate, "ml.parts.0.output", "shape")
    other = answer(kind_problem("mask"), "ml.parts.0.evaluation.metric",
                   "other")
    other = answer(other, "ml.parts.0.evaluation.name", "dice")
    other = answer(other, "ml.parts.0.evaluation.better", "higher")
    text = answer(kind_problem("mask"), "ml.parts.0.target.kind", "text")
    text = answer(text, "ml.parts.0.output", "text")
    text = answer(text, "ml.parts.0.evaluation.metric", "cer")
    for problem, words in ((generate, "まだ扱えません"), (other, "その他"),
                           (text, "まだ扱えません")):
        message = project_error(problem)
        assert message is not None and words in message, message
        with pytest.raises(ValueError) as raised:
            make_project(problem, Path("/nonexistent-ml-quest-project"))
        assert str(raised.value) == message


def test_items_with_data(tmp_path: Path) -> None:
    """
    版 2 のデータセットを入れ, 骨組みに写して学習まで通ることを確かめる
    関数 (ML-Found と PyTorch が無ければ学習は飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 入れると読み方は items, 列は変えず, 概要は空で件数の知らせ
    paths = write_datasets(tmp_path / "datasets")
    trainable = all(importlib.util.find_spec(name) for name in TRAINING)
    for kind, path in paths.items():
        before = kind_problem(kind)
        problem = attach_data(before, str(path),
                              copy_to=str(tmp_path / "store"))
        assert problem["data"]["loader"] == "items"
        assert problem["data"]["origin"] == str(path.resolve())
        assert problem["information"] == before["information"]
        found = inspect_data(problem)
        assert found["overview"] is None
        assert [item["level"] for item in found["findings"]] == ["info"]
        assert project_error(problem) is None

        # 骨組みはデータセットを data/ に写し, その場所を読む
        project = tmp_path / kind
        written = make_project(problem, project)
        assert f"data/{path.name}" in written
        assert (project / "data" / path.name / "dataset.yaml").is_file()
        config = yaml.safe_load((project / "configs/default/train.yaml")
                                .read_text(encoding="utf-8"))
        assert config["data"]["dataset"] == f"data/{path.name}"
        compile((project / "train.py").read_text(encoding="utf-8"),
                "train.py", "exec")
        readme = (project / "README.md").read_text(encoding="utf-8")
        assert "入れた版 2 のデータセット" in readme
        if not trainable:
            continue

        # 2 エポックの学習と, task.yaml とテスト用の評価
        train(project)
        runs = sorted((project / "outputs").glob("*/*/*/task.yaml"))
        assert len(runs) == 1
        task = yaml.safe_load(runs[0].read_text(encoding="utf-8"))
        assert task["target"]["kind"] == kind
        assert task["output"] == {"mask": "region",
                                  "point_cloud": "shape"}[kind]
        evaluations = list(runs[0].parent.glob("evaluations/*.json"))
        assert len(evaluations) == 1


def test_items_mismatch(tmp_path: Path) -> None:
    """
    受け持ちに合わないデータセットを入れると書き出せないことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 点群の問題に領域のデータセット
    paths = write_datasets(tmp_path / "datasets")
    cloud = attach_data(kind_problem("point_cloud"), str(paths["mask"]))
    message = project_error(cloud)
    assert message is not None and "入れたデータセット" in message
    with pytest.raises(ValueError) as raised:
        make_project(cloud, tmp_path / "never")
    assert str(raised.value) == message
    found = inspect_data(cloud)
    assert "error" in [item["level"] for item in found["findings"]]

    # 列の予測はデータセットを見ない (読めない dataset.yaml でも今までどおり)
    # ただし版 2 を入れると, 分析の確かめで使えないと知らせる
    column = attach_data(example_problem("総菜の作る量"), str(paths["mask"]))
    assert project_error(column) is None
    found = inspect_data(column)
    assert any(item["level"] == "error" and "列の予測" in item["message"]
               for item in found["findings"])
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "dataset.yaml").write_text("[: 壊れた", encoding="utf-8")
    assert project_error(answer(example_problem("総菜の作る量"),
                                "data.source", str(broken))) is None

    # 版 1 のデータセットを入れたファイルの組の受け持ちも書き出せない
    old = tmp_path / "old"
    old.mkdir()
    (old / "dataset.yaml").write_text("format_version: 1\n",
                                      encoding="utf-8")
    message = project_error(answer(kind_problem("mask"), "data.source",
                                   str(old)))
    assert message is not None and "版が 1" in message


def test_experiment_name_quoted(tmp_path: Path) -> None:
    """
    実験の名前が YAML の他の型に読める文字でも, 文字のまま書くことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 列の予測とファイルの組の両方
    for problem, folder in ((example_problem("総菜の作る量"), "column"),
                            (kind_problem("mask"), "mask")):
        for name in ("null", "true", "123"):
            project = tmp_path / folder / name
            make_project(problem, project, name=name)
            config = yaml.safe_load(
                (project / "configs/default/train.yaml")
                .read_text(encoding="utf-8")
            )
            assert config["experiment"]["name"] == name


def test_mask_background(tmp_path: Path) -> None:
    """
    名前の無い値 0 を背景のクラスとして足し, 他の名前の無い値は除くことを,
    書き出したスクリプトの関数で確かめる関数 (ML-Found と PyTorch が無けれ
    ば飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 値 0 と 2 に名前を付けない書き出し
    for name in TRAINING:
        pytest.importorskip(name)
    pytest.importorskip("PIL")
    eda = pytest.importorskip("ml_eda")
    synthetic = pytest.importorskip("ml_eda.synthetic")
    synthetic.write_image_masks(tmp_path / "raw", count=6, size=(16, 16))
    eda.export_image_masks(tmp_path / "raw" / "images",
                           tmp_path / "raw" / "masks", tmp_path / "masks",
                           classes={1: "circle"})
    problem = attach_data(kind_problem("mask"), str(tmp_path / "masks"))
    project = tmp_path / "project"
    make_project(problem, project)

    # 書き出したスクリプトを読み込んで, 値の番号の表
    spec = importlib.util.spec_from_file_location(
        "generated_train", project / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lookup, names, foreground, unnamed = module.class_table(
        eda.load_items(project / "data" / "masks")
    )
    assert names == ["background", "circle"]
    assert foreground == [1]
    assert unnamed == [2]
    assert lookup.tolist() == [0, 1]
