import ast
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, apply_recipe, attach_data, check, example_problem, handoff,
    make_project, project_error, recipe_from, record_project
)

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の画像の件数と大きさ
COUNT = 24
SIZE = 8

# 学習と推論の一式まで通すのに要るもの (CI の train のジョブで入れる)
TRAINING = ("torch", "ml_found", "ml_modelcontrol", "ml_inference")

# 骨組みを書き換えずに短く学ぶ config の差分 (少ない時刻と繰り返し)
SHORT = ["trainer.epochs=2", "experiment.device=cpu",
         "diffusion.train_steps=50", "diffusion.sample_steps=5"]

# FID を数える骨組みの差分 (重みを取らない画素の特徴で, 毎エポック検証)
FID_SHORT = ("fid.features=pixels", "trainer.valid_every_n_epochs=1")


def generate_problem() -> dict:
    """
    見本の課題から, 画像の生成の受け持ちの問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        作るで正解の種類と出力の形が画像, 指標が loss の問題の設定 (データ
        は入れていない)
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    for key, value in [("role", "generate"), ("unit", "row"),
                       ("output", "image"), ("target.source", "self"),
                       ("target.kind", "image"), ("method", "scratch"),
                       ("what", "丸の画像"), ("item", "1 枚の画像"),
                       ("baseline", "手で描く"),
                       ("evaluation.metric", "loss"),
                       ("evaluation.link", "見本の画像で確かめる")]:
        problem = answer(problem, f"{PART}.{key}", value)

    return problem


def write_images(root: Path) -> Path:
    """
    Pillow で丸を描いた, クラスの無い画像のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        <画像>.png のディレクトリ
    """
    # 黒地に白の丸 (位置を少しずつずらす)
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    path = root / "images"
    path.mkdir(parents=True)
    for index in range(COUNT):
        image = image_module.new("L", (SIZE, SIZE), "black")
        draw = draw_module.Draw(image)
        shift = index % 3
        draw.ellipse((1 + shift, 1 + shift, 5 + shift, 5 + shift),
                     fill="white")
        image.save(path / f"{index}.png")

    return path


def attached(root: Path) -> dict:
    """
    画像のディレクトリを入れた画像の生成の問題の設定を作る関数

    Parameters
    ----------
    root: Path
        データと写しを置く場所

    Returns
    ----------
    problem: dict
        データを入れた問題の設定
    """
    # ML-EDA の image_folder で読む
    pytest.importorskip("ml_eda")
    source = write_images(root)
    problem = attach_data(generate_problem(), str(source),
                          copy_to=str(root / "copies"))

    return problem


def export(problem: dict, project: Path) -> Path:
    """
    レシピの雛形で, 骨組みの場所にデータセットを書き出す関数

    Parameters
    ----------
    problem: dict
        データを入れた問題の設定
    project: Path
        書き出したプロジェクトのディレクトリ

    Returns
    ----------
    place: Path
        書き出したデータセット (config の data.dataset)
    """
    # record_project の場所 (train.py が読む場所) に書き出し
    eda = pytest.importorskip("ml_eda")
    start = recipe_from(problem)
    load = start["load"]
    data = eda.load_data(load["path"], loader=load["loader"],
                         **(load["options"] or {}))
    place = Path(record_project(problem, project)["project"]["dataset"])
    eda.export_dataset(data, start["recipe"], place)

    return place


def denoiser_spec() -> dict:
    """
    雑音を加えた画像と時刻から雑音を当てる, ML-ModelControl の小さな定義を
    作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        画像を平らにした全結合に時刻の埋め込みを足し, 画像の形に戻す定義
    """
    # 画像 (1, 8, 8) と int64 の時刻の 2 つの入力
    pixels = SIZE * SIZE
    spec = {
        "format_version": 1, "name": "denoiser",
        "inputs": {"noisy": {"shape": [2, 1, SIZE, SIZE]},
                   "step": {"shape": [2], "dtype": "int64",
                            "fill": "arange"}},
        "nodes": {
            "flat": {"unit": "nn.Flatten", "inputs": "noisy"},
            "x_fc": {"unit": "nn.Linear", "inputs": "flat",
                     "params": {"in_features": pixels, "out_features": 32}},
            "t_emb": {"unit": "op.SinusoidalEmbedding", "inputs": "step",
                      "params": {"embedding_dim": 32}},
            "add": {"unit": "op.Add", "inputs": {"xs": ["x_fc", "t_emb"]}},
            "act": {"unit": "nn.SiLU", "inputs": "add"},
            "head": {"unit": "nn.Linear", "inputs": "act",
                     "params": {"in_features": 32, "out_features": pixels}},
            "shape": {"unit": "nn.Unflatten", "inputs": "head",
                      "params": {"dim": 1,
                                 "unflattened_size": [1, SIZE, SIZE]}},
        },
        "outputs": {"noise": "shape"},
    }

    return spec


def train(project: Path, extra: tuple = ()) -> tuple[Path, str]:
    """
    書き出した骨組みを書き換えずに, 別のプロセスで短く学習する関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    extra: tuple = ()
        SHORT に足す config の差分

    Returns
    ----------
    run: Path
        実験のディレクトリ
    stdout: str
        学習のスクリプトの表示
    """
    # config の差分だけを渡す
    done = subprocess.run([sys.executable, "train.py", *SHORT, *extra],
                          cwd=project, capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    runs = sorted((project / "outputs").glob("*/*/*/train.log"))
    assert len(runs) == 1
    run = runs[0].parent

    return run, done.stdout


def history_rows(run: Path) -> list[dict]:
    """
    実験の history.csv を読む関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ

    Returns
    ----------
    rows: list[dict]
        エポックの順の行 (列の名前から値の文字)
    """
    # エポックごとの行
    with open(run / "history.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    return rows


def test_generate_handoff() -> None:
    """
    画像の生成の問題の設定が渡せ, 分割は無作為で, 損失の監視になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 誤りが無く, 書き出せる. 監視は valid/loss
    problem = generate_problem()
    assert [item for item in check(problem) if item["level"] == "error"] == []
    assert project_error(problem) is None
    stages = handoff(problem)
    assert stages["data"]["split"] == "random"
    assert stages["data"]["target"] is None and stages["data"]["task"] is None
    assert (stages["training"]["monitor"], stages["training"]["mode"]) == (
        "valid/loss", "min"
    )
    assert stages["inference"]["decision"] is None
    assert stages["policy"] is None

    # FID を選んでも書き出せ, 監視は valid/fid
    fid = answer(problem, f"{PART}.evaluation.metric", "fid")
    assert project_error(fid) is None
    assert (handoff(fid)["training"]["monitor"],
            handoff(fid)["training"]["mode"]) == ("valid/fid", "min")


def test_generate_errors(tmp_path: Path) -> None:
    """
    画像の生成で書き出せない場合が, make_project と同じ文になることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 画像のディレクトリでないデータ, 画像でない出力の形
    table = generate_problem()
    table["data"] = {"source": str(tmp_path / "a.csv"), "loader": "csv"}
    cases = [(table, "画像のディレクトリとして読んでいません")]
    text = answer(generate_problem(), f"{PART}.target.kind", "text")
    text = answer(text, f"{PART}.output", "text")
    text = answer(text, f"{PART}.evaluation.metric", "loss")
    cases.append((text, "まだ扱えません"))
    for problem, words in cases:
        message = project_error(problem)
        assert message is not None and words in message, message
        with pytest.raises(ValueError) as raised:
            make_project(problem, tmp_path / "project")
        assert str(raised.value) == message
    for raw in ("loader", "csv)", "image_folder"):
        assert raw not in project_error(table), raw
    assert not (tmp_path / "project").exists()


def test_generate_recipe() -> None:
    """
    画像の生成のレシピの雛形が目的の列の無い無作為の分割で, 書き戻すと
    そのまま返り, 食い違いを知らせることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 雛形 (データを入れていないので読み方は無い)
    problem = generate_problem()
    start = recipe_from(problem)
    assert start["load"] is None and start["export"] is None
    assert start["recipe"] == {
        "format_version": 1, "task": "none", "steps": [],
        "split": {"method": "random", "valid": 0.15, "test": 0.15,
                  "seed": 0},
    }
    assert any("resize" in note for note in start["notes"])

    # 書き戻すとそのまま返り, 受け渡しの分割にも出る
    recipe = {**start["recipe"],
              "item_steps": [{"op": "resize", "width": 4, "height": 4}],
              "split": {"method": "none"}}
    result = apply_recipe(problem, recipe)
    assert result["changes"] == [] and result["choices"] == []
    assert [item["level"] for item in result["conflicts"]] == ["warning"]
    assert recipe_from(result["problem"])["recipe"] == recipe
    assert handoff(result["problem"])["data"]["split"] == "none"

    # 目的の列のあるレシピは食い違い
    result = apply_recipe(problem, {"target": "label",
                                    "task": "classification"})
    assert "error" in [item["level"] for item in result["conflicts"]]


def test_generate_project(tmp_path: Path) -> None:
    """
    画像のディレクトリを入れた画像の生成の問題の設定から骨組みを書き出し,
    案内どおりにデータセットを書き出せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出したファイル (画像は data/ に写す)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert set(written) == {
        "problem.yaml", "README.md", "configs/default/train.yaml",
        "configs/experiment/train.yaml", "train.py", "data/images",
    }
    script = (project / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'ROLE = "generate"' in script and "PlaceholderModel" in script
    assert 'PREDICT = "丸の画像"' in script

    # config: 要る機能, 監視, 検証の間隔, EMA, データセットとモデルの定義の
    # 場所, FID の特徴
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol", "inference"]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/loss", "min"
    )
    assert "valid_every_n_epochs" not in config["trainer"]
    assert "fid" not in config
    assert "COUNT_FID = False" in script
    assert config["trainer"]["ema"] == {"decay": 0.995}
    assert config["data"]["dataset"] == "outputs/dataset"
    assert config["model"]["spec"] == "models/model.yaml"
    assert config["model"]["placeholder"] is True
    assert config["diffusion"]["train_steps"] == 1000

    # README: データの分析, モデルの入力, FID, 雑音だけの一式
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("task: none", "data/images", "int64", "FID を選んで",
                  "版 5", "見本", "grayscale", "1 / (1 - decay)"):
        assert words in readme, words

    # 案内どおりの書き出しで, 目的の列の無い画像のデータセット
    place = export(problem, project)
    info = yaml.safe_load((place / "dataset.yaml").read_text(encoding="utf-8"))
    assert info["kind"] == "image" and info["task"] == "none"
    assert "valid" in info["splits"]


def test_generate_placeholder(tmp_path: Path) -> None:
    """
    モデルの定義が無いうちは仮のモデルで学び, 見本と評価を残し, 一式を作ら
    ないことを確かめる関数 (PyTorch と ML-Found が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 骨組みとデータセットだけで学習
    for name in ("torch", "ml_found"):
        pytest.importorskip(name)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    make_project(problem, project)
    export(problem, project)
    run, _ = train(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル (PlaceholderModel) で学習します" in log
    assert "推論の一式を作りません" in log
    assert not (run / "inference").exists()
    assert (run / "evaluations" / "test_loss.json").is_file()
    assert (run / "samples" / "generated_0" / "final.png").is_file()

    # 損失で選ぶので FID は数えない (毎エポック検証)
    rows = history_rows(run)
    assert len(rows) == 2 and all(row["valid/loss"] for row in rows)
    assert "valid/fid" not in rows[0]


def test_generate_fid_monitor(tmp_path: Path) -> None:
    """
    FID を最良を選ぶ指標にした骨組みが, fid の機能と検証の間隔を書き,
    検証するエポックの valid/fid で最良のモデルを選ぶことを確かめる関数
    (PyTorch と ML-Found が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # config: fid の機能, 検証の間隔, FID の設定. README に FID の数え方
    for name in ("torch", "ml_found"):
        pytest.importorskip(name)
    problem = answer(attached(tmp_path), f"{PART}.evaluation.metric", "fid")
    project = tmp_path / "project"
    make_project(problem, project)
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol", "inference", "fid"]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/fid", "min"
    )
    assert config["trainer"]["valid_every_n_epochs"] == 5
    assert config["fid"] == {"features": "inception", "dims": 2048}
    script = (project / "train.py").read_text(encoding="utf-8")
    assert "COUNT_FID = True" in script
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("`valid/fid`", "pytorch-fid", "fid.features: pixels",
                  "小さな画像の確かめ用"):
        assert words in readme, words
    export(problem, project)

    # FID の特徴の誤りは出力先を作る前に止まる
    done = subprocess.run([sys.executable, "train.py", "fid.features=vgg"],
                          cwd=project, capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 1
    assert "fid.features は inception か pixels" in done.stdout
    assert not (project / "outputs").exists() or not any(
        (project / "outputs").glob("*/*/*/train.log")
    )

    # 最良は記録した FID の小さいエポック
    run, stdout = train(project, FID_SHORT)
    values = [float(row["valid/fid"]) for row in history_rows(run)]
    assert len(values) == 2 and min(values) > 0
    assert f"ベストの valid/fid: {min(values)}" in stdout

    # 間隔 5 のままなら, 2 エポックでは最後のエポックだけ数える
    for path in (project / "outputs").glob("*/*/*/train.log"):
        path.unlink()
    run, _ = train(project, ("fid.features=pixels",))
    rows = history_rows(run)
    assert rows[0]["valid/fid"] == "" and float(rows[1]["valid/fid"]) > 0


def test_generate_sample_steps(tmp_path: Path) -> None:
    """
    繰り返しの回数が学習の時刻の数より多い config は, 出力先を作る前に
    止まることを確かめる関数 (PyTorch と ML-Found が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 時刻 50 に繰り返し 60
    for name in ("torch", "ml_found"):
        pytest.importorskip(name)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    make_project(problem, project)
    export(problem, project)
    done = subprocess.run(
        [sys.executable, "train.py", "diffusion.train_steps=50",
         "diffusion.sample_steps=60"],
        cwd=project, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 1
    assert "diffusion.sample_steps は 1 から" in done.stdout
    assert not (project / "outputs").exists() or not any(
        (project / "outputs").glob("*/*/*/train.log")
    )


def test_generate_training(tmp_path: Path) -> None:
    """
    画像の生成の骨組みを書き換えずに学び, 雑音だけの推論の一式が見本と
    同じ画像を作ることを確かめる関数 (PyTorch と段階のライブラリが無ければ
    飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出し, データセット, 定義 (record_project の場所)
    for name in TRAINING:
        pytest.importorskip(name)
    import numpy as np
    from PIL import Image
    from ml_inference import load_bundle, predict
    problem = attached(tmp_path)
    project = tmp_path / "project"
    make_project(problem, project)
    export(problem, project)
    place = Path(record_project(problem, project)["project"]["model_spec"])
    place.write_text(yaml.safe_dump(denoiser_spec(), allow_unicode=True),
                     encoding="utf-8")
    run, _ = train(project)

    # 何の実験か, EMA の重み, テスト用の損失
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"], task["what"]) == (
        "generate", "image", "丸の画像"
    )
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "EMA" in log and "テスト用の損失" in log
    assert "作るもの: 丸の画像" in log and "予測すること" not in log
    table = json.loads((run / "evaluations" / "test_loss.json")
                       .read_text(encoding="utf-8"))
    assert table["split"] == "test" and table["better"] == {"loss": "lower"}
    assert table["rows"][0]["loss"] > 0

    # 見本: 作った画像 8 件と, 見比べる本物の画像
    index = json.loads((run / "samples" / "index.json")
                       .read_text(encoding="utf-8"))
    names = {item["name"] for item in index["samples"]}
    assert {f"generated_{i}" for i in range(8)} <= names
    assert "valid_0_real" in names

    # 雑音だけの一式 (版 5): 学習と同じ雑音の予定の ddim
    bundle_dir = run / "inference"
    info = yaml.safe_load((bundle_dir / "inference.yaml")
                          .read_text(encoding="utf-8"))
    assert info["format_version"] == 5 and info["dataset"] is None
    assert info["role"] == "generate"
    assert info["output"] == {"form": "image", "shape": [1, SIZE, SIZE]}
    assert info["inputs"] == {"noisy": {"source": "state"},
                              "step": {"source": "time"}}
    assert info["procedure"]["name"] == "ddim"
    assert info["procedure"]["steps"] == 5
    assert info["procedure"]["options"]["train_steps"] == 50

    # 一式で作った画像は, 同じ種の見本と同じ (見本は 0 から 1 に収めた PNG)
    frame = predict(load_bundle(bundle_dir), {"count": 8, "seed": 0})
    assert list(frame["status"]) == ["ok"] * 8
    for number in range(8):
        made = np.clip(np.asarray(frame["output"].iloc[number])[0], 0, 1)
        png = np.asarray(Image.open(
            run / "samples" / f"generated_{number}" / "final.png"
        ), dtype=float)
        png = png if png.ndim == 2 else png[..., 0]
        assert np.abs(made - png / 255).max() < 2.5 / 255, number
