import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, attach_data, check, describe, example_problem, handoff,
    inspect_data, is_column_part, make_project, project_error, questions,
    record_project
)

# 受け持ちの場所
PART = "ml.parts.0"

# 見本の名前 (ResNet18 の画像の分類)
EXAMPLE = "写真の仕分け (既製のモデル)"

# 見本の重みの hash (torchvision の resnet18 の IMAGENET1K_V1)
SHA256 = ("8ec1484cdbb88ded96d21c3a5936faf3cdcc665aef92f331f675d0c63038f0"
          "ba")

# 作り物の ML-Inference (make_ready_bundle の口だけ. 受け取った値を記録の
# ファイルに書き, inference.yaml の中身の形を返す)
FAKE_INFERENCE = '''import json
import os
from pathlib import Path


def make_ready_bundle(
    path, model, weights, library="torchvision", sha256=None, name=None,
    license=None, conclusion=None, overwrite=False
):
    # 受け取った値の記録
    Path(os.environ["READY_CALLS"]).write_text(json.dumps({
        "path": str(path), "model": model, "weights": weights,
        "library": library, "sha256": sha256, "name": name,
        "license": license, "conclusion": conclusion,
        "overwrite": overwrite,
    }), encoding="utf-8")
    if sha256 is not None and sha256 != os.environ["READY_SHA256"]:
        raise ValueError("hash が決めておいたものと違います")
    info = {
        "format_version": 6, "name": name or Path(path).name,
        "output": {"form": "class", "classes": ["tench", "goldfish"]},
        "model": {"pretrained": [{"sha256": os.environ["READY_SHA256"]}]},
        "ready": {"library": library, "model": model, "weights": weights},
        "notes": ["重みは出どころから読む"],
    }
    return info
'''

# 既製のモデルの一式を作れない古い ML-Inference の作り物
OLD_INFERENCE = "BUNDLE_FORMAT_VERSION = 5\n"


def ready_problem() -> dict:
    """
    既製のモデルをそのまま使う見本の問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        ResNet18 の画像の分類の問題の設定 (重みの hash は決めていない)
    """
    # 見本の答えから hash を除いたもの
    problem = answer(example_problem(EXAMPLE), f"{PART}.ready.sha256", None)

    return problem


def run_bundle(
    project: Path, fake: str, tmp_path: Path
) -> tuple[subprocess.CompletedProcess, Path]:
    """
    書き出した bundle.py を, 作り物の ML-Inference で動かす関数

    Parameters
    ----------
    project: Path
        書き出したプロジェクトのディレクトリ
    fake: str
        作り物の ml_inference の __init__.py の中身
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    result: tuple[subprocess.CompletedProcess, Path]
        動かした結果と, 受け取った値の記録のファイル
    """
    # 入っている ML-Inference より先に作り物を読む
    package = tmp_path / "fake" / "ml_inference"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(fake, encoding="utf-8")
    calls = tmp_path / "calls.json"
    env = {**os.environ, "PYTHONPATH": str(package.parent),
           "READY_CALLS": str(calls), "READY_SHA256": SHA256}
    done = subprocess.run(
        [sys.executable, str(project / "bundle.py")], capture_output=True,
        text=True, env=env, cwd=str(tmp_path)
    )
    result = (done, calls)

    return result


def test_ready_questions() -> None:
    """
    既製のモデルをそのまま使うと答えると, 一式づくりに渡す値の問いが出て,
    列の予測と学習の指標の問いが要らなくなることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本は error 無しで, 列の予測ではない
    problem = ready_problem()
    assert [item for item in check(problem)
            if item["level"] == "error"] == []
    assert is_column_part(problem) is False
    applies = {item["id"]: item["applies"] for item in questions(problem)}
    for key in ("library", "model", "weights", "sha256"):
        assert applies[f"{PART}.ready.{key}"] is True
    for key in (f"{PART}.target.column", f"{PART}.task",
                "information.columns", "information.known",
                "action.decision", f"{PART}.evaluation.metric",
                f"{PART}.evaluation.link", f"{PART}.evaluation.current"):
        assert applies[key] is False, key
    library = next(item for item in questions(problem)
                   if item["id"] == f"{PART}.ready.library")
    assert [item["value"] for item in library["choices"]] == ["torchvision"]

    # 学習する作り方では出ず, 名前と重みは要る問い
    trained = answer(problem, f"{PART}.method", "scratch")
    applies = {item["id"]: item["applies"] for item in questions(trained)}
    assert applies[f"{PART}.ready.model"] is False
    assert applies[f"{PART}.evaluation.metric"] is True
    missing = answer(problem, f"{PART}.ready.weights", None)
    assert any(item["id"] == f"{PART}.ready.weights"
               and item["level"] == "error" for item in check(missing))


def test_ready_handoff() -> None:
    """
    受け渡しの推論の値に, 一式づくりに渡す値がそろうことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # hash を決めていなければ None, 学習の指標は無し
    stages = handoff(ready_problem())
    assert stages["inference"]["ready"] == {
        "library": "torchvision", "model": "resnet18",
        "weights": "IMAGENET1K_V1", "sha256": None,
    }
    assert stages["inference"]["decision"] is None
    assert stages["training"]["monitor"] is None
    assert stages["data"]["target"] is None
    assert any("resnet18 (重み IMAGENET1K_V1)" in text
               for text in describe(ready_problem()))

    # 大文字と前後の空白の hash は小文字に, 名前は空白を除く
    problem = answer(ready_problem(), f"{PART}.ready.sha256",
                     f" {SHA256.upper()} ")
    problem = answer(problem, f"{PART}.ready.model", " resnet18 ")
    ready = handoff(problem)["inference"]["ready"]
    assert ready["sha256"] == SHA256
    assert ready["model"] == "resnet18"

    # 形の違う hash は error
    wrong = answer(ready_problem(), f"{PART}.ready.sha256", "8ec1")
    assert any(item["id"] == f"{PART}.ready.sha256"
               and item["level"] == "error" for item in check(wrong))
    with pytest.raises(ValueError, match="64 文字"):
        handoff(wrong)

    # 学習する作り方では一式づくりの値は無し
    trained = example_problem("解約の引き止め")
    assert handoff(trained)["inference"]["ready"] is None


def test_ready_project(tmp_path: Path) -> None:
    """
    既製のモデルの骨組みが, 学習のスクリプトの代わりに一式づくりの設定と
    スクリプトを書き出すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 学習のファイルは無し (見本は hash を決めてある)
    problem = example_problem(EXAMPLE)
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert sorted(written) == ["README.md", "bundle.py",
                               "configs/default/bundle.yaml", "problem.yaml"]
    assert not (project / "train.py").exists()

    # config の値は受け渡しの値
    config = yaml.safe_load(
        (project / "configs" / "default" / "bundle.yaml").read_text("utf-8")
    )
    assert config["ready"] == {"library": "torchvision", "model": "resnet18",
                               "weights": "IMAGENET1K_V1", "sha256": SHA256}
    assert config["bundle"] == {"path": "outputs/inference",
                                "name": "resnet18_IMAGENET1K_V1",
                                "overwrite": False}
    assert config["conclusion"] is None

    # スクリプトは Python として読め, 79 文字以内
    script = (project / "bundle.py").read_text("utf-8")
    compile(script, "bundle.py", "exec")
    assert max(len(line) for line in script.splitlines()) <= 79
    readme = (project / "README.md").read_text("utf-8")
    assert "python bundle.py" in readme
    assert "学習の段階は通らない" in readme

    # 記録はデータセットとモデルの定義の無いプロジェクト
    recorded = record_project(problem, project)
    assert recorded["project"] == {"path": str(project.resolve()),
                                   "dataset": None, "model_spec": None}


def test_ready_script_calls_maker(tmp_path: Path) -> None:
    """
    書き出した bundle.py が, config の値を make_ready_bundle に渡すことを
    作り物の ML-Inference で確かめる関数

    ML-Inference の版 6 (make_ready_bundle) が無くても動く

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # hash を決めていない一式 (求めた hash を書き戻す案内)
    project = tmp_path / "project"
    make_project(ready_problem(), project, name="写真 分け")
    done, calls = run_bundle(project, FAKE_INFERENCE, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    received = json.loads(calls.read_text("utf-8"))
    assert received == {
        "path": str((project / "outputs" / "inference").resolve()),
        "model": "resnet18", "weights": "IMAGENET1K_V1",
        "library": "torchvision", "sha256": None, "name": "写真_分け",
        "license": None, "conclusion": None, "overwrite": False,
    }
    assert SHA256 in done.stdout
    assert "ready.sha256" in done.stdout
    assert "種類の数: 2" in done.stdout

    # 決めた hash と結論の決め方を渡し, 違う hash なら止まる
    path = project / "configs" / "default" / "bundle.yaml"
    config = yaml.safe_load(path.read_text("utf-8"))
    config["ready"]["sha256"] = SHA256.upper()
    config["conclusion"] = {"min_confidence": 0.6}
    path.write_text(yaml.safe_dump(config, allow_unicode=True), "utf-8")
    done, calls = run_bundle(project, FAKE_INFERENCE, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    received = json.loads(calls.read_text("utf-8"))
    assert received["sha256"] == SHA256
    assert received["conclusion"] == {"min_confidence": 0.6}
    config["ready"]["sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(config, allow_unicode=True), "utf-8")
    done, _ = run_bundle(project, FAKE_INFERENCE, tmp_path)
    assert done.returncode == 1
    assert "一式を作れません" in done.stdout

    # 名前の無い config は直す所を示して止まる
    config["ready"]["model"] = ""
    path.write_text(yaml.safe_dump(config, allow_unicode=True), "utf-8")
    done, _ = run_bundle(project, FAKE_INFERENCE, tmp_path)
    assert done.returncode == 1
    assert "ready の model" in done.stdout


def test_ready_script_old_inference(tmp_path: Path) -> None:
    """
    make_ready_bundle の無い ML-Inference では, 入れ方を示して止まることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 5 までの ML-Inference の作り物
    project = tmp_path / "project"
    make_project(ready_problem(), project)
    done, calls = run_bundle(project, OLD_INFERENCE, tmp_path)
    assert done.returncode == 1
    assert "make_ready_bundle" in done.stdout
    assert "ml-modelcontrol[torchvision]" in done.stdout
    assert not calls.exists()


def test_ready_project_limits() -> None:
    """
    既製のモデルの骨組みを書き出せない受け持ち, 正解の種類, 出力の形を
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 画像の分類だけ
    assert project_error(ready_problem()) is None
    value = answer(ready_problem(), f"{PART}.output", "value")
    assert "「クラス」の場合だけ" in project_error(value)
    mask = answer(ready_problem(), f"{PART}.target.kind", "mask")
    mask = answer(mask, f"{PART}.output", "region")
    assert "「表の列」の場合だけ" in project_error(mask)

    # 予測するとまとめるでない受け持ちは学習する骨組みも書き出さない
    generate = answer(ready_problem(), f"{PART}.role", "generate")
    generate = answer(generate, f"{PART}.output", "image")
    generate = answer(generate, f"{PART}.target.kind", "image")
    assert "予測するかまとめるの場合だけ" in project_error(generate)


def test_ready_project_with_images(tmp_path: Path) -> None:
    """
    画像のディレクトリを入れた既製のモデルの問題の設定も, 確かめの error
    無しに書き出し, 画像をプロジェクトに写すことを確かめる関数

    ML-EDA と Pillow が入っていなければ飛ばす

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 色の違う 3 枚の画像 (推論で渡す画像)
    pytest.importorskip("ml_eda")
    image_module = pytest.importorskip("PIL.Image")
    source = tmp_path / "photos"
    source.mkdir()
    for index, color in enumerate(("red", "green", "blue")):
        image = image_module.new("RGB", (8, 8), color)
        image.save(source / f"{index}.png")
    problem = attach_data(ready_problem(), str(source),
                          copy_to=str(tmp_path / "copies"))

    # 入れたデータの確かめに error は無く, 写した画像を README で示す
    found = inspect_data(problem)
    assert [item for item in (found or {}).get("findings", [])
            if item["level"] == "error"] == []
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert "data/photos" in written
    assert len(list((project / "data" / "photos").glob("*.png"))) == 3
    assert "`data/photos` に写した" in (project / "README.md").read_text(
        "utf-8"
    )
