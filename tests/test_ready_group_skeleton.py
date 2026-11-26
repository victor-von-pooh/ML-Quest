from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, apply_group, check, describe, example_problem, handoff,
    make_project, project_error, questions, record_project
)

# 受け持ちの場所
PART = "ml.parts.0"

# 見本の名前 (商品の写真を ResNet18 の埋め込みで 4 つの群に分ける)
EXAMPLE = "商品の写真の群分け (既製の埋め込み)"

# 小さな作り物の出どころ (チャンネルの平均を埋め込みにする 1 段の全結合.
# 本物の重みを取りに行かない). bundle.py を動かす前の読み込み
FAKE_SOURCE = '''import torch
from torch import nn

from ml_inference import READY_LIBRARIES
from ml_modelcontrol import register_unit


class TinyNet(nn.Module):
    def __init__(self, name: str = "tiny", weights: str = "V1"):
        # 親クラスのコンストラクタを呼び出す
        super(TinyNet, self).__init__()

        # チャンネルの平均をそのまま種類の logit に
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3, 3)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(3))
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        チャンネルの平均から種類の logit を求める関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 3, 高さ, 幅) の画像

        Returns
        ----------
        out: torch.Tensor
            (件数, 3) の logit
        """
        # 平均して全結合
        out = self.fc(self.flatten(self.pool(x)))

        return out


def tiny_weights(model: str, weights: str) -> dict:
    """
    作り物の出どころの重みの説明を返す関数

    Parameters
    ----------
    model: str
        モデルの名前
    weights: str
        重みの名前

    Returns
    ----------
    found: dict
        READY_LIBRARIES の関数の結果の形
    """
    # 名前は 1 つだけ
    if model != "tiny" or weights != "V1":
        raise ValueError(f"{model} の {weights} はありません")
    found = {
        "unit": "quest.TinyNet",
        "params": {"name": model, "weights": weights},
        "classes": ["red", "green", "blue"], "input": "image",
        "preprocess": {"resize": 10, "crop": 8, "interpolation": "bilinear",
                       "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "url": "https://example.com/tiny-0123abcd.pth",
        "file": "tiny-0123abcd.pth",
    }

    return found


register_unit(TinyNet, name="quest.TinyNet", overwrite=True)
READY_LIBRARIES["test"] = tiny_weights
'''

# 作り物の出どころを読み込んでから bundle.py を動かすスクリプト
RUNNER = '''import runpy
import sys

sys.path.insert(0, sys.argv[1])
import fake_source  # noqa: F401

sys.argv = [sys.argv[2]]
runpy.run_path(sys.argv[0], run_name="__main__")
'''

# 既製の埋め込みの群分けの一式を作れない古い ML-Inference の作り物
OLD_INFERENCE = "BUNDLE_FORMAT_VERSION = 9\n"

# 写真の色 (赤, 緑, 青のチャンネル) と, 色ごとの枚数
COLORS = (0, 1, 2)
PER_COLOR = 4


def write_photos(folder: Path) -> Path:
    """
    赤, 緑, 青の写真 (大きさと濃さは別々) と読めないファイルを書く関数

    Parameters
    ----------
    folder: Path
        書くディレクトリ

    Returns
    ----------
    folder: Path
        書いたディレクトリ
    """
    # 色ごとに少しずつ違う濃さと大きさ
    image = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    folder.mkdir(parents=True)
    for channel in COLORS:
        for index in range(PER_COLOR):
            size = (12 + 2 * index, 10 + channel)
            array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            array[:, :, channel] = 180 + 15 * index
            image.fromarray(array).save(folder / f"{channel}_{index}.png")
    (folder / "broken.png").write_bytes(b"not an image")

    return folder


def tiny_project(project: Path) -> None:
    """
    見本から書き出し, config の既製のモデルを作り物の出どころに替える関数

    Parameters
    ----------
    project: Path
        書き出すディレクトリ

    Returns
    ----------
    None
    """
    # 出どころ, モデル, 重みの名前 (hash は決めない) と群の数は config の
    # 差分だけ (骨組みは書き換えない. 作り物の出どころは問いの選択肢に無い)
    make_project(answer(example_problem(EXAMPLE), f"{PART}.group.k",
                        len(COLORS)), project)
    path = project / "configs/default/bundle.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["ready"] = {"library": "test", "model": "tiny", "weights": "V1",
                       "sha256": None}
    path.write_text(yaml.safe_dump(config, allow_unicode=True),
                    encoding="utf-8")


def run_bundle(
    project: Path, tmp_path: Path, fake: Path
) -> subprocess.CompletedProcess:
    """
    書き出した bundle.py を, 作り物の出どころを読み込んでから動かす関数

    Parameters
    ----------
    project: Path
        書き出したプロジェクトのディレクトリ
    tmp_path: Path
        一時ディレクトリ
    fake: Path
        作り物の出どころの fake_source.py を置いたディレクトリ

    Returns
    ----------
    done: subprocess.CompletedProcess
        動かした結果
    """
    # 動かす場所はプロジェクトの外
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(runner), str(fake), str(project / "bundle.py")],
        capture_output=True, text=True, cwd=str(tmp_path)
    )

    return done


def test_ready_group_questions() -> None:
    """
    まとめるで既製のモデルをそのまま使うと答えると, 一式づくりに渡す値と
    群の数と安定性の数え方を問い, 群を作る列と学習の指標を問わないことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本は error 無し
    problem = example_problem(EXAMPLE)
    assert [item for item in check(problem)
            if item["level"] != "info"] == []
    applies = {item["id"]: item["applies"] for item in questions(problem)}
    for key in ("ready.library", "ready.model", "ready.weights",
                "ready.sha256", "group.k", "group.stability"):
        assert applies[f"{PART}.{key}"] is True, key
    for key in (f"{PART}.group.columns", f"{PART}.group.id",
                "information.columns", f"{PART}.evaluation.metric"):
        assert applies[key] is False, key

    # 学習する作り方では一式づくりの値を問わず, 群を作る列を問う
    trained = answer(problem, f"{PART}.method", "scratch")
    applies = {item["id"]: item["applies"] for item in questions(trained)}
    assert applies[f"{PART}.ready.model"] is False
    assert applies[f"{PART}.group.columns"] is True

    # 群の数と一式づくりの値の確かめ
    for key, value in ((f"{PART}.group.k", 1), (f"{PART}.group.k", None),
                       (f"{PART}.ready.weights", None),
                       (f"{PART}.ready.sha256", "abc")):
        found = check(answer(problem, key, value))
        assert any(item["id"] == key and item["level"] == "error"
                   for item in found), (key, value)


def test_ready_group_handoff() -> None:
    """
    既製の埋め込みの群分けの受け渡しと要約の文を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一式づくりの値は推論の ready, 学習とデータの分析と群の値は空
    stages = handoff(example_problem(EXAMPLE))
    assert stages["inference"]["ready"] == {
        "library": "torchvision", "model": "resnet18",
        "weights": "IMAGENET1K_V1",
        "sha256": ("8ec1484cdbb88ded96d21c3a5936faf3cdcc665aef92f331f675d0"
                   "c63038f0ba"),
    }
    assert stages["inference"]["decision"] is None
    assert stages["group"] is None
    assert stages["data"]["split"] is None
    assert stages["training"]["monitor"] is None
    assert stages["part"]["method"] == "pretrained"
    sentences = describe(example_problem(EXAMPLE))
    assert any(sentence.startswith("学習せずに使う既製のモデル: torchvision")
               for sentence in sentences)
    assert ("まとめ方: 写真を既製のモデルの埋め込みで 4 つの群に分ける "
            "(群の安定性は種を変えて作り直すときの一致で数える)") in sentences


def test_ready_group_apply(tmp_path: Path) -> None:
    """
    既製の埋め込みの群分けに群の数と数え方を書き戻すと, 答え, 確かめと
    骨組みの config が選んだ値になり, 列の選び方と誤った値を断ることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 群の数と数え方の書き戻し (元は変えない. 受け渡しの group は空のまま)
    problem = example_problem(EXAMPLE)
    result = apply_group(problem, {"k": 6.0, "stability": "sample"})
    assert problem["ml"]["parts"][0]["group"]["k"] == 4
    assert result["changes"] == ["群の数を 6 に書き戻した",
                                 "群の安定性の数え方を sample に書き戻した"]
    updated = result["problem"]
    assert [item for item in check(updated) if item["level"] == "error"] == []
    assert handoff(updated)["group"] is None
    make_project(updated, tmp_path / "project")
    config = yaml.safe_load(
        (tmp_path / "project" / "configs/default/bundle.yaml")
        .read_text(encoding="utf-8")
    )
    assert (config["group"]["k"], config["group"]["stability"]) == (
        6, "sample"
    )
    assert apply_group(problem, {"k": 4})["changes"] == []

    # 列の選び方は無く, 群の数と数え方の誤りは断る
    for group, words in (
        ({"columns": ["a"]}, "群を作る列がありません"),
        ({"id": None, "k": 3}, "行を見分ける列がありません"),
        ({"k": 1}, "2 以上の整数"), ({"k": None}, "群の数"),
        ({"stability": "bootstrap"}, "stability"),
    ):
        with pytest.raises(ValueError, match=words):
            apply_group(problem, group)


def test_ready_group_errors(tmp_path: Path) -> None:
    """
    既製の埋め込みの群分けで書き出せない場合が, make_project と同じ文に
    なることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 写真のディレクトリでない入れたデータと, 群でない出力の形
    table = example_problem(EXAMPLE)
    table["data"] = {"source": str(tmp_path / "rows.csv"), "loader": "csv"}
    ranking = answer(example_problem(EXAMPLE), f"{PART}.role", "rank")
    ranking = answer(ranking, f"{PART}.output", "ranking")
    for problem, words in (
        (table, "画像のディレクトリとして読んでいません"),
        (ranking, "「近い順に上位 K 件を返す」"),
    ):
        message = project_error(problem)
        assert message is not None and words in message, message
        with pytest.raises(ValueError) as raised:
            make_project(problem, tmp_path / "project")
        assert str(raised.value) == message
    assert not (tmp_path / "project").exists()


def test_ready_group_project(tmp_path: Path) -> None:
    """
    見本から既製の埋め込みの群分けの骨組みを書き出し, ファイル, config,
    README, スクリプトと記録を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # ファイルの並び (学習の config と train.py は無い)
    project = tmp_path / "project"
    problem = example_problem(EXAMPLE)
    assert project_error(problem) is None
    make_project(problem, project)
    files = sorted(str(path.relative_to(project))
                   for path in project.rglob("*") if path.is_file())
    assert files == ["README.md", "bundle.py", "configs/default/bundle.yaml",
                     "problem.yaml"]

    # config は問題の設定の答えのまま
    config = yaml.safe_load(
        (project / "configs/default/bundle.yaml").read_text(encoding="utf-8")
    )
    assert config["ready"]["model"] == "resnet18"
    assert config["ready"]["sha256"].startswith("8ec1484c")
    assert config["data"] == {"photos": "outputs/photos", "valid": 0.2}
    assert config["group"]["k"] == 4
    assert config["group"]["stability"] == "seed"
    assert config["outputs"] == {"embedder": "outputs/embedder",
                                 "groups": "outputs/groups"}
    assert config["bundle"]["name"] == "resnet18_group_4"

    # README とスクリプトの 79 文字以内と, Python として読めること
    script = (project / "bundle.py").read_text(encoding="utf-8")
    compile(script, "bundle.py", "exec")
    for name in ("bundle.py", "configs/default/bundle.yaml"):
        text = (project / name).read_text(encoding="utf-8")
        assert max(len(line) for line in text.splitlines()) <= 79, name
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "python bundle.py" in readme and "版 10" in readme
    assert "写真を縮めた画素をそのまま k-means で分けた群" in script

    # 記録はプロジェクトの場所だけ (データセットと定義は無い)
    recorded = record_project(problem, project)["project"]
    assert recorded["path"] == str(project.resolve())
    assert recorded["dataset"] is None and recorded["model_spec"] is None


def test_ready_group_script_stops(tmp_path: Path) -> None:
    """
    書き出した bundle.py が, 写真が無い場合と古い ML-Inference で止まる
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # スクリプトの読み込みに要るもの
    for name in ("numpy", "pandas", "torch"):
        pytest.importorskip(name)
    project = tmp_path / "project"
    make_project(example_problem(EXAMPLE), project)
    script = str(project / "bundle.py")

    # 写真の置き場の無い場合
    done = subprocess.run([sys.executable, script], capture_output=True,
                          text=True, cwd=str(tmp_path))
    assert done.returncode == 1
    assert "outputs/photos がありません" in done.stdout

    # 版 10 の一式を作れない ML-Inference
    write_photos(project / "outputs" / "photos")
    package = tmp_path / "fake" / "ml_inference"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(OLD_INFERENCE, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(package.parent)}
    done = subprocess.run([sys.executable, script], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 1
    assert "既製の埋め込みの群分けの一式 (版 10) を作れません" in done.stdout

    # config の群の設定の誤り
    path = project / "configs/default/bundle.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["group"]["k"] = 1
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    done = subprocess.run([sys.executable, script], capture_output=True,
                          text=True, cwd=str(tmp_path))
    assert done.returncode == 1
    assert "config の group.k は 2 以上の整数" in done.stdout


@pytest.mark.skipif(
    any(find_spec(name) is None for name in (
        "torch", "PIL", "ml_eda", "ml_modelcontrol", "ml_inference"
    )),
    reason="PyTorch, Pillow, ML-EDA, ML-ModelControl, ML-Inference が要る"
)
def test_ready_group_run(tmp_path: Path) -> None:
    """
    小さな作り物の出どころと写真で bundle.py を動かし, 群の割り当て, 群の
    要約, 群の安定性と既製の埋め込みの群分けの一式ができることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 10 の一式を作れる ML-Inference
    import numpy as np
    import pandas as pd
    import ml_inference
    from ml_eda import Data, describe_groups
    if ml_inference.BUNDLE_FORMAT_VERSION < 10:
        pytest.fail("ML-Inference の版 10 の一式づくりが要る")

    # 作り物の出どころで書き出して動かす
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "fake_source.py").write_text(FAKE_SOURCE, encoding="utf-8")
    project = tmp_path / "project"
    tiny_project(project)
    write_photos(project / "outputs" / "photos")
    done = run_bundle(project, tmp_path, fake)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "読めない写真 1 枚を除きます" in done.stdout
    assert "割り当て直しの一致: 1.0" in done.stdout
    assert "この hash を config の ready.sha256" in done.stdout

    # 割り当ては読めた写真ごとで, 色ごとに 1 つの群
    groups = project / "outputs" / "groups"
    table = pd.read_csv(groups / "group_assignments.csv")
    assert list(table.columns) == ["path", "group", "split"]
    assert len(table) == len(COLORS) * PER_COLOR
    assert (table["split"] == "valid").sum() == 2
    colors = table["path"].str[0]
    assert table.groupby(colors)["group"].nunique().eq(1).all()
    assert table["group"].nunique() == len(COLORS)

    # 群の中心と既製のモデル
    centers = json.loads((groups / "group_centers.json").read_text())
    assert centers["k"] == len(COLORS)
    assert centers["ready"]["library"] == "test"
    assert centers["ready"]["weights"] == "V1"
    assert len(centers["ready"]["sha256"]) == 64
    assert np.array(centers["centers"]).shape == (len(COLORS), 3)

    # 群の安定性 (比べる相手は縮めた画素) と群の要約 (ML-EDA と同じ行)
    stability = json.loads(
        (groups / "evaluations" / "group_stability.json").read_text()
    )
    row = stability["rows"][0]
    assert stability["baseline"] == (
        "写真を縮めた画素をそのまま k-means で分けた群"
    )
    assert stability["split"] == "all"
    assert stability["better"]["ari_diff"] == "higher"
    assert row["k"] == len(COLORS) and row["method"] == "seed"
    assert row["ari"] == pytest.approx(1.0)
    assert row["ari_diff"] == pytest.approx(row["ari"] - row["ari_baseline"])
    summary = json.loads(
        (groups / "evaluations" / "group_summary.json").read_text()
    )["rows"]
    assert summary[-1]["group"] == "全体" and summary[-1]["stability"] is None
    assert all(item["stability"] == pytest.approx(1.0)
               for item in summary[:-1])
    expected = describe_groups(
        Data(kind="table", frame=pd.DataFrame({
            "width": [12 + 2 * int(name[2]) for name in table["path"]],
            "height": [10 + int(name[0]) for name in table["path"]],
        })), columns=["width", "height"], labels=table["group"].tolist()
    )["groups"]
    for found, wanted in zip(summary, expected):
        assert {key: value for key, value in found.items()
                if key != "stability"} == {
            key: value for key, value in wanted.items() if key != "stability"
        }

    # 一式は版 10 (群の中心から遠い行の閾値を持てば版 17) で, 推論の群が
    # 割り当ての表と同じ
    bundle = project / "outputs" / "inference"
    info = yaml.safe_load((bundle / "inference.yaml").read_text())
    assert info["format_version"] in (10, 17)
    assert info["group"]["check"] == {"rows": len(table), "agreement": 1.0}
    script = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(fake)!r})\n"
        "import fake_source\n"
        "from ml_inference import predict\n"
        f"frame = predict({str(bundle)!r}, "
        f"{str(project / 'outputs' / 'photos')!r})\n"
        "ok = frame[frame['status'] == 'ok']\n"
        "print(json.dumps(dict(zip(ok['path'], ok['group'].astype(int)))))\n"
    )
    done = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    predicted = json.loads(done.stdout.strip().splitlines()[-1])
    assert predicted == dict(zip(table["path"], table["group"]))
