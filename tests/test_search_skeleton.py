from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, check, describe, example_problem, handoff, make_project,
    project_error, questions, record_project
)

# 受け持ちの場所
PART = "ml.parts.0"

# 見本の名前 (商品の写真に似た写真を ResNet18 の埋め込みで上位 5 件返す)
EXAMPLE = "似た商品の写真を探す (既製の埋め込み)"

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

# 検索の一式を作れない古い ML-Inference の作り物
OLD_INFERENCE = "BUNDLE_FORMAT_VERSION = 14\n"

# 写真の種類 (赤, 緑, 青のチャンネル) と, 種類ごとの枚数
COLORS = {"red": 0, "green": 1, "blue": 2}
PER_COLOR = 5


def write_photos(folder: Path, labeled: bool = True) -> Path:
    """
    赤, 緑, 青の写真 (濃さは別々) と読めないファイルを書く関数

    Parameters
    ----------
    folder: Path
        書くディレクトリ
    labeled: bool = True
        種類のサブディレクトリに置くか

    Returns
    ----------
    folder: Path
        書いたディレクトリ
    """
    # 種類ごとに少しずつ違う濃さ (他のチャンネルにも少し)
    image = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    for name, channel in COLORS.items():
        place = folder / name if labeled else folder
        place.mkdir(parents=True, exist_ok=True)
        for index in range(PER_COLOR):
            array = np.full((10, 12, 3), 10 * index, dtype=np.uint8)
            array[:, :, channel] = 160 + 15 * index
            image.fromarray(array).save(place / f"{name}_{index}.png")
    (folder / ("red" if labeled else "") / "broken.png").write_bytes(
        b"not an image"
    )

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
    # 出どころ, モデル, 重みの名前 (hash は決めない) は config の差分だけ
    # (骨組みは書き換えない. 作り物の出どころは問いの選択肢に無い)
    make_project(example_problem(EXAMPLE), project)
    path = project / "configs/default/bundle.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["ready"] = {"library": "test", "model": "tiny", "weights": "V1",
                       "sha256": None}
    config["search"]["k"] = 3
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


def test_search_questions() -> None:
    """
    並べるで既製のモデルをそのまま使うと答えると, 一式づくりに渡す値と
    検索の判断を問い, 反応の表の列と学習の指標を問わないことを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本は確かめの指摘無し
    problem = example_problem(EXAMPLE)
    assert [item for item in check(problem)
            if item["level"] != "info"] == []
    applies = {item["id"]: item["applies"] for item in questions(problem)}
    for key in (f"{PART}.ready.library", f"{PART}.ready.model",
                f"{PART}.ready.weights", f"{PART}.ready.sha256",
                "action.decision"):
        assert applies[key] is True, key
    for key in (f"{PART}.rank.user", f"{PART}.rank.k", "information.columns",
                f"{PART}.evaluation.metric"):
        assert applies[key] is False, key
    choices = {item["id"]: item for item in questions(problem)}
    names = [c["value"] for c in choices["action.decision"]["choices"]]
    assert names == ["search"]

    # 学習する作り方では一式づくりの値と検索の判断を問わず, 反応の表を問う
    trained = answer(problem, f"{PART}.method", "scratch")
    applies = {item["id"]: item["applies"] for item in questions(trained)}
    assert applies[f"{PART}.ready.model"] is False
    assert applies["action.decision"] is False
    assert applies[f"{PART}.rank.user"] is True

    # 検索の判断の確かめ (件数, 上限, 型)
    for decision, words in (
        ({"type": "search"}, "k (問いごとに返す件数)"),
        ({"type": "search", "k": 0}, "1 以上の整数"),
        ({"type": "search", "k": 2.5}, "1 以上の整数"),
        ({"type": "search", "k": 5, "max_latency_ms": 0}, "正の数"),
        ({"type": "order", "shortage_cost": 1, "holding_cost": 1,
          "max_order": 1}, "(search) にしてください"),
    ):
        found = check(answer(problem, "action.decision", decision))
        assert any(item["id"] == "action.decision"
                   and item["level"] == "error" and words in item["message"]
                   for item in found), (decision, found)

    # 列の予測に検索の判断は書けない
    column = example_problem("総菜の作る量")
    found = check(answer(column, "action.decision",
                         {"type": "search", "k": 3}))
    assert any("似た写真を探す" in item["message"] for item in found)


def test_search_handoff() -> None:
    """
    似た写真を探すの受け渡しと要約の文を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 一式づくりの値は推論の ready, 判断は件数だけ, 学習の値は空
    stages = handoff(example_problem(EXAMPLE))
    assert stages["inference"]["ready"]["model"] == "resnet18"
    assert stages["inference"]["decision"] == {"type": "search", "k": 5}
    assert stages["inference"]["guard"] is None
    assert stages["rank"] is None
    assert stages["training"]["monitor"] is None
    assert stages["part"]["method"] == "pretrained"
    sentences = describe(example_problem(EXAMPLE))
    assert ("意思決定: 近い順に上位 K 件を返す (返す件数 5, 待ち時間の上限 "
            "(ms) 200)") in sentences


def test_search_errors(tmp_path: Path) -> None:
    """
    似た写真を探すで書き出せない場合が, make_project と同じ文になることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 写真のディレクトリでない入れたデータと, 判断が無い場合
    table = example_problem(EXAMPLE)
    table["data"] = {"source": str(tmp_path / "rows.csv"), "loader": "csv"}
    undecided = answer(example_problem(EXAMPLE), "action.decision", None)
    for problem, words in (
        (table, "画像のディレクトリとして読んでいません"),
        (undecided, "近い順に上位 K 件を返す"),
    ):
        message = project_error(problem)
        assert message is not None and words in message, message
        with pytest.raises(ValueError) as raised:
            make_project(problem, tmp_path / "project")
        assert str(raised.value) == message
    assert not (tmp_path / "project").exists()


def test_search_project(tmp_path: Path) -> None:
    """
    見本から似た写真を探す骨組みを書き出し, ファイル, config, README,
    スクリプトと記録を確かめる関数

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
    assert config["search"] == {"k": 5, "max_latency_ms": 200}
    assert config["outputs"] == {"embedder": "outputs/embedder",
                                 "search": "outputs/search"}
    assert config["bundle"]["name"] == "resnet18_search_5"

    # README とスクリプトの 79 文字以内と, Python として読めること
    script = (project / "bundle.py").read_text(encoding="utf-8")
    compile(script, "bundle.py", "exec")
    for name in ("bundle.py", "configs/default/bundle.yaml", "README.md"):
        text = (project / name).read_text(encoding="utf-8")
        if name != "README.md":
            assert max(len(line) for line in text.splitlines()) <= 79, name
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "python bundle.py" in readme and "版 15" in readme
    assert "待ち時間の上限 200 ms" in readme
    assert "写真を縮めた画素をそのまま余弦の近さで並べたもの" in script

    # 記録はプロジェクトの場所だけ (データセットと定義は無い)
    recorded = record_project(problem, project)["project"]
    assert recorded["path"] == str(project.resolve())
    assert recorded["dataset"] is None and recorded["model_spec"] is None


def test_search_script_stops(tmp_path: Path) -> None:
    """
    書き出した bundle.py が, 写真が無い場合, 古い ML-Inference と config
    の誤りで止まることを確かめる関数

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

    # 版 15 の一式を作れない ML-Inference
    write_photos(project / "outputs" / "photos")
    package = tmp_path / "fake" / "ml_inference"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(OLD_INFERENCE, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(package.parent)}
    done = subprocess.run([sys.executable, script], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 1
    assert "検索の一式 (版 15) を作れません" in done.stdout

    # config の件数と上限の誤り
    path = project / "configs/default/bundle.yaml"
    original = yaml.safe_load(path.read_text(encoding="utf-8"))
    for change, words in (({"k": 0}, "search.k は 1 以上の整数"),
                          ({"max_latency_ms": -1}, "max_latency_ms")):
        config = {**original, "search": {**original["search"], **change}}
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        done = subprocess.run([sys.executable, script], capture_output=True,
                              text=True, cwd=str(tmp_path))
        assert done.returncode == 1
        assert words in done.stdout, done.stdout


@pytest.mark.skipif(
    any(find_spec(name) is None for name in (
        "torch", "PIL", "ml_eda", "ml_modelcontrol", "ml_inference"
    )),
    reason="PyTorch, Pillow, ML-EDA, ML-ModelControl, ML-Inference が要る"
)
def test_search_run(tmp_path: Path) -> None:
    """
    小さな作り物の出どころと種類のある写真で bundle.py を動かし, 索引, 評価,
    検索の一式ができ, 一式の推論が評価と同じ数え方になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 15 の一式を作れる ML-Inference
    import numpy as np
    import ml_inference
    if ml_inference.BUNDLE_FORMAT_VERSION < 15:
        pytest.fail("ML-Inference の版 15 の一式づくりが要る")

    # 作り物の出どころで書き出して動かす
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "fake_source.py").write_text(FAKE_SOURCE, encoding="utf-8")
    project = tmp_path / "project"
    tiny_project(project)
    photos = write_photos(project / "outputs" / "photos")
    done = run_bundle(project, tmp_path, fake)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "読めない写真 1 枚を除きます" in done.stdout
    assert "この hash を config の ready.sha256" in done.stdout

    # 索引は種類ごとに 4 枚 (1 枚ずつ問い), 種類と hash 付き
    artifacts = project / "outputs" / "search"
    index = json.loads((artifacts / "search_index.json").read_text())
    assert index["k"] == 3 and index["max_latency_ms"] == 200
    assert index["ready"]["library"] == "test"
    assert index["root"] == str(photos.resolve())
    items = index["items"]
    assert len(items) == len(COLORS) * (PER_COLOR - 1)
    assert all(item["path"].split("/")[0] == item["label"] for item in items)
    assert all(len(item["sha256"]) == 64 for item in items)
    assert np.load(artifacts / "search_embeddings.npy").shape == (
        len(items), 3
    )

    # 評価の表 (埋め込みは色を見分け, 比べる相手との差)
    evaluation = json.loads(
        (artifacts / "evaluations" / "search_valid.json").read_text()
    )
    row = evaluation["rows"][0]
    assert evaluation["baseline"] == (
        "写真を縮めた画素をそのまま余弦の近さで並べたもの"
    )
    assert evaluation["better"]["recall_at_k_diff"] == "higher"
    assert row["k"] == 3 and row["queries"] == len(COLORS)
    assert row["ndcg_at_k"] == pytest.approx(1.0)
    assert row["recall_at_k_diff"] == pytest.approx(
        row["recall_at_k"] - row["recall_at_k_baseline"]
    )

    # 一式は版 15 で, 一式の推論の上位 K 件から ML-Found の関数で数えた
    # 値が評価と同じ
    bundle = project / "outputs" / "inference"
    info = yaml.safe_load((bundle / "inference.yaml").read_text())
    assert info["format_version"] == 15
    assert info["decision"] == {"type": "search", "k": 3}
    assert info["search"]["evaluation"]["recall_at_k"] == pytest.approx(
        row["recall_at_k"]
    )
    indexed = {item["path"] for item in items}
    queries = sorted(
        path.relative_to(photos).as_posix() for path in photos.rglob("*.png")
        if path.relative_to(photos).as_posix() not in indexed
        and path.name != "broken.png"
    )
    script = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(fake)!r})\n"
        "import fake_source\n"
        "from ml_inference import predict\n"
        f"root = {str(photos)!r}\n"
        f"found = {{q: predict({str(bundle)!r}, root + '/' + q)['label']"
        f".dropna().tolist() for q in {queries!r}}}\n"
        "print(json.dumps(found))\n"
    )
    done = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert all(len(labels) == 3 for labels in found.values())
    try:
        from ml_found import ndcg_at_k, recall_at_k
    except ImportError:
        return
    top = [[f"{n}:{label}" for n, label in enumerate(found[q])]
           for q in queries]
    relevant = [[f"{n}:{label}" for n, label in enumerate(found[q])
                 if label == q.split("/")[0]]
                + [f"x{n}" for n in range(
                    PER_COLOR - 1 - sum(label == q.split("/")[0]
                                        for label in found[q]))]
                for q in queries]
    assert recall_at_k(top, relevant, 3) == pytest.approx(row["recall_at_k"])
    assert ndcg_at_k(top, relevant, 3) == pytest.approx(row["ndcg_at_k"])
