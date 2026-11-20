import copy
from pathlib import Path

import pytest
import yaml

from ml_quest import (
    answer, apply_recipe, attach_data, handoff, make_project, recipe_from,
    record_project
)
from test_items_skeleton import KIND_STEPS, kind_problem

# 正解の種類ごとの書き出しの関数
FUNCTIONS = {"mask": "export_image_masks",
             "point_cloud": "export_capture_clouds"}


def write_raw(root: Path, kind: str) -> dict:
    """
    ML-EDA の合成の見本から, 書き出す前の入力と正解のディレクトリを書く
    関数

    Parameters
    ----------
    root: Path
        書く場所
    kind: str
        正解の種類 (mask か point_cloud)

    Returns
    ----------
    dirs: dict
        root, inputs, targets (どれも Path)
    """
    # 画像と領域か, 撮影と点群
    pytest.importorskip("PIL")
    synthetic = pytest.importorskip("ml_eda.synthetic")
    if kind == "mask":
        written = synthetic.write_image_masks(root, count=8, size=(16, 16))
        dirs = {"root": root, "inputs": Path(written["images"]),
                "targets": Path(written["masks"])}
        return dirs
    written = synthetic.write_shape_captures(root, count=6, views=3,
                                             size=(16, 16), points=64)
    dirs = {"root": root, "inputs": Path(written["captures"]),
            "targets": Path(written["clouds"])}

    return dirs


@pytest.mark.parametrize("kind", ["mask", "point_cloud"])
def test_draft_without_data(kind: str) -> None:
    """
    データを入れていないファイルの組の受け持ちで, 書き出しの設定の下書きが
    出ることを確かめる関数

    Parameters
    ----------
    kind: str
        正解の種類

    Returns
    ----------
    None
    """
    # レシピは無く, 書き出しは正解の種類から. ディレクトリは分からない
    start = recipe_from(kind_problem(kind))
    assert start["load"] is None and start["recipe"] is None
    export = start["export"]
    assert export["function"] == FUNCTIONS[kind]
    assert (export["inputs"], export["targets"]) == (None, None)
    assert export["split"] == {"method": "random", "valid": 0.15,
                               "test": 0.15, "seed": 0}
    assert export["groups"] is None and export["options"] == {}
    assert any("データが入っていない" in note for note in start["notes"])

    # 正解の出どころ: 今の処理 (領域の見本) は名前と設定, 他は不明
    origin = {"mask": {"type": "process", "name": "今の塗り分け",
                       "settings": {"note": "細かさ 1 画素"}},
              "point_cloud": {"type": "unknown", "name": None,
                              "settings": {}}}[kind]
    assert export["origin"] == origin

    # 受け渡しの分割とグループ
    data = handoff(kind_problem(kind))["data"]
    assert (data["split"], data["groups"]) == ("random", None)


@pytest.mark.parametrize("kind", ["mask", "point_cloud"])
def test_draft_dirs_from_data(tmp_path: Path, kind: str) -> None:
    """
    入れたデータの場所から, 入力と正解のディレクトリが埋まることを確かめる
    関数

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
    # 2 つのディレクトリを持つ場所と, 入力のディレクトリ (隣に正解)
    dirs = write_raw(tmp_path / "raw", kind)
    for place in (dirs["root"], dirs["inputs"]):
        problem = attach_data(kind_problem(kind), str(place))
        start = recipe_from(problem)
        assert start["load"]["path"] == str(place)
        assert start["export"]["inputs"] == str(dirs["inputs"])
        assert start["export"]["targets"] == str(dirs["targets"])
        assert start["notes"] == []

    # 正解が隣に無い入力のディレクトリは, 入力だけ埋めて知らせる
    alone = tmp_path / "alone"
    dirs["inputs"].rename(alone)
    problem = attach_data(kind_problem(kind), str(alone))
    start = recipe_from(problem)
    assert (start["export"]["inputs"], start["export"]["targets"]) == (
        str(alone), None
    )
    assert any("正解のディレクトリ" in note for note in start["notes"])


def test_apply_export_round_trip() -> None:
    """
    書き戻した書き出しの設定が, 下書きと受け渡しに使われることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # グループごとの分割とグループを書き戻す
    problem = kind_problem("mask")
    export = recipe_from(problem)["export"]
    export.update(inputs="/data/images", targets="/data/masks",
                  groups={"a": "site_1", "b": "site_2"},
                  options={"classes": {0: "background", 1: "circle"}})
    export["split"]["method"] = "group"
    result = apply_recipe(problem, export)
    assert result["conflicts"] == [] and result["choices"] == []
    assert len(result["changes"]) == 1
    written = result["problem"]
    assert "export" not in problem.get("data", {})
    assert written["information"] == problem["information"]

    # 往復で同じ (知らせ無し). 受け渡しは書き戻した分割とグループ
    again = recipe_from(written)
    assert again["notes"] == []
    assert again["export"] == written["data"]["export"]
    assert again["export"]["groups"] == {"a": "site_1", "b": "site_2"}
    data = handoff(written)["data"]
    assert data["split"] == "group"
    assert data["groups"] == {"a": "site_1", "b": "site_2"}

    # 分割を選ばなければ ML-EDA の既定 (グループがあればグループごと)
    export["split"].pop("method")
    data = handoff(apply_recipe(problem, export)["problem"])["data"]
    assert data["split"] == "group"

    # 正解の種類を変えたら, 書き戻した設定は使わず下書きに
    cloud = answer(written, "ml.parts.0.target.kind", "point_cloud")
    cloud = answer(cloud, "ml.parts.0.output", "shape")
    cloud = answer(cloud, "ml.parts.0.evaluation.metric", "chamfer")
    start = recipe_from(cloud)
    assert start["export"]["function"] == "export_capture_clouds"
    assert start["export"]["groups"] is None
    assert any("合わない" in note for note in start["notes"])


def test_apply_export_item_steps() -> None:
    """
    書き出しの設定の中身の手順を書き戻し, 下書きに返し, 形の誤りを止める
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 項目ごとの手順を書き戻すと, 往復で同じ (変えた点に項目の名前)
    for kind in ("mask", "point_cloud"):
        problem = kind_problem(kind)
        export = recipe_from(problem)["export"]
        export["options"]["item_steps"] = copy.deepcopy(KIND_STEPS[kind])
        result = apply_recipe(problem, export)
        assert result["conflicts"] == []
        assert all(name in result["changes"][0] for name in KIND_STEPS[kind])
        written = result["problem"]
        again = recipe_from(written)
        assert again["export"]["options"]["item_steps"] == KIND_STEPS[kind]
        assert again["export"] == written["data"]["export"]

    # 空の辞書と空の並びは落とす
    problem = kind_problem("mask")
    base = recipe_from(problem)["export"]
    for steps in ({}, {"image": []}, None):
        export = copy.deepcopy(base)
        export["options"]["item_steps"] = steps
        written = apply_recipe(problem, export)["problem"]
        assert "item_steps" not in written["data"]["export"]["options"]

    # 形の誤り (領域と知らない項目, 並びでない, op の無い手順, 辞書でない)
    for steps, words in (({"mask": [{"op": "resize_mask"}]}, "image"),
                         ({"cloud": []}, "cloud"),
                         ({"image": {"op": "resize"}}, "並び"),
                         ({"image": [{"width": 8}]}, "op"),
                         ({"image": ["resize"]}, "op"),
                         (["image"], "辞書")):
        export = copy.deepcopy(base)
        export["options"]["item_steps"] = steps
        with pytest.raises(ValueError, match=words):
            apply_recipe(problem, export)


def test_apply_export_conflicts() -> None:
    """
    書き出しの設定の食い違いと形の誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正解の種類に合わない書き出し, 検証用の分割が無い, グループが無い
    problem = kind_problem("mask")
    base = recipe_from(problem)["export"]
    cases = [
        ({"function": "export_capture_clouds"}, ("error", "で書き出します")),
        ({"split": {"method": "none"}}, ("warning", "検証用")),
        ({"split": {"valid": 0, "test": 0}}, ("warning", "検証用")),
        ({"split": {"method": "group"}}, ("error", "グループ")),
    ]
    for change, (level, words) in cases:
        export = {**copy.deepcopy(base), **change}
        conflicts = apply_recipe(problem, export)["conflicts"]
        assert [item["level"] for item in conflicts] == [level], conflicts
        assert words in conflicts[0]["message"]

    # 形の誤りは止める
    for change, words in (({"function": "export_dataset"}, "function"),
                          ({"inputs": 1}, "inputs"),
                          ({"groups": ["a"]}, "groups"),
                          ({"options": {"cloud_units": "meter"}},
                           "cloud_units"),
                          ({"name": "x"}, "知らない鍵")):
        with pytest.raises(ValueError, match=words):
            apply_recipe(problem, {**copy.deepcopy(base), **change})
    with pytest.raises(ValueError, match="辞書"):
        apply_recipe(problem, ["export_image_masks"])

    # 列の予測でもファイルの組でもない受け持ちは止める
    text = answer(problem, "ml.parts.0.target.kind", "text")
    text = answer(text, "ml.parts.0.output", "text")
    text = answer(text, "ml.parts.0.evaluation.metric", "cer")
    with pytest.raises(ValueError, match="画像の領域, 点群"):
        recipe_from(text)


@pytest.mark.parametrize("kind", ["mask", "point_cloud"])
@pytest.mark.parametrize("steps", [False, True])
def test_export_into_project(tmp_path: Path, kind: str, steps: bool) -> None:
    """
    下書きで書き出し, 書き戻した問題の設定から骨組みがファイルの組の
    データセットを読むことを通しで確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    kind: str
        正解の種類
    steps: bool
        中身の手順を足して版 3 で書き出すか

    Returns
    ----------
    None
    """
    # 元のディレクトリを入れ, 骨組みを書き出してプロジェクトを記録
    eda = pytest.importorskip("ml_eda")
    dirs = write_raw(tmp_path / "raw", kind)
    problem = attach_data(kind_problem(kind), str(dirs["root"]))
    project = tmp_path / "project"
    make_project(problem, project)
    problem = record_project(problem, project)
    target = Path(handoff(problem)["data"]["project_dataset"])

    # 下書きの引数のまま, プロジェクトのデータセットの場所に書き出す
    # (画面と同じ使い方)
    export = recipe_from(problem)["export"]
    export["split"].update(valid=0.25, test=0.25)
    if steps:
        export["options"]["item_steps"] = copy.deepcopy(KIND_STEPS[kind])
    function = getattr(eda, export["function"])
    function(export["inputs"], export["targets"], target,
             split=export["split"], groups=export["groups"],
             origin=export["origin"], **export["options"])

    # 書き戻し, 書き出し直した骨組みはそのデータセットを読む
    result = apply_recipe(problem, export)
    assert result["conflicts"] == []
    written = result["problem"]
    make_project(written, project, overwrite=True)
    config = yaml.safe_load((project / "configs/default/train.yaml")
                            .read_text(encoding="utf-8"))
    assert (project / config["data"]["dataset"]).resolve() == (
        target.resolve()
    )
    items = eda.load_items(project / config["data"]["dataset"])
    assert items.info["target"]["kind"] == kind
    assert items.info["target"]["origin"]["type"] == export["origin"]["type"]
    assert items.info["format_version"] == (3 if steps else 2)
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "書き戻した設定" in readme and str(dirs["inputs"]) in readme
    assert ("item_steps=" in readme) == steps

    # 書き戻した設定から同じ手順で作り直せる (問題の設定に手順が残る)
    again = recipe_from(written)["export"]
    assert again["options"].get("item_steps") == (
        KIND_STEPS[kind] if steps else None
    )
