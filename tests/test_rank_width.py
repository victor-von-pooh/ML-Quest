import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

from ml_quest import (
    example_problem, make_project, rank_encoding, record_project
)
from ml_quest.rank_width import (
    RANK_TEST, RANK_VALID, make_index, read_reactions, split_by_time
)

from test_rank_skeleton import ITEMS, USERS, attached, rank_problem

# 骨組みの train.py から写しを取り出す関数の名前
COPIED_FUNCTIONS = ("read_reactions", "split_by_time", "make_index")

# 写しを比べる反応の表の列の名前
COLUMNS = {"user": "member", "item": "item_id", "time": "ts",
           "reaction": "qty"}


def script_copies(root: Path) -> dict:
    """
    書き出した並べるの train.py から, 反応の表をそろえて分け, 番号を決める
    関数の写しを取り出して読む関数 (PyTorch などの読み込みを通さない)

    Parameters
    ----------
    root: Path
        骨組みを書き出す場所

    Returns
    ----------
    names: dict
        関数の名前から関数 (load_data は差し替えられる)
    """
    # 写しの関数だけを集めて, pandas だけで動かす
    make_project(rank_problem(), root)
    tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef)
             and node.name in COPIED_FUNCTIONS]
    assert len(nodes) == len(COPIED_FUNCTIONS)
    module = ast.Module(body=nodes, type_ignores=[])
    names: dict = {"pd": pd, "Path": Path, "Any": Any, "LOADER": None,
                   "OPTIONS": {}}
    exec(compile(module, "train.py", "exec"), names)

    return names


def copy_reactions(copies: dict, frame: pd.DataFrame, columns: dict) -> Any:
    """
    写しの read_reactions を, 決まった表を読む load_data で呼ぶ関数

    Parameters
    ----------
    copies: dict
        script_copies の写し
    frame: pd.DataFrame
        読んだことにする反応の表
    columns: dict
        user, item, time, reaction の列の名前

    Returns
    ----------
    result: Any
        写しの read_reactions の結果
    """
    # config の rank の列の名前の代わり
    copies["load_data"] = lambda path, loader, **options: SimpleNamespace(
        frame=frame
    )
    cfg = SimpleNamespace(rank=SimpleNamespace(**columns))
    result = copies["read_reactions"](Path("reactions.csv"), cfg)

    return result


def test_rank_matches_skeleton(tmp_path: Path) -> None:
    """
    反応の表をそろえて分け, 番号を決める関数が, 並べるの骨組みの train.py
    の写しと同じ結果と誤りになり, 分割の割合の既定が config と同じことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 分割の割合の既定は config と同じ
    copies = script_copies(tmp_path / "project")
    config = yaml.safe_load(
        (tmp_path / "project" / "configs" / "default" / "train.yaml")
        .read_text(encoding="utf-8")
    )
    assert (config["data"]["valid"], config["data"]["test"]) == (
        RANK_VALID, RANK_TEST
    )

    # 欠けた行, 反応の強さが 0 以下か数でない行, 日時の時刻, 同じ時刻
    frames = [
        (pd.DataFrame({"member": ["a", "b", None, "a", "c", "b"],
                       "item_id": ["x", "y", "x", "z", "x", "y"],
                       "ts": [3, 1, 2, 2, 5, 4],
                       "qty": [1, 0, 1, "n", 2, 1]}), COLUMNS),
        (pd.DataFrame({"member": list("abcab"), "item_id": list("xyzzx"),
                       "ts": ["2026-01-03", "2026-01-01", "bad",
                              "2026-01-02", "2026-01-02"]}),
         {**COLUMNS, "reaction": None}),
    ]
    for frame, columns in frames:
        ours = read_reactions(frame, columns)
        theirs = copy_reactions(copies, frame, columns)
        pd.testing.assert_frame_equal(ours[0], theirs[0])
        assert ours[1] == theirs[1]

    # 列が無い表と, 時刻を読めない表は同じ文で止まる
    stops = [pd.DataFrame({"member": ["a"], "ts": [1]}),
             pd.DataFrame({"member": ["a"], "item_id": ["x"],
                           "ts": ["morning"], "qty": [1]})]
    for frame in stops:
        with pytest.raises(ValueError) as ours:
            read_reactions(frame, COLUMNS)
        with pytest.raises(ValueError) as theirs:
            copy_reactions(copies, frame, COLUMNS)
        assert str(ours.value) == str(theirs.value)

    # 時刻の順の分割と番号 (テスト用の割合 0 も), 割合の誤りと空の分割
    reactions = pd.DataFrame({"user": list("abacbdcaeb"),
                              "item": list("xyzxwyzvxy"),
                              "time": [0, 1, 2, 3, 4, 5, 6, 6, 8, 9]})
    for valid, test in ((0.2, 0.2), (0.3, 0.0), (0.1, 0.1)):
        cfg = SimpleNamespace(data=SimpleNamespace(valid=valid, test=test))
        ours = split_by_time(reactions, valid, test)
        theirs = copies["split_by_time"](reactions, cfg)
        assert list(ours) == list(theirs)
        for name in ours:
            pd.testing.assert_frame_equal(ours[name], theirs[name])
        assert make_index(ours["train"]) == copies["make_index"](
            theirs["train"]
        )
    same = reactions.assign(time=0)
    for frame, valid, test in ((reactions, 0.6, 0.5), (same, 0.1, 0.1)):
        cfg = SimpleNamespace(data=SimpleNamespace(valid=valid, test=test))
        with pytest.raises(ValueError) as ours:
            split_by_time(frame, valid, test)
        with pytest.raises(ValueError) as theirs:
            copies["split_by_time"](frame, cfg)
        assert str(ours.value) == str(theirs.value)


def test_rank_encoding(tmp_path: Path) -> None:
    """
    並べるの問題の設定と反応の表から, 学習用の利用者と品物の数を求める
    ことを確かめる関数 (ML-EDA が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 並べるでなければ求めない (反応の表も読まない)
    assert rank_encoding(example_problem("総菜の作る量"),
                         tmp_path / "none.csv") is None

    # 書き出したプロジェクトに写した反応の表 (全ての利用者と品物が学習用
    # にいる)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    make_project(problem, project)
    place = record_project(problem, project)["project"]["dataset"]
    encoding = rank_encoding(problem, place)
    assert (encoding["users"], encoding["items"]) == (USERS, ITEMS)
    assert set(encoding["rows"]) == {"train", "valid", "test"}
    assert any("反応の強さ (qty)" in note for note in encoding["notes"])

    # 割合を変えると分割の行の数が変わる (テスト用の割合 0 なら test 無し)
    other = rank_encoding(problem, place, valid=0.3, test=0.0)
    assert set(other["rows"]) == {"train", "valid"}
    assert other["rows"]["train"] < encoding["rows"]["train"]

    # 反応の表の無い場所と, 割合の誤り
    with pytest.raises(FileNotFoundError):
        rank_encoding(problem, tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="data.valid"):
        rank_encoding(problem, place, valid=0.0)
