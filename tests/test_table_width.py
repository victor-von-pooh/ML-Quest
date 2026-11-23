import ast
from pathlib import Path

import pandas as pd
import pytest

from ml_quest import example_problem, make_project, table_encoding
from ml_quest import table_width
from ml_quest.table_width import encoded_width, table_columns

from test_table_skeleton import WIDTH, attached, export, table_problem

# 骨組みの train.py から写しを取り出す関数と定数の名前
COPIED_FUNCTIONS = ("table_columns", "encoded_width")
COPIED_CONSTANTS = ("MAX_CATEGORIES", "RESERVED_COLUMNS",
                    "RESERVED_PREFIXES", "RESERVED_SUFFIX")


def script_copies(root: Path) -> dict:
    """
    書き出した表の生成の train.py から, 列の符号化の関数と定数の写しを
    取り出して読む関数 (PyTorch などの読み込みを通さない)

    Parameters
    ----------
    root: Path
        骨組みを書き出す場所

    Returns
    ----------
    names: dict
        関数と定数の名前から値
    """
    # 写しの関数と定数だけを集めて, pandas だけで動かす
    make_project(table_problem(), root)
    tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef)
            and node.name in COPIED_FUNCTIONS)
        or (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id in COPIED_CONSTANTS
                    for target in node.targets))
    ]
    assert len(nodes) == len(COPIED_FUNCTIONS) + len(COPIED_CONSTANTS)
    module = ast.Module(body=nodes, type_ignores=[])
    names: dict = {"pd": pd}
    exec(compile(module, "train.py", "exec"), names)

    return names


def test_encoding_matches_skeleton(tmp_path: Path) -> None:
    """
    列の符号化の関数と定数が, 表の生成の骨組みの train.py の写しと同じ
    結果と誤りになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 定数は同じ値
    copies = script_copies(tmp_path / "project")
    for name in COPIED_CONSTANTS:
        assert copies[name] == getattr(table_width, name), name

    # 数, 整数, 真偽値, 種類, 値の同じ列と, 上限ちょうどの種類の列
    frames = [
        pd.DataFrame({"same": [5.0, 5.0, 5.0], "count": [1, 2, 3],
                      "city": ["b", "a", "b"], "flag": [True, False, True]}),
        pd.DataFrame({"id": [f"u{i}" for i in range(100)]}),
        pd.DataFrame({"mixed": [1, "a", 2.5], "x": [0.1, 0.2, 0.3]}),
    ]
    for frame in frames:
        columns = table_columns(frame)
        assert columns == copies["table_columns"](frame)
        assert encoded_width(columns) == copies["encoded_width"](columns)
    assert encoded_width(table_columns(frames[0])) == 6
    assert encoded_width(table_columns(frames[1])) == 100

    # 上限を超える種類の列と, 予測の表の列と重なる名前は同じ文で止まる
    stops = [pd.DataFrame({"id": [f"u{i}" for i in range(101)]})]
    stops += [pd.DataFrame({name: [1.0], "ok": [2.0]})
              for name in ("status", "p:a", "q:b", "age (入力)")]
    for frame in stops:
        with pytest.raises(ValueError, match="drop_columns") as ours:
            table_columns(frame)
        with pytest.raises(ValueError) as theirs:
            copies["table_columns"](frame)
        assert str(ours.value) == str(theirs.value)


def test_table_encoding(tmp_path: Path) -> None:
    """
    表の生成の問題の設定とデータセットから, 骨組みと同じ符号化した行の幅を
    求めることを確かめる関数 (ML-EDA が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表の生成でなければ求めない (データセットも読まない)
    assert table_encoding(example_problem("総菜の作る量"),
                          tmp_path / "none") is None

    # 案内どおりに書き出したデータセット (数の列 3, 種類の値 3 と 2)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    make_project(problem, project)
    place = export(problem, project)
    encoding = table_encoding(problem, place)
    assert encoding["width"] == WIDTH
    assert (encoding["numbers"], encoding["categories"]) == (3, 2)
    assert [column["name"] for column in encoding["columns"]] == [
        "age", "income", "visits", "city", "member"
    ]
    assert encoding["columns"][3]["categories"] == ["a", "b", "c"]

    # データセットの無い場所
    with pytest.raises(FileNotFoundError):
        table_encoding(problem, tmp_path / "missing")
