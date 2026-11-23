import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ml_quest import example_problem, make_project, table_encoding
from ml_quest import table_width
from ml_quest.table_width import (
    ENCODING_SPLITS, encoded_width, missing_columns, table_columns
)

from test_table_skeleton import WIDTH, attached, export, table_problem

# 骨組みの train.py から写しを取り出す関数と定数の名前
COPIED_FUNCTIONS = ("table_columns", "encoded_width", "missing_columns")
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

    # main が分割の表を選ぶ組 (split in (...) の並び)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    names["SPLITS"] = [
        ast.literal_eval(node.comparators[0]) for node in ast.walk(main)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name) and node.left.id == "split"
        and isinstance(node.ops[0], ast.In)
        and isinstance(node.comparators[0], ast.Tuple)
    ]

    return names


def write_dataset(root: Path, frame: pd.DataFrame, recipe: dict) -> Path:
    """
    表を ML-EDA のレシピで表のデータセットに書き出す関数 (ML-EDA が無ければ
    飛ばす)

    Parameters
    ----------
    root: Path
        書き出す場所
    frame: pd.DataFrame
        元の表
    recipe: dict
        レシピの task, target, valid の割合 (他は既定)

    Returns
    ----------
    place: Path
        書き出したデータセット
    """
    # 無作為の分割で, 手順は目的の列の行の欠損を落とすだけ
    eda = pytest.importorskip("ml_eda")
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "table.csv", index=False)
    data = eda.load_data(root / "table.csv", loader="csv")
    full = {"format_version": 1, "task": recipe.get("task", "none"),
            "steps": [{"op": "drop_missing"}] if "target" in recipe else [],
            "split": {"method": "random", "valid": recipe.get("valid", 0.2),
                      "test": 0.2, "seed": 0}}
    if "target" in recipe:
        full["target"] = recipe["target"]
    place = root / "dataset"
    eda.export_dataset(data, full, place)

    return place


def fake_eda(info: dict, frames: dict) -> SimpleNamespace:
    """
    決まったデータセットを返す ML-EDA の代わりを作る関数

    Parameters
    ----------
    info: dict
        dataset.yaml の中身
    frames: dict
        分割の名前から表

    Returns
    ----------
    eda: SimpleNamespace
        load_dataset だけを持つ代わり
    """
    # どの場所でも同じデータセット
    dataset = SimpleNamespace(
        info=info, frames=frames, target=info.get("target"),
        features=[item["name"] for item in info.get("features", [])]
    )
    eda = SimpleNamespace(load_dataset=lambda path: dataset)

    return eda


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
    assert copies["SPLITS"] == [ENCODING_SPLITS]

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

    # 欠損のある列を分割をまたいで集める
    for splits in [
        {"train": pd.DataFrame({"a": [1.0, None], "b": ["x", "y"]}),
         "valid": pd.DataFrame({"a": [1.0, 2.0], "b": ["x", None]})},
        {"train": frames[0]},
    ]:
        assert missing_columns(splits) == copies["missing_columns"](splits)


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


def test_table_encoding_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    骨組みが止まるデータセットで止まり, 検証用の分割が無いデータセットでは
    止まらずに幅を返すことを確かめる関数 (ML-EDA が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        ML-EDA の代わりを差す

    Returns
    ----------
    None
    """
    # 数の列 1 つと値の種類 2 の種類の列 (幅 3). 欠損は数の列だけ
    problem = table_problem()
    frame = pd.DataFrame({"a": [1.0, None, 3.0, 4.0] * 10,
                          "b": ["x", "y"] * 20})
    filled = frame.fillna(2.0)

    # 検証用の分割が無くても, 学習用の分割で幅を求める
    place = write_dataset(tmp_path / "novalid", filled, {"valid": 0.0})
    encoding = table_encoding(problem, place)
    assert encoding["width"] == 3

    # 目的の列があるデータセットと, 欠損のある列
    place = write_dataset(tmp_path / "target", frame,
                          {"task": "regression", "target": "a"})
    with pytest.raises(ValueError, match="目的の列"):
        table_encoding(problem, place)
    place = write_dataset(tmp_path / "missing", frame, {})
    with pytest.raises(ValueError, match="欠損のある列があります \\(a\\)"):
        table_encoding(problem, place)

    # 表でないデータセットと, 学習用の分割の無いデータセット (ML-EDA は
    # 書き出さないので代わりで)
    features = {"features": [{"name": "a"}]}
    monkeypatch.setattr(table_width, "import_eda", lambda: fake_eda(
        {"kind": "image", **features}, {"train": filled[["a"]]}
    ))
    with pytest.raises(ValueError, match="表のデータセットではありません"):
        table_encoding(problem, tmp_path)
    monkeypatch.setattr(table_width, "import_eda", lambda: fake_eda(
        {"kind": "table", **features}, {"valid": filled[["a"]]}
    ))
    with pytest.raises(ValueError, match="学習用の分割がありません"):
        table_encoding(problem, tmp_path)
