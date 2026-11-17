import os
from pathlib import Path

import pandas as pd
import pytest

from ml_quest import (
    answer, apply_recipe, attach_data, check, example_problem, get_value,
    handoff, inspect_data, load_problem, make_project, prune_copies,
    recipe_from, resolve_choice
)


def daily_csv(tmp_path: Path) -> Path:
    """
    2 店 × 2 品の 40 日の売れた数と, 売れた数から作った客数の表を書く関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    path: Path
        csv の場所
    """
    # 売れた数は曜日で決まる (7 日前と同じなら損は 0)
    rows = []
    for day in pd.date_range("2025-01-01", periods=40):
        for store in ("east", "west"):
            for item in ("a", "b"):
                sold = 10 + day.day_of_week
                rows.append({
                    "date": day.strftime("%Y-%m-%d"), "store": store,
                    "item": item, "sales": sold,
                    "temperature_forecast": 15.0, "promo": 0,
                    "customers": sold * 3,
                })
    path = tmp_path / "source" / "sozai.csv"
    path.parent.mkdir()
    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def test_copy_and_project(tmp_path: Path) -> None:
    """
    データを写して入れると写した場所を使い, 骨組みのプロジェクトにも
    写すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 写し先に写り, 元の場所も残る
    pytest.importorskip("ml_eda")
    path = daily_csv(tmp_path)
    problem = attach_data(example_problem("総菜の作る量"), path,
                          copy_to=tmp_path / "copy")
    copied = Path(get_value(problem, "data.source"))
    assert copied.name == "sozai.csv"
    assert copied.parent.parent == (tmp_path / "copy").resolve()
    assert get_value(problem, "data.origin") == str(path.resolve())
    assert copied.read_bytes() == path.read_bytes()

    # 骨組みのプロジェクトの data にも写す
    written = make_project(problem, tmp_path / "project")
    assert "data/sozai.csv" in written
    assert (tmp_path / "project" / "data" / "sozai.csv").is_file()
    readme = (tmp_path / "project" / "README.md").read_text("utf-8")
    assert "data/sozai.csv" in readme


def test_current_practice(tmp_path: Path) -> None:
    """
    今のやり方をデータで書くと, 損が求まり指標の今の値の候補になることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 曜日で決まる売れた数なので, 7 日前と同じなら損は 0
    pytest.importorskip("ml_eda")
    problem = attach_data(example_problem("総菜の作る量"),
                          daily_csv(tmp_path))
    weekly = answer(problem, "action.current", {"type": "lag",
                                                "periods": 7})
    found = inspect_data(weekly)
    assert found["baseline"]["total"] == 0
    assert found["baseline"]["rows"] == 4 * 33

    # 1 日前と同じなら, 曜日の差の分だけ損 (足りない 200, 余る 150)
    daily = answer(problem, "action.current", {"type": "lag", "periods": 1})
    found = inspect_data(daily)
    assert found["baseline"]["total"] > 0
    suggested = {s["id"]: s["values"] for s in found["suggestions"]}
    assert suggested["issue.kpi.current"] == [
        f"{found['baseline']['per_30_days']:.0f}"
    ]
    # 平均は ML-EDA と同じく, 窓に値が 1 つでもあれば求める
    mean = answer(problem, "action.current", {"type": "mean", "periods": 7})
    assert inspect_data(mean)["baseline"]["rows"] == 4 * 39

    # 答えの形の誤りと, 振り分けに k 期前は使えない
    with pytest.raises(ValueError, match="1 以上"):
        answer(problem, "action.current", {"type": "lag", "periods": 0})
    with pytest.raises(ValueError, match="列にありません"):
        answer(problem, "action.current", {"type": "column",
                                           "column": "none"})
    churn = answer(example_problem("解約の引き止め"), "action.current",
                   {"type": "lag", "periods": 7})
    assert any(f["id"] == "action.current" and f["level"] == "error"
               for f in check(churn))


def test_triage_current_column(tmp_path: Path) -> None:
    """
    振り分けの今のやり方を行動したかの列で書くと, 見逃しと空振りの損が
    求まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 4 件: 当たりを 1 件見逃し, 外れを 1 件回す
    pytest.importorskip("ml_eda")
    path = tmp_path / "reviews.csv"
    pd.DataFrame({
        "text_length": [10, 20, 30, 40] * 30,
        "label": ["悪い", "悪い", "良い", "良い"] * 30,
        "checked": ["yes", "no", "yes", "no"] * 30,
    }).to_csv(path, index=False)
    problem = example_problem("解約の引き止め")
    problem = answer(problem, "information.columns",
                     ["text_length", "label", "checked"])
    problem = answer(problem, "ml.target", "label")
    problem = answer(problem, "ml.positive", "悪い")
    problem = answer(problem, "information.known", ["text_length"])
    problem = answer(problem, "information.forbidden", None)
    problem = answer(problem, "action.decision", {
        "type": "triage", "miss_cost": 10, "false_alarm_cost": 1,
    })
    problem = attach_data(problem, path)
    problem = answer(problem, "action.current", {"type": "column",
                                                 "column": "checked"})
    found = inspect_data(problem)
    assert found["baseline"]["total"] == 30 * (10 + 1)


def test_recipe_round_trip(tmp_path: Path) -> None:
    """
    問題の設定からのレシピの雛形が ML-EDA で書き出せ, データの分析で変えた
    レシピを書き戻せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 雛形: 除く列, 系列ごとの 2 期前, 時刻の順の分割
    eda = pytest.importorskip("ml_eda")
    path = daily_csv(tmp_path)
    problem = attach_data(example_problem("総菜の作る量"), path)
    start = recipe_from(problem)
    recipe = start["recipe"]
    assert recipe["target"] == "sales" and recipe["task"] == "forecasting"
    assert recipe["steps"][0] == {"op": "drop_columns",
                                  "columns": ["customers"]}
    assert recipe["steps"][1]["lags"] == [2]
    assert recipe["steps"][1]["by"] == ["store", "item"]
    assert recipe["split"]["method"] == "time"
    assert start["load"]["options"]["time_column"] == "date"

    # 同じ読み方とレシピで ML-EDA が書き出せる
    data = eda.load_data(start["load"]["path"], loader=start["load"]["loader"],
                         **start["load"]["options"])
    recipe["steps"] += [{"op": "drop_missing"},
                        {"op": "one_hot", "columns": ["store", "item"]}]
    eda.export_dataset(data, recipe, tmp_path / "dataset")
    assert "sales_lag2" in eda.load_dataset(tmp_path / "dataset").features

    # 客数を使うレシピは, 自動で移さず選ぶ問いにする
    changed = dict(recipe, steps=recipe["steps"][1:])
    result = apply_recipe(problem, changed)
    assert result["changes"] == [] and result["conflicts"] == []
    assert [c["column"] for c in result["choices"]] == ["customers"]
    assert get_value(result["problem"], "information.unknown") == [
        "customers"
    ]
    moved = resolve_choice(result["problem"], "customers", "known")
    assert "customers" in get_value(moved, "information.known")
    dropped = resolve_choice(result["problem"], "customers", "drop")
    assert recipe_from(dropped)["recipe"]["steps"][0]["columns"] == [
        "customers"
    ]

    # 書き戻したレシピは次の雛形になる (往復で同じ)
    assert recipe_from(result["problem"])["recipe"] == changed

    # 目的の列とタスクは書き戻し, 使ってはいけない列と漏れは食い違い
    other = dict(recipe, target="customers", steps=[
        {"op": "lag", "columns": ["customers"], "lags": [1],
         "by": ["store", "item"], "time": "date", "freq": "86400s"},
    ], split={"method": "random"})
    forbidden = answer(problem, "information.forbidden", ["promo"])
    result = apply_recipe(forbidden, other)
    assert get_value(result["problem"], "ml.target") == "customers"
    assert "customers" not in (
        get_value(result["problem"], "information.unknown") or []
    )
    levels = {(f["id"], f["level"]) for f in result["conflicts"]}
    assert ("information.forbidden", "error") in levels
    assert ("ml.available_lag", "warning") in levels
    assert ("ml.task", "warning") in levels


def test_make_project_data_checks(tmp_path: Path) -> None:
    """
    骨組みを書き出すとき, データの写しも上書きの確かめに入り, 写した問題の
    設定がプロジェクトのデータを指すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 既にあるデータは上書きしない. data がファイルなら何も書かない
    pytest.importorskip("ml_eda")
    problem = attach_data(example_problem("総菜の作る量"),
                          daily_csv(tmp_path))
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    (project / "data" / "sozai.csv").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="既にあります"):
        make_project(problem, project)
    assert (project / "data" / "sozai.csv").read_text("utf-8") == "mine"
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "data").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="邪魔"):
        make_project(problem, blocked)
    assert not (blocked / "README.md").exists()

    # 写した問題の設定はプロジェクトのデータを指し, 元の場所を残す
    make_project(problem, tmp_path / "fresh")
    copied = load_problem(tmp_path / "fresh")
    assert copied["data"]["source"] == str(
        (tmp_path / "fresh" / "data" / "sozai.csv").resolve()
    )
    assert copied["data"]["origin"] == get_value(problem, "data.source")


def test_copy_rules(tmp_path: Path) -> None:
    """
    写し先が相対パスか元のディレクトリの中なら誤りにし, 読めなければ写しを
    残さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 相対パスの写し先と, 読めないファイルの写しの後始末
    pytest.importorskip("ml_eda")
    path = daily_csv(tmp_path)
    problem = example_problem("総菜の作る量")
    with pytest.raises(ValueError, match="絶対パス"):
        attach_data(problem, path, copy_to="relative")
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        attach_data(problem, broken, copy_to=tmp_path / "copies")
    assert not list((tmp_path / "copies").rglob("broken.xlsx"))
    with pytest.raises(ValueError, match="中"):
        attach_data(problem, path.parent, copy_to=path.parent / "inside")


def test_current_practice_calendar(tmp_path: Path) -> None:
    """
    休みの日で日付が飛んでも 7 期前は暦の 7 日前で数え, 対象を選ぶ行動は
    得を求めないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 日曜が休み. 売れた数は曜日で決まるので, 7 日前と同じなら損は 0
    pytest.importorskip("ml_eda")
    rows = [
        {"date": day.strftime("%Y-%m-%d"), "store": "east", "item": "a",
         "sales": 10 + day.day_of_week, "temperature_forecast": 15.0,
         "promo": 0, "customers": 0}
        for day in pd.date_range("2025-01-01", periods=42)
        if day.day_of_week != 6
    ]
    path = tmp_path / "holiday.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    problem = attach_data(example_problem("総菜の作る量"), path)
    assert get_value(problem, "data.interval_seconds") == 86400
    weekly = answer(problem, "action.current", {"type": "lag", "periods": 7})
    found = inspect_data(weekly)["baseline"]
    assert found["total"] == 0 and found["rows"] > 0

    # 選ぶ行動は, 行動した行の結果が行動の後なので求めない
    churn = example_problem("解約の引き止め")
    churn["data"] = problem["data"]
    churn = answer(churn, "information.columns", list(rows[0]))
    churn = answer(churn, "action.current", {"type": "column",
                                             "column": "promo"})
    assert "reason" in inspect_data(churn)["baseline"]


def test_copy_again(tmp_path: Path) -> None:
    """
    入れ直しで, 今の写しを壊さず, 古い写しを片付け, 中身の違う同じ名前の
    ファイルを写し直すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # .. で終わる場所は写し先の外に出ない
    pytest.importorskip("ml_eda")
    (tmp_path / "src" / "d1").mkdir(parents=True)
    first = daily_csv(tmp_path / "src" / "d1")
    store = tmp_path / "store"
    store.mkdir()
    (store / "problem.yaml").write_text("keep", encoding="utf-8")
    copies = store / "data"
    problem = example_problem("総菜の作る量")
    try:
        parent = attach_data(problem, tmp_path / "src" / "d1" / "..",
                             copy_to=copies)
        assert Path(get_value(parent, "data.source")).is_relative_to(
            copies.resolve()
        )
    except ValueError:
        pass
    assert (store / "problem.yaml").read_text("utf-8") == "keep"

    # 同じ名前の読めないファイルを入れ直しても, 今の写しは残る
    problem = attach_data(problem, first, copy_to=copies)
    current = Path(get_value(problem, "data.source"))
    broken = tmp_path / "d3" / "sozai.csv"
    broken.parent.mkdir()
    broken.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError):
        attach_data(problem, broken, copy_to=copies)
    assert current.read_bytes() == first.read_bytes()

    # 変わっていない同じファイルは写し直さない. 大きさと時刻が同じでも
    # 別の場所のファイルは写し直す
    again = attach_data(problem, first, copy_to=copies)
    assert get_value(again, "data.source") == str(current)
    other = tmp_path / "d2" / "sozai.csv"
    other.parent.mkdir()
    other.write_bytes(first.read_bytes().replace(b"2025", b"2024", 1))
    stat = first.stat()
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    moved = attach_data(problem, other, copy_to=copies)
    assert Path(get_value(moved, "data.source")).read_bytes() == (
        other.read_bytes()
    )

    # 古い写しの片付けと, 写しが消えたら元の場所
    prune_copies(copies, get_value(moved, "data.source"))
    assert [p.name for p in copies.iterdir()] == [
        Path(get_value(moved, "data.source")).parent.name
    ]
    prune_copies(copies, None)
    assert handoff(moved)["data"]["source"]["source"] == str(other)


def test_prune_keeps_other_files(tmp_path: Path) -> None:
    """
    片付けが写しでないファイルを消さず, 写し先の中の別のファイルを入れると
    前の元の場所を引き継がないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 写し先に利用者のファイルがある
    pytest.importorskip("ml_eda")
    path = daily_csv(tmp_path)
    copies = tmp_path / "copies"
    copies.mkdir()
    (copies / "mine.txt").write_text("keep", encoding="utf-8")
    (copies / "own").mkdir()
    problem = attach_data(example_problem("総菜の作る量"), path,
                          copy_to=copies)
    prune_copies(copies, None)
    assert (copies / "mine.txt").is_file() and (copies / "own").is_dir()
    with pytest.raises(ValueError, match="絶対パス"):
        prune_copies(copies, "relative.csv")
    with pytest.raises(ValueError, match="写し先そのもの"):
        attach_data(problem, copies, copy_to=copies)

    # 写し先の中の別のファイルは元の場所を持たない
    other = copies / "other.csv"
    other.write_bytes(path.read_bytes())
    inside = attach_data(problem, other, copy_to=copies)
    assert get_value(inside, "data.source") == str(other.resolve())
    assert get_value(inside, "data.origin") is None
