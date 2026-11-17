from pathlib import Path

import pandas as pd
import pytest

from ml_quest import (
    answer, apply_recipe, attach_data, check, example_problem, get_value,
    inspect_data, make_project, recipe_from
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
    copied = tmp_path / "copy" / "sozai.csv"
    assert get_value(problem, "data.source") == str(copied)
    assert get_value(problem, "data.origin") == str(path)
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
    mean = answer(problem, "action.current", {"type": "mean", "periods": 7})
    assert inspect_data(mean)["baseline"]["rows"] == 4 * 33

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

    # 客数を使うと決め直したレシピを書き戻すと, 分かる列に移る
    changed = dict(recipe, steps=recipe["steps"][1:])
    updated, changes = apply_recipe(problem, changed)
    assert "customers" in get_value(updated, "information.known")
    assert not get_value(updated, "information.unknown")
    assert any("customers" in sentence for sentence in changes)
    same, nothing = apply_recipe(problem, recipe)
    assert nothing == [] and same == problem
