import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_quest import (
    answer, attach_data, check, get_value, handoff, inspect_data, new_problem
)
from ml_quest.examples import example_problem


def shop_csv(tmp_path: Path) -> Path:
    """
    2 店の日々の売上と, 売上から作った客数と, 天気と ID の列の表を書く
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    path: Path
        csv の場所
    """
    # 120 日 × 2 店. 客数は売上のほぼ 3 倍 (予測の時点で分からない列)
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-01-01", periods=120).strftime("%Y-%m-%d")
    sales = rng.uniform(50, 150, 240).round()
    weather = rng.choice(["晴れ", "雨", "曇り"], 120)
    frame = pd.DataFrame({
        "row_id": np.arange(240), "date": np.r_[days, days],
        "shop": ["a"] * 120 + ["b"] * 120, "sales": sales,
        "customers": (sales * 3 + rng.normal(0, 1, 240)).round(),
        "weather": np.r_[weather, weather],
        "promo": rng.integers(0, 2, 240),
    })
    path = tmp_path / "shop.csv"
    frame.to_csv(path, index=False)

    return path


def ids_of(items: list[dict]) -> dict[str, list]:
    """
    答えの候補を問いの識別子ごとの値にまとめる関数

    Parameters
    ----------
    items: list[dict]
        inspect_data の suggestions

    Returns
    ----------
    found: dict[str, list]
        問いの識別子から候補の値
    """
    # 識別子ごと
    found = {item["id"]: item["values"] for item in items}

    return found


def test_attach_and_inspect(tmp_path: Path) -> None:
    """
    データを入れると場所と読み方と列が入り, 答えの候補と確かめが出る
    ことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # ML-EDA が無ければ飛ばす
    pytest.importorskip("ml_eda")
    path = shop_csv(tmp_path)
    problem = answer(new_problem("店"), "ml.use", True)
    assert inspect_data(problem) is None
    problem = attach_data(problem, path)
    assert get_value(problem, "data.source") == str(path)
    assert get_value(problem, "data.loader") == "csv"
    assert get_value(problem, "data.size") == path.stat().st_size
    assert get_value(problem, "information.columns") == [
        "row_id", "date", "shop", "sales", "customers", "weather", "promo"
    ]

    # 目的の列の前: 目的の列, 時刻の列, 系列 (天気は系列でない), ID
    found = inspect_data(problem)
    assert found["overview"]["rows"] == 240
    suggested = ids_of(found["suggestions"])
    assert "sales" in suggested["ml.target"]
    assert "row_id" not in suggested["ml.target"]
    assert suggested["ml.time_column"] == ["date"]
    assert suggested["ml.series_by"] == ["shop"]
    assert suggested["information.forbidden"] == ["row_id"]

    # タスクを答えると目的の列は数だけ. 漏れの疑いは値でなく確かめ
    problem = answer(problem, "ml.task", "forecasting")
    problem = answer(problem, "ml.target", "sales")
    problem = answer(problem, "ml.time_column", "date")
    problem = answer(problem, "ml.horizon", 30)
    found = inspect_data(problem)
    suggested = ids_of(found["suggestions"])
    assert "shop" not in suggested["ml.target"]
    assert "ml.task" not in suggested
    assert "information.unknown" not in suggested
    by_id = {f["id"]: f for f in found["findings"]}
    assert "customers" in by_id["information.unknown"]["message"]
    assert "期先までの予測" in by_id["ml.horizon"]["message"]
    decided = answer(problem, "information.unknown", ["customers"])
    assert not any(f["id"] == "information.unknown"
                   for f in inspect_data(decided)["findings"])
    assert found["overview"]["kind"] == "timeseries"
    assert handoff_source(problem) == str(path)

    # 場所を変えると読み方は消え (同じ場所なら残る), 無いデータは誤り
    same = answer(problem, "data.source", str(path))
    assert get_value(same, "data.loader") == "csv"
    moved = answer(problem, "data.source", str(tmp_path / "none.csv"))
    assert get_value(moved, "data.loader") is None
    assert inspect_data(moved)["findings"][0]["level"] == "error"
    with pytest.raises(ValueError, match="絶対パス"):
        attach_data(problem, "shop.csv")
    with pytest.raises(ValueError, match="絶対パス"):
        answer(problem, "data.source", "shop.csv")


def handoff_source(problem: dict) -> str:
    """
    受け渡しのデータの分析の値から, データの場所を取り出す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (見本の答えを足して渡せる形にする)

    Returns
    ----------
    source: str
        データの場所
    """
    # 渡せるように, 見本の残りの答えを重ねる
    full = example_problem("総菜の作る量")
    full["data"] = problem["data"]
    source = handoff(full)["data"]["source"]["source"]

    return source


def test_stale_time_column_and_file(tmp_path: Path) -> None:
    """
    答えた時刻の列がデータに無い場合と, タスクを変えた後と, ファイルが
    変わった後を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 時刻の列を答えてからデータを入れる (列 day は無い)
    pytest.importorskip("ml_eda")
    path = shop_csv(tmp_path)
    problem = answer(new_problem("店"), "ml.use", True)
    problem = answer(problem, "ml.task", "forecasting")
    problem = answer(problem, "ml.time_column", "day")
    problem = attach_data(problem, path)
    assert "time_column" not in (get_value(problem, "data.options") or {})
    found = inspect_data(problem)
    assert found["overview"] is not None
    assert any(f["id"] == "ml.time_column" for f in found["findings"])
    assert ids_of(found["suggestions"])["ml.time_column"] == ["date"]

    # 回帰に変えると時刻の列の答えは効かない
    regression = answer(answer(problem, "ml.time_column", "date"),
                        "ml.task", "regression")
    assert not any(f["id"] == "ml.time_column"
                   for f in inspect_data(regression)["findings"])

    # 入れた後にファイルが変わると知らせ
    frame = pd.read_csv(path).rename(columns={"promo": "sale"})
    frame.to_csv(path, index=False)
    messages = [f["message"] for f in inspect_data(problem)["findings"]]
    assert any("ファイルが変わりました" in m for m in messages)
    assert any("列と違います" in m for m in messages)


def test_classification_values(tmp_path: Path) -> None:
    """
    分類の当たりのクラスの候補と, 小数のラベルと, 値に無いクラスと, 少ない
    行の確かめを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 30 行. ラベルは欠損があるので小数の 0.0 と 1.0
    pytest.importorskip("ml_eda")
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 30).astype(float)
    labels[0] = np.nan
    path = tmp_path / "churn.csv"
    pd.DataFrame({"months": rng.integers(1, 40, 30),
                  "churned": labels}).to_csv(path, index=False)
    problem = attach_data(answer(new_problem("解約"), "ml.use", True), path)
    problem = answer(problem, "ml.task", "classification")
    problem = answer(problem, "ml.target", "churned")
    found = inspect_data(problem)
    assert sorted(ids_of(found["suggestions"])["ml.positive"]) == ["0", "1"]
    assert any("少なく" in f["message"] for f in found["findings"])
    right = answer(problem, "ml.positive", "1")
    assert not any(f["id"] == "ml.positive"
                   for f in inspect_data(right)["findings"])
    wrong = answer(problem, "ml.positive", "解約")
    assert any(f["id"] == "ml.positive"
               for f in inspect_data(wrong)["findings"])

    # データの問いは確かめ (check) の足りない答えにはならない (任意)
    assert not any(f["id"] == "data.source" for f in check(problem))


def test_unreadable_and_missing_eda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    読めないファイルを error の確かめにすることと, ML-EDA が無い場合の
    誤りの文を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        読み込みの置き換え

    Returns
    ----------
    None
    """
    # 壊れた Excel は落ちずに error
    pytest.importorskip("ml_eda")
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not a zip")
    problem = answer(new_problem("壊れ"), "ml.use", True)
    problem = answer(problem, "data.source", str(broken))
    found = inspect_data(problem)
    assert found["findings"][0]["level"] == "error"
    with pytest.raises(ValueError, match="読め"):
        attach_data(problem, broken)

    # ML-EDA が無い
    monkeypatch.setitem(sys.modules, "ml_eda", None)
    with pytest.raises(ValueError, match="ML-EDA"):
        attach_data(problem, broken)


def test_series_by_two_columns(tmp_path: Path) -> None:
    """
    店と品の 2 つの列で系列が分かれるデータで, 組み合わせを系列の候補に
    出すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 30 日 × 2 店 × 2 品
    pytest.importorskip("ml_eda")
    days = pd.date_range("2025-01-01", periods=30).strftime("%Y-%m-%d")
    rows = [(d, s, i, float(k)) for k, (d, s, i) in enumerate(
        (d, s, i) for d in days for s in ("a", "b") for i in ("x", "y")
    )]
    path = tmp_path / "panel.csv"
    pd.DataFrame(rows, columns=["date", "store", "item", "sales"]).to_csv(
        path, index=False
    )
    problem = attach_data(answer(new_problem("総菜"), "ml.use", True), path)
    assert ids_of(inspect_data(problem)["suggestions"])["ml.series_by"] == [
        "store", "item"
    ]
    problem = answer(problem, "ml.time_column", "date")
    problem = answer(problem, "ml.task", "forecasting")
    found = inspect_data(problem)
    assert ids_of(found["suggestions"])["ml.series_by"] == ["store", "item"]
    assert not any("間隔が一定ではありません" in f["message"]
                   for f in found["findings"])
