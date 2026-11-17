from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_quest import (
    answer, attach_data, check, get_value, inspect_data, new_problem
)


def shop_csv(tmp_path: Path) -> Path:
    """
    2 店の日々の売上と, 売上から作った客数と ID の列の表を書く関数

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
    frame = pd.DataFrame({
        "row_id": np.arange(240), "date": np.r_[days, days],
        "shop": ["a"] * 120 + ["b"] * 120, "sales": sales,
        "customers": (sales * 3 + rng.normal(0, 1, 240)).round(),
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
    assert get_value(problem, "information.columns")[0] == "row_id"

    # 目的の列の前: 目的の列, 時刻の列, 系列, ID の候補
    found = inspect_data(problem)
    assert found["overview"]["rows"] == 240
    suggested = ids_of(found["suggestions"])
    assert "sales" in suggested["ml.target"]
    assert "row_id" not in suggested["ml.target"]
    assert suggested["ml.time_column"] == ["date"]
    assert "shop" in suggested["ml.series_by"]
    assert suggested["information.forbidden"] == ["row_id"]

    # 目的の列と時刻の列を答えると, タスクと漏れの疑いの候補
    problem = answer(problem, "ml.target", "sales")
    problem = answer(problem, "ml.task", "forecasting")
    problem = answer(problem, "ml.time_column", "date")
    problem = answer(problem, "ml.horizon", 30)
    found = inspect_data(problem)
    suggested = ids_of(found["suggestions"])
    assert suggested["ml.task"] == ["forecasting"]
    assert suggested["information.unknown"] == ["customers"]
    assert found["overview"]["kind"] == "timeseries"
    messages = [f["message"] for f in found["findings"]]
    assert any("期先までの予測" in m for m in messages)

    # 場所を変えると読み方は消え, 無いデータは読めない誤り
    moved = answer(problem, "data.source", str(tmp_path / "none.csv"))
    assert get_value(moved, "data.loader") is None
    broken = inspect_data(moved)
    assert broken["findings"][0]["level"] == "error"
    with pytest.raises(ValueError, match="絶対パス"):
        attach_data(problem, "shop.csv")


def test_classification_and_small_data(tmp_path: Path) -> None:
    """
    分類の当たりのクラスの候補と, クラスが値に無い場合と少ない行の確かめを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 30 行の解約の表
    pytest.importorskip("ml_eda")
    rng = np.random.default_rng(1)
    path = tmp_path / "churn.csv"
    pd.DataFrame({
        "months": rng.integers(1, 40, 30),
        "churned": rng.choice(["yes", "no"], 30),
    }).to_csv(path, index=False)
    problem = attach_data(answer(new_problem("解約"), "ml.use", True), path)
    problem = answer(problem, "ml.target", "churned")
    problem = answer(problem, "ml.task", "classification")
    found = inspect_data(problem)
    suggested = ids_of(found["suggestions"])
    assert sorted(suggested["ml.positive"]) == ["no", "yes"]
    assert any("少なく" in f["message"] for f in found["findings"])
    wrong = answer(problem, "ml.positive", "解約")
    assert any(f["id"] == "ml.positive"
               for f in inspect_data(wrong)["findings"])

    # データの問いは確かめ (check) の足りない答えにはならない (任意)
    assert not any(f["id"] == "data.source" for f in check(problem))
