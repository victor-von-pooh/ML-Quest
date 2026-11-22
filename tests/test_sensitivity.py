from pathlib import Path

import pandas as pd
import pytest

from ml_quest import (
    SENSITIVITY_FIELDS, answer, attach_data, decision_sensitivity,
    example_problem, inspect_data
)
from test_project_data import daily_csv


def triage_problem(tmp_path: Path) -> dict:
    """
    当たりを 1 件見逃し外れを 1 件回す 4 件の並びのデータで, 振り分けの
    問題の設定を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    problem: dict
        問題の設定 (120 行. 当たり 60 行, 回した行 60 行)
    """
    # 見逃しの損 10, 空振りの損 1. 今のやり方は回したかの列
    path = tmp_path / "reviews.csv"
    pd.DataFrame({
        "text_length": [10, 20, 30, 40] * 30,
        "label": ["悪い", "悪い", "良い", "良い"] * 30,
        "checked": ["yes", "no", "yes", "no"] * 30,
    }).to_csv(path, index=False)
    problem = example_problem("解約の引き止め")
    problem = answer(problem, "information.columns",
                     ["text_length", "label", "checked"])
    problem = answer(problem, "ml.parts.0.target.column", "label")
    problem = answer(problem, "ml.parts.0.positive", "悪い")
    problem = answer(problem, "information.known", ["text_length"])
    problem = answer(problem, "information.forbidden", None)
    problem = answer(problem, "action.decision", {
        "type": "triage", "miss_cost": 10, "false_alarm_cost": 1,
    })
    problem = attach_data(problem, path)
    problem = answer(problem, "action.current", {"type": "column",
                                                 "column": "checked"})

    return problem


def test_quantity_rule() -> None:
    """
    量を決める型で, 損の値を変えると置く水準がどう変わるかが分かることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 足りない損 200, 余る損 150 の水準と, 幅の両端を含む点
    result = decision_sensitivity(example_problem("総菜の作る量"),
                                  {"under_cost": [100, 300],
                                   "over_cost": [50, 250]})
    assert result["type"] == "quantity"
    assert result["base"]["rule"] == pytest.approx(200 / 350)
    under, over = result["fields"]
    assert [point["value"] for point in under["points"]] == [
        100, 150, 200, 250, 300
    ]
    assert under["points"][0]["rule"] == pytest.approx(100 / 250)
    assert over["points"][0]["rule"] == pytest.approx(200 / 250)
    assert over["swing"]["rule"] > under["swing"]["rule"]
    assert under["swing"]["loss"] is None
    assert (
        "足りないときの損を 100 から 300 に変えると (他の値は今のまま), "
        "置く分位点の水準は 0.40 から 0.67"
    ) in under["sentence"]
    assert "0.40 から 0.67" in under["sentence"]

    # データが無ければ損は求めず, 予測が無いと量が求まらないと知らせる
    assert result["data"] is None
    assert "loss" not in result["base"]
    assert any("データで書いていない" in note for note in result["notes"])
    assert any("たどり直し" in note for note in result["notes"])

    # 点の数
    three = decision_sensitivity(example_problem("総菜の作る量"),
                                 {"under_cost": (100, 300)}, points=3)
    assert [point["value"] for point in three["fields"][0]["points"]] == [
        100, 200, 300
    ]


def test_quantity_current_loss(tmp_path: Path) -> None:
    """
    量を決める型で, データの上の今のやり方の損が損の値とともに変わり,
    今の値では inspect_data の損と同じになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 1 日前と同じ量を置く今のやり方
    pytest.importorskip("ml_eda")
    problem = attach_data(example_problem("総菜の作る量"),
                          daily_csv(tmp_path))
    problem = answer(problem, "action.current", {"type": "lag",
                                                 "periods": 1})
    baseline = inspect_data(problem)["baseline"]
    result = decision_sensitivity(problem, {"under_cost": [0.5, 400]})
    assert result["data"]["rows"] == baseline["rows"]
    assert result["data"]["days"] == pytest.approx(baseline["days"])
    assert result["base"]["loss"]["current"] == pytest.approx(
        baseline["total"]
    )
    assert result["base"]["per_30_days"]["current"] == pytest.approx(
        baseline["per_30_days"]
    )

    # 足りない損が大きいほど損も大きく, 振れ幅と文に出る
    field = result["fields"][0]
    losses = [point["loss"]["current"] for point in field["points"]]
    assert losses == sorted(losses) and losses[0] < losses[-1]
    assert field["swing"]["loss"] == pytest.approx(losses[-1] - losses[0])
    assert "今のやり方の損は, 30 日あたり" in field["sentence"]
    assert "0.5 から 400" in field["sentence"]


def test_select_rule() -> None:
    """
    対象を選ぶ型で, 得になる確率の下限と件数の上限の変わり方と, どの対象も
    選ばなくなる値が分かることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 価値 24000, 費用 1500, 効果 0.2, 件数の上限 300
    problem = example_problem("解約の引き止め")
    result = decision_sensitivity(problem, {"value": [1000, 30000],
                                            "effect": [0.05, 1]})
    assert result["base"] == {"rule": pytest.approx(1500 / 4800),
                              "acts": True, "limit": 300}
    value, effect = result["fields"]
    assert value["points"][0]["acts"] is False
    assert value["points"][-1]["rule"] == pytest.approx(1500 / 6000)
    assert "どの対象も選ばない" in value["sentence"]
    assert effect["points"][-1]["rule"] == pytest.approx(1500 / 24000)
    assert any("今のやり方の得は求めません" in note
               for note in result["notes"])

    # 費用の上限は, 費用あたりの件数 (件数の上限と一緒には渡せない)
    with pytest.raises(ValueError, match="1 つだけ"):
        decision_sensitivity(problem, {"budget_cost": [1000, 9000]})
    spend = answer(problem, "action.decision", {
        "type": "select", "value": 24000, "cost": 1500, "effect": 0.2,
        "budget_cost": 300000,
    })
    result = decision_sensitivity(spend, {"budget_cost": [150000, 600000],
                                          "cost": [1000, 3000]})
    assert result["base"]["limit"] == 200
    budget, cost = result["fields"]
    assert [point["limit"] for point in budget["points"]][::4] == [100, 400]
    assert "件数の上限は 100 件 から 400 件" in budget["sentence"]
    assert cost["points"][0]["limit"] == 300

    # 確率の差は 費用 / 価値 (効果の列は使わない)
    uplift = answer(problem, "action.decision", {
        "type": "select", "value": 24000, "cost": 1500, "effect": "inquiries",
        "effect_kind": "uplift", "effect_goal": "reduce",
    })
    result = decision_sensitivity(uplift, {"cost": [1500, 3000]})
    assert result["rule"] == "得になる確率の差の大きさの下限"
    assert result["base"]["rule"] == pytest.approx(1500 / 24000)
    assert result["base"]["limit"] is None


def test_triage_alternatives(tmp_path: Path) -> None:
    """
    振り分けの型で, 今のやり方と素朴なやり方の損と, 最も損の小さいやり方の
    変わり方が分かることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 今のやり方 30 × (10 + 1), 回さない 60 × 10, 全てを回す 60 × 1
    pytest.importorskip("ml_eda")
    problem = triage_problem(tmp_path)
    result = decision_sensitivity(problem, {"miss_cost": [0.5, 10],
                                            "capacity": [10, 50]})
    base = result["base"]
    assert base["loss"] == {"current": 330, "none": 600, "all": 60}
    assert base["best"] == "all"
    assert base["rule"] == pytest.approx(1 / 11)
    assert base["per_30_days"] is None
    assert result["data"] == {"rows": 120, "days": None}

    # 見逃しの損が小さいと, 回さないのが最も損が小さい
    miss, capacity = result["fields"]
    assert miss["points"][0]["loss"] == {"current": 45, "none": 30,
                                         "all": 60}
    assert miss["points"][0]["best"] == "none"
    assert "0.5 ではどれも回さない" in miss["sentence"]
    assert "10 では全てを回す" in miss["sentence"]
    assert "120 行で" in miss["sentence"]
    assert [point["limit"] for point in capacity["points"]] == [
        10, 20, 30, 40, 50
    ]
    assert "どの値でも, 最も損が小さいのは全てを回す" in capacity["sentence"]


def test_errors() -> None:
    """
    幅を渡せない意思決定, 項目, 幅の誤りで止まることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 意思決定しない, 直すところのある損得
    problem = example_problem("総菜の作る量")
    none = answer(problem, "action.decision", {"type": "none"})
    with pytest.raises(ValueError, match="意思決定にだけ"):
        decision_sensitivity(none, {"under_cost": [1, 2]})
    broken = answer(problem, "action.decision", {
        "type": "quantity", "under_cost": 200,
    })
    with pytest.raises(ValueError, match="直すところ"):
        decision_sensitivity(broken, {"under_cost": [1, 2]})

    # 幅を渡せない項目, 列で書いた項目, 幅の形と範囲, 点の数
    with pytest.raises(ValueError, match="渡せるのは under_cost"):
        decision_sensitivity(problem, {"stock": [0, 10]})
    rows = answer(problem, "action.decision", {
        "type": "quantity", "under_cost": "promo", "over_cost": 150,
    })
    with pytest.raises(ValueError, match="行ごとの値"):
        decision_sensitivity(rows, {"under_cost": [1, 2]})
    for bounds in ([1], [2, 1], [1, float("nan")], "1-2", [True, 2]):
        with pytest.raises(ValueError, match="幅"):
            decision_sensitivity(problem, {"under_cost": bounds})
    with pytest.raises(ValueError, match="0 より大きく"):
        decision_sensitivity(problem, {"under_cost": [0, 2]})
    churn = example_problem("解約の引き止め")
    with pytest.raises(ValueError, match="1 以下"):
        decision_sensitivity(churn, {"effect": [0.5, 1.5]})
    for points in (1, 0, True):
        with pytest.raises(ValueError, match="points"):
            decision_sensitivity(problem, {"under_cost": [1, 2]},
                                 points=points)
    with pytest.raises(ValueError, match="辞書"):
        decision_sensitivity(problem, {})

    # 渡せる項目の一覧は型ごと
    assert set(SENSITIVITY_FIELDS) == {"quantity", "select", "triage"}


def test_single_point() -> None:
    """
    下と上が等しい幅では, 1 つの点だけを求めることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 1 つの点と, 幅でない文
    result = decision_sensitivity(example_problem("総菜の作る量"),
                                  {"under_cost": [300, 300]})
    field = result["fields"][0]
    assert [point["value"] for point in field["points"]] == [300]
    assert field["swing"]["rule"] == 0
    assert field["sentence"].startswith(
        "足りないときの損を 300 に変えると (他の値は今のまま), "
    )


def test_baseline_reference(tmp_path: Path) -> None:
    """
    今のやり方の損が, 系列ごとに 1 日前の値を置いた損を直に数えた値と
    同じになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 店と品ごとに日の順で 1 つずらした量を置いた, 足りない損と余る損
    pytest.importorskip("ml_eda")
    path = daily_csv(tmp_path)
    frame = pd.read_csv(path).sort_values("date")
    placed = frame.groupby(["store", "item"])["sales"].shift(1)
    keep = placed.notna()
    gap = frame["sales"][keep] - placed[keep]
    expected = float((200 * gap.clip(lower=0)
                      + 150 * (-gap).clip(lower=0)).sum())
    problem = attach_data(example_problem("総菜の作る量"), path)
    problem = answer(problem, "action.current", {"type": "lag",
                                                 "periods": 1})
    baseline = inspect_data(problem)["baseline"]
    assert baseline["total"] == pytest.approx(expected)
    assert baseline["rows"] == int(keep.sum())
    result = decision_sensitivity(problem, {"under_cost": [200, 400]})
    assert result["fields"][0]["points"][0]["loss"]["current"] == (
        pytest.approx(expected)
    )


def test_missing_parts(tmp_path: Path) -> None:
    """
    損得の値の列がデータに無い場合と, 振り分けの当たりのクラスが無い場合は,
    落ちずに損を求めない理由を返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 使えるデータの列にはあるが, データに無い列で書いた損
    pytest.importorskip("ml_eda")
    problem = attach_data(example_problem("総菜の作る量"),
                          daily_csv(tmp_path))
    problem = answer(problem, "action.current", {"type": "lag",
                                                 "periods": 1})
    columns = list(problem["information"]["columns"]) + ["waste"]
    problem = answer(problem, "information.columns", columns)
    problem = answer(problem, "action.decision", {
        "type": "quantity", "under_cost": 200, "over_cost": "waste",
    })
    result = decision_sensitivity(problem, {"under_cost": [100, 300]})
    assert result["data"] is None
    assert "loss" not in result["base"]
    assert any("求められません" in note and "waste" in note
               for note in result["notes"])
    assert "求められません" in inspect_data(problem)["baseline"]["reason"]

    # 当たりのクラスの無い振り分け
    triage = answer(triage_problem(tmp_path), "ml.parts.0.positive", None)
    result = decision_sensitivity(triage, {"miss_cost": [1, 10]})
    assert result["data"] is None
    assert any("当たりのクラスが決まっていない" in note
               for note in result["notes"])
