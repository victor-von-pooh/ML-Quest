import ast
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from ml_quest import (
    QUESTIONS, answer, apply_recipe, attach_data, check, describe,
    example_problem, get_value, handoff, inspect_data, load_problem,
    make_project, open_errors, problem_origin, project_error, prune_copies,
    recipe_from, record_project, resolve_choice, save_problem
)
from ml_quest.check import is_exportable
from ml_quest import data as data_module
from ml_quest.data import data_stat, inspect_key


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
    found = inspect_data(problem)
    assert found["baseline"]["total"] == 30 * (10 + 1)


def test_recipe_read_options(tmp_path: Path) -> None:
    """
    attach_data に渡した読み方の設定を, recipe_from がそのまま返し, その
    読み方で ML-EDA が読めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # セミコロン区切りの Shift_JIS に書き直して入れる. 時刻の列は答えから
    eda = pytest.importorskip("ml_eda")
    path = daily_csv(tmp_path)
    frame = pd.read_csv(path)
    frame.to_csv(path, sep=";", encoding="cp932", index=False)
    options = {"sep": ";", "encoding": "cp932"}
    problem = attach_data(example_problem("総菜の作る量"), path,
                          options=options)
    assert get_value(problem, "data.options") == options
    load = recipe_from(problem)["load"]
    assert load["options"] == {**options, "time_column": "date"}
    assert handoff(problem)["data"]["source"]["options"] == options
    data = eda.load_data(load["path"], loader=load["loader"],
                         **load["options"])
    assert list(data.frame.columns) == list(frame.columns)


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
    assert recipe_from(dropped)["recipe"]["steps"][-1] == {
        "op": "drop_columns", "columns": ["customers"]
    }

    # 書き戻したレシピは次の雛形になる (往復で同じ)
    assert recipe_from(result["problem"])["recipe"] == changed

    # 目的の列とタスクは書き戻し, 使ってはいけない列と漏れは食い違い
    other = dict(recipe, target="customers", steps=[
        {"op": "lag", "columns": ["customers"], "lags": [1],
         "by": ["store", "item"], "time": "date", "freq": "86400s"},
    ], split={"method": "random"})
    forbidden = answer(problem, "information.forbidden", ["promo"])
    result = apply_recipe(forbidden, other)
    assert get_value(
        result["problem"], "ml.parts.0.target.column"
    ) == "customers"
    assert "customers" not in (
        get_value(result["problem"], "information.unknown") or []
    )
    levels = {(f["id"], f["level"]) for f in result["conflicts"]}
    assert [c["column"] for c in result["choices"]] == ["sales"]
    assert ("information.forbidden", "error") in levels
    assert ("ml.parts.0.available_lag", "warning") in levels
    assert ("ml.parts.0.task", "warning") in levels


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


def test_make_project_open_errors(tmp_path: Path) -> None:
    """
    入れたデータの確かめに error が残ると, 確かめ (accept_errors) 無しでは
    書き出さず, 確かめて書き出すと README に一覧を残すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 直すところが無ければ, 今のまま書き出し, README に節を書かない
    pytest.importorskip("ml_eda")
    path = daily_csv(tmp_path)
    problem = attach_data(example_problem("総菜の作る量"), path)
    assert open_errors(problem) == []
    make_project(problem, tmp_path / "clean")
    readme = (tmp_path / "clean" / "README.md").read_text("utf-8")
    assert "残っていた直すところ" not in readme

    # 入れた後に消えたデータは, 問題の設定の確かめ (check) では分からず,
    # 入れたデータの確かめの error
    path.unlink()
    errors = open_errors(problem)
    assert errors and all(item["level"] == "error" for item in errors)
    assert not [item for item in check(problem) if item["level"] == "error"]
    with pytest.raises(ValueError, match="accept_errors"):
        make_project(problem, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()

    # 確かめて書き出すと, README の最後の節に一覧
    make_project(problem, tmp_path / "accepted", accept_errors=True)
    readme = (tmp_path / "accepted" / "README.md").read_text("utf-8")
    section = readme.split("## 書き出した時点で残っていた直すところ")[1]
    assert "\n## " not in section
    for item in errors:
        assert f"- `{item['id']}`: {item['message']}" in section


def test_make_project_check_errors(tmp_path: Path) -> None:
    """
    問題の設定の確かめの error のうち, 骨組みを組めるもの (確かめ方, 損得
    など) は「それでも書き出す」で書き出せ, 組めないもの (目的の列など) は
    止まることと, 覚えておいた分析の結果を渡せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 組める error: 受け渡しは止まり, 書き出しは指定すれば通る
    problem = answer(example_problem("総菜の作る量"),
                     "verification.success", None)
    with pytest.raises(ValueError, match="直すところ"):
        handoff(problem)
    assert project_error(problem) is None
    errors = open_errors(problem)
    assert [item["id"] for item in errors] == ["verification.success"]
    with pytest.raises(ValueError, match="accept_errors"):
        make_project(problem, tmp_path / "blocked")
    make_project(problem, tmp_path / "accepted", accept_errors=True)
    readme = (tmp_path / "accepted" / "README.md").read_text("utf-8")
    assert "- `verification.success`: " in readme

    # 覚えておいた分析の結果 (改行のある文も 1 行の箇条に)
    inspected = {"key": inspect_key(problem), "findings": [
        {"level": "error", "id": "data.source", "message": "読めない\n行"},
        {"level": "warning", "id": "data.source", "message": "少ない"},
    ]}
    assert [item["message"] for item in open_errors(problem, inspected)] == [
        errors[0]["message"], "読めない\n行"
    ]
    make_project(problem, tmp_path / "given", accept_errors=True,
                 inspected=inspected)
    readme = (tmp_path / "given" / "README.md").read_text("utf-8")
    assert "- `data.source`: 読めない 行\n" in readme
    assert "少ない" not in readme
    assert "推論の段階で問題につなげない" in readme

    # 古いか別の問題の設定の結果と, 形の崩れた結果は使わず求め直す
    for stale in ({**inspected, "key": "old"}, [1],
                  {"key": inspect_key(problem), "findings": "x"}):
        assert open_errors(problem, stale) == errors

    # 組めない error: 指定しても止まる
    broken = answer(problem, "ml.parts.0.target.column", None)
    assert "直すところ" in str(project_error(broken))
    with pytest.raises(ValueError, match="直すところ"):
        make_project(broken, tmp_path / "broken", accept_errors=True)
    assert not (tmp_path / "broken").exists()


# 問いと確かめの項目の場所ごとの, 骨組みを組める error か (新しい問いや
# 確かめの場所は, ここに足さないとテストが落ちる)
EXPORTABLE_IDS = {
    "issue.owner", "issue.pain", "issue.current", "issue.kpi.name",
    "issue.kpi.unit", "issue.kpi.current", "issue.kpi.target",
    "issue.kpi.better", "issue.guard.name", "issue.guard.unit",
    "issue.guard.better", "issue.guard.line", "issue.guard.measure",
    "issue.guard.counted",
    "decomposition", "action.who", "action.when", "action.what",
    "action.decision", "action.basis", "ml.constraints",
    "ml.parts.0.what", "ml.parts.0.item", "ml.parts.0.baseline",
    "ml.parts.0.consumer", "ml.parts.0.downstream",
    "ml.parts.0.evaluation.link", "ml.parts.0.evaluation.current",
    "verification.method",
    "verification.backtest", "verification.trial", "verification.success",
    "operation.review", "operation.approver", "operation.stop",
    "operation.remeasure",
}
BLOCKING_IDS = {
    "format_version", "ml.parts", "ml.use", "ml.reason", "ml.gpu",
    "ml.parts.0.role", "ml.parts.0.unit", "ml.parts.0.output",
    "ml.parts.0.target.source", "ml.parts.0.target.process.name",
    "ml.parts.0.target.process.settings", "ml.parts.0.target.kind",
    "ml.parts.0.target.column", "ml.parts.0.method", "ml.parts.0.task",
    "ml.parts.0.positive", "ml.parts.0.group_column",
    "ml.parts.0.split_time_column", "ml.parts.0.time_column",
    "ml.parts.0.series_by", "ml.parts.0.available_lag",
    "ml.parts.0.horizon", "ml.parts.0.autonomous", "ml.parts.0.ros2",
    "ml.parts.0.collect", "ml.parts.0.simulator", "ml.parts.0.env_id",
    "ml.parts.0.evaluation.metric", "ml.parts.0.evaluation.name",
    "ml.parts.0.evaluation.better", "ml.parts.0.policy.action_limits",
    "ml.parts.0.policy.stop", "ml.parts.0.policy.fallback",
    "ml.parts.0.policy.fallback_action", "ml.parts.0.policy.safety_key",
    "ml.parts.0.policy.approver", "information.columns",
    "information.known", "information.unknown", "information.forbidden",
    "data.source", "data.missing", "action.current",
    "ml.parts.0.ready.library", "ml.parts.0.ready.model",
    "ml.parts.0.ready.weights", "ml.parts.0.ready.sha256",
    "ml.parts.0.rank.user", "ml.parts.0.rank.item", "ml.parts.0.rank.time",
    "ml.parts.0.rank.reaction", "ml.parts.0.rank.k",
    "ml.parts.0.rank.shown", "ml.parts.0.rank.presentation",
    "ml.parts.0.group.columns", "ml.parts.0.group.id", "ml.parts.0.group.k",
    "ml.parts.0.group.stability", "ml.parts.0.anomaly.label",
    "ml.parts.0.anomaly.id", "ml.parts.0.anomaly.columns",
    "ml.parts.0.anomaly.score", "ml.parts.0.anomaly.capacity",
    "ml.parts.0.uplift.treatment", "ml.parts.0.uplift.outcome",
    "ml.parts.0.uplift.outcome_kind", "ml.parts.0.uplift.id",
    "ml.parts.0.uplift.columns", "ml.parts.0.uplift.k",
    "ml.parts.0.varied", "ml.parts.0.price.price", "ml.parts.0.price.units",
    "ml.parts.0.price.cost", "ml.parts.0.price.product",
    "ml.parts.0.price.id", "ml.parts.0.price.columns",
    "ml.parts.0.demand.product", "ml.parts.0.demand.date",
    "ml.parts.0.demand.units", "ml.parts.0.demand.stock",
    "ml.parts.0.demand.id", "ml.parts.0.demand.columns",
    "ml.parts.0.demand.horizon", "ml.parts.0.demand.levels",
}


def test_exportable_ids() -> None:
    """
    全ての問いの項目の場所が, 骨組みを組める error か組めない error かの
    どちらかに決まっていることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 問いの場所は全て, 2 つの組のどちらか 1 つにある
    ids = {question["id"] for question in QUESTIONS}
    assert not ids - EXPORTABLE_IDS - BLOCKING_IDS, (
        ids - EXPORTABLE_IDS - BLOCKING_IDS
    )
    assert not EXPORTABLE_IDS & BLOCKING_IDS
    for key in EXPORTABLE_IDS:
        assert is_exportable(key), key
    for key in BLOCKING_IDS:
        assert not is_exportable(key), key

    # 頭の文字だけでは合わせない
    assert not is_exportable("action.decisions")
    assert not is_exportable("issues")


# 骨組みを組める error の種類ごとの, 問題の設定の直し方 (量を決める見本)
EXPORTABLE_CASES = {
    "verification": [("verification.success", None)],
    "issue": [("issue.kpi.target", 1500000)],
    "decision_true": [("action.decision",
                       {"type": "quantity", "under_cost": True,
                        "over_cost": 150})],
    "decision_negative": [("action.decision",
                           {"type": "quantity", "under_cost": -1,
                            "over_cost": 150})],
    "decision_select": [("action.decision",
                         {"type": "select", "value": 100, "cost": 10,
                          "effect": 0.3})],
}


@pytest.mark.parametrize("case", sorted(EXPORTABLE_CASES))
def test_exportable_script(tmp_path: Path, case: str) -> None:
    """
    骨組みを組める error の種類ごとに, 「それでも書き出す」で書いた
    train.py の定数が Python の値として読め, 損得と制約の error では
    意思決定の既定値を付けないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    case: str
        EXPORTABLE_CASES の名前

    Returns
    ----------
    None
    """
    # 手で書いた値も含めて直接入れ, 組める error だけが残る
    problem = example_problem("総菜の作る量")
    for key, value in EXPORTABLE_CASES[case]:
        if key == "action.decision":
            problem["action"]["decision"] = value
        else:
            problem = answer(problem, key, value)
    errors = [item for item in check(problem) if item["level"] == "error"]
    assert errors and all(is_exportable(item["id"]) for item in errors)
    assert project_error(problem) is None
    make_project(problem, tmp_path / "project", accept_errors=True)

    # 頭の大文字の定数は全て Python の値として読める (JSON の true など
    # の名前が残らない)
    script = (tmp_path / "project" / "train.py").read_text("utf-8")
    tree = ast.parse(script)
    constants = {
        node.targets[0].id: node.value for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0],
                                                       ast.Name)
        and node.targets[0].id.isupper()
    }
    for name in ("PROBLEM_NAME", "PREDICT", "DECISION", "CONCLUSION"):
        ast.literal_eval(constants[name])
    dropped = case.startswith("decision")
    assert (ast.literal_eval(constants["DECISION"]) is None) == dropped
    assert ("既定値は付けずに推論の一式" in script) == dropped
    readme = (tmp_path / "project" / "README.md").read_text("utf-8")
    assert ("意思決定の既定値を付けずに" in readme) == dropped

    # データの分析の段階のレシピの下書きは止めず, 知らせに
    start = recipe_from(problem)
    assert any("直すところが残っています" in note for note in start["notes"])
    with pytest.raises(ValueError):
        handoff(problem)


def test_data_decision_error(tmp_path: Path) -> None:
    """
    入れたデータの確かめにだけ損得と制約の error が残ると, 意思決定の既定値
    を付けず, README がデータの確かめの直すところとして書くことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 問題の設定の確かめには error が無く, 受け渡しの値は作れる
    problem = example_problem("総菜の作る量")
    handoff(problem)
    inspected = {"key": inspect_key(problem), "findings": [
        {"level": "error", "id": "action.decision",
         "message": "2 クラスの分類にだけ使えます"},
    ]}
    make_project(problem, tmp_path / "project", accept_errors=True,
                 inspected=inspected)
    readme = (tmp_path / "project" / "README.md").read_text("utf-8")
    assert "推論の段階に渡す値を作るときには確かめない" in readme
    assert "作れないため" not in readme
    assert "意思決定の既定値は付けない (損得と制約に直すところが" in readme
    assert "意思決定の既定値は問題の損得と制約" not in readme
    script = (tmp_path / "project" / "train.py").read_text("utf-8")
    assert "DECISION = None" in script


def test_inspect_key_directory(tmp_path: Path) -> None:
    """
    分析の結果の印が, ディレクトリの中のファイルの変化でも変わることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # サブディレクトリの中のファイルを足すと印が変わる
    folder = tmp_path / "images" / "circle"
    folder.mkdir(parents=True)
    (folder / "a.png").write_bytes(b"a")
    problem = example_problem("総菜の作る量")
    problem["data"] = {"source": str(tmp_path / "images"),
                       "loader": "image_folder"}
    before = inspect_key(problem)
    assert inspect_key(problem) == before
    (folder / "b.png").write_bytes(b"bb")
    assert inspect_key(problem) != before


class VanishingEntry:
    def __init__(self, entry: os.DirEntry, vanish: bool):
        # 元の項目と, 大きさを見る前に消えた扱いにするかを保存
        self.entry = entry
        self.vanish = vanish
        self.name = entry.name
        self.path = entry.path

    def is_dir(self, follow_symlinks: bool = True) -> bool:
        """
        ディレクトリかを返す関数

        Parameters
        ----------
        follow_symlinks: bool = True
            リンクをたどるか

        Returns
        ----------
        found: bool
            ディレクトリか
        """
        # 元の項目のまま
        found = self.entry.is_dir(follow_symlinks=follow_symlinks)

        return found

    def is_file(self) -> bool:
        """
        ファイルかを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        found: bool
            ファイルか
        """
        # 元の項目のまま
        found = self.entry.is_file()

        return found

    def stat(self) -> os.stat_result:
        """
        大きさと時刻を返す関数 (消えた扱いなら誤り). 呼んだ数を数える

        Parameters
        ----------
        None

        Returns
        ----------
        found: os.stat_result
            元の項目の値
        """
        # 呼んだ数と, 消えた扱いの誤り
        STAT_CALLS.append(self.name)
        if self.vanish:
            raise FileNotFoundError(self.path)
        found = self.entry.stat()

        return found


class VanishingScan:
    def __init__(self, folder: str):
        # 元の項目を, gone で始まる名前は消えた扱いにして保存
        with SCANDIR(folder) as found:
            self.entries = [VanishingEntry(entry,
                                           entry.name.startswith("gone"))
                            for entry in found]

    def __enter__(self) -> list:
        """
        項目の並びを返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        entries: list
            項目の並び
        """
        # 保存した並び
        entries = self.entries

        return entries

    def __exit__(self, *args: object) -> None:
        """
        何もしない関数

        Parameters
        ----------
        *args: object
            誤りの情報 (使わない)

        Returns
        ----------
        None
        """
        # 閉じるものは無し
        return None


# VanishingEntry.stat を呼んだ項目の名前と, 置き換える前の os.scandir
STAT_CALLS: list = []
SCANDIR = os.scandir


def test_data_stat_vanished_and_many(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    データの印で, たどった後に消えたファイルを飛ばし, 上限を超えるファイル
    は大きさと時刻を見ずに打ち切ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        os.scandir と上限の置き換え

    Returns
    ----------
    None
    """
    # たどった後に消えたファイルは飛ばし, 無い場所は None
    folder = tmp_path / "images"
    (folder / "sub").mkdir(parents=True)
    for name in ("a.png", "gone.png", "sub/b.png"):
        (folder / name).write_bytes(b"x")
    monkeypatch.setattr(data_module.os, "scandir", VanishingScan)
    stat = data_stat(folder)
    assert [row[0] for row in stat[:-1]] == ["a.png", "sub/b.png"]
    assert stat[-1] == 2
    assert data_stat(tmp_path / "nothing") is None

    # 上限を超えたら, 上限 + 1 件で大きさと時刻を見るのを止める
    for index in range(20):
        (folder / f"f{index:02d}.png").write_bytes(b"x")
    monkeypatch.setattr(data_module, "STAT_FILES", 3)
    STAT_CALLS.clear()
    stat = data_stat(folder)
    assert [row[0] for row in stat[:-1]] == ["a.png", "f00.png", "f01.png"]
    assert stat[-1] == 4
    assert STAT_CALLS == ["a.png", "f00.png", "f01.png", "f02.png"]
    problem = example_problem("総菜の作る量")
    problem["data"] = {"source": str(folder), "loader": "image_folder"}
    before = inspect_key(problem)
    (folder / "f10.png").write_bytes(b"xx")
    assert inspect_key(problem) == before


def test_unanswered_use() -> None:
    """
    機械学習を使うかに答えていない問題の設定は, 答えるよう知らせることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 答えていないときと, 使わないと答えたとき
    problem = answer(example_problem("総菜の作る量"), "ml.use", None)
    assert "「機械学習を使いますか」に答えてください" in str(
        project_error(problem)
    )
    problem = answer(problem, "ml.use", False)
    assert "機械学習を使わない" in str(project_error(problem))


def test_amount_finite() -> None:
    """
    損得の値の非数と無限大を誤りにすることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 手で書いた非数と無限大
    for value in (float("nan"), float("inf")):
        problem = example_problem("総菜の作る量")
        problem["action"]["decision"]["under_cost"] = value
        messages = [item["message"] for item in check(problem)
                    if item["level"] == "error"]
        assert any("有限の数" in message for message in messages)

    # 手で書いた真偽値は, 要約に Python の名前で出さない
    problem = example_problem("総菜の作る量")
    problem["action"]["decision"]["under_cost"] = True
    summary = " ".join(describe(problem))
    assert "true (数でない値)" in summary and "True" not in summary


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


def test_recipe_target_dropped() -> None:
    """
    目的の列を落とすレシピを誤りにし, 前の目的の列を落とすと選ぶと最後の
    手順で落とすことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 目的の列を客数に変え, 前の目的の列 (売れた数) の過去の値を使う
    problem = example_problem("総菜の作る量")
    recipe = {
        "format_version": 1, "target": "customers", "task": "forecasting",
        "steps": [
            {"op": "drop_columns", "columns": ["customers"]},
            {"op": "lag", "columns": ["sales"], "lags": [2]},
        ],
        "split": {"method": "time"},
    }
    result = apply_recipe(problem, recipe)
    assert any(f["id"] == "ml.parts.0.target.column" and f["level"] == "error"
               for f in result["conflicts"])

    # 前の目的の列を落とすと, 過去の値を作った後の最後の手順で落とす
    recipe["steps"] = recipe["steps"][1:]
    result = apply_recipe(problem, recipe)
    assert [c["column"] for c in result["choices"]] == ["sales"]
    dropped = resolve_choice(result["problem"], "sales", "drop")
    steps = get_value(dropped, "data.recipe")["steps"]
    assert steps[-1] == {"op": "drop_columns", "columns": ["sales"]}
    again = resolve_choice(dropped, "sales", "drop")
    assert get_value(again, "data.recipe")["steps"] == steps

    # ワンホットなどの対象からは外し, 過去の値は残す
    recipe["steps"] = [
        {"op": "lag", "columns": ["sales"], "lags": [2]},
        {"op": "one_hot", "columns": ["sales", "store"]},
        {"op": "fill_missing", "column": "sales"},
    ]
    result = apply_recipe(problem, recipe)
    steps = get_value(resolve_choice(result["problem"], "sales", "drop"),
                      "data.recipe")["steps"]
    assert steps == [
        {"op": "lag", "columns": ["sales"], "lags": [2]},
        {"op": "one_hot", "columns": ["store"]},
        {"op": "drop_columns", "columns": ["sales"]},
    ]


def test_record_project(tmp_path: Path) -> None:
    """
    書き出したプロジェクトの記録が config の場所の絶対パスになり, 受け渡し
    に出て, プロジェクトの写しには入らないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 記録は config の data.dataset と model.spec から (元は変えない)
    problem = example_problem("総菜の作る量")
    project = tmp_path / "project"
    make_project(problem, project)
    recorded = record_project(problem, project)
    assert "project" not in problem
    place = project.resolve()
    assert recorded["project"] == {
        "path": str(place), "dataset": str(place / "outputs" / "dataset"),
        "model_spec": str(place / "models" / "model.yaml"),
    }

    # 受け渡しの段階ごとの値と, 保存して読み直しても同じ
    stages = handoff(recorded)
    assert stages["project"] == {"path": str(place)}
    assert stages["data"]["project_dataset"] == recorded["project"][
        "dataset"
    ]
    assert stages["model"]["project_model_spec"] == recorded["project"][
        "model_spec"
    ]
    save_problem(recorded, tmp_path / "problem.yaml")
    assert handoff(load_problem(tmp_path / "problem.yaml")) == stages

    # 書き出し直すと, プロジェクトの写しには記録を入れず, 記録は置き換え
    other = tmp_path / "other"
    make_project(recorded, other)
    assert "project" not in load_problem(other)
    again = record_project(recorded, other)
    assert again["project"]["path"] == str(other.resolve())

    # 崩した記録と相対パスは「記録が無い」. config が無ければ誤り
    broken = dict(recorded, project={"path": "rel", "dataset": 3})
    assert handoff(broken)["project"] == {"path": None}
    assert handoff(broken)["data"]["project_dataset"] is None
    assert handoff(dict(recorded, project="x"))["project"]["path"] is None
    with pytest.raises(ValueError, match="make_project"):
        record_project(problem, tmp_path / "none")


def test_config_requires(tmp_path: Path) -> None:
    """
    骨組みの config の requires に, 学習のスクリプトが読み込むライブラリと
    問題の設定から要る機能が入ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 列の予測は 3 つのライブラリ. GPU が要れば cuda も
    make_project(example_problem("総菜の作る量"), tmp_path / "a")
    config = yaml.safe_load(
        (tmp_path / "a/configs/default/train.yaml").read_text("utf-8")
    )
    assert config["requires"] == ["eda", "modelcontrol", "inference"]
    gpu = answer(example_problem("総菜の作る量"), "ml.gpu", True)
    make_project(gpu, tmp_path / "b")
    config = yaml.safe_load(
        (tmp_path / "b/configs/default/train.yaml").read_text("utf-8")
    )
    assert config["requires"] == ["eda", "modelcontrol", "inference",
                                  "cuda"]


def test_requires_in_ml_found(tmp_path: Path) -> None:
    """
    骨組みの requires の名前が ML-Found の登録簿にあり, 足りない機能が
    あれば学習の前に入れ方を示して止まることを確かめる関数

    ML-Found が入っていなければ飛ばす

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 登録簿の名前 (確かめずに一覧だけ)
    ml_found = pytest.importorskip("ml_found")
    entries = ml_found.describe_requirements(check=False)
    registered = {entry["name"] for entry in entries}
    missing = [name for name in ("eda", "modelcontrol", "inference")
               if name not in registered]
    assert not missing, f"ML-Found の登録簿に無い機能: {missing}"

    # 入っていない機能 (ROS2) を足した骨組みは学習の前に止まる
    problem = answer(example_problem("総菜の作る量"), "ml.gpu", None)
    project = tmp_path / "project"
    make_project(problem, project)
    path = project / "configs" / "default" / "train.yaml"
    config = yaml.safe_load(path.read_text("utf-8"))
    config["requires"].append("ros2")
    path.write_text(yaml.safe_dump(config, allow_unicode=True), "utf-8")
    cfg, diff = ml_found.load_project_config(str(project), argv=[])
    with pytest.raises(Exception, match="ros2"):
        ml_found.start_experiment(cfg, diff, str(project))


def test_baseline_in_skeleton(tmp_path: Path) -> None:
    """
    今のやり方をデータで書いた問題の設定では, 学習のスクリプトが今のやり方
    と比べ, 書いていなければ理由の注記だけになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書いていない: 印 (TODO) でなく理由の注記
    project = tmp_path / "plain"
    make_project(example_problem("総菜の作る量"), project)
    script = (project / "train.py").read_text("utf-8")
    assert "TODO" not in script and "compare_baseline" not in script
    assert "今のやり方をデータで書いていない" in script
    assert '"model_spec.yaml"' not in script
    assert "spec, base_dir = model_file(spec, experiment)" in script
    assert "spec_errors(spec)" in script
    readme = (project / "README.md").read_text("utf-8")
    assert "今のやり方とは比べない" in readme and "TODO" not in readme
    assert "model_inputs" in readme

    # 書いてデータを入れた: 予測は k 期前, 回帰は列
    pytest.importorskip("ml_eda")
    path = daily_csv(tmp_path)
    lag = attach_data(example_problem("総菜の作る量"), path)
    lag = answer(lag, "action.current", {"type": "lag", "periods": 7})
    column = answer(lag, "information.unknown", ["customers"])
    column = answer(column, "ml.parts.0.task", "regression")
    column = answer(column, "action.current", {"type": "column",
                                               "column": "customers"})
    for name, problem in (("lag", lag), ("column", column)):
        make_project(problem, tmp_path / name)
        script = (tmp_path / name / "train.py").read_text("utf-8")
        tree = ast.parse(script)
        assert max(len(line) for line in script.splitlines()) <= 79
        names = {node.name for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
        assert {"current_values", "matched_current",
                "compare_baseline"} <= names
        assert "compare_baseline(trainer.model" in script
        # 一式づくりで止まっても比べの表が残るように, 一式の前に比べる
        assert script.index("compare_baseline(trainer.model") < (
            script.index("info = make_bundle(")
        )
        assert 'DATA_FILE = PROJECT_DIR / "data/sozai.csv"' in script
        readme = (tmp_path / name / "README.md").read_text("utf-8")
        assert "evaluations/baseline_valid.json" in readme
    assert "CURRENT = {'type': 'lag', 'periods': 7}" in (
        tmp_path / "lag" / "train.py"
    ).read_text("utf-8")

    # 評価の表に, 最良を選ぶ指標の向きを書く (分類は正解率で比べる)
    assert "BETTER = {'mae': 'lower'}" in (
        tmp_path / "lag" / "train.py"
    ).read_text("utf-8")
    assert "better=BETTER" in (tmp_path / "lag" / "train.py").read_text(
        "utf-8"
    )

    # 分類で, 今のやり方を列で書いてもデータが無ければ比べない
    churn = answer(example_problem("解約の引き止め"), "action.current",
                   {"type": "column", "column": "inquiries"})
    make_project(churn, tmp_path / "churn")
    script = (tmp_path / "churn" / "train.py").read_text("utf-8")
    assert "手元のデータを入れていない" in script
    assert "compare_baseline" not in script


def test_problem_origin(tmp_path: Path) -> None:
    """
    写した問題の設定が元の場所を絶対パスで持ち, 書き出し直しても残り,
    崩した値と自分を指す場所を受け付けないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 元のディレクトリを渡しても, 中の problem.yaml の絶対パス
    problem = example_problem("総菜の作る量")
    store = tmp_path / "problems" / "one"
    save_problem(problem, store / "problem.yaml")
    assert problem_origin(problem) is None
    project = tmp_path / "project"
    make_project(problem, project, origin=store)
    copied = load_problem(project)
    original = str((store / "problem.yaml").resolve())
    assert problem_origin(copied) == original
    assert "origin" not in problem

    # 写しから書き出し直すと元を指したまま. 渡せば置き換え
    other = tmp_path / "other"
    make_project(copied, other)
    assert problem_origin(load_problem(other)) == original
    third = tmp_path / "third"
    make_project(copied, third, origin=tmp_path / "x.yaml")
    assert problem_origin(load_problem(third)) == str(
        (tmp_path / "x.yaml").resolve()
    )

    # まだ無いディレクトリを渡しても, 中の problem.yaml を指す
    fourth = tmp_path / "fourth"
    make_project(copied, fourth, origin=tmp_path / "later")
    assert problem_origin(load_problem(fourth)) == str(
        (tmp_path / "later" / "problem.yaml").resolve()
    )

    # 元の問題の設定に記録しても, origin は入らない
    assert "origin" not in record_project(problem, project)

    # 崩した値と相対パスは無いものとして扱う. 自分を指す場所と空は誤り
    assert problem_origin(dict(copied, origin="rel/problem.yaml")) is None
    assert problem_origin(dict(copied, origin=3)) is None
    with pytest.raises(ValueError, match="写しです"):
        make_project(problem, tmp_path / "self", origin=tmp_path / "self")
    with pytest.raises(ValueError, match="空です"):
        make_project(problem, tmp_path / "empty", origin=" ")


@pytest.mark.parametrize("key", ["source", "origin"])
@pytest.mark.parametrize("value", [123, True, ["a.csv"],
                                   {"path": "a.csv"}])
def test_data_location_type(tmp_path: Path, key: str, value: Any) -> None:
    """
    文字でないデータの場所が確かめの誤りになり, 分析と書き出しの関数が
    Python の型の例外で止まらないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    key: str
        data の場所の項目 (source か origin)
    value: Any
        文字でない場所

    Returns
    ----------
    None
    """
    # 手で書いた場所 (元の場所の誤りでは, 写しは文字)
    problem = example_problem("総菜の作る量")
    problem["data"] = {"source": str(tmp_path / "missing.csv"),
                       "loader": "csv"}
    problem["data"][key] = value
    found = [item for item in check(problem) if item["level"] == "error"]
    assert [item["id"] for item in found] == [f"data.{key}"]
    assert "場所を文字で" in found[0]["message"]

    # 分析は誤りを返し, 受け渡しと書き出しは確かめの誤りの文で止まる
    assert isinstance(inspect_key(problem), str)
    inspected = inspect_data(problem)
    assert [item["id"] for item in inspected["findings"]] == [f"data.{key}"]
    assert f"data.{key}" in [item["id"] for item in open_errors(problem)]
    message = project_error(problem)
    assert message is not None and "場所を文字で" in message
    for call in (lambda: handoff(problem), lambda: recipe_from(problem),
                 lambda: make_project(problem, tmp_path / "project")):
        with pytest.raises(ValueError, match="場所を文字で"):
            call()
    assert not (tmp_path / "project").exists()

    # ファイルの組の受け持ちの下書きも同じ
    items = answer(answer(problem, "ml.parts.0.target.kind", "mask"),
                   "ml.parts.0.output", "region")
    items = answer(items, "ml.parts.0.evaluation.metric", "iou")
    with pytest.raises(ValueError, match="場所を文字で"):
        recipe_from(items)


@pytest.mark.parametrize("value", ["/data/a.csv", ["a.csv"], 3, True])
def test_data_not_dict(tmp_path: Path, value: Any) -> None:
    """
    手元のデータの項目が辞書でない問題の設定が確かめの誤りになり, 分析と
    書き出しの関数が型の例外で止まらないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    value: Any
        辞書でない data

    Returns
    ----------
    None
    """
    # 形の誤り (他の確かめはしない)
    problem = example_problem("総菜の作る量")
    problem["data"] = value
    found = [item for item in check(problem) if item["level"] == "error"]
    assert [item["id"] for item in found] == ["data"]
    assert "辞書" in found[0]["message"]

    # 分析は誤りを返し, 受け渡しと書き出しは確かめの誤りの文で止まる
    assert isinstance(inspect_key(problem), str)
    assert [item["id"] for item in inspect_data(problem)["findings"]] == [
        "data"
    ]
    assert "data" in [item["id"] for item in open_errors(problem)]
    assert "辞書" in project_error(problem)
    for call in (lambda: handoff(problem), lambda: recipe_from(problem),
                 lambda: make_project(problem, tmp_path / "project")):
        with pytest.raises(ValueError, match="辞書"):
            call()
    assert not (tmp_path / "project").exists()


def test_public_copy_name_and_key(tmp_path: Path) -> None:
    """
    写しの名前の形 (COPY_NAME) と分析の結果の印 (inspect_key) が公開の口
    から取れ, 写しと結果の印に合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 公開の口は中の名前と同じもの
    import ml_quest

    assert ml_quest.COPY_NAME is data_module.COPY_NAME
    assert ml_quest.inspect_key is data_module.inspect_key
    assert {"COPY_NAME", "inspect_key"} <= set(ml_quest.__all__)

    # 写しのサブディレクトリの名前は COPY_NAME の形で, 結果の key と同じ印
    pytest.importorskip("ml_eda")
    source = tmp_path / "sales.csv"
    source.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    problem = attach_data(example_problem("総菜の作る量"), str(source),
                          copy_to=str(tmp_path / "copies"))
    copied = Path(problem["data"]["source"])
    assert ml_quest.COPY_NAME.fullmatch(copied.parent.name)
    assert inspect_data(problem)["key"] == ml_quest.inspect_key(problem)
