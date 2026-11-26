from pathlib import Path

import pytest

from ml_quest import (
    DEMAND_ASSIGNED, DEMAND_LEVELS, MODEL_EXAMPLES, answer, apply_recipe,
    attach_data, check, decision_text, describe, example_problem, handoff,
    model_example, questions, recipe_from
)

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の需要の記録の表の列
COLUMNS = ["record_id", "product_id", "date", "category", "units", "stock",
           "waste"]

# 発注の判断 (水準 τ は 3 / (3 + 1) = 0.75)
DECISION = {"type": "order", "shortage_cost": 3, "holding_cost": 1,
            "max_order": 60}


def demand_problem() -> dict:
    """
    見本の課題から, 商品ごとの日ごとの売れた数の記録から発注の数を決める
    問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        予測するで正解の種類が先の期間の需要の, 列と発注の判断に答えた
        問題の設定 (データは入れていない)
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("解約の引き止め")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = answer(problem, "information.columns", COLUMNS)
    answers = [("role", "predict"), ("unit", "row"),
               ("output", "distribution"), ("target.kind", "demand"),
               ("target.source", "column"), ("method", "scratch"),
               ("what", "先の 7 日の売れる数"),
               ("item", "商品と発注を決める日ごとの 1 行"),
               ("baseline", "前の 7 日と同じ数を発注する"),
               ("demand.product", "product_id"), ("demand.date", "date"),
               ("demand.units", "units"), ("demand.stock", "stock"),
               ("demand.id", "record_id"),
               ("demand.columns", ["category"]),
               ("evaluation.metric", "loss"),
               ("evaluation.link", "分位点の損失が小さいほど発注の数の損の"
                "見込みが当たる")]
    for key, value in answers:
        problem = answer(problem, f"{PART}.{key}", value)
    problem = answer(problem, "action.decision", DECISION)

    return problem


def errors_of(problem: dict) -> list[str]:
    """
    確かめの error の項目の場所を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    ids: list[str]
        error の項目の場所
    """
    # error だけ
    ids = [item["id"] for item in check(problem) if item["level"] == "error"]

    return ids


def test_demand_questions() -> None:
    """
    時間をまたぐ量の判断で, 列, 期間, 水準と発注の判断の問いが要り, 列の
    予測の問いが要らないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 需要の記録の表の列と期間と水準の問いと, 列の予測の問いの有無
    problem = demand_problem()
    items = {item["id"]: item for item in questions(problem)}
    for name in ("product", "date", "units", "stock", "id", "columns",
                 "horizon", "levels"):
        assert items[f"{PART}.demand.{name}"]["applies"], name
    assert items[f"{PART}.demand.levels"]["kind"] == "numbers"
    for name in (f"{PART}.task", f"{PART}.target.column",
                 "information.known", f"{PART}.price.price",
                 f"{PART}.anomaly.columns"):
        assert not items[name]["applies"], name
    kinds = [c["value"] for c in items[f"{PART}.target.kind"]["choices"]]
    assert "demand" in kinds

    # 判断の型は発注の判断だけで, 2 つの損と上限の欄
    choices = items["action.decision"]["choices"]
    assert [c["value"] for c in choices] == ["order"]
    assert [(f["key"], f["required"]) for f in choices[0]["fields"]] == [
        ("shortage_cost", True), ("holding_cost", True), ("max_order", True)
    ]

    # 指標は 0.5 の分位点の誤差と損失
    metrics = [c["value"]
               for c in items[f"{PART}.evaluation.metric"]["choices"]]
    assert "mae" in metrics and "loss" in metrics
    assert "accuracy" not in metrics and "qini" not in metrics

    # 列の予測の判断の型に発注の判断が出ないこと
    column = example_problem("解約の引き止め")
    items = {item["id"]: item for item in questions(column)}
    values = [c["value"] for c in items["action.decision"]["choices"]]
    assert "order" not in values and "select" in values


def test_demand_answer() -> None:
    """
    学ぶ水準の答えが数の並びになり, 形の誤りが ValueError になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # カンマで区切った文字と並び
    problem = answer(demand_problem(), f"{PART}.demand.levels",
                     "0.1, 0.5, 0.9")
    assert problem["ml"]["parts"][0]["demand"]["levels"] == [0.1, 0.5, 0.9]
    for levels in ("a, 0.5", [0.5, None], 3):
        with pytest.raises(ValueError):
            answer(demand_problem(), f"{PART}.demand.levels", levels)


def test_demand_check() -> None:
    """
    時間をまたぐ量の判断の答えが確かめを通り, 合わない答えが error になる
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 在庫の数, 行の id, 商品の特徴が無くても通ることの確認
    assert errors_of(demand_problem()) == []
    bare = demand_problem()
    for name in ("stock", "id", "columns"):
        bare = answer(bare, f"{PART}.demand.{name}", None)
    assert errors_of(bare) == []

    # 列, 正解の出どころ, 出力の形, 期間, 水準と指標
    cases = [
        ("demand.columns", ["category", "units"], f"{PART}.demand.units"),
        ("demand.date", "product_id", f"{PART}.demand.date"),
        ("demand.stock", "units", f"{PART}.demand.stock"),
        ("target.source", "few", f"{PART}.target.source"),
        ("output", "value", f"{PART}.output"),
        ("demand.horizon", 0, f"{PART}.demand.horizon"),
        ("demand.horizon", 2.5, f"{PART}.demand.horizon"),
        ("demand.levels", [0.1, 1], f"{PART}.demand.levels"),
        ("demand.levels", [0.5, 0.5], f"{PART}.demand.levels"),
        ("evaluation.metric", "accuracy", f"{PART}.evaluation.metric"),
    ]
    for key, value, place in cases:
        problem = answer(demand_problem(), f"{PART}.{key}", value)
        assert place in errors_of(problem), (key, value)

    # 商品の id と日付の列の名前の決まりと, ML-EDA の足す列
    for name in DEMAND_ASSIGNED:
        problem = answer(demand_problem(), "information.columns",
                         [*COLUMNS, name])
        problem = answer(problem, f"{PART}.demand.product", name)
        assert f"{PART}.demand.product" in errors_of(problem), name
    problem = answer(demand_problem(), "information.columns",
                     [*COLUMNS, "row_id"])
    problem = answer(problem, f"{PART}.demand.id", None)
    assert f"{PART}.demand.id" in errors_of(problem)
    for name in ("demand", "demand_previous"):
        problem = answer(demand_problem(), "information.columns",
                         [*COLUMNS, name])
        assert "information.columns" in errors_of(problem), name

    # 損と上限の形と, 他の型の誤り
    for decision in ({**DECISION, "shortage_cost": 0},
                     {**DECISION, "holding_cost": -1},
                     {**DECISION, "holding_cost": "waste"},
                     {**DECISION, "max_order": -1},
                     {"type": "order", "shortage_cost": 3,
                      "holding_cost": 1},
                     {"type": "quantity", "under_cost": 3,
                      "over_cost": 1}):
        problem = answer(demand_problem(), "action.decision", decision)
        assert "action.decision" in errors_of(problem), decision
    problem = demand_problem()
    problem["action"]["decision"]["budget"] = 3
    assert "action.decision" in errors_of(problem)

    # 上限が 1 より小さければ warning
    problem = answer(demand_problem(), "action.decision",
                     {**DECISION, "max_order": 0})
    found = [item for item in check(problem)
             if item["id"] == "action.decision"]
    assert [item["level"] for item in found] == ["warning"]

    # 列の予測の発注の判断の誤り
    column = answer(example_problem("解約の引き止め"), "action.decision",
                    DECISION)
    assert "action.decision" in errors_of(column)


def test_demand_handoff() -> None:
    """
    時間をまたぐ量の判断の受け渡しに, 列, 期間, 学ぶ水準, 発注の判断, 雛形
    が同じ形で入ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # データの分析は需要の記録の表のレシピの値
    stages = handoff(demand_problem())
    data = stages["data"]
    assert (data["target"], data["task"], data["split"]) == (
        "units", "demand", "time"
    )
    assert data["known"] == ["category"]
    assert data["drop"] == ["waste"]
    assert (data["id"], data["product"], data["date"], data["stock"],
            data["horizon"], data["unlabeled"]) == (
        "record_id", "product_id", "date", "stock", 7, "drop"
    )

    # 水準の数を出す雛形と, 分位点の損失を最小にする学習
    assert stages["model"]["example"] == "tabular_demand"
    assert model_example(demand_problem()) == "tabular_demand"
    assert MODEL_EXAMPLES[("predict", "demand")] == "tabular_demand"
    assert stages["model"]["outputs"] == len(DEMAND_LEVELS)
    assert (stages["training"]["monitor"], stages["training"]["mode"]) == (
        "valid/loss", "min"
    )

    # 推論の発注の判断 (守る指標と結論は無し)
    assert stages["inference"]["decision"] == DECISION
    assert stages["inference"]["guard"] is None
    assert stages["inference"]["conclusion"] == {}
    assert stages["demand"] == {
        "product": "product_id", "date": "date", "units": "units",
        "stock": "stock", "id": "record_id", "columns": ["category"],
        "horizon": 7, "levels": [0.1, 0.25, 0.5, 0.75, 0.9],
    }
    assert stages["price"] is None and stages["anomaly"] is None
    assert stages["requires"] == []

    # 要約の文
    text = ("意思決定: 損の比で発注の数を決める (品切れの損 3, 売れ残りの損 "
            "1, 発注の数の上限 60)")
    assert decision_text(DECISION) == text
    assert text in describe(demand_problem())

    # 一から学習しない場合の雛形無し
    tuned = answer(demand_problem(), f"{PART}.method", "finetune")
    assert model_example(tuned) is None


def test_demand_levels() -> None:
    """
    学ぶ水準に, 答えた水準と 0.5 と水準 τ が入り, 同じ水準が 1 つにまとまる
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 答えた水準に 0.5 と τ (2 / (2 + 3) = 0.4) を足す
    problem = answer(demand_problem(), f"{PART}.demand.levels", [0.9, 0.1])
    problem = answer(problem, "action.decision",
                     {**DECISION, "shortage_cost": 2, "holding_cost": 3})
    stages = handoff(problem)
    assert stages["demand"]["levels"] == [0.1, 0.4, 0.5, 0.9]
    assert stages["model"]["outputs"] == 4

    # 品切れの損と売れ残りの損が同じなら τ = 0.5 で, 0.5 は 1 つ
    problem = answer(problem, "action.decision",
                     {**DECISION, "shortage_cost": 2.5, "holding_cost": 2.5})
    assert handoff(problem)["demand"]["levels"] == [0.1, 0.5, 0.9]

    # 10 進の表記どおりの τ (0.1 / (0.1 + 0.2) は 1/3) と, 先の期間の日数
    problem = answer(problem, "action.decision",
                     {**DECISION, "shortage_cost": 0.1, "holding_cost": 0.2})
    problem = answer(problem, f"{PART}.demand.horizon", 14)
    stages = handoff(problem)
    assert stages["demand"]["levels"] == [0.1, 1 / 3, 0.5, 0.9]
    assert stages["demand"]["horizon"] == 14
    assert stages["data"]["horizon"] == 14


def test_demand_recipe() -> None:
    """
    時間をまたぐ量の判断のレシピの雛形と書き戻しを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 雛形は需要の記録の表 (task: demand) の発注を決める日の順の分割
    start = recipe_from(demand_problem())
    recipe = start["recipe"]
    assert recipe == {
        "format_version": 1, "task": "demand", "target": "units",
        "product": "product_id", "date": "date", "stock": "stock",
        "horizon": 7, "id": "record_id",
        "steps": [{"op": "drop_columns", "columns": ["waste"]},
                  {"op": "lag", "columns": ["units"], "lags": [1, 7],
                   "windows": [7], "by": ["product_id"], "time": "date",
                   "freq": "1D"},
                  {"op": "datetime_parts", "columns": ["date"],
                   "parts": ["weekday"]},
                  {"op": "standardize", "columns": "auto"}],
        "split": {"method": "time", "valid": 0.2, "test": 0.2,
                  "unlabeled": "drop"},
    }
    assert any("one_hot" in note for note in start["notes"])

    # 書き戻しの往復と, 答えと違う列, 期間と分割の食い違い
    result = apply_recipe(demand_problem(), recipe)
    assert result["conflicts"] == [] and result["choices"] == []
    assert recipe_from(result["problem"])["recipe"] == recipe
    for key, value in (("target", "waste"), ("product", "category"),
                       ("date", "category"), ("stock", None), ("id", None),
                       ("horizon", 14), ("task", "price")):
        changed = {**recipe, key: value}
        conflicts = apply_recipe(demand_problem(), changed)["conflicts"]
        assert [c["level"] for c in conflicts] == ["error"], key
    split = {**recipe, "split": {**recipe["split"], "method": "random"}}
    assert apply_recipe(demand_problem(), split)["conflicts"]
    empty = {**recipe, "split": {**recipe["split"], "valid": 0}}
    levels = [c["level"]
              for c in apply_recipe(demand_problem(), empty)["conflicts"]]
    assert levels == ["warning"]

    # 1 日先なら過去の売れた数は 1 日前だけ
    day = answer(demand_problem(), f"{PART}.demand.horizon", 1)
    lag = recipe_from(day)["recipe"]["steps"][1]
    assert (lag["lags"], lag["windows"]) == ([1], [1])


def test_demand_recipe_matches_ml_eda(tmp_path: Path) -> None:
    """
    レシピの雛形を ML-EDA が読み, 需要の記録の表を書き出せることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 商品と曜日で売れた数の違う小さな表 (ML-EDA が無ければ飛ばす)
    eda = pytest.importorskip("ml_eda")
    pd = pytest.importorskip("pandas")
    days = pd.date_range("2026-01-01", periods=42)
    rows = [{"record_id": f"r{p}_{d:02d}", "product_id": f"p{p}",
             "date": day.strftime("%Y-%m-%d"), "category": p % 2,
             "units": 2 + p + d % 7, "stock": d % 4, "waste": d % 3}
            for p in range(3) for d, day in enumerate(days)]
    path = tmp_path / "daily.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    problem = attach_data(demand_problem(), str(path))
    start = recipe_from(problem)
    data = eda.load_data(start["load"]["path"], loader=start["load"]["loader"],
                         **(start["load"]["options"] or {}))
    eda.export_dataset(data, start["recipe"], tmp_path / "dataset")
    dataset = eda.load_dataset(tmp_path / "dataset")
    assert dataset.info["task"] == "demand"
    assert list(dataset.features) == ["category", "units_lag1",
                                      "units_lag7", "units_mean7",
                                      "date_weekday"]
    assert dataset.stock("train") is not None
    assert len(dataset.previous("train")) == len(dataset.arrays("train")[0])
