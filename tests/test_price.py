from pathlib import Path

import pytest

from ml_quest import (
    MODEL_EXAMPLES, PRICE_ASSIGNED, VARIED, answer, apply_recipe,
    attach_data, check, decision_text, describe, example_problem, handoff,
    model_example, questions, recipe_from
)

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の値段を変えた記録の表の列
COLUMNS = ["record_id", "product_id", "category", "weekday", "price",
           "units", "unit_cost", "leftover"]

# 値段の判断
DECISION = {"type": "price", "candidates": [-0.1, 0, 0.1, 0.2],
            "max_change": 0.1}


def price_problem() -> dict:
    """
    見本の課題から, 試しに値段を変えた記録から値段を選ぶ問題の設定を作る
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        効果を見積もるで方策でなく, 今ある記録で値段を変え, 列と値段の判断
        に答えた問題の設定 (データは入れていない)
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("解約の引き止め")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = answer(problem, "information.columns", COLUMNS)
    answers = [("role", "effect"), ("unit", "row"), ("output", "effect"),
               ("target.source", "intervention"), ("method", "scratch"),
               ("autonomous", False), ("collect", "log"),
               ("varied", "price"),
               ("what", "値段を変えたときの売れた数"),
               ("item", "商品と日ごとの 1 行"),
               ("baseline", "今の値段のまま"),
               ("price.price", "price"), ("price.units", "units"),
               ("price.cost", "unit_cost"),
               ("price.product", "product_id"),
               ("price.id", "record_id"),
               ("price.columns", ["category", "weekday"]),
               ("evaluation.metric", "mae"),
               ("evaluation.link", "売れた数の外れが小さいほど選ぶ値段の"
                "利益の見込みが当たる")]
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


def test_price_questions() -> None:
    """
    値段を変える判断で, 列と値段の判断の問いが要り, 効果の見積もりの問いが
    要らないことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 値段の列の問いと, 処置の列と上位の割合の問いの有無
    problem = price_problem()
    items = {item["id"]: item for item in questions(problem)}
    for name in ("price", "units", "cost", "product", "id", "columns"):
        assert items[f"{PART}.price.{name}"]["applies"]
    assert not items[f"{PART}.uplift.treatment"]["applies"]
    assert not items[f"{PART}.uplift.k"]["applies"]
    assert items[f"{PART}.varied"]["applies"]
    assert [c["value"] for c in items[f"{PART}.varied"]["choices"]] == [
        item["value"] for item in VARIED
    ]

    # 判断の型は値段の判断だけで, 候補は数の並びの欄
    choices = items["action.decision"]["choices"]
    assert [c["value"] for c in choices] == ["price"]
    fields = choices[0]["fields"]
    assert [(f["key"], f["required"]) for f in fields] == [
        ("candidates", True), ("max_change", True)
    ]
    assert fields[0]["list"] is True

    # 指標は売れた数の誤差と損失 (Qini は無し)
    metrics = [c["value"]
               for c in items[f"{PART}.evaluation.metric"]["choices"]]
    assert "mae" in metrics and "loss" in metrics
    assert "qini" not in metrics

    # 配ったかを答えない場合と, 方策の場合の問いの有無
    treat = answer(problem, f"{PART}.varied", None)
    items = {item["id"]: item for item in questions(treat)}
    assert items[f"{PART}.uplift.treatment"]["applies"]
    assert not items[f"{PART}.price.price"]["applies"]
    assert [c["value"] for c in items["action.decision"]["choices"]] == [
        "treat"
    ]
    policy = answer(problem, f"{PART}.autonomous", True)
    items = {item["id"]: item for item in questions(policy)}
    assert not items[f"{PART}.varied"]["applies"]
    assert not items[f"{PART}.price.price"]["applies"]


def test_price_answer() -> None:
    """
    値段の候補の答えが数の並びになり, 形の誤りが ValueError になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # カンマで区切った文字と並び
    problem = answer(price_problem(), "action.decision",
                     {"type": "price", "candidates": "-0.1, 0, 0.1",
                      "max_change": "0.1"})
    assert problem["action"]["decision"] == {
        "type": "price", "candidates": [-0.1, 0, 0.1], "max_change": 0.1
    }
    for candidates in ("a, 0", [0, None], 3, [0, "inf"]):
        with pytest.raises(ValueError):
            answer(price_problem(), "action.decision",
                   {"type": "price", "candidates": candidates,
                    "max_change": 0.1})


def test_price_check() -> None:
    """
    値段を変える判断の答えが確かめを通り, 合わない答えが error になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 原価と商品の id が無くても通ることの確認
    assert errors_of(price_problem()) == []
    bare = price_problem()
    for name in ("cost", "product", "id"):
        bare = answer(bare, f"{PART}.price.{name}", None)
    assert errors_of(bare) == []

    # 列, 正解の出どころと指標
    cases = [
        ("price.columns", ["category", "price"], f"{PART}.price.price"),
        ("price.units", "price", f"{PART}.price.units"),
        ("price.cost", "units", f"{PART}.price.cost"),
        ("target.source", "column", f"{PART}.target.source"),
        ("evaluation.metric", "qini", f"{PART}.evaluation.metric"),
        ("evaluation.metric", "reward", f"{PART}.evaluation.metric"),
    ]
    for key, value, place in cases:
        problem = answer(price_problem(), f"{PART}.{key}", value)
        assert place in errors_of(problem), (key, value)

    # 商品の id と行の id の列の名前の決まりと, 行の位置と元の値段の列
    for name in PRICE_ASSIGNED:
        if name == "price":
            continue
        problem = answer(price_problem(), "information.columns",
                         [*COLUMNS, name])
        problem = answer(problem, f"{PART}.price.product", name)
        assert f"{PART}.price.product" in errors_of(problem), name
    problem = answer(price_problem(), "information.columns",
                     [*COLUMNS, "row_id"])
    problem = answer(problem, f"{PART}.price.id", None)
    assert f"{PART}.price.id" in errors_of(problem)
    problem = answer(price_problem(), "information.columns",
                     [*COLUMNS, "price_raw"])
    assert "information.columns" in errors_of(problem)

    # 候補と上限の形と, 他の型の誤り
    for decision in ({**DECISION, "candidates": [0.1, 0.2]},
                     {**DECISION, "candidates": [0, 0.1, 0.1]},
                     {**DECISION, "candidates": [-1, 0]},
                     {**DECISION, "max_change": -0.1},
                     {"type": "price", "candidates": [0, 0.1]},
                     {"type": "treat", "budget": 10}):
        problem = answer(price_problem(), "action.decision", decision)
        assert "action.decision" in errors_of(problem), decision
    problem = price_problem()
    problem["ml"]["parts"][0]["price"]["cost"] = "missing"
    assert f"{PART}.price.cost" in errors_of(problem)
    problem = price_problem()
    problem["action"]["decision"]["budget"] = 3
    assert "action.decision" in errors_of(problem)

    # 今の値段の他が全て上限の外なら warning
    problem = answer(price_problem(), "action.decision",
                     {**DECISION, "candidates": [0, 0.2], "max_change": 0.1})
    found = [item for item in check(problem)
             if item["id"] == "action.decision"]
    assert [item["level"] for item in found] == ["warning"]

    # 列の予測と効果の見積もりの値段の判断の誤り
    column = answer(example_problem("解約の引き止め"), "action.decision",
                    DECISION)
    assert "action.decision" in errors_of(column)
    treat = answer(price_problem(), f"{PART}.varied", "treat")
    assert "action.decision" in errors_of(treat)


def test_price_handoff() -> None:
    """
    値段を変える判断の受け渡しに, 列, 値段の判断, 雛形が同じ形で入ることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # データの分析は値段を変えた記録の表のレシピの値 (値段の列は最後)
    stages = handoff(price_problem())
    data = stages["data"]
    assert (data["target"], data["task"], data["split"]) == (
        "units", "price", "stratified"
    )
    assert data["known"] == ["category", "weekday", "price"]
    assert data["drop"] == ["leftover"]
    assert (data["id"], data["price"], data["cost"], data["product"],
            data["unlabeled"]) == ("record_id", "price", "unit_cost",
                                   "product_id", "drop")

    # 売れた数の 1 を出す雛形と, 誤差を最小にする学習
    assert stages["model"]["outputs"] == 1
    assert stages["model"]["example"] == "tabular_price"
    assert model_example(price_problem()) == "tabular_price"
    assert MODEL_EXAMPLES[("effect", "price")] == "tabular_price"
    assert (stages["training"]["monitor"], stages["training"]["mode"]) == (
        "valid/mae", "min"
    )

    # 推論の値段の判断 (守る指標と結論は無し)
    assert stages["inference"]["decision"] == DECISION
    assert stages["inference"]["guard"] is None
    assert stages["inference"]["conclusion"] == {}
    assert stages["price"] == {
        "price": "price", "units": "units", "cost": "unit_cost",
        "product": "product_id", "id": "record_id",
        "columns": ["category", "weekday"],
    }
    assert stages["uplift"] is None
    assert stages["requires"] == []

    # 要約の文
    text = ("意思決定: 利益が最も大きい値段を選ぶ (値段の候補 [-0.1, 0, 0.1, "
            "0.2], 値段を変える幅の上限 0.1)")
    assert decision_text(DECISION) == text
    assert text in describe(price_problem())

    # 一から学習しない場合の雛形無し
    tuned = answer(price_problem(), f"{PART}.method", "finetune")
    assert model_example(tuned) is None


def test_price_recipe() -> None:
    """
    値段を変える判断のレシピの雛形と書き戻しを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 雛形は値段を変えた記録の表 (task: price) の値段の段で層を作る分割
    start = recipe_from(price_problem())
    recipe = start["recipe"]
    assert recipe == {
        "format_version": 1, "task": "price", "target": "units",
        "price": "price", "cost": "unit_cost", "product": "product_id",
        "id": "record_id",
        "steps": [{"op": "drop_columns", "columns": ["leftover"]},
                  {"op": "standardize", "columns": "auto"}],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.2,
                  "seed": 0, "unlabeled": "drop"},
    }
    assert any("one_hot" in note for note in start["notes"])

    # 書き戻しの往復と, 答えと違う列と分割の食い違い
    result = apply_recipe(price_problem(), recipe)
    assert result["conflicts"] == [] and result["choices"] == []
    assert recipe_from(result["problem"])["recipe"] == recipe
    for key, value in (("target", "leftover"), ("price", "leftover"),
                       ("cost", None), ("product", None), ("id", None),
                       ("task", "uplift")):
        changed = {**recipe, key: value}
        conflicts = apply_recipe(price_problem(), changed)["conflicts"]
        assert [c["level"] for c in conflicts] == ["error"], key
    split = {**recipe, "split": {**recipe["split"], "method": "random"}}
    assert apply_recipe(price_problem(), split)["conflicts"]
    empty = {**recipe, "split": {**recipe["split"], "valid": 0}}
    levels = [c["level"]
              for c in apply_recipe(price_problem(), empty)["conflicts"]]
    assert levels == ["warning"]


def test_price_recipe_matches_ml_eda(tmp_path: Path) -> None:
    """
    レシピの雛形を ML-EDA が読み, 値段を変えた記録の表を書き出せることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 値段を無作為に変えた小さな表 (ML-EDA が無ければ飛ばす)
    eda = pytest.importorskip("ml_eda")
    pd = pytest.importorskip("pandas")
    rows = [{"record_id": f"r{i:03d}", "product_id": f"p{i % 6}",
             "category": i % 6 % 2, "weekday": i % 7,
             "price": 100 * (1 + 0.1 * (i % 5 - 2)),
             "units": 20 - i % 5, "unit_cost": 60.0, "leftover": i % 3}
            for i in range(150)]
    path = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    problem = attach_data(price_problem(), str(path))
    start = recipe_from(problem)
    data = eda.load_data(start["load"]["path"], loader=start["load"]["loader"],
                         **(start["load"]["options"] or {}))
    eda.export_dataset(data, start["recipe"], tmp_path / "dataset")
    dataset = eda.load_dataset(tmp_path / "dataset")
    assert dataset.info["task"] == "price"
    assert list(dataset.features) == ["category", "weekday", "price"]
    assert dataset.cost("train") is not None
    assert len(dataset.price("train")) == len(dataset.arrays("train")[0])
