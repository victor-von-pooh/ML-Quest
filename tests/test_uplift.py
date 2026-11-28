from pathlib import Path

import pytest

from ml_quest import (
    MODEL_EXAMPLES, UPLIFT_ASSIGNED, answer, apply_recipe, attach_data,
    check, decision_text, describe, example_problem, handoff,
    model_example, questions, recipe_from
)

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の介入の記録の表の列
COLUMNS = ["customer_id", "age", "visits", "coupon", "purchased", "used"]


def uplift_problem(kind: str = "binary") -> dict:
    """
    見本の課題から, 今ある記録から効果を見積もる問題の設定を作る関数

    Parameters
    ----------
    kind: str = "binary"
        結果の種類 (binary か amount)

    Returns
    ----------
    problem: dict
        効果を見積もるで方策でなく, 今ある記録から集め, 列, 上位の割合,
        配る判断に答えた問題の設定 (データは入れていない)
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
               ("what", "クーポンで買う確率がどれだけ上がるか"),
               ("item", "顧客ごとの 1 行"), ("baseline", "全員に配る"),
               ("uplift.treatment", "coupon"),
               ("uplift.outcome", "purchased"),
               ("uplift.outcome_kind", kind),
               ("uplift.id", "customer_id"),
               ("uplift.columns", ["age", "visits"]), ("uplift.k", 0.3),
               ("evaluation.metric", "qini"),
               ("evaluation.link", "効果の順が当たるほど配る費用が実る")]
    for key, value in answers:
        problem = answer(problem, f"{PART}.{key}", value)
    problem = answer(problem, "action.decision",
                     {"type": "treat", "budget": 50})

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


def test_uplift_questions() -> None:
    """
    効果の見積もりで, 列, 上位の割合, 配る判断の問いが要り, 方策では要らない
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 効果の見積もりの問いは要り, 配る判断の型は treat だけ
    problem = uplift_problem()
    items = {item["id"]: item for item in questions(problem)}
    for name in ("treatment", "outcome", "outcome_kind", "id", "columns",
                 "k", "learner"):
        assert items[f"{PART}.uplift.{name}"]["applies"]
    learner = items[f"{PART}.uplift.learner"]
    assert not learner["required"]
    assert [c["value"] for c in learner["choices"]] == ["t_learner",
                                                        "s_learner"]
    assert items["information.columns"]["applies"]
    assert items["action.decision"]["applies"]
    assert [c["value"] for c in items["action.decision"]["choices"]] == [
        "treat"
    ]
    fields = items["action.decision"]["choices"][0]["fields"]
    assert [(f["key"], f["required"]) for f in fields] == [("budget", True)]
    assert not items["action.basis"]["applies"]
    assert not items["action.current"]["applies"]

    # 方策, 試して集める, 既製のモデルで問わないことの確認
    for key, value in (("autonomous", True), ("collect", "online"),
                       ("method", "pretrained")):
        other = answer(problem, f"{PART}.{key}", value)
        items = {item["id"]: item for item in questions(other)}
        assert not items[f"{PART}.uplift.treatment"]["applies"]
        assert not items["action.decision"]["applies"]

    # 列の予測の選択肢の配る判断の無いこと
    column = example_problem("解約の引き止め")
    items = {item["id"]: item for item in questions(column)}
    assert "treat" not in [c["value"]
                           for c in items["action.decision"]["choices"]]


def test_uplift_check() -> None:
    """
    効果の見積もりの答えが確かめを通り, 合わない答えが error になることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 0 と 1 も額も, 2 つの効果の決め方も通ることの確認
    assert errors_of(uplift_problem()) == []
    assert errors_of(uplift_problem("amount")) == []
    assert errors_of(answer(uplift_problem(), f"{PART}.uplift.learner",
                            "s_learner")) == []

    # 列と上位の割合と正解の出どころ
    cases = [
        ("uplift.columns", ["age", "coupon"], f"{PART}.uplift.treatment"),
        ("uplift.outcome", "coupon", f"{PART}.uplift.outcome"),
        ("uplift.k", 1.5, f"{PART}.uplift.k"),
        ("uplift.k", 0, f"{PART}.uplift.k"),
        ("target.source", "column", f"{PART}.target.source"),
        ("evaluation.metric", "reward", f"{PART}.evaluation.metric"),
    ]
    for key, value, place in cases:
        problem = answer(uplift_problem(), f"{PART}.{key}", value)
        assert place in errors_of(problem), (key, value)

    # 手で書いた知らない効果の決め方
    problem = uplift_problem()
    problem["ml"]["parts"][0]["uplift"]["learner"] = "x_learner"
    assert f"{PART}.uplift.learner" in errors_of(problem)

    # 列を答え直した後の古い名前
    problem = uplift_problem()
    problem["ml"]["parts"][0]["uplift"]["id"] = "missing"
    assert f"{PART}.uplift.id" in errors_of(problem)

    # 行の id の列の名前の決まりと, 行の位置の row_id
    for name in UPLIFT_ASSIGNED:
        problem = answer(uplift_problem(), "information.columns",
                         [*COLUMNS, name])
        problem = answer(problem, f"{PART}.uplift.id", name)
        assert f"{PART}.uplift.id" in errors_of(problem)
    problem = answer(uplift_problem(), "information.columns",
                     [*COLUMNS, "row_id"])
    problem = answer(problem, f"{PART}.uplift.id", None)
    assert f"{PART}.uplift.id" in errors_of(problem)

    # 配る数の上限の型と, 他の型と書けない項目の誤り
    for decision in ({"type": "treat", "budget": 0},
                     {"type": "treat", "budget": 2.5},
                     {"type": "treat", "budget": "slots"},
                     {"type": "treat"},
                     {"type": "select", "value": 1, "cost": 1,
                      "effect": 0.1}):
        problem = answer(uplift_problem(), "action.decision", decision)
        assert "action.decision" in errors_of(problem), decision
    problem = uplift_problem()
    problem["action"]["decision"]["capacity"] = 3
    assert "action.decision" in errors_of(problem)

    # 列の予測の配る判断の誤り
    column = answer(example_problem("解約の引き止め"), "action.decision",
                    {"type": "treat", "budget": 10})
    assert "action.decision" in errors_of(column)


def test_uplift_handoff() -> None:
    """
    効果の見積もりの受け渡しに, 列, 配る判断, 雛形が同じ形で入ることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # データの分析は介入の記録の表のレシピの値
    stages = handoff(uplift_problem())
    data = stages["data"]
    assert (data["target"], data["task"], data["split"]) == (
        "purchased", "uplift", "stratified"
    )
    assert data["known"] == ["age", "visits"]
    assert data["drop"] == ["used"]
    assert (data["id"], data["treatment"], data["outcome_kind"],
            data["unlabeled"]) == ("customer_id", "coupon", "binary", "drop")

    # モデルは頭ごとに 1 を出す T-learner の雛形, 学習は Qini を最大に
    assert stages["model"]["outputs"] == 1
    assert stages["model"]["example"] == "tabular_uplift"
    assert model_example(uplift_problem()) == "tabular_uplift"
    assert (stages["training"]["monitor"], stages["training"]["mode"]) == (
        "valid/qini", "max"
    )

    # 推論の配る判断 (守る指標と結論は無し)
    assert stages["inference"]["decision"] == {"type": "treat", "budget": 50}
    assert stages["inference"]["guard"] is None
    assert stages["inference"]["conclusion"] == {}
    assert stages["uplift"] == {
        "treatment": "coupon", "outcome": "purchased",
        "outcome_kind": "binary", "id": "customer_id",
        "columns": ["age", "visits"], "k": 0.3, "learner": "t_learner",
    }
    assert stages["requires"] == []

    # 小数の整数の上限は整数, 要約の文は配る数の上限
    problem = answer(uplift_problem(), "action.decision",
                     {"type": "treat", "budget": 50.0})
    assert handoff(problem)["inference"]["decision"]["budget"] == 50
    assert decision_text({"type": "treat", "budget": 50}) == (
        "意思決定: 効果の上位から配る (配る数の上限 50)"
    )
    assert "意思決定: 効果の上位から配る (配る数の上限 50)" in describe(
        uplift_problem()
    )

    # 方策と一から学習しない場合の雛形無し
    policy = answer(uplift_problem(), f"{PART}.autonomous", True)
    assert model_example(policy) is None
    tuned = answer(uplift_problem(), f"{PART}.method", "finetune")
    assert model_example(tuned) is None
    assert handoff(tuned)["model"]["example"] is None
    assert MODEL_EXAMPLES[("effect", None)] == "tabular_uplift"

    # S-learner の雛形と効果の決め方 (データの分析の値は同じ)
    single = answer(uplift_problem(), f"{PART}.uplift.learner", "s_learner")
    stages = handoff(single)
    assert stages["uplift"]["learner"] == "s_learner"
    assert stages["model"]["example"] == "tabular_uplift_s"
    assert stages["model"]["outputs"] == 1
    assert stages["data"] == handoff(uplift_problem())["data"]
    assert MODEL_EXAMPLES[("effect", "s_learner")] == "tabular_uplift_s"
    explicit = answer(uplift_problem(), f"{PART}.uplift.learner",
                      "t_learner")
    assert handoff(explicit) == handoff(uplift_problem())


def test_uplift_recipe() -> None:
    """
    効果の見積もりのレシピの雛形と書き戻しを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 雛形は介入の記録の表 (task: uplift) の処置で層を作る分割
    start = recipe_from(uplift_problem())
    recipe = start["recipe"]
    assert recipe == {
        "format_version": 1, "task": "uplift", "target": "purchased",
        "treatment": "coupon", "outcome": "binary", "id": "customer_id",
        "steps": [{"op": "drop_columns", "columns": ["used"]},
                  {"op": "standardize", "columns": "auto"}],
        "split": {"method": "stratified", "valid": 0.2, "test": 0.2,
                  "seed": 0, "unlabeled": "drop"},
    }
    assert any("one_hot" in note for note in start["notes"])

    # 書き戻しの往復と, 答えと違う列と分割の食い違い
    result = apply_recipe(uplift_problem(), recipe)
    assert result["conflicts"] == [] and result["choices"] == []
    assert recipe_from(result["problem"])["recipe"] == recipe
    for key, value in (("target", "used"), ("treatment", "used"),
                       ("outcome", "amount"), ("id", None),
                       ("task", "classification")):
        changed = {**recipe, key: value}
        conflicts = apply_recipe(uplift_problem(), changed)["conflicts"]
        assert [c["level"] for c in conflicts] == ["error"], key
    split = {**recipe, "split": {**recipe["split"], "method": "random"}}
    assert apply_recipe(uplift_problem(), split)["conflicts"]
    empty = {**recipe, "split": {**recipe["split"], "valid": 0}}
    levels = [c["level"]
              for c in apply_recipe(uplift_problem(), empty)["conflicts"]]
    assert levels == ["warning"]


def test_uplift_recipe_matches_ml_eda(tmp_path: Path) -> None:
    """
    レシピの雛形を ML-EDA が読み, 介入の記録の表を書き出せることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 無作為に配った小さな表 (ML-EDA が無ければ飛ばす)
    eda = pytest.importorskip("ml_eda")
    pd = pytest.importorskip("pandas")
    rows = [{"customer_id": f"c{i:03d}", "age": 20 + i % 40,
             "visits": i % 7, "coupon": i % 2,
             "purchased": int((i * 7) % 5 == 0), "used": i % 2}
            for i in range(120)]
    path = tmp_path / "log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    problem = attach_data(uplift_problem(), str(path))
    start = recipe_from(problem)
    data = eda.load_data(start["load"]["path"], loader=start["load"]["loader"],
                         **(start["load"]["options"] or {}))
    eda.export_dataset(data, start["recipe"], tmp_path / "dataset")
    dataset = eda.load_dataset(tmp_path / "dataset")
    assert dataset.info["task"] == "uplift"
    assert dataset.info["uplift"]["treatment"] == "coupon"
    assert list(dataset.features) == ["age", "visits"]
    for split in ("train", "valid", "test"):
        assert set(dataset.treatment(split).tolist()) == {0, 1}
