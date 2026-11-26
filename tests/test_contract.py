from typing import Optional

import pytest

from ml_quest import (
    SKELETON_TYPES, answer, check_model_contract, example_problem,
    model_contract, model_example, new_problem, skeleton_type
)

# 受け持ちの場所
PART = "ml.parts.0"

# 骨組みの型ごとの, 列の予測の見本から変える答え
ANSWERS = {
    "column": {},
    "items": {f"{PART}.target.kind": "mask"},
    "anomaly": {f"{PART}.target.kind": "anomaly",
                f"{PART}.anomaly.score": "autoencoder"},
    "demand": {f"{PART}.target.kind": "demand"},
    "generate": {f"{PART}.role": "generate", f"{PART}.target.kind": "image"},
    "table": {f"{PART}.role": "generate", f"{PART}.target.kind": "table"},
    "rank": {f"{PART}.role": "rank"},
    "group": {f"{PART}.role": "group"},
    "search": {f"{PART}.role": "rank", f"{PART}.method": "pretrained"},
    "group_ready": {f"{PART}.role": "group", f"{PART}.method": "pretrained"},
    "ready": {f"{PART}.method": "pretrained", f"{PART}.target.kind": "mask"},
    "policy": {f"{PART}.role": "effect", f"{PART}.autonomous": True},
    "uplift": {f"{PART}.role": "effect", f"{PART}.collect": "log"},
    "price": {f"{PART}.role": "effect", f"{PART}.collect": "log",
              f"{PART}.varied": "price"},
}

# 骨組みの型ごとの, 約束の入力と出力の名前 (None は約束無し)
NAMES = {
    "column": None,
    "items": None,
    "search": None,
    "group_ready": None,
    "ready": None,
    "anomaly": (["x"], ["recon"]),
    "rank": (["user", "item"], ["score"]),
    "group": (["x"], ["embedding", "similarity"]),
    "uplift": (["x"], ["treated", "control"]),
    "price": (["x"], ["units"]),
    "demand": (["x"], ["quantiles"]),
    "table": ([None, None], [None]),
    "generate": ([None, None], [None]),
    "policy": ([None], [None]),
}

# 約束に合う定義 (骨組みの型ごと. 入力の最後の大きさは WIDTHS)
SPECS = {
    "anomaly": {"inputs": {"x": {"shape": [8, 5]}},
                "outputs": {"recon": "dec"}},
    "rank": {"inputs": {"user": {"shape": [8, 3]},
                        "item": {"shape": [8, 4]}},
             "outputs": {"score": "dot", "user_embedding": "u"}},
    "group": {"inputs": {"x": {"shape": [8, 5]}},
              "outputs": {"embedding": "e", "similarity": "s"}},
    "uplift": {"inputs": {"x": {"shape": [8, 5]}},
               "outputs": {"treated": "t", "control": "c"}},
    "price": {"inputs": {"x": {"shape": [8, 5]}},
              "outputs": {"units": "u"}},
    "demand": {"inputs": {"x": {"shape": [8, 5]}},
               "outputs": {"quantiles": "q"}},
    "table": {"inputs": {"row": {"shape": [8, 5], "dtype": "float32"},
                         "step": {"shape": [8], "dtype": "int64"}},
              "outputs": {"noise": "n"}},
    "generate": {"inputs": {"image": {"shape": [8, 3, 4, 4]},
                            "step": {"shape": [8], "dtype": "int64"}},
                 "outputs": {"noise": "n"}},
    "policy": {"inputs": {"obs": {"shape": [1, 5]}},
               "outputs": {"action": "head"}},
}

# 最後の大きさの鍵から数
WIDTHS = {"features": 5, "users": 3, "items": 4, "observations": 5}


def problem_of(kind: str, extra: Optional[dict] = None) -> dict:
    """
    骨組みの型に合う問題の設定を作る関数

    Parameters
    ----------
    kind: str
        骨組みの型 (ANSWERS の鍵)
    extra: Optional[dict] = None
        さらに書く答え

    Returns
    ----------
    problem: dict
        問題の設定 (列の予測の見本から答えを変えたもの)
    """
    # 見本への答えの書き込み
    problem = example_problem("総菜の作る量")
    for key, value in {**ANSWERS[kind], **(extra or {})}.items():
        problem = answer(problem, key, value)

    return problem


def ids_of(found: list[dict]) -> list[str]:
    """
    確かめの場所を並べる関数

    Parameters
    ----------
    found: list[dict]
        確かめ

    Returns
    ----------
    ids: list[str]
        場所 (確かめの順)
    """
    # 全て誤りの重さ
    assert all(item["level"] == "error" for item in found)
    ids = [item["id"] for item in found]

    return ids


def test_model_contract_every_type() -> None:
    """
    骨組みの型ごとの約束の入力と出力の名前を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 全ての型の確かめ
    assert set(ANSWERS) == set(SKELETON_TYPES) == set(NAMES)
    for kind, names in NAMES.items():
        problem = problem_of(kind)
        assert skeleton_type(problem) == kind
        contract = model_contract(problem)
        if names is None:
            assert contract is None, kind
            continue
        assert contract["skeleton"] == kind
        assert [item["name"] for item in contract["inputs"]] == names[0]
        assert [item["name"] for item in contract["outputs"]] == names[1]
        keys = {"name", "dtype", "width", "size", "label", "text"}
        for item in contract["inputs"] + contract["outputs"]:
            assert set(item) == keys
            assert (item["label"] is None) == (item["width"] is None)

    # 機械学習を使わない場合と, 距離に基づく度合い
    assert model_contract(new_problem("x")) is None
    distance = problem_of("anomaly", {f"{PART}.anomaly.score": "distance"})
    assert model_contract(distance) is None


def test_model_contract_widths() -> None:
    """
    約束の最後の大きさの鍵と, 問題の設定で決まる大きさを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 異常の検知は入力と出力が学習に使う列の数
    anomaly = model_contract(problem_of("anomaly"))
    assert anomaly["inputs"][0]["width"] == "features"
    assert anomaly["inputs"][0]["label"] == "学習に使う列の数"
    assert anomaly["outputs"][0]["width"] == "features"
    assert anomaly["extra_outputs"]

    # 並べるは利用者と品物の数, まとめるは符号化した行の幅
    rank = model_contract(problem_of("rank"))
    assert [item["width"] for item in rank["inputs"]] == ["users", "items"]
    group = model_contract(problem_of("group"))
    assert group["inputs"][0]["label"] == "符号化した行の幅"

    # 効果の見積もりと値段は列の数の言葉と, 出力の大きさ 1
    uplift = model_contract(problem_of("uplift"))
    assert "処置の列を除いた" in uplift["inputs"][0]["label"]
    assert [item["size"] for item in uplift["outputs"]] == [1, 1]
    price = model_contract(problem_of("price"))
    assert price["inputs"][0]["label"] == "特徴の列と値段の列の数"
    assert price["outputs"][0]["size"] == 1

    # 需要の水準の数は 0.5 と答えた水準
    demand = model_contract(problem_of(
        "demand", {f"{PART}.demand.levels": [0.1, 0.9]}
    ))
    assert demand["outputs"][0]["width"] == "levels"
    assert demand["outputs"][0]["size"] == 3

    # 生成は時刻を型で見分け, 他の出力を持てない
    table = model_contract(problem_of("table"))
    assert [item["dtype"] for item in table["inputs"]] == [None, "int64"]
    assert table["inputs"][0]["width"] == "features"
    assert not table["extra_outputs"]
    policy = model_contract(problem_of("policy"))
    assert policy["inputs"][0]["width"] == "observations"
    assert policy["outputs"][0]["width"] == "actions"


def test_check_model_contract_matches() -> None:
    """
    約束に合う定義では確かめが空になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 合う定義と最後の大きさ (数を渡さなくても空)
    for kind, spec in SPECS.items():
        problem = problem_of(kind)
        assert check_model_contract(problem, spec, WIDTHS) == [], kind
        assert check_model_contract(problem, spec) == [], kind

    # 約束の無い骨組みは確かめ無し
    assert check_model_contract(problem_of("column"), {}) == []
    assert check_model_contract(new_problem("x"), {}) == []


def test_check_model_contract_named() -> None:
    """
    名前を決めた骨組みで, 名前と最後の大きさの食い違いを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 入力と出力の名前の違い
    problem = problem_of("anomaly")
    spec = {"inputs": {"features": {"shape": [8, 5]}},
            "outputs": {"out": "dec"}}
    found = check_model_contract(problem, spec, WIDTHS)
    assert ids_of(found) == ["inputs", "inputs.features", "outputs"]
    assert "x" in found[0]["message"]
    assert "recon" in found[2]["message"]

    # 最後の大きさの違い (渡した数と比べる)
    spec = {"inputs": {"x": {"shape": [8, 4]}}, "outputs": {"recon": "d"}}
    found = check_model_contract(problem, spec, WIDTHS)
    assert ids_of(found) == ["inputs.x"]
    assert found[0]["message"] == ("入力 x の最後の大きさを学習に使う列の数 "
                                   "5 にしてください (今は 4)")
    assert check_model_contract(problem, spec) == []

    # 並べるの片方の入力の無い定義と, 品物の数の違い
    rank = problem_of("rank")
    spec = {"inputs": {"user": {"shape": [8, 3]}}, "outputs": {"score": "s"}}
    assert ids_of(check_model_contract(rank, spec, WIDTHS)) == ["inputs"]
    spec = {"inputs": {"user": {"shape": [8, 3]}, "item": {"shape": [8, 9]}},
            "outputs": {"score": "s"}}
    found = check_model_contract(rank, spec, WIDTHS)
    assert ids_of(found) == ["inputs.item"]
    assert "学習用の品物の数 4" in found[0]["message"]

    # まとめると効果の見積もりの出力の片方が無い定義
    for kind, name in (("group", "similarity"), ("uplift", "control")):
        spec = {"inputs": {"x": {"shape": [8, 5]}},
                "outputs": {key: "o" for key in SPECS[kind]["outputs"]
                            if key != name}}
        found = check_model_contract(problem_of(kind), spec, WIDTHS)
        assert ids_of(found) == ["outputs"], kind
        assert name in found[0]["message"]

    # 値段と需要の列の数の違い
    for kind in ("price", "demand"):
        spec = {**SPECS[kind], "inputs": {"x": {"shape": [8, 7]}}}
        found = check_model_contract(problem_of(kind), spec, WIDTHS)
        assert ids_of(found) == ["inputs.x"], kind


def test_check_model_contract_free() -> None:
    """
    名前を問わない骨組みで, 数と型と最後の大きさの食い違いを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 表の生成の時刻の無い定義と, 入力の数の違い
    table = problem_of("table")
    spec = {"inputs": {"row": {"shape": [8, 5]}, "t": {"shape": [8]}},
            "outputs": {"noise": "n"}}
    found = check_model_contract(table, spec, WIDTHS)
    assert ids_of(found) == ["inputs"]
    assert "int64" in found[0]["message"]
    spec = {"inputs": {"row": {"shape": [8, 5]}}, "outputs": {"noise": "n"}}
    assert ids_of(check_model_contract(table, spec, WIDTHS)) == ["inputs"]

    # 符号化した行の幅の違いと, 出力の数の違い
    spec = {"inputs": {"row": {"shape": [8, 6]},
                       "t": {"shape": [8], "dtype": "int64"}},
            "outputs": {"noise": "n", "extra": "e"}}
    found = check_model_contract(table, spec, WIDTHS)
    assert ids_of(found) == ["inputs.row", "outputs"]
    assert "符号化した行の幅 5" in found[0]["message"]

    # 方策の観測の要素の数の違い (渡さなければ見ない)
    policy = problem_of("policy")
    spec = {"inputs": {"obs": {"shape": [1, 3]}}, "outputs": {"a": "h"}}
    assert ids_of(check_model_contract(policy, spec, WIDTHS)) == [
        "inputs.obs"
    ]
    assert check_model_contract(policy, spec) == []

    # 崩れた定義は入力と出力の無い定義の扱い
    found = check_model_contract(table, {"inputs": [], "outputs": None})
    assert ids_of(found) == ["inputs", "outputs"]
    assert ids_of(check_model_contract(problem_of("price"), "x")) == [
        "inputs", "outputs"
    ]


def test_contract_examples() -> None:
    """
    モデルの作成の雛形 (ML-ModelControl の同梱の例) が約束に合うことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 雛形の書き換える引数をそのまま最後の大きさに
    modelcontrol = pytest.importorskip("ml_modelcontrol")
    for kind in ("anomaly", "rank", "group", "uplift", "price", "demand",
                 "table"):
        problem = problem_of(kind)
        name = model_example(problem)
        args = modelcontrol.example_args(name)
        spec = modelcontrol.adapt_example(name, args)
        assert check_model_contract(problem, spec, args) == [], kind

        # 入力の幅を変えると食い違い
        changed = {**args, "features": args.get("features", 0) + 1}
        if "features" in args:
            found = check_model_contract(problem, spec, changed)
            assert len(found) == 1, kind
