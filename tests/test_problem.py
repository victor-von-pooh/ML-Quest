import ast
import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ml_quest import (
    EXAMPLES, GUARD_COUNTS, QUESTIONS, SPLITS, answer, apply_recipe, check,
    describe, example_problem, get_value, handoff, is_ready, load_problem,
    make_project, new_problem, questions, read_columns, recipe_from,
    save_problem
)
from ml_quest.problem import DECISION_FIELDS, DECISION_OPTIONS


def levels(problem: dict) -> dict[str, list[str]]:
    """
    確かめを重さごとの項目の場所にまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: dict[str, list[str]]
        error, warning, info ごとの項目の場所
    """
    # 重さごと
    found: dict[str, list[str]] = {"error": [], "warning": [], "info": []}
    for item in check(problem):
        found[item["level"]].append(item["id"])

    return found


def test_examples_are_consistent() -> None:
    """
    見本の問題の設定が, 確かめで error を出さず, 機械学習を使うものは
    渡せることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 意思決定の文は項目の名前で
    summary = describe(example_problem("解約の引き止め"))
    assert any("当たりの価値 24,000" in s and "件数の上限 300" in s
               for s in summary)

    # 4 つの見本
    for name in EXAMPLES:
        problem = example_problem(name)
        assert levels(problem)["error"] == [], name
        uses = get_value(problem, "ml.use")
        assert is_ready(problem) is uses
        assert describe(problem)[0].endswith(
            get_value(problem, "issue.pain")
        )
    with pytest.raises(ValueError, match="見本"):
        example_problem("無い見本")


def test_wording_is_not_business_only() -> None:
    """
    問いと確かめの文が仕事に限る言葉 (業務, 実務, KPI) を使わず, 課題の
    指標と書くことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 問いの見出し, 説明, 選択肢の文
    texts = []
    for item in QUESTIONS:
        texts.append(str(item.get("title", "")))
        texts.append(str(item.get("help", "")))
        for choice in item.get("choices", []):
            texts.append(str(choice.get("label", "")))
    for text in texts:
        for word in ("業務", "実務", "KPI"):
            assert word not in text, text

    # 指標のつながりが空の警告
    problem = answer(example_problem("総菜の作る量"),
                     "ml.parts.0.evaluation.link", "")
    messages = [item["message"] for item in check(problem)
                if item["id"] == "ml.parts.0.evaluation.link"]
    assert len(messages) == 1
    assert "課題の指標" in messages[0]


def test_questions_and_answers() -> None:
    """
    問いの並びが今の答えで要るかを変え, 答えの形を確かめることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 空の問題の設定での, 分類だけの問いと予測だけの問いの不要
    problem = new_problem("試し")
    items = {item["id"]: item for item in questions(problem)}
    assert len(items) == len(QUESTIONS)
    assert items["issue.pain"]["applies"] is True
    assert items["ml.parts.0.target.column"]["applies"] is False
    assert items["ml.parts.0.time_column"]["applies"] is False

    # 予測を選ぶと時刻の列の問いが要り, 指標は回帰と予測のものだけ
    problem = answer(problem, "ml.use", True)
    items = {item["id"]: item for item in questions(problem)}
    assert items["ml.parts.0.role"]["applies"] is True
    assert items["ml.parts.0.task"]["applies"] is False
    problem = answer(problem, "ml.parts.0.role", "predict")
    problem = answer(problem, "ml.parts.0.target.kind", "column")
    problem = answer(problem, "ml.parts.0.task", "forecasting")
    items = {item["id"]: item for item in questions(problem)}
    assert items["ml.parts.0.time_column"]["applies"] is True
    assert [c["value"] for c in items["ml.parts.0.evaluation.metric"][
        "choices"
    ]] == ["mae", "rmse", "other"]

    # 元の問題の設定の不変
    assert get_value(new_problem("試し"), "ml.use") is None
    updated = answer(problem, "issue.kpi.current", "1,200,000")
    assert get_value(updated, "issue.kpi.current") == 1200000
    assert get_value(problem, "issue.kpi.current") is None
    assert get_value(answer(updated, "issue.kpi.current", ""),
                     "issue.kpi.current") is None

    # 形の誤り
    with pytest.raises(ValueError, match="という問い"):
        answer(problem, "no.such", 1)
    with pytest.raises(ValueError, match="数で"):
        answer(problem, "issue.kpi.current", "たくさん")
    with pytest.raises(ValueError, match="どれか"):
        answer(problem, "ml.parts.0.task", "clustering")
    with pytest.raises(ValueError, match="はいかいいえ"):
        answer(problem, "ml.use", "yes")
    with pytest.raises(ValueError, match="書けません"):
        answer(problem, "action.decision",
               {"type": "quantity", "miss_cost": 1})
    columns = answer(problem, "information.columns", ["a", "b", "a"])
    assert get_value(columns, "information.columns") == ["a", "b"]
    with pytest.raises(ValueError, match="列にありません"):
        answer(columns, "ml.parts.0.target.column", "c")


def test_decision_values() -> None:
    """
    損得の値に, 数と行ごとの列の名前を書けることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 数の文字列は数に, それ以外は列の名前
    problem = answer(new_problem("試し"), "action.decision", {
        "type": "select", "value": "yearly_fee", "cost": "1,500",
        "effect": "0.2", "budget": None,
    })
    decision = get_value(problem, "action.decision")
    assert decision == {"type": "select", "value": "yearly_fee",
                        "cost": 1500, "effect": 0.2}


def test_check_finds_inconsistencies() -> None:
    """
    答えの食い違いを確かめで見つけることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 量を決めるのに分類, 目的の列を分かる列に, 目指す値の向きが逆
    problem = example_problem("総菜の作る量")
    problem = answer(problem, "ml.parts.0.task", "classification")
    problem = answer(problem, "information.known",
                     ["sales", "promo", "customers"])
    problem = answer(problem, "issue.kpi.target", 1500000)
    found = levels(problem)
    assert "action.decision" in found["error"]
    assert "information.known" in found["error"]
    assert "information.unknown" in found["error"]
    assert "issue.kpi.target" in found["error"]
    assert "ml.parts.0.evaluation.metric" in found["error"]
    assert is_ready(problem) is False
    with pytest.raises(ValueError, match="直すところ"):
        handoff(problem)

    # 損得の値の範囲と根拠, 選ぶ型の過去のデータだけの確かめ
    churn = example_problem("解約の引き止め")
    churn = answer(churn, "action.decision", {
        "type": "select", "value": 24000, "cost": 1500, "effect": 1.5,
    })
    churn = answer(churn, "action.basis", None)
    churn = answer(churn, "verification.method", "backtest")
    churn = answer(churn, "verification.backtest", "過去 1 年")
    churn = answer(churn, "ml.parts.0.positive", None)
    found = levels(churn)
    assert "action.decision" in found["error"]
    assert "ml.parts.0.positive" in found["error"]
    assert "action.basis" in found["warning"]
    assert "verification.method" in found["warning"]

    # 機械学習を使わない出口 (課題, 分解, 決めること, 理由だけ)
    exit_only = new_problem("出口")
    for key, value in EXAMPLES[
        "問い合わせの振り分け (機械学習を使わない)"
    ]:
        exit_only = answer(exit_only, key, value)
    assert levels(exit_only)["error"] == []
    without = answer(exit_only, "action.what", None)
    assert "action.what" in levels(without)["error"]
    with pytest.raises(ValueError, match="渡すもの"):
        handoff(exit_only)

    # 足りない答え
    missing = levels(new_problem("空"))["error"]
    assert "issue.pain" in missing and "ml.use" in missing

    # 何を決めるかは, 機械学習を使うかより先に, 使わない場合も問う
    order = [question["id"] for question in QUESTIONS]
    assert order.index("action.what") < order.index("ml.use")
    assert order.index("action.when") < order.index("ml.parts.0.what")
    assert "action.what" in missing


def test_handoff_values() -> None:
    """
    受け渡しの値が, 各段階の約束と同じ名前と形になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 予測: 時刻の順の分割, 分からない列を除く, 量の意思決定
    stages = handoff(example_problem("総菜の作る量"))
    assert stages["data"]["split"] == "time"
    assert stages["data"]["drop"] == ["customers"]
    assert stages["training"] == {"metric": "mae", "monitor": "valid/mae",
                                  "mode": "min", "role": "predict",
                                  "target_kind": "column"}
    assert stages["inference"]["decision"]["type"] == "quantity"
    assert stages["inference"]["conclusion"] == {"group_by": "store"}

    # 分類: 選ぶ型に当たりのクラスを足す. 交差エントロピーは小さいほど良い
    stages = handoff(example_problem("解約の引き止め"))
    assert stages["data"]["split"] == "stratified"
    assert stages["model"]["outputs"] is None
    assert stages["training"]["mode"] == "min"
    assert stages["inference"]["decision"]["positive"] == "yes"

    # 正解率は大きいほど良い. 意思決定しないなら None
    problem = answer(example_problem("解約の引き止め"),
                     "ml.parts.0.evaluation.metric", "accuracy")
    problem = answer(problem, "action.decision", {"type": "none"})
    stages = handoff(problem)
    assert stages["training"]["mode"] == "max"
    assert stages["inference"]["decision"] is None


def test_handoff_current() -> None:
    """
    今のやり方の列を, たどり直しで比べる列として渡すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 選ぶ型の行動したかの列は渡す. k 期前は ML-Inference に無いので渡さない
    churn = example_problem("解約の引き止め")
    churn = answer(churn, "information.columns",
                   get_value(churn, "information.columns") + ["contacted"])
    column = answer(churn, "action.current",
                    {"type": "column", "column": "contacted"})
    assert handoff(column)["inference"]["decision"]["current"] == (
        "contacted"
    )

    # 今のやり方の列は学習に使わず落とす (予測の時点で分かる列に入れた
    # 場合は落とさない)
    assert "contacted" in handoff(column)["data"]["drop"]
    assert any(step.get("op") == "drop_columns"
               and "contacted" in step["columns"]
               for step in recipe_from(column)["recipe"]["steps"])
    sozai = example_problem("総菜の作る量")
    weekly = answer(sozai, "action.current", {"type": "lag", "periods": 7})
    assert "current" not in handoff(weekly)["inference"]["decision"]

    # 量の型は, 今ある量が無いときだけ (置いた量と足した量の違い)
    sozai = answer(sozai, "information.columns",
                   get_value(sozai, "information.columns") + ["made"])
    placed = answer(sozai, "action.current",
                    {"type": "column", "column": "made"})
    assert handoff(placed)["inference"]["decision"]["current"] == "made"
    decision = dict(get_value(placed, "action.decision"), stock=5)
    stocked = answer(placed, "action.decision", decision)
    assert "current" not in handoff(stocked)["inference"]["decision"]
    assert any(item["id"] == "action.current" and item["level"] == "info"
               for item in check(stocked))

    # 今のやり方の列を学習に使うと注意
    leaky = answer(column, "information.known",
                   get_value(column, "information.known") + ["contacted"])
    assert "contacted" not in handoff(leaky)["data"]["drop"]
    assert any(item["id"] == "action.current"
               and item["level"] == "warning" for item in check(leaky))


def test_save_and_load(tmp_path: Path) -> None:
    """
    問題の設定を書いて読み直せることと, 新しい版を確かめで知らせることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # ディレクトリへの書き込みと, 更新の時刻だけの変化
    problem = example_problem("総菜の作る量")
    saved = save_problem(problem, tmp_path)
    loaded = load_problem(tmp_path)
    assert loaded == saved
    assert loaded["created_at"] == problem["created_at"]
    assert not list(tmp_path.glob(".*.tmp"))

    # 版の無いファイルは誤り, 新しい版は知らせ
    (tmp_path / "bad.yaml").write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_problem(tmp_path / "bad.yaml")
    newer = dict(loaded, format_version=99)
    assert any(item["id"] == "format_version" for item in check(newer))


def test_read_columns(tmp_path: Path) -> None:
    """
    データのファイルから列の名前を読めることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # UTF-8 と Shift_JIS の csv, tsv
    frame = pd.DataFrame({"日付": ["2026-01-01"], "売上": [1]})
    frame.to_csv(tmp_path / "u.csv", index=False)
    frame.to_csv(tmp_path / "s.csv", index=False, encoding="cp932")
    frame.to_csv(tmp_path / "t.tsv", index=False, sep="\t")
    for name in ("u.csv", "s.csv", "t.tsv"):
        assert read_columns(tmp_path / name) == ["日付", "売上"]
    with pytest.raises(ValueError, match="読めません"):
        read_columns(tmp_path / "x.json")


def test_make_project(tmp_path: Path) -> None:
    """
    骨組みが ML-Found のプロジェクトの形で書き出され, 学習のスクリプトが
    Python として読め, 79 文字以内であることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 予測と分類の 2 つ
    for name in ("総菜の作る量", "解約の引き止め"):
        problem = example_problem(name)
        project = tmp_path / name
        written = make_project(problem, project)
        assert (project / "configs/default/train.yaml").is_file()
        assert (project / "configs/experiment/train.yaml").is_file()
        assert (project / "models").is_dir()
        assert "train.py" in written
        script = (project / "train.py").read_text(encoding="utf-8")
        ast.parse(script)
        assert max(len(line) for line in script.splitlines()) <= 79
        config = yaml.safe_load(
            (project / "configs/default/train.yaml").read_text("utf-8")
        )
        stages = handoff(problem)
        assert config["trainer"]["monitor"] == stages["training"]["monitor"]
        assert config["trainer"]["mode"] == stages["training"]["mode"]
        # 列の予測は定義が無いと止まるので, 仮のモデルの目印を書かない
        assert "placeholder" not in config["model"]
        assert load_problem(project) == yaml.safe_load(
            yaml.safe_dump(problem, allow_unicode=True)
        )
        readme = (project / "README.md").read_text(encoding="utf-8")
        assert stages["data"]["target"] in readme

    # 学習の指標の名前と, 推論の一式の既定値
    script = (tmp_path / "総菜の作る量" / "train.py").read_text("utf-8")
    assert '"mae": error.abs().mean()' in script
    assert '"under_cost": 200' in script
    assert "target_scale={\"center\": center" in script
    churn = (tmp_path / "解約の引き止め" / "train.py").read_text("utf-8")
    assert '"accuracy": accuracy' in churn and "target_scale=None" in churn
    assert config["experiment"]["name"] == "churned_classification"

    # 既にあれば上書きしない. 渡せない問題の設定は誤り
    with pytest.raises(ValueError, match="既にあります"):
        make_project(example_problem("総菜の作る量"),
                     tmp_path / "総菜の作る量")
    make_project(example_problem("総菜の作る量"), tmp_path / "総菜の作る量",
                 overwrite=True)
    with pytest.raises(ValueError, match="渡すもの"):
        make_project(example_problem(
            "問い合わせの振り分け (機械学習を使わない)"
        ), tmp_path / "x")


def test_review_fixes(tmp_path: Path) -> None:
    """
    手で崩したファイル, 古い答え, 列の答え直し, 損得の値の読み方, 埋め込む
    文字の崩れを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 形の崩れは例外でなく error
    broken = example_problem("総菜の作る量")
    broken["action"]["decision"] = {"type": "foo"}
    broken["ml"]["parts"][0]["series_by"] = "store"
    broken["decomposition"] = ["文字"]
    found = levels(broken)
    assert {"action.decision", "ml.parts.0.series_by",
            "decomposition"} <= set(found["error"])
    assert is_ready(broken) is False

    # タスクを変えると, 時刻と系列の古い答えは受け渡さない
    regression = answer(example_problem("総菜の作る量"), "ml.parts.0.task",
                        "regression")
    stages = handoff(regression)
    assert stages["data"]["time_column"] is None
    assert stages["data"]["series_by"] == []
    assert stages["inference"]["conclusion"] == {}
    off = answer(regression, "ml.use", False)
    items = {item["id"]: item for item in questions(off)}
    assert items["ml.parts.0.time_column"]["applies"] is False
    assert items["verification.backtest"]["applies"] is False

    # 列を答え直した後に残った古い名前は error
    narrowed = answer(example_problem("総菜の作る量"),
                      "information.columns", ["date", "x"])
    found = levels(narrowed)
    assert {"ml.parts.0.target.column", "information.known",
            "ml.parts.0.series_by"} <= set(found["error"])

    # 損得の値: 数の文字列は数, 列の名前は使えるデータの列から
    problem = example_problem("総菜の作る量")
    cases = {"1e3": 1000, "+5": 5, " 12 ": 12, "1,200": 1200}
    for text, number in cases.items():
        decided = answer(problem, "action.decision", {
            "type": "quantity", "under_cost": text, "over_cost": 1,
        })
        assert get_value(decided, "action.decision.under_cost") == number
    for text in ("nan", "inf", "1,2,3", "-"):
        with pytest.raises(ValueError):
            answer(problem, "action.decision", {
                "type": "quantity", "under_cost": text, "over_cost": 1,
            })
    by_column = answer(problem, "action.decision", {
        "type": "quantity", "under_cost": "no_such", "over_cost": 1,
    })
    assert "action.decision" in levels(by_column)["error"]

    # 改行や三重の引用符を含む答えでも, 学習のスクリプトは壊れない
    odd = answer(problem, "ml.parts.0.baseline", "前の週\n(手計算) \"\"\" \\")
    odd = answer(odd, "ml.parts.0.what", "売れる数 \"\"\"")
    odd["name"] = "名前 \"\"\" " + "長い" * 60
    make_project(odd, tmp_path / "odd")
    script = (tmp_path / "odd" / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert max(len(line) for line in script.splitlines()) <= 79

    # 文字列の指標の値と, 片方だけの値でも要約は落ちない
    text = answer(problem, "issue.kpi.target", None)
    text["issue"]["kpi"]["current"] = "1200"
    assert "今 1200" in describe(text)[2]

    # 空の答えの消去と, 空になった親の辞書の消去
    empty = answer(new_problem("空"), "issue.kpi.name", "損")
    empty = answer(empty, "issue.kpi.name", None)
    assert "issue" not in empty
    assert get_value(answer(problem, "ml.use", ""), "ml.use") is None


def test_monitor_and_config(tmp_path: Path) -> None:
    """
    RMSE は二乗誤差で選び, 正解率は損失で同じ値を比べる config になることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # RMSE はバッチの平均で書けないので, 二乗誤差を監視
    rmse = answer(example_problem("総菜の作る量"), "ml.parts.0.evaluation.metric",
                  "rmse")
    assert handoff(rmse)["training"]["monitor"] == "valid/mse"
    make_project(rmse, tmp_path / "rmse")
    script = (tmp_path / "rmse" / "train.py").read_text("utf-8")
    assert '"mse": error.pow(2).mean()' in script
    assert "error.pow(2).mean().sqrt()" in script

    # 正解率は同じ値を取りやすい
    accuracy = answer(example_problem("解約の引き止め"),
                      "ml.parts.0.evaluation.metric", "accuracy")
    make_project(accuracy, tmp_path / "acc")
    config = yaml.safe_load(
        (tmp_path / "acc" / "configs/default/train.yaml").read_text("utf-8")
    )
    assert config["trainer"]["monitor_tiebreak"] == "valid/loss"

    # 日本語の目的の列は実験の名前に残す
    japanese = answer(example_problem("総菜の作る量"), "information.columns",
                      ["date", "store", "item", "売上", "temperature_forecast",
                       "promo", "customers"])
    japanese = answer(japanese, "ml.parts.0.target.column", "売上")
    make_project(japanese, tmp_path / "ja")
    config = yaml.safe_load(
        (tmp_path / "ja" / "configs/default/train.yaml").read_text("utf-8")
    )
    assert config["experiment"]["name"] == "売上_forecasting"


def test_read_columns_edges(tmp_path: Path) -> None:
    """
    同じ名前の列, 空のファイル, parquet の列の名前を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ名前の列と空のファイルは誤り
    (tmp_path / "dup.csv").write_text("a,a,b\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="同じ名前"):
        read_columns(tmp_path / "dup.csv")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="空"):
        read_columns(tmp_path / "empty.csv")

    # parquet は形の情報だけ (pyarrow が無ければ飛ばす)
    pytest.importorskip("pyarrow")
    pd.DataFrame({"x": [1], "y": [2]}).to_parquet(tmp_path / "p.parquet")
    assert read_columns(tmp_path / "p.parquet") == ["x", "y"]


def test_scalar_shapes_and_blocked_paths(tmp_path: Path) -> None:
    """
    文字列であるべき値の崩れと, 書き出し先の邪魔なファイルを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 並びや辞書のタスク, 目的の列, 意思決定の型も例外でなく error
    for key, value in (("part", {"task": [1]}),
                       ("part", {"target": {"column": {"a": 1}}}),
                       ("part", {"target": "sales"}),
                       ("action", {"decision": {"type": ["quantity"]}})):
        broken = example_problem("総菜の作る量")
        place = broken["ml"]["parts"][0] if key == "part" else broken[key]
        place.update(value)
        assert levels(broken)["error"], key
        assert is_ready(broken) is False

    # 文の答えの改行は残す
    text = answer(new_problem("文"), "issue.pain", " 一行目 \n\n\n 二行目 ")
    assert get_value(text, "issue.pain") == "一行目\n\n二行目"

    # 書き出し先の configs がファイルなら, 何も書かずに誤り
    (tmp_path / "configs").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="邪魔"):
        make_project(example_problem("総菜の作る量"), tmp_path,
                     overwrite=True)
    assert not (tmp_path / "README.md").exists()


def test_available_lag() -> None:
    """
    決める時点で分かる目的の値の問いが, 時系列の予測だけで要り, 受け渡しの
    いちばん小さいずらし幅になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本は前の日の 17 時に決めるので 2
    problem = example_problem("総菜の作る量")
    assert handoff(problem)["data"]["min_lag"] == 2
    missing = answer(problem, "ml.parts.0.available_lag", None)
    assert "ml.parts.0.available_lag" in levels(missing)["error"]
    zero = answer(problem, "ml.parts.0.available_lag", 0)
    assert "ml.parts.0.available_lag" in levels(zero)["error"]

    # 回帰では問わず, 受け渡しにも出さない
    regression = answer(problem, "ml.parts.0.task", "regression")
    assert handoff(regression)["data"]["min_lag"] is None


def test_uplift_decision() -> None:
    """
    対象を選ぶ型の効果を確率の差 (uplift) で書けて, 向きと範囲を確かめ,
    推論へそのまま渡すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 確率の差は負も書け, 向きと一緒に推論へ渡す
    problem = example_problem("解約の引き止め")
    decision = dict(get_value(problem, "action.decision"), effect=-0.1,
                    effect_kind="uplift", effect_goal="reduce")
    problem = answer(problem, "action.decision", decision)
    assert "action.decision" not in levels(problem)["error"]
    assert handoff(problem)["inference"]["decision"]["effect_goal"] == (
        "reduce"
    )
    assert any("確率の差 -0.1" in text and "当たりを減らす" in text
               and "防げる割合" not in text for text in describe(problem))
    assert any("無作為" in item["message"] for item in check(problem))

    # 向きの書き忘れ, 範囲の外, 防げる割合で向き, 知らない値は誤り
    goalless = {key: value for key, value in decision.items()
                if key != "effect_goal"}
    for wrong in (goalless, dict(decision, effect=1.5),
                  dict(decision, effect_kind="share", effect=0.2)):
        assert "action.decision" in levels(
            answer(problem, "action.decision", wrong)
        )["error"]
    with pytest.raises(ValueError, match="effect_kind"):
        answer(problem, "action.decision", dict(decision, effect_kind="x"))
    with pytest.raises(ValueError, match="書けません"):
        answer(problem, "action.decision",
               {"type": "triage", "miss_cost": 9, "false_alarm_cost": 1,
                "effect_kind": "uplift"})

    # 手で書いたファイル: 空の渡し方は既定, 並びの値と他の型の項目は誤り
    raw = example_problem("解約の引き止め")
    raw["action"]["decision"] = dict(
        raw["action"]["decision"], effect_kind=""
    )
    assert "action.decision" not in levels(raw)["error"]
    raw["action"]["decision"]["effect_kind"] = ["uplift"]
    assert "action.decision" in levels(raw)["error"]
    assert describe(raw)
    raw["action"]["decision"] = {"type": "triage", "miss_cost": 9,
                                 "false_alarm_cost": 1,
                                 "effect_goal": "reduce"}
    assert any("書けません" in item["message"] for item in check(raw)
               if item["id"] == "action.decision")


def test_guard_metric() -> None:
    """
    守る指標を書くと続きの問いが要るようになり, 要約に出て, 成功の基準と
    止める条件へのつなぎを確かめることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 名前を書くまで向きと線は要らない
    problem = example_problem("総菜の作る量")
    needed = {item["id"]: item["applies"] for item in questions(problem)}
    assert needed["issue.guard.name"] and not needed["issue.guard.line"]
    problem = answer(problem, "issue.guard.name", "品切れした品と日の割合")
    found = levels(problem)
    assert {"issue.guard.better", "issue.guard.line"} <= set(found["error"])
    assert "verification.success" in found["info"]

    # 向きと線を書くと要約に出て, 止める条件に入れると知らせが消える
    problem = answer(problem, "issue.guard.better", "lower")
    problem = answer(problem, "issue.guard.line", 5)
    problem = answer(problem, "issue.guard.unit", "%")
    assert any(text.startswith("守る指標: 品切れした品と日の割合")
               and "5 を超えない" in text for text in describe(problem))
    problem = answer(problem, "operation.stop",
                     "品切れした品と日の割合が 5 % を超えたら止める")
    assert "verification.success" not in levels(problem)["info"]

    # 推論の段階へ, たどり直しと測り直しで数える値として渡る
    assert handoff(problem)["inference"]["guard"] == {
        "name": "品切れした品と日の割合", "unit": "%", "better": "lower",
        "line": 5, "measure": None, "counted": None,
    }

    # 解けたと言える指標と同じ名前は見直し. 名前を消すと続きは要らない
    same = answer(problem, "issue.guard.name", "廃棄と欠品の損")
    assert "issue.guard.name" in levels(same)["warning"]
    cleared = answer(problem, "issue.guard.name", None)
    assert levels(cleared)["error"] == []
    assert not any(text.startswith("守る指標") for text in describe(cleared))
    assert handoff(cleared)["inference"]["guard"] is None

    # 見本の解約の引き止めは守る指標を持ち, 止める条件に入っている
    churn = example_problem("解約の引き止め")
    assert "verification.success" not in levels(churn)["info"]
    assert handoff(churn)["inference"]["guard"] == {
        "name": "引き止めの連絡への苦情", "unit": "件 / 月",
        "better": "lower", "line": 5,
        "measure": "窓口に届いた, 連絡についての苦情を数える",
        "counted": None,
    }

    # 手で書いた数でない線と知らない向きは誤り
    raw = example_problem("解約の引き止め")
    raw["issue"]["guard"].update(line="abc", better="up")
    assert {"issue.guard.line", "issue.guard.better"} <= set(
        levels(raw)["error"]
    )


def test_guard_counted() -> None:
    """
    守る指標がたどり直しで数える量を, 意思決定の型の量から選べ, 受け渡しの
    推論の守る指標に入ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 守る指標を書くまで問わない. 選択肢は量を決める型の量と指さない
    problem = example_problem("総菜の作る量")
    item = {item["id"]: item for item in questions(problem)}
    assert not item["issue.guard.counted"]["applies"]
    problem = answer(problem, "issue.guard.name", "品切れした品と日の割合")
    problem = answer(problem, "issue.guard.better", "lower")
    problem = answer(problem, "issue.guard.line", 0.05)
    item = {item["id"]: item for item in questions(problem)}
    counted = item["issue.guard.counted"]
    assert counted["applies"] and not counted["required"]
    assert counted["section"] == 7 and "0.05" in counted["help"]
    assert [choice["value"] for choice in counted["choices"]] == [
        *GUARD_COUNTS["quantity"], "none"
    ]
    assert counted["choices"][0]["label"] == "足りなかった行の割合"
    assert handoff(problem)["inference"]["guard"]["counted"] is None

    # 選んだ量の名前が渡り, 指さないは None
    chosen = answer(problem, "issue.guard.counted", "shortage_rate")
    assert levels(chosen)["error"] == []
    assert handoff(chosen)["inference"]["guard"]["counted"] == "shortage_rate"
    unpointed = answer(chosen, "issue.guard.counted", "none")
    assert handoff(unpointed)["inference"]["guard"]["counted"] is None
    with pytest.raises(ValueError):
        answer(problem, "issue.guard.counted", "complaints")

    # 割合の量の守る線は 0 から 1 (% のままは誤り)
    percent = answer(chosen, "issue.guard.line", 5)
    assert any("0.05" in item["message"] for item in check(percent)
               if item["id"] == "issue.guard.line")
    counts = answer(percent, "issue.guard.counted", "shortage")
    assert "issue.guard.line" not in levels(counts)["error"]

    # 型を変えた後の古い量は誤りで, 受け渡しには渡さない
    raw = copy.deepcopy(chosen)
    raw["action"]["decision"] = {"type": "triage", "miss_cost": 9,
                                 "false_alarm_cost": 1}
    item = {item["id"]: item for item in questions(raw)}
    assert [choice["value"] for choice in item["issue.guard.counted"][
        "choices"]] == [*GUARD_COUNTS["triage"], "none"]
    assert any("選び直して" in item["message"] for item in check(raw)
               if item["id"] == "issue.guard.counted")

    # 意思決定しない型と, 効果を確率の差で渡した対象を選ぶ型では問わない
    churn = example_problem("解約の引き止め")
    assert get_value(churn, "issue.guard.counted") == "none"
    for decision in ({"type": "none"},
                     {"type": "select", "value": 24000, "cost": 1500,
                      "effect_kind": "uplift", "effect": -0.1,
                      "effect_goal": "reduce"}):
        other = answer(churn, "action.decision", decision)
        other = answer(other, "issue.guard.counted", "selected")
        item = {item["id"]: item for item in questions(other)}
        assert not item["issue.guard.counted"]["applies"]
        assert "issue.guard.counted" not in levels(other)["error"]
    other = answer(churn, "issue.guard.counted", "selected")
    assert handoff(other)["inference"]["guard"]["counted"] == "selected"

    # 手で書いた文字列でない答えは形の誤り
    raw = copy.deepcopy(chosen)
    raw["issue"]["guard"]["counted"] = 1
    assert "issue.guard.counted" in levels(raw)["error"]


def test_guard_counts_match_inference() -> None:
    """
    写した数えられる量の一覧が ML-Inference の GUARD_COUNTS と同じことを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ML-Inference が無ければ飛ばす
    inference = pytest.importorskip("ml_inference")
    assert GUARD_COUNTS == inference.GUARD_COUNTS


def test_guard_line_finite() -> None:
    """
    手で書いた NaN と無限大の守る線を, 受け渡しで None にすることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 線だけが None になり, 他の答えは残る
    for line in (float("nan"), float("inf"), float("-inf")):
        endless = example_problem("解約の引き止め")
        endless["issue"]["guard"]["line"] = line
        guard = handoff(endless)["inference"]["guard"]
        assert guard["line"] is None
        assert guard["name"] == "引き止めの連絡への苦情"


def test_output_use() -> None:
    """
    予測の使い道と後段と制約を書けて, 意思決定と後段の食い違いを確かめ,
    要約に出ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 後段に渡すときだけ後段の問いが要り, 書かなければ見直し
    problem = example_problem("総菜の作る量")
    needed = {item["id"]: item["applies"] for item in questions(problem)}
    assert needed["ml.parts.0.consumer"]
    assert not needed["ml.parts.0.downstream"]
    downstream = answer(problem, "ml.parts.0.consumer", "system")
    assert "ml.parts.0.downstream" in levels(downstream)["warning"]
    downstream = answer(downstream, "ml.parts.0.downstream", "発注の仕組み. 個数")
    assert "ml.parts.0.downstream" not in levels(downstream)["warning"]
    assert any("後段が求めるもの: 発注の仕組み" in text
               for text in describe(downstream))

    # 自動で決めるのに意思決定しないは見直し
    none = answer(problem, "action.decision", {"type": "none"})
    assert "action.decision" in levels(none)["warning"]

    # 見本の解約の引き止めは制約が要約に出る
    churn = describe(example_problem("解約の引き止め"))
    assert any(text.startswith("予測の使い道: 決まりで自動に") for text
               in churn)
    assert "守る制約: 会員の情報を社外のサービスに出さない" in churn

    # 手で書いた崩れた値は形の誤りで, 要約は落ちない
    raw = example_problem("解約の引き止め")
    raw["ml"]["parts"][0]["consumer"] = ["rule"]
    raw["ml"]["constraints"] = "外に出さない"
    assert {"ml.parts.0.consumer", "ml.constraints"} <= set(
        levels(raw)["error"]
    )
    assert describe(raw)
    raw["ml"]["parts"][0]["consumer"] = "robot"
    raw["ml"]["constraints"] = ["外に出さない"]
    assert "ml.parts.0.consumer" in levels(raw)["error"]


def test_decision_choices() -> None:
    """
    意思決定の問いが, 型と型ごとの欄を選択肢で返し, 欄が確かめの項目と
    合うことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 欄は型ごとの損得の項目と選んで書く項目のちょうど全て
    problem = example_problem("解約の引き止め")
    item = next(q for q in questions(problem) if q["id"] == "action.decision")
    kinds = {choice["value"]: choice for choice in item["choices"]}
    assert set(kinds) == set(DECISION_FIELDS)
    for kind, required in DECISION_FIELDS.items():
        fields = {field["key"]: field for field in kinds[kind]["fields"]}
        options = DECISION_OPTIONS.get(kind, {})
        assert set(fields) == set(required) | set(options)
        for key, needed in required.items():
            assert fields[key]["required"] is needed
        for key, values in options.items():
            assert [c["value"] for c in fields[key]["choices"]] == list(
                values
            )
    goal = next(field for field in kinds["select"]["fields"]
                if field["key"] == "effect_goal")
    assert goal["when"] == {"key": "effect_kind", "equals": "uplift"}


def test_group_column() -> None:
    """
    同じ相手の行をまとめる列を書くと, 分割を相手ごとにして渡し, 時系列
    では時刻の順のまま列だけを渡すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 分類は相手ごとの分割に (レシピの下書きも)
    problem = answer(example_problem("解約の引き止め"), "ml.parts.0.group_column",
                     "member_id")
    assert levels(problem)["error"] == []
    data = handoff(problem)["data"]
    assert (data["split"], data["group_column"]) == ("group", "member_id")
    split = recipe_from(problem)["recipe"]["split"]
    assert (split["method"], split["column"]) == ("group", "member_id")

    # 時系列は時刻の順のまま, 列は渡す (重なりを知らせるため)
    sozai = answer(example_problem("総菜の作る量"), "ml.parts.0.group_column",
                   "store")
    split = recipe_from(sozai)["recipe"]["split"]
    assert (split["method"], split["column"]) == ("time", "store")

    # 目的の列は誤り, 無い列は答えられない
    wrong = answer(problem, "ml.parts.0.group_column", "churned")
    assert "ml.parts.0.group_column" in levels(wrong)["error"]
    with pytest.raises(ValueError, match="no_such"):
        answer(problem, "ml.parts.0.group_column", "no_such")

    # 書き戻したレシピと相手の列が食い違えば知らせる
    saved = apply_recipe(example_problem("解約の引き止め"),
                         recipe_from(example_problem("解約の引き止め"))[
                             "recipe"])["problem"]
    later = answer(saved, "ml.parts.0.group_column", "member_id")
    assert any("member_id" in note for note in recipe_from(later)["notes"])
    result = apply_recipe(later, recipe_from(later)["recipe"])
    assert [c["id"] for c in result["conflicts"]] == [
        "ml.parts.0.group_column"
    ]


def test_split_time_column() -> None:
    """
    分類でも時刻の順で分けて確かめる列を書くと, 時刻の順の分割と時刻の列を
    渡すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 申し込みの日で分ける (学習に使わない列にしても誤りにしない)
    problem = example_problem("解約の引き止め")
    columns = get_value(problem, "information.columns") + ["applied_at"]
    problem = answer(problem, "information.columns", columns)
    forbidden = get_value(problem, "information.forbidden") or []
    problem = answer(problem, "information.forbidden",
                     forbidden + ["applied_at"])
    problem = answer(problem, "ml.parts.0.split_time_column", "applied_at")
    assert levels(problem)["error"] == []
    data = handoff(problem)["data"]
    assert (data["split"], data["time_column"]) == ("time", "applied_at")
    start = recipe_from(problem)
    assert start["recipe"]["split"]["method"] == "time"

    # 書き戻したレシピが時刻の順でなければ知らせる
    other = dict(start["recipe"], split={"method": "stratified"})
    result = apply_recipe(problem, other)
    assert "ml.parts.0.split_time_column" in [
        c["id"] for c in result["conflicts"]
    ]

    # 目的の列は誤り
    wrong = answer(problem, "ml.parts.0.split_time_column", "churned")
    assert "ml.parts.0.split_time_column" in levels(wrong)["error"]
    same = answer(problem, "ml.parts.0.group_column", "applied_at")
    assert "ml.parts.0.group_column" in levels(same)["error"]

    # 分けるのにだけ使う時刻の列は, 決めていない列の知らせに出さない
    loose = answer(problem, "information.forbidden", forbidden or None)
    assert not any("applied_at" in item["message"]
                   for item in check(loose) if item["level"] == "warning")


def test_group_time_and_stratified_group() -> None:
    """
    ML-EDA のグループと時刻の組 (group_time) とグループの層化
    (stratified_group) の分割を書き戻したとき, 時刻の順を守る分割として
    扱い, 列と分類の要るところだけを食い違いにすることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 時系列の予測で相手ごとと時刻の順は, 偽の警告にならない
    sozai = answer(example_problem("総菜の作る量"), "ml.parts.0.group_column",
                   "store")
    recipe = recipe_from(sozai)["recipe"]
    timed = dict(recipe, split={**recipe["split"], "method": "group_time"})
    result = apply_recipe(sozai, timed)
    assert [(c["level"], c["id"]) for c in result["conflicts"]] == [
        ("info", "ml.parts.0.task")
    ]
    assert "重なりえます" in result["conflicts"][0]["message"]
    assert handoff(result["problem"])["data"]["split"] == "time"

    # 時系列の予測で相手ごとの層別は誤り (と, 時刻の順でない警告)
    layered = dict(recipe, split={**recipe["split"],
                                  "method": "stratified_group"})
    found = {(c["level"], c["id"]) for c in apply_recipe(
        sozai, layered
    )["conflicts"]}
    assert found == {("warning", "ml.parts.0.task"),
                     ("error", "ml.parts.0.task")}

    # 分類で時刻の順を決めたなら group_time は警告しない. 層別は分類なら使える
    problem = example_problem("解約の引き止め")
    columns = get_value(problem, "information.columns") + ["applied_at"]
    problem = answer(problem, "information.columns", columns)
    problem = answer(problem, "information.forbidden",
                     (get_value(problem, "information.forbidden") or [])
                     + ["applied_at"])
    problem = answer(problem, "ml.parts.0.split_time_column", "applied_at")
    problem = answer(problem, "ml.parts.0.group_column", "member_id")
    recipe = recipe_from(problem)["recipe"]
    for method, ids in (("group_time", ["ml.parts.0.split_time_column"]),
                        ("stratified_group",
                         ["ml.parts.0.split_time_column"])):
        written = dict(recipe, split={**recipe["split"], "method": method})
        conflicts = apply_recipe(problem, written)["conflicts"]
        assert [c["id"] for c in conflicts] == ids, method

    # 相手をまとめる列の無いグループの分割は誤り
    plain = example_problem("解約の引き止め")
    recipe = recipe_from(plain)["recipe"]
    for method in ("group", "group_time", "stratified_group"):
        written = dict(recipe, split={"method": method})
        assert ("error", "ml.parts.0.group_column") in {
            (c["level"], c["id"])
            for c in apply_recipe(plain, written)["conflicts"]
        }, method

    # 受け渡しの分割の名前に, 2 つの説明がある
    names = {item["value"]: item["label"] for item in SPLITS}
    assert "時刻" in names["group_time"]
    assert "分類だけ" in names["stratified_group"]

    # グループを使う分割が, 公開の分割の名前の目印から分かる
    grouped = {item["value"] for item in SPLITS if item["grouped"]}
    assert grouped == {"group", "group_time", "stratified_group"}
    assert all(isinstance(item["grouped"], bool) for item in SPLITS)


def test_split_names_match_ml_eda() -> None:
    """
    受け渡しの分割の名前とファイルの組の書き出しの分割が, ML-EDA のレシピ
    と書き出しの分割のしかたにあることを確かめる関数 (ML-EDA が無ければ
    飛ばす)

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # レシピの分割のしかたと, ファイルの組の書き出しの分割のしかた
    recipe = pytest.importorskip("ml_eda.recipe")
    items = pytest.importorskip("ml_eda.items")
    pipeline = pytest.importorskip("ml_eda.pipeline")
    from ml_quest.bridge import SPLIT_METHODS

    names = {item["value"] for item in SPLITS}
    assert names <= set(recipe.SPLIT_METHODS)
    assert set(recipe.SPLIT_METHODS) <= names
    assert set(SPLIT_METHODS) == set(items.SPLIT_METHODS)

    # グループを使う分割の目印が, ML-EDA のグループの分割と同じ組
    grouped = {item["value"] for item in SPLITS if item["grouped"]}
    assert grouped == set(pipeline.GROUP_METHODS)
