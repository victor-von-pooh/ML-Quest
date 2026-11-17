import ast
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ml_quest import (
    EXAMPLES, QUESTIONS, answer, check, describe, example_problem,
    get_value, handoff, is_ready, load_problem, make_project, new_problem,
    questions, read_columns, save_problem
)


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
    # 3 つの見本
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
    # 空の問題の設定では, 分類だけの問いと予測だけの問いは要らない
    problem = new_problem("試し")
    items = {item["id"]: item for item in questions(problem)}
    assert len(items) == len(QUESTIONS)
    assert items["issue.pain"]["applies"] is True
    assert items["ml.target"]["applies"] is False
    assert items["ml.time_column"]["applies"] is False

    # 予測を選ぶと時刻の列の問いが要り, 指標は回帰と予測のものだけ
    problem = answer(problem, "ml.use", True)
    problem = answer(problem, "ml.task", "forecasting")
    items = {item["id"]: item for item in questions(problem)}
    assert items["ml.time_column"]["applies"] is True
    assert [c["value"] for c in items["evaluation.metric"]["choices"]] == [
        "mae", "rmse"
    ]

    # 元の問題の設定は変えない
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
        answer(problem, "ml.task", "clustering")
    with pytest.raises(ValueError, match="はいかいいえ"):
        answer(problem, "ml.use", "yes")
    with pytest.raises(ValueError, match="書けません"):
        answer(problem, "action.decision",
               {"type": "quantity", "miss_cost": 1})
    columns = answer(problem, "information.columns", ["a", "b", "a"])
    assert get_value(columns, "information.columns") == ["a", "b"]
    with pytest.raises(ValueError, match="列にありません"):
        answer(columns, "ml.target", "c")


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
    problem = answer(problem, "ml.task", "classification")
    problem = answer(problem, "information.known",
                     ["sales", "promo", "customers"])
    problem = answer(problem, "issue.kpi.target", 1500000)
    found = levels(problem)
    assert "action.decision" in found["error"]
    assert "information.known" in found["error"]
    assert "information.unknown" in found["error"]
    assert "issue.kpi.target" in found["error"]
    assert "evaluation.metric" in found["error"]
    assert is_ready(problem) is False
    with pytest.raises(ValueError, match="直すところ"):
        handoff(problem)

    # 損得の値の範囲と根拠, 選ぶ型を過去のデータだけで確かめる
    churn = example_problem("解約の引き止め")
    churn = answer(churn, "action.decision", {
        "type": "select", "value": 24000, "cost": 1500, "effect": 1.5,
    })
    churn = answer(churn, "action.basis", None)
    churn = answer(churn, "verification.method", "backtest")
    churn = answer(churn, "verification.backtest", "過去 1 年")
    churn = answer(churn, "ml.positive", None)
    found = levels(churn)
    assert "action.decision" in found["error"]
    assert "ml.positive" in found["error"]
    assert "action.basis" in found["warning"]
    assert "verification.method" in found["warning"]

    # 機械学習を使わない出口は, 課題と分解と理由だけで足りる
    exit_only = new_problem("出口")
    for key, value in EXAMPLES[
        "問い合わせの振り分け (機械学習を使わない)"
    ]:
        exit_only = answer(exit_only, key, value)
    assert levels(exit_only)["error"] == []
    with pytest.raises(ValueError, match="渡すもの"):
        handoff(exit_only)

    # 足りない答え
    missing = levels(new_problem("空"))["error"]
    assert "issue.pain" in missing and "ml.use" in missing


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
                                  "mode": "min"}
    assert stages["inference"]["decision"]["type"] == "quantity"
    assert stages["inference"]["conclusion"] == {"group_by": "store"}

    # 分類: 選ぶ型に当たりのクラスを足す. 交差エントロピーは小さいほど良い
    stages = handoff(example_problem("解約の引き止め"))
    assert stages["data"]["split"] == "random"
    assert stages["model"]["outputs"] is None
    assert stages["training"]["mode"] == "min"
    assert stages["inference"]["decision"]["positive"] == "yes"

    # 正解率は大きいほど良い. 意思決定しないなら None
    problem = answer(example_problem("解約の引き止め"), "evaluation.metric",
                     "accuracy")
    problem = answer(problem, "action.decision", {"type": "none"})
    stages = handoff(problem)
    assert stages["training"]["mode"] == "max"
    assert stages["inference"]["decision"] is None


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
    # ディレクトリにも書ける. 書き直すと更新の時刻だけが変わる
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
