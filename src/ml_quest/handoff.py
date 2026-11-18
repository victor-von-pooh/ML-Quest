from typing import Any, Optional

from ml_quest.check import check
from ml_quest.problem import (
    DECISION_NAMES, DECISION_OPTIONS, FIELD_NAMES, OPTION_NAMES,
    active_value, data_location, get_value
)
from ml_quest.questions import METRICS, TASKS

# 選んで書く意思決定の項目 (値を名前にして文にする)
OPTION_FIELDS = {
    field for options in DECISION_OPTIONS.values() for field in options
}


def decision_of(problem: dict) -> Optional[dict]:
    """
    問題の設定から, ML-Inference の意思決定の形を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    decision: Optional[dict]
        意思決定 (選ぶと振り分けには当たりのクラスを足す. 意思決定しない
        なら None)
    """
    # 損得と制約はそのまま, 当たりのクラスは機械学習の受け持ちから
    raw = active_value(problem, "action.decision")
    if not raw or raw.get("type") == "none":
        return None
    decision = dict(raw)
    if decision["type"] in ("select", "triage"):
        decision["positive"] = active_value(problem, "ml.positive")

    # 今のやり方の列 (量は今ある量の無い場合だけ. 問題の設定の列は置いた
    # 量, ML-Inference は今ある量に足した量)
    current = active_value(problem, "action.current") or {}
    if current.get("type") == "column" and (
        decision["type"] != "quantity" or not decision.get("stock")
    ):
        decision["current"] = current["column"]

    return decision


def conclusion_of(problem: dict) -> dict:
    """
    問題の設定から, ML-Inference の結論の決め方の既定値を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    conclusion: dict
        結論の決め方 (系列を分ける列があれば, 最初の列でまとめる)
    """
    # 系列ごとのまとめ (最初の列. 店と品なら店ごと)
    series = active_value(problem, "ml.series_by") or []
    conclusion = {"group_by": series[0]} if series else {}

    return conclusion


def handoff(problem: dict) -> dict:
    """
    後ろの段階ごとに, 問題の設定から決まる値をまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    stages: dict
        data (データの分析), model (モデルの作成), training (学習・評価),
        inference (推論) ごとの値

    Raises
    ----------
    ValueError
        機械学習を使わない場合と, 確かめに error がある場合
    """
    # 渡せる問題の設定か
    if get_value(problem, "ml.use") is not True:
        raise ValueError("機械学習を使わない問題の設定は, 後ろの段階に渡す"
                         "ものがありません")
    errors = [item for item in check(problem) if item["level"] == "error"]
    if errors:
        raise ValueError(
            "問題の設定に直すところがあります: "
            + " / ".join(item["message"] for item in errors[:3])
        )

    # データの分析: 目的の列, タスク, 分割, 使う列と除く列 (要らなくなった
    # 問いの古い答えは除外)
    value = active_value
    task = value(problem, "ml.task")
    metric = value(problem, "evaluation.metric")
    better = next((item["better"] for item in METRICS
                   if item["value"] == metric), "lower")
    data = {
        "target": value(problem, "ml.target"),
        "task": task,
        "time_column": value(problem, "ml.time_column"),
        "series_by": value(problem, "ml.series_by") or [],
        "split": "time" if task == "forecasting" else "random",
        "min_lag": value(problem, "ml.available_lag"),
        "known": value(problem, "information.known") or [],
        "source": {"source": data_location(problem),
                   "loader": value(problem, "data.loader"),
                   "options": value(problem, "data.options")}
        if value(problem, "data.source") else None,
        "drop": (value(problem, "information.unknown") or [])
        + (value(problem, "information.forbidden") or []),
    }

    # 学習の監視は, バッチの平均で書ける指標 (RMSE は二乗誤差で選ぶ)
    monitor = "valid/mse" if metric == "rmse" else f"valid/{metric}"
    stages = {
        "data": data,
        "model": {"task": task,
                  "outputs": None if task == "classification" else 1},
        "training": {"metric": metric, "monitor": monitor,
                     "mode": "max" if better == "higher" else "min"},
        "inference": {"decision": decision_of(problem),
                      "conclusion": conclusion_of(problem),
                      "horizon": value(problem, "ml.horizon")},
    }

    return stages


def describe(problem: dict) -> list[str]:
    """
    問題の設定を, 課題から行動までの筋道の文にする関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    sentences: list[str]
        要約の文 (答えた項目だけ)
    """
    # 課題と指標 (要らなくなった問いの古い答えは使わない)
    value = active_value
    sentences = []
    if value(problem, "issue.pain"):
        owner = value(problem, "issue.owner") or "誰か"
        sentences.append(f"{owner}の課題: {value(problem, 'issue.pain')}")
    if value(problem, "issue.current"):
        sentences.append(f"今のやり方: {value(problem, 'issue.current')}")
    kpi = value(problem, "issue.kpi") or {}
    if kpi.get("name"):
        sentences.append(kpi_text(kpi))

    # 決めること (予測より先) と機械学習の受け持ち
    if value(problem, "action.what"):
        sentences.append(
            f"{value(problem, 'action.who') or '誰か'}が"
            f"{value(problem, 'action.when') or 'いつか'}に"
            f"{value(problem, 'action.what')}を決める"
        )
    uses = value(problem, "ml.use")
    if uses is False:
        reason = value(problem, "ml.reason")
        sentences.append("機械学習は使わない" + (f": {reason}" if reason
                                                 else ""))
        return sentences
    if value(problem, "ml.predict"):
        tasks = {item["value"]: item["label"].split(" ")[0] for item in TASKS}
        task = tasks.get(str(value(problem, "ml.task")), "")
        target = value(problem, "ml.target")
        details = [part for part in (
            task, f"目的の列 {target}" if target else ""
        ) if part]
        sentences.append(
            f"機械学習で予測すること: {value(problem, 'ml.predict')}"
            + (f" ({'. '.join(details)})" if details else "")
        )
    decision = value(problem, "action.decision")
    if isinstance(decision, dict) and decision.get("type") in DECISION_NAMES:
        sentences.append(decision_text(decision))
    if value(problem, "verification.success"):
        sentences.append(
            f"成功の基準: {value(problem, 'verification.success')}"
        )

    return sentences


def kpi_text(kpi: dict) -> str:
    """
    指標を文にする関数

    Parameters
    ----------
    kpi: dict
        指標 (name, unit, current, target, better)

    Returns
    ----------
    text: str
        「指標: 名前 (単位. 今 x → 目指す y. 小さいほど良い)」の形
    """
    # 分かる値だけの並び
    parts = []
    if kpi.get("unit"):
        parts.append(str(kpi["unit"]))
    if kpi.get("current") is not None:
        parts.append(f"今 {number_text(kpi['current'])}")
    if kpi.get("target") is not None:
        parts.append(f"目指す {number_text(kpi['target'])}")
    if kpi.get("better") in ("lower", "higher"):
        parts.append("小さいほど良い" if kpi["better"] == "lower"
                     else "大きいほど良い")
    text = f"解けたと言える指標: {kpi['name']}" + (
        f" ({'. '.join(parts)})" if parts else ""
    )

    return text


def number_text(value: Any) -> str:
    """
    数を桁区切りで, 小数は 4 桁の有効数字で文にする関数

    Parameters
    ----------
    value: Any
        値 (数でなければそのまま文字列に)

    Returns
    ----------
    text: str
        文
    """
    # 整数は桁区切り, 小数は丸め
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, int):
        text = f"{value:,}"
    else:
        text = f"{value:,.4g}"

    return text


def decision_text(decision: dict) -> str:
    """
    意思決定の型と損得を文にする関数

    Parameters
    ----------
    decision: dict
        意思決定 (ML-Inference の形)

    Returns
    ----------
    text: str
        型と, 書いた損得と制約
    """
    # 型の名前と, 項目の名前と値 (数は桁区切り. 選んだ値は名前)
    items: list[Any] = [
        f"{FIELD_NAMES.get(key, key)} " + number_text(
            OPTION_NAMES.get(value, value) if key in OPTION_FIELDS
            else value
        )
        for key, value in decision.items() if key != "type"
    ]
    text = f"意思決定: {DECISION_NAMES[decision['type']]}" + (
        f" ({', '.join(items)})" if items else ""
    )

    return text
