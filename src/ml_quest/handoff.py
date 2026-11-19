from typing import Any, Optional

from ml_quest.check import check, is_column_part
from ml_quest.problem import (
    DECISION_NAMES, DECISION_OPTIONS, FIELD_NAMES, OPTION_NAMES,
    active_value, data_location, get_value, upgrade_problem
)
from ml_quest.questions import CONSUMERS, METRICS, PART, ROLES, TASKS

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
        decision["positive"] = active_value(problem, f"{PART}.positive")

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
    series = active_value(problem, f"{PART}.series_by") or []
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
    # 渡せる問題の設定か (版 1 は先に移す)
    problem = upgrade_problem(problem)
    if get_value(problem, "ml.use") is not True:
        raise ValueError("機械学習を使わない問題の設定は, 後ろの段階に渡す"
                         "ものがありません")
    errors = [item for item in check(problem) if item["level"] == "error"]
    if errors:
        raise ValueError(
            "問題の設定に直すところがあります: "
            + " / ".join(item["message"] for item in errors[:3])
        )

    # 受け持ちの答えと, 最良を選ぶ指標 (その他は名前と向き)
    value = active_value
    part = part_of(problem)
    training = training_of(problem)
    training.update(role=part["role"], target_kind=part["target_kind"])
    if not is_column_part(problem):
        stages = {
            "part": part, "data": column_data(problem, None),
            "model": {"task": None, "outputs": None, "role": part["role"],
                      "target_kind": part["target_kind"]},
            "training": training,
            "inference": {"decision": None, "conclusion": {},
                          "horizon": None},
        }
        return stages

    # 列の予測: 版 1 と同じ値に, 受け持ちと正解の種類を足す
    task = value(problem, f"{PART}.task")
    stages = {
        "part": part,
        "data": column_data(problem, task),
        "model": {"task": task,
                  "outputs": None if task == "classification" else 1,
                  "role": part["role"], "target_kind": part["target_kind"]},
        "training": training,
        "inference": {"decision": decision_of(problem),
                      "conclusion": conclusion_of(problem),
                      "horizon": value(problem, f"{PART}.horizon")},
    }

    return stages


def part_of(problem: dict) -> dict:
    """
    受け持ちの型に依らない答えをまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    part: dict
        role, unit, output, target_kind, target_source, target_process
        (今の処理の名前と設定. 出どころが今の処理でなければ None), method
    """
    # 要らなくなった問いの古い答えは除外
    value = active_value
    name = value(problem, f"{PART}.target.process.name")
    part = {
        "role": value(problem, f"{PART}.role"),
        "unit": value(problem, f"{PART}.unit"),
        "output": value(problem, f"{PART}.output"),
        "target_kind": value(problem, f"{PART}.target.kind"),
        "target_source": value(problem, f"{PART}.target.source"),
        "target_process": {
            "name": name,
            "settings": value(problem, f"{PART}.target.process.settings"),
        } if name else None,
        "method": value(problem, f"{PART}.method"),
    }

    return part


def training_of(problem: dict) -> dict:
    """
    学習・評価の段階に渡す, 最良を選ぶ指標と監視する指標を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    training: dict
        metric (その他は書いた名前), monitor (valid/<指標>), mode (min か
        max)
    """
    # 学習の監視は, バッチの平均で書ける指標 (RMSE は二乗誤差で選ぶ).
    # その他は書いた名前と向き
    metric = active_value(problem, f"{PART}.evaluation.metric")
    better = next((item["better"] for item in METRICS
                   if item["value"] == metric), "lower")
    if metric == "other":
        metric = active_value(problem, f"{PART}.evaluation.name")
        better = active_value(problem, f"{PART}.evaluation.better")
    monitor = "valid/mse" if metric == "rmse" else f"valid/{metric}"
    training = {"metric": metric, "monitor": monitor,
                "mode": "max" if better == "higher" else "min"}

    return training


def column_data(problem: dict, task: Optional[str]) -> dict:
    """
    データの分析の段階に渡す値を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    task: Optional[str]
        列の予測のタスク (列の予測でなければ None)

    Returns
    ----------
    data: dict
        目的の列, タスク, 分割, 使う列と除く列 (列の予測でなければ同じ
        名前の空の値) と, データの場所と読み方
    """
    # 要らなくなった問いの古い答えは除外
    value = active_value
    data = {
        "target": value(problem, f"{PART}.target.column"),
        "task": task,
        "time_column": value(problem, f"{PART}.time_column")
        or value(problem, f"{PART}.split_time_column"),
        "series_by": value(problem, f"{PART}.series_by") or [],
        "split": None if task is None
        else "time" if task == "forecasting"
        or value(problem, f"{PART}.split_time_column")
        else "group" if value(problem, f"{PART}.group_column")
        else "stratified" if task == "classification" else "random",
        "group_column": value(problem, f"{PART}.group_column"),
        "min_lag": value(problem, f"{PART}.available_lag"),
        "known": value(problem, "information.known") or [],
        "source": {"source": data_location(problem),
                   "loader": value(problem, "data.loader"),
                   "options": value(problem, "data.options")}
        if value(problem, "data.source") else None,
        "drop": (value(problem, "information.unknown") or [])
        + (value(problem, "information.forbidden") or []),
    }

    return data


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
    # 課題と指標 (要らなくなった問いの古い答えは使わない. 版 1 は先に移す)
    problem = upgrade_problem(problem)
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
    guard = value(problem, "issue.guard") or {}
    if isinstance(guard, dict) and guard.get("name"):
        sentences.append(guard_text(guard))

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
    what = value(problem, f"{PART}.what")
    predicts = value(problem, f"{PART}.role") in (None, "predict")
    if what:
        roles = {item["value"]: item["label"].split(" ")[0] for item in ROLES}
        tasks = {item["value"]: item["label"].split(" ")[0] for item in TASKS}
        role = roles.get(str(value(problem, f"{PART}.role")), "")
        task = tasks.get(str(value(problem, f"{PART}.task")), "")
        target = value(problem, f"{PART}.target.column")
        details = [part for part in (
            task, f"目的の列 {target}" if target else ""
        ) if part]
        head = ("機械学習で予測すること" if predicts
                else f"機械学習の受け持ち ({role})" if role
                else "機械学習の受け持ち")
        sentences.append(
            f"{head}: {what}"
            + (f" ({'. '.join(details)})" if details else "")
        )
    consumer = value(problem, f"{PART}.consumer")
    names = {item["value"]: item["label"] for item in CONSUMERS}
    if isinstance(consumer, str) and consumer in names:
        downstream = value(problem, f"{PART}.downstream")
        sentences.append(
            f"{'予測' if predicts else '出力'}の使い道: {names[consumer]}"
            + (f". 後段が求めるもの: {downstream}" if downstream else "")
        )
    decision = value(problem, "action.decision")
    if isinstance(decision, dict) and decision.get("type") in DECISION_NAMES:
        sentences.append(decision_text(decision))
    constraints = value(problem, "ml.constraints")
    if isinstance(constraints, list) and constraints:
        sentences.append("守る制約: " + ", ".join(map(str, constraints)))
    if value(problem, "verification.success"):
        sentences.append(
            f"成功の基準: {value(problem, 'verification.success')}"
        )

    return sentences


def guard_text(guard: dict) -> str:
    """
    守る指標を文にする関数

    Parameters
    ----------
    guard: dict
        守る指標 (name, unit, better, line, measure)

    Returns
    ----------
    text: str
        「守る指標: 名前 (単位. x を超えない. 測り方)」の形
    """
    # 分かる値だけの並び (線の向きは指標の向きから)
    parts = []
    if guard.get("unit"):
        parts.append(str(guard["unit"]))
    line = guard.get("line")
    if line is not None and guard.get("better") in ("lower", "higher"):
        parts.append(f"{number_text(line)} を"
                     + ("超えない" if guard["better"] == "lower"
                        else "下回らない"))
    elif line is not None:
        parts.append(f"守る線 {number_text(line)}")
    if guard.get("measure"):
        parts.append(f"測り方: {guard['measure']}")
    text = f"守る指標: {str(guard['name']).strip()}" + (
        f" ({'. '.join(parts)})" if parts else ""
    )

    return text


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
    # 型の名前と, 項目の名前と値 (数は桁区切り. 選んだ値は名前. 確率の差
    # で書いた効果は防げる割合と呼ばない)
    names = dict(FIELD_NAMES)
    if decision.get("effect_kind") == "uplift":
        names["effect"] = "確率の差"
    items: list[Any] = [
        f"{names.get(key, key)} " + number_text(
            OPTION_NAMES.get(value, value)
            if key in OPTION_FIELDS and isinstance(value, str) else value
        )
        for key, value in decision.items() if key != "type"
    ]
    text = f"意思決定: {DECISION_NAMES[decision['type']]}" + (
        f" ({', '.join(items)})" if items else ""
    )

    return text
