from typing import Any

from ml_quest.problem import (
    DECISION_FIELDS, DECISION_NAMES, FORMAT_VERSION, applies, get_value,
    is_answered
)
from ml_quest.questions import METRICS, QUESTIONS

# 機械学習を使わない場合にも要る節 (課題, 指標, 分解, 使うかとその理由)
WITHOUT_ML = {1, 2, 3}

# 行動の型ごとに合うタスク
DECISION_TASKS = {
    "quantity": ["regression", "forecasting"],
    "select": ["classification"],
    "triage": ["classification"],
}


def finding(level: str, key: str, message: str) -> dict:
    """
    確かめの 1 件を作る関数

    Parameters
    ----------
    level: str
        error, warning, info のどれか
    key: str
        項目の場所
    message: str
        何をすればよいかの文

    Returns
    ----------
    item: dict
        重さと場所と文
    """
    # 画面と文書にそのまま出す
    item = {"level": level, "id": key, "message": message}

    return item


def check_missing(problem: dict) -> list[dict]:
    """
    要る問いの答えの空を一覧にする関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ (error)
    """
    # 機械学習を使わない出口では, 使うかまでの問いだけ
    uses = get_value(problem, "ml.use")
    found = []
    for question in QUESTIONS:
        if not question["required"] or not applies(problem, question):
            continue
        if uses is False and question["section"] not in WITHOUT_ML and (
            question["id"] not in ("ml.use", "ml.reason")
        ):
            continue
        if not is_answered(get_value(problem, question["id"])):
            found.append(finding(
                "error", question["id"],
                f"「{question['title']}」に答えてください"
            ))

    return found


def check_issue(problem: dict) -> list[dict]:
    """
    課題と指標と分解の筋道を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 目指す値の向き
    found = []
    better = get_value(problem, "issue.kpi.better")
    current = get_value(problem, "issue.kpi.current")
    target = get_value(problem, "issue.kpi.target")
    if better and isinstance(current, (int, float)) and isinstance(
        target, (int, float)
    ):
        wrong = target > current if better == "lower" else target < current
        if wrong:
            found.append(finding(
                "error", "issue.kpi.target",
                "目指す値が, 指標の向きと逆です (小さいほど良い指標なら, "
                "目指す値は今の値より小さくしてください)"
            ))
        elif target == current:
            found.append(finding(
                "warning", "issue.kpi.target",
                "目指す値が今の値と同じです. どれだけ良くなれば価値があるか"
                "を決めてください"
            ))

    # 機械学習で手伝う手と, 使うかの答えの食い違い
    levers = get_value(problem, "decomposition") or []
    helped = [lever for lever in levers if lever.get("ml")]
    uses = get_value(problem, "ml.use")
    if uses is True and levers and not helped:
        found.append(finding(
            "warning", "decomposition",
            "機械学習を使うのに, 機械学習で手伝う手がありません. どの手を"
            "手伝うのかを分解に書いてください"
        ))
    if uses is False and helped:
        found.append(finding(
            "info", "ml.use",
            "分解には機械学習で手伝う手がありますが, 使わないと決めました. "
            "代わりにどうするかを理由に書いておくと, 後で見直せます"
        ))
    if uses is True and levers and len(levers) == len(helped):
        found.append(finding(
            "info", "decomposition",
            "どの手も機械学習で手伝う形です. 機械学習を使わない手 (ルール, "
            "やり方の改善) も並べると, 比べて選べます"
        ))

    return found


def check_task(problem: dict) -> list[dict]:
    """
    タスクと行動と時刻の食い違いを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 行動の型に合うタスク
    found = []
    task = get_value(problem, "ml.task")
    decision = get_value(problem, "action.decision") or {}
    kind = decision.get("type")
    if task and kind in DECISION_TASKS and task not in DECISION_TASKS[kind]:
        names = {"classification": "分類", "regression": "回帰",
                 "forecasting": "予測 (時系列)"}
        allowed = "か".join(names[t] for t in DECISION_TASKS[kind])
        found.append(finding(
            "error", "action.decision",
            f"{DECISION_NAMES[kind]}には{allowed}のタスクが要ります "
            f"(今は{names[task]})"
        ))

    # 分類で行動を決めるなら当たりのクラス
    if task == "classification" and kind in ("select", "triage") and (
        not get_value(problem, "ml.positive")
    ):
        found.append(finding(
            "error", "ml.positive",
            f"{DECISION_NAMES[kind]}には当たりのクラス (行動の対象) が要り"
            "ます"
        ))

    # 時刻の列と目的の列
    target = get_value(problem, "ml.target")
    time_column = get_value(problem, "ml.time_column")
    if task == "forecasting" and time_column and time_column == target:
        found.append(finding(
            "error", "ml.time_column",
            "時刻の列が目的の列と同じです"
        ))
    series = get_value(problem, "ml.series_by") or []
    if target and target in series:
        found.append(finding(
            "error", "ml.series_by",
            "系列を分ける列に目的の列が入っています"
        ))

    return found


def check_information(problem: dict) -> list[dict]:
    """
    予測の時点で使える列の食い違いを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 目的の列そのものを使う, 分かる列と分からない列の重なり
    found = []
    target = get_value(problem, "ml.target")
    columns = get_value(problem, "information.columns") or []
    known = get_value(problem, "information.known") or []
    unknown = get_value(problem, "information.unknown") or []
    if target and target in known:
        found.append(finding(
            "error", "information.known",
            f"目的の列 {target} が予測の時点で分かる列に入っています. 分かる"
            "なら予測は要りません (過去の値はデータの分析で作るので, 目的の列"
            "そのものは入れない)"
        ))
    both = [name for name in known if name in unknown]
    if both:
        found.append(finding(
            "error", "information.unknown",
            f"{', '.join(both)} が分かる列と分からない列の両方にあります"
        ))

    # 決めていない列 (目的, 時刻, 系列の列は除く)
    roles = {target, get_value(problem, "ml.time_column"),
             *(get_value(problem, "ml.series_by") or [])}
    undecided = [name for name in columns
                 if name not in known and name not in unknown
                 and name not in roles]
    if undecided and known:
        found.append(finding(
            "warning", "information.known",
            f"{', '.join(undecided)} を予測の時点で使ってよいか決めていません"
            " (分かる列か分からない列に入れてください)"
        ))

    return found


def check_decision(problem: dict) -> list[dict]:
    """
    損得と制約の値と根拠を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 型ごとの要る項目と, 数の値の範囲
    found: list[dict] = []
    decision = get_value(problem, "action.decision")
    if not decision:
        return found
    kind = decision["type"]
    for field, required in DECISION_FIELDS[kind].items():
        value: Any = decision.get(field)
        if value is None:
            if required:
                found.append(finding(
                    "error", "action.decision",
                    f"{DECISION_NAMES[kind]}の {field} を書いてください"
                ))
            continue
        if not isinstance(value, (int, float)):
            continue
        if field == "effect" and not 0 < value <= 1:
            found.append(finding(
                "error", "action.decision",
                "effect (行動で防げる割合) は 0 より大きく 1 以下にしてください"
            ))
        elif field in ("stock",) and value < 0:
            found.append(finding(
                "error", "action.decision", "stock は 0 以上にしてください"
            ))
        elif field not in ("effect", "stock") and value <= 0:
            found.append(finding(
                "error", "action.decision",
                f"{field} は 0 より大きくしてください"
            ))

    # 根拠の無い損得
    if kind != "none" and not get_value(problem, "action.basis"):
        found.append(finding(
            "warning", "action.basis",
            "損得の値の根拠が空です. 根拠の無い値では, 意思決定はもっともら"
            "しいだけになります"
        ))
    if kind == "select":
        found.append(finding(
            "info", "verification.method",
            "対象を選ぶ行動の効果 (effect) は仮定です. 過去のデータからは"
            "分からないので, 実際に試して確かめてください"
        ))

    return found


def check_evaluation(problem: dict) -> list[dict]:
    """
    学習の指標と確かめ方と運用を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 指標がタスクに合うか
    found = []
    task = get_value(problem, "ml.task")
    metric = get_value(problem, "evaluation.metric")
    allowed = {item["value"]: item["tasks"] for item in METRICS}
    if task and metric and task not in allowed.get(metric, []):
        found.append(finding(
            "error", "evaluation.metric",
            f"指標 {metric} はこのタスクに合いません"
        ))
    if metric and not get_value(problem, "evaluation.link"):
        found.append(finding(
            "warning", "evaluation.link",
            "学習の指標と業務の指標のつながりが空です. 指標が良くなっても"
            "課題が解けるとは限りません"
        ))

    # 行動が結果を変える判断を, たどり直しだけで確かめる
    decision = get_value(problem, "action.decision") or {}
    method = get_value(problem, "verification.method")
    if decision.get("type") == "select" and method == "backtest":
        found.append(finding(
            "warning", "verification.method",
            "対象を選ぶ行動 (引き止めなど) の効果は, 過去のデータでたどり直す"
            "だけでは分かりません. 実際に試すことも考えてください"
        ))
    if not get_value(problem, "operation.stop"):
        found.append(finding(
            "warning", "operation.stop",
            "止める条件が空です. 効かなかった場合に今のやり方へ戻す基準を"
            "決めておくと, 試しやすくなります"
        ))

    return found


def check(problem: dict) -> list[dict]:
    """
    問題の設定の, 足りない答えと食い違いを一覧にする関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ (level, id, message) の並び. error, warning, info の順
    """
    # 新しい版は読める項目だけを使う
    found = []
    version = problem.get("format_version")
    if isinstance(version, int) and version > FORMAT_VERSION:
        found.append(finding(
            "info", "format_version",
            f"このファイルは新しい版 ({version}) の ML-Quest で書かれました. "
            "読める項目だけを使います"
        ))

    # 答えの空と筋道 (機械学習を使わない出口では課題と分解まで)
    found += check_missing(problem)
    found += check_issue(problem)
    if get_value(problem, "ml.use") is True:
        found += check_task(problem)
        found += check_information(problem)
        found += check_decision(problem)
        found += check_evaluation(problem)
    found = [item for level in ("error", "warning", "info")
             for item in found if item["level"] == level]

    return found


def is_ready(problem: dict) -> bool:
    """
    問題の設定を後ろの段階に渡せるか (error が無いか) を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    ready: bool
        error が無く, 機械学習を使うと決めたか
    """
    # 使わない出口は渡すものが無い
    ready = get_value(problem, "ml.use") is True and not any(
        item["level"] == "error" for item in check(problem)
    )

    return ready
