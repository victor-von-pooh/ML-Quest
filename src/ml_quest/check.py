from typing import Any

from ml_quest.problem import (
    DECISION_FIELDS, DECISION_NAMES, DECISION_OPTIONS, FORMAT_VERSION,
    active_value, applies, get_value, is_answered
)
from ml_quest.questions import METRICS, QUESTIONS, TASKS

# 行動の型ごとに合うタスク
DECISION_TASKS = {
    "quantity": ["regression", "forecasting"],
    "select": ["classification"],
    "triage": ["classification"],
}

# タスクの名前
TASK_NAMES = {"classification": "分類", "regression": "回帰",
              "forecasting": "時系列の予測"}

# 列の名前の並びの問い
COLUMN_LISTS = ["information.columns", "ml.series_by", "information.known",
                "information.unknown", "information.forbidden"]


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
    # 画面と文書にそのままの形
    item = {"level": level, "id": key, "message": message}

    return item


def is_names(value: Any) -> bool:
    """
    文字列の並びかを返す関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    names: bool
        None か, 文字列だけの並びなら True
    """
    # 手で書いた YAML の文字列 1 つなどは誤り
    names = value is None or (
        isinstance(value, list) and all(isinstance(v, str) for v in value)
    )

    return names


def check_shape(problem: dict) -> list[dict]:
    """
    手で書いたファイルや新しい版の, 値の形の崩れを一覧にする関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ (error. 形が崩れていれば他の確かめはしない)
    """
    # 分解, タスク, 指標, 意思決定, 列の並び
    found = []
    levers = get_value(problem, "decomposition")
    if levers is not None and not (isinstance(levers, list) and all(
        isinstance(lever, dict) for lever in levers
    )):
        found.append(finding("error", "decomposition",
                             "decomposition は要因と手の辞書の並びにして"
                             "ください"))
    for key in ("ml.task", "ml.target", "ml.time_column", "ml.positive",
                "evaluation.metric", "issue.kpi.better"):
        value = get_value(problem, key)
        if value is not None and not isinstance(value, str):
            found.append(finding("error", key,
                                 f"{key} は文字列にしてください"))
    task = get_value(problem, "ml.task")
    if isinstance(task, str) and task not in TASK_NAMES:
        found.append(finding("error", "ml.task",
                             f"ml.task の {task} は知らないタスクです"))
    metric = get_value(problem, "evaluation.metric")
    if isinstance(metric, str) and metric not in [m["value"]
                                                   for m in METRICS]:
        found.append(finding("error", "evaluation.metric",
                             f"evaluation.metric の {metric} は知らない指標"
                             "です"))
    decision = get_value(problem, "action.decision")
    if decision is not None and not (
        isinstance(decision, dict) and isinstance(decision.get("type"), str)
        and decision["type"] in DECISION_FIELDS
    ):
        found.append(finding("error", "action.decision",
                             "action.decision の type は "
                             f"{', '.join(DECISION_FIELDS)} のどれかにして"
                             "ください"))
    for key in COLUMN_LISTS:
        if not is_names(get_value(problem, key)):
            found.append(finding("error", key,
                                 f"{key} は列の名前の並びにしてください"))

    return found


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
        確かめ (error. 機械学習を使わない出口では, 使うかまでの問いだけ)
    """
    # 今の答えのもとで要る問いだけ
    found = []
    for question in QUESTIONS:
        if not question["required"] or not applies(problem, question):
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
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                  for v in (current, target))
    if better in ("lower", "higher") and numbers:
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
    task = active_value(problem, "ml.task")
    decision = active_value(problem, "action.decision") or {}
    kind = decision.get("type")
    if task and kind in DECISION_TASKS and task not in DECISION_TASKS[kind]:
        allowed = "か".join(TASK_NAMES[t] for t in DECISION_TASKS[kind])
        found.append(finding(
            "error", "action.decision",
            f"{DECISION_NAMES[kind]}には{allowed}のタスクが要ります "
            f"(今は{TASK_NAMES[task]})"
        ))

    # 分類で行動を決めるなら当たりのクラス
    if task == "classification" and kind in ("select", "triage") and (
        not active_value(problem, "ml.positive")
    ):
        found.append(finding(
            "error", "ml.positive",
            f"{DECISION_NAMES[kind]}には当たりのクラス (行動の対象) が要り"
            "ます"
        ))

    # 時刻の列と, 目的の列と系列の列の重なり
    target = active_value(problem, "ml.target")
    time_column = active_value(problem, "ml.time_column")
    series = active_value(problem, "ml.series_by") or []
    if time_column and time_column == target:
        found.append(finding("error", "ml.time_column",
                             "時刻の列が目的の列と同じです"))
    if time_column and time_column in series:
        found.append(finding("error", "ml.series_by",
                             "系列を分ける列に時刻の列が入っています"))
    if target and target in series:
        found.append(finding("error", "ml.series_by",
                             "系列を分ける列に目的の列が入っています"))

    # 何期先までと, 何期前まで分かるか (1 以上の整数)
    for key, name in (("ml.horizon", "何期先まで"),
                      ("ml.available_lag", "何期前の分まで分かるか")):
        value = active_value(problem, key)
        if value is not None and not (
            isinstance(value, int) and not isinstance(value, bool)
            and value >= 1
        ):
            found.append(finding("error", key,
                                 f"{name}は 1 以上の整数にしてください"))

    return found


def check_columns(problem: dict) -> list[dict]:
    """
    答えた列の名前が, 使えるデータの列にあるかを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ (列を答え直した後に残った古い名前は error)
    """
    # 使えるデータの列が無ければ確かめなし
    found: list[dict] = []
    columns = active_value(problem, "information.columns") or []
    if not columns:
        return found
    for key in ("ml.target", "ml.time_column"):
        name = active_value(problem, key)
        if name and name not in columns:
            found.append(finding("error", key,
                                 f"{name} は使えるデータの列にありません"))
    for key in COLUMN_LISTS[1:]:
        missing = [name for name in active_value(problem, key) or []
                   if name not in columns]
        if missing:
            found.append(finding(
                "error", key,
                f"{', '.join(missing)} は使えるデータの列にありません"
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
    # 目的の列そのものを使う, 分かる列と使わない列の重なり
    found = []
    target = active_value(problem, "ml.target")
    time_column = active_value(problem, "ml.time_column")
    columns = active_value(problem, "information.columns") or []
    known = active_value(problem, "information.known") or []
    unknown = active_value(problem, "information.unknown") or []
    forbidden = active_value(problem, "information.forbidden") or []
    if target and target in known:
        found.append(finding(
            "error", "information.known",
            f"目的の列 {target} が予測の時点で分かる列に入っています. 分かる"
            "なら予測は要りません (過去の値はデータの分析で作るので, 目的の列"
            "そのものは入れない)"
        ))
    for key, others in (("information.unknown", unknown),
                        ("information.forbidden", forbidden)):
        both = [name for name in known if name in others]
        if both:
            found.append(finding(
                "error", key,
                f"{', '.join(both)} が分かる列と使わない列の両方にあります"
            ))
    if time_column and time_column in unknown + forbidden:
        found.append(finding(
            "error", "ml.time_column",
            "時刻の列が使わない列に入っています (時刻は予測の時点で分かる)"
        ))

    # 決めていない列 (目的, 時刻, 系列の列は除く)
    roles = {target, time_column,
             *(active_value(problem, "ml.series_by") or [])}
    undecided = [name for name in columns
                 if name not in known + unknown + forbidden
                 and name not in roles]
    if undecided and known:
        found.append(finding(
            "warning", "information.known",
            f"{', '.join(undecided)} を予測の時点で使ってよいか決めていません"
            " (分かる列か, 分からない列か, 使ってはいけない列に入れてくださ"
            "い)"
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
    # 型ごとの要る項目と, 数の値の範囲と, 列の名前
    found: list[dict] = []
    decision = active_value(problem, "action.decision")
    if not decision:
        return found
    kind = decision["type"]
    columns = active_value(problem, "information.columns") or []
    uplift = kind == "select" and decision.get("effect_kind") == "uplift"
    for field, required in DECISION_FIELDS[kind].items():
        value: Any = decision.get(field)
        if value is None:
            if required:
                found.append(finding(
                    "error", "action.decision",
                    f"{DECISION_NAMES[kind]}の {field} を書いてください"
                ))
            continue
        if isinstance(value, str):
            found += column_amount(field, value, columns)
            continue
        if field == "effect" and uplift:
            found += uplift_range(value)
            continue
        found += amount_range(field, value)
    found += option_findings(decision)

    # 今のやり方の書き方と行動の型 (k 期前と平均は量だけ)
    current = active_value(problem, "action.current") or {}
    if current.get("type") in ("lag", "mean") and kind != "quantity":
        found.append(finding(
            "error", "action.current",
            f"{DECISION_NAMES[kind]}の今のやり方は, 行動したかの列で書いて"
            "ください (k 期前と平均は量を決める行動だけ)"
        ))
    if current.get("type") == "column":
        if current.get("column") in (
            active_value(problem, "information.known") or []
        ):
            found.append(finding(
                "warning", "action.current",
                f"今のやり方の列 {current['column']} を学習に使ってよい列に"
                "しています. 今のやり方の結果は予測の後に決まるので, ふつう"
                "予測の時点で分かりません (今のやり方をまねるだけになります)"
            ))
        if kind == "quantity" and decision.get("stock"):
            found.append(finding(
                "info", "action.current",
                "今ある量があるので, 今のやり方の列 (置いた量) は推論の"
                "たどり直しには渡しません (推論の段階の今のやり方は, 今ある"
                "量に足した量のため)"
            ))

    # 根拠の無い損得と, 仮定の効果
    if kind != "none" and not get_value(problem, "action.basis"):
        found.append(finding(
            "warning", "action.basis",
            "損得の値の根拠が空です. 根拠の無い値では, 意思決定はもっともら"
            "しいだけになります"
        ))
    if uplift:
        found.append(finding(
            "info", "verification.method",
            "対象を選ぶ行動の効果 (effect) は効果のモデルの見積もりです. "
            "行動を無作為に割り当てて試して確かめてください (行動した人を"
            "選んだ記録だけでは偏ります)"
        ))
    elif kind == "select":
        found.append(finding(
            "info", "verification.method",
            "対象を選ぶ行動の効果 (effect) は仮定です. 過去のデータからは"
            "分からないので, 実際に試して確かめてください"
        ))

    return found


def column_amount(field: str, value: str, columns: list[str]) -> list[dict]:
    """
    列の名前で書いた損得の値を確かめる関数

    Parameters
    ----------
    field: str
        項目
    value: str
        列の名前
    columns: list[str]
        使えるデータの列

    Returns
    ----------
    found: list[dict]
        確かめ (列が無ければ error, あれば行ごとの値として読む知らせ)
    """
    # 使えるデータの列があれば, その中に
    if columns and value not in columns:
        found = [finding("error", "action.decision",
                         f"{field} の {value} は数でなく, 使えるデータの列"
                         "にもありません")]
    else:
        found = [finding("info", "action.decision",
                         f"{field} は列 {value} の行ごとの値として読みます")]

    return found


def uplift_range(value: Any) -> list[dict]:
    """
    確率の差で書いた効果の範囲を確かめる関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    found: list[dict]
        確かめ (数でないか -1 から 1 の外は error)
    """
    # 行動したとき − しないときの当たりの確率なので -1 から 1
    found = []
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        found.append(finding("error", "action.decision",
                             "effect は数か列の名前にしてください"))
    elif not -1 <= value <= 1:
        found.append(finding(
            "error", "action.decision",
            "effect (行動したとき − しないときの当たりの確率) は -1 から 1 "
            "にしてください"
        ))

    return found


def option_findings(decision: dict) -> list[dict]:
    """
    選んで書く項目 (効果の渡し方と向き) を確かめる関数

    Parameters
    ----------
    decision: dict
        意思決定

    Returns
    ----------
    found: list[dict]
        確かめ (知らない値, 確率の差で向きが無い, 防げる割合で向きがある)
    """
    # 手で書いたファイルの知らない値と, 向きの有無
    found = []
    options = DECISION_OPTIONS.get(decision["type"], {})
    for field, choices in options.items():
        value = decision.get(field)
        if value is not None and value not in choices:
            found.append(finding(
                "error", "action.decision",
                f"{field} は {', '.join(choices)} のどれかにしてください"
            ))
    if not options:
        return found
    uplift = decision.get("effect_kind") == "uplift"
    if uplift and decision.get("effect_goal") is None:
        found.append(finding(
            "error", "action.decision",
            "効果を確率の差で書いたので, 向き (effect_goal. 当たりを減らし"
            "たいなら reduce, 増やしたいなら raise) を書いてください"
        ))
    elif not uplift and decision.get("effect_goal") is not None:
        found.append(finding(
            "error", "action.decision",
            "向き (effect_goal) は, 効果を確率の差 (effect_kind: uplift) で"
            "書いたときだけ書いてください"
        ))

    return found


def amount_range(field: str, value: Any) -> list[dict]:
    """
    数で書いた損得と制約の範囲を確かめる関数

    Parameters
    ----------
    field: str
        項目
    value: Any
        値

    Returns
    ----------
    found: list[dict]
        確かめ (範囲の外は error)
    """
    # 効果は 0 から 1, 今ある量は 0 以上, 他は正
    found = []
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        found.append(finding("error", "action.decision",
                             f"{field} は数か列の名前にしてください"))
    elif field == "effect" and not 0 < value <= 1:
        found.append(finding(
            "error", "action.decision",
            "effect (行動で防げる割合) は 0 より大きく 1 以下にしてください"
        ))
    elif field == "stock" and value < 0:
        found.append(finding("error", "action.decision",
                             "stock は 0 以上にしてください"))
    elif field not in ("effect", "stock") and value <= 0:
        found.append(finding("error", "action.decision",
                             f"{field} は 0 より大きくしてください"))

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
    task = active_value(problem, "ml.task")
    metric = active_value(problem, "evaluation.metric")
    allowed = {item["value"]: item["tasks"] for item in METRICS}
    if task and metric and task not in allowed.get(metric, []):
        found.append(finding(
            "error", "evaluation.metric",
            f"指標 {metric} は{TASK_NAMES.get(task, task)}に合いません"
        ))
    if metric and not get_value(problem, "evaluation.link"):
        found.append(finding(
            "warning", "evaluation.link",
            "学習の指標と業務の指標のつながりが空です. 指標が良くなっても"
            "課題が解けるとは限りません"
        ))

    # 行動が結果を変える判断の, たどり直しだけの確かめ
    decision = active_value(problem, "action.decision") or {}
    method = active_value(problem, "verification.method")
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
    # 新しい版の知らせ (読める項目だけを使用)
    found = []
    version = problem.get("format_version")
    if isinstance(version, int) and version > FORMAT_VERSION:
        found.append(finding(
            "info", "format_version",
            f"このファイルは新しい版 ({version}) の ML-Quest で書かれました. "
            "読める項目だけを使います"
        ))

    # 形の崩れ (あれば他は確かめない), 答えの空, 筋道
    shape = check_shape(problem)
    if shape:
        return found + shape
    found += check_missing(problem)
    found += check_issue(problem)
    if get_value(problem, "ml.use") is True and get_value(
        problem, "ml.task"
    ) in (None, *[task["value"] for task in TASKS]):
        found += check_task(problem)
        found += check_columns(problem)
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
