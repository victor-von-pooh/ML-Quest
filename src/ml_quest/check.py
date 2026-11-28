import math
import re
from typing import Any, Optional

from ml_quest.problem import (
    DECISION_FIELDS, DECISION_NAMES, DECISION_OPTIONS,
    active_value, applies, choice_fits, get_value, is_answered,
    upgrade_problem, version_message
)
from ml_quest.questions import (
    ANOMALY_ASSIGNED, ANOMALY_SCORES, COLLECTS, CONSUMERS, DEMAND_ASSIGNED,
    FALLBACKS, GROUP_ASSIGNED, GROUP_STABILITIES, GROUP_TAKEN, GUARD_COUNTS,
    METHODS, METRICS, NOT_COUNTED, OUTPUTS, PART, PRICE_ASSIGNED, QUESTIONS,
    READY_LIBRARIES, ROLES, SIMULATORS, TARGET_KINDS, TARGET_SOURCES, UNITS,
    UPLIFT_ASSIGNED, UPLIFT_OUTCOMES, V1_METRICS, VARIED
)

# その他の指標の名前 (学習の記録の名前と曲線のファイルの名前)
METRIC_NAME = re.compile(r"[A-Za-z0-9_.@-]+")

# Gymnasium の環境の登録名 (CartPole-v1 など. 前に読み込むモジュールの
# "module:" と名前空間の "namespace/" は任意. 名前空間は Gymnasium と同じ
# く英数字と _ と -)
ENV_ID = re.compile(r"(?:[\w.]+:)?(?:[\w-]+/)?[\w.-]+", re.ASCII)

# 既製のモデルの重みの hash (SHA-256 の 16 進の 64 文字)
SHA256 = re.compile(r"[0-9a-fA-F]{64}")

# 骨組みを組める error の項目の場所 (同じ場所か, その下の項目). 課題, 分解,
# 決めることの文, 確かめ方, 運用, 受け持ちの文の答えは, README の要約の文に
# しか使わないので, 「それでも書き出す」と指定すれば書き出せる. 損得と制約
# は推論の一式の既定値になるので, error が残るときは既定値を付けずに書き出す
EXPORTABLE_ERRORS = (
    "issue", "decomposition", "action.who", "action.when", "action.what",
    "action.basis", "action.decision", "verification", "operation",
    "ml.constraints", f"{PART}.what", f"{PART}.item", f"{PART}.baseline",
    f"{PART}.consumer", f"{PART}.downstream", f"{PART}.evaluation.link",
    f"{PART}.evaluation.current"
)

# 0 から 1 の割合の指標の, 画面の呼び方
RATE_NAMES = {"accuracy": "正解率", "iou": "IoU "}

# 列の予測の出力の形
COLUMN_OUTPUTS = ["class", "value", "distribution"]

# 版 2 のデータセットで骨組みを書き出せる正解の種類 (予測するの受け持ち)
ITEMS_KINDS = ("mask", "point_cloud")

# 条件の無い生成の骨組みのある, 作るの正解の種類 (画像と表)
GENERATE_KINDS = ("image", "table")

# 受け持ちの中の, 辞書で書く項目
PART_DICTS = [f"{PART}.target", f"{PART}.target.process",
              f"{PART}.evaluation", f"{PART}.policy", f"{PART}.ready",
              f"{PART}.rank", f"{PART}.group", f"{PART}.anomaly",
              f"{PART}.uplift", f"{PART}.price", f"{PART}.demand"]

# 受け持ちの, 選んで書く項目と選択肢 (手で書いたファイルの確かめ)
PART_CHOICES = {
    f"{PART}.role": ROLES, f"{PART}.unit": UNITS,
    f"{PART}.output": OUTPUTS, f"{PART}.target.source": TARGET_SOURCES,
    f"{PART}.target.kind": TARGET_KINDS, f"{PART}.method": METHODS,
    f"{PART}.collect": COLLECTS, f"{PART}.simulator": SIMULATORS,
    f"{PART}.policy.fallback": FALLBACKS,
    f"{PART}.ready.library": READY_LIBRARIES,
    f"{PART}.group.stability": GROUP_STABILITIES,
    f"{PART}.anomaly.score": ANOMALY_SCORES,
    f"{PART}.uplift.outcome_kind": UPLIFT_OUTCOMES,
    f"{PART}.varied": VARIED,
}

# 方策の守りの, 文で書く項目
POLICY_TEXTS = [f"{PART}.policy.{name}" for name in (
    "action_limits", "stop", "safety_key", "approver"
)] + [f"{PART}.env_id"]

# 既製のモデルの, 文で書く項目 (一式づくりに渡す名前と hash)
READY_TEXTS = [f"{PART}.ready.{name}" for name in (
    "model", "weights", "sha256"
)]

# 並べるの, 反応の表の列の名前の項目 (利用者, 品物, 時刻, 反応の強さ,
# 見せただけの印, 提示)
RANK_COLUMNS = [f"{PART}.rank.{name}" for name in (
    "user", "item", "time", "reaction", "shown", "presentation"
)]

# 効果の見積もりの, 列の名前の項目 (処置, 結果, 行の id) と文の中の呼び方
UPLIFT_COLUMNS = {f"{PART}.uplift.treatment": "処置の列",
                  f"{PART}.uplift.outcome": "結果の列",
                  f"{PART}.uplift.id": "行の id の列"}

# 値段を変える判断の, 列の名前の項目 (値段, 売れた数, 原価, 商品の id,
# 行の id) と文の中の呼び方
PRICE_COLUMNS = {f"{PART}.price.price": "値段の列",
                 f"{PART}.price.units": "売れた数の列",
                 f"{PART}.price.cost": "原価の列",
                 f"{PART}.price.product": "商品の id の列",
                 f"{PART}.price.id": "行の id の列"}

# 時間をまたぐ量の判断の, 列の名前の項目 (商品の id, 日付, 売れた数, 在庫
# の数, 行の id) と文の中の呼び方
DEMAND_COLUMNS = {f"{PART}.demand.product": "商品の id の列",
                  f"{PART}.demand.date": "日付の列",
                  f"{PART}.demand.units": "日ごとの売れた数の列",
                  f"{PART}.demand.stock": "在庫の数の列",
                  f"{PART}.demand.id": "行の id の列"}

# データの分析の段階 (ML-EDA の需要の記録の表) が足す列の名前 (先の期間と
# 前の期間の売れた数の合計)
DEMAND_TAKEN = ("demand", "demand_previous")

# 並べるの列の項目の, 文の中の呼び方
RANK_TITLES = dict(zip(RANK_COLUMNS, ("利用者の列", "品物の列", "時刻の列",
                                      "反応の強さの列", "見せただけの印の列",
                                      "提示の列")))

# はいかいいえで答える項目
BOOLEANS = [f"{PART}.autonomous", f"{PART}.ros2", "ml.gpu"]

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
COLUMN_LISTS = ["information.columns", f"{PART}.series_by",
                "information.known", "information.unknown",
                "information.forbidden", f"{PART}.group.columns",
                f"{PART}.anomaly.columns", f"{PART}.uplift.columns",
                f"{PART}.price.columns", f"{PART}.demand.columns"]


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


def is_action(value: Any) -> bool:
    """
    決めた行動の答えの形 (文, 数, 数の並び) かを返す関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    fits: bool
        文, 数 (真偽値を除く), 数の並びなら True
    """
    # 番号で選ぶ行動は番号, 量で決める行動は数の並び
    items = value if isinstance(value, list) else [value]
    fits = isinstance(value, str) or bool(items) and all(
        isinstance(item, (int, float)) and not isinstance(item, bool)
        for item in items
    )

    return fits


def check_shape(problem: dict) -> list[dict]:
    """
    手で書いたファイルの, 値の形の崩れを一覧にする関数

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
    for key in (f"{PART}.task", f"{PART}.target.column",
                f"{PART}.time_column", f"{PART}.positive",
                f"{PART}.evaluation.metric", "issue.kpi.better",
                f"{PART}.consumer", f"{PART}.group_column",
                f"{PART}.split_time_column", f"{PART}.downstream",
                "issue.guard.name", "issue.guard.counted",
                f"{PART}.evaluation.name",
                f"{PART}.target.process.name", *PART_CHOICES,
                *POLICY_TEXTS, *READY_TEXTS, *RANK_COLUMNS,
                f"{PART}.group.id", f"{PART}.anomaly.label",
                f"{PART}.anomaly.id", *UPLIFT_COLUMNS, *PRICE_COLUMNS,
                *DEMAND_COLUMNS):
        value = get_value(problem, key)
        if value is not None and not isinstance(value, str):
            found.append(finding("error", key,
                                 f"{key} は文字列にしてください"))
    for key in ("data", "data.source", "data.origin"):
        message = location_message(problem, key)
        if message is not None:
            found.append(finding("error", key, message))
    consumer = get_value(problem, f"{PART}.consumer")
    if isinstance(consumer, str) and consumer not in [
        item["value"] for item in CONSUMERS
    ]:
        found.append(finding("error", f"{PART}.consumer",
                             f"使い道の {consumer} は知らない使い道です"))
    constraints = get_value(problem, "ml.constraints")
    if constraints is not None and not is_names(constraints):
        found.append(finding("error", "ml.constraints",
                             "ml.constraints は文の並びにしてください"))
    task = get_value(problem, f"{PART}.task")
    if isinstance(task, str) and task not in TASK_NAMES:
        found.append(finding("error", f"{PART}.task",
                             f"タスクの {task} は知らないタスクです"))
    metric = get_value(problem, f"{PART}.evaluation.metric")
    if isinstance(metric, str) and metric not in [m["value"]
                                                   for m in METRICS]:
        found.append(finding("error", f"{PART}.evaluation.metric",
                             f"evaluation.metric の {metric} は知らない指標"
                             "です"))
    for key, choices in PART_CHOICES.items():
        value = get_value(problem, key)
        if isinstance(value, str) and value not in [
            item["value"] for item in choices
        ]:
            found.append(finding("error", key,
                                 f"{key} の {value} は知らない値です"))
    if get_value(problem, f"{PART}.evaluation.better") not in (
        None, "lower", "higher"
    ):
        found.append(finding("error", f"{PART}.evaluation.better",
                             "指標の向きは lower か higher にしてください"))
    decision = get_value(problem, "action.decision")
    if decision is not None and not (
        isinstance(decision, dict) and isinstance(decision.get("type"), str)
        and decision["type"] in DECISION_FIELDS
    ):
        found.append(finding("error", "action.decision",
                             "action.decision の type は "
                             f"{', '.join(DECISION_FIELDS)} のどれかにして"
                             "ください"))
    action = get_value(problem, f"{PART}.policy.fallback_action")
    if action is not None and not is_action(action):
        found.append(finding("error", f"{PART}.policy.fallback_action",
                             "決めた行動は文, 数か数の並びにしてください"))
    for key in BOOLEANS:
        value = get_value(problem, key)
        if value is not None and not isinstance(value, bool):
            found.append(finding("error", key,
                                 f"{key} は true か false にしてください"))
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
    task = active_value(problem, f"{PART}.task")
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
        not active_value(problem, f"{PART}.positive")
    ):
        found.append(finding(
            "error", f"{PART}.positive",
            f"{DECISION_NAMES[kind]}には当たりのクラス (行動の対象) が要り"
            "ます"
        ))

    # 時刻の列と, 目的の列と系列の列の重なり
    target = active_value(problem, f"{PART}.target.column")
    time_column = active_value(problem, f"{PART}.time_column")
    series = active_value(problem, f"{PART}.series_by") or []
    if time_column and time_column == target:
        found.append(finding("error", f"{PART}.time_column",
                             "時刻の列が目的の列と同じです"))
    if time_column and time_column in series:
        found.append(finding("error", f"{PART}.series_by",
                             "系列を分ける列に時刻の列が入っています"))
    if target and target in series:
        found.append(finding("error", f"{PART}.series_by",
                             "系列を分ける列に目的の列が入っています"))
    split_time = active_value(problem, f"{PART}.split_time_column")
    if split_time and split_time == target:
        found.append(finding("error", f"{PART}.split_time_column",
                             "時刻の順で分ける列が目的の列です"))
    group = active_value(problem, f"{PART}.group_column")
    if group and group in (target, time_column, split_time):
        found.append(finding("error", f"{PART}.group_column",
                             "相手をまとめる列が目的の列か時刻の列です"))

    # 何期先までと, 何期前まで分かるか (1 以上の整数)
    for key, name in ((f"{PART}.horizon", "何期先まで"),
                      (f"{PART}.available_lag", "何期前の分まで分かるか")):
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
    for key in (f"{PART}.target.column", f"{PART}.time_column",
                f"{PART}.group_column", f"{PART}.split_time_column"):
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
    target = active_value(problem, f"{PART}.target.column")
    time_column = active_value(problem, f"{PART}.time_column")
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
            "error", f"{PART}.time_column",
            "時刻の列が使わない列に入っています (時刻は予測の時点で分かる)"
        ))

    # 決めていない列 (目的, 時刻, 系列の列は除く)
    roles = {target, time_column,
             active_value(problem, f"{PART}.split_time_column"),
             *(active_value(problem, f"{PART}.series_by") or [])}
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
    if kind == "order":
        found.append(finding(
            "error", "action.decision",
            f"{DECISION_NAMES[kind]}判断は, 予測するで正解の種類が先の期間の"
            "需要の受け持ちだけです. 列の予測の判断は, 量を決める, 対象を選ぶ"
            ", 振り分けるのどれかにしてください"
        ))
        return found
    if kind == "search":
        found.append(finding(
            "error", "action.decision",
            f"{DECISION_NAMES[kind]}判断は, 並べるで既製のモデルをそのまま"
            "使う受け持ち (似た写真を探す) だけです. 列の予測の判断は, 量を"
            "決める, 対象を選ぶ, 振り分けるのどれかにしてください"
        ))
        return found
    if kind in ("treat", "price"):
        found.append(finding(
            "error", "action.decision",
            f"{DECISION_NAMES[kind]}判断は, 効果を見積もるで方策でなく, 今ある"
            "記録から学ぶ受け持ちだけです. 予測を使う判断は, 量を決める, 対象"
            "を選ぶ, 振り分けるのどれかにしてください"
        ))
        return found
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


def is_finite(value: Any) -> bool:
    """
    値が有限の数 (真偽値でない) かを返す関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    finite: bool
        int か float で, 非数と無限大でなければ True
    """
    # int の仲間の真偽値の除外
    finite = (isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isfinite(value))

    return finite


def range_text(field: str, uplift: bool = False) -> str:
    """
    数で書いた損得と制約の項目の範囲を述べる文を返す関数 (確かめと損得の
    値の幅の誤りの文で同じ範囲を使うため)

    Parameters
    ----------
    field: str
        項目
    uplift: bool = False
        効果を確率の差で渡すか

    Returns
    ----------
    text: str
        範囲の文 (「… にしてください」まで)
    """
    # 確率の差は -1 から 1, 割合の効果は 0 から 1, 今ある量は 0 以上,
    # 他は正
    if field == "effect" and uplift:
        return "-1 から 1 にしてください"
    if field == "effect":
        return "0 より大きく 1 以下にしてください"
    if field == "stock":
        return "0 以上にしてください"
    text = "0 より大きくしてください"

    return text


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
    if not is_finite(value):
        found.append(finding("error", "action.decision",
                             "effect は有限の数か列の名前にしてください"))
    elif not -1 <= value <= 1:
        found.append(finding(
            "error", "action.decision",
            "effect (行動したとき − しないときの当たりの確率) は "
            + range_text("effect", uplift=True)
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
        確かめ (知らない項目と値, 確率の差で向きが無い, 防げる割合で向きが
        ある)
    """
    # 手で書いたファイルの知らない項目と値 (空は既定) と, 向きの有無
    found = []
    options = DECISION_OPTIONS.get(decision["type"], {})
    unknown = sorted(
        str(key) for key in decision
        if key != "type" and key not in DECISION_FIELDS[decision["type"]]
        and key not in options
    )
    if unknown:
        found.append(finding(
            "error", "action.decision",
            f"{DECISION_NAMES[decision['type']]}には {', '.join(unknown)} "
            "を書けません"
        ))
    for field, choices in options.items():
        value = decision.get(field)
        if value not in (None, "") and value not in choices:
            found.append(finding(
                "error", "action.decision",
                f"{field} は {', '.join(choices)} のどれかにしてください"
            ))
    if not options:
        return found
    uplift = decision.get("effect_kind") == "uplift"
    if uplift and decision.get("effect_goal") in (None, ""):
        found.append(finding(
            "error", "action.decision",
            "効果を確率の差で書いたので, 向き (effect_goal. 当たりを減らし"
            "たいなら reduce, 増やしたいなら raise) を書いてください"
        ))
    elif not uplift and decision.get("effect_goal") not in (None, ""):
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
    if not is_finite(value):
        found.append(finding("error", "action.decision",
                             f"{field} は有限の数か列の名前にしてください"))
    elif field == "effect" and not 0 < value <= 1:
        found.append(finding(
            "error", "action.decision",
            "effect (行動で防げる割合) は " + range_text("effect")
        ))
    elif field == "stock" and value < 0:
        found.append(finding("error", "action.decision",
                             "stock は " + range_text("stock")))
    elif field not in ("effect", "stock") and value <= 0:
        found.append(finding("error", "action.decision",
                             f"{field} は {range_text(field)}"))

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
    # 指標が受け持ち, 正解の種類, タスクに合うか
    found = []
    task = active_value(problem, f"{PART}.task")
    metric = active_value(problem, f"{PART}.evaluation.metric")
    if metric and not any(item["value"] == metric
                          and choice_fits(problem, item) for item in METRICS):
        where = TASK_NAMES.get(task, task) if task else (
            "今の受け持ちと正解の種類"
        )
        found.append(finding(
            "error", f"{PART}.evaluation.metric",
            f"指標{choice_label(METRICS, metric)}は{where}に合いません"
        ))
    name = active_value(problem, f"{PART}.evaluation.name")
    if isinstance(name, str) and (
        not METRIC_NAME.fullmatch(name) or name in (".", "..")
    ):
        found.append(finding(
            "error", f"{PART}.evaluation.name",
            f"指標の名前 {name} は英数字と _ . @ - だけで書いてください "
            "(学習の記録の名前 valid/<名前> と, 曲線のファイルの名前にする"
            "ため)"
        ))
    found += reported_range(
        metric, active_value(problem, f"{PART}.evaluation.current")
    )
    if metric and not get_value(problem, f"{PART}.evaluation.link"):
        found.append(finding(
            "warning", f"{PART}.evaluation.link",
            "学習の指標と課題の指標のつながりが空です. 指標が良くなっても"
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


def reported_range(metric: Any, value: Any) -> list[dict]:
    """
    今のやり方での指標の値が, 指標の取りうる値かを確かめる関数

    Parameters
    ----------
    metric: Any
        最良を選ぶ指標
    value: Any
        今のやり方での指標の値 (evaluation.current. 答えていなければ None)

    Returns
    ----------
    found: list[dict]
        確かめ (数でない, 負, 割合の指標で 1 より大きい場合に error)
    """
    # 割合の指標 (正解率と IoU) は 0 から 1, 誤差は 0 以上
    found: list[dict] = []
    key = f"{PART}.evaluation.current"
    if value is None:
        return found
    if not is_finite(value):
        found.append(finding("error", key,
                             "今のやり方での指標の値は有限の数にしてください"))
    elif value < 0:
        found.append(finding("error", key,
                             "今のやり方での指標の値が負です"))
    elif metric in RATE_NAMES and value > 1:
        found.append(finding(
            "error", key,
            f"{RATE_NAMES[metric]}は 0 から 1 の割合で書いてください (85 % "
            "なら 0.85)"
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
    # 版でない値と新しい版は読み込み無し (読める項目だけを使うと黙って
    # 読み違えるため)
    message = version_message(problem, "")
    if message:
        found = [finding("error", "format_version", message)]
        return found

    # 版 2 への移行の後, 版 1 の場所に残した知らない指標, 形の崩れと受け
    # 持ちの数 (あれば他は確かめない), 答えの空, 筋道
    problem = upgrade_problem(problem)
    shape = check_v1(problem) + check_parts(problem)
    if shape:
        return shape
    shape = check_shape(problem)
    if shape:
        return shape
    found = check_missing(problem)
    found += check_issue(problem)
    found += check_guard(problem)
    if get_value(problem, "ml.use") is True:
        if is_column_part(problem):
            found += check_task(problem)
            found += check_columns(problem)
            found += check_information(problem)
            found += check_decision(problem)
        if is_rank_part(problem):
            found += check_rank(problem)
        if is_group_part(problem):
            found += check_group(problem)
        if is_ready_group_part(problem):
            found += check_group_k(problem)
        if is_anomaly_part(problem):
            found += check_anomaly(problem)
        if is_uplift_part(problem):
            found += check_uplift(problem)
        if is_price_part(problem):
            found += check_price(problem)
        if is_demand_part(problem):
            found += check_demand(problem)
        if is_search_part(problem):
            found += search_findings(problem)
        found += check_part(problem)
        found += check_evaluation(problem)
        found += check_consumer(problem)
    found = [item for level in ("error", "warning", "info")
             for item in found if item["level"] == level]

    return found


def is_exportable(key: str) -> bool:
    """
    項目の場所の error が, 骨組みを組めるものかを返す関数

    Parameters
    ----------
    key: str
        項目の場所

    Returns
    ----------
    exportable: bool
        EXPORTABLE_ERRORS のどれかと同じ場所か, その下の項目なら True
    """
    # 頭の文字だけで合わせない (action.decision と action.decisions など)
    exportable = any(key == head or key.startswith(f"{head}.")
                     for head in EXPORTABLE_ERRORS)

    return exportable


def split_errors(
    problem: dict, found: Optional[list[dict]] = None
) -> tuple[list[dict], list[dict]]:
    """
    確かめの error を, 骨組みを組めないものと組めるものに分ける関数

    Parameters
    ----------
    problem: dict
        問題の設定
    found: Optional[list[dict]] = None
        同じ問題の設定の check の結果 (省くと求める)

    Returns
    ----------
    blocking: list[dict]
        骨組みを組めない error (受け渡しの値の形か, 学習のスクリプトの組み
        立てに使う答えの誤り. 版, 形の崩れ, 受け持ちの数の誤りは全て)
    exportable: list[dict]
        骨組みを組める error (EXPORTABLE_ERRORS の項目の場所)
    """
    # 版と形の誤りがあれば, 他の確かめをしていないので全てを組めないものに
    found = check(problem) if found is None else found
    errors = [item for item in found if item["level"] == "error"]
    if version_message(problem, ""):
        return errors, []
    upgraded = upgrade_problem(problem)
    if check_v1(upgraded) + check_parts(upgraded) + check_shape(upgraded):
        return errors, []
    blocking = [item for item in errors
                if not is_exportable(str(item["id"]))]
    exportable = [item for item in errors if item not in blocking]

    return blocking, exportable


def is_column_part(problem: dict) -> bool:
    """
    受け持ちが列の予測 (予測するで, 正解の種類が表の列) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    column: bool
        列の予測か (作り方が既製のモデルをそのままなら False)
    """
    # 受け持ちと正解の種類の答え (要らない問いの古い答えは除外). 既製の
    # モデルをそのまま使う場合は学習しないので除外
    column = active_value(problem, f"{PART}.role") == "predict" and (
        active_value(problem, f"{PART}.target.kind") == "column"
    ) and active_value(problem, f"{PART}.method") != "pretrained"

    return column


def is_items_part(problem: dict) -> bool:
    """
    受け持ちがファイルの組 (予測するで, 正解の種類が画像の領域か点群) かを
    返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    items: bool
        ファイルの組の受け持ちか
    """
    # 受け持ちと正解の種類の答え (要らない問いの古い答えは除外)
    items = active_value(problem, f"{PART}.role") == "predict" and (
        active_value(problem, f"{PART}.target.kind") in ITEMS_KINDS
    )

    return items


def location_message(problem: dict, key: str) -> Optional[str]:
    """
    手元のデータ (data) と, その場所 (data.source か data.origin) の型の
    誤りの文を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    key: str
        data, data.source か data.origin

    Returns
    ----------
    message: Optional[str]
        data が辞書でない場合と, 場所が文字でない場合の誤りの文 (無いか
        合っていれば None)
    """
    # 手で書いたファイルで, data が辞書でない (場所と読み方を持てない) 場合
    if key == "data":
        value = problem.get("data") if isinstance(problem, dict) else None
        message = None
        if value is not None and not isinstance(value, dict):
            message = ("手元のデータの項目は, 場所と読み方の組 (辞書) で"
                       f"書いてください (今は {value!r})")
        return message

    # 手で書いたファイルで数や並びになった場所 (Path にできない)
    value = get_value(problem, key)
    message = None
    if value is not None and not isinstance(value, str):
        name = "元のデータの場所" if key == "data.origin" else "データの場所"
        message = (f"{name}は, ファイルかディレクトリの場所を文字で書いて"
                   f"ください (今は {value!r})")

    return message


def choice_label(
    choices: list[dict], value: Any, problem: Optional[dict] = None,
    short: bool = False
) -> str:
    """
    選択肢の値を, 画面の選択肢と同じ言葉にする関数 (誤りの文のため)

    Parameters
    ----------
    choices: list[dict]
        選択肢 (value と label)
    value: Any
        値
    problem: Optional[dict] = None
        問題の設定 (渡すと, 同じ値の選択肢から今の答えに合うものを選ぶ.
        指標の loss のように同じ値が 2 つあるため)
    short: bool = False
        説明の括弧の前だけにするか (受け持ちの名前など)

    Returns
    ----------
    label: str
        「」で囲んだ選択肢の言葉 (選択肢に無ければ値のまま)
    """
    # 今の答えに合う選択肢を先に, 無ければ同じ値の最初の選択肢
    same = [item for item in choices if item["value"] == value]
    fitting = [item for item in same
               if problem is not None and choice_fits(problem, item)]
    item = (fitting or same or [None])[0]
    if item is None:
        label = f"「{value}」"
        return label
    text = item["label"].split(" (")[0] if short else item["label"]
    label = f"「{text}」"

    return label


def is_pretrained_part(problem: dict) -> bool:
    """
    受け持ちが既製のモデルをそのまま使う予測, まとめるか並べる (予測する,
    まとめるか並べるで, 作り方が既製のモデルをそのまま) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    pretrained: bool
        一式づくりに渡す値を問う受け持ちか (出力の形と正解の種類は問わ
        ない. 骨組みの書き出しが画像の分類か群か順位かを確かめる)
    """
    # 受け持ちと作り方の答え (要らない問いの古い答えは除外)
    pretrained = active_value(problem, f"{PART}.role") in (
        "predict", "group", "rank"
    ) and active_value(problem, f"{PART}.method") == "pretrained"

    return pretrained


def is_ready_group_part(problem: dict) -> bool:
    """
    受け持ちが既製の埋め込みの群分け (まとめるで, 作り方が既製のモデルを
    そのまま) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    group: bool
        既製の埋め込みの群分けの骨組みを書き出す受け持ちか (出力の形は
        問わない. 骨組みの書き出しが群かを確かめる)
    """
    # 受け持ちと作り方の答え (要らない問いの古い答えは除外)
    group = active_value(problem, f"{PART}.role") == "group" and (
        active_value(problem, f"{PART}.method") == "pretrained"
    )

    return group


def sha256_of(problem: dict) -> Optional[str]:
    """
    答えた既製のモデルの重みの hash を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    sha256: Optional[str]
        前後の空白を除き小文字にした hash (要らない問いか, 答えていないか,
        文字でないか, 空白だけの文字なら None)
    """
    # 手で書いた空の文字と空白だけの値も, 答えていない扱い. 大文字の 16 進
    # は ML-Inference の記録 (小文字) と比べられるように小文字
    value = active_value(problem, f"{PART}.ready.sha256")
    sha256 = value.strip().lower() if isinstance(value, str) else None
    sha256 = sha256 or None

    return sha256


def is_generate_part(problem: dict) -> bool:
    """
    受け持ちが条件の無い生成 (作るで, 正解の種類が画像か表) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    generate: bool
        画像か表の生成の受け持ちか (出力の形は問わない. 骨組みの書き出しが
        正解の種類と同じかを確かめる)
    """
    # 受け持ちと正解の種類の答え (要らない問いの古い答えは除外)
    generate = active_value(problem, f"{PART}.role") == "generate" and (
        active_value(problem, f"{PART}.target.kind") in GENERATE_KINDS
    )

    return generate


def is_policy_part(problem: dict) -> bool:
    """
    受け持ちが続けて行動を選ぶ方策 (効果を見積もるで, 方策と答えた) かを
    返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    policy: bool
        方策の受け持ちか
    """
    # 受け持ちと方策かの答え (要らない問いの古い答えは除外)
    policy = active_value(problem, f"{PART}.role") == "effect" and (
        active_value(problem, f"{PART}.autonomous") is True
    )

    return policy


def is_search_part(problem: dict) -> bool:
    """
    受け持ちが似た写真を探す (並べるで, 作り方が既製のモデルをそのまま)
    かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    search: bool
        似た写真を探す骨組みを書き出す受け持ちか (出力の形は問わない.
        骨組みの書き出しが順位かを確かめる)
    """
    # 受け持ちと作り方の答え (要らない問いの古い答えは除外)
    search = active_value(problem, f"{PART}.role") == "rank" and (
        active_value(problem, f"{PART}.method") == "pretrained"
    )

    return search


def search_findings(problem: dict) -> list[dict]:
    """
    似た写真を探すの, 検索の判断の型, 件数と待ち時間の上限を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 並べるで, 既製のモデルをそのまま)

    Returns
    ----------
    found: list[dict]
        確かめ (検索の判断でない, 件数か上限の形の誤り, 書けない項目は
        error)
    """
    # 型 (検索の判断だけ)
    found: list[dict] = []
    decision = active_value(problem, "action.decision")
    if not isinstance(decision, dict):
        return found
    name = DECISION_NAMES["search"]
    if decision.get("type") != "search":
        found.append(finding(
            "error", "action.decision",
            f"似た写真を探すの判断は「{name}」(search) にしてください (問い"
            "の写真ごとに, 登録した写真から見た目の近い順に K 件返す)"
        ))
        return found

    # 件数は 1 以上の整数, 待ち時間の上限は正の数 (任意)
    k = decision.get("k")
    if k is None:
        found.append(finding(
            "error", "action.decision",
            f"{name}判断の k (問いごとに返す件数) を書いてください"
        ))
    elif not is_finite_number(k) or not float(k).is_integer() or k < 1:
        found.append(finding(
            "error", "action.decision",
            "返す件数 (k) は 1 以上の整数にしてください"
        ))
    limit = decision.get("max_latency_ms")
    if limit is not None and (not is_finite_number(limit) or limit <= 0):
        found.append(finding(
            "error", "action.decision",
            "待ち時間の上限 (max_latency_ms) はミリ秒の正の数にしてください"
            " (無ければ空)"
        ))
    found += option_findings(decision)

    return found


def is_rank_part(problem: dict) -> bool:
    """
    受け持ちが学習して並べる (並べるで, 作り方が既製のモデルをそのままで
    ない) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    rank: bool
        並べるの骨組みを書き出す受け持ちか (出力の形は問わない. 骨組みの
        書き出しが順位かを確かめる)
    """
    # 受け持ちと作り方の答え (要らない問いの古い答えは除外)
    rank = active_value(problem, f"{PART}.role") == "rank" and (
        active_value(problem, f"{PART}.method") != "pretrained"
    )

    return rank


def check_rank(problem: dict) -> list[dict]:
    """
    並べるの反応の表の列と, 並べる件数を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 並べるで学習する)

    Returns
    ----------
    found: list[dict]
        確かめ (使えるデータの列に無い列, 同じ列を 2 つに答えた, 件数が
        1 以上の整数でない, は error)
    """
    # 列を答え直した後の古い名前と, 同じ列の答え
    found: list[dict] = []
    columns = active_value(problem, "information.columns") or []
    chosen: dict = {}
    for key in RANK_COLUMNS:
        name = active_value(problem, key)
        if not name:
            continue
        if columns and name not in columns:
            found.append(finding("error", key,
                                 f"{name} は使えるデータの列にありません"))
        elif name in chosen:
            found.append(finding(
                "error", key,
                f"{name} は{RANK_TITLES[chosen[name]]}と同じ列です. 利用者, "
                "品物, 時刻, 反応の強さ, 見せただけの印, 提示は別の列にして"
                "ください"
            ))
        else:
            chosen[name] = key

    # 並べる件数 (Recall@K と NDCG の K) は 1 以上の整数
    k = active_value(problem, f"{PART}.rank.k")
    if k is not None and (
        isinstance(k, bool) or not isinstance(k, (int, float))
        or not float(k).is_integer() or k < 1
    ):
        found.append(finding("error", f"{PART}.rank.k",
                             "並べる件数は 1 以上の整数にしてください"))

    return found


def is_group_part(problem: dict) -> bool:
    """
    受け持ちが学習してまとめる (まとめるで, 作り方が既製のモデルをそのまま
    でない) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    group: bool
        群分けの骨組みを書き出す受け持ちか (出力の形は問わない. 骨組みの
        書き出しが群かを確かめる)
    """
    # 受け持ちと作り方の答え (要らない問いの古い答えは除外)
    group = active_value(problem, f"{PART}.role") == "group" and (
        active_value(problem, f"{PART}.method") != "pretrained"
    )

    return group


def check_group(problem: dict) -> list[dict]:
    """
    群を作る列, 行を見分ける列と, 群の数を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. まとめるで学習する)

    Returns
    ----------
    found: list[dict]
        確かめ (使えるデータの列に無い列, 群を作る列と同じ行を見分ける列,
        決まった名前の群を作る列と行を見分ける列, 群の数が 2 以上の整数で
        ない, は error)
    """
    # 列を答え直した後の古い名前 (使えるデータの列の一覧が無ければ見ない)
    found: list[dict] = []
    columns = active_value(problem, "information.columns") or []
    used = active_value(problem, f"{PART}.group.columns") or []
    missing = [name for name in used if columns and name not in columns]
    if missing:
        found.append(finding(
            "error", f"{PART}.group.columns",
            f"{', '.join(missing)} は使えるデータの列にありません"
        ))
    taken = [name for name in used if name in GROUP_TAKEN]
    if taken:
        found.append(finding(
            "error", f"{PART}.group.columns",
            f"群を作る列 {', '.join(taken)} は群の要約か 1 件の鍵 ("
            f"{', '.join(GROUP_TAKEN)}) と同じ名前です. 名前を変えた列を"
            "作ってから選んでください"
        ))
    key = f"{PART}.group.id"
    name = active_value(problem, key)
    if name and columns and name not in columns:
        found.append(finding("error", key,
                             f"{name} は使えるデータの列にありません"))
    elif name and name in used:
        found.append(finding(
            "error", key,
            f"{name} は群を作るのに使う列にもあります. 行を見分ける列は群を"
            "作るのに使わないので, どちらかから外してください"
        ))
    if name in GROUP_ASSIGNED:
        found.append(finding(
            "error", key,
            f"行を見分ける列 {name} は群の割り当ての表の列 ("
            f"{', '.join(GROUP_ASSIGNED)}) と同じ名前です. 名前を変えた列を"
            "作ってから選んでください"
        ))
    found += check_group_k(problem)

    return found


def check_group_k(problem: dict) -> list[dict]:
    """
    群の数を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. まとめる)

    Returns
    ----------
    found: list[dict]
        確かめ (群の数が 2 以上の整数でない, は error)
    """
    # 群の数は 2 以上の整数
    found: list[dict] = []
    k = active_value(problem, f"{PART}.group.k")
    if k is not None and (
        isinstance(k, bool) or not isinstance(k, (int, float))
        or not float(k).is_integer() or k < 2
    ):
        found.append(finding("error", f"{PART}.group.k",
                             "群の数は 2 以上の整数にしてください"))

    return found


def is_anomaly_part(problem: dict) -> bool:
    """
    受け持ちが異常の検知 (予測するで, 正解の種類が異常. 作り方が既製の
    モデルをそのままでない) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    anomaly: bool
        異常の検知の骨組みを書き出す受け持ちか (出力の形は問わない. 確かめ
        と骨組みの書き出しが度合いかを確かめる)
    """
    # 受け持ち, 正解の種類と作り方の答え (要らない問いの古い答えは除外)
    anomaly = active_value(problem, f"{PART}.role") == "predict" and (
        active_value(problem, f"{PART}.target.kind") == "anomaly"
        and active_value(problem, f"{PART}.method") != "pretrained"
    )

    return anomaly


def check_anomaly(problem: dict) -> list[dict]:
    """
    異常の検知の正解の出どころ, 出力の形, 列と件数の上限を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 異常の検知)

    Returns
    ----------
    found: list[dict]
        確かめ (下の決まりに当たるものは全て error)
    """
    # 正解は無いか少しだけ, 出力の形は値 (行ごとの度合い)
    found: list[dict] = []
    source = active_value(problem, f"{PART}.target.source")
    if source not in (None, "none", "few"):
        found.append(finding(
            "error", f"{PART}.target.source",
            "正解の種類が異常なので, 正解の出どころは"
            f"{choice_label(TARGET_SOURCES, 'none')}か"
            f"{choice_label(TARGET_SOURCES, 'few')}にしてください (正解が多く"
            "あるなら, 正解の種類を表の列にして分類で学びます)"
        ))
    output = active_value(problem, f"{PART}.output")
    if output not in (None, "value"):
        found.append(finding(
            "error", f"{PART}.output",
            "正解の種類が異常なので, 出力の形は"
            f"{choice_label(OUTPUTS, 'value')} (行ごとの異常の度合い) にして"
            "ください (今は"
            f"{choice_label(OUTPUTS, output)})"
        ))

    # 列を答え直した後の古い名前 (使えるデータの列の一覧が無ければ見ない)
    columns = active_value(problem, "information.columns") or []
    used = active_value(problem, f"{PART}.anomaly.columns") or []
    label = active_value(problem, f"{PART}.anomaly.label")
    key = active_value(problem, f"{PART}.anomaly.id")
    missing = [name for name in used if columns and name not in columns]
    if missing:
        found.append(finding(
            "error", f"{PART}.anomaly.columns",
            f"{', '.join(missing)} は使えるデータの列にありません"
        ))
    for item, name in ((f"{PART}.anomaly.label", label),
                       (f"{PART}.anomaly.id", key)):
        if name and columns and name not in columns:
            found.append(finding("error", item,
                                 f"{name} は使えるデータの列にありません"))
        elif name and name in used:
            found.append(finding(
                "error", item,
                f"{name} は度合いに使う列にもあります. 正解の列と行の id の"
                "列は度合いに使いません"
            ))
    if label and key and label == key:
        found.append(finding("error", f"{PART}.anomaly.id",
                             f"{key} は異常の正解の列と同じです. 行の id の"
                             "列は別の列にしてください"))
    if key and key in ANOMALY_ASSIGNED:
        found.append(finding(
            "error", f"{PART}.anomaly.id",
            f"行の id の列 {key} は行ごとの度合いの表の列 ("
            f"{', '.join(ANOMALY_ASSIGNED)}) と同じ名前です. 名前を変えた"
            "列を作ってから選んでください"
        ))
    if not key and "row_id" in columns:
        found.append(finding(
            "error", f"{PART}.anomaly.id",
            "使えるデータの列に row_id があります. データの分析の段階は行の"
            "位置を row_id の列にするので, 行の id の列を答えてください"
        ))

    # 人が確かめる件数の上限は 1 以上の整数
    capacity = active_value(problem, f"{PART}.anomaly.capacity")
    if capacity is not None and (
        isinstance(capacity, bool) or not isinstance(capacity, (int, float))
        or not math.isfinite(capacity) or not float(capacity).is_integer()
        or capacity < 1
    ):
        found.append(finding(
            "error", f"{PART}.anomaly.capacity",
            "人が確かめる件数の上限は 1 以上の整数にしてください"
        ))

    return found


def is_uplift_part(problem: dict) -> bool:
    """
    受け持ちが効果の見積もり (効果を見積もるで, 方策でなく, 今ある記録から
    効果を見積もる. 作り方が既製のモデルをそのままでない) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    uplift: bool
        効果の見積もりの骨組みを書き出す受け持ちか
    """
    # 記録で変えたものが値段でない, 今ある記録から学ぶ受け持ち
    uplift = is_logged_part(problem) and (
        active_value(problem, f"{PART}.varied") != "price"
    )

    return uplift


def is_logged_part(problem: dict) -> bool:
    """
    受け持ちが, 効果を見積もるで方策でなく今ある記録から学ぶ (作り方が既製
    のモデルをそのままでない) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    logged: bool
        効果の見積もりか値段を変える判断か
    """
    # 受け持ち, 方策か, 集め方と作り方の答え (要らない問いの古い答えは除外)
    logged = active_value(problem, f"{PART}.role") == "effect" and (
        active_value(problem, f"{PART}.autonomous") is not True
        and active_value(problem, f"{PART}.collect") == "log"
        and active_value(problem, f"{PART}.method") != "pretrained"
    )

    return logged


def is_price_part(problem: dict) -> bool:
    """
    受け持ちが値段を変える判断 (今ある記録から学び, 記録で変えたものが
    値段) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    price: bool
        値段を変える判断の骨組みを書き出す受け持ちか
    """
    # 記録で変えたものが値段の, 今ある記録から学ぶ受け持ち
    price = is_logged_part(problem) and (
        active_value(problem, f"{PART}.varied") == "price"
    )

    return price


def check_uplift(problem: dict) -> list[dict]:
    """
    効果の見積もりの正解の出どころ, 列, 上位の割合, 配る判断と指標を確かめる
    関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 効果の見積もり)

    Returns
    ----------
    found: list[dict]
        確かめ (下の決まりに当たるものは全て error)
    """
    # 正解は介入の記録 (処置と結果の組)
    found: list[dict] = []
    source = active_value(problem, f"{PART}.target.source")
    if source not in (None, "intervention"):
        found.append(finding(
            "error", f"{PART}.target.source",
            "今ある記録から効果を見積もるので, 正解の出どころは"
            f"{choice_label(TARGET_SOURCES, 'intervention')}にしてください "
            "(処置の列と結果の列を持つ表)"
        ))

    # 列を答え直した後の古い名前と, 処置, 結果, 行の id の列の重なり
    found += uplift_columns(problem)

    # 上位の割合は 0 より大きく 1 以下
    k = active_value(problem, f"{PART}.uplift.k")
    if k is not None and (
        isinstance(k, bool) or not isinstance(k, (int, float))
        or not math.isfinite(k) or not 0 < k <= 1
    ):
        found.append(finding(
            "error", f"{PART}.uplift.k",
            "評価で見る上位の割合 K は 0 より大きく 1 以下の数にしてください "
            "(3 割なら 0.3)"
        ))

    # 配る判断の型と, 配る数の上限 (1 以上の整数)
    found += treat_findings(problem)

    # 方策の報酬は環境の評価のエピソードの値で, 記録からは数えない値
    if active_value(problem, f"{PART}.evaluation.metric") == "reward":
        found.append(finding(
            "error", f"{PART}.evaluation.metric",
            f"{choice_label(METRICS, 'reward')}は方策の指標です. 今ある"
            f"記録から効果を見積もるなら{choice_label(METRICS, 'qini')}か"
            "損失にしてください"
        ))

    return found


def treat_findings(problem: dict) -> list[dict]:
    """
    効果の見積もりの, 配る判断の型と配る数の上限を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 効果の見積もり)

    Returns
    ----------
    found: list[dict]
        確かめ (配る判断でない, 配る数の上限が 1 以上の整数でない, 書けない
        項目. どれも error)
    """
    # 型と上限 (型は配る判断だけ. 上限に列の名前は書けない)
    found: list[dict] = []
    decision = active_value(problem, "action.decision")
    if not isinstance(decision, dict):
        return found
    if decision.get("type") != "treat":
        found.append(finding(
            "error", "action.decision",
            f"効果の見積もりの判断は「{DECISION_NAMES['treat']}」(treat) に"
            "してください (効果の高い順に配る数の上限まで配る)"
        ))
        return found
    budget = decision.get("budget")
    if budget is None:
        found.append(finding(
            "error", "action.decision",
            f"{DECISION_NAMES['treat']}判断の budget (配る数の上限) を書いて"
            "ください"
        ))
    elif (isinstance(budget, bool) or not isinstance(budget, (int, float))
          or not math.isfinite(budget) or not float(budget).is_integer()
          or budget < 1):
        found.append(finding(
            "error", "action.decision",
            "配る数の上限 (budget) は 1 以上の整数にしてください"
        ))
    found += option_findings(decision)

    return found


def uplift_columns(problem: dict) -> list[dict]:
    """
    効果の見積もりの列の答えを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 効果の見積もり)

    Returns
    ----------
    found: list[dict]
        確かめ (使えるデータの列に無い, 見積もりに使う列と重なる, 処置,
        結果, 行の id の列が同じ, 行の id の列の名前の決まり. どれも error)
    """
    # 使えるデータの列の一覧が無いときの, 列に無いかの確かめの省略
    found: list[dict] = []
    columns = active_value(problem, "information.columns") or []
    used = active_value(problem, f"{PART}.uplift.columns") or []
    missing = [name for name in used if columns and name not in columns]
    if missing:
        found.append(finding(
            "error", f"{PART}.uplift.columns",
            f"{', '.join(missing)} は使えるデータの列にありません"
        ))
    named: dict[str, str] = {}
    for item, title in UPLIFT_COLUMNS.items():
        name = active_value(problem, item)
        if not name:
            continue
        if columns and name not in columns:
            found.append(finding("error", item,
                                 f"{name} は使えるデータの列にありません"))
        elif name in used:
            found.append(finding(
                "error", item,
                f"{name} は効果の見積もりに使う列にもあります. {title}は"
                "見積もりに使いません"
            ))
        same = [key for key, value in named.items() if value == name]
        if same:
            found.append(finding(
                "error", item,
                f"{name} は{UPLIFT_COLUMNS[same[0]]}と同じです. 別の列にして"
                "ください"
            ))
        named[item] = name

    # 行の id の列の名前 (行ごとの効果の表の他の列と, ML-EDA の行の位置)
    key = active_value(problem, f"{PART}.uplift.id")
    if key and key in UPLIFT_ASSIGNED:
        found.append(finding(
            "error", f"{PART}.uplift.id",
            f"行の id の列 {key} は行ごとの効果の表の列 ("
            f"{', '.join(UPLIFT_ASSIGNED)}) と同じ名前です. 名前を変えた"
            "列を作ってから選んでください"
        ))
    if not key and "row_id" in columns:
        found.append(finding(
            "error", f"{PART}.uplift.id",
            "使えるデータの列に row_id があります. データの分析の段階は行の"
            "位置を row_id の列にするので, 行の id の列を答えてください"
        ))

    return found


def check_price(problem: dict) -> list[dict]:
    """
    値段を変える判断の正解の出どころ, 列, 値段の判断と指標を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 値段を変える判断)

    Returns
    ----------
    found: list[dict]
        確かめ (候補が全て上限を超える場合だけ warning, 他は error)
    """
    # 正解は試しに値段を変えた記録 (値段と売れた数の組)
    found: list[dict] = []
    source = active_value(problem, f"{PART}.target.source")
    if source not in (None, "intervention"):
        found.append(finding(
            "error", f"{PART}.target.source",
            "試しに値段を変えた記録から学ぶので, 正解の出どころは"
            f"{choice_label(TARGET_SOURCES, 'intervention')}にしてください "
            "(値段の列と売れた数の列を持つ表)"
        ))

    # 列の答えと, 値段の判断の型, 候補, 上限
    found += price_columns(problem)
    found += price_findings(problem)

    # 方策の報酬は環境の評価のエピソードの値で, 記録からは数えない値
    if active_value(problem, f"{PART}.evaluation.metric") == "reward":
        found.append(finding(
            "error", f"{PART}.evaluation.metric",
            f"{choice_label(METRICS, 'reward')}は方策の指標です. 値段を変える"
            f"判断なら{choice_label(METRICS, 'mae')}か損失にしてください"
        ))

    return found


def price_columns(problem: dict) -> list[dict]:
    """
    値段を変える判断の列の答えを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 値段を変える判断)

    Returns
    ----------
    found: list[dict]
        確かめ (使えるデータの列に無い, 予測に使う列と重なる, 列どうしが
        同じ, 商品の id と行の id の列の名前の決まり, ML-EDA の元の値段の
        列の名前. どれも error)
    """
    # 使えるデータの列の一覧が無いときの, 列に無いかの確かめの省略
    found: list[dict] = []
    columns = active_value(problem, "information.columns") or []
    used = active_value(problem, f"{PART}.price.columns") or []
    missing = [name for name in used if columns and name not in columns]
    if missing:
        found.append(finding(
            "error", f"{PART}.price.columns",
            f"{', '.join(missing)} は使えるデータの列にありません"
        ))
    named: dict[str, str] = {}
    for item, title in PRICE_COLUMNS.items():
        name = active_value(problem, item)
        if not name:
            continue
        if columns and name not in columns:
            found.append(finding("error", item,
                                 f"{name} は使えるデータの列にありません"))
        elif name in used:
            found.append(finding(
                "error", item,
                f"{name} は売れ行きの予測に使う列にもあります. {title}は予測に"
                "使う列に入れません (値段の列は骨組みが入力の最後に足します)"
            ))
        same = [key for key, value in named.items() if value == name]
        if same:
            found.append(finding(
                "error", item,
                f"{name} は{PRICE_COLUMNS[same[0]]}と同じです. 別の列にして"
                "ください"
            ))
        named[item] = name

    # 商品の id と行の id の列の名前 (商品ごとの選んだ値段の表の他の列と,
    # ML-EDA の行の位置と元の値段の列)
    for item in (f"{PART}.price.product", f"{PART}.price.id"):
        key = active_value(problem, item)
        if key and key in PRICE_ASSIGNED:
            found.append(finding(
                "error", item,
                f"{PRICE_COLUMNS[item]} {key} は商品ごとの選んだ値段の表の列 "
                f"({', '.join(PRICE_ASSIGNED)}) と同じ名前です. 名前を変えた"
                "列を作ってから選んでください"
            ))
    if not active_value(problem, f"{PART}.price.id") and "row_id" in columns:
        found.append(finding(
            "error", f"{PART}.price.id",
            "使えるデータの列に row_id があります. データの分析の段階は行の"
            "位置を row_id の列にするので, 行の id の列を答えてください"
        ))
    if "price_raw" in columns:
        found.append(finding(
            "error", "information.columns",
            "使えるデータの列に price_raw があります. データの分析の段階は"
            "元の値段を price_raw の列に置くので, 名前を変えた列を作って"
            "ください"
        ))

    return found


def price_findings(problem: dict) -> list[dict]:
    """
    値段を変える判断の, 値段の判断の型, 候補と変える幅の上限を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 値段を変える判断)

    Returns
    ----------
    found: list[dict]
        確かめ (値段の判断でない, 候補か上限の形の誤り, 書けない項目は
        error. 今の値段の他の候補が全て上限を超える場合は warning)
    """
    # 型 (値段の判断だけ)
    found: list[dict] = []
    decision = active_value(problem, "action.decision")
    if not isinstance(decision, dict):
        return found
    name = DECISION_NAMES["price"]
    if decision.get("type") != "price":
        found.append(finding(
            "error", "action.decision",
            f"値段を変える判断の判断は「{name}」(price) にしてください (商品"
            "ごとに, 候補の中で上限を超えない, 利益が最も大きい値段を選ぶ)"
        ))
        return found

    # 候補は -1 より大きい数の並びで, 重ならず, 今の値段 (0) を含む
    candidates = decision.get("candidates")
    ratios = candidates if isinstance(candidates, list) else []
    numbers = all(is_finite_number(value) for value in ratios)
    if candidates is None:
        found.append(finding(
            "error", "action.decision",
            f"{name}判断の candidates (値段の候補の変える割合) を書いて"
            "ください"
        ))
    elif not ratios or not numbers or any(value <= -1 for value in ratios):
        found.append(finding(
            "error", "action.decision",
            "値段の候補 (candidates) は, -1 より大きい数 (今の値段に対する"
            "変える割合. 1 割上げるなら 0.1) の並びにしてください"
        ))
    elif len(set(ratios)) != len(ratios) or 0 not in ratios:
        found.append(finding(
            "error", "action.decision",
            "値段の候補 (candidates) は同じ割合を重ねず, 今の値段 (割合 0) "
            "を含めてください"
        ))

    # 上限は 0 以上の数 (今の値段の他の候補が全て上限を超えると選べない)
    limit = decision.get("max_change")
    if limit is None:
        found.append(finding(
            "error", "action.decision",
            f"{name}判断の max_change (値段を変える幅の上限) を書いて"
            "ください"
        ))
    elif not is_finite_number(limit) or limit < 0:
        found.append(finding(
            "error", "action.decision",
            "値段を変える幅の上限 (max_change) は 0 以上の数 (変える割合の"
            "絶対値. 1 割までなら 0.1) にしてください"
        ))
    elif ratios and numbers and not any(
        value != 0 and abs(value) <= limit for value in ratios
    ):
        found.append(finding(
            "warning", "action.decision",
            "今の値段の他の候補が全て値段を変える幅の上限を超えるので, いつも"
            "今の値段を選びます. 上限の中の候補を足してください"
        ))
    found += option_findings(decision)

    return found


def is_finite_number(value: Any) -> bool:
    """
    値が真偽値でない有限の数かを返す関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    number: bool
        真偽値でない有限の数か
    """
    # 真偽値を除く数
    number = not isinstance(value, bool) and isinstance(
        value, (int, float)
    ) and math.isfinite(value)

    return number


def is_demand_part(problem: dict) -> bool:
    """
    受け持ちが時間をまたぐ量の判断 (予測するで, 正解の種類が先の期間の需要.
    作り方が既製のモデルをそのままでない) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    demand: bool
        時間をまたぐ量の判断の骨組みを書き出す受け持ちか (出力の形は問わ
        ない. 確かめと骨組みの書き出しが分布かを確かめる)
    """
    # 受け持ち, 正解の種類と作り方の答え (要らない問いの古い答えは除外)
    demand = active_value(problem, f"{PART}.role") == "predict" and (
        active_value(problem, f"{PART}.target.kind") == "demand"
        and active_value(problem, f"{PART}.method") != "pretrained"
    )

    return demand


def check_demand(problem: dict) -> list[dict]:
    """
    時間をまたぐ量の判断の正解の出どころ, 出力の形, 列, 期間, 水準と発注の
    判断を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 時間をまたぐ量の判断)

    Returns
    ----------
    found: list[dict]
        確かめ (発注の数の上限が 0 の場合だけ warning, 他は error)
    """
    # 正解は記録の売れた数の列, 出力の形は分布 (分位点)
    found: list[dict] = []
    source = active_value(problem, f"{PART}.target.source")
    if source not in (None, "column"):
        found.append(finding(
            "error", f"{PART}.target.source",
            "日ごとの売れた数の記録から先の期間の合計を学ぶので, 正解の出どころ"
            f"は{choice_label(TARGET_SOURCES, 'column')}にしてください"
        ))
    output = active_value(problem, f"{PART}.output")
    if output not in (None, "distribution"):
        found.append(finding(
            "error", f"{PART}.output",
            "正解の種類が先の期間の需要なので, 出力の形は"
            f"{choice_label(OUTPUTS, 'distribution')} (水準ごとの分位点) に"
            f"してください (今は{choice_label(OUTPUTS, output)})"
        ))

    # 列の答え, 先の期間の日数, 学ぶ水準と, 発注の判断
    found += demand_columns(problem)
    found += demand_periods(problem)
    found += order_findings(problem)

    return found


def demand_columns(problem: dict) -> list[dict]:
    """
    時間をまたぐ量の判断の列の答えを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 時間をまたぐ量の判断)

    Returns
    ----------
    found: list[dict]
        確かめ (使えるデータの列に無い, 予測に使う列と重なる, 列どうしが
        同じ, 商品の id と日付の列の名前の決まり, ML-EDA の足す列の名前.
        どれも error)
    """
    # 使えるデータの列の一覧が無いときの, 列に無いかの確かめの省略
    found: list[dict] = []
    columns = active_value(problem, "information.columns") or []
    used = active_value(problem, f"{PART}.demand.columns") or []
    missing = [name for name in used if columns and name not in columns]
    if missing:
        found.append(finding(
            "error", f"{PART}.demand.columns",
            f"{', '.join(missing)} は使えるデータの列にありません"
        ))
    named: dict[str, str] = {}
    for item, title in DEMAND_COLUMNS.items():
        name = active_value(problem, item)
        if not name:
            continue
        if columns and name not in columns:
            found.append(finding("error", item,
                                 f"{name} は使えるデータの列にありません"))
        elif name in used:
            found.append(finding(
                "error", item,
                f"{name} は需要の予測に使う商品の特徴の列にもあります. {title}"
                "は予測に使う列に入れません (過去の売れた数はデータの分析の"
                "段階でずらした列にします)"
            ))
        same = [key for key, value in named.items() if value == name]
        if same:
            found.append(finding(
                "error", item,
                f"{name} は{DEMAND_COLUMNS[same[0]]}と同じです. 別の列にして"
                "ください"
            ))
        named[item] = name

    # 商品の id と日付の列の名前 (発注の数の表の他の列) と, ML-EDA の行の
    # 位置と先の期間と前の期間の合計の列
    for item in (f"{PART}.demand.product", f"{PART}.demand.date"):
        key = active_value(problem, item)
        if key and key in DEMAND_ASSIGNED:
            found.append(finding(
                "error", item,
                f"{DEMAND_COLUMNS[item]} {key} は発注の数の表の列 ("
                f"{', '.join(DEMAND_ASSIGNED)}) と同じ名前です. 名前を変えた"
                "列を作ってから選んでください"
            ))
    if not active_value(problem, f"{PART}.demand.id") and "row_id" in columns:
        found.append(finding(
            "error", f"{PART}.demand.id",
            "使えるデータの列に row_id があります. データの分析の段階は行の"
            "位置を row_id の列にするので, 行の id の列を答えてください"
        ))
    taken = [name for name in DEMAND_TAKEN if name in columns]
    if taken:
        found.append(finding(
            "error", "information.columns",
            f"使えるデータの列に {', '.join(taken)} があります. データの分析"
            "の段階は先の期間と前の期間の売れた数の合計をこの名前の列に置く"
            "ので, 名前を変えた列を作ってください"
        ))

    return found


def demand_periods(problem: dict) -> list[dict]:
    """
    時間をまたぐ量の判断の先の期間の日数と, 学ぶ分位点の水準を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 時間をまたぐ量の判断)

    Returns
    ----------
    found: list[dict]
        確かめ (日数が 1 以上の整数でない, 水準が 0 より大きく 1 より小さい
        数でないか重なる. どれも error)
    """
    # 先の期間の日数は 1 以上の整数
    found: list[dict] = []
    horizon = active_value(problem, f"{PART}.demand.horizon")
    if horizon is not None and (
        not is_finite_number(horizon) or not float(horizon).is_integer()
        or horizon < 1
    ):
        found.append(finding(
            "error", f"{PART}.demand.horizon",
            "先の期間の日数は 1 以上の整数にしてください"
        ))

    # 水準は 0 より大きく 1 より小さい数の, 重ならない並び
    levels = active_value(problem, f"{PART}.demand.levels")
    if levels is None:
        return found
    if not isinstance(levels, list) or not levels or not all(
        is_finite_number(level) and 0 < level < 1 for level in levels
    ):
        found.append(finding(
            "error", f"{PART}.demand.levels",
            "学ぶ分位点の水準は, 0 より大きく 1 より小さい数の並びにして"
            "ください (9 割の点なら 0.9)"
        ))
    elif len(set(levels)) != len(levels):
        found.append(finding(
            "error", f"{PART}.demand.levels",
            "学ぶ分位点の水準に同じ値が重なっています. 1 つにしてください"
        ))

    return found


def order_findings(problem: dict) -> list[dict]:
    """
    時間をまたぐ量の判断の, 発注の判断の型, 損と上限を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 時間をまたぐ量の判断)

    Returns
    ----------
    found: list[dict]
        確かめ (発注の判断でない, 損か上限の形の誤り, 書けない項目は
        error. 上限が 0 で何も発注しない場合は warning)
    """
    # 型 (発注の判断だけ)
    found: list[dict] = []
    decision = active_value(problem, "action.decision")
    if not isinstance(decision, dict):
        return found
    name = DECISION_NAMES["order"]
    if decision.get("type") != "order":
        found.append(finding(
            "error", "action.decision",
            f"時間をまたぐ量の判断の判断は「{name}」(order) にしてください ("
            "商品ごとに, 損の比から決まる水準の分位点から在庫を引いて, 上限"
            "の中で発注する)"
        ))
        return found

    # 2 つの損は商品 1 つあたりの正の数 (どちらかが 0 だと水準が 0 か 1
    # になり, 分位点で決まらない)
    for field, title in (("shortage_cost", "品切れの損"),
                         ("holding_cost", "売れ残りの損")):
        value = decision.get(field)
        if value is None:
            found.append(finding(
                "error", "action.decision",
                f"{name}判断の {field} ({title}) を書いてください"
            ))
        elif not is_finite_number(value) or value <= 0:
            found.append(finding(
                "error", "action.decision",
                f"{title} ({field}) は商品 1 つあたりの正の数にしてください "
                "(列の名前では書けません. 全ての商品で同じ値)"
            ))

    # 上限は 0 以上の数 (0 なら何も発注しない)
    limit = decision.get("max_order")
    if limit is None:
        found.append(finding(
            "error", "action.decision",
            f"{name}判断の max_order (発注の数の上限) を書いてください"
        ))
    elif not is_finite_number(limit) or limit < 0:
        found.append(finding(
            "error", "action.decision",
            "発注の数の上限 (max_order) は 0 以上の数にしてください (守る"
            "指標なので省けません)"
        ))
    elif limit < 1:
        found.append(finding(
            "warning", "action.decision",
            "発注の数の上限が 1 より小さいので, どの商品も発注しません. 置き"
            "場所の上限を書いてください"
        ))
    found += option_findings(decision)

    return found


def check_v1(problem: dict) -> list[dict]:
    """
    版 1 のファイルに書いた, 4 つの他の指標を確かめる関数

    版 2 へ移すとき, 4 つの他の指標は移さず版 1 の場所に残す

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    found: list[dict]
        確かめ (版 1 の場所に残った指標は error)
    """
    # その他と, 版 2 で足した指標は版 1 では知らない指標
    found = []
    stale = problem.get("evaluation")
    metric = stale.get("metric") if isinstance(stale, dict) else None
    if metric in V1_METRICS:
        found.append(finding(
            "error", f"{PART}.evaluation.metric",
            f"evaluation.metric の {metric} は版 1 の場所です. 版 2 では最良"
            "を選ぶ指標を ml.parts.0.evaluation に書いてください"
        ))
    elif metric is not None:
        found.append(finding(
            "error", f"{PART}.evaluation.metric",
            f"evaluation.metric の {metric} は知らない指標です (版 1 のファ"
            f"イルで書けるのは {', '.join(V1_METRICS)}. 版 2 では "
            "ml.parts.0.evaluation に書く)"
        ))

    return found


def check_parts(problem: dict) -> list[dict]:
    """
    受け持ちの並びの形と数を確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    found: list[dict]
        確かめ (並びでない, 辞書でない, 2 つ以上は error)
    """
    # 今は 1 つだけ (段どうしのつなぎを決めていないため)
    found: list[dict] = []
    parts = get_value(problem, "ml.parts")
    if parts is None:
        return found
    if not isinstance(parts, list) or not all(
        isinstance(part, dict) for part in parts
    ):
        found.append(finding("error", "ml.parts",
                             "ml.parts は受け持ちの辞書の並びにしてください"))
    elif len(parts) > 1:
        found.append(finding(
            "error", "ml.parts",
            f"受け持ちは今は 1 つだけ書けます (ml.parts に {len(parts)} つ"
            "あります). 問題を分けるか, 1 つにしてください"
        ))
    else:
        found += [
            finding("error", key, f"{key} は辞書にしてください")
            for key in PART_DICTS
            if get_value(problem, key) is not None
            and not isinstance(get_value(problem, key), dict)
        ]

    return found


def check_part(problem: dict) -> list[dict]:
    """
    受け持ちと, 出力の形, 正解の種類, タスクの食い違いを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 受け持ちに合わない出力の形と正解の種類 (選択肢に無い値)
    found = []
    for key, choices in ((f"{PART}.output", OUTPUTS),
                         (f"{PART}.target.kind", TARGET_KINDS)):
        value = active_value(problem, key)
        choice = next((item for item in choices if item["value"] == value),
                      None)
        if choice is not None and not choice_fits(problem, choice):
            role = get_value(problem, f"{PART}.role")
            found.append(finding(
                "error", key,
                f"{choice_label(choices, value)}は受け持ち"
                f"{choice_label(ROLES, role, short=True)}に合いません"
            ))

    # 環境の登録名の形 (学習の段階で Gymnasium に渡すため)
    env_id = env_id_of(problem)
    if env_id is not None and not ENV_ID.fullmatch(env_id):
        found.append(finding(
            "error", f"{PART}.env_id",
            f"環境の登録名 {env_id} は Gymnasium の登録名の形 (CartPole-v1, "
            "ALE/Pong-v5 など. 英数字と _ . -. 前に名前空間の namespace/ と, "
            "読み込むモジュールの module: を付けてよい) で書いてください"
        ))

    # 既製のモデルの重みの hash の形 (一式づくりが読んだ重みと比べるため)
    sha256 = sha256_of(problem)
    if sha256 is not None and not SHA256.fullmatch(sha256):
        found.append(finding(
            "error", f"{PART}.ready.sha256",
            f"重みの hash {sha256} は SHA-256 の 16 進の 64 文字で書いて"
            "ください (分からなければ空のまま)"
        ))

    # 列の予測の正解の種類と出力の形 (表の列はクラスか値か分布)
    task = active_value(problem, f"{PART}.task")
    output = active_value(problem, f"{PART}.output")
    if is_column_part(problem) and output not in (
        None, *COLUMN_OUTPUTS
    ):
        found.append(finding(
            "error", f"{PART}.output",
            "正解の種類が表の列なので, 出力の形は "
            + "".join(choice_label(OUTPUTS, name) for name in COLUMN_OUTPUTS)
            + f"のどれかにしてください (今は{choice_label(OUTPUTS, output)})"
        ))

    # 列の予測のタスクと出力の形
    if task == "classification" and output not in (None, "class"):
        found.append(finding("warning", f"{PART}.output",
                             "分類なのに, 出力の形がクラスではありません"))
    if task in ("regression", "forecasting") and output == "class":
        found.append(finding("warning", f"{PART}.output",
                             f"{TASK_NAMES[task]}なのに, 出力の形がクラス"
                             "です"))

    return found


def env_id_of(problem: dict) -> Optional[str]:
    """
    答えた環境の登録名を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    env_id: Optional[str]
        前後の空白を除いた登録名 (要らない問いか, 答えていないか, 空か
        空白だけの文字なら None)
    """
    # 手で書いた空の文字と空白だけの値も, 答えていない扱い
    value = active_value(problem, f"{PART}.env_id")
    env_id = value.strip() if isinstance(value, str) else None
    env_id = env_id or None

    return env_id


def check_consumer(problem: dict) -> list[dict]:
    """
    予測の使い道と, 意思決定と後段の食い違いを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 自動で決めるのに意思決定の無い場合と, 後段に渡すのに求めるものの無い場合
    found: list[dict] = []
    consumer = active_value(problem, f"{PART}.consumer")
    decision = active_value(problem, "action.decision") or {}
    kind = decision.get("type") if isinstance(decision, dict) else None
    if consumer == "rule" and kind == "none":
        found.append(finding(
            "warning", "action.decision",
            "予測を決まりで自動に決めるのに, 意思決定が「しない」です. "
            "量, 選ぶ, 振り分けのどれかと損得を書いてください"
        ))
    if consumer in ("system", "model") and not active_value(
        problem, f"{PART}.downstream"
    ):
        found.append(finding(
            "warning", f"{PART}.downstream",
            "予測を後段に渡すのに, 後段が求める形と精度がありません. 評価"
            "は後段の結果で見るので, 先に書いてください"
        ))

    return found


def check_guard(problem: dict) -> list[dict]:
    """
    守る指標と, 確かめ方と運用へのつなぎを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 解けたと言える指標と同じ名前と, 成功の基準と止める条件への書き忘れ
    found: list[dict] = []
    name = active_value(problem, "issue.guard.name")
    if not isinstance(name, str) or not name.strip():
        return found
    if name.strip() == str(get_value(problem, "issue.kpi.name") or "").strip():
        found.append(finding(
            "warning", "issue.guard.name",
            "守る指標が, 解けたと言える指標と同じです. 良くしたい数字と引き"
            "換えに悪くなりうる別の数字を書いてください"
        ))
    line = active_value(problem, "issue.guard.line")
    if line is not None and (
        not isinstance(line, (int, float)) or isinstance(line, bool)
    ):
        found.append(finding("error", "issue.guard.line",
                             "守る線は数にしてください"))
    if active_value(problem, "issue.guard.better") not in (
        None, "lower", "higher"
    ):
        found.append(finding("error", "issue.guard.better",
                             "守る指標の向きは lower か higher にしてください"))
    found += check_guard_counted(problem, line)
    texts = [str(active_value(problem, key) or "")
             for key in ("verification.success", "operation.stop")]
    if get_value(problem, "ml.use") is True and not any(
        name.strip() in text for text in texts
    ):
        found.append(finding(
            "info", "verification.success",
            f"守る指標 ({name.strip()}) が, 成功の基準 (9) にも止める条件 "
            "(10) にもありません. 守る線を割ったら成功と言わない, 止める, "
            "と書いておくと, 良くしたい数字だけを追わずに済みます"
        ))

    return found


def check_guard_counted(problem: dict, line: Any) -> list[dict]:
    """
    守る指標がたどり直しで数える量が意思決定の型にあるかと, 割合の量の
    守る線が 0 から 1 かを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定
    line: Any
        守る線の答え

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 型を変えた後の古い答えと, % のままの割合の線
    found: list[dict] = []
    counted = active_value(problem, "issue.guard.counted")
    if not isinstance(counted, str) or counted == NOT_COUNTED:
        return found
    kind = get_value(problem, "action.decision.type")
    counts = GUARD_COUNTS.get(kind, {}) if isinstance(kind, str) else {}
    if counted not in counts:
        found.append(finding(
            "error", "issue.guard.counted",
            f"守る指標が数える量の {counted} は, "
            f"{DECISION_NAMES.get(str(kind), kind)}意思決定では数えられません. "
            "選び直してください"
        ))
        return found
    if counted.endswith("_rate") and isinstance(line, (int, float)) and (
        not isinstance(line, bool) and not 0 <= line <= 1
    ):
        found.append(finding(
            "error", "issue.guard.line",
            f"守る指標が数える量 ({counts[counted]['description']}) は 0 から"
            " 1 の割合なので, 守る線も 0 から 1 で書いてください (5 % なら "
            "0.05)"
        ))

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
