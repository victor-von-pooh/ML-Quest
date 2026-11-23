import math
from typing import Any, Optional

from ml_quest.check import (
    amount_range, check_decision, is_finite, location_message, range_text,
    uplift_range
)
from ml_quest.data import (
    ITEMS_LOADER, loss_parts, parts_loss, period_days, read_data
)
from ml_quest.problem import (
    FIELD_NAMES, active_value, data_location, get_value, upgrade_problem
)
from ml_quest.questions import PART

# 意思決定の型ごとの, 幅を渡せる損得と制約の項目 (結論を変える項目だけ.
# 今ある量と量の刻みは, 予測が無いと結論が変わらないので除く)
SENSITIVITY_FIELDS = {
    "quantity": ("under_cost", "over_cost"),
    "select": ("value", "cost", "effect", "budget", "budget_cost"),
    "triage": ("miss_cost", "false_alarm_cost", "capacity"),
}

# 幅の中で求める点の数の範囲 (両端を含む. 多すぎる点で重いデータの損を
# 何度も求めないように)
MIN_POINTS = 2
MAX_POINTS = 21

# 件数の項目 (幅の点を整数にそろえる)
COUNT_FIELDS = ("budget", "capacity")

# 決め方の値の名前 (型ごと. 対象を選ぶ型は効果の渡し方ごと)
RULE_NAMES = {
    "quantity": "置く分位点の水準",
    "share": "得になる当たりの確率の下限",
    "uplift": "得になる確率の差の大きさの下限",
    "triage": "回す確率のしきい値",
}

# 振り分けのやり方の名前 (今のやり方と, 予測の要らない素朴なやり方)
PLAN_NAMES = {"current": "今のやり方", "none": "どれも回さない",
              "all": "全てを回す"}

# 型ごとの, データの上で損を求めるやり方
PLANS = {"quantity": ("current",), "triage": ("current", "none", "all")}

# 予測が無いと求まらないものの知らせ
PREDICTION_NOTE = (
    "行動の件数と量そのものは予測 (確率や外れ方) が無いと求まりません. "
    "ここで求めるのは決め方の値と, データの上の今のやり方の損です. 学習の"
    "後に, 推論の段階のたどり直しで今のやり方との差を確かめてください"
)


def decision_sensitivity(
    problem: dict, ranges: dict, points: int = 5
) -> dict:
    """
    損得と制約の値を幅の中で変えたとき, 結論がどう変わるかを返す関数

    値は 1 つずつ変え, 他の値は問題の設定のまま. 結論は, 決め方の値 (量は
    置く水準, 対象を選ぶは得になる確率の下限, 振り分けは回すしきい値),
    件数の上限, データの上の今のやり方の損 (量と振り分け. 振り分けは
    予測の要らない素朴なやり方の損も)

    Parameters
    ----------
    problem: dict
        問題の設定
    ranges: dict
        項目ごとの幅 ({"under_cost": [100, 300]} など. 下と上の 2 つの数)
    points: int = 5
        幅の中で求める点の数 (両端を含む. 2 から 21)

    Returns
    ----------
    result: dict
        type, rule (決め方の値の名前), base (今の値での結論), fields
        (項目ごとの点と振れ幅と文), data (損を求めた行と日数), notes

    Raises
    ----------
    ValueError
        意思決定が無いか崩れている, 幅を渡せない項目, 幅の誤り
    """
    # 意思決定と幅の確かめ
    problem = upgrade_problem(problem)
    decision = checked_decision(problem)
    kind = decision["type"]
    if not isinstance(points, int) or isinstance(points, bool) or not (
        MIN_POINTS <= points <= MAX_POINTS
    ):
        raise ValueError(f"点の数 (points) は {MIN_POINTS} から "
                         f"{MAX_POINTS} の整数にしてください")
    grids = {field: field_values(decision, field, bounds, points)
             for field, bounds in checked_ranges(decision, ranges).items()}

    # データの上の今のやり方の損の部品と, 知らせ
    parts, notes = data_parts(problem, kind)
    rule = rule_name(decision)
    if rule_value(decision) is None:
        notes.append(f"列で書いた値 (行ごとの値) があるので, {rule}は行ごと"
                     "に違います. 決め方の値は求めません")
    notes.append(PREDICTION_NOTE)
    data = None
    if parts is not None:
        try:
            rows = parts_loss(parts, decision)
        except (KeyError, ValueError, TypeError) as error:
            parts = None
            notes.insert(0, f"今のやり方の損を求められません: {error}")
        else:
            data = {"rows": int(len(rows)),
                    "days": period_days(parts, rows.index)}

    # 今の値と, 項目ごとの点
    base = outcome(decision, parts, data)
    fields = [field_result(decision, field, values, parts, data)
              for field, values in grids.items()]
    result = {"type": kind, "rule": rule, "base": base, "fields": fields,
              "data": data, "notes": notes}

    return result


def checked_decision(problem: dict) -> dict:
    """
    幅を渡せる意思決定かを確かめて返す関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    decision: dict
        意思決定 (量を決める, 対象を選ぶ, 振り分ける)

    Raises
    ----------
    ValueError
        意思決定が無い, 意思決定しない, 損得と制約に直すところがある
    """
    # 型と, 損得と制約の確かめの誤り
    decision = active_value(problem, "action.decision")
    if not isinstance(decision, dict) or (
        decision.get("type") not in SENSITIVITY_FIELDS
    ):
        raise ValueError("損得の値の幅は, 量を決める, 対象を選ぶ, 振り分ける"
                         "の意思決定にだけ渡せます")
    errors = [item["message"] for item in check_decision(problem)
              if item["level"] == "error" and item["id"] == "action.decision"]
    if errors:
        raise ValueError("損得と制約に直すところがあります: "
                         + " / ".join(errors))

    return decision


def checked_ranges(decision: dict, ranges: Any) -> dict:
    """
    項目ごとの幅を確かめる関数

    Parameters
    ----------
    decision: dict
        意思決定
    ranges: Any
        項目ごとの幅

    Returns
    ----------
    checked: dict
        項目ごとの (下, 上)

    Raises
    ----------
    ValueError
        幅を渡せない項目, 列で書いた項目, 幅の形と範囲の誤り
    """
    # 幅を渡せる項目と, 数で書いた項目
    kind = decision["type"]
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError("ranges は, 項目ごとの幅の辞書にしてください")
    checked = {}
    for field, bounds in ranges.items():
        if field not in SENSITIVITY_FIELDS[kind]:
            names = ", ".join(error_name(decision, name)
                              for name in SENSITIVITY_FIELDS[kind])
            raise ValueError(f"{error_name(decision, field)} には幅を渡せ"
                             f"ません (渡せるのは {names})")
        if isinstance(decision.get(field), str):
            raise ValueError(f"{error_name(decision, field)} は列 "
                             f"{decision[field]} の行ごとの値なので, 幅を"
                             "渡せません")
        checked[field] = checked_bounds(decision, field, bounds)

    # 選ぶ型の上限は件数か費用の 1 つだけ
    for field, other in (("budget", "budget_cost"),
                         ("budget_cost", "budget")):
        if field in checked and (other in checked
                                 or decision.get(other) is not None):
            raise ValueError("対象を選ぶ行動の上限は, 件数の上限 (budget) "
                             "か費用の上限 (budget_cost) の 1 つだけにして"
                             "ください")

    return checked


def checked_bounds(decision: dict, field: str, bounds: Any) -> tuple:
    """
    1 つの項目の幅を確かめる関数

    Parameters
    ----------
    decision: dict
        意思決定
    field: str
        項目
    bounds: Any
        幅 (下と上の 2 つの数)

    Returns
    ----------
    checked: tuple
        (下, 上)

    Raises
    ----------
    ValueError
        2 つの数でない, 下が上より大きい, 項目の範囲の外
    """
    # 2 つの有限の数で, 下から上へ
    name = error_name(decision, field)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or not all(
        is_finite(value) for value in bounds
    ):
        raise ValueError(f"{name} の幅は, 下と上の 2 つの数にしてください")
    low, high = bounds
    if low > high:
        raise ValueError(f"{name} の幅は, 下を上以下にしてください")

    # 両端が項目の範囲の中 (問題の設定の確かめと同じ決まり. 文は項目の
    # 名前で)
    uplift = field == "effect" and decision.get("effect_kind") == "uplift"
    for value in (low, high):
        found = uplift_range(value) if uplift else amount_range(field, value)
        if found:
            raise ValueError(f"{name} の幅の {number_text(value)} は, "
                             f"{range_text(field, uplift)}")
    checked = (low, high)

    return checked


def field_label(decision: dict, field: str) -> str:
    """
    損得と制約の項目の日本語の名前を返す関数

    Parameters
    ----------
    decision: dict
        意思決定
    field: str
        項目

    Returns
    ----------
    label: str
        問題の設定の問いの名前 (効果を確率の差で渡すなら「確率の差」.
        知らない項目は鍵のまま)
    """
    # 効果は渡し方ごと
    if field == "effect" and decision.get("effect_kind") == "uplift":
        return "確率の差"
    label = FIELD_NAMES.get(field, field)

    return label


def error_name(decision: dict, field: str) -> str:
    """
    誤りの文に出す項目の名前を返す関数

    Parameters
    ----------
    decision: dict
        意思決定
    field: str
        項目

    Returns
    ----------
    name: str
        日本語の名前と, 括弧で添えた鍵 (知らない項目は鍵だけ)
    """
    # 名前の無い項目は鍵だけ
    label = field_label(decision, field)
    name = label if label == field else f"{label} ({field})"

    return name


def field_values(
    decision: dict, field: str, bounds: tuple, points: int
) -> list:
    """
    幅の中で求める値の並びを作る関数

    Parameters
    ----------
    decision: dict
        意思決定
    field: str
        項目
    bounds: tuple
        (下, 上)
    points: int
        点の数 (両端を含む)

    Returns
    ----------
    values: list
        小さい順の値 (件数の項目は max(1, round(x)) で整数にする. round
        は偶数への丸めなので 2.5 は 2. 重なりは除く)
    """
    # 等しい間隔の点 (件数は整数に丸め, 0 にはしない. 整数の幅の整数の点
    # は整数のまま)
    low, high = bounds
    spaced = [low + (high - low) * step / (points - 1)
              for step in range(points)]
    if field in COUNT_FIELDS:
        spaced = [max(1, round(value)) for value in spaced]
    elif isinstance(low, int) and isinstance(high, int):
        spaced = [int(value) if float(value).is_integer() else value
                  for value in spaced]
    values = sorted(set(spaced))

    return values


def data_parts(problem: dict, kind: str) -> tuple[Optional[dict], list]:
    """
    データを読み, 今のやり方の損の部品を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定
    kind: str
        意思決定の型

    Returns
    ----------
    result: tuple[Optional[dict], list]
        loss_parts の部品 (求められなければ None) と, 求めない理由の知らせ
    """
    # 損を求める型で, 今のやり方を書き, 表のデータを入れた場合だけ読む
    if kind not in PLANS:
        return None, ["対象を選ぶ行動は, 行動した行の結果が行動の後の値なので, "
                      "今のやり方の得は求めません (試して確かめる)"]
    if not active_value(problem, "action.current"):
        return None, ["今のやり方をデータで書いていないので, 今のやり方の損は"
                      "求めません"]
    if any(location_message(problem, key) is not None
           for key in ("data", "data.source", "data.origin")):
        return None, ["手元のデータの場所が崩れているので, 今のやり方の損は"
                      "求めません"]
    source = data_location(problem)
    if not source:
        return None, ["データを入れていないので, 今のやり方の損は求めません"]
    if get_value(problem, "data.loader") == ITEMS_LOADER:
        return None, ["ファイルの組のデータセットでは, 今のやり方の損は求め"
                      "ません"]
    try:
        data, _ = read_data(
            source, get_value(problem, "data.loader"),
            dict(get_value(problem, "data.options") or {}),
            active_value(problem, f"{PART}.time_column")
        )
    except Exception as error:
        return None, [f"データを読めないので, 今のやり方の損は求めません: "
                      f"{error}"]

    # 部品 (求められなければ理由)
    parts = loss_parts(problem, data)
    if parts is None or "reason" in parts:
        return None, [parts["reason"]] if parts else []
    try:
        rows = parts_loss(parts, active_value(problem, "action.decision"))
    except (KeyError, ValueError, TypeError) as error:
        return None, [f"今のやり方の損を求められません: {error}"]
    if rows.empty:
        return None, ["今のやり方の損を求められる行がありません"]
    result = (parts, [])

    return result


def rule_name(decision: dict) -> str:
    """
    意思決定の型の, 決め方の値の名前を返す関数

    Parameters
    ----------
    decision: dict
        意思決定

    Returns
    ----------
    name: str
        決め方の値の名前
    """
    # 対象を選ぶ型は効果の渡し方ごと
    kind = decision["type"]
    if kind == "select":
        kind = "uplift" if decision.get("effect_kind") == "uplift" else (
            "share"
        )
    name = RULE_NAMES[kind]

    return name


def rule_value(decision: dict) -> Optional[float]:
    """
    損得の値から決め方の値を求める関数 (ML-Inference の決め方と同じ式)

    Parameters
    ----------
    decision: dict
        意思決定

    Returns
    ----------
    value: Optional[float]
        量は under / (under + over) (丸めない水準. ML-Inference は百分位
        に丸めて分位点を選ぶ), 対象を選ぶは cost / (value × effect)
        (確率の差は cost / value), 振り分けは false_alarm / (miss +
        false_alarm). 列で書いた値があれば None
    """
    # 使う値が全て数のときだけ (確率の差は効果を使わない)
    kind = decision["type"]
    uplift = decision.get("effect_kind") == "uplift"
    names = {"quantity": ("under_cost", "over_cost"),
             "select": ("value", "cost") if uplift else (
                 "value", "cost", "effect"
             ),
             "triage": ("miss_cost", "false_alarm_cost")}[kind]
    if not all(is_finite(decision.get(name)) for name in names):
        return None
    if kind == "quantity":
        under, over = decision["under_cost"], decision["over_cost"]
        value = under / (under + over)
        return value
    if kind == "triage":
        miss, alarm = decision["miss_cost"], decision["false_alarm_cost"]
        value = alarm / (miss + alarm)
        return value
    worth = decision["value"]
    if not uplift:
        worth *= decision["effect"]
    value = decision["cost"] / worth

    return value


def limit_of(decision: dict) -> Optional[int]:
    """
    件数の上限を求める関数

    Parameters
    ----------
    decision: dict
        意思決定

    Returns
    ----------
    limit: Optional[int]
        件数の上限 (選ぶ型の費用の上限は, 費用が数なら 費用の上限 / 費用
        の切り捨て). 上限が無いか, 費用が列なら None
    """
    # 件数の上限か, 費用の上限を 1 件の費用で割った件数
    kind = decision["type"]
    count = decision.get("capacity" if kind == "triage" else "budget")
    if kind == "quantity":
        return None
    if is_finite(count):
        limit = int(count)
        return limit
    spend, cost = decision.get("budget_cost"), decision.get("cost")
    if kind == "select" and is_finite(spend) and is_finite(cost):
        limit = math.floor(spend / cost)
        return limit

    return None


def outcome(
    decision: dict, parts: Optional[dict], data: Optional[dict]
) -> dict:
    """
    1 つの損得の値の組での結論を求める関数

    Parameters
    ----------
    decision: dict
        意思決定
    parts: Optional[dict]
        今のやり方の損の部品 (無ければ None)
    data: Optional[dict]
        損を求めた行と日数

    Returns
    ----------
    point: dict
        rule (決め方の値), acts (対象を選ぶ型で, 得になる対象がありうるか),
        limit (件数の上限), loss と per_30_days (やり方ごとの損), best
        (振り分けで損の最も小さいやり方. 損が同じなら current, none, all
        の先の順) と best_label (その名前)
    """
    # 決め方の値と, 件数の上限
    kind = decision["type"]
    rule = rule_value(decision)
    point: dict = {"rule": rule}
    if kind == "select":
        # 確率も確率の差の大きさも 1 以下なので, 下限が 1 以上なら選ばない
        point["acts"] = None if rule is None else bool(rule < 1)
    if kind != "quantity":
        point["limit"] = limit_of(decision)
    if parts is None or data is None:
        return point

    # データの上のやり方ごとの損 (と 30 日あたり)
    loss = {plan: float(parts_loss(parts, decision, plan).sum())
            for plan in PLANS[kind]}
    days = data["days"]
    point["loss"] = loss
    point["per_30_days"] = (
        {plan: value / days * 30 for plan, value in loss.items()}
        if days is not None and days >= 1 else None
    )
    if kind == "triage":
        point["best"] = min(loss, key=loss.__getitem__)
        point["best_label"] = PLAN_NAMES[point["best"]]

    return point


def field_result(
    decision: dict, field: str, values: list, parts: Optional[dict],
    data: Optional[dict]
) -> dict:
    """
    1 つの項目を幅の中で変えたときの結論の並びと文を作る関数

    Parameters
    ----------
    decision: dict
        意思決定
    field: str
        項目
    values: list
        求める値の並び
    parts: Optional[dict]
        今のやり方の損の部品
    data: Optional[dict]
        損を求めた行と日数

    Returns
    ----------
    result: dict
        field, label, points (value と結論), swing (決め方の値と今のやり方
        の損の振れ幅), sentence
    """
    # 1 つずつ変えた点
    points = []
    for value in values:
        point = outcome({**decision, field: value}, parts, data)
        points.append({"value": value, **point})
    label = field_label(decision, field)

    # 振れ幅と文
    swing = {"rule": spread([point["rule"] for point in points]),
             "loss": spread([point["loss"]["current"] for point in points
                             if "loss" in point])}
    result = {"field": field, "label": label, "points": points,
              "swing": swing,
              "sentence": field_sentence(decision, label, points, data)}

    return result


def spread(values: list) -> Optional[float]:
    """
    値の振れ幅を求める関数

    Parameters
    ----------
    values: list
        値の並び (None を含んでよい)

    Returns
    ----------
    width: Optional[float]
        最大 − 最小 (数が無ければ None)
    """
    # None を除いた最大と最小の差
    numbers = [value for value in values if value is not None]
    width = max(numbers) - min(numbers) if numbers else None

    return width


def field_sentence(
    decision: dict, label: str, points: list, data: Optional[dict]
) -> str:
    """
    1 つの項目を変えたときの結論の変わり方を述べる文を作る関数

    Parameters
    ----------
    decision: dict
        意思決定
    label: str
        項目の名前
    points: list
        値ごとの結論
    data: Optional[dict]
        損を求めた行と日数

    Returns
    ----------
    sentence: str
        文
    """
    # 幅と決め方の値
    first, last = points[0], points[-1]
    span = (number_text(first["value"]) if len(points) == 1 else
            f"{number_text(first['value'])} から "
            f"{number_text(last['value'])}")
    head = f"{label}を {span} に変えると (他の値は今のまま), "
    lines = []
    if first["rule"] is not None:
        lines.append(f"{rule_name(decision)}は {first['rule']:.2f} から "
                     f"{last['rule']:.2f}")
    stops = [number_text(point["value"]) for point in points
             if point.get("acts") is False]
    if stops:
        lines.append(f"{label}が {', '.join(stops)} では, 確率が 1 でも得に"
                     "ならず, どの対象も選ばない")
    if first.get("limit") != last.get("limit"):
        lines.append(f"件数の上限は {limit_text(first.get('limit'))} から "
                     f"{limit_text(last.get('limit'))}")

    # 今のやり方の損と, 振り分けで最も損の小さいやり方
    if "loss" in first:
        monthly = first["per_30_days"] is not None
        key = "per_30_days" if monthly else "loss"
        period = "30 日あたり" if monthly else f"{data['rows']:,} 行で"
        lines.append(f"今のやり方の損は, {period} "
                     f"{first[key]['current']:,.0f} から "
                     f"{last[key]['current']:,.0f}")
    if "best" in first:
        lines.append(best_text(points))
    if not lines:
        lines.append("決め方の値も今のやり方の損も求めない")
    sentence = head + ". ".join(lines)

    return sentence


def best_text(points: list) -> str:
    """
    振り分けで最も損の小さいやり方の変わり方を述べる文を作る関数

    Parameters
    ----------
    points: list
        値ごとの結論

    Returns
    ----------
    text: str
        文
    """
    # 値ごとの最も損の小さいやり方 (変わらなければ 1 つ)
    plans = [point["best"] for point in points]
    if len(set(plans)) == 1:
        text = f"どの値でも, 最も損が小さいのは{PLAN_NAMES[plans[0]]}"
        return text
    pairs = [f"{number_text(point['value'])} では{PLAN_NAMES[point['best']]}"
             for point in points]
    text = "最も損が小さいのは, " + ", ".join(pairs)

    return text


def number_text(value: float) -> str:
    """
    値を短い文字にする関数

    Parameters
    ----------
    value: float
        値

    Returns
    ----------
    text: str
        整数は桁区切り, 他は小数 4 桁までで後ろの 0 を除く
    """
    # 整数か小数
    if float(value).is_integer():
        text = f"{value:,.0f}"
        return text
    text = f"{value:,.4f}".rstrip("0").rstrip(".")

    return text


def limit_text(limit: Optional[int]) -> str:
    """
    件数の上限を文字にする関数

    Parameters
    ----------
    limit: Optional[int]
        件数の上限

    Returns
    ----------
    text: str
        件数 (無ければ「無し」)
    """
    # 無い上限
    text = "無し" if limit is None else f"{limit:,} 件"

    return text
