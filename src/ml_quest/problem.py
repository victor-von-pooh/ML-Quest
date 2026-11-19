import copy
from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any, Optional, Union
import uuid

import pandas as pd
import yaml

from ml_quest.questions import BY_ID, PART, QUESTIONS, V1_METRICS

# 問題の設定の形の版
FORMAT_VERSION = 2

# 版 1 の場所と, 版 2 で移した先 (版 1 の名前でも読み書きできるように)
LEGACY_KEYS = {
    "ml.predict": f"{PART}.what", "ml.row": f"{PART}.item",
    "ml.task": f"{PART}.task", "ml.target": f"{PART}.target.column",
    "ml.positive": f"{PART}.positive",
    "ml.group_column": f"{PART}.group_column",
    "ml.split_time_column": f"{PART}.split_time_column",
    "ml.time_column": f"{PART}.time_column",
    "ml.series_by": f"{PART}.series_by",
    "ml.available_lag": f"{PART}.available_lag",
    "ml.horizon": f"{PART}.horizon", "ml.baseline": f"{PART}.baseline",
    "ml.consumer": f"{PART}.consumer",
    "ml.downstream": f"{PART}.downstream",
    "evaluation.metric": f"{PART}.evaluation.metric",
    "evaluation.link": f"{PART}.evaluation.link",
}

# 意思決定の型ごとの損得と制約の項目 (True は要る項目. ML-Inference の形)
DECISION_FIELDS = {
    "quantity": {"under_cost": True, "over_cost": True, "stock": False,
                 "step": False},
    "select": {"value": True, "cost": True, "effect": True,
               "budget": False, "budget_cost": False},
    "triage": {"miss_cost": True, "false_alarm_cost": True,
               "capacity": False},
    "none": {},
}

# 意思決定の型ごとの, 選んで書く項目と選択肢 (ML-Inference の形). 対象を
# 選ぶ型の効果の渡し方 (share は防げる割合, uplift は 行動したとき − しない
# ときの当たりの確率) と, uplift の向き (当たりを減らしたいか増やしたいか)
DECISION_OPTIONS = {
    "select": {"effect_kind": ("share", "uplift"),
               "effect_goal": ("reduce", "raise")},
}

# 選んで書く項目の値の名前 (要約の文に使う)
OPTION_NAMES = {
    "share": "防げる割合", "uplift": "確率の差",
    "reduce": "当たりを減らす", "raise": "当たりを増やす",
}

# 桁区切りのカンマ付きの数 (1,200,000 など)
GROUPED_NUMBER = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")

# 数として読まない文字 (無限大と非数)
NOT_NUMBERS = {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}

# 損得と制約の項目の名前 (要約の文に使う)
FIELD_NAMES = {
    "under_cost": "足りないときの損", "over_cost": "余ったときの損",
    "stock": "今ある量", "step": "量の刻み", "value": "当たりの価値",
    "cost": "行動の費用", "effect": "防げる割合", "budget": "件数の上限",
    "budget_cost": "費用の上限", "miss_cost": "見逃しの損",
    "false_alarm_cost": "空振りの損", "capacity": "回せる件数の上限",
    "positive": "当たりのクラス", "effect_kind": "効果の渡し方",
    "effect_goal": "向き",
}

# 意思決定の型の名前
DECISION_NAMES = {
    "quantity": "量を決める", "select": "対象を選ぶ",
    "triage": "振り分ける", "none": "意思決定しない",
}


def now_text() -> str:
    """
    今の時刻を, 時差付きの ISO 8601 の文字列で返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    text: str
        秒までの時刻
    """
    # 手元の時差付き
    text = datetime.now().astimezone().isoformat(timespec="seconds")

    return text


def new_problem(name: str) -> dict:
    """
    空の問題の設定を作る関数

    Parameters
    ----------
    name: str
        問題の名前

    Returns
    ----------
    problem: dict
        版と名前と時刻だけを持つ問題の設定
    """
    # 名前の空は誤り
    if not str(name).strip():
        raise ValueError("問題の名前を書いてください")
    stamp = now_text()
    problem = {"format_version": FORMAT_VERSION, "name": str(name).strip(),
               "created_at": stamp, "updated_at": stamp}

    return problem


def get_value(problem: dict, key: str) -> Any:
    """
    問題の設定の項目の値を, 点でつないだ場所から返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    key: str
        項目の場所 (issue.kpi.name など)

    Returns
    ----------
    value: Any
        値 (無ければ None)
    """
    # 辞書の順のたどり (並びは番号で). 版 2 では版 1 の名前は移した先
    key = current_key(problem, key)
    value: Any = problem
    for part in key.split("."):
        if isinstance(value, list) and part.isdigit():
            if int(part) >= len(value):
                return None
            value = value[int(part)]
            continue
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]

    return value


def current_key(problem: Any, key: str) -> str:
    """
    版 1 の項目の場所を, 版 2 の問題の設定での場所にする関数

    Parameters
    ----------
    problem: Any
        問題の設定
    key: str
        項目の場所

    Returns
    ----------
    resolved: str
        版 2 の問題の設定で版 1 の名前なら移した先, 他はそのまま
    """
    # 版 1 の問題の設定は版 1 の場所のまま
    version = problem.get("format_version") if isinstance(
        problem, dict
    ) else None
    resolved = key
    if key in LEGACY_KEYS and version != 1:
        resolved = LEGACY_KEYS[key]

    return resolved


def set_value(problem: dict, key: str, value: Any) -> None:
    """
    問題の設定の項目に, 点でつないだ場所で値を入れる関数

    値が None なら項目を消す

    Parameters
    ----------
    problem: dict
        問題の設定 (書き換える)
    key: str
        項目の場所
    value: Any
        値

    Returns
    ----------
    None
    """
    # 消す場合は, 空になった親の辞書も消去. 次が番号なら並び (受け持ち).
    # 版 2 では版 1 の名前は移した先
    parts = current_key(problem, key).split(".")
    if value is None:
        remove_value(problem, parts)
        return
    node: Any = problem
    for part, following in zip(parts[:-1], parts[1:]):
        empty: Any = [] if following.isdigit() else {}
        node = child_of(node, part, empty)
    if isinstance(node, list):
        child_of(node, parts[-1], None)
        node[int(parts[-1])] = value
    else:
        node[parts[-1]] = value


def child_of(node: Any, part: str, empty: Any) -> Any:
    """
    辞書か並びの子を, 無ければ空のものを入れて返す関数

    Parameters
    ----------
    node: Any
        辞書か並び (書き換える)
    part: str
        項目の名前か並びの番号
    empty: Any
        無いときに入れる空の辞書か並び

    Returns
    ----------
    child: Any
        子

    Raises
    ----------
    ValueError
        並びの中で, 番号でない名前か, 辞書でない要素の場合
    """
    # 並びは番号まで空の辞書で伸長. 番号でない名前と, 手で書いた辞書で
    # ない要素は誤り (黙って置き換えないため)
    if isinstance(node, list):
        if not part.isdigit():
            raise ValueError(f"並びの中の {part} は番号ではありません")
        index = int(part)
        while len(node) <= index:
            node.append({})
        if empty is not None and not isinstance(node[index], type(empty)):
            raise ValueError(
                f"並びの {index} 番目が辞書でないので, 答えを入れられません"
            )
        child = node[index]
        return child
    if isinstance(node.get(part), list) and isinstance(empty, dict):
        raise ValueError(f"{part} は並びなので, 次は番号で書いてください")
    if not isinstance(node.get(part), type(empty)):
        node[part] = empty
    child = node[part]

    return child


def remove_value(node: Any, parts: list[str]) -> bool:
    """
    点でつないだ場所の項目を消し, 空になった辞書かを返す関数

    Parameters
    ----------
    node: Any
        たどる辞書か並び (書き換える)
    parts: list[str]
        項目の場所を点で分けた並び

    Returns
    ----------
    empty: bool
        消した後に node が空か
    """
    # 末端の消去と, 空になった途中の辞書と並びの消去 (並びは最後の空の
    # 要素だけを除く. 他の要素の番号を変えないため)
    head = parts[0]
    if isinstance(node, list):
        index = int(head) if head.isdigit() else len(node)
        last = index == len(node) - 1
        if last and (len(parts) == 1 or (
            isinstance(node[index], dict)
            and remove_value(node[index], parts[1:])
        )):
            node.pop()
        elif index < len(node) - 1 and len(parts) > 1 and isinstance(
            node[index], dict
        ):
            remove_value(node[index], parts[1:])
        empty = not node
        return empty
    if len(parts) == 1:
        node.pop(head, None)
    elif isinstance(node.get(head), (dict, list)) and remove_value(
        node[head], parts[1:]
    ):
        node.pop(head)
    empty = not node

    return empty


def applies(problem: dict, question: dict) -> bool:
    """
    問いが今の答えのもとで要るかを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    question: dict
        問い

    Returns
    ----------
    needed: bool
        要るか (when の条件が無いか, 満たすか)
    """
    # 条件の並びの全てで, 項目の値が等しいか, 並びに入るか, 答えたか
    when = question.get("when") or []
    conditions = when if isinstance(when, list) else [when]
    needed = True
    for condition in conditions:
        value = get_value(problem, condition["id"])
        if "equals" in condition:
            needed = needed and value == condition["equals"]
        elif "filled" in condition:
            needed = needed and is_answered(value) == condition["filled"]
        else:
            needed = needed and value in condition["in"]

    return needed


def choices_of(problem: dict, question: dict) -> list[dict]:
    """
    問いの選択肢を, 今の答えのもとで選べるものに絞って返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    question: dict
        問い

    Returns
    ----------
    choices: list[dict]
        値と説明の並び (出力の形と指標は受け持ち, 正解の種類, タスクに合う
        ものだけ. 同じ値は最初のもの)
    """
    # 今の答えに合う選択肢. 意思決定の型は欄も
    choices: list[dict] = []
    for choice in question.get("choices", []):
        if not choice_fits(problem, choice) or choice["value"] in [
            item["value"] for item in choices
        ]:
            continue
        choices.append(
            {"value": choice["value"], "label": choice["label"]}
            | ({"fields": copy.deepcopy(choice["fields"])}
               if "fields" in choice else {})
        )

    return choices


def choice_fits(problem: dict, choice: dict) -> bool:
    """
    選択肢が今の答え (受け持ち, 正解の種類, タスク) に合うかを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    choice: dict
        選択肢 (roles か fits を持てば絞る)

    Returns
    ----------
    fits: bool
        合うか (答えていない項目では絞らない)
    """
    # 答えた項目だけでの比較
    answers = {"role": get_value(problem, f"{PART}.role"),
               "kinds": get_value(problem, f"{PART}.target.kind"),
               "tasks": get_value(problem, f"{PART}.task")}
    if "roles" in choice:
        fits = answers["role"] is None or answers["role"] in choice["roles"]
        return fits
    fits = "fits" not in choice or any(
        (answers["role"] is None or item.get("role") in (None,
                                                         answers["role"]))
        and all(answers[name] is None or name not in item
                or answers[name] in item[name] for name in ("kinds", "tasks"))
        for item in choice["fits"]
    )

    return fits


def active_value(problem: dict, key: str) -> Any:
    """
    問いが今の答えのもとで要る場合だけ, その答えを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    key: str
        問いの識別子

    Returns
    ----------
    value: Any
        答え (要らない問いの古い答えは None)
    """
    # タスクを変えた後の時刻の列などの除外 (版 1 は先に版 2 へ移行)
    if isinstance(problem, dict) and problem.get("format_version") == 1:
        problem = upgrade_problem(problem)
    key = current_key(problem, key)
    question = BY_ID.get(key)
    value = get_value(problem, key)
    if question is not None and not applies(problem, question):
        value = None

    return value


def data_location(problem: dict) -> Optional[str]:
    """
    読むデータの場所を返す関数

    写しが消えていれば元の場所 (data.origin) を使う

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    location: Optional[str]
        データの場所 (データを入れていなければ None)
    """
    # 写しが在ればそれ, 無ければ在る元の場所
    location = active_value(problem, "data.source")
    origin = get_value(problem, "data.origin")
    if location and origin and not Path(location).exists() and (
        Path(origin).exists()
    ):
        location = origin

    return location


def is_answered(value: Any) -> bool:
    """
    答えが空でないかを返す関数

    Parameters
    ----------
    value: Any
        答え

    Returns
    ----------
    answered: bool
        None, 空の文字列, 空の並びと辞書でなければ True
    """
    # 0 と False は答え
    answered = value is not None and not (
        isinstance(value, (str, list, dict)) and not value
    )

    return answered


def questions(problem: dict) -> list[dict]:
    """
    問いの並びを, 今の答え, 答えたか, 要るかと一緒に返す関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    items: list[dict]
        問い (id, section, kind, title, help, required, choices) に,
        value, answered, applies を足したものの並び
    """
    # 画面に出す順 (版 1 は先に版 2 へ移行)
    problem = upgrade_problem(problem)
    items = []
    for question in QUESTIONS:
        value = get_value(problem, question["id"])
        item = {key: item for key, item in question.items()
                if key not in ("when", "choices")}
        item["choices"] = choices_of(problem, question)
        item["value"] = copy.deepcopy(value)
        item["answered"] = is_answered(value)
        item["applies"] = applies(problem, question)
        items.append(item)

    return items


def clean_text(value: Any, key: str) -> Optional[str]:
    """
    文の答えを確かめ, 前後の空白を除いて返す関数

    Parameters
    ----------
    value: Any
        答え
    key: str
        項目の場所 (誤りの文に使う)

    Returns
    ----------
    text: Optional[str]
        文 (空なら None)
    """
    # 数も文にする (画面からは文で来る)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{key} は文で答えてください")
    # 行ごとの前後の空白と, 続く空の行を除く (改行は残す)
    lines = [" ".join(line.split()) for line in str(value).splitlines()]
    text = "\n".join(line for index, line in enumerate(lines)
                     if line or (index and lines[index - 1]))
    cleaned = text.strip() or None

    return cleaned


def clean_number(value: Any, key: str) -> Optional[Union[int, float]]:
    """
    数の答えを確かめて返す関数

    Parameters
    ----------
    value: Any
        答え (数か, 数として読める文字列)
    key: str
        項目の場所

    Returns
    ----------
    number: Optional[Union[int, float]]
        数 (空なら None. 整数は整数のまま)
    """
    # 空の文字列は空, 真偽値は数ではない. カンマは桁区切りだけ
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip()
    if isinstance(value, bool) or text.lower() in NOT_NUMBERS or (
        "," in text and not GROUPED_NUMBER.match(text)
    ):
        raise ValueError(f"{key} は有限の数で答えてください")
    try:
        number: Union[int, float] = float(text.replace(",", ""))
    except ValueError:
        raise ValueError(f"{key} は数で答えてください") from None
    if number != number or abs(number) == float("inf"):
        raise ValueError(f"{key} は有限の数で答えてください")
    if float(number).is_integer():
        number = int(number)

    return number


def clean_names(value: Any, key: str) -> Optional[list[str]]:
    """
    文の並びの答えを確かめ, 空白を除き, 重複を除いて返す関数

    Parameters
    ----------
    value: Any
        答え (文字列の並び)
    key: str
        項目の場所

    Returns
    ----------
    names: Optional[list[str]]
        並び (空なら None)
    """
    # 並びでなければ誤り
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} は並びで答えてください")
    names: list[str] = []
    for item in value:
        text = clean_text(item, key)
        if text and text not in names:
            names.append(text)
    cleaned = names or None

    return cleaned


def clean_levers(value: Any) -> Optional[list[dict]]:
    """
    要因と打てる手の並びの答えを確かめて返す関数

    Parameters
    ----------
    value: Any
        答え (cause, lever, ml を持つ辞書の並び)

    Returns
    ----------
    levers: Optional[list[dict]]
        並び (要因も手も空の行は除く. 空なら None)
    """
    # 1 行ずつの確かめ
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("decomposition は要因と手の並びで答えてください")
    levers = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("decomposition の 1 行は cause, lever, ml を持つ"
                             "辞書にしてください")
        cause = clean_text(item.get("cause"), "decomposition の cause")
        lever = clean_text(item.get("lever"), "decomposition の lever")
        if not cause and not lever:
            continue
        ml = item.get("ml", False)
        if not isinstance(ml, bool):
            raise ValueError("decomposition の ml は真偽値にしてください")
        levers.append({"cause": cause or "", "lever": lever or "", "ml": ml})
    cleaned = levers or None

    return cleaned


def clean_decision(value: Any) -> Optional[dict]:
    """
    意思決定の型と損得と制約の答えを確かめて返す関数

    Parameters
    ----------
    value: Any
        答え (type と, 型ごとの項目を持つ辞書. 値は数か列の名前)

    Returns
    ----------
    decision: Optional[dict]
        意思決定 (空の項目は除く. 空なら None)
    """
    # 型と, 型ごとに知っている項目だけ
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("action.decision は辞書で答えてください")
    kind = value.get("type")
    if kind not in DECISION_FIELDS:
        raise ValueError(
            "action.decision の type は "
            f"{', '.join(DECISION_FIELDS)} のどれかにしてください"
        )
    options = DECISION_OPTIONS.get(kind, {})
    unknown = sorted(
        set(value) - {"type"} - set(DECISION_FIELDS[kind]) - set(options)
    )
    if unknown:
        raise ValueError(
            f"{DECISION_NAMES[kind]} には {', '.join(unknown)} を書けません"
        )
    decision: dict = {"type": kind}
    for field in DECISION_FIELDS[kind]:
        item = value.get(field)
        decision_value = clean_amount(item, f"action.decision の {field}")
        if decision_value is not None:
            decision[field] = decision_value
    for field, choices in options.items():
        item = value.get(field)
        if item is None or item == "":
            continue
        if item not in choices:
            raise ValueError(
                f"action.decision の {field} は {', '.join(choices)} の"
                "どれかにしてください"
            )
        decision[field] = item

    return decision


def clean_amount(value: Any, key: str) -> Union[int, float, str, None]:
    """
    損得の値を, 数か行ごとの列の名前として確かめる関数

    Parameters
    ----------
    value: Any
        答え (数, 数として読める文字列, 列の名前)
    key: str
        項目の場所

    Returns
    ----------
    amount: Union[int, float, str, None]
        数か列の名前 (空なら None)
    """
    # 数として読めれば数, 読めない文字列は列の名前 (無限大と非数は誤り)
    try:
        amount: Union[int, float, str, None] = clean_number(value, key)
    except ValueError:
        text = clean_text(value, key) if isinstance(value, str) else None
        if text is None or text.lower() in NOT_NUMBERS or re.fullmatch(
            r"[\d,.+\-\s]+", text
        ) or text[0] in "+-.":
            raise
        amount = text

    return amount


def clean_baseline(problem: dict, value: Any) -> Optional[dict]:
    """
    今のやり方をデータで書いた答えを確かめて返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    value: Any
        答え (type と, periods か column を持つ辞書)

    Returns
    ----------
    baseline: Optional[dict]
        今のやり方 (空なら None)
    """
    # 種類ごとの項目 (期の数は 1 以上の整数, 列は使えるデータの列から)
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("type") not in (
        "lag", "mean", "column"
    ):
        raise ValueError("action.current の type は lag, mean, column の"
                         "どれかにしてください")
    kind = value["type"]
    if kind == "column":
        column = clean_text(value.get("column"), "action.current の column")
        columns = get_value(problem, "information.columns") or []
        if not column:
            raise ValueError("今のやり方の列を書いてください")
        if columns and column not in columns:
            raise ValueError(f"{column} は使えるデータの列にありません")
        baseline: dict = {"type": kind, "column": column}
        return baseline
    periods = clean_number(value.get("periods"), "action.current の periods")
    if not isinstance(periods, int) or periods < 1:
        raise ValueError("期の数は 1 以上の整数にしてください")
    baseline = {"type": kind, "periods": periods}

    return baseline


def clean_answer(problem: dict, question: dict, value: Any) -> Any:
    """
    問いの種類に合わせて答えを確かめ, 入れる形にする関数

    Parameters
    ----------
    problem: dict
        問題の設定
    question: dict
        問い
    value: Any
        答え

    Returns
    ----------
    cleaned: Any
        入れる値 (空なら None)
    """
    # 種類ごとの形 (空の文字列は答えの消去)
    key, kind = question["id"], question["kind"]
    if isinstance(value, str) and not value.strip() and kind != "text":
        value = None
    cleaned: Any
    if kind == "text":
        cleaned = clean_text(value, key)
        return cleaned
    if kind == "number":
        cleaned = clean_number(value, key)
        return cleaned
    if kind == "boolean":
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{key} ははいかいいえで答えてください")
        return value
    if kind == "choice":
        allowed = [choice["value"] for choice in question["choices"]]
        if value is not None and value not in allowed:
            raise ValueError(
                f"{key} は {', '.join(allowed)} のどれかにしてください"
            )
        return value
    if kind == "levers":
        cleaned = clean_levers(value)
        return cleaned
    if kind == "decision":
        cleaned = clean_decision(value)
        return cleaned
    if kind == "baseline":
        cleaned = clean_baseline(problem, value)
        return cleaned
    if kind == "source":
        cleaned = clean_text(value, key)
        if cleaned is not None:
            location = Path(cleaned).expanduser()
            if not location.is_absolute():
                raise ValueError(f"{key} は絶対パスで答えてください")
            cleaned = str(location)
        return cleaned
    if kind == "column":
        cleaned = clean_text(value, key)
        names = [cleaned] if cleaned else []
    else:
        names = clean_names(value, key) or []
        cleaned = names or None

    # 列の名前は, 使える列が決まっていればその中から
    columns = get_value(problem, "information.columns")
    if kind in ("column", "columns") and columns:
        missing = [name for name in names if name not in columns]
        if missing:
            raise ValueError(
                f"{key} の {', '.join(missing)} は使えるデータの列にありません"
            )

    return cleaned


def answer(problem: dict, key: str, value: Any) -> dict:
    """
    問いに答えて, 新しい問題の設定を返す関数

    元の問題の設定は変えない

    Parameters
    ----------
    problem: dict
        問題の設定
    key: str
        問いの識別子 (項目の場所. 版 1 の名前も使える)
    value: Any
        答え (None か空で答えを消す)

    Returns
    ----------
    updated: dict
        答えを入れた問題の設定 (版 2)
    """
    # 版 1 は先に版 2 へ移行 (版 1 の名前の問いは移した先). 知らない問いは誤り
    problem = upgrade_problem(problem)
    key = current_key(problem, key)
    if key not in BY_ID:
        raise ValueError(f"{key} という問いはありません")
    cleaned = clean_answer(problem, BY_ID[key], value)
    updated = copy.deepcopy(problem)
    set_value(updated, key, cleaned)

    # 最良を選ぶ指標に答えたら, 版 1 の場所に残した知らない指標は削除
    stale = updated.get("evaluation")
    if key == f"{PART}.evaluation.metric" and isinstance(stale, dict):
        stale.pop("metric", None)
        if not stale:
            updated.pop("evaluation")

    # データの場所を変えたら, 前の読み方は使わない (attach_data で読み直す)
    if key == "data.source" and get_value(problem, key) != cleaned:
        for name in ("loader", "options", "size", "modified"):
            set_value(updated, f"data.{name}", None)

    return updated


def load_problem(path: Union[str, Path]) -> dict:
    """
    問題の設定のファイルを読む関数

    Parameters
    ----------
    path: Union[str, Path]
        problem.yaml か, それを持つディレクトリ

    Returns
    ----------
    problem: dict
        問題の設定 (版 1 は版 2 に移す. ファイルは変えない)

    Raises
    ----------
    ValueError
        問題の設定のファイルでない場合と, 知っている版より新しい場合
    """
    # ディレクトリなら中の problem.yaml
    path = Path(path)
    if path.is_dir():
        path = path / "problem.yaml"
    with open(path, encoding="utf-8") as file:
        problem = yaml.safe_load(file)
    version = problem.get("format_version") if isinstance(
        problem, dict
    ) else None
    if not isinstance(version, int) or isinstance(version, bool) or (
        version < 1
    ):
        raise ValueError(
            f"{path.name} は問題の設定のファイルではありません "
            "(format_version がありません)"
        )
    problem = upgrade_problem(problem, str(path.name))

    return problem


def upgrade_problem(problem: dict, name: str = "問題の設定") -> dict:
    """
    版 1 の問題の設定を版 2 に移す関数

    版 1 の値は全て移す. 版 1 が前提にしていた答え (列の予測, 1 件は行,
    正解はデータの列, 一から学習) は埋める. 元は変えない

    Parameters
    ----------
    problem: dict
        問題の設定
    name: str = "問題の設定"
        誤りの文に使う名前

    Returns
    ----------
    upgraded: dict
        版 2 の問題の設定 (版 2 なら受け取ったもの)

    Raises
    ----------
    ValueError
        版が 1 以上の整数でない場合と, 知っている版より新しい場合
    """
    # 版 2 はそのまま, 版でない値と新しい版は読み込み無し (黙って読み違え
    # ないように)
    message = version_message(problem, name)
    if message:
        raise ValueError(message)
    if problem["format_version"] != 1:
        return problem
    upgraded = copy.deepcopy(problem)
    upgraded["format_version"] = FORMAT_VERSION

    # 受け持ちの項目の取り出し (空の項目も元の場所から削除. 知らない項目は
    # ml に残す). 版 1 の 4 つの他の指標は移さず, 版 1 の場所に残して確かめ
    # で知らない指標に
    moved: dict = {}
    for old, new in LEGACY_KEYS.items():
        value = get_value(problem, old)
        if old == "evaluation.metric" and isinstance(value, str) and (
            value not in V1_METRICS
        ):
            continue
        if value is not None:
            moved[new] = value
        remove_value(upgraded, old.split("."))
    ml = upgraded.get("ml")
    if not moved and not (isinstance(ml, dict) and ml.get("use") is True):
        return upgraded

    # 版 1 に手で書いた受け持ちの並びが崩れていれば, 移さずに元のまま
    # (確かめで形の誤り)
    parts = ml.get("parts") if isinstance(ml, dict) else None
    if parts not in (None, []) and not (
        isinstance(parts, list) and isinstance(parts[0], dict)
    ):
        upgraded = copy.deepcopy(problem)
        upgraded["format_version"] = FORMAT_VERSION
        return upgraded

    # 版 1 が前提にしていた答えと, 移した値
    task = moved.get(f"{PART}.task")
    defaults = {
        f"{PART}.role": "predict", f"{PART}.unit": "row",
        f"{PART}.output": "class" if task == "classification"
        else "value" if task in ("regression", "forecasting") else None,
        f"{PART}.target.source": "column", f"{PART}.target.kind": "column",
        f"{PART}.method": "scratch",
    }
    if not isinstance(upgraded.get("ml"), dict):
        upgraded["ml"] = {}
    if not upgraded["ml"].get("parts"):
        upgraded["ml"]["parts"] = [{}]
    for key, value in {**defaults, **moved}.items():
        if get_value(upgraded, key) is None:
            set_value(upgraded, key, value)

    return upgraded


def version_message(problem: Any, name: str) -> Optional[str]:
    """
    問題の設定の版が読めないときの誤りの文を作る関数

    Parameters
    ----------
    problem: Any
        問題の設定
    name: str
        誤りの文に使う名前

    Returns
    ----------
    message: Optional[str]
        版が 1 以上の整数でないか, 知っている版より新しい場合の文 (読める
        なら None)
    """
    # 真偽値と小数と文字列は版ではない
    version = problem.get("format_version") if isinstance(
        problem, dict
    ) else None
    message: Optional[str] = None
    if not isinstance(version, int) or isinstance(version, bool) or (
        version < 1
    ):
        message = (f"問題の設定ではありません ({name}. format_version が"
                   " 1 以上の整数でありません)")
    elif version > FORMAT_VERSION:
        message = newer_message(version, name)

    return message


def newer_message(version: int, name: str) -> str:
    """
    知っている版より新しい問題の設定の誤りの文を作る関数

    Parameters
    ----------
    version: int
        ファイルの版
    name: str
        ファイルの名前

    Returns
    ----------
    message: str
        何をすればよいかの文
    """
    # 読める版と, 直し方
    message = (
        f"新しい版 (format_version: {version}) の問題の設定です ({name}). "
        f"この ML-Quest が読めるのは版 {FORMAT_VERSION} までです. ML-Quest "
        "を新しくしてください"
    )

    return message


def save_problem(problem: dict, path: Union[str, Path]) -> dict:
    """
    問題の設定を, 更新の時刻を今にして YAML で書く関数

    Parameters
    ----------
    problem: dict
        問題の設定
    path: Union[str, Path]
        書くファイルか, ディレクトリ (中の problem.yaml)

    Returns
    ----------
    saved: dict
        書いた問題の設定
    """
    # 一時ファイルに書いてから置き換え (途中で止まっても壊れない).
    # 拡張子の無い場所はディレクトリ
    path = Path(path)
    if path.is_dir() or not path.suffix:
        path = path / "problem.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = copy.deepcopy(problem)
    saved["updated_at"] = now_text()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            yaml.safe_dump(saved, file, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    return saved


def read_columns(path: Union[str, Path]) -> list[str]:
    """
    データのファイルの先頭から列の名前を読む関数

    Parameters
    ----------
    path: Union[str, Path]
        csv, tsv, Excel, parquet のファイル

    Returns
    ----------
    columns: list[str]
        列の名前 (UTF-8 で読めない csv は Shift_JIS で読む)
    """
    # 形式は拡張子から. 中身は読まずに列の名前だけ
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        columns = header_of(path, suffix)
    except ImportError as error:
        raise ValueError(
            f"{path.name} を読むライブラリがありません (Excel は openpyxl, "
            f"parquet は pyarrow): {error}"
        ) from error
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{path.name} は空です") from error
    if not columns:
        raise ValueError(f"{path.name} に列がありません")
    repeated = sorted({name for name in columns if columns.count(name) > 1})
    if repeated:
        raise ValueError(
            f"{path.name} に同じ名前の列があります: {', '.join(repeated)}"
        )

    return columns


def header_of(path: Path, suffix: str) -> list[str]:
    """
    形式ごとに, ファイルの先頭から列の名前を読む関数

    Parameters
    ----------
    path: Path
        ファイル
    suffix: str
        小文字の拡張子

    Returns
    ----------
    columns: list[str]
        列の名前 (同じ名前もそのまま)
    """
    # parquet は形の情報だけ, csv は UTF-8 で読めなければ Shift_JIS
    if suffix == ".parquet":
        import pyarrow.parquet

        columns = [str(name) for name in
                   pyarrow.parquet.read_schema(path).names]
    elif suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path, nrows=0, header=None)
        columns = [str(name) for name in pd.read_excel(
            path, nrows=1, header=None
        ).iloc[0]] if frame.shape[1] else []
    elif suffix in (".csv", ".tsv"):
        columns = csv_header(path, "\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError(
            f"{path.name} の列を読めません (csv, tsv, Excel, parquet)"
        )

    return columns


def csv_header(path: Path, separator: str) -> list[str]:
    """
    区切りの文字のファイルの 1 行目を, 同じ名前を変えずに読む関数

    Parameters
    ----------
    path: Path
        ファイル
    separator: str
        区切りの文字

    Returns
    ----------
    columns: list[str]
        列の名前
    """
    # pandas は同じ名前に .1 を付けるので, 1 行目を値として読み込み
    try:
        frame = pd.read_csv(path, nrows=1, header=None, sep=separator,
                            encoding="utf-8-sig", dtype=str)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, nrows=1, header=None, sep=separator,
                            encoding="cp932", dtype=str)
    columns = [str(name) for name in frame.iloc[0]] if len(frame) else []

    return columns
