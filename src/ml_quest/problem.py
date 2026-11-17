import copy
from datetime import datetime
import os
from pathlib import Path
import re
from typing import Any, Optional, Union
import uuid

import pandas as pd
import yaml

from ml_quest.questions import BY_ID, QUESTIONS

# 問題の設定の形の版
FORMAT_VERSION = 1

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

# 桁区切りのカンマ付きの数 (1,200,000 など)
GROUPED_NUMBER = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")

# 数として読まない文字 (無限大と非数)
NOT_NUMBERS = {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}

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
    # 辞書の順のたどり
    value: Any = problem
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]

    return value


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
    # 消す場合は, 空になった親の辞書も消去
    parts = key.split(".")
    if value is None:
        remove_value(problem, parts)
        return
    node = problem
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def remove_value(node: dict, parts: list[str]) -> bool:
    """
    点でつないだ場所の項目を消し, 空になった辞書かを返す関数

    Parameters
    ----------
    node: dict
        たどる辞書 (書き換える)
    parts: list[str]
        項目の場所を点で分けた並び

    Returns
    ----------
    empty: bool
        消した後に node が空か
    """
    # 末端の消去と, 空になった途中の辞書の消去
    head = parts[0]
    if len(parts) == 1:
        node.pop(head, None)
    elif isinstance(node.get(head), dict) and remove_value(node[head],
                                                           parts[1:]):
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
    # 条件の並びの全てで, 項目の値が等しいか並びに入るか
    when = question.get("when") or []
    conditions = when if isinstance(when, list) else [when]
    needed = True
    for condition in conditions:
        value = get_value(problem, condition["id"])
        if "equals" in condition:
            needed = needed and value == condition["equals"]
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
        値と説明の並び (学習の指標はタスクに合うものだけ)
    """
    # タスクを持つ選択肢は, 今のタスクに合うものだけ
    task = get_value(problem, "ml.task")
    choices = [
        {"value": choice["value"], "label": choice["label"]}
        for choice in question.get("choices", [])
        if "tasks" not in choice or task is None or task in choice["tasks"]
    ]

    return choices


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
    # タスクを変えた後の時刻の列などの除外
    question = BY_ID.get(key)
    value = get_value(problem, key)
    if question is not None and not applies(problem, question):
        value = None

    return value


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
    # 画面に出す順
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
    unknown = sorted(set(value) - {"type"} - set(DECISION_FIELDS[kind]))
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
        問いの識別子 (項目の場所)
    value: Any
        答え (None か空で答えを消す)

    Returns
    ----------
    updated: dict
        答えを入れた問題の設定
    """
    # 知らない問いは誤り
    if key not in BY_ID:
        raise ValueError(f"{key} という問いはありません")
    cleaned = clean_answer(problem, BY_ID[key], value)
    updated = copy.deepcopy(problem)
    set_value(updated, key, cleaned)

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
        問題の設定 (新しい版でも読める項目はそのまま. 確かめで知らせる)
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

    return problem


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
