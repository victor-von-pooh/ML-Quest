import copy
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from ml_quest.check import finding
from ml_quest.problem import active_value, answer, get_value, set_value

# 行が少ないとみなす行の数
FEW_ROWS = 100

# 系列を分ける列の候補とする, 値の種類の上限
SERIES_UNIQUE = 50

# 目的の列の候補とする, 少ない種類の値の上限 (分類)
CLASS_UNIQUE = 20


def import_eda() -> Any:
    """
    ML-EDA を読み込む関数 (任意の依存)

    Parameters
    ----------
    None

    Returns
    ----------
    module: Any
        ml_eda

    Raises
    ----------
    ValueError
        入っていない場合 (ml-quest[data] で入れる)
    """
    # 使うときだけ読み込む
    try:
        import ml_eda
    except ImportError as error:
        raise ValueError(
            "データを読むには ML-EDA が要ります (ml-quest[data] で入れる)"
        ) from error
    module = ml_eda

    return module


def read_data(
    path: Union[str, Path], loader: Optional[str], options: dict
) -> Any:
    """
    ML-EDA の読み込みでデータを読む関数

    Parameters
    ----------
    path: Union[str, Path]
        データの場所
    loader: Optional[str]
        読み込みの種類 (None なら ML-EDA が判定)
    options: dict
        読み込みの設定

    Returns
    ----------
    data: Any
        ML-EDA の Data
    """
    # 読めない場合は ML-EDA の誤りの文のまま
    eda = import_eda()
    data = eda.load_data(Path(path).expanduser(), loader=loader, **options)

    return data


def attach_data(problem: dict, path: Union[str, Path]) -> dict:
    """
    手元のデータを読み, 場所と読み方と列の名前を入れた問題の設定を返す
    関数

    Parameters
    ----------
    problem: dict
        問題の設定
    path: Union[str, Path]
        データの場所 (絶対パス)

    Returns
    ----------
    updated: dict
        data (source, loader, options) と information.columns を入れた
        問題の設定 (元は変えない)
    """
    # 時刻の列を答えていれば, 読み込みの日時の列に指定
    location = Path(path).expanduser()
    if not location.is_absolute():
        raise ValueError(f"{path} は絶対パスで書いてください")
    time_column = active_value(problem, "ml.time_column")
    options = {"time_column": time_column} if time_column else {}
    data = read_data(location, None, options)

    # 場所と読み方 (ML-EDA の Data.source) と列の名前
    source = getattr(data, "source", {}) or {}
    updated = answer(problem, "data.source", str(location))
    set_value(updated, "data.loader", source.get("loader"))
    set_value(updated, "data.options",
              copy.deepcopy(source.get("options")) or None)
    columns = [str(name) for name in data.frame.columns]
    updated = answer(updated, "information.columns", columns)

    return updated


def target_candidates(profiled: dict, skip: set) -> list[str]:
    """
    目的の列の候補 (数値か少ない種類の列) を返す関数

    Parameters
    ----------
    profiled: dict
        ML-EDA の profile の結果
    skip: set
        除く列 (ID のような列, 定数の列, 時刻の列)

    Returns
    ----------
    names: list[str]
        候補の列の名前
    """
    # 数値の列と, 種類の少ないカテゴリの列
    names = []
    for name, info in profiled["columns"].items():
        if name in skip:
            continue
        kind = info.get("type")
        if kind == "numeric" or (
            kind in ("categorical", "boolean")
            and (info.get("unique") or 0) <= CLASS_UNIQUE
        ):
            names.append(str(name))

    return names


def is_label(info: dict) -> bool:
    """
    列が文字の値の種類の列 (店や品の名前など) かを返す関数

    Parameters
    ----------
    info: dict
        ML-EDA の profile の列の説明

    Returns
    ----------
    label: bool
        カテゴリの列か, 文字の 2 値の列 (数の 0 と 1 は除く)
    """
    # 2 種類だけの文字の列は boolean と判定される
    dtype = str(info.get("dtype", ""))
    label = info.get("type") == "categorical" or (
        info.get("type") == "boolean" and not dtype.startswith(
            ("int", "float", "bool")
        )
    )

    return label


def suggestion(key: str, values: list, reason: str) -> dict:
    """
    答えの候補の 1 件を作る関数

    Parameters
    ----------
    key: str
        問いの識別子
    values: list
        候補の値
    reason: str
        候補とした理由

    Returns
    ----------
    item: dict
        id, values, reason
    """
    # 画面にそのまま出す形
    item = {"id": key, "values": values, "reason": reason}

    return item


def suggestions_of(
    problem: dict, profiled: dict, data: Any
) -> list[dict]:
    """
    データの分析から, 問いごとの答えの候補を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定
    profiled: dict
        ML-EDA の profile の結果
    data: Any
        ML-EDA の Data

    Returns
    ----------
    items: list[dict]
        答えの候補の並び
    """
    # 気付きの列 (ID のような列, 定数の列, 漏れの疑い)
    flagged: dict[str, set] = {}
    for item in profiled["findings"]:
        flagged.setdefault(item["code"], set()).update(
            str(name) for name in item.get("columns") or []
        )
    columns = profiled["columns"]
    times = [str(n) for n, info in columns.items()
             if info.get("type") == "datetime"]
    target = active_value(problem, "ml.target")
    skip = flagged.get("id_like", set()) | flagged.get("constant", set())
    items = []

    # 目的の列とタスク
    candidates = target_candidates(profiled, skip | set(times))
    if candidates:
        items.append(suggestion(
            "ml.target", candidates,
            "数値の列か, 種類の少ない列 (ID のような列, 定数の列, 時刻の列"
            "は除く)"
        ))
    if target and target in columns:
        task = (profiled.get("target") or {}).get("task")
        if task == "regression" and getattr(data, "kind", "") == "timeseries":
            task = "forecasting"
        if task in ("classification", "regression", "forecasting"):
            items.append(suggestion(
                "ml.task", [task],
                "目的の列の型と, データが時系列か (行動に合うかは 5 と 7 で)"
            ))
        top = columns[target].get("top") or []
        if task == "classification" and top:
            items.append(suggestion(
                "ml.positive", [str(v["value"]) for v in top],
                "目的の列の値 (多い順)"
            ))

    # 時刻の列と系列を分ける列
    if times:
        items.append(suggestion("ml.time_column", times, "日時の列"))
    found_series = (profiled.get("timeseries") or {}).get("series_by")
    series = [found_series] if isinstance(found_series, str) else list(
        found_series or []
    )
    series += [
        str(n) for n, info in columns.items()
        if is_label(info) and n not in skip and n != target
        and str(n) not in series
        and 1 < (info.get("unique") or 0) <= SERIES_UNIQUE
    ]
    if times and series:
        items.append(suggestion(
            "ml.series_by", series,
            "種類の少ない列 (店や品のように, 同じ時刻に複数の系列が並ぶなら)"
        ))

    # 予測の時点で分からない列 (漏れの疑い) と, 使ってはいけない列
    leaks = sorted(flagged.get("target_leak", set()) - {str(target)})
    if leaks:
        items.append(suggestion(
            "information.unknown", leaks,
            "目的の列とほぼ同じ動きの列. 予測の時点で分からないなら学習に"
            "使わない (分かるなら使ってよい)"
        ))
    ids = sorted(flagged.get("id_like", set()))
    if ids:
        items.append(suggestion(
            "information.forbidden", ids,
            "ID のような列 (行ごとに違う値. 学習に使っても汎化しない)"
        ))

    return items


def data_findings(
    problem: dict, profiled: dict
) -> list[dict]:
    """
    データと答えの食い違いと, ML-EDA の気付きを確かめの形で返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    profiled: dict
        ML-EDA の profile の結果

    Returns
    ----------
    found: list[dict]
        確かめ (warning と info)
    """
    # 行の数
    found = []
    rows = profiled["overview"].get("rows", 0)
    if rows < FEW_ROWS:
        found.append(finding(
            "warning", "data.source",
            f"データが {rows} 行と少なく, 機械学習で学べることは限られます. "
            "今のやり方やルールで足りないかも考えてください"
        ))

    # 当たりのクラスがデータにあるか
    target = active_value(problem, "ml.target")
    positive = active_value(problem, "ml.positive")
    columns = profiled["columns"]
    if target in columns and positive:
        values = [str(v["value"]) for v in columns[target].get("top") or []]
        complete = len(values) >= (columns[target].get("unique") or 0)
        if values and complete and positive not in values:
            found.append(finding(
                "warning", "ml.positive",
                f"当たりのクラス {positive} が目的の列 {target} の値に"
                f"ありません (値: {', '.join(values[:10])})"
            ))

    # 何期先までがデータの期間に比べて長い
    series = profiled.get("timeseries") or {}
    horizon = active_value(problem, "ml.horizon")
    interval = series.get("interval_seconds")
    if horizon and interval and series.get("start") and series.get("end"):
        span = (pd.Timestamp(series["end"]) - pd.Timestamp(series["start"])
                ).total_seconds()
        periods = span / interval + 1
        if horizon * 10 > periods:
            found.append(finding(
                "warning", "ml.horizon",
                f"データの期間は {periods:,.0f} 期で, {horizon} 期先までの"
                "予測を確かめるには短めです (予測の先の長さの 10 倍ほどあると"
                "よい)"
            ))

    # ML-EDA の気付き (知らせ)
    for item in profiled["findings"]:
        found.append(finding("info", "data.source",
                             f"データの気付き: {item['message']}"))

    return found


def inspect_data(problem: dict) -> Optional[dict]:
    """
    入れたデータを読み直して分析し, 概要, 答えの候補, 確かめを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    result: Optional[dict]
        overview (行, 列, 種類, 時刻の範囲と間隔), suggestions, findings
        (データを入れていなければ None)
    """
    # 入れたときの読み方. 時刻の列は今の答えを優先
    source = get_value(problem, "data.source")
    if not source:
        return None
    options = dict(get_value(problem, "data.options") or {})
    time_column = active_value(problem, "ml.time_column")
    if time_column:
        options["time_column"] = time_column
    try:
        data = read_data(source, get_value(problem, "data.loader"), options)
        eda = import_eda()
        target = active_value(problem, "ml.target")
        columns = [str(name) for name in data.frame.columns]
        profiled = eda.profile(
            data, target=target if target in columns else None
        )
    except (ValueError, OSError, KeyError) as error:
        result = {"overview": None, "suggestions": [], "findings": [
            finding("error", "data.source",
                    f"データを読めません: {error}")
        ]}
        return result

    # 概要, 候補, 確かめ
    series = profiled.get("timeseries") or {}
    overview = {
        "rows": profiled["overview"].get("rows"),
        "columns": profiled["overview"].get("columns"),
        "kind": profiled["overview"].get("kind"),
        "start": series.get("start"), "end": series.get("end"),
        "interval_seconds": series.get("interval_seconds"),
    }
    result = {
        "overview": overview,
        "suggestions": suggestions_of(problem, profiled, data),
        "findings": data_findings(problem, profiled),
    }

    return result
