import copy
from pathlib import Path
import re
import shutil
from typing import Any, Optional, Union
import uuid

import numpy as np
import pandas as pd
import yaml

from ml_quest.check import (
    DECISION_TASKS, ITEMS_KINDS, finding, is_column_part
)
from ml_quest.problem import (
    DECISION_NAMES, active_value, answer, data_location, get_value,
    set_value, upgrade_problem
)
from ml_quest.questions import PART

# 行が少ないとみなす行の数
FEW_ROWS = 100

# 1 行を表の行として数える ML-EDA の種類 (他の種類は 1 行が画像や撮影の
# 1 件なので件で数える)
ROW_KINDS = ("table", "timeseries")

# 系列を分ける列の候補とする, 値の種類の上限
SERIES_UNIQUE = 50

# 分類の目的の列の候補とする, 値の種類の上限
CLASS_UNIQUE = 20

# 問題の設定に関わる ML-EDA の気付き (重さはそのまま)
RELEVANT_FINDINGS = {
    "target_missing", "class_imbalance", "high_missing", "duplicate_rows",
    "irregular_interval",
}

# 読み方として持たない設定 (時刻の列は答えから毎回付ける)
VOLATILE_OPTIONS = {"time_column"}

# 写しごとのサブディレクトリの名前 (16 進の桁数. 片付けはこの名前だけ)
COPY_DIGITS = 12
COPY_NAME = re.compile(f"[0-9a-f]{{{COPY_DIGITS}}}")

# ML-EDA のファイルの組のデータセットの読み方の名前と, 読める版の組 (ML-EDA
# の ITEMS_FORMAT_VERSIONS の写し. 版 3 は版 2 に中身の手順を足した版)
ITEMS_LOADER = "items"
ITEMS_VERSIONS = (2, 3)

# 1 件の中身で学ぶデータの, ML-EDA の読み込みの名前と種類 (ML-EDA の kind)
CONTENT_LOADERS = {"image_folder": "image", "text_lines": "text"}



def import_eda() -> Any:
    """
    任意の依存の ML-EDA を読み込む関数

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
    # 使うときだけの読み込み
    try:
        import ml_eda
    except ImportError as error:
        raise ValueError(
            "データを読むには ML-EDA が要ります (ml-quest[data] で入れる)"
        ) from error
    module = ml_eda

    return module


def read_data(
    path: Union[str, Path], loader: Optional[str], options: dict,
    time_column: Optional[str]
) -> tuple[Any, Optional[str]]:
    """
    ML-EDA の読み込みでデータを読み, 答えた時刻の列を合わせる関数

    Parameters
    ----------
    path: Union[str, Path]
        データの場所
    loader: Optional[str]
        読み込みの種類 (None なら ML-EDA が判定)
    options: dict
        読み込みの設定 (時刻の列を除く)
    time_column: Optional[str]
        答えた時刻の列

    Returns
    ----------
    result: tuple[Any, Optional[str]]
        ML-EDA の Data と, 時刻の列の知らせ (答えた列が無い場合など)
    """
    # まず推定で読み (系列を分ける列も推定する), 時刻の列が答えと違えば
    # 答えた列を日時の列に指定して読み直し
    eda = import_eda()
    location = Path(path).expanduser()
    data = eda.load_data(location, loader=loader, **options)
    note = None
    columns = [str(name) for name in data.frame.columns]
    if time_column and time_column not in columns:
        note = (f"答えた時刻の列 {time_column} がデータにありません "
                "(時刻の列を答え直すか, データを入れ直してください)")
    elif time_column and data.roles.get("time") != time_column:
        data = eda.load_data(location, loader=loader,
                             **{**options, "time_column": time_column})
    result = (data, note)

    return result


def source_of(data: Any) -> dict:
    """
    ML-EDA の Data から, 問題の設定に持つ読み方を作る関数

    Parameters
    ----------
    data: Any
        ML-EDA の Data

    Returns
    ----------
    source: dict
        loader, options (時刻の列を除く), size, modified
    """
    # 時刻の列は答えから毎回付けるので持たない
    raw = data.source or {}
    options = {key: value for key, value in (raw.get("options") or {}).items()
               if key not in VOLATILE_OPTIONS}
    source = {"loader": raw.get("loader"), "options": options or None,
              "size": raw.get("size"), "modified": raw.get("modified")}

    return source


def content_kind(stages: dict) -> Optional[str]:
    """
    入れたデータが 1 件の中身で学ぶ種類 (画像か文) なら, その種類を返す関数

    Parameters
    ----------
    stages: dict
        受け渡し (データの分析の source の loader から決める)

    Returns
    ----------
    kind: Optional[str]
        image か text (表と時系列と, データを入れていなければ None)
    """
    # 入れたデータの読み込みの名前. 表の中の文の列 (text_column) で読んだ
    # ものも ML-EDA では文
    source = stages["data"].get("source") or {}
    kind = CONTENT_LOADERS.get(source.get("loader"))
    if kind is None and (source.get("options") or {}).get("text_column"):
        kind = "text"

    return kind


def dataset_info(location: Union[str, Path]) -> Optional[dict]:
    """
    ML-EDA の書き出したデータセットの説明を返す関数

    Parameters
    ----------
    location: Union[str, Path]
        データの場所

    Returns
    ----------
    info: Optional[dict]
        dataset.yaml の中身 (dataset.yaml を持つディレクトリでなければ
        None)

    Raises
    ----------
    ValueError
        dataset.yaml が読めないか, 辞書でない場合
    """
    # ML-EDA に頼らない YAML の直接の読み込み (版の確かめは呼ぶ側)
    path = Path(location) / "dataset.yaml"
    if not path.is_file():
        info = None
        return info
    try:
        info = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{path} を読めません: {error}") from error
    if not isinstance(info, dict):
        raise ValueError(f"{path} の中身が辞書ではありません")

    return info


def items_message(problem: dict) -> Optional[str]:
    """
    入れたデータセットが受け持ちに合わない理由を返す関数

    ファイルの組の受け持ち (予測するの mask と point_cloud) には, 受け持ち
    と正解の種類が同じ版 2 か版 3 のデータセットだけが使える. 他の受け持ちでは
    データセットを見ない (列の予測が dataset.yaml で止まらないように)

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    message: Optional[str]
        合わない理由 (合うか, ファイルの組の受け持ちでないか, データセット
        を入れていなければ None)
    """
    # ファイルの組の受け持ちで, データセットを入れた場合だけ (元のファイル
    # は書き出す前)
    role = active_value(problem, f"{PART}.role")
    kind = active_value(problem, f"{PART}.target.kind")
    location = data_location(problem)
    if role != "predict" or kind not in ITEMS_KINDS:
        message: Optional[str] = None
        return message
    try:
        info = dataset_info(location) if location else None
    except ValueError as error:
        message = str(error)
        return message
    if info is None:
        message = None
        return message

    # 版, 受け持ち, 正解の種類の比べ
    version = info.get("format_version")
    target = info.get("target")
    target = target if isinstance(target, dict) else {}
    message = None
    if isinstance(version, int) and version > max(ITEMS_VERSIONS):
        message = (f"入れたデータセットの版 {version} は, この ML-Quest では"
                   "読めません (新しい ML-Quest を使ってください)")
    elif not is_items_version(version):
        message = (f"入れたデータセットの版が {version} です. 正解の種類が "
                   f"{kind} の受け持ちには, ML-EDA の版 2 か版 3 のデータ"
                   "セット (export_image_masks か export_capture_clouds の"
                   "書き出し) を入れてください")
    elif info.get("role") != role or target.get("kind") != kind:
        message = (f"入れたデータセットは受け持ち {info.get('role')}, 正解の"
                   f"種類 {target.get('kind')} ですが, 問題の設定は受け持ち "
                   f"{role}, 正解の種類 {kind} です (データを入れ直すか, "
                   "答えを直してください)")

    return message


def copy_data(location: Path, copy_to: Path) -> Path:
    """
    データのファイルかディレクトリを, 写し先のディレクトリに写す関数

    Parameters
    ----------
    location: Path
        元の場所
    copy_to: Path
        写し先のディレクトリ (絶対パス. 無ければ作る. 同じ名前は置き換える)

    Returns
    ----------
    copied: Path
        写した場所

    Raises
    ----------
    ValueError
        写し先が相対パスか, 元のディレクトリの中の場合, 元の場所に名前が
        無い場合, 空きが足りない場合
    """
    # 写し先の確かめ (相対パスと元のディレクトリの中は誤り. .. などは正規化)
    if not copy_to.is_absolute():
        raise ValueError(f"{copy_to} は絶対パスにしてください")
    location = location.resolve()
    copy_to = copy_to.resolve()
    if not location.name:
        raise ValueError(f"{location} は写せません")
    copied = copy_to / location.name
    if copied == location:
        return copied
    if location.is_dir() and copy_to.is_relative_to(location):
        raise ValueError("写し先が元のディレクトリの中です")

    # 空きの確かめ. 他と重ならない一時の名前に写してから置き換え (途中で
    # 失敗しても前の写しが残る. リンクは指す先を写し, 元から独立)
    copy_to.mkdir(parents=True, exist_ok=True)
    needed = size_of(location)
    free = shutil.disk_usage(copy_to).free
    if needed > free:
        raise ValueError(
            f"写すのに {needed:,} バイト要りますが, 空きは {free:,} バイト"
            "です"
        )
    temporary = copy_to / f".{location.name}.{uuid.uuid4().hex}.copying"
    try:
        if location.is_dir():
            shutil.copytree(location, temporary)
            remove_path(copied)
        else:
            shutil.copy2(location, temporary)
        temporary.replace(copied)
    finally:
        remove_path(temporary)

    return copied


def size_of(location: Path) -> int:
    """
    ファイルかディレクトリの大きさ (バイト) を返す関数

    Parameters
    ----------
    location: Path
        場所 (ディレクトリの中のリンクは指す先の大きさ)

    Returns
    ----------
    size: int
        大きさの合計
    """
    # ディレクトリは中のファイルの合計
    if not location.is_dir():
        size = location.stat().st_size
        return size
    size = sum(item.stat().st_size for item in location.rglob("*")
               if item.is_file())

    return size


def same_file(first: Path, second: Path) -> bool:
    """
    2 つのファイルの大きさと更新の時刻 (ナノ秒) が同じかを返す関数

    Parameters
    ----------
    first: Path
        1 つ目
    second: Path
        2 つ目

    Returns
    ----------
    same: bool
        どちらもファイルで, 大きさと更新の時刻が同じか
    """
    # ディレクトリは中を見ないので同じとしない
    if not (first.is_file() and second.is_file()):
        return False
    old, new = first.stat(), second.stat()
    same = (old.st_size, old.st_mtime_ns) == (new.st_size, new.st_mtime_ns)

    return same


def prune_copies(copy_to: Union[str, Path], keep: Optional[str]) -> None:
    """
    写し先のディレクトリから, 今のデータの場所の写し以外を消す関数

    Parameters
    ----------
    copy_to: Union[str, Path]
        attach_data に渡した写し先のディレクトリ
    keep: Optional[str]
        残すデータの場所 (写し先の外か None なら全て消す)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        残す場所が相対パスの場合
    """
    # attach_data が作った写しごとのサブディレクトリ (12 桁の 16 進の名前)
    # のうち, 今の場所を含むもの以外. 他のファイルは消さない
    root = Path(copy_to).expanduser()
    if keep and not Path(keep).expanduser().is_absolute():
        raise ValueError(f"{keep} は絶対パスにしてください")
    if not root.is_dir():
        return
    kept = Path(keep).expanduser().resolve() if keep else None
    for child in root.iterdir():
        if not COPY_NAME.fullmatch(child.name) or child.is_symlink() or (
            not child.is_dir()
        ):
            continue
        if kept is not None and kept.is_relative_to(child.resolve()):
            continue
        remove_path(child)


def remove_path(path: Path) -> None:
    """
    ファイル, ディレクトリ, シンボリックリンクを消す関数

    Parameters
    ----------
    path: Path
        消す場所 (無ければ何もしない)

    Returns
    ----------
    None
    """
    # リンクは指す先を消さずにリンクだけを消去
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def interval_of(data: Any, time: Optional[str]) -> Optional[float]:
    """
    時刻の列の間隔 (秒) を, 最も多い差から求める関数

    Parameters
    ----------
    data: Any
        ML-EDA の Data
    time: Optional[str]
        時刻の列

    Returns
    ----------
    seconds: Optional[float]
        間隔 (時刻の列が無いか読めなければ None)
    """
    # 重ならない時刻の差の最頻値
    if not time or time not in data.frame.columns:
        return None
    times = pd.to_datetime(data.frame[time], errors="coerce").dropna()
    steps = pd.Series(np.sort(times.unique())).diff().dropna()
    steps = steps[steps > pd.Timedelta(0)]
    if steps.empty:
        return None
    seconds = float(steps.mode().iloc[0].total_seconds())

    return seconds


def attach_data(
    problem: dict, path: Union[str, Path],
    copy_to: Optional[Union[str, Path]] = None
) -> dict:
    """
    手元のデータを読み, 場所と読み方と列の名前を入れる関数

    元の問題の設定は変えない. 写し先を渡すとデータを写し, 写した場所を
    データの場所にする

    Parameters
    ----------
    problem: dict
        問題の設定
    path: Union[str, Path]
        データの場所 (絶対パス)
    copy_to: Optional[Union[str, Path]] = None
        写し先のディレクトリ (絶対パス. None なら写さない)

    Returns
    ----------
    updated: dict
        data (source, origin, loader, options, size, modified,
        interval_seconds) と information.columns を入れた問題の設定

    Raises
    ----------
    ValueError
        相対パスの場合, ML-EDA が無い場合, 読めない場合
    """
    # 絶対パスだけ. 写してから 1 回だけ読み, 読めなければ新しい写しを消す.
    # 版 1 は先に版 2 へ移行
    problem = upgrade_problem(problem)
    location = Path(path).expanduser()
    target = Path(copy_to).expanduser() if copy_to is not None else None
    if not location.is_absolute() or (target and not target.is_absolute()):
        raise ValueError(f"{path} と写し先は絶対パスで書いてください")
    location = location.resolve()
    copied, fresh = copy_place(problem, location, target)
    try:
        if fresh is not None:
            copied = copy_data(location, fresh)
        if is_items(copied or location):
            updated = attach_items(problem, location, copied)
            return updated
        data, _ = read_data(copied or location, None, {}, None)
    except Exception as error:
        if fresh is not None:
            remove_path(fresh)
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"データを読めません: {error}") from error

    # 場所と読み方と列の名前 (写したなら元の場所も. 時刻の間隔も)
    updated = answer(problem, "data.source", str(copied or location))
    origin = None
    if copied == location:
        same = str(location) == get_value(problem, "data.source")
        origin = get_value(problem, "data.origin") if same else None
    elif copied is not None:
        origin = str(location)
    set_value(updated, "data.origin", origin)
    for key, value in source_of(data).items():
        set_value(updated, f"data.{key}", copy.deepcopy(value))
    time = active_value(problem, f"{PART}.time_column") or data.roles.get(
        "time"
    )
    set_value(updated, "data.interval_seconds", interval_of(data, time))
    columns = [str(name) for name in data.frame.columns]
    updated = answer(updated, "information.columns", columns)

    return updated


def is_items_version(version: Any) -> bool:
    """
    dataset.yaml の版がファイルの組の版 (2 か 3) かを返す関数

    Parameters
    ----------
    version: Any
        dataset.yaml の format_version

    Returns
    ----------
    items: bool
        読める版の組にある整数か (真偽値は除く)
    """
    # True == 1 などの取り違えを除く
    items = (isinstance(version, int) and not isinstance(version, bool)
             and version in ITEMS_VERSIONS)

    return items


def is_items(location: Path) -> bool:
    """
    データの場所が ML-EDA のファイルの組のデータセット (版 2 か版 3) かを
    返す関数

    Parameters
    ----------
    location: Path
        データの場所

    Returns
    ----------
    items: bool
        dataset.yaml の版が 2 か 3 のディレクトリか (読めない説明は False.
        読むときに ML-EDA が誤りを出す)
    """
    # 版 1 は ML-EDA の読み込みへ
    try:
        info = dataset_info(location)
    except ValueError:
        info = None
    items = info is not None and is_items_version(info.get("format_version"))

    return items


def attach_items(
    problem: dict, location: Path, copied: Optional[Path]
) -> dict:
    """
    ファイルの組のデータセットを確かめ, 場所と読み方を入れる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    location: Path
        元の場所 (resolve 済み)
    copied: Optional[Path]
        写した場所 (写さないなら None)

    Returns
    ----------
    updated: dict
        data (source, origin, loader, options, size, modified,
        interval_seconds) を入れた問題の設定 (列の名前は入れない)
    """
    # 読めるかの確かめ (読めなければ ML-EDA の誤り)
    place = copied or location
    import_eda().load_items(place)

    # 場所と読み方 (写したなら元の場所も). 列と時刻は持たない
    updated = answer(problem, "data.source", str(place))
    origin = None
    if copied == location:
        same = str(location) == get_value(problem, "data.source")
        origin = get_value(problem, "data.origin") if same else None
    elif copied is not None:
        origin = str(location)
    values = {
        "origin": origin, "loader": ITEMS_LOADER, "options": None,
        "size": size_of(place),
        "modified": (place / "dataset.yaml").stat().st_mtime,
        "interval_seconds": None,
    }
    for key, value in values.items():
        set_value(updated, f"data.{key}", value)

    return updated


def items_overview(problem: dict, location: str) -> dict:
    """
    入れたファイルの組のデータセットの概要と確かめを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    location: str
        データセットの場所

    Returns
    ----------
    result: dict
        overview (表でないので None), suggestions (空), findings (版,
        件数, 正解の種類と出どころの info, 版 3 の中身の手順の info, 受け
        持ちに合わなければ error)
    """
    # 説明から件数, 分割, 正解の種類と出どころ (読めなければ error)
    try:
        info = dataset_info(location) or {}
    except ValueError as error:
        result = {"overview": None, "suggestions": [], "findings": [
            finding("error", "data.source", str(error))
        ]}
        return result
    target = info.get("target")
    target = target if isinstance(target, dict) else {}
    origin = target.get("origin")
    origin = origin if isinstance(origin, dict) else {}
    splits = info.get("splits")
    splits = splits if isinstance(splits, dict) else {}
    counts = ", ".join(f"{name} {value.get('items')}"
                       for name, value in splits.items()
                       if isinstance(value, dict))
    findings = [finding(
        "info", "data.source",
        f"ML-EDA の版 {info.get('format_version')} のデータセット "
        f"{info.get('name')}: 正解の種類 "
        f"{target.get('kind')}, 分割 ({counts or 'なし'}), 正解の出どころ "
        f"{origin.get('type', 'unknown')}"
        + (f" ({origin['name']})" if origin.get("name") else "")
    )]
    steps = entry_steps(info)
    if steps:
        findings.append(finding(
            "info", "data.source",
            "書き出しでかけた中身の手順: " + ". ".join(
                f"{name}: {describe_steps(value)}"
                for name, value in steps.items()
            ) + " (推論でも新しい 1 件に同じ手順をかける)"
        ))

    # 受け持ちに合うか (列の予測には使えないことも) と, 検証用の分割
    message = items_message(problem)
    if is_column_part(problem):
        message = ("ファイルの組のデータセットは列の予測には使えません (受け"
                   "持ちと正解の種類を答え直すか, 表のデータを入れてくださ"
                   "い)")
    if message is not None:
        findings.append(finding("error", "data.source", message))
    if "valid" not in splits:
        findings.append(finding(
            "warning", "data.source",
            "検証用の分割がありません. 最も良いモデルを選ぶために, 検証用の"
            "割合を 0 より大きくして書き出し直してください"
        ))
    result = {"overview": None, "suggestions": [], "findings": findings}

    return result


def entry_steps(info: dict) -> dict:
    """
    データセットの説明から, 項目ごとの中身の手順を集める関数

    Parameters
    ----------
    info: dict
        dataset.yaml の中身

    Returns
    ----------
    steps: dict
        項目の名前から手順の並び (版 3 で手順を持つ項目だけ. 辞書でない
        手順は除く)
    """
    # 版 3 の entries.<名前>.steps (版 2 は持たない)
    entries = info.get("entries")
    entries = entries if isinstance(entries, dict) else {}
    steps = {}
    for name, entry in entries.items():
        value = entry.get("steps") if isinstance(entry, dict) else None
        kept = [step for step in value or [] if isinstance(step, dict)]
        if kept:
            steps[str(name)] = kept

    return steps


def describe_steps(steps: list) -> str:
    """
    中身の手順の並びを, 人が読む文にする関数

    Parameters
    ----------
    steps: list
        手順の並び ({op: resize, width: 64, height: 64} など)

    Returns
    ----------
    text: str
        resize (width 64, height 64), grayscale のような文
    """
    # 手順の名前と引数をかける順に
    parts = []
    for step in steps:
        values = ", ".join(f"{key} {value}" for key, value in step.items()
                           if key != "op")
        parts.append(f"{step.get('op')}" + (f" ({values})" if values else ""))
    text = ", ".join(parts)

    return text


def copy_place(
    problem: dict, location: Path, target: Optional[Path]
) -> tuple[Optional[Path], Optional[Path]]:
    """
    データを写す場所を決める関数

    写しごとに別のサブディレクトリにするので, 入れ直しで読めなくても今の
    写しは残る. 同じ元のファイルが変わっていなければ今の写しを使う

    Parameters
    ----------
    problem: dict
        問題の設定
    location: Path
        元の場所 (resolve 済み)
    target: Optional[Path]
        写し先のディレクトリ (None なら写さない)

    Returns
    ----------
    place: tuple[Optional[Path], Optional[Path]]
        使う今の写し (あれば) と, 新しく写すサブディレクトリ (写さないなら
        None)

    Raises
    ----------
    ValueError
        写し先そのものをデータにする場合
    """
    # 写さない, 写し先の中のデータ, 変わっていない同じ元のファイル
    if target is None:
        place: tuple[Optional[Path], Optional[Path]] = (None, None)
        return place
    root = target.resolve()
    if location == root:
        raise ValueError("写し先そのものはデータにできません")
    if location.is_relative_to(root):
        place = (location, None)
        return place
    source = get_value(problem, "data.source")
    current = Path(source) if isinstance(source, str) else None
    if current is not None and get_value(problem, "data.origin") == str(
        location
    ) and current.is_relative_to(root) and same_file(current, location):
        place = (current, None)
        return place
    place = (None, root / uuid.uuid4().hex[:COPY_DIGITS])

    return place


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
        数でない, カテゴリか 2 値の列 (ML-EDA は種類の少ない整数の列も
        カテゴリとするので, 型の名前で数を除く)
    """
    # 2 種類だけの文字の列は boolean の判定
    dtype = str(info.get("dtype", "")).lower()
    numeric = dtype.startswith(("int", "uint", "float", "bool"))
    label = info.get("type") in ("categorical", "boolean") and not numeric

    return label


def is_panel(frame: pd.DataFrame, time: str, keys: list[str]) -> bool:
    """
    列 (の組み合わせ) が, 同じ時刻に複数の系列を並べる列かを返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        データの表
    time: str
        時刻の列
    keys: list[str]
        確かめる列 (店, 品など)

    Returns
    ----------
    panel: bool
        同じ時刻に複数の組があり, 時刻と組にするとほとんど重ならないか
    """
    # 時刻ごとの組の数と, (時刻, 組) の重なり (5 % 未満)
    pairs = frame[[time, *keys]].dropna()
    if pairs.empty:
        return False
    several = pairs.groupby(time).size().max() > 1
    unique = pairs.duplicated().mean() < 0.05
    panel = bool(several and unique)

    return panel


def series_columns(
    problem: dict, profiled: dict, data: Any, skip: set[str]
) -> list[str]:
    """
    系列を分ける列の候補を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    profiled: dict
        ML-EDA の profile の結果
    data: Any
        ML-EDA の Data
    skip: set[str]
        除く列

    Returns
    ----------
    series: list[str]
        1 つで系列を分ける列か, 組み合わせで分ける列の並び (無ければ空)
    """
    # 時刻の列 (答え, ML-EDA の推定, 日時の列が 1 つならその列). 分から
    # なければ候補なし
    series: list[str] = []
    times = [str(n) for n, info in profiled["columns"].items()
             if info.get("type") == "datetime"]
    time = active_value(problem, f"{PART}.time_column") or data.roles.get(
        "time"
    ) or (times[0] if len(times) == 1 else None)
    target = active_value(problem, f"{PART}.target.column")
    frame = data.frame
    if not time or time not in frame.columns:
        return series
    labels = [
        str(n) for n, info in profiled["columns"].items()
        if is_label(info) and str(n) not in skip and n not in (target, time)
        and (info.get("unique") or 0) <= SERIES_UNIQUE
    ]
    series = [name for name in labels if is_panel(frame, time, [name])]
    if not series and len(labels) > 1 and is_panel(frame, time, labels):
        series = labels

    return series


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


def flagged_columns(profiled: dict) -> dict[str, set[str]]:
    """
    ML-EDA の気付きの種類ごとに, 気付きの付いた列を集める関数

    Parameters
    ----------
    profiled: dict
        ML-EDA の profile の結果

    Returns
    ----------
    flagged: dict[str, set[str]]
        気付きの種類から列の名前
    """
    # 気付きの columns の和
    flagged: dict[str, set[str]] = {}
    for item in profiled["findings"]:
        flagged.setdefault(item["code"], set()).update(
            str(name) for name in item.get("columns") or []
        )

    return flagged


def target_candidates(
    profiled: dict, skip: set[str], task: Optional[str]
) -> list[str]:
    """
    目的の列の候補を, 答えたタスクに合う型で返す関数

    Parameters
    ----------
    profiled: dict
        ML-EDA の profile の結果
    skip: set[str]
        除く列 (ID のような列, 定数の列, 時刻と系列の列)
    task: Optional[str]
        答えたタスク (無ければ両方)

    Returns
    ----------
    names: list[str]
        候補の列の名前
    """
    # 回帰と予測は数, 分類は種類の少ない列
    names = []
    for name, info in profiled["columns"].items():
        if str(name) in skip:
            continue
        numeric = info.get("type") == "numeric" or (
            str(info.get("dtype", "")).lower().startswith(
                ("int", "uint", "float")
            )
        )
        few = info.get("type") in ("categorical", "boolean") and (
            info.get("unique") or 0
        ) <= CLASS_UNIQUE
        if task in ("regression", "forecasting") and numeric:
            names.append(str(name))
        elif task == "classification" and few:
            names.append(str(name))
        elif task is None and (numeric or few):
            names.append(str(name))

    return names


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
        答えの候補の並び (値を入れてよいものだけ. 漏れの疑いは確かめで
        知らせる)
    """
    # 除く列 (ID のような列, 定数の列) と, 時刻の列
    flagged = flagged_columns(profiled)
    columns = profiled["columns"]
    times = [str(n) for n, info in columns.items()
             if info.get("type") == "datetime"]
    time = active_value(problem, f"{PART}.time_column") or data.roles.get(
        "time"
    )
    task = active_value(problem, f"{PART}.task")
    target = active_value(problem, f"{PART}.target.column")
    skip = flagged.get("id_like", set()) | flagged.get("constant", set())
    items = []

    # 系列を分ける列 (同じ時刻に複数の値が並ぶ, 数でない列. 1 つで分け
    # られなければ, 店と品のような組み合わせ)
    series = series_columns(problem, profiled, data, skip)

    # 目的の列 (答えたタスクに合う型)
    candidates = target_candidates(profiled, skip | set(times) | set(series),
                                   task)
    if candidates:
        items.append(suggestion(
            f"{PART}.target.column", candidates,
            "答えたタスクに合う型の列 (回帰と予測は数, 分類は種類の少ない"
            "列. ID のような列, 定数の列, 時刻と系列の列は除く)"
        ))

    # タスク (まだ答えていない場合だけ) と当たりのクラス
    if target and target in columns and not task:
        guess = (profiled.get("target") or {}).get("task")
        if guess == "regression" and data.kind == "timeseries":
            guess = "forecasting"
        if guess in ("classification", "regression", "forecasting"):
            items.append(suggestion(
                f"{PART}.task", [guess], "目的の列の型と, データが時系列か"
            ))
    if task == "classification" and target and target in data.frame.columns:
        values = class_values(data.frame[target])
        if 0 < len(values) <= CLASS_UNIQUE:
            items.append(suggestion(f"{PART}.positive", values,
                                    "目的の列の値 (多い順)"))

    # 時刻の列, 系列を分ける列, 使ってはいけない列
    if times:
        items.append(suggestion(f"{PART}.time_column", times, "日時の列"))
    if series:
        items.append(suggestion(
            f"{PART}.series_by", series,
            "同じ時刻に複数の値が並ぶ列 (店や品ごとの系列)"
        ))
    ids = sorted(flagged.get("id_like", set()))
    if ids:
        items.append(suggestion(
            "information.forbidden", ids,
            "ID のような列 (行ごとに違う値. 学習に使っても汎化しない)"
        ))

    return items


def class_values(column: pd.Series) -> list[str]:
    """
    目的の列の全ての値を, 多い順の文字列で返す関数

    Parameters
    ----------
    column: pd.Series
        目的の列

    Returns
    ----------
    values: list[str]
        値 (整数の小数 1.0 は 1 と書く)
    """
    # 欠損で小数になった整数のラベルは整数の形に
    counts = column.dropna().map(label_text).value_counts()
    values = [str(value) for value in counts.index]

    return values


def binary_message(
    problem: dict, task: Optional[str], values: list[str]
) -> Optional[str]:
    """
    分類で答える意思決定 (対象を選ぶ, 振り分ける) が, 2 クラスでない目的
    の列に付いていれば, その誤りの文を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    task: Optional[str]
        タスク
    values: list[str]
        目的の列の値 (class_values)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (食い違いが無ければ None)
    """
    # 推論の段階 (ML-Inference) は, 分類で答える意思決定を 2 クラスの分類の
    # 一式にだけ使う. 学習の後まで分からないので, データで先に知らせる.
    # 意思決定しない (none) は推論の段階に渡らない
    message: Optional[str] = None
    decision = active_value(problem, "action.decision")
    kind = decision.get("type") if isinstance(decision, dict) else None
    if task == "classification" and "classification" in DECISION_TASKS.get(
        kind, []
    ) and len(values) != 2:
        message = (
            f"{DECISION_NAMES[kind]}意思決定は 2 クラスの分類にだけ使えます"
            f" (目的の列の値は {len(values)} 種類: "
            f"{', '.join(values[:10])}). 目的の列を 2 つの値にまとめるか, "
            "意思決定を見直してください"
        )

    return message


def label_text(value: Any) -> str:
    """
    ラベルの値を文字列にする関数

    Parameters
    ----------
    value: Any
        ラベルの値

    Returns
    ----------
    text: str
        文字列 (整数の値の小数は整数の形)
    """
    # 1.0 は 1
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value)

    return text


def data_findings(
    problem: dict, profiled: dict, data: Any, note: Optional[str],
    suggestions: list[dict]
) -> list[dict]:
    """
    データと答えの食い違いと, 問題の設定に関わる気付きを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    profiled: dict
        ML-EDA の profile の結果
    data: Any
        ML-EDA の Data
    note: Optional[str]
        時刻の列の知らせ
    suggestions: list[dict]
        答えの候補 (系列の候補があるか)

    Returns
    ----------
    found: list[dict]
        確かめ (error, warning と info)
    """
    # 時刻の列, 行の数, 入れた後のデータの変化
    found = []
    if note:
        found.append(finding("warning", f"{PART}.time_column", note))
    rows = len(data.frame)
    unit = "行" if data.kind in ROW_KINDS else "件"
    if rows < FEW_ROWS:
        found.append(finding(
            "warning", "data.source",
            f"データが {rows} {unit}と少なく, 機械学習で学べることは限られ"
            "ます. 今のやり方やルールで足りないかも考えてください"
        ))
    raw = data.source or {}
    if get_value(problem, "data.modified") not in (None, raw.get(
        "modified"
    )) or get_value(problem, "data.size") not in (None, raw.get("size")):
        found.append(finding(
            "warning", "data.source",
            "データを入れた後にファイルが変わりました. 入れ直すと, 使える"
            "データの列も読み直します"
        ))
    columns = [str(name) for name in data.frame.columns]
    answered = get_value(problem, "information.columns") or []
    if answered and answered != columns:
        found.append(finding(
            "warning", "information.columns",
            "使えるデータの列が, 入れたデータの列と違います "
            "(データを入れ直すか, 列を答え直してください)"
        ))

    # 目的の列と当たりのクラス (全ての値で, 数は数として比べる)
    target = active_value(problem, f"{PART}.target.column")
    positive = active_value(problem, f"{PART}.positive")
    if target and target not in columns:
        found.append(finding(
            "warning", f"{PART}.target.column",
            f"目的の列 {target} がデータにありません. 目的の列に関わる候補"
            "は出せません"
        ))
    elif target and positive:
        values = class_values(data.frame[target])
        if label_text(number_or_text(positive)) not in values:
            found.append(finding(
                "warning", f"{PART}.positive",
                f"当たりのクラス {positive} が目的の列 {target} の値に"
                f"ありません (値: {', '.join(values[:10])})"
            ))
        guess = (profiled.get("target") or {}).get("task")
        task = active_value(problem, f"{PART}.task")
        message = binary_message(problem, task, values)
        if message:
            found.append(finding("error", "action.decision", message))
        if task in ("regression", "forecasting") and guess == (
            "classification"
        ):
            found.append(finding(
                "info", f"{PART}.task",
                f"目的の列 {target} は種類の少ない値なので, 分類とも読めます"
                " (量を決めるなら回帰か時系列の予測のまま)"
            ))

    # 漏れの疑い (値は入れさせず, 確かめてもらう. 既に分からない列か使っ
    # てはいけない列に入れた列は除く)
    decided = set(active_value(problem, "information.unknown") or []) | set(
        active_value(problem, "information.forbidden") or []
    )
    for item in profiled["findings"]:
        if item["code"] == "target_leak":
            names = [str(n) for n in item.get("columns") or []
                     if str(n) != target and str(n) not in decided]
            if not names:
                continue
            found.append(finding(
                "warning", "information.unknown",
                f"{', '.join(names)} は目的の列とほぼ同じ動きです. 予測の"
                "時点で分からないなら, 分からない列に入れてください (分かる"
                "なら使ってよい)"
            ))

    # 何期先まで (データの間隔で数えた期の数との比べ)
    found += horizon_findings(problem, profiled)

    # 問題の設定に関わる ML-EDA の気付き (重さはそのまま. 系列が並ぶ
    # データの「同じ時刻」は間隔の乱れではないので除く)
    paneled = bool(active_value(problem, f"{PART}.series_by")) or any(
        item["id"] == f"{PART}.series_by" for item in suggestions
    )
    for item in profiled["findings"]:
        if item["code"] == "irregular_interval" and paneled:
            continue
        if item["code"] in RELEVANT_FINDINGS:
            found.append(finding(
                "warning" if item["severity"] == "warning" else "info",
                "data.source", f"データの気付き: {item['message']}"
            ))

    return found


def number_or_text(text: str) -> Any:
    """
    文字列を, 数として読めれば数に, 読めなければそのまま返す関数

    Parameters
    ----------
    text: str
        文字列

    Returns
    ----------
    value: Any
        数か文字列
    """
    # 1 と 1.0 を同じ値として比べるため
    try:
        value: Any = float(text)
    except ValueError:
        value = text

    return value


def horizon_findings(problem: dict, profiled: dict) -> list[dict]:
    """
    何期先までがデータの期間に比べて長すぎないかを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定
    profiled: dict
        ML-EDA の profile の結果

    Returns
    ----------
    found: list[dict]
        確かめ (warning)
    """
    # データの間隔で数えた期の数 (予測の 1 期と間隔が違えば目安だけ)
    found: list[dict] = []
    series = profiled.get("timeseries") or {}
    horizon = active_value(problem, f"{PART}.horizon")
    interval = series.get("interval_seconds")
    if not (horizon and interval and series.get("start")
            and series.get("end")):
        return found
    span = (pd.Timestamp(series["end"]) - pd.Timestamp(series["start"])
            ).total_seconds()
    periods = span / interval + 1
    if horizon * 10 > periods:
        found.append(finding(
            "warning", f"{PART}.horizon",
            f"データは {interval / 86400:g} 日間隔で {periods:,.0f} 期分です."
            f" {horizon} 期先までの予測を確かめるには短めです (先の長さの "
            "10 倍ほどあるとよい. 予測の 1 期とデータの間隔が違えば目安)"
        ))

    return found


def amount_values(frame: pd.DataFrame, value: Any) -> pd.Series:
    """
    損得の値を, 行ごとの数の並びにする関数

    Parameters
    ----------
    frame: pd.DataFrame
        データの表
    value: Any
        数か列の名前

    Returns
    ----------
    values: pd.Series
        行ごとの値 (列の値の欠損は欠損のまま)
    """
    # 列の名前なら列の値, 数なら全ての行に同じ値
    if isinstance(value, str):
        values = pd.to_numeric(frame[value], errors="coerce")
    else:
        values = pd.Series(float(value), index=frame.index)

    return values


def acted(column: pd.Series) -> pd.Series:
    """
    行動したかの列を真偽値にする関数

    Parameters
    ----------
    column: pd.Series
        列 (真偽値, 数, yes や true の文字)

    Returns
    ----------
    flags: pd.Series
        行動したか (読めない値と欠損は欠損)
    """
    # 数は 0 より大きいか, 文字は決まった言葉 (読めない文字は欠損)
    if pd.api.types.is_bool_dtype(column):
        flags = column.astype("boolean")
        return flags
    if pd.api.types.is_numeric_dtype(column):
        flags = (column > 0).astype("boolean").where(column.notna())
        return flags
    text = column.astype(str).str.strip().str.lower()
    yes = text.isin(["yes", "y", "true", "1", "はい", "済"])
    no = text.isin(["no", "n", "false", "0", "いいえ", "未"])
    flags = yes.astype("boolean").where(yes | no)

    return flags


def baseline_values(
    frame: pd.DataFrame, current: dict, target: str, series: list[str],
    time: str, interval: float
) -> pd.Series:
    """
    今のやり方の決まりから, 行ごとの今のやり方の量を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        データの表
    current: dict
        今のやり方 (lag, mean, column)
    target: str
        目的の列
    series: list[str]
        系列を分ける列
    time: str
        時刻の列
    interval: float
        時刻の間隔 (秒. 1 期)

    Returns
    ----------
    values: pd.Series
        今のやり方の量 (前の期の値が無い行は欠損)
    """
    # 列はそのまま. k 期前と平均は ML-EDA の時刻でずらす手順と同じ扱い
    # (休みで日が飛んでも暦の k 期前)
    if current["type"] == "column":
        values = pd.to_numeric(frame[current["column"]], errors="coerce")
        return values
    eda = import_eda()
    k = int(current["periods"])
    params = {"columns": [target], "by": series, "time": time,
              "freq": f"{int(interval)}s", "match": "exact"}
    if current["type"] == "lag":
        params["lags"] = [k]
        name = f"{target}_lag{k}"
    else:
        params.update(lags=[], windows=[k])
        name = f"{target}_mean{k}"
    shifted = eda.recipe.apply_lag(frame, params, {})
    values = pd.to_numeric(shifted[name], errors="coerce")

    return values


def baseline_result(problem: dict, data: Any) -> Optional[dict]:
    """
    データの上で, 今のやり方の実際の損を問題の設定の損得で求める関数

    Parameters
    ----------
    problem: dict
        問題の設定
    data: Any
        ML-EDA の Data

    Returns
    ----------
    result: Optional[dict]
        total, rows, days, per_30_days, sentence か, 求められない理由
        (reason だけ). 今のやり方を書いていなければ None
    """
    # 目的の列, 行動, 今のやり方が揃うこと
    current = active_value(problem, "action.current")
    if not current:
        return None
    decision = active_value(problem, "action.decision") or {}
    target = active_value(problem, f"{PART}.target.column")
    frame = data.frame
    time = active_value(problem, f"{PART}.time_column") or data.roles.get(
        "time"
    )
    interval = get_value(problem, "data.interval_seconds") or interval_of(
        data, time
    )
    reason = None
    if not target or target not in frame.columns:
        reason = "目的の列がデータに無いので, 今のやり方の損は求めません"
    elif decision.get("type") == "select":
        reason = ("対象を選ぶ行動は, 行動した行の結果が行動の後の値なので, "
                  "今のやり方の得は求めません (試して確かめる)")
    elif current["type"] != "column" and (
        not time or time not in frame.columns or not interval
    ):
        reason = ("k 期前と平均は時刻の列で数えるので, 時刻の列が要ります "
                  "(無ければ今のやり方の値の列で書く)")
    elif current["type"] == "column" and current.get("column") not in (
        frame.columns
    ):
        reason = f"今のやり方の列 {current.get('column')} がデータにありません"
    if reason:
        return {"reason": reason}
    try:
        series = [name for name in active_value(problem, f"{PART}.series_by")
                  or [] if name in frame.columns]
        values = rows_result(problem, frame, decision, current, target,
                             series, time, interval)
    except (KeyError, ValueError, TypeError) as error:
        return {"reason": f"今のやり方の損を求められません: {error}"}
    if values is None or values.empty:
        return {"reason": "今のやり方の損を求められる行がありません"}

    # 期間 (時刻の列があれば, 間隔 1 つ分を足した日数) と 30 日あたり
    total = float(values.sum())
    days = None
    if time and time in frame.columns:
        times = pd.to_datetime(frame.loc[values.index, time], errors="coerce")
        if times.notna().any():
            span = (times.max() - times.min()).total_seconds() + (
                interval or 0
            )
            days = span / 86400
    per_30 = total / days * 30 if days is not None and days >= 1 else None
    period = f"{days:,.0f} 日で" if days is not None else "データの全てで"
    sentence = (f"今のやり方の損は, {period} {total:,.0f}"
                + (f" (30 日あたり {per_30:,.0f})" if per_30 is not None
                   else "")
                + f". {len(values):,} 行で求めた")
    result = {"total": total, "rows": int(len(values)), "days": days,
              "per_30_days": per_30, "sentence": sentence}

    return result


def rows_result(
    problem: dict, frame: pd.DataFrame, decision: dict, current: dict,
    target: str, series: list[str], time: Optional[str],
    interval: Optional[float]
) -> Optional[pd.Series]:
    """
    行ごとの今のやり方の損 (量と振り分け) を求める関数

    Parameters
    ----------
    problem: dict
        問題の設定
    frame: pd.DataFrame
        データの表
    decision: dict
        意思決定の型と損得
    current: dict
        今のやり方
    target: str
        目的の列
    series: list[str]
        系列を分ける列
    time: Optional[str]
        時刻の列
    interval: Optional[float]
        時刻の間隔 (秒)

    Returns
    ----------
    values: Optional[pd.Series]
        求められた行の損 (損得の値か今のやり方が欠けた行は除く)
    """
    # 量: 足りない損と余る損
    kind = decision["type"]
    y = pd.to_numeric(frame[target], errors="coerce")
    if kind == "quantity":
        placed = baseline_values(frame, current, target, series,
                                 str(time), float(interval or 0))
        under = amount_values(frame, decision["under_cost"])
        over = amount_values(frame, decision["over_cost"])
        keep = placed.notna() & y.notna() & under.notna() & over.notna()
        loss = under * np.maximum(y - placed, 0) + over * np.maximum(
            placed - y, 0
        )
        values = loss[keep]
        return values

    # 振り分け: 回したかの列と, 当たりか (読めない値は除く)
    positive = active_value(problem, f"{PART}.positive")
    if kind != "triage" or positive is None:
        return None
    flags = acted(frame[current["column"]])
    hit = frame[target].map(label_text) == label_text(
        number_or_text(str(positive))
    )
    miss = amount_values(frame, decision["miss_cost"])
    alarm = amount_values(frame, decision["false_alarm_cost"])
    keep = (flags.notna() & frame[target].notna() & miss.notna()
            & alarm.notna()).astype(bool)
    flagged = flags.fillna(False).astype(bool)
    loss = miss * (~flagged & hit) + alarm * (flagged & ~hit)
    values = loss[keep]

    return values


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
        overview (行, 列, 種類, 時刻の範囲と間隔, 見本で分析したか),
        suggestions, findings (データを入れていなければ None)
    """
    # 機械学習を使い, データを入れた場合だけ (版 1 は先に版 2 へ移行)
    problem = upgrade_problem(problem)
    source = data_location(problem)
    if not source:
        return None
    if get_value(problem, "data.loader") == ITEMS_LOADER:
        result = items_overview(problem, source)
        return result
    try:
        eda = import_eda()
        data, note = read_data(
            source, get_value(problem, "data.loader"),
            dict(get_value(problem, "data.options") or {}),
            active_value(problem, f"{PART}.time_column")
        )
        target = active_value(problem, f"{PART}.target.column")
        columns = [str(name) for name in data.frame.columns]
        profiled = eda.profile(
            data, target=target if target in columns else None
        )
    except Exception as error:
        result = {"overview": None, "suggestions": [], "findings": [
            finding("error", "data.source", f"データを読めません: {error}")
        ]}
        return result

    # 概要, 候補, 確かめ
    series = profiled.get("timeseries") or {}
    summary = profiled["overview"]
    overview = {
        "rows": summary.get("rows"), "columns": summary.get("columns"),
        "kind": summary.get("kind"), "sampled": summary.get("sampled"),
        "sample_rows": summary.get("sample_rows"),
        "start": series.get("start"), "end": series.get("end"),
        "interval_seconds": series.get("interval_seconds"),
    }
    suggestions = suggestions_of(problem, profiled, data)
    findings = data_findings(problem, profiled, data, note, suggestions)

    # 今のやり方の損 (指標の単位がお金か, まだ書いていなければ, 指標の
    # 今の値の候補にも)
    baseline = baseline_result(problem, data)
    if baseline is not None and "reason" in baseline:
        findings.append(finding("info", "action.current",
                                baseline["reason"]))
    elif baseline is not None:
        findings.append(finding("info", "action.current",
                                baseline["sentence"]))
        unit = str(get_value(problem, "issue.kpi.unit") or "")
        if not unit or "円" in unit:
            monthly = baseline["per_30_days"] is not None
            value = baseline["per_30_days"] if monthly else baseline["total"]
            suggestions.append(suggestion(
                "issue.kpi.current", [f"{value:.0f}"],
                "今のやり方の損をデータで求めた値 ("
                + ("30 日あたり" if monthly else "データの全て")
                + ". 指標が同じ定義で, 単位が合う場合に)"
            ))
    result = {
        "overview": overview, "suggestions": suggestions,
        "findings": findings, "baseline": baseline,
    }

    return result
