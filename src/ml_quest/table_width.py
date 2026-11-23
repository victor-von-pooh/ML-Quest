from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ml_quest.data import import_eda
from ml_quest.problem import active_value, upgrade_problem
from ml_quest.questions import PART

# 種類の列の値の種類の上限 (超える列は ID のような列とみなして止まる.
# 表の生成の骨組みの train.py の MAX_CATEGORIES と同じ値)
MAX_CATEGORIES = 100

# 推論の一式の予測の表の列と重なる列の名前, 頭と終わり (ML-Inference の
# 表の出力の一式が使えない名前. 表の生成の骨組みの train.py と同じ組)
RESERVED_COLUMNS = ("sample", "seed", "status", "reason", "decision",
                    "output", "artifact", "tokens", "prediction",
                    "confidence")
RESERVED_PREFIXES = ("p:", "q:")
RESERVED_SUFFIX = " (入力)"

# 列の符号化を決めるときに欠損を見る分割 (表の生成の骨組みの train.py が
# 使う分割と同じ組)
ENCODING_SPLITS = ("train", "valid", "test")


def is_table_generation(problem: dict) -> bool:
    """
    受け持ちが表の生成 (作るで, 正解の種類が表) かを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 か版 2)

    Returns
    ----------
    table: bool
        表の生成の受け持ちか (要らなくなった問いの古い答えは除外)
    """
    # 受け持ちと正解の種類の答え
    problem = upgrade_problem(problem)
    table = active_value(problem, f"{PART}.role") == "generate" and (
        active_value(problem, f"{PART}.target.kind") == "table"
    )

    return table


def table_columns(frame: pd.DataFrame) -> list[dict]:
    """
    学習用の表から, 列ごとの符号化 (数の列の標準化と, 種類の列のワンホット)
    を決める関数 (表の生成の骨組みの train.py の table_columns と同じ決まり)

    Parameters
    ----------
    frame: pd.DataFrame
        学習用の表 (欠損の無いもの)

    Returns
    ----------
    columns: list[dict]
        列の順の name, type (number か category) と, 数の列は mean, std,
        integer (整数の列か), 種類の列は categories (値の並び)

    Raises
    ----------
    ValueError
        推論の一式の予測の表の列と重なる名前の列がある場合と, 種類の列の
        値の種類が MAX_CATEGORIES より多い場合
    """
    # 推論の一式の予測の表の列と重なる名前
    reserved = [str(name) for name in frame.columns
                if str(name) in RESERVED_COLUMNS
                or str(name).startswith(RESERVED_PREFIXES)
                or str(name).endswith(RESERVED_SUFFIX)]
    if reserved:
        raise ValueError(
            f"列 {', '.join(reserved)} の名前は推論の一式の予測の表の列と"
            f"重なります ({', '.join(RESERVED_COLUMNS)} と, "
            f"{' か '.join(RESERVED_PREFIXES)} で始まる名前, "
            f"「{RESERVED_SUFFIX}」で終わる名前は使えません). レシピの "
            "drop_columns で落とすか, 元の表の列の名前を変えて書き出し直して"
            "ください"
        )

    # 真偽値と数でない列は種類の列. 標準偏差が 0 の列は 1 で割る
    columns = []
    for name in frame.columns:
        values = frame[name]
        numeric = (pd.api.types.is_numeric_dtype(values)
                   and not pd.api.types.is_bool_dtype(values))
        if numeric:
            std = float(values.std(ddof=0))
            columns.append({
                "name": str(name), "type": "number",
                "mean": float(values.mean()),
                "std": std if std > 0 else 1.0,
                "integer": bool(pd.api.types.is_integer_dtype(values)),
            })
            continue
        categories = sorted(pd.unique(values).tolist(), key=str)
        if len(categories) > MAX_CATEGORIES:
            raise ValueError(
                f"種類の列 {name} の値の種類が {len(categories)} で, 上限 "
                f"{MAX_CATEGORIES} より多いです. ID のような列なら, レシピ"
                "の drop_columns で落として書き出し直してください"
            )
        columns.append({"name": str(name), "type": "category",
                        "categories": categories})

    return columns


def encoded_width(columns: list[dict]) -> int:
    """
    符号化した行の幅を求める関数

    Parameters
    ----------
    columns: list[dict]
        列ごとの符号化 (table_columns)

    Returns
    ----------
    width: int
        数の列の数と, 種類の列の値の種類の数の和
    """
    # 数の列は 1, 種類の列は値の種類の数
    width = sum(1 if column["type"] == "number"
                else len(column["categories"]) for column in columns)

    return width


def missing_columns(frames: dict) -> list[str]:
    """
    分割のどれかに欠損のある列を集める関数 (表の生成の骨組みの train.py の
    missing_columns と同じ決まり)

    Parameters
    ----------
    frames: dict
        分割の名前から表

    Returns
    ----------
    names: list[str]
        欠損のある列の名前 (列の順. 無ければ空)
    """
    # 全ての分割で見る
    names: list[str] = []
    for frame in frames.values():
        for name in frame.columns[frame.isna().any()]:
            if str(name) not in names:
                names.append(str(name))

    return names


def table_encoding(
    problem: dict, location: Union[str, Path]
) -> Optional[dict]:
    """
    表の生成の問題の設定なら, データセットの学習用の表から列の符号化と
    符号化した行の幅 (モデルの入力と出力の幅) を求める関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 か版 2)
    location: Union[str, Path]
        ML-EDA の表のデータセットのディレクトリ

    Returns
    ----------
    encoding: Optional[dict]
        width (符号化した行の幅), numbers (数の列の数), categories (種類の
        列の数), columns (列ごとの符号化. table_columns). 表の生成でなければ
        None. 幅は学習用の分割だけで決まるので, 検証用の分割が無くても
        止まらない (骨組みの train.py は学習の前に止まる)

    Raises
    ----------
    ValueError
        ML-EDA が無い場合, 表のデータセットでないか, 目的の列か欠損の
        ある列があるか, 学習用の分割が無い場合と, 列の符号化を決められない
        場合 (table_columns)
    FileNotFoundError
        データセットの説明 (dataset.yaml) が無い場合
    """
    # 表の生成でなければ求めない
    if not is_table_generation(problem):
        encoding = None
        return encoding

    # データセットの読み方は ML-EDA に任せる
    eda = import_eda()
    dataset = eda.load_dataset(Path(location).expanduser())
    if dataset.info.get("kind") != "table":
        raise ValueError(
            f"{location} は表のデータセットではありません (種類 "
            f"{dataset.info.get('kind')})"
        )
    if dataset.target is not None:
        raise ValueError(
            f"{location} には目的の列 ({dataset.target}) があります. 目的の"
            "列の無いレシピ (task: none) で書き出してください"
        )
    if "train" not in dataset.frames:
        raise ValueError(f"{location} に学習用の分割がありません")

    # 分割ごとの学習に使う列 (骨組みと同じく, どの分割の欠損でも止まる)
    frames = {split: frame[list(dataset.features)]
              for split, frame in dataset.frames.items()
              if split in ENCODING_SPLITS}
    missing = missing_columns(frames)
    if missing:
        raise ValueError(
            f"欠損のある列があります ({', '.join(missing)}). レシピに "
            "impute (埋める) か drop_missing (行を落とす) の手順を足して"
            "書き出し直してください"
        )

    # 学習用の表から決めた符号化と幅
    columns = table_columns(frames["train"].reset_index(drop=True))
    numbers = sum(column["type"] == "number" for column in columns)
    encoding = {"width": encoded_width(columns), "numbers": numbers,
                "categories": len(columns) - numbers, "columns": columns}

    return encoding
