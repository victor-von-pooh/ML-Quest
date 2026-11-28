from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ml_quest.check import is_rank_part
from ml_quest.data import import_eda
from ml_quest.problem import active_value, get_value, upgrade_problem
from ml_quest.questions import PART

# 時刻の順に分ける, 検証用とテスト用の反応の行の割合の既定 (並べるの骨組み
# の config の data.valid と data.test と同じ値)
RANK_VALID = 0.1
RANK_TEST = 0.1


def read_reactions(
    frame: pd.DataFrame, columns: dict
) -> tuple[pd.DataFrame, list[str]]:
    """
    読んだ反応の表を, 利用者, 品物, 時刻の列にそろえる関数 (並べるの骨組み
    の train.py の read_reactions と同じ決まり)

    Parameters
    ----------
    frame: pd.DataFrame
        反応の表 (1 行が 1 つの反応)
    columns: dict
        user, item, time, reaction (反応の表の列の名前. 反応の強さは無ければ
        None)

    Returns
    ----------
    reactions: pd.DataFrame
        user, item, time の列 (欠けた行と, 反応の強さが 0 以下か数でない
        行は除く. 時刻の順)
    notes: list[str]
        除いた行の知らせ

    Raises
    ----------
    ValueError
        列が無い場合と, 時刻が数でも日時でもない場合
    """
    # 答えた列があるか
    names = {"user": columns["user"], "item": columns["item"],
             "time": columns["time"]}
    if columns.get("reaction"):
        names["reaction"] = columns["reaction"]
    missing = [name for name in names.values() if name not in frame.columns]
    if missing:
        raise ValueError(f"反応の表に {', '.join(map(str, missing))} の列が"
                         "ありません. config の rank の列の名前を直して"
                         "ください")
    reactions = frame[list(names.values())].copy()
    reactions.columns = list(names)

    # 時刻は数か日時 (日時は数にして順に並べる)
    time = reactions["time"]
    if not pd.api.types.is_numeric_dtype(time):
        parsed = pd.to_datetime(time, errors="coerce")
        if parsed.notna().sum() == 0 and time.notna().any():
            raise ValueError(f"時刻の列 {columns['time']} を数か日時として"
                             "読めません")
        reactions["time"] = parsed

    # 欠けた行と, 反応の強さが 0 以下の行を除く
    notes = []
    before = len(reactions)
    reactions = reactions.dropna(subset=["user", "item", "time"])
    if len(reactions) < before:
        notes.append(f"利用者, 品物か時刻の欠けた {before - len(reactions)} "
                     "行を除きました")
    if "reaction" in reactions:
        strength = pd.to_numeric(reactions["reaction"], errors="coerce")
        kept = strength > 0
        if (~kept).any():
            notes.append(f"反応の強さ ({columns['reaction']}) が 0 以下か数"
                         f"でない {int((~kept).sum())} 行を除きました")
        reactions = reactions[kept]
    reactions = reactions.sort_values("time", kind="stable")
    reactions = reactions[["user", "item", "time"]].reset_index(drop=True)

    return reactions, notes


def split_by_time(
    reactions: pd.DataFrame, valid: float, test: float
) -> dict:
    """
    反応を時刻の順に学習用, 検証用, テスト用に分ける関数 (並べるの骨組みの
    train.py の split_by_time と同じ決まり)

    Parameters
    ----------
    reactions: pd.DataFrame
        時刻の順の反応 (user, item, time)
    valid: float
        検証用の反応の行の割合
    test: float
        テスト用の反応の行の割合

    Returns
    ----------
    splits: dict
        train, valid と, テスト用の割合が 0 より大きければ test の表.
        行の数の割合で切る時刻を決め, 同じ時刻の行は同じ分割に入れる

    Raises
    ----------
    ValueError
        割合の誤りと, 学習用か検証用が空になる場合
    """
    # 割合の確かめ
    if not (0 < valid < 1 and 0 <= test < 1 and valid + test < 1):
        raise ValueError("config の data.valid は 0 より大きく, data.test "
                         "は 0 以上で, 2 つの和は 1 より小さくしてください")

    # 切る時刻 (その時刻から先が次の分割)
    times = reactions["time"]
    count = len(reactions)
    valid_from = times.iloc[min(int(count * (1 - valid - test)), count - 1)]
    splits = {"train": reactions[times < valid_from]}
    if test > 0:
        test_from = times.iloc[min(int(count * (1 - test)), count - 1)]
        splits["valid"] = reactions[(times >= valid_from)
                                    & (times < test_from)]
        splits["test"] = reactions[times >= test_from]
    else:
        splits["valid"] = reactions[times >= valid_from]
    if splits["train"].empty or splits["valid"].empty:
        raise ValueError("時刻で分けると学習用か検証用の反応が空になります "
                         "(同じ時刻の反応が多いか, 反応が少ない). config の "
                         "data.valid と data.test を見直してください")
    splits = {name: frame.reset_index(drop=True)
              for name, frame in splits.items()}

    return splits


def make_index(train: pd.DataFrame) -> dict:
    """
    学習用の反応から, 利用者と品物の番号を決める関数 (並べるの骨組みの
    train.py の make_index と同じ決まり)

    Parameters
    ----------
    train: pd.DataFrame
        学習用の反応 (user, item)

    Returns
    ----------
    index: dict
        users と items (番号の順の元の値. 学習用に初めて出た順)
    """
    # 学習用にいない利用者と品物は番号を持たない
    index = {"users": list(pd.unique(train["user"])),
             "items": list(pd.unique(train["item"]))}

    return index


def rank_encoding(
    problem: dict, location: Union[str, Path], valid: float = RANK_VALID,
    test: float = RANK_TEST
) -> Optional[dict]:
    """
    並べるの問題の設定なら, 反応の表から学習用の利用者と品物の数 (2 塔の
    モデルの入力 user と item の大きさ) を求める関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 か版 2)
    location: Union[str, Path]
        反応の表のファイル (骨組みの config の data.dataset の場所)
    valid: float = RANK_VALID
        検証用の反応の行の割合 (config の data.valid)
    test: float = RANK_TEST
        テスト用の反応の行の割合 (config の data.test)

    Returns
    ----------
    encoding: Optional[dict]
        users (学習用の利用者の数), items (学習用の品物の数), rows (分割の
        名前から反応の行の数), notes (除いた行の知らせ). 見せただけの印の
        列があれば, 印が偽の行 (選んだ反応) だけで数える. 並べるで学習
        しなければ None. 数は学習用の分割だけで決まるので, 検証用に正解の
        ある利用者がいなくても止まらない (骨組みの train.py は学習の前に
        止まる)

    Raises
    ----------
    ValueError
        ML-EDA が無い場合, 反応の表の列が無いか時刻を読めない場合と, 割合
        の誤りか, 時刻で分けると学習用か検証用が空になる場合
    FileNotFoundError
        反応の表が無い場合
    """
    # 並べるで学習しなければ求めない
    problem = upgrade_problem(problem)
    if not is_rank_part(problem):
        encoding = None
        return encoding

    # 問題の手元のデータの読み方で読む (骨組みの LOADER と OPTIONS)
    path = Path(location).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"反応の表 {path} がありません")
    options = get_value(problem, "data.options")
    options = options if isinstance(options, dict) else {}
    loader = get_value(problem, "data.loader")
    columns = {key: active_value(problem, f"{PART}.rank.{key}")
               for key in ("user", "item", "time", "reaction", "shown",
                           "presentation")}

    # 見せただけの印の列があれば, ML-EDA の load_reactions で選んだ反応を
    # 読む (骨組みの train.py と同じ). 無ければ列のそろえ
    if columns["shown"]:
        read = import_eda().load_reactions(
            path, columns["user"], columns["item"], columns["time"],
            columns["reaction"], loader, shown=columns["shown"],
            presentation=columns["presentation"], **options
        )
        reactions = read.frame[["user", "item", "time"]]
        notes = list(read.notes)
    else:
        frame = import_eda().load_data(path, loader=loader, **options).frame
        reactions, notes = read_reactions(frame, columns)

    # 時刻の順に分け, 学習用にいる利用者と品物に番号
    splits = split_by_time(reactions, valid, test)
    index = make_index(splits["train"])
    encoding = {"users": len(index["users"]), "items": len(index["items"]),
                "rows": {name: len(part) for name, part in splits.items()},
                "notes": notes}

    return encoding
