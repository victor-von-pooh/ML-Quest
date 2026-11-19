import copy
from typing import Any

from ml_quest.check import finding, is_column_part
from ml_quest.handoff import handoff
from ml_quest.problem import (
    active_value, answer, get_value, set_value, upgrade_problem
)
from ml_quest.questions import PART

# レシピの分割の割合の既定値 (データの分析の段階で変えられる)
VALID_RATIO = 0.15
TEST_RATIO = 0.15


def require_column_part(problem: dict, purpose: str) -> None:
    """
    列の予測 (予測するで, 正解の種類が表の列) でなければ止める関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    purpose: str
        誤りの文に使う, 何をしようとしたか

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        列の予測でない場合
    """
    # 他の受け持ちのデータの約束は, データの分析の段階の版の上げの後
    if not is_column_part(problem):
        raise ValueError(
            f"{purpose}は, 予測するで正解の種類が表の列の場合だけです (他の"
            "受け持ちと正解の種類はまだ扱えません)"
        )


def recipe_from(problem: dict) -> dict:
    """
    問題の設定から, ML-EDA のレシピの雛形とデータの読み方を作る関数

    データの分析の段階で決めたレシピを書き戻していれば, それを返す

    Parameters
    ----------
    problem: dict
        問題の設定 (受け渡せる形)

    Returns
    ----------
    start: dict
        load (場所, 読み込みの種類と設定. データを入れていなければ None),
        recipe (ML-EDA のレシピの形), notes (知らせの文の並び)

    Raises
    ----------
    ValueError
        受け渡せない問題の設定の場合と, 列の予測でない場合
    """
    # 受け渡しの値 (渡せなければ誤り. 版 1 は先に版 2 へ移行). 列の予測だけ
    problem = upgrade_problem(problem)
    stages = handoff(problem)
    require_column_part(problem, "レシピの下書き")
    data = stages["data"]
    notes = []

    # 書き戻したレシピがあればそれ. 無ければ雛形 (除く列, 系列ごとの決める
    # 時点で分かる最も新しい期の値, 分割)
    saved = get_value(problem, "data.recipe")
    if isinstance(saved, dict):
        recipe = copy.deepcopy(saved)
        notes += [item["message"] for item in recipe_conflicts(
            problem, recipe
        )[0] if item["id"] == f"{PART}.group_column"]
    else:
        recipe = skeleton_recipe(problem, data, notes)

    # データの読み方 (時刻の列は答えから)
    source = data.get("source")
    load = None
    if source:
        options = dict(source.get("options") or {})
        if data["time_column"]:
            options["time_column"] = data["time_column"]
        load = {"path": source["source"], "loader": source.get("loader"),
                "options": options}
    start = {"load": load, "recipe": recipe, "notes": notes}

    return start


def skeleton_recipe(problem: dict, data: dict, notes: list) -> dict:
    """
    受け渡しの値からレシピの雛形を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定
    data: dict
        受け渡しのデータの分析の値
    notes: list
        知らせの文の並び (書き足す)

    Returns
    ----------
    recipe: dict
        ML-EDA のレシピ
    """
    # 除く列
    steps: list[dict] = []
    if data["drop"]:
        steps.append({"op": "drop_columns", "columns": list(data["drop"])})

    # 時系列の過去の値 (時刻の間隔が分かれば時刻で, 分からなければ行で)
    if data["task"] == "forecasting":
        lag: dict[str, Any] = {
            "op": "lag", "columns": [data["target"]],
            "lags": [data["min_lag"] or 1], "by": list(data["series_by"]),
        }
        interval = get_value(problem, "data.interval_seconds")
        if data["time_column"] and interval:
            lag.update(time=data["time_column"], freq=f"{int(interval)}s")
        else:
            notes.append("時刻の間隔が分からないので, 過去の値は行でずらし"
                         "ます (データを入れると時刻でずらせます)")
        steps.append(lag)

    # 分割は受け渡しの値 (時系列は時刻の順, 相手をまとめる列があれば相手
    # ごと, 分類は層別, 他は無作為). 時刻の順でも相手の列を書くと, 重なり
    # を知らせる
    method = data["split"]
    group = data.get("group_column")
    split: dict[str, Any] = {"method": method, "valid": VALID_RATIO,
                             "test": TEST_RATIO, "seed": 0}
    if group:
        split["column"] = group
    recipe = {
        "format_version": 1, "target": data["target"], "task": data["task"],
        "steps": steps, "split": split,
    }

    return recipe


def recipe_columns(recipe: dict) -> tuple[set[str], int]:
    """
    レシピで落とす列と, 過去の値の最も小さいずらし幅を集める関数

    Parameters
    ----------
    recipe: dict
        ML-EDA のレシピ

    Returns
    ----------
    result: tuple[set[str], int]
        drop_columns の列と, lag の手順の最も小さいずらし幅 (移動平均は
        gap. 無ければ 0)
    """
    # drop_columns の列と lag の幅
    dropped: set[str] = set()
    smallest = 0
    for step in recipe.get("steps") or []:
        if step.get("op") == "drop_columns":
            dropped |= {str(name) for name in step.get("columns") or []}
        if step.get("op") == "lag":
            sizes = [int(v) for v in step.get("lags") or []]
            if step.get("windows"):
                sizes.append(int(step.get("gap", 1)))
            if sizes:
                low = min(sizes)
                smallest = low if not smallest else min(smallest, low)
    result = (dropped, smallest)

    return result


def apply_recipe(problem: dict, recipe: dict) -> dict:
    """
    データの分析の段階で決めたレシピを問題の設定に書き戻す関数

    自動で書き戻すのは目的の列とタスクだけ. 列の使い方の食い違いは, 方針を
    黙って変えないように, 食い違いか利用者が選ぶ問いとして返す

    Parameters
    ----------
    problem: dict
        問題の設定
    recipe: dict
        ML-EDA のレシピ

    Returns
    ----------
    result: dict
        problem (書き戻した問題の設定 (版 2). レシピも data.recipe に持つ),
        changes (変えた点の文), conflicts (確かめの形の食い違い),
        choices (列ごとの選ぶ問い. 分からない列をレシピで使った場合)
    """
    # 目的の列とタスク (新しい目的の列は列の使い方から外す. 版 1 は先に
    # 移す). 列の予測だけ
    updated = copy.deepcopy(upgrade_problem(problem))
    require_column_part(updated, "レシピの書き戻し")
    changes = []
    old_target = get_value(updated, f"{PART}.target.column")
    task = recipe.get("task")
    if task and task != "none" and task != get_value(updated, f"{PART}.task"):
        updated = answer(updated, f"{PART}.task", task)
        changes.append(f"タスクを {task} に書き戻した (後の問いを確かめて"
                       "ください)")
    target = recipe.get("target")
    if target and target != old_target:
        updated = answer(updated, f"{PART}.target.column", target)
        for key in ("information.known", "information.unknown",
                    "information.forbidden"):
            names = [n for n in active_value(updated, key) or []
                     if n != target]
            updated = answer(updated, key, names or None)
        changes.append(f"目的の列を {target} に書き戻した ({old_target} は"
                       "使ってよいかを決め直してください)")

    # 列の使い方と過去の値と分割の食い違い. 前の目的の列をレシピで使うなら,
    # 予測の時点で分かるかを選ぶ問い (黙って特徴にしない)
    conflicts, choices = recipe_conflicts(updated, recipe)
    dropped, _ = recipe_columns(recipe)
    if target and old_target and target != old_target and (
        old_target not in dropped
    ):
        choices.append({
            "column": old_target, "options": ["drop", "known"],
            "message": f"{old_target} は前の目的の列で, レシピで使っています. "
                       "予測したい値そのものは, ふつう予測の時点では分かり"
                       "ません (過去の値は過去の値の手順で使えます). レシピ"
                       "で落としますか, 予測の時点で分かる列にしますか",
        })
    set_value(updated, "data.recipe", copy.deepcopy(recipe))
    result = {"problem": updated, "changes": changes,
              "conflicts": conflicts, "choices": choices}

    return result


def recipe_conflicts(problem: dict, recipe: dict) -> tuple[list, list]:
    """
    レシピと問題の設定の, 列の使い方, 過去の値, 分割の食い違いを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    recipe: dict
        ML-EDA のレシピ

    Returns
    ----------
    result: tuple[list, list]
        食い違い (確かめの形) と, 選ぶ問い (列, 選択肢, 文)
    """
    # 目的の列を落とす手順と, 使ってはいけない列を使う (書き換えずに誤り)
    dropped, smallest = recipe_columns(recipe)
    conflicts = []
    target = recipe.get("target")
    if target and target in dropped:
        conflicts.append(finding(
            "error", f"{PART}.target.column",
            f"レシピが目的の列 {target} を落としています. 落とす手順から"
            "外してください"
        ))
    forbidden = [n for n in active_value(problem, "information.forbidden")
                 or [] if n not in dropped]
    if forbidden:
        conflicts.append(finding(
            "error", "information.forbidden",
            f"レシピが使ってはいけない列 {', '.join(forbidden)} を使って"
            "います. レシピで落とすか, 問題の設定で使ってよいと決め直して"
            "ください"
        ))

    # 分からない列を使う (分かる列に移すか, レシピで落とすかを選ぶ)
    choices = [
        {"column": name, "options": ["known", "drop"],
         "message": f"{name} は予測の時点で分からない列ですが, レシピで使って"
                    "います. 分かる列に移しますか, レシピで落としますか"}
        for name in active_value(problem, "information.unknown") or []
        if name not in dropped
    ]

    # 決める時点で分からない期の値と, 時系列の分割
    min_lag = active_value(problem, f"{PART}.available_lag")
    if min_lag and smallest and smallest < min_lag:
        conflicts.append(finding(
            "warning", f"{PART}.available_lag",
            f"レシピの過去の値が {smallest} 期前から始まっていますが, 決める"
            f"時点で分かるのは {min_lag} 期前からです (漏れになります)"
        ))
    method = (recipe.get("split") or {}).get("method")
    group = active_value(problem, f"{PART}.group_column")
    written = (recipe.get("split") or {}).get("column") or None
    if group != written:
        conflicts.append(finding(
            "warning", f"{PART}.group_column",
            f"相手をまとめる列が, 問題の設定では {group or 'なし'}, レシピの"
            f"分割では {written or 'なし'} です (どちらかに合わせてください)"
        ))
    if active_value(problem, f"{PART}.split_time_column") and method and (
        method != "time"
    ):
        conflicts.append(finding(
            "warning", f"{PART}.split_time_column",
            f"時刻の順で分けて確かめると決めたのに, 分割が {method} です"
        ))
    if active_value(problem, f"{PART}.task") == "forecasting" and method and (
        method != "time"
    ):
        conflicts.append(finding(
            "warning", f"{PART}.task",
            f"時系列の予測なのに, 分割が {method} です (先の期間で確かめる"
            "には時刻の順)"
        ))
    result = (conflicts, choices)

    return result


def resolve_choice(problem: dict, column: str, option: str) -> dict:
    """
    書き戻しの選ぶ問い (分からない列をレシピで使った) に答える関数

    Parameters
    ----------
    problem: dict
        問題の設定 (data.recipe を持つ)
    column: str
        列
    option: str
        known (分かる列に移す) か drop (レシピで落とす)

    Returns
    ----------
    updated: dict
        答えた後の問題の設定
    """
    # 分かる列に移すか, 書き戻したレシピから外す. 過去の値の手順はそのまま
    # (過去の値は分かる), 他の手順 (ワンホットなど) の対象から外し, 最後の
    # 落とす手順で元の列を落とす (前の手順が使っても失敗しないように)
    if option not in ("known", "drop"):
        raise ValueError("option は known か drop にしてください")
    updated = copy.deepcopy(upgrade_problem(problem))
    if option == "known":
        unknown = [n for n in active_value(updated, "information.unknown")
                   or [] if n != column]
        known = list(active_value(updated, "information.known") or [])
        updated = answer(updated, "information.unknown", unknown or None)
        updated = answer(updated, "information.known",
                         known + ([column] if column not in known else []))
        return updated
    recipe = copy.deepcopy(get_value(updated, "data.recipe") or {})
    steps = without_column(recipe.setdefault("steps", []), column)
    recipe["steps"] = steps
    dropped, _ = recipe_columns(recipe)
    if column not in dropped:
        last = steps[-1] if steps else None
        if last is None or last.get("op") != "drop_columns":
            last = {"op": "drop_columns", "columns": []}
            steps.append(last)
        last["columns"] = [*(last.get("columns") or []), column]
    set_value(updated, "data.recipe", recipe)

    return updated


def without_column(steps: list, column: str) -> list:
    """
    過去の値と落とす手順の他の手順の対象から, 列を外す関数

    Parameters
    ----------
    steps: list
        レシピの手順
    column: str
        外す列

    Returns
    ----------
    kept: list
        外した後の手順 (対象が無くなった手順は除く)
    """
    # columns を持つ手順は列を外し, 1 つの列 (column) の手順は除く
    kept = []
    for step in steps:
        if step.get("op") in ("lag", "drop_columns"):
            kept.append(step)
            continue
        if step.get("column") == column:
            continue
        names = step.get("columns")
        if isinstance(names, list) and column in names:
            rest = [name for name in names if name != column]
            if not rest:
                continue
            step = {**step, "columns": rest}
        kept.append(step)

    return kept
