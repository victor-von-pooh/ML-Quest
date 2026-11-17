import copy
from typing import Any

from ml_quest.handoff import handoff
from ml_quest.problem import active_value, answer, get_value

# レシピの分割の割合の既定値 (データの分析の段階で変えられる)
VALID_RATIO = 0.15
TEST_RATIO = 0.15


def recipe_from(problem: dict) -> dict:
    """
    問題の設定から, ML-EDA のレシピの雛形とデータの読み方を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (受け渡せる形)

    Returns
    ----------
    start: dict
        load (場所, 読み込みの種類と設定. データを入れていなければ None)
        と recipe (ML-EDA のレシピの形. 目的の列, タスク, 除く列, 時系列の
        過去の値, 分割)
    """
    # 受け渡しの値 (渡せなければ誤り)
    stages = handoff(problem)
    data = stages["data"]

    # 除く列と, 時系列なら系列ごとの決める時点で分かる最も新しい期の値
    steps: list[dict] = []
    if data["drop"]:
        steps.append({"op": "drop_columns", "columns": list(data["drop"])})
    if data["task"] == "forecasting" and data["time_column"]:
        steps.append({
            "op": "lag", "columns": [data["target"]],
            "lags": [data["min_lag"] or 1], "by": list(data["series_by"]),
            "time": data["time_column"],
        })
    recipe = {
        "format_version": 1, "target": data["target"], "task": data["task"],
        "steps": steps,
        "split": {"method": data["split"], "valid": VALID_RATIO,
                  "test": TEST_RATIO, "seed": 0},
    }

    # データの読み方 (時刻の列は答えから)
    source = data.get("source")
    load = None
    if source:
        options = dict(source.get("options") or {})
        if data["time_column"]:
            options["time_column"] = data["time_column"]
        load = {"path": source["source"], "loader": source.get("loader"),
                "options": options}
    start = {"load": load, "recipe": recipe}

    return start


def dropped_columns(recipe: dict) -> set[str]:
    """
    レシピで落とす列を集める関数

    Parameters
    ----------
    recipe: dict
        ML-EDA のレシピ

    Returns
    ----------
    names: set[str]
        drop_columns の手順の列
    """
    # drop_columns の手順だけ
    names = {
        str(name) for step in recipe.get("steps") or []
        if step.get("op") == "drop_columns"
        for name in step.get("columns") or []
    }

    return names


def apply_recipe(problem: dict, recipe: dict) -> tuple[dict, list[str]]:
    """
    データの分析の段階で決めたレシピを問題の設定に書き戻す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    recipe: dict
        ML-EDA のレシピ

    Returns
    ----------
    result: tuple[dict, list[str]]
        書き戻した問題の設定と, 変えた点の文の並び (変えなければ空)
    """
    # 目的の列とタスク
    updated = copy.deepcopy(problem)
    changes = []
    for key, field, name in (("ml.target", "target", "目的の列"),
                             ("ml.task", "task", "タスク")):
        value = recipe.get(field)
        if value and value != get_value(updated, key):
            if value == "none":
                continue
            updated = answer(updated, key, value)
            changes.append(f"{name}を {value} に書き戻した")

    # 落とさなかった列のうち, 分からない列か使ってはいけない列にしていた
    # 列は, 分かる列に移す (使ってはいけない列には注意を添える)
    dropped = dropped_columns(recipe)
    known = list(active_value(updated, "information.known") or [])
    moved_any = False
    for key, warn in (("information.unknown", False),
                      ("information.forbidden", True)):
        names: list[Any] = list(active_value(updated, key) or [])
        kept = [name for name in names if name in dropped]
        moved = [name for name in names if name not in dropped]
        if not moved:
            continue
        updated = answer(updated, key, kept or None)
        known += [name for name in moved if name not in known]
        moved_any = True
        sentence = (f"{', '.join(moved)} をレシピで使うので, 予測の時点で"
                    "分かる列に移した")
        if warn:
            sentence += " (使ってはいけない列にしていた. 使ってよいかを確かめ"
            sentence += "てください)"
        changes.append(sentence)
    if moved_any:
        updated = answer(updated, "information.known", known)
    result = (updated, changes)

    return result
