import copy
from pathlib import Path
from typing import Any, Optional

from ml_quest.check import (
    finding, is_column_part, is_generate_part, is_items_part, is_policy_part
)
from ml_quest.data import EXPORT_FUNCTIONS, ITEMS_LOADER, PAIR_NAMES
from ml_quest.handoff import export_split, export_stages
from ml_quest.problem import (
    active_value, answer, data_location, get_value, set_value,
    upgrade_problem
)
from ml_quest.questions import PART

# レシピの分割の割合の既定値 (データの分析の段階で変えられる)
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 正解の種類から, ML-EDA のファイルの組の書き出しと, 入力と正解のディレク
# トリの名前 (ML-EDA の合成の見本と同じ)
EXPORTS = {kind: (function, *PAIR_NAMES[kind])
           for kind, function in EXPORT_FUNCTIONS.items()}

# レシピの分割のうち, 時刻の順を守るものと, 相手をまとめる列が要るもの
# (ML-EDA のレシピの split の method)
TIME_SPLITS = ("time", "group_time")
GROUP_SPLITS = ("group", "group_time", "stratified_group")

# 書き出しの分割のしかた (ML-EDA の版 2 の書き出しの split の method.
# レシピの分割と違い, 時刻と層化の分け方を持たない) と, 書いていない割合の
# ML-EDA の既定
SPLIT_METHODS = ("random", "group", "none")
SPLIT_DEFAULTS = {"valid": 0.2, "test": 0.0}

# 書き出しの設定の鍵と, 書き出しごとの他の引数 (中身の手順 item_steps を
# 渡すと ML-EDA は版 3 で書く)
EXPORT_KEYS = ("function", "inputs", "targets", "split", "groups", "origin",
               "options")
EXPORT_OPTIONS = {
    "export_image_masks": ("classes", "mask_suffix", "item_steps"),
    "export_capture_clouds": ("cloud_units", "item_steps"),
}

# 書き出しごとの, 中身の手順を渡せる項目の名前 (領域の手順は ML-EDA が画像
# の resize から決める)
STEP_ENTRIES = {
    "export_image_masks": ("image",),
    "export_capture_clouds": ("capture", "cloud"),
}


def require_column_part(problem: dict, purpose: str) -> None:
    """
    列の予測, ファイルの組, 画像の生成の受け持ちでなければ止める関数

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
        方策の受け持ち (データの分析の段階を通らない) の場合と, 列の予測
        でも, ファイルの組でも, 画像の生成の受け持ちでもない場合
    """
    # 方策は環境とのやり取りで集めるので, データの分析の段階が無い. 他の
    # 受け持ちのデータの約束は, データの分析の段階の版の上げの後
    if is_policy_part(problem):
        raise ValueError(
            f"{purpose}はありません. 方策は行動の結果を環境とのやり取りで集め"
            "るので, データの分析の段階 (レシピ) を通りません"
        )
    if not is_column_part(problem) and not is_items_part(problem) and (
        not is_generate_part(problem)
    ):
        raise ValueError(
            f"{purpose}は, 予測するで正解の種類が表の列, 画像の領域, 点群の"
            "場合と, 作るで正解の種類が画像の場合だけです (他の受け持ちと"
            "正解の種類はまだ扱えません)"
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
        recipe (ML-EDA のレシピの形. ファイルの組の受け持ちでは None),
        export (ファイルの組の書き出しの設定. 列の予測では None),
        notes (知らせの文の並び)

    Raises
    ----------
    ValueError
        骨組みを組めない確かめの error がある場合と, 列の予測でもファイル
        の組の受け持ちでもない場合
    """
    # 受け渡しの値 (骨組みを組めない error があれば誤り. 組める error は
    # 骨組みの書き出しと同じく止めず, 知らせに. 版 1 は先に版 2 へ移行)
    problem = upgrade_problem(problem)
    stages, exportable = export_stages(problem)
    require_column_part(problem, "レシピの下書き")
    data = stages["data"]
    remaining = [f"問題の設定に直すところが残っています ({item['id']}): "
                 f"{item['message']}" for item in exportable]
    if is_items_part(problem):
        start = {"load": load_of(data), "recipe": None,
                 "export": export_from(problem, data), "notes": []}
        start["notes"] = remaining + start["export"].pop("notes")
        return start
    notes = remaining

    # 書き戻したレシピがあればそれ. 無ければ雛形 (除く列, 系列ごとの決める
    # 時点で分かる最も新しい期の値, 分割)
    saved = get_value(problem, "data.recipe")
    if isinstance(saved, dict):
        recipe = copy.deepcopy(saved)
        notes += [item["message"] for item in recipe_conflicts(
            problem, recipe
        )[0] if item["id"] == f"{PART}.group_column"
            and not is_generate_part(problem)]
    elif is_generate_part(problem):
        recipe = generate_recipe(notes)
    else:
        recipe = skeleton_recipe(problem, data, notes)

    # データの読み方 (時刻の列は答えから)
    start = {"load": load_of(data), "recipe": recipe, "export": None,
             "notes": notes}

    return start


def load_of(data: dict) -> Optional[dict]:
    """
    受け渡しのデータの分析の値から, データの読み方を作る関数

    Parameters
    ----------
    data: dict
        受け渡しのデータの分析の値

    Returns
    ----------
    load: Optional[dict]
        path, loader, options (時刻の列は答えから. データを入れていなければ
        None)
    """
    # 場所, 読み込みの種類と設定
    source = data.get("source")
    load = None
    if source:
        options = dict(source.get("options") or {})
        if data["time_column"]:
            options["time_column"] = data["time_column"]
        load = {"path": source["source"], "loader": source.get("loader"),
                "options": options}

    return load


def export_from(problem: dict, data: dict) -> dict:
    """
    ファイルの組の受け持ちの, 書き出しの設定を返す関数

    書き戻した設定があり, 書き出しが正解の種類に合えばそれを返す

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. ファイルの組の受け持ち)
    data: dict
        受け渡しのデータの分析の値

    Returns
    ----------
    export: dict
        書き出しの設定 (function, inputs, targets, split, groups, origin,
        options) と, 知らせの文の並び (notes)
    """
    # 書き戻した設定 (書き出しが今の正解の種類に合う場合だけ)
    kind = active_value(problem, f"{PART}.target.kind")
    function, _, _ = EXPORTS[kind]
    saved = get_value(problem, "data.export")
    notes = []
    if isinstance(saved, dict) and saved.get("function") == function:
        export = {**copy.deepcopy(saved), "notes": notes}
        if not export.get("inputs") or not export.get("targets"):
            inputs, targets = pair_dirs(problem, kind, data, notes)
            export["inputs"] = export.get("inputs") or inputs
            export["targets"] = export.get("targets") or targets
        return export
    if isinstance(saved, dict):
        notes.append(f"書き戻した書き出しの設定 ({saved.get('function')}) は"
                     f"正解の種類 {kind} に合わないので, 下書きから始めます")

    # 下書き (ディレクトリ, 受け渡しの分割, 正解の出どころ)
    inputs, targets = pair_dirs(problem, kind, data, notes)
    export = {
        "function": function, "inputs": inputs, "targets": targets,
        "split": {"method": data["split"], "valid": VALID_RATIO,
                  "test": TEST_RATIO, "seed": 0},
        "groups": None, "origin": origin_of(problem), "options": {},
        "notes": notes,
    }

    return export


def pair_dirs(
    problem: dict, kind: str, data: dict, notes: list
) -> tuple[Optional[str], Optional[str]]:
    """
    入れたデータの場所から, 入力と正解のディレクトリを埋める関数

    Parameters
    ----------
    problem: dict
        問題の設定
    kind: str
        正解の種類 (mask か point_cloud)
    data: dict
        受け渡しのデータの分析の値
    notes: list
        知らせの文の並び (書き足す)

    Returns
    ----------
    dirs: tuple[Optional[str], Optional[str]]
        入力と正解のディレクトリ (分からなければ None)
    """
    # データを入れていないか, ファイルの組のデータセットを入れた場合は
    # 埋めない
    source = data.get("source") or {}
    location = data_location(problem)
    _, inputs, targets = EXPORTS[kind]
    dirs: tuple[Optional[str], Optional[str]] = (None, None)
    if not location:
        notes.append("データが入っていないので, 入力と正解のディレクトリを"
                     "選んでください")
        return dirs
    if source.get("loader") == ITEMS_LOADER:
        notes.append("入れたデータは書き出したデータセットです. 書き出し"
                     "直すなら, 元の入力と正解のディレクトリを選んで"
                     "ください")
        return dirs

    # 場所が 2 つのディレクトリを持つか, 入力のディレクトリで隣に正解が
    # あればその 2 つ. 他は場所を入力に
    place = Path(location)
    if (place / inputs).is_dir() and (place / targets).is_dir():
        dirs = (str(place / inputs), str(place / targets))
    elif place.name == inputs and (place.parent / targets).is_dir():
        dirs = (str(place), str(place.parent / targets))
    else:
        dirs = (str(place), None)
        notes.append("正解のディレクトリが分からないので選んでください (入力"
                     "は入れたデータの場所にしました)")

    return dirs


def origin_of(problem: dict) -> dict:
    """
    問題の設定の正解の出どころから, ML-EDA の origin を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    origin: dict
        type, name, settings (今の処理なら process と処理の名前と設定,
        他は unknown)
    """
    # 今の処理だけ ML-EDA の種類と 1 対 1 に当たる
    origin: dict[str, Any] = {"type": "unknown", "name": None, "settings": {}}
    if active_value(problem, f"{PART}.target.source") == "process":
        settings = active_value(problem, f"{PART}.target.process.settings")
        origin = {
            "type": "process",
            "name": active_value(problem, f"{PART}.target.process.name"),
            "settings": {"note": settings} if settings else {},
        }

    return origin


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


def generate_recipe(notes: list) -> dict:
    """
    画像の生成の受け持ちのレシピの雛形を作る関数

    Parameters
    ----------
    notes: list
        知らせの文の並び (書き足す)

    Returns
    ----------
    recipe: dict
        ML-EDA のレシピ (目的の列の無い task: none と, 無作為の分割)
    """
    # 条件の無い生成なので目的の列もタスクも無い. 大きさはデータの分析で
    # 決める
    notes.append("画像の大きさがそろわないなら, 中身の手順 (item_steps) に "
                 "resize (width, height) を足してください")
    split = {"method": "random", "valid": VALID_RATIO, "test": TEST_RATIO,
             "seed": 0}
    recipe = {"format_version": 1, "task": "none", "steps": [],
              "split": split}

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
        ML-EDA のレシピ (ファイルの組の受け持ちでは書き出しの設定)

    Returns
    ----------
    result: dict
        problem (書き戻した問題の設定 (版 2). レシピも data.recipe に持つ.
        書き出しの設定は data.export),
        changes (変えた点の文), conflicts (確かめの形の食い違い),
        choices (列ごとの選ぶ問い. 分からない列をレシピで使った場合)
    """
    # 目的の列とタスク (新しい目的の列は列の使い方から外す. 版 1 は先に
    # 移す). ファイルの組の受け持ちは書き出しの設定
    updated = copy.deepcopy(upgrade_problem(problem))
    require_column_part(updated, "レシピの書き戻し")
    if is_items_part(updated):
        result = apply_export(updated, recipe)
        return result
    if is_generate_part(updated):
        result = apply_generate(updated, recipe)
        return result
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


def apply_generate(problem: dict, recipe: dict) -> dict:
    """
    画像の生成の受け持ちのレシピを問題の設定に書き戻す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 画像の生成の受け持ち. 写したもの)
    recipe: dict
        ML-EDA のレシピ

    Returns
    ----------
    result: dict
        apply_recipe と同じ形 (他の答えは変えず, 選ぶ問いは無い)
    """
    # 条件の無い生成は目的の列を使わない. 検証用の分割が無いと骨組みの
    # 学習が止まる
    conflicts = []
    if recipe.get("target") or recipe.get("task") not in (None, "none"):
        conflicts.append(finding(
            "error", "data.recipe",
            "画像の生成は条件の無い生成なので, レシピに目的の列とタスクを書か"
            "ないでください (task: none)"
        ))
    split = recipe.get("split")
    split = split if isinstance(split, dict) else {}
    if split.get("method", "random") == "none" or split.get("valid", 1) == 0:
        conflicts.append(finding(
            "warning", "data.recipe",
            "検証用の分割がありません. 骨組みの学習は最も良いモデルを選べず"
            "止まるので, 検証用の割合を 0 より大きくしてください"
        ))
    set_value(problem, "data.recipe", copy.deepcopy(recipe))
    result = {"problem": problem, "changes": [], "conflicts": conflicts,
              "choices": []}

    return result


def apply_export(problem: dict, export: dict) -> dict:
    """
    ファイルの組の書き出しの設定を問題の設定に書き戻す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 写した後のもの. ファイルの組の受け持ち)
    export: dict
        書き出しの設定

    Returns
    ----------
    result: dict
        problem (data.export に設定を持つ問題の設定), changes, conflicts,
        choices (いつも空)

    Raises
    ----------
    ValueError
        設定の形の誤り
    """
    # 形を確かめ, 書き出しと分割の食い違いを集めて持つ
    settings = checked_export(export)
    conflicts = export_conflicts(problem, settings)
    set_value(problem, "data.export", settings)
    steps = settings["options"].get("item_steps") or {}
    named = (f". 中身の手順は {', '.join(steps)}" if steps else "")
    result = {
        "problem": problem,
        "changes": [f"ファイルの組の書き出しの設定 ({settings['function']}"
                    f"{named}) を書き戻した"],
        "conflicts": conflicts, "choices": [],
    }

    return result


def checked_export(export: Any) -> dict:
    """
    書き出しの設定の形を確かめ, 決まった形にする関数

    Parameters
    ----------
    export: Any
        書き出しの設定

    Returns
    ----------
    settings: dict
        全ての鍵を持つ設定 (無い鍵は空)

    Raises
    ----------
    ValueError
        辞書でない, 知らない鍵か書き出し, 辞書でない分割, グループ,
        正解の出どころ, 他の引数, 文字でないディレクトリ, 中身の手順の形の
        誤りの場合
    """
    # 鍵と書き出し
    if not isinstance(export, dict):
        raise ValueError("書き出しの設定は辞書にしてください")
    unknown = sorted(str(key) for key in export if key not in EXPORT_KEYS)
    if unknown:
        raise ValueError(f"書き出しの設定の {', '.join(unknown)} は知らない"
                         "鍵です")
    function = export.get("function")
    if function not in EXPORT_OPTIONS:
        raise ValueError("書き出しの設定の function は "
                         f"{', '.join(EXPORT_OPTIONS)} のどれかにしてください")

    # ディレクトリ, 分割, グループ, 正解の出どころ, 他の引数
    for key in ("inputs", "targets"):
        if export.get(key) is not None and not isinstance(export[key], str):
            raise ValueError(f"書き出しの設定の {key} は文字にしてください")
    for key in ("split", "groups", "origin", "options"):
        if export.get(key) is not None and not isinstance(export[key], dict):
            raise ValueError(f"書き出しの設定の {key} は辞書にしてください")
    options = dict(export.get("options") or {})
    extra = sorted(str(key) for key in options
                   if key not in EXPORT_OPTIONS[function])
    if extra:
        raise ValueError(f"{function} の引数に {', '.join(extra)} はあり"
                         "ません")
    steps = checked_steps(function, options.pop("item_steps", None))
    if steps:
        options["item_steps"] = steps
    settings = {
        "function": function,
        "inputs": export.get("inputs") or None,
        "targets": export.get("targets") or None,
        "split": copy.deepcopy(dict(export.get("split") or {})),
        "groups": {str(key): str(value) for key, value in
                   (export.get("groups") or {}).items()} or None,
        "origin": copy.deepcopy(dict(export.get("origin") or {})),
        "options": copy.deepcopy(options),
    }

    return settings


def checked_steps(function: str, item_steps: Any) -> Optional[dict]:
    """
    書き出しの設定の中身の手順の形を確かめ, 写しを返す関数

    手順の名前と引数は ML-EDA が書き出すときに確かめる

    Parameters
    ----------
    function: str
        書き出しの関数の名前
    item_steps: Any
        項目の名前から手順の並び

    Returns
    ----------
    steps: Optional[dict]
        手順の並びが空でない項目だけの写し (無ければ None)

    Raises
    ----------
    ValueError
        辞書でない, 書き出しに無い項目の名前, 並びでない手順, 辞書でない
        手順, op が文字でない手順の場合
    """
    # 無い場合と空の辞書は無いもの
    if item_steps is None:
        steps: Optional[dict] = None
        return steps
    if not isinstance(item_steps, dict):
        raise ValueError("中身の手順 (item_steps) は項目の名前から手順の"
                         "並びの辞書にしてください")
    names = STEP_ENTRIES[function]
    unknown = sorted(str(key) for key in item_steps if key not in names)
    if unknown:
        raise ValueError(f"{function} の中身の手順を渡せる項目は "
                         f"{', '.join(names)} です ({', '.join(unknown)} "
                         "には渡せません)")

    # 項目ごとの並びと, 手順ごとの op
    steps = {}
    for name, value in item_steps.items():
        if not isinstance(value, list):
            raise ValueError(f"中身の手順の {name} は手順の並びにしてくだ"
                             "さい")
        for step in value:
            if not isinstance(step, dict) or not isinstance(
                step.get("op"), str
            ):
                raise ValueError(f"中身の手順の {name} の手順は, 文字の op "
                                 "を持つ辞書にしてください")
        if value:
            steps[name] = copy.deepcopy(value)
    steps = steps or None

    return steps


def is_ratio(value: Any) -> bool:
    """
    分割の割合として書ける値かを返す関数

    Parameters
    ----------
    value: Any
        割合

    Returns
    ----------
    ratio: bool
        0 以上 1 未満の数か (真偽値は除く. 大きな整数も float にせずに
        比べる)
    """
    # nan と無限大は範囲の比べで外れる
    ratio = (isinstance(value, (int, float)) and not isinstance(value, bool)
             and 0 <= value < 1)

    return ratio


def export_conflicts(problem: dict, settings: dict) -> list[dict]:
    """
    書き出しの設定と問題の設定の食い違いを返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (ファイルの組の受け持ち)
    settings: dict
        書き出しの設定 (決まった形)

    Returns
    ----------
    conflicts: list[dict]
        確かめの形の食い違い
    """
    # 書き出しと正解の種類
    kind = active_value(problem, f"{PART}.target.kind")
    function = EXPORTS[kind][0]
    conflicts = []
    if settings["function"] != function:
        conflicts.append(finding(
            "error", f"{PART}.target.kind",
            f"書き出しが {settings['function']} ですが, 正解の種類 {kind} "
            f"は {function} で書き出します"
        ))

    # 分割のしかたと割合の形 (書き出しの時点で初めて誤りにならないように).
    # 割合は ML-EDA と同じく, 書いていなければ既定 (検証用 0.2, テスト用 0)
    # で見る. 分けない (none) なら ML-EDA は割合を使わないので見ない
    split = settings["split"]
    method = split.get("method")
    if method is not None and method not in SPLIT_METHODS:
        conflicts.append(finding(
            "error", "data.export",
            f"分割のしかた {method} は {', '.join(SPLIT_METHODS)} のどれかに"
            "してください"
        ))
    ratios = {key: split.get(key, default)
              for key, default in SPLIT_DEFAULTS.items()}
    wrong = [key for key, value in ratios.items()
             if method != "none" and not is_ratio(value)]
    conflicts += [finding(
        "error", "data.export",
        f"分割の {key} の割合は 0 以上 1 未満の数にしてください (今は "
        f"{ratios[key]!r})"
    ) for key in wrong]
    if method != "none" and not wrong and sum(ratios.values()) >= 1:
        conflicts.append(finding(
            "error", "data.export",
            "検証用とテスト用の割合の和は 1 未満にしてください (今は "
            f"{ratios['valid']} と {ratios['test']}. 書いていない割合は "
            "ML-EDA の既定の検証用 0.2, テスト用 0. 学習用が残らないため)"
        ))

    # 検証用の分割と, グループごとの分割のグループ (ML-EDA の既定の割合は
    # 検証用 2 割)
    method = export_split(settings)
    if method == "none" or split.get("valid", 1) == 0:
        conflicts.append(finding(
            "warning", "data.export",
            "検証用の分割がありません. 骨組みの学習は検証用の分割が無いと"
            "止まります"
        ))
    if method == "group" and not settings["groups"]:
        conflicts.append(finding(
            "error", "data.export",
            "グループごとに分けるのに, グループがありません"
        ))

    return conflicts


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
        method not in TIME_SPLITS
    ):
        conflicts.append(finding(
            "warning", f"{PART}.split_time_column",
            f"時刻の順で分けて確かめると決めたのに, 分割が {method} です"
        ))
    task = active_value(problem, f"{PART}.task")
    if task == "forecasting" and method and method not in TIME_SPLITS:
        conflicts.append(finding(
            "warning", f"{PART}.task",
            f"時系列の予測なのに, 分割が {method} です (先の期間で確かめる"
            "には時刻の順か, 相手ごとと時刻の順)"
        ))

    # 相手ごとと時刻の順は, 相手が現れた順を守るが, 長く続く相手の後半の
    # 行は後から現れた相手の行より後の時刻になりうる (ML-EDA の決まり)
    timed = task == "forecasting" or active_value(
        problem, f"{PART}.split_time_column"
    )
    if timed and method == "group_time":
        conflicts.append(finding(
            "info", f"{PART}.task" if task == "forecasting"
            else f"{PART}.split_time_column",
            "分割が相手ごとと時刻の順 (group_time) です. 後から現れた相手を"
            "検証用とテスト用にしますが, 相手の中の行の時刻は分割をまたいで"
            "重なりえます (長く続く相手の後半の行が, 検証用の相手の行より"
            "後の時刻になることがあります)"
        ))

    # グループでまとめる分割の列と, 層化に要る分類 (ML-EDA が書き出しで
    # 誤りにするものを, 書き戻しの時点で)
    if method in GROUP_SPLITS and not written:
        conflicts.append(finding(
            "error", f"{PART}.group_column",
            f"分割が {method} なのに, 相手をまとめる列 (split.column) が"
            "ありません"
        ))
    if method == "stratified_group" and task in ("regression",
                                                 "forecasting"):
        conflicts.append(finding(
            "error", f"{PART}.task",
            "相手ごとの層別 (stratified_group) は分類だけです. 回帰と時系列"
            "の予測では, 相手ごと (group) か相手ごとと時刻の順 (group_time)"
            " にしてください"
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
