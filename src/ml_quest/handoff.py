from pathlib import PurePath
from typing import Any, Optional

from ml_quest.check import (
    check, env_id_of, is_column_part, is_generate_part, is_items_part,
    is_pretrained_part, sha256_of, split_errors
)
from ml_quest.data import EXPORT_FUNCTIONS
from ml_quest.problem import (
    DECISION_NAMES, DECISION_OPTIONS, FIELD_NAMES, OPTION_NAMES,
    active_value, data_location, get_value, upgrade_problem
)
from ml_quest.questions import (
    CONSUMERS, FALLBACKS, METRICS, PART, ROLES, TASKS
)
from ml_quest.requirements import required_features

# 機械学習を使わないと答えた問題の設定を渡せない理由の文
NO_ML = "機械学習を使わない問題の設定は, 後ろの段階に渡すものがありません"

# 方策の守りの項目 (ML-Found の make_guards と safety_key に渡す答え)
POLICY_KEYS = ["action_limits", "stop", "fallback", "fallback_action",
               "safety_key", "approver"]

# 安全の違反の信号の名前の既定 (ML-Found の safety_key の既定)
DEFAULT_SAFETY_KEY = "cost"

# 選んで書く意思決定の項目 (値を名前にして文にする)
OPTION_FIELDS = {
    field for options in DECISION_OPTIONS.values() for field in options
}


def decision_of(problem: dict) -> Optional[dict]:
    """
    問題の設定から, ML-Inference の意思決定の形を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    decision: Optional[dict]
        意思決定 (選ぶと振り分けには当たりのクラスを足す. 意思決定しない
        なら None)
    """
    # 損得と制約はそのまま, 当たりのクラスは機械学習の受け持ちから
    raw = active_value(problem, "action.decision")
    if not raw or raw.get("type") == "none":
        return None
    decision = dict(raw)
    if decision["type"] in ("select", "triage"):
        decision["positive"] = active_value(problem, f"{PART}.positive")

    # 今のやり方の列 (量は今ある量の無い場合だけ. 問題の設定の列は置いた
    # 量, ML-Inference は今ある量に足した量)
    current = active_value(problem, "action.current") or {}
    if current.get("type") == "column" and (
        decision["type"] != "quantity" or not decision.get("stock")
    ):
        decision["current"] = current["column"]

    return decision


def conclusion_of(problem: dict) -> dict:
    """
    問題の設定から, ML-Inference の結論の決め方の既定値を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    conclusion: dict
        結論の決め方 (系列を分ける列があれば, 最初の列でまとめる)
    """
    # 系列ごとのまとめ (最初の列. 店と品なら店ごと)
    series = active_value(problem, f"{PART}.series_by") or []
    conclusion = {"group_by": series[0]} if series else {}

    return conclusion


def handoff(problem: dict) -> dict:
    """
    後ろの段階ごとに, 問題の設定から決まる値をまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    stages: dict
        data (データの分析), model (モデルの作成), training (学習・評価),
        inference (推論. 既製のモデルをそのまま使う予測は ready に一式
        づくりに渡す値) ごとの値と, part (受け持ちの答え), requires (要る
        機能の名前), policy (方策の守り. 方策でなければ None), project
        (書き出したプロジェクトの path. データセットとモデルの定義の場所は
        data の project_dataset と model の project_model_spec. 記録が無け
        れば None)

    Raises
    ----------
    ValueError
        機械学習を使わない場合と, 確かめに error がある場合
    """
    # 渡せる問題の設定か (版 1 は先に版 2 へ移行)
    problem = upgrade_problem(problem)
    if get_value(problem, "ml.use") is not True:
        raise ValueError(no_ml_message(problem))
    errors = [item for item in check(problem) if item["level"] == "error"]
    if errors:
        raise ValueError(errors_text(errors))
    stages = build_stages(problem)

    return stages


def no_ml_message(problem: dict) -> str:
    """
    機械学習を使わないか, 使うかに答えていない問題の設定を渡せない理由の文
    を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. ml.use が true でないもの)

    Returns
    ----------
    message: str
        誤りの文 (答えていなければ, 答えるよう知らせる)
    """
    # 答えていないか, 使わないと答えたか
    message = (NO_ML if get_value(problem, "ml.use") is False
               else "「機械学習を使いますか」に答えてください")

    return message


def export_stages(
    problem: dict, found: Optional[list[dict]] = None
) -> tuple[dict, list[dict]]:
    """
    骨組みを組めない error が無ければ, 確かめを除いた後ろの段階ごとの値と,
    残っている組める error を返す関数

    骨組みの書き出しと, データの分析の段階のレシピの下書きが使う (推論の
    段階に渡す値は handoff で, 全ての error で止まる)

    Parameters
    ----------
    problem: dict
        問題の設定
    found: Optional[list[dict]] = None
        同じ問題の設定の check の結果 (省くと求める)

    Returns
    ----------
    stages: dict
        handoff と同じ形の値
    exportable: list[dict]
        骨組みを組める確かめの error (無ければ空)

    Raises
    ----------
    ValueError
        機械学習を使わない場合と, 骨組みを組めない error がある場合
    """
    # 組めない error での停止と, 組める error の返却
    problem = upgrade_problem(problem)
    if get_value(problem, "ml.use") is not True:
        raise ValueError(no_ml_message(problem))
    blocking, exportable = split_errors(problem, found)
    if blocking:
        raise ValueError(errors_text(blocking))
    stages = build_stages(problem)

    return stages, exportable


def errors_text(errors: list[dict]) -> str:
    """
    問題の設定の確かめの error で渡せない理由の文を作る関数

    Parameters
    ----------
    errors: list[dict]
        確かめの error の並び (空でない)

    Returns
    ----------
    message: str
        誤りの文 (初めの 3 つ)
    """
    # 初めの 3 つの文
    message = ("問題の設定に直すところがあります: "
               + " / ".join(item["message"] for item in errors[:3]))

    return message


def build_stages(problem: dict) -> dict:
    """
    確かめずに, 後ろの段階ごとの値を作る関数

    handoff と, 「それでも書き出す」と指定した骨組みの書き出しが使う

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 機械学習を使い, 骨組みを組めない error が無い
        もの)

    Returns
    ----------
    stages: dict
        handoff と同じ形の値
    """
    # 受け持ちの答えと, 最良を選ぶ指標 (その他は名前と向き)
    value = active_value
    part = part_of(problem)
    training = training_of(problem)
    training.update(role=part["role"], target_kind=part["target_kind"])
    project = project_of(problem)
    if not is_column_part(problem):
        stages = {
            "part": part,
            "data": {**column_data(problem, None),
                     **items_data(problem),
                     **generate_data(problem),
                     "project_dataset": project["dataset"]},
            "model": {"task": None, "outputs": None, "role": part["role"],
                      "target_kind": part["target_kind"],
                      "project_model_spec": project["model_spec"]},
            "training": training,
            "inference": {"decision": None, "conclusion": {},
                          "horizon": None, "ready": ready_of(problem)},
            "requires": required_features(problem),
            "policy": policy_of(problem),
            "project": {"path": project["path"]},
        }
        return stages

    # 列の予測: 版 1 と同じ値に, 受け持ちと正解の種類を足す
    task = value(problem, f"{PART}.task")
    stages = {
        "part": part,
        "data": {**column_data(problem, task),
                 "project_dataset": project["dataset"]},
        "model": {"task": task,
                  "outputs": None if task == "classification" else 1,
                  "role": part["role"], "target_kind": part["target_kind"],
                  "project_model_spec": project["model_spec"]},
        "training": training,
        "inference": {"decision": decision_of(problem),
                      "conclusion": conclusion_of(problem),
                      "horizon": value(problem, f"{PART}.horizon"),
                      "ready": None},
        "requires": required_features(problem),
        "policy": policy_of(problem),
        "project": {"path": project["path"]},
    }

    return stages


def project_of(problem: dict) -> dict:
    """
    書き出したプロジェクトの記録 (record_project) を読む関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    project: dict
        path, dataset, model_spec (記録が無いか, 文字の絶対パスでない値は
        None)
    """
    # 手で崩した値は使わない (絶対パスの文字だけ)
    record = problem.get("project")
    record = record if isinstance(record, dict) else {}
    project = {
        key: record.get(key) if isinstance(record.get(key), str)
        and PurePath(record[key]).is_absolute() else None
        for key in ("path", "dataset", "model_spec")
    }

    return project


def problem_origin(problem: dict) -> Optional[str]:
    """
    写した問題の設定から, 元の問題の設定のファイルの場所を読む関数

    make_project が骨組みのプロジェクトの problem.yaml に書き, 学習の
    スクリプトが実験に写すときもそのまま残る

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 か版 2)

    Returns
    ----------
    origin: Optional[str]
        元の problem.yaml の絶対パス (写しでないか, 文字の絶対パスでない
        値は None. ファイルがあるかは見ない)
    """
    # 手で崩した値は使わない (絶対パスの文字だけ)
    value = problem.get("origin") if isinstance(problem, dict) else None
    origin = (value if isinstance(value, str) and value.strip()
              and PurePath(value).is_absolute() else None)

    return origin


def part_of(problem: dict) -> dict:
    """
    受け持ちの型に依らない答えをまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    part: dict
        role, unit, output, target_kind, target_source, target_process
        (今の処理の名前と設定. 出どころが今の処理でなければ None), method
    """
    # 要らなくなった問いの古い答えは除外
    value = active_value
    name = value(problem, f"{PART}.target.process.name")
    part = {
        "role": value(problem, f"{PART}.role"),
        "unit": value(problem, f"{PART}.unit"),
        "output": value(problem, f"{PART}.output"),
        "target_kind": value(problem, f"{PART}.target.kind"),
        "target_source": value(problem, f"{PART}.target.source"),
        "target_process": {
            "name": name,
            "settings": value(problem, f"{PART}.target.process.settings"),
        } if name else None,
        "method": value(problem, f"{PART}.method"),
    }

    return part


def ready_of(problem: dict) -> Optional[dict]:
    """
    既製のモデルの一式づくり (ML-Inference の make_ready_bundle) に渡す値
    をまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    ready: Optional[dict]
        library, model, weights (前後の空白を除いた名前), sha256 (決めて
        いなければ None) (既製のモデルをそのまま使う予測でなければ None)
    """
    # 予測するで既製のモデルをそのまま使うときだけ (要らなくなった問いの
    # 古い答えは除外)
    if not is_pretrained_part(problem):
        return None
    ready: dict[str, Any] = {}
    for key in ("library", "model", "weights"):
        found = active_value(problem, f"{PART}.ready.{key}")
        ready[key] = found.strip() if isinstance(found, str) else found
    ready["sha256"] = sha256_of(problem)

    return ready


def policy_of(problem: dict) -> Optional[dict]:
    """
    続けて行動を選ぶ方策の守りの答えをまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    policy: Optional[dict]
        action_limits, stop, fallback, fallback_action, safety_key,
        approver, env_id (方策でなければ None. 信号の名前は答えていなければ
        cost, 環境の登録名は答えていなければ None)
    """
    # 方策と答えたときだけ (要らなくなった問いの古い答えは除外)
    if active_value(problem, f"{PART}.autonomous") is not True:
        return None
    policy = {key: active_value(problem, f"{PART}.policy.{key}")
              for key in POLICY_KEYS}

    # 環境の登録名 (答えていなければ None)
    policy["env_id"] = env_id_of(problem)

    # 信号の名前の既定 (None を渡すと ML-Found は違反を数えないため)
    policy["safety_key"] = policy["safety_key"] or DEFAULT_SAFETY_KEY

    return policy


def training_of(problem: dict) -> dict:
    """
    学習・評価の段階に渡す, 最良を選ぶ指標と監視する指標を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    training: dict
        metric (その他は書いた名前), monitor (valid/<指標>), mode (min か
        max). 既製のモデルをそのまま使う (学習しない) 場合は全て None
    """
    # 学習しない場合は指標を問わない
    metric = active_value(problem, f"{PART}.evaluation.metric")
    if active_value(problem, f"{PART}.method") == "pretrained":
        training: dict[str, Any] = {"metric": None, "monitor": None,
                                    "mode": None}
        return training

    # 学習の監視は, バッチの平均で書ける指標 (RMSE は二乗誤差で選ぶ).
    # その他は書いた名前と向き
    better = next((item["better"] for item in METRICS
                   if item["value"] == metric), "lower")
    monitor = "valid/mse" if metric == "rmse" else f"valid/{metric}"
    if metric == "other":
        metric = active_value(problem, f"{PART}.evaluation.name")
        better = active_value(problem, f"{PART}.evaluation.better")
        monitor = f"valid/{metric}"
    training = {"metric": metric, "monitor": monitor,
                "mode": "max" if better == "higher" else "min"}

    return training


def column_data(problem: dict, task: Optional[str]) -> dict:
    """
    データの分析の段階に渡す値を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    task: Optional[str]
        列の予測のタスク (列の予測でなければ None)

    Returns
    ----------
    data: dict
        目的の列, タスク, 分割, 使う列と除く列 (列の予測でなければ同じ
        名前の空の値) と, データの場所と読み方
    """
    # 要らなくなった問いの古い答えは除外
    value = active_value
    data = {
        "target": value(problem, f"{PART}.target.column"),
        "task": task,
        "time_column": value(problem, f"{PART}.time_column")
        or value(problem, f"{PART}.split_time_column"),
        "series_by": value(problem, f"{PART}.series_by") or [],
        "split": None if task is None
        else "time" if task == "forecasting"
        or value(problem, f"{PART}.split_time_column")
        else "group" if value(problem, f"{PART}.group_column")
        else "stratified" if task == "classification" else "random",
        "group_column": value(problem, f"{PART}.group_column"),
        "min_lag": value(problem, f"{PART}.available_lag"),
        "known": value(problem, "information.known") or [],
        "source": {"source": data_location(problem),
                   "loader": value(problem, "data.loader"),
                   "options": value(problem, "data.options")}
        if value(problem, "data.source") else None,
        "drop": drop_columns(problem),
    }

    return data


def items_data(problem: dict) -> dict:
    """
    ファイルの組の受け持ちの, データの分析の段階に渡す分割とグループを作る
    関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        split と groups (書き戻した書き出しの設定から. ファイルの組の受け
        持ちでなければ空)
    """
    # 書き戻した設定が無ければ無作為 (辞書でない値は無いものとして)
    data: dict = {}
    if not is_items_part(problem):
        return data
    # 書き出しが正解の種類に合う設定だけ (recipe_from と同じ扱い)
    saved = get_value(problem, "data.export")
    function = EXPORT_FUNCTIONS.get(
        active_value(problem, f"{PART}.target.kind")
    )
    saved = (saved if isinstance(saved, dict)
             and saved.get("function") == function else {})
    groups = saved.get("groups")
    data = {"split": export_split(saved),
            "groups": dict(groups) if isinstance(groups, dict) and groups
            else None}

    return data


def generate_data(problem: dict) -> dict:
    """
    画像の生成の受け持ちの, データの分析の段階に渡す分割を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        split (書き戻したレシピの split.method. 無ければ random. 画像の
        生成でなければ空)
    """
    # 目的の列が無いので層別にも時刻の順にもしない
    data: dict = {}
    if not is_generate_part(problem):
        return data
    saved = get_value(problem, "data.recipe")
    split = saved.get("split") if isinstance(saved, dict) else None
    method = split.get("method") if isinstance(split, dict) else None
    data = {"split": method if isinstance(method, str) and method
            else "random"}

    return data


def export_split(export: dict) -> str:
    """
    書き出しの設定の分割のしかたを返す関数

    Parameters
    ----------
    export: dict
        書き出しの設定 (split, groups)

    Returns
    ----------
    method: str
        split の method. 無ければ ML-EDA の既定 (グループがあれば group,
        無ければ random)
    """
    # 選んでいなければ ML-EDA と同じ既定
    split = export.get("split")
    method = split.get("method") if isinstance(split, dict) else None
    if not isinstance(method, str) or not method:
        method = "group" if export.get("groups") else "random"

    return method


def drop_columns(problem: dict) -> list[str]:
    """
    学習に使わずに落とす列を返す関数

    予測の時点で分からない列と使ってはいけない列に加えて, 今のやり方の
    列 (今のやり方で行動したか, 置いた量) も落とす. 今のやり方は決める
    ことの結果なので, 予測の時点では分からず, 学習に入れると答えが漏れる
    ため (比べる相手としてだけ使う). 利用者が予測の時点で分かる列に
    入れた場合は落とさない (確かめが注意を出す)

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    drop: list[str]
        落とす列 (重ならない. 目的の列は含めない)
    """
    # 分からない列, 使ってはいけない列, 今のやり方の列の順
    value = active_value
    names = ((value(problem, "information.unknown") or [])
             + (value(problem, "information.forbidden") or []))
    current = value(problem, "action.current") or {}
    known = value(problem, "information.known") or []
    if (isinstance(current, dict) and current.get("type") == "column"
            and current.get("column") not in known):
        names = names + [current.get("column")]
    target = value(problem, f"{PART}.target.column")
    drop = []
    for name in names:
        if isinstance(name, str) and name != target and name not in drop:
            drop.append(name)

    return drop


def describe(problem: dict) -> list[str]:
    """
    問題の設定を, 課題から行動までの筋道の文にする関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    sentences: list[str]
        要約の文 (答えた項目だけ)
    """
    # 課題と指標 (要らなくなった問いの古い答えは使わない. 版 1 は先に版 2 へ移行)
    problem = upgrade_problem(problem)
    value = active_value
    sentences = []
    if value(problem, "issue.pain"):
        owner = value(problem, "issue.owner") or "誰か"
        sentences.append(f"{owner}の課題: {value(problem, 'issue.pain')}")
    if value(problem, "issue.current"):
        sentences.append(f"今のやり方: {value(problem, 'issue.current')}")
    kpi = value(problem, "issue.kpi") or {}
    if kpi.get("name"):
        sentences.append(kpi_text(kpi))
    guard = value(problem, "issue.guard") or {}
    if isinstance(guard, dict) and guard.get("name"):
        sentences.append(guard_text(guard))

    # 決めること (予測より先) と機械学習の受け持ち
    if value(problem, "action.what"):
        sentences.append(
            f"{value(problem, 'action.who') or '誰か'}が"
            f"{value(problem, 'action.when') or 'いつか'}に"
            f"{value(problem, 'action.what')}を決める"
        )
    uses = value(problem, "ml.use")
    if uses is False:
        reason = value(problem, "ml.reason")
        sentences.append("機械学習は使わない" + (f": {reason}" if reason
                                                 else ""))
        return sentences
    what = value(problem, f"{PART}.what")
    predicts = value(problem, f"{PART}.role") in (None, "predict")
    if what:
        roles = {item["value"]: item["label"].split(" ")[0] for item in ROLES}
        tasks = {item["value"]: item["label"].split(" ")[0] for item in TASKS}
        role = roles.get(str(value(problem, f"{PART}.role")), "")
        task = tasks.get(str(value(problem, f"{PART}.task")), "")
        target = value(problem, f"{PART}.target.column")
        details = [part for part in (
            task, f"目的の列 {target}" if target else ""
        ) if part]
        head = ("機械学習で予測すること" if predicts
                else f"機械学習の受け持ち ({role})" if role
                else "機械学習の受け持ち")
        sentences.append(
            f"{head}: {what}"
            + (f" ({'. '.join(details)})" if details else "")
        )
    consumer = value(problem, f"{PART}.consumer")
    names = {item["value"]: item["label"] for item in CONSUMERS}
    if isinstance(consumer, str) and consumer in names:
        downstream = value(problem, f"{PART}.downstream")
        sentences.append(
            f"{'予測' if predicts else '出力'}の使い道: {names[consumer]}"
            + (f". 後段が求めるもの: {downstream}" if downstream else "")
        )
    decision = value(problem, "action.decision")
    if isinstance(decision, dict) and decision.get("type") in DECISION_NAMES:
        sentences.append(decision_text(decision))
    constraints = value(problem, "ml.constraints")
    if isinstance(constraints, list) and constraints:
        sentences.append("守る制約: " + ", ".join(map(str, constraints)))
    if value(problem, f"{PART}.autonomous") is True and isinstance(
        value(problem, f"{PART}.policy"), dict
    ):
        sentences.append(policy_text(problem))
    env_id = env_id_of(problem)
    if env_id:
        sentences.append(f"方策を学ぶ環境: Gymnasium の {env_id}")
    ready = ready_of(problem)
    if ready and ready["model"]:
        sentences.append(ready_text(ready))
    features = required_features(problem)
    if features:
        sentences.append("要る機能: " + ", ".join(features))
    if value(problem, "verification.success"):
        sentences.append(
            f"成功の基準: {value(problem, 'verification.success')}"
        )

    return sentences


def ready_text(ready: dict) -> str:
    """
    既製のモデルの出どころ, 名前, 重みを文にする関数

    Parameters
    ----------
    ready: dict
        一式づくりに渡す値 (ready_of の結果. モデルの名前があるもの)

    Returns
    ----------
    text: str
        要約の文 (hash は頭の 12 文字)
    """
    # 出どころ, モデル, 重み, hash の答えた部分
    where = f"{ready['library']} の " if ready["library"] else ""
    weights = f" (重み {ready['weights']})" if ready["weights"] else ""
    sha256 = (f". 重みの hash {ready['sha256'][:12]}…" if ready["sha256"]
              else "")
    text = (f"学習せずに使う既製のモデル: {where}{ready['model']}{weights}"
            f"{sha256}")

    return text


def policy_text(problem: dict) -> str:
    """
    方策の守りを文にする関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    text: str
        「方策の守り: 行動の上限 x. 止める条件 y. 外れたら z. 決める人 w」
        の形 (答えた項目だけ)
    """
    # 答えた項目だけの並び (外れたときの動きは名前)
    value = active_value
    names = {item["value"]: item["label"] for item in FALLBACKS}
    fallback = value(problem, f"{PART}.policy.fallback")
    action = value(problem, f"{PART}.policy.fallback_action")
    items = [
        ("行動の上限", value(problem, f"{PART}.policy.action_limits")),
        ("止める条件", value(problem, f"{PART}.policy.stop")),
        ("外れたら", names.get(str(fallback), fallback)
         if fallback else None),
        ("その行動", action),
        ("違反の信号", value(problem, f"{PART}.policy.safety_key")),
        ("決める人", value(problem, f"{PART}.policy.approver")),
    ]
    parts = [f"{name} {item}" for name, item in items
             if item is not None and item != ""]
    text = "方策の守り: " + ". ".join(parts)

    return text


def guard_text(guard: dict) -> str:
    """
    守る指標を文にする関数

    Parameters
    ----------
    guard: dict
        守る指標 (name, unit, better, line, measure)

    Returns
    ----------
    text: str
        「守る指標: 名前 (単位. x を超えない. 測り方)」の形
    """
    # 分かる値だけの並び (線の向きは指標の向きから)
    parts = []
    if guard.get("unit"):
        parts.append(str(guard["unit"]))
    line = guard.get("line")
    if line is not None and guard.get("better") in ("lower", "higher"):
        parts.append(f"{number_text(line)} を"
                     + ("超えない" if guard["better"] == "lower"
                        else "下回らない"))
    elif line is not None:
        parts.append(f"守る線 {number_text(line)}")
    if guard.get("measure"):
        parts.append(f"測り方: {guard['measure']}")
    text = f"守る指標: {str(guard['name']).strip()}" + (
        f" ({'. '.join(parts)})" if parts else ""
    )

    return text


def kpi_text(kpi: dict) -> str:
    """
    指標を文にする関数

    Parameters
    ----------
    kpi: dict
        指標 (name, unit, current, target, better)

    Returns
    ----------
    text: str
        「指標: 名前 (単位. 今 x → 目指す y. 小さいほど良い)」の形
    """
    # 分かる値だけの並び
    parts = []
    if kpi.get("unit"):
        parts.append(str(kpi["unit"]))
    if kpi.get("current") is not None:
        parts.append(f"今 {number_text(kpi['current'])}")
    if kpi.get("target") is not None:
        parts.append(f"目指す {number_text(kpi['target'])}")
    if kpi.get("better") in ("lower", "higher"):
        parts.append("小さいほど良い" if kpi["better"] == "lower"
                     else "大きいほど良い")
    text = f"解けたと言える指標: {kpi['name']}" + (
        f" ({'. '.join(parts)})" if parts else ""
    )

    return text


def number_text(value: Any) -> str:
    """
    数を桁区切りで, 小数は 4 桁の有効数字で文にする関数

    Parameters
    ----------
    value: Any
        値 (数でなければそのまま文字列に)

    Returns
    ----------
    text: str
        文
    """
    # 整数は桁区切り, 小数は丸め. 手で書いた真偽値と空は, 数でないと分かる文
    if isinstance(value, bool) or value is None:
        text = f"{yaml_name(value)} (数でない値)"
    elif not isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, int):
        text = f"{value:,}"
    else:
        text = f"{value:,.4g}"

    return text


def yaml_name(value: Any) -> str:
    """
    真偽値と空を, 問題の設定のファイルでの書き方にする関数

    Parameters
    ----------
    value: Any
        True, False か None

    Returns
    ----------
    name: str
        true, false か null
    """
    # YAML の名前 (Python の True と None を文に出さない)
    name = {True: "true", False: "false"}.get(value, "null")

    return name


def decision_text(decision: dict) -> str:
    """
    意思決定の型と損得を文にする関数

    Parameters
    ----------
    decision: dict
        意思決定 (ML-Inference の形)

    Returns
    ----------
    text: str
        型と, 書いた損得と制約
    """
    # 型の名前と, 項目の名前と値 (数は桁区切り. 選んだ値は名前. 確率の差
    # で書いた効果は防げる割合と呼ばない)
    names = dict(FIELD_NAMES)
    if decision.get("effect_kind") == "uplift":
        names["effect"] = "確率の差"
    items: list[Any] = [
        f"{names.get(key, key)} " + number_text(
            OPTION_NAMES.get(value, value)
            if key in OPTION_FIELDS and isinstance(value, str) else value
        )
        for key, value in decision.items() if key != "type"
    ]
    text = f"意思決定: {DECISION_NAMES[decision['type']]}" + (
        f" ({', '.join(items)})" if items else ""
    )

    return text
