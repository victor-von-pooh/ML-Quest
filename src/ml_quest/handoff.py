from decimal import Decimal
import math
from pathlib import PurePath
from typing import Any, Optional

from ml_quest.check import (
    check, env_id_of, is_anomaly_part, is_column_part, is_demand_part,
    is_finite_number, is_generate_part, is_group_part, is_items_part,
    is_pretrained_part, is_price_part, is_rank_part, is_ready_group_part,
    is_search_part, is_uplift_part, sha256_of, split_errors
)
from ml_quest.data import EXPORT_FUNCTIONS
from ml_quest.problem import (
    DECISION_NAMES, DECISION_OPTIONS, FIELD_NAMES, OPTION_NAMES,
    active_value, data_location, get_value, upgrade_problem
)
from ml_quest.questions import (
    CONSUMERS, DEMAND_HORIZON, DEMAND_LEVELS, FALLBACKS, GROUP_STABILITIES,
    GUARD_COUNTS, METRICS, PART, ROLES, TASKS
)
from ml_quest.requirements import required_features

# 機械学習を使わないと答えた問題の設定を渡せない理由の文
NO_ML = "機械学習を使わない問題の設定は, 後ろの段階に渡すものがありません"

# 方策の守りの項目 (ML-Found の make_guards と safety_key に渡す答え)
POLICY_KEYS = ["action_limits", "stop", "fallback", "fallback_action",
               "safety_key", "approver"]

# 安全の違反の信号の名前の既定 (ML-Found の safety_key の既定)
DEFAULT_SAFETY_KEY = "cost"

# 一から学習する受け持ちと正解の種類ごとの, モデルの作成の雛形
# (ML-ModelControl の同梱の例の名前. 並べる, まとめる, 効果を見積もるは
# 正解の種類を問わないので None. 異常は度合いの決め方が自己符号化器の
# ときだけ. 効果を見積もるは方策でない効果の見積もりのときだけで, 記録で
# 変えたものが値段の値段を変える判断は正解の種類の代わりに price)
MODEL_EXAMPLES = {
    ("predict", "mask"): "unet_segmentation",
    ("predict", "anomaly"): "tabular_autoencoder",
    ("predict", "point_cloud"): "point_cloud_predictor",
    ("generate", "table"): "diffusion_denoiser",
    ("rank", None): "two_tower_retrieval",
    ("group", None): "contrastive_encoder",
    ("effect", None): "tabular_uplift",
    ("effect", "price"): "tabular_price",
    ("predict", "demand"): "tabular_demand",
}

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


def model_example(problem: dict) -> Optional[str]:
    """
    受け持ちに合う, モデルの作成の雛形 (ML-ModelControl の同梱の例) の名前
    を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 か版 2)

    Returns
    ----------
    example: Optional[str]
        同梱の例の名前 (MODEL_EXAMPLES. 一から学習しないか, 合う例が無い
        受け持ちは None)
    """
    # 同梱の例は学習前の定義なので, 一から学習するときだけ (要らなくなった
    # 問いの古い答えは除外)
    problem = upgrade_problem(problem)
    if active_value(problem, f"{PART}.method") != "scratch":
        return None
    key = (active_value(problem, f"{PART}.role"),
           "price" if is_price_part(problem)
           else active_value(problem, f"{PART}.target.kind"))
    example = MODEL_EXAMPLES.get(key)

    # 異常の検知の距離に基づく度合い (モデル無し) と, 方策の頭 (入力と出力
    # が 1 つずつの約束が同梱の例と別の形) の雛形無し
    if is_anomaly_part(problem) and (
        active_value(problem, f"{PART}.anomaly.score") != "autoencoder"
    ):
        example = None
    if key[0] == "effect" and not is_uplift_part(problem) and (
        not is_price_part(problem)
    ):
        example = None

    return example


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


def guard_of(problem: dict) -> Optional[dict]:
    """
    問題の設定から, 推論の段階のたどり直しで数える守る指標を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    guard: Optional[dict]
        name, unit, better (lower か higher), line (守る線. 有限の数),
        measure (測り方), counted (たどり直しで数える量の名前. 指さない
        なら None. どれも答えていなければ None. 名前を答えていなければ
        全体が None)
    """
    # 名前のある守る指標だけ (空の文, 崩れた向き, 有限の数でない線は None)
    name = active_value(problem, "issue.guard.name")
    if not isinstance(name, str) or not name.strip():
        return None
    unit, better, line, measure = (
        active_value(problem, f"issue.guard.{key}")
        for key in ("unit", "better", "line", "measure")
    )
    guard = {
        "name": name.strip(),
        "unit": text_or_none(unit),
        "better": better if better in ("lower", "higher") else None,
        "line": line if isinstance(line, (int, float))
        and not isinstance(line, bool) and math.isfinite(line) else None,
        "measure": text_or_none(measure),
        "counted": counted_of(problem),
    }

    return guard


def text_or_none(text: Any) -> Optional[str]:
    """
    答えの文の前後の空白を除き, 空か文でなければ None を返す関数

    Parameters
    ----------
    text: Any
        答え

    Returns
    ----------
    text: Optional[str]
        前後の空白を除いた文 (空か文でなければ None)
    """
    # 空白だけの文も答えていないとみなす
    text = text.strip() if isinstance(text, str) and text.strip() else None

    return text


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
        data (データの分析), model (モデルの作成. 雛形の名前 example も),
        training (学習・評価), inference (推論. 既製のモデルをそのまま使う
        予測は ready に一式づくりに渡す値, 守る指標は guard) ごとの値と,
        part (受け持ちの答え), requires (要る機能の名前), policy (方策の
        守り. 方策でなければ None), rank (並べるの反応の表の列と件数.
        並べるで学習しなければ None), group (群を作る列, 群の数, 安定性の
        数え方. まとめるで学習しなければ None), anomaly (異常の検知の度合いの
        決め方, 列, 人が確かめる件数の上限. 異常の検知でなければ None),
        uplift (効果の見積もりの処置, 結果, 列と上位の割合. 効果の見積もり
        でなければ None), price (値段を変える判断の値段, 売れた数, 原価,
        商品の id, 行の id の列と予測に使う列. 値段を変える判断でなければ
        None), demand (時間をまたぐ量の判断の商品の id, 日付, 売れた数,
        在庫の数, 行の id の列, 商品の特徴の列, 先の期間の日数と学ぶ水準.
        時間をまたぐ量の判断でなければ None), project (書き出したプロジェクトの
        path. データセットとモデルの定義の場所は data の project_dataset
        と model の project_model_spec. 記録が無ければ None)

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
        # 異常の検知の自己符号化器は入力と同じ幅を出す (ML-EDA の
        # model_inputs の outputs と同じく学習に使う列の数). 効果の見積もり
        # は出力の頭ごとに 1, 値段を変える判断は売れた数の 1
        anomaly = anomaly_of(problem)
        uplift = uplift_of(problem)
        price = price_of(problem)
        demand = demand_of(problem)
        outputs = (len(anomaly["columns"]) if anomaly
                   else 1 if uplift or price
                   else len(demand["levels"]) if demand else None)
        stages = {
            "part": part,
            "data": {**column_data(problem, None),
                     **items_data(problem),
                     **generate_data(problem),
                     **rank_data(problem),
                     **group_data(problem),
                     **anomaly_data(problem),
                     **uplift_data(problem),
                     **price_data(problem),
                     **demand_data(problem),
                     "project_dataset": project["dataset"]},
            "model": {"task": None, "outputs": outputs,
                      "role": part["role"],
                      "target_kind": part["target_kind"],
                      "example": model_example(problem),
                      "project_model_spec": project["model_spec"]},
            "training": training,
            "inference": {"decision": (anomaly_decision(problem)
                                       or treat_decision(problem)
                                       or price_decision(problem)
                                       or order_decision(problem)
                                       or search_decision(problem)),
                          "conclusion": {},
                          "horizon": None, "guard": None,
                          "ready": ready_of(problem)},
            "requires": required_features(problem),
            "policy": policy_of(problem),
            "rank": rank_of(problem),
            "group": group_of(problem),
            "anomaly": anomaly,
            "uplift": uplift,
            "price": price,
            "demand": demand,
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
                  "example": model_example(problem),
                  "project_model_spec": project["model_spec"]},
        "training": training,
        "inference": {"decision": decision_of(problem),
                      "conclusion": conclusion_of(problem),
                      "horizon": value(problem, f"{PART}.horizon"),
                      "guard": guard_of(problem), "ready": None},
        "requires": required_features(problem),
        "policy": policy_of(problem),
        "rank": None,
        "group": None,
        "anomaly": None,
        "uplift": None,
        "price": None,
        "demand": None,
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
        いなければ None) (既製のモデルをそのまま使う予測, まとめるか並べる
        でなければ None. まとめると並べるは埋め込みの一式づくりに渡す)
    """
    # 予測する, まとめるか並べるで既製のモデルをそのまま使うときだけ
    # (要らなくなった問いの古い答えは除外)
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


def rank_of(problem: dict) -> Optional[dict]:
    """
    並べるの反応の表の列と, 並べる件数をまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    rank: Optional[dict]
        user, item, time, reaction (列の名前. 反応の強さは答えていなければ
        None), k (並べる件数. 整数) (並べるで学習しなければ None)
    """
    # 並べるで学習するときだけ (要らなくなった問いの古い答えは除外)
    if not is_rank_part(problem):
        return None
    rank: dict[str, Any] = {
        key: active_value(problem, f"{PART}.rank.{key}")
        for key in ("user", "item", "time", "reaction")
    }
    k = active_value(problem, f"{PART}.rank.k")
    rank["k"] = (int(k) if isinstance(k, (int, float))
                 and not isinstance(k, bool) and float(k).is_integer()
                 else k)

    return rank


def rank_data(problem: dict) -> dict:
    """
    並べるの受け持ちの, データの分析の段階に渡す分割と時刻の列を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        split (time. 時刻の順) と time_column (反応の時刻の列) (並べるで
        学習しなければ空)
    """
    # 骨組みは反応を時刻の順に分ける
    data: dict = {}
    if not is_rank_part(problem):
        return data
    data = {"split": "time",
            "time_column": active_value(problem, f"{PART}.rank.time")}

    return data


def group_of(problem: dict) -> Optional[dict]:
    """
    まとめるの群を作る列, 群の数と, 群の安定性の数え方をまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    group: Optional[dict]
        columns (群を作る列の名前の並び), id (行を見分ける列. 答えて
        いなければ None), k (群の数. 整数), stability (seed か sample)
        (まとめるで学習しなければ None)
    """
    # まとめるで学習するときだけ (要らなくなった問いの古い答えは除外)
    if not is_group_part(problem):
        return None
    k = active_value(problem, f"{PART}.group.k")
    group: dict[str, Any] = {
        "columns": list(active_value(problem, f"{PART}.group.columns")
                        or []),
        "id": active_value(problem, f"{PART}.group.id"),
        "k": (int(k) if isinstance(k, (int, float))
              and not isinstance(k, bool) and float(k).is_integer()
              else k),
        "stability": active_value(problem, f"{PART}.group.stability"),
    }

    return group


def group_data(problem: dict) -> dict:
    """
    まとめるの受け持ちの, データの分析の段階に渡す分割を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        split (random. 骨組みが行を無作為に学習用と検証用に分ける) (まとめる
        で学習しなければ空)
    """
    # 正解が無いので, 行の順に意味を持たせない
    data: dict = {}
    if not is_group_part(problem):
        return data
    data = {"split": "random"}

    return data


def anomaly_of(problem: dict) -> Optional[dict]:
    """
    異常の検知の度合いの決め方, 列と, 人が確かめる件数の上限をまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    anomaly: Optional[dict]
        score (autoencoder か distance), label (異常の正解の列. 無ければ
        None), id (行の id の列. 無ければ None), columns (度合いに使う列),
        capacity (件数の上限. 整数), k (評価の K. capacity と同じ) (異常の
        検知でなければ None)
    """
    # 異常の検知のときだけ (要らなくなった問いの古い答えは除外)
    if not is_anomaly_part(problem):
        return None
    capacity = whole(active_value(problem, f"{PART}.anomaly.capacity"))
    anomaly: dict[str, Any] = {
        "score": active_value(problem, f"{PART}.anomaly.score"),
        "label": active_value(problem, f"{PART}.anomaly.label"),
        "id": active_value(problem, f"{PART}.anomaly.id"),
        "columns": list(active_value(problem, f"{PART}.anomaly.columns")
                        or []),
        "capacity": capacity,
        "k": capacity,
    }

    return anomaly


def whole(value: Any) -> Any:
    """
    整数の値の小数を整数にする関数

    Parameters
    ----------
    value: Any
        答えの値

    Returns
    ----------
    number: Any
        整数の値なら int, 他はそのまま
    """
    # 4.0 から 4 への直し (真偽値は対象外)
    number = (int(value) if isinstance(value, (int, float))
              and not isinstance(value, bool) and math.isfinite(value)
              and float(value).is_integer() else value)

    return number


def anomaly_data(problem: dict) -> dict:
    """
    異常の検知の, データの分析の段階に渡す値を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        target, task (anomaly), split (random), known (度合いに使う列),
        drop (他の使えるデータの列), id, unlabeled (正解の列があれば
        normal) (異常の検知でなければ空)
    """
    # ML-EDA の異常の検知の表のレシピの値
    data: dict = {}
    anomaly = anomaly_of(problem)
    if anomaly is None:
        return data
    kept = [*anomaly["columns"], anomaly["label"], anomaly["id"]]
    columns = active_value(problem, "information.columns") or []
    data = {
        "target": anomaly["label"], "task": "anomaly", "split": "random",
        "known": list(anomaly["columns"]),
        "drop": [name for name in columns if name not in kept],
        "id": anomaly["id"],
        "unlabeled": "normal" if anomaly["label"] else None,
    }

    return data


def anomaly_decision(problem: dict) -> Optional[dict]:
    """
    異常の検知の, 推論の段階の振り分けを作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    decision: Optional[dict]
        type (triage) と capacity (人が確かめる件数の上限) だけの振り分け
        (異常の検知でなければ None)
    """
    # 度合いの上位から上限まで回す (損得の値と当たりのクラスは持たない)
    anomaly = anomaly_of(problem)
    if anomaly is None:
        return None
    decision = {"type": "triage", "capacity": anomaly["capacity"]}

    return decision


def uplift_of(problem: dict) -> Optional[dict]:
    """
    効果の見積もりの処置, 結果, 列と, 評価の上位の割合をまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    uplift: Optional[dict]
        treatment (処置の列), outcome (結果の列), outcome_kind (binary か
        amount), id (行の id の列. 無ければ None), columns (見積もりに使う
        列), k (上位 K 割の K) (効果の見積もりでなければ None)
    """
    # 効果の見積もりのときだけ (要らなくなった問いの古い答えは除外)
    if not is_uplift_part(problem):
        return None
    uplift: dict[str, Any] = {
        "treatment": active_value(problem, f"{PART}.uplift.treatment"),
        "outcome": active_value(problem, f"{PART}.uplift.outcome"),
        "outcome_kind": active_value(problem, f"{PART}.uplift.outcome_kind"),
        "id": active_value(problem, f"{PART}.uplift.id"),
        "columns": list(active_value(problem, f"{PART}.uplift.columns")
                        or []),
        "k": active_value(problem, f"{PART}.uplift.k"),
    }

    return uplift


def uplift_data(problem: dict) -> dict:
    """
    効果の見積もりの, データの分析の段階に渡す値を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        target (結果の列), task (uplift), split (stratified), known
        (見積もりに使う列), drop (他の使えるデータの列), id, treatment,
        outcome_kind, unlabeled (drop) (効果の見積もりでなければ空)
    """
    # ML-EDA の介入の記録の表のレシピの値
    data: dict = {}
    uplift = uplift_of(problem)
    if uplift is None:
        return data
    kept = [*uplift["columns"], uplift["treatment"], uplift["outcome"],
            uplift["id"]]
    columns = active_value(problem, "information.columns") or []
    data = {
        "target": uplift["outcome"], "task": "uplift",
        "split": "stratified", "known": list(uplift["columns"]),
        "drop": [name for name in columns if name not in kept],
        "id": uplift["id"], "treatment": uplift["treatment"],
        "outcome_kind": uplift["outcome_kind"], "unlabeled": "drop",
    }

    return data


def treat_decision(problem: dict) -> Optional[dict]:
    """
    効果の見積もりの, 推論の段階の配る判断を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    decision: Optional[dict]
        type (treat) と budget (配る数の上限. 整数) だけの配る判断 (効果の
        見積もりでないか, 配る判断でなければ None)
    """
    # 効果の上位から上限まで配る判断 (損得の値と当たりのクラスは無し)
    raw = active_value(problem, "action.decision")
    if not is_uplift_part(problem) or not isinstance(raw, dict) or (
        raw.get("type") != "treat"
    ):
        return None
    decision = {"type": "treat", "budget": whole(raw.get("budget"))}

    return decision


def price_of(problem: dict) -> Optional[dict]:
    """
    値段を変える判断の値段, 売れた数, 原価, 商品の id, 行の id の列と予測に
    使う列をまとめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    price: Optional[dict]
        price (値段の列), units (売れた数の列), cost (原価の列), product
        (商品の id の列), id (行の id の列. 答えていない列は None),
        columns (予測に使う列. 値段の列は含まない) (値段を変える判断で
        なければ None)
    """
    # 値段を変える判断のときだけ (要らなくなった問いの古い答えは除外)
    if not is_price_part(problem):
        return None
    price: dict[str, Any] = {
        name: active_value(problem, f"{PART}.price.{name}")
        for name in ("price", "units", "cost", "product", "id")
    }
    price["columns"] = list(
        active_value(problem, f"{PART}.price.columns") or []
    )

    return price


def price_data(problem: dict) -> dict:
    """
    値段を変える判断の, データの分析の段階に渡す値を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        target (売れた数の列), task (price), split (stratified), known
        (予測に使う列と, 最後に値段の列), drop (他の使えるデータの列), id,
        price, cost, product, unlabeled (drop) (値段を変える判断でなければ
        空)
    """
    # ML-EDA の値段を変えた記録の表のレシピの値 (値段の列は学習に使う列の
    # 最後)
    data: dict = {}
    price = price_of(problem)
    if price is None:
        return data
    kept = [*price["columns"], price["price"], price["units"],
            price["cost"], price["product"], price["id"]]
    columns = active_value(problem, "information.columns") or []
    data = {
        "target": price["units"], "task": "price", "split": "stratified",
        "known": [*price["columns"], price["price"]],
        "drop": [name for name in columns if name not in kept],
        "id": price["id"], "price": price["price"], "cost": price["cost"],
        "product": price["product"], "unlabeled": "drop",
    }

    return data


def price_decision(problem: dict) -> Optional[dict]:
    """
    値段を変える判断の, 推論の段階の値段の判断を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    decision: Optional[dict]
        type (price), candidates (今の値段に対する変える割合の並び),
        max_change (変える幅の上限) だけの値段の判断 (値段を変える判断で
        ないか, 値段の判断でなければ None)
    """
    # 候補の中で上限を超えない, 利益が最も大きい値段を選ぶ判断
    raw = active_value(problem, "action.decision")
    if not is_price_part(problem) or not isinstance(raw, dict) or (
        raw.get("type") != "price"
    ):
        return None
    candidates = raw.get("candidates")
    decision = {
        "type": "price",
        "candidates": (list(candidates) if isinstance(candidates, list)
                       else None),
        "max_change": raw.get("max_change"),
    }

    return decision


def demand_of(problem: dict) -> Optional[dict]:
    """
    時間をまたぐ量の判断の列, 先の期間の日数と学ぶ分位点の水準をまとめる
    関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    demand: Optional[dict]
        product (商品の id の列), date (日付の列), units (日ごとの売れた数の
        列), stock (在庫の数の列), id (行の id の列. 答えていない列は
        None), columns (商品の特徴の列), horizon (先の期間の日数. 答えて
        いなければ 7), levels (学ぶ水準. 小さい順で, 0.5 と水準 τ を含む)
        (時間をまたぐ量の判断でなければ None)
    """
    # 時間をまたぐ量の判断のときだけ (要らなくなった問いの古い答えは除外)
    if not is_demand_part(problem):
        return None
    demand: dict[str, Any] = {
        name: active_value(problem, f"{PART}.demand.{name}")
        for name in ("product", "date", "units", "stock", "id")
    }
    horizon = active_value(problem, f"{PART}.demand.horizon")
    demand.update(
        columns=list(active_value(problem, f"{PART}.demand.columns") or []),
        horizon=DEMAND_HORIZON if horizon is None else int(horizon),
        levels=learned_levels(problem),
    )

    return demand


def order_level(decision: Any) -> Optional[float]:
    """
    発注の判断の品切れの損と売れ残りの損から, 発注に使う水準 τ を求める
    関数

    Parameters
    ----------
    decision: Any
        意思決定 (action.decision)

    Returns
    ----------
    tau: Optional[float]
        τ = shortage_cost / (shortage_cost + holding_cost) (10 進の表記
        どおりに求めた値. 発注の判断でないか, 損が正の数でなければ None)
    """
    # 2 つの損が正の数の発注の判断だけ (10 進の表記どおり. ML-Inference の
    # 水準の求め方と同じ)
    if not isinstance(decision, dict) or decision.get("type") != "order":
        return None
    costs = [decision.get(key) for key in ("shortage_cost", "holding_cost")]
    if not all(is_finite_number(value) and value > 0 for value in costs):
        return None
    shortage, holding = (Decimal(str(value)) for value in costs)
    tau = float(shortage / (shortage + holding))

    return tau


def learned_levels(problem: dict) -> list[float]:
    """
    時間をまたぐ量の判断で学ぶ分位点の水準を決める関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 時間をまたぐ量の判断)

    Returns
    ----------
    levels: list[float]
        答えた水準 (答えていなければ DEMAND_LEVELS) に 0.5 と水準 τ (損が
        書けていれば) を足し, 同じ水準を 1 つにまとめた小さい順の並び
    """
    # 品切れの損と売れ残りの損が同じなら τ は 0.5 で, 重ならないよう 1 つ
    # にまとめる (ML-Found の PinballLoss は重なった水準を誤りにする)
    answered = active_value(problem, f"{PART}.demand.levels")
    levels = {float(level) for level in (answered or DEMAND_LEVELS)}
    levels.add(0.5)
    tau = order_level(active_value(problem, "action.decision"))
    if tau is not None:
        levels.add(tau)
    levels = sorted(levels)

    return levels


def demand_data(problem: dict) -> dict:
    """
    時間をまたぐ量の判断の, データの分析の段階に渡す値を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        target (日ごとの売れた数の列), task (demand), split (time), known
        (商品の特徴の列), drop (他の使えるデータの列), id, product, date,
        stock, horizon, unlabeled (drop) (時間をまたぐ量の判断でなければ
        空)
    """
    # ML-EDA の需要の記録の表のレシピの値 (過去の売れた数のずらしと曜日は
    # レシピの手順)
    data: dict = {}
    demand = demand_of(problem)
    if demand is None:
        return data
    kept = [*demand["columns"], demand["product"], demand["date"],
            demand["units"], demand["stock"], demand["id"]]
    columns = active_value(problem, "information.columns") or []
    data = {
        "target": demand["units"], "task": "demand", "split": "time",
        "known": list(demand["columns"]),
        "drop": [name for name in columns if name not in kept],
        "id": demand["id"], "product": demand["product"],
        "date": demand["date"], "stock": demand["stock"],
        "horizon": demand["horizon"], "unlabeled": "drop",
    }

    return data


def order_decision(problem: dict) -> Optional[dict]:
    """
    時間をまたぐ量の判断の, 推論の段階の発注の判断を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    decision: Optional[dict]
        type (order), shortage_cost, holding_cost, max_order だけの発注の
        判断 (時間をまたぐ量の判断でないか, 発注の判断でなければ None)
    """
    # 損の比から決まる水準の分位点から在庫を引き, 上限の中で発注する判断
    raw = active_value(problem, "action.decision")
    if not is_demand_part(problem) or not isinstance(raw, dict) or (
        raw.get("type") != "order"
    ):
        return None
    decision = {"type": "order",
                **{key: raw.get(key) for key in ("shortage_cost",
                                                 "holding_cost",
                                                 "max_order")}}

    return decision


def search_decision(problem: dict) -> Optional[dict]:
    """
    似た写真を探すの, 推論の段階の検索の判断を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    decision: Optional[dict]
        type (search) と k だけの検索の判断 (似た写真を探すでないか, 検索
        の判断でなければ None. 待ち時間の上限は骨組みの config で渡す)
    """
    # 問いごとに近い順に K 件を返す判断 (整数の値の小数は整数)
    raw = active_value(problem, "action.decision")
    if not is_search_part(problem) or not isinstance(raw, dict) or (
        raw.get("type") != "search"
    ):
        return None
    k = raw.get("k")
    if isinstance(k, float) and k.is_integer():
        k = int(k)
    decision = {"type": "search", "k": k}

    return decision


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
    # 学習しない場合 (既製のモデルと, 異常の検知の距離に基づく度合い) は
    # 指標を問わない
    metric = active_value(problem, f"{PART}.evaluation.metric")
    distance = is_anomaly_part(problem) and (
        active_value(problem, f"{PART}.anomaly.score") == "distance"
    )
    if active_value(problem, f"{PART}.method") == "pretrained" or distance:
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
    画像か表の生成の受け持ちの, データの分析の段階に渡す分割を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    data: dict
        split (書き戻したレシピの split.method. 無ければ random. 画像か
        表の生成でなければ空)
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
    if isinstance(guard, dict) and value(problem, "issue.guard.name"):
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
    rank = rank_of(problem)
    if rank and rank["user"] and rank["item"]:
        sentences.append(
            f"並べ方: 利用者 ({rank['user']}) ごとに品物 ({rank['item']}) を"
            + (f"上位 {rank['k']} 件" if rank["k"] else "上位から")
            + "並べる"
            + (f" (反応の時刻 {rank['time']} の順に分けて確かめる)"
               if rank["time"] else "")
        )
    group = group_of(problem)
    stabilities = {item["value"]: item["label"].split(" ")[0]
                   for item in GROUP_STABILITIES}
    if group and group["columns"]:
        method = stabilities.get(str(group["stability"]))
        sentences.append(
            f"まとめ方: {', '.join(map(str, group['columns']))} の列から"
            + (f" {group['k']} つの群" if group["k"] else "群")
            + "を作る"
            + (f" (群の安定性は{method}ときの一致で数える)" if method
               else "")
        )
    if is_ready_group_part(problem):
        k = value(problem, f"{PART}.group.k")
        method = stabilities.get(
            str(value(problem, f"{PART}.group.stability"))
        )
        sentences.append(
            "まとめ方: 写真を既製のモデルの埋め込みで"
            + (f" {k} つの群" if k else "群") + "に分ける"
            + (f" (群の安定性は{method}ときの一致で数える)" if method
               else "")
        )
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


def counted_of(problem: dict) -> Optional[str]:
    """
    守る指標がたどり直しで数える量の名前を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    counted: Optional[str]
        ML-Inference の GUARD_COUNTS の, 意思決定の型の量の名前 (指さない,
        答えていない, 要らない問い, 型に無い量なら None)
    """
    # 今の意思決定の型の量だけ
    counted = active_value(problem, "issue.guard.counted")
    kind = get_value(problem, "action.decision.type")
    counts = GUARD_COUNTS.get(kind, {}) if isinstance(kind, str) else {}
    if not isinstance(counted, str) or counted not in counts:
        return None

    return counted


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
    elif isinstance(value, list):
        text = "[" + ", ".join(number_text(item) for item in value) + "]"
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
    if decision.get("type") == "treat":
        names["budget"] = "配る数の上限"
    if decision.get("type") == "price":
        names["max_change"] = "値段を変える幅の上限"
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
