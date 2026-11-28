from typing import Any, Optional

from ml_quest.check import finding
from ml_quest.handoff import learned_levels, uplift_learner
from ml_quest.model_names import (
    CONTROL_OUTPUT, EMBEDDING_OUTPUT, FEATURE_INPUT, ITEM_INPUT,
    OUTCOME_OUTPUT, QUANTILES_OUTPUT, RECON_OUTPUT, SCORE_OUTPUT,
    SIMILARITY_OUTPUT, TREATED_OUTPUT, UNITS_OUTPUT, USER_INPUT
)
from ml_quest.problem import active_value, upgrade_problem
from ml_quest.questions import PART
from ml_quest.skeleton import skeleton_type

# 表と画像の生成の時刻の入力の型 (骨組みの noise_inputs と同じ. もう 1 つの
# 入力が雑音を加えた行か画像)
TIME_DTYPE = "int64"

# 最後の大きさの鍵ごとの説明 (骨組みの確かめの文と同じ言葉). 鍵は
# ML-ModelControl の同梱の例の書き換える引数 (example_args) と同じ名前
WIDTH_LABELS = {
    "features": "学習に使う列の数",
    "users": "学習用の利用者の数",
    "items": "学習用の品物の数",
    "levels": "学ぶ水準の数",
    "observations": "観測の要素の数",
    "actions": "行動の数か次元の数",
}


def entry(
    name: Optional[str], text: str, width: Optional[str] = None,
    size: Optional[int] = None, label: Optional[str] = None,
    dtype: Optional[str] = None, offset: int = 0
) -> dict:
    """
    骨組みの約束の入力か出力の 1 項目を作る関数

    Parameters
    ----------
    name: Optional[str]
        名前 (None は名前を問わない)
    text: str
        項目の説明
    width: Optional[str] = None
        形の最後の大きさの鍵 (WIDTH_LABELS の鍵. 決まらなければ None)
    size: Optional[int] = None
        問題の設定か骨組みで決まる最後の大きさ (決まらなければ None)
    label: Optional[str] = None
        最後の大きさの説明 (None なら鍵の説明. 鍵が無ければ None)
    dtype: Optional[str] = None
        骨組みが型で入力を見分ける場合の型 (他は None)
    offset: int = 0
        渡した鍵の数に足す数 (効果の見積もりの s_learner の処置の列の 1)

    Returns
    ----------
    item: dict
        name, dtype, width, size, label, text, offset
    """
    # 説明は鍵の説明を既定に
    if label is None and width is not None:
        label = WIDTH_LABELS[width]
    item = {"name": name, "dtype": dtype, "width": width, "size": size,
            "label": label, "text": text, "offset": offset}

    return item


def contract_of(
    kind: str, inputs: list[dict], outputs: list[dict], extra_outputs: bool
) -> dict:
    """
    骨組みの約束の辞書を作る関数

    Parameters
    ----------
    kind: str
        骨組みの型 (SKELETON_TYPES のどれか)
    inputs: list[dict]
        入力の項目の並び
    outputs: list[dict]
        出力の項目の並び
    extra_outputs: bool
        約束に無い出力を持ってよいか

    Returns
    ----------
    contract: dict
        skeleton, inputs, outputs, extra_outputs
    """
    # 画面と文書にそのままの形
    contract = {"skeleton": kind, "inputs": inputs, "outputs": outputs,
                "extra_outputs": extra_outputs}

    return contract


def named_contract(problem: dict, kind: str) -> Optional[dict]:
    """
    入力と出力の名前を決めた骨組みの約束を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    kind: str
        骨組みの型

    Returns
    ----------
    contract: Optional[dict]
        骨組みの約束 (名前を決めた骨組みでなければ None)
    """
    # 骨組みの型ごとの入力と出力 (他の出力も持てる)
    row = "行の特徴"
    encoded = "符号化した行の幅"
    if kind == "anomaly":
        inputs = [entry(FEATURE_INPUT, row, "features")]
        outputs = [entry(RECON_OUTPUT, "入力と同じ形の再構成", "features")]
    elif kind == "rank":
        inputs = [entry(USER_INPUT, "利用者の番号のワンホット", "users"),
                  entry(ITEM_INPUT, "品物の番号のワンホット", "items")]
        outputs = [entry(SCORE_OUTPUT, "利用者と品物の全ての組の点数の表")]
    elif kind == "group":
        inputs = [entry(FEATURE_INPUT, row, "features", label=encoded)]
        outputs = [entry(EMBEDDING_OUTPUT, "行の埋め込み"),
                   entry(SIMILARITY_OUTPUT, "全ての組の類似度の表")]
    elif kind == "uplift" and uplift_learner(problem) == "s_learner":
        inputs = [entry(FEATURE_INPUT, "行の特徴と, 最後の列の処置 (1 か 0)",
                        "features",
                        label="特徴の列と処置の列の数 (学習に使う列の数 + 1)",
                        offset=1)]
        outputs = [entry(OUTCOME_OUTPUT, "結果の予測", size=1)]
    elif kind == "uplift":
        inputs = [entry(FEATURE_INPUT, "行の特徴 (処置の列は入れない)",
                        "features", label="処置の列を除いた学習に使う列の数")]
        outputs = [entry(TREATED_OUTPUT, "処置したときの結果の予測", size=1),
                   entry(CONTROL_OUTPUT, "処置しないときの結果の予測", size=1)]
    elif kind == "price":
        inputs = [entry(FEATURE_INPUT, "行の特徴 (最後の列が値段)",
                        "features", label="特徴の列と値段の列の数")]
        outputs = [entry(UNITS_OUTPUT, "売れた数の予測", size=1)]
    elif kind == "demand":
        inputs = [entry(FEATURE_INPUT, row, "features")]
        outputs = [entry(QUANTILES_OUTPUT, "水準の小さい順の分位点",
                         "levels", size=len(learned_levels(problem)))]
    else:
        return None
    contract = contract_of(kind, inputs, outputs, True)

    return contract


def free_contract(kind: str) -> Optional[dict]:
    """
    入力と出力の名前を問わない骨組みの約束を返す関数

    Parameters
    ----------
    kind: str
        骨組みの型

    Returns
    ----------
    contract: Optional[dict]
        骨組みの約束 (名前を問わない骨組みでなければ None)
    """
    # 生成は時刻を型で見分け, 方策は観測と行動を 1 つずつ (他の出力は
    # 持てない)
    time = entry(None, "時刻", dtype=TIME_DTYPE)
    encoded = "符号化した行の幅"
    if kind == "table":
        inputs = [entry(None, "雑音を加えた行", "features", label=encoded),
                  time]
        outputs = [entry(None, "雑音の予測", "features", label=encoded)]
    elif kind == "generate":
        inputs = [entry(None, "雑音を加えた画像"), time]
        outputs = [entry(None, "雑音の予測")]
    elif kind == "policy":
        inputs = [entry(None, "観測", "observations")]
        outputs = [entry(None, "行動", "actions")]
    else:
        return None
    contract = contract_of(kind, inputs, outputs, False)

    return contract


def model_contract(problem: dict) -> Optional[dict]:
    """
    骨組みの学習のスクリプトがモデルの定義に求める入力と出力の約束を返す
    関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 は版 2 に移して見る)

    Returns
    ----------
    contract: Optional[dict]
        skeleton (骨組みの型), inputs と outputs (name, dtype, width, size,
        label, text, offset の項目の並び), extra_outputs (約束に無い出力を持って
        よいか). 約束の無い骨組み (定義を持たないか, 入力の形をデータ
        セットの model_inputs で決める) と
        機械学習を使わない場合は None
    """
    # 機械学習を使わない場合と, 定義の無い距離に基づく度合いの除外
    upgraded = upgrade_problem(problem)
    kind = skeleton_type(upgraded)
    if kind is None:
        return None
    if kind == "anomaly" and (
        active_value(upgraded, f"{PART}.anomaly.score") != "autoencoder"
    ):
        return None

    # 名前を決めた骨組みか, 名前を問わない骨組み
    contract = named_contract(upgraded, kind) or free_contract(kind)

    return contract


def spec_entries(spec: Any, key: str) -> dict:
    """
    定義の辞書から入力か出力の辞書を取り出す関数

    Parameters
    ----------
    spec: Any
        ML-ModelControl の定義を読んだもの
    key: str
        inputs か outputs

    Returns
    ----------
    entries: dict
        名前から中身 (辞書でなければ空)
    """
    # 崩れた定義は空の辞書
    entries = spec.get(key) if isinstance(spec, dict) else None
    if not isinstance(entries, dict):
        entries = {}

    return entries


def free_names(
    free: list[dict], rest: list[str], inputs: dict
) -> tuple[list[tuple[dict, str]], list[dict]]:
    """
    名前を問わない入力の項目に, 定義の入力の名前を当てる関数

    Parameters
    ----------
    free: list[dict]
        名前を問わない入力の項目
    rest: list[str]
        約束の名前に無い, 定義の入力の名前
    inputs: dict
        定義の入力 (名前から shape と dtype)

    Returns
    ----------
    matched: list[tuple[dict, str]]
        項目と当てた名前の組
    found: list[dict]
        確かめ (数か型が合わない)
    """
    # 数が違う場合は当てずに誤り
    texts = ", ".join(item["text"] for item in free)
    if len(rest) != len(free):
        found = [finding("error", "inputs",
                         f"定義の入力を {len(free)} つ ({texts}) にして"
                         f"ください (今は {list(inputs)})")]
        return [], found

    # 型で見分ける項目から当て, 残りは順に当てる形
    matched = []
    found = []
    left = list(rest)
    typed = [item for item in free if item["dtype"]]
    for item in typed:
        hits = [name for name in rest if isinstance(inputs[name], dict)
                and inputs[name].get("dtype") == item["dtype"]]
        if len(hits) != 1:
            found.append(finding(
                "error", "inputs",
                f"定義の入力のうち dtype が {item['dtype']} のもの "
                f"({item['text']}) を 1 つにしてください (今は {hits})"
            ))
            continue
        matched.append((item, hits[0]))
        left.remove(hits[0])
    if not found:
        matched += list(zip(
            [item for item in free if not item["dtype"]], left
        ))

    return matched, found


def width_findings(
    matched: list[tuple[dict, str]], inputs: dict, widths: dict
) -> list[dict]:
    """
    定義の入力の形の最後の大きさを, 約束の数と比べる関数

    Parameters
    ----------
    matched: list[tuple[dict, str]]
        項目と定義の入力の名前の組
    inputs: dict
        定義の入力 (名前から shape と dtype)
    widths: dict
        最後の大きさの鍵から数

    Returns
    ----------
    found: list[dict]
        確かめ (数の分からない鍵と形の無い入力は見ない)
    """
    # 渡した数 (足す数を足す) を, 問題の設定か骨組みで決まる数より先に
    found = []
    for item, name in matched:
        expected = (widths[item["width"]] + item["offset"]
                    if item["width"] in widths else item["size"])
        value = inputs[name]
        shape = value.get("shape") if isinstance(value, dict) else None
        if expected is None or not isinstance(shape, list) or not shape:
            continue
        if shape[-1] != expected:
            found.append(finding(
                "error", f"inputs.{name}",
                f"入力 {name} の最後の大きさを{item['label']} {expected} に"
                f"してください (今は {shape[-1]})"
            ))

    return found


def input_findings(contract: dict, inputs: dict, widths: dict) -> list[dict]:
    """
    定義の入力を骨組みの約束と比べる関数

    Parameters
    ----------
    contract: dict
        骨組みの約束
    inputs: dict
        定義の入力 (名前から shape と dtype)
    widths: dict
        最後の大きさの鍵から数

    Returns
    ----------
    found: list[dict]
        確かめ
    """
    # 名前を決めた入力の有無
    named = [item for item in contract["inputs"] if item["name"]]
    names = [item["name"] for item in named]
    found = [finding("error", "inputs",
                     f"定義の入力に {item['name']} ({item['text']}) を入れ"
                     "てください")
             for item in named if item["name"] not in inputs]
    matched = [(item, item["name"]) for item in named
               if item["name"] in inputs]

    # 約束に無い名前の入力か, 名前を問わない入力
    free = [item for item in contract["inputs"] if not item["name"]]
    rest = [name for name in inputs if name not in names]
    if free:
        more, wrong = free_names(free, rest, inputs)
        matched += more
        found += wrong
    else:
        found += [finding("error", f"inputs.{name}",
                          f"入力 {name} は骨組みが渡さないので除いてください "
                          f"(入力は {', '.join(names)})")
                  for name in rest]

    # 形の最後の大きさ
    found += width_findings(matched, inputs, widths)

    return found


def output_findings(contract: dict, outputs: dict) -> list[dict]:
    """
    定義の出力を骨組みの約束と比べる関数

    Parameters
    ----------
    contract: dict
        骨組みの約束
    outputs: dict
        定義の出力 (名前から出どころ)

    Returns
    ----------
    found: list[dict]
        確かめ (出力は形を持たないので, 名前と数だけ)
    """
    # 名前を決めた出力の有無と, 他の出力を持てない骨組みの数
    found = [finding("error", "outputs",
                     f"定義の出力に {item['name']} ({item['text']}) を入れ"
                     "てください")
             for item in contract["outputs"]
             if item["name"] and item["name"] not in outputs]
    count = len(contract["outputs"])
    if not contract["extra_outputs"] and len(outputs) != count:
        texts = ", ".join(item["text"] for item in contract["outputs"])
        found.append(finding("error", "outputs",
                             f"定義の出力を {count} つ ({texts}) にして"
                             f"ください (今は {list(outputs)})"))

    return found


def check_model_contract(
    problem: dict, spec: Any, widths: Optional[dict] = None
) -> list[dict]:
    """
    モデルの定義が骨組みの約束に合うかを確かめる関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 は版 2 に移して見る)
    spec: Any
        ML-ModelControl の定義を読んだもの (辞書でなければ入力と出力が無い
        とみなす. inputs は名前から shape と
        dtype, outputs は名前から出どころ)
    widths: Optional[dict] = None
        最後の大きさの鍵 (features, users, items, levels, observations,
        actions) から数 (無い鍵は約束の size だけで見る)

    Returns
    ----------
    found: list[dict]
        確かめ (level, id, message. level は error, id は直す場所. 約束が
        無ければ空)
    """
    # 約束の無い骨組みは確かめ無し
    contract = model_contract(problem)
    if contract is None:
        return []

    # 入力と出力と約束の比べ
    found = input_findings(contract, spec_entries(spec, "inputs"),
                           widths or {})
    found += output_findings(contract, spec_entries(spec, "outputs"))

    return found
