from pathlib import Path
from pprint import pformat
from string import Template
from typing import Any, Callable, Optional

from ml_quest.bridge import recipe_from
from ml_quest.data import content_kind
from ml_quest.problem import active_value, get_value
from ml_quest.questions import PART

# 比べる相手の評価の名前の頭 (evaluations/baseline_<分割>.json)
EVALUATION_PREFIX = "baseline_"

# 比べる相手の名前を答えていない場合の名前
DEFAULT_NAME = "今のやり方"

# 今のやり方の値を作り, 検証用とテスト用で学習したモデルと比べる部分
BASELINE = '''

# 比べる相手 (今のやり方). 問題の設定の今のやり方 (action.current) を,
# プロジェクトに写した元のデータの上で作り, 検証用とテスト用で学習した
# モデルと同じ行で比べる
BASELINE_NAME = $name
CURRENT = $current
DATA_FILE = PROJECT_DIR / $data_file
LOADER = $loader
OPTIONS = $options
TARGET = $target
SERIES_BY = $series_by
TIME_COLUMN = $time_column
INTERVAL_SECONDS = $interval
POSITIVE = $positive

# 前処理した値を同じ行とみなす桁 (小数の誤差の除外)
DIGITS = 6
$values

def row_keys(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    行を見分ける値 (数にして丸めた値) の表を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        表
    columns: list[str]
        使う列

    Returns
    ----------
    keys: pd.DataFrame
        数にして DIGITS 桁に丸めた表 (日時は数, 数にできない値は欠損)
    """
    # 列ごとに数にしてから丸め
    keys = frame[columns].apply(pd.to_numeric, errors="coerce")
    keys = keys.astype("float64").round(DIGITS)

    return keys


def matched_current(
    dataset: Dataset, split: str, frame: pd.DataFrame,
    current: pd.Series, target: pd.Series
) -> pd.Series:
    """
    分割の行ごとに, 元のデータの同じ行の今のやり方の値を引く関数

    元のデータにデータセットの前処理をかけ, 学習に使う列と目的の値が
    そろう行を同じ行とみなす

    Parameters
    ----------
    dataset: Dataset
        ML-EDA のデータセット
    split: str
        分割の名前
    frame: pd.DataFrame
        元のデータ
    current: pd.Series
        元のデータの行ごとの今のやり方の値
    target: pd.Series
        元のデータの行ごとの目的の値 (分類はクラスの番号)

    Returns
    ----------
    found: pd.Series
        分割の行の順の今のやり方の値 (見つからない行と, 同じ値の行で
        今のやり方が違う行は欠損)
    """
    # 元のデータの前処理した値 (データセットに時刻の列が残っていれば時刻も)
    # と, 同じ値の行で 1 つに決まる今のやり方
    columns = [*dataset.features, TARGET]
    raw = dataset.transform(frame)
    raw[TARGET] = target.to_numpy()
    if TIME_COLUMN in dataset.frames[split] and TIME_COLUMN in frame:
        columns.append(TIME_COLUMN)
        raw[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN]).to_numpy()
    raw = row_keys(raw, columns)
    raw["__current__"] = current.to_numpy()
    grouped = raw.dropna().groupby(columns)["__current__"]
    table = pd.DataFrame({"count": grouped.nunique(),
                          "__current__": grouped.first()})
    table = table[table["count"] == 1].drop(columns="count").reset_index()

    # 分割の行の順に引く (left の merge は左の並びのまま)
    rows = dataset.frames[split].copy()
    if TIME_COLUMN in columns:
        rows[TIME_COLUMN] = pd.to_datetime(rows[TIME_COLUMN])
    rows = row_keys(rows, columns)
    found = rows.merge(table, on=columns, how="left")["__current__"]

    return found


def compare_baseline(
    model: nn.Module, dataset: Dataset, experiment: Experiment,
    center: float, scale: float
) -> None:
    """
    検証用とテスト用で学習したモデルと今のやり方を同じ行で比べ, 評価の表
    (evaluations/baseline_<分割>.json) に書く関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    dataset: Dataset
        ML-EDA のデータセット
    experiment: Experiment
        実験 (train.log と evaluations/ に書く)
    center: float
        目的の値の中心 (回帰と予測. 分類は 0)
    scale: float
        目的の値の幅 (回帰と予測. 分類は 1)

    Returns
    ----------
    None
    """
    # 比べられる形か (学習に使う列が無いと, 目的の値で行を選んでしまう)
    reason = None
    if not DATA_FILE.exists():
        reason = f"{DATA_FILE} がありません"
    elif dataset.target != TARGET:
        reason = (f"データセットの目的の列 {dataset.target} が問題の設定"
                  f"の {TARGET} と違います")
    elif not dataset.features:
        reason = "学習に使う列が無いので, 元のデータの行とそろえられません"
    if reason:
        experiment.logger.info(f"今のやり方とは比べません: {reason}")
        return

    # 元のデータと, 行ごとの今のやり方と目的の値, 分割ごとの行のそろえ
    # (作れなければ知らせだけ. 学習と推論の一式は済んでいる)
    try:
        data = load_data(DATA_FILE, loader=LOADER, **OPTIONS)
        frame = data.frame.reset_index(drop=True)
        classes = dataset.info.get("classes") or []
        current = current_values(frame, classes)
        target = target_values(frame, classes)
        found_rows = {
            split: matched_current(dataset, split, frame, current, target)
            for split in ("valid", "test") if split in dataset.frames
        }
    except (OSError, KeyError, ValueError, TypeError) as error:
        experiment.logger.info(f"今のやり方とは比べません: {error}")
        return

    # 分割ごとに, 今のやり方の分かる行だけで比べる
    model = model.cpu().eval()
    for split, found in found_rows.items():
        keep = torch.tensor(found.notna().to_numpy())
        if not bool(keep.any()):
            experiment.logger.info(
                f"{split}: 今のやり方と比べられる行がありません (元のデータ"
                "とデータセットの行がそろわないか, 今のやり方が欠けている)"
            )
            continue
        x, y = dataset.tensors(split)
        with torch.no_grad():
            output = model(x[keep])
        base = torch.tensor(found[found.notna()].to_numpy(dtype="float64"))
        row = {"rows": int(keep.sum()), "split_rows": len(found),
               **scores(output, y[keep], base, center, scale)}
        save_evaluation(
            experiment, f"$prefix{split}", row, split=split,
            baseline=BASELINE_NAME,
            description=(
                f"今のやり方 ({BASELINE_NAME}. 書き方 {CURRENT}) との比べ. "
                f"元のデータの同じ行が分かる {row['rows']} 行 ({len(found)} "
                "行のうち) で求めた"
            )
        )
        experiment.logger.info(f"{split} の今のやり方との比べ: {row}")'''

# 回帰と予測: 今のやり方の値は目的の値と同じ単位
REGRESSION_VALUES = '''

def current_values(frame: pd.DataFrame, classes: list) -> pd.Series:
    """
    元のデータの行ごとに, 今のやり方の値を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        元のデータ
    classes: list
        分類とそろえるための引数 (使わない)

    Returns
    ----------
    values: pd.Series
        今のやり方の値 (前の期の値が無い行は欠損)
    """
    # 列はそのまま. k 期前と平均は ML-EDA の過去の値の手順を時刻でずらす
    # 扱い (休みで日が飛んでも暦の k 期前. 系列ごと)
    if CURRENT["type"] == "column":
        values = pd.to_numeric(frame[CURRENT["column"]], errors="coerce")
        return values
    k = int(CURRENT["periods"])
    params = {"columns": [TARGET], "by": SERIES_BY, "time": TIME_COLUMN,
              "freq": f"{int(INTERVAL_SECONDS)}s", "match": "exact"}
    if CURRENT["type"] == "lag":
        params["lags"] = [k]
        name = f"{TARGET}_lag{k}"
    else:
        params.update(lags=[], windows=[k])
        name = f"{TARGET}_mean{k}"
    shifted = STEPS["lag"].apply(frame, params, {})
    values = pd.to_numeric(shifted[name], errors="coerce")

    return values


def target_values(frame: pd.DataFrame, classes: list) -> pd.Series:
    """
    元のデータの行ごとの目的の値を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        元のデータ
    classes: list
        分類とそろえるための引数 (使わない)

    Returns
    ----------
    values: pd.Series
        目的の値 (数でない値は欠損)
    """
    # 数の値
    values = pd.to_numeric(frame[TARGET], errors="coerce")

    return values


def scores(
    output: torch.Tensor, y: torch.Tensor, current: torch.Tensor,
    center: float, scale: float
) -> dict:
    """
    モデルと今のやり方の${label}と, その差を求める関数

    Parameters
    ----------
    output: torch.Tensor
        モデルの出力 (標準化した値)
    y: torch.Tensor
        目的の値
    current: torch.Tensor
        今のやり方の値
    center: float
        目的の値の中心
    scale: float
        目的の値の幅

    Returns
    ----------
    result: dict
        $key (モデル), ${key}_baseline (今のやり方), ${key}_diff (モデル -
        今のやり方. 負ならモデルが良い)
    """
    # 元の単位に戻した誤差
    pred = output.squeeze(-1).double() * scale + center
    y = y.double()
    error = pred - y
    model_value = $expression.item()
    error = current - y
    base_value = $expression.item()
    result = {"$key": model_value, "${key}_baseline": base_value,
              "${key}_diff": model_value - base_value}

    return result
'''

# 分類: 今のやり方は当たりのクラスに回したか (行動したか) の列
CLASSIFICATION_VALUES = '''

def label_text(value: object) -> str:
    """
    クラスの値を, 比べるための文字にする関数

    Parameters
    ----------
    value: object
        クラスの値

    Returns
    ----------
    text: str
        文字 (整数の値の小数は整数の形)
    """
    # 1.0 は 1 (numpy の小数も)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value)

    return text


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


def current_values(frame: pd.DataFrame, classes: list) -> pd.Series:
    """
    元のデータの行ごとに, 今のやり方が選んだクラスの番号を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        元のデータ
    classes: list
        データセットのクラス (番号の順)

    Returns
    ----------
    values: pd.Series
        行動した行は当たりのクラス, しなかった行はもう 1 つのクラスの番号
        (読めない行は欠損)

    Raises
    ----------
    ValueError
        2 クラスでない場合と, 当たりのクラスがクラスに無い場合
    """
    # 2 クラスで, 当たりのクラスがあること
    names = [label_text(value) for value in classes]
    if len(names) != 2 or label_text(POSITIVE) not in names:
        raise ValueError(
            f"当たりのクラス {POSITIVE} のある 2 クラスの分類だけ比べます "
            f"(クラス: {names})"
        )
    hit = names.index(label_text(POSITIVE))

    # 行動したかから, 選んだクラス
    flags = acted(frame[CURRENT["column"]])
    chosen = flags.fillna(False).astype(bool)
    known = flags.notna().astype(bool)
    values = pd.Series(float(1 - hit), index=frame.index).where(
        ~chosen, float(hit)
    ).where(known)

    return values


def target_values(frame: pd.DataFrame, classes: list) -> pd.Series:
    """
    元のデータの行ごとのクラスの番号を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        元のデータ
    classes: list
        データセットのクラス (番号の順)

    Returns
    ----------
    values: pd.Series
        クラスの番号 (クラスに無い値は欠損)
    """
    # クラスの文字から番号
    index = {label_text(value): number
             for number, value in enumerate(classes)}
    values = frame[TARGET].map(label_text).map(index).astype("float64")

    return values


def scores(
    output: torch.Tensor, y: torch.Tensor, current: torch.Tensor,
    center: float, scale: float
) -> dict:
    """
    モデルと今のやり方の正解率と, その差を求める関数

    Parameters
    ----------
    output: torch.Tensor
        モデルの出力 (softmax を付けない値)
    y: torch.Tensor
        クラスの番号
    current: torch.Tensor
        今のやり方が選んだクラスの番号
    center: float
        回帰とそろえるための引数 (使わない)
    scale: float
        回帰とそろえるための引数 (使わない)

    Returns
    ----------
    result: dict
        accuracy (モデル), accuracy_baseline (今のやり方), accuracy_diff
        (モデル - 今のやり方. 正ならモデルが良い)
    """
    # 正解率
    model_value = (output.argmax(dim=1) == y).float().mean().item()
    base_value = (current.long() == y).float().mean().item()
    result = {"accuracy": model_value, "accuracy_baseline": base_value,
              "accuracy_diff": model_value - base_value}

    return result
'''

# 回帰と予測の指標 (名前, 式, 名前の文)
METRIC_EXPRESSIONS = {
    "mae": ("error.abs().mean()", "平均絶対誤差"),
    "rmse": ("error.pow(2).mean().sqrt()", "二乗平均平方根誤差"),
}


def python_value(value: Any) -> str:
    """
    値を, 79 文字に収まる Python の式にする関数

    Parameters
    ----------
    value: Any
        辞書, 並び, 文字, 数, 真偽値か None

    Returns
    ----------
    text: str
        式 (長い値は括弧の中で折り返す)
    """
    # pprint の形 (Python の式として有効. 真偽値と None もそのまま)
    text = pformat(value, width=60, sort_dicts=False)

    return text


def baseline_reason(
    problem: dict, stages: dict, data_file: Optional[Path]
) -> Optional[str]:
    """
    学習のスクリプトで今のやり方と比べられない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    data_file: Optional[Path]
        プロジェクトに写すデータ (無ければ None)

    Returns
    ----------
    reason: Optional[str]
        理由の文 (比べられるなら None)
    """
    # 画像と文か, 今のやり方をデータで書いたか, データを入れたか
    current = active_value(problem, "action.current")
    task = stages["data"]["task"]
    reason: Optional[str] = None
    if content_kind(stages) is not None:
        reason = ("入れたデータが 1 件の中身で学ぶ画像か文 (データセットに"
                  "学習に使う列が無く, 元のデータの同じ行が分からない)")
    elif not isinstance(current, dict) or "type" not in current:
        reason = ("問題の設定に今のやり方をデータで書いていない (今のやり方"
                  "をデータで書く問い, action.current)")
    elif data_file is None:
        reason = "問題の設定に手元のデータを入れていない"

    # k 期前と平均は時刻の列と間隔, 分類は当たりのクラスと列
    elif current["type"] != "column" and (
        not stages["data"]["time_column"]
        or not get_value(problem, "data.interval_seconds")
    ):
        reason = "k 期前と平均を数える時刻の列か, 時刻の間隔が分からない"
    elif task == "classification" and (
        current["type"] != "column"
        or active_value(problem, f"{PART}.positive") is None
    ):
        reason = ("分類は, 当たりのクラス (positive) と, 当たりのクラスに"
                  "回したかの列で書いた今のやり方だけと比べる")

    return reason


def baseline_note(
    problem: dict, reason: str, comment_lines: Callable, retry: bool = True
) -> str:
    """
    今のやり方と比べない理由を, 学習のスクリプトのコメントにする関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    reason: str
        比べない理由
    comment_lines: Callable
        文を 1 行ずつのコメントにする関数 (skeleton.comment_lines)
    retry: bool = True
        答えて書き出し直せば比べることも書くか (画像と文では False)

    Returns
    ----------
    note: str
        コメントの行
    """
    # 比べる相手の名前と理由と, 比べるための手
    name = active_value(problem, f"{PART}.baseline") or DEFAULT_NAME
    text = f"比べる相手 ({name}): {reason} ので, 今のやり方とは比べない"
    if retry:
        text += (". 答えて骨組みを書き出し直すと, 検証用とテスト用で今のやり方"
                 "と比べた評価の表 (evaluations/"
                 f"{EVALUATION_PREFIX}<分割>.json) を書く")
    note = "\n" + comment_lines(text, indent="")

    return note


def baseline_code(
    problem: dict, stages: dict, data_file: Path, literal: Callable
) -> str:
    """
    今のやり方と比べる部分の学習のスクリプトを作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 比べられるもの)
    stages: dict
        受け渡し
    data_file: Path
        プロジェクトに写すデータ (プロジェクトからの相対)
    literal: Callable
        文字を Python の式にする関数 (skeleton.literal)

    Returns
    ----------
    code: str
        定数, 今のやり方の値, 行のそろえ, 比べる関数
    """
    # 今のやり方の値と比べ方 (タスクごと)
    data = stages["data"]
    metric = stages["training"]["metric"]
    if data["task"] == "classification":
        values = CLASSIFICATION_VALUES
    else:
        expression, label = METRIC_EXPRESSIONS[metric]
        values = Template(REGRESSION_VALUES).substitute(
            key=metric, expression=expression, label=label
        )

    # 定数 (元のデータの読み方はデータの分析の段階と同じ)
    load = recipe_from(problem)["load"] or {}
    name = active_value(problem, f"{PART}.baseline") or DEFAULT_NAME
    code = Template(BASELINE).substitute(
        name=literal(str(name)),
        current=python_value(active_value(problem, "action.current")),
        data_file=literal(data_file.as_posix()),
        loader=literal(load.get("loader")),
        options=python_value(load.get("options") or {}),
        target=literal(data["target"]),
        series_by=python_value(list(data["series_by"])),
        time_column=literal(data["time_column"]),
        interval=python_value(get_value(problem, "data.interval_seconds")),
        positive=python_value(active_value(problem, f"{PART}.positive")),
        values=values, prefix=EVALUATION_PREFIX,
    )

    return code
