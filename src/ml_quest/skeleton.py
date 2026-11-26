import copy
import json
import math
from pathlib import Path
import re
from string import Template
import textwrap
from typing import Any, Optional, Union

import yaml

from ml_quest.anomaly_skeleton import (
    ANOMALY_DATASET, ANOMALY_LIBRARIES, anomaly_config, anomaly_name,
    anomaly_readme, anomaly_script
)
from ml_quest.baseline_skeleton import (
    EVALUATION_PREFIX, baseline_code, baseline_note, baseline_reason,
    predict_log, reported_code, reported_hint, reported_sentence,
    reported_value
)
from ml_quest.check import (
    choice_label, is_anomaly_part, is_column_part, is_generate_part,
    is_group_part,
    is_policy_part, is_price_part, is_rank_part, is_ready_group_part,
    is_uplift_part, split_errors
)
from ml_quest.data import (
    ITEMS_KINDS, TABLE_LOADERS, content_kind, copy_data, dataset_info,
    inspect_data, inspect_key, items_message
)
from ml_quest.generate_skeleton import (
    FID_LIBRARIES, FID_MONITOR, GENERATE_LIBRARIES, generate_config,
    generate_readme, generate_script
)
from ml_quest.group_skeleton import (
    GROUP_DATASET, GROUP_LIBRARIES, group_config, group_name, group_readme,
    group_script
)
from ml_quest.handoff import build_stages, describe, export_stages
from ml_quest.items_skeleton import (
    ITEMS_LIBRARIES, dataset_place, items_config, items_name, items_readme,
    items_script
)
from ml_quest.policy_skeleton import (
    POLICY_LIBRARIES, POLICY_MODEL, policy_config, policy_env, policy_name,
    policy_readme, policy_script
)
from ml_quest.price_skeleton import (
    PRICE_DATASET, PRICE_LIBRARIES, price_config, price_name, price_readme,
    price_script
)
from ml_quest.problem import (
    active_value, data_location, get_value, set_value, upgrade_problem
)
from ml_quest.rank_skeleton import (
    RANK_DATASET, RANK_LIBRARIES, rank_config, rank_name, rank_readme,
    rank_script
)
from ml_quest.table_skeleton import (
    TABLE_LIBRARIES, table_config, table_readme, table_script
)
from ml_quest.questions import METRICS, OUTPUTS, PART
from ml_quest.ready_group_skeleton import (
    READY_GROUP_PHOTOS, ready_group_config, ready_group_message,
    ready_group_readme, ready_group_script
)
from ml_quest.ready_skeleton import (
    READY_CONFIG, READY_SCRIPT, ready_config, ready_message, ready_readme,
    ready_script
)
from ml_quest.uplift_skeleton import (
    UPLIFT_DATASET, UPLIFT_LIBRARIES, uplift_config, uplift_name,
    uplift_readme, uplift_script
)

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は, コードを壊さないように JSON の文字列か 1 行のコメントで埋め込み
HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

骨組みなので, 課題に合わせて書き足す
"""
from pathlib import Path
import shutil
import sys
from typing import Optional
$pandas
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import $eda_names
from ml_found import (
    $found_names
)
from ml_inference import make_bundle
from ml_modelcontrol import build_model, validate

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 予測すること
PROBLEM_NAME = $name
PREDICT = $predict

# 推論の一式の意思決定と結論の既定値 (問題の設定の損得と制約から)
DECISION = $decision
CONCLUSION = $conclusion
'''

# 回帰と予測: 目的の値を標準化して学び, 元の単位の誤差を記録
REGRESSION_STEP = '''

class Step:
    def __init__(self, scale: float):
        # 目的の値の標準化の幅を保存
        self.scale = scale

    def __call__(self, model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の損失と, 元の単位の誤差を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            特徴量と標準化した目的の値
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            二乗誤差 (損失) と, 元の単位の$metric_label ($key)
        """
        # 標準化した値で学び, 元の単位に直した誤差の記録 (バッチの平均で
        # 書ける指標だけ. RMSE はエポック全体の二乗誤差から求める)
        x, y = batch
        pred = model(x).squeeze(-1)
        mse = nn.MSELoss()
        loss = mse(pred, y)
        error = (pred - y) * self.scale
        outputs = {"loss": loss, "$key": $expression}

        return outputs


def evaluate_test(
    model: nn.Module, dataset: Dataset, experiment: Experiment,
    center: float, scale: float
) -> None:
    """
    テスト用の分割の誤差を記録する関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    dataset: Dataset
        ML-EDA のデータセット
    experiment: Experiment
        実験 (train.log に記録する)
    center: float
        目的の値の中心 (学習用の平均)
    scale: float
        目的の値の幅 (学習用の標準偏差)

    Returns
    ----------
    None
    """
    # テスト用の分割が無い場合は知らせだけ
    if "test" not in dataset.frames:
        experiment.logger.info("テスト用の分割が無いので, 評価$no_test")
        return
    model = model.cpu().eval()
    x, y = dataset.tensors("test")
    with torch.no_grad():
        pred = model(x).squeeze(-1) * scale + center
    error = pred - y
    value = $test_expression
    experiment.logger.info(f"テスト用の$test_label: {value.item():,.4f}")$reported
$baseline
'''

# 分類: 交差エントロピーで学び, 正解率を記録
CLASSIFICATION_STEP = '''

class Step:
    def __init__(self, scale: float):
        # 回帰とそろえるための引数 (分類では使わない) を保存
        self.scale = scale

    def __call__(self, model: nn.Module, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の交差エントロピーと正解率を求める関数

        Parameters
        ----------
        model: nn.Module
            学習するモデル
        batch: tuple
            特徴量とクラスの番号
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            交差エントロピー (loss) と正解率 (accuracy)
        """
        # クラスの番号との交差エントロピー
        x, y = batch
        logits = model(x)
        ce = nn.CrossEntropyLoss()
        loss = ce(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        outputs = {"loss": loss, "accuracy": accuracy}

        return outputs


def evaluate_test(
    model: nn.Module, dataset: Dataset, experiment: Experiment,
    center: float, scale: float
) -> None:
    """
    テスト用の分割の正解率を記録する関数

    Parameters
    ----------
    model: nn.Module
        学習した後のモデル
    dataset: Dataset
        ML-EDA のデータセット
    experiment: Experiment
        実験 (train.log に記録する)
    center: float
        回帰とそろえるための引数 (使わない)
    scale: float
        回帰とそろえるための引数 (使わない)

    Returns
    ----------
    None
    """
    # テスト用の分割が無い場合は知らせだけ
    if "test" not in dataset.frames:
        experiment.logger.info("テスト用の分割が無いので, 評価$no_test")
        return
    model = model.cpu().eval()
    x, y = dataset.tensors("test")
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == y).float().mean()
    experiment.logger.info(f"テスト用の正解率: {accuracy.item():.4f}")$reported
$baseline
'''

# 学習の流れ (データセット, モデル, 学習, 評価, 推論の一式)
MAIN = '''

def spec_errors(spec: Path) -> list[str]:
    """
    ML-ModelControl の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Path
        config の model.spec の, プロジェクトの定義

    Returns
    ----------
    errors: list[str]
        誤りの文 (無ければ空. 外部の重みのファイルが読めるかは, 組み立てる
        ときに分かる)
    """
    # 読めない定義も誤りとして返る (例外にならない)
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]

    return errors


def model_file(
    spec: Path, experiment: Experiment
) -> tuple[Path, Optional[Path]]:
    """
    モデルを組み立てる ML-ModelControl の定義のファイルと, 定義の中の相対
    の参照 (外部の重みなど) の起点を返す関数

    Parameters
    ----------
    spec: Path
        config の model.spec の, プロジェクトの定義
    experiment: Experiment
        実験 (ML-Found が実験に写した定義の場所 model_spec_path と, 写した
        元の定義 model_spec_source)

    Returns
    ----------
    path: Path
        ML-Found が実験に写した定義 (学習を始めたときの定義で, source.yaml
        の sha256 と同じ中身). 写せなければプロジェクトの定義
    base_dir: Optional[Path]
        相対の参照の起点 (写しなら元の定義のディレクトリ. プロジェクトの
        定義なら None で, 定義のディレクトリ)
    """
    # 写しの相対の参照は元の定義の場所から (写しの隣には重みが無い)
    if experiment.model_spec_path is not None:
        path = Path(experiment.model_spec_path)
        base_dir = Path(experiment.model_spec_source).parent
        return path, base_dir

    # 写しが無ければ理由を知らせて, プロジェクトの定義
    experiment.logger.warning(
        "モデルの定義を実験に写せなかったので, プロジェクトの定義から"
        "組み立てます (理由は source.yaml の model_spec の status)"
    )
    path = spec
    base_dir = None

    return path, base_dir


def main() -> None:
    """
    データセットとモデルの定義から学習し, 推論の一式を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config とデータセット (無ければ作り方の案内)
    cfg, diff = load_project_config(str(PROJECT_DIR))
    path = PROJECT_DIR / cfg.data.dataset
    if not (path / "dataset.yaml").is_file():
        print(f"{path} がありません. README の「次にすること」の 1 で, "
              "データセットを書き出してください")
        sys.exit(1)
    if not cfg.model.spec:
        print("config の model.spec に定義の場所を書いてください (既定は "
              "models/model.yaml)")
        sys.exit(1)
    spec = PROJECT_DIR / cfg.model.spec
    if not spec.is_file():
        print(f"{spec} がありません. README の「次にすること」の 2 で, "
              "モデルの定義を置いてください")
        sys.exit(1)
    errors = spec_errors(spec)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\\n".join(f"- {message}" for message in errors))
        sys.exit(1)
    dataset = load_dataset(path)
    if "valid" not in dataset.frames:
        print("データセットに検証用の分割がありません. 最も良いモデルを選ぶ"
              "ために, 検証用の割合を 0 より大きくして書き出してください")
        sys.exit(1)

    # 学習用と検証用 ($scaling)
    x_train, y_train = dataset.tensors("train")
    x_valid, y_valid = dataset.tensors("valid")
$scale_lines
    train_loader = DataLoader(
        TensorDataset(x_train, $train_target),
        batch_size=cfg.data.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(x_valid, $valid_target),
        batch_size=cfg.data.batch_size
    )

    # 実験に写したモデルの定義からの組み立てと学習
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    spec, base_dir = model_file(spec, experiment)
    model = build_model(spec, seed=cfg.experiment.seed, base_dir=base_dir)
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}$inputs_log"
    )
    trainer = Trainer(model, experiment, Step(scale))
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の設定の場所
    # origin も残る. モデルの定義は ML-Found が start_experiment で写す)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")

    # 結果の表示とテスト用の評価
    best_epoch = trainer.checkpoint.best_epoch
    best_value = trainer.checkpoint.best_value
    print(f"保存先: {run_dir}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの {cfg.trainer.monitor}: {best_value}")
    evaluate_test(trainer.model, dataset, experiment, center, scale)
$compare
    # 実験の中に推論の一式 (問題の設定の意思決定と結論を既定値に)
    info = make_bundle(
        run_dir / "inference", spec, path, run_dir,
        target_scale=$target_scale, conclusion=CONCLUSION,
        decision=DECISION, name=cfg.experiment.name
    )$decision_log
    check = info["check"]
    if check and check["recorded"] is None:
        experiment.logger.info(
            f"推論の一式: {check['split']} の {check['metric']} "
            f"{check['value']:,.4f} (監視した指標が accuracy か mae でない"
            "ので, 学習の記録とは比べない)"
        )
    elif check:
        experiment.logger.info(
            f"推論の一式: {check['split']} の {check['metric']} "
            f"{check['value']:,.4f} (記録 {check['recorded']}, 合う: "
            f"{check['matches']})"
        )


if __name__ == "__main__":
    main()
'''

# 損得と制約に直すところが残り, 意思決定の既定値を付けないことの記録
DECISION_LOG = (
    "\n    experiment.logger.info(\n        \"問題の設定の損得と制約に直す"
    "ところが残るので, 意思決定の\"\n        \"既定値は付けずに推論の一式を作り"
    "ました\"\n    )"
)

# 学習の記録の入力の文 (表と時系列は学習に使う列, 画像, 文, 音声は 1 件の
# 形)
INPUTS_LOG = {
    None: ('学習に使う列 "\n        f"{len(dataset.features)}: '
           '{dataset.features}'),
    "image": ('1 件の画像の形 (チャンネル, 高さ, 幅): "\n        '
              'f"{tuple(x_train.shape[1:])}'),
    "text": ('1 件の文の形 (語の番号の長さ): "\n        '
             'f"{tuple(x_train.shape[1:])}'),
    "audio": ('1 件の音声の形 (チャンネル, 標本の数): "\n        '
              'f"{tuple(x_train.shape[1:])}'),
}

# README のデータの分析の, 画像, 文, 音声で学ぶもの (表の列でなく 1 件の
# 中身)
CONTENT_DATA = {
    "image": ("学ぶのは 1 件の画像の中身で, 読み込みが作る表の列 (`path`, "
              "`width`, `height` など) は学習に使わない. 画像の大きさがそろ"
              "わないなら, レシピの中身の手順 (`item_steps`) に `resize` "
              "(`width`, `height`) を足してそろえる (そろわないまま ML-EDA は"
              "書き出さない)"),
    "text": ("学ぶのは 1 件の文の中身で, 読み込みが作る表の列 (`file`, "
             "`length`, `words` など) は学習に使わない. 文はレシピの中身の"
             "手順 (`item_steps`) の `vocabulary` で語の番号の並びにする "
             "(例 `[{op: vocabulary, length: 64}]`. `length` は 1 件の長さ,"
             " `max_tokens` は語彙の上限. 無いと ML-EDA が書き出さない)"),
    "audio": ("学ぶのは 1 件の音声の中身で, 読み込みが作る表の列 (`path`, "
              "`sample_rate`, `duration` など) は学習に使わない. 標本化の"
              "周波数, チャンネルの数, 長さがそろわないなら, レシピの中身の"
              "手順 (`item_steps`) に `resample` (`rate`), `mix_channels` "
              "(`channels`), `fix_length` (`seconds`) をこの順に足してそろ"
              "える (例 `[{op: resample, rate: 16000}, {op: mix_channels, "
              "channels: 1}, {op: fix_length, seconds: 1.0}]`. そろわない"
              "まま ML-EDA は書き出さない). 読めない件 (`error` の列に理由"
              "のある件) は表の手順 `filter_rows` で除く. WAVE の他の形式 "
              "(FLAC, MP3, Ogg, AIFF) を読むには soundfile が要る (ML-EDA "
              "の任意の依存 `ml-eda[audio]`. `ml-quest[data]` でも入る)"),
}

# README のモデルの作成の, 入力の形の案内 (データの種類ごと. None は表と
# 時系列か, データを入れていない場合)
CONTENT_INPUTS = {
    None: ("入力の形は書き出したデータセットから決まる (ML-EDA の "
           "`model_inputs(<データセット>)` の `inputs` の `shape`). 表と時系列"
           "では学習に使う列の数 (`dataset.yaml` の features. ワンホットや"
           "過去の値の列で元の列より増える), 画像, 文, 音声, 点群の列では 1 件"
           "の配列の形 (`dataset.yaml` の splits の `shape` の 2 つ目から. 列の"
           "数ではない)"),
    "image": ("入力の形は書き出したデータセットから決まる (ML-EDA の "
              "`model_inputs(<データセット>)` の `inputs` の `shape` と "
              "`dtype`). 画像は `[None, チャンネル, 高さ, 幅]` の float32 "
              "(値は 0 から 1. 学習に使う列の数ではない)"),
    "text": ("入力の形は書き出したデータセットから決まる (ML-EDA の "
             "`model_inputs(<データセット>)` の `inputs` の `shape` と "
             "`dtype`). 文は `[None, 長さ]` の int64 (語の番号). 埋め込み "
             "(`nn.Embedding`) の数 `num_embeddings` は語彙の大きさで, "
             "`model_inputs` の `inputs` の `vocabulary` (埋め草 `<pad>` と"
             "語彙に無い語 `<unk>` を含む. ML-CreateArchitecture の「データ"
             "セットから入力と出力を作る」は, 入力に直につながる埋め込みの数"
             "をこれにする)"),
    "audio": ("入力の形は書き出したデータセットから決まる (ML-EDA の "
              "`model_inputs(<データセット>)` の `inputs` の `shape` と "
              "`dtype`). 音声は `[None, チャンネル, 標本の数]` の float32 "
              "(値は -1 から 1 ほど. 標本の数は周波数 × 秒. 学習に使う列の"
              "数ではない)"),
}

# 回帰と予測の指標 (step の名前, step の式, テスト用の式, 名前)
REGRESSION_METRICS = {
    "mae": ("mae", "error.abs().mean()", "error.abs().mean()",
            "平均絶対誤差"),
    "rmse": ("mse", "error.pow(2).mean()", "error.pow(2).mean().sqrt()",
             "二乗平均平方根誤差"),
}

# config の基準 (ML-Found のプロジェクトの形)
CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題の設定から書き出した骨組み)
$requires
experiment:
  name: $experiment
  seed: 0

optimizer:
  name: Adam
  params:
    lr: 0.001

trainer:
  epochs: 50
  # 問題の設定の「学習で見る指標」$monitor_note
  monitor: $monitor
  mode: $mode$tiebreak
  early_stopping:
    use: true
    patience: 10

# プロジェクト固有の設定
data:
  # ML-EDA (ML-Fathom) で書き出すデータセット (プロジェクトのディレクトリから)
  dataset: outputs/dataset
  batch_size: 64

model:
  # ML-ModelControl の形のモデルの定義 (ML-CreateArchitecture で組む)
  spec: models/model.yaml
''')

# 列の予測の学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の
# 機能の名前 (ml_eda, ml_modelcontrol, ml_inference)
COLUMN_LIBRARIES = ["eda", "modelcontrol", "inference"]

# 差分の config (既定は差分なし)
EXPERIMENT = "# 普段使う差分 (書かなければ基準のまま)\n{}\n"


def literal(value: Any) -> str:
    """
    値を, 学習のスクリプトに書ける Python の式にする関数

    Parameters
    ----------
    value: Any
        辞書, 並び, 文字列, 数, 真偽値か None

    Returns
    ----------
    text: str
        式 (文字は二重引用符. 長い辞書は 1 項目ずつ折り返し, 長い文字列は
        括弧の中で空白か句読点の後で折り返す)
    """
    # 長い辞書は 1 項目ずつ, 長い文字列は括弧の中でつなぐ形
    text = expression(value)
    if isinstance(value, dict) and len(text) > 60:
        items = [f"    {expression(key)}: {expression(item)}"
                 for key, item in value.items()]
        text = "{\n" + ",\n".join(items) + "\n}"
    elif isinstance(value, str) and len(text) > 60:
        pieces = [json.dumps(piece, ensure_ascii=False)
                  for piece in text_pieces(value, 30)]
        text = "(\n" + "\n".join(f"    {piece}" for piece in pieces) + "\n)"

    return text


def expression(value: Any) -> str:
    """
    値を 1 行の Python の式にする関数

    Parameters
    ----------
    value: Any
        辞書, 並び, 文字列, 数, 真偽値か None

    Returns
    ----------
    text: str
        式 (文字は JSON の文字列. 真偽値と None は Python の名前. 有限で
        ない数は float の式. 他の値は文字にする)
    """
    # 型ごとの式 (JSON の true, null は Python では名前の誤りになるため)
    if value is None or isinstance(value, bool):
        text = repr(value)
    elif isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        text = repr(value) if math.isfinite(value) else f'float("{value}")'
    elif isinstance(value, (list, tuple)):
        text = "[" + ", ".join(expression(item) for item in value) + "]"
    elif isinstance(value, dict):
        text = "{" + ", ".join(f"{expression(key)}: {expression(item)}"
                               for key, item in value.items()) + "}"
    else:
        text = json.dumps(str(value), ensure_ascii=False)

    return text


def text_pieces(value: str, size: int) -> list[str]:
    """
    文字列を, 決まった長さほどの切れ目の良い片に分ける関数

    Parameters
    ----------
    value: str
        文字列
    size: int
        片の長さの上限

    Returns
    ----------
    pieces: list[str]
        片の並び (つなぐと元に戻る. 空白か句読点の後で切り, 無ければ長さ
        で切る)
    """
    # 上限の半分より後ろの, 最後の空白か句読点の後
    pieces = []
    rest = value
    while len(rest) > size:
        head = rest[:size]
        cut = max(head.rfind(mark, size // 2) for mark in " ,.、。)")
        cut = cut + 1 if cut >= 0 else size
        pieces.append(rest[:cut])
        rest = rest[cut:]
    pieces.append(rest)

    return pieces


def comment_lines(text: str, indent: str = "    ") -> str:
    """
    文を, 79 文字に収まる 1 行ずつのコメントにする関数

    Parameters
    ----------
    text: str
        文 (改行は空白に)
    indent: str = "    "
        行頭の空白

    Returns
    ----------
    lines: str
        「# 」で始まる行
    """
    # 日本語も 1 文字と数えた折り返し
    flat = " ".join(str(text).split())
    wrapped = textwrap.wrap(flat, width=70 - len(indent)) or [""]
    lines = "\n".join(f"{indent}# {line}" for line in wrapped)

    return lines


def training_script(
    problem: dict, stages: dict, data_file: Optional[Path] = None,
    dropped: bool = False
) -> str:
    """
    問題の設定から学習のスクリプトを作る関数

    Parameters
    ----------
    problem: dict
        問題の設定
    stages: dict
        受け渡し (handoff の結果)
    data_file: Optional[Path] = None
        プロジェクトに写すデータ (プロジェクトからの相対. 無ければ None)
    dropped: bool = False
        損得と制約に直すところが残り, 意思決定の既定値を付けないか (train.log
        にそう書く)

    Returns
    ----------
    script: str
        train.py の中身
    """
    # 比べる相手: 比べられれば今のやり方と比べる部分, 無理なら理由の注記
    task = stages["data"]["task"]
    metric = stages["training"]["metric"]
    inference = stages["inference"]
    kind = content_kind(stages)
    reason = baseline_reason(problem, stages, data_file)
    reported = reported_value(problem) if kind is not None else None
    call = ""
    if reason is None and data_file is not None:
        baseline = baseline_code(problem, stages, data_file, literal)
        compare = ("\n    # 今のやり方との比べ (一式づくりで止まっても表が残る"
                   "ように, 推論の一式の前.\n    # 比べられなくても止めない)"
                   "\n    compare_baseline(trainer.model, dataset, "
                   "experiment, center, scale)\n")
    elif reported is not None:
        # 1 件の中身で学ぶデータの, 答えた今のやり方での指標の値とテスト用の
        # 値の並べ (テスト用の分割が無ければ, 並べないことを train.log に)
        key = "accuracy" if task == "classification" else metric
        value = "accuracy" if task == "classification" else "value"
        baseline = "\n\n" + reported_code(problem, literal, key)
        call = ("\n\n    # 答えた今のやり方での指標の値と並べる\n    "
                f'save_reported(experiment, "{key}", {value}.item())')
        compare = ""
    elif kind is not None:
        # 1 件の中身で学ぶデータで答えていなければ, 比べない理由をコメント
        # と train.log に
        hint = content_hint(problem, task)
        baseline = "\n" + baseline_note(problem, str(reason), comment_lines,
                                        retry=False, hint=hint)
        baseline += ("\n\n# 今のやり方と比べない理由 (train.log に書く)\n"
                     f"BASELINE_REASON = {literal(f'{reason}. {hint}')}")
        compare = ("\n    # 今のやり方と比べない理由\n    experiment.logger."
                   "info(f\"今のやり方とは比べません: {BASELINE_REASON}\")\n")
    else:
        baseline = baseline_note(problem, str(reason), comment_lines)
        compare = ""

    # テスト用の分割が無いときの記録 (答えた値と並べるなら, それも)
    no_test = "せず, 答えた値とも並べません" if call else "しません"

    # 頭 (比べるときだけ pandas, 元のデータの読み込み, 評価の表の書き出し)
    head = Template(HEAD).substitute(
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        decision=literal(inference["decision"]),
        conclusion=literal(inference["conclusion"]),
        pandas=("\nimport numpy as np" if compare and task ==
                "classification" else "")
        + ("\nimport pandas as pd" if compare else ""),
        eda_names=("Dataset, load_data, load_dataset" if task ==
                   "classification" else "STEPS, Dataset, load_data, "
                   "load_dataset") if compare else "Dataset, load_dataset",
        found_names=("Experiment, Trainer, load_project_config, "
                     "save_evaluation,\n    start_experiment"
                     if compare or call else "Experiment, Trainer, "
                     "load_project_config, start_experiment"),
    )

    # タスクごとの step と評価
    if task == "classification":
        body = Template(CLASSIFICATION_STEP).substitute(
            baseline=baseline, reported=call, no_test=no_test
        )
        scaling = "分類は番号のまま"
        scale_lines = "    center, scale = 0.0, 1.0"
        train_target, valid_target = "y_train", "y_valid"
        target_scale = "None"
    else:
        key, expression, test_expression, label = REGRESSION_METRICS[metric]
        body = Template(REGRESSION_STEP).substitute(
            key=key, expression=expression, metric_label=label
            if key == metric else "二乗誤差", test_expression=test_expression,
            test_label=label, baseline=baseline, reported=call,
            no_test=no_test
        )
        scaling = "目的の値は学習用の平均と標準偏差で標準化"
        scale_lines = ("    center, scale = y_train.mean().item(), "
                       "y_train.std().item()")
        train_target = "(y_train - center) / scale"
        valid_target = "(y_valid - center) / scale"
        target_scale = '{"center": center, "scale": scale}'
    tail = Template(MAIN).substitute(
        scaling=scaling, scale_lines=scale_lines,
        train_target=train_target, valid_target=valid_target,
        target_scale=target_scale, compare=compare,
        inputs_log=INPUTS_LOG[kind], decision_log=DECISION_LOG if dropped
        else "", predict_log=predict_log(problem),
    )
    script = head + body + tail

    return script


def content_hint(problem: dict, task: Optional[str]) -> str:
    """
    1 件の中身で学ぶデータで, 今のやり方での指標の値を答えれば並べられる
    ことの文を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    task: Optional[str]
        タスク

    Returns
    ----------
    hint: str
        文 (分類は正解率, 回帰と予測は誤差の指標と並べる)
    """
    # 骨組みのテスト用の評価が求める指標
    hint = reported_hint(problem, "accuracy (正解率)" if task ==
                         "classification" else "mae か rmse")

    return hint


def experiment_name(problem: dict, name: Optional[str]) -> str:
    """
    実験の名前 (ディレクトリの名前) を決める関数

    Parameters
    ----------
    problem: dict
        問題の設定
    name: Optional[str]
        指定した名前

    Returns
    ----------
    text: str
        文字と数字と _ と - だけの名前 (省略すると目的の列とタスクか,
        正解の種類と受け持ちから. 日本語の列の名前も残す)
    """
    # 区切りや空白などのディレクトリに使いにくい文字の _ への置き換え.
    # ファイルの組の既定は正解の種類と受け持ち
    target = active_value(problem, f"{PART}.target.column")
    task = active_value(problem, f"{PART}.task")
    default = (f"{target}_{task}" if is_column_part(problem)
               else policy_name(problem) if is_policy_part(problem)
               else rank_name(problem) if is_rank_part(problem)
               else group_name(problem) if is_group_part(problem)
               else anomaly_name(problem) if is_anomaly_part(problem)
               else uplift_name(problem) if is_uplift_part(problem)
               else price_name(problem) if is_price_part(problem)
               else items_name(problem))
    raw = name or default
    text = re.sub(r"[^\w-]+", "_", raw).strip("_") or "problem"

    return text


def requires_text(libraries: list[str], stages: dict) -> str:
    """
    config の一番上の requires (学習に要る機能の名前) を作る関数

    Parameters
    ----------
    libraries: list[str]
        学習のスクリプトが読み込むライブラリの機能の名前
    stages: dict
        受け渡し (requires に問題の設定から要る機能の名前)

    Returns
    ----------
    text: str
        コメントと requires の行 (末尾に空行)
    """
    # 読み込むライブラリと問題の設定から要る機能 (同じ名前は 1 回)
    names = list(dict.fromkeys([*libraries, *stages["requires"]]))
    text = ("# 学習に要る機能 (ML-Found の登録簿の名前). 学習の前に確かめ, "
            "足りなければ\n# 入れ方を示して止まる\n"
            f"requires: [{', '.join(names)}]\n")

    return text


def config_text(problem: dict, stages: dict, name: Optional[str]) -> str:
    """
    config の基準を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定
    stages: dict
        受け渡し
    name: Optional[str]
        実験の名前

    Returns
    ----------
    text: str
        configs/default/train.yaml の中身
    """
    # 正解率は同じ値を取りやすいので, 同じ値なら損失で比較
    training = stages["training"]
    tiebreak = ("\n  # 正解率が同じなら損失で比べる\n"
                "  monitor_tiebreak: valid/loss\n"
                "  monitor_tiebreak_mode: min"
                if training["metric"] == "accuracy" else "")
    note = (" (RMSE は二乗誤差で選ぶ. 並びは同じ)"
            if training["metric"] == "rmse" else "")
    text = CONFIG.substitute(
        experiment=json.dumps(experiment_name(problem, name),
                              ensure_ascii=False),
        monitor=training["monitor"], mode=training["mode"],
        tiebreak=tiebreak, monitor_note=note,
        requires=requires_text(COLUMN_LIBRARIES, stages),
    )

    return text


def column_lines(data: dict) -> list[str]:
    """
    README のデータの分析の, 学習に使ってよい列と使わない列の行を作る関数

    Parameters
    ----------
    data: dict
        受け渡しのデータの分析の値

    Returns
    ----------
    lines: list[str]
        字下げした箇条の行 (答えが無ければ空)
    """
    # 予測の時点で分かる列と, 分からない列と使ってはいけない列
    lines = []
    if data["known"]:
        lines.append("    - 学習に使ってよい列 (予測の時点で分かる列): "
                     + ", ".join(f"`{name}`" for name in data["known"]))
    if data["drop"]:
        lines.append("    - 学習に使わない列 (予測の時点で分からない列と, "
                     "使ってはいけない列): "
                     + ", ".join(f"`{name}`" for name in data["drop"]))

    return lines


def project_readme(
    problem: dict, stages: dict, data_file: Optional[Path] = None,
    dropped: bool = False
) -> str:
    """
    プロジェクトの README (課題の要約と次にすること) を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定
    stages: dict
        受け渡し
    data_file: Optional[Path] = None
        プロジェクトに写すデータ (プロジェクトからの相対. 無ければ None)
    dropped: bool = False
        損得と制約に直すところが残り, 意思決定の既定値を付けないか

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約 (データを入れていれば, その名前)
    data = stages["data"]
    source = (data.get("source") or {}).get("source")
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, "
             "ML-Found のプロジェクトの骨組み.", "", "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 次にすること (受け渡しの値を各段階で選ぶ)
    split = {"time": "時刻の順 (先の期間を検証用とテスト用に)",
             "group": f"相手ごと (`{data.get('group_column')}` の同じ値を"
                      "同じ分割に)",
             "stratified": "層別 (クラスの割合をそろえて)"}.get(
                 data["split"], "無作為")
    lines += [
        "", "## 次にすること", "",
        "学習に要るもの (ML-Found と, `configs/default/train.yaml` の "
        "`requires` に書いた機能) は, ML-Dashboard の「学習の準備」で確かめて"
        "入れられる. `python train.py` も, 足りなければ入れ方を示して止まる.",
        "",
        "1. **データの分析** (ML-Fathom. コードなら ML-EDA の "
        "`export_dataset`): データを読み, 次の値でレシピを作り, このディレクトリ"
        "の `outputs/dataset` に書き出す. 検証用の分割は必ず作る"
        + (f". データはこのディレクトリの `data/{Path(source).name}` に写した"
           if source and Path(source).exists() else ""),
        f"    - 目的の列: `{data['target']}`, タスク: `{data['task']}`, "
        f"分割: {split}",
    ]
    if data["time_column"]:
        series = ", ".join(f"`{name}`" for name in data["series_by"])
        lines.append(
            f"    - 時刻の列: `{data['time_column']}` (読み込みで日時の列に"
            "指定する. 系列を分ける列が複数あると推定できないため)"
            + (f", 系列を分ける列: {series}" if series else "")
        )
    if data.get("min_lag"):
        lines.append(
            f"    - 過去の値の列は {data['min_lag']} 期前より古い値だけで作る "
            f"(ML-EDA の過去の値の手順の lags は {data['min_lag']} 以上, "
            f"移動平均の gap は {data['min_lag']}). 決める時点で分からない値で"
            "学習すると, 学習のときだけ当たる"
        )
    kind = content_kind(stages)
    lines += column_lines(data) if kind is None else [
        f"    - {CONTENT_DATA[kind]}"
    ]
    outputs = ("クラスの数 (softmax を付けない値)"
               if stages["model"]["outputs"] is None else "1")
    horizon = stages["inference"].get("horizon")
    reason = baseline_reason(problem, stages, data_file)
    if reason is None:
        compare = (
            "今のやり方との比べを, 検証用とテスト用の評価の表 "
            f"(`evaluations/{EVALUATION_PREFIX}valid.json` と "
            f"`{EVALUATION_PREFIX}test.json`. 元のデータの同じ行が分かる行"
            "だけで求める) に書く"
        )
    elif kind is not None and reported_value(problem) is not None:
        compare = reported_sentence(
            problem, "accuracy" if data["task"] == "classification"
            else stages["training"]["metric"]
        )
    elif kind is not None:
        compare = (f"今のやり方とは比べない ({reason}ため). "
                   f"{content_hint(problem, data['task'])}")
    else:
        compare = f"今のやり方とは比べない ({reason}ため)"
    lines += [
        "2. **モデルの作成** (ML-CreateArchitecture): 入力 1 つと出力 1 つの"
        f"モデルを組み, `models/model.yaml` に置く. {CONTENT_INPUTS[kind]}. "
        f"出力の大きさは {outputs}. 回帰の目的の値は train.py が標準化する",
        "3. **学習** (ML-Found, ML-Dashboard): `python train.py` "
        "(`python train.py trainer.epochs=10` のように「項目=値」で上書き"
        "できる). 結果は `outputs/<実験の名前>/<日付>/<時刻>/` に残る. "
        f"検証用の `{stages['training']['monitor']}` が最も良いモデルを選ぶ. "
        + compare,
        "4. **推論** (ML-Provide): 実験の中の `inference/` を登録する. "
        + ("意思決定の既定値は付けない (損得と制約に直すところが残るため. 下"
           "の「書き出した時点で残っていた直すところ」)" if dropped else
           "意思決定の既定値は問題の設定の損得と制約")
        + (f". {horizon} 期先まで予測するなら「何段も先まで予測する」を使う"
           if horizon and horizon > 1 else ""),
        "", "`train.py` は出発点なので, 課題に合わせて書き足す. 問題の設定を"
        "変えたら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text


def project_error(problem: dict) -> Optional[str]:
    """
    骨組みを書き出せない理由を, make_project が出す誤りの文で返す関数

    project_check の文だけ

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 も)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (書き出せるなら None. 書き出し先のファイルは見ない)
    """
    # 骨組みを組めない理由の文だけ
    message = project_check(problem)[0]

    return message


def project_check(problem: dict) -> tuple[Optional[str], list[dict]]:
    """
    骨組みを書き出せない理由と, 残っている組める確かめの error を返す関数

    受け渡しの値を作れるか (骨組みを組めない確かめの error が無いか),
    書き出せる受け持ちと正解の種類か, 指標がその他でないか, 入れたデータ
    セットが受け持ちに合うか, の順に確かめる. 「それでも書き出す」と指定
    すれば書き出せる error (open_errors) は含めない

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 も)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (書き出せるなら None. 書き出し先のファイルは見ない)
    exportable: list[dict]
        骨組みを組める確かめの error (check は 1 回だけ求める)
    """
    # 受け渡しの値を作れるか (機械学習を使い, 骨組みを組めない確かめの
    # error が無い. 組める error は「それでも書き出す」と指定すれば書き出
    # せる). 渡せない理由は ValueError の文のまま
    message: Optional[str] = None
    exportable: list[dict] = []
    try:
        upgraded = upgrade_problem(problem)
        exportable = export_stages(upgraded)[1]
    except ValueError as error:
        message = str(error)
        return message, exportable

    # 既製のモデルをそのまま使う (学習しない) のは, 予測するの画像の分類
    # と, まとめるの写真の群分けだけ (学習しないので, 指標と入れたデータ
    # セットの確かめは無し)
    if active_value(upgraded, f"{PART}.method") == "pretrained":
        message = (ready_group_message(upgraded)
                   if is_ready_group_part(upgraded)
                   else ready_message(upgraded))
        return message, exportable

    # 列の予測と, 予測するの画像の領域と点群と異常と, 作るの画像と表と,
    # 並べると, まとめると, 効果を見積もるの方策と今ある記録からの効果の
    # 見積もりだけ
    generate = is_generate_part(upgraded)
    policy = is_policy_part(upgraded)
    rank = is_rank_part(upgraded)
    group = is_group_part(upgraded)
    anomaly = is_anomaly_part(upgraded)
    uplift = is_uplift_part(upgraded)
    price = is_price_part(upgraded)
    if not is_column_part(upgraded) and skeleton_kind(upgraded) is None and (
        not generate and not policy and not rank and not group
        and not anomaly and not uplift and not price
    ):
        message = ("骨組みを書き出せるのは, 予測するで正解の種類が表の列, "
                   "画像の領域, 点群, 異常の場合と, 作るで正解の種類が画像か表の場合"
                   "と, 並べるとまとめるの場合と, 効果を見積もるで続けて行動を"
                   "選ぶ方策か今ある記録から効果を見積もるか値段を選ぶ場合と, "
                   "予測するで"
                   "既製のモデルをそのまま使う画像の分類の場合だけです (方策で"
                   "なく試して集める効果の見積もりと, 他の正解の種類はまだ扱え"
                   "ません)")
        return message, exportable
    if generate or policy or rank or group or anomaly or uplift or price:
        message = (policy_message(upgraded) if policy
                   else rank_message(upgraded) if rank
                   else group_message(upgraded) if group
                   else anomaly_message(upgraded) if anomaly
                   else uplift_message(upgraded) if uplift
                   else price_message(upgraded) if price
                   else table_message(upgraded)
                   if active_value(upgraded, f"{PART}.target.kind") == "table"
                   else generate_message(upgraded))
        if message is not None:
            return message, exportable

    # 骨組みが記録できる指標だけ
    if active_value(upgraded, f"{PART}.evaluation.metric") == "other":
        message = ("最良を選ぶ指標がその他の場合は, 骨組みを書き出せません "
                   "(学習のスクリプトがその指標の求め方を知らないため). 候補"
                   "の指標 (「検証用の分割での損失」など) で書き出し, その"
                   "指標は学習のスクリプトに足して記録してください")
        return message, exportable

    # 入れたデータセットの版, 受け持ち, 正解の種類
    message = items_message(upgraded)

    return message, exportable


def generate_message(problem: dict) -> Optional[str]:
    """
    画像の生成の骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 作るで正解の種類が画像)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が画像でない, 指標が損失か FID でない, 入れた
        データが画像のディレクトリでない. 書き出せるなら None)
    """
    # 条件の無い拡散のモデルで, 画像を作り, 損失か FID で選び, 画像を読む
    output = active_value(problem, f"{PART}.output")
    metric = active_value(problem, f"{PART}.evaluation.metric")
    loader = active_value(problem, "data.loader")
    source = get_value(problem, "data.source")
    loss = choice_label(METRICS, "loss", problem)
    fid = choice_label(METRICS, "fid", problem)
    message = None
    if output != "image":
        message = ("画像の生成の骨組みは, 出力の形が「画像」の場合だけです (今"
                   f"は{choice_label(OUTPUTS, output, problem)})")
    elif metric not in ("loss", "fid", "other"):
        message = (f"画像の生成の骨組みは, 最良を選ぶ指標が{loss}か{fid}の"
                   "場合だけです (今は"
                   f"{choice_label(METRICS, metric, problem)})")
    elif source and loader != "image_folder":
        message = ("画像の生成の骨組みは, 画像のディレクトリのデータで学びます"
                   f" (入れたデータ {Path(str(source)).name} は画像のディレクト"
                   "リとして読んでいません). 画像を並べたディレクトリを入れて"
                   "ください")

    return message


def table_message(problem: dict) -> Optional[str]:
    """
    表の生成の骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 作るで正解の種類が表)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が表でない, 指標が損失か列ごとの分布の近さで
        ない, 入れたデータが表でない. 書き出せるなら None)
    """
    # 条件の無い拡散のモデルで, 表を作り, 損失か分布の近さで選び, 表を読む
    output = active_value(problem, f"{PART}.output")
    metric = active_value(problem, f"{PART}.evaluation.metric")
    loader = active_value(problem, "data.loader")
    options = get_value(problem, "data.options")
    source = get_value(problem, "data.source")
    loss = choice_label(METRICS, "loss", problem)
    distance = choice_label(METRICS, "column_distance", problem)
    text = isinstance(options, dict) and options.get("text_column")
    message = None
    if output != "table":
        message = ("表の生成の骨組みは, 出力の形が「表 (合成のデータ)」の場合"
                   f"だけです (今は{choice_label(OUTPUTS, output, problem)})")
    elif metric not in ("loss", "column_distance", "other"):
        message = (f"表の生成の骨組みは, 最良を選ぶ指標が{loss}か{distance}の"
                   "場合だけです (今は"
                   f"{choice_label(METRICS, metric, problem)})")
    elif source and (loader not in TABLE_LOADERS or text):
        message = ("表の生成の骨組みは, 表のデータで学びます (入れたデータ "
                   f"{Path(str(source)).name} は表として読んでいません). "
                   "表のファイル (CSV, Parquet, Excel など) を入れてください")

    return message


def rank_message(problem: dict) -> Optional[str]:
    """
    並べるの骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 並べるで学習する)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が順位でない, 入れたデータが表でない. 書き出せる
        なら None)
    """
    # 2 塔のモデルで順位を出し, 反応の表を読む. 指標は受け持ちの候補
    # (Recall@K, NDCG, 損失) が全て書ける
    output = active_value(problem, f"{PART}.output")
    loader = active_value(problem, "data.loader")
    options = get_value(problem, "data.options")
    source = get_value(problem, "data.source")
    text = isinstance(options, dict) and options.get("text_column")
    message = None
    if output != "ranking":
        message = ("並べるの骨組みは, 出力の形が「順位」の場合だけです (今"
                   f"は{choice_label(OUTPUTS, output, problem)})")
    elif source and (loader not in TABLE_LOADERS or text):
        message = ("並べるの骨組みは, 反応の表で学びます (入れたデータ "
                   f"{Path(str(source)).name} は表として読んでいません). "
                   "利用者, 品物, 時刻の列を持つ表のファイル (CSV, Parquet "
                   "など) を入れてください")

    return message


def group_message(problem: dict) -> Optional[str]:
    """
    まとめるの骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. まとめるで学習する)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が群でない, 入れたデータが表でない. 書き出せる
        なら None)
    """
    # 1 行が 1 つの対象の表を読む. 指標は受け持ちの候補 (シルエット係数,
    # 損失) が全て書ける
    output = active_value(problem, f"{PART}.output")
    loader = active_value(problem, "data.loader")
    options = get_value(problem, "data.options")
    source = get_value(problem, "data.source")
    text = isinstance(options, dict) and options.get("text_column")
    message = None
    if output != "cluster":
        message = ("まとめるの骨組みは, 出力の形が「群」の場合だけです (今"
                   f"は{choice_label(OUTPUTS, output, problem)})")
    elif source and (loader not in TABLE_LOADERS or text):
        message = ("まとめるの骨組みは, 1 行が 1 つの対象の表で群を作ります "
                   f"(入れたデータ {Path(str(source)).name} は表として読んで"
                   "いません). 群を作る列を持つ表のファイル (CSV, Parquet "
                   "など) を入れてください")

    return message


def anomaly_message(problem: dict) -> Optional[str]:
    """
    異常の検知の骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 異常の検知)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が値でない, 入れたデータが表でない. 書き出せる
        なら None)
    """
    # 異常の検知の表を読む. 指標は受け持ちの候補が全て書ける
    output = active_value(problem, f"{PART}.output")
    loader = active_value(problem, "data.loader")
    options = get_value(problem, "data.options")
    source = get_value(problem, "data.source")
    text = isinstance(options, dict) and options.get("text_column")
    message = None
    if output != "value":
        message = ("異常の検知の骨組みは, 出力の形が「値」(行ごとの異常の"
                   "度合い) の場合だけです (今は"
                   f"{choice_label(OUTPUTS, output, problem)})")
    elif source and (loader not in TABLE_LOADERS or text):
        message = ("異常の検知の骨組みは, 正常の行が多い表で度合いを作ります "
                   f"(入れたデータ {Path(str(source)).name} は表として読んで"
                   "いません). 度合いに使う列を持つ表のファイル (CSV, Parquet "
                   "など) を入れてください")

    return message


def uplift_message(problem: dict) -> Optional[str]:
    """
    効果の見積もりの骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 効果の見積もり)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が行動ごとの差でない, 入れたデータが表でない.
        書き出せるなら None)
    """
    # 介入の記録の表 (指標は Qini の係数か損失. 方策の報酬は確かめの
    # error)
    output = active_value(problem, f"{PART}.output")
    loader = active_value(problem, "data.loader")
    options = get_value(problem, "data.options")
    source = get_value(problem, "data.source")
    text = isinstance(options, dict) and options.get("text_column")
    message = None
    if output != "effect":
        message = ("効果の見積もりの骨組みは, 出力の形が「行動ごとの差」の"
                   "場合だけです (今は"
                   f"{choice_label(OUTPUTS, output, problem)})")
    elif source and (loader not in TABLE_LOADERS or text):
        message = ("効果の見積もりの骨組みは, 過去に行動した記録の表で効果を"
                   f"見積もります (入れたデータ {Path(str(source)).name} は表"
                   "として読んでいません). 処置の列と結果の列を持つ表のファイル"
                   " (CSV, Parquet など) を入れてください")

    return message


def price_message(problem: dict) -> Optional[str]:
    """
    値段を変える判断の骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 値段を変える判断)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が行動ごとの差でない, 入れたデータが表でない.
        書き出せるなら None)
    """
    # 値段を変えた記録の表 (指標は売れた数の誤差か損失. Qini と方策の報酬は
    # 確かめの error)
    output = active_value(problem, f"{PART}.output")
    loader = active_value(problem, "data.loader")
    options = get_value(problem, "data.options")
    source = get_value(problem, "data.source")
    text = isinstance(options, dict) and options.get("text_column")
    message = None
    if output != "effect":
        message = ("値段を変える判断の骨組みは, 出力の形が「行動ごとの差」の"
                   "場合だけです (今は"
                   f"{choice_label(OUTPUTS, output, problem)})")
    elif source and (loader not in TABLE_LOADERS or text):
        message = ("値段を変える判断の骨組みは, 試しに値段を変えた売れ行きの"
                   f"記録の表で学びます (入れたデータ {Path(str(source)).name}"
                   " は表として読んでいません). 値段の列と売れた数の列を持つ表"
                   "のファイル (CSV, Parquet など) を入れてください")

    return message


def policy_message(problem: dict) -> Optional[str]:
    """
    方策の骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 効果を見積もるで方策)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (今ある記録から学ぶ, 指標が報酬でも損失でもない. 書き出せる
        なら None)
    """
    # 環境とやり取りして学び, 評価の報酬 (valid/reward. valid/loss はその
    # 符号の反転) で選ぶ
    collect = active_value(problem, f"{PART}.collect")
    metric = active_value(problem, f"{PART}.evaluation.metric")
    message = None
    if collect == "log":
        message = ("方策の骨組みは, 環境で行動を試して学びます. 今ある記録"
                   "から方策を学ぶ骨組みはまだありません (行動の結果の集め方"
                   "を, 実際に試すかシミュレータにしてください)")
    elif metric not in ("reward", "loss", "other"):
        message = (
            "方策の骨組みは, 最良を選ぶ指標が"
            f"{choice_label(METRICS, 'reward', problem)}か"
            f"{choice_label(METRICS, 'loss', problem)}の場合だけです (今は"
            f"{choice_label(METRICS, metric, problem)}. 方策の報酬は評価の"
            "エピソードの報酬の和の平均, 損失はその符号を反転した値です)"
        )

    return message


def open_errors(
    problem: dict, inspected: Optional[dict] = None
) -> list[dict]:
    """
    「それでも書き出す」と指定すれば書き出せる確かめの error を返す関数

    問題の設定の確かめ (check) の, 骨組みを組める error と, 入れたデータの
    確かめ (inspect_data の findings) の error

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 も)
    inspected: Optional[dict] = None
        画面が覚えておいた inspect_data の結果 (省くか, 今の問題の設定と
        データから求めたものでなければ求め直す)

    Returns
    ----------
    errors: list[dict]
        check と同じ形の確かめの並び (問題の設定の確かめの後に, 入れた
        データの確かめ)
    """
    # 問題の設定の確かめの組める error と, 入れたデータの確かめの error
    errors = split_errors(problem)[1] + data_errors(problem, inspected)

    return errors


def data_errors(problem: dict, inspected: Optional[dict]) -> list[dict]:
    """
    入れたデータの確かめ (inspect_data の findings) の error を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    inspected: Optional[dict]
        覚えておいた inspect_data の結果 (印 key が今のものと同じで,
        findings が辞書の並びのときだけ使う)

    Returns
    ----------
    errors: list[dict]
        確かめの並び (データを入れていなければ空)
    """
    # 古いか別の問題の設定の結果と, 形の崩れた結果の求め直し
    fresh = (
        isinstance(inspected, dict)
        and inspected.get("key") == inspect_key(problem)
        and isinstance(inspected.get("findings"), list)
        and all(isinstance(item, dict) for item in inspected["findings"])
    )
    found = inspected if fresh else inspect_data(problem)
    errors = [item for item in (found or {}).get("findings", [])
              if item.get("level") == "error"]

    return errors


def errors_message(errors: list[dict]) -> str:
    """
    残っている確かめの error で書き出さない理由の文を作る関数

    Parameters
    ----------
    errors: list[dict]
        確かめの error の並び (空でない)

    Returns
    ----------
    message: str
        誤りの文 (初めの 3 つと, 書き出すための手)
    """
    # 初めの 3 つと残りの数
    shown = " / ".join(item["message"] for item in errors[:3])
    rest = f" (ほか {len(errors) - 3} 件)" if len(errors) > 3 else ""
    message = (f"確かめに直すところが残っています: {shown}{rest}. 直して"
               "から書き出すか, 残ったまま書き出すなら accept_errors=True "
               "を渡してください (README に一覧を残す)")

    return message


def errors_section(
    errors: list[dict], dropped: bool = False, checked: bool = True
) -> str:
    """
    README の, 書き出した時点で残っていた直すところの節を作る関数

    Parameters
    ----------
    errors: list[dict]
        確かめの error の並び
    dropped: bool = False
        損得と制約に直すところが残り, 意思決定の既定値を付けなかったか
    checked: bool = True
        問題の設定の確かめ (check) の直すところが残るか (残れば推論の段階
        に渡す値を作れない. 入れたデータの確かめだけなら作れる)

    Returns
    ----------
    text: str
        Markdown の節 (残っていなければ空)
    """
    # 無ければ節は無し
    if not errors:
        return ""
    lines = ["", "## 書き出した時点で残っていた直すところ", "",
             "問題の設定と入れたデータの確かめで直すところ (error) が残った"
             "まま, 「それでも書き出す」と指定して書き出した. 学習が失敗したり"
             "結果がおかしかったりしたときは, まずこれを見る. 直したら書き出し"
             "直す. 直すまで, 推論の段階 (ML-Provide) で問題の設定につなげない"
             + (" (推論の段階に渡す値は, 問題の設定の確かめの直すところが残る"
                "と作れないため)." if checked else
                " (入れたデータの確かめの直すところは, 推論の段階に渡す値を作る"
                "ときには確かめないので, つなぐと同じ誤りのまま使われるため)."
                )]
    if dropped:
        lines.append("")
        lines.append("損得と制約 (`action.decision`) に直すところが残るので, "
                     "`train.py` は意思決定の既定値を付けずに推論の一式を作る "
                     "(`train.log` にもそう書く).")
    lines.append("")
    lines += [f"- `{item['id']}`: {' '.join(str(item['message']).split())}"
              for item in errors]
    text = "\n".join(lines) + "\n"

    return text


def skeleton_kind(problem: dict) -> Optional[str]:
    """
    ファイルの組の骨組みを書き出す正解の種類を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    kind: Optional[str]
        予測するで正解の種類が mask か point_cloud ならその種類, 他は None
    """
    # 予測するの受け持ちだけ
    kind = active_value(problem, f"{PART}.target.kind")
    if active_value(problem, f"{PART}.role") != "predict" or (
        kind not in ITEMS_KINDS
    ):
        kind = None

    return kind


def project_files(
    problem: dict, stages: dict, name: Optional[str],
    data_file: Optional[Path], info: Optional[dict], dropped: bool = False,
    spec_exists: bool = False
) -> dict:
    """
    受け持ちに合う README, config の基準, 学習のスクリプトを作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. データの場所はプロジェクトの写し)
    stages: dict
        受け渡し
    name: Optional[str]
        実験の名前
    data_file: Optional[Path]
        プロジェクトに写すデータ (プロジェクトからの相対. 無ければ None)
    info: Optional[dict]
        入れたデータセットの説明 (データセットでなければ None)
    dropped: bool = False
        損得と制約に直すところが残り, 意思決定の既定値を付けないか
    spec_exists: bool = False
        書き出し先に定義 (models/model.yaml) が既にあるか (まとめるの
        config に仮のモデルの目印を書かない)

    Returns
    ----------
    files: dict
        README.md, configs/default/train.yaml, train.py の中身 (既製の
        モデルは configs/default/bundle.yaml と bundle.py. 方策は model.py
        と env.py も)
    """
    # 既製のモデルの埋め込みで写真を群に分ける (学習せず, 群と推論の一式
    # を作る. 入れた写真は data に写してそのまま読む. 入れていなければ
    # 置き場の outputs/photos)
    if is_ready_group_part(problem):
        photos = data_file.as_posix() if data_file else READY_GROUP_PHOTOS
        files = {
            "README.md": ready_group_readme(
                problem, stages, photos,
                data_file.as_posix() if data_file else None
            ),
            READY_CONFIG: ready_group_config(stages, problem, name, photos),
            READY_SCRIPT: ready_group_script(problem, literal),
        }
        return files

    # 既製のモデルをそのまま使う (学習せず, 推論の一式だけを作る)
    if stages["inference"].get("ready") is not None:
        files = {
            "README.md": ready_readme(
                problem, stages, data_file.as_posix() if data_file else None
            ),
            READY_CONFIG: ready_config(stages, name),
            READY_SCRIPT: ready_script(problem, literal),
        }
        return files

    # 方策 (環境の登録名が無ければ自作の環境の雛形も)
    if is_policy_part(problem):
        files = {
            "README.md": policy_readme(problem, stages),
            "configs/default/train.yaml": policy_config(
                problem, experiment_name(problem, name), stages,
                requires_text(POLICY_LIBRARIES, stages)
            ),
            "train.py": policy_script(problem, literal),
            "model.py": POLICY_MODEL,
        }
        env = policy_env(stages)
        if env is not None:
            files["env.py"] = env
        return files

    # 並べる (入れた反応の表は data に写してそのまま読む. 入れていなければ
    # 置き場の outputs/interactions.csv)
    if is_rank_part(problem):
        dataset = data_file.as_posix() if data_file else RANK_DATASET
        files = {
            "README.md": rank_readme(
                problem, stages, dataset,
                data_file.as_posix() if data_file else None
            ),
            "configs/default/train.yaml": rank_config(
                experiment_name(problem, name), stages, dataset,
                requires_text(RANK_LIBRARIES, stages)
            ),
            "train.py": rank_script(problem, literal),
        }
        return files

    # まとめる (入れた表は data に写してそのまま読む. 入れていなければ
    # 置き場の outputs/rows.csv)
    if is_group_part(problem):
        dataset = data_file.as_posix() if data_file else GROUP_DATASET
        files = {
            "README.md": group_readme(
                problem, stages, dataset,
                data_file.as_posix() if data_file else None
            ),
            "configs/default/train.yaml": group_config(
                experiment_name(problem, name), stages, dataset,
                requires_text(GROUP_LIBRARIES, stages), spec_exists
            ),
            "train.py": group_script(problem, literal),
        }
        return files

    # 異常の検知 (入れた表は data に写し, 異常の検知の表はデータの分析の
    # 段階が outputs/dataset に書き出す)
    if is_anomaly_part(problem):
        score = stages["anomaly"]["score"]
        files = {
            "README.md": anomaly_readme(
                problem, stages, ANOMALY_DATASET,
                data_file.as_posix() if data_file else None
            ),
            "configs/default/train.yaml": anomaly_config(
                experiment_name(problem, name), stages, ANOMALY_DATASET,
                requires_text(ANOMALY_LIBRARIES[score], stages), spec_exists,
                data_file.as_posix() if data_file else None
            ),
            "train.py": anomaly_script(problem, literal),
        }
        return files

    # 効果の見積もり (入れた表は data に写し, 介入の記録の表はデータの分析の
    # 段階が outputs/dataset に書き出す)
    if is_uplift_part(problem):
        files = {
            "README.md": uplift_readme(
                problem, stages, UPLIFT_DATASET,
                data_file.as_posix() if data_file else None
            ),
            "configs/default/train.yaml": uplift_config(
                experiment_name(problem, name), stages, UPLIFT_DATASET,
                requires_text(UPLIFT_LIBRARIES, stages), spec_exists,
                data_file.as_posix() if data_file else None
            ),
            "train.py": uplift_script(problem, literal),
        }
        return files

    # 値段を変える判断 (入れた表は data に写し, 値段を変えた記録の表は
    # データの分析の段階が outputs/dataset に書き出す)
    if is_price_part(problem):
        files = {
            "README.md": price_readme(
                problem, stages, PRICE_DATASET,
                data_file.as_posix() if data_file else None
            ),
            "configs/default/train.yaml": price_config(
                experiment_name(problem, name), stages, PRICE_DATASET,
                requires_text(PRICE_LIBRARIES, stages), spec_exists,
                data_file.as_posix() if data_file else None
            ),
            "train.py": price_script(problem, literal),
        }
        return files

    # 表の生成 (入れた表は data に写し, データセットはデータの分析の段階が
    # outputs/dataset に書き出す)
    table = active_value(problem, f"{PART}.target.kind") == "table"
    if is_generate_part(problem) and table:
        files = {
            "README.md": table_readme(
                problem, stages, "outputs/dataset",
                data_file.as_posix() if data_file else None
            ),
            "configs/default/train.yaml": table_config(
                experiment_name(problem, name), stages, "outputs/dataset",
                requires_text(TABLE_LIBRARIES, stages)
            ),
            "train.py": table_script(problem, literal),
        }
        return files

    # 画像の生成 (入れた画像のディレクトリは data に写し, データセットは
    # データの分析の段階が outputs/dataset に書き出す)
    if is_generate_part(problem):
        files = {
            "README.md": generate_readme(
                problem, stages, "outputs/dataset",
                data_file.as_posix() if data_file else None
            ),
            "configs/default/train.yaml": generate_config(
                experiment_name(problem, name), stages, "outputs/dataset",
                requires_text(GENERATE_LIBRARIES + (
                    FID_LIBRARIES
                    if stages["training"]["monitor"] == FID_MONITOR else []
                ), stages)
            ),
            "train.py": generate_script(problem, literal),
        }
        return files

    # 列の予測
    kind = skeleton_kind(problem)
    if kind is None:
        files = {
            "README.md": project_readme(problem, stages, data_file,
                                        dropped),
            "configs/default/train.yaml": config_text(problem, stages, name),
            "train.py": training_script(problem, stages, data_file,
                                        dropped),
        }
        return files

    # ファイルの組 (入れたデータがデータセットなら, その写しを読む)
    dataset = dataset_place(info, data_file)
    raw = data_file.as_posix() if data_file and info is None else None
    files = {
        "README.md": items_readme(problem, stages, kind, dataset, info, raw),
        "configs/default/train.yaml": items_config(
            experiment_name(problem, name), stages, kind, dataset,
            requires_text(ITEMS_LIBRARIES, stages)
        ),
        "train.py": items_script(problem, kind, literal, comment_lines),
    }

    return files


def origin_file(origin: Union[str, Path], path: Path) -> str:
    """
    元の問題の設定の場所を, problem.yaml の絶対パスにする関数

    Parameters
    ----------
    origin: Union[str, Path]
        元の problem.yaml か, それを持つディレクトリ
    path: Path
        書き出すプロジェクトのディレクトリ

    Returns
    ----------
    place: str
        元の problem.yaml の絶対パス

    Raises
    ----------
    ValueError
        空の場合と, 書き出すプロジェクトの problem.yaml そのものの場合
    """
    # ディレクトリ (YAML のファイルの名前でない場所) なら中の
    # problem.yaml. 写しが自分を指さないように
    if not str(origin).strip():
        raise ValueError("元の問題の設定の場所が空です")
    target = Path(origin).expanduser().resolve()
    if (target.is_dir() or target == path.resolve()
            or target.suffix.lower() not in (".yaml", ".yml")):
        target = target / "problem.yaml"
    if target == (path / "problem.yaml").resolve():
        raise ValueError(
            f"{target} は書き出すプロジェクトの写しです. 元の問題の設定の"
            "場所を渡してください"
        )
    place = str(target)

    return place


def make_project(
    problem: dict, path: Union[str, Path], name: Optional[str] = None,
    overwrite: bool = False, origin: Optional[Union[str, Path]] = None,
    accept_errors: bool = False, inspected: Optional[dict] = None
) -> list[str]:
    """
    問題の設定から, ML-Found のプロジェクトの骨組みを書き出す関数

    Parameters
    ----------
    problem: dict
        問題の設定
    path: Union[str, Path]
        プロジェクトのディレクトリ
    name: Optional[str] = None
        実験の名前 (省略すると目的の列とタスクから. 既製のモデルをそのまま
        使う場合は一式の名前で, 省略するとモデルと重みの名前から)
    overwrite: bool = False
        既にあるファイルを上書きするか
    origin: Optional[Union[str, Path]] = None
        元の問題の設定 (problem.yaml か, それを持つディレクトリ). 写しの
        origin に絶対パスで残す (省略すると, 渡した問題の設定の origin を
        そのまま残す)
    accept_errors: bool = False
        「それでも書き出す」. 書き出せる確かめの error (open_errors) が
        残ったまま書き出すか (書き出すなら README に一覧を残す)
    inspected: Optional[dict] = None
        同じ問題の設定の inspect_data の結果 (省くと求める)

    Returns
    ----------
    written: list[str]
        書いたファイル (プロジェクトからの相対パス)

    Raises
    ----------
    ValueError
        渡せない問題の設定の場合, 書き出せない受け持ちか指標がその他の
        場合, 入れたデータセットが受け持ちに合わない場合, 確かめの error
        (open_errors) が残り accept_errors でない場合, 上書きしない既に
        あるファイルがある場合と, 元の場所が書き出す写しそのものの場合
    """
    # 書き出せるか (project_error と同じ決まり). 受け渡しの値 (版 1 は先に
    # 版 2 へ移行). データはプロジェクトの data に写し, 写した問題の設定は
    # その場所を指す. プロジェクトの記録は写しに持たせない (プロジェクトを
    # 動かしても古い場所を指さないように). 元の問題の設定の場所は写しの
    # origin に (推論の段階が元の今の値を読めるように)
    error, exportable = project_check(problem)
    if error is not None:
        raise ValueError(error)

    # 書き出せる確かめの error は「それでも書き出す」のときだけ残したまま.
    # 損得と制約の error が残れば, 推論の一式の既定値は無し (学習と一式が
    # 最後まで通るように)
    errors = exportable + data_errors(problem, inspected)
    if errors and not accept_errors:
        raise ValueError(errors_message(errors))
    problem = upgrade_problem(problem)
    if "project" in problem:
        problem = {key: value for key, value in problem.items()
                   if key != "project"}
    stages = build_stages(problem)
    dropped = any(item["id"] == "action.decision"
                  or str(item["id"]).startswith("action.decision.")
                  for item in errors)
    if dropped:
        stages = copy.deepcopy(stages)
        stages["inference"]["decision"] = None
    path = Path(path)
    if origin is not None:
        problem = {**problem, "origin": origin_file(origin, path)}
    source = data_location(problem)
    data_file = (Path("data") / Path(source).name
                 if isinstance(source, str) and Path(source).exists()
                 else None)
    info = (dataset_info(source)
            if data_file is not None and skeleton_kind(problem) else None)
    if data_file is not None:
        problem = copy.deepcopy(problem)
        set_value(problem, "data.origin",
                  get_value(problem, "data.origin") or source)
        set_value(problem, "data.source", str((path / data_file).resolve()))
    files = {
        "problem.yaml": yaml.safe_dump(problem, allow_unicode=True,
                                       sort_keys=False),
        **project_files(problem, stages, name, data_file, info, dropped,
                        (path / "models" / "model.yaml").is_file()),
    }
    if stages["inference"].get("ready") is None:
        files["configs/experiment/train.yaml"] = EXPERIMENT
    files["README.md"] += errors_section(errors, dropped, bool(exportable))

    # 既にあるファイルと, ディレクトリを作れない場所の確かめ (全てを確かめ
    # てから書く. 途中で止まって一部だけが残らないように)
    names = [*files, *([str(data_file)] if data_file else [])]
    existing = [name for name in names if (path / name).exists()]
    if existing and not overwrite:
        raise ValueError(
            f"{', '.join(existing)} が既にあります (上書きするなら overwrite)"
        )
    blocked = sorted({
        str(parent) for name in [*names, "models/x"]
        for parent in [path, *(path / name).parents]
        if parent.exists() and not parent.is_dir()
    } | {str(path / name) for name in files if (path / name).is_dir()})
    if blocked:
        raise ValueError(
            f"{', '.join(blocked)} がディレクトリかファイルとして邪魔をして"
            "いるので書き出せません"
        )
    for relative, text in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (path / "models").mkdir(exist_ok=True)
    written = list(files)

    # データを入れていれば, プロジェクトの data への写し (確かめの後)
    if data_file is not None and isinstance(source, str):
        copy_data(Path(source), (path / "data").resolve())
        written.append(str(data_file))

    return written


def project_place(path: Path, value: Any) -> Optional[str]:
    """
    config に書いた場所を, プロジェクトのディレクトリから絶対パスにする関数

    Parameters
    ----------
    path: Path
        プロジェクトのディレクトリ (絶対パス)
    value: Any
        config の値 (相対パスか絶対パス)

    Returns
    ----------
    place: Optional[str]
        絶対パス (空か文字でなければ None)
    """
    # train.py と同じく, プロジェクトのディレクトリからの場所
    if not isinstance(value, str) or not value.strip():
        return None
    place = str((path / value).resolve())

    return place


def record_project(problem: dict, path: Union[str, Path]) -> dict:
    """
    書き出したプロジェクトの場所を, 問題の設定に記録する関数

    元の問題の設定は変えない. 場所は書き出した config から読む (train.py
    がデータセットとモデルの定義を読む場所と同じにするため)

    Parameters
    ----------
    problem: dict
        問題の設定
    path: Union[str, Path]
        make_project で書き出したプロジェクトのディレクトリ

    Returns
    ----------
    updated: dict
        project (path, dataset, model_spec. どれも絶対パス. モデルの定義を
        config に書いていなければ model_spec は None) を入れた問題の設定

    Raises
    ----------
    ValueError
        config が無いか読めない場合
    """
    # 書き出した config の data.dataset と model.spec (既製のモデルの
    # 一式づくりの設定はどちらも持たない)
    place = Path(path).expanduser().resolve()
    config = place / "configs" / "default" / "train.yaml"
    if not config.exists() and (place / READY_CONFIG).is_file():
        config = place / READY_CONFIG
    try:
        loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"{config} を読めません. 先に make_project で書き出してください"
            f" ({error})"
        ) from error
    loaded = loaded if isinstance(loaded, dict) else {}
    data = loaded.get("data") if isinstance(loaded.get("data"), dict) else {}
    model = (loaded.get("model") if isinstance(loaded.get("model"), dict)
             else {})

    # 版 1 は先に版 2 へ移行. 前の記録は置き換え
    updated = copy.deepcopy(upgrade_problem(problem))
    updated["project"] = {
        "path": str(place),
        "dataset": project_place(place, data.get("dataset")),
        "model_spec": project_place(place, model.get("spec")),
    }

    return updated
