import copy
import json
from pathlib import Path
import re
from string import Template
import textwrap
from typing import Any, Optional, Union

import yaml

from ml_quest.baseline_skeleton import (
    EVALUATION_PREFIX, baseline_code, baseline_note, baseline_reason
)
from ml_quest.check import is_column_part
from ml_quest.data import (
    ITEMS_KINDS, content_kind, copy_data, dataset_info, items_message
)
from ml_quest.handoff import describe, handoff
from ml_quest.items_skeleton import (
    ITEMS_LIBRARIES, dataset_place, items_config, items_name, items_readme,
    items_script
)
from ml_quest.problem import (
    active_value, data_location, get_value, set_value, upgrade_problem
)
from ml_quest.questions import PART

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は, コードを壊さないように JSON の文字列か 1 行のコメントで埋め込み
HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

骨組みなので, 課題に合わせて書き足す
"""
from pathlib import Path
import shutil
import sys
$pandas
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_eda import $eda_names
from ml_found import (
    $found_names
)
from ml_inference import make_bundle
from ml_modelcontrol import build_model

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
        experiment.logger.info("テスト用の分割が無いので, 評価しません")
        return
    model = model.cpu().eval()
    x, y = dataset.tensors("test")
    with torch.no_grad():
        pred = model(x).squeeze(-1) * scale + center
    error = pred - y
    value = $test_expression
    experiment.logger.info(f"テスト用の$test_label: {value.item():,.4f}")
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
        experiment.logger.info("テスト用の分割が無いので, 評価しません")
        return
    model = model.cpu().eval()
    x, y = dataset.tensors("test")
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == y).float().mean()
    experiment.logger.info(f"テスト用の正解率: {accuracy.item():.4f}")
$baseline
'''

# 学習の流れ (データセット, モデル, 学習, 評価, 推論の一式)
MAIN = '''

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
    spec = PROJECT_DIR / cfg.model.spec
    if not spec.is_file():
        print(f"{spec} がありません. README の「次にすること」の 2 で, "
              "モデルの定義を置いてください")
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

    # モデルの定義からの組み立てと学習
    model = build_model(spec, seed=cfg.experiment.seed)
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. 予測すること: {PREDICT}. $inputs_log"
    )
    trainer = Trainer(model, experiment, Step(scale))
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定とモデルの定義の写し (元の問題の
    # 設定の場所 origin も残る)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    shutil.copy(spec, run_dir / "model_spec.yaml")

    # 結果の表示とテスト用の評価
    best_epoch = trainer.checkpoint.best_epoch
    best_value = trainer.checkpoint.best_value
    print(f"保存先: {run_dir}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの {cfg.trainer.monitor}: {best_value}")
    evaluate_test(trainer.model, dataset, experiment, center, scale)

    # 実験の中に推論の一式 (問題の設定の意思決定と結論を既定値に)
    info = make_bundle(
        run_dir / "inference", spec, path, run_dir,
        target_scale=$target_scale, conclusion=CONCLUSION,
        decision=DECISION, name=cfg.experiment.name
    )
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
$compare

if __name__ == "__main__":
    main()
'''

# 学習の記録の入力の文 (表と時系列は学習に使う列, 画像と文は 1 件の形)
INPUTS_LOG = {
    None: ('学習に使う列 "\n        f"{len(dataset.features)}: '
           '{dataset.features}'),
    "image": ('1 件の画像の形 (チャンネル, 高さ, 幅): "\n        '
              'f"{tuple(x_train.shape[1:])}'),
    "text": ('1 件の文の形 (語の番号の長さ): "\n        '
             'f"{tuple(x_train.shape[1:])}'),
}

# README のデータの分析の, 画像と文で学ぶもの (表の列でなく 1 件の中身)
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
}

# README のモデルの作成の, 入力の形の案内 (データの種類ごと. None は表と
# 時系列か, データを入れていない場合)
CONTENT_INPUTS = {
    None: ("入力の形は書き出したデータセットから決まる (ML-EDA の "
           "`model_inputs(<データセット>)` の `inputs` の `shape`). 表と時系列"
           "では学習に使う列の数 (`dataset.yaml` の features. ワンホットや"
           "過去の値の列で元の列より増える), 画像, 文, 点群の列では 1 件の"
           "配列の形 (`dataset.yaml` の splits の `shape` の 2 つ目から. 列の"
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
        辞書, 文字列か None

    Returns
    ----------
    text: str
        式 (二重引用符. 長い辞書は 1 項目ずつ折り返す)
    """
    # JSON の文字列と辞書 (Python の式としても有効). 長い文字列は括弧の中で
    # つなぐ形に折り返し
    if value is None:
        return "None"
    text = json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict) and len(text) > 60:
        text = json.dumps(value, ensure_ascii=False, indent=4)
    elif isinstance(value, str) and len(text) > 60:
        pieces = [json.dumps(value[i:i + 30], ensure_ascii=False)
                  for i in range(0, len(value), 30)]
        text = "(\n" + "\n".join(f"    {piece}" for piece in pieces) + "\n)"

    return text


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
    problem: dict, stages: dict, data_file: Optional[Path] = None
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
    if reason is None and data_file is not None:
        baseline = baseline_code(problem, stages, data_file, literal)
        compare = ("\n    # 今のやり方との比べ (推論の一式の後. 比べられなくても"
                   "止めない)\n    compare_baseline(trainer.model, dataset, "
                   "experiment, center, scale)\n")
    else:
        baseline = baseline_note(problem, str(reason), comment_lines,
                                 retry=kind is None)
        compare = ""

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
                     "save_evaluation,\n    start_experiment" if compare
                     else "Experiment, Trainer, load_project_config, "
                     "start_experiment"),
    )

    # タスクごとの step と評価
    if task == "classification":
        body = Template(CLASSIFICATION_STEP).substitute(baseline=baseline)
        scaling = "分類は番号のまま"
        scale_lines = "    center, scale = 0.0, 1.0"
        train_target, valid_target = "y_train", "y_valid"
        target_scale = "None"
    else:
        key, expression, test_expression, label = REGRESSION_METRICS[metric]
        body = Template(REGRESSION_STEP).substitute(
            key=key, expression=expression, metric_label=label
            if key == metric else "二乗誤差", test_expression=test_expression,
            test_label=label, baseline=baseline
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
        inputs_log=INPUTS_LOG[kind],
    )
    script = head + body + tail

    return script


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
    problem: dict, stages: dict, data_file: Optional[Path] = None
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
    compare = (
        "今のやり方との比べを, 検証用とテスト用の評価の表 "
        f"(`evaluations/{EVALUATION_PREFIX}valid.json` と "
        f"`{EVALUATION_PREFIX}test.json`. 元のデータの同じ行が分かる行だけで"
        "求める) に書く" if reason is None else
        f"今のやり方とは比べない ({reason} ため)"
    )
    lines += [
        "2. **モデルの作成** (ML-CreateArchitecture): 入力 1 つと出力 1 つの"
        f"モデルを組み, `models/model.yaml` に置く. {CONTENT_INPUTS[kind]}. "
        f"出力の大きさは {outputs}. 回帰の目的の値は train.py が標準化する",
        "3. **学習** (ML-Found, ML-Dashboard): `python train.py` "
        "(`python train.py trainer.epochs=10` のように「項目=値」で上書き"
        "できる). 結果は `outputs/<実験の名前>/<日付>/<時刻>/` に残る. "
        f"検証用の `{stages['training']['monitor']}` が最も良いモデルを選ぶ. "
        + compare,
        "4. **推論** (ML-Provide): 実験の中の `inference/` を登録する. 意思"
        "決定の既定値は問題の設定の損得と制約"
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

    受け渡し, 書き出せる受け持ちと正解の種類か, 指標がその他でないか,
    入れたデータセットが受け持ちに合うか, の順に確かめる

    Parameters
    ----------
    problem: dict
        問題の設定 (版 1 も)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (書き出せるなら None. 書き出し先のファイルは見ない)
    """
    # 受け渡し (渡せない理由は ValueError の文のまま)
    message: Optional[str] = None
    try:
        upgraded = upgrade_problem(problem)
        handoff(upgraded)
    except ValueError as error:
        message = str(error)
        return message

    # 列の予測と, 予測するの画像の領域と点群だけ
    if not is_column_part(upgraded) and skeleton_kind(upgraded) is None:
        message = ("骨組みを書き出せるのは, 予測するで正解の種類が表の列, "
                   "画像の領域, 点群の場合だけです (作る, 並べる, まとめる, "
                   "効果を見積もるの受け持ちと, 他の正解の種類はまだ扱えま"
                   "せん)")
        return message

    # 骨組みが記録できる指標だけ
    if active_value(upgraded, f"{PART}.evaluation.metric") == "other":
        message = ("最良を選ぶ指標がその他の場合は, 骨組みを書き出せません "
                   "(学習のスクリプトがその指標の求め方を知らないため). 候補"
                   "の指標 (loss など) で書き出し, その指標は学習のスクリプト"
                   "に足して記録してください")
        return message

    # 入れたデータセットの版, 受け持ち, 正解の種類
    message = items_message(upgraded)

    return message


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
    data_file: Optional[Path], info: Optional[dict]
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

    Returns
    ----------
    files: dict
        README.md, configs/default/train.yaml, train.py の中身
    """
    # 列の予測
    kind = skeleton_kind(problem)
    if kind is None:
        files = {
            "README.md": project_readme(problem, stages, data_file),
            "configs/default/train.yaml": config_text(problem, stages, name),
            "train.py": training_script(problem, stages, data_file),
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
    overwrite: bool = False, origin: Optional[Union[str, Path]] = None
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
        実験の名前 (省略すると目的の列とタスクから)
    overwrite: bool = False
        既にあるファイルを上書きするか
    origin: Optional[Union[str, Path]] = None
        元の問題の設定 (problem.yaml か, それを持つディレクトリ). 写しの
        origin に絶対パスで残す (省略すると, 渡した問題の設定の origin を
        そのまま残す)

    Returns
    ----------
    written: list[str]
        書いたファイル (プロジェクトからの相対パス)

    Raises
    ----------
    ValueError
        渡せない問題の設定の場合, 書き出せない受け持ちか指標がその他の
        場合, 入れたデータセットが受け持ちに合わない場合, 上書きしない
        既にあるファイルがある場合と, 元の場所が書き出す写しそのものの
        場合
    """
    # 書き出せるか (project_error と同じ決まり). 受け渡しの値 (版 1 は先に
    # 版 2 へ移行). データはプロジェクトの data に写し, 写した問題の設定は
    # その場所を指す. プロジェクトの記録は写しに持たせない (プロジェクトを
    # 動かしても古い場所を指さないように). 元の問題の設定の場所は写しの
    # origin に (推論の段階が元の今の値を読めるように)
    error = project_error(problem)
    if error is not None:
        raise ValueError(error)
    problem = upgrade_problem(problem)
    if "project" in problem:
        problem = {key: value for key, value in problem.items()
                   if key != "project"}
    stages = handoff(problem)
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
        **project_files(problem, stages, name, data_file, info),
        "configs/experiment/train.yaml": EXPERIMENT,
    }

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
    # 書き出した config の data.dataset と model.spec
    place = Path(path).expanduser().resolve()
    config = place / "configs" / "default" / "train.yaml"
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
