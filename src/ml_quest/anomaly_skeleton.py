import json
from string import Template
from typing import Callable, Optional

from ml_quest.handoff import describe
from ml_quest.problem import active_value, get_value
from ml_quest.questions import ANOMALY_ASSIGNED, ANOMALY_SCORES, PART

# 学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の機能の名前
# (データセットは ML-EDA で読む. 自己符号化器は config の model.spec の
# 定義を置くと ML-ModelControl の読み込み. 推論の一式は ML-Inference)
ANOMALY_LIBRARIES = {"autoencoder": ["eda", "modelcontrol", "inference"],
                     "distance": ["eda", "inference"]}

# データの分析の段階が書き出す, 異常の検知の表の場所 (プロジェクトの
# ディレクトリから)
ANOMALY_DATASET = "outputs/dataset"

# 距離に基づく度合いの, 共分散の対角に足す値の既定 (config の
# anomaly.ridge)
ANOMALY_RIDGE = 0.001

# spec のファイルが無いときの, 仮のモデルで学習することの目印 (config の
# model.placeholder)
ANOMALY_PLACEHOLDER = """  # spec のファイルが無いうちは train.py の仮のモデルで学習することの目印
  # (ML-Dashboard の学習の準備が読む. train.py は読まず, 仮のモデルを使うか
  # は spec のファイルがあるかで決まる)
  placeholder: true
"""

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
ANOMALY_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

正常の行が多い表 (ML-EDA の異常の検知の表) の学習用の正常の行から, 自己
符号化器の再構成の誤差か距離に基づく度合いを作り, 行ごとの異常の度合いと
その評価, 推論の段階が読む成果物を残す骨組みなので, 印 (TODO) の所を書き
足す
"""
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from ml_eda import Dataset, load_dataset
from ml_found import (
    Experiment, Trainer, average_precision, load_project_config,
    precision_at_k, save_evaluation, start_experiment, write_task
)

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 度合いを出すもの
PROBLEM_NAME = $name
PREDICT = $predict

# task.yaml の受け持ち, 出力の形と作り方 (問題の設定の答え)
ROLE = "predict"
OUTPUT = "value"
METHOD = $method

# 元の表の読み方 (問題の設定の手元のデータの読み方. 推論の一式づくりが
# ML-EDA の load_data に渡す. None は ML-EDA の推定)
LOADER = $loader
OPTIONS = $options

# 推論の一式を作る場所 (実験のディレクトリの中)
INFERENCE_DIR = "inference"

# 度合いの決め方の名前
SCORE_NAMES = $score_names

# モデルの入力と出力の名前 (ML-ModelControl の同梱の例 tabular_autoencoder
# と同じ)
FEATURE_INPUT = "x"
RECON_OUTPUT = "recon"

# 正解の要る最良を選ぶ指標 (検証用に異常の行が無いと数えられない)
LABEL_METRICS = ("precision_at_k", "average_precision")

# 行ごとの度合いと, 度合いの決め方 (実験のディレクトリの中)
SCORES_FILE = "anomaly_scores.csv"
SCORER_FILE = "anomaly_scorer.json"

# 成果物の版 (anomaly_scorer.json の format_version. anomaly_scores.csv の
# 列もこの版に従う) と, 実験に写す前処理のレシピ (ML-EDA の当てはめた値を
# 含む recipe.yaml) の名前
SCORER_VERSION = 1
RECIPE_FILE = "anomaly_recipe.yaml"

# 行ごとの度合いの表と上位の行の要約の, 行の id の列にできない名前
ASSIGNED_COLUMNS = $assigned_columns

# 評価の名前 (evaluations/<名前>_<分割>.json), 説明, 向き
EVALUATION = "anomaly"
EVALUATION_TEXT = (
    "{split}の行 ({rows} 行, 異常 {anomalies} 行) の度合い ({score}) の, "
    "上位 K 件 (K = {k}. 人が 1 回に確かめる件数の上限は {capacity}) の"
    "適合率 precision_at_k と PR-AUC average_precision ({weights}). 大きい"
    "ほど良い. 異常の割合が小さいと上位 K 件の適合率は低く出る"
)
EVALUATION_BETTER = {"precision_at_k": "higher",
                     "average_precision": "higher"}
DISTRIBUTION = "anomaly_scores"
DISTRIBUTION_TEXT = (
    "{split}の行 ({rows} 行. 異常の正解が無い) の度合い ({score}) の分布 "
    "(quantile の分位点の度合い score. {weights})"
)
TOP = "anomaly_top"
TOP_TEXT = (
    "{split}の行 (異常の正解が無い) の, 度合い ({score}) の高い順の上位 "
    "{k} 行 (人が 1 回に確かめる件数の上限は {capacity}). column は度合い"
    "のうち最も大きい分の列, column_share はその割合 ({weights})"
)

# 評価する分割の名前
SPLIT_NAMES = {"valid": "検証用", "test": "テスト用",
               "unlabeled": "正解の欠けた"}
'''

# 自己符号化器と距離の部品, Trainer, 度合いと評価, 学習の流れ
ANOMALY_MAIN = '''

class PlaceholderModel(nn.Module):
    def __init__(self, features: int, width: int):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 学習に使う列の数と, 縮めた幅を保存
        self.features = features
        self.width = width

        # TODO: 課題に合うモデルに置き換える. 今は動くことを確かめるための,
        # 全結合で縮めて戻す小さな自己符号化器
        self.encoder = nn.Linear(self.features, self.width)
        self.decoder = nn.Linear(self.width, self.features)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> dict:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 学習に使う列の数) の行の特徴

        Returns
        ----------
        outputs: dict
            recon (入力と同じ形の再構成)
        """
        # 縮めて戻す順伝播
        outputs = {RECON_OUTPUT: self.decoder(self.relu(self.encoder(x)))}

        return outputs


class AnomalyTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment,
        valid: tuple[torch.Tensor, np.ndarray], cfg: Any
    ):
        # step を上書きするので step_fn は None
        # 親クラスのコンストラクタを呼び出す
        super(AnomalyTrainer, self).__init__(model, experiment, None)

        # 検証用の全ての行の特徴と正解, K, まとまりの大きさを保存
        self.valid_x, self.valid_y = valid
        self.k = cfg.anomaly.k
        self.batch_size = cfg.data.batch_size

        # 再構成と入力の二乗誤差
        self.criterion = nn.MSELoss()

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の, 再構成の二乗誤差を求める関数

        Parameters
        ----------
        batch: tuple
            (行の特徴,)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            再構成と入力の二乗誤差 (loss)
        """
        # 学習は学習しているモデル, 検証は評価のモデル (EMA があれば EMA)
        (x,) = batch
        model = self.model if phase == "train" else self.eval_model()
        outputs = {"loss": self.criterion(reconstruct(model, x), x)}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 検証用の行の度合いの上位 K 件の適合率と PR-AUC を
        数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証で検証用に異常の行があれば precision_at_k と
            average_precision (valid/<名前> として記録. 大きいほど良い).
            他は空
        """
        # 検証用の全ての行 (正常と異常) の度合い
        if phase != "valid" or not self.valid_y.any():
            return {}
        parts = autoencoder_parts(self.eval_model(), self.valid_x,
                                  self.batch_size)
        scores = parts.mean(axis=1)
        metrics = {
            "precision_at_k": precision_at_k(scores, self.valid_y, self.k),
            "average_precision": average_precision(scores, self.valid_y),
        }

        return metrics


def reconstruct(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    行の特徴を入力の名前でモデルに渡し, 再構成を返す関数

    Parameters
    ----------
    model: nn.Module
        自己符号化器
    x: torch.Tensor
        (件数, 学習に使う列の数) の行の特徴

    Returns
    ----------
    recon: torch.Tensor
        入力と同じ形の再構成
    """
    # 出力が 1 つの定義はテンソル, 仮のモデルと出力が複数の定義は辞書
    outputs = model(**{FEATURE_INPUT: x})
    recon = (outputs[RECON_OUTPUT] if isinstance(outputs, dict)
             else outputs)

    return recon


def autoencoder_parts(
    model: nn.Module, x: torch.Tensor, batch_size: int
) -> np.ndarray:
    """
    行ごと, 列ごとの再構成の二乗誤差を, まとまりごとに求める関数

    Parameters
    ----------
    model: nn.Module
        自己符号化器
    x: torch.Tensor
        (件数, 学習に使う列の数) の行の特徴
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    parts: np.ndarray
        (件数, 学習に使う列の数) の float64 (行の平均が度合い)
    """
    # 評価のモードへの切り替え (学習に戻すのは Trainer)
    device = next(model.parameters()).device
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size].to(device)
            error = (reconstruct(model, chunk) - chunk) ** 2
            chunks.append(error.cpu())
    parts = torch.cat(chunks).double().numpy()

    return parts


def fit_distance(x: np.ndarray, ridge: float) -> dict:
    """
    学習用の正常の行の中心と, 共分散の逆の精度行列を数える関数

    Parameters
    ----------
    x: np.ndarray
        (学習用の行の数, 学習に使う列の数) の行の特徴
    ridge: float
        共分散の対角に足す値 (散らばりの無い列でも逆行列を作れるように)

    Returns
    ----------
    distance: dict
        center (中心), precision (精度行列), ridge
    """
    # 中心と, 対角に ridge を足した共分散の逆 (行が 1 つなら散らばりは 0)
    points = x.astype(np.float64)
    center = points.mean(axis=0)
    diff = points - center
    covariance = diff.T @ diff / max(len(points) - 1, 1)
    covariance += np.eye(points.shape[1]) * ridge
    distance = {"center": center, "precision": np.linalg.inv(covariance),
                "ridge": float(ridge)}

    return distance


def precision_root(precision: np.ndarray) -> np.ndarray:
    """
    精度行列の対称な平方根を求める関数

    Parameters
    ----------
    precision: np.ndarray
        (学習に使う列の数, 学習に使う列の数) の float64 の精度行列

    Returns
    ----------
    root: np.ndarray
        (学習に使う列の数, 学習に使う列の数) の float64 の対称な行列 (二乗
        が精度行列. 負の固有値は 0)
    """
    # 対称にした精度行列の固有値の平方根での組み直し
    values, vectors = np.linalg.eigh((precision + precision.T) / 2)
    root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T

    return root


def distance_parts(x: np.ndarray, distance: dict) -> np.ndarray:
    """
    行ごと, 列ごとの, 二乗の Mahalanobis 距離の分を求める関数

    Parameters
    ----------
    x: np.ndarray
        (件数, 学習に使う列の数) の行の特徴
    distance: dict
        center と precision

    Returns
    ----------
    parts: np.ndarray
        (件数, 学習に使う列の数) の float64 の 0 以上の値 (行の平均が度合
        い. 行の和が二乗の Mahalanobis 距離)
    """
    # 精度行列の対称な平方根で白色化した差の二乗
    center = np.asarray(distance["center"], dtype=np.float64)
    precision = np.asarray(distance["precision"], dtype=np.float64)
    diff = x.astype(np.float64) - center
    parts = (diff @ precision_root(precision)) ** 2

    return parts


def read_splits(dataset: Dataset) -> dict:
    """
    分割ごとの行の特徴, 正解, 行の id を読む関数

    Parameters
    ----------
    dataset: Dataset
        ML-EDA の異常の検知の表

    Returns
    ----------
    splits: dict
        分割の名前から x (float32 の配列), y (0 か 1 の配列. 正解が無ければ
        全て 0), ids (行の id), anomalies (説明の異常の行の数)

    Raises
    ----------
    ValueError
        数でない学習に使う列か, 欠けた値がある場合
    """
    # 学習に使う列は数だけ (種類の列はデータの分析の段階で one_hot に)
    key = dataset.info["anomaly"]["id"]
    rows = dataset.info["anomaly"].get("rows") or {}
    splits = {}
    for split in dataset.frames:
        try:
            x, y = dataset.arrays(split)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{split} の学習に使う列に数でない列があります. レシピに "
                "one_hot の手順を足して書き出し直してください "
                f"({error})"
            ) from error
        x = np.array(x, dtype=np.float32)
        if np.isnan(x).any():
            raise ValueError(
                f"{split} の学習に使う列に欠けた値があります. レシピに "
                "impute (埋める) か drop_missing (行を落とす) の手順を足して"
                "書き出し直してください"
            )
        labels = (np.zeros(len(x), dtype=np.int64) if y is None
                  else np.asarray(y, dtype=np.int64))
        counts = rows.get(split) or {}
        splits[split] = {"x": x, "y": labels,
                         "ids": dataset.frames[split][key].to_numpy(),
                         "anomalies": int(counts.get("anomaly") or 0)}

    return splits


def check_config(cfg: Any) -> Optional[str]:
    """
    config の異常の検知の設定の誤りを返す関数

    Parameters
    ----------
    cfg: Any
        config (anomaly と model)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # 決め方は名前の中から, K と件数の上限は 1 以上の整数
    anomaly = cfg.anomaly
    message = None
    numbers = {"anomaly.k": anomaly.k, "anomaly.capacity": anomaly.capacity}
    wrong = [name for name, value in numbers.items()
             if not isinstance(value, int) or isinstance(value, bool)
             or value < 1]
    if anomaly.score not in SCORE_NAMES:
        message = (f"config の anomaly.score は "
                   f"{' か '.join(SCORE_NAMES)} にしてください (今は "
                   f"{anomaly.score})")
    elif wrong:
        message = (f"config の {', '.join(wrong)} は 1 以上の整数にして"
                   "ください")
    elif anomaly.score == "distance" and not (
        isinstance(anomaly.get("ridge"), (int, float))
        and anomaly.get("ridge") > 0
    ):
        message = "config の anomaly.ridge は 0 より大きい数にしてください"
    elif anomaly.score == "autoencoder" and "model" not in cfg:
        message = ("config に model がありません. 度合いの決め方を変えた"
                   "なら, 問題の設定から書き出し直してください")

    return message


def check_splits(splits: dict, cfg: Any) -> Optional[str]:
    """
    分割の行の数と, 最良を選ぶ指標に要る検証用の行を確かめる関数

    Parameters
    ----------
    splits: dict
        read_splits の結果
    cfg: Any
        config (度合いの決め方と監視する指標)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # 学習用の正常の行の数と, 自己符号化器の検証用の正常の行と, 正解の
    # 要る指標の検証用の異常の行の確かめ
    message = None
    train = splits.get("train")
    if train is None or len(train["x"]) < 2:
        message = ("学習用の行が 2 行より少ないので, 度合いを作れません. 行を"
                   "足すか, 検証用とテスト用の割合を見直して書き出し直して"
                   "ください")
        return message
    if cfg.anomaly.score != "autoencoder":
        return message
    valid = splits.get("valid")
    metric = str(cfg.trainer.monitor).split("/", 1)[-1]
    if valid is None or not (valid["y"] == 0).any():
        message = ("検証用の正常の行がありません. 最も良いモデルを選ぶ"
                   "ために, 検証用の割合を 0 より大きくして書き出し直して"
                   "ください")
    elif metric in LABEL_METRICS and not valid["anomalies"]:
        message = (f"検証用に異常の行が無いので, {cfg.trainer.monitor} を"
                   "数えられません. 最良を選ぶ指標を損失 (valid/loss) に"
                   "するか, 異常の行が検証用に入るように書き出し直して"
                   "ください")

    return message


def project_spec(cfg: Any) -> Optional[Path]:
    """
    config の model.spec の, プロジェクトの ML-ModelControl の定義を返す関数

    Parameters
    ----------
    cfg: Any
        config (model.spec に定義の場所. プロジェクトのディレクトリから)

    Returns
    ----------
    spec: Optional[Path]
        定義のファイル (場所を書いていないか, ファイルがまだ無ければ None)
    """
    # 定義のファイル (ML-CreateArchitecture で組んで保存するまでは None)
    spec = PROJECT_DIR / cfg.model.spec if cfg.model.spec else None
    if spec is not None and not spec.is_file():
        spec = None

    return spec


def spec_errors(spec: Optional[Path], width: int) -> list[str]:
    """
    ML-ModelControl の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)
    width: int
        学習に使う列の数

    Returns
    ----------
    errors: list[str]
        誤りの文 (定義が無いか誤りが無ければ空)
    """
    # ML-ModelControl の確かめ (定義の無いうちは読み込み無し. 読めない
    # 定義も例外でなく誤りの文)
    if spec is None:
        return []
    from ml_modelcontrol import validate
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]
    if errors:
        return errors

    # 入力は行の特徴, 出力に再構成
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    inputs = loaded.get("inputs") or {}
    if list(inputs) != [FEATURE_INPUT]:
        errors.append(f"定義の入力は {FEATURE_INPUT} の 1 つにしてください "
                      f"(今は {list(inputs)})")
    if RECON_OUTPUT not in (loaded.get("outputs") or {}):
        errors.append(f"定義の出力に {RECON_OUTPUT} を入れてください")
    entry = inputs.get(FEATURE_INPUT)
    shape = entry.get("shape") if isinstance(entry, dict) else None
    if isinstance(shape, list) and shape and shape[-1] != width:
        errors.append(f"入力 {FEATURE_INPUT} の最後の大きさを学習に使う列の"
                      f"数 {width} にしてください (今は {shape[-1]})")

    return errors


def spec_file(
    spec: Optional[Path], experiment: Experiment
) -> tuple[Optional[Path], Optional[Path]]:
    """
    モデルを組み立てる ML-ModelControl の定義のファイルと, 定義の中の相対
    の参照 (外部の重みなど) の起点を返す関数

    Parameters
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)
    experiment: Experiment
        実験 (ML-Found が実験に写した定義の場所 model_spec_path と, 写した
        元の定義 model_spec_source)

    Returns
    ----------
    path: Optional[Path]
        ML-Found が実験に写した定義. 写せなければプロジェクトの定義. 定義
        が無ければ None
    base_dir: Optional[Path]
        相対の参照の起点 (写しなら元の定義のディレクトリ. 他は None)
    """
    # 定義が無いうちは仮のモデル
    if spec is None:
        return None, None

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


def make_model(
    width: int, cfg: Any, spec: Optional[Path], base_dir: Optional[Path]
) -> nn.Module:
    """
    モデルを作る関数

    Parameters
    ----------
    width: int
        学習に使う列の数
    cfg: Any
        config (仮のモデルの縮めた幅と乱数の種)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    base_dir: Optional[Path]
        定義の中の相対の参照の起点 (None なら定義のディレクトリ)

    Returns
    ----------
    model: nn.Module
        定義から組み立てたモデルか, 仮のモデル
    """
    # 定義があれば組み立て (定義の無いうちは ML-ModelControl を読まない)
    if spec is not None:
        from ml_modelcontrol import build_model
        model = build_model(spec, seed=cfg.experiment.seed,
                            base_dir=base_dir)
        return model
    torch.manual_seed(cfg.experiment.seed)
    model = PlaceholderModel(width, cfg.model.width)

    return model


def train_autoencoder(
    splits: dict, cfg: Any, spec: Optional[Path], experiment: Experiment
) -> tuple[nn.Module, str, Any]:
    """
    学習用の正常の行で自己符号化器を学ぶ関数

    Parameters
    ----------
    splits: dict
        read_splits の結果
    cfg: Any
        config
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)
    experiment: Experiment
        実験

    Returns
    ----------
    model: nn.Module
        学習した後の重みのモデル (CPU)
    weights: str
        評価する重みの名前 (最良のモデルか, 最後のエポックの重み)
    epoch: Any
        評価する重みのエポック
    """
    # 学習用 (正常の行だけ) と, 検証の損失を数える検証用の正常の行
    train_x = torch.tensor(splits["train"]["x"])
    valid = splits["valid"]
    valid_x = torch.tensor(valid["x"])
    normal = torch.tensor(valid["x"][valid["y"] == 0])
    train_loader = DataLoader(TensorDataset(train_x),
                              batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(normal),
                              batch_size=cfg.data.batch_size)

    # モデルの組み立てと学習 (定義がまだ無ければ仮のモデル)
    spec, base_dir = spec_file(spec, experiment)
    if spec is None:
        experiment.logger.info(
            f"モデルの定義 ({cfg.model.spec or '場所なし'}) が無いので, "
            "仮のモデル (PlaceholderModel) で学習します. ML-CreateArchitecture"
            " で組んだ定義をその場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(train_x.shape[1], cfg, spec, base_dir)
    trainer = AnomalyTrainer(model, experiment, (valid_x, valid["y"]), cfg)
    trainer.fit(train_loader, valid_loader)

    # 学習した後の重みは, 最良のモデルに戻した場合だけ最良のエポック
    # (select_best: false などは最後)
    best = trainer.final_weights == "best"
    epoch = trainer.checkpoint.best_epoch if best else trainer.epoch
    weights = "最良のモデル" if best else "最後のエポックの重み"
    print(f"評価する重み: {weights} (エポック {epoch})")
    if best:
        print(f"ベストの {cfg.trainer.monitor}: "
              f"{trainer.checkpoint.best_value}")
    model = trainer.model.cpu().eval()

    return model, weights, epoch


def normal_summary(scores: np.ndarray) -> dict:
    """
    学習用の正常の行の度合いの分布をまとめる関数

    Parameters
    ----------
    scores: np.ndarray
        学習用の行の度合い

    Returns
    ----------
    summary: dict
        rows, mean, p50, p90, p95, p99, max (度合いの読み方の目安)
    """
    # 分位点は線形の補間
    summary = {"rows": int(len(scores)), "mean": float(scores.mean())}
    for point in (50, 90, 95, 99):
        summary[f"p{point}"] = float(np.percentile(scores, point))
    summary["max"] = float(scores.max())

    return summary


def distribution_rows(scores: np.ndarray) -> list[dict]:
    """
    度合いの分布の曲線 (0 から 1 の 0.05 刻みの分位点) の行を作る関数

    Parameters
    ----------
    scores: np.ndarray
        分割の行の度合い

    Returns
    ----------
    rows: list[dict]
        quantile と score の行
    """
    # 21 点の分位点
    rows = [{"quantile": round(step * 0.05, 2),
             "score": float(np.quantile(scores, step * 0.05))}
            for step in range(21)]

    return rows


def top_rows(
    part: dict, scores: np.ndarray, parts: np.ndarray, key: str,
    features: list, k: int
) -> list[dict]:
    """
    度合いの高い順の上位 K 行の要約を作る関数

    Parameters
    ----------
    part: dict
        分割の ids
    scores: np.ndarray
        分割の行の度合い
    parts: np.ndarray
        分割の行ごと, 列ごとの度合いの分
    key: str
        行の id の列の名前
    features: list
        学習に使う列の名前
    k: int
        行の数

    Returns
    ----------
    rows: list[dict]
        rank, 行の id, score, column (最も大きい分の列), column_share
        (度合いのうちその列の分の割合)
    """
    # 度合いの高い順 (同じ度合いは前の行からの安定な並べ替え)
    order = np.argsort(-scores, kind="stable")[:k]
    rows = []
    for rank, index in enumerate(order, 1):
        column = int(np.argmax(parts[index]))
        total = float(parts[index].sum())
        value = part["ids"][index]
        rows.append({
            "rank": rank,
            key: value.item() if isinstance(value, np.generic) else value,
            "score": float(scores[index]), "column": features[column],
            "column_share": (float(parts[index][column]) / total
                             if total else None),
        })

    return rows


def evaluate_splits(
    experiment: Experiment, splits: dict, found: dict, info: dict, cfg: Any
) -> None:
    """
    学習用の他の分割ごとに, 度合いの評価を evaluations/ に書く関数

    Parameters
    ----------
    experiment: Experiment
        実験
    splits: dict
        read_splits の結果
    found: dict
        分割の名前から scores と parts
    info: dict
        id (行の id の列), features, weights, epoch, score (決め方の名前)
    cfg: Any
        config (K と件数の上限)

    Returns
    ----------
    None
    """
    # 異常の行があれば上位 K 件の適合率と PR-AUC, 無ければ度合いの分布と
    # 上位の行の要約
    k = cfg.anomaly.k
    for split, title in SPLIT_NAMES.items():
        if split not in splits:
            continue
        part = splits[split]
        scores = found[split]["scores"]
        names = {"split": title, "rows": len(scores), "k": k,
                 "anomalies": part["anomalies"], "score": info["score"],
                 "capacity": cfg.anomaly.capacity,
                 "weights": info["weights"]}
        if part["anomalies"]:
            row = {"rows": len(scores), "anomalies": part["anomalies"],
                   "k": k,
                   "precision_at_k": precision_at_k(scores, part["y"], k),
                   "average_precision": average_precision(scores,
                                                          part["y"])}
            experiment.logger.info(
                f"{title}の上位 {k} 件の適合率 {row['precision_at_k']:.4f}, "
                f"PR-AUC {row['average_precision']:.4f}"
            )
            save_evaluation(
                experiment, f"{EVALUATION}_{split}", row, split,
                description=EVALUATION_TEXT.format(**names),
                epoch=info["epoch"], better=EVALUATION_BETTER
            )
            continue
        experiment.logger.info(
            f"{title}の行に異常の正解が無いので, 度合いの分布と上位 {k} 行"
            "の要約を書きます"
        )
        save_evaluation(
            experiment, f"{DISTRIBUTION}_{split}", distribution_rows(scores),
            split, description=DISTRIBUTION_TEXT.format(**names),
            kind="curve", x="quantile", epoch=info["epoch"]
        )
        save_evaluation(
            experiment, f"{TOP}_{split}",
            top_rows(part, scores, found[split]["parts"], info["id"],
                     info["features"], k),
            split, description=TOP_TEXT.format(**names), epoch=info["epoch"]
        )


def write_results(
    run_dir: Path, dataset: Path, splits: dict, found: dict, info: dict,
    distance: Optional[dict], labeled: bool, cfg: Any
) -> None:
    """
    行ごとの度合い, 度合いの決め方と前処理のレシピを実験のディレクトリに
    書く関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    dataset: Path
        ML-EDA の異常の検知の表のディレクトリ (recipe.yaml を写す)
    splits: dict
        read_splits の結果
    found: dict
        分割の名前から scores
    info: dict
        id (行の id の列), features
    distance: Optional[dict]
        距離の中心と精度行列 (自己符号化器は None)
    labeled: bool
        データセットに異常の正解の列があるか
    cfg: Any
        config (度合いの決め方, K と件数の上限)

    Returns
    ----------
    None
    """
    # 行の id, 分割, 度合い, 正解の表 (正解の無い表と unlabeled の正解は空欄)
    tables = []
    for split, part in splits.items():
        known = labeled and split != "unlabeled"
        tables.append(pd.DataFrame({
            info["id"]: part["ids"], "split": split,
            "score": found[split]["scores"],
            "anomaly": pd.array(part["y"] if known
                                else [None] * len(part["y"]),
                                dtype="Int64"),
        }))
    pd.concat(tables, ignore_index=True).to_csv(run_dir / SCORES_FILE,
                                                index=False)

    # 推論の段階が読む前処理のレシピの写しと度合いの決め方 (新しい行は
    # レシピの前処理の後の features から)
    shutil.copy(dataset / "recipe.yaml", run_dir / RECIPE_FILE)
    scorer = {
        "format_version": SCORER_VERSION, "score": str(cfg.anomaly.score),
        "recipe": RECIPE_FILE, "scores": SCORES_FILE,
        "features": list(info["features"]),
        "id": info["id"], "capacity": int(cfg.anomaly.capacity),
        "k": int(cfg.anomaly.k),
        "normal": normal_summary(found["train"]["scores"]),
        "model": ({"input": FEATURE_INPUT, "output": RECON_OUTPUT}
                  if distance is None else None),
        "distance": None if distance is None else {
            "center": distance["center"].tolist(),
            "precision": distance["precision"].tolist(),
            "ridge": distance["ridge"],
        },
    }
    text = json.dumps(scorer, ensure_ascii=False, indent=2)
    (run_dir / SCORER_FILE).write_text(text + "\\n", encoding="utf-8")


def make_inference(
    cfg: Any, spec: Optional[Path], run_dir: Path, experiment: Experiment
) -> None:
    """
    ML-Inference の異常の度合いの一式を, 実験の中の inference/ に作る関数

    Parameters
    ----------
    cfg: Any
        config (度合いの決め方, 元の表の場所, 一式の名前)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (自己符号化器. 無ければ None)
    run_dir: Path
        実験のディレクトリ (行ごとの度合いと度合いの決め方を書いた後)
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    None
    """
    # 定義が無い仮のモデルの一式の省略
    autoencoder = cfg.anomaly.score == "autoencoder"
    if autoencoder and spec is None:
        experiment.logger.info(
            "仮のモデル (PlaceholderModel) は ML-ModelControl の定義が無い"
            "ので, 推論の一式を作りません. config の model.spec の場所に"
            "定義を置くと, 学習の後に作ります"
        )
        return

    # 定義を写せなかった実験の一式の省略 (一式は実験に写した定義を読む)
    if autoencoder and experiment.model_spec_path is None:
        experiment.logger.info(
            "モデルの定義を実験に写せなかったので, 推論の一式を作りません "
            "(理由は source.yaml の model_spec の status)"
        )
        return

    # 元の表 (config の data.source) との度合いの比べ (元の表が無ければ
    # 省略)
    data = None
    if cfg.data.source:
        path = PROJECT_DIR / str(cfg.data.source)
        data = path if path.is_file() else None
    if data is None:
        experiment.logger.info(
            f"元の表 (config の data.source: {cfg.data.source}) が無いので,"
            " 推論の一式の度合いを実験の度合いと比べません"
        )

    # 約束に合わない場合の ML-Inference の誤りの文の警告と, 学習の結果の
    # 保存
    from ml_inference import make_anomaly_bundle
    options = {**OPTIONS, **({"loader": LOADER} if LOADER else {})}
    try:
        info = make_anomaly_bundle(
            run_dir / INFERENCE_DIR, run_dir, name=cfg.experiment.name,
            data=data, options=options if data is not None else None
        )
    except (ValueError, FileNotFoundError) as error:
        experiment.logger.warning(
            "成果物か定義が異常の度合いの一式の約束に合わないので, 推論の"
            f"一式を作れません: {error}"
        )
        return
    check = info["anomaly"]["check"]
    compared = ("度合いは比べず" if check is None
                else f"度合いの一致 {check['agreement'] * 100:.1f} % "
                f"({check['rows']} 行. 差の最大 {check['max_diff']:.3g})")
    experiment.logger.info(
        f"推論の一式 (異常の度合い): {INFERENCE_DIR}/ ({compared}. 人の"
        f"確認に回す件数の上限 {info['decision']['capacity']}. 版 "
        f"{info['format_version']})"
    )
    for note in info["notes"]:
        experiment.logger.warning(f"推論の一式の注意: {note}")


def main() -> None:
    """
    学習用の正常の行から度合いの決め方を作り, 行ごとの度合いと評価と成果物
    を残す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config とデータセット (無ければ書き出しの案内)
    cfg, diff = load_project_config(str(PROJECT_DIR))
    path = PROJECT_DIR / cfg.data.dataset
    if not (path / "dataset.yaml").is_file():
        print(f"{path} がありません. README の「次にすること」の 1 で, "
              "異常の検知の表を書き出してください")
        sys.exit(1)
    message = check_config(cfg)
    if message:
        print(message)
        sys.exit(1)
    dataset = load_dataset(path)
    if dataset.info.get("task") != "anomaly" or not isinstance(
        dataset.info.get("anomaly"), dict
    ):
        print(f"{path} は異常の検知の表ではありません (タスク "
              f"{dataset.info.get('task')}). task: anomaly のレシピで書き出"
              "してください")
        sys.exit(1)
    key = str(dataset.info["anomaly"]["id"])
    if key in ASSIGNED_COLUMNS:
        print(f"行の id の列 {key} は行ごとの度合いの表の列 ("
              f"{', '.join(ASSIGNED_COLUMNS)}) と同じ名前です. 名前を変えて"
              "書き出し直してください")
        sys.exit(1)

    # 分割ごとの行の特徴, 正解, 行の id (数でない列と欠けた値で停止)
    try:
        splits = read_splits(dataset)
    except ValueError as error:
        print(error)
        sys.exit(1)
    message = check_splits(splits, cfg)
    if message:
        print(message)
        sys.exit(1)
    features = list(dataset.features)
    width = len(features)

    # モデルの定義の確かめ (自己符号化器だけ. 誤りなら出力先を作る前の
    # 停止)
    autoencoder = cfg.anomaly.score == "autoencoder"
    spec = project_spec(cfg) if autoencoder else None
    errors = spec_errors(spec, width)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\\n".join(f"- {message}" for message in errors))
        sys.exit(1)

    # 何の実験か (受け持ち, 出力の形, 度合いを出すもの, 作り方は問題の
    # 設定から)
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    write_task(experiment, {"role": ROLE, "output": OUTPUT,
                            "what": PREDICT, "method": METHOD})
    counts = ", ".join(f"{split} {len(part['x'])} 行 (異常 "
                       f"{part['anomalies']} 行)"
                       for split, part in splits.items())
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}度合いの決め方: "
        f"{SCORE_NAMES[cfg.anomaly.score]}. 行: {counts}. 学習に使う列: "
        f"{width}. K: {cfg.anomaly.k}"
    )

    # 度合いの決め方 (自己符号化器は学習, 距離は中心と精度行列) と, 全ての
    # 分割の行ごと, 列ごとの度合いの分
    distance = None
    if autoencoder:
        model, weights, epoch = train_autoencoder(splits, cfg, spec,
                                                  experiment)
        found = {}
        for split, part in splits.items():
            parts = autoencoder_parts(model, torch.tensor(part["x"]),
                                      cfg.data.batch_size)
            found[split] = {"parts": parts, "scores": parts.mean(axis=1)}
    else:
        distance = fit_distance(splits["train"]["x"], cfg.anomaly.ridge)
        weights, epoch = "学習用の正常の行の中心と精度行列", None
        found = {}
        for split, part in splits.items():
            parts = distance_parts(part["x"], distance)
            found[split] = {"parts": parts, "scores": parts.mean(axis=1)}

    # 実験だけで見直すための問題の設定の写し (元の問題の設定の場所
    # origin も. モデルの定義の写しは ML-Found の start_experiment)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    print(f"保存先: {run_dir}")

    # 度合いの評価の evaluations/ への書き出しと, 行ごとの度合いと決め方の
    # 実験への書き出し
    info = {"id": key, "features": features, "weights": weights,
            "epoch": epoch, "score": SCORE_NAMES[cfg.anomaly.score]}
    evaluate_splits(experiment, splits, found, info, cfg)
    write_results(run_dir, path, splits, found, info, distance,
                  dataset.target is not None, cfg)
    experiment.logger.info(
        f"行ごとの度合い: {SCORES_FILE}. 度合いの決め方: {SCORER_FILE} "
        f"(人が 1 回に確かめる件数の上限 {cfg.anomaly.capacity})"
    )

    # 成果物から ML-Inference の推論の一式
    make_inference(cfg, spec, run_dir, experiment)


if __name__ == "__main__":
    main()
'''

# config の基準 (ML-Found のプロジェクトの形)
ANOMALY_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題の設定から書き出した骨組み)
$requires
experiment:
  name: $experiment
  seed: 0
$training
# プロジェクト固有の設定
data:
  # ML-EDA の異常の検知の表 (task: anomaly. プロジェクトのディレクトリから.
  # 学習用は正常の行だけ)
  dataset: $dataset
  # 入れた表の写し (プロジェクトのディレクトリから. 推論の一式づくりが
  # 度合いを実験の度合いと比べる元の表. 入れていなければ null)
  source: $source
  # 行のバッチの大きさ
  batch_size: 128

anomaly:
  # 度合いの決め方 (autoencoder は自己符号化器の再構成の誤差, distance は
  # 学習用の正常の行の中心からの距離)
  score: $score
  # 評価の上位 K 件の K (既定は人が確かめる件数の上限)
  k: $k
  # 人が 1 回に確かめる件数の上限 (度合いの上位から人の確認に回す)
  capacity: $capacity
$tail''')

# 自己符号化器の学習の設定
ANOMALY_TRAINING = Template('''
optimizer:
  name: AdamW
  params:
    lr: 0.001

trainer:
  epochs: 30
  grad_clip_norm: 1.0
  # 問題の設定の「最良を選ぶ指標」(valid/loss は検証用の正常の行の再構成の
  # 誤差で, 小さいほど良い. valid/average_precision と
  # valid/precision_at_k は検証用の行の度合いの PR-AUC と上位 K 件の適合率
  # で, 大きいほど良い)
  monitor: $monitor
  mode: $mode
''')

# 自己符号化器のモデルの設定
ANOMALY_MODEL = Template('''
model:
  # ML-ModelControl の形のモデルの定義 (ML-CreateArchitecture で組んで, この
  # 場所に保存する. 入力は行の特徴 x, 出力に再構成 recon)
  spec: models/model.yaml
$placeholder  # 仮のモデルの縮めた幅
  width: 8
''')

# 距離に基づく度合いの設定
ANOMALY_DISTANCE = Template('''  # 共分散の対角に足す値 (散らばりの無い列でも精度行列を作れるように)
  ridge: $ridge
''')


def anomaly_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 異常の度合いを作って評価するスクリプトを作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    literal: Callable
        値を Python の式にする関数 (skeleton.literal)

    Returns
    ----------
    script: str
        train.py の中身
    """
    # 頭 (元の表の読み方は問題の設定の手元のデータの読み方) と学習の流れ
    names = {item["value"]: item["label"] for item in ANOMALY_SCORES}
    options = get_value(problem, "data.options")
    head = Template(ANOMALY_HEAD).substitute(
        name=literal(str(problem.get("name", ""))),
        loader=literal(get_value(problem, "data.loader")),
        options=literal(options if isinstance(options, dict) else {}),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        method=literal(active_value(problem, f"{PART}.method")),
        score_names=literal(names),
        assigned_columns="[\n" + "".join(
            f"    {json.dumps(name)},\n" for name in ANOMALY_ASSIGNED
        ) + "]",
    )
    # train.log の初めの行の度合いを出すもの (空なら省略)
    made = ("異常の度合い: {PREDICT}. "
            if active_value(problem, f"{PART}.what") else "")
    script = head + Template(ANOMALY_MAIN).substitute(predict_log=made)

    return script


def anomaly_config(
    experiment: str, stages: dict, dataset: str, requires: str,
    spec_exists: bool = False, source: Optional[str] = None
) -> str:
    """
    異常の検知の config の基準を作る関数

    Parameters
    ----------
    experiment: str
        実験の名前
    stages: dict
        受け渡し
    dataset: str
        異常の検知の表の場所 (プロジェクトのディレクトリから)
    requires: str
        一番上の requires の行 (skeleton.requires_text)
    spec_exists: bool = False
        書き出し先に定義 (models/model.yaml) が既にあるか (あれば
        model.placeholder を書かない)
    source: Optional[str] = None
        写した表の場所 (プロジェクトのディレクトリから. 入れていなければ
        None)

    Returns
    ----------
    text: str
        configs/default/train.yaml の中身
    """
    # 自己符号化器は学習とモデル, 距離は対角に足す値
    anomaly = stages["anomaly"]
    training = stages["training"]
    if anomaly["score"] == "autoencoder":
        middle = ANOMALY_TRAINING.substitute(monitor=training["monitor"],
                                             mode=training["mode"])
        tail = ANOMALY_MODEL.substitute(
            placeholder="" if spec_exists else ANOMALY_PLACEHOLDER
        )
    else:
        middle = ""
        tail = ANOMALY_DISTANCE.substitute(ridge=ANOMALY_RIDGE)
    text = ANOMALY_CONFIG.substitute(
        requires=requires, experiment=json.dumps(experiment,
                                                 ensure_ascii=False),
        training=middle, dataset=json.dumps(dataset, ensure_ascii=False),
        source=json.dumps(source, ensure_ascii=False),
        score=json.dumps(anomaly["score"]), k=json.dumps(anomaly["k"]),
        capacity=json.dumps(anomaly["capacity"]), tail=tail,
    )

    return text


def anomaly_name(problem: dict) -> str:
    """
    異常の検知の骨組みの, 実験の名前の元を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    raw: str
        受け持ちと度合いの決め方 (anomaly_autoencoder など)
    """
    # 目的の列とタスクの代わり
    raw = f"anomaly_{active_value(problem, f'{PART}.anomaly.score')}"

    return raw


def anomaly_readme(
    problem: dict, stages: dict, dataset: str, raw: Optional[str]
) -> str:
    """
    異常の検知のプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    dataset: str
        異常の検知の表の場所 (プロジェクトのディレクトリから)
    raw: Optional[str]
        写した表の場所 (プロジェクトのディレクトリから. 入れていなければ
        None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    anomaly = stages["anomaly"]
    autoencoder = anomaly["score"] == "autoencoder"
    names = {item["value"]: item["label"] for item in ANOMALY_SCORES}
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, "
             "ML-Found のプロジェクトの骨組み (正常の行が多い表の, 行ごとの"
             "異常の度合いを作り, 度合いの評価と推論の段階が読む成果物を"
             "残す).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1. データ (ML-EDA の異常の検知の表)
    columns = ", ".join(f"`{name}`" for name in anomaly["columns"])
    label = (f"異常の正解の列は `{anomaly['label']}` (正解の欠けた行は"
             "正常と見なす `split.unlabeled: normal`)" if anomaly["label"]
             else "異常の正解は無い (全ての行を正常と見なす)")
    key = (f"`{anomaly['id']}`" if anomaly["id"]
           else "行の位置 (`row_id`)")
    libraries = ("ML-EDA, ML-Found, ML-ModelControl, ML-Inference"
                 if autoencoder else "ML-EDA, ML-Found, ML-Inference")
    lines += [
        "", "## 次にすること", "",
        f"学習する環境には {libraries} を入れる (config の `requires` の"
        "機能. 足りなければ学習の前に入れ方を示して止まる"
        + (". ML-ModelControl は定義を置くと使う)." if autoencoder
           else ")."), "",
        "1. **データの分析** (ML-EDA, ML-Fathom): ML-EDA の異常の検知の表 "
        "(版 1 の `task: anomaly`) を config の `data.dataset` の場所 "
        f"(`{dataset}`) に書き出す. ML-Fathom の「問題の設定から始める」か, "
        "ML-Quest の `recipe_from` のレシピの雛形で ML-EDA の "
        f"`export_dataset` を使う. 度合いに使う列は {columns}, {label}, "
        f"行の id は {key}. 学習用は正常の行だけで, 異常の行は検証用とテスト"
        "用に入る. 骨組みは数の列だけを読むので, 欠けた値は `impute` か "
        "`drop_missing` で無くし, 種類の列は `one_hot` にする"
        + (f". 入れた表はこのディレクトリの `{raw}` に写した" if raw
           else ""),
    ]

    # 2. モデル (自己符号化器だけ)
    if autoencoder:
        lines.append(
            "2. **モデルの作成** (ML-ModelControl, ML-CreateArchitecture): "
            "ML-CreateArchitecture で組んだ定義を, config の `model.spec` の"
            "場所 (`models/model.yaml`) に保存する. 入力は行の特徴 `x` ((件"
            "数, 学習に使う列の数) の float32) の 1 つ, 出力に入力と同じ形の"
            "再構成 `recon`. 学習に使う列の数は `train.log` の初めの行に"
            "出る. ML-ModelControl の同梱の例 `tabular_autoencoder` の入力と"
            "出力の幅を列の数に合わせて使える. 定義が無いうちは, `train.py` "
            "の `PlaceholderModel` (印 TODO. 動くことを確かめるだけの小さな"
            "モデル) で学習する (config の `model.placeholder: true` がその"
            "目印)"
        )
    else:
        lines.append(
            "2. **モデルの作成**: 要らない (距離に基づく度合いは, 学習用の"
            "正常の行の中心と精度行列を数えるだけで, モデルの定義を使わない)"
        )

    # 3. 学習と評価
    monitor = stages["training"]["monitor"]
    k = anomaly["k"]
    how = (
        "`python train.py`. 学習用の正常の行の再構成と入力の二乗誤差で"
        "自己符号化器を学び, 検証用の正常の行の損失と, 検証用に異常の行が"
        "あれば度合いの上位 K 件の適合率と PR-AUC を数える. 検証用の "
        f"`{monitor}` が最も良いモデルを選ぶ"
        if autoencoder else
        "`python train.py`. 学習用の正常の行の平均 (中心) と, 共分散の"
        "対角に config の `anomaly.ridge` を足した行列の逆 (精度行列) を"
        "数える (学習の繰り返しは無い)"
    )
    lines.append(
        f"3. **学習と評価** (ML-Found, ML-Dashboard): {how}. 度合いは "
        f"{names[anomaly['score']]}で, 大きいほど異常. 学習の後に, 検証用, "
        "テスト用と正解の欠けた行の分割ごとに, 異常の行があれば上位 K 件 "
        f"(K = {k}. config の `anomaly.k`. 既定は人が確かめる件数の上限) の"
        "適合率と PR-AUC を `evaluations/anomaly_<分割>.json` に, 無ければ"
        "度合いの分布 (`evaluations/anomaly_scores_<分割>.json`) と上位 K "
        "行の要約 (`evaluations/anomaly_top_<分割>.json`. 行の id, 度合い,"
        " 最も外れた列) を書く. 結果は `outputs/<実験の名前>/<日付>/<時刻>/` "
        "に残る (`task.yaml`, "
        + ("学習した重み `final_model.pt`, " if autoencoder else "")
        + "行ごとの度合い `anomaly_scores.csv`, 度合いの決め方 "
        "`anomaly_scorer.json` と前処理のレシピの写し `anomaly_recipe.yaml`)"
    )

    # 4. 推論
    lines += [
        "4. **推論** (ML-Inference, ML-Provide): 学習の後に, 実験の度合いの"
        "決め方 `anomaly_scorer.json` と前処理のレシピの写し "
        "`anomaly_recipe.yaml` "
        + ("と, 学習した重みと写した定義 " if autoencoder else "")
        + "から, ML-Inference の異常の度合いの一式を実験の `inference/` に"
        "作る (config の `data.source` の元の表があれば, 一式の度合いを"
        "実験の度合いと比べて `train.log` に書く"
        + (". 仮のモデルでは作らない" if autoencoder else "")
        + "). 一式は新しい行にレシピの前処理をかけて度合いを出す. 推論の 1 回"
        "で, 度合いの高い順に人が確かめる件数の上限 "
        f"({anomaly['capacity']} 件. config の `anomaly.capacity`) まで人の"
        "確認に回す (受け渡しの推論の意思決定 `{type: triage, capacity}` "
        "は, ML-Inference の異常の度合いの一式が読む形)",
        "", "`train.py` の印 (TODO) の所 (モデル) は書き足す. 問題の設定を"
        "変えたら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
