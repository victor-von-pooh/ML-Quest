import json
from string import Template
from typing import Callable, Optional

from ml_quest.handoff import describe
from ml_quest.model_names import (
    CONTROL_OUTPUT, FEATURE_INPUT, TREATED_OUTPUT
)
from ml_quest.problem import active_value, get_value
from ml_quest.questions import PART, UPLIFT_ASSIGNED, UPLIFT_OUTCOMES

# 学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の機能の名前
# (データセットは ML-EDA で読む. config の model.spec の定義を置くと
# ML-ModelControl の読み込みと, ML-Inference の推論の一式)
UPLIFT_LIBRARIES = ["eda", "modelcontrol", "inference"]

# データの分析の段階が書き出す, 介入の記録の表の場所 (プロジェクトの
# ディレクトリから)
UPLIFT_DATASET = "outputs/dataset"

# spec のファイルが無いときの, 仮のモデルで学習することの目印 (config の
# model.placeholder)
UPLIFT_PLACEHOLDER = """  # spec のファイルが無いうちは train.py の仮のモデルで学習することの目印
  # (学習・評価の段階の学習の準備が読む. train.py は読まず, 仮のモデルを使うか
  # は spec のファイルがあるかで決まる)
  placeholder: true
"""

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
UPLIFT_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

無作為に配った記録の表 (ML-EDA の介入の記録の表) で, 処置の群と対照の群の
結果の 2 つの予測 (T-learner) を学び, 行ごとの効果 (処置したときの結果 −
しないときの結果) とその評価 (Qini の曲線と係数, 上位 K 割の効果の差),
推論の段階が読む成果物と推論の一式を残す骨組みなので, 印 (TODO) の所を
書き足す
"""
import json
import math
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
    Experiment, Trainer, load_project_config, qini_coefficient, qini_curve,
    save_evaluation, start_experiment, uplift_at_k, write_task
)

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 効果を見積もるもの
PROBLEM_NAME = $name
PREDICT = $predict

# task.yaml の受け持ち, 出力の形と作り方 (問題の設定の答え)
ROLE = "effect"
OUTPUT = "effect"
METHOD = $method

# 結果の種類の名前
OUTCOME_NAMES = $outcome_names

# 元の表の読み方 (問題の手元のデータの読み方. 推論の一式づくりが
# ML-EDA の load_data に渡す. None は ML-EDA の推定)
LOADER = $loader
OPTIONS = $options

# 推論の一式を作る場所 (実験のディレクトリの中)
INFERENCE_DIR = "inference"

# モデルの入力と出力の名前 (ML-ModelControl の同梱の例 tabular_uplift と
# 同じ) と, 効果の決め方の名前 (処置の群と対照の群の 2 つの予測)
FEATURE_INPUT = $feature_input
TREATED_OUTPUT = $treated_output
CONTROL_OUTPUT = $control_output
LEARNER = "t_learner"

# 行ごとの効果と, 効果の決め方 (実験のディレクトリの中)
EFFECTS_FILE = "uplift_effects.csv"
ESTIMATOR_FILE = "uplift_estimator.json"

# 成果物の版 (uplift_estimator.json の format_version. uplift_effects.csv
# の列もこの版に従う) と, 実験に写す前処理のレシピ (ML-EDA の当てはめた値を
# 含む recipe.yaml) の名前
ESTIMATOR_VERSION = 1
RECIPE_FILE = "uplift_recipe.yaml"

# 行ごとの効果の表の, 行の id の列にできない名前
ASSIGNED_COLUMNS = $assigned_columns

# 評価の名前 (evaluations/<名前>_<分割>.json), 説明, 向きと, Qini の曲線の
# 残す点の数の目安
EVALUATION = "uplift"
EVALUATION_TEXT = (
    "{split}の行 ({rows} 行. 処置の群 {treated} 行, 対照の群 {control} 行) "
    "の効果の見積もり ({outcome}) の, Qini の係数 qini と上位 {percent} % "
    "(K = {k}) の処置の群と対照の群の結果の平均の差 uplift_at_k ("
    "{weights}). "
    "大きいほど良い. effect は分割の全ての行の結果の平均の差 (無作為に配った"
    "場合), estimated は見積もりの平均. 配る数の上限は {budget}"
)
EVALUATION_BETTER = {"qini": "higher", "uplift_at_k": "higher"}
CURVE = "qini_curve"
CURVE_TEXT = (
    "{split}の行の Qini の曲線 (fraction は効果の見積もりの高い順の上位の"
    "行の割合, qini は処置で増えた結果 (対照の群の数を処置の群に合わせた"
    "結果の和の差), random は無作為に配った場合の直線. {weights})"
)
CURVE_POINTS = 101

# 評価する分割の名前
SPLIT_NAMES = {"valid": "検証用", "test": "テスト用"}
'''

# T-learner の部品, Trainer, 効果と評価, 学習の流れ
UPLIFT_MAIN = '''

class PlaceholderModel(nn.Module):
    def __init__(self, features: int, width: int):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 学習に使う列の数と, 隠れ層の幅を保存
        self.features = features
        self.width = width

        # TODO: 課題に合うモデルに置き換える. 今は動くことを確かめるための,
        # 重みを共有しない 2 つの全結合の枝 (処置の群と対照の群)
        self.treated = nn.Sequential(nn.Linear(self.features, self.width),
                                     nn.ReLU(), nn.Linear(self.width, 1))
        self.control = nn.Sequential(nn.Linear(self.features, self.width),
                                     nn.ReLU(), nn.Linear(self.width, 1))

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
            treated と control (どちらも (件数, 1) の予測)
        """
        # 2 つの枝の順伝播
        outputs = {TREATED_OUTPUT: self.treated(x),
                   CONTROL_OUTPUT: self.control(x)}

        return outputs


class UpliftTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment,
        valid: dict, effect: dict, cfg: Any
    ):
        # step を上書きするので step_fn は None
        # 親クラスのコンストラクタを呼び出す
        super(UpliftTrainer, self).__init__(model, experiment, None)

        # 検証用の全ての行と, 効果の数え方と, K を保存
        self.valid = valid
        self.effect = effect
        self.k = cfg.uplift.k
        self.batch_size = cfg.data.batch_size

        # 行ごとの損失 (0 と 1 はロジットの二値の交差エントロピー, 量は
        # 標準化した値の二乗誤差)
        self.criterion = (
            nn.BCEWithLogitsLoss(reduction="none")
            if effect["outcome"] == "binary" else nn.MSELoss(reduction="none")
        )

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の, 行の群の予測の損失を求める関数

        Parameters
        ----------
        batch: tuple
            (行の特徴, 処置, 学習の目的の値)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            処置の群の行は treated, 対照の群の行は control の損失の平均
            (loss)
        """
        # 学習のモデルと, 検証の評価のモデル (EMA があれば EMA)
        x, treatment, y = batch
        model = self.model if phase == "train" else self.eval_model()
        treated, control = predict(model, x)
        chosen = torch.where(treatment == 1, treated, control)
        outputs = {"loss": self.criterion(chosen, y).mean()}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 検証用の行の効果の見積もりの Qini の係数を数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証では qini (valid/qini として記録. 大きいほど良い). 他は空
        """
        # 検証用の全ての行の効果の見積もり
        if phase != "valid":
            return {}
        found = estimate(self.eval_model(), self.valid["x"], self.effect,
                         self.batch_size)
        metrics = {"qini": qini_coefficient(found["effect"],
                                            self.valid["t"],
                                            self.valid["y"])}

        return metrics


def predict(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    行の特徴を入力の名前でモデルに渡し, 2 つの群の予測を返す関数

    Parameters
    ----------
    model: nn.Module
        T-learner のモデル
    x: torch.Tensor
        (件数, 学習に使う列の数) の行の特徴

    Returns
    ----------
    predictions: tuple[torch.Tensor, torch.Tensor]
        処置したときとしないときの (件数,) の予測 (0 と 1 ならロジット,
        量なら標準化した値)
    """
    # 出力が複数の定義と仮のモデルは辞書
    outputs = model(**{FEATURE_INPUT: x})
    predictions = (outputs[TREATED_OUTPUT].reshape(-1),
                   outputs[CONTROL_OUTPUT].reshape(-1))

    return predictions


def estimate(
    model: nn.Module, x: torch.Tensor, effect: dict, batch_size: int
) -> dict:
    """
    行ごとの 2 つの群の予測と効果を, まとまりごとに求める関数

    Parameters
    ----------
    model: nn.Module
        T-learner のモデル
    x: torch.Tensor
        (件数, 学習に使う列の数) の行の特徴
    effect: dict
        outcome (binary か amount) と scale (量の標準化の mean と std.
        0 と 1 は None)
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    found: dict
        treated, control (結果の予測. 0 と 1 は確率, 量は元の単位), effect
        (treated − control) の (件数,) の float64 の配列
    """
    # 評価のモードへの切り替え (学習に戻すのは Trainer)
    device = next(model.parameters()).device
    sigmoid = nn.Sigmoid()
    treated, control = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size].to(device)
            pair = predict(model, chunk)
            treated.append(pair[0].cpu())
            control.append(pair[1].cpu())
    treated = torch.cat(treated).double()
    control = torch.cat(control).double()

    # 0 と 1 はシグモイドで確率にしてから, 量は元の単位に戻してからの差
    if effect["outcome"] == "binary":
        treated, control = sigmoid(treated), sigmoid(control)
    else:
        mean, std = effect["scale"]["mean"], effect["scale"]["std"]
        treated, control = treated * std + mean, control * std + mean
    found = {"treated": treated.numpy(), "control": control.numpy(),
             "effect": (treated - control).numpy()}

    return found


def read_splits(dataset: Dataset) -> dict:
    """
    分割ごとの行の特徴, 処置, 結果, 行の id を読む関数

    Parameters
    ----------
    dataset: Dataset
        ML-EDA の介入の記録の表

    Returns
    ----------
    splits: dict
        分割の名前から x (float32 の配列), t (0 か 1 の配列), y (結果の
        float64 の配列. unlabeled は None), ids (行の id)

    Raises
    ----------
    ValueError
        数でない学習に使う列か, 欠けた値がある場合
    """
    # 学習に使う列 (数だけ. 種類の列の one_hot はデータの分析の段階)
    key = dataset.info["uplift"]["id"]
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
        splits[split] = {
            "x": x, "t": dataset.treatment(split),
            "y": None if y is None else np.asarray(y, dtype=np.float64),
            "ids": dataset.frames[split][key].to_numpy(),
        }

    return splits


def check_config(cfg: Any) -> Optional[str]:
    """
    config の効果の見積もりの設定の誤りを返す関数

    Parameters
    ----------
    cfg: Any
        config (uplift と model)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # K は 0 より大きく 1 以下, 配る数の上限は空か 1 以上の整数
    uplift = cfg.uplift
    budget = uplift.get("budget")
    message = None
    if isinstance(uplift.k, bool) or not isinstance(
        uplift.k, (int, float)
    ) or not 0 < uplift.k <= 1:
        message = ("config の uplift.k は 0 より大きく 1 以下の数にして"
                   "ください")
    elif budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
    ):
        message = ("config の uplift.budget は 1 以上の整数 (決めていなければ"
                   " null) にしてください")
    elif "model" not in cfg:
        message = ("config に model がありません. 問題から書き出し直して"
                   "ください")

    return message


def check_splits(splits: dict) -> Optional[str]:
    """
    学習用と検証用の分割に, 処置の群と対照の群があるかを確かめる関数

    Parameters
    ----------
    splits: dict
        read_splits の結果

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # 2 つの群の予測を学ぶ学習用と, 最も良いモデルを選ぶ検証用
    message = None
    for split, title in (("train", "学習用"), ("valid", "検証用")):
        part = splits.get(split)
        groups = set() if part is None else set(part["t"].tolist())
        if part is None or part["y"] is None or groups != {0, 1}:
            message = (f"{title}の分割に処置の群と対照の群の両方がありません. "
                       "検証用の割合を 0 より大きくし, stratified の分割で"
                       "書き出し直してください")
            return message

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

    # 入力は行の特徴, 出力に 2 つの群の予測
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    inputs = loaded.get("inputs") or {}
    if list(inputs) != [FEATURE_INPUT]:
        errors.append(f"定義の入力は {FEATURE_INPUT} の 1 つにしてください "
                      f"(今は {list(inputs)})")
    for name in (TREATED_OUTPUT, CONTROL_OUTPUT):
        if name not in (loaded.get("outputs") or {}):
            errors.append(f"定義の出力に {name} を入れてください")
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
        "モデルの定義を実験に写せなかったので, 学習のプロジェクトの定義から"
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
        config (仮のモデルの隠れ層の幅と乱数の種)
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


def effect_scale(splits: dict, outcome: str) -> Optional[dict]:
    """
    量の結果を学習用の行の平均と標準偏差で標準化する値を求める関数

    Parameters
    ----------
    splits: dict
        read_splits の結果
    outcome: str
        結果の種類 (binary か amount)

    Returns
    ----------
    scale: Optional[dict]
        mean と std (標準偏差が 0 なら 1). 0 と 1 は None
    """
    # 0 と 1 はロジットで学ぶので標準化無し
    if outcome == "binary":
        return None
    y = splits["train"]["y"]
    std = float(y.std())
    scale = {"mean": float(y.mean()), "std": std if std > 0 else 1.0}

    return scale


def loader_of(part: dict, effect: dict, batch_size: int,
              shuffle: bool) -> DataLoader:
    """
    分割の行の特徴, 処置, 学習の目的の値のまとまりを作る関数

    Parameters
    ----------
    part: dict
        分割の x, t, y
    effect: dict
        outcome と scale
    batch_size: int
        まとまりの大きさ
    shuffle: bool
        行を混ぜるか

    Returns
    ----------
    loader: DataLoader
        (行の特徴, 処置, 学習の目的の値) のまとまり
    """
    # 量は標準化した値の学習
    y = part["y"]
    if effect["scale"] is not None:
        y = (y - effect["scale"]["mean"]) / effect["scale"]["std"]
    tensors = TensorDataset(torch.tensor(part["x"]), torch.tensor(part["t"]),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)

    return loader


def train_uplift(
    splits: dict, effect: dict, cfg: Any, spec: Optional[Path],
    experiment: Experiment
) -> tuple[nn.Module, str, Any]:
    """
    学習用の行で, 処置の群と対照の群の結果の予測を学ぶ関数

    Parameters
    ----------
    splits: dict
        read_splits の結果
    effect: dict
        outcome と scale
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
    # 学習用と検証用のまとまり (検証の Qini の係数は検証用の全ての行で)
    train_loader = loader_of(splits["train"], effect, cfg.data.batch_size,
                             True)
    valid_loader = loader_of(splits["valid"], effect, cfg.data.batch_size,
                             False)
    valid = {"x": torch.tensor(splits["valid"]["x"]),
             "t": splits["valid"]["t"], "y": splits["valid"]["y"]}

    # モデルの組み立てと学習 (定義がまだ無ければ仮のモデル)
    spec, base_dir = spec_file(spec, experiment)
    if spec is None:
        experiment.logger.info(
            f"モデルの定義 ({cfg.model.spec or '場所なし'}) が無いので, "
            "仮のモデル (PlaceholderModel) で学習します. モデルの作成・選定の段階で組んだ定義を"
            "その場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(splits["train"]["x"].shape[1], cfg, spec, base_dir)
    trainer = UpliftTrainer(model, experiment, valid, effect, cfg)
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


def effect_summary(effects: np.ndarray) -> dict:
    """
    学習用の行の効果の見積もりの分布をまとめる関数

    Parameters
    ----------
    effects: np.ndarray
        学習用の行の効果の見積もり

    Returns
    ----------
    summary: dict
        rows, mean, min, p10, p50, p90, max, positive (効果が 0 より大きい
        行の割合. 効果の読み方の目安)
    """
    # 分位点は線形の補間
    summary = {"rows": int(len(effects)), "mean": float(effects.mean()),
               "min": float(effects.min())}
    for point in (10, 50, 90):
        summary[f"p{point}"] = float(np.percentile(effects, point))
    summary["max"] = float(effects.max())
    summary["positive"] = float((effects > 0).mean())

    return summary


def finite(value: float) -> Optional[float]:
    """
    有限でない値を空にする関数

    Parameters
    ----------
    value: float
        値

    Returns
    ----------
    number: Optional[float]
        有限ならその値, nan と無限大は None
    """
    # 上位に片方の群が無い場合の nan
    number = float(value) if math.isfinite(value) else None

    return number


def evaluate_splits(
    experiment: Experiment, splits: dict, found: dict, info: dict, cfg: Any
) -> None:
    """
    検証用とテスト用の分割ごとに, 効果の評価を evaluations/ に書く関数

    Parameters
    ----------
    experiment: Experiment
        実験
    splits: dict
        read_splits の結果
    found: dict
        分割の名前から estimate の結果
    info: dict
        weights, epoch, outcome (結果の種類の名前)
    cfg: Any
        config (K と配る数の上限)

    Returns
    ----------
    None
    """
    # Qini の係数と上位 K 割の効果の差の表と, Qini の曲線
    k = cfg.uplift.k
    for split, title in SPLIT_NAMES.items():
        if split not in splits:
            continue
        part = splits[split]
        effects = found[split]["effect"]
        treated = part["t"] == 1
        row = {
            "rows": len(effects), "treated": int(treated.sum()),
            "control": int((~treated).sum()), "k": k,
            "qini": qini_coefficient(effects, part["t"], part["y"]),
            "uplift_at_k": finite(uplift_at_k(effects, part["t"],
                                              part["y"], k)),
            "effect": float(part["y"][treated].mean()
                            - part["y"][~treated].mean()),
            "estimated": float(effects.mean()),
        }
        names = {"split": title, "rows": row["rows"],
                 "treated": row["treated"], "control": row["control"],
                 "k": k, "percent": f"{k * 100:g}",
                 "outcome": info["outcome"],
                 "weights": info["weights"],
                 "budget": cfg.uplift.get("budget")}
        lift = ("無し (上位に片方の群が無い)" if row["uplift_at_k"] is None
                else f"{row['uplift_at_k']:.4f}")
        experiment.logger.info(
            f"{title}の Qini の係数 {row['qini']:.4f}, 上位 {k * 100:g} % の"
            f"効果の差 {lift} (分割の全ての行の差 {row['effect']:.4f})"
        )
        save_evaluation(
            experiment, f"{EVALUATION}_{split}", row, split,
            description=EVALUATION_TEXT.format(**names),
            epoch=info["epoch"], better=EVALUATION_BETTER
        )
        save_evaluation(
            experiment, f"{CURVE}_{split}",
            qini_curve(effects, part["t"], part["y"], points=CURVE_POINTS),
            split, description=CURVE_TEXT.format(**names), kind="curve",
            x="fraction", epoch=info["epoch"]
        )


def write_results(
    run_dir: Path, dataset: Path, splits: dict, found: dict, info: dict,
    effect: dict, cfg: Any
) -> None:
    """
    行ごとの効果, 効果の決め方と前処理のレシピを実験のディレクトリに書く
    関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    dataset: Path
        ML-EDA の介入の記録の表のディレクトリ (recipe.yaml を写す)
    splits: dict
        read_splits の結果
    found: dict
        分割の名前から estimate の結果
    info: dict
        id (行の id の列), features, treatment (処置の列の名前)
    effect: dict
        outcome と scale
    cfg: Any
        config (K と配る数の上限)

    Returns
    ----------
    None
    """
    # 行の id, 分割, 効果, 2 つの群の予測, 処置, 結果の表 (unlabeled の
    # 結果は空欄)
    tables = []
    binary = effect["outcome"] == "binary"
    for split, part in splits.items():
        outcome = ([None] * len(part["t"]) if part["y"] is None
                   else part["y"].tolist())
        tables.append(pd.DataFrame({
            info["id"]: part["ids"], "split": split,
            "effect": found[split]["effect"],
            "treated": found[split]["treated"],
            "control": found[split]["control"],
            "treatment": part["t"],
            "outcome": pd.array(
                [None if v is None else int(v) for v in outcome]
                if binary else outcome,
                dtype="Int64" if binary else "Float64"
            ),
        }))
    pd.concat(tables, ignore_index=True).to_csv(run_dir / EFFECTS_FILE,
                                                index=False)

    # 推論の段階が読む前処理のレシピの写しと効果の決め方 (新しい行は
    # レシピの前処理の後の features から)
    shutil.copy(dataset / "recipe.yaml", run_dir / RECIPE_FILE)
    budget = cfg.uplift.get("budget")
    estimator = {
        "format_version": ESTIMATOR_VERSION, "learner": LEARNER,
        "outcome": effect["outcome"], "recipe": RECIPE_FILE,
        "effects": EFFECTS_FILE, "features": list(info["features"]),
        "id": info["id"], "treatment": info["treatment"],
        "budget": None if budget is None else int(budget),
        "k": float(cfg.uplift.k),
        "model": {"input": FEATURE_INPUT, "treated": TREATED_OUTPUT,
                  "control": CONTROL_OUTPUT,
                  "link": "sigmoid" if binary else "identity"},
        "scale": effect["scale"],
        "train": effect_summary(found["train"]["effect"]),
    }
    text = json.dumps(estimator, ensure_ascii=False, indent=2)
    (run_dir / ESTIMATOR_FILE).write_text(text + "\\n", encoding="utf-8")


def make_inference(
    cfg: Any, spec: Optional[Path], run_dir: Path, experiment: Experiment
) -> None:
    """
    ML-Inference の効果の一式を, 実験の中の inference/ に作る関数

    Parameters
    ----------
    cfg: Any
        config (元の表の場所, 一式の名前)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    run_dir: Path
        実験のディレクトリ (行ごとの効果と効果の決め方を書いた後)
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    None
    """
    # 定義が無い仮のモデルの一式の省略
    if spec is None:
        experiment.logger.info(
            "仮のモデル (PlaceholderModel) は ML-ModelControl の定義が無い"
            "ので, 推論の一式を作りません. config の model.spec の場所に"
            "定義を置くと, 学習の後に作ります"
        )
        return

    # 定義を写せなかった実験の一式の省略 (一式は実験に写した定義を読む)
    if experiment.model_spec_path is None:
        experiment.logger.info(
            "モデルの定義を実験に写せなかったので, 推論の一式を作りません "
            "(理由は source.yaml の model_spec の status)"
        )
        return

    # 元の表 (config の data.source) との効果の比べ (元の表が無ければ
    # 省略)
    data = None
    if cfg.data.source:
        path = PROJECT_DIR / str(cfg.data.source)
        data = path if path.is_file() else None
    if data is None:
        experiment.logger.info(
            f"元の表 (config の data.source: {cfg.data.source}) が無いので,"
            " 推論の一式の効果を実験の効果と比べません"
        )

    # 約束に合わない場合の ML-Inference の誤りの文の警告と, 学習の結果の
    # 保存
    from ml_inference import make_uplift_bundle
    options = {**OPTIONS, **({"loader": LOADER} if LOADER else {})}
    try:
        info = make_uplift_bundle(
            run_dir / INFERENCE_DIR, run_dir, name=cfg.experiment.name,
            data=data, options=options if data is not None else None
        )
    except (ValueError, FileNotFoundError) as error:
        experiment.logger.warning(
            "成果物か定義が効果の一式の約束に合わないので, 推論の一式を"
            f"作れません: {error}"
        )
        return
    check = info["uplift"]["check"]
    compared = ("効果は比べず" if check is None
                else f"効果の一致 {check['agreement'] * 100:.1f} % "
                f"({check['rows']} 行. 差の最大 {check['max_diff']:.3g})")
    decision = info["decision"]
    budget = ("配る数の上限は無し" if decision is None
              else f"配る数の上限 {decision['budget']}")
    experiment.logger.info(
        f"推論の一式 (効果): {INFERENCE_DIR}/ ({compared}. {budget}. 版 "
        f"{info['format_version']})"
    )
    for note in info["notes"]:
        experiment.logger.warning(f"推論の一式の注意: {note}")


def main() -> None:
    """
    介入の記録の表で 2 つの群の予測を学び, 行ごとの効果と評価と成果物を
    残す関数

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
              "介入の記録の表を書き出してください")
        sys.exit(1)
    message = check_config(cfg)
    if message:
        print(message)
        sys.exit(1)
    dataset = load_dataset(path)
    if dataset.info.get("task") != "uplift" or not isinstance(
        dataset.info.get("uplift"), dict
    ):
        print(f"{path} は介入の記録の表ではありません (タスク "
              f"{dataset.info.get('task')}). task: uplift のレシピで書き出"
              "してください")
        sys.exit(1)
    key = str(dataset.info["uplift"]["id"])
    if key in ASSIGNED_COLUMNS:
        print(f"行の id の列 {key} は行ごとの効果の表の列 ("
              f"{', '.join(ASSIGNED_COLUMNS)}) と同じ名前です. 名前を変えて"
              "書き出し直してください")
        sys.exit(1)

    # 分割ごとの行の特徴, 処置, 結果, 行の id (数でない列と欠けた値で停止)
    try:
        splits = read_splits(dataset)
    except ValueError as error:
        print(error)
        sys.exit(1)
    message = check_splits(splits)
    if message:
        print(message)
        sys.exit(1)
    features = list(dataset.features)
    width = len(features)
    outcome = str(dataset.info["uplift"]["outcome"])
    effect = {"outcome": outcome, "scale": effect_scale(splits, outcome)}

    # モデルの定義の確かめ (誤りなら出力先を作る前の停止)
    spec = project_spec(cfg)
    errors = spec_errors(spec, width)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\\n".join(f"- {message}" for message in errors))
        sys.exit(1)

    # 何の実験か (受け持ち, 出力の形, 効果を見積もるもの, 作り方は問題の
    # 設定から)
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    write_task(experiment, {"role": ROLE, "output": OUTPUT,
                            "what": PREDICT, "method": METHOD})
    counts = ", ".join(f"{split} {len(part['t'])} 行 (処置 "
                       f"{int(part['t'].sum())} 行)"
                       for split, part in splits.items())
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}結果の種類: "
        f"{OUTCOME_NAMES[outcome]}. 行: {counts}. 学習に使う列: {width}. "
        f"K: {cfg.uplift.k}. 配る数の上限: {cfg.uplift.get('budget')}"
    )

    # 2 つの群の予測の学習と, 全ての分割の行ごとの効果
    model, weights, epoch = train_uplift(splits, effect, cfg, spec,
                                         experiment)
    found = {split: estimate(model, torch.tensor(part["x"]), effect,
                             cfg.data.batch_size)
             for split, part in splits.items()}

    # 実験だけで見直すための問題の設定の写し (元の問題の場所
    # origin も. モデルの定義の写しは ML-Found の start_experiment)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    print(f"保存先: {run_dir}")

    # 効果の評価の evaluations/ への書き出しと, 行ごとの効果と決め方の
    # 実験への書き出し
    info = {"id": key, "features": features, "weights": weights,
            "epoch": epoch, "outcome": OUTCOME_NAMES[outcome],
            "treatment": str(dataset.info["uplift"]["treatment"])}
    evaluate_splits(experiment, splits, found, info, cfg)
    write_results(run_dir, path, splits, found, info, effect, cfg)
    experiment.logger.info(
        f"行ごとの効果: {EFFECTS_FILE}. 効果の決め方: {ESTIMATOR_FILE} "
        f"(配る数の上限 {cfg.uplift.get('budget')})"
    )

    # 成果物から ML-Inference の推論の一式
    make_inference(cfg, spec, run_dir, experiment)


if __name__ == "__main__":
    main()
'''

# config の基準 (ML-Found のプロジェクトの形)
UPLIFT_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題から書き出した骨組み)
$requires
experiment:
  name: $experiment
  seed: 0

optimizer:
  name: AdamW
  params:
    lr: 0.001

trainer:
  epochs: 30
  grad_clip_norm: 1.0
  # 問題の「最良を選ぶ指標」(valid/qini は検証用の行の効果の見積もり
  # の Qini の係数で, 大きいほど良い. valid/loss は検証用の行の, 処置の群の
  # 行は treated, 対照の群の行は control の予測の損失で, 小さいほど良い)
  monitor: $monitor
  mode: $mode
$tiebreak
# プロジェクト固有の設定
data:
  # ML-EDA の介入の記録の表 (task: uplift. プロジェクトのディレクトリから.
  # 全ての分割に処置の群と対照の群がある)
  dataset: $dataset
  # 入れた表の写し (プロジェクトのディレクトリから. 入れていなければ null)
  source: $source
  # 行のバッチの大きさ
  batch_size: 128

uplift:
  # 評価の上位 K 割の K (効果の見積もりの高い順の上位の行の割合)
  k: $k
  # 配る数の上限 (推論の段階が効果の高い順に上限まで配る. 守る指標)
  budget: $budget

model:
  # ML-ModelControl の形のモデルの定義 (モデルの作成・選定の段階で組んで, この
  # 場所に保存する. 入力は行の特徴 x, 出力に処置したときの予測 treated と
  # しないときの予測 control)
  spec: models/model.yaml
$placeholder  # 仮のモデルの隠れ層の幅
  width: 8
''')

# Qini の係数が並んだときに比べる指標
UPLIFT_TIEBREAK = """  # Qini の係数が並んだときに比べる指標
  monitor_tiebreak: valid/loss
  monitor_tiebreak_mode: min
"""


def uplift_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 行ごとの効果を見積もって評価するスクリプトを作る関数

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
    # 頭と学習の流れ
    names = {item["value"]: item["label"] for item in UPLIFT_OUTCOMES}
    options = get_value(problem, "data.options")
    head = Template(UPLIFT_HEAD).substitute(
        feature_input=literal(FEATURE_INPUT),
        treated_output=literal(TREATED_OUTPUT),
        control_output=literal(CONTROL_OUTPUT),
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        method=literal(active_value(problem, f"{PART}.method")),
        outcome_names=literal(names),
        loader=literal(get_value(problem, "data.loader")),
        options=literal(options if isinstance(options, dict) else {}),
        assigned_columns="[\n" + "".join(
            f"    {json.dumps(name)},\n" for name in UPLIFT_ASSIGNED
        ) + "]",
    )
    # train.log の初めの行の効果を見積もるもの (空なら省略)
    made = ("効果: {PREDICT}. "
            if active_value(problem, f"{PART}.what") else "")
    script = head + Template(UPLIFT_MAIN).substitute(predict_log=made)

    return script


def uplift_config(
    experiment: str, stages: dict, dataset: str, requires: str,
    spec_exists: bool = False, source: Optional[str] = None
) -> str:
    """
    効果の見積もりの config の基準を作る関数

    Parameters
    ----------
    experiment: str
        実験の名前
    stages: dict
        受け渡し
    dataset: str
        介入の記録の表の場所 (プロジェクトのディレクトリから)
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
    # 配る数の上限は推論の配る判断から (損得と制約の直すところが残れば空)
    training = stages["training"]
    decision = stages["inference"]["decision"]
    budget = decision["budget"] if decision else None
    text = UPLIFT_CONFIG.substitute(
        requires=requires, experiment=json.dumps(experiment,
                                                 ensure_ascii=False),
        monitor=training["monitor"], mode=training["mode"],
        tiebreak=(UPLIFT_TIEBREAK if training["monitor"] == "valid/qini"
                  else ""),
        dataset=json.dumps(dataset, ensure_ascii=False),
        source=json.dumps(source, ensure_ascii=False),
        k=json.dumps(stages["uplift"]["k"]), budget=json.dumps(budget),
        placeholder="" if spec_exists else UPLIFT_PLACEHOLDER,
    )

    return text


def uplift_name(problem: dict) -> str:
    """
    効果の見積もりの骨組みの, 実験の名前の元を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    raw: str
        結果の列と受け持ち (purchased_uplift など)
    """
    # 目的の列とタスクの代わり
    raw = f"{active_value(problem, f'{PART}.uplift.outcome')}_uplift"

    return raw


def uplift_readme(
    problem: dict, stages: dict, dataset: str, raw: Optional[str]
) -> str:
    """
    効果の見積もりのプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    dataset: str
        介入の記録の表の場所 (プロジェクトのディレクトリから)
    raw: Optional[str]
        写した表の場所 (プロジェクトのディレクトリから. 入れていなければ
        None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    uplift = stages["uplift"]
    decision = stages["inference"]["decision"]
    names = {item["value"]: item["label"] for item in UPLIFT_OUTCOMES}
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest で問題 (`problem.yaml`) から書き出した, "
             "ML-Found の学習のプロジェクトの骨組み (無作為に配った記録の表で, "
             "顧客ごとの効果を見積もり, 効果の評価と推論の段階が読む成果物を"
             "残す).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1. データ (ML-EDA の介入の記録の表)
    columns = ", ".join(f"`{name}`" for name in uplift["columns"])
    key = (f"`{uplift['id']}`" if uplift["id"]
           else "行の位置 (`row_id`)")
    lines += [
        "", "## 次にすること", "",
        "学習する環境には ML-EDA, ML-Found, ML-ModelControl, ML-Inference "
        "を入れる (config の `requires` の機能. 足りなければ学習の前に入れ方"
        "を示して止まる. ML-ModelControl と ML-Inference は定義を置くと"
        "使う).", "",
        "1. **データの分析** (ML-EDA): ML-EDA の介入の記録の表 "
        "(版 1 の `task: uplift`) を config の `data.dataset` の場所 "
        f"(`{dataset}`) に書き出す. データの分析の段階の「この問題から始める」か, "
        "ML-Quest の `recipe_from` のレシピの雛形で ML-EDA の "
        f"`export_dataset` を使う. 処置の列は `{uplift['treatment']}`, 結果"
        f"の列は `{uplift['outcome']}` ({names[uplift['outcome_kind']]}), "
        f"見積もりに使う列は {columns}, 行の id は {key}. 処置は無作為に"
        "配った記録を前提にする (ML-EDA の `describe_uplift` で群の釣り合い"
        "と, 配った後に決まる列を確かめる). 分割は処置で層を作る "
        "`stratified` で, 全ての分割に処置の群と対照の群が入る. 骨組みは数"
        "の列だけを読むので, 欠けた値は `impute` か `drop_missing` で無く"
        "し, 種類の列は `one_hot` にする"
        + (f". 入れた表はこのディレクトリの `{raw}` に写した" if raw
           else ""),
    ]

    # 2. モデル
    lines.append(
        "2. **モデルの作成・選定** (ML-ModelControl): モデルの作成・選定の段階で組んだ定義を"
        ", config の `model.spec` の"
        "場所 (`models/model.yaml`) に保存する. 入力は行の特徴 `x` ((件"
        "数, 学習に使う列の数) の float32. 処置の列は入れない) の 1 つ, "
        "出力に処置したときの結果の予測 `treated` としないときの予測 "
        "`control` (どちらも (件数, 1). 結果が 0 と 1 ならロジット, 量なら"
        "学習用の行の平均と標準偏差で標準化した値). 学習に使う列の数は "
        "`train.log` の初めの行に出る. ML-ModelControl の同梱の例 "
        "`tabular_uplift` の入力の幅を列の数に合わせて使える. 定義が無い"
        "うちは, `train.py` の `PlaceholderModel` (印 TODO. 動くことを確か"
        "めるだけの小さなモデル) で学習する (config の "
        "`model.placeholder: true` がその目印)"
    )

    # 3. 学習と評価
    monitor = stages["training"]["monitor"]
    lines.append(
        "3. **学習・評価** (ML-Found): `python train.py`. "
        "処置の群の行は `treated`, 対照の群の行は `control` だけに損失 "
        "(0 と 1 はロジットの二値の交差エントロピー, 量は標準化した値の"
        "二乗誤差) を付けて学び, 検証のたびに検証用の行の効果の見積もりの "
        f"Qini の係数を `valid/qini` に記録する. 検証用の `{monitor}` が最も"
        "良いモデルを選ぶ. 行ごとの効果は `treated` から `control` を引いた"
        "値 (0 と 1 はシグモイドで確率にしてから. 量は元の単位に戻してから)"
        ". 学習の後に, 検証用とテスト用の分割ごとに, Qini の係数と上位 K "
        f"割 (K = {uplift['k']} で上位 {uplift['k'] * 100:g} %. config の "
        "`uplift.k`) の効果の差 (上位の行"
        "の処置の群と対照の群の結果の平均の差) を "
        "`evaluations/uplift_<分割>.json` に, Qini の曲線を "
        "`evaluations/qini_curve_<分割>.json` に書く. 結果は "
        "`outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, 学習した"
        "重み `final_model.pt`, 行ごとの効果 `uplift_effects.csv`, 効果の"
        "決め方 `uplift_estimator.json` と前処理のレシピの写し "
        "`uplift_recipe.yaml`)"
    )

    # 4. 推論
    budget = (f"{decision['budget']} 件. config の `uplift.budget`"
              if decision else "損得と制約に直すところが残っているので空")
    lines += [
        "4. **推論** (ML-Inference): 学習の後に, 実験の効果の"
        "決め方 `uplift_estimator.json`, 前処理のレシピの写し "
        "`uplift_recipe.yaml`, 学習した重みと写した定義から, ML-Inference "
        "の効果の一式を実験の `inference/` に作る (config の `data.source` "
        "の元の表があれば, 一式の効果を実験の効果と比べて `train.log` に"
        "書く. 仮のモデルでは作らない). 一式は新しい行にレシピの前処理を"
        "かけて効果を出す. 推論の 1 回で, 効果の高い順に配る数の上限 "
        f"({budget}) まで配る (受け渡しの推論の意思決定 "
        "`{type: treat, budget}` は, ML-Inference の効果の一式が読む形). "
        "効果は見積もりで, 無作為に配った記録を前提にする. 配った効果は, "
        "配る相手を無作為に分けて試して確かめる",
        "", "`train.py` の印 (TODO) の所 (モデル) は書き足す. 問題を変えた"
        "ら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
