import json
from string import Template
from typing import Callable, Optional

from ml_quest.handoff import describe
from ml_quest.model_names import FEATURE_INPUT, QUANTILES_OUTPUT
from ml_quest.problem import active_value, get_value
from ml_quest.questions import DEMAND_ASSIGNED, PART

# 学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の機能の名前
# (データセットは ML-EDA で読む. config の model.spec の定義を置くと
# ML-ModelControl の読み込みと, ML-Inference の推論の一式)
DEMAND_LIBRARIES = ["eda", "modelcontrol", "inference"]

# データの分析の段階が書き出す, 需要の記録の表の場所 (プロジェクトの
# ディレクトリから)
DEMAND_DATASET = "outputs/dataset"

# spec のファイルが無いときの, 仮のモデルで学習することの目印 (config の
# model.placeholder)
DEMAND_PLACEHOLDER = """  # spec のファイルが無いうちは train.py の仮のモデルで学習することの目印
  # (ML-Dashboard の学習の準備が読む. train.py は読まず, 仮のモデルを使うか
  # は spec のファイルがあるかで決まる)
  placeholder: true
"""

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
DEMAND_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

商品ごとの日ごとの売れた数の記録 (ML-EDA の需要の記録の表) で, 先の期間の
需要 (売れる数の合計) を水準ごとの分位点で学び, 品切れの損と売れ残りの損の
比から決まる水準 τ の分位点から在庫の数を引いた数を, 上限の中で商品ごとに
発注して, その評価 (0.5 の分位点の誤差, 分位点の損失, 発注の数の損と前の
期間と同じ数の発注の比べ), 推論の段階が読む成果物と推論の一式を残す骨組み
なので, 印 (TODO) の所を書き足す
"""
from decimal import Decimal
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
    Experiment, PinballLoss, Trainer, load_project_config, order_cost,
    pinball_loss, save_evaluation, start_experiment, write_task
)

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 予測するもの
PROBLEM_NAME = $name
PREDICT = $predict

# task.yaml の受け持ち, 出力の形と作り方 (問題の設定の答え)
ROLE = "predict"
OUTPUT = "distribution"
METHOD = $method

# 元の表の読み方 (問題の設定の手元のデータの読み方. 推論の一式づくりが
# ML-EDA の load_data に渡す. None は ML-EDA の推定)
LOADER = $loader
OPTIONS = $options

# 推論の一式を作る場所 (実験のディレクトリの中)
INFERENCE_DIR = "inference"

# モデルの入力と出力の名前 (ML-ModelControl の同梱の例 tabular_demand と
# 同じ. 出力は水準の小さい順の分位点)
FEATURE_INPUT = $feature_input
QUANTILES_OUTPUT = $quantiles_output

# 商品ごとの発注の数と, 需要の決め方 (実験のディレクトリの中)
ORDERS_FILE = "demand_orders.csv"
ESTIMATOR_FILE = "demand_estimator.json"

# 成果物の版 (demand_estimator.json の format_version. demand_orders.csv
# の列もこの版に従う) と, 実験に写す前処理のレシピ (ML-EDA の当てはめた値を
# 含む recipe.yaml) の名前
ESTIMATOR_VERSION = 1
RECIPE_FILE = "demand_recipe.yaml"

# 発注の数の表の, 商品の id と日付の列にできない名前
ASSIGNED_COLUMNS = $assigned_columns

# 0.5 の分位点 (予測の点. 誤差を数える水準)
MEDIAN = 0.5

# 評価の名前 (evaluations/<名前>_<分割>.json), 説明と向き
EVALUATION = "demand"
EVALUATION_TEXT = (
    "{split}の発注を決める日の行 ({rows} 行. 商品 {products}) の, 先の "
    "{horizon} 日の需要の 0.5 の分位点の平均絶対誤差 (mae), 学んだ水準 "
    "{levels} の分位点の損失の平均 (pinball_loss) と, 水準 τ = {level} の"
    "分位点から在庫の数を引き, 0 から上限 {max_order} の間の整数にした発注"
    "の数での実際の損の和 (order_cost. 品切れの損 {shortage} × 足りない数 "
    "+ 売れ残りの損 {holding} × 余った数. 売れる数は在庫の数 + 発注の数) と"
    ", 前の期間と同じ数を発注したときの損の和 (baseline_cost), その差 "
    "(order_cost_saving. 大きいほど良い) と比 (order_cost_ratio. 小さいほど"
    "良い. 前の期間の損が 0 なら空). capped は上限で切った行の数 "
    "({weights})"
)
EVALUATION_BETTER = {"mae": "lower", "pinball_loss": "lower",
                     "order_cost": "lower", "baseline_cost": "lower",
                     "order_cost_saving": "higher",
                     "order_cost_ratio": "lower"}

# 水準ごとの分位点の損失の評価の名前と説明
LEVELS_EVALUATION = "demand_levels"
LEVELS_TEXT = ("{split}の発注を決める日の行の, 学んだ水準ごとの分位点の損失 "
               "(小さいほど良い. {weights})")

# 評価する分割の名前
SPLIT_NAMES = {"valid": "検証用", "test": "テスト用"}
'''

# 需要の予測の部品, Trainer, 発注の数の決め方と評価, 学習の流れ
DEMAND_MAIN = '''

class PlaceholderModel(nn.Module):
    def __init__(self, features: int, width: int, levels: int,
                 scale: float):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 学習に使う列の数, 隠れ層の幅, 水準の数と, 先の期間の需要の目安を
        # 保存
        self.features = features
        self.width = width
        self.levels = levels
        self.scale = scale

        # TODO: 課題に合うモデルに置き換える. 今は動くことを確かめるための,
        # 全結合の層と, 負にならない最も小さい水準の値と隣どうしの差の
        # softplus (前から足して, 水準の順に下がらない分位点にする)
        self.steps = nn.Sequential(nn.Linear(self.features, self.width),
                                   nn.ReLU(),
                                   nn.Linear(self.width, self.levels),
                                   nn.Softplus())

        # 初めの分位点を, 学習用の行の需要の平均の半分から平均の 1.5 倍に
        # 並べる (softplus の逆の値の最後の層の偏り)
        first = 0.5 * self.scale
        step = self.scale / max(self.levels - 1, 1)
        with torch.no_grad():
            for index in range(self.levels):
                value = first if index == 0 else step
                self.steps[2].bias[index] = value + math.log(
                    -math.expm1(-value)
                )

    def forward(self, x: torch.Tensor) -> dict:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 学習に使う列の数) の行

        Returns
        ----------
        outputs: dict
            quantiles ((件数, 水準の数) の水準の小さい順の分位点)
        """
        # 最も小さい水準の値と隣どうしの差を前から足す
        outputs = {QUANTILES_OUTPUT: torch.cumsum(self.steps(x), dim=1)}

        return outputs


class DemandTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, valid: dict,
        levels: list, cfg: Any
    ):
        # step を上書きするので step_fn は None
        # 親クラスのコンストラクタを呼び出す
        super(DemandTrainer, self).__init__(model, experiment, None)

        # 検証用の全ての行と, 学ぶ水準を保存
        self.valid = valid
        self.levels = levels
        self.batch_size = cfg.data.batch_size

        # 水準ごとの分位点の損失の平均 (ML-Found の pinball_loss と同じ数え方)
        self.criterion = PinballLoss(levels)

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の, 先の期間の需要の分位点の損失を求める関数

        Parameters
        ----------
        batch: tuple
            (行, 先の期間の需要)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            水準と行の分位点の損失の平均 (loss)
        """
        # 学習のモデルと, 検証の評価のモデル (EMA があれば EMA)
        x, y = batch
        model = self.model if phase == "train" else self.eval_model()
        outputs = {"loss": self.criterion(y, predict(model, x))}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 検証用の行の 0.5 の分位点の誤差と分位点の損失を数える
        関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証では mae (0.5 の分位点の平均絶対誤差. 需要の単位) と
            pinball_loss (valid/mae, valid/pinball_loss として記録. 小さい
            ほど良い). 他は空
        """
        # 検証用の全ての行の分位点 (負は 0 に切り, 行ごとに小さい順)
        if phase != "valid":
            return {}
        found = quantiles_of(self.eval_model(), self.valid["x"],
                             self.batch_size)
        middle = found[:, self.levels.index(MEDIAN)]
        metrics = {
            "mae": float(np.abs(middle - self.valid["y"]).mean()),
            "pinball_loss": pinball_loss(self.valid["y"], found,
                                         self.levels)["mean"],
        }

        return metrics


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    行を入力の名前でモデルに渡し, 分位点の予測を返す関数

    Parameters
    ----------
    model: nn.Module
        需要のモデル
    x: torch.Tensor
        (件数, 学習に使う列の数) の行

    Returns
    ----------
    quantiles: torch.Tensor
        (件数, 水準の数) の水準の小さい順の分位点
    """
    # 仮のモデルの辞書と, 出力が 1 つの定義のテンソル
    outputs = model(**{FEATURE_INPUT: x})
    if isinstance(outputs, dict):
        outputs = outputs[QUANTILES_OUTPUT]
    quantiles = outputs.reshape(len(x), -1)

    return quantiles


def quantiles_of(model: nn.Module, x: Any, batch_size: int) -> np.ndarray:
    """
    行ごとの分位点の予測を, まとまりごとに求める関数

    Parameters
    ----------
    model: nn.Module
        需要のモデル
    x: Any
        (件数, 学習に使う列の数) の行 (テンソルか配列)
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    values: np.ndarray
        (件数, 水準の数) の float64 の分位点 (負の値は 0 に切り, 行ごとに
        小さい順に並べ直す. ML-Inference の需要の一式と同じ)

    Raises
    ----------
    ValueError
        予測に数でない値がある場合
    """
    # 評価のモードへの切り替え (学習に戻すのは Trainer)
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    device = next(model.parameters()).device
    found = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size].to(device)
            found.append(predict(model, chunk).cpu())
    values = torch.cat(found).double().numpy()
    if not np.isfinite(values).all():
        raise ValueError("分位点の予測に数でない値があります. 学習率を下げる"
                         "か, モデルを見直してください")
    values = np.sort(np.clip(values, 0.0, None), axis=1)

    return values


def read_splits(dataset: Dataset) -> dict:
    """
    分割ごとの行, 先の期間の需要, 前の期間の合計, 在庫の数, 商品の id,
    日付を読む関数

    Parameters
    ----------
    dataset: Dataset
        ML-EDA の需要の記録の表

    Returns
    ----------
    splits: dict
        分割の名前から x (float32 の配列), y (先の期間の需要の float64 の
        配列. unlabeled は None), previous (前の期間の合計), stock (在庫の
        数. 在庫の数の列が無ければ None), products (商品の id), dates
        (発注を決める日)

    Raises
    ----------
    ValueError
        数でない学習に使う列か, 欠けた値がある場合
    """
    # 学習に使う列 (数だけ. 種類の列の one_hot はデータの分析の段階)
    info = dataset.info["demand"]
    splits = {}
    for split in dataset.frames:
        try:
            x, _ = dataset.arrays(split)
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
        frame = dataset.frames[split]
        demand = frame[info["target"]].to_numpy(dtype=np.float64)
        splits[split] = {
            "x": x,
            "y": None if np.isnan(demand).all() else demand,
            "previous": dataset.previous(split),
            "stock": dataset.stock(split),
            "products": frame[info["product"]].astype(str).to_numpy(),
            "dates": pd.to_datetime(frame[info["date"]]).dt.strftime(
                "%Y-%m-%d"
            ).to_numpy(),
        }

    return splits


def order_level(cfg: Any) -> float:
    """
    品切れの損と売れ残りの損から, 発注に使う水準 τ を求める関数

    Parameters
    ----------
    cfg: Any
        config (order の shortage_cost と holding_cost)

    Returns
    ----------
    tau: float
        τ = shortage_cost / (shortage_cost + holding_cost) (10 進の表記
        どおりに求めた値. ML-Inference の発注の判断と同じ)
    """
    # 2 進の丸めを持ち込まない比
    shortage = Decimal(str(cfg.order.shortage_cost))
    holding = Decimal(str(cfg.order.holding_cost))
    tau = float(shortage / (shortage + holding))

    return tau


def is_number(value: Any) -> bool:
    """
    値が真偽値でない有限の数かを返す関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    number: bool
        真偽値でない有限の数か
    """
    # 真偽値を除く数
    number = not isinstance(value, bool) and isinstance(
        value, (int, float)
    ) and math.isfinite(value)

    return number


def check_config(cfg: Any) -> Optional[str]:
    """
    config の学ぶ水準と発注の判断の設定の誤りを返す関数

    Parameters
    ----------
    cfg: Any
        config (demand, order と model)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # 水準は 0 より大きく 1 より小さい数の小さい順の並びで 0.5 を含む, 2 つ
    # の損は正の数, 上限は 0 以上の数
    levels = cfg.demand.get("levels")
    if levels is not None and not isinstance(levels, str):
        levels = list(levels)
    order = cfg.order
    message = None
    if not isinstance(levels, list) or not levels or not all(
        is_number(level) and 0 < level < 1 for level in levels
    ) or any(a >= b for a, b in zip(levels, levels[1:])) or (
        MEDIAN not in levels
    ):
        message = ("config の demand.levels は, 0 より大きく 1 より小さい数"
                   "を小さい順に重ねずに並べ, 0.5 を含めてください")
    elif not all(is_number(order.get(key)) and order.get(key) > 0
                 for key in ("shortage_cost", "holding_cost")):
        message = ("config の order.shortage_cost (品切れの損) と "
                   "order.holding_cost (売れ残りの損) は商品 1 つあたりの正"
                   "の数にしてください")
    elif not is_number(order.get("max_order")) or order.max_order < 0:
        message = ("config の order.max_order は 0 以上の数 (発注の数の上限)"
                   " にしてください")
    elif "model" not in cfg:
        message = ("config に model がありません. 問題の設定から書き出し直して"
                   "ください")

    return message


def check_splits(splits: dict) -> Optional[str]:
    """
    学習用と検証用の分割に, 先の期間の需要の分かる行があるかを確かめる関数

    Parameters
    ----------
    splits: dict
        read_splits の結果

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # 需要を学ぶ学習用と, 最も良いモデルを選ぶ検証用
    message = None
    for split, title in (("train", "学習用"), ("valid", "検証用")):
        part = splits.get(split)
        if part is None or part["y"] is None or len(part["y"]) == 0:
            message = (f"{title}の分割に先の期間の需要の分かる行がありません. "
                       "記録の日数を増やすか, 検証用の割合を 0 より大きくして"
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


def spec_errors(spec: Optional[Path], width: int, levels: int) -> list[str]:
    """
    ML-ModelControl の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)
    width: int
        学習に使う列の数
    levels: int
        学ぶ水準の数

    Returns
    ----------
    errors: list[str]
        誤りの文 (定義が無いか誤りが無ければ空)
    """
    # ML-ModelControl の確かめ (定義の無いうちは読み込み無し. 読めない
    # 定義も例外でなく誤りの文)
    if spec is None:
        return []
    from ml_modelcontrol import build_model, validate
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]
    if errors:
        return errors

    # 入力は行, 出力に分位点
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    inputs = loaded.get("inputs") or {}
    if list(inputs) != [FEATURE_INPUT]:
        errors.append(f"定義の入力は {FEATURE_INPUT} の 1 つにしてください "
                      f"(今は {list(inputs)})")
    if QUANTILES_OUTPUT not in (loaded.get("outputs") or {}):
        errors.append(f"定義の出力に {QUANTILES_OUTPUT} を入れてください")
    entry = inputs.get(FEATURE_INPUT)
    shape = entry.get("shape") if isinstance(entry, dict) else None
    if isinstance(shape, list) and shape and shape[-1] != width:
        errors.append(f"入力 {FEATURE_INPUT} の最後の大きさを学習に使う列の"
                      f"数 {width} にしてください (今は {shape[-1]})")
    if errors:
        return errors

    # 0 の行を 2 件渡した出力の形 (水準の数の列. 外部の重みは読まない)
    model = build_model(spec, seed=0, load_weights=False,
                        base_dir=spec.parent)
    model.eval()
    with torch.no_grad():
        found = predict(model, torch.zeros(2, width))
    if tuple(found.shape) != (2, levels):
        errors.append(f"出力 {QUANTILES_OUTPUT} の形を (件数, 学ぶ水準の数 "
                      f"{levels}) にしてください (今は 2 件で "
                      f"{tuple(found.shape)})")

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
    width: int, levels: int, cfg: Any, spec: Optional[Path],
    base_dir: Optional[Path], scale: float
) -> nn.Module:
    """
    モデルを作る関数

    Parameters
    ----------
    width: int
        学習に使う列の数
    levels: int
        学ぶ水準の数
    cfg: Any
        config (仮のモデルの隠れ層の幅と乱数の種)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    base_dir: Optional[Path]
        定義の中の相対の参照の起点 (None なら定義のディレクトリ)
    scale: float
        学習用の行の先の期間の需要の平均 (仮のモデルの初めの分位点の目安)

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
    model = PlaceholderModel(width, cfg.model.width, levels, scale)

    return model


def demand_scale(splits: dict) -> float:
    """
    仮のモデルの初めの分位点の目安を求める関数

    Parameters
    ----------
    splits: dict
        read_splits の結果

    Returns
    ----------
    scale: float
        学習用の行の先の期間の需要の平均 (0 なら 1)
    """
    # 初めの予測を需要の大きさに合わせ, 短い学習でも分位点に届くように
    mean = float(splits["train"]["y"].mean())
    scale = mean if mean > 0 else 1.0

    return scale


def loader_of(part: dict, batch_size: int, shuffle: bool) -> DataLoader:
    """
    分割の行と, 先の期間の需要のまとまりを作る関数

    Parameters
    ----------
    part: dict
        分割の x, y
    batch_size: int
        まとまりの大きさ
    shuffle: bool
        行を混ぜるか

    Returns
    ----------
    loader: DataLoader
        (行, 先の期間の需要) のまとまり
    """
    # 需要のままの正解
    tensors = TensorDataset(
        torch.tensor(part["x"]),
        torch.tensor(part["y"], dtype=torch.float32)
    )
    loader = DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)

    return loader


def train_demand(
    splits: dict, levels: list, cfg: Any, spec: Optional[Path],
    experiment: Experiment
) -> tuple[nn.Module, str, Any]:
    """
    学習用の行で, 先の期間の需要の分位点を学ぶ関数

    Parameters
    ----------
    splits: dict
        read_splits の結果
    levels: list
        学ぶ水準 (小さい順)
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
    # 学習用と検証用のまとまり (検証の誤差は検証用の全ての行で)
    train_loader = loader_of(splits["train"], cfg.data.batch_size, True)
    valid_loader = loader_of(splits["valid"], cfg.data.batch_size, False)
    valid = {"x": splits["valid"]["x"], "y": splits["valid"]["y"]}

    # モデルの組み立てと学習 (定義がまだ無ければ仮のモデル)
    spec, base_dir = spec_file(spec, experiment)
    if spec is None:
        experiment.logger.info(
            f"モデルの定義 ({cfg.model.spec or '場所なし'}) が無いので, "
            "仮のモデル (PlaceholderModel) で学習します. ML-CreateArchitecture"
            " で組んだ定義をその場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(splits["train"]["x"].shape[1], len(levels), cfg,
                       spec, base_dir, demand_scale(splits))
    trainer = DemandTrainer(model, experiment, valid, levels, cfg)
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


def level_values(levels: list, values: np.ndarray, tau: float) -> np.ndarray:
    """
    水準 τ の分位点を, 学んだ水準の分位点から求める関数

    Parameters
    ----------
    levels: list
        学んだ水準 (小さい順)
    values: np.ndarray
        (行の数, 水準の数) の分位点 (行ごとに小さい順)
    tau: float
        求める水準

    Returns
    ----------
    found: np.ndarray
        行ごとの τ の分位点 (隣の 2 つの水準の間は直線で補い, 学んだ水準の
        範囲の外は端の水準の値. ML-Inference の発注の判断と同じ)
    """
    # 端の水準の外
    points = [float(level) for level in levels]
    if tau <= points[0]:
        found = values[:, 0].copy()
        return found
    if tau >= points[-1]:
        found = values[:, -1].copy()
        return found

    # 隣の 2 つの水準の直線の補い
    upper = int(np.searchsorted(points, tau, side="left"))
    low, high = points[upper - 1], points[upper]
    weight = (tau - low) / (high - low)
    found = (1 - weight) * values[:, upper - 1] + weight * values[:, upper]

    return found


def expected_costs(
    levels: list, values: np.ndarray, supply: np.ndarray, cfg: Any
) -> np.ndarray:
    """
    予測した分位点の分布での, 行ごとの見込みの損を数える関数

    Parameters
    ----------
    levels: list
        学んだ水準 (小さい順)
    values: np.ndarray
        (行の数, 水準の数) の分位点 (行ごとに小さい順)
    supply: np.ndarray
        行ごとの売れる数 (在庫の数 + 発注の数)
    cfg: Any
        config (品切れの損と売れ残りの損)

    Returns
    ----------
    cost: np.ndarray
        行ごとの損の, 分位点の関数 (水準の間は直線, 外は端の水準の値) で
        決まる需要の分布での期待値 (ML-Inference の発注の判断と同じ)
    """
    # 水準 u の区間ごとの, 分位点の関数の両端 (端の外は平らな区間)
    shortage = float(cfg.order.shortage_cost)
    holding = float(cfg.order.holding_cost)
    points = [0.0, *[float(level) for level in levels], 1.0]
    ends = np.concatenate([values[:, :1], values, values[:, -1:]], axis=1)
    cost = np.zeros(len(values))

    # 区間の中で損は需要の直線の 1 つか 2 つの部分 (売れる数を超える点で
    # 分ける) なので, 台形で正しく積む
    for low, high, start, end in zip(points, points[1:], ends.T,
                                     ends.T[1:]):
        width = high - low
        if width <= 0:
            continue
        start_cost = (shortage * np.clip(start - supply, 0.0, None)
                      + holding * np.clip(supply - start, 0.0, None))
        end_cost = (shortage * np.clip(end - supply, 0.0, None)
                    + holding * np.clip(supply - end, 0.0, None))
        crossing = (start < supply) & (supply < end)
        with np.errstate(divide="ignore", invalid="ignore"):
            cut = np.where(crossing, (supply - start) / (end - start), 0.0)
        whole = width * (start_cost + end_cost) / 2
        split = width * (cut * start_cost + (1 - cut) * end_cost) / 2
        cost += np.where(crossing, split, whole)

    return cost


def choose_orders(
    values: np.ndarray, part: dict, levels: list, tau: float, cfg: Any
) -> dict:
    """
    分割の行ごとに, 水準 τ の分位点から在庫の数を引いた発注の数と, 前の
    期間と同じ数の発注を決める関数

    Parameters
    ----------
    values: np.ndarray
        (行の数, 水準の数) の分位点
    part: dict
        read_splits の分割の値
    levels: list
        学んだ水準 (小さい順)
    tau: float
        水準 τ
    cfg: Any
        config (品切れの損, 売れ残りの損, 発注の数の上限)

    Returns
    ----------
    found: dict
        held (在庫の数. 列が無ければ 0), quantile (τ の分位点), order (0
        から上限の間の整数の発注の数), capped (上限を超える数を上限で切った
        印), baseline (前の期間の合計から在庫の数を引いて四捨五入し, 0 から
        上限の間に収めた数)
    """
    # τ の分位点から在庫を引いた数の切り捨てと切り上げ (0 より下は 0) の
    # うち見込みの損の小さい方 (同じなら少ない方)
    held = (np.zeros(len(values)) if part["stock"] is None
            else part["stock"])
    quantile = level_values(levels, values, tau)
    wanted = quantile - held
    low = np.clip(np.floor(wanted), 0, None)
    high = np.clip(np.ceil(wanted), 0, None)
    low_cost = expected_costs(levels, values, held + low, cfg)
    high_cost = expected_costs(levels, values, held + high, cfg)
    wished = np.where(high_cost < low_cost, high, low)

    # 上限での切り (見込みの損は発注の数の凸な関数なので, 上限を超えるなら
    # 上限が上限の中で最も良い) と, 同じ在庫と上限の扱いの前の期間と同じ
    # 数の発注
    top = math.floor(cfg.order.max_order)
    found = {
        "held": held, "quantile": quantile,
        "order": np.minimum(wished, top), "capped": wished > top,
        "baseline": np.clip(np.floor(part["previous"] - held + 0.5), 0, top),
    }

    return found


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
    # 前の期間と同じ数の発注の損の和が 0 の比の nan
    number = float(value) if math.isfinite(value) else None

    return number


def evaluate_splits(
    experiment: Experiment, splits: dict, chosen: dict, levels: list,
    info: dict, cfg: Any
) -> None:
    """
    検証用とテスト用の分割ごとに, 分位点の誤差と発注の数の損の評価を
    evaluations/ に書く関数

    Parameters
    ----------
    experiment: Experiment
        実験
    splits: dict
        read_splits の結果
    chosen: dict
        分割の名前から, values (分位点) と choose_orders の結果
    levels: list
        学んだ水準 (小さい順)
    info: dict
        weights, epoch, horizon, tau
    cfg: Any
        config (損と上限)

    Returns
    ----------
    None
    """
    # 0.5 の分位点の誤差, 分位点の損失と, 実際の需要での発注の数の損
    # (売れる数は在庫の数 + 発注の数)
    for split, title in SPLIT_NAMES.items():
        part = splits.get(split)
        if part is None or part["y"] is None or len(part["y"]) == 0:
            continue
        actual = part["y"]
        found = chosen[split]
        values = found["values"]
        pinball = pinball_loss(actual, values, levels)
        cost = order_cost(
            actual, found["held"] + found["order"],
            shortage_cost=cfg.order.shortage_cost,
            holding_cost=cfg.order.holding_cost,
            baseline=found["held"] + found["baseline"]
        )
        middle = values[:, levels.index(MEDIAN)]
        row = {
            "rows": len(actual), "products": len(set(part["products"])),
            "mae": float(np.abs(middle - actual).mean()),
            "pinball_loss": pinball["mean"], "order_cost": cost["total"],
            "baseline_cost": cost["baseline_total"],
            "order_cost_saving": cost["saving"],
            "order_cost_ratio": finite(cost["ratio"]),
            "capped": int(found["capped"].sum()),
            "level": info["tau"],
            "max_order": float(cfg.order.max_order),
        }
        names = {"split": title, "rows": row["rows"],
                 "products": row["products"], "horizon": info["horizon"],
                 "levels": levels, "level": info["tau"],
                 "max_order": cfg.order.max_order,
                 "shortage": cfg.order.shortage_cost,
                 "holding": cfg.order.holding_cost,
                 "weights": info["weights"]}
        experiment.logger.info(
            f"{title}の 0.5 の分位点の平均絶対誤差 {row['mae']:.4f}, 分位点"
            f"の損失 {row['pinball_loss']:.4f}. 発注の数の損 "
            f"{row['order_cost']:.4g} (前の期間と同じ数の発注 "
            f"{row['baseline_cost']:.4g}). 上限で切った行 {row['capped']}"
        )
        save_evaluation(
            experiment, f"{EVALUATION}_{split}", row, split,
            description=EVALUATION_TEXT.format(**names),
            epoch=info["epoch"], better=EVALUATION_BETTER
        )
        save_evaluation(
            experiment, f"{LEVELS_EVALUATION}_{split}", pinball["levels"],
            split, description=LEVELS_TEXT.format(**names),
            epoch=info["epoch"], better={"loss": "lower"}
        )


def write_results(
    run_dir: Path, dataset: Dataset, path: Path, splits: dict,
    chosen: dict, levels: list, cfg: Any
) -> None:
    """
    商品ごとの発注の数, 需要の決め方と前処理のレシピを実験のディレクトリ
    に書く関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    dataset: Dataset
        ML-EDA の需要の記録の表 (列の名前, 期間, 学習用の分割の要約)
    path: Path
        需要の記録の表のディレクトリ (recipe.yaml を写す)
    splits: dict
        read_splits の結果
    chosen: dict
        分割の名前から, values (分位点) と choose_orders の結果
    levels: list
        学んだ水準 (小さい順)
    cfg: Any
        config (損と上限)

    Returns
    ----------
    None
    """
    # 分割ごと発注を決める日の行ごとの, 在庫の数 (列が無ければ空), τ の
    # 分位点, 発注の数, 前の期間と先の期間の合計 (分からなければ空)
    info = dataset.info["demand"]
    tables = []
    for split, part in splits.items():
        found = chosen[split]
        rows = len(part["x"])
        tables.append(pd.DataFrame({
            info["product"]: part["products"], info["date"]: part["dates"],
            "split": split,
            "stock": pd.array([None] * rows if part["stock"] is None
                              else part["stock"].tolist(), dtype="Float64"),
            "quantile": found["quantile"], "order": found["order"],
            "previous": part["previous"],
            "demand": pd.array([None] * rows if part["y"] is None
                               else part["y"].tolist(), dtype="Float64"),
        }))
    pd.concat(tables, ignore_index=True).to_csv(run_dir / ORDERS_FILE,
                                                index=False)

    # 推論の段階が読む前処理のレシピの写しと需要の決め方 (新しい記録は
    # レシピの前処理の後の features から)
    shutil.copy(path / "recipe.yaml", run_dir / RECIPE_FILE)
    train = info["splits"]["train"]
    estimator = {
        "format_version": ESTIMATOR_VERSION, "recipe": RECIPE_FILE,
        "orders": ORDERS_FILE, "features": list(dataset.features),
        "product": info["product"], "date": info["date"],
        "units": info["units"], "stock": info["stock"], "id": info["id"],
        "horizon": int(info["horizon"]), "interval": int(info["interval"]),
        "levels": [float(level) for level in levels],
        "shortage_cost": cfg.order.shortage_cost,
        "holding_cost": cfg.order.holding_cost,
        "max_order": cfg.order.max_order,
        "model": {"input": FEATURE_INPUT, "quantiles": QUANTILES_OUTPUT},
        "train": {"rows": int(train["rows"]),
                  "products": int(train["products"]),
                  "demand_mean": train["demand_mean"]},
    }
    text = json.dumps(estimator, ensure_ascii=False, indent=2)
    (run_dir / ESTIMATOR_FILE).write_text(text + "\\n", encoding="utf-8")


def make_inference(
    cfg: Any, spec: Optional[Path], run_dir: Path, experiment: Experiment
) -> None:
    """
    ML-Inference の需要の一式を, 実験の中の inference/ に作る関数

    Parameters
    ----------
    cfg: Any
        config (元の表の場所, 一式の名前)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    run_dir: Path
        実験のディレクトリ (商品ごとの発注の数と需要の決め方を書いた後)
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

    # 元の表 (config の data.source) との発注の数の比べ (元の表が無けれ
    # ば省略)
    data = None
    if cfg.data.source:
        path = PROJECT_DIR / str(cfg.data.source)
        data = path if path.is_file() else None
    if data is None:
        experiment.logger.info(
            f"元の表 (config の data.source: {cfg.data.source}) が無いので,"
            " 推論の一式の発注の数を実験の発注の数と比べません"
        )

    # 約束に合わない場合の ML-Inference の誤りの文の警告と, 学習の結果の
    # 保存
    from ml_inference import make_demand_bundle

    options = {**OPTIONS, **({"loader": LOADER} if LOADER else {})}
    try:
        info = make_demand_bundle(
            run_dir / INFERENCE_DIR, run_dir, name=cfg.experiment.name,
            data=data, options=options if data is not None else None
        )
    except (ValueError, FileNotFoundError) as error:
        experiment.logger.warning(
            "成果物か定義が需要の一式の約束に合わないので, 推論の一式を"
            f"作れません: {error}"
        )
        return

    check = info["demand"]["check"]
    compared = ("発注の数は比べず" if check is None
                else f"発注の数の一致 {check['agreement'] * 100:.1f} % "
                f"({check['rows']} 件. τ の分位点の差の最大 "
                f"{check['max_diff']:.3g})")
    decision = info["decision"]
    experiment.logger.info(
        f"推論の一式 (需要): {INFERENCE_DIR}/ ({compared}. 品切れの損 "
        f"{decision['shortage_cost']}, 売れ残りの損 "
        f"{decision['holding_cost']}, 発注の数の上限 "
        f"{decision['max_order']}. 版 {info['format_version']})"
    )
    for note in info["notes"]:
        experiment.logger.warning(f"推論の一式の注意: {note}")


def main() -> None:
    """
    需要の記録の表で先の期間の需要の分位点を学び, 商品ごとの発注の数と
    評価と成果物を残す関数

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
              "需要の記録の表を書き出してください")
        sys.exit(1)
    message = check_config(cfg)
    if message:
        print(message)
        sys.exit(1)
    dataset = load_dataset(path)
    if dataset.info.get("task") != "demand" or not isinstance(
        dataset.info.get("demand"), dict
    ):
        print(f"{path} は需要の記録の表ではありません (タスク "
              f"{dataset.info.get('task')}). task: demand のレシピで書き出"
              "してください")
        sys.exit(1)
    demand_info = dataset.info["demand"]
    taken = [str(demand_info[key]) for key in ("product", "date")
             if str(demand_info[key]) in ASSIGNED_COLUMNS]
    if taken:
        print(f"商品の id か日付の列 {', '.join(taken)} は発注の数の表の列 "
              f"({', '.join(ASSIGNED_COLUMNS)}) と同じ名前です. 名前を変えて"
              "書き出し直してください")
        sys.exit(1)

    # 分割ごとの行, 需要, 前の期間の合計, 在庫の数, 商品 (数でない列と
    # 欠けた値で停止)
    try:
        splits = read_splits(dataset)
    except ValueError as error:
        print(error)
        sys.exit(1)
    message = check_splits(splits)
    if message:
        print(message)
        sys.exit(1)
    levels = [float(level) for level in cfg.demand.levels]
    tau = order_level(cfg)
    width = len(dataset.features)

    # モデルの定義の確かめ (誤りなら出力先を作る前の停止)
    spec = project_spec(cfg)
    errors = spec_errors(spec, width, len(levels))
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\\n".join(f"- {message}" for message in errors))
        sys.exit(1)

    # 何の実験か (受け持ち, 出力の形, 予測するもの, 作り方は問題の設定
    # から)
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    write_task(experiment, {"role": ROLE, "output": OUTPUT,
                            "what": PREDICT, "method": METHOD})
    counts = ", ".join(f"{split} {len(part['x'])} 行"
                       for split, part in splits.items())
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}行: {counts}. 学習に使う列: "
        f"{width}. 先の期間: {demand_info['horizon']} 日. 学ぶ水準: "
        f"{levels}. 発注の水準 τ: {tau:.4g}. 発注の数の上限: "
        f"{cfg.order.max_order}"
    )
    if tau not in levels:
        experiment.logger.warning(
            f"発注の水準 τ ({tau:.4g}) が学ぶ水準に無いので, "
            + ("隣の 2 つの水準の分位点を直線で補います"
               if levels[0] < tau < levels[-1]
               else "端の水準の分位点を使います (分布の端は学んでいないので"
               ", 見込みの損は少なめに出ます)")
        )

    # 需要の学習と, 全ての分割の発注を決める日の行ごとの発注の数
    model, weights, epoch = train_demand(splits, levels, cfg, spec,
                                         experiment)
    chosen = {}
    for split, part in splits.items():
        values = quantiles_of(model, part["x"], cfg.data.batch_size)
        chosen[split] = {"values": values,
                         **choose_orders(values, part, levels, tau, cfg)}

    # 実験だけで見直すための問題の設定の写し (元の問題の設定の場所
    # origin も. モデルの定義の写しは ML-Found の start_experiment)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    print(f"保存先: {run_dir}")

    # 評価の evaluations/ への書き出しと, 発注の数と需要の決め方の実験への
    # 書き出し
    info = {"weights": weights, "epoch": epoch, "tau": tau,
            "horizon": demand_info["horizon"]}
    evaluate_splits(experiment, splits, chosen, levels, info, cfg)
    write_results(run_dir, dataset, path, splits, chosen, levels, cfg)
    experiment.logger.info(
        f"商品ごとの発注の数: {ORDERS_FILE}. 需要の決め方: "
        f"{ESTIMATOR_FILE} (発注の数の上限 {cfg.order.max_order})"
    )

    # 成果物から ML-Inference の推論の一式
    make_inference(cfg, spec, run_dir, experiment)


if __name__ == "__main__":
    main()
'''

# config の基準 (ML-Found のプロジェクトの形)
DEMAND_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題の設定から書き出した骨組み)
$requires
experiment:
  name: $experiment
  seed: 0

optimizer:
  name: AdamW
  params:
    lr: 0.01

trainer:
  epochs: 30
  grad_clip_norm: 1.0
  # 問題の設定の「最良を選ぶ指標」(valid/loss は検証用の行の, 学んだ水準の
  # 分位点の損失の平均. valid/mae は検証用の行の, 0.5 の分位点の平均絶対
  # 誤差 (先の期間の需要の単位). どちらも小さいほど良い)
  monitor: $monitor
  mode: $mode

# プロジェクト固有の設定
data:
  # ML-EDA の需要の記録の表 (task: demand. プロジェクトのディレクトリから.
  # 1 行が商品と発注を決める日)
  dataset: $dataset
  # 入れた表の写し (プロジェクトのディレクトリから. 入れていなければ null)
  source: $source
  # 行のバッチの大きさ
  batch_size: 128

demand:
  # 学ぶ分位点の水準 (小さい順. 発注に使う水準 τ と 0.5 を含み, 同じ水準は
  # 1 つ. モデルの出力 quantiles の列の順)
  levels: $levels

order:
  # 1 つ足りないときの損と 1 つ余ったときの損 (商品 1 つあたりの同じお金の
  # 単位. 発注に使う水準 τ = shortage_cost / (shortage_cost + holding_cost))
  shortage_cost: $shortage_cost
  holding_cost: $holding_cost
  # 発注の数の上限 (守る指標. 置き場所の上限)
  max_order: $max_order

model:
  # ML-ModelControl の形のモデルの定義 (ML-CreateArchitecture で組んで, この
  # 場所に保存する. 入力は行 x, 出力に水準の小さい順の分位点 quantiles)
  spec: models/model.yaml
$placeholder  # 仮のモデルの隠れ層の幅
  width: 16
''')


def demand_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 先の期間の需要の分位点を学んで商品ごとの発注の数を決める
    スクリプトを作る関数

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
    options = get_value(problem, "data.options")
    head = Template(DEMAND_HEAD).substitute(
        feature_input=literal(FEATURE_INPUT),
        quantiles_output=literal(QUANTILES_OUTPUT),
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        method=literal(active_value(problem, f"{PART}.method")),
        loader=literal(get_value(problem, "data.loader")),
        options=literal(options if isinstance(options, dict) else {}),
        assigned_columns="[\n" + "".join(
            f"    {json.dumps(name)},\n" for name in DEMAND_ASSIGNED
        ) + "]",
    )
    # train.log の初めの行の予測するもの (空なら省略)
    made = ("予測: {PREDICT}. "
            if active_value(problem, f"{PART}.what") else "")
    script = head + Template(DEMAND_MAIN).substitute(predict_log=made)

    return script


def demand_config(
    experiment: str, stages: dict, dataset: str, requires: str,
    spec_exists: bool = False, source: Optional[str] = None
) -> str:
    """
    時間をまたぐ量の判断の config の基準を作る関数

    Parameters
    ----------
    experiment: str
        実験の名前
    stages: dict
        受け渡し
    dataset: str
        需要の記録の表の場所 (プロジェクトのディレクトリから)
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
    # 損と上限は推論の発注の判断から (損得と制約の直すところが残れば空).
    # 水準は受け渡しの学ぶ水準
    training = stages["training"]
    decision = stages["inference"]["decision"] or {}
    text = DEMAND_CONFIG.substitute(
        requires=requires, experiment=json.dumps(experiment,
                                                 ensure_ascii=False),
        monitor=training["monitor"], mode=training["mode"],
        dataset=json.dumps(dataset, ensure_ascii=False),
        source=json.dumps(source, ensure_ascii=False),
        levels=json.dumps(stages["demand"]["levels"]),
        shortage_cost=json.dumps(decision.get("shortage_cost")),
        holding_cost=json.dumps(decision.get("holding_cost")),
        max_order=json.dumps(decision.get("max_order")),
        placeholder="" if spec_exists else DEMAND_PLACEHOLDER,
    )

    return text


def demand_name(problem: dict) -> str:
    """
    時間をまたぐ量の判断の骨組みの, 実験の名前の元を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    raw: str
        日ごとの売れた数の列と受け持ち (units_demand など)
    """
    # 目的の列とタスクの代わり
    raw = f"{active_value(problem, f'{PART}.demand.units')}_demand"

    return raw


def demand_readme(
    problem: dict, stages: dict, dataset: str, raw: Optional[str]
) -> str:
    """
    時間をまたぐ量の判断のプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    dataset: str
        需要の記録の表の場所 (プロジェクトのディレクトリから)
    raw: Optional[str]
        写した表の場所 (プロジェクトのディレクトリから. 入れていなければ
        None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    demand = stages["demand"]
    decision = stages["inference"]["decision"]
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, "
             "ML-Found のプロジェクトの骨組み (商品ごとの日ごとの売れた数の"
             "記録で, 先の期間の需要の分位点を学び, 品切れの損と売れ残りの損"
             "の比から決まる水準の分位点から在庫を引いて商品ごとの発注の数を"
             "決め, その評価と推論の段階が読む成果物を残す).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1. データ (ML-EDA の需要の記録の表)
    columns = (", ".join(f"`{name}`" for name in demand["columns"])
               if demand["columns"] else "無し")
    stock = (f"在庫の数の列は `{demand['stock']}`" if demand["stock"]
             else "在庫の数の列は無し (在庫を 0 として発注する)")
    lines += [
        "", "## 次にすること", "",
        "学習する環境には ML-EDA, ML-Found, ML-ModelControl, ML-Inference "
        "を入れる (config の `requires` の機能. 足りなければ学習の前に入れ方"
        "を示して止まる. ML-ModelControl と ML-Inference は定義を置くと"
        "使う).", "",
        "1. **データの分析** (ML-EDA, ML-Fathom): ML-EDA の需要の記録の表 "
        "(版 1 の `task: demand`) を config の `data.dataset` の場所 "
        f"(`{dataset}`) に書き出す. ML-Fathom の「問題の設定から始める」か, "
        "ML-Quest の `recipe_from` のレシピの雛形で ML-EDA の "
        f"`export_dataset` を使う. 商品の id の列は `{demand['product']}`, "
        f"日付の列は `{demand['date']}`, 日ごとの売れた数の列は "
        f"`{demand['units']}`, {stock}, 商品の特徴の列は {columns}, 先の期間"
        f"は {demand['horizon']} 日. 1 件は商品と発注を決める日の 1 行で, "
        "正解は先の期間の売れた数の合計 (`demand`), 比べる相手は前の期間の"
        "合計 (`demand_previous`). 過去の売れた数は `lag` でずらした列にし,"
        " 曜日は `datetime_parts` で列にする (ML-EDA の `describe_demand` "
        "で, 商品ごとの記録の日数, 抜けの日と在庫が 0 の日を確かめる). 分割"
        "は発注を決める日の順の `time`. 骨組みは数の列だけを読むので, 欠けた"
        "値は `impute` か `drop_missing` で無くし, 種類の列は `one_hot` に"
        "する"
        + (f". 入れた表はこのディレクトリの `{raw}` に写した" if raw
           else ""),
    ]

    # 2. モデル
    lines.append(
        "2. **モデルの作成** (ML-ModelControl, ML-CreateArchitecture): "
        "ML-CreateArchitecture で組んだ定義を, config の `model.spec` の"
        "場所 (`models/model.yaml`) に保存する. 入力は行 `x` ((件数, 学習に"
        "使う列の数) の float32) の 1 つ, 出力に水準の小さい順の分位点 "
        f"`quantiles` ((件数, 学ぶ水準の数 {len(demand['levels'])}). 先の"
        "期間の需要のまま). 学習に使う列の数は `train.log` の初めの行に出る"
        ". ML-ModelControl の同梱の例 `tabular_demand` の入力の幅を列の数に,"
        " 最後の全結合の出力の幅を水準の数に合わせて使える. 定義が無いうち"
        "は, `train.py` の `PlaceholderModel` (印 TODO. 動くことを確かめる"
        "だけの小さなモデル) で学習する (config の `model.placeholder: true`"
        " がその目印)"
    )

    # 3. 学習と評価
    monitor = stages["training"]["monitor"]
    lines.append(
        "3. **学習と評価** (ML-Found, ML-Dashboard): `python train.py`. "
        f"学ぶ水準 {demand['levels']} (config の `demand.levels`. 発注に使う"
        "水準 τ と 0.5 を含み, 同じ水準は 1 つにまとめる) の分位点の損失を, "
        "学習の損失 (ML-Found の `PinballLoss`) にして学び, 検証のたびに検証"
        "用の行の 0.5 の分位点の平均絶対誤差を `valid/mae` に, 分位点の損失 "
        "(ML-Found の `pinball_loss`) を `valid/pinball_loss` に記録する. "
        f"検証用の `{monitor}` が最も良いモデルを選ぶ. 学習の後に, 発注を"
        "決める日の行ごとに, 水準 τ (config の `order` の品切れの損 / (品切"
        "れの損 + 売れ残りの損)) の分位点から在庫の数を引き, 0 から上限 "
        "(config の `order.max_order`) の間の整数で発注の数を決める. 検証用"
        "とテスト用の分割ごとに, 0.5 の分位点の誤差, 分位点の損失, 実際の"
        "需要での発注の数の損と前の期間と同じ数の発注の損の比べ (ML-Found "
        "の `order_cost`) を `evaluations/demand_<分割>.json` に, 水準ごとの"
        "分位点の損失を `evaluations/demand_levels_<分割>.json` に書く. 結果"
        "は `outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, 学習"
        "した重み `final_model.pt`, 商品ごとの発注の数 `demand_orders.csv`, "
        "需要の決め方 `demand_estimator.json` と前処理のレシピの写し "
        "`demand_recipe.yaml`)"
    )

    # 4. 推論
    rule = ("損得と制約に直すところが残っているので空" if not decision
            else f"品切れの損 {decision['shortage_cost']}, 売れ残りの損 "
            f"{decision['holding_cost']}, 発注の数の上限 "
            f"{decision['max_order']}. config の `order`")
    lines += [
        "4. **推論** (ML-Inference, ML-Provide): 学習の後に, 実験の需要の"
        "決め方 `demand_estimator.json`, 前処理のレシピの写し "
        "`demand_recipe.yaml`, 学習した重みと写した定義から, ML-Inference "
        "の需要の一式を実験の `inference/` に作る (config の `data.source` "
        "の元の表があれば, 一式の発注の数を実験の発注の数と比べて "
        "`train.log` に書く. 仮のモデルでは作らない). 一式は新しい日ごとの"
        "記録の商品ごとに最も遅い日を発注を決める日とし, 先の期間の需要の"
        "分位点から在庫を引いて, 上限の中で発注する "
        f"({rule}. 受け渡しの推論の意思決定 "
        "`{type: order, shortage_cost, holding_cost, max_order}` は, "
        "ML-Inference の需要の一式が読む形). 在庫が 0 の日の売れた数は需要"
        "より少ないので, 品切れの多い商品の分位点は少なめに出る. 実際の損"
        "は発注の後の売れた数で確かめる",
        "", "`train.py` の印 (TODO) の所 (モデル) は書き足す. 問題の設定を"
        "変えたら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
