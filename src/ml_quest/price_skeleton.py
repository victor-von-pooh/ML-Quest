import json
from string import Template
from typing import Callable, Optional

from ml_quest.handoff import describe
from ml_quest.model_names import FEATURE_INPUT, UNITS_OUTPUT
from ml_quest.problem import active_value, get_value
from ml_quest.questions import PART, PRICE_ASSIGNED

# 学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の機能の名前
# (データセットは ML-EDA で読む. config の model.spec の定義を置くと
# ML-ModelControl の読み込みと, ML-Inference の推論の一式)
PRICE_LIBRARIES = ["eda", "modelcontrol", "inference"]

# データの分析の段階が書き出す, 値段を変えた記録の表の場所 (プロジェクトの
# ディレクトリから)
PRICE_DATASET = "outputs/dataset"

# spec のファイルが無いときの, 仮のモデルで学習することの目印 (config の
# model.placeholder)
PRICE_PLACEHOLDER = """  # spec のファイルが無いうちは train.py の仮のモデルで学習することの目印
  # (学習・評価の段階の学習の準備が読む. train.py は読まず, 仮のモデルを使うか
  # は spec のファイルがあるかで決まる)
  placeholder: true
"""

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
PRICE_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

試しに値段を変えた売れ行きの記録 (ML-EDA の値段を変えた記録の表) で, 値段を
入力の最後の列に入れて売れた数を学び, 商品ごとに値段の候補の中で上限を
超えない, 利益 ((値段 − 原価) × 予測した数) が最も大きい値段を選んで, その
評価 (売れた数の誤差, 値段の弾力性, 見込みの利益の比べ), 推論の段階が読む
成果物と推論の一式を残す骨組みなので, 印 (TODO) の所を書き足す
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
    Experiment, Trainer, load_project_config, price_elasticity, profit_lift,
    save_evaluation, start_experiment, write_task
)

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 予測するもの
PROBLEM_NAME = $name
PREDICT = $predict

# task.yaml の受け持ち, 出力の形と作り方 (問題の設定の答え)
ROLE = "effect"
OUTPUT = "effect"
METHOD = $method

# 元の表の読み方 (問題の手元のデータの読み方. 推論の一式づくりが
# ML-EDA の load_data に渡す. None は ML-EDA の推定)
LOADER = $loader
OPTIONS = $options

# 推論の一式を作る場所 (実験のディレクトリの中)
INFERENCE_DIR = "inference"

# モデルの入力と出力の名前 (ML-ModelControl の同梱の例 tabular_price と
# 同じ) と, 出力の決まり (出力がそのまま売れた数. 最後に softplus)
FEATURE_INPUT = $feature_input
UNITS_OUTPUT = $units_output
LINK = "softplus"

# 商品ごとの選んだ値段と, 値段の決め方 (実験のディレクトリの中)
CHOICES_FILE = "price_choices.csv"
ESTIMATOR_FILE = "price_estimator.json"

# 成果物の版 (price_estimator.json の format_version. price_choices.csv
# の列もこの版に従う) と, 実験に写す前処理のレシピ (ML-EDA の当てはめた値を
# 含む recipe.yaml) の名前
ESTIMATOR_VERSION = 1
RECIPE_FILE = "price_recipe.yaml"

# 商品ごとの選んだ値段の表の, 商品の id と行の id の列にできない名前
ASSIGNED_COLUMNS = $assigned_columns

# 弾力性を数える値段の上げ幅 (今の値段の 1 %)
ELASTICITY_STEP = 0.01

# 評価の名前 (evaluations/<名前>_<分割>.json), 説明と向き
EVALUATION = "price"
EVALUATION_TEXT = (
    "{split}の行 ({rows} 行. 商品 {products}) の, 記録の値段での売れた数の"
    "予測の誤差 (mae, rmse. 小さいほど良い) と, 商品ごとの今の値段 ({current}"
    ") で 1 % 上げたときに予測した数が何 % 変わるかの平均 (elasticity), "
    "値段の候補 {candidates} の中で変える幅の上限 {max_change} を超えない, "
    "見込みの利益が最も大きい値段を選んだときの見込みの利益 (商品ごとの 1 行"
    "あたりの売れた数の予測から数えた, 1 行あたりの利益) の和 (profit) と, "
    "今の値段での和 (current_profit), その差 (profit_gain) と比 "
    "(profit_ratio. 大きいほど良い. 今の利益の和が 0 以下なら空). changed "
    "は値段を変える商品の数 ({weights})"
)
EVALUATION_BETTER = {"mae": "lower", "rmse": "lower",
                     "profit_gain": "higher", "profit_ratio": "higher"}

# 評価する分割の名前
SPLIT_NAMES = {"valid": "検証用", "test": "テスト用"}
'''

# 売れ行きの予測の部品, Trainer, 値段の選び方と評価, 学習の流れ
PRICE_MAIN = '''

class PlaceholderModel(nn.Module):
    def __init__(self, features: int, width: int, scale: float):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 学習に使う列の数 (値段の列を含む), 隠れ層の幅と, 売れた数の目安
        # を保存
        self.features = features
        self.width = width
        self.scale = scale

        # TODO: 課題に合うモデルに置き換える. 今は動くことを確かめるための,
        # 全結合の層と, 負にならない売れた数の softplus
        self.units = nn.Sequential(nn.Linear(self.features, self.width),
                                   nn.ReLU(), nn.Linear(self.width, 1),
                                   nn.Softplus())

        # 初めの出力を学習用の行の売れた数の平均に (softplus の逆の値の
        # 最後の層の偏り)
        with torch.no_grad():
            self.units[2].bias.fill_(
                self.scale + math.log(-math.expm1(-self.scale))
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 学習に使う列の数) の行 (最後の列が値段)

        Returns
        ----------
        outputs: dict
            units ((件数, 1) の売れた数の予測)
        """
        # 売れた数の予測
        outputs = {UNITS_OUTPUT: self.units(x)}

        return outputs


class PriceTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, valid: dict,
        scale: float, cfg: Any
    ):
        # step を上書きするので step_fn は None
        # 親クラスのコンストラクタを呼び出す
        super(PriceTrainer, self).__init__(model, experiment, None)

        # 検証用の全ての行と, 損失で売れた数を割る値を保存
        self.valid = valid
        self.scale = scale
        self.batch_size = cfg.data.batch_size

        # 売れた数と予測を scale で割った値の二乗誤差
        self.criterion = nn.MSELoss()

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の, 売れた数の予測の損失を求める関数

        Parameters
        ----------
        batch: tuple
            (行, 売れた数)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            売れた数と予測を scale で割った値の二乗誤差の平均 (loss)
        """
        # 学習のモデルと, 検証の評価のモデル (EMA があれば EMA)
        x, y = batch
        model = self.model if phase == "train" else self.eval_model()
        outputs = {"loss": self.criterion(predict(model, x) / self.scale,
                                          y / self.scale)}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 検証用の行の売れた数の平均絶対誤差を数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証では mae (valid/mae として記録. 売れた数の単位. 小さいほど
            良い). 他は空
        """
        # 検証用の全ての行の, 記録の値段での売れた数の予測
        if phase != "valid":
            return {}
        found = units_of(self.eval_model(), self.valid["x"],
                         self.batch_size)
        metrics = {"mae": float(np.abs(found - self.valid["y"]).mean())}

        return metrics


def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    行を入力の名前でモデルに渡し, 売れた数の予測を返す関数

    Parameters
    ----------
    model: nn.Module
        売れ行きのモデル
    x: torch.Tensor
        (件数, 学習に使う列の数) の行

    Returns
    ----------
    units: torch.Tensor
        (件数,) の売れた数の予測
    """
    # 仮のモデルの辞書と, 出力が 1 つの定義のテンソル
    outputs = model(**{FEATURE_INPUT: x})
    if isinstance(outputs, dict):
        outputs = outputs[UNITS_OUTPUT]
    units = outputs.reshape(-1)

    return units


def units_of(model: nn.Module, x: Any, batch_size: int) -> np.ndarray:
    """
    行ごとの売れた数の予測を, まとまりごとに求める関数

    Parameters
    ----------
    model: nn.Module
        売れ行きのモデル
    x: Any
        (件数, 学習に使う列の数) の行 (テンソルか配列)
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    units: np.ndarray
        (件数,) の float64 の売れた数の予測 (負の値は 0 に切る)
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
    units = torch.cat(found).double().numpy()
    units = np.clip(units, 0.0, None)

    return units


def read_splits(dataset: Dataset) -> dict:
    """
    分割ごとの行, 売れた数, 元の値段, 原価, 商品の id, 行の id を読む関数

    Parameters
    ----------
    dataset: Dataset
        ML-EDA の値段を変えた記録の表

    Returns
    ----------
    splits: dict
        分割の名前から x (float32 の配列), y (売れた数の float64 の配列.
        unlabeled は None), price (元の値段), cost (原価. 無ければ None),
        products (商品の id. 無ければ行の id), ids (行の id)

    Raises
    ----------
    ValueError
        数でない学習に使う列か, 欠けた値がある場合
    """
    # 学習に使う列 (数だけ. 種類の列の one_hot はデータの分析の段階)
    info = dataset.info["price"]
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
        frame = dataset.frames[split]
        ids = frame[info["id"]].to_numpy()
        splits[split] = {
            "x": x,
            "y": None if y is None else np.asarray(y, dtype=np.float64),
            "price": dataset.price(split), "cost": dataset.cost(split),
            "products": (frame[info["product"]].to_numpy()
                         if info.get("product") else ids),
            "ids": ids,
        }

    return splits


def current_prices(splits: dict, by_product: bool) -> dict:
    """
    商品ごとの今の値段を決める関数

    Parameters
    ----------
    splits: dict
        read_splits の結果
    by_product: bool
        商品の id の列があるか

    Returns
    ----------
    current: dict
        分割の名前から, 行ごとの今の値段 (float64 の配列). 商品の id の列
        があれば全ての分割のその商品の元の値段の中央値, 無ければ行の元の
        値段
    """
    # 全ての分割の記録の値段の, 商品ごとの中央値
    if not by_product:
        current = {split: part["price"].copy()
                   for split, part in splits.items()}
        return current
    records = pd.concat([
        pd.DataFrame({"product": part["products"], "price": part["price"]})
        for part in splits.values()
    ], ignore_index=True)
    medians = records.groupby("product", sort=False)["price"].median()
    current = {split: medians.loc[part["products"]].to_numpy(np.float64)
               for split, part in splits.items()}

    return current


def check_config(cfg: Any) -> Optional[str]:
    """
    config の値段の判断の設定の誤りを返す関数

    Parameters
    ----------
    cfg: Any
        config (price と model)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # 候補は -1 より大きい数の並び (重ならず 0 を含む), 上限は 0 以上の数
    price = cfg.price
    candidates = price.get("candidates")
    limit = price.get("max_change")
    if candidates is not None and not isinstance(candidates, str):
        candidates = list(candidates)
    numbers = isinstance(candidates, list) and all(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and value > -1 for value in candidates
    )
    message = None
    if not numbers or len(set(candidates)) != len(candidates) or (
        0 not in candidates
    ):
        message = ("config の price.candidates は, -1 より大きい数 (今の値段"
                   "に対する変える割合) を重ねずに並べ, 0 を含めてください")
    elif isinstance(limit, bool) or not isinstance(limit, (int, float)) or (
        limit < 0
    ):
        message = ("config の price.max_change は 0 以上の数 (変える割合の"
                   "絶対値の上限) にしてください")
    elif "model" not in cfg:
        message = ("config に model がありません. 問題から書き出し直して"
                   "ください")

    return message


def check_splits(splits: dict) -> Optional[str]:
    """
    学習用と検証用の分割に, 売れた数の分かる行があるかを確かめる関数

    Parameters
    ----------
    splits: dict
        read_splits の結果

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # 売れ行きを学ぶ学習用と, 最も良いモデルを選ぶ検証用
    message = None
    for split, title in (("train", "学習用"), ("valid", "検証用")):
        part = splits.get(split)
        if part is None or part["y"] is None or len(part["y"]) == 0:
            message = (f"{title}の分割に売れた数の分かる行がありません. 検証用"
                       "の割合を 0 より大きくして書き出し直してください")
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
        学習に使う列の数 (値段の列を含む)

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

    # 入力は行, 出力に売れた数の予測
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    inputs = loaded.get("inputs") or {}
    if list(inputs) != [FEATURE_INPUT]:
        errors.append(f"定義の入力は {FEATURE_INPUT} の 1 つにしてください "
                      f"(今は {list(inputs)})")
    if UNITS_OUTPUT not in (loaded.get("outputs") or {}):
        errors.append(f"定義の出力に {UNITS_OUTPUT} を入れてください")
    entry = inputs.get(FEATURE_INPUT)
    shape = entry.get("shape") if isinstance(entry, dict) else None
    if isinstance(shape, list) and shape and shape[-1] != width:
        errors.append(f"入力 {FEATURE_INPUT} の最後の大きさを学習に使う列の"
                      f"数 {width} (値段の列を含む) にしてください (今は "
                      f"{shape[-1]})")

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
    width: int, cfg: Any, spec: Optional[Path], base_dir: Optional[Path],
    scale: float
) -> nn.Module:
    """
    モデルを作る関数

    Parameters
    ----------
    width: int
        学習に使う列の数 (値段の列を含む)
    cfg: Any
        config (仮のモデルの隠れ層の幅と乱数の種)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    base_dir: Optional[Path]
        定義の中の相対の参照の起点 (None なら定義のディレクトリ)
    scale: float
        学習用の行の売れた数の平均 (仮のモデルの初めの出力)

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
    model = PlaceholderModel(width, cfg.model.width, scale)

    return model


def units_scale(splits: dict) -> float:
    """
    損失で売れた数と予測を割る値を求める関数

    Parameters
    ----------
    splits: dict
        read_splits の結果

    Returns
    ----------
    scale: float
        学習用の行の売れた数の平均 (0 なら 1)
    """
    # 売れた数の大きさで学習率を変えずに済むように
    mean = float(splits["train"]["y"].mean())
    scale = mean if mean > 0 else 1.0

    return scale


def loader_of(part: dict, batch_size: int, shuffle: bool) -> DataLoader:
    """
    分割の行と, 売れた数のまとまりを作る関数

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
        (行, 売れた数) のまとまり
    """
    # 売れた数のままの正解 (scale で割るのは損失の中だけ)
    tensors = TensorDataset(
        torch.tensor(part["x"]),
        torch.tensor(part["y"], dtype=torch.float32)
    )
    loader = DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)

    return loader


def train_units(
    splits: dict, scale: float, cfg: Any, spec: Optional[Path],
    experiment: Experiment
) -> tuple[nn.Module, str, Any]:
    """
    学習用の行で, 値段を入れた売れた数の予測を学ぶ関数

    Parameters
    ----------
    splits: dict
        read_splits の結果
    scale: float
        売れた数の戻し方
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
            "仮のモデル (PlaceholderModel) で学習します. モデルの作成・選定の段階で組んだ定義を"
            "その場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(splits["train"]["x"].shape[1], cfg, spec, base_dir,
                       scale)
    trainer = PriceTrainer(model, experiment, valid, scale, cfg)
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


def product_means(products: np.ndarray, values: np.ndarray) -> tuple:
    """
    行ごとの値を, 商品ごとの平均にまとめる関数

    Parameters
    ----------
    products: np.ndarray
        行ごとの商品の id
    values: np.ndarray
        (行の数,) か (行の数, 候補の数) の値

    Returns
    ----------
    keys: np.ndarray
        初めに出た順の商品の id
    means: np.ndarray
        商品ごとの平均 (keys の順)
    rows: np.ndarray
        商品ごとの行の数
    """
    # 初めに出た順の商品の番号
    keys, first, codes = np.unique(products, return_index=True,
                                   return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    codes = rank[codes.reshape(-1)]
    keys = keys[order]

    # 商品ごとの和を行の数で割った平均
    rows = np.bincount(codes, minlength=len(keys)).astype(np.float64)
    flat = values.reshape(len(values), -1)
    sums = np.zeros((len(keys), flat.shape[1]))
    np.add.at(sums, codes, flat)
    means = sums / rows[:, None]
    if values.ndim == 1:
        means = means[:, 0]

    return keys, means, rows.astype(np.int64)


def choose_prices(
    model: nn.Module, dataset: Dataset, split: str, part: dict,
    current: np.ndarray, cfg: Any
) -> dict:
    """
    分割の商品ごとに, 候補の値段での売れた数を予測し, 弾力性と選んだ値段を
    求める関数

    Parameters
    ----------
    model: nn.Module
        売れ行きのモデル
    dataset: Dataset
        ML-EDA の値段を変えた記録の表 (候補の値段での入力)
    split: str
        分割の名前
    part: dict
        read_splits の分割の値
    current: np.ndarray
        行ごとの今の値段
    cfg: Any
        config (値段の候補, 上限, バッチの大きさ)

    Returns
    ----------
    found: dict
        keys, rows, current (商品ごとの今の値段), cost (商品ごとの原価の平均.
        無ければ None), elasticity (price_elasticity の結果), lift
        (profit_lift の結果)
    """
    # 行ごと候補ごとの予測した数 (値段の列だけを候補の値段に置き換え)
    candidates = list(cfg.price.candidates)
    batch = cfg.data.batch_size
    units = np.stack([
        units_of(model, dataset.price_inputs(split, current * (1 + change)),
                 batch)
        for change in candidates
    ], axis=1)
    raised = units_of(
        model, dataset.price_inputs(split, current * (1 + ELASTICITY_STEP)),
        batch
    )

    # 商品ごとの平均 (1 行あたりの売れた数) と, 今の値段と原価
    keys, means, rows = product_means(part["products"], units)
    _, raised, _ = product_means(part["products"], raised)
    _, prices, _ = product_means(part["products"], current)
    cost = (None if part["cost"] is None
            else product_means(part["products"], part["cost"])[1])
    now = means[:, candidates.index(0)]
    found = {
        "keys": keys, "rows": rows, "current": prices, "cost": cost,
        "elasticity": price_elasticity(now, raised),
        "lift": profit_lift(prices, candidates, means, cost=cost,
                            max_change=cfg.price.max_change),
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
    # 予測が 0 の商品だけの弾力性と, 今の利益の和が 0 以下の比の nan
    number = float(value) if math.isfinite(value) else None

    return number


def evaluate_splits(
    experiment: Experiment, splits: dict, chosen: dict, model: nn.Module,
    info: dict, cfg: Any
) -> None:
    """
    検証用とテスト用の分割ごとに, 予測の誤差と値段の選び方の評価を
    evaluations/ に書く関数

    Parameters
    ----------
    experiment: Experiment
        実験
    splits: dict
        read_splits の結果
    chosen: dict
        分割の名前から choose_prices の結果
    model: nn.Module
        売れ行きのモデル
    info: dict
        weights, epoch, current (今の値段の決め方の文)
    cfg: Any
        config (値段の候補と上限)

    Returns
    ----------
    None
    """
    # 記録の値段での売れた数の誤差と, 弾力性と見込みの利益の比べ
    for split, title in SPLIT_NAMES.items():
        if split not in splits:
            continue
        part = splits[split]
        found = units_of(model, part["x"], cfg.data.batch_size)
        errors = found - part["y"]
        lift = chosen[split]["lift"]
        elasticity = chosen[split]["elasticity"]
        row = {
            "rows": len(errors), "products": len(chosen[split]["keys"]),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "elasticity": finite(elasticity["mean"]),
            "elasticity_counted": elasticity["counted"],
            "profit": lift["total"], "current_profit": lift["current_total"],
            "profit_gain": lift["gain"],
            "profit_ratio": finite(lift["ratio"]),
            "changed": lift["changed"],
            "max_change": float(cfg.price.max_change),
        }
        names = {"split": title, "rows": row["rows"],
                 "products": row["products"], "current": info["current"],
                 "candidates": list(cfg.price.candidates),
                 "max_change": cfg.price.max_change,
                 "weights": info["weights"]}
        experiment.logger.info(
            f"{title}の売れた数の平均絶対誤差 {row['mae']:.4f}. 見込みの"
            f"利益 {row['profit']:.4g} (今の値段 {row['current_profit']:.4g}"
            f"). 値段を変える商品 {row['changed']} / {row['products']}"
        )
        save_evaluation(
            experiment, f"{EVALUATION}_{split}", row, split,
            description=EVALUATION_TEXT.format(**names),
            epoch=info["epoch"], better=EVALUATION_BETTER
        )


def write_results(
    run_dir: Path, dataset: Path, chosen: dict, info: dict, cfg: Any,
    scale: float
) -> None:
    """
    商品ごとの選んだ値段, 値段の決め方と前処理のレシピを実験のディレクトリ
    に書く関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    dataset: Path
        ML-EDA の値段を変えた記録の表のディレクトリ (recipe.yaml を写す)
    chosen: dict
        分割の名前から choose_prices の結果
    info: dict
        key (商品の id か行の id の列の名前), features, price, cost,
        product, id (列の名前), current (今の値段の決め方の名前)
    cfg: Any
        config (値段の候補と上限)
    scale: float
        損失で売れた数と予測を割った値 (学習用の行の売れた数の平均)

    Returns
    ----------
    None
    """
    # 分割ごと商品ごとの, 今の値段, 原価, 選んだ割合と値段, 見込みの利益,
    # 弾力性 (原価が無ければ空欄)
    tables = []
    for split, found in chosen.items():
        items = found["lift"]["items"]
        cost = found["cost"]
        tables.append(pd.DataFrame({
            info["key"]: found["keys"], "split": split,
            "rows": found["rows"], "current_price": found["current"],
            "cost": pd.array([None] * len(found["keys"]) if cost is None
                             else cost.tolist(), dtype="Float64"),
            "change": items["change"], "price": items["price"],
            "profit": items["profit"],
            "current_profit": items["current_profit"],
            "elasticity": pd.array(
                [finite(value) for value in
                 found["elasticity"]["items"]["elasticity"]],
                dtype="Float64"
            ),
        }))
    pd.concat(tables, ignore_index=True).to_csv(run_dir / CHOICES_FILE,
                                                index=False)

    # 推論の段階が読む前処理のレシピの写しと値段の決め方 (新しい行は
    # レシピの前処理の後の features から. 値段の列は最後)
    shutil.copy(dataset / "recipe.yaml", run_dir / RECIPE_FILE)
    train = chosen["train"]
    estimator = {
        "format_version": ESTIMATOR_VERSION, "recipe": RECIPE_FILE,
        "choices": CHOICES_FILE, "features": list(info["features"]),
        "price": info["price"], "cost": info["cost"],
        "product": info["product"], "id": info["id"],
        "current": info["current"],
        "candidates": [float(value) for value in cfg.price.candidates],
        "max_change": float(cfg.price.max_change),
        "elasticity_step": ELASTICITY_STEP,
        "model": {"input": FEATURE_INPUT, "units": UNITS_OUTPUT,
                  "link": LINK},
        "scale": scale,
        "train": {"products": len(train["keys"]),
                  "changed": train["lift"]["changed"],
                  "elasticity": finite(train["elasticity"]["mean"])},
    }
    text = json.dumps(estimator, ensure_ascii=False, indent=2)
    (run_dir / ESTIMATOR_FILE).write_text(text + "\\n", encoding="utf-8")


def make_inference(
    cfg: Any, spec: Optional[Path], run_dir: Path, experiment: Experiment
) -> None:
    """
    ML-Inference の値段の一式を, 実験の中の inference/ に作る関数

    Parameters
    ----------
    cfg: Any
        config (元の表の場所, 一式の名前)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    run_dir: Path
        実験のディレクトリ (商品ごとの選んだ値段と値段の決め方を書いた後)
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

    # 元の表 (config の data.source) との選んだ値段の比べ (元の表が無けれ
    # ば省略)
    data = None
    if cfg.data.source:
        path = PROJECT_DIR / str(cfg.data.source)
        data = path if path.is_file() else None
    if data is None:
        experiment.logger.info(
            f"元の表 (config の data.source: {cfg.data.source}) が無いので,"
            " 推論の一式の選んだ値段を実験の選んだ値段と比べません"
        )

    # 約束に合わない場合の ML-Inference の誤りの文の警告と, 学習の結果の
    # 保存
    from ml_inference import make_price_bundle

    options = {**OPTIONS, **({"loader": LOADER} if LOADER else {})}
    try:
        info = make_price_bundle(
            run_dir / INFERENCE_DIR, run_dir, name=cfg.experiment.name,
            data=data, options=options if data is not None else None
        )
    except (ValueError, FileNotFoundError) as error:
        experiment.logger.warning(
            "成果物か定義が値段の一式の約束に合わないので, 推論の一式を"
            f"作れません: {error}"
        )
        return

    check = info["price"]["check"]
    compared = ("選んだ値段は比べず" if check is None
                else f"選んだ値段の一致 {check['agreement'] * 100:.1f} % "
                f"({check['rows']} 件. 見込みの利益の差の最大 "
                f"{check['max_diff']:.3g})")
    decision = info["decision"]
    experiment.logger.info(
        f"推論の一式 (値段): {INFERENCE_DIR}/ ({compared}. 候補 "
        f"{decision['candidates']}, 変える幅の上限 "
        f"{decision['max_change']}. 版 {info['format_version']})"
    )
    for note in info["notes"]:
        experiment.logger.warning(f"推論の一式の注意: {note}")


def main() -> None:
    """
    値段を変えた記録の表で売れ行きを学び, 商品ごとの選んだ値段と評価と
    成果物を残す関数

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
              "値段を変えた記録の表を書き出してください")
        sys.exit(1)
    message = check_config(cfg)
    if message:
        print(message)
        sys.exit(1)
    dataset = load_dataset(path)
    if dataset.info.get("task") != "price" or not isinstance(
        dataset.info.get("price"), dict
    ):
        print(f"{path} は値段を変えた記録の表ではありません (タスク "
              f"{dataset.info.get('task')}). task: price のレシピで書き出"
              "してください")
        sys.exit(1)
    price_info = dataset.info["price"]
    key = str(price_info.get("product") or price_info["id"])
    if key in ASSIGNED_COLUMNS:
        print(f"商品の id か行の id の列 {key} は商品ごとの選んだ値段の表の列"
              f" ({', '.join(ASSIGNED_COLUMNS)}) と同じ名前です. 名前を変えて"
              "書き出し直してください")
        sys.exit(1)

    # 分割ごとの行, 売れた数, 値段, 原価, 商品 (数でない列と欠けた値で
    # 停止) と, 商品ごとの今の値段
    try:
        splits = read_splits(dataset)
    except ValueError as error:
        print(error)
        sys.exit(1)
    message = check_splits(splits)
    if message:
        print(message)
        sys.exit(1)
    by_product = bool(price_info.get("product"))
    current = current_prices(splits, by_product)
    features = list(dataset.features)
    width = len(features)
    scale = units_scale(splits)

    # モデルの定義の確かめ (誤りなら出力先を作る前の停止)
    spec = project_spec(cfg)
    errors = spec_errors(spec, width)
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
        f"{width} (最後が値段の列 {price_info['price']}). 値段の候補: "
        f"{list(cfg.price.candidates)}. 変える幅の上限: "
        f"{cfg.price.max_change}"
    )

    # 売れ行きの学習と, 全ての分割の商品ごとの選んだ値段
    model, weights, epoch = train_units(splits, scale, cfg, spec,
                                        experiment)
    chosen = {split: choose_prices(model, dataset, split, part,
                                   current[split], cfg)
              for split, part in splits.items()}

    # 実験だけで見直すための問題の設定の写し (元の問題の場所
    # origin も. モデルの定義の写しは ML-Found の start_experiment)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    print(f"保存先: {run_dir}")

    # 評価の evaluations/ への書き出しと, 商品ごとの選んだ値段と決め方の
    # 実験への書き出し
    rule = "median" if by_product else "row"
    info = {"key": key, "features": features, "weights": weights,
            "epoch": epoch, "price": str(price_info["price"]),
            "cost": price_info.get("cost"),
            "product": price_info.get("product"),
            "id": str(price_info["id"]), "current": rule,
            "text": ("商品ごとの記録の値段の中央値" if by_product
                     else "行の記録の値段")}
    evaluate_splits(experiment, splits, chosen, model,
                    {**info, "current": info["text"]}, cfg)
    write_results(run_dir, path, chosen, info, cfg, scale)
    experiment.logger.info(
        f"商品ごとの選んだ値段: {CHOICES_FILE}. 値段の決め方: "
        f"{ESTIMATOR_FILE} (変える幅の上限 {cfg.price.max_change})"
    )

    # 成果物から ML-Inference の推論の一式
    make_inference(cfg, spec, run_dir, experiment)


if __name__ == "__main__":
    main()
'''

# config の基準 (ML-Found のプロジェクトの形)
PRICE_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題から書き出した骨組み)
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
  # 問題の「最良を選ぶ指標」(valid/mae は検証用の行の, 記録の値段での
  # 売れた数の平均絶対誤差 (売れた数の単位). valid/loss は検証用の行の,
  # 売れた数と予測を学習用の行の売れた数の平均で割った値の二乗誤差. どちらも
  # 小さいほど良い)
  monitor: $monitor
  mode: $mode

# プロジェクト固有の設定
data:
  # ML-EDA の値段を変えた記録の表 (task: price. プロジェクトのディレクトリ
  # から. 学習に使う列の最後が値段の列)
  dataset: $dataset
  # 入れた表の写し (プロジェクトのディレクトリから. 入れていなければ null)
  source: $source
  # 行のバッチの大きさ
  batch_size: 128

price:
  # 値段の候補 (今の値段に対する変える割合の並び. 0 は今の値段)
  candidates: $candidates
  # 値段を変える幅の上限 (変える割合の絶対値. 守る指標)
  max_change: $max_change

model:
  # ML-ModelControl の形のモデルの定義 (モデルの作成・選定の段階で組んで, この
  # 場所に保存する. 入力は行 x (最後の列が値段), 出力に売れた数の予測
  # units)
  spec: models/model.yaml
$placeholder  # 仮のモデルの隠れ層の幅
  width: 16
''')


def price_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 売れ行きを学んで商品ごとの値段を選ぶスクリプトを作る
    関数

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
    head = Template(PRICE_HEAD).substitute(
        feature_input=literal(FEATURE_INPUT),
        units_output=literal(UNITS_OUTPUT),
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        method=literal(active_value(problem, f"{PART}.method")),
        loader=literal(get_value(problem, "data.loader")),
        options=literal(options if isinstance(options, dict) else {}),
        assigned_columns="[\n" + "".join(
            f"    {json.dumps(name)},\n" for name in PRICE_ASSIGNED
        ) + "]",
    )
    # train.log の初めの行の予測するもの (空なら省略)
    made = ("予測: {PREDICT}. "
            if active_value(problem, f"{PART}.what") else "")
    script = head + Template(PRICE_MAIN).substitute(predict_log=made)

    return script


def price_config(
    experiment: str, stages: dict, dataset: str, requires: str,
    spec_exists: bool = False, source: Optional[str] = None
) -> str:
    """
    値段を変える判断の config の基準を作る関数

    Parameters
    ----------
    experiment: str
        実験の名前
    stages: dict
        受け渡し
    dataset: str
        値段を変えた記録の表の場所 (プロジェクトのディレクトリから)
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
    # 候補と上限は推論の値段の判断から (損得と制約の直すところが残れば空)
    training = stages["training"]
    decision = stages["inference"]["decision"] or {}
    text = PRICE_CONFIG.substitute(
        requires=requires, experiment=json.dumps(experiment,
                                                 ensure_ascii=False),
        monitor=training["monitor"], mode=training["mode"],
        dataset=json.dumps(dataset, ensure_ascii=False),
        source=json.dumps(source, ensure_ascii=False),
        candidates=json.dumps(decision.get("candidates")),
        max_change=json.dumps(decision.get("max_change")),
        placeholder="" if spec_exists else PRICE_PLACEHOLDER,
    )

    return text


def price_name(problem: dict) -> str:
    """
    値段を変える判断の骨組みの, 実験の名前の元を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    raw: str
        売れた数の列と受け持ち (units_price など)
    """
    # 目的の列とタスクの代わり
    raw = f"{active_value(problem, f'{PART}.price.units')}_price"

    return raw


def price_readme(
    problem: dict, stages: dict, dataset: str, raw: Optional[str]
) -> str:
    """
    値段を変える判断のプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    dataset: str
        値段を変えた記録の表の場所 (プロジェクトのディレクトリから)
    raw: Optional[str]
        写した表の場所 (プロジェクトのディレクトリから. 入れていなければ
        None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    price = stages["price"]
    decision = stages["inference"]["decision"]
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest で問題 (`problem.yaml`) から書き出した, "
             "ML-Found の学習のプロジェクトの骨組み (試しに値段を変えた売れ行きの"
             "記録で, 値段を入れて売れた数を学び, 商品ごとに利益が最も大きい"
             "値段を選び, その評価と推論の段階が読む成果物を残す).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1. データ (ML-EDA の値段を変えた記録の表)
    columns = ", ".join(f"`{name}`" for name in price["columns"])
    cost = (f"原価の列は `{price['cost']}`" if price["cost"]
            else "原価の列は無し (利益は売上で数える)")
    product = (f"商品の id は `{price['product']}`" if price["product"]
               else "商品の id は無し (行ごとに値段を選ぶ)")
    lines += [
        "", "## 次にすること", "",
        "学習する環境には ML-EDA, ML-Found, ML-ModelControl, ML-Inference "
        "を入れる (config の `requires` の機能. 足りなければ学習の前に入れ方"
        "を示して止まる. ML-ModelControl と ML-Inference は定義を置くと"
        "使う).", "",
        "1. **データの分析** (ML-EDA): ML-EDA の値段を変えた記録"
        "の表 (版 1 の `task: price`) を config の `data.dataset` の場所 "
        f"(`{dataset}`) に書き出す. データの分析の段階の「この問題から始める」か, "
        "ML-Quest の `recipe_from` のレシピの雛形で ML-EDA の "
        f"`export_dataset` を使う. 値段の列は `{price['price']}`, 売れた数の"
        f"列は `{price['units']}`, {cost}, {product}, 予測に使う列は "
        f"{columns}. 値段は試しに変えた記録 (無作為に近い) を前提にする "
        "(ML-EDA の `describe_price` で, 商品ごとの試した値段と, 値段と他の"
        "列の偏りを確かめる). 分割は値段の段で層を作る `stratified`. 骨組みは"
        "数の列だけを読むので, 欠けた値は `impute` か `drop_missing` で無く"
        "し, 種類の列は `one_hot` にする"
        + (f". 入れた表はこのディレクトリの `{raw}` に写した" if raw
           else ""),
    ]

    # 2. モデル
    lines.append(
        "2. **モデルの作成・選定** (ML-ModelControl): モデルの作成・選定の段階で組んだ定義を"
        ", config の `model.spec` の"
        "場所 (`models/model.yaml`) に保存する. 入力は行 `x` ((件数, 学習に"
        "使う列の数) の float32. 最後の列が前処理をかけた値段) の 1 つ, "
        "出力に売れた数の予測 `units` ((件数, 1). 売れた数のまま. 負に"
        "ならないよう最後に softplus をかける). 学習に"
        "使う列の数は `train.log` の初めの行に出る. ML-ModelControl の同梱"
        "の例 `tabular_price` の入力の幅を列の数に合わせて使える. 定義が無い"
        "うちは, `train.py` の `PlaceholderModel` (印 TODO. 動くことを確か"
        "めるだけの小さなモデル) で学習する (config の "
        "`model.placeholder: true` がその目印)"
    )

    # 3. 学習と評価
    monitor = stages["training"]["monitor"]
    lines.append(
        "3. **学習・評価** (ML-Found): `python train.py`. "
        "売れた数と予測を, 学習用の行の売れた数の平均で割った値の二乗誤差を"
        "損失にして学び, "
        "検証のたびに検証用の行の売れた数の平均絶対誤差を `valid/mae` に"
        f"記録する. 検証用の `{monitor}` が最も良いモデルを選ぶ. 学習の後に"
        ", 商品ごとの今の値段 ("
        + ("商品ごとの, 全ての分割の記録の値段の中央値" if price["product"]
           else "行の記録の値段")
        + ") に候補の割合 (config の `price.candidates`) をかけた値段で値段の"
        "列だけを置き換えて売れた数を予測し (ML-EDA の `price_inputs`), "
        "変える幅の上限 (config の `price.max_change`) を超えない候補の中で"
        "見込みの利益 ((値段 − 原価) × 予測した数) が最も大きい値段を選ぶ "
        "(ML-Found の `profit_lift`). 検証用とテスト用の分割ごとに, 売れた"
        "数の誤差, 値段の弾力性 (ML-Found の `price_elasticity`), 見込みの"
        "利益の比べを `evaluations/price_<分割>.json` に書く. 結果は "
        "`outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, 学習した"
        "重み `final_model.pt`, 商品ごとの選んだ値段 `price_choices.csv`, "
        "値段の決め方 `price_estimator.json` と前処理のレシピの写し "
        "`price_recipe.yaml`)"
    )

    # 4. 推論
    rule = ("損得と制約に直すところが残っているので空" if not decision
            else f"候補 {decision['candidates']}, 変える幅の上限 "
            f"{decision['max_change']}. config の `price`")
    lines += [
        "4. **推論** (ML-Inference): 学習の後に, 実験の値段の"
        "決め方 `price_estimator.json`, 前処理のレシピの写し "
        "`price_recipe.yaml`, 学習した重みと写した定義から, ML-Inference "
        "の値段の一式を実験の `inference/` に作る (config の `data.source` "
        "の元の表があれば, 一式の選んだ値段を実験の選んだ値段と比べて "
        "`train.log` に書く. 仮のモデルでは作らない). 一式は新しい行の値段"
        "の列を今の値段とし, 候補の値段で売れた数を予測して, 上限の中で"
        f"利益が最も大きい値段を選ぶ ({rule}. 受け渡しの推論の意思決定 "
        "`{type: price, candidates, max_change}` は, ML-Inference の値段の"
        "一式が読む形). 選ぶ値段は見積もりで, 試しに値段を変えた記録を前提"
        "にする. 値段を変えた効き目は, 値段を試して確かめる",
        "", "`train.py` の印 (TODO) の所 (モデル) は書き足す. 問題を変えた"
        "ら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
