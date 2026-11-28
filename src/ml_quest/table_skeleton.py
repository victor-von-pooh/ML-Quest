import json
from string import Template
from typing import Callable, Optional

from ml_quest.generate_skeleton import generation_origin, origin_readme
from ml_quest.handoff import describe
from ml_quest.problem import active_value
from ml_quest.questions import PART

# 学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の機能の名前
# (画像の生成と同じ. config の model.spec の定義が無いうちは
# ml_modelcontrol と ml_inference を読まない)
TABLE_LIBRARIES = ["eda", "modelcontrol", "inference"]

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
TABLE_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

ML-EDA の表のデータセットで, 条件の無い拡散のモデル (数の列を標準化し,
種類の列をワンホットにした行に雑音を加え, 加えた雑音を当てる) を学び,
元の表に似た行を作る骨組みなので, 印 (TODO) の所を書き足す
"""
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from ml_eda import Dataset, load_dataset
from ml_found import (
    Experiment, Trainer, load_project_config, save_evaluation, save_sample,
    start_experiment, write_task
)
from ml_found.experiment import task as found_task

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 作るもの
PROBLEM_NAME = $name
PREDICT = $predict

# task.yaml の受け持ち, 出力の形と作り方 (問題の設定の答え)
ROLE = "generate"
OUTPUT = "table"
METHOD = $method

# task.yaml の正解の種類と出どころの種類 (問題の設定の正解の出どころが
# 自分自身なら, 学んだ表そのものが正解なので self. 他は unknown)
TARGET_KIND = "table"
ORIGIN_TYPE = $origin_type

# 推論の一式の出力の形 (ML-Inference の表の出力の雑音だけの一式. 列の
# 符号化を一式に写し, 列の名前と値の表を返す)
BUNDLE_FORM = "table"

# 列の符号化の記録 (実験のディレクトリの中. 推論の一式に写す)
COLUMNS_FILE = "table_columns.json"

# 検証用とテスト用の評価の名前 (evaluations/<名前>_<分割>.json), 説明,
# 指標の向き
EVALUATION = "column_distance"
EVALUATION_TEXT = ("作った表と{split}の本物の表の列ごとの分布の近さ ({weights}. "
                   "本物と同じ行の数を見本と同じ種で作る. distance は数の列が "
                   "KS 統計量, 種類の列が全変動距離で, 0 から 1. wasserstein "
                   "は数の列の Wasserstein 距離を学習用の標準偏差で割った値. "
                   "train_distance は学習用の本物との distance で, 同じ分布の"
                   "行の数の違いから出る目安)")
EVALUATION_BETTER = {"distance": "lower", "wasserstein": "lower"}

# 全ての列の平均の行の名前
MEAN_ROW = "(全ての列の平均)"

# 種類の列の値の種類の上限 (超える列は ID のような列とみなして止まる)
MAX_CATEGORIES = 100

# 推論の一式の予測の表の列と重なる列の名前, 頭と終わり (ML-Inference の
# 表の出力の一式が使えない名前. 学習の前に止まる)
RESERVED_COLUMNS = ("sample", "seed", "status", "reason", "decision",
                    "output", "artifact", "tokens", "prediction",
                    "confidence")
RESERVED_PREFIXES = ("p:", "q:")
RESERVED_SUFFIX = " (入力)"

# 学習の後に作る見本の行の数 (見比べる検証用の本物の行も同じ数まで)
SAMPLES = 100
'''

# 拡散のモデルの部品, 列の符号化と分布の近さ, Trainer, 学習の流れ
TABLE_MAIN = '''

class PlaceholderModel(nn.Module):
    def __init__(self, columns: int, width: int):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 符号化した行の幅と, 中の幅を保存
        self.columns = columns
        self.width = width

        # TODO: 課題に合うモデル (幅と段の多い全結合など) に置き換える. 今は
        # 動くことを確かめるための, 時刻の埋め込みを足す全結合 3 段
        self.time_fc = nn.Linear(self.width, self.width)
        self.fc1 = nn.Linear(self.columns, self.width)
        self.fc2 = nn.Linear(self.width, self.width)
        self.fc3 = nn.Linear(self.width, self.columns)

        # 活性化関数の定義
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 幅) の雑音を加えた行
        t: torch.Tensor
            (バッチ,) の時刻 (0 から学習の時刻の数 - 1 の整数)

        Returns
        ----------
        noise: torch.Tensor
            (バッチ, 幅) の加えた雑音の予測
        """
        # 時刻の正弦波の埋め込みを, 1 段目の全結合の後に足す
        emb = self.time_fc(time_embedding(t, self.width))
        h = self.silu(self.fc1(x) + emb)
        h = self.silu(self.fc2(h))
        noise = self.fc3(h)

        return noise


class TableDiffusionTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, inputs: tuple,
        schedule: torch.Tensor, columns: list, real: pd.DataFrame, cfg: Any
    ):
        # 親クラスのコンストラクタを呼び出す (step を上書きするので
        # step_fn は None)
        super(TableDiffusionTrainer, self).__init__(model, experiment, None)

        # モデルの入力の名前 (雑音を加えた行, 時刻), 時刻ごとの累積の
        # alpha, 検証の乱数の種と, 検証で毎回同じ時刻と雑音を引く乱数の
        # 生成器を保存
        self.inputs = inputs
        self.schedule = schedule
        self.seed = cfg.experiment.seed
        self.generator = torch.Generator()

        # 列の符号化, 分布の近さを比べる検証用の本物の表, 作る繰り返しの
        # 回数と種, バッチの大きさを保存
        self.columns = columns
        self.real = real
        self.sample_steps = cfg.diffusion.sample_steps
        self.sample_seed = cfg.diffusion.sample_seed
        self.batch_size = cfg.data.batch_size

    def on_epoch_start(self, phase: str) -> None:
        """
        検証の前に, 検証の乱数を同じ状態に戻す関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        None
        """
        # 検証の損失をエポックの間で同じ時刻と雑音で比べるため
        if phase == "valid":
            self.generator.manual_seed(self.seed)

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の雑音の予測の損失を求める関数

        Parameters
        ----------
        batch: tuple
            (符号化した行,)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            加えた雑音との平均二乗誤差 (loss)
        """
        # 学習は全体の乱数, 検証は決まった乱数で時刻と雑音を選ぶ. 学習は
        # 学習しているモデル, 検証は評価のモデル (EMA があれば EMA)
        x0 = batch[0]
        generator = self.generator if phase == "valid" else None
        model = self.model if phase == "train" else self.eval_model()
        loss = noise_loss(model, self.inputs, self.schedule, x0, generator)
        outputs = {"loss": loss}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 作った表と検証用の本物の表の列ごとの分布の近さを
        数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証なら column_distance (valid/column_distance として記録.
            小さいほど良い). 他は空
        """
        # 見本と同じ種と回数で本物と同じ行の数を作る (始まりの雑音が同じ
        # なのでエポックの間で比べられる)
        if phase != "valid":
            return {}
        made = generate_table(self.eval_model(), self.inputs, self.schedule,
                              self.columns, len(self.real), self.sample_seed,
                              self.sample_steps, self.batch_size)
        rows = column_distances(self.real, made, self.columns)
        metrics = {"column_distance": mean_distance(rows)}

        return metrics


def time_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    """
    時刻を正弦波の埋め込みにする関数

    Parameters
    ----------
    t: torch.Tensor
        (バッチ,) の時刻
    width: int
        埋め込みの幅 (偶数)

    Returns
    ----------
    emb: torch.Tensor
        (バッチ, 幅) の sin と cos の並び
    """
    # 周波数の違う正弦波
    half = width // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    return emb


def make_schedule(cfg: Any) -> torch.Tensor:
    """
    線形の雑音の予定から, 時刻ごとの累積の alpha を求める関数

    Parameters
    ----------
    cfg: Any
        config (diffusion の train_steps, beta_start, beta_end)

    Returns
    ----------
    schedule: torch.Tensor
        (学習の時刻の数,) の累積の alpha (ML-Inference の ddim と同じ予定)
    """
    # 1 - beta の累積の積 (推論の一式の ddim の設定と同じ値を使う)
    betas = torch.linspace(
        cfg.diffusion.beta_start, cfg.diffusion.beta_end,
        cfg.diffusion.train_steps, dtype=torch.float64
    )
    schedule = torch.cumprod(1 - betas, dim=0).float()

    return schedule


def call_model(
    model: nn.Module, inputs: tuple, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    雑音を加えた行と時刻を, 入力の名前でモデルに渡す関数

    Parameters
    ----------
    model: nn.Module
        雑音を当てるモデル
    inputs: tuple
        (雑音を加えた行の入力の名前, 時刻の入力の名前)
    x: torch.Tensor
        雑音を加えた行
    t: torch.Tensor
        (バッチ,) の時刻

    Returns
    ----------
    noise: torch.Tensor
        加えた雑音の予測 (形は x と同じ)
    """
    # ML-ModelControl の定義の入力の名前か, 仮のモデルの引数の名前
    noise = model(**{inputs[0]: x, inputs[1]: t})

    return noise


def noise_loss(
    model: nn.Module, inputs: tuple, schedule: torch.Tensor,
    x0: torch.Tensor, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """
    時刻と雑音を選んで行に雑音を加え, 加えた雑音を当てた誤差を求める関数

    Parameters
    ----------
    model: nn.Module
        雑音を当てるモデル
    inputs: tuple
        モデルの入力の名前 (雑音を加えた行, 時刻)
    schedule: torch.Tensor
        時刻ごとの累積の alpha
    x0: torch.Tensor
        (バッチ, 幅) の符号化した行
    generator: Optional[torch.Generator]
        時刻と雑音を選ぶ乱数 (None なら全体の乱数)

    Returns
    ----------
    loss: torch.Tensor
        予測した雑音と加えた雑音の平均二乗誤差
    """
    # 時刻ごとの混ぜ方で, 元の行と雑音を一度に混ぜる
    count = len(x0)
    t = torch.randint(0, len(schedule), (count,), generator=generator)
    noise = torch.randn(x0.shape, generator=generator)
    t, noise = t.to(x0.device), noise.to(x0.device)
    alpha = schedule.to(x0.device)[t][:, None]
    x_t = alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise
    loss = nn.MSELoss()(call_model(model, inputs, x_t, t), noise)

    return loss


def start_noise(width: int, count: int, seed: int) -> torch.Tensor:
    """
    作る行の始まりの雑音を引く関数

    Parameters
    ----------
    width: int
        符号化した行の幅
    count: int
        作る行の数
    seed: int
        始まりの雑音の乱数の種

    Returns
    ----------
    noise: torch.Tensor
        (行の数, 幅) の cpu の雑音
    """
    # 推論の一式と同じく, 種の生成器から 1 行ずつ引く (先頭の行は行の数に
    # よらず同じ)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.cat([torch.randn((1, width), generator=generator)
                       for _ in range(count)])

    return noise


@torch.no_grad()
def sample(
    model: nn.Module, inputs: tuple, schedule: torch.Tensor,
    noise: torch.Tensor, steps: int
) -> torch.Tensor:
    """
    雑音から DDIM のサンプリング (決まった道) で行を作る関数

    Parameters
    ----------
    model: nn.Module
        雑音を当てるモデル
    inputs: tuple
        モデルの入力の名前
    schedule: torch.Tensor
        時刻ごとの累積の alpha
    noise: torch.Tensor
        (行の数, 幅) の始まりの雑音 (start_noise)
    steps: int
        繰り返しの回数 (学習の時刻の数以下)

    Returns
    ----------
    x: torch.Tensor
        (行の数, 幅) の作った符号化した行 (モデルのデバイス)
    """
    # 雑音をモデルのデバイスへ. 間引いた時刻 (大きい順) で, 見積もった元の
    # 行から 1 つ前の時刻へ (推論の一式の ddim の eta 0 と同じ道)
    model.eval()
    device = next(model.parameters(), torch.empty(0)).device
    x = noise.to(device)
    times = torch.linspace(len(schedule) - 1, 0, steps).round().long()
    times = torch.unique_consecutive(times).tolist()
    for index, t in enumerate(times):
        now = float(schedule[t])
        before = (float(schedule[times[index + 1]])
                  if index + 1 < len(times) else 1.0)
        eps = call_model(model, inputs, x,
                         torch.full((len(x),), t, dtype=torch.long,
                                    device=device))
        x0 = (x - math.sqrt(1 - now) * eps) / math.sqrt(now)
        x = math.sqrt(before) * x0 + math.sqrt(1 - before) * eps

    return x


def table_columns(frame: pd.DataFrame) -> list[dict]:
    """
    学習用の表から, 列ごとの符号化 (数の列の標準化と, 種類の列のワンホット)
    を決める関数

    Parameters
    ----------
    frame: pd.DataFrame
        学習用の表 (欠損の無いもの)

    Returns
    ----------
    columns: list[dict]
        列の順の name, type (number か category) と, 数の列は mean, std,
        integer (整数の列か), 種類の列は categories (値の並び)

    Raises
    ----------
    ValueError
        推論の一式の予測の表の列と重なる名前の列がある場合と, 種類の列の
        値の種類が MAX_CATEGORIES より多い場合
    """
    # 推論の一式の予測の表の列と重なる名前
    reserved = [str(name) for name in frame.columns
                if str(name) in RESERVED_COLUMNS
                or str(name).startswith(RESERVED_PREFIXES)
                or str(name).endswith(RESERVED_SUFFIX)]
    if reserved:
        raise ValueError(
            f"列 {', '.join(reserved)} の名前は推論の一式の予測の表の列と"
            f"重なります ({', '.join(RESERVED_COLUMNS)} と, "
            f"{' か '.join(RESERVED_PREFIXES)} で始まる名前, "
            f"「{RESERVED_SUFFIX}」で終わる名前は使えません). レシピの "
            "drop_columns で落とすか, 元の表の列の名前を変えて書き出し直して"
            "ください"
        )

    # 真偽値と数でない列は種類の列. 標準偏差が 0 の列は 1 で割る
    columns = []
    for name in frame.columns:
        values = frame[name]
        numeric = (pd.api.types.is_numeric_dtype(values)
                   and not pd.api.types.is_bool_dtype(values))
        if numeric:
            std = float(values.std(ddof=0))
            columns.append({
                "name": str(name), "type": "number",
                "mean": float(values.mean()),
                "std": std if std > 0 else 1.0,
                "integer": bool(pd.api.types.is_integer_dtype(values)),
            })
            continue
        categories = sorted(pd.unique(values).tolist(), key=str)
        if len(categories) > MAX_CATEGORIES:
            raise ValueError(
                f"種類の列 {name} の値の種類が {len(categories)} で, 上限 "
                f"{MAX_CATEGORIES} より多いです. ID のような列なら, レシピ"
                "の drop_columns で落として書き出し直してください"
            )
        columns.append({"name": str(name), "type": "category",
                        "categories": categories})

    return columns


def encoded_width(columns: list[dict]) -> int:
    """
    符号化した行の幅を求める関数

    Parameters
    ----------
    columns: list[dict]
        列ごとの符号化 (table_columns)

    Returns
    ----------
    width: int
        数の列の数と, 種類の列の値の種類の数の和
    """
    # 数の列は 1, 種類の列は値の種類の数
    width = sum(1 if column["type"] == "number"
                else len(column["categories"]) for column in columns)

    return width


def encode_table(frame: pd.DataFrame, columns: list[dict]) -> torch.Tensor:
    """
    表を符号化した行にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        表 (列は columns の名前を全て持つ)
    columns: list[dict]
        列ごとの符号化 (table_columns)

    Returns
    ----------
    x: torch.Tensor
        (行の数, 幅) の float32. 学習用に無い種類の値は全て 0
    """
    # 数の列は標準化, 種類の列は値ごとの 0 と 1
    parts = []
    for column in columns:
        values = frame[column["name"]]
        if column["type"] == "number":
            scaled = (values.to_numpy(dtype=np.float64) - column["mean"]) / (
                column["std"]
            )
            parts.append(scaled[:, None])
            continue
        texts = values.astype(str).to_numpy()
        names = np.array([str(value) for value in column["categories"]])
        parts.append((texts[:, None] == names[None, :]).astype(np.float64))
    x = torch.from_numpy(np.concatenate(parts, axis=1).astype(np.float32))

    return x


def decode_table(x: torch.Tensor, columns: list[dict]) -> pd.DataFrame:
    """
    符号化した行を表に戻す関数

    Parameters
    ----------
    x: torch.Tensor
        (行の数, 幅) の符号化した行
    columns: list[dict]
        列ごとの符号化 (table_columns)

    Returns
    ----------
    frame: pd.DataFrame
        数の列は値 × std + mean (整数の列は丸める), 種類の列はワンホットの
        区間で最も大きい値の種類
    """
    # 列の順に区間を切り出して戻す
    array = x.detach().cpu().numpy().astype(np.float64)
    data = {}
    start = 0
    for column in columns:
        if column["type"] == "number":
            values = array[:, start] * column["std"] + column["mean"]
            data[column["name"]] = (np.round(values).astype(np.int64)
                                    if column["integer"] else values)
            start += 1
            continue
        width = len(column["categories"])
        picked = array[:, start:start + width].argmax(axis=1)
        data[column["name"]] = [column["categories"][index]
                                for index in picked]
        start += width
    frame = pd.DataFrame(data)

    return frame


def number_distances(
    real: np.ndarray, made: np.ndarray, std: float
) -> tuple[float, float]:
    """
    数の列の, 本物と作った値の分布の近さを求める関数

    Parameters
    ----------
    real: np.ndarray
        本物の値
    made: np.ndarray
        作った値
    std: float
        学習用の標準偏差 (Wasserstein 距離を割る)

    Returns
    ----------
    ks: float
        KS 統計量 (累積分布の差の最大. 0 から 1)
    wasserstein: float
        Wasserstein 距離 (累積分布の差の面積) を標準偏差で割った値
    """
    # 2 つの値を合わせた点ごとの累積分布の差
    points = np.sort(np.concatenate([real, made]))
    real_cdf = np.searchsorted(np.sort(real), points, side="right") / len(
        real
    )
    made_cdf = np.searchsorted(np.sort(made), points, side="right") / len(
        made
    )
    gap = np.abs(real_cdf - made_cdf)
    ks = float(gap.max())
    wasserstein = float(np.sum(gap[:-1] * np.diff(points)) / std)

    return ks, wasserstein


def category_distance(real: pd.Series, made: pd.Series) -> float:
    """
    種類の列の, 本物と作った値の分布の全変動距離を求める関数

    Parameters
    ----------
    real: pd.Series
        本物の値
    made: pd.Series
        作った値

    Returns
    ----------
    tvd: float
        値ごとの割合の差の絶対値の和の半分 (0 から 1)
    """
    # 値は文字にして比べる (学習用に無い本物の値も, その値のまま種類)
    real_share = real.astype(str).value_counts(normalize=True)
    made_share = made.astype(str).value_counts(normalize=True)
    shares = pd.concat([real_share, made_share], axis=1).fillna(0.0)
    tvd = float((shares.iloc[:, 0] - shares.iloc[:, 1]).abs().sum() / 2)

    return tvd


def column_distances(
    real: pd.DataFrame, made: pd.DataFrame, columns: list[dict]
) -> list[dict]:
    """
    本物の表と作った表の, 列ごとの分布の近さを求める関数

    Parameters
    ----------
    real: pd.DataFrame
        本物の表
    made: pd.DataFrame
        作った表
    columns: list[dict]
        列ごとの符号化 (table_columns)

    Returns
    ----------
    rows: list[dict]
        列ごとの column, type, distance (数の列は KS 統計量, 種類の列は
        全変動距離), wasserstein (数の列だけ. 種類の列は None)
    """
    # 列の種類ごとの近さ
    rows = []
    for column in columns:
        name = column["name"]
        if column["type"] == "number":
            ks, wasserstein = number_distances(
                real[name].to_numpy(dtype=np.float64),
                made[name].to_numpy(dtype=np.float64), column["std"]
            )
            rows.append({"column": name, "type": "number", "distance": ks,
                         "wasserstein": wasserstein})
            continue
        rows.append({"column": name, "type": "category",
                     "distance": category_distance(real[name], made[name]),
                     "wasserstein": None})

    return rows


def mean_distance(rows: list[dict]) -> float:
    """
    列ごとの分布の近さを, 全ての列で平均する関数

    Parameters
    ----------
    rows: list[dict]
        列ごとの近さ (column_distances)

    Returns
    ----------
    distance: float
        distance の平均 (0 から 1. 小さいほど良い)
    """
    # 列を同じ重みで
    distance = float(np.mean([row["distance"] for row in rows]))

    return distance


def generate_table(
    model: nn.Module, inputs: tuple, schedule: torch.Tensor,
    columns: list[dict], count: int, seed: int, steps: int, batch_size: int
) -> pd.DataFrame:
    """
    雑音から行を作り, 表に戻す関数

    Parameters
    ----------
    model: nn.Module
        雑音を当てるモデル
    inputs: tuple
        モデルの入力の名前
    schedule: torch.Tensor
        時刻ごとの累積の alpha
    columns: list[dict]
        列ごとの符号化 (table_columns)
    count: int
        作る行の数
    seed: int
        始まりの雑音の乱数の種
    steps: int
        DDIM の繰り返しの回数
    batch_size: int
        一度に作る行の数

    Returns
    ----------
    frame: pd.DataFrame
        作った表 (推論の一式に同じ種と行の数を渡すと同じ行)
    """
    # 始まりの雑音は全ての行の分を先に引き, バッチごとに作る
    noise = start_noise(encoded_width(columns), count, seed)
    made = torch.cat([sample(model, inputs, schedule, chunk, steps).cpu()
                      for chunk in noise.split(batch_size)])
    frame = decode_table(made, columns)

    return frame


def missing_columns(frames: dict) -> list[str]:
    """
    分割のどれかに欠損のある列を集める関数

    Parameters
    ----------
    frames: dict
        分割の名前から表

    Returns
    ----------
    names: list[str]
        欠損のある列の名前 (列の順. 無ければ空)
    """
    # 全ての分割で見る
    names: list[str] = []
    for frame in frames.values():
        for name in frame.columns[frame.isna().any()]:
            if str(name) not in names:
                names.append(str(name))

    return names


def noise_inputs(spec: Optional[Path]) -> tuple:
    """
    モデルの入力の名前 (雑音を加えた行, 時刻) を決める関数

    Parameters
    ----------
    spec: Optional[Path]
        ML-ModelControl の定義 (無ければ None で, 仮のモデルの x と t)

    Returns
    ----------
    names: tuple
        (雑音を加えた行の入力, 時刻の入力). 定義では dtype が int64 の
        入力が時刻, もう 1 つが行

    Raises
    ----------
    ValueError
        定義の入力が 2 つでないか, int64 の入力が 1 つでない場合
    """
    # 仮のモデルは引数の名前
    if spec is None:
        return ("x", "t")
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    entries = loaded.get("inputs") or {}
    times = [name for name, entry in entries.items()
             if isinstance(entry, dict) and entry.get("dtype") == "int64"]
    if len(entries) != 2 or len(times) != 1:
        raise ValueError(
            "定義の入力は, 雑音を加えた行と時刻 (dtype: int64) の 2 つに"
            f"してください (今は {list(entries)})"
        )
    names = (next(name for name in entries if name != times[0]), times[0])

    return names


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
    # ML-CreateArchitecture で組んで保存するまでは無い
    spec = PROJECT_DIR / cfg.model.spec if cfg.model.spec else None
    if spec is not None and not spec.is_file():
        spec = None

    return spec


def spec_errors(spec: Optional[Path]) -> list[str]:
    """
    ML-ModelControl の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)

    Returns
    ----------
    errors: list[str]
        誤りの文 (定義が無いか誤りが無ければ空)
    """
    # 定義の無いうちは ML-ModelControl を読まない. 読めない定義も誤りと
    # して返る (例外にならない). 入力の名前も確かめる
    if spec is None:
        return []
    from ml_modelcontrol import validate
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]
    if not errors:
        try:
            noise_inputs(spec)
        except ValueError as error:
            errors.append(str(error))

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
        符号化した行の幅
    cfg: Any
        config (仮のモデルの幅と乱数の種)
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
    model = PlaceholderModel(width, cfg.model.width)

    return model


def evaluate_split(
    trainer: TableDiffusionTrainer, real: pd.DataFrame, train: pd.DataFrame,
    split: str, cfg: Any, experiment: Experiment
) -> list[dict]:
    """
    学習した後の重みで本物と同じ行の数を作り, 列ごとの分布の近さを数える
    関数

    Parameters
    ----------
    trainer: TableDiffusionTrainer
        学習した後の Trainer
    real: pd.DataFrame
        比べる分割の本物の表
    train: pd.DataFrame
        学習用の本物の表 (目安の train_distance を数える)
    split: str
        分割の名前 (valid か test)
    cfg: Any
        config (作る繰り返しの回数と種, バッチの大きさ)
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    rows: list[dict]
        列ごとの column, type, distance, wasserstein, train_distance と,
        最後に全ての列の平均の行
    """
    # 見本と同じ種と回数で, 分割と同じ行の数
    model = trainer.model.cpu().eval()
    made = generate_table(model, trainer.inputs, trainer.schedule,
                          trainer.columns, len(real),
                          cfg.diffusion.sample_seed,
                          cfg.diffusion.sample_steps, cfg.data.batch_size)
    rows = column_distances(real, made, trainer.columns)
    reference = column_distances(real, train, trainer.columns)
    for row, base in zip(rows, reference):
        row["train_distance"] = base["distance"]
    rows.append({"column": MEAN_ROW, "type": None,
                 "distance": mean_distance(rows), "wasserstein": None,
                 "train_distance": mean_distance(reference)})
    experiment.logger.info(
        f"{split} の列ごとの分布の近さ (平均): {rows[-1]['distance']:.4f} "
        f"(学習用の本物との近さ {rows[-1]['train_distance']:.4f})"
    )

    return rows


def save_samples(
    trainer: TableDiffusionTrainer, valid: pd.DataFrame, cfg: Any,
    experiment: Experiment
) -> None:
    """
    学習した後の重みで作った表と, 見比べる検証用の本物の表を見本に書く関数

    Parameters
    ----------
    trainer: TableDiffusionTrainer
        学習した後の Trainer
    valid: pd.DataFrame
        検証用の本物の表
    cfg: Any
        config (作る繰り返しの回数と種, バッチの大きさ)
    experiment: Experiment
        実験 (samples/ に書く)

    Returns
    ----------
    None
    """
    # 推論の一式と同じ種と回数で作り, CSV のファイルにして写す
    model = trainer.model.cpu().eval()
    made = generate_table(model, trainer.inputs, trainer.schedule,
                          trainer.columns, SAMPLES,
                          cfg.diffusion.sample_seed,
                          cfg.diffusion.sample_steps, cfg.data.batch_size)
    with tempfile.TemporaryDirectory() as folder:
        generated = Path(folder) / "generated.csv"
        made.to_csv(generated, index=False)
        save_sample(experiment, "generated", generated, kind="file",
                    caption=f"作った表 {SAMPLES} 行 (種 "
                    f"{cfg.diffusion.sample_seed})")
        real = Path(folder) / "valid_real.csv"
        valid.head(SAMPLES).to_csv(real, index=False)
        save_sample(experiment, "valid_real", real, kind="file",
                    split="valid", caption="見比べる検証用の本物の表 (先頭 "
                    f"{min(SAMPLES, len(valid))} 行)")


def write_columns(run_dir: Path, columns: list[dict]) -> None:
    """
    列の符号化を実験のディレクトリに書く関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    columns: list[dict]
        列ごとの符号化 (table_columns)

    Returns
    ----------
    None
    """
    # 種類の値は JSON にできないものを文字に
    text = json.dumps({"columns": columns}, ensure_ascii=False, indent=2,
                      default=str)
    (run_dir / COLUMNS_FILE).write_text(text + "\\n", encoding="utf-8")


def origin_error() -> Optional[str]:
    """
    入れた ML-Found が task.yaml の正解の出どころの値を書けるかを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    message: Optional[str]
        書けない場合の文, 書ける場合は None
    """
    # 出どころ self を知らない古い ML-Found では write_task が分かりにくい
    # 誤りで止まるので, 出力先を作る前に確かめる
    message = None
    if ORIGIN_TYPE not in getattr(found_task, "ORIGIN_TYPES", ()):
        message = (
            f"この骨組みは正解の出どころ {ORIGIN_TYPE} を持つ ML-Found が"
            "要ります. ML-Found を git の main に上げてから学習し直して"
            "ください"
        )

    return message


def make_inference(
    cfg: Any, spec: Optional[Path], run_dir: Path, experiment: Experiment
) -> None:
    """
    学習した重みから, 実験の中に ML-Inference の雑音だけの一式を作る関数

    Parameters
    ----------
    cfg: Any
        config (一式の名前と ddim の設定)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    run_dir: Path
        実験のディレクトリ (列の符号化 COLUMNS_FILE を書いた後)
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    None
    """
    # 仮のモデルは ML-ModelControl の定義が無いので一式にできない
    if spec is None:
        experiment.logger.info(
            "仮のモデル (PlaceholderModel) は ML-ModelControl の定義が無い"
            "ので, 推論の一式を作りません. config の model.spec の場所に"
            "定義を置くと, 学習の後に作ります"
        )
        return

    # 学習と同じ雑音の予定の ddim で, データセットを持たない一式 (行の数
    # と種で表を作る). 列の符号化は一式に写すので, 1 件の形は省く
    from ml_inference import make_bundle
    state, time = noise_inputs(spec)
    procedure = {
        "name": "ddim", "steps": cfg.diffusion.sample_steps,
        "seed": cfg.diffusion.sample_seed,
        "options": {"train_steps": cfg.diffusion.train_steps,
                    "beta_start": cfg.diffusion.beta_start,
                    "beta_end": cfg.diffusion.beta_end, "eta": 0.0},
    }
    info = make_bundle(
        run_dir / "inference", spec, None, run_dir, role=ROLE,
        form=BUNDLE_FORM,
        inputs={state: {"source": "state"}, time: {"source": "time"}},
        output_options={"columns": str(run_dir / COLUMNS_FILE)},
        procedure=procedure, name=cfg.experiment.name
    )
    experiment.logger.info(
        "推論の一式 (雑音だけ): inference/ (行の数と種で, 列の名前と値の"
        f"表を作る. 版 {info['format_version']})"
    )
    for note in info["notes"]:
        experiment.logger.warning(f"推論の一式の注意: {note}")


def main() -> None:
    """
    ML-EDA の表のデータセットで拡散のモデルを学習し, 検証用とテスト用の
    分布の近さ, 見本と推論の一式を残す関数

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
    dataset: Dataset = load_dataset(path)
    if dataset.info.get("kind") != "table":
        print(f"{path} は表のデータセットではありません (種類 "
              f"{dataset.info.get('kind')})")
        sys.exit(1)
    if dataset.target is not None:
        print(f"{path} には目的の列 ({dataset.target}) があります. 目的の列"
              "の無いレシピ (task: none) で書き出してください")
        sys.exit(1)
    if "valid" not in dataset.frames:
        print("データセットに検証用の分割がありません. 最も良いモデルを選ぶ"
              "ために, 検証用の割合を 0 より大きくして書き出してください")
        sys.exit(1)

    # 分割ごとの表 (学習に使う列だけ). 欠損があれば止まる
    frames = {split: frame[list(dataset.features)].reset_index(drop=True)
              for split, frame in dataset.frames.items()
              if split in ("train", "valid", "test")}
    missing = missing_columns(frames)
    if missing:
        print(f"欠損のある列があります ({', '.join(missing)}). レシピに "
              "impute (埋める) か drop_missing (行を落とす) の手順を足して"
              "書き出し直してください")
        sys.exit(1)

    # 列の符号化 (学習用の表から決める)
    try:
        columns = table_columns(frames["train"])
    except ValueError as error:
        print(error)
        sys.exit(1)
    width = encoded_width(columns)
    x_train = encode_table(frames["train"], columns)
    x_valid = encode_table(frames["valid"], columns)

    # 繰り返しの回数と, モデルの定義の確かめ (誤りなら出力先を作る前に
    # 止まる)
    if not 1 <= cfg.diffusion.sample_steps <= cfg.diffusion.train_steps:
        print("config の diffusion.sample_steps は 1 から "
              f"diffusion.train_steps ({cfg.diffusion.train_steps}) に"
              f"してください (今は {cfg.diffusion.sample_steps})")
        sys.exit(1)
    spec = project_spec(cfg)
    errors = spec_errors(spec)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\\n".join(f"- {message}" for message in errors))
        sys.exit(1)

    # 入れた ML-Found が正解の出どころを書けるか
    origin = origin_error()
    if origin is not None:
        print(origin)
        sys.exit(1)

    # 何の実験か (受け持ち, 出力の形, 作るもの, 作り方は問題の設定から.
    # 学習に使ったデータセットは場所と dataset.yaml の名前と版 (load_dataset
    # が読む版 1). 版 1 は受け持ちと出どころを持たず, パスで渡すと ML-Found
    # が止まるので辞書で渡す).
    # TODO: 既製のモデルを元にするなら, 元にした実験か重みを task["base"]
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    write_task(experiment, {
        "role": ROLE, "output": OUTPUT, "what": PREDICT,
        "target": {"kind": TARGET_KIND, "origin": {"type": ORIGIN_TYPE}},
        "method": METHOD,
        "dataset": {"path": str(path), "name": dataset.info.get("name"),
                    "format_version": dataset.info.get("format_version")},
    })

    # 列の数, 符号化した行の幅, 行の数と学習と検証の loader
    numbers = sum(column["type"] == "number" for column in columns)
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}列: 数の列 {numbers}, 種類の列 "
        f"{len(columns) - numbers}. 符号化した行の幅: {width}. 学習用 "
        f"{len(x_train)} 行, 検証用 {len(x_valid)} 行"
    )
    train_loader = DataLoader(TensorDataset(x_train),
                              batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid),
                              batch_size=cfg.data.batch_size)

    # モデルの組み立てと学習 (定義がまだ無ければ仮のモデル. 重みの指数
    # 移動平均は ML-Found の trainer.ema)
    spec, base_dir = spec_file(spec, experiment)
    if spec is None:
        experiment.logger.info(
            f"モデルの定義 ({cfg.model.spec or '場所なし'}) が無いので, "
            "仮のモデル (PlaceholderModel) で学習します. モデルの作成・選定の段階で組んだ定義を"
            "その場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(width, cfg, spec, base_dir)
    trainer = TableDiffusionTrainer(model, experiment, noise_inputs(spec),
                                    make_schedule(cfg), columns,
                                    frames["valid"], cfg)
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の場所
    # origin も残る. モデルの定義は ML-Found が start_experiment で写す) と
    # 列の符号化
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    write_columns(run_dir, columns)

    # 結果の表示. 学習した後の重みは, 最良のモデルに戻した場合だけ最良の
    # エポック (select_best: false などは最後)
    best = trainer.final_weights == "best"
    epoch = trainer.checkpoint.best_epoch if best else trainer.epoch
    weights = "最良のモデル" if best else "最後のエポックの重み"
    print(f"保存先: {run_dir}")
    print(f"評価する重み: {weights} (エポック {epoch})")
    if best:
        print(f"ベストの {cfg.trainer.monitor}: "
              f"{trainer.checkpoint.best_value}")

    # 検証用とテスト用 (あれば) の列ごとの分布の近さを evaluations/ へ保存
    for split, title in (("valid", "検証用"), ("test", "テスト用")):
        if split not in frames:
            experiment.logger.info(
                f"{title}の分割が無いので, {title}の分布の近さは数えません"
            )
            continue
        rows = evaluate_split(trainer, frames[split], frames["train"], split,
                              cfg, experiment)
        save_evaluation(
            experiment, f"{EVALUATION}_{split}", rows, split,
            description=EVALUATION_TEXT.format(split=title, weights=weights),
            epoch=epoch, better=EVALUATION_BETTER
        )

    # 作った表と本物の表の見本 (ML-Dashboard の見本のタブ). 書けなくても
    # 学習の結果は残すので, 警告だけ
    try:
        save_samples(trainer, frames["valid"], cfg, experiment)
    except ValueError as error:
        experiment.logger.warning(f"見本を書けません: {error}")

    # 実験の中に雑音だけの推論の一式 (ML-ModelControl の定義がある場合
    # だけ. 上で書いた列の符号化を一式に写す)
    make_inference(cfg, spec, run_dir, experiment)


if __name__ == "__main__":
    main()
'''

# config の基準 (ML-Found のプロジェクトの形)
TABLE_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題から書き出した骨組み)
$requires
experiment:
  name: $experiment
  seed: 0

optimizer:
  name: AdamW
  params:
    lr: 0.001

trainer:
  epochs: 100
  grad_clip_norm: 1.0
  # 問題の「最良を選ぶ指標」(valid/loss は雑音の予測の損失. 検証は
  # 毎回同じ時刻と雑音で数える. valid/column_distance は作った表と検証用の
  # 表の列ごとの分布の近さ. どちらも小さいほど良い)
  monitor: $monitor
  mode: $mode
  # 重みの指数移動平均. 検証, 最良のモデル, final_model.pt はこの重み.
  # 更新の回数 (エポック数 × 1 エポックのバッチ数) が 1 / (1 - decay) (0.995
  # なら 200) より十分多くなるようにする (少ないと初めの重みのまま)
  ema:
    decay: 0.995

# プロジェクト固有の設定
data:
  # ML-EDA の表のデータセット (プロジェクトのディレクトリから)
  dataset: $dataset
  batch_size: 64

model:
  # ML-ModelControl の形のモデルの定義 (モデルの作成・選定の段階で組んで, この
  # 場所に保存する. 入力は雑音を加えた行と int64 の時刻の 2 つ)
  spec: models/model.yaml
  # spec のファイルが無いうちは train.py の仮のモデルで学習することの目印
  # (学習・評価の段階の学習の準備が読む. train.py は読まず, 仮のモデルを使うか
  # は spec のファイルがあるかで決まる)
  placeholder: true
  # 仮のモデルの中の幅 (時刻の正弦波の埋め込みの幅にも使うので偶数)
  width: 64

diffusion:
  # 線形の雑音の予定 (学習の時刻の数と beta の始まりと終わり). train.py が
  # 同じ値を推論の一式の ddim に書く
  train_steps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  # 見本, 分布の近さと推論の一式の ddim の繰り返しの回数 (1 から
  # train_steps) と, 始まりの雑音の乱数の種
  sample_steps: 50
  sample_seed: 0
''')


def table_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 表の生成の拡散のモデルを学ぶスクリプトを作る関数

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
    head = Template(TABLE_HEAD).substitute(
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        method=literal(active_value(problem, f"{PART}.method")),
        origin_type=literal(generation_origin(problem)),
    )
    # train.log の初めの行の作るもの (空なら省く)
    made = ("作るもの: {PREDICT}. "
            if active_value(problem, f"{PART}.what") else "")
    script = head + Template(TABLE_MAIN).substitute(predict_log=made)

    return script


def table_config(
    experiment: str, stages: dict, dataset: str, requires: str
) -> str:
    """
    表の生成の config の基準を作る関数

    Parameters
    ----------
    experiment: str
        実験の名前
    stages: dict
        受け渡し
    dataset: str
        データセットの場所 (プロジェクトのディレクトリから)
    requires: str
        一番上の requires の行 (skeleton.requires_text)

    Returns
    ----------
    text: str
        configs/default/train.yaml の中身
    """
    # 監視は受け渡しの値 (valid/loss か valid/column_distance と min)
    training = stages["training"]
    text = TABLE_CONFIG.substitute(
        experiment=json.dumps(experiment, ensure_ascii=False),
        monitor=training["monitor"], mode=training["mode"],
        dataset=json.dumps(dataset, ensure_ascii=False), requires=requires,
    )

    return text


def table_readme(
    problem: dict, stages: dict, dataset: str, raw: Optional[str]
) -> str:
    """
    表の生成のプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    dataset: str
        データセットの場所 (プロジェクトのディレクトリから)
    raw: Optional[str]
        写した元の表のファイルの場所 (プロジェクトのディレクトリから.
        入れていなければ None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # ML-Found の要件 (正解の出どころが self なら)
    origin_note = origin_readme(problem)

    # 筋道の要約
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest で問題 (`problem.yaml`) から書き出した, "
             "ML-Found の学習のプロジェクトの骨組み (ML-EDA の表のデータセットで, "
             "条件の無い拡散のモデルを学び, 元の表に似た行を作る).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1 から 4. データ, モデル, 学習, 推論
    monitor = stages["training"]["monitor"]
    lines += [
        "", "## 次にすること", "",
        "学習する環境には ML-EDA, ML-Found, ML-ModelControl, ML-Inference を"
        " 入れる (config の `requires` の機能. 足りなければ学習の前に入れ方を"
        "示して止まる. ML-ModelControl と ML-Inference は定義を置くと使う)."
        + origin_note, "",
        "1. **データの分析** (ML-EDA): 表のファイルを ML-EDA で"
        "読み, 目的の列の無いレシピ (`task: none`. `recipe_from` の雛形) で"
        "このディレクトリの `" + dataset + "` に書き出す. 検証用の分割は"
        "必ず作る. 書き出した列を全て作るので, ID のような列 (行ごとに違う"
        "値) と作らない列は `drop_columns` で落とし, 欠損は `impute` (埋める)"
        " か `drop_missing` (行を落とす) で無くす (欠損のある表, 値の種類が "
        "100 を超える種類の列と, 推論の一式の予測の表の列と重なる名前 "
        "(`sample`, `status`, `prediction` など) の列では `train.py` が"
        "止まる). 標準化とワン"
        "ホットは `train.py` が学習用の分割から決めるので, レシピには足さない"
        + (f". 入れた表はこのディレクトリの `{raw}` に写した" if raw
           else ""),
        "2. **モデルの作成・選定** (ML-ModelControl): モデルの作成・選定の段階で組んだ定義を"
        ", config の `model.spec` の場所"
        " (`models/model.yaml`) に保存する. 入力は雑音を加えた行 ((バッチ, "
        "幅) の float32. 幅は数の列の数と種類の列の値の種類の数の和で, "
        "`train.log` の初めの行に出る) と時刻 ((バッチ,) の int64. "
        "`dtype: int64` の入力を時刻とみなす) の 2 つ, 出力は行と同じ形の"
        "加えた雑音の予測. ML-ModelControl の同梱の例 `diffusion_denoiser` "
        "の幅を合わせて使える. 定義が無いうちは, `train.py` の "
        "`PlaceholderModel` (印 TODO. 動くことを確かめるだけの小さなモデル) "
        "で学習する (config の `model.placeholder: true` がその目印)",
        "3. **学習・評価** (ML-Found): `python train.py`. 数の列は"
        "学習用の分割の平均と標準偏差で標準化し, 種類の列は学習用の分割の値"
        "でワンホットにした行に, 線形の雑音の予定 (config の `diffusion`) で"
        "雑音を加え, 加えた雑音を当てる (平均二乗誤差). 重みの指数移動平均 "
        "(`trainer.ema.decay`. 既定 0.995) の重みで検証し (更新の回数 "
        "(エポック数 × 1 エポックのバッチ数) が 1 / (1 - decay) より十分"
        "多くなるようにする), 検証のたびに, 見本と同じ種と回数で検証用と"
        "同じ行の数を作り, 列ごとの分布の近さ (`valid/column_distance`. 数の"
        "列は KS 統計量, 種類の列は全変動距離の, 列の平均. 0 から 1 で小さい"
        "ほど良い) を数える. 損失 (`valid/loss`) は毎回同じ時刻と雑音で"
        f"数える. 検証用の `{monitor}` が最も良いモデルを選ぶ. 結果は "
        "`outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, 列の"
        "符号化 `table_columns.json`, 検証用とテスト用 (あれば) の列ごとの"
        "分布の近さ `evaluations/column_distance_valid.json` と "
        "`column_distance_test.json` (列ごとの `distance`, 数の列の "
        "Wasserstein 距離を学習用の標準偏差で割った `wasserstein`, 学習用の"
        "本物との `distance` の `train_distance` と, 全ての列の平均の行), "
        "作った表 100 行と見比べる検証用の本物の表の見本 `samples/` の "
        "CSV). 分布の近さは行の数が少ないと大きく出るので, 同じ行の数で"
        "比べ, `train_distance` (同じ分布でも行の数の違いから出る目安) と"
        "見比べる. 列どうしの関係 (同時の分布) は数えない. 今のやり方とは"
        "比べない",
        "4. **推論** (ML-Inference): `model.spec` の場所に定義を置いて"
        "学習すると, `train.py` が学習の後に実験の中の `inference/` に"
        "雑音だけの推論の一式 (版 7. データセットを持たず, 行の数と種で"
        "作る. 手順は学習と同じ雑音の予定の `ddim`. 出力の形は `table` で, "
        "列の符号化 `table_columns.json` を一式に写す) を作る. 版 7 の一式を"
        "作れる ML-Inference が要る. その `inference/` に行の数と種を渡すと, "
        "件ごとの `sample`, `seed`, `status`, `reason` と, 列の名前と値の"
        "表が返る (同じ種と行の数なら見本の表とほぼ同じ行. 整数と種類は "
        "95% 以上, 数は 1e-3 以内で一致). 仮のモデル (`PlaceholderModel`) は"
        "定義が無いので一式を作らない",
        "", "`train.py` の印 (TODO) の所 (モデルなど) は書き足す. 問題を"
        "変えたら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
