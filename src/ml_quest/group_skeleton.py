import json
from string import Template
from typing import Callable, Optional

from ml_quest.handoff import describe
from ml_quest.problem import active_value, get_value
from ml_quest.questions import GROUP_STABILITIES, PART

# 学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の機能の名前
# (表は ML-EDA の読み方で読む. config の model.spec の定義が無いうちは
# ml_modelcontrol を読まない)
GROUP_LIBRARIES = ["eda", "modelcontrol"]

# 書き出していないときの, 群を作る表の場所 (プロジェクトのディレクトリから)
GROUP_DATASET = "outputs/rows.csv"

# 無作為に分ける, 検証用の行の割合の既定 (config の data.valid)
GROUP_VALID = 0.2

# 群の要約の表の, 群を作る列と同じ名前にできない列
SUMMARY_COLUMNS = ["group", "rows", "share", "stability"]

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
GROUP_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

1 行が 1 つの対象の表で, 行の特徴から埋め込みを作る符号化器を対照学習
(同じ行の 2 通りの崩し方の埋め込みを近づける) で学び, 埋め込みを k-means
で群に分けて, 群の安定性と群の要約を残す骨組みなので, 印 (TODO) の所を
書き足す
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

from ml_eda import load_data
from ml_found import (
    Experiment, Trainer, load_project_config, save_evaluation,
    start_experiment, write_task
)

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, まとめるもの
PROBLEM_NAME = $name
PREDICT = $predict

# task.yaml の受け持ち, 出力の形と作り方 (問題の設定の答え)
ROLE = "group"
OUTPUT = "cluster"
METHOD = $method

# 表の読み方 (問題の設定の手元のデータの読み方. ML-EDA の load_data に
# 渡す. None は ML-EDA の推定)
LOADER = $loader
OPTIONS = $options

# モデルの入力と出力の名前 (ML-ModelControl の同梱の例 contrastive_encoder
# と同じ)
FEATURE_INPUT = "x"
EMBEDDING_OUTPUT = "embedding"
SIMILARITY_OUTPUT = "similarity"

# 種類の列の, 欠けた値の呼び方
MISSING = "(欠損)"

# 群の要約の表の列 (群を作る列と同じ名前にはできない)
SUMMARY_COLUMNS = $summary_columns

# 群の安定性の数え方の名前
STABILITY_NAMES = {"seed": "種を変えて作り直す",
                   "sample": "標本を変えて作り直す"}

# 行ごとの群の割り当てと, 列の符号化と群の中心 (実験のディレクトリの中)
ASSIGNMENT_FILE = "group_assignments.csv"
CENTERS_FILE = "group_centers.json"

# 群の安定性と群の要約の評価の名前 (evaluations/<名前>.json), 説明, 向き
STABILITY = "group_stability"
STABILITY_TEXT = (
    "全ての行の群と, 学習用の行で作り直した群 ({method}, {repeats} 回) の"
    "一致 "
    "(調整ランド指数 ari の平均と最も低い値 ari_min. 1 は同じ群, 0 は偶然"
    "と同じ) と, 全ての行のシルエット係数 (群のまとまり. それぞれの群を"
    "作った空間で数える) ({weights}). _baseline は列の特徴をそのまま "
    "k-means で分けた値, _diff はモデル − そのままで, 大きいほど良い"
)
STABILITY_BETTER = {"ari": "higher", "ari_min": "higher",
                    "silhouette": "higher"}
SUMMARY = "group_summary"
SUMMARY_TEXT = (
    "群ごとの行の数 (rows), 割合 (share), 作り直したときの群の一致 "
    "(stability. 作り直した群のうち最も重なる群との Jaccard 係数の平均) と,"
    " 群を作った列の, 数の列は平均, 種類の列は最も多い値と割合 ({weights}."
    " 最後の行は全ての行)"
)
SUMMARY_BETTER = {"stability": "higher"}

# 比べる相手の名前
BASELINE_NAME = "列の特徴をそのまま k-means で分けた群"
'''

# 符号化器の部品, 表の符号化, Trainer, k-means と安定性, 学習の流れ
GROUP_MAIN = '''

class PlaceholderModel(nn.Module):
    def __init__(self, features: int, width: int):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 符号化した行の幅と, 埋め込みの幅を保存
        self.features = features
        self.width = width

        # TODO: 課題に合うモデルに置き換える. 今は動くことを確かめるための,
        # 全結合 2 段の符号化器
        self.fc1 = nn.Linear(self.features, self.width * 2)
        self.fc2 = nn.Linear(self.width * 2, self.width)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> dict:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 符号化した行の幅) の行の特徴

        Returns
        ----------
        outputs: dict
            embedding (L2 正規化した埋め込み) と similarity (全ての組の
            内積の 10 倍)
        """
        # 埋め込みを L2 正規化し, 全ての組の内積を温度 0.1 で割る
        h = self.fc2(self.relu(self.fc1(x)))
        e = h / h.norm(dim=1, keepdim=True).clamp_min(1e-12)
        outputs = {EMBEDDING_OUTPUT: e, SIMILARITY_OUTPUT: e @ e.T * 10.0}

        return outputs


class GroupTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment,
        features: torch.Tensor, valid: torch.Tensor, cfg: Any
    ):
        # 親クラスのコンストラクタを呼び出す (step を上書きするので
        # step_fn は None)
        super(GroupTrainer, self).__init__(model, experiment, None)

        # 学習用と検証用の行の特徴 (崩すときに値を借りる行と, 群を作る行),
        # 群の設定, 埋め込みを求めるまとまりの大きさ, 乱数の種を保存
        self.features = features
        self.valid = valid
        self.group = cfg.group
        self.batch_size = cfg.data.batch_size
        self.seed = cfg.experiment.seed
        self.generator: Optional[torch.Generator] = None

        # 同じ行の 2 つの崩し方を正解にする交差エントロピー
        self.criterion = nn.CrossEntropyLoss()

    def on_epoch_start(self, phase: str) -> None:
        """
        検証の崩し方を, エポックごとに同じにする関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        None
        """
        # 検証は同じ種の乱数 (損失をエポックの間で比べられるように)
        self.generator = (torch.Generator().manual_seed(self.seed)
                          if phase != "train" else None)

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の, 対照学習の損失を求める関数

        Parameters
        ----------
        batch: tuple
            (行の特徴,)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            同じ行の 2 つの崩し方を当てる交差エントロピー (loss)
        """
        # 学習は学習しているモデル, 検証は評価のモデル (EMA があれば EMA)
        (x,) = batch
        model = self.model if phase == "train" else self.eval_model()
        rate = self.group.corruption
        views = torch.cat([corrupt(x, self.features, rate, self.generator),
                           corrupt(x, self.features, rate, self.generator)])
        similarity = encode(model, views)[SIMILARITY_OUTPUT]

        # 自分自身との類似度は隠し, i 番目の正解は同じ行のもう一方
        count = len(x)
        same = torch.eye(2 * count, dtype=torch.bool,
                         device=similarity.device)
        similarity = similarity.masked_fill(same, -1e9)
        target = (torch.arange(2 * count, device=similarity.device)
                  + count) % (2 * count)
        outputs = {"loss": self.criterion(similarity, target)}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 検証用の行の埋め込みの群のシルエット係数を数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証なら silhouette (valid/silhouette として記録. 大きいほど
            良い). 他は空
        """
        # 学習用の行の埋め込みで群の中心を決め, 検証用の行を割り当てる
        if phase != "valid":
            return {}
        model = self.eval_model()
        train = embed(model, self.features, self.batch_size)
        valid = embed(model, self.valid, self.batch_size)
        rng = np.random.default_rng(self.seed)
        centers = kmeans(train, self.group.k, rng, self.group.iterations,
                         self.group.inits)
        labels = assign(valid, centers)
        metrics = {"silhouette": silhouette(valid, labels, self.group.sample,
                                            rng)}

        return metrics


def corrupt(
    x: torch.Tensor, pool: torch.Tensor, rate: float,
    generator: Optional[torch.Generator]
) -> torch.Tensor:
    """
    行の特徴の一部を, 他の行の同じ特徴の値に置き換えて崩す関数

    Parameters
    ----------
    x: torch.Tensor
        (件数, 符号化した行の幅) の行の特徴
    pool: torch.Tensor
        値を借りる行の特徴 (学習用の行)
    rate: float
        置き換える特徴の割合
    generator: Optional[torch.Generator]
        乱数 (None なら torch の乱数)

    Returns
    ----------
    view: torch.Tensor
        崩した行の特徴 (x と同じデバイス)
    """
    # TODO: 課題に合う崩し方 (数の列に雑音を足すなど) に置き換えられる.
    # 今は特徴ごとに, 学習用の無作為な行の値を借りる
    rows = torch.randint(len(pool), tuple(x.shape), generator=generator)
    columns = torch.arange(x.shape[1]).expand_as(rows)
    borrowed = pool[rows, columns].to(x.device)
    mask = torch.rand(tuple(x.shape), generator=generator) < rate
    view = torch.where(mask.to(x.device), borrowed, x)

    return view


def encode(model: nn.Module, x: torch.Tensor) -> Any:
    """
    行の特徴を, 入力の名前でモデルに渡して出力を返す関数

    Parameters
    ----------
    model: nn.Module
        符号化器
    x: torch.Tensor
        (件数, 符号化した行の幅) の行の特徴

    Returns
    ----------
    outputs: Any
        出力の名前の辞書 (embedding と similarity)
    """
    # ML-ModelControl の定義も仮のモデルも入力の名前で渡す
    outputs = model(**{FEATURE_INPUT: x})

    return outputs


def embed(model: nn.Module, x: torch.Tensor, batch_size: int) -> np.ndarray:
    """
    行の特徴の埋め込みを, まとまりごとに求める関数

    Parameters
    ----------
    model: nn.Module
        符号化器
    x: torch.Tensor
        (件数, 符号化した行の幅) の行の特徴
    batch_size: int
        1 回に渡す行の数

    Returns
    ----------
    points: np.ndarray
        (件数, 埋め込みの幅) の float64
    """
    # 学習の切り替えは Trainer がする (検証の後は学習のエポックに戻る)
    device = next(model.parameters()).device
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size].to(device)
            parts.append(encode(model, chunk)[EMBEDDING_OUTPUT].cpu())
    points = torch.cat(parts).double().numpy()

    return points


def assign(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    点を最も近い群の中心に割り当てる関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    centers: np.ndarray
        (群の数, 幅) の群の中心

    Returns
    ----------
    labels: np.ndarray
        (件数,) の群の番号
    """
    # 二乗距離を内積で求める
    points = np.asarray(points, dtype=np.float64)
    distance = ((points ** 2).sum(1)[:, None] - 2 * points @ centers.T
                + (centers ** 2).sum(1)[None, :])
    labels = distance.argmin(1)

    return labels


def kmeans(
    points: np.ndarray, k: int, rng: np.random.Generator, iterations: int,
    inits: int = 1
) -> np.ndarray:
    """
    中心の置き方を変えて k-means を繰り返し, 群の中心までの二乗距離の和が
    最も小さい群の中心を返す関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    k: int
        群の数
    rng: np.random.Generator
        中心の初めの置き方の乱数
    iterations: int
        Lloyd 法の繰り返しの上限
    inits: int = 1
        中心の置き方を変える回数

    Returns
    ----------
    centers: np.ndarray
        (群の数, 幅) の群の中心
    """
    # 悪い置き方から始めた局所解を避ける
    points = np.asarray(points, dtype=np.float64)
    best, smallest = None, np.inf
    for _ in range(max(int(inits), 1)):
        centers = kmeans_once(points, k, rng, iterations)
        inertia = ((points - centers[assign(points, centers)]) ** 2).sum()
        if inertia < smallest:
            best, smallest = centers, inertia

    return best


def kmeans_once(
    points: np.ndarray, k: int, rng: np.random.Generator, iterations: int
) -> np.ndarray:
    """
    k-means++ で中心を置き, Lloyd 法で群の中心を求める関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    k: int
        群の数
    rng: np.random.Generator
        中心の初めの置き方の乱数
    iterations: int
        Lloyd 法の繰り返しの上限

    Returns
    ----------
    centers: np.ndarray
        (群の数, 幅) の群の中心 (点の無い群の中心は動かさない)
    """
    # k-means++ (近い中心から遠い点ほど選ばれやすい)
    points = np.asarray(points, dtype=np.float64)
    first = int(rng.integers(len(points)))
    chosen = [points[first]]
    closest = ((points - points[first]) ** 2).sum(1)
    for _ in range(1, k):
        total = closest.sum()
        index = (int(rng.choice(len(points), p=closest / total))
                 if total > 0 else int(rng.integers(len(points))))
        chosen.append(points[index])
        closest = np.minimum(closest, ((points - points[index]) ** 2).sum(1))
    centers = np.stack(chosen)

    # 割り当てと群の平均を, 中心が動かなくなるまで
    for _ in range(iterations):
        labels = assign(points, centers)
        moved = centers.copy()
        for group in range(k):
            members = points[labels == group]
            if len(members):
                moved[group] = members.mean(0)
        if np.allclose(moved, centers):
            break
        centers = moved

    return centers


def silhouette(
    points: np.ndarray, labels: np.ndarray, size: int,
    rng: np.random.Generator
) -> float:
    """
    群のシルエット係数 (群のまとまり) を数える関数

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点
    labels: np.ndarray
        (件数,) の群の番号
    size: int
        数える点の数の上限 (多ければ無作為に選ぶ. 距離の表が点の数の二乗の
        大きさのため)
    rng: np.random.Generator
        点を選ぶ乱数

    Returns
    ----------
    score: float
        -1 から 1 (大きいほど群がまとまり, 他の群と離れている. 群が 1 つ
        なら 0. 1 点だけの群の点は 0)
    """
    # 点を選び, 全ての組の距離
    points = np.asarray(points, dtype=np.float64)
    if len(points) > size:
        picked = rng.choice(len(points), size, replace=False)
        points, labels = points[picked], labels[picked]
    groups = np.unique(labels)
    if len(groups) < 2:
        return 0.0
    squared = (points ** 2).sum(1)
    distance = np.sqrt(np.maximum(
        squared[:, None] - 2 * points @ points.T + squared[None, :], 0.0
    ))

    # 自分の群の他の点までの平均 a と, 最も近い他の群の点までの平均 b
    member = labels[:, None] == groups[None, :]
    sums = distance @ member
    counts = member.sum(0)
    own = member.argmax(1)
    rows = np.arange(len(points))
    a = sums[rows, own] / np.maximum(counts[own] - 1, 1)
    other = sums / counts
    other[rows, own] = np.inf
    b = other.min(1)
    scores = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    scores[counts[own] == 1] = 0.0
    score = float(scores.mean())

    return score


def adjusted_rand(first: np.ndarray, second: np.ndarray) -> float:
    """
    2 つの群の割り当ての一致 (調整ランド指数) を数える関数

    Parameters
    ----------
    first: np.ndarray
        (件数,) の群の番号
    second: np.ndarray
        (件数,) の群の番号 (番号の付け方は first と違ってよい)

    Returns
    ----------
    score: float
        1 は同じ群, 0 は偶然と同じ (負もある)
    """
    # 同じ群に入る組の数を, 群の組の表から数える
    first = np.unique(first, return_inverse=True)[1]
    second = np.unique(second, return_inverse=True)[1]
    table = np.zeros((first.max() + 1, second.max() + 1))
    np.add.at(table, (first, second), 1)
    pairs = (table * (table - 1) / 2).sum()
    rows = (table.sum(1) * (table.sum(1) - 1) / 2).sum()
    columns = (table.sum(0) * (table.sum(0) - 1) / 2).sum()
    total = len(first) * (len(first) - 1) / 2
    expected = rows * columns / total if total else 0.0
    top = (rows + columns) / 2
    score = 1.0 if top == expected else float(
        (pairs - expected) / (top - expected)
    )

    return score


def group_jaccard(
    labels: np.ndarray, again: np.ndarray, k: int
) -> np.ndarray:
    """
    群ごとに, 作り直した群のうち最も重なる群との Jaccard 係数を数える関数

    Parameters
    ----------
    labels: np.ndarray
        (件数,) の元の群の番号
    again: np.ndarray
        (件数,) の作り直した群の番号
    k: int
        群の数

    Returns
    ----------
    scores: np.ndarray
        (群の数,) の 0 から 1 (行の無い群は NaN)
    """
    # 元の群と作り直した群の, 共通の行と合わせた行の数の比の最も大きい値
    scores = np.full(k, np.nan)
    for group in range(k):
        members = labels == group
        if not members.any():
            continue
        scores[group] = max(
            (members & (again == other)).sum()
            / (members | (again == other)).sum()
            for other in np.unique(again)
        )

    return scores


def stability(
    points: np.ndarray, train: np.ndarray, labels: np.ndarray, cfg: Any
) -> dict:
    """
    学習用の行で群を作り直し, 全ての行の群との一致を数える関数

    Parameters
    ----------
    points: np.ndarray
        (全ての行の数, 幅) の点
    train: np.ndarray
        学習用の行の番号
    labels: np.ndarray
        (全ての行の数,) の元の群の番号
    cfg: Any
        config (group.stability, repeats, fraction, k, iterations と
        乱数の種)

    Returns
    ----------
    found: dict
        ari (調整ランド指数の平均), ari_min (最も低い値), jaccard (群ごと
        の Jaccard 係数の平均. 行の無い群は NaN)
    """
    # 作り直すたびに別の種. 標本を変えるなら学習用の行の一部を引き直す
    scores, overlaps = [], []
    for repeat in range(cfg.group.repeats):
        rng = np.random.default_rng(cfg.experiment.seed + 1 + repeat)
        rows = train
        if cfg.group.stability == "sample":
            size = max(cfg.group.k, int(len(train) * cfg.group.fraction))
            rows = rng.choice(train, size, replace=False)
        centers = kmeans(points[rows], cfg.group.k, rng,
                         cfg.group.iterations, cfg.group.inits)
        again = assign(points, centers)
        scores.append(adjusted_rand(labels, again))
        overlaps.append(group_jaccard(labels, again, cfg.group.k))
    overlaps = np.array(overlaps)
    jaccard = np.array([
        column[~np.isnan(column)].mean() if (~np.isnan(column)).any()
        else np.nan for column in overlaps.T
    ])
    found = {"ari": float(np.mean(scores)), "ari_min": float(np.min(scores)),
             "jaccard": jaccard}

    return found


def category_text(value: Any) -> str:
    """
    種類の列の値を, 符号化に使う文字にする関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    text: str
        文字 (欠けた値は MISSING)
    """
    # 欠けた値も 1 つの種類にする
    text = MISSING if pd.isna(value) else str(value)

    return text


def is_number(column: pd.Series) -> bool:
    """
    列を数の列として符号化するかを返す関数

    Parameters
    ----------
    column: pd.Series
        列

    Returns
    ----------
    number: bool
        数の列 (真偽値の列は種類の列にする)
    """
    # 真偽値は 2 つの種類
    number = (pd.api.types.is_numeric_dtype(column)
              and not pd.api.types.is_bool_dtype(column))

    return number


def make_encoding(frame: pd.DataFrame) -> dict:
    """
    学習用の行から, 列の符号化を決める関数

    Parameters
    ----------
    frame: pd.DataFrame
        学習用の行の, 群を作る列

    Returns
    ----------
    encoding: dict
        numeric (数の列ごとの mean と std. 標準化する) と categories
        (種類の列ごとの値の並び. 学習用に初めて出た順. 値ごとに 0 と 1 の
        列にする)
    """
    # 散らばりの無い数の列は 1 で割る
    numeric: dict = {}
    categories: dict = {}
    for name in frame.columns:
        column = frame[name]
        if is_number(column):
            values = column.astype(float)
            std = float(values.std(ddof=0)) if values.notna().any() else 0.0
            numeric[name] = {
                "mean": float(values.mean()) if values.notna().any() else 0.0,
                "std": std if std > 0 else 1.0,
            }
        else:
            categories[name] = list(dict.fromkeys(column.map(category_text)))
    encoding = {"numeric": numeric, "categories": categories}

    return encoding


def encode_rows(frame: pd.DataFrame, encoding: dict) -> np.ndarray:
    """
    行を, 列の符号化で数の並びにする関数

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列
    encoding: dict
        make_encoding の符号化

    Returns
    ----------
    features: np.ndarray
        (行の数, 符号化した行の幅) の float32 (数の列は標準化して欠けた
        値は 0, 種類の列は値ごとの 0 と 1. 学習用に無い値は全て 0)
    """
    # TODO: 課題に合う特徴 (比の列, 対数など) を足せる
    parts = []
    for name, stats in encoding["numeric"].items():
        values = (frame[name].astype(float) - stats["mean"]) / stats["std"]
        parts.append(values.fillna(0.0).to_numpy()[:, None])
    for name, values in encoding["categories"].items():
        text = frame[name].map(category_text).to_numpy()
        parts.append(np.stack([text == value for value in values], 1))
    features = np.concatenate(parts, 1).astype(np.float32)

    return features


def read_rows(
    path: Path, cfg: Any
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    群を作る表を読み, 群を作る列と行を見分ける列を取り出す関数

    Parameters
    ----------
    path: Path
        表 (1 行が 1 つの対象)
    cfg: Any
        config (group.columns と group.id)

    Returns
    ----------
    frame: pd.DataFrame
        群を作る列
    ids: Optional[pd.Series]
        行を見分ける列 (答えていなければ None)

    Raises
    ----------
    ValueError
        列が無い場合と, 群の要約の列と同じ名前の列がある場合
    """
    # 問題の設定の読み方で読み, 答えた列があるか
    table = load_data(path, loader=LOADER, **OPTIONS).frame
    names = list(cfg.group.columns or [])
    if not names:
        raise ValueError("config の group.columns に群を作る列を書いて"
                         "ください")
    wanted = names + ([cfg.group.id] if cfg.group.id else [])
    missing = [name for name in wanted if name not in table.columns]
    if missing:
        raise ValueError(f"表に {', '.join(map(str, missing))} の列があり"
                         "ません. config の group の列の名前を直してください")
    taken = [name for name in names if name in SUMMARY_COLUMNS]
    if taken:
        raise ValueError(f"群を作る列 {', '.join(taken)} は群の要約の列と同じ"
                         "名前です. 名前を変えた列を作ってから選んでください")
    frame = table[names].reset_index(drop=True)
    ids = (table[cfg.group.id].reset_index(drop=True) if cfg.group.id
           else None)

    return frame, ids


def split_rows(count: int, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    行を無作為に学習用と検証用に分ける関数

    Parameters
    ----------
    count: int
        行の数
    cfg: Any
        config (data.valid の割合, group.k と乱数の種)

    Returns
    ----------
    train: np.ndarray
        学習用の行の番号 (小さい順)
    valid: np.ndarray
        検証用の行の番号 (小さい順)

    Raises
    ----------
    ValueError
        割合の誤りと, 検証用が 2 行より少ないか学習用が群の数より少ない場合
    """
    # 割合の確かめ
    if not 0 < cfg.data.valid < 1:
        raise ValueError("config の data.valid は 0 より大きく 1 より小さく"
                         "してください")

    # 乱数の種で並べ替えて, 前を検証用に
    order = np.random.default_rng(cfg.experiment.seed).permutation(count)
    size = int(round(count * cfg.data.valid))
    valid, train = np.sort(order[:size]), np.sort(order[size:])
    if len(valid) < 2 or len(train) < cfg.group.k:
        raise ValueError(f"行が少なく ({count} 行), 検証用が 2 行以上と, "
                         f"学習用が群の数 ({cfg.group.k}) 以上になりません. "
                         "行を足すか, config の data.valid か group.k を見直"
                         "してください")

    return train, valid


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


def spec_errors(spec: Optional[Path], width: int) -> list[str]:
    """
    ML-ModelControl の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)
    width: int
        符号化した行の幅

    Returns
    ----------
    errors: list[str]
        誤りの文 (定義が無いか誤りが無ければ空)
    """
    # 定義の無いうちは ML-ModelControl を読まない. 読めない定義も誤りと
    # して返る (例外にならない)
    if spec is None:
        return []
    from ml_modelcontrol import validate
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]
    if errors:
        return errors

    # 入力は行の特徴, 出力に埋め込みと類似度の表
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    inputs = loaded.get("inputs") or {}
    if list(inputs) != [FEATURE_INPUT]:
        errors.append(f"定義の入力は {FEATURE_INPUT} の 1 つにしてください "
                      f"(今は {list(inputs)})")
    for name in (EMBEDDING_OUTPUT, SIMILARITY_OUTPUT):
        if name not in (loaded.get("outputs") or {}):
            errors.append(f"定義の出力に {name} を入れてください")
    entry = inputs.get(FEATURE_INPUT)
    shape = entry.get("shape") if isinstance(entry, dict) else None
    if isinstance(shape, list) and shape and shape[-1] != width:
        errors.append(f"入力 {FEATURE_INPUT} の最後の大きさを符号化した行の"
                      f"幅 {width} にしてください (今は {shape[-1]})")

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
        符号化した行の幅
    cfg: Any
        config (仮のモデルの埋め込みの幅と乱数の種)
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


def summary_rows(
    frame: pd.DataFrame, labels: np.ndarray, jaccard: np.ndarray, k: int
) -> list[dict]:
    """
    群ごとの行の数, 割合, 安定性と, 群を作った列の要約を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        全ての行の, 群を作る列 (元の値)
    labels: np.ndarray
        (全ての行の数,) の群の番号
    jaccard: np.ndarray
        (群の数,) の群ごとの Jaccard 係数の平均
    k: int
        群の数

    Returns
    ----------
    rows: list[dict]
        群の番号の順の行と, 最後に全ての行 (group は「全体」)
    """
    # 数の列は平均, 種類の列は最も多い値と割合 (行の無い群は None)
    rows = []
    for group in [*range(k), None]:
        part = frame if group is None else frame[labels == group]
        overlap = None if group is None else jaccard[group]
        row: dict = {
            "group": "全体" if group is None else group, "rows": len(part),
            "share": len(part) / len(frame),
            "stability": (None if overlap is None or np.isnan(overlap)
                          else float(overlap)),
        }
        for name in frame.columns:
            column = part[name]
            if is_number(frame[name]):
                row[name] = (float(column.astype(float).mean())
                             if column.notna().any() else None)
                continue
            counts = column.map(category_text).value_counts()
            row[name] = (f"{counts.index[0]} "
                         f"({counts.iloc[0] / len(part) * 100:.0f} %)"
                         if len(part) else None)
        rows.append(row)

    return rows


def write_results(
    run_dir: Path, ids: Optional[pd.Series], labels: np.ndarray,
    valid: np.ndarray, encoding: dict, centers: np.ndarray, cfg: Any
) -> None:
    """
    行ごとの群の割り当てと, 列の符号化と群の中心を実験のディレクトリに書く
    関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    ids: Optional[pd.Series]
        行を見分ける列 (無ければ行の番号)
    labels: np.ndarray
        (全ての行の数,) の群の番号
    valid: np.ndarray
        検証用の行の番号
    encoding: dict
        列の符号化
    centers: np.ndarray
        (群の数, 埋め込みの幅) の群の中心
    cfg: Any
        config (群の数)

    Returns
    ----------
    None
    """
    # 行の番号か行を見分ける列, 群, 分割
    split = np.full(len(labels), "train", dtype=object)
    split[valid] = "valid"
    key = str(cfg.group.id) if ids is not None else "row"
    table = pd.DataFrame({key: ids if ids is not None
                          else np.arange(len(labels)),
                          "group": labels, "split": split})
    table.to_csv(run_dir / ASSIGNMENT_FILE, index=False)

    # 群の中心は埋め込みの空間 (学習した重みで行を埋め込んで割り当てる)
    text = json.dumps({"k": int(cfg.group.k), "encoding": encoding,
                       "centers": centers.tolist()},
                      ensure_ascii=False, indent=2, default=str)
    (run_dir / CENTERS_FILE).write_text(text + "\\n", encoding="utf-8")


def compare_row(found: dict, base: dict, cfg: Any) -> dict:
    """
    群の安定性とまとまりを, 列の特徴をそのまま分けた群と並べた 1 行にする
    関数

    Parameters
    ----------
    found: dict
        埋め込みの群の ari, ari_min, silhouette
    base: dict
        列の特徴をそのまま分けた群の ari, ari_min, silhouette
    cfg: Any
        config (群の数と安定性の数え方)

    Returns
    ----------
    row: dict
        k, method, repeats と, 指標ごとの値, _baseline と _diff
    """
    # 差はモデル − そのまま
    row: dict = {"k": int(cfg.group.k), "method": str(cfg.group.stability),
                 "repeats": int(cfg.group.repeats)}
    for name in ("ari", "ari_min", "silhouette"):
        row[name] = found[name]
        row[f"{name}_baseline"] = base[name]
        row[f"{name}_diff"] = found[name] - base[name]

    return row


def check_config(cfg: Any) -> Optional[str]:
    """
    config の群の設定の誤りを返す関数

    Parameters
    ----------
    cfg: Any
        config (group)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (無ければ None)
    """
    # 群の数は 2 以上の整数, 作り直しは 1 回以上, 数え方は名前の中から
    group = cfg.group
    message = None
    if not isinstance(group.k, int) or isinstance(group.k, bool) or (
        group.k < 2
    ):
        message = (f"config の group.k は 2 以上の整数にしてください (今は "
                   f"{group.k})")
    elif not isinstance(group.repeats, int) or group.repeats < 1:
        message = ("config の group.repeats は 1 以上の整数にしてください "
                   f"(今は {group.repeats})")
    elif group.stability not in STABILITY_NAMES:
        message = (f"config の group.stability は "
                   f"{' か '.join(STABILITY_NAMES)} にしてください (今は "
                   f"{group.stability})")
    elif not 0 < group.fraction <= 1:
        message = ("config の group.fraction は 0 より大きく 1 以下にして"
                   "ください")

    return message


def main() -> None:
    """
    表の行の符号化器を対照学習で学び, 埋め込みを群に分けて, 群の安定性と
    群の要約を残す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config と表 (無ければ置き場の案内)
    cfg, diff = load_project_config(str(PROJECT_DIR))
    path = PROJECT_DIR / cfg.data.dataset
    if not path.is_file():
        print(f"{path} がありません. README の「次にすること」の 1 で, "
              "群を作る表を置いてください")
        sys.exit(1)
    message = check_config(cfg)
    if message:
        print(message)
        sys.exit(1)

    # 群を作る列を読み, 無作為に分け, 学習用の行で決めた符号化をかける
    try:
        frame, ids = read_rows(path, cfg)
        train, valid = split_rows(len(frame), cfg)
    except ValueError as error:
        print(error)
        sys.exit(1)
    encoding = make_encoding(frame.iloc[train])
    features = encode_rows(frame, encoding)
    width = features.shape[1]

    # モデルの定義の確かめ (誤りなら出力先を作る前に止まる)
    spec = project_spec(cfg)
    errors = spec_errors(spec, width)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\\n".join(f"- {message}" for message in errors))
        sys.exit(1)

    # 何の実験か (受け持ち, 出力の形, まとめるもの, 作り方は問題の設定から).
    # TODO: 既製のモデルを元にするなら, 元にした実験か重みを task["base"]
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    write_task(experiment, {"role": ROLE, "output": OUTPUT,
                            "what": PREDICT, "method": METHOD})

    # 行の数, 列の数, 符号化した行の幅, 群の数
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}行: train {len(train)} 行, "
        f"valid {len(valid)} 行. 列: 数 {len(encoding['numeric'])}, 種類 "
        f"{len(encoding['categories'])} (符号化した行の幅: {width}). 群の"
        f"数: {cfg.group.k}"
    )
    train_x = torch.tensor(features[train])
    valid_x = torch.tensor(features[valid])
    train_loader = DataLoader(TensorDataset(train_x),
                              batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_x),
                              batch_size=cfg.data.batch_size, shuffle=False)

    # モデルの組み立てと学習 (定義がまだ無ければ仮のモデル)
    spec, base_dir = spec_file(spec, experiment)
    if spec is None:
        experiment.logger.info(
            f"モデルの定義 ({cfg.model.spec or '場所なし'}) が無いので, "
            "仮のモデル (PlaceholderModel) で学習します. ML-CreateArchitecture"
            " で組んだ定義をその場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(width, cfg, spec, base_dir)
    trainer = GroupTrainer(model, experiment, train_x, valid_x, cfg)
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の設定の場所
    # origin も残る. モデルの定義は ML-Found が start_experiment で写す)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")

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

    # 全ての行の埋め込みを, 学習用の行で決めた群の中心に割り当てる
    model = trainer.model.cpu().eval()
    points = embed(model, torch.tensor(features), cfg.data.batch_size)
    rng = np.random.default_rng(cfg.experiment.seed)
    centers = kmeans(points[train], cfg.group.k, rng, cfg.group.iterations,
                     cfg.group.inits)
    labels = assign(points, centers)
    found = stability(points, train, labels, cfg)
    pick = np.random.default_rng(cfg.experiment.seed)
    found["silhouette"] = silhouette(points, labels, cfg.group.sample, pick)

    # 比べる相手: 符号化した列の特徴をそのまま同じ k-means で分けた群
    plain = features.astype(np.float64)
    rng = np.random.default_rng(cfg.experiment.seed)
    plain_labels = assign(plain, kmeans(plain[train], cfg.group.k, rng,
                                        cfg.group.iterations,
                                        cfg.group.inits))
    base = stability(plain, train, plain_labels, cfg)
    pick = np.random.default_rng(cfg.experiment.seed)
    base["silhouette"] = silhouette(plain, plain_labels, cfg.group.sample,
                                    pick)

    # 群の安定性と群の要約を evaluations/ へ, 割り当てと中心を実験へ
    method = STABILITY_NAMES[cfg.group.stability]
    experiment.logger.info(
        f"群の安定性 ({method}, {cfg.group.repeats} 回): 一致 (ari) "
        f"{found['ari']:.4f} (列の特徴をそのまま {base['ari']:.4f}), "
        f"シルエット係数 {found['silhouette']:.4f}"
    )
    experiment.logger.info(
        "群ごとの行の数: " + ", ".join(
            f"{group} {int((labels == group).sum())} 行"
            for group in range(cfg.group.k)
        )
    )
    save_evaluation(
        experiment, STABILITY, [compare_row(found, base, cfg)], "all",
        description=STABILITY_TEXT.format(
            method=method, repeats=cfg.group.repeats, weights=weights
        ),
        baseline=BASELINE_NAME, epoch=epoch, better=STABILITY_BETTER
    )
    save_evaluation(
        experiment, SUMMARY,
        summary_rows(frame, labels, found["jaccard"], cfg.group.k), "all",
        description=SUMMARY_TEXT.format(weights=weights), epoch=epoch,
        better=SUMMARY_BETTER
    )
    write_results(run_dir, ids, labels, valid, encoding, centers, cfg)

    # 推論の一式は作らない
    experiment.logger.info(
        "推論の一式は作りません. 実験に, 学習した重み, 行ごとの群の割り当て "
        f"({ASSIGNMENT_FILE}) と, 列の符号化と群の中心 ({CENTERS_FILE}) が"
        "残ります"
    )


if __name__ == "__main__":
    main()
'''

# config の基準 (ML-Found のプロジェクトの形)
GROUP_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題の設定から書き出した骨組み)
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
  # 問題の設定の「最良を選ぶ指標」(valid/silhouette は検証用の行の埋め込みの
  # 群のシルエット係数で, 大きいほど良い. valid/loss は検証用の行の対照学習の
  # 損失で, 小さいほど良い)
  monitor: $monitor
  mode: $mode

# プロジェクト固有の設定
data:
  # 群を作る表 (1 行が 1 つの対象. プロジェクトのディレクトリから. 問題の
  # 設定の手元のデータの読み方で読む)
  dataset: $dataset
  # 行のバッチの大きさ (バッチの中の他の行が対照学習の負例)
  batch_size: 128
  # 無作為に分ける, 検証用の行の割合
  valid: $valid

group:
  # 群を作る列 (数の列は標準化, 他の列は値ごとに 0 と 1 の列) と, 行を
  # 見分ける列 (群の割り当ての表に書く. null なら行の番号)
  columns: $columns
  id: $id
  # 群の数
  k: $k
  # 群の安定性の数え方 (seed は種を変えて, sample は学習用の行の fraction
  # の割合を引き直して, repeats 回作り直す)
  stability: $stability
  repeats: 5
  fraction: 0.8
  # k-means (Lloyd 法) の繰り返しの上限と, 中心の置き方を変える回数 (群の
  # 中心までの二乗距離の和が最も小さいものを使う)
  iterations: 100
  inits: 5
  # 対照学習で崩す (他の行の値に置き換える) 特徴の割合
  corruption: 0.3
  # シルエット係数を数える行の数の上限 (多ければ無作為に選ぶ)
  sample: 2000

model:
  # ML-ModelControl の形のモデルの定義 (ML-CreateArchitecture で組んで, この
  # 場所に保存する. 入力は行の特徴 x, 出力に埋め込み embedding と類似度の表
  # similarity)
  spec: models/model.yaml
  # spec のファイルが無いうちは train.py の仮のモデルで学習することの目印
  # (ML-Dashboard の学習の準備が読む. train.py は読まず, 仮のモデルを使うか
  # は spec のファイルがあるかで決まる)
  placeholder: true
  # 仮のモデルの埋め込みの幅
  width: 16
''')


def group_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 行の符号化器を学んで群に分けるスクリプトを作る関数

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
    # 頭 (表の読み方は問題の設定の手元のデータの読み方) と学習の流れ
    options = get_value(problem, "data.options")
    head = Template(GROUP_HEAD).substitute(
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        method=literal(active_value(problem, f"{PART}.method")),
        loader=literal(get_value(problem, "data.loader")),
        options=literal(options if isinstance(options, dict) else {}),
        summary_columns=literal(SUMMARY_COLUMNS),
    )
    # train.log の初めの行のまとめるもの (空なら省く)
    made = ("まとめるもの: {PREDICT}. "
            if active_value(problem, f"{PART}.what") else "")
    script = head + Template(GROUP_MAIN).substitute(predict_log=made)

    return script


def group_config(
    experiment: str, stages: dict, dataset: str, requires: str
) -> str:
    """
    まとめるの config の基準を作る関数

    Parameters
    ----------
    experiment: str
        実験の名前
    stages: dict
        受け渡し
    dataset: str
        群を作る表の場所 (プロジェクトのディレクトリから)
    requires: str
        一番上の requires の行 (skeleton.requires_text)

    Returns
    ----------
    text: str
        configs/default/train.yaml の中身
    """
    # 監視は受け渡しの値, 列と群の数と安定性の数え方は受け渡しの group
    training = stages["training"]
    group = stages["group"]
    text = GROUP_CONFIG.substitute(
        experiment=json.dumps(experiment, ensure_ascii=False),
        monitor=training["monitor"], mode=training["mode"],
        dataset=json.dumps(dataset, ensure_ascii=False), requires=requires,
        valid=GROUP_VALID,
        columns=json.dumps(group["columns"], ensure_ascii=False),
        id=json.dumps(group["id"], ensure_ascii=False),
        k=json.dumps(group["k"]),
        stability=json.dumps(group["stability"], ensure_ascii=False),
    )

    return text


def group_name(problem: dict) -> str:
    """
    まとめるの骨組みの, 実験の名前の元を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    raw: str
        群の数と受け持ち (group_4 など)
    """
    # 目的の列とタスクの代わり
    raw = f"group_{active_value(problem, f'{PART}.group.k')}"

    return raw


def group_readme(
    problem: dict, stages: dict, dataset: str, raw: Optional[str]
) -> str:
    """
    まとめるのプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    dataset: str
        群を作る表の場所 (プロジェクトのディレクトリから)
    raw: Optional[str]
        写した表の場所 (プロジェクトのディレクトリから. 入れていなければ
        None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    group = stages["group"]
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, "
             "ML-Found のプロジェクトの骨組み (1 行が 1 つの対象の表で, 行の"
             "埋め込みを学び, 群に分けて, 群の安定性と群の要約を残す).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1 から 4. データ, モデル, 学習, 推論
    monitor = stages["training"]["monitor"]
    columns = ", ".join(f"`{name}`" for name in group["columns"])
    names = {item["value"]: item["label"] for item in GROUP_STABILITIES}
    method = names.get(str(group["stability"]), str(group["stability"]))
    key = (f"行を見分ける列 `{group['id']}` (群の割り当ての表に書く)"
           if group["id"] else "行の番号")
    lines += [
        "", "## 次にすること", "",
        "学習する環境には ML-EDA, ML-Found (と, 定義を使うなら "
        "ML-ModelControl) を入れる.", "",
        "1. **データ**: 1 行が 1 つの対象 (顧客など) の表を config の "
        f"`data.dataset` の場所 (`{dataset}`) に置く. 群を作る列は {columns}"
        " (config の `group.columns`) で, 他の列は読まない (群の割り当てには"
        f"{key}). 数の列は標準化して欠けた値を 0 に, 他の列は値ごとに 0 と "
        "1 の列 (欠けた値も 1 つの値) にする (学習用の行で決める). 読み方は"
        "問題の設定の手元のデータの読み方 (ML-EDA の `load_data`)"
        + (f". 入れた表はこのディレクトリの `{raw}` に写した" if raw
           else ""),
        "2. **モデルの作成** (ML-ModelControl, ML-CreateArchitecture): "
        "ML-CreateArchitecture で組んだ定義を, config の `model.spec` の場所"
        " (`models/model.yaml`) に保存する. 入力は行の特徴 `x` ((件数, 符号"
        "化した行の幅) の float32) の 1 つ, 出力に埋め込み `embedding` と, "
        "全ての組の類似度の表 `similarity` ((件数, 件数)). 符号化した行の幅"
        "は `train.log` の初めの行に出る. ML-ModelControl の同梱の例 "
        "`contrastive_encoder` の入力の大きさと 1 段目の全結合の "
        "`in_features` を幅に合わせて使える. 定義が無いうちは, `train.py` "
        "の `PlaceholderModel` (印 TODO. 動くことを確かめるだけの小さな"
        "モデル) で学習する (config の `model.placeholder: true` がその目印)",
        "3. **学習** (ML-Found, ML-Dashboard): `python train.py`. 行を無作為"
        "に学習用と検証用 (config の `data.valid` の割合) に分ける. 学習は, "
        "行の特徴の一部 (config の `group.corruption` の割合) を他の行の値に"
        "置き換えた 2 通りの崩し方を, 同じ行どうしで近い埋め込みにする対照"
        "学習 (バッチの中の他の行と比べる交差エントロピー). 検証のたびに, "
        f"学習用の行の埋め込みを k-means で {group['k']} つの群 (config の "
        "`group.k`) に分け, 検証用の行を割り当てたシルエット係数 (群の"
        f"まとまり) を数える. 検証用の `{monitor}` が最も良いモデルを選ぶ. "
        "学習の後に, 全ての行の埋め込みを群に割り当て, 群の安定性を数える ("
        f"{method}. config の `group.stability` と `group.repeats` の回数. "
        "作り直した群と元の群の一致を調整ランド指数で数える). 結果は "
        "`outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, 学習した"
        "重み `final_model.pt`, 群の安定性の "
        "`evaluations/group_stability.json` (調整ランド指数の平均 `ari` と"
        "最も低い値 `ari_min`, シルエット係数 `silhouette` と, 列の特徴を"
        "そのまま k-means で分けた `_baseline` と差 `_diff`), 群の要約の "
        "`evaluations/group_summary.json` (群ごとの行の数, 割合, 群ごとの"
        "安定性 `stability` と, 数の列の平均, 種類の列の最も多い値. 最後の"
        "行は全ての行), 行ごとの群の割り当て `group_assignments.csv`, 列の"
        "符号化と群の中心 `group_centers.json`)",
        "4. **推論**: 推論の一式は作らない. 新しい行の群は, "
        "`group_centers.json` の符号化をかけ, 学習した重みで埋め込み, 最も"
        "近い群の中心に割り当てて求める",
        "", "`train.py` の印 (TODO) の所 (モデル, 特徴, 崩し方) は書き足す. "
        "問題の設定を変えたら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
