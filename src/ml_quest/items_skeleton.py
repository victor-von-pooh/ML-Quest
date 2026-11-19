import json
from pathlib import Path
from string import Template
from typing import Callable, Optional

from ml_quest.handoff import describe
from ml_quest.problem import active_value
from ml_quest.questions import PART

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
ITEMS_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

ML-EDA の版 2 のデータセット (ファイルの組) で学習する骨組みなので, 印
(TODO) の所を書き足す
"""
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml_eda import ItemDataset, ItemSplit, load_items
from ml_found import (
    Experiment, Trainer, describe_origin, load_project_config,
    save_evaluation, start_experiment, task_from_dataset, write_task
)

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 予測すること
PROBLEM_NAME = $name
PREDICT = $predict

# 正解の種類と, task.yaml に足す出力の形と作り方 (問題の設定の答え)
TARGET_KIND = "$kind"
OUTPUT = $output
METHOD = $method

# 分割の名前 (学習用, 検証用, テスト用)
SPLITS = ("train", "valid", "test")
'''

# 画像の領域: 画素ごとの交差エントロピーと, 背景を除いた IoU の平均
MASK_PART = '''
# テスト用の評価の名前と説明 (evaluations/<名前>.json)
EVALUATION = "test_iou"
EVALUATION_TEXT = ("テスト用の分割のクラスごとの IoU (最良のモデル). mean は"
                   "背景 (値 0) を除いた平均")


class MaskDataset(Dataset):
    def __init__(self, items: ItemSplit, names: tuple, lookup: np.ndarray):
        # 親クラスのコンストラクタを呼び出す
        super(MaskDataset, self).__init__()

        # ML-EDA の分割の件の並び, 入力と正解の項目の名前, 領域の値から
        # クラスの番号の表を保存
        self.items = items
        self.names = names
        self.lookup = lookup

    def __len__(self) -> int:
        """
        件数を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        count: int
            分割の件数
        """
        # ML-EDA の並びの件数
        count = len(self.items)

        return count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1 件の画像と領域をテンソルにする関数

        Parameters
        ----------
        index: int
            分割の中の位置

        Returns
        ----------
        pair: tuple[torch.Tensor, torch.Tensor]
            (チャンネル, 高さ, 幅) の 0 から 1 の画像と, (高さ, 幅) の
            クラスの番号 (知らない値は -1)
        """
        # (高さ, 幅, チャンネル) の uint8 と (高さ, 幅) の整数. 表の外の値
        # (名前の無い値, 負の値) は -1. TODO: 大きさのそろわない画像なら,
        # ここで大きさの揃え
        item = self.items[index]
        image = torch.from_numpy(item[self.names[0]]).permute(2, 0, 1)
        values = np.asarray(item[self.names[1]]).astype(np.int64)
        inside = (values >= 0) & (values < len(self.lookup))
        labels = np.where(
            inside, self.lookup[np.clip(values, 0, len(self.lookup) - 1)], -1
        )
        pair = (image.float() / 255.0, torch.from_numpy(labels))

        return pair


class PlaceholderModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 入力のチャンネル数とクラス数を保存
        self.in_channels = in_channels
        self.num_classes = num_classes

        # TODO: 課題に合うモデル (U-Net など) に置き換える. 今は動くことを
        # 確かめるための, 3x3 の畳み込み 2 段
        self.conv1 = nn.Conv2d(self.in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, self.num_classes, 3, padding=1)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, チャンネル, 高さ, 幅) の画像

        Returns
        ----------
        logits: torch.Tensor
            (バッチ, クラス数, 高さ, 幅) の画素ごとのスコア
        """
        # 畳み込み, 活性化, 畳み込み
        logits = self.conv2(self.relu(self.conv1(x)))

        return logits


class MaskTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, names: list[str],
        foreground: list[int]
    ):
        # 親クラスのコンストラクタを呼び出す (step を上書きするので
        # step_fn は None)
        super(MaskTrainer, self).__init__(model, experiment, None)

        # クラスの名前, 背景でないクラスの番号, 損失関数 (知らない値の -1 は
        # 除く) と, エポックごとの重なりと和の画素数を保存
        self.names = names
        self.foreground = foreground
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.intersection = torch.zeros(len(names))
        self.union = torch.zeros(len(names))

    def on_epoch_start(self, phase: str) -> None:
        """
        エポックの始めに画素数の入れ物を空にする関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        None
        """
        # クラスごとの重なりと和の 0 への戻し
        self.intersection = torch.zeros(len(self.names))
        self.union = torch.zeros(len(self.names))

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の損失を求め, 重なりと和の画素数を貯める関数

        Parameters
        ----------
        batch: tuple
            画像とクラスの番号
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            画素ごとの交差エントロピー (loss)
        """
        # 画素ごとの分類の損失と, IoU のための画素数の貯め置き
        x, y = batch
        logits = self.model(x)
        loss = self.ce(logits, y)
        counts = overlap_counts(logits.argmax(dim=1), y, len(self.names))
        self.intersection += counts[0]
        self.union += counts[1]
        outputs = {"loss": loss}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        エポック全体の画素数から IoU を求める関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            背景を除いたクラスの IoU の平均 (iou)
        """
        # 最良のモデルの選びは検証の段階の値 (valid/iou)
        value = mean_iou(self.intersection, self.union, self.foreground)
        metrics = {"iou": value}

        return metrics


def class_table(dataset: ItemDataset) -> tuple[np.ndarray, list, list]:
    """
    説明の classes (無ければ values) から, 領域の値の番号の表を作る関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット

    Returns
    ----------
    table: tuple[np.ndarray, list, list]
        値から番号の表 (知らない値は -1), 番号の順の名前, 背景 (値 0) で
        ない番号
    """
    # 値の小さい順の番号 (名前も値の一覧も無ければ止める)
    entry = dataset.entries[entry_names(dataset)[1]]
    if not entry.get("classes") and not entry.get("values"):
        raise ValueError("領域の値の名前 (classes) も値の一覧 (values) も"
                         "ありません. 書き出しで classes を渡してください")
    classes = entry.get("classes") or [
        {"value": value, "name": str(value)} for value in entry["values"]
    ]
    values = sorted(int(item["value"]) for item in classes)
    names = {int(item["value"]): str(item["name"]) for item in classes}
    lookup = np.full(max(values) + 1, -1, dtype=np.int64)
    lookup[values] = np.arange(len(values))
    table = (lookup, [names[value] for value in values],
             [index for index, value in enumerate(values) if value != 0])

    return table


def overlap_counts(
    predicted: torch.Tensor, target: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    クラスごとの重なりと和の画素数を数える関数

    Parameters
    ----------
    predicted: torch.Tensor
        予測したクラスの番号
    target: torch.Tensor
        正解のクラスの番号 (知らない値の -1 は数えない)
    count: int
        クラス数

    Returns
    ----------
    counts: tuple[torch.Tensor, torch.Tensor]
        クラスごとの重なり (予測と正解の両方) と和 (どちらか) の画素数
    """
    # クラスごとの予測と正解の画素の集合の比べ
    known = target >= 0
    intersection = torch.zeros(count)
    union = torch.zeros(count)
    for index in range(count):
        guess = (predicted == index) & known
        truth = target == index
        intersection[index] = (guess & truth).sum().item()
        union[index] = (guess | truth).sum().item()
    counts = (intersection, union)

    return counts


def mean_iou(
    intersection: torch.Tensor, union: torch.Tensor, foreground: list[int]
) -> float:
    """
    背景でないクラスの IoU の平均を求める関数

    Parameters
    ----------
    intersection: torch.Tensor
        クラスごとの重なりの画素数
    union: torch.Tensor
        クラスごとの和の画素数
    foreground: list[int]
        背景でないクラスの番号

    Returns
    ----------
    value: float
        IoU の平均 (和が 0 のクラスは除く. 1 つも無ければ 0. 監視する
        指標が nan だと ML-Found が止まるため)
    """
    # 予測にも正解にも無いクラスは平均の対象外
    values = [(intersection[index] / union[index]).item()
              for index in foreground if union[index] > 0]
    value = float(np.mean(values)) if values else 0.0

    return value


def make_datasets(dataset: ItemDataset, cfg: object) -> dict:
    """
    load_items の分割を torch の Dataset に包む関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット
    cfg: object
        config (使わない. 点群とそろえるための引数)

    Returns
    ----------
    datasets: dict
        分割の名前から MaskDataset (件のある分割だけ)
    """
    # 値の番号の表は全ての分割で同じ
    lookup = class_table(dataset)[0]
    datasets = {}
    for name in SPLITS:
        if name in dataset.info["splits"]:
            datasets[name] = MaskDataset(
                dataset.split(name), entry_names(dataset), lookup
            )

    return datasets


def placeholder(dataset: ItemDataset, cfg: object) -> nn.Module:
    """
    仮のモデルを作る関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット
    cfg: object
        config (使わない. 点群とそろえるための引数)

    Returns
    ----------
    model: nn.Module
        入力のチャンネル数とクラス数を合わせた PlaceholderModel
    """
    # チャンネル数とクラス数はデータセットの説明から
    shape = dataset.entries[entry_names(dataset)[0]]["shape"]
    model = PlaceholderModel(shape[2] or 3, len(class_table(dataset)[1]))

    return model


def make_trainer(
    model: nn.Module, experiment: Experiment, dataset: ItemDataset
) -> Trainer:
    """
    損失と指標を持つ Trainer を作る関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    experiment: Experiment
        実験
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット

    Returns
    ----------
    trainer: Trainer
        MaskTrainer
    """
    # クラスの名前と背景でない番号
    _, names, foreground = class_table(dataset)
    trainer = MaskTrainer(model, experiment, names, foreground)

    return trainer


def evaluate_test(
    trainer: Trainer, loader: DataLoader, experiment: Experiment
) -> list[dict]:
    """
    テスト用の分割のクラスごとの IoU を記録する関数

    Parameters
    ----------
    trainer: Trainer
        学習した後の MaskTrainer (最良のモデルの重み)
    loader: DataLoader
        テスト用の DataLoader
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    rows: list[dict]
        class と iou (和が 0 のクラスは None) の行と, 最後に mean の行
    """
    # 学習と同じ数え方で, 最良のモデルの予測の重なりと和
    model = trainer.model.cpu().eval()
    count = len(trainer.names)
    intersection, union = torch.zeros(count), torch.zeros(count)
    with torch.no_grad():
        for x, y in loader:
            counts = overlap_counts(model(x).argmax(dim=1), y, count)
            intersection += counts[0]
            union += counts[1]
    rows = [{"class": name, "iou": (intersection[index] / union[index]).item()
             if union[index] > 0 else None}
            for index, name in enumerate(trainer.names)]
    rows.append({"class": "mean", "iou": mean_iou(intersection, union,
                                                  trainer.foreground)})
    experiment.logger.info(f"テスト用の IoU (背景を除く平均): "
                           f"{rows[-1]['iou']:.4f}")

    return rows

$baseline_comment
'''

# 点群: 2 乗の Chamfer で学び, 検証したエポックだけ Chamfer 距離を記録
CLOUD_PART = '''
# テスト用の評価の名前と説明 (evaluations/<名前>.json)
EVALUATION = "test_chamfer"
EVALUATION_TEXT = ("テスト用の分割の Chamfer 距離 (最良のモデル. 単位は"
                   "データセットのまま)")


class CloudDataset(Dataset):
    def __init__(
        self, items: ItemSplit, names: tuple, counts: tuple, seed: int
    ):
        # 親クラスのコンストラクタを呼び出す
        super(CloudDataset, self).__init__()

        # ML-EDA の分割の件の並び, 入力と正解の項目の名前, 選ぶ点の数 (入力
        # と正解), 点を選ぶ乱数の種を保存
        self.items = items
        self.names = names
        self.counts = counts
        self.seed = seed

    def __len__(self) -> int:
        """
        件数を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        count: int
            分割の件数
        """
        # ML-EDA の並びの件数
        count = len(self.items)

        return count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1 件の入力の点と正解の点群をテンソルにする関数

        Parameters
        ----------
        index: int
            分割の中の位置

        Returns
        ----------
        pair: tuple[torch.Tensor, torch.Tensor]
            (入力の点の数, 3) と (正解の点の数, 3) の float32
        """
        # 件ごとに決まった乱数での選び. 撮影 (辞書) は深度から点に
        item = self.items[index]
        rng = np.random.default_rng([self.seed, index])
        source = item[self.names[0]]
        points = backproject(source) if isinstance(source, dict) else source
        pair = (
            torch.from_numpy(choose(points, self.counts[0], rng)),
            torch.from_numpy(choose(item[self.names[1]], self.counts[1], rng)),
        )

        return pair


class PlaceholderModel(nn.Module):
    def __init__(self, num_points: int):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 出す点の数を保存
        self.num_points = num_points

        # TODO: 課題に合うモデルに置き換える. 今は動くことを確かめるための,
        # 点ごとの全結合と最大値のまとめから決まった数の点を出す小さな層
        self.encode1 = nn.Linear(3, 64)
        self.encode2 = nn.Linear(64, 128)
        self.decode1 = nn.Linear(128, 256)
        self.decode2 = nn.Linear(256, self.num_points * 3)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 入力の点の数, 3) の座標

        Returns
        ----------
        points: torch.Tensor
            (バッチ, 出す点の数, 3) の座標
        """
        # 点ごとの特徴, 点の並びに依らない最大値のまとめ, 座標への戻し
        h = self.relu(self.encode2(self.relu(self.encode1(x))))
        feature = h.max(dim=1).values
        h = self.decode2(self.relu(self.decode1(feature)))
        points = h.view(-1, self.num_points, 3)

        return points


class CloudTrainer(Trainer):
    def __init__(self, model: nn.Module, experiment: Experiment):
        # 親クラスのコンストラクタを呼び出す (step を上書きするので
        # step_fn は None)
        super(CloudTrainer, self).__init__(model, experiment, None)

        # 検証の予測と正解の入れ物
        self.predictions: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

    def on_epoch_start(self, phase: str) -> None:
        """
        エポックの始めに予測と正解の入れ物を空にする関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        None
        """
        # 入れ物の片付け
        self.predictions = []
        self.targets = []

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の Chamfer の損失を求め, 検証では予測を貯める関数

        Parameters
        ----------
        batch: tuple
            入力の点と正解の点群
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            2 乗の距離の Chamfer (loss) とバッチの件数
        """
        # 予測と, 2 乗の距離の Chamfer (学習の損失)
        x, y = batch
        predicted = self.model(x)
        loss = chamfer(predicted, y, True).mean()
        if phase == "valid":
            self.predictions.append(predicted.detach().cpu())
            self.targets.append(y.detach().cpu())
        outputs = {"loss": loss, "batch_size": len(y)}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証したエポックだけ, Chamfer 距離を求める関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証では chamfer (距離の平均), 他は空
        """
        # 重い指標は検証の段階だけ. 検証の段階は trainer.valid_every_n_epochs
        # ごとと最後のエポックだけ
        if phase != "valid" or not self.predictions:
            metrics = {}
            return metrics
        with torch.no_grad():
            distance = chamfer(
                torch.cat(self.predictions), torch.cat(self.targets), False
            )
        metrics = {"chamfer": distance.mean().item()}

        return metrics


def chamfer(
    predicted: torch.Tensor, target: torch.Tensor, squared: bool
) -> torch.Tensor:
    """
    1 件ごとの対称な Chamfer 距離を求める関数

    Parameters
    ----------
    predicted: torch.Tensor
        (バッチ, 予測の点の数, 3)
    target: torch.Tensor
        (バッチ, 正解の点の数, 3)
    squared: bool
        最も近い点までの距離を 2 乗するか (学習の損失は 2 乗)

    Returns
    ----------
    distance: torch.Tensor
        (バッチ,) の, 両向きの最も近い点までの距離の平均の和の半分
    """
    # 全ての組の距離から, 両向きに最も近い点
    pairs = torch.cdist(predicted, target)
    if squared:
        pairs = pairs ** 2
    forward = pairs.min(dim=2).values.mean(dim=1)
    backward = pairs.min(dim=1).values.mean(dim=1)
    distance = (forward + backward) / 2

    return distance


def backproject(capture: dict) -> np.ndarray:
    """
    撮影の深度を, 撮影の世界の座標の点に戻す関数

    Parameters
    ----------
    capture: dict
        load_items の撮影 (intrinsics, poses, depth, convention)

    Returns
    ----------
    points: np.ndarray
        (点の数, 3) の点

    Raises
    ----------
    ValueError
        扱えない座標の決まり, 深度が無い場合, 点が無い場合
    """
    # 扱うのは OpenGL の軸 (+X 右, +Y 上, -Z 前) と, カメラの前の向きの
    # 距離の深度 (z) だけ
    convention = capture["convention"]
    axes = str(convention.get("camera_axes", "OpenGL"))
    if not axes.startswith("OpenGL") or convention.get("depth") != "z":
        raise ValueError(f"扱えない撮影の座標の決まりです: {convention}")
    if capture["depth"] is None:
        raise ValueError("深度の無い枚があります")
    intr = capture["intrinsics"]
    center = float(convention.get("pixel_center", 0.5))
    clouds = []
    for pose, depth in zip(capture["poses"], capture["depth"]):
        if not np.all(np.isfinite(pose)):
            continue
        rows, columns = np.nonzero(depth > 0)
        value = depth[rows, columns].astype(np.float64)

        # 深度の大きさが画像と違う場合の, カメラの内部の値の縮め
        sx = depth.shape[1] / intr["w"] if intr.get("w") else 1.0
        sy = depth.shape[0] / intr["h"] if intr.get("h") else 1.0

        # 画素の中心を通る光線の点の, カメラの座標から世界の座標への変換
        camera = np.stack([
            (columns + center - intr["cx"] * sx) / (intr["fl_x"] * sx)
            * value,
            -(rows + center - intr["cy"] * sy) / (intr["fl_y"] * sy)
            * value,
            -value,
        ], axis=1)
        clouds.append(camera @ pose[:3, :3].T + pose[:3, 3])
    points = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    if len(points) == 0:
        raise ValueError("深度から戻せる点がありません")

    return points


def choose(
    points: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """
    点群から決まった数の点を選ぶ関数

    Parameters
    ----------
    points: np.ndarray
        (点の数, 3) の点
    count: int
        選ぶ数 (足りなければ重ねて選ぶ)
    rng: np.random.Generator
        選ぶ乱数

    Returns
    ----------
    chosen: np.ndarray
        (count, 3) の float32
    """
    # 点が足りないときだけ重ねての選び
    if len(points) == 0:
        raise ValueError("点の無い点群があります")
    index = rng.choice(len(points), count, replace=len(points) < count)
    chosen = np.asarray(points)[index].astype(np.float32)

    return chosen


def make_datasets(dataset: ItemDataset, cfg: object) -> dict:
    """
    load_items の分割を torch の Dataset に包む関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット
    cfg: object
        config (入力と正解の点の数と乱数の種)

    Returns
    ----------
    datasets: dict
        分割の名前から CloudDataset (件のある分割だけ)
    """
    # 分割ごとに別の乱数の種
    counts = (cfg.data.input_points, cfg.data.target_points)
    datasets = {}
    for offset, name in enumerate(SPLITS):
        if name in dataset.info["splits"]:
            datasets[name] = CloudDataset(
                dataset.split(name), entry_names(dataset), counts,
                cfg.data.seed + offset
            )

    return datasets


def placeholder(dataset: ItemDataset, cfg: object) -> nn.Module:
    """
    仮のモデルを作る関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット (使わない. 領域とそろえるための引数)
    cfg: object
        config (出す点の数)

    Returns
    ----------
    model: nn.Module
        PlaceholderModel
    """
    # 出す点の数は config から
    model = PlaceholderModel(cfg.model.output_points)

    return model


def make_trainer(
    model: nn.Module, experiment: Experiment, dataset: ItemDataset
) -> Trainer:
    """
    損失と指標を持つ Trainer を作る関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    experiment: Experiment
        実験
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット (使わない. 領域とそろえるための引数)

    Returns
    ----------
    trainer: Trainer
        CloudTrainer
    """
    # 点群の Trainer
    trainer = CloudTrainer(model, experiment)

    return trainer


def evaluate_test(
    trainer: Trainer, loader: DataLoader, experiment: Experiment
) -> list[dict]:
    """
    テスト用の分割の Chamfer 距離を記録する関数

    Parameters
    ----------
    trainer: Trainer
        学習した後の CloudTrainer (最良のモデルの重み)
    loader: DataLoader
        テスト用の DataLoader
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    rows: list[dict]
        answer (model) と chamfer の行
    """
    # 最良のモデルの予測と正解の距離の平均
    model = trainer.model.cpu().eval()
    distances = []
    with torch.no_grad():
        for x, y in loader:
            distances.append(chamfer(model(x), y, False))
    value = torch.cat(distances).mean().item()
    experiment.logger.info(f"テスト用の Chamfer 距離: {value:.5f}")
    rows = [{"answer": "model", "chamfer": value}]

    return rows

$baseline_comment
'''

# 学習の流れ (データセット, モデル, 学習, task.yaml, 評価)
ITEMS_MAIN = '''

def entry_names(dataset: ItemDataset) -> tuple[str, str]:
    """
    入力と正解の項目の名前を返す関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット

    Returns
    ----------
    names: tuple[str, str]
        入力 (role が input の最初の項目) と正解 (target.entries の最初)
    """
    # 説明の項目の役から
    inputs = [name for name, entry in dataset.entries.items()
              if entry["role"] == "input"]
    names = (inputs[0], dataset.info["target"]["entries"][0])

    return names


def make_model(dataset: ItemDataset, cfg: object) -> nn.Module:
    """
    モデルを作る関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA の版 2 のデータセット
    cfg: object
        config (model.spec に ML-ModelControl の定義の場所)

    Returns
    ----------
    model: nn.Module
        定義から組み立てたモデルか, 仮のモデル
    """
    # ML-ModelControl の定義を書いていれば組み立て (ML-ModelControl は任意
    # の依存なので, 使うときだけの読み込み)
    if cfg.model.spec:
        from ml_modelcontrol import build_model
        model = build_model(PROJECT_DIR / cfg.model.spec,
                            seed=cfg.experiment.seed)
        return model
    model = placeholder(dataset, cfg)

    return model


def main() -> None:
    """
    ML-EDA の版 2 のデータセットで学習し, テスト用の評価を残す関数

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
    dataset = load_items(path)
    info = dataset.info
    if info["target"]["kind"] != TARGET_KIND:
        print(f"正解の種類が {TARGET_KIND} ではありません: "
              f"{info['target']['kind']}")
        sys.exit(1)
    if "valid" not in info["splits"]:
        print("データセットに検証用の分割がありません. 最も良いモデルを選ぶ"
              "ために, 検証用の割合を 0 より大きくして書き出してください")
        sys.exit(1)

    # 何の実験か (受け持ち, 正解の種類と出どころはデータセットから, 出力の
    # 形, 何を出すか, 作り方は問題の設定から. 誤りは出力先を作る前に).
    # TODO: 既製のモデルを元にするなら, 元にした実験か重みを task["base"]
    task = task_from_dataset(cfg.data.dataset, project_dir=str(PROJECT_DIR))
    task.update({"output": OUTPUT, "what": PREDICT, "method": METHOD})
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    task = write_task(experiment, task)
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. 予測すること: {PREDICT}. データセット: "
        f"{info['name']} (版 {info['format_version']}), 分割 "
        f"{info['splits']}"
    )

    # load_items の分割の torch の Dataset への包み
    datasets = make_datasets(dataset, cfg)
    loaders = {
        name: DataLoader(part, batch_size=cfg.data.batch_size,
                         shuffle=name == "train")
        for name, part in datasets.items()
    }

    # モデルの組み立てと学習 (検証は trainer.valid_every_n_epochs ごとと
    # 最後のエポック)
    model = make_model(dataset, cfg)
    trainer = make_trainer(model, experiment, dataset)
    trainer.fit(loaders["train"], loaders["valid"])

    # 実験だけで見直せるように, 問題の設定とモデルの定義の写し
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    if cfg.model.spec:
        shutil.copy(PROJECT_DIR / cfg.model.spec, run_dir / "model_spec.yaml")

    # 結果の表示と, 指標が何との一致か
    best_epoch = trainer.checkpoint.best_epoch
    print(f"保存先: {run_dir}")
    print(f"ベストモデルのエポック: {best_epoch}")
    print(f"ベストの {cfg.trainer.monitor}: {trainer.checkpoint.best_value}")
    print(describe_origin(task["target"]["origin"]))

    # テスト用の分割があれば, 最良のモデルの評価を evaluations/ へ保存
    if "test" not in loaders:
        experiment.logger.info("テスト用の分割が無いので, 評価しません")
        return
    rows = evaluate_test(trainer, loaders["test"], experiment)
    save_evaluation(experiment, EVALUATION, rows, "test",
                    description=EVALUATION_TEXT, epoch=best_epoch)


if __name__ == "__main__":
    main()
'''

# 正解の種類ごとの部分, 出力の形の既定, 書き出しの関数の名前と入力
PARTS = {"mask": MASK_PART, "point_cloud": CLOUD_PART}
EXPORTERS = {
    "mask": ("export_image_masks", "画像のディレクトリ, 領域のディレクトリ",
             ", classes=<値から名前の辞書>"),
    "point_cloud": ("export_capture_clouds",
                    "撮影のディレクトリ, 点群のディレクトリ",
                    ", cloud_units=<点群の単位>"),
}

# config の基準 (ML-Found のプロジェクトの形)
ITEMS_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題の設定から書き出した骨組み)
experiment:
  name: $experiment
  seed: 0

optimizer:
  name: Adam
  params:
    lr: 0.001

trainer:
  epochs: 30$every
  # 問題の設定の「最良を選ぶ指標」
  monitor: $monitor
  mode: $mode
  early_stopping:
    use: true$patience_note
    patience: $patience

# プロジェクト固有の設定
data:
  # ML-EDA の版 2 のデータセット (プロジェクトのディレクトリから)
  dataset: $dataset
  batch_size: 8$cloud_data

model:
  # ML-ModelControl の形のモデルの定義 (ML-CreateArchitecture で組む. 空なら
  # train.py の仮のモデル)
  spec: null$cloud_model
''')

# 点群の検証の間隔と, 点の数
CLOUD_EVERY = ("\n  # Chamfer 距離は重いので, 2 エポックごとと最後のエポックだけ"
               "の検証\n  valid_every_n_epochs: 2")
CLOUD_DATA = ("\n  # 入力 (深度から戻した点) と正解から選ぶ点の数と, 選ぶ乱数の"
              "種\n  input_points: 512\n  target_points: 1024\n  seed: 0")
CLOUD_MODEL = "\n  # 仮のモデルの出す点の数\n  output_points: 256"


def items_script(
    problem: dict, kind: str, literal: Callable, comment_lines: Callable
) -> str:
    """
    問題の設定から, 版 2 のデータセットで学習するスクリプトを作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    kind: str
        正解の種類 (mask か point_cloud)
    literal: Callable
        値を Python の式にする関数 (skeleton.literal)
    comment_lines: Callable
        文を 1 行ずつのコメントにする関数 (skeleton.comment_lines)

    Returns
    ----------
    script: str
        train.py の中身
    """
    # 頭, 正解の種類ごとの部分, 学習の流れ
    head = Template(ITEMS_HEAD).substitute(
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        kind=kind,
        output=literal(active_value(problem, f"{PART}.output")),
        method=literal(active_value(problem, f"{PART}.method")),
    )
    baseline = comment_lines(
        "TODO: 比べる相手 ("
        f"{active_value(problem, f'{PART}.baseline')}) の値も評価の表に足す",
        indent=""
    )
    body = Template(PARTS[kind]).substitute(baseline_comment=baseline)
    script = head + body + ITEMS_MAIN

    return script


def items_config(
    experiment: str, stages: dict, kind: str, dataset: str
) -> str:
    """
    版 2 のデータセットで学習する config の基準を作る関数

    Parameters
    ----------
    experiment: str
        実験の名前
    stages: dict
        受け渡し
    kind: str
        正解の種類 (mask か point_cloud)
    dataset: str
        データセットの場所 (プロジェクトのディレクトリから)

    Returns
    ----------
    text: str
        configs/default/train.yaml の中身
    """
    # 点群だけ検証の間隔と点の数 (打ち切りの待つ回数は検証の回数)
    training = stages["training"]
    cloud = kind == "point_cloud"
    text = ITEMS_CONFIG.substitute(
        experiment=experiment, monitor=training["monitor"],
        mode=training["mode"],
        dataset=json.dumps(dataset, ensure_ascii=False),
        every=CLOUD_EVERY if cloud else "",
        patience_note="\n    # 検証の回数で数えた回数 (5 回 = 10 エポック)"
        if cloud else "",
        patience=5 if cloud else 10,
        cloud_data=CLOUD_DATA if cloud else "",
        cloud_model=CLOUD_MODEL if cloud else "",
    )

    return text


def origin_lines(problem: dict, info: Optional[dict]) -> list[str]:
    """
    README の, 正解の出どころの節を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    info: Optional[dict]
        入れたデータセットの説明 (無ければ None)

    Returns
    ----------
    lines: list[str]
        Markdown の行
    """
    # 問題の設定の答えと, 入れたデータセットの target.origin
    source = active_value(problem, f"{PART}.target.source")
    name = active_value(problem, f"{PART}.target.process.name")
    settings = active_value(problem, f"{PART}.target.process.settings")
    lines = ["", "## 正解の出どころ", ""]
    if source == "process":
        lines.append(
            f"正解は今の処理 ({name}"
            + (f". 設定: {settings}" if settings else "")
            + ") の出力. 指標はその処理との一致を測るので, モデルはその処理を"
            "超えられない (評価と結論でもそう書く)."
        )
    else:
        lines.append(f"正解の出どころ (問題の設定): `{source}`.")
    target = (info or {}).get("target")
    origin = target.get("origin") if isinstance(target, dict) else None
    if isinstance(origin, dict):
        lines.append(
            f"入れたデータセットの `target.origin`: `{origin.get('type')}`"
            + (f" ({origin['name']})" if origin.get("name") else "")
            + ". 学習のスクリプトが `task.yaml` に写す."
        )
    if source == "process" and isinstance(origin, dict) and (
        origin.get("type") != "process"
    ):
        lines.append(
            "問題の設定では今の処理の出力だが, データセットの出どころは "
            "process でない. 書き出し直すときに `origin` に処理の名前と設定を"
            "渡すと, `task.yaml` と評価に正しく残る."
        )

    return lines


def items_readme(
    problem: dict, stages: dict, kind: str, dataset: str,
    info: Optional[dict], raw: Optional[str]
) -> str:
    """
    版 2 のデータセットで学習するプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    kind: str
        正解の種類 (mask か point_cloud)
    dataset: str
        データセットの場所 (プロジェクトのディレクトリから)
    info: Optional[dict]
        入れたデータセットの説明 (版 2 のデータセットを入れていなければ
        None)
    raw: Optional[str]
        写した元のデータの場所 (版 2 のデータセットでないデータを写した
        場合. プロジェクトのディレクトリから)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約と, 正解の出どころ
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, "
             "ML-Found のプロジェクトの骨組み (ML-EDA の版 2 のデータセット"
             "で学習する).", "", "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]
    lines += origin_lines(problem, info)

    # 1. データ (入れた版 2 のデータセットか, 書き出しの案内)
    function, inputs, extra = EXPORTERS[kind]
    process = active_value(problem, f"{PART}.target.process.name")
    origin = (', origin={"type": "process", "name": ..., "settings": '
              '{"note": ...}}' if process else "")
    lines += ["", "## 次にすること", "",
              "学習する環境には ML-EDA, ML-Found (と, 定義を使うなら "
              "ML-ModelControl) を入れる.", ""]
    if info is not None:
        lines.append(f"1. **データ**: 入れた版 2 のデータセットを `{dataset}`"
                     " に写した. 学習はこれを読む. 書き出し直したら写し直す")
    else:
        lines.append(
            f"1. **データの分析** (コードなら ML-EDA の `{function}`): "
            f"`{function}({inputs}, \"{dataset}\"{extra}{origin})` で, この"
            "ディレクトリの `" + dataset + "` に書き出す. 検証用の分割は必ず"
            "作る" + (f". 入れたデータはこのディレクトリの `{raw}` に写した"
                      if raw else "")
        )
    if process and info is None:
        lines.append("    - 正解の出どころ `origin` に今の処理の名前と設定を"
                     "渡す (ML-EDA の `settings` は辞書. 問題の設定の文は "
                     "`{\"note\": ...}` で). `task.yaml` に写り, 指標が何と"
                     "の一致かが実験に残る")

    # 2 から 4. モデル, 学習, 推論
    every = ("検証 (Chamfer 距離) は `trainer.valid_every_n_epochs` ごとと"
             "最後のエポックだけ. " if kind == "point_cloud" else "")
    lines += [
        "2. **モデルの作成**: `train.py` の `PlaceholderModel` (印 TODO. 動く"
        "ことを確かめるだけの小さなモデル) を課題に合うモデルに置き換えるか,"
        " ML-CreateArchitecture で組んだ定義を `models/` に置き, config の "
        "`model.spec` にその場所を書く"
        + (". 入力は (バッチ, チャンネル, 高さ, 幅) の 0 から 1 の画像, 出力は"
           " (バッチ, クラス数, 高さ, 幅) のスコア" if kind == "mask" else
           ". 入力は (バッチ, 点の数, 3) の点, 出力は (バッチ, 点の数, 3) の"
           "点"),
        "3. **学習** (ML-Found, ML-Dashboard): `python train.py`. 結果は "
        "`outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, テスト用の"
        f"分割があれば `evaluations/`). {every}検証用の "
        f"`{stages['training']['monitor']}` が最も良いモデルを選ぶ",
        "4. **推論**: ML-Inference の一式がまだこの出力の形を持たないので, "
        "推論の一式は作らない",
        "", "`train.py` の印 (TODO) の所 (モデル, 比べる相手の評価など) は"
        "書き足す. 問題の設定を変えたら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text


def items_name(problem: dict) -> str:
    """
    版 2 のデータセットの骨組みの, 実験の名前の元を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    raw: str
        正解の種類と受け持ち (mask_predict など)
    """
    # 目的の列とタスクの代わり
    raw = (f"{active_value(problem, f'{PART}.target.kind')}_"
           f"{active_value(problem, f'{PART}.role')}")

    return raw


def dataset_place(info: Optional[dict], copied: Optional[Path]) -> str:
    """
    config のデータセットの場所を決める関数

    Parameters
    ----------
    info: Optional[dict]
        入れたデータセットの説明 (版 2 のデータセットでなければ None)
    copied: Optional[Path]
        プロジェクトに写すデータの場所 (プロジェクトからの相対)

    Returns
    ----------
    place: str
        入れた版 2 のデータセットなら写した場所, 他は outputs/dataset
    """
    # 版 2 のデータセットは書き出し済み
    place = copied.as_posix() if info is not None and copied else (
        "outputs/dataset"
    )

    return place
