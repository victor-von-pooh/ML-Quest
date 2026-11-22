import json
from pathlib import Path
from string import Template
from typing import Callable, Optional

from ml_quest.baseline_skeleton import (
    predict_log, reported_code, reported_hint, reported_sentence,
    reported_value
)
from ml_quest.data import EXPORT_FUNCTIONS, describe_steps, entry_steps
from ml_quest.handoff import describe
from ml_quest.problem import active_value, get_value
from ml_quest.questions import PART

# 学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の機能の名前
# (ml_eda, ml_modelcontrol, ml_inference. config の model.spec の定義が
# 無いうちは ml_modelcontrol と ml_inference を読まない)
ITEMS_LIBRARIES = ["eda", "modelcontrol", "inference"]

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
ITEMS_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

ML-EDA のファイルの組のデータセット (版 2 か版 3) で学習する骨組みなので,
印 (TODO) の所を書き足す
"""
from pathlib import Path
import shutil
import sys
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml_eda import ItemDataset, ItemSplit, load_items
from ml_found import (
    Experiment, Trainer, describe_origin, load_project_config,
    save_evaluation, save_sample, start_experiment, task_from_dataset,
    write_task
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

# 見本 (入力, 予測, 正解) を書く検証用の件数
SAMPLES = 3
'''

# 画像の領域: 画素ごとの交差エントロピーと, 背景を除いた IoU の平均
MASK_PART = '''
# テスト用の評価の名前, 説明, 指標の向き (evaluations/<名前>.json)
EVALUATION = "test_iou"
EVALUATION_TEXT = ("テスト用の分割のクラスごとの IoU ({weights}). mean は"
                   "背景 (値 0) を除いた平均. 予測にも正解にも無いクラスは 1")
EVALUATION_BETTER = {"iou": "higher"}


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
        # 除く) と, エポックごとの重なり, 和, 正解の画素数を保存
        self.names = names
        self.foreground = foreground
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.counts = torch.zeros(3, len(names))
        self.warned = False

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
        # クラスごとの重なり, 和, 正解の画素数の 0 への戻し
        self.counts = torch.zeros(3, len(self.names))

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
        self.counts += overlap_counts(logits.argmax(dim=1), y,
                                      len(self.names))
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
        # 検証用の分割の正解に背景しか無い場合の警告 (IoU は全て 1 に)
        present = self.counts[2, self.foreground].sum()
        if phase == "valid" and present == 0 and not self.warned:
            self.logger.warning(
                "検証用の分割の正解に背景しかありません. valid/iou では"
                "モデルの良し悪しが分かりません"
            )
            self.warned = True

        # 最良のモデルの選びは検証の段階の値 (valid/iou)
        metrics = {"iou": mean_iou(self.counts, self.foreground)}

        return metrics


def class_table(
    dataset: ItemDataset
) -> tuple[np.ndarray, list, list, list]:
    """
    説明の classes (無ければ values) から, 領域の値の番号の表を作る関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット

    Returns
    ----------
    table: tuple[np.ndarray, list, list, list]
        値から番号の表 (名前の無い値は -1), 番号の順の名前, 背景 (値 0) で
        ない番号, 名前の無い値

    Raises
    ----------
    ValueError
        名前も値の一覧も無い場合と, クラスが 2 つ未満の場合
    """
    # 名前の付いた値 (名前が無ければ値の一覧の全て)
    entry = dataset.entries[entry_names(dataset)[1]]
    values = [int(value) for value in entry.get("values") or []]
    classes = entry.get("classes") or [
        {"value": value, "name": str(value)} for value in values
    ]
    if not classes:
        raise ValueError("領域の値の名前 (classes) も値の一覧 (values) も"
                         "ありません. 書き出しで classes を渡してください")
    names = {int(item["value"]): str(item["name"]) for item in classes}

    # 名前の無い値 0 は背景のクラスとして追加 (除くと正解が全て背景の外に
    # なり, 損失 0, IoU 1 の見かけだけの学習になるため)
    if 0 in values and 0 not in names:
        names[0] = "background"
    if len(names) < 2:
        raise ValueError(f"領域のクラスが 2 つ未満です: {names}")

    # 値の小さい順の番号
    ordered = sorted(names)
    lookup = np.full(max(ordered) + 1, -1, dtype=np.int64)
    lookup[ordered] = np.arange(len(ordered))
    table = (lookup, [names[value] for value in ordered],
             [index for index, value in enumerate(ordered) if value != 0],
             sorted(set(values) - set(names)))

    return table


def overlap_counts(
    predicted: torch.Tensor, target: torch.Tensor, count: int
) -> torch.Tensor:
    """
    クラスごとの重なり, 和, 正解の画素数を数える関数

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
    counts: torch.Tensor
        (3, クラス数) の, 重なり (予測と正解の両方), 和 (どちらか), 正解の
        画素数
    """
    # クラスごとの予測と正解の画素の集合の比べ
    known = target >= 0
    counts = torch.zeros(3, count)
    for index in range(count):
        guess = (predicted == index) & known
        truth = target == index
        counts[0, index] = (guess & truth).sum().item()
        counts[1, index] = (guess | truth).sum().item()
        counts[2, index] = truth.sum().item()

    return counts


def mean_iou(counts: torch.Tensor, foreground: list[int]) -> float:
    """
    背景でないクラスの IoU の平均を求める関数

    Parameters
    ----------
    counts: torch.Tensor
        overlap_counts の (3, クラス数) の画素数
    foreground: list[int]
        背景でないクラスの番号

    Returns
    ----------
    value: float
        IoU の平均 (予測にも正解にも無いクラスは 1. 当てるものが無く, 外し
        てもいないため)
    """
    # 和が 0 のクラス (空と空) は 1
    values = [(counts[0, index] / counts[1, index]).item()
              if counts[1, index] > 0 else 1.0 for index in foreground]
    value = float(np.mean(values)) if values else 1.0

    return value


def make_datasets(dataset: ItemDataset, cfg: object) -> dict:
    """
    load_items の分割を torch の Dataset に包む関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット
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
        ML-EDA のファイルの組のデータセット
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
        ML-EDA のファイルの組のデータセット

    Returns
    ----------
    trainer: Trainer
        MaskTrainer
    """
    # クラスの名前と背景でない番号. 名前の無い値は除くことの警告
    _, names, foreground, unnamed = class_table(dataset)
    if unnamed:
        experiment.logger.warning(
            f"名前の無い領域の値 {unnamed} は, 損失と IoU から除きます"
        )
    trainer = MaskTrainer(model, experiment, names, foreground)

    return trainer


def bundle_items(dataset: ItemDataset, cfg: object) -> dict:
    """
    推論の一式に書く, 学習と同じ 1 件の扱いを返す関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット
    cfg: object
        config (使わない. 点群とそろえるための引数)

    Returns
    ----------
    items: dict
        入力と正解の項目の名前と, クラスの番号ごとの領域の値 (値の番号の
        表の順)
    """
    # 番号の表の, 名前のある値の小さい順 (番号 0 から)
    lookup = class_table(dataset)[0]
    names = entry_names(dataset)
    items = {"input": names[0], "target": names[1], "prepare": "image",
             "values": [int(v) for v in np.flatnonzero(lookup >= 0)]}

    return items


def evaluate_test(
    trainer: Trainer, loader: DataLoader, experiment: Experiment
) -> list[dict]:
    """
    テスト用の分割のクラスごとの IoU を記録する関数

    Parameters
    ----------
    trainer: Trainer
        学習した後の MaskTrainer
    loader: DataLoader
        テスト用の DataLoader
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    rows: list[dict]
        class と iou (予測にも正解にも無いクラスは None) の行と, 最後に
        mean の行
    """
    # 学習と同じ数え方で, 学習した後の予測の重なりと和
    model = trainer.model.cpu().eval()
    counts = torch.zeros(3, len(trainer.names))
    with torch.no_grad():
        for x, y in loader:
            counts += overlap_counts(model(x).argmax(dim=1), y,
                                     len(trainer.names))
    rows = [{"class": name, "iou": (counts[0, index] / counts[1, index])
             .item() if counts[1, index] > 0 else None}
            for index, name in enumerate(trainer.names)]
    rows.append({"class": "mean",
                 "iou": mean_iou(counts, trainer.foreground)})
    experiment.logger.info(f"テスト用の IoU (背景を除く平均): "
                           f"{rows[-1]['iou']:.4f}")$reported

    return rows


def save_samples(
    trainer: Trainer, part: Dataset, dataset: ItemDataset,
    experiment: Experiment
) -> None:
    """
    検証用の数件の入力, 予測, 正解を見本に書く関数

    Parameters
    ----------
    trainer: Trainer
        学習した後の MaskTrainer
    part: Dataset
        検証用の MaskDataset
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット
    experiment: Experiment
        実験 (samples/ に書く)

    Returns
    ----------
    None
    """
    # 予測の番号を領域の値に戻す表と, 値の名前 (データセットと同じ値で
    # 書き, 凡例に名前を出す)
    lookup, names = class_table(dataset)[:2]
    values = np.flatnonzero(lookup >= 0)
    classes = {int(value): name for value, name in zip(values, names)}
    model = trainer.model.cpu().eval()
    for index in range(min(SAMPLES, len(part))):
        # 1 件の画像の予測と, データセットのままの正解
        item = part.items[index]
        image, _ = part[index]
        with torch.no_grad():
            predicted = model(image.unsqueeze(0)).argmax(dim=1)[0]
        target = np.asarray(item[part.names[1]]).astype(np.int64)
        label = f"検証用の {item['id']}"

        # 3 つとも書けるかを先に確かめ, 書けない件は 1 つも書かない (入力
        # だけの半端な見本を残さないように)
        shown = values[predicted.numpy()]
        reason = (sample_problem("image", image.numpy())
                  or sample_problem("mask", shown)
                  or sample_problem("mask", target))
        if reason is not None:
            experiment.logger.warning(
                f"見本 valid_{index} ({label}) を書きません: {reason}"
            )
            continue
        save_sample(experiment, f"valid_{index}_input", image.numpy(),
                    kind="image", split="valid", caption=f"{label} の入力")
        save_sample(experiment, f"valid_{index}_predicted", shown,
                    kind="mask", split="valid",
                    caption=f"{label} の予測", classes=classes)
        save_sample(experiment, f"valid_{index}_target", target,
                    kind="mask", split="valid", caption=f"{label} の正解",
                    classes=classes)

$baseline_comment
'''

# 点群: 2 乗の Chamfer で学び, 検証したエポックだけ Chamfer 距離を記録
CLOUD_PART = '''
# テスト用の評価の名前, 説明, 指標の向き (evaluations/<名前>.json)
EVALUATION = "test_chamfer"
EVALUATION_TEXT = ("テスト用の分割の Chamfer 距離 ({weights}. 単位は"
                   "データセットのまま)")
EVALUATION_BETTER = {"chamfer": "lower"}


class CloudDataset(Dataset):
    def __init__(
        self, items: ItemSplit, names: tuple, counts: tuple, seed: int,
        resample: bool
    ):
        # 親クラスのコンストラクタを呼び出す
        super(CloudDataset, self).__init__()

        # ML-EDA の分割の件の並び, 入力と正解の項目の名前, 選ぶ点の数 (入力
        # と正解), 点を選ぶ乱数の種, 読むたびに選び直すかを保存
        self.items = items
        self.names = names
        self.counts = counts
        self.seed = seed
        self.resample = resample

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
        # 学習用は読むたびに選び直し (torch の乱数の種から. 実験の種で同じ
        # 並び), 検証用とテスト用は件ごとに決まった選び. 撮影 (辞書) は
        # 深度から点に
        item = self.items[index]
        seed = (int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
                if self.resample else self.seed)
        rng = np.random.default_rng([seed, index])
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

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の Chamfer の損失と, 学習以外で Chamfer 距離を求める関数

        Parameters
        ----------
        batch: tuple
            入力の点と正解の点群
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            2 乗の距離の Chamfer (loss), バッチの件数と, 学習以外では
            距離の Chamfer (chamfer. ML-Found が件数の重みで平均)
        """
        # 予測と, 2 乗の距離の Chamfer (学習の損失). 検証の段階は
        # trainer.valid_every_n_epochs ごとと最後のエポックだけ
        x, y = batch
        predicted = self.model(x)
        outputs = {"loss": chamfer(predicted, y, True).mean(),
                   "batch_size": len(y)}
        if phase != "train":
            with torch.no_grad():
                distance = chamfer(predicted, y, False).mean().item()
            outputs["chamfer"] = distance

        return outputs


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
        ML-EDA のファイルの組のデータセット
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
                cfg.data.seed + offset, name == "train"
            )

    return datasets


def placeholder(dataset: ItemDataset, cfg: object) -> nn.Module:
    """
    仮のモデルを作る関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット (使わない. 領域とそろえるための引数)
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
        ML-EDA のファイルの組のデータセット (使わない. 領域とそろえるための引数)

    Returns
    ----------
    trainer: Trainer
        CloudTrainer
    """
    # 点群の Trainer
    trainer = CloudTrainer(model, experiment)

    return trainer


def bundle_items(dataset: ItemDataset, cfg: object) -> dict:
    """
    推論の一式に書く, 学習と同じ 1 件の扱いを返す関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット
    cfg: object
        config (入力と正解の点の数と乱数の種)

    Returns
    ----------
    items: dict
        入力と正解の項目の名前, 扱い方 (撮影は深度から点に), 点の数と,
        検証用の分割と同じ乱数の種 (一式の確かめが学習の記録と合うように)
    """
    # 検証用の分割の種は make_datasets と同じ (分割の順の番号を足す)
    names = entry_names(dataset)
    kind = dataset.entries[names[0]]["kind"]
    items = {
        "input": names[0], "target": names[1],
        "prepare": "capture_points" if kind == "capture" else "points",
        "options": {"points": cfg.data.input_points,
                    "target_points": cfg.data.target_points,
                    "seed": cfg.data.seed + SPLITS.index("valid")},
    }

    return items


def evaluate_test(
    trainer: Trainer, loader: DataLoader, experiment: Experiment
) -> list[dict]:
    """
    テスト用の分割の Chamfer 距離を記録する関数

    Parameters
    ----------
    trainer: Trainer
        学習した後の CloudTrainer
    loader: DataLoader
        テスト用の DataLoader
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    rows: list[dict]
        answer (model) と chamfer の行
    """
    # 学習した後の予測と正解の距離の平均
    model = trainer.model.cpu().eval()
    distances = []
    with torch.no_grad():
        for x, y in loader:
            distances.append(chamfer(model(x), y, False))
    value = torch.cat(distances).mean().item()
    experiment.logger.info(f"テスト用の Chamfer 距離: {value:.5f}")$reported
    rows = [{"answer": "model", "chamfer": value}]

    return rows


def cloud_convention(
    part: Dataset, dataset: ItemDataset, index: int
) -> dict:
    """
    見本の点群の値の決まり (単位と, 正規化で足した量) を返す関数

    Parameters
    ----------
    part: Dataset
        検証用の CloudDataset
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット
    index: int
        分割の中の位置

    Returns
    ----------
    convention: dict
        units と, 正規化した場合は件の shift (ML-EDA の表の値のまま.
        正規化した座標 = 元の座標 + shift)
    """
    # 正規化の単位 (無ければ正解の項目の単位) と, 件の表の shift の列
    normalization = dataset.info.get("normalization") or {}
    units = (normalization.get("units")
             or dataset.entries[part.names[1]].get("units"))
    columns = normalization.get("columns") or []
    row = part.items.rows.iloc[index]
    convention = {"units": units} if units else {}
    if columns and all(column in row.index for column in columns):
        convention["shift"] = [float(row[column]) for column in columns]

    return convention


def save_samples(
    trainer: Trainer, part: Dataset, dataset: ItemDataset,
    experiment: Experiment
) -> None:
    """
    検証用の数件の入力, 予測, 正解の点を見本に書く関数

    Parameters
    ----------
    trainer: Trainer
        学習した後の CloudTrainer
    part: Dataset
        検証用の CloudDataset
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット
    experiment: Experiment
        実験 (samples/ に書く)

    Returns
    ----------
    None
    """
    # 検証と同じ選びの点と予測. 値の決まりで元の座標に戻して見られる
    model = trainer.model.cpu().eval()
    for index in range(min(SAMPLES, len(part))):
        points, target = part[index]
        with torch.no_grad():
            predicted = model(points.unsqueeze(0))[0]
        convention = cloud_convention(part, dataset, index)
        label = f"検証用の {part.items.rows.iloc[index]['id']}"

        # 3 つとも書けるかを先に確かめ, 書けない件は 1 つも書かない
        clouds = (("input", "入力の点", points),
                  ("predicted", "予測", predicted), ("target", "正解", target))
        reason = next((problem for problem in (
            sample_problem("point_cloud", value.numpy())
            for _, _, value in clouds
        ) if problem is not None), None)
        if reason is not None:
            experiment.logger.warning(
                f"見本 valid_{index} ({label}) を書きません: {reason}"
            )
            continue
        for name, text, value in clouds:
            save_sample(experiment, f"valid_{index}_{name}", value.numpy(),
                        kind="point_cloud", split="valid",
                        caption=f"{label} の{text}",
                        convention=convention)

$baseline_comment
'''

# 学習の流れ (データセット, モデル, 学習, task.yaml, 評価)
ITEMS_MAIN = '''

def sample_problem(kind: str, value: np.ndarray) -> Optional[str]:
    """
    見本の値が ML-Found の save_sample で書けるかを確かめる関数

    Parameters
    ----------
    kind: str
        見本の種類 (image, mask, point_cloud)
    value: np.ndarray
        値

    Returns
    ----------
    reason: Optional[str]
        書けない理由 (書けるなら None)
    """
    # 画像は (チャンネル, 高さ, 幅) か (高さ, 幅) の有限の値, 領域は (高さ,
    # 幅) の 0 から 255 の整数, 点群は (点の数, 3) の有限の座標
    array = np.asarray(value)
    reason = None
    if kind == "image" and (array.ndim not in (2, 3)
                            or not np.isfinite(array).all()):
        reason = "画像の形か値が書けません (有限でない値があるなど)"
    elif kind == "mask" and (
        array.ndim != 2 or array.size == 0
        or not np.issubdtype(array.dtype, np.integer)
        or array.min() < 0 or array.max() > 255
    ):
        reason = ("領域の値が 0 から 255 の整数でないので, 見本の PNG に"
                  "書けません")
    elif kind == "point_cloud" and (
        array.ndim != 2 or array.shape[-1] != 3
        or not np.isfinite(array).all()
    ):
        reason = "点群の形が (点の数, 3) でないか, 有限でない座標があります"

    return reason

def entry_names(dataset: ItemDataset) -> tuple[str, str]:
    """
    入力と正解の項目の名前を返す関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット

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


def project_spec(cfg: object) -> Optional[Path]:
    """
    config の model.spec の, プロジェクトの ML-ModelControl の定義を返す関数

    Parameters
    ----------
    cfg: object
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
        誤りの文 (定義が無いか誤りが無ければ空. 外部の重みのファイルが
        読めるかは, 組み立てるときに分かる)
    """
    # 定義の無いうちは ML-ModelControl を読まない. 読めない定義も誤りと
    # して返る (例外にならない)
    if spec is None:
        return []
    from ml_modelcontrol import validate
    errors = [issue["message"] for issue in validate(spec)
              if issue["severity"] == "error"]

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
        ML-Found が実験に写した定義 (学習を始めたときの定義で, source.yaml
        の sha256 と同じ中身). 写せなければプロジェクトの定義. 定義が無け
        れば None
    base_dir: Optional[Path]
        相対の参照の起点 (写しなら元の定義のディレクトリ. 他は None で,
        定義のディレクトリ)
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
    dataset: ItemDataset, cfg: object, spec: Optional[Path],
    base_dir: Optional[Path]
) -> nn.Module:
    """
    モデルを作る関数

    Parameters
    ----------
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット
    cfg: object
        config (仮のモデルの大きさと乱数の種)
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
    model = placeholder(dataset, cfg)

    return model


def make_inference(
    cfg: object, spec: Optional[Path], path: Path, dataset: ItemDataset,
    run_dir: Path, experiment: Experiment
) -> None:
    """
    学習した重みから, 実験の中に ML-Inference の推論の一式を作る関数

    Parameters
    ----------
    cfg: object
        config (一式の名前と 1 件の扱い)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    path: Path
        ML-EDA のファイルの組のデータセットの場所
    dataset: ItemDataset
        ML-EDA のファイルの組のデータセット
    run_dir: Path
        実験のディレクトリ
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

    # 学習と同じ 1 件の扱いで一式を作り, 検証用の分割の評価を記録と比べる
    from ml_inference import make_bundle
    info = make_bundle(
        run_dir / "inference", spec, path, run_dir,
        items=bundle_items(dataset, cfg), name=cfg.experiment.name
    )
    check = info["check"]
    if check and check["recorded"] is None:
        experiment.logger.info(
            f"推論の一式: {check['split']} の {check['metric']} "
            f"{check['value']:,.4f} (学習の記録とは比べない. 監視した指標"
            "か最後の重みが違うため)"
        )
    elif check:
        experiment.logger.info(
            f"推論の一式: {check['split']} の {check['metric']} "
            f"{check['value']:,.4f} (記録 {check['recorded']}, 合う: "
            f"{check['matches']})"
        )
    for note in info["notes"]:
        experiment.logger.warning(f"推論の一式の注意: {note}")


def main() -> None:
    """
    ML-EDA のファイルの組のデータセットで学習し, テスト用の評価, 見本と
    推論の一式を残す関数

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
    spec = project_spec(cfg)
    errors = spec_errors(spec)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\\n".join(f"- {message}" for message in errors))
        sys.exit(1)
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    task = write_task(experiment, task)
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}データセット: "
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

    # モデルの組み立てと学習 (定義がまだ無ければ仮のモデル. 検証は
    # trainer.valid_every_n_epochs ごとと最後のエポック)
    spec, base_dir = spec_file(spec, experiment)
    if spec is None:
        experiment.logger.info(
            f"モデルの定義 ({cfg.model.spec or '場所なし'}) が無いので, "
            "仮のモデル (PlaceholderModel) で学習します. ML-CreateArchitecture"
            " で組んだ定義をその場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(dataset, cfg, spec, base_dir)
    trainer = make_trainer(model, experiment, dataset)
    trainer.fit(loaders["train"], loaders["valid"])

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の設定の場所
    # origin も残る. モデルの定義は ML-Found が start_experiment で写す)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")

    # 結果の表示と, 指標が何との一致か. 学習した後の重みは, 最良のモデル
    # に戻した場合だけ最良のエポック (select_best: false などは最後)
    best = trainer.final_weights == "best"
    epoch = trainer.checkpoint.best_epoch if best else trainer.epoch
    weights = "最良のモデル" if best else "最後のエポックの重み"
    print(f"保存先: {run_dir}")
    print(f"評価する重み: {weights} (エポック {epoch})")
    if best:
        print(f"ベストの {cfg.trainer.monitor}: "
              f"{trainer.checkpoint.best_value}")
    print(describe_origin(task["target"]["origin"]))

    # テスト用の分割があれば, 学習した後の重みの評価を evaluations/ へ保存
    if "test" in loaders:
        rows = evaluate_test(trainer, loaders["test"], experiment)
        save_evaluation(experiment, EVALUATION, rows, "test",
                        description=EVALUATION_TEXT.format(weights=weights),
                        epoch=epoch, better=EVALUATION_BETTER)
    else:
        experiment.logger.info("テスト用の分割が無いので, 評価$no_test")$reason_log

    # 検証用の数件の見本 (ML-Dashboard の見本のタブ). 書けなくても
    # 学習の結果は残すので, 警告だけ
    try:
        save_samples(trainer, datasets["valid"], dataset, experiment)
    except ValueError as error:
        experiment.logger.warning(f"見本を書けません: {error}")

    # 実験の中に推論の一式 (ML-ModelControl の定義がある場合だけ)
    make_inference(cfg, spec, path, dataset, run_dir, experiment)


if __name__ == "__main__":
    main()
'''

# 正解の種類ごとの部分, 出力の形の既定, 書き出しの関数の名前と入力
PARTS = {"mask": MASK_PART, "point_cloud": CLOUD_PART}

# 並べる指標の呼び方
ITEMS_METRIC_NAMES = {"mask": "IoU", "point_cloud": "Chamfer 距離"}

# テスト用の評価で, 答えた今のやり方での指標の値と並べる呼び出し
REPORTED_CALLS = {
    "mask": ("\n\n    # 答えた今のやり方での指標の値と並べる (背景を除く平均)"
             '\n    save_reported(experiment, "iou", rows[-1]["iou"])'),
    "point_cloud": ("\n\n    # 答えた今のやり方での指標の値と並べる"
                    '\n    save_reported(experiment, "chamfer", value)'),
}
EXPORTERS = {
    "mask": ("export_image_masks", "画像のディレクトリ, 領域のディレクトリ",
             ", classes=<値から名前の辞書>"),
    "point_cloud": ("export_capture_clouds",
                    "撮影のディレクトリ, 点群のディレクトリ",
                    ", cloud_units=<点群の単位>"),
}

# config の基準 (ML-Found のプロジェクトの形)
ITEMS_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題の設定から書き出した骨組み)
$requires
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
  # ML-EDA のファイルの組のデータセット (プロジェクトのディレクトリから)
  dataset: $dataset
  batch_size: 8$cloud_data

model:
  # ML-ModelControl の形のモデルの定義 (ML-CreateArchitecture で組んで, この
  # 場所に保存する)
  spec: models/model.yaml
  # spec のファイルが無いうちは train.py の仮のモデルで学習することの目印
  # (ML-Dashboard の学習の準備が読む. train.py は読まず, 仮のモデルを使うか
  # は spec のファイルがあるかで決まる)
  placeholder: true$cloud_model
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
    問題の設定から, ファイルの組のデータセットで学習するスクリプトを
    作る関数

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
    # 比べる相手: 答えた今のやり方での指標の値があれば並べる部分, 無ければ
    # 並べるための手の注記
    # (テスト用の分割が無い場合と, 答えていない場合は train.log にも)
    metric = "iou" if kind == "mask" else "chamfer"
    reported, no_test, reason_log = "", "しません", ""
    if reported_value(problem) is not None:
        baseline = "\n" + reported_code(problem, literal, metric)
        reported = REPORTED_CALLS[kind]
        no_test = "せず, 答えた値とも並べません"
    else:
        reason = ("ファイルの組の受け持ちには今のやり方をデータで書く問いが"
                  "無いので, 今のやり方の値とは並べない. " + reported_hint(
                      problem, f"{metric} ({ITEMS_METRIC_NAMES[kind]})"
                  ))
        baseline = "\n" + comment_lines(
            f"比べる相手 ({active_value(problem, f'{PART}.baseline')}): "
            f"{reason}", indent=""
        ) + ("\n\n# 今のやり方と比べない理由 (train.log に書く)\n"
             f"BASELINE_REASON = {literal(reason)}")
        reason_log = ("\n\n    # 今のやり方と比べない理由\n    experiment."
                      "logger.info(f\"今のやり方とは比べません: "
                      "{BASELINE_REASON}\")")
    body = Template(PARTS[kind]).substitute(baseline_comment=baseline,
                                            reported=reported)
    script = head + body + Template(ITEMS_MAIN).substitute(
        no_test=no_test, reason_log=reason_log,
        predict_log=predict_log(problem)
    )

    return script


def items_config(
    experiment: str, stages: dict, kind: str, dataset: str, requires: str
) -> str:
    """
    ファイルの組のデータセットで学習する config の基準を作る関数

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
    requires: str
        一番上の requires の行 (skeleton.requires_text)

    Returns
    ----------
    text: str
        configs/default/train.yaml の中身
    """
    # 点群だけ検証の間隔と点の数 (打ち切りの待つ回数は検証の回数)
    training = stages["training"]
    cloud = kind == "point_cloud"
    text = ITEMS_CONFIG.substitute(
        experiment=json.dumps(experiment, ensure_ascii=False),
        monitor=training["monitor"],
        mode=training["mode"],
        dataset=json.dumps(dataset, ensure_ascii=False),
        every=CLOUD_EVERY if cloud else "",
        patience_note="\n    # 検証の回数で数えた回数 (5 回 = 10 エポック)"
        if cloud else "",
        patience=5 if cloud else 10,
        cloud_data=CLOUD_DATA if cloud else "",
        cloud_model=CLOUD_MODEL if cloud else "",
        requires=requires,
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
    ファイルの組のデータセットで学習するプロジェクトの README を作る
    関数

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
        入れたデータセットの説明 (ファイルの組のデータセットを入れて
        いなければ None)
    raw: Optional[str]
        写した元のデータの場所 (ファイルの組のデータセットでないデータを
        写した
        場合. プロジェクトのディレクトリから)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約と, 正解の出どころ
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, "
             "ML-Found のプロジェクトの骨組み (ML-EDA のファイルの組のデータセット"
             "で学習する).", "", "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]
    lines += origin_lines(problem, info)

    # 1. データ (入れたデータセットか, 書き出しの案内)
    function, inputs, extra = EXPORTERS[kind]
    process = active_value(problem, f"{PART}.target.process.name")
    origin = (', origin={"type": "process", "name": ..., "settings": '
              '{"note": ...}}' if process else "")
    lines += ["", "## 次にすること", "",
              "学習する環境には ML-EDA, ML-Found (と, 定義を使うなら "
              "ML-ModelControl) を入れる.", ""]
    if info is not None:
        lines.append(f"1. **データ**: 入れた版 {info.get('format_version')} "
                     f"のデータセットを `{dataset}` に写した. 学習はこれを"
                     "読む. 書き出し直したら写し直す")
        lines += [f"    - 書き出しでかけた中身の手順 ({name}): "
                  f"{describe_steps(steps)}. ファイルはかけた後なので学習は"
                  "そのまま読む. 推論でも新しい 1 件に同じ手順をかける"
                  for name, steps in entry_steps(info).items()]
    else:
        lines.append(
            f"1. **データの分析** (コードなら ML-EDA の `{function}`): "
            f"`{function}({inputs}, \"{dataset}\"{extra}{origin})` で, この"
            "ディレクトリの `" + dataset + "` に書き出す. 検証用の分割は必ず"
            "作る" + (f". 入れたデータはこのディレクトリの `{raw}` に写した"
                      if raw else "")
        )
    if info is None:
        lines += export_lines(problem, dataset)
    if process and info is None:
        lines.append("    - 正解の出どころ `origin` に今の処理の名前と設定を"
                     "渡す (ML-EDA の `settings` は辞書. 問題の設定の文は "
                     "`{\"note\": ...}` で). `task.yaml` に写り, 指標が何と"
                     "の一致かが実験に残る")

    # 2 から 4. モデル, 学習, 推論
    every = ("検証 (Chamfer 距離) は `trainer.valid_every_n_epochs` ごとと"
             "最後のエポックだけ. " if kind == "point_cloud" else "")
    lines += [
        "2. **モデルの作成** (ML-ModelControl, ML-CreateArchitecture): "
        "ML-CreateArchitecture で組んだ定義を, config の `model.spec` の場所"
        " (`models/model.yaml`) に保存する. 定義が無いうちは, `train.py` の "
        "`PlaceholderModel` (印 TODO. 動くことを確かめるだけの小さなモデル)"
        " で学習する (config の `model.placeholder: true` がその目印で, "
        "ML-Dashboard の学習の準備は定義が無いことを知らせるだけにする)"
        + (". 入力は (バッチ, チャンネル, 高さ, 幅) の 0 から 1 の画像, 出力は"
           " (バッチ, クラス数, 高さ, 幅) のスコア. クラスは `dataset.yaml` の"
           " `classes` (無ければ `values`). 名前の無い値 0 は背景 "
           "(`background`) のクラスとして足し, 他の名前の無い値は損失と IoU "
           "から除いて `train.log` に警告を出す" if kind == "mask" else
           ". 入力は (バッチ, 点の数, 3) の点, 出力は (バッチ, 点の数, 3) の"
           "点"),
        "3. **学習** (ML-Found, ML-Dashboard): `python train.py`. 結果は "
        "`outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, テスト用の"
        "分割があれば `evaluations/`, 検証用の 3 件の入力と予測と正解の見本 "
        "`samples/`" + (". 領域は値の名前の凡例付き" if kind == "mask" else
                        ". 点群は正規化を戻した元の座標でも見られる")
        + f"). {every}検証用の "
        f"`{stages['training']['monitor']}` が最も良いモデルを選ぶ"
        + ". " + (reported_sentence(
            problem, "iou" if kind == "mask" else "chamfer"
        ) if reported_value(problem) is not None else reported_hint(
            problem, "iou (IoU)" if kind == "mask"
            else "chamfer (Chamfer 距離)"
        )),
        "4. **推論** (ML-Inference, ML-Provide): `model.spec` の場所に定義を"
        "置いて学習すると, `train.py` が学習の後に実験の中の "
        "`inference/` に推論の一式 (出力の形 `" + kind + "`) を作る. 1 件"
        "の扱いは学習と同じもの "
        + ("(選ぶ点の数と検証用の分割と同じ乱数の種)" if kind ==
           "point_cloud" else "(クラスの番号ごとの領域の値)")
        + "を一式に書くので, 一式の確かめ (`train.log` の「推論の一式」) が"
        "学習の記録と合う. ML-Provide でその `inference/` を登録し, 新しい"
        + ("撮影を推論すると, 元の撮影の座標の点群が返る (学習と同じ正規化"
           "を推論でかけて戻す)" if kind == "point_cloud" else
           "画像を推論すると, データセットと同じ領域の値の画像が返る")
        + ". 仮のモデル (`PlaceholderModel`) は定義が無いので一式を作らない",
        "", "`train.py` の印 (TODO) の所 (モデルなど) は書き足す. 問題の"
        "設定を変えたら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text


def export_lines(problem: dict, dataset: str) -> list[str]:
    """
    書き戻した書き出しの設定を, README の案内の行にする関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    dataset: str
        データセットの場所 (プロジェクトのディレクトリから)

    Returns
    ----------
    lines: list[str]
        案内の行 (書き戻した設定が無ければ空)
    """
    # 書き出しの関数と, ディレクトリ, 分割, グループの数
    # 書き出しが今の正解の種類に合う設定だけ (recipe_from と受け渡しと同じ)
    lines: list[str] = []
    saved = get_value(problem, "data.export")
    kind = active_value(problem, f"{PART}.target.kind")
    if not isinstance(saved, dict) or (
        saved.get("function") != EXPORT_FUNCTIONS.get(kind)
    ):
        return lines
    split = saved.get("split") or {}
    groups = saved.get("groups") or {}
    options = saved.get("options")
    steps = options.get("item_steps") if isinstance(options, dict) else None
    lines = [
        "    - データの分析 (ML-Fathom) で書き戻した設定: "
        f"`{saved['function']}({place_text(saved.get('inputs'), '入力')}, "
        f"{place_text(saved.get('targets'), '正解')}, \"{dataset}\", "
        f"split={json.dumps(split, ensure_ascii=False)}"
        + (f", item_steps={json.dumps(steps, ensure_ascii=False)}"
           if steps else "") + ")`"
        + (f" (グループは {len(groups)} 件の対応)" if groups else ""),
        "    - ML-Fathom の「問題の設定から始める」で書き出していれば, "
        f"`{dataset}` に入っている",
    ]

    return lines


def place_text(value: Optional[str], name: str) -> str:
    """
    書き出しの案内の, 入力か正解のディレクトリの書き方を返す関数

    Parameters
    ----------
    value: Optional[str]
        ディレクトリ (選んでいなければ None)
    name: str
        入力か正解

    Returns
    ----------
    text: str
        JSON の文字列か, 選んでいなければ <入力のディレクトリ> の印
    """
    # 選んでいないディレクトリを null と書かない
    text = (json.dumps(value, ensure_ascii=False) if value
            else f"<{name}のディレクトリ>")

    return text


def items_name(problem: dict) -> str:
    """
    ファイルの組のデータセットの骨組みの, 実験の名前の元を返す関数

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
        入れたデータセットの説明 (ファイルの組のデータセットでなければ
        None)
    copied: Optional[Path]
        プロジェクトに写すデータの場所 (プロジェクトからの相対)

    Returns
    ----------
    place: str
        入れたファイルの組のデータセットなら写した場所, 他は
        outputs/dataset
    """
    # ファイルの組のデータセットは書き出し済み
    place = copied.as_posix() if info is not None and copied else (
        "outputs/dataset"
    )

    return place
