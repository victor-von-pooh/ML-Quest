import json
from string import Template
from typing import Callable, Optional

from ml_quest.handoff import describe
from ml_quest.model_names import ITEM_INPUT, SCORE_OUTPUT, USER_INPUT
from ml_quest.problem import active_value, get_value
from ml_quest.questions import PART
from ml_quest.rank_width import RANK_TEST, RANK_VALID

# 学習のスクリプトが読み込むライブラリの, ML-Found の登録簿の機能の名前
# (反応の表は ML-EDA の読み方で読む. config の model.spec の定義が無い
# うちは ml_modelcontrol と ml_inference を読まない)
RANK_LIBRARIES = ["eda", "modelcontrol", "inference"]

# 書き出していないときの, 反応の表の場所 (プロジェクトのディレクトリから)
RANK_DATASET = "outputs/interactions.csv"

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
RANK_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

利用者と品物の反応の表で, 2 塔のモデル (利用者と品物を別の塔で埋め込みに
し, 内積を点数にする) を学び, 利用者ごとに上位 K の品物を並べて Recall@K
と NDCG で確かめる骨組みなので, 印 (TODO) の所を書き足す
"""
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from ml_eda import load_data
from ml_found import (
    Experiment, Trainer, load_project_config, ndcg_at_k, recall_at_k,
    save_evaluation, start_experiment, write_task
)

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 並べるもの
PROBLEM_NAME = $name
PREDICT = $predict

# task.yaml の受け持ち, 出力の形と作り方 (問題の設定の答え)
ROLE = "rank"
OUTPUT = "ranking"
METHOD = $method

# 反応の表の読み方 (問題の設定の手元のデータの読み方. ML-EDA の load_data
# に渡す. None は ML-EDA の推定)
LOADER = $loader
OPTIONS = $options

# モデルの入力と点数の出力の名前 (ML-ModelControl の同梱の例
# two_tower_retrieval と同じ)
USER_INPUT = $user_input
ITEM_INPUT = $item_input
SCORE_OUTPUT = $score_output

# 埋め込みの出力の名前 (両方があれば, 点数を利用者と品物の埋め込みの内積
# で求める. 品物の埋め込みは 1 度だけ作る)
USER_EMBEDDING = "user_embedding"
ITEM_EMBEDDING = "item_embedding"

# 利用者と品物の番号の表 (実験のディレクトリの中. 重みの行と列の順)
INDEX_FILE = "rank_index.json"

# 検証用とテスト用の評価の名前 (evaluations/<名前>_<分割>.json), 説明,
# 指標の向き
EVALUATION = "ranking"
EVALUATION_TEXT = (
    "{split}の反応で数えた, 利用者ごとの上位 {k} 件の Recall@K と NDCG "
    "({weights}. 前の分割で反応した品物は並べない. 前の分割で反応して"
    "いない品物に{split}で反応した, 学習用にいる利用者だけを数える). "
    "_baseline は学習用の反応の多い順 (人気順) に並べた値, _diff は"
    "モデル − 人気順で, 大きいほど良い"
)
EVALUATION_BETTER = {"recall_at_k": "higher", "ndcg": "higher"}

# 比べる相手の名前
BASELINE_NAME = "人気順 (学習用の反応の多い順)"
'''

# 2 塔のモデルの部品, 反応の表の分け方, Trainer, 指標, 学習の流れ
RANK_MAIN = '''

class PlaceholderModel(nn.Module):
    def __init__(self, users: int, items: int, width: int):
        # 親クラスのコンストラクタを呼び出す
        super(PlaceholderModel, self).__init__()

        # 利用者と品物の数と, 埋め込みの幅を保存
        self.users = users
        self.items = items
        self.width = width

        # TODO: 課題に合うモデル (利用者と品物の他の列の特徴を足すなど) に
        # 置き換える. 今は動くことを確かめるための, 番号のワンホットから
        # 埋め込みを作る全結合 2 段の塔 2 つ
        self.user_fc1 = nn.Linear(self.users, self.width)
        self.user_fc2 = nn.Linear(self.width, self.width)
        self.item_fc1 = nn.Linear(self.items, self.width)
        self.item_fc2 = nn.Linear(self.width, self.width)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> dict:
        """
        順伝播を行う関数

        Parameters
        ----------
        user: torch.Tensor
            (利用者の件数, 利用者の数) の番号のワンホット
        item: torch.Tensor
            (品物の件数, 品物の数) の番号のワンホット

        Returns
        ----------
        outputs: dict
            score ((利用者の件数, 品物の件数) の点数. 正規化した内積の 20
            倍), user_embedding, item_embedding
        """
        # 塔ごとの埋め込みを L2 正規化し, 全ての組の内積を温度 0.05 で割る
        u = self.user_fc2(self.relu(self.user_fc1(user)))
        v = self.item_fc2(self.relu(self.item_fc1(item)))
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        outputs = {SCORE_OUTPUT: u @ v.T * 20.0, USER_EMBEDDING: u,
                   ITEM_EMBEDDING: v}

        return outputs


class RankTrainer(Trainer):
    def __init__(
        self, model: nn.Module, experiment: Experiment, sizes: tuple,
        valid: dict, cfg: Any
    ):
        # 親クラスのコンストラクタを呼び出す (step を上書きするので
        # step_fn は None)
        super(RankTrainer, self).__init__(model, experiment, None)

        # 利用者と品物の数, 検証用の並べる相手 (利用者ごとの正解と除く
        # 品物), 上位の件数, 点数を数える利用者のまとまりの大きさを保存
        self.sizes = sizes
        self.valid = valid
        self.k = cfg.rank.k
        self.batch_size = cfg.data.batch_size

        # 組ごとの交差エントロピー (バッチの中の他の品物が負例)
        self.criterion = nn.CrossEntropyLoss()

    def step(self, batch: tuple, phase: str) -> dict:
        """
        1 バッチ分の, 対になった利用者と品物の損失を求める関数

        Parameters
        ----------
        batch: tuple
            (利用者の番号, 品物の番号)
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        outputs: dict
            点数の表の対角を正解とする交差エントロピー (loss)
        """
        # 学習は学習しているモデル, 検証は評価のモデル (EMA があれば EMA)
        users, items = batch
        model = self.model if phase == "train" else self.eval_model()
        score = pair_scores(model, users, items, self.sizes)
        target = torch.arange(len(users), device=score.device)
        outputs = {"loss": self.criterion(score, target)}

        return outputs

    def on_epoch_end(self, phase: str) -> dict:
        """
        検証のときに, 利用者ごとの上位 K 件の Recall@K と NDCG を数える関数

        Parameters
        ----------
        phase: str
            train, valid, train_eval のどれか

        Returns
        ----------
        metrics: dict
            検証なら recall_at_k と ndcg (valid/recall_at_k, valid/ndcg
            として記録. 大きいほど良い). 他は空
        """
        # 検証用の正解のある利用者の点数から並べる
        if phase != "valid":
            return {}
        top = top_items(self.eval_model(), self.valid, self.sizes, self.k,
                        self.batch_size)
        metrics = rank_metrics(top, self.valid, self.k)

        return metrics


def one_hot(index: torch.Tensor, size: int) -> torch.Tensor:
    """
    番号をワンホットにする関数

    Parameters
    ----------
    index: torch.Tensor
        (件数,) の番号
    size: int
        番号の数

    Returns
    ----------
    x: torch.Tensor
        (件数, 番号の数) の float32 (番号の所だけ 1)
    """
    # 番号と同じデバイス
    x = torch.zeros(len(index), size, device=index.device)
    x[torch.arange(len(index), device=index.device), index] = 1.0

    return x


def pair_scores(
    model: nn.Module, users: torch.Tensor, items: torch.Tensor, sizes: tuple
) -> torch.Tensor:
    """
    利用者と品物の番号を, 入力の名前でモデルに渡して点数の表を返す関数

    Parameters
    ----------
    model: nn.Module
        2 塔のモデル
    users: torch.Tensor
        (利用者の件数,) の利用者の番号
    items: torch.Tensor
        (品物の件数,) の品物の番号
    sizes: tuple
        (利用者の数, 品物の数)

    Returns
    ----------
    score: torch.Tensor
        (利用者の件数, 品物の件数) の点数
    """
    # ML-ModelControl の定義の出力が複数なら出力の名前の辞書
    outputs = tower_outputs(model, users, items, sizes)
    score = outputs[SCORE_OUTPUT] if isinstance(outputs, dict) else outputs

    return score


def tower_outputs(
    model: nn.Module, users: torch.Tensor, items: torch.Tensor, sizes: tuple
) -> Any:
    """
    利用者と品物の番号を, 入力の名前でモデルに渡して出力を返す関数

    Parameters
    ----------
    model: nn.Module
        2 塔のモデル
    users: torch.Tensor
        (利用者の件数,) の利用者の番号
    items: torch.Tensor
        (品物の件数,) の品物の番号
    sizes: tuple
        (利用者の数, 品物の数)

    Returns
    ----------
    outputs: Any
        出力 (複数なら出力の名前の辞書, 1 つならテンソル)
    """
    # 番号のワンホット
    outputs = model(**{USER_INPUT: one_hot(users, sizes[0]),
                       ITEM_INPUT: one_hot(items, sizes[1])})

    return outputs


def item_embeddings(
    model: nn.Module, sizes: tuple, batch_size: int, device: Any
) -> Optional[torch.Tensor]:
    """
    全ての品物の埋め込みを, 品物のまとまりごとに 1 度だけ求める関数

    Parameters
    ----------
    model: nn.Module
        2 塔のモデル
    sizes: tuple
        (利用者の数, 品物の数)
    batch_size: int
        1 回に渡す品物の数
    device: Any
        モデルのデバイス

    Returns
    ----------
    embeddings: Optional[torch.Tensor]
        (品物の数, 埋め込みの幅). 出力に user_embedding と item_embedding
        の無い定義 (2 塔でない) なら None
    """
    # 利用者は 1 人だけ渡し, 品物の塔の出力を集める
    user = torch.zeros(1, dtype=torch.int64, device=device)
    probe = tower_outputs(model, user, user, sizes)
    if not (isinstance(probe, dict) and USER_EMBEDDING in probe
            and ITEM_EMBEDDING in probe):
        return None
    parts = []
    for start in range(0, sizes[1], batch_size):
        items = torch.arange(start, min(start + batch_size, sizes[1]),
                             device=device)
        parts.append(tower_outputs(model, user, items, sizes)[ITEM_EMBEDDING])
    embeddings = torch.cat(parts)

    return embeddings


def chunk_scores(
    model: nn.Module, users: torch.Tensor, sizes: tuple, batch_size: int,
    items: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    利用者のまとまりの, 全ての品物の点数を求める関数

    Parameters
    ----------
    model: nn.Module
        2 塔のモデル
    users: torch.Tensor
        (まとまりの利用者の数,) の利用者の番号
    sizes: tuple
        (利用者の数, 品物の数)
    batch_size: int
        2 塔でない定義で, 1 回に渡す品物の数
    items: Optional[torch.Tensor]
        品物の埋め込み (item_embeddings. 2 塔でなければ None)

    Returns
    ----------
    scores: torch.Tensor
        (まとまりの利用者の数, 品物の数) の点数
    """
    # 2 塔なら利用者の埋め込みと品物の埋め込みの内積 (点数と同じ順)
    if items is not None:
        one = torch.zeros(1, dtype=torch.int64, device=users.device)
        user = tower_outputs(model, users, one, sizes)[USER_EMBEDDING]
        scores = user @ items.T
        return scores

    # 2 塔でなければ, 品物をまとまりに分けて点数の出力をつなぐ
    parts = []
    for start in range(0, sizes[1], batch_size):
        chunk = torch.arange(start, min(start + batch_size, sizes[1]),
                             device=users.device)
        outputs = tower_outputs(model, users, chunk, sizes)
        parts.append(outputs[SCORE_OUTPUT] if isinstance(outputs, dict)
                     else outputs)
    scores = torch.cat(parts, dim=1)

    return scores


def top_items(
    model: nn.Module, target: dict, sizes: tuple, k: int, batch_size: int
) -> list[list[int]]:
    """
    利用者のまとまりごとに点数を求め, 前の分割で反応した品物を除いて上位
    K 件の品物を選ぶ関数

    Parameters
    ----------
    model: nn.Module
        2 塔のモデル
    target: dict
        rank_target の users と seen
    sizes: tuple
        (利用者の数, 品物の数)
    k: int
        上位の件数
    batch_size: int
        1 回に点数を求める利用者 (と, 2 塔でない定義の品物) の数

    Returns
    ----------
    top: list[list[int]]
        利用者ごとの上位の品物の番号 (target の users の順. 全ての利用者
        と品物の点数の表は持たない)
    """
    # 品物の埋め込みは 1 度だけ. 利用者のまとまりの点数は選んだら捨てる
    device = next(model.parameters()).device
    users = target["users"]
    top: list[list[int]] = []
    model.eval()
    with torch.no_grad():
        items = item_embeddings(model, sizes, batch_size, device)
        for start in range(0, len(users), batch_size):
            chunk = users[start:start + batch_size]
            scores = chunk_scores(model, torch.tensor(chunk, device=device),
                                  sizes, batch_size, items)
            for row, user in enumerate(chunk):
                seen = list(target["seen"].get(user, ()))
                if seen:
                    scores[row, seen] = -math.inf
            picked = torch.topk(scores, min(k, sizes[1]), dim=1).indices
            top += picked.cpu().tolist()

    return top


def rank_metrics(top: list[list[int]], target: dict, k: int) -> dict:
    """
    利用者ごとの上位 K 件の品物から, Recall@K と NDCG を数える関数

    Parameters
    ----------
    top: list[list[int]]
        利用者ごとの上位の品物の番号 (target の users の順)
    target: dict
        users (利用者の番号), relevant (利用者ごとの正解の品物の番号の
        集合)
    k: int
        上位の件数

    Returns
    ----------
    metrics: dict
        recall_at_k (上位 K 件に入った正解の数を正解の全数で割った値の,
        利用者の平均) と ndcg (上位 K 件の NDCG の, 利用者の平均). 数え方は
        ML-Found の recall_at_k と ndcg_at_k (正解の無い利用者は数えず,
        数える利用者がいなければ 0)
    """
    # 利用者ごとの正解の品物 (番号の小さい順の並び) と, ML-Found の数え方
    relevant = [sorted(target["relevant"][user]) for user in target["users"]]
    metrics = {"recall_at_k": recall_at_k(top, relevant, k),
               "ndcg": ndcg_at_k(top, relevant, k)}

    return metrics


def popular_top(
    train: pd.DataFrame, target: dict, items: int, k: int
) -> list[list[int]]:
    """
    学習用の反応の多い順 (人気順) に, 利用者ごとの上位 K 件を選ぶ関数

    Parameters
    ----------
    train: pd.DataFrame
        学習用の反応 (user と item の番号の列)
    target: dict
        rank_target の users と seen
    items: int
        品物の数
    k: int
        上位の件数

    Returns
    ----------
    top: list[list[int]]
        利用者ごとの上位の品物の番号 (target の users の順. 前の分割で
        反応した品物は除く)
    """
    # 全ての利用者で同じ 1 つの並び (同じ数なら番号の小さい品物が先) を
    # 前から見て, 除く品物を飛ばす
    counts = train["item"].value_counts().reindex(range(items), fill_value=0)
    ranked = pd.DataFrame({"item": range(items),
                           "count": counts.to_numpy()})
    order = ranked.sort_values(["count", "item"], ascending=[False, True],
                               kind="stable")["item"].tolist()
    top: list[list[int]] = []
    for user in target["users"]:
        seen = target["seen"].get(user, set())
        picked: list[int] = []
        for item in order:
            if len(picked) == k:
                break
            if item not in seen:
                picked.append(item)
        top.append(picked)

    return top


def read_reactions(path: Path, cfg: Any) -> tuple[pd.DataFrame, list[str]]:
    """
    反応の表を読み, 利用者, 品物, 時刻の列にそろえる関数

    Parameters
    ----------
    path: Path
        反応の表 (1 行が 1 つの反応)
    cfg: Any
        config (rank の列の名前)

    Returns
    ----------
    reactions: pd.DataFrame
        user, item, time の列 (欠けた行と, 反応の強さが 0 以下か数でない
        行は除く. 時刻の順)
    notes: list[str]
        除いた行の知らせ

    Raises
    ----------
    ValueError
        列が無い場合と, 時刻が数でも日時でもない場合
    """
    # 問題の設定の読み方で読み, 答えた列があるか
    frame = load_data(path, loader=LOADER, **OPTIONS).frame
    names = {"user": cfg.rank.user, "item": cfg.rank.item,
             "time": cfg.rank.time}
    if cfg.rank.reaction:
        names["reaction"] = cfg.rank.reaction
    missing = [name for name in names.values() if name not in frame.columns]
    if missing:
        raise ValueError(f"反応の表に {', '.join(map(str, missing))} の列が"
                         "ありません. config の rank の列の名前を直して"
                         "ください")
    reactions = frame[list(names.values())].copy()
    reactions.columns = list(names)

    # 時刻は数か日時 (日時は数にして順に並べる)
    time = reactions["time"]
    if not pd.api.types.is_numeric_dtype(time):
        parsed = pd.to_datetime(time, errors="coerce")
        if parsed.notna().sum() == 0 and time.notna().any():
            raise ValueError(f"時刻の列 {cfg.rank.time} を数か日時として"
                             "読めません")
        reactions["time"] = parsed

    # 欠けた行と, 反応の強さが 0 以下の行を除く
    notes = []
    before = len(reactions)
    reactions = reactions.dropna(subset=["user", "item", "time"])
    if len(reactions) < before:
        notes.append(f"利用者, 品物か時刻の欠けた {before - len(reactions)} "
                     "行を除きました")
    if "reaction" in reactions:
        strength = pd.to_numeric(reactions["reaction"], errors="coerce")
        kept = strength > 0
        if (~kept).any():
            notes.append(f"反応の強さ ({cfg.rank.reaction}) が 0 以下か数で"
                         f"ない {int((~kept).sum())} 行を除きました")
        reactions = reactions[kept]
    reactions = reactions.sort_values("time", kind="stable")
    reactions = reactions[["user", "item", "time"]].reset_index(drop=True)

    return reactions, notes


def split_by_time(reactions: pd.DataFrame, cfg: Any) -> dict:
    """
    反応を時刻の順に学習用, 検証用, テスト用に分ける関数

    Parameters
    ----------
    reactions: pd.DataFrame
        時刻の順の反応 (user, item, time)
    cfg: Any
        config (data.valid と data.test の割合)

    Returns
    ----------
    splits: dict
        train, valid と, テスト用の割合が 0 より大きければ test の表.
        行の数の割合で切る時刻を決め, 同じ時刻の行は同じ分割に入れる

    Raises
    ----------
    ValueError
        割合の誤りと, 学習用か検証用が空になる場合
    """
    # 割合の確かめ
    valid, test = cfg.data.valid, cfg.data.test
    if not (0 < valid < 1 and 0 <= test < 1 and valid + test < 1):
        raise ValueError("config の data.valid は 0 より大きく, data.test "
                         "は 0 以上で, 2 つの和は 1 より小さくしてください")

    # 切る時刻 (その時刻から先が次の分割)
    times = reactions["time"]
    count = len(reactions)
    valid_from = times.iloc[min(int(count * (1 - valid - test)), count - 1)]
    splits = {"train": reactions[times < valid_from]}
    if test > 0:
        test_from = times.iloc[min(int(count * (1 - test)), count - 1)]
        splits["valid"] = reactions[(times >= valid_from)
                                    & (times < test_from)]
        splits["test"] = reactions[times >= test_from]
    else:
        splits["valid"] = reactions[times >= valid_from]
    if splits["train"].empty or splits["valid"].empty:
        raise ValueError("時刻で分けると学習用か検証用の反応が空になります "
                         "(同じ時刻の反応が多いか, 反応が少ない). config の "
                         "data.valid と data.test を見直してください")
    splits = {name: frame.reset_index(drop=True)
              for name, frame in splits.items()}

    return splits


def make_index(train: pd.DataFrame) -> dict:
    """
    学習用の反応から, 利用者と品物の番号を決める関数

    Parameters
    ----------
    train: pd.DataFrame
        学習用の反応 (user, item)

    Returns
    ----------
    index: dict
        users と items (番号の順の元の値. 学習用に初めて出た順)
    """
    # 学習用にいない利用者と品物は番号を持たない
    index = {"users": list(pd.unique(train["user"])),
             "items": list(pd.unique(train["item"]))}

    return index


def to_numbers(frame: pd.DataFrame, index: dict) -> pd.DataFrame:
    """
    反応の利用者と品物を番号にする関数

    Parameters
    ----------
    frame: pd.DataFrame
        反応 (user, item)
    index: dict
        make_index の番号

    Returns
    ----------
    numbered: pd.DataFrame
        user と item の番号の列 (学習用にいない利用者か品物の行は除く)
    """
    # 番号の無い値は欠損にして除く
    users = {value: number for number, value in enumerate(index["users"])}
    items = {value: number for number, value in enumerate(index["items"])}
    numbered = pd.DataFrame({"user": frame["user"].map(users),
                             "item": frame["item"].map(items)}).dropna()
    numbered = numbered.astype("int64").reset_index(drop=True)

    return numbered


def rank_target(frame: pd.DataFrame, before: list[pd.DataFrame]) -> dict:
    """
    分割の反応から, 利用者ごとの正解と並べない品物を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        数える分割の反応 (番号)
    before: list[pd.DataFrame]
        前の分割の反応 (番号. 反応した品物は並べない)

    Returns
    ----------
    target: dict
        users (正解のある利用者の番号), relevant (利用者ごとの, 前の分割で
        反応していない品物の集合), seen (利用者ごとの, 前の分割で反応した
        品物の集合)
    """
    # 前の分割で反応した品物と, その分割で新しく反応した品物
    seen: dict = {}
    for earlier in before:
        for user, item in zip(earlier["user"], earlier["item"]):
            seen.setdefault(int(user), set()).add(int(item))
    relevant: dict = {}
    for user, item in zip(frame["user"], frame["item"]):
        if int(item) not in seen.get(int(user), set()):
            relevant.setdefault(int(user), set()).add(int(item))
    target = {"users": sorted(relevant), "relevant": relevant,
              "seen": seen}

    return target


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


def spec_errors(spec: Optional[Path], sizes: tuple) -> list[str]:
    """
    ML-ModelControl の定義の誤りを, 実験の出力先を作る前に確かめる関数

    Parameters
    ----------
    spec: Optional[Path]
        プロジェクトの定義 (無ければ None)
    sizes: tuple
        (利用者の数, 品物の数)

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

    # 入力は利用者と品物のワンホット, 出力に点数
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    inputs = loaded.get("inputs") or {}
    if sorted(inputs) != sorted([USER_INPUT, ITEM_INPUT]):
        errors.append(f"定義の入力は {USER_INPUT} と {ITEM_INPUT} の 2 つに"
                      f"してください (今は {list(inputs)})")
    if SCORE_OUTPUT not in (loaded.get("outputs") or {}):
        errors.append(f"定義の出力に, 利用者と品物の点数の表 {SCORE_OUTPUT} "
                      "を入れてください")
    for name, size, title in ((USER_INPUT, sizes[0], "利用者"),
                              (ITEM_INPUT, sizes[1], "品物")):
        entry = inputs.get(name)
        shape = entry.get("shape") if isinstance(entry, dict) else None
        if isinstance(shape, list) and shape and shape[-1] != size:
            errors.append(f"入力 {name} の最後の大きさを{title}の数 {size} "
                          f"にしてください (今は {shape[-1]})")

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
    sizes: tuple, cfg: Any, spec: Optional[Path], base_dir: Optional[Path]
) -> nn.Module:
    """
    モデルを作る関数

    Parameters
    ----------
    sizes: tuple
        (利用者の数, 品物の数)
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
    model = PlaceholderModel(sizes[0], sizes[1], cfg.model.width)

    return model


def pair_loader(
    frame: pd.DataFrame, batch_size: int, shuffle: bool
) -> DataLoader:
    """
    対になった利用者と品物の番号の loader を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        反応 (番号)
    batch_size: int
        バッチの大きさ
    shuffle: bool
        エポックごとに並べ替えるか (学習用だけ)

    Returns
    ----------
    loader: DataLoader
        (利用者の番号, 品物の番号) のバッチ
    """
    # 番号は int64
    pairs = TensorDataset(torch.tensor(frame["user"].to_numpy()),
                          torch.tensor(frame["item"].to_numpy()))
    loader = DataLoader(pairs, batch_size=batch_size, shuffle=shuffle)

    return loader


def evaluate_split(
    trainer: RankTrainer, target: dict, train: pd.DataFrame, split: str,
    cfg: Any, experiment: Experiment
) -> dict:
    """
    学習した後の重みで, 分割の Recall@K と NDCG を人気順と並べて数える関数

    Parameters
    ----------
    trainer: RankTrainer
        学習した後の Trainer
    target: dict
        分割の rank_target
    train: pd.DataFrame
        学習用の反応 (番号. 人気順を決める)
    split: str
        分割の名前 (valid か test)
    cfg: Any
        config (上位の件数と, 利用者のまとまりの大きさ)
    experiment: Experiment
        実験 (train.log に記録する)

    Returns
    ----------
    row: dict
        users (数えた利用者の数), k, recall_at_k, ndcg と, 人気順の
        _baseline と, 差の _diff
    """
    # モデルと人気順で同じ利用者を並べる (前の分割の品物は除く)
    model = trainer.model.cpu().eval()
    top = top_items(model, target, trainer.sizes, cfg.rank.k,
                    cfg.data.batch_size)
    found = rank_metrics(top, target, cfg.rank.k)
    popular = rank_metrics(
        popular_top(train, target, trainer.sizes[1], cfg.rank.k), target,
        cfg.rank.k
    )
    row = {"users": len(target["users"]), "k": cfg.rank.k}
    for name in ("recall_at_k", "ndcg"):
        row[name] = found[name]
        row[f"{name}_baseline"] = popular[name]
        row[f"{name}_diff"] = found[name] - popular[name]
    experiment.logger.info(
        f"{split} の上位 {cfg.rank.k} 件 ({len(target['users'])} 人): "
        f"Recall@K {found['recall_at_k']:.4f} (人気順 "
        f"{popular['recall_at_k']:.4f}), NDCG {found['ndcg']:.4f} (人気順 "
        f"{popular['ndcg']:.4f})"
    )

    return row


def write_index(run_dir: Path, index: dict) -> None:
    """
    利用者と品物の番号の表を実験のディレクトリに書く関数

    Parameters
    ----------
    run_dir: Path
        実験のディレクトリ
    index: dict
        make_index の番号

    Returns
    ----------
    None
    """
    # JSON にできない値 (日時など) は文字に
    text = json.dumps({"users": index["users"], "items": index["items"]},
                      ensure_ascii=False, indent=2, default=str)
    (run_dir / INDEX_FILE).write_text(text + "\\n", encoding="utf-8")


def make_inference(
    cfg: Any, spec: Optional[Path], run_dir: Path, experiment: Experiment
) -> None:
    """
    学習した重みから, 実験の中に ML-Inference の並べるの一式を作る関数

    Parameters
    ----------
    cfg: Any
        config (一式の名前, 上位の件数と反応の表の列)
    spec: Optional[Path]
        ML-ModelControl の定義のファイル (無ければ None)
    run_dir: Path
        実験のディレクトリ (重みと番号の表 INDEX_FILE を書いた後)
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

    # 一式は品物の埋め込みを索引にするので, 埋め込みの出力が要る.
    # ML-Inference も作る前に確かめるが, 骨組みは score だけの定義でも
    # 学べるので, 誤りの警告でなく知らせの 1 行にするため先に見る
    loaded = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    missing = [name for name in (USER_EMBEDDING, ITEM_EMBEDDING)
               if name not in (loaded.get("outputs") or {})]
    if missing:
        experiment.logger.info(
            f"定義の出力に {' と '.join(missing)} が無いので, 推論の一式を"
            "作りません. 一式は品物の埋め込みを索引にし, 利用者の埋め込みとの"
            "内積で並べます"
        )
        return

    # 上位の件数と反応の表の列は config の rank から. 定義か設定が約束に
    # 合わない (点数が内積の定数倍でない, k や列の名前の誤りなど) ときは,
    # ML-Inference の誤りの文をそのまま残し, 学習の結果は残す
    from ml_inference import make_rank_bundle
    columns = {"user": cfg.rank.user, "item": cfg.rank.item,
               "reaction": cfg.rank.reaction}
    try:
        info = make_rank_bundle(
            run_dir / "inference", run_dir, spec=spec, k=cfg.rank.k,
            columns=columns, name=cfg.experiment.name
        )
    except ValueError as error:
        experiment.logger.warning(
            "定義か設定が並べるの一式の約束に合わないので, 推論の一式を"
            f"作れません: {error}"
        )
        return
    rank = info["rank"]
    experiment.logger.info(
        f"推論の一式 (並べる): inference/ (利用者 {rank['users']} 人, 品物 "
        f"{rank['items']} 件の索引. 利用者ごとに上位 {rank['k']} 件を並べる. "
        f"版 {info['format_version']})"
    )
    for note in info["notes"]:
        experiment.logger.warning(f"推論の一式の注意: {note}")


def main() -> None:
    """
    反応の表で 2 塔のモデルを学習し, 検証用とテスト用の Recall@K と NDCG と
    推論の一式を残す関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config と反応の表 (無ければ置き場の案内)
    cfg, diff = load_project_config(str(PROJECT_DIR))
    path = PROJECT_DIR / cfg.data.dataset
    if not path.is_file():
        print(f"{path} がありません. README の「次にすること」の 1 で, "
              "反応の表を置いてください")
        sys.exit(1)
    if not isinstance(cfg.rank.k, int) or isinstance(cfg.rank.k, bool) or (
        cfg.rank.k < 1
    ):
        print(f"config の rank.k は 1 以上の整数にしてください (今は "
              f"{cfg.rank.k})")
        sys.exit(1)

    # 利用者, 品物, 時刻の列にそろえ, 時刻の順に分ける
    try:
        reactions, notes = read_reactions(path, cfg)
        splits = split_by_time(reactions, cfg)
    except ValueError as error:
        print(error)
        sys.exit(1)

    # 番号 (学習用にいる利用者と品物) と, 検証用とテスト用の正解
    index = make_index(splits["train"])
    sizes = (len(index["users"]), len(index["items"]))
    numbered = {name: to_numbers(frame, index)
                for name, frame in splits.items()}
    targets = {"valid": rank_target(numbered["valid"], [numbered["train"]])}
    if "test" in numbered:
        targets["test"] = rank_target(numbered["test"],
                                      [numbered["train"], numbered["valid"]])
    if numbered["valid"].empty or not targets["valid"]["users"]:
        print("検証用の反応に, 学習用にいる利用者が前に反応していない品物へ"
              "反応した行がありません. 最も良いモデルを選ぶために, config の"
              " data.valid を大きくしてください")
        sys.exit(1)

    # モデルの定義の確かめ (誤りなら出力先を作る前に止まる)
    spec = project_spec(cfg)
    errors = spec_errors(spec, sizes)
    if errors:
        print(f"{spec} の定義に誤りがあります. README の「次にすること」の "
              "2 で直してください")
        print("\\n".join(f"- {message}" for message in errors))
        sys.exit(1)

    # 何の実験か (受け持ち, 出力の形, 並べるもの, 作り方は問題の設定から).
    # TODO: 既製のモデルを元にするなら, 元にした実験か重みを task["base"]
    experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    write_task(experiment, {"role": ROLE, "output": OUTPUT,
                            "what": PREDICT, "method": METHOD})

    # 利用者と品物の数, 分割ごとの反応の数
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}利用者の数: {sizes[0]}, 品物の"
        f"数: {sizes[1]}. 反応: "
        + ", ".join(f"{name} {len(frame)} 行"
                    for name, frame in splits.items())
        + f". 上位 {cfg.rank.k} 件で数える"
    )
    for note in notes:
        experiment.logger.info(note)
    for name, frame in splits.items():
        dropped = len(frame) - len(numbered[name])
        if dropped:
            experiment.logger.info(
                f"{name} の, 学習用にいない利用者か品物の {dropped} 行は"
                "数えません"
            )
    train_loader = pair_loader(numbered["train"], cfg.data.batch_size, True)
    valid_loader = pair_loader(numbered["valid"], cfg.data.batch_size, False)

    # モデルの組み立てと学習 (定義がまだ無ければ仮のモデル)
    spec, base_dir = spec_file(spec, experiment)
    if spec is None:
        experiment.logger.info(
            f"モデルの定義 ({cfg.model.spec or '場所なし'}) が無いので, "
            "仮のモデル (PlaceholderModel) で学習します. ML-CreateArchitecture"
            " で組んだ定義をその場所に置くと, 定義のモデルで学習します"
        )
    model = make_model(sizes, cfg, spec, base_dir)
    trainer = RankTrainer(model, experiment, sizes, targets["valid"], cfg)
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の設定の場所
    # origin も残る. モデルの定義は ML-Found が start_experiment で写す) と
    # 利用者と品物の番号の表
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")
    write_index(run_dir, index)

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

    # 検証用とテスト用 (あれば) の Recall@K と NDCG を evaluations/ へ保存
    for split, title in (("valid", "検証用"), ("test", "テスト用")):
        if split not in targets or not targets[split]["users"]:
            experiment.logger.info(
                f"{title}の正解のある利用者が無いので, {title}の Recall@K "
                "と NDCG は数えません"
            )
            continue
        row = evaluate_split(trainer, targets[split], numbered["train"],
                             split, cfg, experiment)
        save_evaluation(
            experiment, f"{EVALUATION}_{split}", [row], split,
            description=EVALUATION_TEXT.format(split=title, k=cfg.rank.k,
                                               weights=weights),
            baseline=BASELINE_NAME, epoch=epoch, better=EVALUATION_BETTER
        )

    # 実験の中に並べるの推論の一式 (ML-ModelControl の定義がある場合だけ.
    # 上で書いた番号の表を一式に写す)
    make_inference(cfg, spec, run_dir, experiment)


if __name__ == "__main__":
    main()
'''

# config の基準 (ML-Found のプロジェクトの形)
RANK_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題の設定から書き出した骨組み)
$requires
experiment:
  name: $experiment
  seed: 0

optimizer:
  name: AdamW
  params:
    lr: 0.01

trainer:
  epochs: 20
  grad_clip_norm: 1.0
  # 問題の設定の「最良を選ぶ指標」(valid/recall_at_k と valid/ndcg は検証用
  # の反応で数えた上位 rank.k 件の値で, 大きいほど良い. valid/loss は検証用の
  # 反応の組の交差エントロピーで, 小さいほど良い)
  monitor: $monitor
  mode: $mode

# プロジェクト固有の設定
data:
  # 反応の表 (1 行が 1 つの反応. プロジェクトのディレクトリから. 問題の
  # 設定の手元のデータの読み方で読む)
  dataset: $dataset
  # 反応の組のバッチの大きさ (バッチの中の他の品物が負例)
  batch_size: 256
  # 時刻の順に分ける, 検証用とテスト用の反応の行の割合 (新しい反応から.
  # 同じ時刻の反応は同じ分割)
  valid: $valid
  test: $test

rank:
  # 反応の表の, 利用者, 品物, 時刻の列と, 反応の強さの列 (0 より大きい行
  # だけを反応とみなす. null なら全ての行)
  user: $user
  item: $item
  time: $time
  reaction: $reaction
  # 利用者ごとに並べる品物の数 (Recall@K と NDCG の K)
  k: $k

model:
  # ML-ModelControl の形のモデルの定義 (ML-CreateArchitecture で組んで, この
  # 場所に保存する. 入力は利用者と品物の番号のワンホットの user と item,
  # 出力に点数の表 score)
  spec: models/model.yaml
  # spec のファイルが無いうちは train.py の仮のモデルで学習することの目印
  # (ML-Dashboard の学習の準備が読む. train.py は読まず, 仮のモデルを使うか
  # は spec のファイルがあるかで決まる)
  placeholder: true
  # 仮のモデルの埋め込みの幅
  width: 32
''')


def rank_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 並べる 2 塔のモデルを学ぶスクリプトを作る関数

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
    # 頭 (反応の表の読み方は問題の設定の手元のデータの読み方) と学習の流れ
    options = get_value(problem, "data.options")
    head = Template(RANK_HEAD).substitute(
        user_input=literal(USER_INPUT),
        item_input=literal(ITEM_INPUT),
        score_output=literal(SCORE_OUTPUT),
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        method=literal(active_value(problem, f"{PART}.method")),
        loader=literal(get_value(problem, "data.loader")),
        options=literal(options if isinstance(options, dict) else {}),
    )
    # train.log の初めの行の並べるもの (空なら省く)
    made = ("並べるもの: {PREDICT}. "
            if active_value(problem, f"{PART}.what") else "")
    script = head + Template(RANK_MAIN).substitute(predict_log=made)

    return script


def rank_config(
    experiment: str, stages: dict, dataset: str, requires: str
) -> str:
    """
    並べるの config の基準を作る関数

    Parameters
    ----------
    experiment: str
        実験の名前
    stages: dict
        受け渡し
    dataset: str
        反応の表の場所 (プロジェクトのディレクトリから)
    requires: str
        一番上の requires の行 (skeleton.requires_text)

    Returns
    ----------
    text: str
        configs/default/train.yaml の中身
    """
    # 監視は受け渡しの値, 列の名前と K は受け渡しの rank
    training = stages["training"]
    rank = stages["rank"]
    text = RANK_CONFIG.substitute(
        experiment=json.dumps(experiment, ensure_ascii=False),
        monitor=training["monitor"], mode=training["mode"],
        dataset=json.dumps(dataset, ensure_ascii=False), requires=requires,
        user=json.dumps(rank["user"], ensure_ascii=False),
        item=json.dumps(rank["item"], ensure_ascii=False),
        time=json.dumps(rank["time"], ensure_ascii=False),
        reaction=json.dumps(rank["reaction"], ensure_ascii=False),
        k=json.dumps(rank["k"]), valid=RANK_VALID, test=RANK_TEST,
    )

    return text


def rank_name(problem: dict) -> str:
    """
    並べるの骨組みの, 実験の名前の元を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    raw: str
        品物の列と受け持ち (item_id_rank など)
    """
    # 目的の列とタスクの代わり
    raw = f"{active_value(problem, f'{PART}.rank.item')}_rank"

    return raw


def rank_readme(
    problem: dict, stages: dict, dataset: str, raw: Optional[str]
) -> str:
    """
    並べるのプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    dataset: str
        反応の表の場所 (プロジェクトのディレクトリから)
    raw: Optional[str]
        写した反応の表の場所 (プロジェクトのディレクトリから. 入れて
        いなければ None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    rank = stages["rank"]
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, "
             "ML-Found のプロジェクトの骨組み (利用者と品物の反応の表で, "
             "2 塔のモデルを学び, 利用者ごとに上位 K の品物を並べる).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1 から 4. データ, モデル, 学習, 推論
    monitor = stages["training"]["monitor"]
    reaction = (f"反応の強さの列 `{rank['reaction']}` が 0 より大きい行だけを"
                "反応とみなす" if rank["reaction"]
                else "全ての行を反応とみなす")
    lines += [
        "", "## 次にすること", "",
        "学習する環境には ML-EDA, ML-Found, ML-ModelControl, ML-Inference を"
        " 入れる (config の `requires` の機能. 足りなければ学習の前に入れ方を"
        "示して止まる. ML-ModelControl と ML-Inference は定義を置くと使う).", "",
        "1. **データ**: 反応の表 (1 行が 1 つの反応) を config の "
        f"`data.dataset` の場所 (`{dataset}`) に置く. 列は利用者 "
        f"`{rank['user']}`, 品物 `{rank['item']}`, 時刻 `{rank['time']}` "
        f"(数か日時) で, {reaction}. 他の列は読まない. 利用者, 品物か時刻の"
        "欠けた行は除く. 読み方は問題の設定の手元のデータの読み方 (ML-EDA "
        "の `load_data`)"
        + (f". 入れた表はこのディレクトリの `{raw}` に写した" if raw
           else ""),
        "2. **モデルの作成** (ML-ModelControl, ML-CreateArchitecture): "
        "ML-CreateArchitecture で組んだ定義を, config の `model.spec` の場所"
        " (`models/model.yaml`) に保存する. 入力は利用者の番号のワンホット "
        "`user` ((件数, 利用者の数) の float32) と品物の番号のワンホット "
        "`item` ((件数, 品物の数)) の 2 つ, 出力に全ての組の点数の表 "
        "`score` ((利用者の件数, 品物の件数)). 利用者と品物の数は学習用の"
        "反応にいる数で, `train.log` の初めの行に出る. ML-ModelControl の"
        "同梱の例 `two_tower_retrieval` の入力の大きさと 1 段目の全結合の "
        "`in_features` を利用者と品物の数に合わせて使える. 定義が無いうちは, "
        "`train.py` の `PlaceholderModel` (印 TODO. 動くことを確かめるだけの"
        "小さなモデル) で学習する (config の `model.placeholder: true` が"
        "その目印)",
        "3. **学習** (ML-Found, ML-Dashboard): `python train.py`. 反応を"
        "時刻の順に並べ, 古い反応を学習用, 新しい反応を検証用とテスト用 "
        "(config の `data.valid` と `data.test` の行の割合. 同じ時刻の反応は"
        "同じ分割) に分ける. 番号を持つのは学習用にいる利用者と品物だけで, "
        "他の反応は数えない. 学習は対になった利用者と品物の点数を, バッチの"
        "中の他の品物と比べる交差エントロピー. 検証のたびに, 検証用の反応の"
        f"ある利用者ごとに上位 {rank['k']} 件 (config の `rank.k`) を並べ, "
        "Recall@K (検証用で新しく反応した品物のうち上位 K 件に入った割合) と"
        " NDCG (上位 K 件の順位の良さ) を利用者で平均して数える (前の分割で"
        "反応した品物は並べず, 正解にもしない). 検証用の "
        f"`{monitor}` が最も良いモデルを選ぶ. 結果は "
        "`outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, 学習した"
        "重み `final_model.pt`, 利用者と品物の番号の表 `rank_index.json` "
        "(重みの番号の順の元の値), 検証用とテスト用 (あれば) の "
        "`evaluations/ranking_valid.json` と `ranking_test.json` (数えた"
        "利用者の数, K, `recall_at_k`, `ndcg` と, 学習用の反応の多い順 "
        "(人気順) に並べた `_baseline` と差 `_diff`))",
        "4. **推論** (ML-Inference): `model.spec` の場所に定義を置いて"
        "学習すると, `train.py` が学習の後に実験の中の `inference/` に"
        "並べるの推論の一式 (版 8. 番号の表と品物の埋め込みの索引を持ち, "
        f"上位の件数の既定は `rank.k` の {rank['k']}, 反応の表の列は "
        "config の `rank` の列) を作る. 版 8 の一式を作れる ML-Inference が"
        "要る. 一式を作るには, 定義の出力に利用者の埋め込み "
        "`user_embedding` と品物の埋め込み `item_embedding` も持ち, 点数が"
        "その内積の定数倍であること (同梱の例 `two_tower_retrieval` の形)."
        " その `inference/` に利用者の値の並びを渡すと, 利用者と順位ごとの "
        "`user`, `rank`, `item`, `score` の表が返る (前に反応した品物の表を"
        "渡すと除く). 仮のモデル (`PlaceholderModel`) は定義が無いので一式を"
        "作らない",
        "", "`train.py` の印 (TODO) の所 (モデルなど) は書き足す. 問題の"
        "設定を変えたら, 書き出し直すか手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
