import json
import re
from pathlib import Path
from string import Template
from typing import Callable, Optional

from ml_quest.check import choice_label
from ml_quest.handoff import describe
from ml_quest.problem import active_value, get_value
from ml_quest.questions import OUTPUTS, PART
from ml_quest.ready_skeleton import (
    READY_BUNDLE, READY_CONFIG, READY_SETTINGS, yaml_value
)

# 書き出していないときの, 写真のディレクトリ (プロジェクトのディレクトリ
# から)
SEARCH_PHOTOS = "outputs/photos"

# 埋め込みの一式と成果物のディレクトリの既定 (プロジェクトのディレクトリ
# から)
SEARCH_EMBEDDER = "outputs/embedder"
SEARCH_ARTIFACTS = "outputs/search"

# 検証用の問いにする写真の割合の既定
SEARCH_VALID = 0.2

# 索引と評価を作り一式を作るスクリプトの頭 (読み込みと, 問題の設定から
# 決まる既定値). 答えの文字は JSON の文字列で埋め込み
SEARCH_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した, 既製のモデルの埋め込み
で写真の索引を作り, 似た写真を探す推論の一式を作るスクリプト

学習はしない. config の ready の既製のモデルの分類の最後の層の手前の出力で
写真を埋め込み, 索引 (登録した写真) と検証用の問いに分けて, 余弦の近さの
上位 K 件の Recall@K と NDCG@K を数え, ML-Inference の検索の一式を作る
"""
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

# このプロジェクトのディレクトリと, 一式づくりの設定の場所
PROJECT_DIR = Path(__file__).resolve().parent
CONFIG = PROJECT_DIR / $config

# 問題の名前と, 探すもの
PROBLEM_NAME = $name
PREDICT = $predict

# 一式づくりに渡す, 要る名前 (config の ready. sha256 は任意)
READY_KEYS = ("library", "model", "weights")

# ML-Inference と, torchvision の重みを読む ML-ModelControl の入れ方
# (検索の一式は ML-Inference の推論の一式の版 15 から)
INSTALL = (
    "pip install \\"ml-inference @ git+https://github.com/victor-von-pooh/"
    "ML-Inference.git\\" \\"ml-modelcontrol[torchvision] @ git+https://"
    "github.com/victor-von-pooh/ML-ModelControl.git\\""
)
SEARCH_VERSION = 15

# 成果物の名前 (ML-Inference の検索の一式が読む)
INDEX_FILE = "search_index.json"
EMBEDDINGS_FILE = "search_embeddings.npy"
EVALUATIONS_DIR = "evaluations"
EVALUATION = "search_valid"

# 比べる相手の, チャンネルごとに縮めた写真の辺
PIXELS = 8

# 評価の説明, 向き, 比べる相手の名前
EVALUATION_TEXT = (
    "検証用の問いの写真ごとに, 索引の写真を余弦の近さで上位 {k} 件並べ, 問い"
    "と同じ種類の索引の写真を正解にした Recall@K と NDCG@K の, 問いの平均 "
    "(既製のモデル {model} の埋め込み). _baseline は写真を縮めた画素をその"
    "まま並べた値, _diff は埋め込み − そのままで, 大きいほど良い"
)
EVALUATION_BETTER = {"recall_at_k": "higher", "ndcg_at_k": "higher"}
BASELINE_NAME = "写真を縮めた画素をそのまま余弦の近さで並べたもの"
'''

# 設定の読み方, ライブラリ, 埋め込み, 分け方, 評価, 成果物と一式づくり
SEARCH_MAIN = '''

def search_settings(config: dict) -> SimpleNamespace:
    """
    config の seed, data と search から, 分け方と検索の設定を読む関数

    Parameters
    ----------
    config: dict
        一式づくりの設定

    Returns
    ----------
    cfg: SimpleNamespace
        seed, photos, valid, k, max_latency_ms. 誤りなら直す所を示して
        止まる
    """
    # 種は整数, 割合は 0 から 1 の間, 件数は 1 以上の整数, 上限は正の数
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    search = (config.get("search") if isinstance(config.get("search"), dict)
              else {})
    cfg = SimpleNamespace(
        seed=config.get("seed", 0), photos=data.get("photos"),
        valid=data.get("valid", 0.2), k=search.get("k"),
        max_latency_ms=search.get("max_latency_ms"),
    )
    message = None
    if not isinstance(cfg.seed, int) or isinstance(cfg.seed, bool):
        message = f"config の seed は整数にしてください (今は {cfg.seed})"
    elif isinstance(cfg.valid, bool) or not isinstance(
        cfg.valid, (int, float)
    ) or not 0 < cfg.valid < 1:
        message = ("config の data.valid は 0 より大きく 1 より小さくして"
                   f"ください (今は {cfg.valid})")
    elif not isinstance(cfg.k, int) or isinstance(cfg.k, bool) or cfg.k < 1:
        message = ("config の search.k は 1 以上の整数にしてください (今は "
                   f"{cfg.k})")
    elif cfg.max_latency_ms is not None and (
        isinstance(cfg.max_latency_ms, bool)
        or not isinstance(cfg.max_latency_ms, (int, float))
        or not math.isfinite(cfg.max_latency_ms) or cfg.max_latency_ms <= 0
    ):
        message = ("config の search.max_latency_ms はミリ秒の正の数か null"
                   f"にしてください (今は {cfg.max_latency_ms})")
    if message:
        print(message)
        sys.exit(1)

    return cfg


def output_places(config: dict) -> tuple[Path, Path]:
    """
    config の outputs から, 埋め込みの一式と成果物の場所を読む関数

    Parameters
    ----------
    config: dict
        一式づくりの設定

    Returns
    ----------
    places: tuple[Path, Path]
        埋め込みの一式と成果物のディレクトリ (プロジェクトのディレクトリ
        から)
    """
    # 既定は outputs/embedder と outputs/search
    raw = config.get("outputs")
    raw = raw if isinstance(raw, dict) else {}
    places = (PROJECT_DIR / str(raw.get("embedder") or "outputs/embedder"),
              PROJECT_DIR / str(raw.get("search") or "outputs/search"))

    return places


def load_libraries() -> SimpleNamespace:
    """
    ML-Inference の, 埋め込みと検索の一式づくりに使う関数を読み込む関数

    Parameters
    ----------
    None

    Returns
    ----------
    libraries: SimpleNamespace
        make_ready_bundle, predict, read_images, make_search_bundle (無けれ
        ば入れ方を示して止まる)
    """
    # 入っていないか, 検索の一式を作れない古い ML-Inference
    try:
        inference = importlib.import_module("ml_inference")
    except ImportError as error:
        print(f"ML-Inference を読み込めません ({error}). 入れてから動かして"
              f"ください:\\n{INSTALL}")
        sys.exit(1)
    version = getattr(inference, "BUNDLE_FORMAT_VERSION", 0)
    if not isinstance(version, int) or version < SEARCH_VERSION:
        print(f"この ML-Inference は検索の一式 (版 {SEARCH_VERSION}) を作れ"
              f"ません. 新しくしてください:\\n{INSTALL}")
        sys.exit(1)
    ready = importlib.import_module("ml_inference.ready")
    libraries = SimpleNamespace(
        make_ready_bundle=inference.make_ready_bundle,
        predict=inference.predict, read_images=ready.read_images,
        make_search_bundle=inference.make_search_bundle,
    )

    return libraries


def shrink(images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    前処理の後の写真を, チャンネルごとに PIXELS × PIXELS に平均で縮める
    関数

    Parameters
    ----------
    images: np.ndarray
        (件数, 3, crop, crop) の写真
    batch_size: int = 256
        1 回に縮める件数

    Returns
    ----------
    pixels: np.ndarray
        (件数, 3 × PIXELS × PIXELS) の画素の値
    """
    # 平均の縮め方 (割り切れない大きさも同じ数の区画)
    pool = nn.AdaptiveAvgPool2d(PIXELS)
    parts = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size])
            parts.append(pool(batch).flatten(1).numpy())
    pixels = (np.concatenate(parts).astype(np.float64) if parts
              else np.zeros((0, 3 * PIXELS * PIXELS)))

    return pixels


def embed_photos(
    libraries: SimpleNamespace, ready: dict, place: Path, photos: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """
    埋め込みの一式を作り, 写真を埋め込みと縮めた画素にする関数

    Parameters
    ----------
    libraries: SimpleNamespace
        load_libraries の関数
    ready: dict
        config の ready (ready_values の結果)
    place: Path
        埋め込みの一式のディレクトリ (いつも置き換える)
    photos: Path
        写真のディレクトリ

    Returns
    ----------
    found: tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]
        読めた写真の読み込みの表 (path, digest と, 種類のディレクトリなら
        label. 添字は 0 から), (枚数, 埋め込みの幅) の埋め込み, (枚数,
        192) の縮めた画素, 埋め込みの一式の inference.yaml の中身
    """
    # 既製のモデルの埋め込みの一式 (重みの出どころからの読み込み. 初回は
    # 取得)
    try:
        info = libraries.make_ready_bundle(
            place, ready["model"], ready["weights"],
            library=ready["library"], sha256=ready["sha256"],
            name="embedder", overwrite=True, form="vector"
        )
        result = libraries.predict(place, str(photos))
        frame, images, _ = libraries.read_images(
            str(photos), info["ready"]["preprocess"]
        )
    except (FileExistsError, ValueError) as error:
        print(f"写真を埋め込めません: {error}")
        sys.exit(1)
    if list(frame["path"]) != list(result["path"]):
        print("埋め込みと前処理の写真の並びが違います. ML-Inference を同じ版"
              "にそろえてください")
        sys.exit(1)

    # 読めた写真だけの絞り込み (読めない写真の理由の表示)
    ok = (result["status"] == "ok").to_numpy()
    skipped = result.loc[~ok]
    if len(skipped):
        print(f"読めない写真 {len(skipped)} 枚を除きます: " + ", ".join(
            f"{row.path} ({row.reason})"
            for row in skipped.head(5).itertuples()
        ))
    if not ok.any():
        print(f"{photos} に読める写真がありません")
        sys.exit(1)
    points = np.stack(result.loc[ok, "output"].to_list()).astype(np.float64)
    found = (frame.loc[ok].reset_index(drop=True), points,
             shrink(images[ok]), info)

    return found


def split_photos(
    labels: Optional[list], cfg: Any
) -> tuple[np.ndarray, np.ndarray]:
    """
    写真を無作為に索引と検証用の問いに分ける関数

    種類があれば種類ごとに分け, 1 枚しか無い種類は索引に入れる

    Parameters
    ----------
    labels: Optional[list]
        写真ごとの種類 (種類が無ければ None で, 枚数は cfg.count)
    cfg: Any
        search_settings の設定 (valid の割合と乱数の種. count に枚数)

    Returns
    ----------
    index: np.ndarray
        索引の写真の番号 (小さい順)
    queries: np.ndarray
        検証用の問いの写真の番号 (小さい順)
    """
    # 種類ごと (種類が無ければ全体を 1 つ) の並べ替えと, 前の問い
    rng = np.random.default_rng(cfg.seed)
    names = labels if labels is not None else [None] * cfg.count
    queries = []
    for name in dict.fromkeys(names):
        rows = np.flatnonzero(np.array([v == name for v in names]))
        rows = rows[rng.permutation(len(rows))]
        size = min(int(round(len(rows) * cfg.valid)), len(rows) - 1)
        queries.extend(rows[:size].tolist())
    queries_array = np.sort(np.array(queries, dtype=np.int64))
    index = np.setdiff1d(np.arange(len(names)), queries_array)
    if len(queries_array) < 1 or len(index) < 1:
        print(f"読めた写真が少なく ({len(names)} 枚), 索引と検証用の問いが"
              "それぞれ 1 枚以上になりません. 写真を足すか, config の "
              "data.valid を見直してください")
        sys.exit(1)

    return index, queries_array


def unit_rows(points: np.ndarray) -> np.ndarray:
    """
    点を長さで割る関数 (長さが 0 の点は 0 のまま)

    Parameters
    ----------
    points: np.ndarray
        (件数, 幅) の点

    Returns
    ----------
    unit: np.ndarray
        長さ 1 の点
    """
    # 余弦の近さは長さで割った点の内積
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    unit = points / np.where(norm > 0, norm, 1.0)

    return unit


def top_items(
    points: np.ndarray, index: np.ndarray, queries: np.ndarray,
    digests: list, k: int
) -> list[list[int]]:
    """
    問いごとに, 余弦の近さで近い索引の写真を上位 K 件並べる関数

    ML-Inference の検索の一式の並べ方と同じ (同じ近さなら索引の並びの前の
    写真が先. 問いと中身が同じ写真は除く)

    Parameters
    ----------
    points: np.ndarray
        (枚数, 幅) の点
    index: np.ndarray
        索引の写真の番号
    queries: np.ndarray
        問いの写真の番号
    digests: list
        写真ごとのファイルの hash
    k: int
        並べる件数

    Returns
    ----------
    top: list[list[int]]
        問いごとの, 近い順の索引の写真の番号
    """
    # 長さで割った点の内積と, 安定な並べ替え
    unit = unit_rows(points)
    similarity = unit[queries] @ unit[index].T
    top = []
    for row, query in enumerate(queries):
        kept = np.array([digests[i] != digests[query] for i in index])
        places = np.flatnonzero(kept)
        order = np.argsort(-similarity[row, places], kind="stable")[:k]
        top.append([int(index[places[i]]) for i in order])

    return top


def rank_scores(
    top: list[list[int]], relevant: list[set], k: int
) -> dict:
    """
    問いごとの上位 K 件から Recall@K と NDCG@K を数える関数

    ML-Found の recall_at_k と ndcg_at_k と同じ数え方 (正解の無い問いは
    数えない. 数える問いが無ければ 0)

    Parameters
    ----------
    top: list[list[int]]
        問いごとの近い順の写真の番号
    relevant: list[set]
        問いごとの正解の写真の番号
    k: int
        上位から数える件数

    Returns
    ----------
    scores: dict
        queries (数えた問いの数), recall_at_k, ndcg_at_k
    """
    # 正解のある問いごとの値の平均
    recalls, gains = [], []
    for found, wanted in zip(top, relevant):
        if not wanted:
            continue
        places = [r for r, item in enumerate(found[:k], start=1)
                  if item in wanted]
        recalls.append(len(places) / len(wanted))
        dcg = sum(1.0 / math.log2(r + 1) for r in places)
        ideal = sum(1.0 / math.log2(r + 1)
                    for r in range(1, min(k, len(wanted)) + 1))
        gains.append(dcg / ideal)
    scores = {
        "queries": len(recalls),
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(gains)) if gains else 0.0,
    }

    return scores


def evaluate(
    points: np.ndarray, pixels: np.ndarray, index: np.ndarray,
    queries: np.ndarray, frame: pd.DataFrame, k: int
) -> dict:
    """
    埋め込みと縮めた画素の, 検証用の問いの Recall@K と NDCG@K を数える関数

    Parameters
    ----------
    points: np.ndarray
        (枚数, 幅) の埋め込み
    pixels: np.ndarray
        (枚数, 192) の縮めた画素
    index: np.ndarray
        索引の写真の番号
    queries: np.ndarray
        問いの写真の番号
    frame: pd.DataFrame
        読めた写真の読み込みの表 (label と digest)
    k: int
        上位から数える件数

    Returns
    ----------
    row: dict
        k, queries, 指標とその _baseline と _diff
    """
    # 問いと同じ種類の索引の写真 (中身が同じ写真は除く) が正解
    labels = frame["label"].astype(str).tolist()
    digests = frame["digest"].tolist()
    relevant = [
        {int(i) for i in index
         if labels[i] == labels[q] and digests[i] != digests[q]}
        for q in queries
    ]
    model = rank_scores(top_items(points, index, queries, digests, k),
                        relevant, k)
    base = rank_scores(top_items(pixels, index, queries, digests, k),
                       relevant, k)
    row: dict = {"k": int(k), "queries": model["queries"]}
    for name in ("recall_at_k", "ndcg_at_k"):
        row[name] = model[name]
        row[f"{name}_baseline"] = base[name]
        row[f"{name}_diff"] = model[name] - base[name]

    return row


def save_table(
    folder: Path, name: str, rows: list[dict], description: str,
    better: dict, baseline: Optional[str] = None
) -> None:
    """
    評価の表を ML-Found の evaluations の形で書く関数

    Parameters
    ----------
    folder: Path
        成果物の evaluations のディレクトリ
    name: str
        評価の名前
    rows: list[dict]
        表の行
    description: str
        何の評価か
    better: dict
        列ごとの指標の向き (_baseline と _diff にも広げる)
    baseline: Optional[str] = None
        比べる相手の名前

    Returns
    ----------
    None
    """
    # 列は行の鍵の順, 向きは表にある列だけ
    columns = list(dict.fromkeys(key for row in rows for key in row))
    directions = {
        column: direction for key, direction in better.items()
        for column in (key, f"{key}_baseline", f"{key}_diff")
        if column in columns
    }
    evaluation = {
        "name": name, "split": "valid", "kind": "table", "x": None,
        "columns": columns, "rows": rows, "description": description,
        "baseline": baseline, "guard": False, "better": directions,
        "epoch": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(evaluation, ensure_ascii=False, indent=2)
    (folder / f"{name}.json").write_text(text + "\\n", encoding="utf-8")


def write_index(
    place: Path, frame: pd.DataFrame, points: np.ndarray, index: np.ndarray,
    photos: Path, info: dict, cfg: Any, labeled: bool
) -> None:
    """
    索引の説明と埋め込みを成果物に書く関数

    Parameters
    ----------
    place: Path
        成果物のディレクトリ
    frame: pd.DataFrame
        読めた写真の読み込みの表
    points: np.ndarray
        (枚数, 幅) の埋め込み
    index: np.ndarray
        索引の写真の番号
    photos: Path
        写真のディレクトリ
    info: dict
        埋め込みの一式の inference.yaml の中身
    cfg: Any
        search_settings の設定
    labeled: bool
        写真に種類があるか

    Returns
    ----------
    None
    """
    # 埋め込みを作った既製のモデル (版の決まった重みの名前と hash) と,
    # 索引の写真の並び (場所, 種類, ファイルの hash)
    place.mkdir(parents=True, exist_ok=True)
    ready = info["ready"]
    sha256 = info["model"]["pretrained"][0]["sha256"]
    items = [{"path": str(frame["path"].iloc[i]),
              "label": str(frame["label"].iloc[i]) if labeled else None,
              "sha256": str(frame["digest"].iloc[i])} for i in index]
    text = json.dumps({
        "ready": {"library": ready["library"], "model": ready["model"],
                  "weights": ready["weights"], "sha256": sha256},
        "width": int(points.shape[1]), "k": int(cfg.k),
        "max_latency_ms": cfg.max_latency_ms,
        "root": str(photos.resolve()), "items": items,
    }, ensure_ascii=False, indent=2)
    (place / INDEX_FILE).write_text(text + "\\n", encoding="utf-8")
    np.save(place / EMBEDDINGS_FILE, points[index].astype(np.float32))


def build_bundle(
    libraries: SimpleNamespace, settings: tuple, artifacts: Path
) -> dict:
    """
    成果物から, 検索の一式を作る関数

    Parameters
    ----------
    libraries: SimpleNamespace
        load_libraries の関数
    settings: tuple
        bundle_settings の結果 (一式の場所, 名前, 置き換えるか)
    artifacts: Path
        成果物のディレクトリ

    Returns
    ----------
    info: dict
        一式の inference.yaml の中身 (作れなければ理由を示して止まる)
    """
    # 重みの扱い (一式には写さず, 使うときに出どころからの読み込み)
    place, name, overwrite = settings
    try:
        info = libraries.make_search_bundle(place, artifacts, name=name,
                                            overwrite=overwrite)
    except (FileExistsError, ValueError) as error:
        print(f"一式を作れません: {error}")
        sys.exit(1)

    return info


def report(
    info: dict, place: Path, row: Optional[dict], pinned: Optional[str]
) -> None:
    """
    評価と, 作った一式の索引の数, 版, 注意を表示する関数

    Parameters
    ----------
    info: dict
        make_search_bundle が返した inference.yaml の中身
    place: Path
        一式のディレクトリ
    row: Optional[dict]
        評価の行 (種類が無ければ None)
    pinned: Optional[str]
        config に決めておいた hash (無ければ None)

    Returns
    ----------
    None
    """
    # 索引の数と件数, 評価 (hash を決めていなければ書き戻す案内)
    search = info.get("search") or {}
    records = (info.get("model") or {}).get("pretrained") or [{}]
    sha256 = records[0].get("sha256")
    print(f"問題: {PROBLEM_NAME}. 一式を作りました: {place} (版 "
          f"{info.get('format_version')})")
    if PREDICT:
        print(f"探すもの: {PREDICT}")
    print(f"索引の写真: {search.get('items')} 枚. 返す件数: {search.get('k')}"
          f". 待ち時間の上限: {search.get('max_latency_ms')} ms")
    if row is not None:
        print(f"検証用の問い {row['queries']} 枚: Recall@K "
              f"{row['recall_at_k']:.4f} (縮めた画素 "
              f"{row['recall_at_k_baseline']:.4f}), NDCG@K "
              f"{row['ndcg_at_k']:.4f} (縮めた画素 "
              f"{row['ndcg_at_k_baseline']:.4f})")
    else:
        print("写真に種類のディレクトリが無いので評価は無し. 一式で問いの写真"
              "を推論し, 並んだ写真と近さを見て確かめる")
    if sha256:
        print(f"重みの hash (SHA-256): {sha256}")
    if sha256 and not pinned:
        print("この hash を config の ready.sha256 (と問題の設定の重みの "
              "hash) に書くと, 次から違う重みでは作りません")
    for note in info.get("notes") or []:
        print(f"注意: {note}")


def main() -> None:
    """
    config の既製のモデルの埋め込みで索引と評価を作り, 検索の一式を作る
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 設定と写真の場所 (無ければ置き場の案内)
    config = read_config(CONFIG)
    ready = ready_values(config)
    cfg = search_settings(config)
    settings = bundle_settings(config)
    embedder, artifacts = output_places(config)
    photos = PROJECT_DIR / str(cfg.photos or "outputs/photos")
    if not photos.is_dir():
        print(f"{photos} がありません. README の「次にすること」の 1 で, "
              "索引にする写真を置いてください")
        sys.exit(1)

    # 写真の埋め込みと縮めた画素, 索引と問いへの分け方
    libraries = load_libraries()
    frame, points, pixels, info = embed_photos(libraries, ready, embedder,
                                               photos)
    labeled = "label" in frame.columns and frame["label"].notna().all()
    cfg.count = len(frame)
    index, queries = split_photos(
        frame["label"].astype(str).tolist() if labeled else None, cfg
    )
    print(f"写真: 索引 {len(index)} 枚, 検証用の問い {len(queries)} 枚. "
          f"埋め込みの幅: {points.shape[1]}. 返す件数: {cfg.k}")

    # 種類があれば, 埋め込みと縮めた画素の Recall@K と NDCG@K
    row = None
    if labeled:
        row = evaluate(points, pixels, index, queries, frame, cfg.k)
        model = f"{info['ready']['library']} の {info['ready']['model']}"
        save_table(artifacts / EVALUATIONS_DIR, EVALUATION, [row],
                   EVALUATION_TEXT.format(k=cfg.k, model=model),
                   EVALUATION_BETTER, baseline=BASELINE_NAME)
    else:
        stale = artifacts / EVALUATIONS_DIR / f"{EVALUATION}.json"
        if stale.exists():
            stale.unlink()
    write_index(artifacts, frame, points, index, photos, info, cfg, labeled)
    print(f"成果物: {artifacts}")

    # 検索の一式 (問いの写真に近い索引の写真を上位 K 件返す)
    bundle = build_bundle(libraries, settings, artifacts)
    report(bundle, settings[0], row, ready["sha256"])


if __name__ == "__main__":
    main()
'''

# 索引と一式づくりの設定
SEARCH_CONFIG = Template('''# 既製のモデルの埋め込みで写真の索引を作り, 似た写真を探す推論の一式を作る
# 設定 (ML-Quest が問題の設定から書き出した骨組み). bundle.py が読む. 学習は
# しない

# 埋め込みに使う既製のモデル (問題の設定の既製のモデルの答え)
ready:
  # 出どころ (ML-Inference の READY_LIBRARIES の名前)
  library: $library
  # 出どころでのモデルの名前 (画像の分類のモデル. 分類の最後の層の手前の
  # 出力を埋め込みにする)
  model: $model
  # 重みの名前 (版. 名前ごとに重みが決まる). DEFAULT は作るときの出どころの
  # 既定の重みを使い, 成果物と一式にはその版の名前を残す
  weights: $weights
  # 決めておいた重みの hash (SHA-256). 読んだ重みと違えば作らない. null は
  # 比べずに作り, 求めた hash を表示する
  sha256: $sha256

# 索引と問いに分ける乱数の種
seed: 0

data:
  # 写真のディレクトリ (プロジェクトのディレクトリから. <種類>/<写真> の形
  # なら種類を評価の正解に使う)
  photos: $photos
  # 無作為に分ける, 検証用の問いにする写真の割合 (種類ごと. 1 枚しか無い
  # 種類は索引に入れる)
  valid: $valid

search:
  # 問いごとに返す件数 K (Recall@K と NDCG@K もこの K で数える)
  k: $k
  # 1 件の問いの待ち時間の上限 (守る指標. ミリ秒. null は上限無し)
  max_latency_ms: $max_latency_ms

outputs:
  # 埋め込みの一式 (動かすたびに作り直す) と, 成果物 (索引の説明と埋め込み,
  # 評価の表) のディレクトリ (プロジェクトのディレクトリから)
  embedder: $embedder
  search: $search

bundle:
  # 検索の一式を作るディレクトリ (プロジェクトのディレクトリから)
  path: $path
  # 一式の名前 (null はディレクトリの名前)
  name: $name
  # 前に作った一式を置き換えるか
  overwrite: false
''')


def search_message(problem: dict) -> Optional[str]:
    """
    似た写真を探す骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 並べるで, 作り方が既製のモデルをそのまま)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が順位でない, 入れたデータが画像のディレクトリ
        でない, 検索の判断の件数が無い. 書き出せるなら None)
    """
    # 写真のディレクトリを既製のモデルで埋め込み, 近い順に並べる
    output = active_value(problem, f"{PART}.output")
    loader = active_value(problem, "data.loader")
    source = get_value(problem, "data.source")
    decision = active_value(problem, "action.decision")
    k = decision.get("k") if isinstance(decision, dict) else None
    message = None
    if output != "ranking":
        message = ("似た写真を探す骨組みは, 出力の形が「順位」の場合だけです "
                   f"(今は{choice_label(OUTPUTS, output, problem)})")
    elif source and loader != "image_folder":
        message = ("似た写真を探す骨組みは, 写真のディレクトリから索引を作り"
                   f"ます (入れたデータ {Path(str(source)).name} は画像の"
                   "ディレクトリとして読んでいません). 写真のディレクトリを"
                   "入れてください")
    elif not isinstance(decision, dict) or (
        decision.get("type") != "search" or k is None
    ):
        message = ("似た写真を探す骨組みには, 7 の損得と制約で「近い順に上位"
                   " K 件を返す」と返す件数 K を答えてください")

    return message


def search_name(stages: dict, problem: dict, name: Optional[str]) -> str:
    """
    検索の一式の名前を決める関数

    Parameters
    ----------
    stages: dict
        受け渡し (inference.ready に一式づくりに渡す値)
    problem: dict
        問題の設定 (版 2)
    name: Optional[str]
        指定した名前

    Returns
    ----------
    text: str
        文字と数字と _ と - だけの名前 (省略するとモデルの名前と件数から)
    """
    # 区切りや空白などのディレクトリに使いにくい文字の _ への置き換え
    ready = stages["inference"]["ready"]
    k = (stages["inference"].get("decision") or {}).get("k")
    raw = name or f"{ready['model']}_search_{k}"
    text = re.sub(r"[^\w-]+", "_", raw).strip("_") or "search"

    return text


def number_text(value: object) -> str:
    """
    数を config の値の文字にする関数 (整数の小数は整数)

    Parameters
    ----------
    value: object
        数か None

    Returns
    ----------
    text: str
        JSON の数か null
    """
    # 整数の値の小数は整数
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = json.dumps(value)

    return text


def search_config(
    stages: dict, problem: dict, name: Optional[str], photos: str
) -> str:
    """
    似た写真を探す設定を作る関数

    Parameters
    ----------
    stages: dict
        受け渡し (inference.ready と検索の判断)
    problem: dict
        問題の設定 (版 2. 待ち時間の上限)
    name: Optional[str]
        一式の名前 (省略するとモデルの名前と件数から)
    photos: str
        写真のディレクトリ (プロジェクトのディレクトリから)

    Returns
    ----------
    text: str
        configs/default/bundle.yaml の中身
    """
    # 文字は JSON の文字列 (YAML としても読める), 件数は整数
    ready = stages["inference"]["ready"]
    decision = active_value(problem, "action.decision") or {}
    text = SEARCH_CONFIG.substitute(
        **{key: yaml_value(ready[key])
           for key in ("library", "model", "weights", "sha256")},
        photos=yaml_value(photos), valid=SEARCH_VALID,
        k=number_text(decision.get("k")),
        max_latency_ms=number_text(decision.get("max_latency_ms")),
        embedder=yaml_value(SEARCH_EMBEDDER),
        search=yaml_value(SEARCH_ARTIFACTS),
        path=yaml_value(READY_BUNDLE),
        name=yaml_value(search_name(stages, problem, name)),
    )

    return text


def search_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 既製の埋め込みで索引を作り検索の一式を作るスクリプトを
    作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    literal: Callable
        値を Python の式にする関数 (skeleton.literal)

    Returns
    ----------
    script: str
        bundle.py の中身
    """
    # 頭, 設定の読み方 (既製のモデルの骨組みと同じ), 索引と評価と一式
    # づくりの流れ
    head = Template(SEARCH_HEAD).substitute(
        config=literal(READY_CONFIG),
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
    )
    script = head + "\n" + READY_SETTINGS + SEARCH_MAIN

    return script


def search_readme(
    problem: dict, stages: dict, photos: str, raw: Optional[str]
) -> str:
    """
    似た写真を探すプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    photos: str
        写真のディレクトリ (プロジェクトのディレクトリから)
    raw: Optional[str]
        写した写真のディレクトリ (プロジェクトのディレクトリから. 入れて
        いなければ None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    ready = stages["inference"]["ready"]
    decision = active_value(problem, "action.decision") or {}
    k = decision.get("k")
    limit = decision.get("max_latency_ms")
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, 既製の"
             "モデルの埋め込みで似た写真を探す骨組み (学習しない. 索引と評価"
             "を残し, ML-Inference の推論の一式を作る).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1 から 3. 写真, 索引と評価と一式づくり, 推論
    pinned = (f"決めておいた hash (`{ready['sha256'][:12]}…`) と読んだ重み"
              "が違えば作らない" if ready["sha256"] else
              "hash を決めていないので比べずに作り, 求めた hash を表示する. "
              "それを config の `ready.sha256` と問題の設定の重みの hash に"
              "書くと, 次から同じ重みに固定できる")
    guard = (f"待ち時間の上限 {limit} ms (config の `search.max_latency_ms`."
             " 守る指標) を超えた問いの数を知らせる" if limit is not None
             else "待ち時間の上限は決めていない (config の "
             "`search.max_latency_ms` に書くと, 超えた問いの数を知らせる)")
    lines += [
        "", "## 次にすること", "",
        "データの分析, モデルの作成, 学習の段階は通らない. 動かす環境には "
        "ML-Inference (ML-EDA と PyTorch も入る) と, torchvision の重みを"
        "読む ML-ModelControl の追加の依存 (`ml-modelcontrol[torchvision]`)"
        " を入れる.", "",
        "1. **写真**: 索引にする写真のディレクトリを config の `data.photos`"
        f" の場所 (`{photos}`) に置く (ML-EDA の画像の読み込みで読む. 読め"
        "ない写真は除く). `<種類>/<写真>` の形なら, 種類を評価の正解に使う"
        + (f". 入れた写真はこのディレクトリの `{raw}` に写した" if raw
           else ""),
        "2. **索引と評価と一式づくり** (ML-Inference): `python bundle.py`. "
        f"config (`{READY_CONFIG}`) の `ready` の既製のモデル (出どころ "
        f"`{ready['library']}`, モデル `{ready['model']}`, 重み "
        f"`{ready['weights']}`) の埋め込みの一式を `outputs.embedder` に作り"
        " (重みは一式に写さず出どころから読む. 初回は取りに行き, PyTorch の"
        "置き場 `torch.hub.get_dir()` の `checkpoints` に置く. ネットワーク"
        f"を使えない場所では, そこに重みのファイルを置く. {pinned}), 写真を"
        "分類の最後の層の手前の出力で埋め込む. 写真を無作為に索引と検証用の"
        "問い (config の `data.valid` の割合. 種類ごとに分け, 1 枚しか無い"
        "種類は索引に入れる) に分ける. 種類があれば, 問いごとに索引の写真を"
        f"余弦の近さで上位 {k} 件 (config の `search.k`) 並べ, 問いと同じ"
        "種類の写真を正解にした Recall@K と NDCG@K を数え, 写真を縮めた画素を"
        "そのまま並べた値と比べる. 成果物は `outputs.search` (既製のモデル"
        "と索引の写真の並びの `search_index.json`, 索引の埋め込みの "
        "`search_embeddings.npy`, 種類があれば評価の "
        "`evaluations/search_valid.json`) に残る. 続けて `bundle.path` (`"
        + READY_BUNDLE + "`) に ML-Inference の検索の一式 (版 15) を作る",
        "3. **推論** (ML-Inference, ML-Provide): ML-Provide で作った一式を"
        "登録し, 問いの写真のファイルかディレクトリを渡すと, 問いごとに索引"
        f"の写真から見た目の近い順に {k} 件 (場所, 種類, 余弦の近さ) と, 1 "
        f"件の問いの待ち時間が返る. {guard}. 一式を作るときに重みのライセンス"
        "の注意が出たら, 使ってよいかを確かめる",
        "", "返す件数を変えるなら config の `search.k` を変えて動かし直す. 問題"
        "の設定の既製のモデルの答えを変えたら, 書き出し直すか config を手で"
        "直す.", "",
    ]
    text = "\n".join(lines)

    return text
