import json
import re
from pathlib import Path
from string import Template
from typing import Callable, Optional

from ml_quest.check import choice_label
from ml_quest.group_skeleton import GROUP_CLUSTER, GROUP_VALID
from ml_quest.handoff import describe
from ml_quest.problem import active_value, get_value
from ml_quest.questions import GROUP_STABILITIES, OUTPUTS, PART
from ml_quest.ready_skeleton import (
    READY_BUNDLE, READY_CONFIG, READY_SETTINGS, yaml_value
)

# 書き出していないときの, 群に分ける写真のディレクトリ (プロジェクトの
# ディレクトリから)
READY_GROUP_PHOTOS = "outputs/photos"

# 埋め込みの一式と成果物のディレクトリの既定 (プロジェクトのディレクトリ
# から)
READY_GROUP_EMBEDDER = "outputs/embedder"
READY_GROUP_ARTIFACTS = "outputs/groups"

# 群に分けて一式を作るスクリプトの頭 (読み込みと, 問題の設定から決まる
# 既定値). 答えの文字は JSON の文字列で埋め込み
READY_GROUP_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した, 既製のモデルの埋め込み
で写真を群に分け, 推論の一式を作るスクリプト

学習はしない. config の ready の既製のモデルの分類の最後の層の手前の出力で
写真を埋め込み, k-means で群に分けて群の安定性と群の要約を残し,
ML-Inference の既製の埋め込みの群分けの一式を作る
"""
from datetime import datetime, timezone
import importlib
import json
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

# 問題の名前と, まとめるもの
PROBLEM_NAME = $name
PREDICT = $predict

# 一式づくりに渡す, 要る名前 (config の ready. sha256 は任意)
READY_KEYS = ("library", "model", "weights")

# ML-Inference と, torchvision の重みを読む ML-ModelControl の入れ方
# (既製の埋め込みの群分けの一式は ML-Inference の推論の一式の版 10 から)
INSTALL = (
    "pip install \\"ml-inference @ git+https://github.com/victor-von-pooh/"
    "ML-Inference.git\\" \\"ml-modelcontrol[torchvision] @ git+https://"
    "github.com/victor-von-pooh/ML-ModelControl.git\\""
)
GROUP_VERSION = 10

# 群の安定性の数え方の名前
STABILITY_NAMES = {"seed": "種を変えて作り直す",
                   "sample": "標本を変えて作り直す"}

# config の group の, 問題の設定で問わない値の既定
GROUP_DEFAULTS = {"repeats": 5, "fraction": 0.8, "iterations": 100,
                  "inits": 5, "sample": 2000}

# 成果物の名前 (ML-Inference の既製の埋め込みの群分けの一式が読む)
ASSIGNMENT_FILE = "group_assignments.csv"
CENTERS_FILE = "group_centers.json"
EVALUATIONS_DIR = "evaluations"

# 群の要約の列 (画像の読み込みの表の画素の数)
SUMMARY_COLUMNS = ["width", "height"]

# 比べる相手の, チャンネルごとに縮めた写真の辺
PIXELS = 8

# 群の安定性と群の要約の評価の名前 (evaluations/<名前>.json), 説明, 向き
STABILITY = "group_stability"
STABILITY_TEXT = (
    "全ての写真の群と, 学習用の写真で作り直した群 ({method}, {repeats} 回) "
    "の一致 (調整ランド指数 ari の平均と最も低い値 ari_min. 1 は同じ群, 0 "
    "は偶然と同じ) と, 全ての写真のシルエット係数 (群のまとまり. それぞれの"
    "群を作った空間で数える) (既製のモデル {model} の埋め込み). _baseline "
    "は写真を縮めた画素をそのまま k-means で分けた値, _diff は埋め込み − "
    "そのままで, 大きいほど良い"
)
STABILITY_BETTER = {"ari": "higher", "ari_min": "higher",
                    "silhouette": "higher"}
SUMMARY = "group_summary"
SUMMARY_TEXT = (
    "群ごとの写真の数 (rows), 割合 (share), 作り直したときの群の一致 "
    "(stability. 作り直した群のうち最も重なる群との Jaccard 係数の平均) と,"
    " 写真の幅と高さの平均 (既製のモデル {model} の埋め込み. 最後の行は全て"
    "の写真)"
)
SUMMARY_BETTER = {"stability": "higher"}

# 比べる相手の名前
BASELINE_NAME = "写真を縮めた画素をそのまま k-means で分けた群"
'''

# 群分けの設定の読み方, ライブラリ, 埋め込み, 成果物と一式づくりの流れ
READY_GROUP_MAIN = '''

def group_settings(config: dict) -> SimpleNamespace:
    """
    config の seed, data と group から, 分け方と群の設定を読む関数

    Parameters
    ----------
    config: dict
        一式づくりの設定

    Returns
    ----------
    cfg: SimpleNamespace
        experiment.seed, data.photos と data.valid, group (k, stability と
        GROUP_DEFAULTS の値). 誤りなら直す所を示して止まる
    """
    # 問わない値は既定のまま. 群の設定の確かめはまとめるの骨組みと同じ
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    group = (config.get("group") if isinstance(config.get("group"), dict)
             else {})
    seed = config.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        print(f"config の seed は整数にしてください (今は {seed})")
        sys.exit(1)
    valid = data.get("valid", 0.2)
    if isinstance(valid, bool) or not isinstance(valid, (int, float)) or (
        not 0 < valid < 1
    ):
        print("config の data.valid は 0 より大きく 1 より小さくしてください "
              f"(今は {valid})")
        sys.exit(1)
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(seed=seed),
        data=SimpleNamespace(photos=data.get("photos"), valid=valid),
        group=SimpleNamespace(**{**GROUP_DEFAULTS, **group}),
    )
    for key in ("k", "stability"):
        if not hasattr(cfg.group, key):
            setattr(cfg.group, key, None)
    counts = [key for key in ("iterations", "inits", "sample")
              if not isinstance(getattr(cfg.group, key), int)
              or isinstance(getattr(cfg.group, key), bool)
              or getattr(cfg.group, key) < 1]
    message = None
    if counts:
        message = (f"config の group の {', '.join(counts)} は 1 以上の整数"
                   "にしてください")
    elif isinstance(cfg.group.fraction, bool) or not isinstance(
        cfg.group.fraction, (int, float)
    ):
        message = "config の group.fraction は数にしてください"
    message = message or check_config(cfg)
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
    # 既定は outputs/embedder と outputs/groups
    raw = config.get("outputs")
    raw = raw if isinstance(raw, dict) else {}
    places = (PROJECT_DIR / str(raw.get("embedder") or "outputs/embedder"),
              PROJECT_DIR / str(raw.get("groups") or "outputs/groups"))

    return places


def load_libraries() -> SimpleNamespace:
    """
    ML-Inference と ML-EDA の, 群分けに使う関数を読み込む関数

    Parameters
    ----------
    None

    Returns
    ----------
    libraries: SimpleNamespace
        make_ready_bundle, predict, read_images, make_group_bundle, Data,
        describe_groups (無ければ入れ方を示して止まる)
    """
    # 入っていないか, 既製の埋め込みの群分けの一式を作れない古い
    # ML-Inference
    try:
        inference = importlib.import_module("ml_inference")
        eda = importlib.import_module("ml_eda")
    except ImportError as error:
        print(f"ML-Inference を読み込めません ({error}). 入れてから動かして"
              f"ください:\\n{INSTALL}")
        sys.exit(1)
    version = getattr(inference, "BUNDLE_FORMAT_VERSION", 0)
    if not isinstance(version, int) or version < GROUP_VERSION:
        print("この ML-Inference は既製の埋め込みの群分けの一式 (版 "
              f"{GROUP_VERSION}) を作れません. 新しくしてください:\\n"
              f"{INSTALL}")
        sys.exit(1)
    ready = importlib.import_module("ml_inference.ready")
    libraries = SimpleNamespace(
        make_ready_bundle=inference.make_ready_bundle,
        predict=inference.predict, read_images=ready.read_images,
        make_group_bundle=inference.make_group_bundle, Data=eda.Data,
        describe_groups=eda.describe_groups,
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
    # 平均の縮め方 (割り切れない大きさも同じ数の区画に)
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
        群に分ける写真のディレクトリ

    Returns
    ----------
    found: tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]
        読めた写真の読み込みの表 (path, width, height など. 添字は 0
        から), (枚数, 埋め込みの幅) の埋め込み, (枚数, 192) の縮めた画素,
        埋め込みの一式の inference.yaml の中身
    """
    # 既製のモデルの埋め込みの一式 (重みは出どころから読む. 初回は取りに
    # 行く)
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

    # 読めた写真だけ (読めない写真は理由を表示して除く)
    ok = (result["status"] == "ok").to_numpy()
    skipped = result.loc[~ok]
    if len(skipped):
        print(f"読めない写真 {len(skipped)} 枚を除きます: " + ", ".join(
            f"{row.path} ({row.reason})"
            for row in skipped.head(5).itertuples()
        ))
    points = np.stack(result.loc[ok, "output"].to_list()).astype(np.float64)
    found = (frame.loc[ok].reset_index(drop=True), points,
             shrink(images[ok]), info)

    return found


def split_photos(count: int, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    写真を無作為に学習用と検証用に分ける関数

    Parameters
    ----------
    count: int
        読めた写真の数
    cfg: Any
        group_settings の設定 (data.valid の割合, group.k と乱数の種)

    Returns
    ----------
    train: np.ndarray
        学習用の写真の番号 (小さい順)
    valid: np.ndarray
        検証用の写真の番号 (小さい順)
    """
    # 乱数の種で並べ替えて, 前を検証用に
    order = np.random.default_rng(cfg.experiment.seed).permutation(count)
    size = int(round(count * cfg.data.valid))
    valid, train = np.sort(order[:size]), np.sort(order[size:])
    if len(valid) < 2 or len(train) < cfg.group.k:
        print(f"読めた写真が少なく ({count} 枚), 検証用が 2 枚以上と, 学習用"
              f"が群の数 ({cfg.group.k}) 以上になりません. 写真を足すか, "
              "config の data.valid か group.k を見直してください")
        sys.exit(1)

    return train, valid


def make_groups(
    points: np.ndarray, train: np.ndarray, cfg: Any
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    学習用の写真で群の中心を決め, 全ての写真を割り当てて, 群の安定性と
    まとまりを数える関数

    Parameters
    ----------
    points: np.ndarray
        (枚数, 幅) の点 (埋め込みか縮めた画素)
    train: np.ndarray
        学習用の写真の番号
    cfg: Any
        group_settings の設定

    Returns
    ----------
    found: tuple[np.ndarray, np.ndarray, dict]
        (枚数,) の群の番号, (群の数, 幅) の群の中心, ari, ari_min,
        jaccard と silhouette
    """
    # まとめるで学習する骨組みと同じ k-means と作り直し
    rng = np.random.default_rng(cfg.experiment.seed)
    centers = kmeans(points[train], cfg.group.k, rng, cfg.group.iterations,
                     cfg.group.inits)
    labels = assign(points, centers)
    scores = stability(points, train, labels, cfg)
    pick = np.random.default_rng(cfg.experiment.seed)
    scores["silhouette"] = silhouette(points, labels, cfg.group.sample, pick)
    found = (labels, centers, scores)

    return found


def summary_rows(
    libraries: SimpleNamespace, frame: pd.DataFrame, labels: np.ndarray,
    jaccard: np.ndarray
) -> list[dict]:
    """
    ML-EDA の群の要約に, 群ごとの安定性を入れる関数

    Parameters
    ----------
    libraries: SimpleNamespace
        load_libraries の関数
    frame: pd.DataFrame
        読めた写真の読み込みの表
    labels: np.ndarray
        (枚数,) の群の番号
    jaccard: np.ndarray
        (群の数,) の群ごとの Jaccard 係数の平均 (写真の無い群は NaN)

    Returns
    ----------
    rows: list[dict]
        群の番号の順の行と, 最後に全ての写真の行 (group は「全体」)
    """
    # 列は写真の幅と高さ. 全体の行の安定性は None のまま
    table = libraries.Data(kind="table",
                           frame=frame[SUMMARY_COLUMNS].reset_index(
                               drop=True
                           ))
    rows = libraries.describe_groups(table, columns=SUMMARY_COLUMNS,
                                     labels=labels.tolist())["groups"]
    for row in rows[:-1]:
        overlap = jaccard[int(row["group"])]
        row["stability"] = None if np.isnan(overlap) else float(overlap)

    return rows


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
        "name": name, "split": "all", "kind": "table", "x": None,
        "columns": columns, "rows": rows, "description": description,
        "baseline": baseline, "guard": False, "better": directions,
        "epoch": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(evaluation, ensure_ascii=False, indent=2)
    (folder / f"{name}.json").write_text(text + "\\n", encoding="utf-8")


def write_artifacts(
    place: Path, frame: pd.DataFrame, labels: np.ndarray, valid: np.ndarray,
    centers: np.ndarray, info: dict, cfg: Any
) -> None:
    """
    写真ごとの群の割り当てと, 既製のモデルと群の中心を成果物に書く関数

    Parameters
    ----------
    place: Path
        成果物のディレクトリ
    frame: pd.DataFrame
        読めた写真の読み込みの表
    labels: np.ndarray
        (枚数,) の群の番号
    valid: np.ndarray
        検証用の写真の番号
    centers: np.ndarray
        (群の数, 埋め込みの幅) の群の中心
    info: dict
        埋め込みの一式の inference.yaml の中身
    cfg: Any
        group_settings の設定 (群の数)

    Returns
    ----------
    None
    """
    # 写真の場所, 群, 分割
    place.mkdir(parents=True, exist_ok=True)
    split = np.full(len(labels), "train", dtype=object)
    split[valid] = "valid"
    pd.DataFrame({"path": frame["path"], "group": labels,
                  "split": split}).to_csv(place / ASSIGNMENT_FILE,
                                          index=False)

    # 埋め込みを作った既製のモデル (版の決まった重みの名前と hash)
    ready = info["ready"]
    sha256 = info["model"]["pretrained"][0]["sha256"]
    text = json.dumps({
        "k": int(cfg.group.k),
        "ready": {"library": ready["library"], "model": ready["model"],
                  "weights": ready["weights"], "sha256": sha256},
        "centers": centers.tolist(),
    }, ensure_ascii=False, indent=2)
    (place / CENTERS_FILE).write_text(text + "\\n", encoding="utf-8")


def build_bundle(
    libraries: SimpleNamespace, settings: tuple, artifacts: Path,
    photos: Path
) -> dict:
    """
    成果物と写真から, 既製の埋め込みの群分けの一式を作る関数

    Parameters
    ----------
    libraries: SimpleNamespace
        load_libraries の関数
    settings: tuple
        bundle_settings の結果 (一式の場所, 名前, 置き換えるか)
    artifacts: Path
        成果物のディレクトリ
    photos: Path
        群に分けた写真のディレクトリ (割り当て直して確かめる)

    Returns
    ----------
    info: dict
        一式の inference.yaml の中身 (作れなければ理由を示して止まる)
    """
    # 重みは一式に写さず, 使うときに出どころから読む
    place, name, overwrite = settings
    try:
        info = libraries.make_group_bundle(place, artifacts, name=name,
                                           data=str(photos),
                                           overwrite=overwrite)
    except (FileExistsError, ValueError) as error:
        print(f"一式を作れません: {error}")
        sys.exit(1)

    return info


def report(
    info: dict, place: Path, scores: dict, base: dict, pinned: Optional[str]
) -> None:
    """
    群の安定性と, 作った一式の群の数, 割り当て直しの一致, 注意を表示する
    関数

    Parameters
    ----------
    info: dict
        make_group_bundle が返した inference.yaml の中身
    place: Path
        一式のディレクトリ
    scores: dict
        埋め込みの群の ari, ari_min, silhouette
    base: dict
        縮めた画素の群の ari, ari_min, silhouette
    pinned: Optional[str]
        config に決めておいた hash (無ければ None)

    Returns
    ----------
    None
    """
    # 群の数, 写真の数, 一致と版 (hash を決めていなければ書き戻す案内)
    group = info.get("group") or {}
    check = group.get("check") or {}
    records = (info.get("model") or {}).get("pretrained") or [{}]
    sha256 = records[0].get("sha256")
    print(f"問題: {PROBLEM_NAME}. 一式を作りました: {place} (版 "
          f"{info.get('format_version')})")
    if PREDICT:
        print(f"まとめるもの: {PREDICT}")
    print(f"群の数: {group.get('k')}, 写真: {group.get('rows')} 枚. 割り当て"
          f"直しの一致: {check.get('agreement')}")
    print(f"群の安定性: 一致 (ari) {scores['ari']:.4f} (縮めた画素 "
          f"{base['ari']:.4f}), シルエット係数 {scores['silhouette']:.4f} "
          f"(縮めた画素 {base['silhouette']:.4f})")
    if sha256:
        print(f"重みの hash (SHA-256): {sha256}")
    if sha256 and not pinned:
        print("この hash を config の ready.sha256 (と問題の設定の重みの "
              "hash) に書くと, 次から違う重みでは作りません")
    for note in info.get("notes") or []:
        print(f"注意: {note}")


def main() -> None:
    """
    config の既製のモデルの埋め込みで写真を群に分け, 群分けの一式を作る
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
    cfg = group_settings(config)
    settings = bundle_settings(config)
    embedder, artifacts = output_places(config)
    photos = PROJECT_DIR / str(cfg.data.photos or "outputs/photos")
    if not photos.is_dir():
        print(f"{photos} がありません. README の「次にすること」の 1 で, "
              "群に分ける写真を置いてください")
        sys.exit(1)

    # 写真の埋め込みと縮めた画素, 無作為な分け方
    libraries = load_libraries()
    frame, points, pixels, info = embed_photos(libraries, ready, embedder,
                                               photos)
    train, valid = split_photos(len(frame), cfg)
    print(f"写真: train {len(train)} 枚, valid {len(valid)} 枚. 埋め込みの"
          f"幅: {points.shape[1]}. 群の数: {cfg.group.k}")

    # 埋め込みの群と, 比べる相手の縮めた画素の群
    labels, centers, scores = make_groups(points, train, cfg)
    _, _, base = make_groups(pixels, train, cfg)

    # 群の安定性と群の要約, 割り当てと群の中心を成果物へ
    method = STABILITY_NAMES[cfg.group.stability]
    model = f"{info['ready']['library']} の {info['ready']['model']}"
    save_table(
        artifacts / EVALUATIONS_DIR, STABILITY,
        [compare_row(scores, base, cfg)],
        STABILITY_TEXT.format(method=method, repeats=cfg.group.repeats,
                              model=model),
        STABILITY_BETTER, baseline=BASELINE_NAME
    )
    save_table(
        artifacts / EVALUATIONS_DIR, SUMMARY,
        summary_rows(libraries, frame, labels, scores["jaccard"]),
        SUMMARY_TEXT.format(model=model), SUMMARY_BETTER
    )
    write_artifacts(artifacts, frame, labels, valid, centers, info, cfg)
    print(f"成果物: {artifacts}")

    # 既製の埋め込みの群分けの一式 (新しい写真を最も近い群に割り当てる)
    bundle = build_bundle(libraries, settings, artifacts, photos)
    report(bundle, settings[0], scores, base, ready["sha256"])


if __name__ == "__main__":
    main()
'''

# 群分けと一式づくりの設定
READY_GROUP_CONFIG = Template('''# 既製のモデルの埋め込みで写真を群に分け, 推論の一式を作る設定 (ML-Quest が
# 問題の設定から書き出した骨組み). bundle.py が読む. 学習はしない

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

# 分け方, k-means と作り直しの乱数の種
seed: 0

data:
  # 群に分ける写真のディレクトリ (プロジェクトのディレクトリから)
  photos: $photos
  # 無作為に分ける, 検証用の写真の割合
  valid: $valid

group:
  # 群の数
  k: $k
  # 群の安定性の数え方 (seed は種を変えて, sample は学習用の写真の fraction
  # の割合を引き直して, repeats 回作り直す)
  stability: $stability
  repeats: 5
  fraction: 0.8
  # k-means (Lloyd 法) の繰り返しの上限と, 中心の置き方を変える回数 (群の
  # 中心までの二乗距離の和が最も小さいものを使う)
  iterations: 100
  inits: 5
  # シルエット係数を数える写真の数の上限 (多ければ無作為に選ぶ)
  sample: 2000

outputs:
  # 埋め込みの一式 (動かすたびに作り直す) と, 成果物 (群の中心, 割り当て,
  # 評価の表) のディレクトリ (プロジェクトのディレクトリから)
  embedder: $embedder
  groups: $groups

bundle:
  # 群分けの一式を作るディレクトリ (プロジェクトのディレクトリから)
  path: $path
  # 一式の名前 (null はディレクトリの名前)
  name: $name
  # 前に作った一式を置き換えるか
  overwrite: false
''')


def ready_group_message(problem: dict) -> Optional[str]:
    """
    既製の埋め込みの群分けの骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. まとめるで, 作り方が既製のモデルをそのまま)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (出力の形が群でない, 入れたデータが画像のディレクトリ
        でない. 書き出せるなら None)
    """
    # 写真のディレクトリを既製のモデルで埋め込む
    output = active_value(problem, f"{PART}.output")
    loader = active_value(problem, "data.loader")
    source = get_value(problem, "data.source")
    message = None
    if output != "cluster":
        message = ("既製の埋め込みの群分けの骨組みは, 出力の形が「群」の場合"
                   f"だけです (今は{choice_label(OUTPUTS, output, problem)})")
    elif source and loader != "image_folder":
        message = ("既製の埋め込みの群分けの骨組みは, 写真のディレクトリを"
                   f"群に分けます (入れたデータ {Path(str(source)).name} は"
                   "画像のディレクトリとして読んでいません). 写真のディレ"
                   "クトリを入れてください")

    return message


def ready_group_name(stages: dict, problem: dict, name: Optional[str]) -> str:
    """
    既製の埋め込みの群分けの一式の名前を決める関数

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
        文字と数字と _ と - だけの名前 (省略するとモデルの名前と群の数
        から)
    """
    # 区切りや空白などのディレクトリに使いにくい文字の _ への置き換え
    ready = stages["inference"]["ready"]
    k = active_value(problem, f"{PART}.group.k")
    raw = name or f"{ready['model']}_group_{k}"
    text = re.sub(r"[^\w-]+", "_", raw).strip("_") or "group"

    return text


def ready_group_config(
    stages: dict, problem: dict, name: Optional[str], photos: str
) -> str:
    """
    既製の埋め込みの群分けの設定を作る関数

    Parameters
    ----------
    stages: dict
        受け渡し (inference.ready に一式づくりに渡す値)
    problem: dict
        問題の設定 (版 2. 群の数と安定性の数え方)
    name: Optional[str]
        一式の名前 (省略するとモデルの名前と群の数から)
    photos: str
        群に分ける写真のディレクトリ (プロジェクトのディレクトリから)

    Returns
    ----------
    text: str
        configs/default/bundle.yaml の中身
    """
    # 文字は JSON の文字列 (YAML としても読める), 群の数は整数
    ready = stages["inference"]["ready"]
    k = active_value(problem, f"{PART}.group.k")
    text = READY_GROUP_CONFIG.substitute(
        **{key: yaml_value(ready[key])
           for key in ("library", "model", "weights", "sha256")},
        photos=yaml_value(photos), valid=GROUP_VALID,
        k=json.dumps(int(k) if isinstance(k, float) else k),
        stability=yaml_value(
            active_value(problem, f"{PART}.group.stability")
        ),
        embedder=yaml_value(READY_GROUP_EMBEDDER),
        groups=yaml_value(READY_GROUP_ARTIFACTS),
        path=yaml_value(READY_BUNDLE),
        name=yaml_value(ready_group_name(stages, problem, name)),
    )

    return text


def ready_group_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 既製の埋め込みで写真を群に分けて一式を作るスクリプト
    を作る関数

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
    # 頭, 設定の読み方 (既製のモデルの骨組みと同じ), k-means と安定性
    # (まとめるの骨組みと同じ), 群分けと一式づくりの流れ
    head = Template(READY_GROUP_HEAD).substitute(
        config=literal(READY_CONFIG),
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
    )
    script = (head + "\n" + READY_SETTINGS + "\n" + GROUP_CLUSTER
              + READY_GROUP_MAIN)

    return script


def ready_group_readme(
    problem: dict, stages: dict, photos: str, raw: Optional[str]
) -> str:
    """
    既製の埋め込みの群分けのプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    photos: str
        群に分ける写真のディレクトリ (プロジェクトのディレクトリから)
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
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, 既製の"
             "モデルの埋め込みで写真を群に分ける骨組み (学習しない. 群の安定"
             "性と群の要約を残し, ML-Inference の推論の一式を作る).", "",
             "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1 から 3. 写真, 群分けと一式づくり, 推論
    k = active_value(problem, f"{PART}.group.k")
    names = {item["value"]: item["label"] for item in GROUP_STABILITIES}
    stability = active_value(problem, f"{PART}.group.stability")
    method = names.get(str(stability), str(stability))
    pinned = (f"決めておいた hash (`{ready['sha256'][:12]}…`) と読んだ重み"
              "が違えば作らない" if ready["sha256"] else
              "hash を決めていないので比べずに作り, 求めた hash を表示する. "
              "それを config の `ready.sha256` と問題の設定の重みの hash に"
              "書くと, 次から同じ重みに固定できる")
    lines += [
        "", "## 次にすること", "",
        "データの分析, モデルの作成, 学習の段階は通らない. 動かす環境には "
        "ML-Inference (ML-EDA と PyTorch も入る) と, torchvision の重みを"
        "読む ML-ModelControl の追加の依存 (`ml-modelcontrol[torchvision]`)"
        " を入れる.", "",
        "1. **写真**: 群に分ける写真のディレクトリを config の "
        f"`data.photos` の場所 (`{photos}`) に置く (ML-EDA の画像の読み込み"
        "で読む. 読めない写真は除く)"
        + (f". 入れた写真はこのディレクトリの `{raw}` に写した" if raw
           else ""),
        "2. **群分けと一式づくり** (ML-Inference): `python bundle.py`. "
        f"config (`{READY_CONFIG}`) の `ready` の既製のモデル (出どころ "
        f"`{ready['library']}`, モデル `{ready['model']}`, 重み "
        f"`{ready['weights']}`) の埋め込みの一式を `outputs.embedder` に作り"
        " (重みは一式に写さず出どころから読む. 初回は取りに行き, PyTorch の"
        "置き場 `torch.hub.get_dir()` の `checkpoints` に置く. ネットワーク"
        f"を使えない場所では, そこに重みのファイルを置く. {pinned}), 写真を"
        "分類の最後の層の手前の出力で埋め込む. 写真を無作為に学習用と検証用"
        " (config の `data.valid` の割合) に分け, 学習用の写真の埋め込みで "
        f"k-means の {k} つの群 (config の `group.k`) の中心を決め, 全ての"
        "写真を割り当てる. 群の安定性は, 作り直した群と元の群の一致 (調整"
        f"ランド指数. {method}. config の `group.stability` と "
        "`group.repeats` の回数) とシルエット係数で数え, 写真を縮めた画素を"
        "そのまま k-means で分けた群と並べる. 成果物は `outputs.groups` "
        "(群の中心と既製のモデルの `group_centers.json`, 写真ごとの群の割り"
        "当て `group_assignments.csv`, 群の安定性の "
        "`evaluations/group_stability.json`, 写真の幅と高さの群の要約の "
        "`evaluations/group_summary.json`) に残る. 続けて `bundle.path` (`"
        + READY_BUNDLE + "`) に ML-Inference の既製の埋め込みの群分けの一式 "
        "(版 10) を作り, 写真を割り当て直して成果物の割り当てと比べる",
        "3. **推論** (ML-Inference, ML-Provide): ML-Provide で作った一式を"
        "登録し, 写真のファイルかディレクトリを渡すと, 写真ごとに最も近い群"
        "の番号 `group` とその中心までの距離 `distance` が返り, 群の要約と"
        "群の中心に近い写真も返る. 一式を作るときに重みのライセンスの注意が"
        "出たら, 使ってよいかを確かめる",
        "", "群の数を変えるなら config の `group.k` を変えて動かし直し, 群の"
        "まとまりと安定性を比べる. 問題の設定の既製のモデルの答えを変えたら,"
        " 書き出し直すか config を手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
