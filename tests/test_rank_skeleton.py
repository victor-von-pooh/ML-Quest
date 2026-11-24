import ast
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Iterator

import pytest
import yaml

from ml_quest import (
    answer, attach_data, check, describe, example_problem, handoff,
    make_project, project_error, questions, rank_encoding, record_project
)
from ml_quest.handoff import model_example

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の反応の表の, 利用者, 品物, 行の数
USERS = 60
ITEMS = 40
ROWS = 3000

# 合成の反応の表の列
COLUMNS = ["member", "item_id", "ts", "qty"]

# 骨組みを書き換えずに短く学ぶ config の差分
SHORT = ["trainer.epochs=3", "experiment.device=cpu"]


def rank_problem() -> dict:
    """
    見本の課題から, 並べるの受け持ちの問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        並べるで出力の形が順位, 指標が NDCG で, 反応の表の列と件数に
        答えた問題の設定 (データは入れていない)
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = answer(problem, "information.columns", COLUMNS)
    for key, value in [("role", "rank"), ("unit", "query"),
                       ("output", "ranking"),
                       ("target.source", "feedback"), ("method", "scratch"),
                       ("what", "会員ごとに勧める品"),
                       ("item", "会員ごとの品の並び"),
                       ("baseline", "人気順"),
                       ("evaluation.metric", "ndcg"),
                       ("evaluation.link", "上位に買う品が来るほど売上が"
                        "増える"),
                       ("rank.user", "member"), ("rank.item", "item_id"),
                       ("rank.time", "ts"), ("rank.reaction", "qty"),
                       ("rank.k", 10)]:
        problem = answer(problem, f"{PART}.{key}", value)

    return problem


def write_reactions(root: Path) -> Path:
    """
    利用者のまとまりごとに好む品物のある, 合成の反応の表の CSV を作る関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        CSV のファイル (member, item_id, ts, qty)
    """
    # 利用者は番号を 4 で割った余りのまとまりの品物に 8 割で反応する.
    # 反応の強さ 0 の行は反応でない
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    rows = []
    for number in range(ROWS):
        user = int(rng.integers(0, USERS))
        item = (int(rng.choice(np.arange(user % 4, ITEMS, 4)))
                if rng.random() < 0.8 else int(rng.integers(0, ITEMS)))
        rows.append({"member": f"m{user}", "item_id": f"i{item}",
                     "ts": number, "qty": int(rng.integers(0, 3))})
    root.mkdir(parents=True, exist_ok=True)
    path = root / "reactions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def attached(root: Path) -> dict:
    """
    反応の表を入れた並べるの問題の設定を作る関数

    Parameters
    ----------
    root: Path
        データと写しを置く場所

    Returns
    ----------
    problem: dict
        データを入れた問題の設定
    """
    # ML-EDA の csv で読む
    pytest.importorskip("ml_eda")
    source = write_reactions(root / "raw")
    problem = attach_data(rank_problem(), str(source),
                          copy_to=str(root / "copies"))

    return problem


def tower_spec(users: int, items: int) -> dict:
    """
    ML-ModelControl の同梱の例 two_tower_retrieval の入力の大きさを, 利用者
    と品物の数に合わせた定義を作る関数

    Parameters
    ----------
    users: int
        利用者の数
    items: int
        品物の数

    Returns
    ----------
    spec: dict
        2 塔のモデルの定義
    """
    # 同梱の例の入力と 1 段目の全結合を書き換え
    spec_module = pytest.importorskip("ml_modelcontrol.spec")
    spec = json.loads(json.dumps(spec_module.load_example(
        "two_tower_retrieval"
    )))
    spec["inputs"]["user"]["shape"] = [8, users]
    spec["inputs"]["item"]["shape"] = [8, items]
    spec["nodes"]["user_fc1"]["params"]["in_features"] = users
    spec["nodes"]["item_fc1"]["params"]["in_features"] = items

    return spec


def run_script(
    project: Path, extra: tuple = ()
) -> subprocess.CompletedProcess:
    """
    書き出した骨組みを書き換えずに, 別のプロセスで短く学習する関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    extra: tuple = ()
        SHORT に足す config の差分

    Returns
    ----------
    done: subprocess.CompletedProcess
        終わったプロセス (表示と終わりの番号)
    """
    # config の差分だけを渡す
    done = subprocess.run([sys.executable, "train.py", *SHORT, *extra],
                          cwd=project, capture_output=True, text=True,
                          timeout=600)

    return done


def train(project: Path, extra: tuple = ()) -> tuple[Path, str]:
    """
    骨組みで短く学習し, 実験のディレクトリを返す関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    extra: tuple = ()
        SHORT に足す config の差分

    Returns
    ----------
    run: Path
        実験のディレクトリ
    stdout: str
        学習のスクリプトの表示
    """
    # 実験は 1 つだけ
    done = run_script(project, extra)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    runs = sorted((project / "outputs").glob("*/*/*/train.log"))
    assert len(runs) == 1
    run = runs[0].parent

    return run, done.stdout


def history_rows(run: Path) -> list[dict]:
    """
    実験の history.csv を読む関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ

    Returns
    ----------
    rows: list[dict]
        エポックの順の行 (列の名前から値の文字)
    """
    # エポックごとの行
    with open(run / "history.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    return rows


def test_rank_questions() -> None:
    """
    並べるの問いが, 並べるで学習するときだけ出て, 足りない答えと食い違い
    を確かめることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 並べるでは使えるデータの列と反応の表の列と件数を問い, 列の予測の問い
    # は問わない
    problem = rank_problem()
    shown = {item["id"]: item for item in questions(problem)
             if item["applies"]}
    for key in ("user", "item", "time", "reaction", "k"):
        assert f"{PART}.rank.{key}" in shown
    assert "information.columns" in shown
    assert f"{PART}.target.column" not in shown
    assert shown[f"{PART}.rank.reaction"]["required"] is False

    # 列の予測と, 既製のモデルをそのまま使う並べるでは問わない
    for other in (example_problem("総菜の作る量"),
                  answer(problem, f"{PART}.method", "pretrained")):
        asked = {item["id"] for item in questions(other)
                  if item["applies"]}
        assert f"{PART}.rank.user" not in asked

    # 足りない答え, 同じ列, 1 以上の整数でない件数
    empty = answer(problem, f"{PART}.rank.user", None)
    assert f"{PART}.rank.user" in [
        item["id"] for item in check(empty) if item["level"] == "error"
    ]
    same = answer(problem, f"{PART}.rank.item", "member")
    found = [item for item in check(same) if item["level"] == "error"]
    assert [item["id"] for item in found] == [f"{PART}.rank.item"]
    assert "利用者の列と同じ列" in found[0]["message"]
    for k in (0, 2.5, -1):
        wrong = answer(problem, f"{PART}.rank.k", k)
        assert f"{PART}.rank.k" in [
            item["id"] for item in check(wrong) if item["level"] == "error"
        ]

    # 列を答え直した後の古い名前
    renamed = answer(problem, "information.columns",
                     ["user", "item_id", "ts", "qty"])
    assert f"{PART}.rank.user" in [
        item["id"] for item in check(renamed) if item["level"] == "error"
    ]


def test_rank_handoff() -> None:
    """
    並べるの問題の設定が渡せ, 分割は時刻の順で, 反応の表の列と件数が
    渡り, モデルの作成の雛形が 2 塔の例になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 誤りが無く, 書き出せる. 監視は valid/ndcg
    problem = rank_problem()
    assert [item for item in check(problem) if item["level"] == "error"] == []
    assert project_error(problem) is None
    stages = handoff(problem)
    assert stages["data"]["split"] == "time"
    assert stages["data"]["time_column"] == "ts"
    assert stages["data"]["target"] is None and stages["data"]["task"] is None
    assert stages["rank"] == {"user": "member", "item": "item_id",
                              "time": "ts", "reaction": "qty", "k": 10}
    assert (stages["training"]["monitor"], stages["training"]["mode"]) == (
        "valid/ndcg", "max"
    )
    assert stages["model"]["example"] == "two_tower_retrieval"
    assert model_example(problem) == "two_tower_retrieval"
    assert stages["inference"]["decision"] is None
    assert any("上位 10 件" in sentence for sentence in describe(problem))

    # Recall@K と損失でも書き出せる. 反応の強さは任意
    recall = answer(problem, f"{PART}.evaluation.metric", "recall_at_k")
    assert project_error(recall) is None
    assert handoff(recall)["training"]["monitor"] == "valid/recall_at_k"
    loss = answer(problem, f"{PART}.evaluation.metric", "loss")
    assert handoff(loss)["training"]["mode"] == "min"
    plain = answer(problem, f"{PART}.rank.reaction", None)
    assert handoff(plain)["rank"]["reaction"] is None

    # 列の予測では None
    assert handoff(example_problem("総菜の作る量"))["rank"] is None


def test_rank_errors(tmp_path: Path) -> None:
    """
    並べるで書き出せない場合が, make_project と同じ文になることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表でないデータと, 既製のモデルをそのまま使う並べる
    images = rank_problem()
    images["data"] = {"source": str(tmp_path / "images"),
                      "loader": "image_folder"}
    ready = answer(rank_problem(), f"{PART}.method", "pretrained")
    for problem, words in ((images, "表として読んでいません"),
                           (ready, "既製のモデル")):
        message = project_error(problem)
        assert message is not None and words in message, message
        with pytest.raises(ValueError) as raised:
            make_project(problem, tmp_path / "project")
        assert str(raised.value) == message
    assert not (tmp_path / "project").exists()


def test_rank_project(tmp_path: Path) -> None:
    """
    データを入れない並べるの問題の設定から骨組みを書き出し, config,
    README, スクリプトが反応の表の置き場と列を指すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出したファイル
    problem = rank_problem()
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert set(written) == {
        "problem.yaml", "README.md", "configs/default/train.yaml",
        "configs/experiment/train.yaml", "train.py",
    }
    script = (project / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'ROLE = "rank"' in script and 'OUTPUT = "ranking"' in script
    assert 'PREDICT = "会員ごとに勧める品"' in script
    assert "PlaceholderModel" in script
    assert "make_bundle" not in script
    assert all(len(line) <= 79 for line in script.splitlines())

    # config: 要る機能, 監視, 反応の表の場所, 列と件数, モデルの定義
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol", "inference"]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/ndcg", "max"
    )
    assert config["data"]["dataset"] == "outputs/interactions.csv"
    assert (config["data"]["valid"], config["data"]["test"]) == (0.1, 0.1)
    assert config["rank"] == {"user": "member", "item": "item_id",
                              "time": "ts", "reaction": "qty", "k": 10}
    assert config["model"]["spec"] == "models/model.yaml"
    assert config["model"]["placeholder"] is True
    assert config["experiment"]["name"] == "item_id_rank"
    place = record_project(problem, project)["project"]["dataset"]
    assert place == str((project / "outputs/interactions.csv").resolve())

    # README: 反応の表の列, 2 塔の例, 時刻の順の分割, 指標, 並べるの一式
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("outputs/interactions.csv", "`member`", "`item_id`",
                  "`ts`", "`qty`", "two_tower_retrieval", "時刻の順",
                  "Recall@K", "NDCG", "ranking_test.json", "人気順",
                  "rank_index.json", "並べるの推論の一式 (版 8",
                  "`user_embedding`", "`rank.k` の 10"):
        assert words in readme, words

    # 反応の表が無ければ, 置き場を案内して止まる
    for name in ("torch", "ml_eda", "ml_found", "ml_inference"):
        pytest.importorskip(name)
    done = run_script(project)
    assert done.returncode == 1
    assert "反応の表を置いてください" in done.stdout


def test_rank_placeholder(tmp_path: Path) -> None:
    """
    モデルの定義が無いうちは仮のモデルで学び, NDCG で最良を選び, 検証用と
    テスト用の Recall@K と NDCG を人気順と並べて残し, 一式を作らないことを
    確かめる関数 (PyTorch と ML-Found が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 入れた反応の表は data/ に写して読む
    for name in ("torch", "ml_found", "ml_inference"):
        pytest.importorskip(name)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert "data/reactions.csv" in written
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["data"]["dataset"] == "data/reactions.csv"
    run, stdout = train(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル (PlaceholderModel) で学習します" in log
    assert "並べるもの: 会員ごとに勧める品" in log
    assert f"利用者の数: {USERS}" in log and f"品物の数: {ITEMS}" in log
    assert "反応の強さ (qty) が 0 以下か数でない" in log
    assert "推論の一式を作りません" in log
    assert not (run / "inference").exists()

    # 検証のたびの NDCG で最良を選ぶ
    rows = history_rows(run)
    values = [float(row["valid/ndcg"]) for row in rows]
    assert len(values) == 3 and all(0 <= value <= 1 for value in values)
    assert all(0 <= float(row["valid/recall_at_k"]) <= 1 for row in rows)
    assert f"ベストの valid/ndcg: {max(values)}" in stdout

    # 検証用とテスト用の評価: 人気順との差, 向き
    for split in ("valid", "test"):
        table = json.loads((run / "evaluations" / f"ranking_{split}.json")
                           .read_text(encoding="utf-8"))
        assert table["split"] == split
        assert table["baseline"] == "人気順 (学習用の反応の多い順)"
        assert table["better"]["ndcg"] == "higher"
        assert table["better"]["recall_at_k_diff"] == "higher"
        row = table["rows"][0]
        assert row["k"] == 10 and row["users"] > 0
        for name in ("recall_at_k", "ndcg"):
            assert 0 <= row[name] <= 1 and 0 <= row[f"{name}_baseline"] <= 1
            assert row[f"{name}_diff"] == pytest.approx(
                row[name] - row[f"{name}_baseline"]
            )
    valid = json.loads((run / "evaluations" / "ranking_valid.json")
                       .read_text(encoding="utf-8"))
    assert valid["rows"][0]["ndcg"] == pytest.approx(max(values))

    # 何の実験かと, 利用者と品物の番号の表
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"]) == ("rank", "ranking")
    index = json.loads((run / "rank_index.json").read_text(encoding="utf-8"))
    assert len(index["users"]) == USERS and len(index["items"]) == ITEMS


def test_rank_training(tmp_path: Path) -> None:
    """
    並べるの骨組みを書き換えずに, 同梱の例 two_tower_retrieval の大きさを
    合わせた定義で学び, 並べるの一式で利用者ごとに上位 K 件を並べ,
    一式の Recall@K と NDCG がテスト用の評価と同じになることと, 大きさの
    合わない定義では出力先を作る前に止まることを確かめる関数 (PyTorch と
    段階のライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 大きさの合わない定義 (同梱の例のまま) は止まる
    for name in ("torch", "ml_found", "ml_modelcontrol", "ml_inference"):
        pytest.importorskip(name)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    make_project(problem, project)
    place = Path(record_project(problem, project)["project"]["model_spec"])
    place.parent.mkdir(parents=True, exist_ok=True)
    place.write_text(yaml.safe_dump(tower_spec(12, 20), allow_unicode=True),
                     encoding="utf-8")
    done = run_script(project)
    assert done.returncode == 1
    assert f"利用者の数 {USERS}" in done.stdout
    assert not (project / "outputs").exists()

    # 大きさを合わせた定義で学び, 実験に定義が写る
    place.write_text(yaml.safe_dump(tower_spec(USERS, ITEMS),
                                    allow_unicode=True), encoding="utf-8")
    run, _ = train(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "PlaceholderModel" not in log
    assert (run / "model_spec.yaml").is_file()

    # 学習の前に求めた数と分割の行の数が, train.log の初めの行と同じ
    dataset = record_project(problem, project)["project"]["dataset"]
    encoding = rank_encoding(problem, dataset)
    assert (f"利用者の数: {encoding['users']}, 品物の数: "
            f"{encoding['items']}. 反応: " + ", ".join(
                f"{name} {rows} 行"
                for name, rows in encoding["rows"].items())) in log
    table = json.loads((run / "evaluations" / "ranking_test.json")
                       .read_text(encoding="utf-8"))
    row = table["rows"][0]
    assert 0 <= row["ndcg"] <= 1
    check_bundle(run, dataset, row)


def check_bundle(run: Path, dataset: str, row: dict) -> None:
    """
    学習の後に作った並べるの一式で利用者ごとに上位 K 件を並べ, 一式の
    Recall@K と NDCG がテスト用の評価の表と同じになることを確かめる関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ
    dataset: str
        反応の表の場所 (骨組みが読んだ写し)
    row: dict
        テスト用の評価の表の行

    Returns
    ----------
    None
    """
    # 一式は版 8 で, 番号の表と config の rank.k と列を持つ
    from ml_inference import infer, predict, rank_metrics
    pd = pytest.importorskip("pandas")
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "推論の一式 (並べる): inference/" in log
    bundle = run / "inference"
    info = yaml.safe_load((bundle / "inference.yaml")
                          .read_text(encoding="utf-8"))
    assert info["format_version"] == 8
    assert info["output"]["form"] == "ranking"
    assert (info["rank"]["users"], info["rank"]["items"]) == (USERS, ITEMS)
    assert info["rank"]["k"] == 10
    assert info["rank"]["columns"] == {"user": "member", "item": "item_id",
                                       "reaction": "qty"}

    # 骨組みと同じ分け方 (反応の強さが 0 より大きい行を, 行の数の割合で
    # 時刻の順に切る). 前の分割の反応は除き, テスト用の反応を正解にする
    frame = pd.read_csv(dataset)
    kept = frame[frame["qty"] > 0].sort_values("ts", kind="stable")
    cut = kept["ts"].iloc[int(len(kept) * 0.9)]
    seen, truth = kept[kept["ts"] < cut], kept[kept["ts"] >= cut]

    # 学習用にいる全ての利用者を並べ, 番号の表に無い利用者は飛ばす
    index = json.loads((run / "rank_index.json").read_text(encoding="utf-8"))
    users = [*index["users"], "m_unknown"]
    ranked = predict(bundle, users, seen=seen)
    done = ranked[ranked["status"] == "ok"]
    assert done.groupby("user").size().eq(10).all()
    assert set(done["user"]) <= set(index["users"])
    assert ranked[ranked["user"] == "m_unknown"]["status"].tolist() == [
        "skipped"
    ]
    found = rank_metrics(bundle, ranked, truth, seen=seen)
    assert found["users"] == row["users"] and found["k"] == 10
    assert found["recall_at_k"] == pytest.approx(row["recall_at_k"])
    assert found["ndcg"] == pytest.approx(row["ndcg"])

    # k を変えて並べ, 結論は利用者の人数だけ
    result = infer(bundle, index["users"][:2], k=3)
    assert len(result["frame"]) == 6
    assert result["sentences"][0] == ("利用者 2 人のうち 2 人に, 上位 3 件の"
                                      "品物を並べた")


class FixedScores:
    def __init__(self, scores: object):
        # 利用者ごとの全ての品物の点数を保存
        self.scores = scores

    def parameters(self) -> Iterator:
        """
        デバイスを決めるための値を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        values: Iterator
            点数の表だけを返す iterator
        """
        # 点数の表と同じデバイス
        values = iter([self.scores])

        return values

    def eval(self) -> "FixedScores":
        """
        評価の形にする関数 (何もしない)

        Parameters
        ----------
        None

        Returns
        ----------
        model: FixedScores
            自分自身
        """
        # 変える状態は無し
        model = self

        return model

    def __call__(self, user: object, item: object) -> object:
        """
        ワンホットの利用者と品物の点数の表だけを返す関数 (2 塔でない定義)

        Parameters
        ----------
        user: object
            (利用者の件数, 利用者の数) のワンホット
        item: object
            (品物の件数, 品物の数) のワンホット

        Returns
        ----------
        score: object
            (利用者の件数, 品物の件数) の点数
        """
        # ワンホットで行と列を選ぶ
        score = user @ self.scores @ item.T

        return score


def script_module(root: Path) -> ModuleType:
    """
    書き出した train.py を, 純粋な関数を呼ぶためにモジュールとして読む関数
    (PyTorch, ML-EDA, ML-Found が無ければ飛ばす)

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    module: ModuleType
        読み込んだ train.py
    """
    # main は呼ばない
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    project = root / "project"
    make_project(rank_problem(), project)
    spec = importlib.util.spec_from_file_location("rank_train",
                                                  project / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_rank_functions(tmp_path: Path) -> None:
    """
    書き出した train.py の, 時刻の順の分割と Recall@K と NDCG の数え方を
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ時刻の反応は同じ分割に入る
    module = script_module(tmp_path)
    import pandas as pd
    import torch
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"data": {"valid": 0.2, "test": 0.2}})
    reactions = pd.DataFrame({"user": ["a"] * 10, "item": list("abcdefghij"),
                              "time": [0, 1, 2, 3, 4, 5, 6, 6, 8, 9]})
    splits = module.split_by_time(reactions, cfg)
    assert [len(splits[name]) for name in ("train", "valid", "test")] == [
        6, 2, 2
    ]
    assert list(splits["valid"]["time"]) == [6, 6]
    with pytest.raises(ValueError):
        module.split_by_time(reactions, OmegaConf.create(
            {"data": {"valid": 0.6, "test": 0.5}}
        ))

    # テスト用は学習用と検証用の両方で反応した品物を並べず, 正解にもしない
    train = pd.DataFrame({"user": [0, 0], "item": [0, 1]})
    valid = pd.DataFrame({"user": [0, 1], "item": [2, 0]})
    test = pd.DataFrame({"user": [0, 0, 1, 1], "item": [2, 3, 0, 3]})
    target = module.rank_target(test, [train, valid])
    assert target["users"] == [0, 1]
    assert target["relevant"] == {0: {3}, 1: {3}}
    assert target["seen"] == {0: {0, 1, 2}, 1: {0}}

    # 2 塔でない定義 (点数の表だけを返す) も, 利用者と品物をまとまりに
    # 分けて除いてから選ぶ. 利用者 0 は品物 3 だけが残り, 利用者 1 は
    # 品物 0 を除いて 2, 3 の順
    scores = torch.tensor([[9.0, 8.0, 7.0, 1.0], [5.0, 0.0, 4.0, 3.0]])
    fixed = FixedScores(scores)
    top = module.top_items(fixed, target, (2, 4), 2, 1)
    assert [items[:1] for items in top] == [[3], [2]]
    assert top[1] == [2, 3]

    # 利用者 0 は品物 3 が 1 位, 利用者 1 は品物 3 が 2 位. Recall@2 は 1 と
    # 1, NDCG は 1 と 1 / log2(3)
    metrics = module.rank_metrics(top, target, 2)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["ndcg"] == pytest.approx((1 + 1 / 1.5849625) / 2,
                                            rel=1e-6)
    metrics = module.rank_metrics(top, target, 1)
    assert metrics["recall_at_k"] == pytest.approx(0.5)

    # 正解が K より多い利用者の Recall@K の分母は正解の全数. NDCG は上から
    # K 件の正解で 1
    many = {"users": [0], "relevant": {0: {0, 1, 2, 3}}, "seen": {}}
    metrics = module.rank_metrics([[0, 1]], many, 2)
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["ndcg"] == pytest.approx(1.0)

    # 人気順: 同じ数なら番号の小さい品物が先で, 利用者ごとに前の分割の品物
    # を除く
    counts = pd.DataFrame({"user": [0] * 5, "item": [2, 2, 1, 3, 0]})
    popular = module.popular_top(counts, target, 4, 3)
    assert popular == [[3], [2, 1, 3]]

    # 2 塔の定義は品物の埋め込みを 1 度だけ作り, 内積で選ぶ. 全ての組の
    # 点数の表から選んだものと同じ
    torch.manual_seed(0)
    tower = module.PlaceholderModel(3, 5, 4).eval()
    every = {"users": [0, 1, 2], "relevant": {}, "seen": {1: {0}}}
    top = module.top_items(tower, every, (3, 5), 2, 2)
    with torch.no_grad():
        full = module.pair_scores(tower, torch.arange(3), torch.arange(5),
                                  (3, 5))
    full[1, 0] = -float("inf")
    assert top == torch.topk(full, 2, dim=1).indices.tolist()


class Messages:
    def __init__(self):
        # 段階ごとの文の並び
        self.lines = {"info": [], "warning": []}

    def info(self, text: str) -> None:
        """
        知らせの文を残す関数

        Parameters
        ----------
        text: str
            文

        Returns
        ----------
        None
        """
        # 知らせの並びに足す
        self.lines["info"].append(text)

    def warning(self, text: str) -> None:
        """
        警告の文を残す関数

        Parameters
        ----------
        text: str
            文

        Returns
        ----------
        None
        """
        # 警告の並びに足す
        self.lines["warning"].append(text)


def test_rank_inference_skips(tmp_path: Path) -> None:
    """
    書き出した train.py が, 仮のモデルと埋め込みの出力の無い定義では一式を
    作らず, 点数が埋め込みの内積の定数倍でない定義では警告を残して止まらない
    ことを確かめる関数 (ML-ModelControl と ML-Inference が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 仮のモデルと, 出力が score だけの定義は作らない
    for name in ("ml_modelcontrol", "ml_inference"):
        pytest.importorskip(name)
    module = script_module(tmp_path)
    import torch
    from types import SimpleNamespace
    from ml_modelcontrol import build_model
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "experiment": {"name": "shop"},
        "rank": {"user": "member", "item": "item_id", "reaction": "qty",
                 "k": 3},
    })
    run = tmp_path / "run"
    run.mkdir()
    messages = Messages()
    experiment = SimpleNamespace(logger=messages)
    module.make_inference(cfg, None, run, experiment)
    assert "推論の一式を作りません" in messages.lines["info"][-1]
    spec = tower_spec(4, 6)
    only = {**spec, "outputs": {"score": spec["outputs"]["score"]}}
    place = tmp_path / "only.yaml"
    place.write_text(yaml.safe_dump(only), encoding="utf-8")
    module.make_inference(cfg, place, run, experiment)
    assert "user_embedding と item_embedding が無い" in (
        messages.lines["info"][-1]
    )

    # 点数が内積の定数倍でない定義 (点数に sigmoid を通したもの) は,
    # ML-Inference の誤りを警告に残し, 一式を作らない
    squashed = json.loads(json.dumps(spec))
    squashed["nodes"]["squash"] = {"unit": "nn.Sigmoid", "inputs": "score"}
    squashed["outputs"]["score"] = "squash"
    place = tmp_path / "squashed.yaml"
    place.write_text(yaml.safe_dump(squashed), encoding="utf-8")
    torch.save(build_model(place, seed=0).state_dict(),
               run / "final_model.pt")
    (run / "rank_index.json").write_text(json.dumps({
        "users": list("abcd"), "items": list("uvwxyz")
    }), encoding="utf-8")
    module.make_inference(cfg, place, run, experiment)
    assert messages.lines["warning"][-1].startswith("推論の一式を作れません")
    assert not (run / "inference").exists()
