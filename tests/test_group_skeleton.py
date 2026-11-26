import ast
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from ml_quest import (
    GROUP_TAKEN, answer, apply_group, apply_recipe, attach_data, check,
    describe, example_problem, handoff, make_project, project_error,
    questions, record_project
)

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の表の, 群の数と群ごとの行の数
GROUPS = 3
ROWS = 80

# 合成の表の列
COLUMNS = ["customer", "spend", "visits", "region"]

# 骨組みを書き換えずに短く学ぶ config の差分
SHORT = ["trainer.epochs=3", "experiment.device=cpu", "group.repeats=3"]

# 守る指標の問い
GUARD_IDS = ["issue.guard.name", "issue.guard.unit", "issue.guard.better",
             "issue.guard.line", "issue.guard.measure",
             "issue.guard.counted"]


def group_problem() -> dict:
    """
    見本の課題から, まとめるの受け持ちの問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        まとめるで出力の形が群, 指標がシルエット係数で, 群を作る列, 群の数,
        安定性の数え方に答えた問題の設定 (データは入れていない. 見本の
        守る指標の答えは残す)
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("解約の引き止め")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = answer(problem, "information.columns", COLUMNS)
    for key, value in [("role", "group"), ("unit", "row"),
                       ("output", "cluster"), ("target.source", "none"),
                       ("method", "scratch"),
                       ("what", "顧客の群"), ("item", "顧客 1 人"),
                       ("baseline", "今の区分 (地域)"),
                       ("evaluation.metric", "silhouette"),
                       ("evaluation.link", "群ごとに連絡の仕方を変える"),
                       ("group.columns", ["spend", "visits", "region"]),
                       ("group.id", "customer"), ("group.k", GROUPS),
                       ("group.stability", "seed")]:
        problem = answer(problem, f"{PART}.{key}", value)

    return problem


def write_rows(root: Path) -> Path:
    """
    3 つのはっきりした群のある, 合成の顧客の表の CSV を作る関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        CSV のファイル (customer, spend, visits, region)
    """
    # 群ごとに使う額, 来た回数, 地域が違う. 欠けた値も含む
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    rows = []
    for group in range(GROUPS):
        for number in range(ROWS):
            rows.append({
                "customer": f"c{group}_{number}",
                "spend": float(rng.normal(1000 * (group + 1), 50)),
                "visits": int(rng.poisson(2 + 6 * group)),
                "region": ("north", "south", "east")[group],
            })
    rows[0]["spend"] = None
    root.mkdir(parents=True, exist_ok=True)
    path = root / "customers.csv"
    pd.DataFrame(rows).sample(frac=1.0, random_state=0).to_csv(path,
                                                               index=False)

    return path


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


def load_script(project: Path) -> dict:
    """
    書き出した train.py を, 実行せずに関数として読む関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    names: dict
        train.py の名前 (main は呼ばない)
    """
    # __name__ が __main__ でないので main は動かない
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    names: dict = {"__name__": "group_train",
                   "__file__": str(project / "train.py")}
    code = (project / "train.py").read_text(encoding="utf-8")
    exec(compile(code, str(project / "train.py"), "exec"), names)

    return names


def test_group_questions() -> None:
    """
    まとめるの問いが, まとめるで学習するときだけ出て, 守る指標の問いが
    出ず, 足りない答えと食い違いを確かめることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # まとめるでは使えるデータの列と群の問いを問い, 列の予測の問いと
    # 守る指標の問いは問わない
    problem = group_problem()
    shown = {item["id"]: item for item in questions(problem)
             if item["applies"]}
    for key in ("columns", "id", "k", "stability"):
        assert f"{PART}.group.{key}" in shown
    assert "information.columns" in shown
    assert f"{PART}.target.column" not in shown
    assert shown[f"{PART}.group.id"]["required"] is False
    assert [item["value"] for item in
            shown[f"{PART}.group.stability"]["choices"]] == ["seed", "sample"]
    assert not set(GUARD_IDS) & set(shown)
    order = [item["id"] for item in questions(problem)]
    assert order.index("information.columns") < order.index(
        f"{PART}.group.columns"
    )

    # 見本に残した守る指標の答えは使わない (確かめにも要約にも出ない)
    assert problem["issue"]["guard"]["name"]
    assert not [item for item in check(problem)
                if item["id"].startswith("issue.guard")]
    assert not any(sentence.startswith("守る指標")
                   for sentence in describe(problem))

    # 機械学習を使わない場合と他の受け持ちでは守る指標を問い, 群の問いは
    # 問わない
    for other in (example_problem("解約の引き止め"),
                  answer(problem, "ml.use", False),
                  answer(problem, f"{PART}.method", "pretrained")):
        asked = {item["id"] for item in questions(other)
                 if item["applies"]}
        assert f"{PART}.group.columns" not in asked
    assert "issue.guard.name" in {
        item["id"] for item in questions(answer(problem, "ml.use", False))
        if item["applies"]
    }
    assert "issue.guard.line" in {
        item["id"] for item in questions(example_problem("解約の引き止め"))
        if item["applies"]
    }

    # 足りない答え, 使えるデータの列に無い列, 群を作る列と同じ行を見分ける
    # 列, 2 以上の整数でない群の数, 知らない数え方
    for key in ("columns", "k", "stability"):
        empty = answer(problem, f"{PART}.group.{key}", None)
        assert f"{PART}.group.{key}" in [
            item["id"] for item in check(empty) if item["level"] == "error"
        ]
    same = answer(problem, f"{PART}.group.id", "spend")
    found = [item for item in check(same) if item["level"] == "error"]
    assert [item["id"] for item in found] == [f"{PART}.group.id"]
    assert "群を作るのに使う列にもあります" in found[0]["message"]

    # 群の割り当ての表の列と同じ名前の行を見分ける列と, ML-EDA の群の要約
    # が受け付けない名前の群を作る列
    for name in ("group", "split"):
        taken = answer(problem, "information.columns", [*COLUMNS, name])
        taken = answer(taken, f"{PART}.group.id", name)
        found = [item for item in check(taken) if item["level"] == "error"]
        assert [item["id"] for item in found] == [f"{PART}.group.id"]
        assert "群の割り当ての表の列" in found[0]["message"]
    for name in ("group", "rows", "share", "stability", "id", "split"):
        taken = answer(problem, "information.columns", [*COLUMNS, name])
        taken = answer(taken, f"{PART}.group.columns", ["spend", name])
        found = [item for item in check(taken) if item["level"] == "error"]
        assert [item["id"] for item in found] == [f"{PART}.group.columns"]
        assert f"群を作る列 {name} は群の要約か 1 件の鍵" in (
            found[0]["message"]
        )
    unknown = json.loads(json.dumps(problem))
    unknown["ml"]["parts"][0]["group"]["stability"] = "bootstrap"
    assert [item["id"] for item in check(unknown)
            if item["level"] == "error"] == [f"{PART}.group.stability"]

    # 列を答え直した後の古い名前
    renamed = answer(problem, "information.columns",
                     ["id", "spend", "region"])
    found = [item for item in check(renamed) if item["level"] == "error"]
    assert sorted(item["id"] for item in found) == [
        f"{PART}.group.columns", f"{PART}.group.id"
    ]
    assert any("visits は使えるデータの列にありません" in item["message"]
               for item in found)
    for k in (1, 2.5, -3):
        wrong = answer(problem, f"{PART}.group.k", k)
        assert f"{PART}.group.k" in [
            item["id"] for item in check(wrong) if item["level"] == "error"
        ]


def test_group_names_match_eda() -> None:
    """
    群を作る列にできない名前が, ML-EDA の群の要約が受け付けない名前と同じ
    ことを確かめる関数 (ML-EDA が無ければ飛ばす)

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 群の要約の鍵と 1 件の決まった鍵
    groups = pytest.importorskip("ml_eda.groups")
    items = pytest.importorskip("ml_eda.items")
    assert set(GROUP_TAKEN) == {*groups.SUMMARY_COLUMNS, *items.RESERVED}
    assert len(GROUP_TAKEN) == len(set(GROUP_TAKEN))


def test_group_handoff() -> None:
    """
    まとめるの問題の設定が渡せ, 分割は無作為で, 群を作る列, 群の数と安定性
    の数え方が渡り, 守る指標と意思決定が None になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 誤りが無く, 書き出せる. 監視は valid/silhouette
    problem = group_problem()
    assert [item for item in check(problem) if item["level"] == "error"] == []
    assert project_error(problem) is None
    stages = handoff(problem)
    assert stages["data"]["split"] == "random"
    assert stages["data"]["target"] is None and stages["data"]["task"] is None
    assert stages["group"] == {"columns": ["spend", "visits", "region"],
                               "id": "customer", "k": 3, "stability": "seed"}
    assert (stages["training"]["monitor"], stages["training"]["mode"]) == (
        "valid/silhouette", "max"
    )
    assert stages["model"]["example"] == "contrastive_encoder"
    assert stages["inference"]["guard"] is None
    assert stages["inference"]["decision"] is None
    assert stages["rank"] is None
    assert any("3 つの群" in sentence and "種を変えて作り直す" in sentence
               for sentence in describe(problem))

    # 損失でも書き出せる. 行を見分ける列は任意. 群の数は整数で渡る
    loss = answer(problem, f"{PART}.evaluation.metric", "loss")
    assert project_error(loss) is None
    assert handoff(loss)["training"]["mode"] == "min"
    plain = answer(problem, f"{PART}.group.id", None)
    assert handoff(plain)["group"]["id"] is None
    assert handoff(answer(problem, f"{PART}.group.k", 4.0))["group"]["k"] == 4

    # 他の受け持ちと, 既製のモデルをそのまま使うまとめるでは None
    assert handoff(example_problem("総菜の作る量"))["group"] is None
    ready = answer(problem, f"{PART}.method", "pretrained")
    for key, value in (("library", "torchvision"), ("model", "resnet18"),
                       ("weights", "IMAGENET1K_V1")):
        ready = answer(ready, f"{PART}.ready.{key}", value)
    assert handoff(ready)["group"] is None


def test_group_errors(tmp_path: Path) -> None:
    """
    まとめるで書き出せない場合が, make_project と同じ文になることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表でないデータと, 一式づくりの値を答えていない既製のモデルをそのまま
    # 使うまとめる
    images = group_problem()
    images["data"] = {"source": str(tmp_path / "images"),
                      "loader": "image_folder"}
    ready = answer(group_problem(), f"{PART}.method", "pretrained")
    for problem, words in ((images, "表として読んでいません"),
                           (ready, "既製のモデル")):
        message = project_error(problem)
        assert message is not None and words in message, message
        with pytest.raises(ValueError) as raised:
            make_project(problem, tmp_path / "project")
        assert str(raised.value) == message
    assert not (tmp_path / "project").exists()


def test_apply_group(tmp_path: Path) -> None:
    """
    群の選び方を書き戻すと, 答え, 確かめ, 受け渡しと骨組みの config が
    選んだ値と同じになり, 変えた点の文が残ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 全ての鍵を書き戻す (元は変えない. 群の数の 5.0 は 5)
    problem = group_problem()
    chosen = {"columns": ["visits", "region"], "id": "spend", "k": 5.0,
              "stability": "sample"}
    result = apply_group(problem, chosen)
    assert problem["ml"]["parts"][0]["group"]["k"] == GROUPS
    assert (result["conflicts"], result["choices"]) == ([], [])
    updated = result["problem"]
    assert updated["ml"]["parts"][0]["group"] == {
        "columns": ["visits", "region"], "id": "spend", "k": 5,
        "stability": "sample",
    }
    assert result["changes"] == [
        "群を作る列を visits, region に書き戻した",
        "行を見分ける列を spend に書き戻した", "群の数を 5 に書き戻した",
        "群の安定性の数え方を sample に書き戻した",
    ]
    assert [item for item in check(updated) if item["level"] == "error"] == []
    expected = {"columns": ["visits", "region"], "id": "spend", "k": 5,
                "stability": "sample"}
    assert handoff(updated)["group"] == expected
    make_project(updated, tmp_path / "project")
    config = yaml.safe_load(
        (tmp_path / "project" / "configs" / "default" / "train.yaml")
        .read_text(encoding="utf-8")
    )
    assert {key: config["group"][key] for key in expected} == expected

    # 書いた鍵だけ. 同じ値は文にしない. id の None は外す
    same = apply_group(problem, {"k": GROUPS, "id": None})
    assert same["changes"] == ["行を見分ける列を外した"]
    assert same["problem"]["ml"]["parts"][0]["group"]["k"] == GROUPS
    assert handoff(same["problem"])["group"]["id"] is None
    assert apply_group(problem, {})["changes"] == []

    # 書き出したプロジェクトの記録があれば, 書き出し直しの案内を添える
    recorded = record_project(problem, tmp_path / "project")
    moved = apply_group(recorded, {"k": 4})
    assert moved["changes"][0] == "群の数を 4 に書き戻した"
    assert "書き出し直して" in moved["changes"][-1]
    assert apply_group(recorded, {"k": GROUPS})["changes"] == []


def test_apply_group_refuses() -> None:
    """
    群の選び方の形の誤りと, まとめるの確かめを通らない値と, まとめるで
    学習しない受け持ちを断ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 形の誤り
    problem = group_problem()
    for group, words in (
        (["spend"], "辞書"), ({"size": 3}, "size"),
        ({"columns": []}, "群を作る列"), ({"columns": [""]}, "群を作る列"),
        ({"columns": ["", None]}, "群を作る列"), ({"k": None}, "群の数"),
        ({"k": ""}, "群の数"), ({"k": "  "}, "群の数"),
        ({"k": True}, "2 以上の整数"), ({"stability": ""}, "数え方"),
        ({"stability": None}, "数え方"),
        ({"stability": "bootstrap"}, "stability"),
        ({"columns": ["spend", "price"]}, "price"),
    ):
        with pytest.raises(ValueError, match=words):
            apply_group(problem, group)

    # まとめるの確かめ (決まった名前, 重なり, 群の数) を通らない値
    named = answer(problem, "information.columns", [*COLUMNS, *GROUP_TAKEN])
    for group, words in (
        ({"columns": ["spend", "share"]}, "群の要約"),
        ({"id": "split"}, "割り当ての表"),
        ({"id": "spend"}, "群を作るのに使う列にも"),
        ({"k": 1}, "2 以上の整数"), ({"k": 2.5}, "2 以上の整数"),
    ):
        with pytest.raises(ValueError, match=words):
            apply_group(named, group)

    # 前から残る古い列の名前も, 選び方で直さなければ断る
    stale = group_problem()
    stale["information"]["columns"] = ["customer", "spend", "visits"]
    with pytest.raises(ValueError, match="region"):
        apply_group(stale, {"k": 4})
    fixed = apply_group(stale, {"columns": ["spend", "visits"]})
    assert fixed["problem"]["ml"]["parts"][0]["group"]["columns"] == [
        "spend", "visits"
    ]

    # まとめるでない受け持ちと, レシピの書き戻しの誤り
    with pytest.raises(ValueError, match="まとめるで学習する"):
        apply_group(example_problem("総菜の作る量"), {"k": 4})
    with pytest.raises(ValueError):
        apply_recipe(problem, {"target": "spend"})


def test_group_project(tmp_path: Path) -> None:
    """
    データを入れないまとめるの問題の設定から骨組みを書き出し, config,
    README, スクリプトが表の置き場と列を指すことと, 表が無ければ案内して
    止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出したファイル
    problem = group_problem()
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert set(written) == {
        "problem.yaml", "README.md", "configs/default/train.yaml",
        "configs/experiment/train.yaml", "train.py",
    }
    script = (project / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'ROLE = "group"' in script and 'OUTPUT = "cluster"' in script
    assert 'PREDICT = "顧客の群"' in script
    assert "PlaceholderModel" in script
    assert "make_group_bundle" in script and "guard" not in script
    assert all(len(line) <= 79 for line in script.splitlines())

    # config: 要る機能, 監視, 表の場所, 群の設定, モデルの定義
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol", "inference"]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/silhouette", "max"
    )
    assert config["data"]["dataset"] == "outputs/rows.csv"
    assert config["data"]["valid"] == 0.2
    group = config["group"]
    assert group["columns"] == ["spend", "visits", "region"]
    assert (group["id"], group["k"], group["stability"]) == (
        "customer", 3, "seed"
    )
    assert config["model"]["spec"] == "models/model.yaml"
    assert config["model"]["placeholder"] is True
    assert config["experiment"]["name"] == "group_3"
    place = record_project(problem, project)["project"]["dataset"]
    assert place == str((project / "outputs/rows.csv").resolve())

    # README: 表の列, 例, 安定性, 群の要約, 群分けの一式
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("outputs/rows.csv", "`spend`", "`region`", "`customer`",
                  "contrastive_encoder", "調整ランド指数",
                  "group_stability.json", "group_summary.json",
                  "group_assignments.csv", "inference/", "ML-Provide",
                  "仮のモデルでは一式を作らない",
                  "種を変えて作り直す"):
        assert words in readme, words
    assert "守る指標" not in readme

    # 表が無ければ, 置き場を案内して止まる
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    done = run_script(project)
    assert done.returncode == 1
    assert "群を作る表を置いてください" in done.stdout


def test_group_functions(tmp_path: Path) -> None:
    """
    書き出した train.py の, k-means, 一致とまとまりの数え方, 列の符号化,
    行の分け方を確かめる関数 (PyTorch と段階のライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 骨組みを書き出して読む
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    project = tmp_path / "project"
    make_project(group_problem(), project)
    names = load_script(project)

    # 調整ランド指数: 番号の付け方に依らず同じ群は 1, 群が 1 つずつでも 1
    same = names["adjusted_rand"](np.array([0, 0, 1, 1, 2, 2]),
                                  np.array([5, 5, 3, 3, 9, 9]))
    assert same == pytest.approx(1.0)
    mixed = names["adjusted_rand"](np.array([0, 0, 0, 1, 1, 1]),
                                   np.array([0, 1, 0, 1, 0, 1]))
    assert mixed < 0.2
    assert names["adjusted_rand"](np.zeros(4), np.zeros(4)) == 1.0

    # 離れた 2 つの点の塊: k-means で分かれ, シルエット係数は 1 に近い
    rng = np.random.default_rng(0)
    points = np.concatenate([rng.normal(0, 0.1, (30, 2)),
                             rng.normal(5, 0.1, (30, 2))])
    centers = names["kmeans"](points, 2, rng, 50)
    labels = names["assign"](points, centers)
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert labels[0] != labels[30]
    score = names["silhouette"](points, labels, 2000, rng)
    assert 0.9 < score <= 1.0
    assert names["silhouette"](points, np.zeros(60, dtype=int), 2000,
                               rng) == 0.0
    jaccard = names["group_jaccard"](labels, 1 - labels, 2)
    assert jaccard.tolist() == [1.0, 1.0]

    # 符号化: 数の列は学習用で標準化して欠けた値は 0, 種類の列は値ごと.
    # 学習用に無い値は全て 0
    frame = pd.DataFrame({"spend": [1.0, 3.0, None],
                          "region": ["a", "b", "c"]})
    encoding = names["make_encoding"](frame.iloc[:2])
    assert encoding["numeric"]["spend"] == {"mean": 2.0, "std": 1.0}
    assert encoding["categories"]["region"] == ["a", "b"]
    features = names["encode_rows"](frame, encoding)
    assert features.tolist() == [[-1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
                                 [0.0, 0.0, 0.0]]

    # 行の分け方: 検証用は割合の数で, 学習用と重ならない
    cfg = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                         .read_text(encoding="utf-8"))
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.create(cfg)
    train, valid = names["split_rows"](100, cfg)
    assert len(valid) == 20 and len(train) == 80
    assert not set(train) & set(valid)
    with pytest.raises(ValueError):
        names["split_rows"](3, cfg)

    # 表の読み方: 決まった名前の群を作る列と行を見分ける列は, config を手で
    # 直しても止まる
    path = tmp_path / "named.csv"
    pd.DataFrame({"spend": [1.0, 2.0], "id": [3, 4], "split": ["a", "b"],
                  "customer": ["x", "y"]}).to_csv(path, index=False)
    plain = omegaconf.OmegaConf.merge(
        cfg, {"group": {"columns": ["spend"], "id": "customer"}}
    )
    frame, ids = names["read_rows"](path, plain)
    assert list(frame.columns) == ["spend"] and ids.tolist() == ["x", "y"]
    for change, words in (({"columns": ["spend", "id"]}, "群を作る列 id"),
                          ({"id": "split"}, "行を見分ける列 split")):
        changed = omegaconf.OmegaConf.merge(plain, {"group": change})
        with pytest.raises(ValueError, match=words):
            names["read_rows"](path, changed)


def test_group_training(tmp_path: Path) -> None:
    """
    まとめるの骨組みを書き換えずに仮のモデルで学び, シルエット係数で最良を
    選び, 群の安定性と群の要約, 群の割り当てを残すことと, 標本を変えて
    数える場合と, 大きさの合わない定義で出力先を作る前に止まることを確かめ
    る関数 (PyTorch と段階のライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 入れた表は data/ に写して読む
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    source = write_rows(tmp_path / "raw")
    problem = attach_data(group_problem(), str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert "data/customers.csv" in written
    done = run_script(project)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    runs = sorted((project / "outputs").glob("*/*/*/train.log"))
    assert len(runs) == 1
    run = runs[0].parent
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル (PlaceholderModel) で学習します" in log
    assert "まとめるもの: 顧客の群" in log
    assert "train 192 行, valid 48 行" in log
    assert "符号化した行の幅: 5" in log
    assert "推論の一式を作りません" in log
    assert not (run / "inference").exists()

    # 検証のたびのシルエット係数で最良を選ぶ
    with open(run / "history.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    values = [float(row["valid/silhouette"]) for row in rows]
    assert len(values) == 3 and all(-1 <= value <= 1 for value in values)
    assert f"ベストの valid/silhouette: {max(values)}" in done.stdout

    # 群の安定性: 列の特徴をそのまま分けた群と並べる
    table = json.loads((run / "evaluations" / "group_stability.json")
                       .read_text(encoding="utf-8"))
    assert table["split"] == "all"
    assert table["baseline"] == "列の特徴をそのまま k-means で分けた群"
    assert table["better"]["ari"] == "higher"
    row = table["rows"][0]
    assert (row["k"], row["method"], row["repeats"]) == (3, "seed", 3)
    for name in ("ari", "ari_min", "silhouette"):
        assert row[f"{name}_diff"] == pytest.approx(
            row[name] - row[f"{name}_baseline"]
        )
    assert row["ari_min"] <= row["ari"] <= 1
    assert row["ari_baseline"] > 0.9

    # 群の要約: 群ごとの行の数と割合と全体の行
    summary = json.loads((run / "evaluations" / "group_summary.json")
                         .read_text(encoding="utf-8"))
    groups = summary["rows"]
    assert [item["group"] for item in groups] == [0, 1, 2, "全体"]
    assert sum(item["rows"] for item in groups[:3]) == 240
    assert groups[-1]["rows"] == 240 and groups[-1]["share"] == 1.0
    assert all(0 <= item["stability"] <= 1 for item in groups[:3]
               if item["rows"])
    assert "%" in groups[-1]["region"]
    assert groups[-1]["spend"] > 0

    # 群の割り当て, 符号化と群の中心, 何の実験か
    with open(run / "group_assignments.csv", encoding="utf-8",
              newline="") as file:
        assigned = list(csv.DictReader(file))
    assert len(assigned) == 240 and set(assigned[0]) == {"customer", "group",
                                                         "split"}
    centers = json.loads((run / "group_centers.json")
                         .read_text(encoding="utf-8"))
    assert centers["k"] == 3 and len(centers["centers"]) == 3
    assert list(centers["encoding"]["numeric"]) == ["spend", "visits"]
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"]) == ("group", "cluster")

    # 群の要約の行は, ML-EDA の群の要約 (割り当てた群を labels で渡す) の
    # 行と, 安定性の他は同じ鍵, 順, 値
    from ml_eda import describe_groups
    described = describe_groups(
        project / "data" / "customers.csv",
        columns=["spend", "visits", "region"],
        labels=[int(item["group"]) for item in assigned]
    )["groups"]
    assert len(described) == len(groups)
    for mine, theirs in zip(groups, described):
        assert list(mine) == list(theirs)
        assert {key: value for key, value in mine.items()
                if key != "stability"} == pytest.approx(
            {key: value for key, value in theirs.items()
             if key != "stability"}
        )

    # 標本を変えて数える. 学習用の 192 行で最後のバッチが 1 行になる大きさ
    # でも学ぶ (config の差分だけ)
    sample = tmp_path / "sample"
    make_project(problem, sample)
    done = run_script(sample, ("group.stability=sample",
                               "data.batch_size=191"))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    found = sorted((sample / "outputs").glob(
        "*/*/*/evaluations/group_stability.json"
    ))
    assert json.loads(found[0].read_text(encoding="utf-8"))["rows"][0][
        "method"
    ] == "sample"

    # 大きさの合わない定義 (同梱の例のまま) は出力先を作る前に止まる
    spec_module = pytest.importorskip("ml_modelcontrol.spec")
    spec = json.loads(json.dumps(spec_module.load_example(
        "contrastive_encoder"
    )))
    wrong = tmp_path / "wrong"
    make_project(problem, wrong)
    place = Path(record_project(problem, wrong)["project"]["model_spec"])
    place.parent.mkdir(parents=True, exist_ok=True)
    place.write_text(yaml.safe_dump(spec, allow_unicode=True),
                     encoding="utf-8")
    done = run_script(wrong)
    assert done.returncode == 1
    assert "符号化した行の幅 5" in done.stdout
    assert not (wrong / "outputs").exists()

    # 大きさを合わせた定義で学び, 実験に定義が写る
    spec["inputs"]["x"]["shape"] = [8, 5]
    spec["nodes"]["fc1"]["params"]["in_features"] = 5
    place.write_text(yaml.safe_dump(spec, allow_unicode=True),
                     encoding="utf-8")
    done = run_script(wrong)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    runs = sorted((wrong / "outputs").glob("*/*/*/train.log"))
    assert "PlaceholderModel" not in runs[0].read_text(encoding="utf-8")
    assert (runs[0].parent / "model_spec.yaml").is_file()
    check_bundle(runs[0].parent, project / "data" / "customers.csv")
    check_bundle_skips(wrong, runs[0].parent, tmp_path / "skips")

    # 定義を置いて書き出し直すと, 仮のモデルの目印を書かない
    make_project(problem, wrong, overwrite=True)
    config = yaml.safe_load((wrong / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert "placeholder" not in config["model"]
    assert config["model"]["spec"] == "models/model.yaml"


def check_bundle(run: Path, dataset: Path) -> None:
    """
    定義で学んだ実験の中に群分けの一式 (版 9) ができ, 学習の表の割り当て
    直しと一式の推論の群と群の要約が, 実験の割り当てと群の要約と同じこと
    を確かめる関数 (ML-Inference が無ければ飛ばす)

    Parameters
    ----------
    run: Path
        定義で学んだ実験のディレクトリ
    dataset: Path
        学習に使った表

    Returns
    ----------
    None
    """
    # 一式は版 9 で, 学習の表の割り当て直しが全て同じ
    pytest.importorskip("ml_inference")
    from ml_inference import group_summary, predict
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "推論の一式 (まとめる): inference/" in log
    assert "割り当て直しの一致 100.0 %" in log
    bundle = run / "inference"
    info = yaml.safe_load((bundle / "inference.yaml")
                          .read_text(encoding="utf-8"))
    assert info["format_version"] == 9
    assert info["output"]["form"] == "cluster"
    group = info["group"]
    assert (group["k"], group["rows"], group["id"]) == (3, 240, "customer")
    assert group["columns"] == ["spend", "visits", "region"]
    assert group["check"] == {"rows": 240, "agreement": 1.0}

    # 一式の推論の群は実験の割り当てと同じで, 群の要約も同じ
    with open(run / "group_assignments.csv", encoding="utf-8",
              newline="") as file:
        assigned = {row["customer"]: int(row["group"])
                    for row in csv.DictReader(file)}
    prediction = predict(bundle, dataset)
    assert (prediction["status"] == "ok").all()
    assert dict(zip(prediction["customer"],
                    prediction["group"].astype(int))) == assigned
    summary = json.loads((run / "evaluations" / "group_summary.json")
                         .read_text(encoding="utf-8"))["rows"]
    found = group_summary(bundle, prediction)
    assert len(found) == len(summary)
    for mine, theirs in zip(found, summary):
        assert list(mine) == list(theirs)
        assert mine == pytest.approx(theirs)


def check_bundle_skips(project: Path, run: Path, root: Path) -> None:
    """
    書き出した train.py が, 仮のモデルと定義を写せなかった実験では一式を
    作らず, 埋め込みの出力の無い定義と成果物の欠けた実験では警告を残して
    止まらないことを確かめる関数

    Parameters
    ----------
    project: Path
        定義を置いたプロジェクトのディレクトリ
    run: Path
        定義で学んだ実験のディレクトリ
    root: Path
        写した実験を置く場所

    Returns
    ----------
    None
    """
    # 知らせと警告の文を残す記録
    from omegaconf import OmegaConf
    names = load_script(project)
    info: list = []
    warning: list = []
    experiment = SimpleNamespace(
        logger=SimpleNamespace(info=info.append, warning=warning.append),
        model_spec_path=None
    )
    cfg = OmegaConf.create({"experiment": {"name": "group_3"}})
    copied = root / "run"
    root.mkdir(parents=True)
    dataset = project / "data" / "customers.csv"

    # 仮のモデルと, 定義を写せなかった実験は作らない
    names["make_inference"](cfg, None, dataset, copied, experiment)
    assert "仮のモデル (PlaceholderModel)" in info[-1]
    spec = project / "models" / "model.yaml"
    names["make_inference"](cfg, spec, dataset, copied, experiment)
    assert "定義を実験に写せなかった" in info[-1]
    assert not copied.exists()

    # 埋め込みの出力の無い定義は, ML-Inference の誤りを警告に残す
    shutil.copytree(run, copied, ignore=shutil.ignore_patterns("inference"))
    written = yaml.safe_load((copied / "model_spec.yaml")
                             .read_text(encoding="utf-8"))
    written["outputs"].pop("embedding")
    (copied / "model_spec.yaml").write_text(
        yaml.safe_dump(written, allow_unicode=True), encoding="utf-8"
    )
    experiment.model_spec_path = str(copied / "model_spec.yaml")
    names["make_inference"](cfg, spec, dataset, copied, experiment)
    assert warning[-1].startswith(
        "定義か成果物が群分けの一式の約束に合わないので, 推論の一式を"
        "作れません: "
    )
    assert "embedding" in warning[-1]
    assert not (copied / "inference").exists()

    # 成果物の欠けた実験も, ML-Inference の誤りを警告に残す
    lacking = root / "lacking"
    shutil.copytree(run, lacking, ignore=shutil.ignore_patterns("inference"))
    (lacking / "group_centers.json").unlink()
    experiment.model_spec_path = str(lacking / "model_spec.yaml")
    names["make_inference"](cfg, spec, dataset, lacking, experiment)
    assert "group_centers.json" in warning[-1]
    assert not (lacking / "inference").exists()
