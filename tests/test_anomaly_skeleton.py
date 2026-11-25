import ast
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    ANOMALY_ASSIGNED, answer, apply_recipe, attach_data, check,
    example_problem, handoff, make_project, model_example, project_error,
    questions, recipe_from, record_project
)

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の表の列と, 正常の行と異常の行の数
COLUMNS = ["reading_id", "temp", "vibration", "current", "is_anomaly"]
NORMAL = 300
ANOMALIES = 12

# 骨組みを書き換えずに短く学ぶ config の差分
SHORT = ["trainer.epochs=3", "experiment.device=cpu"]


def anomaly_problem(source: str = "few", score: str = "autoencoder") -> dict:
    """
    見本の課題から, 異常の検知の問題の設定を作る関数

    Parameters
    ----------
    source: str = "few"
        正解の出どころ (few か none)
    score: str = "autoencoder"
        度合いの決め方

    Returns
    ----------
    problem: dict
        予測するで正解の種類が異常, 出力の形が値で, 列と件数の上限に
        答えた問題の設定 (データは入れていない)
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("解約の引き止め")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = answer(problem, "information.columns", COLUMNS)
    metric = "average_precision" if source == "few" else "loss"
    answers = [("role", "predict"), ("unit", "row"), ("output", "value"),
               ("target.source", source), ("method", "scratch"),
               ("target.kind", "anomaly"), ("what", "機械の異常の度合い"),
               ("item", "1 回の計測"), ("baseline", "決まった値を超えたら"),
               ("anomaly.id", "reading_id"),
               ("anomaly.columns", ["temp", "vibration", "current"]),
               ("anomaly.score", score), ("anomaly.capacity", 5)]
    if source == "few":
        answers.append(("anomaly.label", "is_anomaly"))
    if score == "autoencoder":
        answers += [("evaluation.metric", metric),
                    ("evaluation.link", "上位に異常が多いほど見逃しが減る")]
    for key, value in answers:
        problem = answer(problem, f"{PART}.{key}", value)

    return problem


def write_rows(root: Path, labeled: bool = True) -> Path:
    """
    正常の行と外れた少しの異常の行の, 合成のセンサの表の CSV を作る関数

    Parameters
    ----------
    root: Path
        書き出す場所
    labeled: bool = True
        異常の行にだけ印 (is_anomaly が 1. 正常の行は空) を付けるか

    Returns
    ----------
    path: Path
        CSV のファイル (reading_id, temp, vibration, current, is_anomaly)
    """
    # 正常の行は 3 つの列が連れ立って動き, 異常の行はその関係から外れる
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    rows = []
    for number in range(NORMAL + ANOMALIES):
        base = rng.normal(0, 1)
        broken = number >= NORMAL
        rows.append({
            "reading_id": f"r{number:04d}",
            "temp": 50 + 5 * base + rng.normal(0, 0.3),
            "vibration": 2 + base + rng.normal(0, 0.1)
            + (6 if broken else 0),
            "current": 10 - 2 * base + rng.normal(0, 0.2),
            "is_anomaly": 1 if broken and labeled else None,
        })
    root.mkdir(parents=True, exist_ok=True)
    path = root / "readings.csv"
    pd.DataFrame(rows).sample(frac=1.0, random_state=0).to_csv(path,
                                                               index=False)

    return path


def export_rows(problem: dict, project: Path) -> Path:
    """
    データを入れた問題の設定のレシピの雛形で, 骨組みの場所に異常の検知の表
    を書き出す関数

    Parameters
    ----------
    problem: dict
        データを入れた問題の設定
    project: Path
        書き出したプロジェクトのディレクトリ

    Returns
    ----------
    place: Path
        書き出したデータセット (config の data.dataset)
    """
    # record_project の場所 (train.py が読む場所) に書き出し
    eda = pytest.importorskip("ml_eda")
    start = recipe_from(problem)
    load = start["load"]
    data = eda.load_data(load["path"], loader=load["loader"],
                         **(load["options"] or {}))
    place = Path(record_project(problem, project)["project"]["dataset"])
    eda.export_dataset(data, start["recipe"], place)

    return place


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
        config の差分

    Returns
    ----------
    done: subprocess.CompletedProcess
        終わったプロセス (表示と終わりの番号)
    """
    # config の差分だけを渡す
    done = subprocess.run([sys.executable, "train.py", *extra],
                          cwd=project, capture_output=True, text=True,
                          timeout=600)

    return done


def only_run(project: Path) -> Path:
    """
    プロジェクトの 1 つだけの実験のディレクトリを返す関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # outputs/<実験の名前>/<日付>/<時刻>/
    runs = sorted((project / "outputs").glob("*/*/*/train.log"))
    assert len(runs) == 1
    run = runs[0].parent

    return run


def read_evaluation(run: Path, name: str) -> dict:
    """
    実験の評価の表を読む関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ
    name: str
        評価の名前

    Returns
    ----------
    table: dict
        evaluations/<名前>.json の中身
    """
    # ML-Found の save_evaluation の形
    table = json.loads((run / "evaluations" / f"{name}.json")
                       .read_text(encoding="utf-8"))

    return table


def test_anomaly_questions() -> None:
    """
    異常の検知の問いと選択肢と, 確かめの誤りを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正解が少しある自己符号化器は全て答えて誤りが無い
    problem = anomaly_problem()
    errors = [item for item in check(problem) if item["level"] == "error"]
    assert errors == []
    needed = {item["id"] for item in questions(problem) if item["applies"]}
    for key in ("anomaly.label", "anomaly.id", "anomaly.columns",
                "anomaly.score", "anomaly.capacity", "evaluation.metric"):
        assert f"{PART}.{key}" in needed, key
    assert "information.columns" in needed
    assert f"{PART}.task" not in needed and "action.decision" not in needed
    by_id = {item["id"]: item for item in questions(problem)}
    metrics = [item["value"]
               for item in by_id[f"{PART}.evaluation.metric"]["choices"]]
    assert metrics[:3] == ["loss", "average_precision", "precision_at_k"]
    kinds = [item["value"] for item in by_id[f"{PART}.target.kind"]["choices"]]
    assert "anomaly" in kinds
    assert ANOMALY_ASSIGNED == ("split", "score", "anomaly", "rank",
                                "column", "column_share")

    # 正解が無いと正解の列を問わず, 指標は損失だけ
    plain = anomaly_problem("none")
    needed = {item["id"] for item in questions(plain) if item["applies"]}
    assert f"{PART}.anomaly.label" not in needed
    by_id = {item["id"]: item for item in questions(plain)}
    assert [item["value"] for item in
            by_id[f"{PART}.evaluation.metric"]["choices"]][:-1] == ["loss"]
    wrong = answer(plain, f"{PART}.evaluation.metric", "average_precision")
    assert any(item["id"] == f"{PART}.evaluation.metric"
               for item in check(wrong) if item["level"] == "error")

    # 距離に基づく度合いは最良を選ぶ指標を問わない
    distance = anomaly_problem(score="distance")
    needed = {item["id"] for item in questions(distance) if item["applies"]}
    assert f"{PART}.evaluation.metric" not in needed
    assert [item for item in check(distance) if item["level"] == "error"] == []

    # 列と件数の上限と, 正解の出どころと出力の形の誤り
    for key, value, words in (
        ("anomaly.columns", ["temp", "is_anomaly"], "度合いに使う列にも"),
        ("anomaly.id", "is_anomaly", "異常の正解の列と同じ"),
        ("anomaly.capacity", 0, "1 以上の整数"),
        ("anomaly.capacity", 2.5, "1 以上の整数"),
        ("target.source", "column", "正解の出どころは"),
        ("output", "class", "出力の形は"),
    ):
        changed = answer(problem, f"{PART}.{key}", value)
        messages = [item["message"] for item in check(changed)
                    if item["level"] == "error"]
        assert any(words in message for message in messages), (key, value)
    stale = answer(problem, "information.columns", COLUMNS[:3] + COLUMNS[4:])
    assert any("current は使えるデータの列にありません" in item["message"]
               for item in check(stale))
    for name in ANOMALY_ASSIGNED:
        named = answer(answer(problem, "information.columns",
                              [*COLUMNS, name]), f"{PART}.anomaly.id", name)
        assert any("行ごとの度合いの表の列" in item["message"]
                   for item in check(named) if item["level"] == "error"), name
    rowid = answer(answer(problem, "information.columns",
                          [*COLUMNS, "row_id"]), f"{PART}.anomaly.id", None)
    assert any("row_id" in item["message"] for item in check(rowid)
               if item["level"] == "error")
    missing = answer(problem, f"{PART}.anomaly.label", None)
    assert any(item["id"] == f"{PART}.anomaly.label"
               for item in check(missing) if item["level"] == "error")
    shape = answer(problem, "information.columns", COLUMNS)
    shape["ml"]["parts"][0]["anomaly"]["score"] = "forest"
    assert any(item["id"] == f"{PART}.anomaly.score"
               for item in check(shape))


def test_anomaly_handoff() -> None:
    """
    異常の検知の受け渡しの値を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正解が少しある自己符号化器
    stages = handoff(anomaly_problem())
    assert stages["anomaly"] == {
        "score": "autoencoder", "label": "is_anomaly", "id": "reading_id",
        "columns": ["temp", "vibration", "current"], "capacity": 5, "k": 5,
    }
    data = stages["data"]
    assert (data["target"], data["task"], data["split"]) == (
        "is_anomaly", "anomaly", "random"
    )
    assert data["known"] == ["temp", "vibration", "current"]
    assert data["drop"] == []
    assert (data["id"], data["unlabeled"]) == ("reading_id", "normal")
    assert stages["model"]["example"] == "tabular_autoencoder"
    assert stages["model"]["outputs"] == 3
    assert (stages["training"]["monitor"], stages["training"]["mode"]) == (
        "valid/average_precision", "max"
    )
    inference = stages["inference"]
    assert inference["decision"] == {"type": "triage", "capacity": 5}
    assert inference["guard"] is None and inference["ready"] is None
    assert stages["group"] is None and stages["rank"] is None

    # 正解の無い距離に基づく度合い (使わない列は落とす)
    problem = anomaly_problem("none", "distance")
    problem = answer(problem, "information.columns", [*COLUMNS, "memo"])
    stages = handoff(problem)
    assert stages["anomaly"]["label"] is None
    assert stages["data"]["target"] is None
    assert stages["data"]["unlabeled"] is None
    assert stages["data"]["drop"] == ["is_anomaly", "memo"]
    assert stages["model"]["example"] is None
    assert model_example(problem) is None
    assert stages["training"] == {"metric": None, "monitor": None,
                                  "mode": None, "role": "predict",
                                  "target_kind": "anomaly"}

    # 列の予測の受け渡しは異常の検知の値を持たない
    assert handoff(example_problem("解約の引き止め"))["anomaly"] is None


def test_anomaly_recipe() -> None:
    """
    異常の検知のレシピの雛形と書き戻しを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 雛形: task: anomaly, 正解の列, 行の id, 正解の欠けた行は正常
    problem = answer(anomaly_problem(), "information.columns",
                     [*COLUMNS, "memo"])
    start = recipe_from(problem)
    recipe = start["recipe"]
    assert recipe["task"] == "anomaly" and recipe["target"] == "is_anomaly"
    assert recipe["id"] == "reading_id"
    assert recipe["steps"] == [
        {"op": "drop_columns", "columns": ["memo"]},
        {"op": "standardize", "columns": "auto"},
    ]
    assert recipe["split"] == {"method": "random", "valid": 0.15,
                               "test": 0.15, "seed": 0,
                               "unlabeled": "normal"}
    assert any("one_hot" in note for note in start["notes"])
    plain = recipe_from(anomaly_problem("none"))["recipe"]
    assert "target" not in plain and "unlabeled" not in plain["split"]

    # 書き戻し: レシピを持ち, 答えと違う列と分割は食い違い
    result = apply_recipe(problem, recipe)
    assert result["conflicts"] == [] and result["changes"] == []
    assert result["problem"]["data"]["recipe"] == recipe
    assert recipe_from(result["problem"])["recipe"] == recipe
    for change, words in (
        ({"task": "classification"}, "タスクは anomaly"),
        ({"target": "temp"}, "異常の正解の列"),
        ({"id": None}, "行の id の列"),
        ({"split": {"method": "time"}}, "random"),
    ):
        conflicts = apply_recipe(problem, {**recipe, **change})["conflicts"]
        assert any(words in item["message"] and item["level"] == "error"
                   for item in conflicts), change
    conflicts = apply_recipe(problem, {
        **recipe, "split": {"method": "random", "valid": 0}
    })["conflicts"]
    assert [item["level"] for item in conflicts] == ["warning"]


def test_anomaly_project(tmp_path: Path) -> None:
    """
    データを入れない異常の検知の問題の設定から骨組みを書き出し, config,
    README, スクリプトを確かめ, データセットが無ければ案内して止まることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 自己符号化器: 書き出したファイル, スクリプト, config, README
    problem = anomaly_problem()
    assert project_error(problem) is None
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert set(written) == {
        "problem.yaml", "README.md", "configs/default/train.yaml",
        "configs/experiment/train.yaml", "train.py",
    }
    script = (project / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'ROLE = "predict"' in script and 'OUTPUT = "value"' in script
    assert "precision_at_k" in script and "PlaceholderModel" in script
    assert all(len(line) <= 79 for line in script.splitlines())
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol", "inference"]
    assert config["trainer"]["monitor"] == "valid/average_precision"
    assert config["data"]["dataset"] == "outputs/dataset"
    assert config["data"]["source"] is None
    assert "LOADER = None" in script and "OPTIONS = {}" in script
    assert config["anomaly"] == {"score": "autoencoder", "k": 5,
                                 "capacity": 5}
    assert config["model"]["placeholder"] is True
    assert config["experiment"]["name"] == "anomaly_autoencoder"
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("outputs/dataset", "`temp`", "`is_anomaly`",
                  "`reading_id`", "tabular_autoencoder",
                  "anomaly_scorer.json", "anomaly_scores.csv",
                  "evaluations/anomaly_<分割>.json", "5 件"):
        assert words in readme, words
    place = record_project(problem, project)["project"]["dataset"]
    assert place == str((project / "outputs/dataset").resolve())

    # 距離: 学習とモデルの設定を書かない
    other = tmp_path / "distance"
    make_project(anomaly_problem("none", "distance"), other)
    config = yaml.safe_load((other / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "inference"]
    assert "trainer" not in config and "model" not in config
    assert config["anomaly"]["ridge"] == 0.001
    readme = (other / "README.md").read_text(encoding="utf-8")
    assert "モデルの定義を使わない" in readme
    assert "異常の正解は無い" in readme

    # 書き出せない出力の形
    wrong = answer(problem, f"{PART}.output", "distribution")
    assert project_error(wrong) is not None

    # データセットが無ければ, 書き出しを案内して止まる
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 1
    assert "異常の検知の表を書き出してください" in done.stdout


def test_anomaly_training(tmp_path: Path) -> None:
    """
    正解が少しある異常の検知の骨組みを, 自己符号化器の仮のモデルと距離に
    基づく度合いで書き換えずに動かし, 上位 K 件の適合率と PR-AUC, 行ごとの
    度合い, 度合いの決め方の成果物を確かめる関数 (PyTorch と段階のライブ
    ラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 入れた表をレシピの雛形で書き出し, 仮のモデルで 3 エポック
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    source = write_rows(tmp_path / "raw")
    problem = attach_data(anomaly_problem(), str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    assert "data/readings.csv" in make_project(problem, project)
    export_rows(problem, project)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル (PlaceholderModel) で学習します" in log
    assert "推論の一式を作りません" in log
    assert not (run / "inference").exists()
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["data"]["source"] == "data/readings.csv"
    assert "異常の度合い: 機械の異常の度合い" in log
    assert "学習に使う列: 3" in log

    # 検証のたびの PR-AUC で最良を選ぶ
    with open(run / "history.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    values = [float(row["valid/average_precision"]) for row in rows]
    assert len(values) == 3 and all(0 <= value <= 1 for value in values)
    assert "valid/precision_at_k" in rows[0]

    # 検証用とテスト用の上位 K 件の適合率と PR-AUC
    for split in ("valid", "test"):
        table = read_evaluation(run, f"anomaly_{split}")
        assert table["split"] == split
        assert table["better"]["average_precision"] == "higher"
        row = table["rows"][0]
        assert row["k"] == 5 and row["anomalies"] >= 1
        assert 0 <= row["precision_at_k"] <= 1
        assert 0 <= row["average_precision"] <= 1

    # 行ごとの度合いと, 度合いの決め方
    with open(run / "anomaly_scores.csv", encoding="utf-8",
              newline="") as file:
        scores = list(csv.DictReader(file))
    assert len(scores) == NORMAL + ANOMALIES
    assert list(scores[0]) == ["reading_id", "split", "score", "anomaly"]
    assert sum(row["anomaly"] == "1" for row in scores) == ANOMALIES
    assert all(row["anomaly"] == "0" for row in scores
               if row["split"] == "train")
    scorer = json.loads((run / "anomaly_scorer.json")
                        .read_text(encoding="utf-8"))
    assert scorer["format_version"] == 1
    assert scorer["score"] == "autoencoder"
    assert (scorer["recipe"], scorer["scores"]) == (
        "anomaly_recipe.yaml", "anomaly_scores.csv"
    )
    recipe = yaml.safe_load((run / "anomaly_recipe.yaml")
                            .read_text(encoding="utf-8"))
    assert recipe["task"] == "anomaly" and recipe["id"] == "reading_id"
    assert scorer["features"] == ["temp", "vibration", "current"]
    assert (scorer["id"], scorer["capacity"], scorer["k"]) == (
        "reading_id", 5, 5
    )
    assert scorer["model"] == {"input": "x", "output": "recon"}
    assert scorer["distance"] is None
    assert scorer["normal"]["p50"] <= scorer["normal"]["p99"]
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"]) == ("predict", "value")
    assert (run / "final_model.pt").is_file()

    # 距離に基づく度合い: 異常の行が上位に来る
    other = tmp_path / "distance"
    distance = attach_data(anomaly_problem(score="distance"), str(source),
                           copy_to=str(tmp_path / "copies2"))
    make_project(distance, other)
    export_rows(distance, other)
    done = run_script(other, ("experiment.device=cpu",))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(other)
    for split in ("valid", "test"):
        row = read_evaluation(run, f"anomaly_{split}")["rows"][0]
        assert row["average_precision"] > 0.9, row
    scorer = json.loads((run / "anomaly_scorer.json")
                        .read_text(encoding="utf-8"))
    assert scorer["model"] is None
    assert len(scorer["distance"]["center"]) == 3
    assert len(scorer["distance"]["precision"]) == 3
    assert not (run / "final_model.pt").exists()

    # 実験の中の推論の一式の度合いが, 行ごとの度合いの表と同じ
    check_inference(run, source, NORMAL + ANOMALIES)


def test_anomaly_unlabeled(tmp_path: Path) -> None:
    """
    正解の無い異常の検知の骨組みが, 度合いの分布と上位の行の要約を書く
    ことと, ML-ModelControl の同梱の例の大きさを合わせた定義で学ぶことを
    確かめる関数 (PyTorch と段階のライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 正解の無い表を, 損失で選ぶ自己符号化器で
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    source = write_rows(tmp_path / "raw", labeled=False)
    problem = attach_data(anomaly_problem("none"), str(source),
                          copy_to=str(tmp_path / "copies"))
    problem = answer(problem, "information.columns", COLUMNS)
    project = tmp_path / "project"
    make_project(problem, project)
    export_rows(problem, project)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)
    assert not (run / "evaluations" / "anomaly_test.json").exists()
    for split in ("valid", "test"):
        curve = read_evaluation(run, f"anomaly_scores_{split}")
        assert curve["kind"] == "curve" and curve["x"] == "quantile"
        points = curve["rows"]
        assert len(points) == 21 and points[0]["quantile"] == 0
        assert all(a["score"] <= b["score"]
                   for a, b in zip(points, points[1:]))
        top = read_evaluation(run, f"anomaly_top_{split}")["rows"]
        assert [item["rank"] for item in top] == [1, 2, 3, 4, 5]
        assert all(a["score"] >= b["score"] for a, b in zip(top, top[1:]))
        assert all(item["column"] in ("temp", "vibration", "current")
                   for item in top)
        assert all(item["reading_id"].startswith("r") for item in top)
    with open(run / "anomaly_scores.csv", encoding="utf-8",
              newline="") as file:
        assert all(row["anomaly"] == "" for row in csv.DictReader(file))

    # 正解の要る指標を config で選ぶと, 出力先を作る前に止まる
    done = run_script(project, (*SHORT, "trainer.monitor=valid/precision_at_k",
                                "trainer.mode=max"))
    assert done.returncode == 1
    assert "検証用に異常の行が無い" in done.stdout

    # 列の数に合わせた同梱の例の定義で学ぶ
    control = pytest.importorskip("ml_modelcontrol")
    spec = control.load_example("tabular_autoencoder")
    spec["inputs"]["x"]["shape"][1] = 3
    spec["nodes"]["enc1"]["params"]["in_features"] = 3
    spec["nodes"]["dec2"]["params"]["out_features"] = 3
    shutil = pytest.importorskip("shutil")
    shutil.rmtree(project / "outputs" / "anomaly_autoencoder")
    (project / "models" / "model.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "PlaceholderModel" not in log
    assert (run / "model_spec.yaml").is_file()

    # 定義のモデルの自己符号化器は, 学習の後に推論の一式を作る
    check_inference(run, source, NORMAL + ANOMALIES)


def check_inference(run: Path, source: Path, rows: int) -> None:
    """
    推論の一式の度合いが実験の度合いと同じかを確かめる関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ
    source: Path
        元の表の CSV
    rows: int
        元の表の行の数

    Returns
    ----------
    None
    """
    # 一式づくりの比べと, 元の表を一式で推論した度合い
    inference = pytest.importorskip("ml_inference")
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "推論の一式 (異常の度合い): inference/ (度合いの一致 100.0 %" in log
    bundle = inference.load_bundle(run / "inference")
    assert bundle.info["format_version"] == 11
    assert bundle.info["decision"] == {"type": "triage", "capacity": 5}
    assert bundle.info["anomaly"]["check"]["rows"] == rows
    pd = pytest.importorskip("pandas")
    frame = inference.predict(bundle, source)
    scores = pd.read_csv(run / "anomaly_scores.csv",
                         dtype={"reading_id": str})
    found = frame.set_index("reading_id")["score"]
    recorded = scores.set_index("reading_id")["score"]
    gap = (found.loc[recorded.index] - recorded).abs()
    assert float(gap.max()) <= 1e-6 + 1e-4 * float(recorded.abs().max())
    result = inference.infer(bundle, source)
    assert int(result["frame"]["action"].sum()) == 5


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
    names: dict = {"__name__": "anomaly_train",
                   "__file__": str(project / "train.py")}
    code = (project / "train.py").read_text(encoding="utf-8")
    exec(compile(code, str(project / "train.py"), "exec"), names)

    return names


def test_anomaly_functions(tmp_path: Path) -> None:
    """
    書き出した train.py の, 距離に基づく度合いの式と ridge の境界と, 上位の
    行の要約の列を確かめる関数 (PyTorch と段階のライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 手で書いた (x - 中心) P (x - 中心) / 列の数 との一致
    np = pytest.importorskip("numpy")
    project = tmp_path / "project"
    make_project(anomaly_problem(score="distance"), project)
    names = load_script(project)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    distance = names["fit_distance"](x, 0.01)
    points = x.astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False) + np.eye(3) * 0.01
    precision = np.linalg.inv(covariance)
    assert np.allclose(distance["center"], center)
    assert np.allclose(distance["precision"], precision)
    expected = np.array([(row - center) @ precision @ (row - center) / 3
                         for row in points])
    scores = names["distance_parts"](x, distance).mean(axis=1)
    assert np.allclose(scores, expected)

    # 散らばりの無い列と, 列の数が行の数より多い表でも有限
    flat = x.copy()
    flat[:, 1] = 5.0
    wide = rng.normal(size=(4, 10)).astype(np.float32)
    for table in (flat, wide):
        fitted = names["fit_distance"](table, 0.001)
        parts = names["distance_parts"](table, fitted)
        assert np.isfinite(parts).all()
        assert (parts.sum(axis=1) >= -1e-6).all()

    # 上位の行の要約は度合いの高い順で, 列の名前が重ならない
    part = {"ids": np.array(["a", "b", "c"])}
    rows = names["top_rows"](part, np.array([0.1, 0.9, 0.5]),
                             np.array([[1.0, 0.0], [0.2, 0.8], [0.5, 0.5]]),
                             "reading_id", ["temp", "vibration"], 2)
    assert [row["reading_id"] for row in rows] == ["b", "c"]
    assert rows[0] == {"rank": 1, "reading_id": "b", "score": 0.9,
                       "column": "vibration",
                       "column_share": pytest.approx(0.8)}
    assert set(names["ASSIGNED_COLUMNS"]) == set(ANOMALY_ASSIGNED)
