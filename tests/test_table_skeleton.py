import ast
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import yaml

from ml_quest import (
    answer, apply_recipe, attach_data, check, example_problem, handoff,
    make_project, project_error, recipe_from, record_project
)
from ml_quest.data import TABLE_LOADERS
from ml_quest.handoff import model_example

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の表の行の数
ROWS = 120

# 学習と推論の一式まで通すのに要るもの (CI の train のジョブで入れる)
TRAINING = ("torch", "ml_found", "ml_modelcontrol", "ml_inference")

# 骨組みを書き換えずに短く学ぶ config の差分 (少ない時刻と繰り返し)
SHORT = ["trainer.epochs=2", "experiment.device=cpu",
         "diffusion.train_steps=50", "diffusion.sample_steps=5"]

# 合成の表の列 (数の列 2 つ, 整数の列 1 つ, 種類の列 2 つ) と, 符号化した
# 行の幅 (数の列 3 と, 種類の値 3 と 2)
COLUMNS = ["age", "income", "visits", "city", "member"]
WIDTH = 8


def table_problem() -> dict:
    """
    見本の課題から, 表の生成の受け持ちの問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        作るで正解の種類と出力の形が表, 指標が列ごとの分布の近さの問題の
        設定 (データは入れていない)
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    for key, value in [("role", "generate"), ("unit", "row"),
                       ("output", "table"), ("target.source", "self"),
                       ("target.kind", "table"), ("method", "scratch"),
                       ("what", "顧客の表に似た行"), ("item", "顧客の 1 行"),
                       ("baseline", "手で作る"),
                       ("evaluation.metric", "column_distance"),
                       ("evaluation.link", "近いほど元の表の代わりに使える")]:
        problem = answer(problem, f"{PART}.{key}", value)

    return problem


def write_table(root: Path) -> Path:
    """
    数と種類の列を持つ合成の表の CSV を作る関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        CSV のファイル
    """
    # 決まった乱数で, 正規分布, 対数正規分布, 整数, 偏った種類の列
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "age": rng.normal(40, 10, ROWS).round(1),
        "income": rng.lognormal(10, 0.5, ROWS),
        "visits": rng.integers(0, 10, ROWS),
        "city": rng.choice(["a", "b", "c"], ROWS, p=[0.5, 0.3, 0.2]),
        "member": rng.choice(["yes", "no"], ROWS),
    })
    root.mkdir(parents=True, exist_ok=True)
    path = root / "customers.csv"
    frame.to_csv(path, index=False)

    return path


def attached(root: Path) -> dict:
    """
    表を入れた表の生成の問題の設定を作る関数

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
    source = write_table(root / "raw")
    problem = attach_data(table_problem(), str(source),
                          copy_to=str(root / "copies"))

    return problem


def export(problem: dict, project: Path) -> Path:
    """
    レシピの雛形で, 骨組みの場所にデータセットを書き出す関数

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


def denoiser_spec() -> dict:
    """
    ML-ModelControl の同梱の例 diffusion_denoiser の幅を, 符号化した行の幅
    に合わせた定義を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    spec: dict
        雑音を加えた行と時刻から雑音を当てる全結合の定義
    """
    # 同梱の例の入力と出力の幅を書き換え
    spec_module = pytest.importorskip("ml_modelcontrol.spec")
    spec = spec_module.load_example("diffusion_denoiser")
    spec = json.loads(json.dumps(spec))
    spec["inputs"]["x"]["shape"] = [8, WIDTH]
    spec["nodes"]["x_fc"]["params"]["in_features"] = WIDTH
    spec["nodes"]["head"]["params"]["out_features"] = WIDTH

    return spec


def train(project: Path, extra: tuple = ()) -> tuple[Path, str]:
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
    run: Path
        実験のディレクトリ
    stdout: str
        学習のスクリプトの表示
    """
    # config の差分だけを渡す
    done = subprocess.run([sys.executable, "train.py", *SHORT, *extra],
                          cwd=project, capture_output=True, text=True,
                          timeout=600)
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


def test_table_handoff() -> None:
    """
    表の生成の問題の設定が渡せ, 分割は無作為で, 分布の近さの監視になり,
    モデルの作成の雛形が拡散の例になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 誤りが無く, 書き出せる. 監視は valid/column_distance
    problem = table_problem()
    assert [item for item in check(problem) if item["level"] == "error"] == []
    assert project_error(problem) is None
    stages = handoff(problem)
    assert stages["data"]["split"] == "random"
    assert stages["data"]["target"] is None and stages["data"]["task"] is None
    assert (stages["training"]["monitor"], stages["training"]["mode"]) == (
        "valid/column_distance", "min"
    )
    assert stages["model"]["example"] == "diffusion_denoiser"
    assert model_example(problem) == "diffusion_denoiser"
    assert stages["inference"]["decision"] is None

    # 損失でも書き出せる. 分布の近さは画像の生成には合わない
    loss = answer(problem, f"{PART}.evaluation.metric", "loss")
    assert project_error(loss) is None
    assert handoff(loss)["training"]["monitor"] == "valid/loss"
    image = answer(problem, f"{PART}.target.kind", "image")
    image = answer(image, f"{PART}.output", "image")
    assert f"{PART}.evaluation.metric" in [
        item["id"] for item in check(image) if item["level"] == "error"
    ]


def test_table_errors(tmp_path: Path) -> None:
    """
    表の生成で書き出せない場合が, make_project と同じ文になることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 表でないデータ, 表でない出力の形, 文の列を読む表
    images = table_problem()
    images["data"] = {"source": str(tmp_path / "images"),
                      "loader": "image_folder"}
    cases = [(images, "表として読んでいません")]
    texts = table_problem()
    texts["data"] = {"source": str(tmp_path / "a.csv"), "loader": "csv",
                     "options": {"text_column": "body"}}
    cases.append((texts, "表として読んでいません"))
    shape = answer(table_problem(), f"{PART}.output", "shape")
    cases.append((shape, "出力の形が「表 (合成のデータ)」"))
    for problem, words in cases:
        message = project_error(problem)
        assert message is not None and words in message, message
        with pytest.raises(ValueError) as raised:
            make_project(problem, tmp_path / "project")
        assert str(raised.value) == message
    assert not (tmp_path / "project").exists()


def test_table_recipe() -> None:
    """
    表の生成のレシピの雛形が目的の列の無い無作為の分割で, 書き戻すと
    そのまま返り, 食い違いを知らせることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 雛形 (データを入れていないので読み方は無い)
    problem = table_problem()
    start = recipe_from(problem)
    assert start["load"] is None and start["export"] is None
    assert start["recipe"] == {
        "format_version": 1, "task": "none", "steps": [],
        "split": {"method": "random", "valid": 0.15, "test": 0.15,
                  "seed": 0},
    }
    assert any("drop_columns" in note for note in start["notes"])
    assert not any("resize" in note for note in start["notes"])

    # 書き戻すとそのまま返り, 受け渡しの分割にも出る
    recipe = {**start["recipe"],
              "steps": [{"op": "drop_columns", "columns": ["id"]}]}
    result = apply_recipe(problem, recipe)
    assert result["changes"] == [] and result["conflicts"] == []
    assert recipe_from(result["problem"])["recipe"] == recipe

    # 目的の列のあるレシピは食い違い
    result = apply_recipe(problem, {"target": "city",
                                    "task": "classification"})
    assert "error" in [item["level"] for item in result["conflicts"]]


def test_table_loaders() -> None:
    """
    写して持つ表の読み込みの名前が, ML-EDA の登録簿で kind が表のものと
    同じことを確かめる関数 (ML-EDA が無ければ飛ばす)

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # ML-EDA の登録簿
    eda = pytest.importorskip("ml_eda")
    names = [name for name, loader in eda.LOADERS.items()
             if getattr(loader, "kind", None) == "table"]
    assert sorted(names) == sorted(TABLE_LOADERS)


def test_table_project(tmp_path: Path) -> None:
    """
    表を入れた表の生成の問題の設定から骨組みを書き出し, 案内どおりに
    データセットを書き出せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出したファイル (表は data/ に写す)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert set(written) == {
        "problem.yaml", "README.md", "configs/default/train.yaml",
        "configs/experiment/train.yaml", "train.py", "data/customers.csv",
    }
    script = (project / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'ROLE = "generate"' in script and 'OUTPUT = "table"' in script
    assert 'PREDICT = "顧客の表に似た行"' in script
    assert 'TARGET_KIND = "table"' in script
    assert 'ORIGIN_TYPE = "measurement"' in script
    assert "PlaceholderModel" in script

    # config: 要る機能, 監視, EMA, データセットとモデルの定義の場所
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol", "inference"]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/column_distance", "min"
    )
    assert config["trainer"]["ema"] == {"decay": 0.995}
    assert config["data"]["dataset"] == "outputs/dataset"
    assert config["model"]["spec"] == "models/model.yaml"
    assert config["model"]["placeholder"] is True
    assert config["diffusion"]["train_steps"] == 1000

    # README: データの分析, モデルの入力, 分布の近さ, 雑音だけの一式
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("task: none", "data/customers.csv", "drop_columns",
                  "int64", "diffusion_denoiser", "KS 統計量", "全変動距離",
                  "Wasserstein", "column_distance_test.json", "版 7",
                  "`table`", "table_columns.json"):
        assert words in readme, words

    # 案内どおりの書き出しで, 目的の列の無い表のデータセット
    place = export(problem, project)
    info = yaml.safe_load((place / "dataset.yaml").read_text(encoding="utf-8"))
    assert info["kind"] == "table" and info["task"] == "none"
    assert "valid" in info["splits"]


def test_table_placeholder(tmp_path: Path) -> None:
    """
    モデルの定義が無いうちは仮のモデルで学び, 分布の近さで最良を選び,
    検証用とテスト用の評価と見本を残し, 一式を作らないことを確かめる関数
    (PyTorch と ML-Found が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 骨組みとデータセットだけで学習
    for name in ("torch", "ml_found"):
        pytest.importorskip(name)
    problem = attached(tmp_path)
    project = tmp_path / "project"
    make_project(problem, project)
    export(problem, project)
    run, stdout = train(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル (PlaceholderModel) で学習します" in log
    assert "推論の一式を作りません" in log
    assert f"符号化した行の幅: {WIDTH}" in log
    assert "作るもの: 顧客の表に似た行" in log
    assert not (run / "inference").exists()

    # 検証のたびの分布の近さで最良を選ぶ
    rows = history_rows(run)
    values = [float(row["valid/column_distance"]) for row in rows]
    assert len(values) == 2 and all(0 <= value <= 1 for value in values)
    assert f"ベストの valid/column_distance: {min(values)}" in stdout

    # 検証用とテスト用の評価: 列ごとの行と平均の行, 向き
    for split in ("valid", "test"):
        table = json.loads((run / "evaluations"
                            / f"column_distance_{split}.json")
                           .read_text(encoding="utf-8"))
        assert table["split"] == split
        assert table["better"] == {"distance": "lower",
                                   "wasserstein": "lower"}
        names = [row["column"] for row in table["rows"]]
        assert names == [*COLUMNS, "(全ての列の平均)"]
        for row in table["rows"]:
            assert 0 <= row["distance"] <= 1
            assert 0 <= row["train_distance"] <= 1
            assert (row["wasserstein"] is None) == (
                row["type"] != "number"
            )
    valid = json.loads((run / "evaluations" / "column_distance_valid.json")
                       .read_text(encoding="utf-8"))
    assert valid["rows"][-1]["distance"] == pytest.approx(min(values))

    # 見本: 作った表と本物の表の CSV, 列の符号化
    made = (run / "samples" / "generated" / "final.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert made[0].split(",") == COLUMNS and len(made) == 101
    assert (run / "samples" / "valid_real" / "final.csv").is_file()
    columns = json.loads((run / "table_columns.json")
                         .read_text(encoding="utf-8"))["columns"]
    assert [column["type"] for column in columns] == [
        "number", "number", "number", "category", "category"
    ]
    assert columns[2]["integer"] is True
    assert columns[3]["categories"] == ["a", "b", "c"]


def test_table_training(tmp_path: Path) -> None:
    """
    表の生成の骨組みを書き換えずに同梱の例の幅を合わせた定義で学び, 雑音
    だけの推論の一式の返す表が見本の表とほぼ同じになることを確かめる関数
    (PyTorch と段階のライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出し, データセット, 定義 (record_project の場所)
    for name in TRAINING:
        pytest.importorskip(name)
    import numpy as np
    import pandas as pd
    from ml_inference import load_bundle, predict
    problem = attached(tmp_path)
    project = tmp_path / "project"
    make_project(problem, project)
    export(problem, project)
    place = Path(record_project(problem, project)["project"]["model_spec"])
    place.parent.mkdir(parents=True, exist_ok=True)
    place.write_text(yaml.safe_dump(denoiser_spec(), allow_unicode=True),
                     encoding="utf-8")
    run, _ = train(project)

    # 何の実験か (正解は学んだ表で出どころは測った値, データセットは
    # 書き出した場所と名前)
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"], task["what"]) == (
        "generate", "table", "顧客の表に似た行"
    )
    name = yaml.safe_load((project / "outputs" / "dataset" / "dataset.yaml")
                          .read_text(encoding="utf-8"))["name"]
    assert task["target"] == {
        "kind": "table",
        "origin": {"type": "measurement", "name": None, "settings": None},
    }
    assert task["dataset"] == {"path": "outputs/dataset", "name": name,
                               "format_version": None}
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "正解は測った値" in log and "出どころは分からない" not in log

    # 雑音だけの一式 (版 7): 学習と同じ雑音の予定の ddim で, 列の符号化を
    # 写した表の出力
    bundle_dir = run / "inference"
    info = yaml.safe_load((bundle_dir / "inference.yaml")
                          .read_text(encoding="utf-8"))
    columns = json.loads((run / "table_columns.json")
                         .read_text(encoding="utf-8"))["columns"]
    assert info["format_version"] == 7 and info["dataset"] is None
    assert info["role"] == "generate"
    assert info["output"] == {"form": "table", "shape": [WIDTH],
                              "columns": columns}
    assert info["inputs"] == {"x": {"source": "state"},
                              "t": {"source": "time"}}
    assert info["procedure"]["steps"] == 5
    assert info["procedure"]["options"]["train_steps"] == 50

    # 一式の表は, 同じ種の見本の表とほぼ同じ
    frame = predict(load_bundle(bundle_dir), {"count": 100, "seed": 0})
    assert list(frame.columns) == ["sample", "seed", "status", "reason",
                                   *COLUMNS]
    assert list(frame["status"]) == ["ok"] * 100
    made = pd.read_csv(run / "samples" / "generated" / "final.csv")
    for column in columns:
        name = column["name"]
        if column["type"] == "number" and not column["integer"]:
            assert np.allclose(frame[name].astype(float), made[name],
                               atol=1e-3)
            continue
        same = frame[name].astype(str).to_numpy() == made[name].astype(
            str
        ).to_numpy()
        assert same.mean() > 0.95, name


def script_module(root: Path) -> ModuleType:
    """
    書き出した train.py を, 純粋な関数を呼ぶためにモジュールとして読む関数
    (PyTorch, ML-EDA, ML-Found が無ければ飛ばす)

    Parameters
    ----------
    root: Path
        骨組みを書き出す場所

    Returns
    ----------
    module: ModuleType
        読んだ train.py (main は呼ばない)
    """
    # 読み込みに要るライブラリ
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    make_project(table_problem(), root)
    spec = importlib.util.spec_from_file_location(
        "table_train", root / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_table_functions(tmp_path: Path) -> None:
    """
    骨組みの列の符号化と分布の近さの関数を, 小さな表と手計算で確かめる関数
    (PyTorch, ML-EDA, ML-Found が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 標準偏差が 0 の列は 1 で割り, 整数の列は戻すときに丸める
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    module = script_module(tmp_path / "project")
    frame = pd.DataFrame({"same": [5.0, 5.0, 5.0], "count": [1, 2, 3],
                          "city": ["b", "a", "b"]})
    columns = module.table_columns(frame)
    assert columns[0]["std"] == 1.0 and columns[0]["mean"] == 5.0
    assert columns[1]["integer"] is True
    assert columns[2]["categories"] == ["a", "b"]
    assert module.encoded_width(columns) == 4
    back = module.decode_table(module.encode_table(frame, columns), columns)
    assert back.to_dict("list") == frame.to_dict("list")

    # 学習用に無い種類の値は全て 0 に符号化する
    unseen = module.encode_table(
        pd.DataFrame({"same": [5.0], "count": [1], "city": ["z"]}), columns
    )
    assert unseen[0, 2:].tolist() == [0.0, 0.0]

    # 推論の一式の予測の表の列と重なる名前の列は止まる
    for name in ("status", "p:a", "q:b", "age (入力)"):
        with pytest.raises(ValueError, match="drop_columns"):
            module.table_columns(pd.DataFrame({name: [1.0], "ok": [2.0]}))

    # 種類の値が上限を超える列は止まる
    many = pd.DataFrame({"id": [f"u{i}" for i in range(101)]})
    with pytest.raises(ValueError, match="drop_columns"):
        module.table_columns(many)

    # 欠損のある列を分割をまたいで集める
    frames = {"train": pd.DataFrame({"a": [1.0, None], "b": ["x", "y"]}),
              "valid": pd.DataFrame({"a": [1.0, 2.0], "b": ["x", None]})}
    assert module.missing_columns(frames) == ["a", "b"]
    assert module.missing_columns({"train": frame}) == []

    # 検証用だけにある種類の値は, その値のまま種類として数える
    real = pd.Series(["a", "a", "z", "z"])
    made = pd.Series(["a", "a", "a", "a"])
    assert module.category_distance(real, made) == pytest.approx(0.5)
    assert module.category_distance(real, real) == 0.0

    # KS と Wasserstein 距離の手計算 ({0, 1} と {2, 3} は KS 1, 距離 2)
    ks, wasserstein = module.number_distances(
        np.array([0.0, 1.0]), np.array([2.0, 3.0]), 2.0
    )
    assert ks == pytest.approx(1.0)
    assert wasserstein == pytest.approx(1.0)
    ks, wasserstein = module.number_distances(
        np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0]), 1.0
    )
    assert ks == pytest.approx(0.25)
    assert wasserstein == pytest.approx(0.5)
