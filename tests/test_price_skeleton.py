import ast
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, attach_data, make_project, project_error, recipe_from,
    record_project
)

# 受け持ちの場所
PART = "ml.parts.0"

# 合成の値段を変えた記録の表の商品の数と, 商品ごとの記録の数
PRODUCTS = 30
RECORDS = 50

# 骨組みを書き換えずに短く学ぶ config の差分
SHORT = ["trainer.epochs=8", "optimizer.params.lr=0.01",
         "experiment.device=cpu"]

# 値段を変える判断の問題の設定の作り方 (tests/test_price.py と同じ)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_price import DECISION, price_problem  # noqa: E402


def write_sales(root: Path) -> Path:
    """
    特徴で弾力性の違う商品の値段を無作為に変えた, 合成の売れ行きの記録の
    CSV を作る関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        CSV のファイル (record_id, product_id, category, weekday, price,
        units, unit_cost, leftover)
    """
    # 種類 0 は値段で売れた数が大きく変わり (弾力性 4), 種類 1 はほぼ
    # 変わらない (弾力性 0.3). 値段は商品の基の値段の 0.8 から 1.2 倍
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    rows = []
    for product in range(PRODUCTS):
        category = product % 2
        base = 100.0
        power = 4.0 if category == 0 else 0.3
        for record in range(RECORDS):
            price = base * (1 + 0.1 * int(rng.integers(-2, 3)))
            weekday = int(rng.integers(0, 7))
            mean = 20 * (price / base) ** (-power) * (1 + 0.05 * weekday)
            rows.append({
                "record_id": f"r{product:02d}{record:02d}",
                "product_id": f"p{product:02d}", "category": category,
                "weekday": weekday, "price": price,
                "units": int(rng.poisson(mean)), "unit_cost": 60.0,
                "leftover": int(rng.integers(0, 3)),
            })
    root.mkdir(parents=True, exist_ok=True)
    path = root / "sales.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def export_sales(problem: dict, project: Path) -> Path:
    """
    データを入れた問題の設定のレシピの雛形で, 骨組みの場所に値段を変えた
    記録の表を書き出す関数

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
    # config の差分の引数
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


def test_price_project(tmp_path: Path) -> None:
    """
    データを入れない値段を変える判断の問題の設定から骨組みを書き出し,
    config, README, スクリプトを確かめ, データセットが無ければ案内して
    止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 書き出したファイル, スクリプト, config, README
    problem = price_problem()
    assert project_error(problem) is None
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert set(written) == {
        "problem.yaml", "README.md", "configs/default/train.yaml",
        "configs/experiment/train.yaml", "train.py",
    }
    script = (project / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'ROLE = "effect"' in script and 'OUTPUT = "effect"' in script
    assert "profit_lift" in script and "price_elasticity" in script
    assert "PlaceholderModel" in script
    assert all(len(line) <= 79 for line in script.splitlines())
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol", "inference"]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/mae", "min"
    )
    assert config["data"] == {"dataset": "outputs/dataset", "source": None,
                              "batch_size": 128}
    assert config["price"] == {"candidates": [-0.1, 0, 0.1, 0.2],
                               "max_change": 0.1}
    assert config["model"]["placeholder"] is True
    assert config["experiment"]["name"] == "units_price"
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("outputs/dataset", "`price`", "`units`", "`unit_cost`",
                  "`product_id`", "`category`", "tabular_price",
                  "price_estimator.json", "price_choices.csv",
                  "evaluations/price_<分割>.json", "stratified",
                  "中央値", "price_inputs", "inference/",
                  "ML-Inference を入れる"):
        assert words in readme, words

    # 損得と制約の直すところを残した場合の空の候補
    broken = answer(problem, "action.decision",
                    {**DECISION, "max_change": -1})
    with pytest.raises(ValueError, match="accept_errors"):
        make_project(broken, tmp_path / "broken")
    make_project(broken, tmp_path / "broken", accept_errors=True)
    config = yaml.safe_load((tmp_path / "broken" / "configs" / "default"
                             / "train.yaml").read_text(encoding="utf-8"))
    assert config["price"] == {"candidates": None, "max_change": None}

    # 書き出せない出力の形
    wrong = answer(problem, f"{PART}.output", "value")
    assert project_error(wrong) is not None

    # データセットが無いときの, 書き出しの案内と停止
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 1
    assert "値段を変えた記録の表を書き出してください" in done.stdout


def test_price_training(tmp_path: Path) -> None:
    """
    特徴で弾力性の違う商品の合成の表で, 値段を変える判断の骨組みを仮の
    モデルで書き換えずに動かし, 評価, 商品ごとの選んだ値段と値段の決め方の
    成果物を確かめる関数 (PyTorch と段階のライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 入れた表のレシピの雛形での書き出しと, 仮のモデルでの短い学習
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    source = write_sales(tmp_path / "raw")
    problem = attach_data(price_problem(), str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    assert "data/sales.csv" in make_project(problem, project)
    export_sales(problem, project)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル (PlaceholderModel) で学習します" in log
    assert "予測: 値段を変えたときの売れた数" in log
    assert "学習に使う列: 3 (最後が値段の列 price)" in log
    assert "仮のモデル (PlaceholderModel) は ML-ModelControl の定義が無い" \
        "ので, 推論の一式を作りません" in log
    assert not (run / "inference").exists()

    # 検証のたびの売れた数の誤差
    with open(run / "history.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 8
    assert all(row["valid/mae"] not in ("", None) for row in rows)

    # 検証用とテスト用の誤差, 弾力性, 見込みの利益の比べ
    for split in ("valid", "test"):
        table = read_evaluation(run, f"price_{split}")
        assert table["split"] == split
        assert table["better"]["profit_gain"] == "higher"
        row = table["rows"][0]
        assert row["products"] == PRODUCTS and row["max_change"] == 0.1
        assert row["mae"] >= 0 and row["rmse"] >= row["mae"]
        assert row["profit_gain"] >= 0 and row["profit"] >= (
            row["current_profit"]
        )
        assert row["elasticity"] < 0, row

    # 商品ごとの選んだ値段 (今の値段は商品の記録の値段の中央値. 上限を
    # 超えず, 弾力性の大きい種類 0 ほど下げる)
    with open(run / "price_choices.csv", encoding="utf-8",
              newline="") as file:
        choices = list(csv.DictReader(file))
    assert list(choices[0]) == ["product_id", "split", "rows",
                                "current_price", "cost", "change", "price",
                                "profit", "current_profit", "elasticity"]
    assert {row["split"] for row in choices} == {"train", "valid", "test"}
    for row in choices:
        change = float(row["change"])
        assert abs(change) <= 0.1 + 1e-9 and change in (-0.1, 0, 0.1)
        current = float(row["current_price"])
        assert 80 - 1e-9 <= current <= 120 + 1e-9
        assert float(row["price"]) == pytest.approx(current * (1 + change))
        assert float(row["cost"]) == pytest.approx(60.0)
    train = [row for row in choices if row["split"] == "train"]
    elastic = [float(row["change"]) for row in train
               if int(row["product_id"][1:]) % 2 == 0]
    flat = [float(row["change"]) for row in train
            if int(row["product_id"][1:]) % 2 == 1]
    assert sum(elastic) / len(elastic) < sum(flat) / len(flat)

    # 値段の決め方と前処理のレシピの写し
    estimator = json.loads((run / "price_estimator.json")
                           .read_text(encoding="utf-8"))
    assert estimator["format_version"] == 1
    assert (estimator["recipe"], estimator["choices"]) == (
        "price_recipe.yaml", "price_choices.csv"
    )
    assert estimator["features"] == ["category", "weekday", "price"]
    assert (estimator["price"], estimator["cost"], estimator["product"],
            estimator["id"]) == ("price", "unit_cost", "product_id",
                                 "record_id")
    assert estimator["current"] == "median"
    assert (estimator["candidates"], estimator["max_change"]) == (
        [-0.1, 0.0, 0.1, 0.2], 0.1
    )
    assert estimator["model"] == {"input": "x", "units": "units",
                                  "link": "softplus"}
    assert estimator["scale"] > 0
    assert estimator["train"]["products"] == PRODUCTS
    recipe = yaml.safe_load((run / "price_recipe.yaml")
                            .read_text(encoding="utf-8"))
    assert recipe["task"] == "price" and recipe["price"] == "price"
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"]) == ("effect", "effect")
    assert (run / "final_model.pt").is_file()


def test_price_rows(tmp_path: Path) -> None:
    """
    商品の id と原価の列の無い表では, 行ごとに記録の値段を今の値段にして
    売上で値段を選ぶことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 商品の id と原価を答えない問題の設定 (PyTorch と段階のライブラリが
    # 無ければ飛ばす)
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    source = write_sales(tmp_path / "raw")
    problem = price_problem()
    for key in ("product", "cost"):
        problem = answer(problem, f"{PART}.price.{key}", None)
    problem = attach_data(problem, str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    make_project(problem, project)
    export_sales(problem, project)
    done = run_script(project, ("trainer.epochs=2", "experiment.device=cpu"))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)
    with open(run / "price_choices.csv", encoding="utf-8",
              newline="") as file:
        choices = list(csv.DictReader(file))
    assert list(choices[0])[0] == "record_id"
    assert len(choices) == PRODUCTS * RECORDS
    assert all(row["cost"] == "" for row in choices)
    assert all(row["rows"] == "1" for row in choices)
    estimator = json.loads((run / "price_estimator.json")
                           .read_text(encoding="utf-8"))
    assert (estimator["current"], estimator["cost"],
            estimator["product"]) == ("row", None, None)


def place_spec(project: Path) -> tuple[dict, Path]:
    """
    ML-ModelControl の同梱の例 tabular_price の入力の幅を 3 列にした定義を,
    プロジェクトの models/model.yaml に置く関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    placed: tuple[dict, Path]
        置いた定義と, 置いた場所
    """
    # 入力の形と最初の層の入力の幅
    from ml_modelcontrol import load_example
    spec = load_example("tabular_price")
    spec["inputs"]["x"]["shape"] = [8, 3]
    spec["nodes"]["units1"]["params"]["in_features"] = 3
    place = project / "models" / "model.yaml"
    place.parent.mkdir(parents=True, exist_ok=True)
    place.write_text(yaml.safe_dump(spec, allow_unicode=True),
                     encoding="utf-8")
    placed = (spec, place)

    return placed


def check_bundle(run: Path) -> object:
    """
    実験の中に値段の一式ができ, 一式の既定の判断が値段の判断かを確かめる
    関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ

    Returns
    ----------
    bundle: object
        読み込んだ ML-Inference の一式
    """
    # 一式の版, 受け持ちと既定の判断 (ML-Inference が無ければ飛ばす)
    inference = pytest.importorskip("ml_inference")
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "変える幅の上限 0.1. 版 13)" in log
    bundle = inference.load_bundle(run / "inference")
    assert bundle.info["format_version"] == 13
    assert bundle.info["role"] == "effect"
    assert bundle.info["decision"] == {
        "type": "price", "candidates": [-0.1, 0.0, 0.1, 0.2],
        "max_change": 0.1,
    }

    return bundle


def test_price_rows_bundle(tmp_path: Path) -> None:
    """
    商品の id の無い表を tabular_price の定義で学び, 推論の一式の選んだ
    値段が実験の選んだ値段の表と合うことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 商品の id を答えない問題の設定と, 定義での短い学習 (段階のライブラリ
    # が無ければ飛ばす)
    for name in ("torch", "ml_eda", "ml_found", "ml_modelcontrol",
                 "ml_inference"):
        pytest.importorskip(name)
    pd = pytest.importorskip("pandas")
    source = write_sales(tmp_path / "raw")
    problem = answer(price_problem(), f"{PART}.price.product", None)
    problem = attach_data(problem, str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    make_project(problem, project)
    export_sales(problem, project)
    place_spec(project)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)

    # 一式づくりの比べ (全ての行が 1 つの分割の 1 行)
    bundle = check_bundle(run)
    check = bundle.info["price"]["check"]
    assert check["rows"] == PRODUCTS * RECORDS
    assert check["agreement"] == 1.0
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "推論の一式 (値段): inference/ (選んだ値段の一致 100.0 % (" \
        f"{PRODUCTS * RECORDS} 件" in log

    # 元の表を一式で決めた値段と, 実験の選んだ値段の表の行ごとの比べ
    import ml_inference
    result = ml_inference.infer(bundle, source)
    found = result["frame"].set_index("record_id")["change"]
    choices = pd.read_csv(run / "price_choices.csv")
    recorded = choices.set_index("record_id")["change"]
    gap = (found.loc[recorded.index] - recorded).abs()
    assert float(gap.max()) <= 1e-9


def test_price_example_spec(tmp_path: Path) -> None:
    """
    ML-ModelControl の同梱の例 tabular_price の入力の幅を列の数に合わせた
    定義で学び, 合わない定義は出力先を作る前に止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同梱の例の入力の幅を 3 列に (ML-ModelControl が無ければ飛ばす)
    for name in ("torch", "ml_eda", "ml_found", "ml_modelcontrol"):
        pytest.importorskip(name)
    source = write_sales(tmp_path / "raw")
    problem = attach_data(price_problem(), str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    make_project(problem, project)
    export_sales(problem, project)
    spec, place = place_spec(project)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル" not in log
    assert (run / "model_spec.yaml").is_file()
    row = read_evaluation(run, "price_test")["rows"][0]
    assert row["profit_gain"] >= 0

    # 定義のモデルの推論の一式 (商品は分割をまたぐので選んだ値段は比べ
    # ない)
    bundle = check_bundle(run)
    assert bundle.info["price"]["check"] is None
    assert "推論の一式 (値段): inference/ (選んだ値段は比べず" in log

    # 出力の無い定義の, 出力先を作る前の停止
    spec["outputs"] = {"count": "units_softplus"}
    spec["ui"]["outputs"] = {"count": {"x": 1220, "y": 100}}
    place.write_text(yaml.safe_dump(spec, allow_unicode=True),
                     encoding="utf-8")
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 1
    assert "定義の出力に units を入れてください" in done.stdout
    assert len(list((project / "outputs").glob("*/*/*/train.log"))) == 1
