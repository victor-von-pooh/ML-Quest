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

# 合成の需要の記録の表の商品の数と日の数
PRODUCTS = 6
DAYS = 120

# 学ぶ水準 (受け渡しの既定. 水準 τ = 0.75 を含む) と, 予測に使う列 (前処理
# の後)
LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
FEATURES = ["category", "units_lag1", "units_lag7", "units_mean7",
            "date_weekday"]

# 骨組みを書き換えずに短く学ぶ config の差分
SHORT = ["trainer.epochs=8", "experiment.device=cpu"]

# 時間をまたぐ量の判断の問題の設定の作り方 (tests/test_demand.py と同じ)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_demand import DECISION, demand_problem  # noqa: E402


def write_daily(root: Path) -> Path:
    """
    商品ごとに曜日と水準の違う, 合成の日ごとの売れた数の記録の CSV を作る
    関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        CSV のファイル (record_id, product_id, date, category, units,
        stock, waste)
    """
    # 商品ごとの水準 (種類 1 は多く売れる) と, 週末の多い売れ行き
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    rows = []
    days = pd.date_range("2026-01-01", periods=DAYS)
    for product in range(PRODUCTS):
        category = product % 2
        for number, day in enumerate(days):
            mean = (2 + product + 4 * category) * (
                1.8 if day.weekday() >= 5 else 1.0
            )
            rows.append({
                "record_id": f"r{product}_{number:03d}",
                "product_id": f"p{product:02d}",
                "date": day.strftime("%Y-%m-%d"), "category": category,
                "units": int(rng.poisson(mean)),
                "stock": int(rng.integers(0, 6)),
                "waste": int(rng.integers(0, 3)),
            })
    root.mkdir(parents=True, exist_ok=True)
    path = root / "daily.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def export_daily(problem: dict, project: Path) -> Path:
    """
    データを入れた問題の設定のレシピの雛形で, 骨組みの場所に需要の記録の
    表を書き出す関数

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


def read_orders(run: Path) -> list[dict]:
    """
    実験の商品ごとの発注の数の表を読む関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ

    Returns
    ----------
    rows: list[dict]
        demand_orders.csv の行
    """
    # 列の名前からの辞書
    with open(run / "demand_orders.csv", encoding="utf-8",
              newline="") as file:
        rows = list(csv.DictReader(file))

    return rows


def trained_run(
    tmp_path: Path, problem: dict, spec: bool, extra: tuple = ()
) -> Path:
    """
    合成の表を入れた問題の設定から骨組みを書き出し, 需要の記録の表を書き
    出して短く学ぶ関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    problem: dict
        データを入れていない問題の設定
    spec: bool
        ML-ModelControl の同梱の例 tabular_demand の定義を置くか
    extra: tuple = ()
        config の差分 (SHORT の後)

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 入れた表のレシピの雛形での書き出しと, 書き換えない骨組みの学習
    source = write_daily(tmp_path / "raw")
    problem = attach_data(problem, str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    assert "data/daily.csv" in make_project(problem, project)
    export_daily(problem, project)
    if spec:
        place_spec(project, len(problem_levels(project)))
    done = run_script(project, (*SHORT, *extra))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)

    return run


def problem_levels(project: Path) -> list:
    """
    書き出した config の学ぶ水準を読む関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    levels: list
        config の demand.levels
    """
    # 骨組みの config
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    levels = config["demand"]["levels"]

    return levels


def place_spec(project: Path, levels: int) -> tuple[dict, Path]:
    """
    ML-ModelControl の同梱の例 tabular_demand の入力の幅と水準の数を合わせた
    定義を, プロジェクトの models/model.yaml に置く関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    levels: int
        学ぶ水準の数

    Returns
    ----------
    placed: tuple[dict, Path]
        置いた定義と, 置いた場所
    """
    # 入力の形と, 最初と最後の全結合の幅
    from ml_modelcontrol import load_example
    spec = load_example("tabular_demand")
    spec["inputs"]["x"]["shape"] = [8, len(FEATURES)]
    spec["nodes"]["demand1"]["params"]["in_features"] = len(FEATURES)
    spec["nodes"]["demand2"]["params"]["out_features"] = levels
    place = project / "models" / "model.yaml"
    place.parent.mkdir(parents=True, exist_ok=True)
    place.write_text(yaml.safe_dump(spec, allow_unicode=True),
                     encoding="utf-8")
    placed = (spec, place)

    return placed


def test_demand_project(tmp_path: Path) -> None:
    """
    データを入れない時間をまたぐ量の判断の問題の設定から骨組みを書き出し,
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
    problem = demand_problem()
    assert project_error(problem) is None
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert set(written) == {
        "problem.yaml", "README.md", "configs/default/train.yaml",
        "configs/experiment/train.yaml", "train.py",
    }
    script = (project / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'ROLE = "predict"' in script
    assert 'OUTPUT = "distribution"' in script
    for name in ("PinballLoss", "pinball_loss", "order_cost",
                 "make_demand_bundle", "PlaceholderModel"):
        assert name in script, name
    assert all(len(line) <= 79 for line in script.splitlines())
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol", "inference"]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/loss", "min"
    )
    assert config["data"] == {"dataset": "outputs/dataset", "source": None,
                              "batch_size": 128}
    assert config["demand"] == {"levels": LEVELS}
    assert config["order"] == {"shortage_cost": 3, "holding_cost": 1,
                               "max_order": 60}
    assert config["model"]["placeholder"] is True
    assert config["experiment"]["name"] == "units_demand"
    assert config["trainer"]["epochs"] == 100
    assert config["trainer"]["early_stopping"] == {"use": True,
                                                   "patience": 10}
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("outputs/dataset", "`product_id`", "`date`", "`units`",
                  "`stock`", "`category`", "tabular_demand", "PinballLoss",
                  "`pinball_loss`", "`order_cost`", "demand_estimator.json",
                  "demand_orders.csv", "evaluations/demand_<分割>.json",
                  "`time`", "inference/", "ML-Inference を入れる"):
        assert words in readme, words

    # 損得と制約の直すところを残した場合の空の損と上限
    broken = answer(problem, "action.decision",
                    {**DECISION, "max_order": -1})
    with pytest.raises(ValueError, match="accept_errors"):
        make_project(broken, tmp_path / "broken")
    make_project(broken, tmp_path / "broken", accept_errors=True)
    config = yaml.safe_load((tmp_path / "broken" / "configs" / "default"
                             / "train.yaml").read_text(encoding="utf-8"))
    assert config["order"] == {"shortage_cost": None, "holding_cost": None,
                               "max_order": None}

    # 書き出せない出力の形
    wrong = answer(problem, f"{PART}.output", "value")
    assert project_error(wrong) is not None

    # データセットが無いときの, 書き出しの案内と停止と, 空の損での停止
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 1
    assert "需要の記録の表を書き出してください" in done.stdout


def test_demand_training(tmp_path: Path) -> None:
    """
    商品ごとに曜日と水準の違う合成の表で, 時間をまたぐ量の判断の骨組みを
    仮のモデルで書き換えずに動かし, 評価, 商品ごとの発注の数と需要の決め方
    の成果物を確かめる関数 (PyTorch と段階のライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 仮のモデルでの短い学習
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    run = trained_run(tmp_path, demand_problem(), spec=False)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル (PlaceholderModel) で学習します" in log
    assert "予測: 先の 7 日の売れる数" in log
    assert f"学習に使う列: {len(FEATURES)}. 先の期間: 7 日" in log
    assert "発注の水準 τ: 0.75" in log
    assert "仮のモデル (PlaceholderModel) は ML-ModelControl の定義が無い" \
        "ので, 推論の一式を作りません" in log
    assert not (run / "inference").exists()

    # 検証のたびの 0.5 の分位点の誤差と分位点の損失
    with open(run / "history.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 8
    assert all(row["valid/mae"] not in ("", None) for row in rows)
    assert all(row["valid/pinball_loss"] not in ("", None) for row in rows)

    # 検証用とテスト用の誤差, 分位点の損失, 発注の数の損の比べ
    for split in ("valid", "test"):
        table = read_evaluation(run, f"demand_{split}")
        assert table["split"] == split
        assert table["better"]["order_cost_saving"] == "higher"
        row = table["rows"][0]
        assert row["products"] == PRODUCTS and row["max_order"] == 60
        assert row["level"] == 0.75
        assert row["mae"] >= 0 and row["pinball_loss"] >= 0
        assert row["order_cost"] >= 0 and row["baseline_cost"] >= 0
        assert row["order_cost_saving"] == pytest.approx(
            row["baseline_cost"] - row["order_cost"]
        )
        levels = read_evaluation(run, f"demand_levels_{split}")["rows"]
        assert [item["level"] for item in levels] == LEVELS

    # 発注を決める日の行ごとの発注の数 (0 から上限の間の整数)
    orders = read_orders(run)
    assert list(orders[0]) == ["product_id", "date", "split", "stock",
                               "quantile", "order", "previous", "demand"]
    assert {row["split"] for row in orders} == {"train", "valid", "test"}
    for row in orders:
        order = float(row["order"])
        assert 0 <= order <= 60 and order.is_integer()
        assert float(row["quantile"]) >= 0 and row["demand"] != ""
        assert row["stock"] != "" and row["previous"] != ""

    # 需要の決め方と前処理のレシピの写し
    estimator = json.loads((run / "demand_estimator.json")
                           .read_text(encoding="utf-8"))
    assert estimator["format_version"] == 1
    assert (estimator["recipe"], estimator["orders"]) == (
        "demand_recipe.yaml", "demand_orders.csv"
    )
    assert estimator["features"] == FEATURES
    assert (estimator["product"], estimator["date"], estimator["units"],
            estimator["stock"], estimator["id"]) == (
        "product_id", "date", "units", "stock", "record_id"
    )
    assert (estimator["horizon"], estimator["interval"]) == (7, 7)
    assert estimator["levels"] == LEVELS
    assert (estimator["shortage_cost"], estimator["holding_cost"],
            estimator["max_order"]) == (3, 1, 60)
    assert estimator["model"] == {"input": "x", "quantiles": "quantiles"}
    assert estimator["train"]["products"] == PRODUCTS
    recipe = yaml.safe_load((run / "demand_recipe.yaml")
                            .read_text(encoding="utf-8"))
    assert recipe["task"] == "demand" and recipe["product"] == "product_id"
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"]) == ("predict", "distribution")
    assert (run / "final_model.pt").is_file()


def test_demand_same_costs(tmp_path: Path) -> None:
    """
    品切れの損と売れ残りの損が同じ (τ = 0.5) で在庫の数の列の無い表を,
    学ぶ水準の重なりを 1 つにまとめて学び, 上限で切った発注の数を確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同じ損と小さな上限で, 在庫の数を答えない問題の設定 (PyTorch と段階の
    # ライブラリが無ければ飛ばす)
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    problem = answer(demand_problem(), f"{PART}.demand.stock", None)
    problem = answer(problem, f"{PART}.demand.levels", [0.2, 0.5, 0.8])
    problem = answer(problem, "action.decision",
                     {**DECISION, "shortage_cost": 2, "holding_cost": 2,
                      "max_order": 5})
    run = trained_run(tmp_path, problem, spec=False,
                      extra=("trainer.epochs=2",))
    estimator = json.loads((run / "demand_estimator.json")
                           .read_text(encoding="utf-8"))
    assert estimator["levels"] == [0.2, 0.5, 0.8]
    assert estimator["stock"] is None
    orders = read_orders(run)
    assert all(row["stock"] == "" for row in orders)
    assert all(0 <= float(row["order"]) <= 5 for row in orders)
    capped = sum(read_evaluation(run, f"demand_{split}")["rows"][0]["capped"]
                 for split in ("valid", "test"))
    assert capped > 0


def test_demand_bundle(tmp_path: Path) -> None:
    """
    ML-ModelControl の同梱の例 tabular_demand の定義で学び, 推論の一式の
    発注の数が実験の発注の数と合い, 既定の判断が発注の判断であることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 定義での短い学習 (段階のライブラリが無ければ飛ばす)
    for name in ("torch", "ml_eda", "ml_found", "ml_modelcontrol",
                 "ml_inference"):
        pytest.importorskip(name)
    run = trained_run(tmp_path, demand_problem(), spec=True)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル" not in log
    assert "合わせられませんでした" not in log
    assert (run / "model_spec.yaml").is_file()

    # 一式の版, 受け持ちと既定の判断と, 元の表での発注の数の比べ
    import ml_inference
    bundle = ml_inference.load_bundle(run / "inference")
    assert bundle.info["format_version"] == 14
    assert bundle.info["role"] == "predict"
    assert bundle.info["decision"] == DECISION
    assert bundle.info["output"]["quantiles"] == LEVELS
    check = bundle.info["demand"]["check"]
    assert check["rows"] == len(read_orders(run))
    assert check["agreement"] == 1.0
    assert "推論の一式 (需要): inference/ (発注の数の一致 100.0 % (" in log
    assert "発注の数の上限 60. 版 14)" in log
    assert set(bundle.info["demand"]["evaluations"]) == {"valid", "test"}

    # 一式の発注の数が 0 から上限の間
    source = tmp_path / "project" / "data" / "daily.csv"
    result = ml_inference.infer(bundle, source)
    orders = result["frame"]["order"].dropna()
    assert len(orders) == PRODUCTS
    assert ((orders >= 0) & (orders <= 60)).all()


def test_demand_spec_errors(tmp_path: Path) -> None:
    """
    水準の数に合わない定義は, 出力先を作る前に止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 出力の幅を水準の数と違う 3 にした定義 (ML-ModelControl が無ければ
    # 飛ばす)
    for name in ("torch", "ml_eda", "ml_found", "ml_modelcontrol"):
        pytest.importorskip(name)
    source = write_daily(tmp_path / "raw")
    problem = attach_data(demand_problem(), str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    make_project(problem, project)
    export_daily(problem, project)
    place_spec(project, 3)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 1
    assert "の形を (件数, 学ぶ水準の数 5) にしてください" in done.stdout
    assert not list((project / "outputs").glob("*/*/*/train.log"))
