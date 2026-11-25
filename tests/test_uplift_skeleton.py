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

# 合成の介入の記録の表の行の数
ROWS = 1500

# 骨組みを書き換えずに短く学ぶ config の差分
SHORT = ["trainer.epochs=8", "optimizer.params.lr=0.01",
         "experiment.device=cpu"]

# 効果の見積もりの問題の設定の作り方 (tests/test_uplift.py と同じ)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_uplift import uplift_problem  # noqa: E402


def write_log(root: Path, kind: str = "binary") -> Path:
    """
    特徴で効果の違う顧客に無作為に配った, 合成の介入の記録の表の CSV を
    作る関数

    Parameters
    ----------
    root: Path
        書き出す場所
    kind: str = "binary"
        結果の種類 (binary は買ったか, amount は買った額)

    Returns
    ----------
    path: Path
        CSV のファイル (customer_id, age, visits, coupon, purchased, used)
    """
    # visits が多い顧客ほど配ると買う表 (age は効果に無関係)
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    rows = []
    for number in range(ROWS):
        visits = float(rng.uniform(0, 10))
        coupon = int(rng.integers(0, 2))
        lift = (0.6 if visits > 5 else 0.0) * coupon
        if kind == "binary":
            outcome = int(rng.uniform() < 0.1 + lift)
        else:
            outcome = round(10 + 50 * lift + float(rng.normal(0, 2)), 2)
        rows.append({"customer_id": f"c{number:04d}",
                     "age": float(rng.uniform(20, 60)), "visits": visits,
                     "coupon": coupon, "purchased": outcome,
                     "used": coupon * int(rng.integers(0, 2))})
    root.mkdir(parents=True, exist_ok=True)
    path = root / "log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def export_log(problem: dict, project: Path) -> Path:
    """
    データを入れた問題の設定のレシピの雛形で, 骨組みの場所に介入の記録の
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


def test_uplift_project(tmp_path: Path) -> None:
    """
    データを入れない効果の見積もりの問題の設定から骨組みを書き出し, config,
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
    # 書き出したファイル, スクリプト, config, README
    problem = uplift_problem()
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
    assert "qini_coefficient" in script and "PlaceholderModel" in script
    assert all(len(line) <= 79 for line in script.splitlines())
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["eda", "modelcontrol"]
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/qini", "max"
    )
    assert config["trainer"]["monitor_tiebreak"] == "valid/loss"
    assert config["data"] == {"dataset": "outputs/dataset", "source": None,
                              "batch_size": 128}
    assert config["uplift"] == {"k": 0.3, "budget": 50}
    assert config["model"]["placeholder"] is True
    assert config["experiment"]["name"] == "purchased_uplift"
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("outputs/dataset", "`coupon`", "`purchased`", "`age`",
                  "`customer_id`", "tabular_uplift", "uplift_estimator.json",
                  "uplift_effects.csv", "evaluations/uplift_<分割>.json",
                  "50 件", "stratified"):
        assert words in readme, words

    # 損失で選ぶ場合の並んだときの指標の無いことと, 損得と制約の直す
    # ところを残した場合の空の配る数の上限
    loss = answer(problem, f"{PART}.evaluation.metric", "loss")
    make_project(loss, tmp_path / "loss")
    config = yaml.safe_load((tmp_path / "loss" / "configs" / "default"
                             / "train.yaml").read_text(encoding="utf-8"))
    assert config["trainer"]["monitor"] == "valid/loss"
    assert "monitor_tiebreak" not in config["trainer"]
    broken = answer(problem, "action.decision",
                    {"type": "treat", "budget": 0})
    with pytest.raises(ValueError, match="accept_errors"):
        make_project(broken, tmp_path / "broken")
    make_project(broken, tmp_path / "broken", accept_errors=True)
    config = yaml.safe_load((tmp_path / "broken" / "configs" / "default"
                             / "train.yaml").read_text(encoding="utf-8"))
    assert config["uplift"]["budget"] is None

    # 書き出せない出力の形と, 試して集める効果の見積もり
    wrong = answer(problem, f"{PART}.output", "value")
    assert project_error(wrong) is not None
    online = answer(problem, f"{PART}.collect", "online")
    assert "まだ扱えません" in project_error(online)

    # データセットが無ければ, 書き出しを案内して止まる
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 1
    assert "介入の記録の表を書き出してください" in done.stdout


@pytest.mark.parametrize("kind", ["binary", "amount"])
def test_uplift_training(tmp_path: Path, kind: str) -> None:
    """
    特徴で効果の違う顧客の合成の表で, 効果の見積もりの骨組みを仮のモデルで
    書き換えずに動かし, 行ごとの効果, Qini の曲線と係数, 上位 K 割の効果の
    差, 効果の決め方の成果物を確かめる関数 (PyTorch と段階のライブラリが
    無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    kind: str
        結果の種類

    Returns
    ----------
    None
    """
    # 入れた表のレシピの雛形での書き出しと, 仮のモデルでの短い学習
    for name in ("torch", "ml_eda", "ml_found"):
        pytest.importorskip(name)
    source = write_log(tmp_path / "raw", kind)
    problem = attach_data(uplift_problem(kind), str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    assert "data/log.csv" in make_project(problem, project)
    export_log(problem, project)
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "仮のモデル (PlaceholderModel) で学習します" in log
    assert "効果: クーポンで買う確率がどれだけ上がるか" in log
    assert "学習に使う列: 2" in log

    # 検証のたびの Qini の係数での最良の選択
    with open(run / "history.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 8
    assert all(row["valid/qini"] not in ("", None) for row in rows)

    # 検証用とテスト用の Qini の係数と上位 K 割の効果の差と曲線 (効果は
    # visits で決まるので, 見積もりの順は無作為より良い)
    for split in ("valid", "test"):
        table = read_evaluation(run, f"uplift_{split}")
        assert table["split"] == split
        assert table["better"]["qini"] == "higher"
        row = table["rows"][0]
        assert row["k"] == 0.3 and row["treated"] >= 1 and row["control"] >= 1
        assert row["qini"] > 0, row
        assert row["uplift_at_k"] > row["effect"], row
        curve = read_evaluation(run, f"qini_curve_{split}")
        assert curve["kind"] == "curve" and curve["x"] == "fraction"

    # 行ごとの効果と効果の決め方
    with open(run / "uplift_effects.csv", encoding="utf-8",
              newline="") as file:
        effects = list(csv.DictReader(file))
    assert len(effects) == ROWS
    assert list(effects[0]) == ["customer_id", "split", "effect", "treated",
                                "control", "treatment", "outcome"]
    first = effects[0]
    assert abs(float(first["effect"]) - (float(first["treated"])
                                         - float(first["control"]))) < 1e-9
    if kind == "binary":
        assert all(0 <= float(row["treated"]) <= 1 for row in effects)
    estimator = json.loads((run / "uplift_estimator.json")
                           .read_text(encoding="utf-8"))
    assert estimator["format_version"] == 1
    assert (estimator["learner"], estimator["outcome"]) == (
        "t_learner", kind
    )
    assert (estimator["recipe"], estimator["effects"]) == (
        "uplift_recipe.yaml", "uplift_effects.csv"
    )
    assert estimator["features"] == ["age", "visits"]
    assert (estimator["id"], estimator["treatment"]) == ("customer_id",
                                                         "coupon")
    assert (estimator["budget"], estimator["k"]) == (50, 0.3)
    assert estimator["model"] == {
        "input": "x", "treated": "treated", "control": "control",
        "link": "sigmoid" if kind == "binary" else "identity",
    }
    assert (estimator["scale"] is None) == (kind == "binary")
    assert estimator["train"]["p10"] <= estimator["train"]["p90"]
    recipe = yaml.safe_load((run / "uplift_recipe.yaml")
                            .read_text(encoding="utf-8"))
    assert recipe["task"] == "uplift" and recipe["treatment"] == "coupon"
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"]) == ("effect", "effect")
    assert (run / "final_model.pt").is_file()


def test_uplift_example_spec(tmp_path: Path) -> None:
    """
    ML-ModelControl の同梱の例 tabular_uplift の入力の幅を列の数に合わせた
    定義で学び, 合わない定義は出力先を作る前に止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 同梱の例の入力の幅を 2 列に (ML-ModelControl が無ければ飛ばす)
    for name in ("torch", "ml_eda", "ml_found", "ml_modelcontrol"):
        pytest.importorskip(name)
    from ml_modelcontrol import load_example
    source = write_log(tmp_path / "raw")
    problem = attach_data(uplift_problem(), str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    make_project(problem, project)
    export_log(problem, project)
    spec = load_example("tabular_uplift")
    spec["inputs"]["x"]["shape"] = [8, 2]
    for node in ("treated1", "control1"):
        spec["nodes"][node]["params"]["in_features"] = 2
    place = project / "models" / "model.yaml"
    place.parent.mkdir(parents=True, exist_ok=True)
    place.write_text(yaml.safe_dump(spec, allow_unicode=True),
                     encoding="utf-8")
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    run = only_run(project)
    assert "仮のモデル" not in (run / "train.log").read_text(encoding="utf-8")
    assert (run / "model_spec.yaml").is_file()
    assert read_evaluation(run, "uplift_test")["rows"][0]["qini"] > 0

    # 出力が 1 つの定義の, 出力先を作る前の停止
    del spec["outputs"]["control"]
    place.write_text(yaml.safe_dump(spec, allow_unicode=True),
                     encoding="utf-8")
    done = run_script(project, tuple(SHORT))
    assert done.returncode == 1
    assert "定義の出力に control を入れてください" in done.stdout
    assert len(list((project / "outputs").glob("*/*/*/train.log"))) == 1
