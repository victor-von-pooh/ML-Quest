import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, check, example_problem, make_project, project_error,
    questions, recipe_from
)
from ml_quest.policy_skeleton import fallback_value

# 受け持ちの場所
PART = "ml.parts.0"

# 学習と方策の一式まで通すのに要るもの (CI の train のジョブで入れる)
TRAINING = ("torch", "ml_found", "ml_modelcontrol", "gymnasium")

# 骨組みを書き換えずに短く学ぶ config の差分 (少ないエピソードと歩数)
SHORT = ["trainer.epochs=2", "rollout.episodes=4", "rollout.batch_episodes=2",
         "evaluation.episodes=2", "evaluation.guarded_episodes=2",
         "env.max_episode_steps=30"]

# 方策の守りの答え
GUARDS = [("policy.action_limits", "左右に押すだけ"),
          ("policy.stop", "台車が中心から 2 m を出たら"),
          ("policy.fallback", "action"), ("policy.fallback_action", "0"),
          ("policy.approver", "設備の責任者")]


def policy_problem(env_id: str = None, metric: str = "reward") -> dict:
    """
    見本の課題から, 続けて行動を選ぶ方策の問題の設定を作る関数

    Parameters
    ----------
    env_id: str = None
        Gymnasium の環境の登録名 (None なら答えない)
    metric: str = "reward"
        最良を選ぶ指標 (方策の既定の reward か, 古い設定の loss)

    Returns
    ----------
    problem: dict
        効果を見積もるで方策, シミュレータで集める問題の設定
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    for key, value in [("role", "effect"), ("unit", "row"),
                       ("output", "effect"), ("target.source", "feedback"),
                       ("method", "scratch"), ("what", "押す向きを選ぶ方策"),
                       ("item", "1 歩"), ("baseline", "人の操作"),
                       ("evaluation.metric", metric),
                       ("evaluation.link", "報酬が大きいほど損が減る"),
                       ("autonomous", True), ("collect", "simulator"),
                       ("simulator", "other"), *GUARDS]:
        problem = answer(problem, f"{PART}.{key}", value)
    if env_id is not None:
        problem = answer(problem, f"{PART}.env_id", env_id)

    return problem


def train(project: Path) -> Path:
    """
    書き出した骨組みを書き換えずに, 別のプロセスで短く学習する関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # config の差分だけを渡す
    done = subprocess.run([sys.executable, "train.py", *SHORT], cwd=project,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    runs = sorted((project / "outputs").glob("*/*/*/train.log"))
    assert len(runs) == 1
    run = runs[0].parent

    return run


def test_policy_project(tmp_path: Path) -> None:
    """
    環境の登録名を答えた方策の問題の設定から, Gymnasium の環境で学ぶ
    骨組みを書き出せることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 誤りが無く, 書き出せる. 自作の環境の雛形は書かない
    problem = policy_problem("CartPole-v1")
    assert [item for item in check(problem) if item["level"] == "error"] == []
    assert project_error(problem) is None
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert set(written) == {
        "problem.yaml", "README.md", "configs/default/train.yaml",
        "configs/experiment/train.yaml", "train.py", "model.py",
    }
    for name in ("train.py", "model.py"):
        ast.parse((project / name).read_text(encoding="utf-8"))
    script = (project / "train.py").read_text(encoding="utf-8")
    assert 'APPROVER = "設備の責任者"' in script

    # config: 要る機能, 監視, 環境, 違反の信号, 守り
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["requires"] == ["rl", "modelcontrol"]
    assert config["model"]["spec"] == "models/policy.yaml"
    assert not (project / "models" / "policy.yaml").exists()
    assert config["experiment"]["name"] == "CartPole-v1_policy"
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/reward", "max"
    )
    assert config["trainer"]["monitor_tiebreak"] == "train/reward"
    assert config["env"]["id"] == "CartPole-v1"
    assert config["algorithm"] == "auto"
    assert config["safety_key"] == "cost"
    assert config["guards"] == {
        "action_low": None, "action_high": None, "allowed_actions": None,
        "observation_low": None, "observation_high": None,
        "fallback": "action", "fallback_action": 0,
    }
    text = (project / "configs" / "default" / "train.yaml").read_text(
        encoding="utf-8"
    )
    assert "左右に押すだけ" in text and "台車が中心から 2 m" in text

    # README: 環境, 守り, 方策の一式
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("`CartPole-v1`", "設備の責任者", "policy/",
                  "`valid/reward` (評価のエピソードの報酬の和の平均. 大きい",
                  "guarded_episodes", "データの分析の段階は無い",
                  "ML-ModelControl (`modelcontrol`)",
                  "全結合と Tanh を 2 段と, 行動の数の全結合",
                  "既にある定義は書き直さない"):
        assert words in readme, words
    assert "env.py" not in readme


def test_policy_loss_metric(tmp_path: Path) -> None:
    """
    指標に損失 (loss) を選んだ古い方策の問題の設定も, 選択肢が強化学習の
    言葉で, 今まで通り valid/loss を min で監視する骨組みを書き出せること
    を確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 指標の選択肢の loss は強化学習の言葉で 1 つ (方策でなければ損失)
    problem = policy_problem("CartPole-v1", metric="loss")
    for value, words in ((True, "評価のエピソードの報酬の和の平均の符号を"
                                "反転した値"),
                         (False, "検証用の分割での損失")):
        items = {item["id"]: item for item in questions(
            answer(problem, f"{PART}.autonomous", value)
        )}
        labels = [choice["label"] for choice
                  in items[f"{PART}.evaluation.metric"]["choices"]
                  if choice["value"] == "loss"]
        assert len(labels) == 1 and labels[0].startswith(words), labels

    # 誤りが無く, 監視と README の説明が損失の向き
    assert [item for item in check(problem) if item["level"] == "error"] == []
    assert project_error(problem) is None
    project = tmp_path / "project"
    make_project(problem, project)
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/loss", "min"
    )
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "`valid/loss` (評価のエピソードの報酬の和の平均の符号" in readme


def test_policy_env_template(tmp_path: Path) -> None:
    """
    環境の登録名を答えない方策では, 自作の環境の雛形を書き出すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 雛形の env.py に違反の信号の名前を書く. 決めた行動が文なら空にする
    problem = answer(policy_problem(), f"{PART}.policy.safety_key", "unsafe")
    problem = answer(problem, f"{PART}.policy.fallback_action", "止める")
    project = tmp_path / "project"
    written = make_project(problem, project)
    assert "env.py" in written
    env = (project / "env.py").read_text(encoding="utf-8")
    ast.parse(env)
    assert 'SAFETY_KEY = "unsafe"' in env and "class ProjectEnv" in env
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["env"]["id"] is None
    assert config["experiment"]["name"] == "env_policy"
    assert config["safety_key"] == "unsafe"
    assert config["guards"]["fallback_action"] is None
    text = (project / "configs" / "default" / "train.yaml").read_text(
        encoding="utf-8"
    )
    assert "決めた行動 (問題の設定): 止める" in text
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "env.py" in readme and "ProjectEnv" in readme


def test_policy_errors(tmp_path: Path) -> None:
    """
    方策の骨組みを書き出せない場合が make_project と同じ文になり, レシピの
    下書きが無いことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 今ある記録から集める, 指標が qini, 方策でない効果の見積もり
    logged = answer(policy_problem(), f"{PART}.collect", "log")
    qini = answer(policy_problem(), f"{PART}.evaluation.metric", "qini")
    uplift = answer(policy_problem(metric="loss"), f"{PART}.autonomous",
                    False)
    assert "「評価のエピソードの報酬の和の平均" in project_error(qini)
    assert "「評価のエピソードの報酬の和の平均の符号を反転した値" in (
        project_error(qini)
    )
    assert "検証用の分割での損失" not in project_error(qini)
    assert "今は「効果の順の当たり (Qini" in project_error(qini)
    for problem, words in ((logged, "今ある記録"), (qini, "Qini"),
                           (uplift, "まだ扱えません")):
        message = project_error(problem)
        assert message is not None and words in message, message
        with pytest.raises(ValueError) as raised:
            make_project(problem, tmp_path / "project")
        assert str(raised.value) == message
    assert not (tmp_path / "project").exists()

    # 環境とやり取りして集めるので, データの分析の段階 (レシピ) は無い
    with pytest.raises(ValueError, match="データの分析の段階"):
        recipe_from(policy_problem())


def test_policy_approver_lines(tmp_path: Path) -> None:
    """
    改行のある承認する人と守りの文が, config の YAML を崩さないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 改行の後に YAML の行に見える文
    problem = answer(policy_problem(), f"{PART}.policy.approver",
                     "設備の\nfallback: stop\n責任者")
    problem = answer(problem, f"{PART}.policy.stop",
                     "2 m を出たら\nguards: null")
    project = tmp_path / "project"
    make_project(problem, project)
    config = yaml.safe_load((project / "configs" / "default" / "train.yaml")
                            .read_text(encoding="utf-8"))
    assert config["guards"]["fallback"] == "action"
    assert isinstance(config["guards"], dict)
    assert "fallback" not in config
    script = (project / "train.py").read_text(encoding="utf-8")
    ast.parse(script)
    assert 'APPROVER = "設備の fallback: stop 責任者"' in script

    # README の箇条も 1 行のまま
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert "設備の fallback: stop 責任者 が決める" in readme
    assert "\nfallback: stop" not in readme and "\nguards: null" not in readme


def test_policy_config_errors(tmp_path: Path) -> None:
    """
    学習の方法の名前, 決めた行動, 空の定義の場所の誤りが, 出力先を作る
    前に止まることを確かめる関数 (ML-Found と Gymnasium が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 知らない方法, 番号で選ぶ行動で決めた行動が無い, 定義の場所が空
    for name in TRAINING:
        pytest.importorskip(name)
    project = tmp_path / "project"
    make_project(policy_problem("CartPole-v1"), project)
    for override, words in (("algorithm=ppo", "知らない方法"),
                            ("guards.fallback_action=null",
                             "guards.fallback_action"),
                            ("model.spec=null",
                             "model.spec に定義の場所を書いてください")):
        done = subprocess.run([sys.executable, "train.py", override],
                              cwd=project, capture_output=True, text=True,
                              timeout=600)
        assert done.returncode == 1, override
        assert words in done.stdout, done.stdout
        assert "Traceback" not in done.stderr, done.stderr[-2000:]
    assert not (project / "outputs").exists()


def test_policy_definition_errors(tmp_path: Path) -> None:
    """
    方策の頭の定義の誤り (ノードと出力が無い, 出力が 2 つ, 行動の数と違う,
    観測を入れられない) が, 出力先を作る前に止まることを確かめる関数 (ML-Found, ML-ModelControl,
    Gymnasium が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # CartPole は観測 4, 行動 2
    for name in TRAINING:
        pytest.importorskip(name)
    project = tmp_path / "project"
    make_project(policy_problem("CartPole-v1"), project)
    (project / "models").mkdir(exist_ok=True)
    linear = {"unit": "nn.Linear", "inputs": "x",
              "params": {"in_features": 4, "out_features": 3}}
    cases = (
        ({"inputs": {"x": {"shape": [1, 4]}}, "nodes": {}},
         "outputs に 1 つ以上の出力を書いてください"),
        ({"inputs": {"x": {"shape": [1, 4]}}, "nodes": {"fc": linear},
          "outputs": {"a": "fc", "b": "fc"}}, "1 つずつ"),
        ({"inputs": {"x": {"shape": [1, 4]}}, "nodes": {"fc": linear},
          "outputs": {"a": "fc"}}, "行動の数 2"),
        ({"inputs": {"x": {"shape": [1, 4]}},
          "nodes": {"fc": {**linear, "params": {"in_features": 5,
                                                 "out_features": 2}}},
          "outputs": {"a": "fc"}}, "入力の形を (1, 4) に"),
    )
    for definition, words in cases:
        (project / "models" / "policy.yaml").write_text(
            yaml.safe_dump(definition), encoding="utf-8"
        )
        done = subprocess.run([sys.executable, "train.py"], cwd=project,
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 1, done.stderr[-2000:]
        assert "定義に誤りがあります" in done.stdout, done.stdout
        assert "「次にすること」の 3" in done.stdout
        assert words in done.stdout, done.stdout
        assert "Traceback" not in done.stderr
    assert not (project / "outputs").exists()


# 扱えない空間の自作の環境 ({observation} と {action} に空間の式)
SPACE_ENV = """
import gymnasium
from gymnasium import spaces
import numpy as np


class ProjectEnv(gymnasium.Env):
    def __init__(self):
        super(ProjectEnv, self).__init__()
        self.observation_space = {observation}
        self.action_space = {action}
"""

# 扱える観測と, 扱えない行動の空間と, 扱えない観測の空間
BOX = "spaces.Box(-1.0, 1.0, (2,), np.float32)"
SPACES = {
    "dict": (BOX, 'spaces.Dict({"a": spaces.Discrete(2)})'),
    "scalar_box": (BOX, "spaces.Box(-1.0, 1.0, (), np.float32)"),
    "box_2d": (BOX, "spaces.Box(-1.0, 1.0, (2, 2), np.float32)"),
    "discrete_start": (BOX, "spaces.Discrete(3, start=1)"),
    "tuple_obs": ("spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))",
                  "spaces.Discrete(2)"),
}


@pytest.mark.parametrize("env_id", ["FrozenLake-v1", *SPACES])
def test_policy_space_errors(tmp_path: Path, env_id: str) -> None:
    """
    扱えない観測の空間 (Discrete) と行動の空間 (Dict) が, 出力先を作る前
    に「環境の空間を扱えません」で止まることを確かめる関数 (ML-Found と
    Gymnasium が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    env_id: str
        観測が Discrete の環境の登録名か, 自作の環境の空間の組の名前
        (SPACES)

    Returns
    ----------
    None
    """
    # 書き出して, 自作の環境は Dict の行動の空間に書き換える
    for name in TRAINING:
        pytest.importorskip(name)
    project = tmp_path / "project"
    make_project(policy_problem(None if env_id in SPACES else env_id),
                 project)
    if env_id in SPACES:
        observation, action = SPACES[env_id]
        (project / "env.py").write_text(
            SPACE_ENV.replace("{observation}", observation)
            .replace("{action}", action), encoding="utf-8"
        )
    done = subprocess.run([sys.executable, "train.py"], cwd=project,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 1, done.stderr[-2000:]
    assert "環境の空間を扱えません" in done.stdout, done.stdout
    assert "guards" not in done.stdout and "形 None" not in done.stdout
    assert "Traceback" not in done.stderr
    assert not (project / "outputs").exists()


def test_policy_guards_numbers(tmp_path: Path) -> None:
    """
    守りに数を書くと, 守り付きの評価で clipped と fallback が数えられ,
    一式に守りを数にした印が残ることを確かめる関数 (PyTorch, ML-Found,
    Gymnasium が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 選んでよい行動を 0 だけ, 棒の角度を狭く (外れたら決めた行動 0)
    for name in TRAINING:
        pytest.importorskip(name)
    project = tmp_path / "project"
    make_project(policy_problem("CartPole-v1"), project)
    done = subprocess.run(
        [sys.executable, "train.py", *SHORT, "guards.allowed_actions=[0]",
         "guards.observation_low=[null,null,-0.01,null]",
         "guards.observation_high=[null,null,0.01,null]"],
        cwd=project, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-2000:]
    run = next((project / "outputs").glob("*/*/*/policy")).parent
    spec = json.loads((run / "policy" / "policy.json")
                      .read_text(encoding="utf-8"))
    assert spec["metadata"]["guards_set"] is True
    assert spec["guards"]["allowed_actions"] == [0]
    table = json.loads((run / "evaluations" / "guarded_episodes.json")
                       .read_text(encoding="utf-8"))
    assert sum(row["clipped"] + row["fallback"]
               for row in table["rows"]) > 0
    assert "数で書いていない" not in table["description"]
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "数で書いていない" not in log


def test_fallback_value() -> None:
    """
    決めた行動の答えを config の数か数の並びにすることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 数, 数の文, 区切りの数の文, 読めない文
    assert fallback_value(1) == 1
    assert fallback_value("0") == 0
    assert fallback_value("0.5, -1") == [0.5, -1]
    assert fallback_value([0, 1.5]) == [0, 1.5]
    for value in ("止める", None, True, "nan", "1,", ["a"]):
        assert fallback_value(value) is None, value


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1", None])
def test_policy_training(tmp_path: Path, env_id: str) -> None:
    """
    方策の骨組みを書き換えずに学び, 守りと一緒に書き出した方策の一式を
    守り付きで確かめることを確かめる関数 (PyTorch, ML-Found, Gymnasium が
    無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    env_id: str
        Gymnasium の環境の登録名 (None は自作の環境の雛形)

    Returns
    ----------
    None
    """
    # 書き出しと学習 (連続の行動の Pendulum は決めた行動を 0 の行動に)
    for name in TRAINING:
        pytest.importorskip(name)
    problem = policy_problem(env_id)
    if env_id == "Pendulum-v1":
        problem = answer(problem, f"{PART}.policy.fallback_action", None)
    project = tmp_path / "project"
    make_project(problem, project)
    run = train(project)

    # 評価の報酬 (valid/reward) を大きいほど良い向きで監視し, 記録する
    status = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert (status["monitor"], status["mode"]) == ("valid/reward", "max")
    header = (run / "history.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "valid/reward" in header.split(",")

    # 何の実験か (環境の報酬)
    task = yaml.safe_load((run / "task.yaml").read_text(encoding="utf-8"))
    assert (task["role"], task["output"]) == ("effect", "action")
    assert task["target"]["kind"] == "reward"
    assert task["target"]["origin"]["type"] == "environment"
    assert task["target"]["origin"]["name"] == (env_id or "env:ProjectEnv")

    # 方策の一式: 守り, 組み立て方, 学習の方法
    spec = json.loads((run / "policy" / "policy.json")
                      .read_text(encoding="utf-8"))
    assert spec["guards"]["fallback"]["kind"] == "action"
    metadata = spec["metadata"]
    assert metadata["model"] == {"spec": "model.yaml"}
    assert metadata["approver"] == "設備の責任者"
    assert metadata["env_id"] == env_id
    assert ("entry_point" in metadata) == (env_id is None)
    assert metadata["guards_set"] is False
    assert metadata["guards_text"] == {
        "action_limits": "左右に押すだけ", "stop": "台車が中心から 2 m を出たら"
    }
    assert metadata["algorithm"] == (
        "actor_critic" if env_id == "Pendulum-v1" else "reinforce"
    )
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "守り付きの評価" in log and "数で書いていない" in log
    assert "方策が出すもの: 押す向きを選ぶ方策" in log
    assert "予測すること" not in log

    # 守る指標の表 (守りを当てたエピソード)
    table = json.loads((run / "evaluations" / "guarded_episodes.json")
                       .read_text(encoding="utf-8"))
    assert table["guard"] is True and table["split"] == "environment"
    assert "数で書いていない" in table["description"]
    assert "  " not in table["description"]
    assert table["better"] == {"reward": "higher", "fallback": "lower",
                               "stop": "lower", "handover": "lower"}
    assert len(table["rows"]) == 2
    assert {"reward", "length", "end", "ok", "fallback"} <= set(
        table["rows"][0]
    )

    # 既定の定義をプロジェクトに書き, 一式には実験に写した定義を写す
    definition = (project / "models" / "policy.yaml").read_bytes()
    assert (run / "model_spec.yaml").read_bytes() == definition
    assert (run / "policy" / "model.yaml").read_bytes() == definition

    # ML-Inference がモデルを渡さずに定義から組み立て, ML-Found と同じ守り
    # の行動を返す (方策の頭の重みだけなので余る重みは無い)
    ml_inference = pytest.importorskip("ml_inference")
    from ml_found.rl import load_policy
    from ml_modelcontrol import build_model
    read = ml_inference.read_policy_spec(run / "policy")
    assert read["metadata"]["model"]["spec"] == "model.yaml"
    inferred = ml_inference.load_policy(run / "policy")
    assert inferred.unused_keys == []
    found = load_policy(run / "policy", build_model(
        run / "policy" / "model.yaml", seed=0, load_weights=False
    ))
    size = read["observation_space"]["shape"][0]
    for observation in ([0.0] * size,
                        [0.1 * (index + 1) * (-1) ** index
                         for index in range(size)]):
        got = inferred.act(observation)
        want = found.act(observation)
        assert (got["status"], got["reasons"]) == (
            want["status"], want["reasons"]
        )
        assert got["action"] == pytest.approx(want["action"], abs=1e-5)
