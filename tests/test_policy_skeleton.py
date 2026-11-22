import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from ml_quest import (
    answer, check, example_problem, make_project, project_error,
    recipe_from
)
from ml_quest.policy_skeleton import fallback_value

# 受け持ちの場所
PART = "ml.parts.0"

# 学習と方策の一式まで通すのに要るもの (CI の train のジョブで入れる)
TRAINING = ("torch", "ml_found", "gymnasium")

# 骨組みを書き換えずに短く学ぶ config の差分 (少ないエピソードと歩数)
SHORT = ["trainer.epochs=2", "rollout.episodes=4", "rollout.batch_episodes=2",
         "evaluation.episodes=2", "evaluation.guarded_episodes=2",
         "env.max_episode_steps=30"]

# 方策の守りの答え
GUARDS = [("policy.action_limits", "左右に押すだけ"),
          ("policy.stop", "台車が中心から 2 m を出たら"),
          ("policy.fallback", "action"), ("policy.fallback_action", "0"),
          ("policy.approver", "設備の責任者")]


def policy_problem(env_id: str = None) -> dict:
    """
    見本の課題から, 続けて行動を選ぶ方策の問題の設定を作る関数

    Parameters
    ----------
    env_id: str = None
        Gymnasium の環境の登録名 (None なら答えない)

    Returns
    ----------
    problem: dict
        効果を見積もるで方策, シミュレータで集め, 指標が loss の問題の設定
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
                       ("evaluation.metric", "loss"),
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
    assert config["requires"] == ["rl"]
    assert config["experiment"]["name"] == "CartPole-v1_policy"
    assert (config["trainer"]["monitor"], config["trainer"]["mode"]) == (
        "valid/loss", "min"
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
                  "guarded_episodes", "データの分析の段階は無い"):
        assert words in readme, words
    assert "env.py" not in readme


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
    uplift = answer(policy_problem(), f"{PART}.autonomous", False)
    for problem, words in ((logged, "今ある記録"), (qini, "qini"),
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
    assert metadata["model"]["class"] == "model.PolicyNet"
    assert metadata["approver"] == "設備の責任者"
    assert metadata["algorithm"] == (
        "actor_critic" if env_id == "Pendulum-v1" else "reinforce"
    )
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "守り付きの評価" in log
    assert "方策が出すもの: 押す向きを選ぶ方策" in log
    assert "予測すること" not in log

    # 守る指標の表 (守りを当てたエピソード)
    table = json.loads((run / "evaluations" / "guarded_episodes.json")
                       .read_text(encoding="utf-8"))
    assert table["guard"] is True and table["split"] == "environment"
    assert len(table["rows"]) == 2
    assert {"reward", "length", "end", "ok", "fallback"} <= set(
        table["rows"][0]
    )

    # ML-Inference でも同じ守りの行動を返す (入っていれば)
    if env_id != "CartPole-v1":
        return
    ml_inference = pytest.importorskip("ml_inference")
    from ml_found.rl import load_policy
    sys.path.insert(0, str(project))
    try:
        from model import PolicyNet
    finally:
        sys.path.remove(str(project))
    sys.modules.pop("model", None)
    args = metadata["model"]["args"]
    found = load_policy(run / "policy", PolicyNet(**args))
    inferred = ml_inference.load_policy(run / "policy", PolicyNet(**args))
    for observation in ([0.0, 0.1, 0.02, -0.1], [0.5, -0.3, -0.05, 0.2]):
        assert inferred.act(observation) == found.act(observation)
