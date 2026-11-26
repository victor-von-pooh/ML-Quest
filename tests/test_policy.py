import copy
from pathlib import Path

import pytest

from ml_quest import (
    FEATURES, FORMAT_VERSION, answer, check, describe, example_problem,
    handoff, is_ready, load_problem, new_problem, questions,
    required_features, save_problem
)

# 受け持ちの場所
PART = "ml.parts.0"

# 方策の守りの答え
GUARDS = [("policy.action_limits", "各関節の速さは毎秒 0.5 rad まで"),
          ("policy.stop", "腕の先が台の上 0 から 0.3 m を出たら"),
          ("policy.fallback", "handover"),
          ("policy.approver", "設備の責任者")]


def effect_problem() -> dict:
    """
    見本の課題から, 効果を見積もるの受け持ちの問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        問題の設定 (方策かどうかは答えていない)
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    for key, value in [("role", "effect"), ("unit", "row"),
                       ("output", "effect"),
                       ("target.source", "intervention"),
                       ("method", "scratch"), ("what", "試しの出力"),
                       ("item", "1 件"), ("baseline", "今のやり方"),
                       ("evaluation.metric", "qini"),
                       ("evaluation.link", "良くなると損が減る")]:
        problem = answer(problem, f"{PART}.{key}", value)

    return problem


def policy_problem() -> dict:
    """
    方策と答え, 守りを全て答えた問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        問題の設定 (シミュレータで集める)
    """
    # 方策, 集め方, 守り
    problem = effect_problem()
    for key, value in [("autonomous", True), ("collect", "simulator"),
                       ("simulator", "mujoco"), *GUARDS]:
        problem = answer(problem, f"{PART}.{key}", value)

    return problem


def errors_of(problem: dict) -> list[str]:
    """
    確かめの error の項目の場所を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    ids: list[str]
        error の項目の場所
    """
    # error だけ
    ids = [item["id"] for item in check(problem) if item["level"] == "error"]

    return ids


def test_effect_without_new_answers() -> None:
    """
    方策と集め方に答えていない効果を見積もるのファイルが, 誤り無く
    渡せることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 方策と集め方の問いは任意なので誤りにならず, 受け渡しは空と None
    problem = effect_problem()
    assert is_ready(problem)
    stages = handoff(problem)
    assert stages["requires"] == [] and stages["policy"] is None

    # 列の予測の見本も, 要る機能は空で方策の守りは None
    sample = handoff(example_problem("総菜の作る量"))
    assert sample["requires"] == [] and sample["policy"] is None
    assert set(sample) == {"part", "data", "model", "training",
                           "inference", "requires", "policy", "rank",
                           "group", "anomaly", "uplift", "price",
                           "project"}


def test_policy_guards_handoff() -> None:
    """
    方策と答えると守りの問いが要り, 答えると受け渡しの policy になる
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 方策と答えただけでは, 守りと承認の答えが足りない
    problem = answer(effect_problem(), f"{PART}.autonomous", True)
    missing = {f"{PART}.{key}" for key, _ in GUARDS}
    assert missing <= set(errors_of(problem))
    with pytest.raises(ValueError):
        handoff(problem)

    # 全て答えると渡せる (信号の名前と決めた行動は任意)
    problem = policy_problem()
    assert is_ready(problem)
    stages = handoff(problem)
    assert stages["policy"] == {
        "action_limits": "各関節の速さは毎秒 0.5 rad まで",
        "stop": "腕の先が台の上 0 から 0.3 m を出たら",
        "fallback": "handover", "fallback_action": None,
        "safety_key": "cost", "approver": "設備の責任者", "env_id": None,
    }
    assert stages["requires"] == ["rl", "mujoco"]
    assert stages["part"]["role"] == "effect"

    # 決めた行動は action のときだけ渡す
    problem = answer(problem, f"{PART}.policy.fallback_action", "止める")
    problem = answer(problem, f"{PART}.policy.safety_key", "unsafe")
    assert handoff(problem)["policy"]["fallback_action"] is None
    problem = answer(problem, f"{PART}.policy.fallback", "action")
    policy = handoff(problem)["policy"]
    assert policy["fallback_action"] == "止める"
    assert policy["safety_key"] == "unsafe"

    # 手で書いた番号の行動は数のまま渡す
    numbered = copy.deepcopy(problem)
    numbered["ml"]["parts"][0]["policy"]["fallback_action"] = 2
    assert is_ready(numbered)
    assert handoff(numbered)["policy"]["fallback_action"] == 2
    numbered["ml"]["parts"][0]["policy"]["fallback_action"] = [0.0, 0.5]
    assert is_ready(numbered)

    # 方策でないと答え直すと守りは渡さない (答えはファイルに残る)
    problem = answer(problem, f"{PART}.autonomous", False)
    assert is_ready(problem)
    assert handoff(problem)["policy"] is None
    assert problem["ml"]["parts"][0]["policy"]["fallback"] == "action"


def test_required_features() -> None:
    """
    要る機能が答えから決まり, 要らなくなった問いの古い答えを使わない
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 答えの途中 (新しい問題の設定) でも呼べる
    assert required_features(new_problem("試し")) == []

    # 記録からは要らず, 実際に試すなら rl
    problem = effect_problem()
    problem = answer(problem, f"{PART}.collect", "log")
    assert required_features(problem) == []
    problem = answer(problem, f"{PART}.collect", "online")
    assert required_features(problem) == ["rl"]

    # シミュレータは選ぶまで足りない答え. その他なら rl だけ
    problem = answer(problem, f"{PART}.collect", "simulator")
    assert f"{PART}.simulator" in errors_of(problem)
    problem = answer(problem, f"{PART}.simulator", "other")
    assert required_features(problem) == ["rl"]

    # ROS2 は方策のときだけ, GPU は全ての受け持ちで
    problem = answer(problem, f"{PART}.ros2", True)
    assert required_features(problem) == ["rl"]
    problem = answer(problem, f"{PART}.autonomous", True)
    problem = answer(problem, "ml.gpu", True)
    assert required_features(problem) == ["rl", "ros2", "cuda"]

    # 確かめに誤りがあっても返す (守りの答えが足りない)
    assert errors_of(problem)
    for key, value in GUARDS:
        problem = answer(problem, f"{PART}.{key}", value)
    assert handoff(problem)["requires"] == ["rl", "ros2", "cuda"]

    # 記録に戻すと, シミュレータの古い答えは使わない
    simulated = answer(problem, f"{PART}.simulator", "mujoco")
    assert required_features(simulated) == ["rl", "mujoco", "ros2", "cuda"]
    simulated = answer(simulated, f"{PART}.collect", "log")
    assert required_features(simulated) == ["ros2", "cuda"]

    # 受け持ちを変えると, 効果を見積もるの答えは使わない
    problem = answer(problem, f"{PART}.role", "group")
    assert required_features(problem) == ["cuda"]

    # 機械学習を使わないなら何も要らない
    assert required_features(answer(problem, "ml.use", False)) == []

    # 版 1 の問題の設定も読める
    old = {"format_version": 1, "ml": {"use": True}}
    assert required_features(old) == []
    old["ml"]["gpu"] = True
    assert required_features(old) == ["cuda"]


def test_policy_questions() -> None:
    """
    方策の問いが効果を見積もるの受け持ちだけで要り, 選択肢を持つこと
    を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 列の予測では要らない
    items = {item["id"]: item
             for item in questions(example_problem("総菜の作る量"))}
    assert not items[f"{PART}.autonomous"]["applies"]
    assert not items[f"{PART}.collect"]["applies"]
    assert items["ml.gpu"]["applies"]

    # 効果を見積もるで方策なら守りの問いが要る
    items = {item["id"]: item for item in questions(policy_problem())}
    for key in ("autonomous", "collect", "simulator", "ros2",
                "policy.action_limits", "policy.stop", "policy.fallback",
                "policy.safety_key", "policy.approver"):
        assert items[f"{PART}.{key}"]["applies"], key
    assert not items[f"{PART}.policy.fallback_action"]["applies"]
    assert [choice["value"] for choice in
            items[f"{PART}.policy.fallback"]["choices"]] == [
        "stop", "handover", "action"
    ]
    assert items[f"{PART}.policy.approver"]["section"] == 10

    # 知らない値と形の誤り
    with pytest.raises(ValueError):
        answer(policy_problem(), f"{PART}.policy.fallback", "retreat")
    with pytest.raises(ValueError):
        answer(policy_problem(), f"{PART}.autonomous", "yes")


def metric_choices(problem: dict) -> list[str]:
    """
    最良を選ぶ指標の選択肢の値を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    values: list[str]
        画面に出す選択肢の値 (並びのまま)
    """
    # questions の choices
    items = {item["id"]: item for item in questions(problem)}
    values = [choice["value"] for choice in
              items[f"{PART}.evaluation.metric"]["choices"]]

    return values


def test_policy_reward_metric() -> None:
    """
    最良を選ぶ指標の報酬 (reward) が方策だけで選べ, 方策の選択肢の先頭に
    なり, 大きいほど良い監視で受け渡すことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 方策では先頭, 方策でない効果の見積もりと方策かを答えていない場合
    assert metric_choices(policy_problem())[:2] == ["reward", "loss"]
    uplift = answer(effect_problem(), f"{PART}.autonomous", False)
    assert "reward" not in metric_choices(uplift)
    assert "reward" in metric_choices(effect_problem())
    assert "reward" not in metric_choices(example_problem("総菜の作る量"))

    # 方策で選ぶと誤りが無く, valid/reward を max で監視する
    problem = answer(policy_problem(), f"{PART}.evaluation.metric",
                     "reward")
    errors = [item for item in check(problem) if item["level"] == "error"
              and item["id"] == f"{PART}.evaluation.metric"]
    assert errors == []
    training = handoff(problem)["training"]
    assert (training["monitor"], training["mode"]) == ("valid/reward", "max")

    # 方策でなくなると合わない指標の誤り. 古い loss は今まで通り
    stale = answer(problem, f"{PART}.autonomous", False)
    errors = [item for item in check(stale) if item["level"] == "error"
              and item["id"] == f"{PART}.evaluation.metric"]
    assert len(errors) == 1 and "合いません" in errors[0]["message"]
    old = answer(policy_problem(), f"{PART}.evaluation.metric", "loss")
    assert [item for item in check(old) if item["level"] == "error"
            and item["id"] == f"{PART}.evaluation.metric"] == []
    training = handoff(old)["training"]
    assert (training["monitor"], training["mode"]) == ("valid/loss", "min")


def test_policy_shape_errors() -> None:
    """
    手で書いたファイルの, 方策の答えの形の崩れを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 知らない値, 真偽値でない値, 文字列でない守り, 辞書でない守り
    cases = [
        (["autonomous"], 1, f"{PART}.autonomous"),
        (["collect"], "sim", f"{PART}.collect"),
        (["simulator"], "gazebo3", f"{PART}.simulator"),
        (["policy", "fallback"], "retreat", f"{PART}.policy.fallback"),
        (["policy", "stop"], 3, f"{PART}.policy.stop"),
        (["policy", "fallback_action"], ["0"],
         f"{PART}.policy.fallback_action"),
        (["policy", "fallback_action"], True,
         f"{PART}.policy.fallback_action"),
        (["policy"], "止める", f"{PART}.policy"),
        (["env_id"], 1, f"{PART}.env_id"),
    ]
    for path, value, key in cases:
        problem = copy.deepcopy(policy_problem())
        node = problem["ml"]["parts"][0]
        for name in path[:-1]:
            node = node[name]
        node[path[-1]] = value
        assert errors_of(problem) == [key], key
    problem = copy.deepcopy(policy_problem())
    problem["ml"]["gpu"] = "はい"
    assert errors_of(problem) == ["ml.gpu"]


def test_simulator_choices() -> None:
    """
    シミュレータのその他の選択肢が, Gymnasium の環境の道でもあると読める
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 選択肢の言葉
    items = {item["id"]: item for item in questions(policy_problem())}
    labels = {choice["value"]: choice["label"]
              for choice in items[f"{PART}.simulator"]["choices"]}
    assert "Gymnasium の環境" in labels["other"]


def test_env_id() -> None:
    """
    環境の登録名の問いが方策で今ある記録から集めない場合だけ要り, 形を
    確かめ, 受け渡しと要約に出ることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 方策で, 集め方が今ある記録でなければ問う (任意の問い)
    items = {item["id"]: item for item in questions(policy_problem())}
    assert items[f"{PART}.env_id"]["applies"]
    assert not items[f"{PART}.env_id"]["required"]
    assert items[f"{PART}.env_id"]["section"] == 5
    logged = answer(policy_problem(), f"{PART}.collect", "log")
    items = {item["id"]: item for item in questions(logged)}
    assert not items[f"{PART}.env_id"]["applies"]
    unanswered = answer(policy_problem(), f"{PART}.collect", None)
    items = {item["id"]: item for item in questions(unanswered)}
    assert items[f"{PART}.env_id"]["applies"]
    items = {item["id"]: item for item in questions(effect_problem())}
    assert not items[f"{PART}.env_id"]["applies"]

    # 答えると受け渡しの policy と要約に出る. 要らなくなると渡さない
    for name in ("CartPole-v1", "ALE/Pong-v5", "Pendulum-v1",
                 "my_package.envs:Arm-v0"):
        problem = answer(policy_problem(), f"{PART}.env_id", name)
        assert not errors_of(problem), name
        assert handoff(problem)["policy"]["env_id"] == name
    assert "方策を学ぶ環境: Gymnasium の my_package.envs:Arm-v0" in (
        describe(problem)
    )
    problem = answer(problem, f"{PART}.collect", "log")
    assert handoff(problem)["policy"]["env_id"] is None
    assert not any(text.startswith("方策を学ぶ環境")
                   for text in describe(problem))

    # 手で書いた空の文字と空白だけの値は答えていない扱い. 前後の空白は除く
    for raw in ("", "   "):
        problem = copy.deepcopy(policy_problem())
        problem["ml"]["parts"][0]["env_id"] = raw
        assert not errors_of(problem), raw
        assert handoff(problem)["policy"]["env_id"] is None
    problem = copy.deepcopy(policy_problem())
    problem["ml"]["parts"][0]["env_id"] = " CartPole-v1 "
    assert not errors_of(problem)
    assert handoff(problem)["policy"]["env_id"] == "CartPole-v1"

    # 登録名の形でない名前は誤り (名前空間に . は使えない)
    for name in ("Cart Pole", "CartPole-v1;rm", "環境", "a/b/c",
                 "a.b/Env-v0"):
        problem = answer(policy_problem(), f"{PART}.env_id", name)
        assert errors_of(problem) == [f"{PART}.env_id"], name


def test_describe_and_version(tmp_path: Path) -> None:
    """
    要約に守りと要る機能が出て, 版を上げずに読み書きできることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 要約
    problem = policy_problem()
    sentences = describe(problem)
    assert any(text.startswith("方策の守り: 行動の上限 各関節")
               and "元の制御" in text and "決める人 設備の責任者" in text
               for text in sentences)
    assert "要る機能: rl, mujoco" in sentences

    # 版は 2 のまま, 読み直しても同じ
    assert FORMAT_VERSION == 2
    path = tmp_path / "problem.yaml"
    save_problem(problem, path)
    loaded = load_problem(path)
    assert loaded["format_version"] == 2
    assert handoff(loaded) == handoff(problem)


def test_features_match_ml_found() -> None:
    """
    要る機能の名前が ML-Found の登録簿にあることを確かめる関数

    ML-Found が入っていなければ飛ばす

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 確かめずに一覧だけ (重いライブラリを読み込まない)
    ml_found = pytest.importorskip("ml_found")
    entries = ml_found.describe_requirements(check=False)
    registered = {entry["name"] for entry in entries}
    names = [item["value"] for item in FEATURES]
    missing = [name for name in names if name not in registered]
    assert not missing, f"ML-Found の登録簿に無い機能: {missing}"


def test_question_lines() -> None:
    """
    問いが答えの欄の形 (multiline) を返し, 名前の問いが 1 行で, 1 行の問い
    の答えの改行は空白になることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 全ての問いが真偽値を持ち, 文でない問いは偽
    from ml_quest.questions import ONE_LINE, QUESTIONS

    items = {item["id"]: item for item in questions(policy_problem())}
    assert all(isinstance(item["multiline"], bool) for item in items.values())
    assert not any(item["multiline"] for item in items.values()
                   if item["kind"] != "text")
    kinds = {question["id"]: question["kind"] for question in QUESTIONS}
    assert all(kinds[key] == "text" for key in ONE_LINE)

    # 名前の問いは 1 行, 文の問いは複数行
    for key in (f"{PART}.env_id", f"{PART}.policy.safety_key",
                f"{PART}.evaluation.name", f"{PART}.policy.approver",
                "issue.owner", "issue.kpi.unit"):
        assert items[key]["multiline"] is False, key
    for key in ("issue.pain", f"{PART}.what", f"{PART}.policy.stop",
                "verification.success", "operation.stop"):
        assert items[key]["multiline"] is True, key

    # 1 行の問いの改行は空白に, 複数行の問いの改行は残す
    problem = answer(policy_problem(), f"{PART}.policy.approver",
                     "設備の\n責任者")
    assert problem["ml"]["parts"][0]["policy"]["approver"] == "設備の 責任者"
    problem = answer(problem, "issue.pain", "廃棄が多い\n品切れも出る")
    assert problem["issue"]["pain"] == "廃棄が多い\n品切れも出る"
