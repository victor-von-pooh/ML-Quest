from ml_quest.problem import active_value, upgrade_problem
from ml_quest.questions import PART

# 問題の設定から要るかが決まる機能. 名前は ML-Found の登録簿
# (describe_requirements) の名前と同じ (ML-Found には依存しないので,
# 名前が登録簿にあるかはテストで確認)
FEATURES = [
    {"value": "rl", "label": "強化学習 (環境とのやり取り. Gymnasium)"},
    {"value": "mujoco", "label": "MuJoCo (物理のシミュレータ)"},
    {"value": "ros2", "label": "ROS2 (機械とのやり取り)"},
    {"value": "cuda", "label": "CUDA (GPU で学習)"},
]

# 試して集める集め方 (環境とのやり取りが要る)
TRIAL_COLLECTS = ["online", "simulator"]


def required_features(problem: dict) -> list[str]:
    """
    問題の設定の答えから, 要る機能の名前を返す関数

    答えの途中でも返す (確かめの誤りがあっても, 画面が早めに入れ方の
    カードを出せるように)

    Parameters
    ----------
    problem: dict
        問題の設定

    Returns
    ----------
    names: list[str]
        要る機能の名前 (FEATURES の順. 要らなくなった問いの古い答えは
        使わない)

    Raises
    ----------
    ValueError
        版が 1 以上の整数でないか, 知っている版より新しい場合
    """
    # 版 1 は先に版 2 へ移行
    problem = upgrade_problem(problem)
    value = active_value

    # 答えごとに要る機能
    needed = {
        "rl": value(problem, f"{PART}.collect") in TRIAL_COLLECTS,
        "mujoco": value(problem, f"{PART}.simulator") == "mujoco",
        "ros2": value(problem, f"{PART}.ros2") is True,
        "cuda": value(problem, "ml.gpu") is True,
    }
    names = [item["value"] for item in FEATURES if needed[item["value"]]]

    return names
