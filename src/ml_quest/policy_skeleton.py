import json
import math
from string import Template
from typing import Any, Callable, Optional

from ml_quest.check import env_id_of
from ml_quest.handoff import describe
from ml_quest.problem import active_value
from ml_quest.questions import FALLBACKS, PART

# 学習のスクリプトが読み込む機能の, ML-Found の登録簿の名前 (Gymnasium)
POLICY_LIBRARIES = ["rl"]

# 骨組みの方策の行動の空間 (ML-Found の RolloutLoader と方策の一式が扱える
# もの. 番号で選ぶ行動と連続の行動)
SPACES = ("Discrete", "Box")

# 方策のモデル (観測から行動のスコアか平均と, 状態の価値)
POLICY_MODEL = '''"""
方策のモデル (ML-Quest が問題の設定から書き出した骨組み)

train.py が観測と行動の空間から大きさを決めて作り, 方策の一式の metadata
に組み立て方 (model.PolicyNet と引数) を残す. 印 (TODO) の所を書き足す
"""
import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(
        self, obs_dim: int, hidden_dim: int, action_dim: int,
        continuous: bool
    ):
        # 親クラスのコンストラクタを呼び出す
        super(PolicyNet, self).__init__()

        # 観測の次元数, 隠れ層の次元数, 行動の数 (連続の行動は要素の数),
        # 連続の行動かを保存
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # TODO: 課題に合う大きさと形に直す. 今は共通の隠れ層 2 つから, 方策
        # の頭 (番号で選ぶ行動はスコア, 連続の行動は平均) と価値の頭
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.policy = nn.Linear(self.hidden_dim, self.action_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.tanh = nn.Tanh()

        # 連続の行動の, 観測によらない対数の標準偏差 (ML-Found の決まり)
        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def body(self, x: torch.Tensor) -> torch.Tensor:
        """
        方策と価値で共通の隠れ層を求める関数

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 観測の次元数) の観測

        Returns
        ----------
        hidden: torch.Tensor
            (バッチ, 隠れ層の次元数) の特徴
        """
        # 全結合と活性化を 2 段
        hidden = self.tanh(self.fc2(self.tanh(self.fc1(x))))

        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数 (書き出した方策もこの出力だけを使う)

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 観測の次元数) の観測

        Returns
        ----------
        output: torch.Tensor
            (バッチ, 行動の数) の行動ごとのスコアか, 連続の行動の平均
        """
        # 方策の頭
        output = self.policy(self.body(x))

        return output

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """
        状態の価値を求める関数 (ActorCriticStep が使う)

        Parameters
        ----------
        x: torch.Tensor
            (バッチ, 観測の次元数) の観測

        Returns
        ----------
        values: torch.Tensor
            (バッチ, 1) の状態の価値
        """
        # 価値の頭
        values = self.critic(self.body(x))

        return values
'''

# 自作の環境の雛形 (環境の登録名を答えていない場合)
POLICY_ENV = Template('''"""
自作の環境の雛形 (ML-Quest が問題の設定から書き出した骨組み)

問題の設定で Gymnasium の環境の登録名を答えていないので, train.py はこの
ProjectEnv で方策を学ぶ. 今は動くことを確かめるための小さな環境 (台の上の
点を 0 に寄せる) なので, 印 (TODO) の所を課題の環境 (自作のシミュレータ,
実機とのつなぎ) に書き換える. 実機につなぐ場合は, 外れたときの損に気を
付ける (学習は環境で行動を試す)
"""
from typing import Any, Optional

import gymnasium
from gymnasium import spaces
import numpy as np

# 安全の違反を知らせる info の名前 (問題の設定の答え. 学習と評価で違反の
# 数を数える)
SAFETY_KEY = $safety_key


class ProjectEnv(gymnasium.Env):
    def __init__(self, max_steps: int = 100):
        # 親クラスのコンストラクタを呼び出す
        super(ProjectEnv, self).__init__()

        # 1 エピソードの歩数の上限を保存
        self.max_steps = max_steps

        # TODO: 課題の観測と行動の空間に直す. 今は観測が (位置, 速さ),
        # 行動が 0: 左に押す, 1: 押さない, 2: 右に押す
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # 今の状態と歩数
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """
        エピソードの初めの観測を返す関数

        Parameters
        ----------
        seed: Optional[int] = None
            環境の乱数の種 (最初のエピソードだけ渡される)
        options: Optional[dict] = None
            使わない (Gymnasium の決まり)

        Returns
        ----------
        result: tuple[np.ndarray, dict]
            初めの観測と info
        """
        # 種は親クラスが np_random に (ML-Found は学習の再開でこの乱数を
        # 戻す). TODO: 課題の初めの状態に直す
        super(ProjectEnv, self).reset(seed=seed)
        position = self.np_random.uniform(-1.0, 1.0)
        self.state = np.array([position, 0.0], dtype=np.float32)
        self.steps = 0
        result = (self.state.copy(), {})

        return result

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        行動を 1 歩進める関数

        Parameters
        ----------
        action: Any
            行動の番号

        Returns
        ----------
        result: tuple[np.ndarray, float, bool, bool, dict]
            観測, 報酬, 終了 (台から落ちた), 打ち切り (歩数の上限), info
            (安全の違反の信号)
        """
        # TODO: 課題の動きと報酬に直す. 今は押した向きに速さが変わり, 台の
        # 上にいる歩は中心に近いほど大きい正の報酬 (0.5 から 1), 台 (位置の
        # 絶対値 2) から落ちたら大きな負の報酬で終える (早く落ちると得に
        # ならず, 中心に寄せて長く留まるのが学ばれる)
        position, velocity = (float(value) for value in self.state)
        velocity = 0.9 * velocity + 0.05 * (int(action) - 1)
        position = position + velocity
        self.state = np.array([position, velocity], dtype=np.float32)
        self.steps += 1
        terminated = abs(position) > 2.0
        reward = -10.0 if terminated else 1.0 - 0.25 * abs(position)
        truncated = self.steps >= self.max_steps

        # 台の端に近い (位置の絶対値 1.5 より外) 歩を安全の違反として知らせる
        info = {SAFETY_KEY: 1.0 if abs(position) > 1.5 else 0.0}
        result = (self.state.copy(), reward, terminated, truncated, info)

        return result
''')

# 学習のスクリプトの頭 (読み込みと, 問題の設定から決まる既定値). 答えの
# 文字は JSON の文字列で埋め込み
POLICY_HEAD = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した学習のスクリプト

Gymnasium の形の環境で, 観測から行動を続けて選ぶ方策を学び, 問題の設定の
守りと一緒に書き出して, 守りを当てたまま環境で確かめる骨組みなので, 印
(TODO) の所を書き足す
"""
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

from ml_found import (
    Experiment, MissingRequirementError, Trainer, load_project_config,
    require, save_evaluation, start_experiment, write_task
)
from model import PolicyNet

# このプロジェクトのディレクトリ (ML-Found のプロジェクト)
PROJECT_DIR = Path(__file__).resolve().parent

# 問題の名前と, 方策が出すもの
PROBLEM_NAME = $name
PREDICT = $predict

# task.yaml の作り方 (問題の設定の答え)
METHOD = $method

# 方策を使ってよいかと守りの範囲を決める人 (問題の設定の答え)
APPROVER = $approver

# 問題の設定の守りの文 (行動の上限と止める条件. config の guards に数で
# 書く元)
ACTION_LIMITS = $action_limits
STOP = $stop

# 守りの結果の種類 (ML-Found の evaluate_policy の列) と, 行動を返さずに
# エピソードを終える結果
STATUSES = ("ok", "clipped", "fallback", "stop", "handover")
STOP_STATUSES = ("stop", "handover")

# 守る指標の表の列ごとの向き
GUARDED_BETTER = {"reward": "higher", "fallback": "lower", "stop": "lower",
                  "handover": "lower"}

# 学習の方法の名前
ALGORITHMS = ("auto", "reinforce", "actor_critic")

# 守りの数が無いことの知らせ
NO_LIMITS = ("config の guards に行動の範囲も止める条件も数で書いていない"
             "ので, 外れたときの動きは観測やモデルの出力が読めないときだけ"
             "です. 問題の設定の守り (ACTION_LIMITS と STOP) を観測と行動"
             "の空間に合わせて数にしてください")
'''

# 環境, 守り, 学習の流れ
POLICY_MAIN = '''

def make_environment(cfg: Any) -> Any:
    """
    config の環境を作る関数

    Parameters
    ----------
    cfg: Any
        config (env.id に Gymnasium の環境の登録名. null なら env.py の
        ProjectEnv)

    Returns
    ----------
    env: Any
        Gymnasium の形の環境
    """
    # 追加の依存 rl が要る関数なので, 確認の後に関数の中で読み込み
    from ml_found.rl import make_env

    # 登録名があれば Gymnasium から, 無ければ自作の環境
    if cfg.env.id:
        env = make_env(cfg.env.id)
        return env
    from env import ProjectEnv
    env = ProjectEnv()

    return env


def environment_name(cfg: Any) -> str:
    """
    記録に残す環境の名前を返す関数

    Parameters
    ----------
    cfg: Any
        config (env.id)

    Returns
    ----------
    name: str
        Gymnasium の環境の登録名か, 自作の環境の env:ProjectEnv
    """
    # 自作の環境は Gymnasium の entry_point の書き方 (登録名ではない)
    name = cfg.env.id or "env:ProjectEnv"

    return name


def space_kind(env: Any) -> str:
    """
    行動の空間の種類を返す関数

    Parameters
    ----------
    env: Any
        環境

    Returns
    ----------
    kind: str
        Discrete (番号で選ぶ行動) か Box (連続の行動)

    Raises
    ----------
    ValueError
        他の空間の場合 (ML-Found の強化学習が扱えない)
    """
    # 空間のクラスの名前
    kind = type(env.action_space).__name__
    if kind not in ("Discrete", "Box"):
        raise ValueError(
            f"行動の空間 {kind} は扱えません (Discrete か Box にしてください)"
        )

    return kind


def model_args(env: Any, cfg: Any) -> dict:
    """
    観測と行動の空間から, 方策のモデルの引数を決める関数

    Parameters
    ----------
    env: Any
        環境
    cfg: Any
        config (model.hidden_dim)

    Returns
    ----------
    args: dict
        PolicyNet の引数 (方策の一式の metadata にも残す)
    """
    # 番号で選ぶ行動は行動の数, 連続の行動は要素の数
    continuous = space_kind(env) == "Box"
    action_dim = (int(env.action_space.shape[0]) if continuous
                  else int(env.action_space.n))
    args = {"obs_dim": int(env.observation_space.shape[0]),
            "hidden_dim": int(cfg.model.hidden_dim),
            "action_dim": action_dim, "continuous": continuous}

    return args


def algorithm_of(env: Any, cfg: Any) -> str:
    """
    学習の方法を決める関数

    Parameters
    ----------
    env: Any
        環境
    cfg: Any
        config (algorithm. auto なら空間から)

    Returns
    ----------
    algorithm: str
        reinforce か actor_critic (auto は番号で選ぶ行動なら reinforce,
        連続の行動なら actor_critic)
    """
    # 連続の行動は価値の関数で利得を見積もる方が安定しやすい
    algorithm = str(cfg.algorithm)
    if algorithm == "auto":
        algorithm = ("actor_critic" if space_kind(env) == "Box"
                     else "reinforce")
    if algorithm not in ("reinforce", "actor_critic"):
        raise ValueError(
            f"algorithm の {algorithm} は知らない方法です (auto, reinforce, "
            "actor_critic)"
        )

    return algorithm


def make_step(algorithm: str, cfg: Any) -> Any:
    """
    学習の方法の step を作る関数

    Parameters
    ----------
    algorithm: str
        reinforce か actor_critic
    cfg: Any
        config (loss の係数)

    Returns
    ----------
    step: Any
        ML-Found の ReinforceStep か ActorCriticStep
    """
    # 追加の依存 rl が要るので, 関数の中で読み込み
    from ml_found.rl import ActorCriticStep, ReinforceStep

    if algorithm == "actor_critic":
        step = ActorCriticStep(
            value_coef=cfg.loss.value_coef,
            entropy_coef=cfg.loss.entropy_coef,
            gae_lambda=cfg.loss.gae_lambda
        )
        return step
    step = ReinforceStep(entropy_coef=cfg.loss.entropy_coef)

    return step


def optional_list(value: Any) -> Optional[list]:
    """
    config の並びを, 守りに渡す並びにする関数

    Parameters
    ----------
    value: Any
        config の値 (null か並び)

    Returns
    ----------
    items: Optional[list]
        並び (null なら None. 要素の null は制限なし)
    """
    # OmegaConf の並びを Python の並びに
    items = None if value is None else list(value)

    return items


def build_guards(env: Any, cfg: Any) -> dict:
    """
    config の守りから, 方策の一式に書く守りを作る関数

    Parameters
    ----------
    env: Any
        環境 (観測と行動の空間)
    cfg: Any
        config (guards)

    Returns
    ----------
    guards: dict
        ML-Found の make_guards の守り
    """
    # 追加の依存 rl が要るので, 関数の中で読み込み
    from ml_found.rl import make_guards

    setting = cfg.guards
    action = setting.fallback_action
    guards = make_guards(
        env.observation_space, env.action_space,
        action_low=optional_list(setting.action_low),
        action_high=optional_list(setting.action_high),
        allowed_actions=optional_list(setting.allowed_actions),
        observation_low=optional_list(setting.observation_low),
        observation_high=optional_list(setting.observation_high),
        fallback=setting.fallback,
        fallback_action=(list(action) if hasattr(action, "__len__")
                         else action)
    )

    return guards


def guards_numbered(guards: dict) -> bool:
    """
    守りに数の範囲が 1 つでもあるかを返す関数

    Parameters
    ----------
    guards: dict
        make_guards の守り

    Returns
    ----------
    numbered: bool
        行動の範囲か止める条件の要素に, null でない値があるか
    """
    # 範囲の項目の要素のどれかが数
    keys = ("action_low", "action_high", "allowed_actions",
            "observation_low", "observation_high")
    numbered = any(value is not None for key in keys
                   for value in (guards.get(key) or []))

    return numbered


def describe_guards(guards: dict) -> str:
    """
    守りを 1 行の文にする関数 (評価の表と train.log のため)

    Parameters
    ----------
    guards: dict
        make_guards の守り

    Returns
    ----------
    text: str
        守りの項目と値 (JSON の書き方. 制限なしは null)
    """
    # 外れたときの動きと, 他の項目
    fallback = guards["fallback"]
    items = [f"{key} {json.dumps(value)}" for key, value in guards.items()
             if key != "fallback"]
    items.append(f"fallback {fallback['kind']}"
                 + (f" ({fallback['action']})"
                    if fallback["kind"] == "action" else ""))
    text = ", ".join(items)

    return text


def summarize(rows: list[dict]) -> dict:
    """
    evaluate_policy のエピソードごとの行を, 1 つの要約にする関数

    Parameters
    ----------
    rows: list[dict]
        evaluate_policy の行

    Returns
    ----------
    summary: dict
        reward (報酬の和の平均), steps (歩数の合計), stopped (stop か
        handover で終えたエピソードの数), 守りの結果ごとの数の合計
    """
    # エピソードの平均と合計
    summary = {
        "reward": sum(row["reward"] for row in rows) / len(rows),
        "steps": sum(row["length"] for row in rows),
        "stopped": sum(row["end"] in STOP_STATUSES for row in rows),
    }
    for status in STATUSES:
        summary[status] = sum(row[status] for row in rows)

    return summary


def evaluate_guards(
    model: Any, env: Any, cfg: Any, experiment: Experiment, args: dict,
    guards: dict, metadata: dict
) -> list[dict]:
    """
    方策を守りと一緒に書き出し, 読み直して守りを当てたまま環境で動かす
    関数

    Parameters
    ----------
    model: Any
        学習した後の方策
    env: Any
        守り付きの評価の環境 (観測と行動の空間もここから)
    cfg: Any
        config (evaluation)
    experiment: Experiment
        実験 (policy/ に書き出し, train.log に記録)
    args: dict
        PolicyNet の引数 (読み直しに使う)
    guards: dict
        守り
    metadata: dict
        方策の一式の metadata

    Returns
    ----------
    rows: list[dict]
        エピソードごとの evaluate_policy の行 (guards の文を足す)
    """
    # 追加の依存 rl が要るので, 関数の中で読み込み
    from ml_found.rl import evaluate_policy, export_policy, load_policy

    # 実験の中の policy/ に書き出し, 新しいモデルに読み直して (重みは
    # 一式から) 守りを当てたエピソードを動かす
    place = Path(experiment.run_dir) / "policy"
    export_policy(place, model, env.observation_space, env.action_space,
                  guards, metadata=metadata)
    policy = load_policy(place, PolicyNet(**args))
    episodes = evaluate_policy(
        policy, env, episodes=cfg.evaluation.guarded_episodes,
        seed=cfg.evaluation.guarded_seed,
        max_steps=cfg.env.max_episode_steps
    )
    text = describe_guards(guards)
    rows = [{"guards": text, **row} for row in episodes]
    summary = summarize(episodes)
    message = (
        f"守り付きの評価: 報酬の和の平均 {summary['reward']:.2f}, 歩数 "
        f"{summary['steps']}, 止めたエピソード {summary['stopped']}, "
        + ", ".join(f"{status} {summary[status]}" for status in STATUSES)
    )
    experiment.logger.info(message)
    print(message)

    return rows


def main() -> None:
    """
    環境で方策を学び, 守りと一緒に書き出して守り付きで確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # config. 学習の方法の名前の確かめ
    cfg, diff = load_project_config(str(PROJECT_DIR))
    if str(cfg.algorithm) not in ALGORITHMS:
        print(f"config の algorithm の {cfg.algorithm} は知らない方法です "
              f"({', '.join(ALGORITHMS)} のどれかにしてください)")
        sys.exit(1)

    # 学習用, 検証用, 守り付きの評価用の環境は別に作る. 守りも学習の前に
    # 作る. Gymnasium が足りないか守りが誤りなら, 出力先を作る前に止める
    try:
        require("rl")
        train_env = make_environment(cfg)
        valid_env = make_environment(cfg)
        guard_env = make_environment(cfg)
        guards = build_guards(guard_env, cfg)
        args = model_args(train_env, cfg)
    except MissingRequirementError as error:
        print(error)
        sys.exit(1)
    except ValueError as error:
        print(f"config の guards を直してください ({error}). 外れたときの"
              "動きが action なら, 番号で選ぶ行動では guards.fallback_action"
              " に決めた行動の番号を書きます")
        sys.exit(1)
    from ml_found.rl import EvaluationLoader, Exploration, RolloutLoader
    algorithm = algorithm_of(train_env, cfg)
    name = environment_name(cfg)
    numbered = guards_numbered(guards)

    # 実験. config の requires が足りなければ, start_experiment が出力先を
    # 作る前に止まるので, 入れ方を示して終了
    try:
        experiment = start_experiment(cfg, diff, str(PROJECT_DIR))
    except MissingRequirementError as error:
        print(error)
        sys.exit(1)

    # 何の実験か. 方策は受け持ちでは効果を見積もる (effect), 出力の形は
    # 行動 (action), 正解の出どころは環境の報酬 (environment)
    write_task(experiment, {
        "role": "effect", "output": "action", "what": PREDICT,
        "target": {"kind": "reward",
                   "origin": {"type": "environment", "name": name}},
        "method": METHOD,
    })
    experiment.logger.info(
        f"問題: {PROBLEM_NAME}. ${predict_log}環境: {name}, 行動の空間: "
        f"{space_kind(train_env)}, 学習の方法: {algorithm}, 守り: "
        f"{describe_guards(guards)}"
    )
    if not numbered:
        experiment.logger.warning(NO_LIMITS)

    # 1 エポックは rollout.episodes のエピソード (batch_episodes ごとに
    # 更新), 検証は別の種のエピソードを最もスコアの高い行動 (連続の行動は
    # 平均) で
    model = PolicyNet(**args)
    train_loader = RolloutLoader(
        train_env, model, episodes=cfg.rollout.episodes,
        batch_episodes=cfg.rollout.batch_episodes, gamma=cfg.rollout.gamma,
        exploration=Exploration(cfg.rollout.temperature,
                                cfg.rollout.epsilon),
        seed=cfg.experiment.seed,
        max_episode_steps=cfg.env.max_episode_steps,
        safety_key=cfg.safety_key
    )
    valid_loader = EvaluationLoader(
        valid_env, model, episodes=cfg.evaluation.episodes,
        seed=cfg.env.valid_seed,
        max_episode_steps=cfg.env.max_episode_steps,
        safety_key=cfg.safety_key
    )
    trainer = Trainer(model, experiment, make_step(algorithm, cfg))
    trainer.fit(train_loader, valid_loader)

    # 実験だけで見直せるように, 問題の設定の写し (元の問題の設定の場所
    # origin も残る)
    run_dir = Path(experiment.run_dir)
    shutil.copy(PROJECT_DIR / "problem.yaml", run_dir / "problem.yaml")

    # 結果の表示. 書き出す重みは, 最良のモデルに戻した場合だけ最良の
    # エポック (select_best: false などは最後)
    best = trainer.final_weights == "best"
    epoch = trainer.checkpoint.best_epoch if best else trainer.epoch
    weights = "最良のモデル" if best else "最後のエポックの重み"
    print(f"保存先: {run_dir}")
    print(f"書き出す重み: {weights} (エポック {epoch})")

    # 守りと一緒に方策の一式 (policy/) に書き出し, 読み直して守り付きで
    # 動かした結果を守る指標の表に
    # metadata には, 守りを数にしたか (guards_set) と問題の設定の守りの
    # 文も残す (数が無いまま書き出した一式と分かるように)
    metadata = {
        "env_id": cfg.env.id or None, "algorithm": algorithm,
        "epoch": epoch,
        "model": {"class": "model.PolicyNet", "args": args},
        "approver": APPROVER, "guards_set": numbered,
        "guards_text": {"action_limits": ACTION_LIMITS, "stop": STOP},
    }
    if not cfg.env.id:
        metadata["entry_point"] = name
    rows = evaluate_guards(trainer.model, guard_env, cfg, experiment, args,
                           guards, metadata)
    save_evaluation(
        experiment, "guarded_episodes", rows, "environment",
        description="書き出した方策を読み直し, 守りを当てて環境で動かした"
        f"エピソード ({weights}). 1 行が 1 エピソードで, 歩ごとの守りの"
        "結果の数. 実機での結果は別に確かめる"
        + ("" if numbered else f". {NO_LIMITS}"),
        guard=True, epoch=epoch, better=GUARDED_BETTER
    )
    for env in (train_env, valid_env, guard_env):
        env.close()
    print(f"方策の一式: {run_dir / 'policy'} (使ってよいかと守りの範囲は "
          f"{APPROVER or '承認する人'} が決める)")


if __name__ == "__main__":
    main()
'''

# config の基準 (ML-Found のプロジェクトの形)
POLICY_CONFIG = Template('''# ML-Found が使う設定 (ML-Quest が問題の設定から書き出した骨組み)
$requires
experiment:
  name: $experiment
  seed: 0
  device: cpu

optimizer:
  name: Adam
  params:
    lr: 0.01

trainer:
  epochs: 20
  # 問題の設定の「最良を選ぶ指標」. 強化学習の valid/loss は評価の
  # エピソードの報酬の和の平均の符号を反転した値 (小さいほど報酬が大きい)
  monitor: $monitor
  mode: $mode
  # 評価の報酬が上限で並んだ場合は, 学習のエピソードの報酬が大きい方
  monitor_tiebreak: train/reward
  monitor_tiebreak_mode: max

# プロジェクト固有の設定
env:
  # Gymnasium の環境の登録名 (問題の設定の答え). null なら env.py の
  # ProjectEnv
  id: $env_id
  # 検証 (評価のエピソード) の乱数の種と, 1 エピソードの歩数の上限 (null
  # なら環境の上限)
  valid_seed: 10000
  max_episode_steps: null

# 学習の方法 (auto は番号で選ぶ行動なら reinforce, 連続の行動なら
# actor_critic)
algorithm: auto

rollout:
  # 1 エポックのエピソード数と, 1 回の更新のエピソード数
  episodes: 20
  batch_episodes: 5
  gamma: 0.99
  temperature: 1.0
  epsilon: 0.0

evaluation:
  # 学習中の検証のエピソード数
  episodes: 10
  # 書き出した方策を守り付きで動かすエピソード数と乱数の種
  guarded_episodes: 10
  guarded_seed: 20000

loss:
  entropy_coef: 0.01
  # actor_critic だけで使う係数
  value_coef: 0.5
  gae_lambda: 0.95

model:
  hidden_dim: 64

# 安全の違反を知らせる info の名前 (問題の設定の答え. 無ければ cost)
safety_key: $safety_key

# 書き出す方策の守り (policy/). 範囲は$approver決めて承認する. 問題の
# 設定の答えは文なので, 観測と行動の空間 (要素の並び) に合わせて数の並び
# にする (null は制限なし)
guards:
$limits_comment
  action_low: null
  action_high: null
  allowed_actions: null
$stop_comment
  observation_low: null
  observation_high: null
  # 外れたときの動き (stop, handover, action. 問題の設定の答え) と, action
  # の場合の行動 (番号で選ぶ行動なら要る)$fallback_comment
  fallback: $fallback
  fallback_action: $fallback_action
''')


def fallback_value(value: Any) -> Any:
    """
    問題の設定の決めた行動を, config に書ける数か数の並びにする関数

    Parameters
    ----------
    value: Any
        決めた行動の答え (文, 数か数の並び)

    Returns
    ----------
    action: Any
        数か数の並び (読めない文と空なら None)
    """
    # 数と数の並びはそのまま, 文は数か , 区切りの数として読む
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        numbers = [fallback_value(item) for item in value]
        return numbers if numbers and None not in numbers else None
    pieces = [piece.strip() for piece in str(value).split(",")]
    numbers = []
    for piece in pieces:
        try:
            number = float(piece)
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        numbers.append(int(number) if number.is_integer() else number)
    action = numbers[0] if len(numbers) == 1 else numbers

    return action


def yaml_comment(title: str, text: Optional[str]) -> str:
    """
    問題の設定の文の答えを, config の 1 行ずつのコメントにする関数

    Parameters
    ----------
    title: str
        項目の名前
    text: Optional[str]
        文の答え (無ければ「答えなし」)

    Returns
    ----------
    lines: str
        「  # 」で始まる行
    """
    # 改行は空白にし, 70 文字ほどで折り返す
    flat = " ".join(str(text).split()) if text else "答えなし"
    body = f"{title} (問題の設定): {flat}"
    width = 70
    lines = "\n".join(f"  # {body[start:start + width]}"
                      for start in range(0, len(body), width))

    return lines


def policy_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 方策を学ぶスクリプトを作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    literal: Callable
        値を Python の式にする関数 (skeleton.literal)

    Returns
    ----------
    script: str
        train.py の中身
    """
    # 頭と学習の流れ
    head = Template(POLICY_HEAD).substitute(
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
        method=literal(active_value(problem, f"{PART}.method")),
        approver=literal(active_value(problem, f"{PART}.policy.approver")),
        action_limits=literal(
            active_value(problem, f"{PART}.policy.action_limits")
        ),
        stop=literal(active_value(problem, f"{PART}.policy.stop")),
    )
    # train.log の初めの行の方策が出すもの (空なら省く)
    made = ("方策が出すもの: {PREDICT}. "
            if active_value(problem, f"{PART}.what") else "")
    script = head + Template(POLICY_MAIN).substitute(predict_log=made)

    return script


def policy_env(stages: dict) -> Optional[str]:
    """
    自作の環境の雛形を作る関数

    Parameters
    ----------
    stages: dict
        受け渡し (policy の env_id と safety_key)

    Returns
    ----------
    script: Optional[str]
        env.py の中身 (環境の登録名を答えていれば None)
    """
    # 登録名があれば Gymnasium の環境を使う
    policy = stages["policy"]
    if policy["env_id"]:
        return None
    script = POLICY_ENV.substitute(
        safety_key=json.dumps(policy["safety_key"], ensure_ascii=False)
    )

    return script


def policy_config(
    problem: dict, experiment: str, stages: dict, requires: str
) -> str:
    """
    方策を学ぶ config の基準を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    experiment: str
        実験の名前
    stages: dict
        受け渡し (training の監視と, policy の守り)
    requires: str
        一番上の requires の行 (skeleton.requires_text)

    Returns
    ----------
    text: str
        configs/default/train.yaml の中身
    """
    # 守りの答え (文はコメント, 外れたときの動きと決めた行動は値)
    policy = stages["policy"]
    training = stages["training"]
    approver = policy["approver"]
    action = fallback_value(policy["fallback_action"])
    fallback_comment = ""
    if policy["fallback"] == "action" and action is None:
        fallback_comment = "\n" + yaml_comment(
            "決めた行動", policy["fallback_action"]
        ) + ("\n  # TODO: 番号で選ぶ行動なら, 決めた行動の番号を "
             "fallback_action に書く")
    text = POLICY_CONFIG.substitute(
        requires=requires,
        experiment=json.dumps(experiment, ensure_ascii=False),
        monitor=training["monitor"], mode=training["mode"],
        env_id=json.dumps(policy["env_id"]) if policy["env_id"] else "null",
        safety_key=json.dumps(policy["safety_key"], ensure_ascii=False),
        approver=(f" {' '.join(str(approver).split())} が" if approver
                  else "人が"),
        limits_comment=yaml_comment("行動の上限", policy["action_limits"])
        + "\n  # TODO: 連続の行動は action_low と action_high, 番号で選ぶ"
        "行動は allowed_actions",
        stop_comment=yaml_comment("止める条件", policy["stop"])
        + "\n  # TODO: 観測の要素ごとの範囲",
        fallback_comment=fallback_comment,
        fallback=policy["fallback"] or "stop",
        fallback_action=json.dumps(action) if action is not None else "null",
    )

    return text


def policy_name(problem: dict) -> str:
    """
    方策の骨組みの, 実験の名前の元を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)

    Returns
    ----------
    raw: str
        環境の登録名と policy (自作の環境なら env_policy)
    """
    # 目的の列とタスクの代わり
    raw = f"{env_id_of(problem) or 'env'}_policy"

    return raw


def policy_readme(problem: dict, stages: dict) -> str:
    """
    方策を学ぶプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    policy = stages["policy"]
    names = {item["value"]: item["label"] for item in FALLBACKS}
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, "
             "ML-Found のプロジェクトの骨組み (Gymnasium の形の環境で方策を"
             "学び, 守りと一緒に書き出す).", "", "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1 から 4. 環境, 守り, 学習, 推論
    environment = (
        f"1. **環境**: Gymnasium の `{policy['env_id']}` (問題の設定の答え."
        " config の `env.id`). 学習用, 検証用, 守り付きの評価用に別々に作る"
        if policy["env_id"] else
        "1. **環境**: 問題の設定で Gymnasium の環境の登録名を答えていないの"
        "で, 自作の環境の雛形 `env.py` (`ProjectEnv`. 印 TODO) を書き出した. "
        "今は動くことを確かめるだけの小さな環境なので, 観測と行動の空間, "
        "初めの状態, 動き, 報酬を課題の環境 (自作のシミュレータ, 実機との"
        "つなぎ) に書き換える. 実機につなぐなら, 学習は行動を試すので外れた"
        "ときの損に気を付ける. Gymnasium に登録した環境なら, config の "
        "`env.id` にその名前を書けば `env.py` は使わない"
    )
    approver = policy["approver"] or "承認する人"
    lines += [
        "", "## 次にすること", "",
        "学習する環境には ML-Found と Gymnasium (ML-Found の追加の依存 `rl`)"
        " を入れる (config の `requires` で足りなければ入れ方を示して止まる)"
        ". データの分析の段階は無い (行動の結果は環境とのやり取りで集める).",
        "",
        environment,
        "2. **守り**: config の `guards` に, 問題の設定の守り (文) を観測と"
        "行動の空間に合わせた数の並びで書く (行動の上限: "
        f"{policy['action_limits'] or '答えなし'}. 止める条件: "
        f"{policy['stop'] or '答えなし'}). 外れたときの動きは "
        f"`{policy['fallback'] or 'stop'}` "
        f"({names.get(str(policy['fallback']), '止める')}). 範囲と方策を"
        f"使ってよいかは {approver} が決める. 安全の違反は環境が info の "
        f"`{policy['safety_key']}` に 0 より大きい値を入れた歩で数える",
        "3. **学習** (ML-Found, ML-Dashboard): `python train.py`. 方策の"
        "モデルは `model.py` の `PolicyNet` (印 TODO). 学習の方法は config "
        "の `algorithm` (`auto` は番号で選ぶ行動なら REINFORCE の "
        "`ReinforceStep`, 連続の行動なら `ActorCriticStep`). 検証用の "
        f"`{stages['training']['monitor']}` (評価のエピソードの報酬の和の"
        "平均の符号を反転した値) が最も良いモデルを選ぶ. 結果は "
        "`outputs/<実験の名前>/<日付>/<時刻>/` に残る (`task.yaml`, 学習の"
        "記録)",
        "4. **方策の一式と守り付きの評価** (ML-Inference, ML-Provide): 学習"
        "の後に, 守りと一緒に実験の中の `policy/` に方策の一式を書き出し, "
        "読み直して守りを当てたまま環境で動かした結果を守る指標の表 "
        "`evaluations/guarded_episodes.json` に書く. 一式は ML-Inference の "
        "`load_policy(<policy/>, PolicyNet(**args))` で読め (組み立て方は "
        "`policy.json` の `metadata.model`. `model.PolicyNet` を読むには, "
        "このプロジェクトのディレクトリを `sys.path` に入れる), 観測から"
        "守りを当てた行動を返す. 守りを数にしていない (`metadata` の "
        "`guards_set` が false) 一式は, 範囲の守りが入っていない. "
        "実機での実行は道具の外で, 環境での結果は実機での結果と別に確かめる",
        "", "`train.py`, `model.py`"
        + (", `env.py`" if not policy["env_id"] else "")
        + " の印 (TODO) の所は書き足す. 問題の設定を変えたら, 書き出し直す"
        "か手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
