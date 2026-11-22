import copy
from pathlib import Path

import pytest
import yaml

from ml_quest import (
    LEGACY_KEYS, answer, check, describe, example_problem, get_value,
    handoff, is_ready, load_problem, make_project, new_problem,
    project_error, questions, recipe_from, save_problem, upgrade_problem
)
from ml_quest.problem import active_value, set_value

# 版 1 の問題の設定 (版 1 の全ての項目と, 知らない項目)
V1_TEXT = """
format_version: 1
name: 総菜の作る量
created_at: "2026-11-17T21:30:00+09:00"
updated_at: "2026-11-17T21:45:00+09:00"
issue:
  owner: 店長
  pain: 総菜の廃棄が多く, 夕方には品切れも出る
  current: 前の週の同じ曜日と同じ数を作る
  kpi: {name: 廃棄と欠品の損, unit: 円 / 月, current: 1200000,
        target: 960000, better: lower}
decomposition:
  - {cause: 作る量が勘で決まる, lever: 作る量を予測から決める, ml: true}
  - {cause: 値引きが遅い, lever: 値引きの時刻を決める, ml: false}
ml:
  use: true
  reason: 品と日による売れ方の違いが大きい
  predict: 翌日の品ごとの売れる数
  target: sales
  task: forecasting
  row: 店, 品, 日ごとの 1 行
  positive: null
  group_column: shop_group
  split_time_column: null
  time_column: date
  series_by: [store, item]
  available_lag: 2
  horizon: 1
  baseline: 前の週の同じ曜日と同じ数
  consumer: system
  downstream: 発注の仕組みが品ごとの数を求める
  constraints: [データを社外に出さない]
  future_field: 残す
information:
  columns: [date, store, item, sales, promo, customers, shop_group]
  known: [store, item, promo]
  unknown: [customers]
  forbidden: []
action:
  who: 店長
  when: 前の日の 17 時
  what: 品ごとの作る量
  decision: {type: quantity, under_cost: 200, over_cost: 150}
  basis: 欠品は粗利 200 円, 廃棄は原価 150 円
evaluation:
  metric: mae
  link: 誤差が小さいほど損が減る
verification:
  method: both
  backtest: 過去 3 か月
  trial: 1 店で 4 週間
  success: 損が 2 割少ない
operation:
  review: 月に 1 回
  stop: 損が悪い月が 2 回続いたら止める
"""

# 受け持ちごとに, 型に依らない答えと最小の続く答え
ROLE_ANSWERS = {
    "generate": [("unit", "pair"), ("output", "image"),
                 ("target.source", "files"), ("method", "finetune"),
                 ("target.kind", "image"), ("evaluation.metric", "fid")],
    "rank": [("unit", "query"), ("output", "ranking"),
             ("target.source", "feedback"), ("method", "scratch"),
             ("evaluation.metric", "ndcg")],
    "group": [("unit", "row"), ("output", "cluster"),
              ("target.source", "none"), ("method", "pretrained"),
              ("evaluation.metric", "silhouette")],
    "effect": [("unit", "row"), ("output", "effect"),
               ("target.source", "intervention"), ("method", "scratch"),
               ("evaluation.metric", "qini")],
}


def v1_problem() -> dict:
    """
    版 1 の問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict
        版 1 の問題の設定
    """
    # YAML から
    problem = yaml.safe_load(V1_TEXT)

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


def role_problem(role: str, answers: list) -> dict:
    """
    見本の課題から, 受け持ちを入れ替えた問題の設定を作る関数

    Parameters
    ----------
    role: str
        受け持ち
    answers: list
        受け持ちの中の場所と答えの並び

    Returns
    ----------
    problem: dict
        問題の設定
    """
    # 1 から 4 と 9, 10 は見本のまま, 受け持ちは作り直し
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    problem = answer(problem, "ml.parts.0.role", role)
    for key, value in [("what", "試しの出力"), ("item", "1 件"),
                       ("baseline", "今のやり方"),
                       ("evaluation.link", "良くなると損が減る"), *answers]:
        problem = answer(problem, f"ml.parts.0.{key}", value)

    return problem


def test_upgrade_round_trip(tmp_path: Path) -> None:
    """
    版 1 を版 2 に移すと全ての値が残り, 版 1 の名前で読め, ファイルの
    読み書きでも同じになることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 1 の全ての値が移した先にある (元は変えない)
    old = v1_problem()
    kept = copy.deepcopy(old)
    new = upgrade_problem(old)
    assert old == kept
    assert new["format_version"] == 2
    for legacy, moved in LEGACY_KEYS.items():
        assert get_value(new, moved) == get_value(old, legacy), legacy
        assert get_value(new, legacy) == get_value(old, legacy), legacy
    for section in ("issue", "decomposition", "information", "action",
                    "verification", "operation", "name", "created_at"):
        assert new[section] == old[section]
    assert new["ml"]["future_field"] == "残す"
    assert new["ml"]["constraints"] == ["データを社外に出さない"]
    assert "evaluation" not in new and "predict" not in new["ml"]

    # 版 1 の前提の答えを埋め, 版 2 はそのまま
    part = new["ml"]["parts"][0]
    assert {key: part[key] for key in ("role", "unit", "output", "method")} \
        == {"role": "predict", "unit": "row", "output": "value",
            "method": "scratch"}
    assert part["target"] == {"source": "column", "kind": "column",
                              "column": "sales"}
    assert upgrade_problem(new) is new
    assert check(old) == check(new)
    assert is_ready(new)

    # ファイルの版 1 は読むと版 2, 書いて読み直しても同じ
    path = tmp_path / "problem.yaml"
    path.write_text(V1_TEXT, encoding="utf-8")
    loaded = load_problem(path)
    assert loaded == new
    saved = save_problem(loaded, tmp_path / "again.yaml")
    assert load_problem(tmp_path / "again.yaml") == saved
    assert yaml.safe_load(path.read_text(encoding="utf-8"))[
        "format_version"
    ] == 1

    # 機械学習を使わない版 1 は受け持ち無し
    plain = {"format_version": 1, "name": "x", "created_at": "t",
             "ml": {"use": False, "reason": "規則で足りる"}}
    assert "parts" not in upgrade_problem(plain)["ml"]

    # 空の版 1 の項目は残さず, 版 1 に手で書いた受け持ちは残す
    sparse = {"format_version": 1, "name": "x", "created_at": "t",
              "ml": {"use": True, "task": None, "positive": None,
                     "parts": [{"note": "手書き"}]}}
    moved = upgrade_problem(sparse)["ml"]
    assert "task" not in moved and "positive" not in moved
    assert moved["parts"][0]["note"] == "手書き"
    assert moved["parts"][0]["role"] == "predict"


def test_legacy_names_on_version2() -> None:
    """
    版 1 の名前で答えて読めること (後ろの段階の今の呼び方) を確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 版 1 の問題の設定への答えの後は版 2
    problem = answer(v1_problem(), "ml.target", "promo")
    assert problem["format_version"] == 2
    assert get_value(problem, "ml.parts.0.target.column") == "promo"
    assert get_value(problem, "ml.target") == "promo"
    problem = answer(problem, "evaluation.metric", "rmse")
    assert handoff(answer(problem, "ml.target", "sales"))["training"][
        "monitor"
    ] == "valid/mse"


def test_column_handoff_unchanged() -> None:
    """
    列の予測の受け渡しが版 1 と同じ値で, 受け持ちと正解の種類が足される
    ことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 見本の時系列の予測 (版 1 のときの値)
    stages = handoff(example_problem("総菜の作る量"))
    assert stages["data"] == {
        "target": "sales", "task": "forecasting", "time_column": "date",
        "series_by": ["store", "item"], "split": "time",
        "group_column": None, "min_lag": 2,
        "known": ["store", "item", "temperature_forecast", "promo"],
        "source": None, "drop": ["customers"], "project_dataset": None,
    }
    assert stages["model"] == {"task": "forecasting", "outputs": 1,
                               "role": "predict", "target_kind": "column",
                               "project_model_spec": None}
    assert stages["training"] == {"metric": "mae", "monitor": "valid/mae",
                                  "mode": "min", "role": "predict",
                                  "target_kind": "column"}
    assert stages["inference"] == {
        "decision": {"type": "quantity", "under_cost": 200,
                     "over_cost": 150},
        "conclusion": {"group_by": "store"}, "horizon": 1,
    }
    assert stages["part"] == {
        "role": "predict", "unit": "row", "output": "value",
        "target_kind": "column", "target_source": "column",
        "target_process": None, "method": "scratch",
    }

    # 版 1 の見本の分類も同じ
    churn = handoff(example_problem("解約の引き止め"))
    assert churn["data"]["split"] == "stratified"
    assert churn["model"]["outputs"] is None
    assert churn["inference"]["decision"]["positive"] == "yes"


def test_point_cloud_handoff() -> None:
    """
    予測するで正解の種類が点群の受け持ちを, 確かめ, 渡せることを確かめる
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 正解は今の処理の出力 (名前と設定), 指標は Chamfer
    problem = role_problem("predict", [
        ("unit", "bundle"), ("output", "shape"),
        ("target.source", "process"), ("target.process.name", "今の再構成"),
        ("target.process.settings", "細かさ 1 mm"), ("method", "partial"),
        ("target.kind", "point_cloud"), ("evaluation.metric", "chamfer"),
    ])
    items = {item["id"]: item for item in questions(problem)}
    assert not items["ml.parts.0.task"]["applies"]
    assert not items["information.columns"]["applies"]
    assert not items["action.decision"]["applies"]
    assert [c["value"] for c in items["ml.parts.0.evaluation.metric"][
        "choices"
    ]] == ["loss", "chamfer", "other"]
    assert errors_of(problem) == []
    stages = handoff(problem)
    assert stages["training"] == {"metric": "chamfer",
                                  "monitor": "valid/chamfer", "mode": "min",
                                  "role": "predict",
                                  "target_kind": "point_cloud"}
    assert stages["part"]["target_process"] == {"name": "今の再構成",
                                                "settings": "細かさ 1 mm"}
    assert stages["data"]["target"] is None
    assert stages["data"]["split"] == "random"
    assert stages["data"]["groups"] is None
    assert stages["inference"] == {"decision": None, "conclusion": {},
                                   "horizon": None}
    assert any("予測すること: 試しの出力" in text
               for text in describe(problem))

    # 今の処理の名前は必須. レシピの下書きは書き出しの設定
    nameless = answer(problem, "ml.parts.0.target.process.name", None)
    assert "ml.parts.0.target.process.name" in errors_of(nameless)
    start = recipe_from(problem)
    assert start["recipe"] is None
    assert start["export"]["function"] == "export_capture_clouds"


def test_each_role_checks() -> None:
    """
    並べる, まとめる, 効果を見積もる, 作るの受け持ちが最小の答えで渡せ,
    合わない答えを確かめることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 受け持ちごとの候補の指標での受け渡し
    for role, answers in ROLE_ANSWERS.items():
        problem = role_problem(role, answers)
        assert errors_of(problem) == [], role
        stages = handoff(problem)
        assert stages["training"]["role"] == role
        assert stages["part"]["role"] == role
        assert "機械学習の受け持ち (" in " ".join(describe(problem))

    # 受け持ちに合わない出力の形, 正解の種類, 指標
    rank = role_problem("rank", ROLE_ANSWERS["rank"])
    assert "ml.parts.0.output" in errors_of(
        answer(rank, "ml.parts.0.output", "class")
    )
    assert "ml.parts.0.evaluation.metric" in errors_of(
        answer(rank, "ml.parts.0.evaluation.metric", "mae")
    )
    generate = role_problem("generate", ROLE_ANSWERS["generate"])
    assert "ml.parts.0.target.kind" in errors_of(
        answer(generate, "ml.parts.0.target.kind", "column")
    )

    # 誤りの文は画面の選択肢と同じ言葉
    found = {item["id"]: item["message"] for item in check(
        answer(rank, "ml.parts.0.output", "class")
    )}
    assert found["ml.parts.0.output"] == (
        "「クラス」は受け持ち「並べる」に合いません"
    )
    found = {item["id"]: item["message"] for item in check(
        answer(rank, "ml.parts.0.evaluation.metric", "mae")
    )}
    assert "指標「外れの大きさの平均 (平均絶対誤差" in found[
        "ml.parts.0.evaluation.metric"
    ]

    # 受け持ちを答えるまで型ごとの問いは出ず, 知らない値は形の誤り
    fresh = answer(new_problem("試し"), "ml.use", True)
    assert "ml.parts.0.role" in errors_of(fresh)
    broken = example_problem("総菜の作る量")
    broken["ml"]["parts"][0]["role"] = "translate"
    assert errors_of(broken) == ["ml.parts.0.role"]


def test_one_part_only() -> None:
    """
    受け持ちが 2 つ以上なら, はっきりした誤りになることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 2 つの受け持ち
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"].append(copy.deepcopy(problem["ml"]["parts"][0]))
    found = check(problem)
    assert [item["id"] for item in found] == ["ml.parts"]
    assert "1 つだけ" in found[0]["message"]
    with pytest.raises(ValueError, match="1 つだけ"):
        handoff(problem)

    # 並びでない受け持ちと, 辞書でない指標
    problem["ml"]["parts"] = {"role": "predict"}
    assert errors_of(problem) == ["ml.parts"]
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"][0]["evaluation"] = "rmse"
    assert errors_of(problem) == ["ml.parts.0.evaluation"]


def test_other_metric(tmp_path: Path) -> None:
    """
    最良を選ぶ指標のその他が, 名前と向きで監視する指標として渡り, 版 1 の
    ファイルでは知らない指標になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # その他は名前と向きが要る
    problem = answer(example_problem("解約の引き止め"),
                     "ml.parts.0.evaluation.metric", "other")
    assert {"ml.parts.0.evaluation.name",
            "ml.parts.0.evaluation.better"} <= set(errors_of(problem))
    problem = answer(problem, "ml.parts.0.evaluation.name", "f1")
    problem = answer(problem, "ml.parts.0.evaluation.better", "higher")
    assert errors_of(problem) == []
    training = handoff(problem)["training"]
    assert (training["metric"], training["monitor"], training["mode"]) == (
        "f1", "valid/f1", "max"
    )
    with pytest.raises(ValueError, match="その他"):
        make_project(problem, Path("/nonexistent-ml-quest-project"))

    # 空白と / のある名前は誤り
    spaced = answer(problem, "ml.parts.0.evaluation.name", "f 1")
    assert "ml.parts.0.evaluation.name" in errors_of(spaced)

    # 版 1 のファイルでは 4 つの他は知らない指標
    for metric in ("other", "chamfer"):
        old = v1_problem()
        old["evaluation"]["metric"] = metric
        found = check(old)
        assert [item["id"] for item in found] == [
            "ml.parts.0.evaluation.metric"
        ]
        assert "知らない指標" in found[0]["message"]

    # 読み込んだ版 1 のファイルでも同じで, 指標に答え直すと直る
    path = tmp_path / "problem.yaml"
    old = v1_problem()
    old["evaluation"]["metric"] = "other"
    path.write_text(yaml.safe_dump(old, allow_unicode=True),
                    encoding="utf-8")
    loaded = load_problem(path)
    assert "知らない指標" in check(loaded)[0]["message"]
    fixed = answer(loaded, "ml.parts.0.evaluation.metric", "mae")
    assert "evaluation" not in fixed
    assert errors_of(fixed) == []


def test_newer_version_stops(tmp_path: Path) -> None:
    """
    知っている版より新しい問題の設定で, はっきり止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 読み込み, 確かめ, 受け渡し, 答え
    newer = example_problem("総菜の作る量")
    newer["format_version"] = 3
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(newer, allow_unicode=True),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="新しい版"):
        load_problem(path)
    found = check(newer)
    assert [(item["level"], item["id"]) for item in found] == [
        ("error", "format_version")
    ]
    assert not is_ready(newer)
    with pytest.raises(ValueError, match="新しい版"):
        handoff(newer)
    with pytest.raises(ValueError, match="新しい版"):
        answer(newer, "issue.owner", "誰か")


def test_review_fixes() -> None:
    """
    版 1 の dict の古い答え, その他の名前, 並びの書き込み, 版の値, 列の
    予測の出力の形の確かめを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 版 1 の dict でも, 要らない問いの古い答えは active_value で除外
    old = v1_problem()
    old["ml"].update(task="classification", positive="yes")
    assert active_value(old, "ml.time_column") is None
    assert active_value(old, "ml.horizon") is None
    assert active_value(old, "ml.positive") == "yes"
    old["ml"]["use"] = False
    assert active_value(old, "ml.target") is None

    # その他の名前は英数字と _ . @ - だけ. 古い名前と向きは使わない
    problem = answer(example_problem("解約の引き止め"),
                     "ml.parts.0.evaluation.metric", "other")
    problem = answer(problem, "ml.parts.0.evaluation.better", "higher")
    for name, bad in (("recall@10", False), ("f1.macro", False),
                      ("a/b", True), ("..", True), ("名前", True)):
        named = answer(problem, "ml.parts.0.evaluation.name", name)
        assert ("ml.parts.0.evaluation.name" in errors_of(named)) is bad
    back = answer(named, "ml.parts.0.evaluation.metric", "loss")
    assert handoff(back)["training"]["monitor"] == "valid/loss"

    # 並びの番号でない名前と, 辞書でない受け持ちは誤り (黙って置き換えない)
    with pytest.raises(ValueError, match="番号で"):
        set_value(copy.deepcopy(problem), "ml.parts.x", 1)
    hand = copy.deepcopy(problem)
    hand["ml"]["parts"] = ["手書き"]
    with pytest.raises(ValueError, match="辞書でない"):
        answer(hand, "ml.parts.0.role", "rank")
    assert errors_of(hand) == ["ml.parts"]

    # 版 1 に手で書いた受け持ちの答えは上書きしない
    mixed = v1_problem()
    mixed["ml"]["parts"] = [{"unit": "window"}]
    assert get_value(upgrade_problem(mixed), "ml.parts.0.unit") == "window"

    # 版 2 の版 1 の場所の指標は, 書く場所の知らせ
    misplaced = example_problem("総菜の作る量")
    misplaced["evaluation"] = {"metric": "mae"}
    message = check(misplaced)[0]["message"]
    assert "版 2 では" in message and "知らない" not in message

    # 1 以上の整数でない版は誤り
    for version in ("2", 2.0, True, 0, None):
        bad_version = example_problem("総菜の作る量")
        bad_version["format_version"] = version
        assert [item["id"] for item in check(bad_version)] == [
            "format_version"
        ]
        with pytest.raises(ValueError, match="format_version"):
            upgrade_problem(bad_version)

    # 列の予測の出力の形はクラスか値か分布
    shaped = answer(example_problem("総菜の作る量"), "ml.parts.0.output",
                    "shape")
    assert "ml.parts.0.output" in errors_of(shaped)


def test_project_error() -> None:
    """
    骨組みを書き出せない理由が, make_project の誤りの文と同じになることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 列の予測は書き出せる (版 1 も)
    assert project_error(example_problem("総菜の作る量")) is None
    assert project_error(v1_problem()) is None

    # 列の予測でない受け持ち, その他の指標, 渡せない問題の設定
    other = answer(example_problem("解約の引き止め"),
                   "ml.parts.0.evaluation.metric", "other")
    other = answer(other, "ml.parts.0.evaluation.name", "f1")
    other = answer(other, "ml.parts.0.evaluation.better", "higher")
    plain = example_problem("総菜の作る量")
    plain["ml"]["parts"][0]["role"] = "translate"
    for problem, words in (
        (role_problem("rank", ROLE_ANSWERS["rank"]), "表の列"),
        (other, "その他"), (plain, "直すところ"),
    ):
        message = project_error(problem)
        assert message is not None and words in message
        with pytest.raises(ValueError) as raised:
            make_project(problem, Path("/nonexistent-ml-quest-project"))
        assert str(raised.value) == message


def test_screen_names() -> None:
    """
    画面が写していた名前 (受け渡しの分割の名前, 今のやり方の書き方の名前)
    が, 説明付きでこのライブラリから取れることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 分割: 受け渡しとファイルの組の書き出しの分け方を全て含み, 説明付き
    from ml_quest import BASELINE_TYPES, SPLITS
    from ml_quest.bridge import SPLIT_METHODS

    names = {item["value"] for item in SPLITS}
    assert {"time", "group", "random", "stratified"} <= names
    assert set(SPLIT_METHODS) <= names
    assert all(item["label"] for item in SPLITS)
    for name in ("総菜の作る量", "解約の引き止め"):
        assert handoff(example_problem(name))["data"]["split"] in names

    # 今のやり方の書き方: 問いの選択肢として, 答えた意思決定の型に使える
    # ものだけと, 欄
    assert [item["value"] for item in BASELINE_TYPES] == [
        "lag", "mean", "column"
    ]
    items = {item["id"]: item
             for item in questions(example_problem("総菜の作る量"))}
    choices = items["action.current"]["choices"]
    assert [choice["value"] for choice in choices] == ["lag", "mean",
                                                       "column"]
    assert choices[0]["fields"][0]["key"] == "periods"
    items = {item["id"]: item
             for item in questions(example_problem("解約の引き止め"))}
    choices = items["action.current"]["choices"]
    assert [choice["value"] for choice in choices] == ["column"]
    assert choices[0]["fields"][0]["key"] == "column"
    with pytest.raises(ValueError, match="lag, mean, column"):
        answer(example_problem("総菜の作る量"), "action.current",
               {"type": "median", "periods": 3})
