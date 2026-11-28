import json
from pathlib import Path
import sys

import pytest

from ml_quest import (
    answer, attach_data, check, handoff, make_project, questions,
    rank_encoding, record_project
)

# 並べるの問題の設定の作り方と短い学習 (tests/test_rank_skeleton.py と同じ)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rank_skeleton import (  # noqa: E402
    ITEMS, PART, USERS, rank_problem, script_module, train
)

# 合成の提示の回数と, 1 回に見せる品物の数
SHOWS = 800
PER_SHOW = 5

# 見せただけの記録のある反応の表の列
COLUMNS = ["member", "item_id", "ts", "qty", "shown_only", "pid"]


def shown_problem(presentation: bool = True) -> dict:
    """
    見せただけの印の列 (と提示の列) に答えた並べるの問題の設定を作る関数

    Parameters
    ----------
    presentation: bool = True
        提示の列にも答えるか

    Returns
    ----------
    problem: dict
        並べるの問題の設定 (データは入れていない)
    """
    # 使えるデータの列に印と提示の列を足す
    problem = answer(rank_problem(), "information.columns", COLUMNS)
    problem = answer(problem, f"{PART}.rank.shown", "shown_only")
    if presentation:
        problem = answer(problem, f"{PART}.rank.presentation", "pid")

    return problem


def write_shown(root: Path) -> Path:
    """
    提示ごとに 1 件を選び, 残りを見せただけにした合成の反応の表を作る関数

    Parameters
    ----------
    root: Path
        書き出す場所

    Returns
    ----------
    path: Path
        CSV のファイル (member, item_id, ts, qty, shown_only, pid)
    """
    # 利用者は番号を 4 で割った余りのまとまりの品物を選び, 見せただけの
    # 品物は他のまとまりから. 選んだ行の印は空
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    rows = []
    for number in range(SHOWS):
        user = int(rng.integers(0, USERS))
        chosen = int(rng.choice(np.arange(user % 4, ITEMS, 4)))
        others = rng.choice([item for item in range(ITEMS)
                             if item % 4 != user % 4], PER_SHOW - 1,
                            replace=False)
        rows.append({"member": f"m{user}", "item_id": f"i{chosen}",
                     "ts": number, "qty": 1, "shown_only": None,
                     "pid": f"p{number}"})
        rows += [{"member": f"m{user}", "item_id": f"i{int(item)}",
                  "ts": number, "qty": 0, "shown_only": 1,
                  "pid": f"p{number}"} for item in others]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "shown.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def test_shown_questions() -> None:
    """
    見せただけの印と提示の列の問いが任意で, 受け渡しと確かめに入ることを
    確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 印の列は任意, 提示の列は印の列に答えたときだけ問う
    items = {item["id"]: item for item in questions(rank_problem())}
    assert items[f"{PART}.rank.shown"]["applies"]
    assert not items[f"{PART}.rank.shown"]["required"]
    assert not items[f"{PART}.rank.presentation"]["applies"]
    items = {item["id"]: item for item in questions(shown_problem())}
    assert items[f"{PART}.rank.presentation"]["applies"]

    # 受け渡しの列の名前 (答えなければ None)
    assert handoff(shown_problem())["rank"]["shown"] == "shown_only"
    assert handoff(shown_problem())["rank"]["presentation"] == "pid"
    assert handoff(shown_problem(False))["rank"]["presentation"] is None
    assert handoff(rank_problem())["rank"]["shown"] is None

    # 他の役割と同じ列は誤り
    same = answer(shown_problem(), f"{PART}.rank.presentation", "ts")
    found = [item["id"] for item in check(same) if item["level"] == "error"]
    assert f"{PART}.rank.presentation" in found


def test_shown_training(tmp_path: Path) -> None:
    """
    見せただけの記録のある表で, 骨組みが見せただけの品物を負の例にし,
    1 回の提示の中の順位を人気順と並べて評価し, 学習の前に求めた数と
    train.log の数が合うことを確かめる関数 (PyTorch と段階のライブラリが
    無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 入れた表のレシピ無しの読み込みと, 仮のモデルでの短い学習
    for name in ("torch", "ml_found", "ml_eda"):
        pytest.importorskip(name)
    source = write_shown(tmp_path / "raw")
    problem = attach_data(shown_problem(), str(source),
                          copy_to=str(tmp_path / "copies"))
    project = tmp_path / "project"
    make_project(problem, project)
    config = (project / "configs" / "default" / "train.yaml").read_text(
        encoding="utf-8"
    )
    assert 'shown: "shown_only"' in config and 'presentation: "pid"' in config
    readme = (project / "README.md").read_text(encoding="utf-8")
    for words in ("`shown_only`", "`pid`", "load_reactions",
                  "presentation_auc"):
        assert words in readme, words
    run, _ = train(project)
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "見せただけの記録: train" in log and "学習の負の例: train" in log
    assert "の提示の中の順位" in log

    # 反応は選んだ行だけ (学習の前に求めた数と同じ)
    dataset = record_project(problem, project)["project"]["dataset"]
    encoding = rank_encoding(problem, dataset)
    assert sum(encoding["rows"].values()) == SHOWS
    assert (f"利用者の数: {encoding['users']}, 品物の数: "
            f"{encoding['items']}. 反応: " + ", ".join(
                f"{name} {rows} 行"
                for name, rows in encoding["rows"].items())) in log

    # 提示の中の順位: 選んだ品物を上に並べるほど大きく, 好むまとまりを
    # 学んだモデルは人気順より良い
    for split in ("valid", "test"):
        table = json.loads((run / "evaluations" / f"ranking_{split}.json")
                           .read_text(encoding="utf-8"))
        row = table["rows"][0]
        assert row["presentations"] > 0
        assert 0 <= row["presentation_auc"] <= 1
        assert row["presentation_auc_diff"] == pytest.approx(
            row["presentation_auc"] - row["presentation_auc_baseline"]
        )
        assert row["presentation_auc"] > row["presentation_auc_baseline"]
        assert table["better"]["presentation_auc"] == "higher"
        assert "presentation_auc は" in table["description"]


def test_shown_by_time(tmp_path: Path) -> None:
    """
    提示の列が無いときに, 利用者と時刻の組を 1 回の提示として読み, 選んだ
    反応だけを数えることを確かめる関数 (ML-EDA が無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 印の列だけに答えた問題の設定の, 学習の前の数
    pytest.importorskip("ml_eda")
    source = write_shown(tmp_path / "raw")
    problem = attach_data(shown_problem(False), str(source),
                          copy_to=str(tmp_path / "copies"))
    encoding = rank_encoding(problem, source)
    assert sum(encoding["rows"].values()) == SHOWS

    # 印に答えなければ, 見せただけの行も反応の強さの決まりで読む (今どおり)
    plain = answer(rank_problem(), "information.columns", COLUMNS)
    plain = attach_data(plain, str(source),
                        copy_to=str(tmp_path / "plain"))
    assert sum(rank_encoding(plain, source)["rows"].values()) == SHOWS


def test_shown_functions(tmp_path: Path) -> None:
    """
    書き出した train.py の, 負の例の付け方, 提示の組, 提示の中の順位,
    負の例の点数の隠し方を小さな表で確かめる関数 (PyTorch と段階の
    ライブラリが無ければ飛ばす)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 番号にした反応と見せただけの記録 (提示 a は 2 件, b は無し)
    module = script_module(tmp_path)
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    frame = pd.DataFrame({"user": [0, 1], "item": [0, 1],
                          "key": [("u0", 1), ("u1", 2)]})
    shown = pd.DataFrame({"user": [0, 0, 0], "item": [2, 3, 0],
                          "key": [("u0", 1)] * 3})

    # 見せただけの品物が無い提示の行は -1, 同じ種で同じ負の例
    paired = module.attach_negatives(frame, shown, 0)
    assert paired["negative"].iloc[1] == -1
    assert paired["negative"].iloc[0] in (0, 2, 3)
    assert paired.equals(module.attach_negatives(frame, shown, 0))

    # 提示の組は選んだ品物と同じ品物を見せただけから除く
    pairs = module.presentation_pairs(frame, shown)
    assert pairs == [(0, [0], [2, 3])]

    # 同じ点数は半分, 上なら 1
    assert module.presentation_auc({0: np.array([1.0, 0, 1.0, 0])},
                                   pairs) == pytest.approx(0.75)
    assert module.presentation_auc({0: np.array([2.0, 0, 1.0, 0])},
                                   pairs) == pytest.approx(1.0)

    # 行の正解と同じ品物の負の例の列は -inf
    model = module.PlaceholderModel(2, 4, 3)
    score = module.negative_scores(
        model, torch.tensor([0, 1]), torch.tensor([0, 1]),
        torch.tensor([1, 2]), (2, 4)
    )
    assert score.shape == (2, 4)
    assert torch.isinf(score[1, 2]) and not torch.isinf(score[0, 2])
    assert not torch.isinf(score[:, 3]).any()
