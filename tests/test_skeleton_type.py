from ml_quest import (
    EXAMPLES, SKELETON_TYPES, answer, example_problem, new_problem,
    skeleton_type
)

# 見本ごとの骨組みの型
EXPECTED = {
    "総菜の作る量": "column",
    "写真の仕分け (既製のモデル)": "ready",
    "商品の写真の群分け (既製の埋め込み)": "group_ready",
    "似た商品の写真を探す (既製の埋め込み)": "search",
    "問い合わせの振り分け (機械学習を使わない)": None,
}

# 受け持ちの場所
PART = "ml.parts.0"


def test_skeleton_type_examples() -> None:
    """
    見本の問題の設定の骨組みの型が, 書き出す骨組みと合うことを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 名前を決めた見本と, 全ての見本が型の一覧の値か None
    for name, kind in EXPECTED.items():
        assert skeleton_type(example_problem(name)) == kind, name
    for name in EXAMPLES:
        found = skeleton_type(example_problem(name))
        assert found is None or found in SKELETON_TYPES, name
    assert skeleton_type(new_problem("x")) is None


def test_skeleton_type_answers() -> None:
    """
    受け持ち, 作り方, 正解の種類の答えで型が変わることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 列の予測から, 正解の種類と受け持ちと作り方を変える
    base = example_problem("総菜の作る量")
    cases = [
        ({f"{PART}.target.kind": "mask"}, "items"),
        ({f"{PART}.target.kind": "anomaly"}, "anomaly"),
        ({f"{PART}.target.kind": "demand"}, "demand"),
        ({f"{PART}.role": "generate", f"{PART}.target.kind": "image"},
         "generate"),
        ({f"{PART}.role": "generate", f"{PART}.target.kind": "table"},
         "table"),
        ({f"{PART}.role": "rank"}, "rank"),
        ({f"{PART}.role": "group"}, "group"),
        ({f"{PART}.role": "rank", f"{PART}.method": "pretrained"}, "search"),
        ({f"{PART}.role": "group", f"{PART}.method": "pretrained"},
         "group_ready"),
        ({f"{PART}.method": "pretrained", f"{PART}.target.kind": "mask"},
         "ready"),
        ({f"{PART}.role": "effect", f"{PART}.autonomous": True}, "policy"),
        ({f"{PART}.role": "effect", f"{PART}.collect": "log"}, "uplift"),
        ({f"{PART}.role": "effect", f"{PART}.collect": "log",
          f"{PART}.varied": "price"}, "price"),
    ]
    for answers, kind in cases:
        problem = base
        for key, value in answers.items():
            problem = answer(problem, key, value)
        assert skeleton_type(problem) == kind, answers
