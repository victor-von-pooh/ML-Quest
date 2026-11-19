from typing import Any

from ml_quest.problem import answer, new_problem
from ml_quest.questions import PART

# 見本の答え (架空の課題. 問いの順). 量を決める予測, 対象を選ぶ分類,
# 機械学習を使わない出口
EXAMPLES: dict[str, list[tuple[str, Any]]] = {
    "総菜の作る量": [
        ("issue.owner", "店長"),
        ("issue.pain", "総菜の廃棄が多く, 夕方には品切れも出る"),
        ("issue.current", "前の週の同じ曜日と同じ数を作る"),
        ("issue.kpi.name", "廃棄と欠品の損"),
        ("issue.kpi.unit", "円 / 月"),
        ("issue.kpi.better", "lower"),
        ("issue.kpi.current", 1200000),
        ("issue.kpi.target", 960000),
        ("decomposition", [
            {"cause": "作る量が勘で決まる",
             "lever": "作る量を売れる数の予測から決める", "ml": True},
            {"cause": "夕方の値引きが遅い",
             "lever": "値引きの時刻を決めておく", "ml": False},
        ]),
        ("action.what", "品ごとの作る量"),
        ("action.who", "店長"),
        ("action.when", "前の日の 17 時"),
        ("ml.use", True),
        ("ml.reason", "品と日による売れ方の違いが大きく, 前の週と同じでは"
                      "追えない"),
        (f"{PART}.role", "predict"),
        (f"{PART}.unit", "row"),
        (f"{PART}.output", "value"),
        (f"{PART}.target.source", "column"),
        (f"{PART}.method", "scratch"),
        (f"{PART}.target.kind", "column"),
        (f"{PART}.what", "翌日の品ごとの売れる数"),
        (f"{PART}.item", "店, 品, 日ごとの 1 行"),
        ("information.columns", ["date", "store", "item", "sales",
                                 "temperature_forecast", "promo",
                                 "customers"]),
        (f"{PART}.task", "forecasting"),
        (f"{PART}.target.column", "sales"),
        (f"{PART}.time_column", "date"),
        (f"{PART}.series_by", ["store", "item"]),
        (f"{PART}.available_lag", 2),
        (f"{PART}.horizon", 1),
        (f"{PART}.baseline", "前の週の同じ曜日と同じ数"),
        (f"{PART}.consumer", "rule"),
        ("information.known", ["store", "item", "temperature_forecast",
                               "promo"]),
        ("information.unknown", ["customers"]),
        ("action.decision", {"type": "quantity", "under_cost": 200,
                             "over_cost": 150}),
        ("action.basis", "欠品は粗利 200 円, 廃棄は原価 150 円"),
        (f"{PART}.evaluation.metric", "mae"),
        (f"{PART}.evaluation.link", "誤差が小さいほど作る量の外れが減り, 廃棄と欠品"
                            "の損が減る"),
        ("verification.method", "both"),
        ("verification.backtest", "過去 3 か月で, 今のやり方と損を比べる"),
        ("verification.trial", "1 店で 4 週間"),
        ("verification.success", "廃棄と欠品の損が今のやり方より 2 割少ない"),
        ("operation.review", "月に 1 回"),
        ("operation.approver", "本部"),
        ("operation.stop", "損が今のやり方より悪い月が 2 回続いたら止める"),
        ("operation.remeasure", "月に 1 回, 前の月の実績で外れ方を確かめ, "
                                "本部が承認したら書き戻す"),
    ],
    "解約の引き止め": [
        ("issue.owner", "会員サービスの担当"),
        ("issue.pain", "毎月の解約が多く, 会員が減っている"),
        ("issue.current", "解約の申し込みがあってから引き止めの連絡をする"),
        ("issue.kpi.name", "月の解約の件数"),
        ("issue.kpi.unit", "件 / 月"),
        ("issue.kpi.better", "lower"),
        ("issue.kpi.current", 400),
        ("issue.kpi.target", 340),
        ("issue.guard.name", "引き止めの連絡への苦情"),
        ("issue.guard.unit", "件 / 月"),
        ("issue.guard.better", "lower"),
        ("issue.guard.line", 5),
        ("issue.guard.measure", "窓口に届いた, 連絡についての苦情を数える"),
        ("decomposition", [
            {"cause": "解約しそうな会員に気付くのが遅い",
             "lever": "解約しそうな会員を先に見つけて連絡する", "ml": True},
            {"cause": "料金の案内が分かりにくい",
             "lever": "料金の案内を書き直す", "ml": False},
        ]),
        ("action.what", "引き止めの連絡をする会員"),
        ("action.who", "会員サービスの担当"),
        ("action.when", "毎月 1 日"),
        ("ml.use", True),
        ("ml.reason", "連絡できるのは月に 300 人までで, 誰に連絡するかを"
                      "選ぶ必要がある"),
        (f"{PART}.role", "predict"),
        (f"{PART}.unit", "row"),
        (f"{PART}.output", "class"),
        (f"{PART}.target.source", "column"),
        (f"{PART}.method", "scratch"),
        (f"{PART}.target.kind", "column"),
        (f"{PART}.what", "来月に解約するか"),
        (f"{PART}.item", "会員ごとの 1 行"),
        ("information.columns", ["member_id", "plan", "months", "inquiries",
                                 "monthly_fee", "churned"]),
        (f"{PART}.task", "classification"),
        (f"{PART}.target.column", "churned"),
        (f"{PART}.positive", "yes"),
        (f"{PART}.baseline", "問い合わせの多い順に連絡する"),
        (f"{PART}.consumer", "rule"),
        ("ml.constraints", ["会員の情報を社外のサービスに出さない"]),
        ("information.known", ["plan", "months", "inquiries",
                               "monthly_fee"]),
        ("information.forbidden", ["member_id"]),
        ("action.decision", {"type": "select", "value": 24000,
                             "cost": 1500, "effect": 0.2, "budget": 300}),
        ("action.basis", "1 人の 1 年分の料金 24,000 円. 連絡 1 件の人件費"
                         " 1,500 円. 防げる割合は前の年の試しから 2 割"),
        (f"{PART}.evaluation.metric", "loss"),
        (f"{PART}.evaluation.link", "確率が当たるほど, 連絡する会員を損得で正しく"
                            "選べる"),
        ("verification.method", "trial"),
        ("verification.trial", "会員を 2 つに分け, 片方だけに 2 か月連絡する"),
        ("verification.success", "連絡した側の解約が 15 % 少ない"),
        ("operation.review", "月に 1 回"),
        ("operation.stop", "2 か月続けて差が出ないか, 引き止めの連絡への"
                           "苦情が月 5 件を超えたら止める"),
    ],
    "問い合わせの振り分け (機械学習を使わない)": [
        ("issue.owner", "サポートの窓口"),
        ("issue.pain", "問い合わせを担当に回すのに時間がかかる"),
        ("issue.current", "受けた人が中身を読んで担当を決める"),
        ("issue.kpi.name", "回すまでの時間"),
        ("issue.kpi.unit", "分 / 件"),
        ("issue.kpi.better", "lower"),
        ("issue.kpi.current", 12),
        ("issue.kpi.target", 5),
        ("decomposition", [
            {"cause": "問い合わせの種類を書く欄が無い",
             "lever": "受付の画面に種類を選ぶ欄を足す", "ml": False},
            {"cause": "種類が分かっても担当の表が古い",
             "lever": "担当の表を更新する", "ml": False},
        ]),
        ("action.what", "問い合わせを回す担当"),
        ("action.who", "受けた人"),
        ("action.when", "問い合わせを受けたとき"),
        ("ml.use", False),
        ("ml.reason", "種類を選ぶ欄と担当の表で決まる. 機械学習で種類を当て"
                      "る前に, 受付で聞けばよい"),
    ],
}


def example_problem(name: str) -> dict:
    """
    見本の問題の設定を作る関数

    Parameters
    ----------
    name: str
        見本の名前 (EXAMPLES のキー)

    Returns
    ----------
    problem: dict
        答えを順に入れた問題の設定
    """
    # 問いの順の答え
    if name not in EXAMPLES:
        raise ValueError(f"{name} という見本はありません")
    problem = new_problem(name)
    for key, value in EXAMPLES[name]:
        problem = answer(problem, key, value)

    return problem
