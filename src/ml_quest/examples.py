from typing import Any

from ml_quest.problem import answer, new_problem
from ml_quest.questions import PART

# 見本の答え (架空の課題. 問いの順). 量を決める予測, 対象を選ぶ分類,
# 既製のモデルをそのまま使う画像の分類, 既製の埋め込みの写真の群分け,
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
        (f"{PART}.evaluation.link", "誤差が小さいほど作る量の外れが減り, "
                                    "廃棄と欠品の損が減る"),
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
        ("issue.guard.counted", "none"),
        ("action.basis", "1 人の 1 年分の料金 24,000 円. 連絡 1 件の人件費"
                         " 1,500 円. 防げる割合は前の年の試しから 2 割"),
        (f"{PART}.evaluation.metric", "loss"),
        (f"{PART}.evaluation.link", "確率が当たるほど, 連絡する会員を損得で"
                                    "正しく選べる"),
        ("verification.method", "trial"),
        ("verification.trial", "会員を 2 つに分け, 片方だけに 2 か月連絡する"),
        ("verification.success", "連絡した側の解約が 15 % 少ない"),
        ("operation.review", "月に 1 回"),
        ("operation.stop", "2 か月続けて差が出ないか, 引き止めの連絡への"
                           "苦情が月 5 件を超えたら止める"),
    ],
    "写真の仕分け (既製のモデル)": [
        ("issue.owner", "写真の整理の担当"),
        ("issue.pain", "届く写真に何が写っているかを人が見て仕分けていて, "
                       "1 日に何時間もかかる"),
        ("issue.current", "担当が 1 枚ずつ見て, 写っているものの名前を付ける"),
        ("issue.kpi.name", "仕分けにかかる時間"),
        ("issue.kpi.unit", "時間 / 日"),
        ("issue.kpi.better", "lower"),
        ("issue.kpi.current", 3),
        ("issue.kpi.target", 1),
        ("decomposition", [
            {"cause": "写っているものの名前を人が 1 枚ずつ付ける",
             "lever": "既製のモデルで名前の候補を付け, 人は確かめるだけに"
                      "する", "ml": True},
            {"cause": "仕分け先の決まりが人ごとに違う",
             "lever": "種類ごとの仕分け先の表を作る", "ml": False},
        ]),
        ("action.what", "写真の仕分け先"),
        ("action.who", "写真の整理の担当"),
        ("action.when", "写真が届いたとき"),
        ("ml.use", True),
        ("ml.reason", "写っているものは多くの種類があり, 決まりでは分けられ"
                      "ない. ImageNet の種類で足りるので学習しない"),
        (f"{PART}.role", "predict"),
        (f"{PART}.unit", "row"),
        (f"{PART}.output", "class"),
        (f"{PART}.target.source", "none"),
        (f"{PART}.method", "pretrained"),
        (f"{PART}.target.kind", "column"),
        (f"{PART}.ready.library", "torchvision"),
        (f"{PART}.ready.model", "resnet18"),
        (f"{PART}.ready.weights", "IMAGENET1K_V1"),
        (f"{PART}.ready.sha256", "8ec1484cdbb88ded96d21c3a5936faf3cdcc665aef"
                                 "92f331f675d0c63038f0ba"),
        (f"{PART}.what", "写真に写っているものの種類 (ImageNet の 1000 種類)"),
        (f"{PART}.item", "写真 1 枚"),
        (f"{PART}.baseline", "担当が 1 枚ずつ見て名前を付ける"),
        (f"{PART}.consumer", "human"),
        ("verification.method", "trial"),
        ("verification.trial", "1 週間, 候補を見ながら仕分ける"),
        ("verification.success", "仕分けにかかる時間が 1 日 1 時間になる"),
        ("operation.review", "月に 1 回"),
        ("operation.stop", "候補の直しが半分を超えたら止める"),
    ],
    "商品の写真の群分け (既製の埋め込み)": [
        ("issue.owner", "通販の売り場の担当"),
        ("issue.pain", "商品の写真が増え, 似た見た目の商品をまとめた売り場の"
                       "棚を作るのに時間がかかる"),
        ("issue.current", "担当が写真を見比べて, 似た商品を手で棚に分ける"),
        ("issue.kpi.name", "棚を作るのにかかる時間"),
        ("issue.kpi.unit", "時間 / 月"),
        ("issue.kpi.better", "lower"),
        ("issue.kpi.current", 20),
        ("issue.kpi.target", 5),
        ("decomposition", [
            {"cause": "似た見た目の商品を人が全て見比べる",
             "lever": "既製のモデルの埋め込みで写真を群に分け, 人は群を"
                      "見て名前を付けるだけにする", "ml": True},
            {"cause": "棚の名前の付け方が決まっていない",
             "lever": "棚の名前の決まりを作る", "ml": False},
        ]),
        ("action.what", "商品を並べる売り場の棚"),
        ("action.who", "通販の売り場の担当"),
        ("action.when", "新しい商品の写真が届いたとき"),
        ("ml.use", True),
        ("ml.reason", "見た目の似方は決まりで書けない. 正解の群は無いが, "
                      "画像の分類の既製のモデルの埋め込みで似た写真が近く"
                      "なるので学習しない"),
        (f"{PART}.role", "group"),
        (f"{PART}.unit", "row"),
        (f"{PART}.output", "cluster"),
        (f"{PART}.target.source", "none"),
        (f"{PART}.method", "pretrained"),
        (f"{PART}.ready.library", "torchvision"),
        (f"{PART}.ready.model", "resnet18"),
        (f"{PART}.ready.weights", "IMAGENET1K_V1"),
        (f"{PART}.ready.sha256", "8ec1484cdbb88ded96d21c3a5936faf3cdcc665aef"
                                 "92f331f675d0c63038f0ba"),
        (f"{PART}.group.k", 4),
        (f"{PART}.group.stability", "seed"),
        (f"{PART}.what", "見た目の似た商品の群"),
        (f"{PART}.item", "商品の写真 1 枚"),
        (f"{PART}.baseline", "担当が写真を見比べて手で分ける"),
        (f"{PART}.consumer", "human"),
        ("verification.method", "trial"),
        ("verification.trial", "1 か月, 群を見ながら棚を作る"),
        ("verification.success", "棚を作るのにかかる時間が月 5 時間になる"),
        ("operation.review", "月に 1 回"),
        ("operation.stop", "群を見ても棚に使えないと担当が判断したら止める"),
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
