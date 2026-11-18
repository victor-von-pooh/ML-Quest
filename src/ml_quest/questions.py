from typing import Any

# 問題の設定の 10 の節 (上から順に決める. 各節は次の節の根拠)
SECTIONS = {
    1: "課題",
    2: "解けたと言える指標",
    3: "課題の分解",
    4: "決めること",
    5: "機械学習の受け持ち",
    6: "予測の時点で使える情報",
    7: "損得と制約",
    8: "学習で見る指標",
    9: "効果の確かめ方",
    10: "運用",
}

# 機械学習を使う場合だけの問い (使わない出口では要らない). when は条件の
# 並びで, 全てを満たすときに必要
USES_ML = {"id": "ml.use", "equals": True}

# 守る指標を書いた場合だけの問い
HAS_GUARD = {"id": "issue.guard.name", "filled": True}

# 予測の使い道の選択肢 (後段の仕組みと後段の機械学習は, 求める形を問う)
CONSUMERS = [
    {"value": "human", "label": "人が見て決める (判断の材料)"},
    {"value": "rule", "label": "決まりで自動に決める (量, 選ぶ, 振り分け)"},
    {"value": "system", "label": "後段の仕組みに渡す (機械学習でない計算や"
     "検査)"},
    {"value": "model", "label": "後段の機械学習に渡す (表現, 候補など)"},
    {"value": "deliver", "label": "そのまま届ける (作った中身)"},
]

# 後段のある使い道
DOWNSTREAM = {"id": "ml.consumer", "in": ["system", "model"]}

# 意思決定の型と, 型ごとの欄 (画面はこれだけで描く. 項目は ML-Inference の
# 形. required は要る欄, choices は選んで書く欄, unset は選ばないときの意味,
# when は他の欄の値で要るかが決まる欄)
DECISION_TYPES = [
    {"value": "quantity", "label": "量を決める (仕入れ, 補充, 人員)",
     "fields": [
         {"key": "under_cost", "required": True,
          "label": "1 単位足りないときの損 (欠品の粗利など)"},
         {"key": "over_cost", "required": True,
          "label": "1 単位余ったときの損 (捨てるなら原価など)"},
         {"key": "stock", "required": False,
          "label": "今ある量 (任意. 行ごとに違えば列の名前)"},
         {"key": "step", "required": False, "label": "量の刻み (任意)"},
     ]},
    {"value": "select", "label": "対象を選ぶ (引き止め, 営業先)",
     "fields": [
         {"key": "value", "required": True,
          "label": "当たり 1 件の価値 (行ごとに違えば列の名前)"},
         {"key": "cost", "required": True, "label": "行動 1 件の費用"},
         {"key": "effect_kind", "required": False,
          "label": "効果の渡し方 (防げる割合か, 行動したとき − しないとき"
                   "の当たりの確率)",
          "choices": [
              {"value": "share", "label": "防げる割合 (仮定)"},
              {"value": "uplift", "label": "確率の差 (効果のモデル)"},
          ],
          "unset": "既定 (防げる割合 (仮定))"},
         {"key": "effect", "required": True,
          "label": "効果 (割合は 0 から 1, 確率の差は -1 から 1)"},
         {"key": "effect_goal", "required": True,
          "label": "当たりをどうしたいか (減らすなら差が負, 増やすなら正が"
                   "得)",
          "choices": [
              {"value": "reduce", "label": "減らしたい (解約など)"},
              {"value": "raise", "label": "増やしたい (成約など)"},
          ],
          "unset": "選ぶ",
          "when": {"key": "effect_kind", "equals": "uplift"}},
         {"key": "budget", "required": False, "label": "件数の上限 (任意)"},
         {"key": "budget_cost", "required": False,
          "label": "費用の上限 (任意)"},
     ]},
    {"value": "triage", "label": "振り分ける (人が確かめる, 再検査)",
     "fields": [
         {"key": "miss_cost", "required": True, "label": "当たりを見逃す損"},
         {"key": "false_alarm_cost", "required": True,
          "label": "外れを回す損 (空振り)"},
         {"key": "capacity", "required": False,
          "label": "回せる件数の上限 (任意)"},
     ]},
    {"value": "none", "label": "意思決定しない (人が予測を見て決める)",
     "fields": []},
]

# タスクの選択肢
TASKS = [
    {"value": "classification",
     "label": "分類 (どれに当たるか. 解約するか, 不良か)"},
    {"value": "regression",
     "label": "回帰 (値. 時刻の順に関係なく, 行ごとの値)"},
    {"value": "forecasting",
     "label": "時系列の予測 (時刻の順に, 過去の値から先の値)"},
]

# 学習で見る指標 (タスクごと. better は大きいほど良いか小さいほど良いか)
METRICS = [
    {"value": "accuracy", "label": "当たった割合 (正解率. 大きいほど良い)",
     "tasks": ["classification"], "better": "higher"},
    {"value": "loss", "label": "確率の当たり具合 (交差エントロピー. 小さい"
     "ほど良い. 確率で行動を決めるなら)", "tasks": ["classification"],
     "better": "lower"},
    {"value": "mae", "label": "外れの大きさの平均 (平均絶対誤差. 小さいほど"
     "良い)", "tasks": ["regression", "forecasting"], "better": "lower"},
    {"value": "rmse", "label": "大きな外れを重く見た外れ (二乗平均平方根誤差"
     ". 小さいほど良い)", "tasks": ["regression", "forecasting"],
     "better": "lower"},
]

# 問いの並び (上から順)
QUESTIONS: list[dict[str, Any]] = [
    # 1. 課題
    {"id": "issue.owner", "section": 1, "kind": "text", "required": True,
     "title": "誰の課題ですか",
     "help": "課題を抱えていて, 解けたら嬉しい人や部署. 例: 店長, 本部の"
             "仕入れの担当"},
    {"id": "issue.pain", "section": 1, "kind": "text", "required": True,
     "title": "何に困っていますか",
     "help": "機械学習の言葉ではなく, 業務の言葉で. 例: 総菜の廃棄が多く, "
             "夕方には品切れも出る"},
    {"id": "issue.current", "section": 1, "kind": "text", "required": True,
     "title": "今はどうしていますか",
     "help": "今のやり方. 後で比べる相手になる. 例: 前の週の同じ曜日と同じ"
             "数を作る"},

    # 2. 指標
    {"id": "issue.kpi.name", "section": 2, "kind": "text",
     "required": True, "title": "何が変われば解けたと言えますか",
     "help": "業務の数字. 例: 廃棄と欠品の損, 解約の件数, 確かめの手間"},
    {"id": "issue.kpi.unit", "section": 2, "kind": "text",
     "required": False, "title": "その数字の単位は",
     "help": "例: 円 / 月, 件 / 週, 時間 / 日"},
    {"id": "issue.kpi.better", "section": 2, "kind": "choice",
     "required": True, "title": "小さいほど良いですか, 大きいほど良いですか",
     "choices": [{"value": "lower", "label": "小さいほど良い"},
                 {"value": "higher", "label": "大きいほど良い"}],
     "help": "指標の向き"},
    {"id": "issue.kpi.current", "section": 2, "kind": "number",
     "required": False, "title": "今の値は (分かれば)",
     "help": "今の値が分かると, 目指す値と比べて, どれだけ良くなれば価値が"
             "あるかが決まる"},
    {"id": "issue.kpi.target", "section": 2, "kind": "number",
     "required": False, "title": "目指す値は (分かれば)",
     "help": "例: 今の 2 割減"},
    {"id": "issue.guard.name", "section": 2, "kind": "text",
     "required": False,
     "title": "引き換えに悪くしてはいけない数字はありますか (守る指標)",
     "help": "良くしたい数字を追うと悪くなりうる数字. 例: 廃棄を減らすと"
             "増えうる品切れ, 引き止めの連絡への苦情, 確かめる件数を減らす"
             "と増えうる見逃し. 時間のかかる今の処理を速くするなら, 今の処理"
             "と同じ結果になる割合. 無ければ空のまま"},
    {"id": "issue.guard.unit", "section": 2, "kind": "text",
     "required": False, "when": HAS_GUARD, "title": "その数字の単位は",
     "help": "例: %, 件 / 月"},
    {"id": "issue.guard.better", "section": 2, "kind": "choice",
     "required": True, "when": HAS_GUARD,
     "title": "守る指標は, 小さいほど良いですか, 大きいほど良いですか",
     "choices": [{"value": "lower", "label": "小さいほど良い"},
                 {"value": "higher", "label": "大きいほど良い"}],
     "help": "向き. 守る線をどちら側に割ると困るかが決まる"},
    {"id": "issue.guard.line", "section": 2, "kind": "number",
     "required": True, "when": HAS_GUARD,
     "title": "どこまでなら許せますか (守る線)",
     "help": "これを割ったら, 良くしたい数字が良くなっても失敗とする値. "
             "例: 品切れは今と同じ 5 % まで"},
    {"id": "issue.guard.measure", "section": 2, "kind": "text",
     "required": False, "when": HAS_GUARD, "title": "どう測りますか",
     "help": "確かめ (9) と運用 (10) で同じ測り方を使う. 例: 閉店の 2 時間"
             "前に棚が空の品と日を数える"},

    # 3. 分解
    {"id": "decomposition", "section": 3, "kind": "levers",
     "required": True, "title": "課題を生んでいる要因と, 打てる手は",
     "help": "要因ごとに打てる手を書き, その手を機械学習で手伝えるかを選ぶ. "
             "機械学習で手伝わない手も書く (比べるため). 例: 作る量が勘で"
             "決まる → 作る量を予測から決める (機械学習で手伝う)"},

    # 4. 決めること (機械学習を使うかに関わらず, 課題を解くために決めること)
    {"id": "action.what", "section": 4, "kind": "text", "required": True,
     "title": "課題を解くために, 何を決めますか",
     "help": "行動. 予測は決めるために使うので, 何を予測するかより先に"
             "決める. 例: 品ごとの作る量, 引き止めの連絡をする人"},
    {"id": "action.who", "section": 4, "kind": "text", "required": True,
     "title": "誰が決めますか",
     "help": "例: 店長"},
    {"id": "action.when", "section": 4, "kind": "text", "required": True,
     "title": "いつ決めますか",
     "help": "決める時点. 機械学習を使うなら予測の時点になり, 6 の節の"
             "「予測の時点で分かる列」を決める. 例: 前の日の 17 時"},

    # 5. 機械学習の受け持ち
    {"id": "ml.use", "section": 5, "kind": "boolean", "required": True,
     "title": "機械学習を使いますか",
     "help": "分解した手のうち, 機械学習でしか手伝えないものがあるか. ルール"
             "や集計, 今のやり方の改善で足りるなら, 使わないで終えてよい"},
    {"id": "ml.reason", "section": 5, "kind": "text", "required": True,
     "title": "そう決めた理由は",
     "help": "使う場合は, 今のやり方やルールで足りない理由. 使わない場合は, "
             "代わりに何をするか"},
    {"id": "ml.predict", "section": 5, "kind": "text", "required": True,
     "when": USES_ML, "title": "何を予測しますか",
     "help": "4 で決めることのために, 何が分かればよいか. 文で. 例: 作る量"
             "を決めるために, 翌日の品ごとの売れる数"},
    {"id": "ml.row", "section": 5, "kind": "text", "required": True,
     "when": USES_ML, "title": "1 行 (1 件) は何ですか",
     "help": "予測の単位. 例: 店, 品, 日ごとの 1 行"},
    {"id": "data.source", "section": 5, "kind": "source",
     "required": False, "when": USES_ML, "title": "手元のデータは",
     "help": "あれば入れると, 後の問いの答えの候補 (目的の列, 時刻の列, 漏れ"
             "の疑いのある列など) と確かめが出る. 画面ではデータを問題の"
             "設定の置き場に写す (元のファイルは変えない). 課題, 指標, 分解, "
             "決めることはデータ無しで答える (データから考え始めると, 何が"
             "予測できるかが先に立つため)"},
    {"id": "information.columns", "section": 5, "kind": "list",
     "required": True, "when": USES_ML, "title": "使えるデータの列は",
     "help": "データのファイルから読める (画面では「ファイルから読む」). "
             "後の問いで, この中から目的の列と使ってよい列を選ぶ"},
    {"id": "ml.task", "section": 5, "kind": "choice", "required": True,
     "when": USES_ML, "choices": TASKS, "title": "どの種類の問題ですか",
     "help": "目的の値がクラスなら分類, 値なら回帰. 時刻の順に過去から先を"
             "予測するなら時系列の予測. 4 で決めることが量なら回帰か時系列の"
             "予測, 対象を選ぶか振り分けるなら分類"},
    {"id": "ml.target", "section": 5, "kind": "column", "required": True,
     "when": USES_ML, "title": "目的の列は",
     "help": "予測する値の列"},
    {"id": "ml.positive", "section": 5, "kind": "text", "required": False,
     "when": [USES_ML, {"id": "ml.task", "equals": "classification"}],
     "title": "当たりのクラスは (2 クラスの分類で行動を決める場合)",
     "help": "行動の対象になるクラス. 例: 解約, 不良"},
    {"id": "ml.time_column", "section": 5, "kind": "column",
     "required": True,
     "when": [USES_ML, {"id": "ml.task", "equals": "forecasting"}],
     "title": "時刻の列は", "help": "時刻の順に並べ, 分けるのに使う列"},
    {"id": "ml.series_by", "section": 5, "kind": "columns",
     "required": False,
     "when": [USES_ML, {"id": "ml.task", "equals": "forecasting"}],
     "title": "系列を分ける列は (複数の系列が並ぶ場合)",
     "help": "例: 店, 品. 過去の値は系列ごとにずらす"},
    {"id": "ml.available_lag", "section": 5, "kind": "number",
     "required": True,
     "when": [USES_ML, {"id": "ml.task", "equals": "forecasting"}],
     "title": "決める時点 (4 の「いつ決めますか」) で, 目的の値は何期前の"
              "分まで分かっていますか",
     "help": "過去の値の列に使ってよい, 最も新しい値. 例: 前の日の 17 時に"
             "翌日の量を決めるなら, 前の日の売れた数はまだ確定しないので 2 "
             "(2 日前まで). 前の日が締まってから決めるなら 1. 分からない値を"
             "学習に使うと, 学習のときだけ当たり, 使うときに外れる"},
    {"id": "ml.horizon", "section": 5, "kind": "number",
     "required": True,
     "when": [USES_ML, {"id": "ml.task", "equals": "forecasting"}],
     "title": "何期先まで予測しますか",
     "help": "時刻の列の 1 つ分を 1 期として. 例: 翌日だけなら 1, 1 週間分を"
             "まとめて決めるなら 7. 先ほど外れやすく, 予測の時点で分かる列も"
             "変わる"},
    {"id": "ml.baseline", "section": 5, "kind": "text", "required": True,
     "when": USES_ML, "title": "機械学習を使わない比べる相手は",
     "help": "機械学習が要るかは, 素朴なやり方より良いかで決まる. 多くは"
             "1 の「今のやり方」と同じ. 例: 前の週の同じ曜日と同じ数"},
    {"id": "ml.consumer", "section": 5, "kind": "choice",
     "required": False, "when": USES_ML, "choices": CONSUMERS,
     "title": "予測を誰が (何が) 使いますか",
     "help": "使い道で, 確かめ方と要る精度が変わる. 人が見るなら説明, 自動"
             "で決めるなら損得 (7 の節), 後段に渡すなら後段が求める形と精度"},
    {"id": "ml.downstream", "section": 5, "kind": "text",
     "required": False, "when": [USES_ML, DOWNSTREAM],
     "title": "後段は何で, 予測に何を求めますか",
     "help": "形, 単位, 要る精度. 評価は後段の結果で見る (予測の誤差が小さく"
             "ても後段の判定が変わることがある). 例: 点群から寸法を測る決"
             "まり. 1 mm の差が分かる精度が要る"},

    # 6. 使える情報
    {"id": "information.known", "section": 6, "kind": "columns",
     "required": True, "when": USES_ML,
     "title": "予測の時点で分かっている列は",
     "help": "学習に使ってよい列. 予測の時点で分からない列 (当日の客数など)"
             "で学習すると, 学習のときだけ当たり, 使うときに当たらない. 目的"
             "の列そのものは入れない (過去の値はデータの分析で作る)"},
    {"id": "information.unknown", "section": 6, "kind": "columns",
     "required": False, "when": USES_ML,
     "title": "予測の時点では分からない列は",
     "help": "学習に使わない列 (当日の客数など. 予測のときに分からない)"},
    {"id": "data.missing", "section": 6, "kind": "list", "required": False,
     "when": USES_ML, "title": "要るのに手元に無いデータは",
     "help": "課題を解くのに要るが, まだ集めていないデータ. 例: 天気予報の"
             "記録. 要るデータが無いなら, まずデータを集めることが答えになる"
             "こともある"},
    {"id": "information.forbidden", "section": 6, "kind": "columns",
     "required": False, "when": USES_ML,
     "title": "分かっていても使ってはいけない列は",
     "help": "ID, 個人を特定できる情報, 使うと公平でない情報など. 学習に"
             "使わない"},

    # 7. 損得と制約
    {"id": "action.decision", "section": 7, "kind": "decision",
     "required": True, "when": USES_ML, "choices": DECISION_TYPES,
     "title": "行動の損得と制約は",
     "help": "量を決める: 1 単位足りないときの損と余ったときの損. 対象を選ぶ: "
             "当たり 1 件の価値, 行動 1 件の費用, 行動で防げる割合 (仮定), "
             "件数か費用の上限. 振り分ける: 見逃しの損と空振りの損, 回せる件数"
             ". 人が予測を見て決めるなら「意思決定しない」"},
    {"id": "action.current", "section": 7, "kind": "baseline",
     "required": False,
     "when": [USES_ML, {"id": "action.decision.type",
                        "in": ["quantity", "select", "triage"]}],
     "title": "今のやり方をデータで書くと",
     "help": "1 の「今のやり方」を, データの上の決まりにする. 量なら k 期前と同じ"
             " (前の週の同じ曜日なら 7) か, 直前の k 期の平均か, 今のやり方の"
             "値の列. 選ぶと振り分けなら, 行動したかの列. 書くと, データから"
             "今のやり方の損 (指標の今の値の候補) が出る"},
    {"id": "action.basis", "section": 7, "kind": "text", "required": False,
     "when": USES_ML, "title": "損得の値の根拠は",
     "help": "例: 欠品は粗利 200 円, 廃棄は原価 150 円. 根拠の無い値では, "
             "意思決定はもっともらしいだけになる"},
    {"id": "ml.constraints", "section": 7, "kind": "list",
     "required": False, "when": USES_ML,
     "title": "機械学習の仕組みが守る制約は",
     "help": "1 行に 1 つ. 作り方 (既製のモデル, 外のサービス) を選ぶ前に"
             "決める. 例: データを社外に出せない, 商用で使えるライセンス"
             "だけ, 1 件 1 秒以内に返す, 手元の計算機で動く"},

    # 8. 学習の指標
    {"id": "evaluation.metric", "section": 8, "kind": "choice",
     "required": True, "when": USES_ML, "choices": METRICS,
     "title": "学習で何を見ますか",
     "help": "検証用の分割で見て, 最も良いモデルを選ぶ指標"},
    {"id": "evaluation.link", "section": 8, "kind": "text",
     "required": False, "when": USES_ML,
     "title": "その指標が良くなると, 業務の指標はどう良くなりますか",
     "help": "例: 誤差が小さいほど作る量の外れが減り, 廃棄と欠品の損が減る"},

    # 9. 確かめ方
    {"id": "verification.method", "section": 9, "kind": "choice",
     "required": True, "when": USES_ML,
     "choices": [
         {"value": "backtest", "label": "過去のデータでたどり直す"},
         {"value": "trial", "label": "実際に試す"},
         {"value": "both", "label": "両方 (たどり直してから試す)"},
     ],
     "title": "効果をどう確かめますか",
     "help": "行動が結果を変える判断 (引き止めなど) は, 過去のデータからは"
             "効果が分からないので, 試すことが要る"},
    {"id": "verification.backtest", "section": 9, "kind": "text",
     "required": True,
     "when": [USES_ML,
              {"id": "verification.method", "in": ["backtest", "both"]}],
     "title": "どの期間でたどり直しますか", "help": "例: 過去 3 か月"},
    {"id": "verification.trial", "section": 9, "kind": "text",
     "required": True,
     "when": [USES_ML,
              {"id": "verification.method", "in": ["trial", "both"]}],
     "title": "どこで, どれだけの期間試しますか",
     "help": "例: 1 店で 4 週間"},
    {"id": "verification.success", "section": 9, "kind": "text",
     "required": True, "when": USES_ML,
     "title": "どうなれば成功ですか",
     "help": "業務の指標で. 例: 廃棄と欠品の損が今のやり方より 2 割少ない"},

    # 10. 運用
    {"id": "operation.review", "section": 10, "kind": "text",
     "required": True, "when": USES_ML,
     "title": "どれくらいの頻度で見直しますか", "help": "例: 月に 1 回"},
    {"id": "operation.approver", "section": 10, "kind": "text",
     "required": False, "when": USES_ML,
     "title": "変更 (予測の外れ方の見積もりを新しい実績で直すなど) は誰が"
              "承認しますか",
     "help": "例: 本部"},
    {"id": "operation.stop", "section": 10, "kind": "text",
     "required": False, "when": USES_ML, "title": "どうなったら止めますか",
     "help": "例: 損が今のやり方より悪い月が 2 回続いたら"},
    {"id": "operation.remeasure", "section": 10, "kind": "text",
     "required": False, "when": USES_ML,
     "title": "予測の外れ方をいつ確かめ直しますか",
     "help": "外れ方は時間とともに変わる. 例: 月に 1 回, 前の月の実績で"},
]

# 問いの識別子から問い
BY_ID = {question["id"]: question for question in QUESTIONS}
