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

# 受け持ちの場所 (今は 1 つだけ書ける. 問いの識別子は並びの番号を持つ)
PART = "ml.parts.0"

# 作り方が既製のモデルをそのまま (学習しない) か
PRETRAINED = {"id": f"{PART}.method", "equals": "pretrained"}

# 異常の検知の, 度合いの決め方が距離 (学習しない) か
DISTANCE = [{"id": f"{PART}.role", "equals": "predict"},
            {"id": f"{PART}.target.kind", "equals": "anomaly"},
            {"id": f"{PART}.anomaly.score", "equals": "distance"}]

# 学習する場合だけの問い (最良を選ぶ指標). 既製のモデルをそのまま使う場合
# と, 異常の検知の距離に基づく度合いは学習しない
TRAINED = [USES_ML, {"none": [PRETRAINED]},
           {"any": [{"none": [item]} for item in DISTANCE]}]

# 予測するで, 正解の種類が表の列 (列の予測) の場合だけの問い. 既製のモデル
# をそのまま使う場合は学習もデータの分析もしないので含めない
COLUMN_TARGET = [USES_ML, {"id": f"{PART}.role", "equals": "predict"},
                 {"id": f"{PART}.target.kind", "equals": "column"},
                 {"none": [PRETRAINED]}]

# 異常の検知 (予測するで, 正解の種類が異常) の場合だけの問い. 既製の
# モデルをそのまま使う場合は骨組みが無いので含めない
ANOMALY = [USES_ML, {"id": f"{PART}.role", "equals": "predict"},
           {"id": f"{PART}.target.kind", "equals": "anomaly"},
           {"none": [PRETRAINED]}]

# 異常の検知の度合いの決め方
ANOMALY_SCORES = [
    {"value": "autoencoder", "label": "正常の行だけで学ぶ自己符号化器の再構成"
     "の誤差 (行ごとの二乗誤差の平均)"},
    {"value": "distance", "label": "学習用の正常の行の中心からの距離 "
     "(Mahalanobis 距離. 学習しない)"},
]

# 異常の検知の行の id の列にできない名前 (行ごとの度合いの表と, 上位の
# 行の要約の他の列)
ANOMALY_ASSIGNED = ("split", "score", "anomaly", "rank", "column",
                    "column_share")

# 効果を見積もるで, 続けて行動を選ぶ方策でなく, 今ある記録から学ぶ場合
# だけの問い (記録で変えたもの. 既製のモデルをそのまま使う場合は骨組みが
# 無いので含めない)
LOGGED = [USES_ML, {"id": f"{PART}.role", "equals": "effect"},
          {"none": [{"id": f"{PART}.autonomous", "equals": True}]},
          {"id": f"{PART}.collect", "equals": "log"},
          {"none": [PRETRAINED]}]

# 今ある記録で変えたもの (答えなければ配ったか)
VARIED = [
    {"value": "treat", "label": "配ったか (クーポンを配った行と配らなかった"
     "行. 処置の列)"},
    {"value": "price", "label": "値段 (試しに値段を変えた売れ行き. 値段の"
     "列)"},
]

# 記録で変えたものが値段か
PRICED = {"id": f"{PART}.varied", "equals": "price"}

# 今ある記録から, 配ったかの効果を見積もる場合だけの問い (以下「効果の見積
# もり」)
UPLIFT = [*LOGGED, {"none": [PRICED]}]

# 今ある記録から, 商品ごとに利益が最も大きい値段を選ぶ場合だけの問い (以下
# 「値段を変える判断」)
PRICE = [*LOGGED, PRICED]

# 値段を変える判断の商品の id の列と行の id の列にできない名前 (商品ごとの
# 選んだ値段の表の他の列と, ML-EDA の元の値段の列)
PRICE_ASSIGNED = ("split", "rows", "current_price", "cost", "change",
                  "price", "profit", "current_profit", "elasticity",
                  "price_raw")

# 予測するで, 正解の種類が先の期間の需要 (時間をまたぐ量の判断) の場合
# だけの問い. 既製のモデルをそのまま使う場合は骨組みが無いので含めない
DEMAND = [USES_ML, {"id": f"{PART}.role", "equals": "predict"},
          {"id": f"{PART}.target.kind", "equals": "demand"},
          {"none": [PRETRAINED]}]

# 時間をまたぐ量の判断の商品の id の列と日付の列にできない名前 (発注の数の
# 表の他の列. ML-Inference の需要の一式が読む demand_orders.csv の列)
DEMAND_ASSIGNED = ("split", "stock", "quantile", "order", "previous",
                   "demand")

# 時間をまたぐ量の判断の, 学ぶ分位点の水準の既定 (答えなければこれに 0.5
# と水準 τ を足す) と, 先の期間の日数の既定 (ML-EDA の horizon の既定)
DEMAND_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
DEMAND_HORIZON = 7

# 効果の見積もりの結果の種類 (ML-EDA の介入の記録の表の outcome)
UPLIFT_OUTCOMES = [
    {"value": "binary", "label": "起きたか (買ったかの 0 と 1. 効果は確率の"
     "差)"},
    {"value": "amount", "label": "量 (買った額. 効果は値の差)"},
]

# 効果の見積もりの効果の決め方 (答えなければ t_learner. 名前は成果物の
# learner と同じ)
UPLIFT_LEARNERS = [
    {"value": "t_learner", "label": "処置の群と対照の群で 2 つの予測を別々"
     "に学ぶ (T-learner)"},
    {"value": "s_learner", "label": "処置の列を入力の終わりに足して 1 つの"
     "予測で学ぶ (S-learner)"},
]

# 効果の見積もりの行の id の列にできない名前 (行ごとの効果の表の他の列)
UPLIFT_ASSIGNED = ("split", "effect", "treated", "control", "treatment",
                   "outcome")

# 予測するかまとめるで, 既製のモデルをそのまま使う場合だけの問い (一式
# づくりに渡す値. まとめるは既製のモデルの埋め込みで群を作る)
READY = [USES_ML, {"id": f"{PART}.role", "in": ["predict", "group", "rank"]},
         PRETRAINED]

# 似た写真を探す (並べるで, 作り方が既製のモデルをそのまま) の場合だけの
# 問い (検索の判断)
SEARCH = [USES_ML, {"id": f"{PART}.role", "equals": "rank"}, PRETRAINED]

# 入れたデータが画像, 文か音声 (1 件の中身で学ぶ) の場合と, 骨組みが今の
# やり方の結果を 1 件ずつ並べられない場合 (画像, 文か音声の列の予測と, 画像
# の領域と点群). 読み込みの名前は data.CONTENT_LOADERS と同じ (食い違いは
# テストで確かめる)
CONTENT_DATA = [{"id": "data.loader",
                 "in": ["image_folder", "text_lines", "audio_folder"]},
                {"id": "data.options.text_column", "filled": True}]
REPORTED_BASELINE = [
    *TRAINED, {"id": f"{PART}.role", "equals": "predict"},
    {"id": f"{PART}.evaluation.metric",
     "in": ["accuracy", "mae", "rmse", "iou", "chamfer"]},
    {"any": [*CONTENT_DATA,
             {"id": f"{PART}.target.kind", "in": ["mask", "point_cloud"]}]},
]

# 問いの題で名指す, 最良を選ぶ指標の短い名前 (title_metric の問いの
# 「この指標」を置き換える)
METRIC_TITLES = {"accuracy": "正解率", "mae": "平均絶対誤差",
                 "rmse": "二乗平均平方根誤差", "iou": "IoU",
                 "chamfer": "Chamfer 距離"}

# 守る指標を問う場合 (まとめるの受け持ちでは要らない. 群は人が解釈して
# 使い, 使われたかは運用の後に見るため)
GUARD_ASKED = [{"any": [{"none": [USES_ML]},
                        {"none": [{"id": f"{PART}.role",
                                   "equals": "group"}]}]}]

# 守る指標を書いた場合だけの問い
HAS_GUARD = [*GUARD_ASKED, {"id": "issue.guard.name", "filled": True}]

# たどり直しで数えられる守る指標の量 (意思決定の型ごと. 名前から説明と
# 向き. lower は小さいほど良い. 割合は名前が _rate で終わり, 0 から 1).
# ML-Inference の GUARD_COUNTS を写す (食い違いはテストで確かめる)
GUARD_COUNTS = {
    "quantity": {
        "shortage_rate": {"description": "足りなかった行の割合",
                          "better": "lower"},
        "shortage": {"description": "足りなかった量の合計 (目的の列の単位)",
                     "better": "lower"},
        "surplus": {"description": "余った量の合計 (目的の列の単位)",
                    "better": "lower"},
    },
    "select": {
        "selected": {"description": "選んだ件数", "better": "lower"},
        "selected_cost": {"description": "選んだ行の費用の合計",
                          "better": "lower"},
        "selected_hit_rate": {"description": "選んだ行のうち当たりの割合",
                              "better": "higher"},
    },
    "triage": {
        "routed": {"description": "回した件数", "better": "lower"},
        "missed": {"description": "見逃した件数 (回さなかった当たり)",
                   "better": "lower"},
        "false_alarms": {"description": "空振りの件数 (回した外れ)",
                         "better": "lower"},
    },
}

# 守る指標が数える量を指さない答え (受け渡しの counted は None)
NOT_COUNTED = "none"

# たどり直しで数える量の選択肢 (decisions は使える意思決定の型. 指さない
# は全ての型)
GUARD_COUNTED = [
    {"value": name, "label": count["description"], "decisions": [kind]}
    for kind, counts in GUARD_COUNTS.items()
    for name, count in counts.items()
] + [{"value": NOT_COUNTED, "label": "指さない (過去のデータでは数えられ"
      "ない. 課題の場で数える)", "decisions": list(GUARD_COUNTS)}]

# 守る指標をたどり直しで数えられる場合 (列の予測で, 数えられる量のある
# 意思決定の型. 効果を確率の差で渡した対象を選ぶ型はたどり直せない)
COUNTABLE_GUARD = [
    *COLUMN_TARGET, *HAS_GUARD,
    {"id": "action.decision.type", "in": list(GUARD_COUNTS)},
    {"none": [{"id": "action.decision.effect_kind", "equals": "uplift"}]},
]

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
DOWNSTREAM = {"id": f"{PART}.consumer", "in": ["system", "model"]}

# 受け持ち (課題から見た, 機械学習の最後の出力)
ROLES = [
    {"value": "predict", "label": "予測する (正解が 1 つに決まるものを当てる"
     ". 値, クラス, 領域, 3D の形, 異常も)"},
    {"value": "generate", "label": "作る (正解が 1 つに決まらない中身. 文章"
     "の下書き, 画像, 合成データ)"},
    {"value": "rank", "label": "並べる (多くの候補から要求に合うものを順に."
     " 推薦, 検索)"},
    {"value": "group", "label": "まとめる (正解の無いデータを, 人が分かる"
     "まとまりに. 顧客の群分け)"},
    {"value": "effect", "label": "効果を見積もる (行動したときとしないとき"
     "の差. 誰に行動するか)"},
]

# 1 件の単位
UNITS = [
    {"value": "row", "label": "行 (表の 1 行)"},
    {"value": "bundle", "label": "まとまり (1 回の撮影, 文書と添付)"},
    {"value": "pair", "label": "対 (入力と正解のファイル)"},
    {"value": "query", "label": "要求と候補の組"},
    {"value": "collection", "label": "集まり (文書の束)"},
    {"value": "window", "label": "時間の窓"},
]

# 出力の形 (roles は合う受け持ち)
OUTPUTS = [
    {"value": "class", "label": "クラス", "roles": ["predict"]},
    {"value": "value", "label": "値", "roles": ["predict"]},
    {"value": "distribution", "label": "分布 (分位点, 区間)",
     "roles": ["predict"]},
    {"value": "region", "label": "位置や領域", "roles": ["predict"]},
    {"value": "shape", "label": "3D の形 (点群, メッシュ)",
     "roles": ["predict", "generate"]},
    {"value": "text", "label": "文", "roles": ["predict", "generate"]},
    {"value": "image", "label": "画像", "roles": ["generate"]},
    {"value": "table", "label": "表 (合成のデータ)", "roles": ["generate"]},
    {"value": "ranking", "label": "順位", "roles": ["rank"]},
    {"value": "cluster", "label": "群", "roles": ["group"]},
    {"value": "effect", "label": "行動ごとの差", "roles": ["effect"]},
]

# 正解の出どころ
TARGET_SOURCES = [
    {"value": "column", "label": "データの列"},
    {"value": "files", "label": "対のファイル (入力ごとの正解のファイル)"},
    {"value": "self", "label": "自分自身 (隠して当てる, 元に戻す)"},
    {"value": "none", "label": "無い"},
    {"value": "few", "label": "少し (確かめの組だけ)"},
    {"value": "process", "label": "今の処理の出力 (シミュレーション, 今の"
     "計算)"},
    {"value": "objective", "label": "目的の式 (方程式の残差, 再描画の誤差)"},
    {"value": "feedback", "label": "行動の結果 (見せたものにだけ反応)"},
    {"value": "intervention", "label": "介入の記録 (無作為の試験, 記録した"
     "方策の確率)"},
    {"value": "human", "label": "人の判断と好み"},
    {"value": "instruction", "label": "指示 (プロンプト) と例"},
]

# 正解の種類 (予測すると作るで問う. roles は合う受け持ち)
TARGET_KINDS = [
    {"value": "column", "label": "表の列 (クラスか値)", "roles": ["predict"]},
    {"value": "mask", "label": "画像の領域", "roles": ["predict"]},
    {"value": "point_cloud", "label": "点群",
     "roles": ["predict", "generate"]},
    {"value": "text", "label": "文", "roles": ["predict", "generate"]},
    {"value": "image", "label": "画像", "roles": ["generate"]},
    {"value": "table", "label": "表 (合成のデータ)", "roles": ["generate"]},
    {"value": "anomaly", "label": "異常か正常か (正常の行が多い表. 正解は"
     "無いか少しだけ)", "roles": ["predict"]},
    {"value": "demand", "label": "先の期間の需要 (商品ごとの日ごとの売れた"
     "数の記録から, 先の期間の売れる数の合計を分位点で)",
     "roles": ["predict"]},
]

# 作り方
METHODS = [
    {"value": "rule", "label": "規則 (機械学習でない)"},
    {"value": "pretrained", "label": "既製のモデルをそのまま (指示, 例, "
     "検索を足す)"},
    {"value": "head", "label": "既製の特徴の上に頭だけ学習"},
    {"value": "partial", "label": "一部を微調整"},
    {"value": "finetune", "label": "全体を微調整"},
    {"value": "scratch", "label": "一から学習"},
]

# 既製のモデルの出どころ (ML-Inference の READY_LIBRARIES の名前)
READY_LIBRARIES = [
    {"value": "torchvision", "label": "torchvision (画像の分類の学習済みの"
     "重み)"},
]

# 効果を見積もるの受け持ちの場合だけの問い
EFFECT = [USES_ML, {"id": f"{PART}.role", "equals": "effect"}]

# 並べるで学習する場合だけの問い (反応の表の列と, 並べる件数). 既製の
# モデルをそのまま使う場合は骨組みが無いので含めない
RANK = [USES_ML, {"id": f"{PART}.role", "equals": "rank"},
        {"none": [PRETRAINED]}]

# まとめるで学習する場合だけの問い (群を作る列, 群の数, 安定性の数え方).
# 既製のモデルをそのまま使う場合は骨組みが無いので含めない
GROUP = [USES_ML, {"id": f"{PART}.role", "equals": "group"},
         {"none": [PRETRAINED]}]

# まとめるの, 作り方に依らない問い (群の数と安定性の数え方. 既製の埋め込み
# の群分けも群の中心を作り直して数える)
GROUP_ANY = [USES_ML, {"id": f"{PART}.role", "equals": "group"}]

# 使えるデータの列を問う場合 (列の予測と異常の検知と, 並べるかまとめるで
# 学習する場合と, 効果の見積もりと, 値段を変える判断)
COLUMNS_ASKED = [{"any": [
    {"all": [USES_ML, {"none": [PRETRAINED]},
             {"id": f"{PART}.role", "in": ["predict", "rank", "group"]},
             {"any": [{"id": f"{PART}.role", "in": ["rank", "group"]},
                      {"id": f"{PART}.target.kind",
                       "in": ["column", "anomaly", "demand"]}]}]},
    {"all": LOGGED},
]}]

# 意思決定の型と損得を問う場合 (列の予測と, 効果の見積もりと, 値段を変える
# 判断と, 時間をまたぐ量の判断)
DECIDED = [{"any": [{"all": COLUMN_TARGET}, {"all": LOGGED},
                    {"all": DEMAND}, {"all": SEARCH}]}]

# 群の安定性の数え方 (作り直し方)
GROUP_STABILITIES = [
    {"value": "seed", "label": "種を変えて作り直す (同じ行で, 群の中心の"
     "初めの置き方を変える)"},
    {"value": "sample", "label": "標本を変えて作り直す (行の 8 割を引き"
     "直す)"},
]

# 群を作る列にできない名前 (群の要約の鍵と, ML-EDA の 1 件の決まった鍵.
# ML-EDA の群の要約の関数 describe_groups が受け付けない名前と同じ値)
GROUP_TAKEN = ("group", "rows", "share", "stability", "id", "split")

# 行を見分ける列にできない名前 (群の割り当ての表の他の列)
GROUP_ASSIGNED = ("group", "split")

# 続けて行動を選ぶ方策の場合だけの問い (守りの問い)
AUTONOMOUS = [*EFFECT, {"id": f"{PART}.autonomous", "equals": True}]

# 行動の結果の集め方 (online と simulator は試して集める)
COLLECTS = [
    {"value": "log", "label": "今ある記録から (新しく試さない)"},
    {"value": "online", "label": "実際に試しながら集める"},
    {"value": "simulator", "label": "シミュレータで試して集める"},
]

# シミュレータ
SIMULATORS = [
    {"value": "mujoco", "label": "MuJoCo (物理のシミュレータ)"},
    {"value": "other", "label": "その他 (Gymnasium の環境, 自分で作った環境,"
     " 他の道具)"},
]

# 方策が守りを外れたときの動き (名前は ML-Found の make_guards の fallback)
FALLBACKS = [
    {"value": "stop", "label": "その場で止める"},
    {"value": "handover", "label": "元の制御 (今のやり方) に戻す"},
    {"value": "action", "label": "決めた行動にする"},
]

# 意思決定の型と, 型ごとの欄 (画面はこれだけで描く. 項目は ML-Inference の
# 形. required は要る欄, choices は選んで書く欄, unset は選ばないときの意味,
# when は他の欄の値で要るかが決まる欄. fits は合う受け持ち (方策でない効果の
# 見積もりは配る判断だけ))
DECISION_TYPES = [
    {"value": "quantity", "label": "量を決める (仕入れ, 補充, 人員)",
     "fits": [{"role": "predict", "kinds": ["column"]}], "fields": [
         {"key": "under_cost", "required": True,
          "label": "1 単位足りないときの損 (欠品の粗利など)"},
         {"key": "over_cost", "required": True,
          "label": "1 単位余ったときの損 (捨てるなら原価など)"},
         {"key": "stock", "required": False,
          "label": "今ある量 (任意. 行ごとに違えば列の名前)"},
         {"key": "step", "required": False, "label": "量の刻み (任意)"},
     ]},
    {"value": "select", "label": "対象を選ぶ (引き止め, 営業先)",
     "fits": [{"role": "predict", "kinds": ["column"]}], "fields": [
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
     "fits": [{"role": "predict", "kinds": ["column"]}], "fields": [
         {"key": "miss_cost", "required": True, "label": "当たりを見逃す損"},
         {"key": "false_alarm_cost", "required": True,
          "label": "外れを回す損 (空振り)"},
         {"key": "capacity", "required": False,
          "label": "回せる件数の上限 (任意)"},
     ]},
    {"value": "none", "label": "意思決定しない (人が予測を見て決める)",
     "fits": [{"role": "predict", "kinds": ["column"]}], "fields": []},
    {"value": "treat", "label": "効果の上位から配る (クーポン, 案内)",
     "fits": [{"role": "effect", "policy": False, "varied": ["treat"]}],
     "fields": [
         {"key": "budget", "required": True,
          "label": "配る数の上限 (守る指標. 1 以上の整数)"},
     ]},
    {"value": "price", "label": "利益が最も大きい値段を選ぶ (商品ごとの値段)",
     "fits": [{"role": "effect", "policy": False, "varied": ["price"]}],
     "fields": [
         {"key": "candidates", "required": True, "list": True,
          "label": "値段の候補 (今の値段に対する変える割合の並び. 0 を含む. "
                   "1 割上げるなら 0.1)"},
         {"key": "max_change", "required": True,
          "label": "値段を変える幅の上限 (守る指標. 変える割合の絶対値. "
                   "1 割までなら 0.1)"},
     ]},
    {"value": "order", "label": "損の比で発注の数を決める (商品ごとの先の"
     "期間の発注)",
     "fits": [{"role": "predict", "kinds": ["demand"]}], "fields": [
         {"key": "shortage_cost", "required": True,
          "label": "1 つ足りないときの損 (売り逃した利益など. 商品 1 つ"
                   "あたりの正の数)"},
         {"key": "holding_cost", "required": True,
          "label": "1 つ余ったときの損 (捨てる原価か持ち越す費用. 同じお金"
                   "の単位の正の数)"},
         {"key": "max_order", "required": True,
          "label": "発注の数の上限 (守る指標. 置き場所の上限. 0 以上の数)"},
     ]},
    {"value": "search", "label": "近い順に上位 K 件を返す (問いの写真に似た"
     "写真)",
     "fits": [{"role": "rank", "methods": ["pretrained"]}], "fields": [
         {"key": "k", "required": True,
          "label": "問いごとに返す件数 K (1 以上の整数. Recall@K と NDCG@K "
                   "もこの K で数える)"},
         {"key": "max_latency_ms", "required": False,
          "label": "1 件の問いの待ち時間の上限 (守る指標. ミリ秒の正の数. "
                   "任意)"},
     ]},
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

# 最良のモデルを選ぶ指標. fits は合う受け持ち, 正解の種類, タスク, 正解の
# 出どころの組の並び (kinds, tasks, sources が無ければ問わない). better は大きいほど良いか小さい
# ほど良いか. tasks は版 1 の形 (列の予測のタスク). policy は続けて行動を
# 選ぶ方策 (方策かの答え) だけに合う目印. loss は 3 つ (分類の交差
# エントロピーと, 他の受け持ちの損失と, 評価のエピソードの報酬の和の平均の符号を反転した値.
# 監視する名前は同じ valid/loss なので値も同じ). 選択肢は questions の choices (今の答えで絞り, 同じ値は
# 1 つ) の利用. reward は方策の既定なので, 方策の選択肢の先頭
METRICS = [
    {"value": "accuracy", "label": "当たった割合 (正解率. 大きいほど良い)",
     "tasks": ["classification"], "better": "higher",
     "fits": [{"role": "predict", "kinds": ["column"],
               "tasks": ["classification"]}]},
    {"value": "loss", "label": "確率の当たり具合 (交差エントロピー. 小さい"
     "ほど良い. 確率で行動を決めるなら)", "tasks": ["classification"],
     "better": "lower",
     "fits": [{"role": "predict", "kinds": ["column"],
               "tasks": ["classification"]}]},
    {"value": "mae", "label": "外れの大きさの平均 (平均絶対誤差. 小さいほど"
     "良い)", "tasks": ["regression", "forecasting"], "better": "lower",
     "fits": [{"role": "predict", "kinds": ["column"],
               "tasks": ["regression", "forecasting"]},
              {"role": "effect", "policy": False, "varied": ["price"]},
              {"role": "predict", "kinds": ["demand"]}]},
    {"value": "rmse", "label": "大きな外れを重く見た外れ (二乗平均平方根誤差"
     ". 小さいほど良い)", "tasks": ["regression", "forecasting"],
     "better": "lower",
     "fits": [{"role": "predict", "kinds": ["column"],
               "tasks": ["regression", "forecasting"]}]},
    {"value": "reward", "label": "評価のエピソードの報酬の和の平均 (大きい"
     "ほど良い. 方策の既定)", "better": "higher",
     "fits": [{"role": "effect", "policy": True}]},
    {"value": "loss", "label": "検証用の分割での損失 (小さいほど良い)",
     "better": "lower",
     "fits": [{"role": "predict",
               "kinds": ["mask", "point_cloud", "text", "anomaly",
                         "demand"]},
              {"role": "generate"}, {"role": "rank"}, {"role": "group"},
              {"role": "effect", "policy": False}]},
    {"value": "loss", "label": "評価のエピソードの報酬の和の平均の符号を"
     "反転した値 (小さいほど良い)", "better": "lower",
     "fits": [{"role": "effect", "policy": True}]},
    {"value": "iou", "label": "領域の重なり (IoU. 大きいほど良い)",
     "better": "higher", "fits": [{"role": "predict", "kinds": ["mask"]}]},
    {"value": "chamfer", "label": "点群どうしの近さ (Chamfer 距離. 小さい"
     "ほど良い)", "better": "lower",
     "fits": [{"role": "predict", "kinds": ["point_cloud"]},
              {"role": "generate", "kinds": ["point_cloud"]}]},
    {"value": "cer", "label": "文字の誤り率 (CER. 小さいほど良い)",
     "better": "lower", "fits": [{"role": "predict", "kinds": ["text"]}]},
    {"value": "fid", "label": "作った画像と本物の分布の近さ (FID. 小さい"
     "ほど良い)", "better": "lower",
     "fits": [{"role": "generate", "kinds": ["image"]}]},
    {"value": "column_distance", "label": "作った表と本物の表の列ごとの分布"
     "の近さ (数の列は KS 統計量, 種類の列は全変動距離の列の平均. 小さい"
     "ほど良い)", "better": "lower",
     "fits": [{"role": "generate", "kinds": ["table"]}]},
    {"value": "average_precision", "label": "度合いの順の異常の当たり (PR-AUC"
     ". 大きいほど良い)", "better": "higher",
     "fits": [{"role": "predict", "kinds": ["anomaly"], "sources": ["few"]}]},
    {"value": "precision_at_k", "label": "度合いの上位 K 件 (人が確かめる"
     "件数の上限) のうち異常の割合 (上位 K 件の適合率. 大きいほど良い)",
     "better": "higher",
     "fits": [{"role": "predict", "kinds": ["anomaly"], "sources": ["few"]}]},
    {"value": "recall_at_k", "label": "上位 K 件に正解が入る割合 (Recall@K"
     ". 大きいほど良い)", "better": "higher", "fits": [{"role": "rank"}]},
    {"value": "ndcg", "label": "順位の良さ (NDCG. 大きいほど良い)",
     "better": "higher", "fits": [{"role": "rank"}]},
    {"value": "silhouette", "label": "群のまとまり (シルエット係数. 大きい"
     "ほど良い)", "better": "higher", "fits": [{"role": "group"}]},
    {"value": "qini", "label": "効果の順の当たり (Qini 係数. 大きいほど"
     "良い)", "better": "higher",
     "fits": [{"role": "effect", "varied": ["treat"]}]},
    {"value": "other", "label": "その他 (名前と, 大きいほど良いか小さい"
     "ほど良いかを書く)", "better": None, "fits": [{}]},
]

# 今のやり方をデータで書く種類 (action.current の type). 欄 (fields) は
# 意思決定の型の欄と同じ形, decisions は使える意思決定の型
BASELINE_TYPES = [
    {"value": "lag", "label": "k 期前と同じ (前の週の同じ曜日なら 7)",
     "decisions": ["quantity"],
     "fields": [{"key": "periods", "label": "期の数 k (1 以上の整数)",
                 "required": True}]},
    {"value": "mean", "label": "直前の k 期の平均", "decisions": ["quantity"],
     "fields": [{"key": "periods", "label": "期の数 k (1 以上の整数)",
                 "required": True}]},
    {"value": "column", "label": "今のやり方の値の列 (置いた量, 行動したか)",
     "decisions": ["quantity", "select", "triage"],
     "fields": [{"key": "column", "label": "今のやり方の値の列",
                 "required": True}]},
]

# 受け渡しのデータの分析の分割の名前 (handoff の data.split. ML-EDA の
# split の method). grouped は相手をまとめる列の同じ値を同じ分割に入れるか
SPLITS = [
    {"value": "time", "label": "時刻の順 (後ろの期間を検証用とテスト用に)",
     "grouped": False},
    {"value": "group", "label": "相手ごと (同じ相手は同じ分割に)",
     "grouped": True},
    {"value": "random", "label": "無作為", "grouped": False},
    {"value": "stratified", "label": "層別 (クラスの割合をそろえて無作為に)",
     "grouped": False},
    {"value": "group_time",
     "label": "相手ごとと時刻の順 (後から現れた相手を検証用とテスト用に)",
     "grouped": True},
    {"value": "stratified_group",
     "label": "相手ごとの層別 (同じ相手は同じ分割に, クラスの割合をそろえて"
     ". 分類だけ)", "grouped": True},
    {"value": "none", "label": "分けない (全て学習用)", "grouped": False},
]

# 版 1 のファイルで書ける指標
V1_METRICS = ["accuracy", "loss", "mae", "rmse"]

# 問いの並び (上から順)
QUESTIONS: list[dict[str, Any]] = [
    # 1. 課題
    {"id": "issue.owner", "section": 1, "kind": "text", "required": True,
     "title": "誰の課題ですか",
     "help": "課題を抱えていて, 解けたら嬉しい人や部署. 例: 店長, 本部の"
             "仕入れの担当 (仕事). 計測の研究室 (研究). ゲームの作り手 "
             "(娯楽)"},
    {"id": "issue.pain", "section": 1, "kind": "text", "required": True,
     "title": "何に困っていますか",
     "help": "機械学習の言葉ではなく, 課題の言葉で. 例: 総菜の廃棄が多く, "
             "夕方には品切れも出る (仕事). 測ったものから元の形を作り直す"
             "と誤差が大きく, 1 件に時間もかかる (研究). 遊ぶ人がすぐに"
             "飽きてやめる (娯楽)"},
    {"id": "issue.current", "section": 1, "kind": "text", "required": True,
     "title": "今はどうしていますか",
     "help": "今のやり方. 後で比べる相手になる. 例: 前の週の同じ曜日と同じ"
             "数を作る"},

    # 2. 指標
    {"id": "issue.kpi.name", "section": 2, "kind": "text",
     "required": True, "title": "何が変われば解けたと言えますか",
     "help": "課題の数字. 例: 廃棄と欠品の損, 解約の件数 (仕事). 再構成の"
             "誤差, 1 件の処理時間 (研究). 続けて遊ぶ時間 (娯楽)"},
    {"id": "issue.kpi.unit", "section": 2, "kind": "text",
     "required": False, "title": "その数字の単位は",
     "help": "例: 円 / 月, 件 / 週, mm, 秒 / 件, 分 / 回"},
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
     "required": False, "when": GUARD_ASKED,
     "title": "引き換えに悪くしてはいけない数字はありますか (守る指標)",
     "help": "良くしたい数字を追うと悪くなりうる数字. 例: 廃棄を減らすと"
             "増えうる品切れ, 引き止めの連絡への苦情, 確かめる件数を減らす"
             "と増えうる見逃し. 時間のかかる今の処理を速くするなら, 今の処理"
             "と同じ結果になる割合. 無ければ空のまま"},
    {"id": "issue.guard.unit", "section": 2, "kind": "text",
     "required": False, "when": HAS_GUARD, "title": "その数字の単位は",
     "help": "例: 件 / 月. たどり直しで割合の量 (7) を数えるなら, 値と線"
             "は 0 から 1 なので, 空のままか「割合 (0 から 1)」と書く (% と"
             "書くと線の 0.05 が 0.05 % に見える)"},
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
             "例: 品切れは今と同じ 5 % まで (たどり直しで割合の量を数えるなら, "
             "0 から 1 で 0.05)"},
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
    {"id": f"{PART}.role", "section": 5, "kind": "choice", "required": True,
     "when": USES_ML, "choices": ROLES,
     "title": "機械学習は何を受け持ちますか",
     "help": "課題から見た, 機械学習の最後の出力. 正解が 1 つに決まるなら"
             "予測する (訳や要約のように多くの答えが許されるなら作る). 今の"
             "時間のかかる処理を速くするなら, 予測するか作るで, 正解の出どころ"
             "を今の処理の出力にする. 今は 1 つだけ書ける"},
    {"id": f"{PART}.unit", "section": 5, "kind": "choice", "required": True,
     "when": USES_ML, "choices": UNITS, "title": "1 件の単位は",
     "help": "データの書き出しの形, 推論の出力の形, 分割の単位が決まる. 例: "
             "表の 1 行, 1 回の撮影 (まとまり), 画像と正解の領域 (対)"},
    {"id": f"{PART}.output", "section": 5, "kind": "choice",
     "required": True, "when": USES_ML, "choices": OUTPUTS,
     "title": "出力の形は",
     "help": "推論の一式, 結論と意思決定, 要る精度の決め方が変わる"},
    {"id": f"{PART}.target.source", "section": 5, "kind": "choice",
     "required": True, "when": USES_ML, "choices": TARGET_SOURCES,
     "title": "正解は誰が, いつ, どう確定させますか (正解の出どころ)",
     "help": "確かめ方と, 漏れの型がほぼこれで決まる. 例: データの列にある, "
             "今の処理で作れる, 無い"},
    {"id": f"{PART}.target.process.name", "section": 5, "kind": "text",
     "required": True,
     "when": [USES_ML, {"id": f"{PART}.target.source",
                        "equals": "process"}],
     "title": "正解を作る今の処理の名前は",
     "help": "評価と結論で, その処理を超えられないことを示すために残す. "
             "例: 今の計算の道具の名前と版"},
    {"id": f"{PART}.target.process.settings", "section": 5, "kind": "text",
     "required": False,
     "when": [USES_ML, {"id": f"{PART}.target.source",
                        "equals": "process"}],
     "title": "その処理の設定は",
     "help": "正解の質を決める設定 (細かさ, 回数など). 無ければ空のまま"},
    {"id": f"{PART}.method", "section": 5, "kind": "choice",
     "required": True, "when": USES_ML, "choices": METHODS,
     "title": "どう作りますか (作り方)",
     "help": "規則と既製のモデルから試し, 足りなければ学習を足す. 学習しない"
             "なら, モデルの作成と学習の段階を通らない. 例: 表の予測は一から"
             "学習, 少ない画像は既製の特徴の上に頭だけ"},
    {"id": f"{PART}.target.kind", "section": 5, "kind": "choice",
     "required": True,
     "when": [USES_ML, {"id": f"{PART}.role",
                        "in": ["predict", "generate"]}],
     "choices": TARGET_KINDS, "title": "正解 (出力) の種類は",
     "help": "表の列なら, 続けて目的の列とタスクを問う"},
    {"id": f"{PART}.ready.library", "section": 5, "kind": "choice",
     "required": True, "when": READY, "choices": READY_LIBRARIES,
     "title": "既製のモデルはどこのものですか (出どころ)",
     "help": "学習済みの重みを配っているライブラリ. 今は torchvision の画像"
             "の分類だけ"},
    {"id": f"{PART}.ready.model", "section": 5, "kind": "text",
     "required": True, "when": READY, "title": "既製のモデルの名前は",
     "help": "出どころでのモデルの名前. 例: resnet18 (torchvision)"},
    {"id": f"{PART}.ready.weights", "section": 5, "kind": "text",
     "required": True, "when": READY, "title": "重みの名前 (版) は",
     "help": "出どころでの学習済みの重みの名前. 名前ごとに重みが決まる. "
             "例: IMAGENET1K_V1 (torchvision の ImageNet で学んだ重み). "
             "DEFAULT と書くと, 一式を作るときの出どころの既定の重みを使い, "
             "一式にはその版の名前 (IMAGENET1K_V1 など) を残す. 出どころの"
             "既定が変わると別の重みになるので, 決まった重みを使うなら版"
             "まで書く"},
    {"id": f"{PART}.ready.sha256", "section": 5, "kind": "text",
     "required": False, "when": READY,
     "title": "重みの hash (SHA-256) は (決めておくなら)",
     "help": "書くと, 一式を作るときに読んだ重みの hash と比べ, 違えば作ら"
             "ない. 16 進の 64 文字. 分からなければ空のまま (一式を作ると"
             "求めた hash が出るので, それを書くと次から同じ重みに固定でき"
             "る)"},
    {"id": f"{PART}.what", "section": 5, "kind": "text", "required": True,
     "when": USES_ML, "title": "何を出させますか",
     "help": "4 で決めることのために, 何が分かればよいか. 文で. 例: 作る量"
             "を決めるために, 翌日の品ごとの売れる数"},
    {"id": f"{PART}.item", "section": 5, "kind": "text", "required": True,
     "when": USES_ML, "title": "1 件は何ですか",
     "help": "出力の単位. 例: 店, 品, 日ごとの 1 行"},
    {"id": f"{PART}.autonomous", "section": 5, "kind": "boolean",
     "required": False, "when": EFFECT,
     "title": "機械学習が, 続けて自分で行動を選びますか (方策)",
     "help": "はい: 機械の動かし方のように, 観測を見て次の行動を次々に決め"
             "る (人は 1 回ごとに確かめない). 守りを問う. いいえ: 誰に行動"
             "するかを選び, 人や決まりが行動する"},
    {"id": f"{PART}.collect", "section": 5, "kind": "choice",
     "required": False, "when": EFFECT, "choices": COLLECTS,
     "title": "行動の結果のデータは, どう集めますか",
     "help": "試して集めるなら, 学習で環境とのやり取り (強化学習) の道具が"
             "要る. 実際に試すなら, 外れたときの損に気を付ける"},
    {"id": f"{PART}.varied", "section": 5, "kind": "choice",
     "required": False, "when": LOGGED, "choices": VARIED,
     "title": "記録で変えたのは, 配ったかですか, 値段ですか",
     "help": "配ったか (クーポンを配った行と配らなかった行) なら, 顧客ごとの"
             "効果を見積もり, 効果の高い順に配る. 値段 (試しに値段を変えた"
             "売れ行き) なら, 値段を入力に入れて売れた数を予測し, 商品ごとに"
             "候補の中で利益が最も大きい値段を選ぶ. 答えなければ配ったか"},
    {"id": f"{PART}.simulator", "section": 5, "kind": "choice",
     "required": True,
     "when": [*EFFECT, {"id": f"{PART}.collect", "equals": "simulator"}],
     "choices": SIMULATORS, "title": "どのシミュレータを使いますか",
     "help": "MuJoCo は pip で入るが, 描画には OpenGL が要る"},
    {"id": f"{PART}.env_id", "section": 5, "kind": "text",
     "required": False,
     "when": [*AUTONOMOUS, {"none": [{"id": f"{PART}.collect",
                                      "equals": "log"}]}],
     "title": "Gymnasium の環境の登録名は (あれば)",
     "help": "例: CartPole-v1, Pendulum-v1. 答えると, 書き出す学習の"
             "スクリプトがその環境で方策を学ぶ. 自分で作る環境 (実機との"
             "つなぎ, 自作のシミュレータ) なら空のまま. 骨組みに環境の雛形 "
             "(env.py) を書き出す"},
    {"id": f"{PART}.ros2", "section": 5, "kind": "boolean",
     "required": False, "when": AUTONOMOUS,
     "title": "機械を ROS2 で動かしますか",
     "help": "ROS2 は pip では入らないので, 公式の手順で先に入れる. 実機で"
             "動かすのは道具の外 (方策と守りを書き出すところまで)"},
    {"id": "data.source", "section": 5, "kind": "source",
     "required": False, "when": USES_ML, "title": "手元のデータは",
     "help": "あれば入れると, 後の問いの答えの候補 (目的の列, 時刻の列, 漏れ"
             "の疑いのある列など) と確かめが出る. 画面ではデータを問題の"
             "設定の置き場に写す (元のファイルは変えない). 課題, 指標, 分解, "
             "決めることはデータ無しで答える (データから考え始めると, 何が"
             "予測できるかが先に立つため)"},
    {"id": "information.columns", "section": 5, "kind": "list",
     "required": True, "when": COLUMNS_ASKED, "title": "使えるデータの列は",
     "help": "データのファイルから読める (画面では「ファイルから読む」). "
             "後の問いで, この中から目的の列と使ってよい列 (並べるなら, "
             "利用者, 品物, 時刻の列) を選ぶ"},
    {"id": f"{PART}.rank.user", "section": 5, "kind": "column",
     "required": True, "when": RANK, "title": "利用者の列は",
     "help": "反応 (利用者が品物を買った, 見たなど) の表 (1 行が 1 つの"
             "反応) の, 反応した利用者の列. 利用者ごとに品物を並べる. 例: "
             "member_id"},
    {"id": f"{PART}.rank.item", "section": 5, "kind": "column",
     "required": True, "when": RANK, "title": "品物の列は",
     "help": "反応の表の, 反応された品物の列. 例: item_id"},
    {"id": f"{PART}.rank.time", "section": 5, "kind": "column",
     "required": True, "when": RANK, "title": "反応の時刻の列は",
     "help": "反応を時刻の順に並べ, 古い反応で学び, 新しい反応で確かめる "
             "(無作為に分けると, 先の反応を学習に使って確かめが甘くなる). "
             "数か日時の列"},
    {"id": f"{PART}.rank.reaction", "section": 5, "kind": "column",
     "required": False, "when": RANK,
     "title": "反応の強さの列は (あれば)",
     "help": "数の列. 0 より大きい行だけを反応とみなす (例: 買った数). "
             "評価の点数のように全ての行が 0 より大きい列では, 見たい反応を "
             "1, 他を 0 にした列を作ってから選ぶ. 無ければ空のまま (全ての"
             "行を反応とみなす)"},
    {"id": f"{PART}.rank.shown", "section": 5, "kind": "column",
     "required": False, "when": RANK,
     "title": "見せただけの印の列は (あれば)",
     "help": "利用者に見せたが選ばれなかった品物の行に真 (true か 1) を"
             "付けた列. 選んだ行は偽 (false か 0) か空のまま. 1 行が見せた"
             "品物 1 件で, 選んだ反応と見せただけの記録を 1 つの表に置く. "
             "あれば, 見せただけの品物を学習の負の例にし, 1 回の提示の中で"
             "選んだ品物を上に並べられたかも確かめる. 無ければ空のまま "
             "(選んだ反応だけで学ぶ). 例: shown_only"},
    {"id": f"{PART}.rank.presentation", "section": 5, "kind": "column",
     "required": False,
     "when": [*RANK, {"id": f"{PART}.rank.shown", "filled": True}],
     "title": "提示の列は (あれば)",
     "help": "1 回の提示 (1 度に見せた品物の並び) の ID の列. 同じ利用者と"
             "提示の行を 1 回の提示とみなす. 無ければ空のまま (同じ利用者と"
             "時刻の行を 1 回の提示とみなす. 選んだ行の時刻が見せた時刻より"
             "後になる表では, 提示の列を選ぶ). 提示の欠けた行は選んだ行も除く. "
             "例: impression_id"},
    {"id": f"{PART}.rank.k", "section": 5, "kind": "number",
     "required": True, "when": RANK,
     "title": "利用者ごとに上位何件を並べますか (K)",
     "help": "並べる品物の数 (1 以上の整数). Recall@K と NDCG もこの K "
             "で数える. 例: 10"},
    {"id": f"{PART}.group.columns", "section": 5, "kind": "columns",
     "required": True, "when": GROUP, "title": "群を作るのに使う列は",
     "help": "1 行が 1 つの対象 (顧客など) の表の, 群を作るのに使う列. 数の"
             "列は標準化し, 他の列は値ごとに 0 と 1 の列にして使う. ID や"
             "名前のように行ごとに違う列は入れない. group, rows, share, "
             "stability, id, split の名前の列は使えない"},
    {"id": f"{PART}.group.id", "section": 5, "kind": "column",
     "required": False, "when": GROUP, "title": "行を見分ける列は (あれば)",
     "help": "群の割り当ての表に, 行の番号の代わりに書く列. 群を作るのには"
             "使わない. group, split の名前の列は使えない. 例: customer_id"},
    {"id": f"{PART}.group.k", "section": 5, "kind": "number",
     "required": True, "when": GROUP_ANY,
     "title": "いくつの群に分けますか",
     "help": "2 以上の整数. 人が見分けて扱える数にする. 分からなければ 4 "
             "などで始め, 学習の段階 (既製の埋め込みでは一式づくり) で "
             "config の group.k を変えて, 群のまとまりと安定性を比べる"},
    {"id": f"{PART}.group.stability", "section": 5, "kind": "choice",
     "required": True, "when": GROUP_ANY, "choices": GROUP_STABILITIES,
     "title": "群の安定性は, どう作り直して数えますか",
     "help": "作り直した群と元の群の一致 (調整ランド指数. 1 は同じ群, 0 は"
             "偶然と同じ) を数える. 作り直すと変わる群は, 人が解釈しても"
             "当てにならない"},
    {"id": f"{PART}.anomaly.label", "section": 5, "kind": "column",
     "required": True,
     "when": [*ANOMALY, {"id": f"{PART}.target.source", "equals": "few"}],
     "title": "異常の正解の列は",
     "help": "異常の行は 1 か真, 正常の行は 0 か偽の列. 正解の無い行は空の"
             "まま (正常と見なす). 異常の行にだけ印を付けた表でもよい. 例: "
             "is_anomaly"},
    {"id": f"{PART}.anomaly.id", "section": 5, "kind": "column",
     "required": False, "when": ANOMALY,
     "title": "行の id の列は (あれば)",
     "help": "人が確かめた結果の記録が 1 行を指すための列. 欠けも同じ値も"
             "無い列. 度合いには使わない. split, score, anomaly, rank, "
             "column, column_share の名前の列は使えない. 無ければ空のまま "
             "(行の位置で指す). 例: reading_id"},
    {"id": f"{PART}.anomaly.columns", "section": 5, "kind": "columns",
     "required": True, "when": ANOMALY,
     "title": "異常の度合いを出すのに使う列は",
     "help": "正常の行の値の組み合わせから外れたかを見る列 (センサの値, 取引"
             "の額など). 正解の列と行の id の列は入れない"},
    {"id": f"{PART}.anomaly.score", "section": 5, "kind": "choice",
     "required": True, "when": ANOMALY, "choices": ANOMALY_SCORES,
     "title": "異常の度合いを, どう出しますか",
     "help": "自己符号化器は正常の行だけで行を縮めて戻すことを学び, うまく"
             "戻せない行ほど異常とする (列の組み合わせの複雑な外れも見る). "
             "距離は学習用の正常の行の中心と散らばりから, 遠い行ほど異常と"
             "する (学習しないので速く, 列の少ない表で始めやすい)"},
    {"id": f"{PART}.uplift.treatment", "section": 5, "kind": "column",
     "required": True, "when": UPLIFT, "title": "処置の列は",
     "help": "行動した (クーポンを配った) 行は 1 か真, しなかった行は 0 か偽"
             "の列. 無作為に配った記録を前提にする (配る相手を選んだ記録"
             "では効果を取り違える). 例: coupon"},
    {"id": f"{PART}.uplift.outcome", "section": 5, "kind": "column",
     "required": True, "when": UPLIFT, "title": "結果の列は",
     "help": "行動の後の結果の列 (買ったか, 買った額). 処置の群と対照の群の"
             "この列の差が効果. 例: purchased"},
    {"id": f"{PART}.uplift.outcome_kind", "section": 5, "kind": "choice",
     "required": True, "when": UPLIFT, "choices": UPLIFT_OUTCOMES,
     "title": "結果は, 起きたかですか, 量ですか",
     "help": "起きたか (0 と 1) なら効果は確率の差, 量 (額) なら値の差"},
    {"id": f"{PART}.uplift.id", "section": 5, "kind": "column",
     "required": False, "when": UPLIFT,
     "title": "行の id の列は (あれば)",
     "help": "行ごとの効果の表と配る判断が 1 行 (顧客) を指すための列. 欠けも"
             "同じ値も無い列. 効果の見積もりには使わない. split, effect, "
             "treated, control, treatment, outcome の名前の列は使えない. 無け"
             "れば空のまま (行の位置で指す). 例: customer_id"},
    {"id": f"{PART}.uplift.columns", "section": 5, "kind": "columns",
     "required": True, "when": UPLIFT,
     "title": "効果の見積もりに使う列は",
     "help": "配る前に分かっている顧客の特徴の列 (年齢, 来店の回数など). 処置"
             "の列, 結果の列, 行の id の列と, 配った後に決まる列 (クーポンを"
             "使ったかなど) は入れない"},
    {"id": f"{PART}.uplift.learner", "section": 5, "kind": "choice",
     "required": False, "when": UPLIFT, "choices": UPLIFT_LEARNERS,
     "title": "効果を, どう学びますか (答えなければ 2 つの予測)",
     "help": "2 つの予測は処置の群と対照の群の結果を別々に学び, 差を効果に"
             "する. 1 つの予測は処置の列 (1 か 0) を入力の終わりに足して"
             "学び, 処置を 1 にした予測から 0 にした予測を引いて効果にする "
             "(片方の群の行が少なく, 2 つの予測が揺れる記録で合うことが"
             "ある)"},
    {"id": f"{PART}.price.price", "section": 5, "kind": "column",
     "required": True, "when": PRICE, "title": "値段の列は",
     "help": "売ったときの値段の列 (0 より大きい数). 試しに値段を変えた記録 "
             "(無作為に近い) を前提にする (売れ行きに合わせて値段を決めた"
             "記録では, 値段の効き方を取り違える). 売れ行きの予測の入力の"
             "最後の列になる. 例: price"},
    {"id": f"{PART}.price.units", "section": 5, "kind": "column",
     "required": True, "when": PRICE, "title": "売れた数の列は",
     "help": "その値段で売れた数の列 (0 以上の数). 予測する値. 例: units"},
    {"id": f"{PART}.price.cost", "section": 5, "kind": "column",
     "required": False, "when": PRICE, "title": "原価の列は (あれば)",
     "help": "商品 1 つを仕入れるか作るのにかかる値段の列 (0 以上の数). 利益"
             "は (値段 − 原価) × 売れた数. 無ければ空のまま (原価を 0 とし, "
             "売上で選ぶ). 予測の入力には入れない. 例: unit_cost"},
    {"id": f"{PART}.price.product", "section": 5, "kind": "column",
     "required": False, "when": PRICE, "title": "商品の id の列は (あれば)",
     "help": "値段を選ぶ単位の商品を指す列 (同じ商品の行は何行あってもよい)"
             ". 今の値段は商品ごとの記録の値段の中央値にする. 無ければ空の"
             "まま (行ごとに選び, 今の値段はその行の値段). 予測の入力には"
             "入れない. 例: product_id"},
    {"id": f"{PART}.price.id", "section": 5, "kind": "column",
     "required": False, "when": PRICE, "title": "行の id の列は (あれば)",
     "help": "記録の 1 行を指す列. 欠けも同じ値も無い列. 予測には使わない. "
             "無ければ空のまま (行の位置で指す). 例: record_id"},
    {"id": f"{PART}.price.columns", "section": 5, "kind": "columns",
     "required": True, "when": PRICE,
     "title": "売れ行きの予測に使う列は (値段の列の他)",
     "help": "値段を決める前に分かっている商品の特徴の列 (種類, 曜日など). "
             "値段の列, 売れた数の列, 原価の列, 商品の id の列, 行の id の列"
             "と, 売れた後に決まる列 (残った数など) は入れない"},
    {"id": f"{PART}.demand.product", "section": 5, "kind": "column",
     "required": True, "when": DEMAND, "title": "商品の id の列は",
     "help": "発注の数を決める単位の商品を指す列. 1 行が (商品, 日) の記録の"
             "表で, 商品ごとに日ごとの行が並ぶ. 予測の入力には入れない (商品"
             "の特徴は別の列に書く). split, stock, quantile, order, "
             "previous, demand の名前の列は使えない. 例: product_id"},
    {"id": f"{PART}.demand.date", "section": 5, "kind": "column",
     "required": True, "when": DEMAND, "title": "日付の列は",
     "help": "記録の日 (日ごと. 時刻は書かない). 同じ商品の同じ日の行は 1 行"
             "にまとめる. 曜日などはデータの分析の段階で列にする. 例: date"},
    {"id": f"{PART}.demand.units", "section": 5, "kind": "column",
     "required": True, "when": DEMAND, "title": "日ごとの売れた数の列は",
     "help": "その日に売れた数の列 (0 以上の数). 先の期間の合計が予測する値"
             "で, 過去の売れた数はずらした列にして予測に使う. 売れなかった日"
             "は 0 の行を書く (記録の無い日は合計を作らない). 例: units"},
    {"id": f"{PART}.demand.stock", "section": 5, "kind": "column",
     "required": False, "when": DEMAND, "title": "在庫の数の列は (あれば)",
     "help": "その日の始めの在庫の数 (0 以上の数). 発注の数は, 先の期間の"
             "需要の分位点から在庫の数を引いた数にする. 予測の入力には入れ"
             "ない. 無ければ空のまま (在庫を 0 として発注する). 例: stock"},
    {"id": f"{PART}.demand.id", "section": 5, "kind": "column",
     "required": False, "when": DEMAND, "title": "行の id の列は (あれば)",
     "help": "記録の 1 行を指す列. 欠けも同じ値も無い列. 予測には使わない. "
             "無ければ空のまま (行の位置で指す). 例: record_id"},
    {"id": f"{PART}.demand.columns", "section": 5, "kind": "columns",
     "required": False, "when": DEMAND,
     "title": "需要の予測に使う商品の特徴の列は (あれば)",
     "help": "発注を決める前に分かっている商品の特徴の列 (種類, 大きさなど)"
             ". 過去の売れた数と曜日は, データの分析の段階で列にするので入れ"
             "ない. 商品の id, 日付, 売れた数, 在庫の数, 行の id の列と, 売れ"
             "た後に決まる列 (捨てた数など) は入れない. 無ければ空のまま"},
    {"id": f"{PART}.demand.horizon", "section": 5, "kind": "number",
     "required": False, "when": DEMAND,
     "title": "何日先までの需要の合計を予測しますか (先の期間の日数)",
     "help": "発注してから次の発注で届くまでの日数 (届くまでの日数と発注の"
             "周期を合わせた日数). 1 以上の整数. 無ければ空のまま (7 日)"},
    {"id": f"{PART}.demand.levels", "section": 5, "kind": "numbers",
     "required": False, "when": DEMAND,
     "title": "学ぶ分位点の水準は",
     "help": "先の期間の需要の分布のどの点を学ぶか (0 より大きく 1 より小さい"
             "数の並び. 0.9 は 9 割の日でこれより少ない数). 発注に使う水準 "
             "τ (品切れの損 / (品切れの損 + 売れ残りの損)) と 0.5 (予測の"
             "誤差を数える点) は答えに無くても足し, 同じ水準は 1 つにまとめる"
             ". 無ければ空のまま (0.1, 0.25, 0.5, 0.75, 0.9)"},
    {"id": f"{PART}.task", "section": 5, "kind": "choice", "required": True,
     "when": COLUMN_TARGET, "choices": TASKS, "title": "どの種類の問題ですか",
     "help": "目的の値がクラスなら分類, 値なら回帰. 時刻の順に過去から先を"
             "予測するなら時系列の予測. 4 で決めることが量なら回帰か時系列の"
             "予測, 対象を選ぶか振り分けるなら分類"},
    {"id": f"{PART}.target.column", "section": 5, "kind": "column",
     "required": True,
     "when": COLUMN_TARGET, "title": "目的の列は",
     "help": "予測する値の列"},
    {"id": f"{PART}.group_column", "section": 5, "kind": "column",
     "required": False, "when": COLUMN_TARGET,
     "title": "同じ相手の行をまとめる列は (分割をまたがせない)",
     "help": "1 人の会員, 1 台の機械に行が何行もあるとき, 学習と確かめに同じ"
             "相手が入ると, 確かめが甘くなる (覚えた相手を当てるだけになる)."
             " 書くと, データの分析の段階で相手ごとに分ける. 例: member_id. "
             "無ければ空のまま"},
    {"id": f"{PART}.positive", "section": 5, "kind": "text",
     "required": False,
     "when": [*COLUMN_TARGET,
              {"id": f"{PART}.task", "equals": "classification"}],
     "title": "当たりのクラスは (2 クラスの分類で行動を決める場合)",
     "help": "行動の対象になるクラス. 例: 解約, 不良"},
    {"id": f"{PART}.time_column", "section": 5, "kind": "column",
     "required": True,
     "when": [*COLUMN_TARGET, {"id": f"{PART}.task", "equals": "forecasting"}],
     "title": "時刻の列は", "help": "時刻の順に並べ, 分けるのに使う列"},
    {"id": f"{PART}.split_time_column", "section": 5, "kind": "column",
     "required": False,
     "when": [*COLUMN_TARGET,
              {"id": f"{PART}.task",
               "in": ["classification", "regression"]}],
     "title": "時刻の順で分けて確かめるなら, その時刻の列は",
     "help": "使うときは過去から先を当てるなら (今月の申し込みで来月の解約を"
             "当てるなど), 先の期間を検証用とテスト用に残すと, 使うときの当たり"
             "具合に近い確かめになる (無作為に分けると, 先の様子を学習に使って"
             "甘くなる). 時刻の列を学習に使わない列にしてもよい (分けるのにだけ"
             "使う). 無ければ空のまま"},
    {"id": f"{PART}.series_by", "section": 5, "kind": "columns",
     "required": False,
     "when": [*COLUMN_TARGET, {"id": f"{PART}.task", "equals": "forecasting"}],
     "title": "系列を分ける列は (複数の系列が並ぶ場合)",
     "help": "例: 店, 品. 過去の値は系列ごとにずらす"},
    {"id": f"{PART}.available_lag", "section": 5, "kind": "number",
     "required": True,
     "when": [*COLUMN_TARGET, {"id": f"{PART}.task", "equals": "forecasting"}],
     "title": "決める時点 (4 の「いつ決めますか」) で, 目的の値は何期前の"
              "分まで分かっていますか",
     "help": "過去の値の列に使ってよい, 最も新しい値. 例: 前の日の 17 時に"
             "翌日の量を決めるなら, 前の日の売れた数はまだ確定しないので 2 "
             "(2 日前まで). 前の日が締まってから決めるなら 1. 分からない値を"
             "学習に使うと, 学習のときだけ当たり, 使うときに外れる"},
    {"id": f"{PART}.horizon", "section": 5, "kind": "number",
     "required": True,
     "when": [*COLUMN_TARGET, {"id": f"{PART}.task", "equals": "forecasting"}],
     "title": "何期先まで予測しますか",
     "help": "時刻の列の 1 つ分を 1 期として. 例: 翌日だけなら 1, 1 週間分を"
             "まとめて決めるなら 7. 先ほど外れやすく, 予測の時点で分かる列も"
             "変わる"},
    {"id": f"{PART}.baseline", "section": 5, "kind": "text", "required": True,
     "when": USES_ML, "title": "機械学習を使わない比べる相手は",
     "help": "機械学習が要るかは, 素朴なやり方より良いかで決まる. 多くは"
             "1 の「今のやり方」と同じ. 例: 前の週の同じ曜日と同じ数"},
    {"id": f"{PART}.consumer", "section": 5, "kind": "choice",
     "required": False, "when": USES_ML, "choices": CONSUMERS,
     "title": "出力を誰が (何が) 使いますか",
     "help": "使い道で, 確かめ方と要る精度が変わる. 人が見るなら説明, 自動"
             "で決めるなら損得 (7 の節), 後段に渡すなら後段が求める形と精度"},
    {"id": f"{PART}.downstream", "section": 5, "kind": "text",
     "required": False, "when": [USES_ML, DOWNSTREAM],
     "title": "後段は何で, 出力に何を求めますか",
     "help": "形, 単位, 要る精度. 評価は後段の結果で見る (予測の誤差が小さく"
             "ても後段の判定が変わることがある). 例: 点群から寸法を測る決"
             "まり. 1 mm の差が分かる精度が要る"},

    # 6. 使える情報
    {"id": "information.known", "section": 6, "kind": "columns",
     "required": True, "when": [*COLUMN_TARGET, {"none": CONTENT_DATA}],
     "title": "予測の時点で分かっている列は",
     "help": "学習に使ってよい列. 予測の時点で分からない列 (当日の客数など)"
             "で学習すると, 学習のときだけ当たり, 使うときに当たらない. 目的"
             "の列そのものは入れない (過去の値はデータの分析で作る)"},
    {"id": "information.unknown", "section": 6, "kind": "columns",
     "required": False, "when": COLUMN_TARGET,
     "title": "予測の時点では分からない列は",
     "help": "学習に使わない列 (当日の客数など. 予測のときに分からない)"},
    {"id": "data.missing", "section": 6, "kind": "list", "required": False,
     "when": USES_ML, "title": "要るのに手元に無いデータは",
     "help": "課題を解くのに要るが, まだ集めていないデータ. 例: 天気予報の"
             "記録. 要るデータが無いなら, まずデータを集めることが答えになる"
             "こともある"},
    {"id": "information.forbidden", "section": 6, "kind": "columns",
     "required": False, "when": COLUMN_TARGET,
     "title": "分かっていても使ってはいけない列は",
     "help": "ID, 個人を特定できる情報, 使うと公平でない情報など. 学習に"
             "使わない"},

    # 7. 損得と制約
    {"id": "action.decision", "section": 7, "kind": "decision",
     "required": True, "when": DECIDED, "choices": DECISION_TYPES,
     "title": "行動の損得と制約は",
     "help": "量を決める: 1 単位足りないときの損と余ったときの損. 対象を選ぶ: "
             "当たり 1 件の価値, 行動 1 件の費用, 行動で防げる割合 (仮定), "
             "件数か費用の上限. 振り分ける: 見逃しの損と空振りの損, 回せる件数"
             ". 人が予測を見て決めるなら「意思決定しない」. 効果の上位から"
             "配る: 配る数の上限 (守る指標. 1 回に配れる数. 効果の高い順に"
             "上限まで配る). 利益が最も大きい値段を選ぶ: 値段の候補 (今の値段"
             "に対する変える割合の並び. 0 を含む) と, 値段を変える幅の上限 "
             "(守る指標. 上限を超える候補は選ばない). 損の比で発注の数を決め"
             "る: 1 つ足りないときの損と 1 つ余ったときの損 (同じお金の単位) "
             "と, 発注の数の上限 (守る指標. 置き場所の上限). 近い順に上位 K "
             "件を返す: 問いごとに返す件数 K と, 1 件の問いの待ち時間の上限 "
             "(守る指標. ミリ秒. 任意)"},
    {"id": f"{PART}.anomaly.capacity", "section": 7, "kind": "number",
     "required": True, "when": ANOMALY,
     "title": "人が 1 回に確かめられる件数の上限は (守る指標)",
     "help": "推論の 1 回 (1 日分の記録など) で, 度合いの高い順に人の確かめに"
             "回す件数の上限 (1 以上の整数). これを超えて回すと確かめきれ"
             "ない. 評価の上位 K 件の適合率もこの K で数える. 例: 20"},
    {"id": "action.current", "section": 7, "kind": "baseline",
     "required": False,
     "when": [*COLUMN_TARGET, {"id": "action.decision.type",
                        "in": ["quantity", "select", "triage"]}],
     "choices": BASELINE_TYPES, "title": "今のやり方をデータで書くと",
     "help": "1 の「今のやり方」を, データの上の決まりにする. 量なら k 期前と同じ"
             " (前の週の同じ曜日なら 7) か, 直前の k 期の平均か, 今のやり方の"
             "値の列. 選ぶと振り分けなら, 行動したかの列. 書くと, データから"
             "今のやり方の損 (指標の今の値の候補) が出る"},
    {"id": "issue.guard.counted", "section": 7, "kind": "choice",
     "required": False, "when": COUNTABLE_GUARD, "choices": GUARD_COUNTED,
     "title": "守る指標は, たどり直しで数えられる量のどれに当たりますか",
     "help": "たどり直しは, 過去のデータで, 予測から決めた行動をなぞり"
             "直すこと. 当たる量を選ぶと, 推論の段階のたどり直しが, その"
             "行動でその量を数え, 守る線と一緒に出す. 当たる量が無ければ"
             "「指さない」(課題の場で数える). 割合の量は 0 から 1 で数える"
             "ので, 守る線 (2) も 0 から 1 で書く (5 % なら 0.05)"},
    {"id": "action.basis", "section": 7, "kind": "text", "required": False,
     "when": COLUMN_TARGET, "title": "損得の値の根拠は",
     "help": "例: 欠品は粗利 200 円, 廃棄は原価 150 円. 根拠の無い値では, "
             "意思決定はもっともらしいだけになる"},
    {"id": "ml.constraints", "section": 7, "kind": "list",
     "required": False, "when": USES_ML,
     "title": "機械学習の仕組みが守る制約は",
     "help": "1 行に 1 つ. 作り方 (既製のモデル, 外のサービス) を選ぶ前に"
             "決める. 例: データを社外に出せない, 商用で使えるライセンス"
             "だけ, 1 件 1 秒以内に返す, 手元の計算機で動く"},
    {"id": "ml.gpu", "section": 7, "kind": "boolean", "required": False,
     "when": USES_ML, "title": "GPU (CUDA) が要りますか",
     "help": "画像, 文, 大きなモデルの学習は, 多くの場合要る. NVIDIA の"
             "ドライバと CUDA 付きの PyTorch を先に入れる. 分からなければ"
             "空のまま"},
    {"id": f"{PART}.policy.action_limits", "section": 7, "kind": "text",
     "required": True, "when": AUTONOMOUS,
     "title": "行動の上限は (守り)",
     "help": "方策が出してよい行動の範囲. 書き出す方策は, 行動をこの範囲に"
             "収める. 例: 各関節の速さは毎秒 0.5 rad まで, 選んでよいのは"
             "左右と止まるだけ"},
    {"id": f"{PART}.policy.stop", "section": 7, "kind": "text",
     "required": True, "when": AUTONOMOUS,
     "title": "何がどの範囲を外れたら, その場で方策の出力を止めますか (止める"
              "条件)",
     "help": "観測 (位置, 角度, 温度など) ごとの範囲. 外れたら下の「外れた"
             "ときの動き」にする. 例: 腕の先が台の上 0 から 0.3 m を出たら"},
    {"id": f"{PART}.policy.fallback", "section": 7, "kind": "choice",
     "required": True, "when": AUTONOMOUS, "choices": FALLBACKS,
     "title": "外れたときは, どうしますか",
     "help": "止める条件に当たったときと, 観測やモデルの出力が読めないとき"
             "の動き"},
    {"id": f"{PART}.policy.fallback_action", "section": 7, "kind": "text",
     "required": False,
     "when": [*AUTONOMOUS, {"id": f"{PART}.policy.fallback",
                            "equals": "action"}],
     "title": "その決めた行動は",
     "help": "例: 全ての関節を止める. 番号で選ぶ行動なら, その番号 (要る)."
             " 量で決める行動なら, 空のときは全ての量を 0 にする"},
    {"id": f"{PART}.policy.safety_key", "section": 7, "kind": "text",
     "required": False, "when": AUTONOMOUS,
     "title": "安全の違反を知らせる信号の名前は",
     "help": "シミュレータや記録が, 違反した歩に 0 より大きい値を入れる項目"
             "の名前. 学習と評価で違反の数を数える. 無ければ空のまま (学習の"
             "段階の既定は cost)"},

    # 8. 学習の指標
    {"id": f"{PART}.evaluation.metric", "section": 8, "kind": "choice",
     "required": True, "when": TRAINED, "choices": METRICS,
     "title": "最も良いモデルを何で選びますか",
     "help": "検証用の分割で見て, 最も良いモデルを選ぶ指標 (学習の損失の"
             "種類はモデルの作成と学習の段階で決める). 候補に無ければその他"},
    {"id": f"{PART}.evaluation.name", "section": 8, "kind": "text",
     "required": True,
     "when": [*TRAINED, {"id": f"{PART}.evaluation.metric",
                         "equals": "other"}],
     "title": "その指標の名前は",
     "help": "学習の記録の名前 (空白と / は使えない). 検証用の分割の値 "
             "valid/<名前> を監視する指標として渡す. 例: f1, map50"},
    {"id": f"{PART}.evaluation.better", "section": 8, "kind": "choice",
     "required": True,
     "when": [*TRAINED, {"id": f"{PART}.evaluation.metric",
                         "equals": "other"}],
     "choices": [{"value": "lower", "label": "小さいほど良い"},
                 {"value": "higher", "label": "大きいほど良い"}],
     "title": "その指標は, 小さいほど良いですか, 大きいほど良いですか",
     "help": "最良のモデルを選ぶ向き"},
    {"id": f"{PART}.evaluation.link", "section": 8, "kind": "text",
     "required": False, "when": TRAINED,
     "title": "その指標が良くなると, 課題の指標はどう良くなりますか",
     "help": "例: 誤差が小さいほど作る量の外れが減り, 廃棄と欠品の損が減る"},
    {"id": f"{PART}.uplift.k", "section": 8, "kind": "number",
     "required": True, "when": UPLIFT,
     "title": "評価で見る上位の割合 K は (0 より大きく 1 以下)",
     "help": "効果の見積もりの高い順の上位 K 割の行で, 処置の群と対照の群の"
             "結果の平均の差 (上位 K 割の効果の差) を数える. 配る数の上限が"
             "顧客の何割かに当たるかで決める. 例: 3 割に配れるなら 0.3"},
    {"id": f"{PART}.evaluation.current", "section": 8, "kind": "number",
     "required": False, "when": REPORTED_BASELINE,
     "title": "今のやり方 (人の目視や今の仕組み) で, この指標は幾つですか "
              "(分かれば)", "title_metric": True,
     "help": "画像や文, 領域, 点群の問題では, 今のやり方の結果を 1 件ずつ並べ"
             "られないので, ここに答えた値を学習したモデルのテスト用の値と並べ"
             "ます. 数えた件が違うので目安です. 例: 人が目で見て分けると 100 "
             "件中 85 件当たる → 0.85. 割合 (正解率, IoU) は 0 から 1, 誤差は"
             "目的の値 (点群は座標) の単位. 2 の「今の値」は課題の指標 (円など)"
             "で, これとは別. 分からなければ空のまま"},

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
     "help": "課題の指標で. 例: 廃棄と欠品の損が今のやり方より 2 割少ない"
             " (仕事). 再構成の誤差が今の手法より小さい (研究). 続けて"
             "遊ぶ時間が 1 割延びる (娯楽)"},

    # 10. 運用
    {"id": "operation.review", "section": 10, "kind": "text",
     "required": True, "when": USES_ML,
     "title": "どれくらいの頻度で見直しますか", "help": "例: 月に 1 回"},
    {"id": "operation.approver", "section": 10, "kind": "text",
     "required": False, "when": USES_ML,
     "title": "変更 (予測の外れ方の見積もりを新しい実績で直すなど) は誰が"
              "承認しますか",
     "help": "例: 本部"},
    {"id": f"{PART}.policy.approver", "section": 10, "kind": "text",
     "required": True, "when": AUTONOMOUS,
     "title": "方策を使ってよいかと, 守りの範囲は誰が決めますか",
     "help": "人が決めるのは 1 回 1 回の行動ではなく, 方策を使ってよいかと"
             "守り (行動の上限, 止める条件, 外れたときの動き) の範囲. 例: "
             "設備の責任者"},
    {"id": "operation.stop", "section": 10, "kind": "text",
     "required": False, "when": USES_ML, "title": "どうなったら止めますか",
     "help": "例: 損が今のやり方より悪い月が 2 回続いたら"},
    {"id": "operation.remeasure", "section": 10, "kind": "text",
     "required": False, "when": USES_ML,
     "title": "予測の外れ方をいつ確かめ直しますか",
     "help": "外れ方は時間とともに変わる. 例: 月に 1 回, 前の月の実績で"},
]

# 1 行の名前で答える文の問い (人や部署, 数字と単位, 処理と指標と環境と
# 信号の名前, クラス, 比べる相手, 頻度, 決めた行動, 既製のモデルと重みの
# 名前と hash). 他の文の問いは複数行
ONE_LINE = {
    "issue.owner", "issue.kpi.name", "issue.kpi.unit", "issue.guard.name",
    "issue.guard.unit", "action.who", "action.when",
    f"{PART}.target.process.name", f"{PART}.env_id", f"{PART}.positive",
    f"{PART}.baseline", f"{PART}.policy.fallback_action",
    f"{PART}.policy.safety_key", f"{PART}.policy.approver",
    f"{PART}.evaluation.name", "operation.review", "operation.approver",
    f"{PART}.ready.model", f"{PART}.ready.weights", f"{PART}.ready.sha256",
}

# 答えの欄の形 (文の問いだけ multiline が真か偽. 他の種類の問いは偽)
for question in QUESTIONS:
    question["multiline"] = (question["kind"] == "text"
                             and question["id"] not in ONE_LINE)

# 問いの識別子から問い
BY_ID = {question["id"]: question for question in QUESTIONS}
