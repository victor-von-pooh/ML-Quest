# 骨組みの学習のスクリプトがモデルの定義に求める入力と出力の名前. 骨組みの
# 書き出し (学習のスクリプトの定数) と骨組みの約束 (model_contract) が同じ
# 値を使う (ML-ModelControl の同梱の例と同じ名前)

# 行の特徴の入力 (異常の検知, まとめる, 効果の見積もり, 値段を変える判断,
# 時間をまたぐ量の判断)
FEATURE_INPUT = "x"

# 異常の検知の自己符号化器の再構成の出力
RECON_OUTPUT = "recon"

# 並べるの利用者と品物の番号のワンホットの入力と, 全ての組の点数の出力
USER_INPUT = "user"
ITEM_INPUT = "item"
SCORE_OUTPUT = "score"

# まとめるの埋め込みと全ての組の類似度の出力
EMBEDDING_OUTPUT = "embedding"
SIMILARITY_OUTPUT = "similarity"

# 効果の見積もりの, 処置したときとしないときの結果の予測の出力
TREATED_OUTPUT = "treated"
CONTROL_OUTPUT = "control"

# 効果の見積もりの S-learner の, 処置の列を入力に足した結果の予測の出力
OUTCOME_OUTPUT = "outcome"

# 値段を変える判断の売れた数の予測の出力
UNITS_OUTPUT = "units"

# 時間をまたぐ量の判断の分位点の出力
QUANTILES_OUTPUT = "quantiles"
