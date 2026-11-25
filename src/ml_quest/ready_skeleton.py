import json
import re
from string import Template
from typing import Callable, Optional

from ml_quest.handoff import describe
from ml_quest.problem import active_value
from ml_quest.questions import OUTPUTS, PART, TARGET_KINDS

# 一式づくりの設定とスクリプトの場所 (プロジェクトのディレクトリから)
READY_CONFIG = "configs/default/bundle.yaml"
READY_SCRIPT = "bundle.py"

# 一式を作るディレクトリの既定 (プロジェクトのディレクトリから)
READY_BUNDLE = "outputs/inference"

# 一式づくりの設定を読む関数 (既製の埋め込みの群分けの骨組みも同じ文を
# 書き出す)
READY_SETTINGS = '''
def read_config(path: Path) -> dict:
    """
    一式づくりの設定を読む関数

    Parameters
    ----------
    path: Path
        設定のファイル (configs/default/bundle.yaml)

    Returns
    ----------
    config: dict
        設定 (読めなければ理由を示して止まる)
    """
    # YAML の辞書だけ
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        print(f"{path} を読めません ({error})")
        sys.exit(1)
    if not isinstance(config, dict):
        print(f"{path} は辞書の形にしてください")
        sys.exit(1)

    return config


def ready_values(config: dict) -> dict:
    """
    config の ready から, 一式づくりに渡す値を取り出す関数

    Parameters
    ----------
    config: dict
        一式づくりの設定

    Returns
    ----------
    ready: dict
        library, model, weights (空でない文字) と sha256 (文字か None).
        足りなければ直す所を示して止まる
    """
    # 名前は空でない文字, hash は空なら比べない
    raw = config.get("ready")
    raw = raw if isinstance(raw, dict) else {}
    missing = [key for key in READY_KEYS
               if not isinstance(raw.get(key), str) or not raw[key].strip()]
    if missing:
        print(f"config の ready の {', '.join(missing)} を書いてください "
              "(問題の設定の既製のモデルの答え)")
        sys.exit(1)
    ready = {key: raw[key].strip() for key in READY_KEYS}
    sha256 = raw.get("sha256")
    if sha256 is not None and not isinstance(sha256, str):
        print("config の ready.sha256 は 16 進の文字か null にしてください")
        sys.exit(1)
    ready["sha256"] = (sha256.strip().lower() or None) if sha256 else None

    return ready


def bundle_settings(config: dict) -> tuple[Path, Optional[str], bool]:
    """
    config の bundle から, 一式を作る場所, 名前, 置き換えるかを読む関数

    Parameters
    ----------
    config: dict
        一式づくりの設定

    Returns
    ----------
    settings: tuple[Path, Optional[str], bool]
        一式のディレクトリ (プロジェクトのディレクトリから), 名前 (None
        はディレクトリの名前), 前に作った一式を置き換えるか
    """
    # 場所の既定は outputs/inference
    raw = config.get("bundle")
    raw = raw if isinstance(raw, dict) else {}
    place = raw.get("path") or "outputs/inference"
    name = raw.get("name") if isinstance(raw.get("name"), str) else None
    settings = (PROJECT_DIR / str(place), name or None,
                raw.get("overwrite") is True)

    return settings
'''

# 一式を作るスクリプト. 学習はせず, config の ready の値を ML-Inference の
# make_ready_bundle に渡す. 答えの文字は JSON の文字列で埋め込み
READY_SCRIPT_TEXT = '''"""
問題の設定 (problem.yaml) から ML-Quest が書き出した, 既製のモデルの推論の
一式を作るスクリプト

学習はしない. config の ready の値 (出どころ, モデルの名前, 重みの名前と
hash) を ML-Inference の make_ready_bundle に渡し, 推論の一式を作る
"""
import importlib
from pathlib import Path
import sys
from typing import Any, Callable, Optional

import yaml

# このプロジェクトのディレクトリと, 一式づくりの設定の場所
PROJECT_DIR = Path(__file__).resolve().parent
CONFIG = PROJECT_DIR / $config

# 問題の名前と, 何を出すか
PROBLEM_NAME = $name
PREDICT = $predict

# 一式づくりに渡す, 要る名前 (config の ready. sha256 は任意)
READY_KEYS = ("library", "model", "weights")

# ML-Inference と, torchvision の重みを読む ML-ModelControl の入れ方
# (既製のモデルの一式は ML-Inference の推論の一式の版 6 から)
INSTALL = (
    "pip install \\"ml-inference @ git+https://github.com/victor-von-pooh/"
    "ML-Inference.git\\" \\"ml-modelcontrol[torchvision] @ git+https://"
    "github.com/victor-von-pooh/ML-ModelControl.git\\""
)

''' + READY_SETTINGS + '''

def load_maker() -> Callable:
    """
    ML-Inference の make_ready_bundle を読み込む関数

    Parameters
    ----------
    None

    Returns
    ----------
    maker: Callable
        make_ready_bundle (無ければ入れ方を示して止まる)
    """
    # 入っていないか, 既製のモデルの一式を作れない古い ML-Inference
    try:
        module = importlib.import_module("ml_inference")
    except ImportError as error:
        print(f"ML-Inference を読み込めません ({error}). 入れてから動かして"
              f"ください:\\n{INSTALL}")
        sys.exit(1)
    maker = getattr(module, "make_ready_bundle", None)
    if maker is None:
        print("この ML-Inference は既製のモデルの一式 (make_ready_bundle) を"
              f"作れません. 新しくしてください:\\n{INSTALL}")
        sys.exit(1)

    return maker


def report(info: dict, place: Path, pinned: Optional[str]) -> None:
    """
    作った一式の, 種類の数, 重みの hash, 注意を表示する関数

    Parameters
    ----------
    info: dict
        make_ready_bundle が返した inference.yaml の中身
    place: Path
        一式のディレクトリ
    pinned: Optional[str]
        config に決めておいた hash (無ければ None)

    Returns
    ----------
    None
    """
    # 種類の名前の数と重みの hash (決めていなければ書き戻す案内)
    output = info.get("output") or {}
    classes = output.get("classes") or []
    records = (info.get("model") or {}).get("pretrained") or [{}]
    sha256 = records[0].get("sha256")
    ready = info.get("ready") or {}
    print(f"問題: {PROBLEM_NAME}. 一式を作りました: {place}")
    if PREDICT:
        print(f"出すもの: {PREDICT}")
    print(f"既製のモデル: {ready.get('library')} の {ready.get('model')} "
          f"(重み {ready.get('weights')}). 種類の数: {len(classes)}")
    if sha256:
        print(f"重みの hash (SHA-256): {sha256}")
    if sha256 and not pinned:
        print("この hash を config の ready.sha256 (と問題の設定の重みの "
              "hash) に書くと, 次から違う重みでは作りません")
    for note in info.get("notes") or []:
        print(f"注意: {note}")


def main() -> None:
    """
    config の値で, 学習しない既製のモデルの推論の一式を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 設定と, 一式づくりに渡す値
    config = read_config(CONFIG)
    ready = ready_values(config)
    place, name, overwrite = bundle_settings(config)
    conclusion: Any = config.get("conclusion")
    if conclusion is not None and not isinstance(conclusion, dict):
        print("config の conclusion は辞書か null にしてください")
        sys.exit(1)

    # ML-Inference の一式づくり (重みは一式に写さず, 使うときに出どころ
    # から読む. 初回は取りに行く)
    maker = load_maker()
    try:
        info = maker(
            place, ready["model"], ready["weights"],
            library=ready["library"], sha256=ready["sha256"], name=name,
            conclusion=conclusion, overwrite=overwrite
        )
    except (FileExistsError, ValueError) as error:
        print(f"一式を作れません: {error}")
        sys.exit(1)

    # 作った一式の要約
    report(info, place, ready["sha256"])


if __name__ == "__main__":
    main()
'''

# 一式づくりの設定
READY_CONFIG_TEXT = Template('''# 既製のモデルの推論の一式づくりの設定 (ML-Quest が問題の設定から書き出した
# 骨組み). bundle.py が読み, ML-Inference の make_ready_bundle に渡す.
# 学習はしない

# 一式づくりに渡す値 (問題の設定の既製のモデルの答え)
ready:
  # 出どころ (ML-Inference の READY_LIBRARIES の名前)
  library: $library
  # 出どころでのモデルの名前
  model: $model
  # 重みの名前 (版. 名前ごとに重みが決まる). DEFAULT は作るときの出どころの
  # 既定の重みを使い, 一式にはその版の名前を残す (既定が変わると別の重みに
  # なるので, 決まった重みを使うなら版まで書く)
  weights: $weights
  # 決めておいた重みの hash (SHA-256). 読んだ重みと違えば作らない. null は
  # 比べずに作り, 求めた hash を表示する
  sha256: $sha256

bundle:
  # 一式を作るディレクトリ (プロジェクトのディレクトリから)
  path: $path
  # 一式の名前 (null はディレクトリの名前)
  name: $name
  # 前に作った一式を置き換えるか
  overwrite: false

# 結論の決め方の既定値 (ML-Inference の結論. 例: {min_confidence: 0.6}.
# null は既定のまま)
conclusion: null
''')


def ready_message(problem: dict) -> Optional[str]:
    """
    既製のモデルの骨組みを書き出せない理由を返す関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2. 作り方が既製のモデルをそのまま)

    Returns
    ----------
    message: Optional[str]
        誤りの文 (予測するでない, 正解の種類が表の列でない, 出力の形が
        クラスでない. 書き出せるなら None)
    """
    # 今の一式づくり (ML-Inference) は画像の分類だけ
    role = active_value(problem, f"{PART}.role")
    kind = active_value(problem, f"{PART}.target.kind")
    output = active_value(problem, f"{PART}.output")
    message = None
    if role != "predict":
        message = ("既製のモデルをそのまま使う骨組みは, 受け持ちが予測するか"
                   "まとめるの場合だけです (作り方を学習するものにするか, "
                   "受け持ちを予測するかまとめるにしてください)")
    elif kind not in (None, "column"):
        message = ("既製のモデルをそのまま使う骨組みは, 正解の種類が"
                   f"{label_of(TARGET_KINDS, 'column')}の場合だけです (今は"
                   f"{label_of(TARGET_KINDS, kind)}. 今は画像の分類だけ)")
    elif output not in (None, "class"):
        message = ("既製のモデルをそのまま使う骨組みは, 出力の形が"
                   f"{label_of(OUTPUTS, 'class')}の場合だけです (今は"
                   f"{label_of(OUTPUTS, output)}. 今は画像の分類だけ)")

    return message


def label_of(choices: list[dict], value: Optional[str]) -> str:
    """
    選択肢の値を「」付きの短い呼び方にする関数

    Parameters
    ----------
    choices: list[dict]
        選択肢
    value: Optional[str]
        値

    Returns
    ----------
    text: str
        「呼び方」(選択肢に無ければ値のまま)
    """
    # 説明の括弧の前まで
    label = next((item["label"] for item in choices
                  if item["value"] == value), str(value))
    text = f"「{label.split(' (')[0]}」"

    return text


def ready_name(stages: dict, name: Optional[str]) -> str:
    """
    一式の名前を決める関数

    Parameters
    ----------
    stages: dict
        受け渡し (inference.ready に一式づくりに渡す値)
    name: Optional[str]
        指定した名前

    Returns
    ----------
    text: str
        文字と数字と _ と - だけの名前 (省略するとモデルと重みの名前から)
    """
    # 区切りや空白などのディレクトリに使いにくい文字の _ への置き換え
    ready = stages["inference"]["ready"]
    raw = name or f"{ready['model']}_{ready['weights']}"
    text = re.sub(r"[^\w-]+", "_", raw).strip("_") or "ready"

    return text


def ready_config(stages: dict, name: Optional[str]) -> str:
    """
    既製のモデルの一式づくりの設定を作る関数

    Parameters
    ----------
    stages: dict
        受け渡し (inference.ready に一式づくりに渡す値)
    name: Optional[str]
        一式の名前 (省略するとモデルと重みの名前から)

    Returns
    ----------
    text: str
        configs/default/bundle.yaml の中身
    """
    # 文字は JSON の文字列 (YAML としても読める), 無い値は null
    ready = stages["inference"]["ready"]
    text = READY_CONFIG_TEXT.substitute(
        **{key: yaml_value(ready[key])
           for key in ("library", "model", "weights", "sha256")},
        path=yaml_value(READY_BUNDLE),
        name=yaml_value(ready_name(stages, name)),
    )

    return text


def yaml_value(value: Optional[str]) -> str:
    """
    文字を YAML の値にする関数

    Parameters
    ----------
    value: Optional[str]
        文字 (無ければ None)

    Returns
    ----------
    text: str
        JSON の文字列 (None は null)
    """
    # 引用符で囲み, 数や真偽値に読まれないように
    text = "null" if value is None else json.dumps(value, ensure_ascii=False)

    return text


def ready_script(problem: dict, literal: Callable) -> str:
    """
    問題の設定から, 既製のモデルの一式を作るスクリプトを作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    literal: Callable
        値を Python の式にする関数 (skeleton.literal)

    Returns
    ----------
    script: str
        bundle.py の中身
    """
    # 頭の値 (問題の名前と何を出すか)
    script = Template(READY_SCRIPT_TEXT).substitute(
        config=literal(READY_CONFIG),
        name=literal(str(problem.get("name", ""))),
        predict=literal(str(active_value(problem, f"{PART}.what") or "")),
    )

    return script


def ready_readme(problem: dict, stages: dict, raw: Optional[str]) -> str:
    """
    既製のモデルのプロジェクトの README を作る関数

    Parameters
    ----------
    problem: dict
        問題の設定 (版 2)
    stages: dict
        受け渡し
    raw: Optional[str]
        写した入れたデータの場所 (プロジェクトのディレクトリから. 入れて
        いなければ None)

    Returns
    ----------
    text: str
        Markdown の文書
    """
    # 筋道の要約
    ready = stages["inference"]["ready"]
    lines = [f"# {problem.get('name', '')}", "",
             "ML-Quest の問題の設定 (`problem.yaml`) から書き出した, 既製の"
             "モデルをそのまま使う骨組み (学習しない. ML-Inference の推論の"
             "一式を作る).", "", "## 課題から行動まで", ""]
    lines += [f"- {sentence}" for sentence in describe(problem)]

    # 1 から 2. 一式づくりと推論
    pinned = (f"決めておいた hash (`{ready['sha256'][:12]}…`) と読んだ重み"
              "が違えば作らない" if ready["sha256"] else
              "hash を決めていないので比べずに作り, 求めた hash を表示する. "
              "それを config の `ready.sha256` と問題の設定の重みの hash に"
              "書くと, 次から同じ重みに固定できる")
    lines += [
        "", "## 次にすること", "",
        "データの分析, モデルの作成, 学習の段階は通らない. 一式を作る環境"
        "には ML-Inference と, torchvision の重みを読む ML-ModelControl の"
        "追加の依存 (`ml-modelcontrol[torchvision]`) を入れる.", "",
        "1. **一式づくり** (ML-Inference): `python bundle.py`. config "
        f"(`{READY_CONFIG}`) の `ready` の値 (出どころ `{ready['library']}`, "
        f"モデル `{ready['model']}`, 重み `{ready['weights']}`) を "
        "`make_ready_bundle` に渡し, `bundle.path` (`" + READY_BUNDLE
        + "`) に推論の一式 (版 6) を作る. 重みは一式に写さず, 使うときに"
        "出どころから読む (初回は取りに行き, PyTorch の置き場 "
        "`torch.hub.get_dir()` の `checkpoints` に置く. ネットワークを使え"
        f"ない場所では, そこに重みのファイルを置く). {pinned}. 種類の名前"
        "の表と前処理は重みの説明から読む",
        "2. **推論** (ML-Inference, ML-Provide): ML-Provide で作った一式を"
        "登録し, 画像のファイルかディレクトリを渡すと, 種類の名前と確信の"
        "度合いが返る"
        + (f" (入れたデータはこのディレクトリの `{raw}` に写した)" if raw
           else "")
        + ". 一式を作るときに重みのライセンスの注意が出たら, 使ってよいか"
        "を確かめる. 比べる正解が無いので, 当たり具合は確かめ方 (9 の節) "
        "で別に確かめる",
        "", "問題の設定の既製のモデルの答えを変えたら, 書き出し直すか "
        "config を手で直す.", "",
    ]
    text = "\n".join(lines)

    return text
