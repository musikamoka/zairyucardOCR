# -*- coding: utf-8 -*-
"""
在留カードOCR一括処理ツール v4.6 (DeepSeek-OCR-2)
==================================================
v4.6の改善（国籍精度の根本対策 — 画像前処理＋辞書照合）:
  - ホログラム抑制前処理: 在留カードの彩紋/ホログラム光彩により
    片仮名国籍（ベトナム等）が潰れOCRが誤読する問題に対し、RGB最小値
    （着色光彩を除去し黒文字を残す）＋コントラスト強調を導入。
    国籍が要再確認のとき抑制パスで再OCRし補正する。
  - 国籍ファジー照合: 編集距離で実在国名へ安全に補正
    （ナカノナム→ベトナム, パンクラデシュ→バングラデシュ 等）。
    曖昧値・ハルシネーション（メンタル/日本/大和等）は補正せず要確認へ。
  - マージ規則更新: 旧値が辞書外ゴミなら妥当な国名で上書き可
    （正規国名どうしのサイレント置換は引き続き禁止）。

v4.5の修正（氏名ノイズ混入の根絶）:
  - 氏名候補に「氏名らしさ」検証を追加。DATE OF BIRTH / RESIDENCE CARD /
    Y M D などのラベル語ノイズが氏名欄に出力される不具合を修正
  - 候補先頭・末尾のラベル語を自動除去し、純ノイズ候補は破棄
  - ローマ字氏名は全語が大文字英字かつ非ラベル語、漢字氏名は2〜4字に限定

v4.4の変更（速度の正しい最適化）:
  ■ 既定を逐次（num_workers=1）に変更
    - 自己回帰デコードは「显存帯域」という単一物理資源で律速される。
      同一GPUへ2プロセスでモデルを2基ロードすると、帯域を奪い合い
      かつVRAMが逼迫してKVキャッシュが溢れ、却って大幅に遅くなる
      （実測: 並列で78秒/枚まで悪化）。並列は既定で無効化。
  ■ 単一モデルを速くする最適化を導入
    - torch.inference_mode() で推論（勾配計算・余分なメモリ確保を排除）
    - TF32高速演算を許可（Blackwell世代で有効、OCR精度への影響は軽微）
    - cudnn.benchmark 有効化
    - sdpa attention（融合カーネル）を既定化
  ■ さらに速くしたい場合（GUIの環境設定で調整可）
    - base_size / image_size を 1024 → 768 に下げる
      （トークン数は解像度の二乗で増えるため、最も効く。
       低解像度カードでは精度低下に注意。auto+局部再スキャンが緩衝）
    - scan_mode=auto なら md だけで妥当な札は Free OCR を省略（既定で有効）

v4.3〜v4.1の機能（抽出・正規化・検証）はすべて維持。

v4.1の修正:
  - 国籍ラベル語「地域」の誤捕捉を排除
  - 住所「未定（届出後裏面に記載）」を定型文言として標準化出力
  - 2行に跨るローマ字氏名の連結
  - 日付の前導ゼロ統一（2025年5月7日→2025年05月07日）
  - 国名の一意サフィックス補正（シール/メタール→ネパール等、曖昧時は保留）
  - 要確認リストCSV出力（番号形式・国名辞書・空欄の検証に通らない項目を列挙）

v4の機能（光学的弱点への対策）:
  1. カード自動切り出し＋拡大前処理
     - ページ内のカード領域を自動検出して切り出し
     - 有効幅が min_card_width 未満なら LANCZOS で拡大
     - 低解像度PDF（カードが頁の一部しかない等）の認識失敗を防止
  2. ヘッダー局部再スキャン
     - 番号/氏名/国籍が疑わしい場合、mdモードのbbox座標から
       該当領域を切り出し3倍拡大してFree OCRを再実行
     - 片仮名国名（ベトナム/ネパール等）×防伪地紋の誤読対策
  3. 検証駆動: 番号形式・国名辞書で妥当性を判定し、必要時のみ再試行

実行:
  conda activate deepseek-ocr
  python ZairyuCardOCR_GUI.py
"""

import os
import re
import csv
import json
import time
import base64
import urllib.request
import urllib.error
import threading
import unicodedata
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

# ============================================================
# 設定管理
# ============================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

DEFAULT_CONFIG = {
    # ── vLLM サーバ設定（WSL内で動くvLLMへHTTPで投げる）────────────────
    #   このGUIはWindows側で動き、推論はWSLのvLLMサーバが担当する。
    #   サーバ起動例（WSLのdsocr環境内）:
    #     export VLLM_USE_FLASHINFER_MOE_FP16=0
    #     export VLLM_USE_FLASHINFER_SAMPLER=0
    #     vllm serve deepseek-ai/DeepSeek-OCR \
    #       --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
    #       --no-enable-prefix-caching --mm-processor-cache-gb 0 --enforce-eager
    "server_url":   "http://localhost:8000/v1",  # WSL2はlocalhostがWindowsへ転送される
    "served_model": "deepseek-ai/DeepSeek-OCR",  # vllm serve で立てたモデル名と一致させる
    "request_timeout": 600,                       # 1枚あたりの最大待ち時間(秒)
    "model_path": r"D:\desk\26OCR\DeepSeek-OCR-2",
    "output_dir": r"D:\desk\26OCR\output",
    "temp_dir":   r"D:\desk\26OCR\temp_images",
    "pdf_dpi":    600,           # カードの「ネイティブ解像度」を決める最重要パラメータ。
                                 #   低DPI(250等)だと国籍「中国」等の小さな漢字が潰れ、モデルが
                                 #   "N"などと誤読する（実機で確認）。元の高DPI設定では国籍まで
                                 #   読めていた。下の max_card_width でOCR入力上限を抑えるので、
                                 #   高DPIでもタイル爆発(=旧130秒)は起きない。国籍が読めない時は
                                 #   さらに 700〜800 へ上げる。速度優先なら 400 へ下げる。
    # DeepSeek-OCR の解像度モード。重要: 本モデル構築で実際に動く組合せは限られる。
    #   - Gundam: base_size=1024, image_size=640, crop_mode=True
    #       → これだけが実機で出力を出せた（生OCRダンプの実績あり）。採用。
    #   - image_size=1024 + crop_mode=True は非正規の組合せで、タイルのtoken数が
    #     モデル想定を超え CUDA device-side assert を誘発した（実機で確認）。使用禁止。
    #   - image_size=1024 + crop_mode=False（Base）は本構築で空出力になった。使わない。
    # 角の番号など小文字は image_size=640 のタイルで拾える。広告等の幻覚や反復ループは
    # モデルに触れず後処理（_collapse_repeats + ラベル限定抽出）で吸収する。
    "base_size":  1024,
    "image_size": 640,
    "crop_mode":  True,
    "scan_mode":  "region",      # region=固定版式モード（既定/推奨）。本スキャン1回で左カラム、
                                 #   番号/性別/国籍だけ固定位置クロップで個別OCR。無駄なモード総当りなし。
                                 #   single=フルスキャンのみ / auto=条件付き補完パス。
    "region_upscale": 1.5,       # 領域クロップの拡大率（小さい字の可読性）。
    "region_max_px":  1024,      # 領域クロップの最長辺上限(px)。これより大きい切片は縮小。
                                 #   実測: 1024で十分な精度かつ高速(prefill軽量)。大きくしても
                                 #   生成内容は変わらず時間だけ増える（中央行は縮小で国籍改善）。
    "enable_garbled_retry": False,  # 崩札検出時の grounding 再読。GPU不安定化の恐れがあり既定OFF。
    "single_prompt": "free",     # singleモードの読み方:
                                 #   "free"=Free OCR。版面に縛られず全列を線形に読む。
                                 #     写真隣の中央列（性別/国籍）も拾う。人の目で全部読む方式。
                                 #   "grounding"=markdown。住所/日付は綺麗だが中央列を落とすことあり。
                                 #   "both"=両方読んで統合（確実だが2回スキャン）。
    "enable_rescan": True,       # auto時のみ有効。single時は無視。
                                 #   小さく印字される番号(右上)/住所を局部拡大で補う。
    "min_card_width": 1600,      # カード最小有効幅(px)。未満なら拡大
    "max_card_width": 2800,      # カード最大有効幅(px)。超過なら縮小（OCR入力=タイル数を上限制御）。
                                 #   これにより高DPIでもタイル数=処理時間が暴れない。
                                 #   国籍など小さい字を残すため広め(2800)。さらに必要なら3200。
                                 #   番号が読めない場合はここを2400〜2800へ上げる手もある。
    "anti_repeat": True,         # 反復ループ抑制。DeepSeek-OCR公式 vLLM の
                                 #   NoRepeatNGramLogitsProcessor（窓付きn-gram）を移植し、
                                 #   GenerationMixin.generateをクラス単位でパッチして
                                 #   全generate経路へlogits_processorとして注入。出力logits
                                 #   のみ制御で安全。完全一致のno_repeat_ngram_sizeと違い、
                                 #   ループのドリフト（EXPIRATION→EXPIRED…）も早期に断てる。
    "ngram_size": 30,            # 公式既定=30（窓=90）。ドリフトが残るなら20前後へ下げる。
    "max_new_tokens": 1024,      # 暴走長の上限（公式コミュニティ対策 issue #89）。
                                 #   在留カード本文は十分収まる。さらに速くしたいなら768等へ。
    "hologram_pass": True,       # auto時の国籍ホログラム抑制リトライ。single時は無視。
    "num_workers": 1,            # 既定は逐次（1）。GPUの帯域が単一資源のため、
                                 #   同一GPUへの多プロセスは却って遅くなる環境が多い
    "attn_impl":  "sdpa",        # sdpa(高速)/eager(互換)。sdpa失敗時は自動eager
    "fast_math":  True,          # TF32等の高速演算を許可（精度影響は軽微）
}

def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
                cfg.update(json.load(fp))
        except Exception:
            pass
    return cfg

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)

CSV_HEADERS = [
    "在留カード番号", "氏名", "生年月日", "性別", "国籍",
    "住所", "在留資格", "在留期間", "許可年月日"
]

# ============================================================
# 正規化 + フィールド抽出
# ============================================================
_KANJI_MAP = str.maketrans({
    "资": "資", "别": "別", "劳": "労", "间": "間", "满": "満",
    "请": "請", "样": "様", "动": "動",
})
_KANJI_MULTI = [("滿", "満"), ("·", "・"), ("･", "・")]

_COUNTRIES = [
    "ベトナム", "中国", "韓国", "フィリピン", "ネパール", "インドネシア",
    "ミャンマー", "スリランカ", "バングラデシュ", "カンボジア", "モンゴル",
    "タイ", "インド", "台湾", "ブラジル", "ペルー", "米国", "カナダ",
    "マレーシア", "パキスタン", "ウズベキスタン",
]

_STATUS_LIST = [
    "技術・人文知識・国際業務", "日本人の配偶者等", "永住者の配偶者等",
    "家族滞在", "特定技能", "技能実習", "経営・管理", "特定活動",
    "永住者", "定住者", "留学", "研究", "教育", "介護", "医療", "興行",
]
_STATUS_EN = {"student": "留学", "college student": "留学", "dependent": "家族滞在"}

_ADDR_LINE = re.compile(
    r'^(?:東京都|北海道|京都府|大阪府|'
    r'[一-龥]{2,3}県)[^\r\n]{3,}|^未定[（(]届出後裏面[にの]?記載[）)]')

_NUM_RE = re.compile(r'^[A-Z]{2}\d{8}[A-Z]{2}$')


def _norm_date(s: str) -> str:
    """日付の前導ゼロ統一: 2025年5月7日 → 2025年05月07日"""
    return re.sub(
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',
        lambda m: f"{m.group(1)}年{int(m.group(2)):02d}月{int(m.group(3)):02d}日",
        s)

_NAME_PATTERNS = [
    r'氏名[\s:]*([A-Z][A-Z\s.\-]{3,}?)(?=\s*(?:NAME|\n|$))',
    r'氏名[\s:]*([一-龥々]{1,4}(?:[ ][一-龥々]{1,4}){0,3})',
    r'(?<![0-9A-Za-z])([A-Z]{2,}(?:\s+[A-Z.\-]+)+)\s*\n\s*NAME',
    r'NAME\s*\n\s*([A-Z]{2,}(?:\s+[A-Z.\-]+)+)',
    # 「氏名/Name」ラベル行の次行にある漢字氏名（例: 氏名→Name→洪吉童 / 洪 吉 童）
    # 単字スペース区切り（洪 吉 童）も拾えるよう先頭を {1,4} にする。
    r'(?:氏名|NAME|Name)[\s:]*\n\s*([一-龥々]{1,4}(?:[ 　][一-龥々]{1,4}){0,3})\s*(?:\n|$)',
    # 版面崩れ対策: ローマ字氏名がカード種別ヘッダ(RESIDENCE/RESTRICTED CARD)の
    # 直下に浮く札（氏名/NAMEラベルから離れる）。通常札では次行が「番号」等の和文
    # なのでローマ字条件により発火しない。
    r'(?:RESIDENCE|RESTRICTED|RESIDENT)\s+CARD\s*\n+\s*([A-Z][A-Z\s.\-]{3,}?)(?=\s*\n)',
]
_NAME_REJECT = ("在留", "生年", "住居", "国籍", "出生", "性別", "日本国")

# 氏名候補から除外すべき定型ラベル語（OCRが氏名欄に紛れ込ませる英字ノイズ）
_NAME_STOPWORDS = {
    "NAME", "STATUS", "SEX", "ADDRESS", "NO", "DATE", "BIRTH", "OF",
    "RESIDENCE", "CARD", "GOVERNMENT", "JAPAN", "NATIONALITY", "REGION",
    "PERIOD", "STAY", "EXPIRATION", "PERMIT", "TYPE", "VALIDITY", "THIS",
    "STUDENT", "DEPENDENT", "MOJ", "DIPLOMA", "Y", "M", "D", "F",
    "ANDDRESS", "ADDESS", "STATISTIC", "STATS", "DONEMENT", "COVERNMENT",
}


def _looks_like_name(v: str) -> bool:
    """氏名らしさの検証。ラベルノイズ（DATE OF BIRTH 等）を弾く。"""
    if not v:
        return False
    # 漢字氏名（2〜4字、姓名の区切りスペース可）はOK
    if re.fullmatch(r'[一-龥々]{1,4}(?:[ ][一-龥々]{1,4}){0,3}', v):
        return True
    # ローマ字氏名: 全語が大文字英字で、いずれもラベル語でないこと
    words = v.split()
    if not words or len(words) > 5:
        return False
    if not all(re.fullmatch(r"[A-Z][A-Z.\-']*", w) for w in words):
        return False
    # ストップワードを含む候補は氏名ではない（DATE/OF/BIRTH 等）
    if any(w in _NAME_STOPWORDS for w in words):
        return False
    # 1語だけかつ3文字未満は信頼しない
    if len(words) == 1 and len(words[0]) < 3:
        return False
    return True


def _common_suffix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


def _collapse_repeats(s: str, max_repeat: int = 3) -> str:
    """連続して繰り返される行を圧縮する（モデルの反復ループ対策）。
    数字を無視して類似判定するため、"1 年(Year of issue: 1)"・"2 年(…2)"…の
    ような連番暴走も同一視して max_repeat 回で打ち切る。
    "就労就労…" のような完全反復も対象。GPUに触れない純後処理で安全。"""
    if not s:
        return s
    out, prev_key, count = [], None, 0
    for line in s.split("\n"):
        key = re.sub(r'\d+', '#', line.strip())
        if key and key == prev_key:
            count += 1
            if count <= max_repeat:
                out.append(line)
        else:
            prev_key, count = key, 1
            out.append(line)
    # 同一行内での短い反復（"就労就労就労…"）も圧縮
    res = "\n".join(out)
    res = re.sub(r'(.{2,30}?)\1{3,}', r'\1\1\1', res)
    return res


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = _collapse_repeats(s)                     # 反復ループを先に圧縮
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_KANJI_MAP)
    for a, b in _KANJI_MULTI:
        s = s.replace(a, b)
    s = s.replace("**", "").replace("__", "")   # markdown強調記号を除去（ラベル抽出を妨げる）
    s = s.replace("性别", "性別")                  # 簡体「别」→「別」（SEXラベル誤読対策）
    s = s.replace("八イツ", "ハイツ")
    # 日付の数字が空白で分断される版面対策:
    #   "2 9 9 6 年 1 0 月 0 1 日" → "2996年10月01日"
    # 年・月・日を含むひとまとまりを丸ごと捉え、内部空白を全除去（住所等は無傷）。
    # 量化子は {0,N} で上限を付ける（崩札の超長出力での暴走/灾难的回溯を防止）。
    s = re.sub(r'\d[\d\s]{0,24}?年[\d\s]{0,12}?月[\d\s]{0,12}?日',
               lambda m: re.sub(r'\s+', '', m.group(0)), s)
    # 「2 年 1 1 月」のような期間プレフィックスの空白も詰める（日付を伴わない箇所）
    s = re.sub(r'\d[\d\s]{0,8}?年(?:[\d\s]{0,8}?月)?(?=[\s（(])',
               lambda m: re.sub(r'\s+', '', m.group(0)), s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if not m:
        return n
    if not n:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def _fix_country(v: str) -> str:
    """在留カードの国籍は必ず実在の国名。OCR誤読を辞書へ安全に補正する。
    ホログラム/彩紋による片仮名誤読を編集距離で照合するが、
    曖昧な値（メンタル等のハルシネーション）は補正せず要確認に回す。"""
    if not v:
        return ""
    v = re.sub(r'国籍$', '', v).strip()
    if v in _COUNTRIES:
        return v
    # 部分一致（辞書の国名が誤読値に含まれる）
    for c in _COUNTRIES:
        if c in v:
            return c
    # 「ナム」を含む片仮名 → ベトナム（最頻出・識別力が高い）
    if "ナム" in v and not v.startswith("カナ"):
        return "ベトナム"
    # 片仮名のみの値に限りファジー照合（漢字ハルシネーションは弾く）
    if re.fullmatch(r'[ァ-ヴー・]+', v) and len(v) >= 3:
        scored = sorted(((_levenshtein(v, c), c) for c in _COUNTRIES),
                        key=lambda x: x[0])
        best_d, best_c = scored[0]
        second_d = scored[1][0]
        # 厳格な受理条件（誤マッチ＝サイレントエラーを避ける）:
        #   長い国名(≥6字)は距離≤2、それ以外は距離≤1、
        #   かつ2位との差が2以上（一意に近い場合のみ）
        ok = (len(best_c) >= 6 and best_d <= 2) or (best_d <= 1)
        if ok and (second_d - best_d) >= 2:
            return best_c
    return v


def _pick(*patterns, text="", group=1):
    for p in patterns:
        m = re.search(p, text, re.MULTILINE)
        if m:
            return m.group(group).strip()
    return ""


def _extract_name(text: str) -> str:
    """全パターン×全マッチから候補を収集し、氏名らしさを検証した上で
    語数最多の候補を採用（md側がJOHNで切れfree側にJOHN SMITHがある等に対応）"""
    cands = []
    for pi, p in enumerate(_NAME_PATTERNS):
        for m in re.finditer(p, text, re.MULTILINE):
            v = re.sub(r'\s*NAME\s*$', '', m.group(1).strip()).strip()
            # ラベル語が混入した候補（RESIDENCE CARD…等）は末尾から削る
            words = v.split()
            while words and words[-1] in _NAME_STOPWORDS:
                words.pop()
            while words and words[0] in _NAME_STOPWORDS:
                words.pop(0)
            v = " ".join(words)
            if not v or any(r in v for r in _NAME_REJECT):
                continue
            # 氏名が1語のみの場合、次行が大文字短語なら連結（2行跨ぎ対策）
            if pi == 0 and " " not in v:
                nm = re.match(r'\s*\n([A-Z]{2,10})\s*(?:\n|$)', text[m.end():])
                if nm and nm.group(1) not in _NAME_STOPWORDS:
                    v = v + " " + nm.group(1)
            if _looks_like_name(v):
                cands.append(v)
    if not cands:
        return ""
    best = max(cands, key=lambda x: (len(x.split()), -cands.index(x)))
    # 単字スペース区切りの漢字氏名（洪 吉 童）は詰めて 洪吉童 に（姓名スペースの
    # 「山田 太郎」のような複数字区切りは温存する）。
    parts = best.split()
    if len(parts) > 1 and all(re.fullmatch(r'[一-龥々]', p) for p in parts):
        best = "".join(parts)
    return best


def _extract_card_number(text: str) -> str:
    """在留カード番号（英2+数8+英2）を堅牢に抽出する。

    番号は『英字2 + 数字8 + 英字2』。連続8桁の数字はカード上で番号にしか
    現れない（生年月日・各種日付は年月日で区切られる）ため、これを構造
    アンカーにする。「番号」ラベルやタイトル『在留カード/RESIDENCE CARD』
    の位置に依存しないので、タイトル領域を誤って拾うことがない。
    OCRが前後に余分な文字を付けても（例: LAB12345678CD）数字コアの前後
    から正しい2文字を取り直して復元する。末尾欠け（AB12345678C など）は
    11桁のまま返し、検証側（_NUM_RE）が要確認に回す。"""
    if not text:
        return ""
    # Tier1: 「番号」ラベル直後の正規形（最も信頼度が高い）
    m = re.search(r'番号[\s:：]*(?:No\.?)?[\s:：]*([A-Z]{2}\d{8}[A-Z]{2})', text)
    if m:
        return m.group(1)
    # Tier2: 単独の正規形（英数字に挟まれていない12文字）
    m = re.search(r'(?<![A-Z0-9])([A-Z]{2}\d{8}[A-Z]{2})(?![A-Z0-9])', text)
    if m:
        return m.group(1)
    # Tier3: 連続8桁をアンカーに前後の英字を取り直す（前置のゴミ文字を除去）
    best = ""
    for m in re.finditer(r'([A-Z]{0,3})(\d{8})([A-Z]{0,3})', text):
        head = m.group(1)[-2:]
        digits = m.group(2)
        tail = m.group(3)[:2]
        if len(head) == 2 and len(tail) == 2:
            cand = head + digits + tail
            if _NUM_RE.match(cand):
                return cand
        if len(head) == 2 and len(tail) >= 1 and not best:
            best = head + digits + tail   # 末尾欠け等は11桁候補として保持
    return best


_ADDR_LABEL = r'(?:住居地|在居地|居住地|住所地|出生地|在住|住所)'


def _address_from_text(text: str) -> str:
    """1つのOCRテキストから最良の住所を1件返す（パス単位の抽出）。
    クロスパス投票（md/free/header/footer）で使う基本部品。"""
    if not text:
        return ""
    if re.search(r'未定[\s（(]*届出後?裏面[にの仁]?記載', text):
        return "未定（届出後裏面に記載）"
    cands = []
    for m in re.finditer(
            _ADDR_LABEL + r'[\s:]*(?:A[DN]DRESS[\s:]*)?([^\r\n]+)', text, re.IGNORECASE):
        c = re.sub(r'^A[DN]DRESS[\s:]*', '', m.group(1), flags=re.IGNORECASE).strip()
        if _ADDR_LINE.match(c):
            cands.append(c)
    for m in re.finditer(_ADDR_LABEL + r'[^\r\n]*\n((?:[^\r\n]*\n?){0,3})', text):
        for line in m.group(1).splitlines():
            line = re.sub(r'^A[DN]DRESS[\s:]*', '', line, flags=re.IGNORECASE).strip()
            if _ADDR_LINE.match(line):
                cands.append(line)
    for line in text.splitlines():
        s = re.sub(r'^A[DN]DRESS[\s:]*', '', line.strip(), flags=re.IGNORECASE).strip()
        if _ADDR_LINE.match(s):
            cands.append(s)
    if not cands:
        return ""
    complete = [c for c in cands if re.search(r'[丁目番地号]', c)]
    pool = complete or cands
    a = pool[0]
    a = re.sub(r'\s*A[DN]DRESS\s*$', '', a).strip()
    a = re.sub(r'(?<=\d) (?=\d)', '', a)   # 数字間の空白除去（0 0号→00号）
    return a


def _addr_complete(a: str) -> bool:
    return bool(a) and bool(re.search(r'[丁目番地号]', a))


def _addr_norm(a: str) -> str:
    return re.sub(r'\s+', '', a or '')


def _extract_country(md: str, free: str) -> str:
    """国籍ラベル(国籍/NATIONALITY)以降の窓内で辞書の正式国名を直接探す。
    英語ラベル行「NATIONALITY (BANGLADESH)」等を挟んでも片仮名値へ到達できる。
    窓内に正式国名が無ければ旧ロジック（ラベル直後語＋辞書ファジー補正）へ。"""
    texts = [t for t in (md, free) if t]
    # 在留カードの国籍は決して「日本」ではない。表頭「日本国政府」を国籍と誤認しないよう
    # 全段で日本を除外し、表頭文字列も走査対象から除く。
    def _scrub(t):
        return t.replace("日本国政府", "").replace("GOVERNMENT OF JAPAN", "")
    # ① 国籍ラベル以降の窓内に正式国名（窓を80字に拡大、日本は除外）
    for t in texts:
        st = _scrub(t)
        for m in re.finditer(r'国籍|NATIONALITY', st):
            window = st[m.start(): m.start() + 80]
            hit = next((c for c in _COUNTRIES if c != "日本" and c in window), "")
            if hit:
                return hit
    # ② ラベル直後の漢字/片仮名を拾い、辞書へファジー補正
    cands = []
    for t in texts:
        c = _pick(r'国籍[・:]*\s*(?:地域)?[\s:／/・]*([一-龥ァ-ヶー]+)', text=t)
        if c and c not in ("地域", "日本"):
            cands.append(c)
    exact = next((c for c in cands for k in _COUNTRIES if k != "日本" and k in c), "")
    if exact:
        return exact
    # ③ 値がラベルから離れて出力された版面の救済：全文に既知国名があれば採用
    #    （住所等は日本語地名のみで他国名は出ない前提。日本・表頭は除外済み）
    for t in texts:
        body = _scrub(t)
        hit = next((c for c in _COUNTRIES if c != "日本" and c in body), "")
        if hit:
            return hit
    # ④ 最後にラベル直後語（辞書外でも人手確認用に返す。日本は返さない）
    return _fix_country(cands[0] if cands else "")


def _looks_garbled(text: str) -> bool:
    """繰り返しループ等でモデル出力が崩れているかの簡易判定。
    同一行が多数回繰り返す/ユニーク率が極端に低い場合に True。"""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if len(lines) < 12:
        return False
    from collections import Counter
    c = Counter(lines)
    top = c.most_common(1)[0][1]
    uniq_ratio = len(c) / len(lines)
    # 「在留力一下」等の在留カード誤読が複数回出るのも崩れの兆候
    misread = sum(1 for l in lines if "在留力" in l or "力一下" in l)
    return top >= 8 or uniq_ratio < 0.45 or misread >= 2


def extract_fields(free_text: str, md_text: str) -> dict:
    md   = normalize_text(md_text)
    free = normalize_text(free_text)
    combined = md + "\n" + free
    f = {}

    # 在留カード番号（連続8桁を構造アンカーにした堅牢抽出）
    f["在留カード番号"] = _extract_card_number(combined)

    # 氏名
    f["氏名"] = _extract_name(combined)

    # 生年月日
    f["生年月日"] = _norm_date(_pick(
        r'(?:生年月日|出生年月|出生日期)[\s:]*(\d{4}年\d{1,2}月\d{1,2}日)',
        r'(?:生年月日|出生年月|出生日期)[^\d]{0,20}?(\d{4}年\d{1,2}月\d{1,2}日)',
        # ラベルが値と離れる版面/領域クロップ用: 日付の直後に 性別/男/女/M/F または
        # DATE OF BIRTH が来るのは生年月日だけ（許可/交付/在留期間は後続が異なる）。
        r'(\d{4}年\d{1,2}月\d{1,2}日)[\s　]*\n?[\s　]*'
        r'(?:性別|[男女][\s　]*[MFＭＦ]|[男女]\b|DATE\s*OF\s*BIRTH|SEX)',
        text=combined))

    # 性別（「性別→SEX→男」のようにラベルが挟まる出力にも対応）
    v = _pick(r'性別[\s:]*([男女])',
              r'性別\s*\n?\s*(?:SEX|Sex)\s*\n?\s*([男女])',
              text=combined)
    if not v:
        # ラベルが日付に巻き込まれて落ちる出力（例「…31日别男 M.」「…31日 男 M.国籍」）。
        # DeepSeek-OCRの中央行は [男/女] の直後に英字性別コード M./F. か「国籍」が続くので
        # それを錨に拾う（カード上の男/女はこの欄のみ＝誤検出しにくい）。
        v = _pick(r'([男女])[\s　]*[ＭMＦF][\.．]',
                  r'([男女])[\s　]*国籍',
                  text=combined)
    if not v:
        mf = _pick(r'性別[\s:]*\(?([MF])\)?', text=combined)
        v = {"M": "男", "F": "女"}.get(mf, "")
    f["性別"] = v

    # 国籍: ラベル(国籍/NATIONALITY)以降の窓内で辞書の正式国名を直接探す。
    #   モデルが「国籍・地域 / NATIONALITY (BANGLADESH) / バングラデシュ」のように
    #   英語ラベル行を挟んでも、片仮名の実値（バングラデシュ等）まで到達できる。
    f["国籍"] = _extract_country(md, free)

    # 住所: markdown(grounding)を主、free を従。クロスパス投票は infer_card 側。
    #   長い住所はモデルがパスごとに別々に誤読するため、合意を最優先する。
    md_addr = _address_from_text(md)
    free_addr = _address_from_text(free)
    if md_addr and free_addr and _addr_norm(md_addr) == _addr_norm(free_addr):
        addr = md_addr                          # ① md/free 合意 → 高信頼
    elif _addr_complete(md_addr):
        addr = md_addr                          # ② grounding(md) を優先
    elif _addr_complete(free_addr):
        addr = free_addr                        # ③ md が不完全なら free
    else:
        addr = md_addr or free_addr
    f["住所"] = addr
    # 投票・要確認判定用にパス別候補を保持（CSVには出力されない private キー）
    f["_addr_md"] = md_addr
    f["_addr_free"] = free_addr

    # 在留資格
    v = ""
    m = re.search(r'在留資格[\s:]*([^\s\r\n]+)', combined)
    if m:
        cand = m.group(1)
        v = next((s for s in _STATUS_LIST if cand.startswith(s)), "")
    if not v:
        v = next((s for s in _STATUS_LIST if s in combined), "")
    if not v:
        for en, ja in _STATUS_EN.items():
            if re.search(r'STATUS[\s:]*' + en, combined, re.IGNORECASE):
                v = ja
                break
    f["在留資格"] = v

    # 在留期間
    m = re.search(
        r'(\d+年\d+月|\d+年|\d+月)\s*[（(]\s*(\d{4}年\d{1,2}月\d{1,2}日)\s*[）)]',
        combined)
    f["在留期間"] = _norm_date(f"{m.group(1)}（{m.group(2)}）") if m else ""

    # 許可年月日。ラベルと値の間に英語ラベル(PERIOD OF VALIDITY…)が挟まる版面が
    # 多いので、ラベル以降の最初の日付を許可日として拾う。ただし「…まで有効」の
    # 有効期限日付は許可日ではないので除外（直後の「まで」を否定先読みで弾く）。
    # 交付年月日は許可日より後に出るため、最初の非「まで」日付＝許可年月日。
    f["許可年月日"] = _norm_date(_pick(
        r'許可年月日[\s:]*(\d{4}年\d{1,2}月\d{1,2}日)(?!まで)',
        r'許可年月日[^\d]{0,60}?(\d{4}年\d{1,2}月\d{1,2}日)(?!まで)',
        text=combined))

    return f


def suspicious_fields(f: dict) -> list:
    """妥当性検証: 形式・辞書に基づき「疑わしい」項目を列挙"""
    bad = []
    if not _NUM_RE.match(f.get("在留カード番号", "")):
        bad.append("在留カード番号")
    # 形式は合うが二重読みで割れた番号（B→R 等の静默誤読）も要確認へ
    if f.get("_番号_uncertain") and "在留カード番号" not in bad:
        bad.append("在留カード番号")
    if not f.get("氏名"):
        bad.append("氏名")
    if f.get("国籍") not in _COUNTRIES:
        bad.append("国籍")
    for k in ("生年月日", "性別", "住所", "在留資格", "在留期間", "許可年月日"):
        if not f.get(k):
            bad.append(k)
    # 住所が非空でも「丁目/番地/号」もハイフン番地（例:円山町5-5, 2-3-1）も
    # 欠く截断値（例:「東京都世田谷区」）のみ不完全とみなし要確認に回す。
    # 「未定（届出後裏面に記載）」は法定の定型文言なので除外。
    addr = f.get("住所", "")
    _addr_ok = re.search(r'[丁目番地号]', addr) or re.search(r'\d+\s*-\s*\d+', addr)
    if (addr and "住所" not in bad
            and addr != "未定（届出後裏面に記載）"
            and not _addr_ok):
        bad.append("住所")
    # クロスパス投票で合意が得られなかった住所（誤読の疑い）も要確認に回す
    if f.get("_addr_uncertain") and "住所" not in bad:
        bad.append("住所")
    return bad


# ============================================================
# 画像前処理: カード自動切り出し＋拡大
# ============================================================
def suppress_hologram(pil_img):
    """在留カードのホログラム/彩紋オーバーレイを抑制する。
    着色ホログラムは少なくとも1チャンネルで明るいが、黒文字は全チャンネルで
    暗い。RGBの最小値を取ることで着色光彩を除去し文字を残す。
    片仮名国籍がホログラム光彩で潰れる問題に有効。"""
    import numpy as np
    arr = np.asarray(pil_img.convert("RGB")).astype(np.int16)
    mn = arr.min(axis=2)
    lo = np.percentile(mn, 10)
    hi = np.percentile(mn, 72)
    st = np.clip((mn - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype("uint8")
    from PIL import Image
    return Image.fromarray(st)


def normalize_card(img_path: str, out_path: str, log=None):
    """カードを検出→傾き補正→ID-1規格(86:54)の統一キャンバス1720x1080へ正規化。
    原画のフレーム差（位置/余白/縦横比/わずかな傾き）を吸収し、全カードで版面位置が
    一致するようにする。これにより固定領域クロップが原画差でズレなくなる。
    OpenCV必須。失敗/未導入時は None を返し、呼び側は従来処理にフォールバック。"""
    try:
        import cv2
        import numpy as np
    except Exception as e:
        if log:
            log(f"  正規化スキップ（OpenCV未導入）: pip install opencv-python  [{e}]")
        return None
    try:
        CW, CH = 1720, 1080            # 86:54 標準カード
        im = cv2.imread(img_path)
        if im is None:
            return None
        H, W = im.shape[:2]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # 非白（カード）領域: ページ余白の白を除外。内部の白窓はモルフォロジで充填。
        mask = (gray < 238).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < W * H * 0.12:    # カードが小さすぎ＝検出失敗
            return None
        rect = cv2.minAreaRect(c)                # 傾き込みの最小外接矩形
        box = cv2.boxPoints(rect).astype("float32")
        s = box.sum(1); d = np.diff(box, 1).ravel()
        o = np.array([box[np.argmin(s)], box[np.argmin(d)],
                      box[np.argmax(s)], box[np.argmax(d)]], dtype="float32")
        wA = np.linalg.norm(o[1] - o[0]); hA = np.linalg.norm(o[3] - o[0])
        if wA < hA:                              # 縦向きなら横長へ回す
            o = o[[1, 2, 3, 0]]
            wA, hA = hA, wA
        # 縦横比が標準カード(1.585)から極端に外れたら検出失敗とみなす
        ar = wA / max(hA, 1)
        if not (1.30 < ar < 1.95):
            return None
        dst = np.array([[0, 0], [CW, 0], [CW, CH], [0, CH]], dtype="float32")
        M = cv2.getPerspectiveTransform(o, dst)
        warp = cv2.warpPerspective(im, M, (CW, CH),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(out_path, warp)
        if log:
            log(f"  前処理: カード正規化 {CW}x{CH}px（傾き補正・統一フレーム）")
        return out_path
    except Exception as e:
        if log:
            log(f"  正規化失敗→従来処理へ: {e}")
        return None


def preprocess_card(img_path: str, min_width: int, max_width: int = 0,
                    log=None) -> str:
    """カードを統一フレームに正規化して返す（OpenCV）。正規化に成功すれば、
    全カードで版面位置が一致するため固定領域クロップが安定する。正規化が使えない
    場合は従来の「非白bbox切り出し＋幅制御」にフォールバックする。"""
    # ① まず正規化（傾き補正＋86:54統一キャンバス）を試みる
    norm = img_path.rsplit(".", 1)[0] + "_norm.png"
    if normalize_card(img_path, norm, log=log):
        try:
            from PIL import Image
            im = Image.open(norm).convert("RGB")
            im.save(img_path.rsplit(".", 1)[0] + "_cardhi.png")   # クロップ用フル解像度
            out = img_path.rsplit(".", 1)[0] + "_card.png"
            if max_width and im.width > max_width:
                sc = max_width / im.width
                im = im.resize((int(im.width * sc), int(im.height * sc)),
                               Image.LANCZOS)
            im.save(out)
            return out
        except Exception:
            pass
    # ② フォールバック: 従来の非白bbox切り出し＋幅制御
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(img_path).convert("RGB")
        g = np.asarray(im.convert("L"), dtype=np.int16)
        H, W = g.shape
        border = np.concatenate([g[:3].ravel(), g[-3:].ravel(),
                                 g[:, :3].ravel(), g[:, -3:].ravel()])
        bg = int(np.median(border))
        ys, xs = np.where(np.abs(g - bg) > 25)
        if len(xs) > 500:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            mw, mh = int((x1 - x0) * 0.02), int((y1 - y0) * 0.02)
            x0, y0 = max(0, x0 - mw), max(0, y0 - mh)
            x1, y1 = min(W, x1 + mw), min(H, y1 + mh)
            if (x1 - x0) > W * 0.15 and (y1 - y0) > H * 0.08:
                im = im.crop((x0, y0, x1, y1))
        try:
            im.save(img_path.rsplit(".", 1)[0] + "_cardhi.png")
        except Exception:
            pass
        note = ""
        if im.width < min_width:
            sc = min_width / im.width
            im = im.resize((int(im.width * sc), int(im.height * sc)),
                           Image.LANCZOS)
            note = "（拡大適用）"
        elif max_width and im.width > max_width:
            sc = max_width / im.width
            im = im.resize((int(im.width * sc), int(im.height * sc)),
                           Image.LANCZOS)
            note = "（縮小適用=高速化）"
        out = img_path.rsplit(".", 1)[0] + "_card.png"
        im.save(out)
        if log:
            log(f"  前処理: カード切出 {im.width}x{im.height}px" + note)
        return out
    except Exception as e:
        if log:
            log(f"  前処理スキップ: {e}")
        return img_path


# ============================================================
# 固定版式の領域座標（正規化キャンバス1720x1080=86:54 に対する比率 l,t,r,b）
#   recheck30の10カードを正規化し全数で目視検証済み。値はやや広めに取り、
#   正規化の残差(±数%)を吸収する。最重要は center（性別+国籍の一行を丸ごと）。
# ============================================================
_DEFAULT_REGIONS = {
    "center":  [0.38, 0.205, 0.97, 0.330],  # 性別+国籍 一行（全数検証済み）
    "number":  [0.72, 0.050, 0.96, 0.150],  # 番号 AB...（右上、グリッド実測）
    "name":    [0.115, 0.140, 0.62, 0.215], # 氏名（表頭GOVERNMENT行を避け y14%から）
    "birth":   [0.12, 0.215, 0.45, 0.305],  # 生年月日 1985年12月31日
    "address": [0.11, 0.325, 0.72, 0.420],  # 住居地（太字帯）
    "period":  [0.10, 0.600, 0.74, 0.710],  # 在留期間（満了日）
    "kyoka":   [0.10, 0.780, 0.58, 0.890],  # 許可年月日（左下・実測80〜87%）
    # 後方互換（個別sex/natを参照する旧コード用。centerに含まれる）
    "sex":     [0.40, 0.205, 0.60, 0.330],
    "nat":     [0.70, 0.205, 0.97, 0.330],
}


# ============================================================
# bbox解析（mdモード出力の<|det|>タグ）
# ============================================================
_DET_RE = re.compile(
    r'<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', )

def parse_header_bbox(md_raw: str):
    """番号/氏名/生年月日/国籍を含む行のbbox（0-999正規化）の合併領域を返す"""
    if not md_raw or "<|det|>" not in md_raw:
        return None
    blocks = re.findall(
        r'<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]?<\|/det\|>\s*\n?([^<]*)',
        md_raw)
    keys = ("番号", "氏名", "生年月日", "国籍")
    boxes = [(int(a), int(b), int(c), int(d))
             for a, b, c, d, txt in blocks if any(k in txt for k in keys)]
    if not boxes:
        return None
    pad = 15
    return (max(0,   min(b[0] for b in boxes) - pad),
            max(0,   min(b[1] for b in boxes) - pad),
            min(999, max(b[2] for b in boxes) + pad),
            min(999, max(b[3] for b in boxes) + pad))


# ============================================================
# 反復ループ抑制（DeepSeek-OCR 公式 vLLM 実装の移植 + クラス単位パッチ）
#   DeepSeek-OCR は既知の反復バグ（Whisper類似）を持つ。公式の対策は
#   vLLM 用 NoRepeatNGramLogitsProcessor（窓付きn-gram, ngram=30/window=90,
#   whitelist={<td>,</td>}）。ところが本モデルの Transformers infer() 経路には
#   この processor が組み込まれておらず、それがループの直接原因。
#   そこで公式 process/ngram_norepeat.py のアルゴリズムを忠実に移植し
#   （vLLMの1次元シグネチャ→Transformersのバッチ[batch,seq]/[batch,vocab]へ適合）、
#   GenerationMixin.generate をクラス単位で1度だけパッチして全generate経路へ
#   logits_processor として注入する。さらにコミュニティ既知の対策（issue #89）に
#   従い max_new_tokens を上限化して暴走長を確実に有界化する。
#   ※ no_repeat_ngram_size（完全一致）はループのドリフト
#     （PERIOD OF EXPIRATION→EXPIRED→EXPIRING…）を捕まえられず無効だった。
#     公式は「窓内での部分一致n-gram禁止」なので早期にループの芽を断てる。
# ============================================================
_GEN_PATCHED = False
_NGRAM_SIZE = 30        # 公式既定。小さくするとドリフトにも強いが正当な反復も禁じうる
_NGRAM_WINDOW = 90      # 公式既定
_NGRAM_WHITELIST = {128821, 128822}   # <td>, </td>（表セルの正当な反復を除外）
_MAX_NEW_TOKENS = 1024  # 暴走長の上限（在留カード本文は十分収まる。0で無効）
_NGRAM_PROC = None


def _make_official_ngram_processor():
    """DeepSeek-OCR-vllm/process/ngram_norepeat.py の NoRepeatNGramLogitsProcessor を
    Transformers のバッチ版 LogitsProcessor シグネチャに忠実移植。"""
    from transformers import LogitsProcessor

    class _OfficialNoRepeatNGram(LogitsProcessor):
        def __init__(self, ngram_size, window_size, whitelist_token_ids):
            self.n = int(ngram_size)
            self.w = int(window_size)
            self.wl = set(whitelist_token_ids or ())

        def __call__(self, input_ids, scores):
            # input_ids: LongTensor[batch, seq] / scores: FloatTensor[batch, vocab]
            n, w = self.n, self.w
            for b in range(input_ids.shape[0]):
                seq = input_ids[b].tolist()
                L = len(seq)
                if L < n:
                    continue
                prefix = tuple(seq[L - (n - 1):])        # 直近(n-1)トークン
                start = max(0, L - w)                     # 窓: 直近 window_size
                end = L - n + 1
                banned = set()
                for i in range(start, end):
                    ng = tuple(seq[i:i + n])
                    if ng[:-1] == prefix:                 # 部分一致 → 次トークン禁止
                        banned.add(ng[-1])
                banned -= self.wl
                for t in banned:
                    scores[b, t] = -float("inf")
            return scores

    return _OfficialNoRepeatNGram(_NGRAM_SIZE, _NGRAM_WINDOW, _NGRAM_WHITELIST)


def _install_no_repeat_classlevel(ngram_size=30, max_new_tokens=1024, log=None):
    """全generate経路へ公式 n-gram processor を注入し、max_new_tokens を上限化する。"""
    global _GEN_PATCHED, _NGRAM_SIZE, _MAX_NEW_TOKENS, _NGRAM_PROC
    _NGRAM_SIZE = int(ngram_size)
    _MAX_NEW_TOKENS = int(max_new_tokens)
    try:
        _NGRAM_PROC = _make_official_ngram_processor()
    except Exception as e:
        if log:
            log(f"  反復抑制processor生成失敗: {e}")
        _NGRAM_PROC = None
    if _GEN_PATCHED:
        if log:
            log(f"  反復ループ抑制 適用済み（窓付きn-gram={_NGRAM_SIZE}/{_NGRAM_WINDOW}, "
                f"max_new_tokens≤{_MAX_NEW_TOKENS}）")
        return
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception as e:
        if log:
            log(f"  反復抑制パッチ不可（GenerationMixin取得失敗）: {e}")
        return
    _orig = GenerationMixin.generate

    def _patched(self, *args, **kwargs):
        # 1) 公式 窓付きn-gram processor を logits_processor へ合流
        if _NGRAM_PROC is not None:
            lp = kwargs.get("logits_processor")
            if not lp:
                kwargs["logits_processor"] = [_NGRAM_PROC]
            else:
                try:
                    if _NGRAM_PROC not in list(lp):
                        kwargs["logits_processor"] = list(lp) + [_NGRAM_PROC]
                except Exception:
                    kwargs["logits_processor"] = [_NGRAM_PROC]
        # 2) max_new_tokens を上限化（暴走を確実に有界化, issue #89）
        if _MAX_NEW_TOKENS:
            cur = kwargs.get("max_new_tokens")
            kwargs["max_new_tokens"] = min(cur, _MAX_NEW_TOKENS) if cur else _MAX_NEW_TOKENS
        return _orig(self, *args, **kwargs)

    _patched._zairyu_orig = _orig
    GenerationMixin.generate = _patched
    _GEN_PATCHED = True
    if log:
        log(f"  反復ループ抑制 有効化（公式 窓付きn-gram {_NGRAM_SIZE}/{_NGRAM_WINDOW} を"
            f"全generate経路へ注入, max_new_tokens≤{_MAX_NEW_TOKENS}）")


# ============================================================
# OCRエンジン
# ============================================================
class OCREngine:
    def __init__(self, log_fn):
        self.model = None          # vLLM利用時は「準備済み」を表す番兵として使う
        self.tokenizer = None
        self.log = log_fn
        self.loaded_path = None
        self.server_url = "http://localhost:8000/v1"
        self.served_model = "deepseek-ai/DeepSeek-OCR"
        self.request_timeout = 600
        self._last_raw = {"markdown": "", "free": "", "rescan": ""}

    def load(self, model_path: str = None, attn_impl: str = "eager",
             fast_math: bool = True, anti_repeat: bool = True,
             ngram_size: int = 30, max_new_tokens: int = 1024):
        # ローカルにモデルは読み込まない。推論はWSL内のvLLMサーバが担当する。
        #   （torch / transformers はこのGUIには不要。Windows側はHTTPクライアントに徹する）
        # ここではサーバへの疎通だけ確認しておき、落ちていれば早めに気づけるようにする。
        if self.model is not None and self.loaded_path == (model_path, attn_impl):
            return
        url = self.server_url.rstrip("/") + "/models"
        self.log(f"vLLMサーバへ接続確認中: {self.server_url}")
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
            ids = [m.get("id") for m in data.get("data", [])]
            self.log(f"  サーバ接続OK ✓  モデル: {', '.join(filter(None, ids)) or '(不明)'}")
            if self.served_model and ids and self.served_model not in ids:
                self.log(f"  ⚠ 設定の served_model='{self.served_model}' がサーバ側に"
                         f"見当たりません。先頭の '{ids[0]}' を使用します。")
                self.served_model = ids[0]
        except Exception as e:
            # 疎通失敗でもここでは止めない（サーバ起動直後など）。実際の推論時に明示エラーを出す。
            self.log(f"  ⚠ サーバへ接続できませんでした: {e}")
            self.log(f"    WSLで vllm serve が起動しているか、server_url"
                     f"（{self.server_url}）が正しいか確認してください。")
        self.model = True          # 準備済み番兵
        self.loaded_path = (model_path, attn_impl)
        self.log("準備完了 ✓（推論はvLLMサーバが実行します）")

    def _infer(self, prompt, image_file, out_dir, cfg, max_tokens=None):
        # WSLのvLLMサーバ（OpenAI互換 /chat/completions）へ画像を送りOCR結果を得る。
        #   prompt は "<image>\n<|grounding|>..." 形式で来るので、先頭の <image> と改行を
        #   取り除いた指示文だけを渡す（画像プレースホルダはvLLMが自動で挿入する）。
        #   max_tokens: 領域クロップ等の短文では小さく絞り、無駄な生成を抑える。
        self.server_url = cfg.get("server_url", self.server_url)
        self.served_model = cfg.get("served_model", self.served_model)
        timeout = int(cfg.get("request_timeout", getattr(self, "request_timeout", 600)))
        # out_dir は旧経路の名残。work_dir を確実に作る副作用を保つため残す（領域クロップ保存先）。
        os.makedirs(out_dir, exist_ok=True)

        instr = re.sub(r"^\s*<image>\s*", "", prompt).strip()
        if not instr:
            instr = "Free OCR."

        with open(image_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        ext = os.path.splitext(image_file)[1].lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

        body = {
            "model": self.served_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": instr},
                ],
            }],
            "max_tokens": (int(max_tokens) if max_tokens
                           else int(cfg.get("max_new_tokens", 1024))),
            "temperature": 0.0,
            # DeepSeek-OCR公式の反復抑制（窓付きn-gram）をサーバ側ロジットプロセッサで有効化。
            #   サーバ起動時に --logits_processors ...NGramPerReqLogitsProcessor を付けておくこと。
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": int(cfg.get("ngram_size", 30)),
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],  # <td>, </td>
            },
        }
        data = json.dumps(body).encode("utf-8")
        url = self.server_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "ignore")[:500]
            except Exception:
                detail = ""
            raise RuntimeError(f"vLLMサーバ HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"vLLMサーバへ接続できません（{self.server_url}）。"
                f"WSLで vllm serve が起動しているか確認してください: {e}") from e
        # トークン使用量を診断用に積算（サーバ応答のusage）。
        try:
            u = resp.get("usage") or {}
            self._tok_prompt = getattr(self, "_tok_prompt", 0) + int(u.get("prompt_tokens", 0) or 0)
            self._tok_completion = getattr(self, "_tok_completion", 0) + int(u.get("completion_tokens", 0) or 0)
            self._tok_calls = getattr(self, "_tok_calls", 0) + 1
        except Exception:
            pass
        try:
            return resp["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            return ""

    def infer_card(self, image_path: str, work_dir: str, cfg: dict,
                   preprocessed: bool = False) -> dict:
        # 生OCRテキストを診断用に保持（モデルが何を読めたかを後で出力）
        self._last_raw = {"markdown": "", "free": "", "rescan": "", "footer": ""}
        # ── 前処理: カード切り出し＋拡大（プリフェッチ済みならスキップ）──
        if preprocessed:
            proc_path = image_path
        else:
            proc_path = preprocess_card(
                image_path, cfg.get("min_card_width", 1600),
                cfg.get("max_card_width", 1900), self.log)

        scan_mode = cfg.get("scan_mode", "single")

        # ── region モード: 固定版式の座標で要点だけ個別に読む（既定/推奨）─────
        #   在留カードは版式が固定。実機データが示す通り、本スキャン1回で左カラム
        #   （氏名/住所/在留資格/在留期間/許可年月日 等）はちゃんと読める。問題は
        #   毎回ループに食われたり位置がブレたりする「番号(右上の小印字)」と
        #   「性別・国籍(中央列)」だけ。そこを全解像度カードの固定位置でピンポイントに
        #   切り出してFree OCRする。クロップは小さく短文なので循環しない（カード画像で
        #   座標検証済み）。無駄なモード総当りは一切しない。
        if scan_mode == "region":
            main_cap = int(cfg.get("main_max_tokens", 512))
            region_cap = int(cfg.get("region_max_tokens", 96))
            # 切片保存先: 出力直下の単一フォルダ「切片」にまとめ、ファイル名を
            # <カード名>_<フィールド>.png にする（カードごとにサブフォルダを作らない）。
            self._slice_dir = ""
            self._slice_prefix = ""
            if cfg.get("save_slices", True) and (cfg.get("run_dir") or cfg.get("output_dir")):
                self._slice_prefix = os.path.splitext(os.path.basename(image_path))[0]
                base_dir = cfg.get("run_dir") or cfg["output_dir"]
                self._slice_dir = os.path.join(base_dir, "切片")
                try:
                    os.makedirs(self._slice_dir, exist_ok=True)
                except Exception:
                    self._slice_dir = ""
            # 主スキャンは Free OCR（線形読み）。grounding は難図で整張を画像と誤判定して
            # 本文ゼロを返すうえ、版面解析が固定座標前提＝原画のフレーム差に弱い。Free は
            # 「見えている文字を全部読む」のでフレーム差に強く、番号・左カラムも安定して拾う。
            # 落としやすい中央列(性別/国籍)等だけ後段の領域クロップで補完（＝主スキャン1回）。
            free_text = self._infer(
                "<image>\nFree OCR. ",
                proc_path, os.path.join(work_dir, "mainfree"), cfg,
                max_tokens=main_cap)
            self._last_raw["free"] = free_text
            fields = extract_fields(free_text, "")

            # 崩れ(繰り返しループ)検出時の自動回避: 別プロンプト(grounding/markdown)で
            # 1回だけ読み直す。ただし grounding は Free OCR と別のGPUコードパスを通り、
            # sm_120(Blackwell)等の環境でドライバを不安定化させ得るため【既定で無効】。
            # 安定動作を最優先し、必要時のみ config の enable_garbled_retry=true で有効化。
            _filled = sum(1 for h in CSV_HEADERS if fields.get(h, "").strip())
            if (cfg.get("enable_garbled_retry", False)
                    and (_looks_garbled(free_text) or _filled <= 3)):
                self._last_raw["main_garbled"] = True
                try:
                    alt = self._infer(
                        "<image>\n<|grounding|>Convert the document to markdown. ",
                        proc_path, os.path.join(work_dir, "mainalt"), cfg,
                        max_tokens=main_cap)
                    af = extract_fields(alt, "")
                    if sum(1 for h in CSV_HEADERS if af.get(h, "").strip()) > _filled:
                        self._last_raw["free"] = (
                            free_text + "\n---[崩れ検出→grounding再読]---\n" + alt)
                        free_text, fields = alt, af
                except Exception:
                    pass

            hi = image_path.rsplit(".", 1)[0] + "_cardhi.png"
            if not os.path.exists(hi):
                hi = proc_path
            rc = cfg.get("region_coords", _DEFAULT_REGIONS)
            up = float(cfg.get("region_upscale", 1.5))

            # 番号（右上）: 本スキャンで妥当な番号が出ていなければ領域OCR。
            #   この分岐に入る＝主スキャンは番号を読めていない＝佐証なし。番号は最重要
            #   フィールドゆえ、領域OCRのみが出所のときは必ず要確認へ回す（人手で一瞥）。
            #   さらに異なる拡大率で二重読みし、割れたら「誤読の疑い」として強めの理由に。
            #   （実機 22TE475: B→R を両拡大率で安定誤読 → 一致でもすり抜けるため）
            if not fields.get("在留カード番号"):
                nt = self._ocr_region(hi, rc.get("number"), work_dir,
                                      "number", cfg, up, region_cap)
                self._last_raw["region_number"] = nt
                num = extract_fields(nt, "").get("在留カード番号", "")
                if num:
                    nt2 = self._ocr_region(hi, rc.get("number"), work_dir,
                                           "number2", cfg, up * 1.3, region_cap)
                    self._last_raw["region_number2"] = nt2
                    num2 = extract_fields(nt2, "").get("在留カード番号", "")
                    fields["在留カード番号"] = num
                    fields["_番号_uncertain"] = True      # 領域のみ＝佐証なし→必ず要確認
                    if num2 and num2 == num:
                        fields["_番号_src"] = "region_only"
                        self.log(f"  → 番号 領域OCR: {num}（二重読み一致・領域のみ→要確認）")
                    else:
                        fields["_番号_src"] = "mismatch"
                        self.log(f"  → 番号 領域OCR: {num}"
                                 f"（二重読み不一致 {num!r}≠{num2!r} → 要確認）")

            # 氏名（左上）: 左カラムは主スキャンが拾うのが基本だが、grounding が
            #   部分的に氏名行を落とすことがあるため固定位置で補完（bbox中央値で座標導出）。
            if not fields.get("氏名"):
                nmx = self._ocr_region(hi, rc.get("name"), work_dir,
                                       "name", cfg, up, region_cap)
                self._last_raw["region_name"] = nmx
                nm = extract_fields(nmx, "").get("氏名", "")
                if nm:
                    fields["氏名"] = nm
                    self.log(f"  → 氏名 領域OCR: {nm}")

            # 生年月日: 主スキャンが落としたら固定位置で補完（実測 y22〜30%）
            if not fields.get("生年月日"):
                bt = self._ocr_region(hi, rc.get("birth"), work_dir,
                                      "birth", cfg, up, region_cap)
                self._last_raw["region_birth"] = bt
                bd = extract_fields(bt, "").get("生年月日", "")
                if bd:
                    fields["生年月日"] = bd
                    self.log(f"  → 生年月日 領域OCR: {bd}")

            # 性別＋国籍（写真隣の細い中央行）: 主スキャンが最も落とす/誤る箇所。
            #   正規化済みカードなら固定位置の「中央行」を1回読むだけで両方確実に取れる
            #   （recheck30全数で検証）。中央行は値の確度が高いので、辞書内の国籍は主スキャン
            #   値より優先採用する（主が「日本」等の誤りでも中央行で上書き）。
            need_center = (not fields.get("性別") or not fields.get("国籍")
                           or fields.get("国籍") not in _COUNTRIES)
            if need_center:
                ctx = self._ocr_region(hi, rc.get("center"), work_dir,
                                       "center", cfg, up, region_cap)
                self._last_raw["region_center"] = ctx
                cf = extract_fields(ctx, "")
                if cf.get("性別") and not fields.get("性別"):
                    fields["性別"] = cf["性別"]
                    self.log(f"  → 性別 中央行OCR: {cf['性別']}")
                cnat = cf.get("国籍", "")
                if cnat and (cnat in _COUNTRIES or not fields.get("国籍")):
                    if fields.get("国籍") != cnat:
                        self.log(f"  → 国籍 中央行OCR: {cnat}"
                                 + ("（主スキャン値を置換）" if fields.get("国籍") else ""))
                    fields["国籍"] = cnat

            # 住所（中央の太字帯）: 本スキャンが住所値を出さないことがあるため固定位置で補完
            if not fields.get("住所"):
                at = self._ocr_region(hi, rc.get("address"), work_dir,
                                     "address", cfg, up, region_cap)
                self._last_raw["region_address"] = at
                addr = extract_fields(at, "").get("住所", "")
                if addr:
                    fields["住所"] = addr
                    self.log(f"  → 住所 領域OCR: {addr}")

            # 在留期間: 同上、固定位置で補完
            if not fields.get("在留期間"):
                pt = self._ocr_region(hi, rc.get("period"), work_dir,
                                    "period", cfg, up, region_cap)
                self._last_raw["region_period"] = pt
                per = extract_fields(pt, "").get("在留期間", "")
                if per:
                    fields["在留期間"] = per
                    self.log(f"  → 在留期間 領域OCR: {per}")

            # 許可年月日（左下）: 主スキャンが下部を落とすことがあるため固定位置で補完
            #   （bbox中央値で座標導出。交付年月日も帯に入るが extract は許可側を採用）
            if not fields.get("許可年月日"):
                kt = self._ocr_region(hi, rc.get("kyoka"), work_dir,
                                      "kyoka", cfg, up, region_cap)
                self._last_raw["region_kyoka"] = kt
                ky = extract_fields(kt, "").get("許可年月日", "")
                if ky:
                    fields["許可年月日"] = ky
                    self.log(f"  → 許可年月日 領域OCR: {ky}")

            self._vote_address(fields, self._last_raw)
            return fields

        # ── single モード: フルスキャンのみ（再スキャン・局部スキャンなし）──
        #   重要: grounding markdown は版面解析が綺麗だが、写真の隣の細い中央列
        #   （性別・国籍）を取り逃すことがある（実機で確認: 性別/国籍が生出力に出ない）。
        #   Free OCR は版面に縛られず「見えている文字を線形に全部読む」ため、中央列の
        #   性別・国籍も拾う（人の目で一度に全部読むのに近い）。公式 窓付きn-gram +
        #   max_new_tokens 上限でループは抑制済み。single_prompt で選択可:
        #     "free"      : Free OCR 1回。性別/国籍を含め全列を読む（既定）。
        #     "grounding" : markdown 1回。住所/日付が綺麗だが中央列を落とすことがある。
        #     "both"      : 両方読んで統合（住所/日付=markdown, 性別/国籍=free）。確実だが2回。
        if scan_mode == "single":
            sp = cfg.get("single_prompt", "free")
            md_text = free_text = ""
            if sp in ("grounding", "both"):
                md_text = self._infer(
                    "<image>\n<|grounding|>Convert the document to markdown. ",
                    proc_path, os.path.join(work_dir, "markdown"), cfg)
                self._last_raw["markdown"] = md_text
            if sp in ("free", "both"):
                free_text = self._infer(
                    "<image>\nFree OCR. ",
                    proc_path, os.path.join(work_dir, "free"), cfg)
                self._last_raw["free"] = free_text
            self.log(f"  → singleモード（{sp}）: フルスキャンのみ")
            # free優先で中央列(性別/国籍)を拾い、markdownで住所/日付を補完
            return extract_fields(free_text, md_text)

        # ── パス1: Markdown構造化（auto モード）──
        md_text = self._infer(
            "<image>\n<|grounding|>Convert the document to markdown. ",
            proc_path, os.path.join(work_dir, "markdown"), cfg)
        self._last_raw["markdown"] = md_text

        fields = extract_fields("", md_text)
        sus = suspicious_fields(fields)

        if scan_mode == "auto" and not sus:
            self.log("  → Markdownのみで全項目妥当（Free OCRスキップ）")
            return fields

        # ヘッダー局部再スキャン（拡大切り出し）で確実に読める上部項目。
        # 実測: 拡大した上帯はフルカードのFree OCRより上部項目を綺麗に読む
        # （番号/氏名/生年月日/性別/住所が1パスで揃う）。
        # 残る要確認がこれらだけなら、フルカードFree OCR（≈1パス丸ごと=最大の遅延）は
        # 冗長。スキップして拡大ヘッダー再スキャンへ直行し、約1パス分を短縮する。
        _HEADER_COVERS = {"在留カード番号", "氏名", "生年月日", "性別", "国籍", "住所"}
        skip_free = (bool(sus) and set(sus).issubset(_HEADER_COVERS)
                     and cfg.get("enable_rescan", True))

        if skip_free:
            self.log(f"  → 要確認は上部項目のみ（{', '.join(sus)}）。"
                     "Free OCRを省略し拡大ヘッダー再スキャンへ直行")
        else:
            # ── パス2: Free OCR ──
            if sus:
                self.log(f"  → 要確認{len(sus)}項目をFree OCRで補完: {', '.join(sus)}")
            free_text = self._infer(
                "<image>\nFree OCR. ",
                proc_path, os.path.join(work_dir, "free"), cfg)
            self._last_raw["free"] = free_text
            fields = extract_fields(free_text, md_text)
            sus = suspicious_fields(fields)

        # ── パス3: ヘッダー局部再スキャン（番号/氏名/国籍/生年月日/性別/住所）──
        target = [k for k in sus if k in ("在留カード番号", "氏名", "国籍",
                                          "生年月日", "性別", "住所")]
        if target and cfg.get("enable_rescan", True):
            self.log(f"  → ヘッダー局部再スキャン実行: {', '.join(target)}")
            patch = self._rescan_header(proc_path, work_dir, cfg, md_text)
            if patch:
                # 氏名は常にマージ対象（語数増のみ採用なので安全）
                self._merge(fields, patch, list(set(target) | {"氏名"}))
            sus = suspicious_fields(fields)

        # ── パス4: フッター局部再スキャン（許可年月日/在留期間のみ）──
        #   反復ループで日付類が欠落した場合の救済に限定する。
        #   住所はここでは扱わない（フッターの3倍拡大は表を誤生成しやすく、
        #   実測でハルシネーションの<table>を量産して時間を浪費した）。
        #   住所はヘッダー再スキャン＋住所投票で補完する。
        foot = [k for k in sus if k in ("許可年月日", "在留期間")]
        if foot and cfg.get("enable_rescan", True):
            self.log(f"  → フッター局部再スキャン実行: {', '.join(foot)}")
            patch = self._rescan_footer(proc_path, work_dir, cfg)
            if patch:
                self._merge(fields, patch, foot)

        # ── 住所のクロスパス投票 ──
        #   md/free/ヘッダー再扫/フッター再扫 の各住所を集め、正規化して投票。
        #   ≥2パスが一致 → その住所を高信頼で採用。
        #   全て不一致 → 主pass値を残しつつ要確認フラグ（静默誤読を人手検出へ）。
        self._vote_address(fields, self._last_raw)
        return fields

    @staticmethod
    def _vote_address(fields: dict, raw: dict):
        import collections
        vals = [
            fields.get("_addr_md", ""),
            fields.get("_addr_free", ""),
            _address_from_text((raw or {}).get("rescan", "")),
            _address_from_text((raw or {}).get("footer", "")),
        ]
        vals = [v for v in vals if _addr_complete(v)]
        if not vals:
            # 主pass/footer等に「完全な」住所候補が無い。fields["住所"]が領域OCR等の
            # 単一ソースで完全（丁目/番地/号あり）なら“不一致”は存在しない→フラグしない。
            # （実機: grounding主扫が整図失敗→裁块だけが正しい住所を出した場合の誤報を防ぐ）
            # 不完全な住所は suspicious_fields の不完全判定が「不完全」として拾う。
            return
        cnt = collections.Counter(_addr_norm(v) for v in vals)
        top, n = cnt.most_common(1)[0]
        if n >= 2:
            # 過半数一致 → その住所を高信頼で採用
            fields["住所"] = next(v for v in vals if _addr_norm(v) == top)
            fields.pop("_addr_uncertain", None)
        elif len(cnt) >= 2:
            # 完全な候補が複数あるが全て異なる＝各passが別々に誤読＝信頼不可。
            # 現値を維持しつつ要確認に回す（静默誤読を人手検出へ）。
            fields["_addr_uncertain"] = True
        else:
            # 完全な候補が単一passのみ（矛盾なし）→ 採用、フラグなし
            fields["住所"] = vals[0]
            fields.pop("_addr_uncertain", None)

    def _ocr_region(self, hi_path, frac_box, work_dir, name, cfg,
                    upscale=1.5, max_tokens=96):
        """全解像度カードから固定位置の領域を比率で切り出し、拡大してFree OCR。
        領域が小さく内容も短い。生成上限(max_tokens)を小さく絞るので、万一同一行を
        繰り返しても即座に頭打ちになり時間を浪費しない。対象フィールドだけ鮮明に読む。"""
        if not frac_box:
            return ""
        try:
            from PIL import Image
            im = Image.open(hi_path)
            W, H = im.size
            l, t, r, b = frac_box
            box = (max(0, int(l * W)), max(0, int(t * H)),
                   min(W, int(r * W)), min(H, int(b * H)))
            crop = im.crop(box)
            # 各クロップは呼び出し側が指定した倍率で拡大する。中央行(国籍/性別)だけは
            # 過去に2倍へ拡大しており、resize時に片仮名が不鮮明化していた（実測2028px）。
            # → 中央行のみ呼び出し側で控えめな倍率(center_upscale,既定1.0=原寸)を渡す。
            # それ以外(kyoka/period/address等)は従来の1.5倍を維持（縮小しすぎて許可日付を
            # 落とす回帰を防ぐ）。巨大化だけは安全弁として上限を設ける。
            cap = int(cfg.get("region_max_px", 1800))
            if upscale and abs(upscale - 1.0) > 0.01:
                nw, nh = int(crop.width * upscale), int(crop.height * upscale)
                if max(nw, nh) > cap:                  # 安全弁: 過大化のみ抑制
                    s2 = cap / max(crop.width, crop.height)
                    nw, nh = int(crop.width * s2), int(crop.height * s2)
                crop = crop.resize((max(1, nw), max(1, nh)), Image.LANCZOS)
            cp = os.path.join(work_dir, f"region_{name}.png")
            crop.save(cp)
            # 目視検証用に切片を出力フォルダにも残す（座標ズレをユーザーが確認できる）
            sd = getattr(self, "_slice_dir", "")
            if sd:
                try:
                    pre = getattr(self, "_slice_prefix", "") or "card"
                    crop.save(os.path.join(sd, f"{pre}_{name}.png"))
                except Exception:
                    pass
            return self._infer("<image>\nFree OCR. ", cp,
                               os.path.join(work_dir, f"region_{name}_out"),
                               cfg, max_tokens=max_tokens)
        except Exception as e:
            self.log(f"  領域OCR({name})失敗: {e}")
            return ""

    def _rescan_footer(self, image_path, work_dir, cfg):
        """カード下部（住居地〜許可年月日）を切り出し3倍拡大して再OCR。
        住居地は概ねカード高さの35〜43%に位置するため、確実に含めるよう
        上端を30%から取る（上端を42%にすると住所行を切り落としていた不具合を修正）。
        反復ループで本文下部が欠落したカードの住所・日付類を救済する。"""
        try:
            from PIL import Image
            im = Image.open(image_path)
            W, H = im.size
            # 上端30%（住居地ラベルの上）〜最下部。左カラムのみで番号画像列を除外。
            region = (0, int(H * 0.30), int(W * 0.72), H)
            crop = im.crop(region)
            crop = crop.resize((crop.width*2, crop.height*2), Image.LANCZOS)  # 3倍→2倍: 90MP級の巨大画像化を防止
            crop_path = os.path.join(work_dir, "footer_zoom.png")
            crop.save(crop_path)
            text = self._infer("<image>\nFree OCR. ",
                               crop_path, os.path.join(work_dir, "rescan_footer"),
                               cfg)
            self._last_raw["footer"] = text
            return extract_fields(text, "")
        except Exception as e:
            self.log(f"  フッター再スキャン失敗: {e}")
            return None

    def _rescan_header(self, image_path, work_dir, cfg, md_raw, targets=None):
        """カード上部を全幅で切り出し拡大して再OCR。
        重要: 在留カード番号は右上に小さく印字される。氏名/生年月日のbboxだけ
        切り出すと左カラムのみで番号を取り逃すため、x方向は常に全幅にする。
        国籍が対象なら片仮名の光彩潰れ対策にホログラム抑制パスも実行。"""
        try:
            from PIL import Image
            im = Image.open(image_path)
            W, H = im.size
            bbox = parse_header_bbox(md_raw)
            # y方向はヘッダー下端まで（bboxがあればその下端＋余白、無ければ45%）。
            # x方向は右上の番号を確実に含めるため常に全幅。
            if bbox:
                y_bottom = min(H, bbox[3] / 1000 * H + 0.04 * H)
            else:
                y_bottom = H * 0.45
            region = (0, 0, W, int(y_bottom))
            crop = im.crop(tuple(map(int, region)))
            crop = crop.resize((crop.width*2, crop.height*2), Image.LANCZOS)  # 全幅上帯を2倍（90MP化を防止）
            crop_path = os.path.join(work_dir, "header_zoom.png")
            crop.save(crop_path)
            text = self._infer("<image>\nFree OCR. ",
                               crop_path, os.path.join(work_dir, "rescan"), cfg)
            self._last_raw["rescan"] = text
            result = extract_fields(text, "")
            # 国籍が要再確認なら、ホログラム抑制パスで再挑戦。
            #   ただし本人カードの国籍欄は公的ホログラム（星章）が重なり、拡大でも
            #   読めない個体が多い（MOJ見本も同様）。読めない個体で毎回これを回すと
            #   約1パス分を浪費して結局空になるため、config で無効化できるようにする。
            #   既定ON（読める個体では救済になる）。確実に読めない運用なら false に。
            need_nat = (targets is None) or ("国籍" in targets)
            if (need_nat and cfg.get("hologram_pass", True)
                    and result.get("国籍") not in _COUNTRIES):
                try:
                    sup = suppress_hologram(crop)
                    sup_path = os.path.join(work_dir, "header_holo.png")
                    sup.save(sup_path)
                    t2 = self._infer("<image>\nFree OCR. ", sup_path,
                                     os.path.join(work_dir, "rescan_holo"), cfg)
                    r2 = extract_fields(t2, "")
                    if r2.get("国籍") in _COUNTRIES:
                        result["国籍"] = r2["国籍"]
                        self.log("  → ホログラム抑制で国籍を補正: "
                                 + r2["国籍"])
                except Exception as e:
                    self.log(f"  ホログラム抑制スキップ: {e}")
            return result
        except Exception as e:
            self.log(f"  再スキャン失敗: {e}")
            return None

    @staticmethod
    def _merge(fields: dict, patch: dict, targets: list):
        """再スキャン結果のマージ。妥当な値のみ採用（安全側）。
        国籍: 辞書内でも、主抽出候補と無関係な値は採用しない
              （誤読→別の正規国名への置換=サイレントエラーを防ぐ。
               不一致なら元の値を保持し要確認リストに回す）"""
        for k in targets:
            new = patch.get(k, "")
            if not new:
                continue
            if k == "在留カード番号":
                if _NUM_RE.match(new):
                    fields[k] = new
            elif k == "国籍":
                old = fields.get(k, "")
                # 妥当な国名のみ採用。旧値が辞書外（誤読ゴミ）なら上書き可。
                # 旧値も妥当な国名の場合のみ、サイレント置換を防ぐため
                # 末尾2字以上一致を要求する。
                if new in _COUNTRIES and (
                        not old or old not in _COUNTRIES
                        or new == old or _common_suffix(old, new) >= 2):
                    fields[k] = new
            elif k == "氏名":
                old = fields.get(k, "")
                # 空欄、または再スキャン側の語数が多い（=切れていた）場合に採用
                if not old or len(new.split()) > len(old.split()):
                    fields[k] = new
            else:
                # 住所を含むその他は「空欄のときのみ補完」。再スキャンの
                # 誤読版で正しい主pass値を上書きしない（安全側）。
                if not fields.get(k):
                    fields[k] = new

    @staticmethod
    def _read_mmd(d):
        p = os.path.join(d, "result.mmd")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                return fp.read()
        return ""


# ============================================================
# PDF → 画像変換
# ============================================================
def pdf_to_images(pdf_path: str, out_dir: str, dpi: int) -> list:
    import fitz
    os.makedirs(out_dir, exist_ok=True)
    images = []
    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(out_dir, f"{base}_p{i+1}.png")
        pix.save(img_path)
        images.append(img_path)
    doc.close()
    return images



# ============================================================
# 並列ワーカー（multiprocessing: 各プロセスがモデルを1基ロード）
# 同一GPU上で2ストリーム推論し、デコードの帯域待ちを相互に埋める
# ============================================================
_WORKER_ENGINE = None
_WORKER_LOGS = []


def _worker_log(msg):
    _WORKER_LOGS.append(str(msg))


def _worker_init(model_path: str, attn_impl: str, fast_math: bool = True,
                 anti_repeat: bool = True, ngram_size: int = 30,
                 max_new_tokens: int = 1024):
    """各ワーカープロセス起動時に1回だけ実行（モデルロード）"""
    global _WORKER_ENGINE
    _WORKER_ENGINE = OCREngine(_worker_log)
    _WORKER_ENGINE.load(model_path, attn_impl, fast_math, anti_repeat,
                        ngram_size, max_new_tokens)


def _worker_task(args):
    """1枚分の推論。(idx, fields, 所要秒, エラー, ログ, 生OCR) を返す"""
    idx, img, work_dir, cfg = args
    global _WORKER_LOGS
    _WORKER_LOGS = []
    t0 = time.time()
    try:
        fields = _WORKER_ENGINE.infer_card(img, work_dir, cfg)
        raw = dict(getattr(_WORKER_ENGINE, "_last_raw", {}) or {})
        err = ""
    except Exception as e:
        fields = {h: "" for h in CSV_HEADERS}
        raw = {}
        err = str(e)
    return idx, fields, time.time() - t0, err, list(_WORKER_LOGS), raw


# ============================================================
# 環境チェック
# ============================================================
def run_env_check(cfg: dict) -> list:
    results = []
    import sys
    results.append(("Python", "OK",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))
    # ── vLLMサーバへの疎通（推論はここが担当。torch等はWindows側に不要）──
    server_url = cfg.get("server_url", "http://localhost:8000/v1")
    try:
        url = server_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        ids = [m.get("id") for m in data.get("data", [])]
        results.append(("vLLMサーバ", "OK",
                        f"{server_url} / モデル: {', '.join(filter(None, ids)) or '(不明)'}"))
        want = cfg.get("served_model")
        if want and ids and want not in ids:
            results.append(("served_model", "NG",
                            f"設定='{want}' がサーバに無い。候補: {', '.join(filter(None, ids))}"))
    except Exception as e:
        results.append(("vLLMサーバ", "NG",
            f"{server_url} へ接続不可: {e} ／ WSLで vllm serve を起動してください"))
    # ── クライアント側で必要なライブラリ（画像/PDF処理）──
    try:
        import PIL  # noqa
        results.append(("Pillow", "OK", getattr(PIL, "__version__", "インストール済み")))
    except ImportError:
        results.append(("Pillow", "NG", "未インストール → pip install pillow"))
    try:
        import numpy  # noqa
        results.append(("numpy", "OK", numpy.__version__))
    except ImportError:
        results.append(("numpy", "NG", "未インストール → pip install numpy"))
    try:
        import fitz  # noqa
        results.append(("PyMuPDF", "OK", "インストール済み（PDF対応）"))
    except ImportError:
        results.append(("PyMuPDF", "NG", "未インストール → pip install pymupdf（PDF入力に必要）"))
    try:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        t = os.path.join(cfg["output_dir"], ".write_test")
        open(t, "w").write("ok"); os.remove(t)
        results.append(("出力先", "OK", cfg["output_dir"]))
    except Exception as e:
        results.append(("出力先", "NG", f"書き込み不可: {e}"))
    return results


# ============================================================
# GUI本体
# ============================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("在留カードOCR一括処理ツール v5.0  (vLLM / DeepSeek-OCR)")
        root.geometry("820x680")

        self.cfg = load_config()
        self.pdf_files = []
        self.engine = OCREngine(self.log)
        self.engine.server_url = self.cfg.get("server_url", self.engine.server_url)
        self.engine.served_model = self.cfg.get("served_model", self.engine.served_model)
        self.engine.request_timeout = int(self.cfg.get("request_timeout", 600))
        self.running = False
        self._pool = None
        self._pool_key = None
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        self.tab_main = ttk.Frame(nb)
        self.tab_cfg  = ttk.Frame(nb)
        nb.add(self.tab_main, text="  OCR処理  ")
        nb.add(self.tab_cfg,  text="  環境設定  ")
        self._build_main_tab()
        self._build_config_tab()

    # ── タブ1 ──
    def _build_main_tab(self):
        t = self.tab_main
        frm = ttk.Frame(t, padding=10); frm.pack(fill="x")
        ttk.Button(frm, text="PDFファイルを選択（複数可）",
                   command=self.select_files).pack(side="left")
        self.lbl_count = ttk.Label(frm, text="選択ファイル: 0件")
        self.lbl_count.pack(side="left", padx=12)
        ttk.Button(frm, text="クリア", command=self.clear_files).pack(side="right")

        self.listbox = tk.Listbox(t, height=7)
        self.listbox.pack(fill="x", padx=10)

        # 精度モード（いつでも切替可）: 高速=single/free, 高精度=region
        frm_mode = ttk.LabelFrame(t, text=" 精度モード（いつでも切替可） ", padding=8)
        frm_mode.pack(fill="x", padx=10, pady=(6, 0))
        init_mode = "region" if self.cfg.get("scan_mode") == "region" else "single"
        self.var_mode = tk.StringVar(value=init_mode)
        ttk.Radiobutton(frm_mode, text="高速  （single/free・全面1回・約8秒/枚）",
                        value="single", variable=self.var_mode).pack(side="left", padx=6)
        ttk.Radiobutton(frm_mode, text="高精度（region・番号/性別/国籍を個別再読・約15〜25秒/枚）",
                        value="region", variable=self.var_mode).pack(side="left", padx=6)

        frm2 = ttk.Frame(t, padding=10); frm2.pack(fill="x")
        self.btn_run = ttk.Button(frm2, text="OCR処理を開始", command=self.start)
        self.btn_run.pack(side="left")
        # サーバ操作（コード変更後の再適用を容易にする）
        ttk.Button(frm2, text="サーバ停止",
                   command=self.stop_server).pack(side="right")
        ttk.Button(frm2, text="サーバ再起動",
                   command=self.restart_server).pack(side="right", padx=6)
        self.progress = ttk.Progressbar(frm2, length=360, mode="determinate")
        self.progress.pack(side="left", padx=12, fill="x", expand=True)

        self.txt = tk.Text(t, height=18, state="disabled", font=("Consolas", 9))
        self.txt.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def stop_server(self):
        """WSL上のvLLMサーバを停止する（コード変更後の再適用を容易にする）。
        WSL以外の環境では何もしない（失敗してもアプリは継続）。"""
        import subprocess
        try:
            subprocess.run(
                ["wsl.exe", "-e", "bash", "-lic", "pkill -f 'vllm serve' || true"],
                timeout=20, capture_output=True)
            self.log("vLLMサーバへ停止信号を送信しました（pkill vllm serve）。"
                     "再起動は「サーバ再起動」または 起動.bat で。")
        except Exception as e:
            self.log(f"サーバ停止に失敗: {e}（WSL環境のWindowsからのみ動作します）")

    def restart_server(self):
        """vLLMを停止→新ウィンドウで再起動（モデルロードに1〜2分）。"""
        import subprocess
        self.stop_server()
        try:
            subprocess.Popen(
                'start "vLLM OCR Server" wsl.exe -e bash -lic '
                '"bash ~/start_ocr_server.sh"', shell=True)
            self.log("vLLMサーバを再起動中…（モデルロードに1〜2分）。"
                     "起動完了までOCR開始はお待ちください。")
        except Exception as e:
            self.log(f"サーバ再起動に失敗: {e}")

    # ── タブ2 ──
    def _build_config_tab(self):
        t = self.tab_cfg
        pad = {"padx": 10, "pady": 6}

        frm = ttk.LabelFrame(t, text=" パス設定 ", padding=12)
        frm.pack(fill="x", padx=10, pady=10)
        self.var_model  = tk.StringVar(value=self.cfg["model_path"])
        self.var_output = tk.StringVar(value=self.cfg["output_dir"])
        self.var_temp   = tk.StringVar(value=self.cfg["temp_dir"])
        self._path_row(frm, 0, "モデルパス（DeepSeek-OCR-2）", self.var_model)
        self._path_row(frm, 1, "出力先フォルダ（CSV）",        self.var_output)
        self._path_row(frm, 2, "一時フォルダ（PDF画像）",      self.var_temp)

        frm2 = ttk.LabelFrame(t, text=" 推論パラメータ ", padding=12)
        frm2.pack(fill="x", padx=10, pady=4)
        self.var_dpi  = tk.IntVar(value=self.cfg["pdf_dpi"])
        self.var_base = tk.IntVar(value=self.cfg["base_size"])
        self.var_img  = tk.IntVar(value=self.cfg["image_size"])
        ttk.Label(frm2, text="PDF描画DPI").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(frm2, from_=100, to=600, increment=50,
                    textvariable=self.var_dpi, width=8).grid(row=0, column=1, **pad)
        ttk.Label(frm2, text="base_size").grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(frm2, from_=512, to=2048, increment=256,
                    textvariable=self.var_base, width=8).grid(row=1, column=1, **pad)
        ttk.Label(frm2, text="image_size").grid(row=2, column=0, sticky="w", **pad)
        ttk.Spinbox(frm2, from_=512, to=2048, increment=256,
                    textvariable=self.var_img, width=8).grid(row=2, column=1, **pad)

        self.var_scan = tk.StringVar(value=self.cfg.get("scan_mode", "single"))  # 互換保持（未使用）
        ttk.Label(frm2, text="スキャンモード").grid(row=3, column=0, sticky="w", **pad)
        ttk.Label(frm2, text="→ メイン画面の「精度モード」で切替",
                  foreground="#666").grid(row=3, column=1, sticky="w", **pad)

        self.var_rescan = tk.BooleanVar(value=self.cfg.get("enable_rescan", True))
        ttk.Checkbutton(frm2, text="ヘッダー局部再スキャン（番号/氏名/国籍の誤読対策）",
                        variable=self.var_rescan).grid(row=4, column=0,
                        columnspan=2, sticky="w", **pad)

        self.var_minw = tk.IntVar(value=self.cfg.get("min_card_width", 1600))
        ttk.Label(frm2, text="カード最小有効幅px（未満なら拡大）").grid(row=5, column=0, sticky="w", **pad)
        ttk.Spinbox(frm2, from_=800, to=3200, increment=200,
                    textvariable=self.var_minw, width=8).grid(row=5, column=1, **pad)

        self.var_nw = tk.IntVar(value=int(self.cfg.get("num_workers", 1)))
        ttk.Label(frm2, text="並列ワーカー数（通常は1を推奨。2は多くの環境で逆効果）").grid(row=6, column=0, sticky="w", **pad)
        ttk.Spinbox(frm2, from_=1, to=2,
                    textvariable=self.var_nw, width=8).grid(row=6, column=1, **pad)

        self.var_attn = tk.StringVar(value=self.cfg.get("attn_impl", "sdpa"))
        ttk.Label(frm2, text="attention実装（sdpa=高速 / eager=互換）").grid(row=7, column=0, sticky="w", **pad)
        ttk.Combobox(frm2, textvariable=self.var_attn, width=10, state="readonly",
                     values=["sdpa", "eager"]).grid(row=7, column=1, **pad)

        self.var_fast = tk.BooleanVar(value=self.cfg.get("fast_math", True))
        ttk.Checkbutton(frm2, text="高速演算（TF32）を許可 ※推奨。精度への影響は軽微",
                        variable=self.var_fast).grid(row=8, column=0,
                        columnspan=2, sticky="w", **pad)

        frm3 = ttk.Frame(t, padding=10); frm3.pack(fill="x")
        ttk.Button(frm3, text="設定を保存", command=self.save_settings).pack(side="left", padx=4)
        ttk.Button(frm3, text="環境チェック実行", command=self.env_check).pack(side="left", padx=4)
        ttk.Button(frm3, text="既定値に戻す", command=self.reset_settings).pack(side="left", padx=4)

        self.txt_check = tk.Text(t, height=11, state="disabled", font=("Consolas", 9))
        self.txt_check.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    def _path_row(self, parent, row, label, var):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(parent, textvariable=var, width=56).grid(row=row, column=1, padx=6, pady=5)
        ttk.Button(parent, text="参照...",
                   command=lambda v=var: self._browse_dir(v)).grid(row=row, column=2, padx=6)

    def _browse_dir(self, var):
        d = filedialog.askdirectory(title="フォルダを選択")
        if d:
            var.set(d.replace("/", os.sep))

    def _on_close(self):
        try:
            if self._pool is not None:
                self._pool.terminate()
        except Exception:
            pass
        self.root.destroy()

    def _get_pool(self, cfg):
        """並列ワーカープールを取得（設定が同じなら再利用＝モデル再ロード回避）"""
        import multiprocessing as mp
        key = (cfg["model_path"], int(cfg["num_workers"]),
               cfg.get("attn_impl", "eager"), cfg.get("fast_math", True),
               bool(cfg.get("anti_repeat", True)),
               int(cfg.get("ngram_size", 30)),
               int(cfg.get("max_new_tokens", 1024)))
        if self._pool is not None and self._pool_key == key:
            return self._pool
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None
        self.log(f"並列ワーカー {key[1]}基を起動中...")
        self.log(f"  各ワーカーがモデルをロードします（初回のみ1〜2分・VRAM約{key[1]*7}GB使用）")
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(key[1], initializer=_worker_init,
                              initargs=(cfg["model_path"],
                                        cfg.get("attn_impl", "eager"),
                                        cfg.get("fast_math", True),
                                        bool(cfg.get("anti_repeat", True)),
                                        int(cfg.get("ngram_size", 30)),
                                        int(cfg.get("max_new_tokens", 1024))))
        self._pool_key = key
        return self._pool

    # ── 設定 ──
    def collect_cfg(self) -> dict:
        # 精度モードはメイン画面のトグルを唯一の正とする（region=高精度 / single=高速）
        _scan = self.var_mode.get() if hasattr(self, "var_mode") else \
            self.cfg.get("scan_mode", "single")
        _sp = self.cfg.get("single_prompt", "free")   # singleモードの読み方（free既定）
        return {
            "model_path": self.var_model.get().strip(),
            "output_dir": self.var_output.get().strip(),
            "temp_dir":   self.var_temp.get().strip(),
            "pdf_dpi":    int(self.var_dpi.get()),
            "base_size":  int(self.var_base.get()),
            "image_size": int(self.var_img.get()),
            "crop_mode":  bool(self.cfg.get("crop_mode", True)),
            "enable_rescan": bool(self.var_rescan.get()),
            "min_card_width": int(self.var_minw.get()),
            "max_card_width": int(self.cfg.get("max_card_width", 1900)),
            "num_workers": int(self.var_nw.get()),
            "attn_impl":  self.var_attn.get(),
            "fast_math":  bool(self.var_fast.get()),
            "anti_repeat": bool(self.cfg.get("anti_repeat", True)),
            "ngram_size": int(self.cfg.get("ngram_size", 30)),
            "max_new_tokens": int(self.cfg.get("max_new_tokens", 1024)),
            "server_url": self.cfg.get("server_url", "http://localhost:8000/v1"),
            "served_model": self.cfg.get("served_model", "deepseek-ai/DeepSeek-OCR"),
            "request_timeout": int(self.cfg.get("request_timeout", 600)),
            "scan_mode": _scan,
            "single_prompt": _sp,
        }

    def save_settings(self):
        self.cfg = self.collect_cfg()
        save_config(self.cfg)
        messagebox.showinfo("保存完了", f"設定を保存しました。\n{CONFIG_PATH}")

    def reset_settings(self):
        for k, var in [("model_path", self.var_model), ("output_dir", self.var_output),
                       ("temp_dir", self.var_temp), ("pdf_dpi", self.var_dpi),
                       ("base_size", self.var_base), ("image_size", self.var_img),
                       ("scan_mode", self.var_scan), ("enable_rescan", self.var_rescan),
                       ("min_card_width", self.var_minw), ("num_workers", self.var_nw),
                       ("attn_impl", self.var_attn), ("fast_math", self.var_fast)]:
            var.set(DEFAULT_CONFIG[k])

    def env_check(self):
        cfg = self.collect_cfg()
        self.txt_check.config(state="normal")
        self.txt_check.delete("1.0", "end")
        self.txt_check.insert("end", "環境チェック実行中...\n\n")
        self.txt_check.config(state="disabled")
        self.root.update()

        def _do():
            results = run_env_check(cfg)
            lines, ng = [], 0
            for item, status, detail in results:
                if status != "OK":
                    ng += 1
                lines.append(f" {'✓' if status=='OK' else '✗'} [{status}] {item:14s} : {detail}")
            lines.append("")
            lines.append("→ すべてOKです。このマシンで実行できます。" if ng == 0
                         else f"→ {ng}件のNGがあります。上記の指示に従って解決してください。")
            text = "\n".join(lines)
            def _show():
                self.txt_check.config(state="normal")
                self.txt_check.delete("1.0", "end")
                self.txt_check.insert("end", text)
                self.txt_check.config(state="disabled")
            self.root.after(0, _show)
        threading.Thread(target=_do, daemon=True).start()

    # ── OCR処理 ──
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="在留カードPDFを選択（複数選択可）",
            filetypes=[("PDFファイル", "*.pdf")])
        if files:
            self.pdf_files = list(files)
            self.listbox.delete(0, "end")
            for f in self.pdf_files:
                self.listbox.insert("end", os.path.basename(f))
            self.lbl_count.config(text=f"選択ファイル: {len(self.pdf_files)}件")

    def clear_files(self):
        self.pdf_files = []
        self.listbox.delete(0, "end")
        self.lbl_count.config(text="選択ファイル: 0件")

    def log(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        try:
            self._log_lines.append(line)
        except AttributeError:
            self._log_lines = [line]
        def _do():
            self.txt.config(state="normal")
            self.txt.insert("end", line + "\n")
            self.txt.see("end")
            self.txt.config(state="disabled")
        self.root.after(0, _do)

    def start(self):
        if self.running:
            return
        if not self.pdf_files:
            messagebox.showwarning("警告", "PDFファイルを選択してください。")
            return
        self.cfg = self.collect_cfg()
        save_config(self.cfg)
        self._log_lines = []
        _eng = getattr(self, "engine", None)
        if _eng is not None:
            for _a in ("_tok_completion", "_tok_prompt", "_tok_calls"):
                if hasattr(_eng, _a):
                    setattr(_eng, _a, 0)
        self.running = True
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        cfg = self.cfg
        try:
            os.makedirs(cfg["output_dir"], exist_ok=True)
            os.makedirs(cfg["temp_dir"], exist_ok=True)

            self.log("PDFを画像に変換中...")
            all_images = []
            for pdf in self.pdf_files:
                imgs = pdf_to_images(pdf, cfg["temp_dir"], cfg["pdf_dpi"])
                self.log(f"  {os.path.basename(pdf)} → {len(imgs)}ページ")
                all_images.extend(imgs)

            total = len(all_images)
            if total == 0:
                self.log("変換できるページがありません。")
                return
            self.root.after(0, lambda: self.progress.config(maximum=total, value=0))

            # 実行結果フォルダを先に作成し、切片もこの中に保存する（出力直下に散らさない）
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(cfg["output_dir"], f"OCR結果_{run_ts}")
            os.makedirs(run_dir, exist_ok=True)
            cfg["run_dir"] = run_dir

            rows_map, times, review, raw_map = {}, [], [], {}
            names = {i: os.path.basename(img)
                     for i, img in enumerate(all_images, 1)}

            def record(idx, fields, elapsed, raw=None):
                times.append(elapsed)
                rows_map[idx] = fields
                if raw is not None:
                    raw_map[idx] = raw
                for k in suspicious_fields(fields):
                    if k == "在留カード番号" and fields.get(k) and fields.get("_番号_uncertain"):
                        if fields.get("_番号_src") == "mismatch":
                            reason = "番号が二重読みで不一致（誤読の疑い・要目視）"
                        else:
                            reason = "番号が領域OCRのみ・主スキャン未確認（要目視）"
                    elif k == "在留カード番号":
                        reason = "番号形式不一致（英2+数8+英2）"
                    elif k == "国籍":
                        reason = "国名辞書外"
                    elif k == "住所" and fields.get(k) and fields.get("_addr_uncertain"):
                        reason = "住所がパス間で不一致（誤読の疑い・要目視）"
                    elif k == "住所" and fields.get(k):
                        reason = "住所が不完全（丁目/番地/号なし）"
                    else:
                        reason = "空欄"
                    review.append([idx, names[idx], k, fields.get(k, ""), reason])
                self.log(f"  ✓ ({idx}) {names[idx]} 完了 ({elapsed:.1f}秒)"
                         f"  番号={fields.get('在留カード番号','-')}"
                         f"  国籍={fields.get('国籍','-')}")
                done = len(rows_map)
                self.root.after(0, lambda v=done: self.progress.config(value=v))

            use_parallel = int(cfg.get("num_workers", 1)) >= 2 and total >= 2
            if use_parallel:
                # ── 並列モード: 2プロセスが同一GPUで同時推論 ──
                tasks = [(idx, img,
                          os.path.join(cfg["temp_dir"], f"work_{idx}"), cfg)
                         for idx, img in enumerate(all_images, 1)]
                try:
                    pool = self._get_pool(cfg)
                    t_batch = time.time()
                    for idx, fields, dt, err, wlogs, raw in pool.imap_unordered(
                            _worker_task, tasks):
                        if err:
                            self.log(f"  ✗ ({idx}) {names[idx]} エラー: {err}")
                        record(idx, fields, dt, raw)
                    self.log(f"並列バッチ所要: {time.time()-t_batch:.1f}秒"
                             f"（実効 {(time.time()-t_batch)/total:.1f}秒/枚）")
                except Exception as e:
                    self.log(f"並列処理エラー: {e}")
                    self.log("⚠ VRAM不足の可能性があります。環境設定タブで"
                             "並列ワーカー数を1に変更して再実行してください。")
                    raise
            else:
                # ── 逐次モード ──
                self.engine.load(cfg["model_path"], cfg.get("attn_impl", "eager"),
                                 cfg.get("fast_math", True),
                                 cfg.get("anti_repeat", True),
                                 int(cfg.get("ngram_size", 30)),
                                 int(cfg.get("max_new_tokens", 1024)))
                for idx, img in enumerate(all_images, 1):
                    self.log(f"({idx}/{total}) 認識中: {names[idx]}")
                    t0 = time.time()
                    tok0 = (getattr(self.engine, "_tok_completion", 0),
                            getattr(self.engine, "_tok_prompt", 0),
                            getattr(self.engine, "_tok_calls", 0))
                    work = os.path.join(cfg["temp_dir"], f"work_{idx}")
                    cuda_dead = False
                    try:
                        fields = self.engine.infer_card(img, work, cfg)
                        raw = dict(getattr(self.engine, "_last_raw", {}) or {})
                        raw["_tokens"] = {
                            "completion": getattr(self.engine, "_tok_completion", 0) - tok0[0],
                            "prompt": getattr(self.engine, "_tok_prompt", 0) - tok0[1],
                            "calls": getattr(self.engine, "_tok_calls", 0) - tok0[2],
                        }
                    except Exception as e:
                        self.log(f"  ✗ エラー: {e}")
                        fields = {h: "" for h in CSV_HEADERS}
                        raw = {}
                        if "CUDA" in str(e) or "device-side" in str(e):
                            cuda_dead = True
                    record(idx, fields, time.time() - t0, raw)
                    if cuda_dead:
                        # CUDAコンテキストが壊れると以降は全て失敗する。残り全部を
                        # 空で「完了」させるより、ここで中断して原因を明示する。
                        self.log("  ⚠ CUDAエラーが発生しました。CUDAコンテキストが"
                                 "破損したため、残りの処理を中断します。")
                        self.log("  → アプリを再起動し、もう一度実行してください。"
                                 "繰り返す場合はそのカード画像を確認してください。")
                        self.engine.model = None
                        break

            def _gakuseki(i):
                s = os.path.splitext(names.get(i, ""))[0]
                return re.sub(r'_p\d+$', '', s)          # 22TE492_p1 → 22TE492
            rows = [[_gakuseki(i)] + [rows_map[i].get(h, "") for h in CSV_HEADERS]
                    for i in sorted(rows_map)]
            review.sort(key=lambda r: r[0])

            ts = run_ts                                   # 先に作成済みの実行フォルダを再利用
            # ── 実行フォルダに3つのCSVと生OCR・切片をまとめて出力（切片は処理中に作成済み）──

            csv_path = os.path.join(run_dir, f"在留カード認識結果_{ts}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
                w = csv.writer(fp)
                w.writerow(["学籍番号"] + CSV_HEADERS)
                w.writerows(rows)
            self.log(f"CSV出力完了 → {csv_path}")

            stat_path = os.path.join(run_dir, f"処理統計_{ts}.csv")
            with open(stat_path, "w", newline="", encoding="utf-8-sig") as fp:
                w = csv.writer(fp)
                w.writerow(["項目", "値"])
                w.writerow(["対象枚数", total])
                w.writerow(["平均処理時間（秒）", f"{sum(times)/len(times):.2f}"])
                w.writerow(["最短処理時間（秒）", f"{min(times):.2f}"])
                w.writerow(["最長処理時間（秒）", f"{max(times):.2f}"])
            self.log(f"統計出力完了 → {stat_path}")

            # 要確認リスト出力（人手確認で100%運用を支援）
            rev_msg = "要確認項目なし"
            if review:
                rev_path = os.path.join(run_dir, f"要確認リスト_{ts}.csv")
                with open(rev_path, "w", newline="", encoding="utf-8-sig") as fp:
                    w = csv.writer(fp)
                    w.writerow(["行番号", "ファイル", "項目", "抽出値", "理由"])
                    w.writerows(review)
                rev_msg = f"要確認 {len(review)}項目 → {rev_path}"
                self.log(f"要確認リスト出力 → {rev_path}")

            # ── 生OCRダンプ: モデルが各カードで実際に読めた文字を保存 ──
            #    どこで誤ったか（OCR未読取 vs 抽出失敗）を即座に切り分けられる。
            raw_dir = os.path.join(run_dir, "生OCR")
            os.makedirs(raw_dir, exist_ok=True)
            dumped = 0
            for idx in sorted(rows_map):
                raw = raw_map.get(idx, {})
                if not raw:
                    continue
                fields = rows_map[idx]
                base = os.path.splitext(names[idx])[0]
                txt_path = os.path.join(raw_dir, f"{idx:03d}_{base}.txt")
                lines = [
                    f"ファイル: {names[idx]}",
                    f"行番号: {idx}",
                    "抽出結果: " + " | ".join(
                        f"{h}={fields.get(h,'') or '(空)'}" for h in CSV_HEADERS),
                    "=" * 60,
                    "【Markdownモード生出力】", raw.get("markdown", "") or "(なし)",
                    "=" * 60,
                    "【Free OCRモード生出力】", raw.get("free", "") or "(実行なし)",
                ]
                if raw.get("rescan"):
                    lines += ["=" * 60,
                              "【ヘッダー局部再スキャン生出力】", raw["rescan"]]
                if raw.get("footer"):
                    lines += ["=" * 60,
                              "【フッター局部再スキャン生出力】", raw["footer"]]
                tk_ = raw.get("_tokens") or {}
                if tk_:
                    lines += ["=" * 60,
                              "【トークン使用量】 "
                              f"生成={tk_.get('completion','?')} / "
                              f"プロンプト={tk_.get('prompt','?')} / "
                              f"API呼出={tk_.get('calls','?')}回"]
                try:
                    with open(txt_path, "w", encoding="utf-8") as fp:
                        fp.write("\n".join(lines))
                    dumped += 1
                except Exception as e:
                    self.log(f"  生OCR保存失敗({names[idx]}): {e}")
            self.log(f"生OCRダンプ出力完了 → {raw_dir}（{dumped}件）")

            self.log("=" * 50)
            self.log(f"全処理完了！ 対象枚数: {total}枚 / "
                     f"平均: {sum(times)/len(times):.1f}秒 / "
                     f"最短: {min(times):.1f}秒 / 最長: {max(times):.1f}秒")
            self.log(f"出力フォルダ: {run_dir}")
            self.log(f"検証結果: {rev_msg}")

            # ── 診断ログ: 設定・サーバflag・トークン・GUIログを1ファイルに集約 ──
            try:
                diag_path = os.path.join(run_dir, f"診断ログ_{ts}.txt")
                eng = self.engine
                tot_c = getattr(eng, "_tok_completion", 0)
                tot_p = getattr(eng, "_tok_prompt", 0)
                tot_n = getattr(eng, "_tok_calls", 0)
                dlines = [
                    "================ 実行診断ログ ================",
                    f"実行時刻: {ts}",
                    f"対象枚数: {total}",
                    f"サーバURL: {cfg.get('server_url','')}",
                    f"モデル: {cfg.get('served_model','')}",
                    f"スキャンモード: {cfg.get('scan_mode','')}",
                    "",
                    "---- リクエストflag（GUI→サーバ） ----",
                    f"main_max_tokens: {cfg.get('main_max_tokens', cfg.get('max_new_tokens',1024))}",
                    f"region_max_tokens: {cfg.get('region_max_tokens', 96)}",
                    f"max_new_tokens: {cfg.get('max_new_tokens',1024)}",
                    f"ngram_size: {cfg.get('ngram_size',30)}",
                    f"temperature: 0.0  / skip_special_tokens: False",
                    f"base_size: {cfg.get('base_size','')} / image_size: {cfg.get('image_size','')}",
                    f"pdf_dpi: {cfg.get('pdf_dpi','')} / 最小有効幅: {cfg.get('min_card_width','')}",
                    f"並列ワーカー: {cfg.get('num_workers',1)} / attn: {cfg.get('attn_impl','')}",
                    "",
                    "---- トークン使用量（サーバ応答usageの合計） ----",
                    f"生成トークン合計: {tot_c}",
                    f"プロンプトトークン合計: {tot_p}",
                    f"API呼出回数: {tot_n}",
                    (f"1枚あたり平均生成: {tot_c/total:.0f} tok" if total else ""),
                    "",
                    "---- 1枚ごと（時間・トークン・空欄項目） ----",
                ]
                for i in sorted(rows_map):
                    fl = rows_map[i]
                    tkn = (raw_map.get(i, {}) or {}).get("_tokens", {}) or {}
                    empties = [h for h in CSV_HEADERS if not fl.get(h, "").strip()]
                    dlines.append(
                        f"  {i:3d} {names[i]:18s} "
                        f"生成={tkn.get('completion','?'):>4} 呼出={tkn.get('calls','?')} "
                        f"空欄={','.join(empties) if empties else 'なし'}")
                dlines += ["", "================ GUIログ全文 ================"]
                dlines += getattr(self, "_log_lines", [])
                with open(diag_path, "w", encoding="utf-8") as fp:
                    fp.write("\n".join(str(x) for x in dlines))
                self.log(f"診断ログ出力 → {diag_path}")
            except Exception as e:
                self.log(f"  診断ログ保存失敗: {e}")
            messagebox.showinfo("完了",
                f"処理が完了しました。\n\n出力フォルダ:\n{run_dir}\n\n"
                f"・在留カード認識結果_{ts}.csv\n"
                f"・処理統計_{ts}.csv\n"
                f"・要確認リスト_{ts}.csv\n"
                f"・生OCR\\（各カードの生スキャン）\n\n{rev_msg}")
        except Exception as e:
            self.log(f"致命的エラー: {e}")
            messagebox.showerror("エラー", str(e))
        finally:
            self.running = False
            self.root.after(0, lambda: self.btn_run.config(state="normal"))


if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()
