# 在留カードOCR / Residence Card OCR

視覚言語モデル **DeepSeek-OCR**（[arXiv:2510.18234](https://arxiv.org/abs/2510.18234)）と **vLLM** を用いた、在留カードの記載項目自動抽出システム。
An automated field-extraction system for Japanese Residence Cards (Zairyu Card), built on **DeepSeek-OCR** served via **vLLM**, with a Windows GUI client.

スキャン画像（PDF / 画像）から **9項目**（在留カード番号・氏名・生年月日・性別・国籍・住所・在留資格・在留期間・許可年月日）を読み取り、CSVに出力します。

---

## 特長 / Features

- **ローカル完結**：DeepSeek-OCR を vLLM 推論サーバとして自分の GPU 上で稼働。外部APIに画像を送りません。
- **カード正規化**：OpenCV でカードの傾き・余白を補正し、規格化フレームへ射影変換 → 安定した項目切り出し。
- **2段階OCR**：カード全体の本スキャンで大半を取得し、落とした項目（番号・性別・国籍など）だけ領域を切り出して再認識。
- **後処理正規化**：英語ラベル除去／簡体字→日本語漢字／全角半角統一／国名辞書補正。
- **要確認キュー**：番号形式不一致・住所途切れ・国籍辞書外などを自動で一覧化し、人手確認を効率化。
- **診断ログ出力**：各項目の切片画像・生OCR出力・トークン使用量・処理時間を実行フォルダにまとめて保存。
- **GUI**：Tkinter 製。ワンクリック起動バッチ、サーバ停止/再起動ボタン付き。

---

## 構成 / Architecture

```
┌─────────────────────────────┐        HTTP (OpenAI互換API)        ┌──────────────────────────┐
│   Windows GUI クライアント       │  ───────────────────────────────▶ │   vLLM 推論サーバ (WSL2)        │
│   ZairyuCardOCR_GUI.py        │                                    │   DeepSeek-OCR             │
│                               │ ◀─────────────────────────────── │   (RTX GPU)               │
│  PDF→画像 → カード正規化(OpenCV)  │          認識テキスト                └──────────────────────────┘
│   → 本スキャン → 領域補完          │
│   → 項目抽出 + 正規化             │
│   → CSV / 要確認 / 診断ログ       │
└─────────────────────────────┘
```

処理フロー / Pipeline:

1. **PDF→画像** 変換（PyMuPDF, 300 dpi）
2. **カード正規化**（OpenCV：輪郭検出 → 傾き補正 → 規格化フレームへ射影変換）
3. **本スキャン**（カード全体を1回 Free OCR）
4. **領域補完**（不足項目のみ該当領域を切り出して再認識）
5. **項目抽出 + 正規化**（ラベル除去・字体統一・全半角統一・辞書補正）
6. **出力**（CSV ＋ 要確認リスト ＋ 切片画像 ＋ 生OCR ＋ 診断ログ）

---

## 動作環境 / Environment

| 項目 | 内容 |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu) |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Python | 3.12 |
| 推論基盤 | vLLM (PagedAttention) |
| モデル | DeepSeek-OCR (DeepEncoder + DeepSeek3B-MoE-A570M) |
| 入力解像度 | **300 dpi** |
| 視覚解像度 | base_size = 1024 / image_size = 640 |
| 生成上限 | 本スキャン 1024 / 領域補完 96 tokens |
| 復号設定 | temperature = 0.0 / n-gram block = 30 / workers = 1 |

> 注 / Note: RTX 50シリーズ（Blackwell, sm_120）では CUDA / PyTorch の対応版（cu128 以降）が必要です。
> RTX 50-series (Blackwell) requires a CUDA-matched PyTorch build (cu128+).

---

## セットアップ / Setup

### 1. vLLM サーバ（WSL2 / Ubuntu）

```bash
conda create -n dsocr python=3.12 -y
conda activate dsocr
pip install vllm   # GPUに合うCUDAビルドを使用 (sm_120はcu128+)
```

`~/start_ocr_server.sh`（同梱）でサーバを起動:

```bash
bash ~/start_ocr_server.sh
# 内部で実行されるコマンド例:
# vllm serve deepseek-ai/DeepSeek-OCR \
#   --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
#   --no-enable-prefix-caching --mm-processor-cache-gb 0 --no-async-scheduling
```

起動完了まで 1〜2 分（モデルロード）。`http://localhost:8000/v1` で待受します。

### 2. GUI クライアント（Windows）

```bash
pip install pillow opencv-python pymupdf   # Tkinterは標準同梱
python ZairyuCardOCR_GUI.py
```

または同梱の **`起動.bat`**（サーバ自動起動＋GUI起動）/ **`GUIのみ起動.bat`**（サーバはそのまま、GUIだけ再起動）をダブルクリック。

---

## 使い方 / Usage

1. GUI で在留カードの **PDF / 画像** を選択
2. 「OCR処理を開始」
3. 出力フォルダ `OCR結果_<日時>/` に以下が生成されます:

```
OCR結果_<日時>/
├── 在留カード認識結果_<日時>.csv     ← 9項目の抽出結果（学籍番号列付き）
├── 要確認リスト_<日時>.csv          ← 信頼度の低い項目（人手確認用）
├── 処理統計_<日時>.csv             ← 枚数・平均/最短/最長処理時間
├── 診断ログ_<日時>.txt             ← 設定・サーバflag・トークン・GUIログ
├── 切片/                          ← 各項目の切り出し画像
└── 生OCR/                         ← カードごとのモデル生出力
```

---

## 評価結果 / Evaluation

スキャン画像 **83枚**（合計 747項目）での項目別正解率（有効性基準による評価）:

| 項目 | 生OCR | 正規化後 | 改善幅 |
|---|---:|---:|---:|
| 在留カード番号 | 95.2% | 96.4% | +1.2 |
| 氏名 | 98.8% | **98.8%** | 0.0 |
| 生年月日 | 97.6% | **98.8%** | +1.2 |
| 性別 | 94.0% | 96.4% | +2.4 |
| 国籍 | 84.3% | 94.0% | **+9.7** |
| 住所 | 94.0% | 94.0% | 0.0 |
| 在留資格 | 97.6% | 97.6% | 0.0 |
| 在留期間 | 95.2% | **98.8%** | +3.6 |
| 許可年月日 | 90.4% | 90.4% | 0.0 |
| **全項目平均** | **94.1%** | **96.1%** | **+2.0** |

- 処理時間：平均 **6.9 秒/枚**（最短 5.4 秒、最長 16.9 秒）
- 住所 CER（簡易）：平均 **0.060**（有効抽出 78/83 枚）

> 評価は出力の妥当性（番号形式・国名辞書・日付形式・住所完全性など）に基づく簡易評価です。内容の最終確認には原本との照合を推奨します。

---

## 限界 / Limitations

- **文字レベルの誤読**（番号の英字誤読・国籍カナ誤読など）はモデル側の限界で、後処理では補正不可。
- **認識の非決定性**：同一画像でも実行ごとに結果がわずかに変動することがあります（GPU並列演算に起因）。
- **想定用途**：実運用では「大多数を自動確定 ＋ 要確認の少数を人手確認」の運用を推奨します。100%全自動を保証するものではありません。

---

## ライセンス / License

本リポジトリのコードは学習・研究目的で公開しています。利用するモデル（DeepSeek-OCR）・ライブラリ（vLLM 等）はそれぞれのライセンスに従ってください。
Code in this repository is released for research/educational use. The underlying model (DeepSeek-OCR) and libraries (vLLM, etc.) are subject to their own licenses.

> 在留カードは個人情報を含みます。実データの取り扱いには十分注意し、適切な管理下で使用してください。
> Residence cards contain personal data — handle real data responsibly and under appropriate safeguards.

---

## 参考 / References

- Wei, H., Sun, Y., Li, Y. *DeepSeek-OCR: Contexts Optical Compression.* arXiv:2510.18234 (2025).
- Kwon, W., et al. *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP '23 (2023). arXiv:2309.06180.
