# GFANC-custom

Generative Fixed-filter Active Noise Control（GFANC）を扱う Python 研究プロジェクトです。公開リポジトリ [Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control](https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control) のコードをもとに、Python モジュールを `src/gfanc/` に整理し、ノートブックから再利用できる形にした派生版です。

## 研究背景

GFANC は入力ノイズに応じてニューラルネットワークがサブ制御フィルタを選択・合成し、アクティブノイズ制御を行う手法です。手法の詳細と評価については、論文 [Deep Generative Fixed-Filter Active Noise Control](https://arxiv.org/abs/2303.05788) と [上流リポジトリ](https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control) を参照してください。

## ディレクトリ構成

```text
.
├── models/                         # 同梱済みモデル
├── notebooks/
│   ├── 1.Labeling_Dataset_Code/    # データ生成・ラベル作成
│   ├── 2.Training_1DNetwork_Code/  # 1D CNN の学習・評価
│   ├── 3.Noise_Cancellation_Code/  # GFANC / FxLMS の実験
│   └── Temp/                       # 試作・一時的なノートブック
├── Pz and Sz/                      # 音響経路データ
├── Real Noise Examples/            # 実ノイズ音源
├── src/gfanc/
│   ├── common/                     # 共通処理・音源や経路の読み込み
│   ├── data/                       # データセット、ラベル、フィルタ生成
│   ├── fxlms/                      # FxLMS アルゴリズム
│   ├── gfanc/                      # GFANC のモデル・制御処理
│   ├── sfanc_fxlms/                # SFANC と FxNLMS の処理
│   └── training/                   # データローダー、学習・評価
├── tests/                          # パッケージの import 確認
└── pyproject.toml                  # ビルド、依存関係、プロジェクト設定
```

ノートブックの入口は次のとおりです。

| 作業 | ノートブック |
|---|---|
| データ生成・ラベル作成 | `notebooks/1.Labeling_Dataset_Code/` |
| ネットワーク学習・評価 | `notebooks/2.Training_1DNetwork_Code/Train_Testing_M5_Network.ipynb` |
| ノイズキャンセリング実験 | `notebooks/3.Noise_Cancellation_Code/` |
| 試作・一時コード | `notebooks/Temp/` |

`notebooks/Temp/` は実験途中のノートブックを含むため、再現可能な実行手順としては扱わないでください。各ノートブックに記録済みの出力が含まれていても、現在の環境での実行成功を保証するものではありません。

詳しいモジュール配置とノートブックからの読み込みについては [`src/README.md`](src/README.md) を参照してください。

## 必要環境

- Python: `pyproject.toml` では Python 3.9 以上
- 依存関係とバージョン: `pyproject.toml` の optional dependency グループで管理
- ビルドバックエンド: `uv_build`

## 必要環境とインストール

`pyproject.toml` では、パッケージ名を `gfanc`、バージョンを `0.1.0`、Python の下限を `3.9` としています。ビルドには PEP 517 の `uv_build`（`>=0.12.19,<0.13.0`）を使います。必須依存関係は `[project.dependencies]` には定義されておらず、依存パッケージは `[project.optional-dependencies]` の `normal`、`titanx`、`dev` extras に分けて定義されています。メタデータ上の開発段階は Alpha です。`[project.urls]` テーブルはありますが、現在 URL は登録されていません。

### uv を使う場合

`uv` をインストールしたうえで、プロジェクトのルートで実行します。プロジェクト本体をeditable モードでインストールします。

```bash
# 通常の実行環境
uv pip install -e ".[normal]"

# Titan X 向けの依存セットを使う場合は、normal の代わりに選択
uv pip install -e ".[titanx]"

# 開発ツールも追加する場合（pytest、ruff）
uv pip install -e ".[normal]" ".[dev]"
```

`normal` と `titanx` は、それぞれ異なる PyTorch / torchaudio のバージョンを指定する環境セットです。通常はどちらか一方を選んでください。`dev` も optional extra として定義されているため、使う場合は `--extra dev` を明示します。

### pip を使う場合

Python 3.9 以上の仮想環境を有効にし、リポジトリのルートでインストールします。

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell の場合: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[normal]"
```

pip は `pyproject.toml` の build-system 設定に従って `uv_build` をビルド環境に用意します。Titan X 向けまたは開発用 extras は、最後のコマンドを `-e ".[titanx]"`、`-e ".[normal,dev]"` のように必要なセットへ変更してください。

`[tool.setuptools.packages.find]` も `pyproject.toml` にありますが、現在指定されているビルドバックエンドは `uv_build` です。その setuptools 用設定は `uv_build` のパッケージ検出を構成するものではありません。プロジェクトのパッケージ構造は `src/deepanc/` にあります。

GPU を使う場合、PyTorch / torchaudio の CUDA 対応状況は OS、Python、CUDA 環境によって異なります。必要に応じて環境に合う PyTorch の配布元・ビルドを確認してください。

## ノートブックの実行

1. 上記のいずれかの依存関係セットをインストールします。
2. Jupyter 環境に `gfanc` 用の Python 環境をカーネルとして登録・選択します。
3. リポジトリ内の `notebooks/` から実行したいノートブックを開きます。
4. 必要に応じて、ノートブック内のデータセットやモデルのパス、実験条件を設定します。

データ生成や学習を行うノートブックでは、リポジトリに含まれない合成データセットや生成物を参照する場合があります。実行前にノートブック内のパス設定と入力ファイルを確認してください。

## データ・モデル

リポジトリには一部のモデル、音響経路データ、実ノイズ音源が含まれています。合成データセット、学習出力、図表などの実験生成物はすべて同梱されているわけではありません。`models/` にあるチェックポイントと必要なデータが、各ノートブックの設定で指定された名前・形式と一致することを確認してください。

## ライセンスと引用

コード・モデル・データを再利用または再配布する際は、上流プロジェクトおよび各ファイルの権利条件を確認してください。上流研究の引用情報は論文ページを参照してください。

## 関連研究

- [Delayless Generative Fixed-filter Active Noise Control based on Deep Learning and Bayesian Filter](https://ieeexplore.ieee.org/document/10339836/)
- [GFANC-Kalman: Generative Fixed-Filter Active Noise Control with CNN-Kalman Filtering](https://ieeexplore.ieee.org/document/10323505)
- [A hybrid SFANC-FxNLMS algorithm for active noise control based on deep learning](https://arxiv.org/abs/2208.08082)
- [Performance Evaluation of Selective Fixed-filter Active Noise Control based on Different Convolutional Neural Networks](https://arxiv.org/abs/2208.08440)
