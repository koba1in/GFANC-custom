# GFANC-custom

Generative Fixed-filter Active Noise Control（GFANC）を扱う Python プロジェクトです。このリポジトリは、公開リポジトリ [Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control](https://github.com/Luo-Zhengding/GFANC-Generative-fixed-filter-active-noise-control) のコードをもとに、ファイル構成や Python モジュールの整理などを加えた派生版です。上流プロジェクトの説明・研究成果と、このリポジトリで行った変更を区別して参照してください。

## 上流プロジェクトと研究

上流プロジェクトは、入力ノイズに応じて 1D CNN がサブ制御フィルタの重みを決め、それらを組み合わせて制御フィルタを生成する GFANC 手法を実装しています。詳細は論文 [Deep Generative Fixed-Filter Active Noise Control](https://arxiv.org/abs/2303.05788) および上流リポジトリを参照してください。手法、論文、上流コードの著作権やライセンスはそれぞれの権利者に帰属します。

## このリポジトリの構成

- `src/`: 再利用する Python コードをパッケージとして整理しています。データ・ラベル生成は `data/`、学習と評価は `training/`、GFANC は `gfanc/`、FxLMS は `fxlms/`、SFANC と FxNLMS の連携は `sfanc_fxlms/`、共通処理は `common/` にあります。
- `Hard_Label/1.Labeling_Dataset_Code/`: データセットやハードラベルを生成する実験ノートブックです。
- `Hard_Label/2.Training_1DNetwork_Code/`: 1D CNN の学習・評価ノートブックです。
- `Hard_Label/3.Noise_Cancellation_Code/`: GFANC、FxLMS などによるノイズキャンセリングの実験ノートブックです。
- `Hard_Label/Temp/`: 試作・一時保管のノートブックです。現時点では動作確認が済んでいないため、実行可能な手順として扱わないでください。
- `models/`: リポジトリに含めている学習済みモデル。
- `Pz and Sz/`, `Real Noise Examples/`: 音響パスと実験用の実ノイズ音源。
- `pyproject.toml`, `requirements.txt`: Python プロジェクト設定と依存パッケージ。

`src/README.md` に Python モジュールの配置とノートブックからの読み込み方法を記載しています。元のスクリプト群を `src/` に整理し、重複コードを統合したほか、実装差があるものは `_v1`、`_v2` などの名前で区別しています。

## 環境とセットアップ

`pyproject.toml` では Python 3.9 以上を指定しています。依存パッケージはプロジェクト設定または `requirements.txt` からインストールできます。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
```

ノートブックを実行する場合は、Jupyter 対応環境でリポジトリ内の `Hard_Label/` にあるノートブックを開いてください。`Temp/` 内のノートブックは未確認です。Python の依存関係を入れた後も、PyTorch の CUDA 利用可否は OS、GPU、CUDA ドライバーに応じて別途確認してください。

## データと実行時の注意

リポジトリには一部のモデル、音響パス、実験用音源が含まれます。上流で案内されている合成データセットや、各ノートブックが生成する中間ファイルは、すべてがこのリポジトリに含まれるとは限りません。ノートブック内の入力パスや必要ファイルを確認してから実行してください。

## 関連研究

- [Delayless Generative Fixed-filter Active Noise Control based on Deep Learning and Bayesian Filter](https://ieeexplore.ieee.org/document/10339836/)
- [GFANC-Kalman: Generative Fixed-Filter Active Noise Control with CNN-Kalman Filtering](https://ieeexplore.ieee.org/document/10323505)
- [A hybrid SFANC-FxNLMS algorithm for active noise control based on deep learning](https://arxiv.org/abs/2208.08082)
- [Performance Evaluation of Selective Fixed-filter Active Noise Control based on Different Convolutional Neural Networks](https://arxiv.org/abs/2208.08440)
