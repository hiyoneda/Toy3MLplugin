# Toy3MLPlugin -- threeMLを用いた天体同時解析のデモンストレーション

**SMILE-3 や NinjaSat-2 などの全天サーベイ型の高エネルギー天体解析に共通する
「型」を抜き出したデモンストレーションコードです。
シンプルな**1次元スキャン**のデータ形式(漏れ込みで少し太ったライトカーブを、点源として解析する)で、
以下が一式動きます。

ここでデモンストレーションしていること

| # | 内容 | どこで | 実行 |
|---|---|---|---|
| 0 | **レスポンス生成を明示**(軌道+装置固有応答→畳み込み) | `Toy3MLPlugin/instrument.py` | `examples/01_plugin_and_fitting.ipynb` |
| 1 | **3ML プラグインの書き方** | `Toy3MLPlugin/base.py`, `Toy3MLPlugin/toy/scanner.py` | `examples/01_plugin_and_fitting.ipynb` |
| 2 | **複数天体の同時フィッティング** | `catalog.py`(YAML→Model), クロージャ | `examples/01_plugin_and_fitting.ipynb` |
| 3 | **複数 dataset を束ねられる** | `ToyScanLike` × 複数軌道 + `DataList` | `examples/02_multi_dataset.ipynb` |
| 4 | **TS マップ(localization)の理解** | `Toy3MLPlugin/tsmap.py` | `examples/03_ts_map.ipynb` |
| 5 | **cosipy 流 deconvolution**(cosipy を使う。DataIF だけ用意) | `Toy3MLPlugin/cosipy_dataif.py` | `examples/04_cosipy_deconvolution.ipynb` |
| 6 | **MCMC で事後分布(triangle plot)** | 3ML `BayesianAnalysis` | `examples/05_mcmc_posterior.ipynb` |

`toy/scanner.py` は「答え」ではなく、埋めるべき2メソッドの**お手本**です。
自分の装置は、データ空間の軸とレスポンスの中身だけ差し替えれば同じ骨格に乗ります。

### エネルギー軸について

エネルギーは**軸としては用意**してありますが(`energy_edges`)、混乱を避けるため
デモは基本 **1要素に潰して**(`energy_edges=[2,10]`, `n_e=1`)動かしています。
多ビンに増やしても骨格は不変(`tests/test_response.py` は 3 ビンで軸が生きることを確認)。

### レスポンス生成を明示的に分ける

解析の前段(`Toy3MLPlugin/instrument.py`)で、レスポンスを次のように組み立てます:

    衛星軌道(pointing履歴)  +  装置固有の応答 R0(theta, e)
        ── fold_response ──►  畳み込み済みレスポンス R[time, sky, e]  → ToyScanLike に入れる

- 装置固有応答 `intrinsic_response(theta, e)` は pointing にも軌道にもよらない装置の性質。
- 軌道は `make_scan_orbit` で作り、`save_orbit_fits` / `load_orbit_fits` で **FITS** に保存/読込(デモ)。
- `fold_response(orbit, R0, ...)` が両者を畳み込み、前方作用素 `R[time, sky, e]` を返す。

> NOTE: いまは R0 / R を numpy or astropy(単位つき) の array で扱います。
> FITS なり HDF5 でレスポンスを保存・配布するのが実用上はよいでしょう。その差し替え口が
> `intrinsic_response` / `fold_response` です。

### cosipy を使った画像再構成

deconvolution 本体(RL/MAP-RL)は **cosipy をそのまま使います**。この教材で用意するのは
cosipy に差し込む **データインタフェース(DataIF)** のトイ実装1つだけ
(`Toy3MLPlugin/cosipy_dataif.py` の `DataIF_ToyScan`)。これは cosipy の
`ImageDeconvolutionDataInterfaceBase` を継承し、実装すべきは5つの抽象メソッドだけ:

    calc_source_expectation(model)  # R·model   (前方: モデル空間 → データ空間)
    calc_bkg_expectation(bkg_norm)  # 背景の期待カウント
    calc_T_product(H)               # R^T·H      (随伴: データ空間 → モデル空間)
    calc_bkg_model_product(key, H)  # Σ_i B_i H_i
    calc_log_likelihood(expectation)

`examples/04_cosipy_deconvolution.ipynb` は、このトイ DataIF を作って cosipy の正式な入口
`ImageDeconvolution` に、**YAML パラメータファイル**(`examples/imagedeconvolution_toy.yml`、
実物の `imagedeconvolution_parfile_test.yml` と同じ構造)とともに渡し、
`set_dataset → read_parameterfile → initialize → run_deconvolution` で再構成を回します
(モデル空間は healpix 天球 × エネルギー = cosipy の `AllSkyImageModel`、データ空間はライトカーブ)。
cosipy の `DataIF_COSI_DC2` と同じ骨格で、CDS レスポンスをトイのスキャンレスポンスに置き換えただけ
── 自分の装置の DataIF を書くときの最小お手本になります。

---

## 自分の観測に応じて自分で実装すべきこと

`Toy3MLPlugin.base.ForwardFoldingLike` を継承し、以下の2つを埋めます。
尤度・多天体の重ね合わせ・背景・3ML 連携は基底がすべて面倒を見ます。

```python
class MyInstrumentLike(ForwardFoldingLike):

    def _load_response(self):
        # self._obs から、応答計算に必要な補助データを取り出して属性に持つ。
        # 例(NinjaSat): 姿勢履歴・スリット角度応答・有効面積。
        ...

    def _response_for_source(self, source) -> np.ndarray:
        # 位置固定の点源1個 → データ空間の応答行列 R。
        # 戻り値 shape は必ず (n_data_bins, n_e)。
        # R[i, e] = 「エネルギービン e の photon flux 1単位がデータビン i に落とすカウント数」。
        ...
```

これが全体を貫く1つの式です:

```
mu[data_bin] = Σ_source  R_src[data_bin, e] @ F_src[e]  +  Σ_bkg  norm · template[data_bin]
```

- `data_bin` … 装置のデータ空間を1次元に潰したベクトル。
  NinjaSat = (検出器, 時間ビン) を連結した2本のライトカーブ。SMILE = CDS × エネルギー等。
- `e` … エネルギービン。NinjaSat は `n_e=1`(1バンド)という退化ケースとして扱えます。
- `F_src[e]` … astromodels のスペクトルをエネルギービンで積分した photon flux。
- `R_src` … その位置の点源への装置応答。**位置固定なら基底が自動キャッシュ**。

## なぜソースリストを自作しないか

多天体同時フィットの枠組みは **astromodels.Model がそのまま担います**
(位置・スペクトル形・free/fixed・prior まで内蔵、3ML がそのままフィット)。
Fermi の XML に相当する薄い編集レイヤだけ `catalog.py`(YAML → Model)に用意しました。

## ファイル構成

```
Toy3MLPlugin/
  base.py           ForwardFoldingLike    ← 型の中心(3ML プラグイン基底)。
  background.py     BackgroundComponent   ← データ空間テンプレ × 自由 norm
  observation.py    Observation           ← データ空間カウント + 露出(最小コンテナ)
  catalog.py        load_sources()        ← YAML の点源リスト → astromodels.Model
  instrument.py     intrinsic_response /  ← レスポンス生成を明示(軌道FITS I/O + 畳み込み)
                    make_scan_orbit / save_orbit_fits / load_orbit_fits / fold_response
  simulate.py       simulate_counts()     ← 期待値 → Poisson 実現(クロージャ用)
  tsmap.py          ts_map_1d()           ← 1次元 TS マップ(localization)
  cosipy_dataif.py  DataIF_ToyScan        ← cosipy に差し込む DataIF のトイ実装(目標5)
  toy/scanner.py    ToyScanLike           ← 畳み込み済みレスポンスを使うプラグイン(お手本)
examples/                                  ← Jupyter notebook(実行済み・図つき)
  sources.yaml                    点源リストの書式例
  imagedeconvolution_toy.yml      cosipy ImageDeconvolution のパラメータ(目標5)
  01_plugin_and_fitting.ipynb     レスポンス生成 + 3ML プラグイン + 多天体フィット(目標0,1,2)
  02_multi_dataset.ipynb          複数軌道を DataList で同時フィット(目標3)
  03_ts_map.ipynb                 1次元 TS マップで localization(目標4)
  04_cosipy_deconvolution.ipynb   トイ DataIF を cosipy ImageDeconvolution+YAML に渡す(目標5)
  05_mcmc_posterior.ipynb         MCMC で事後分布の triangle plot(目標6)
tests/
  test_poisson.py       Poisson カーネル単体(scipy と一致)
  test_response.py      応答サニティ(transit / 位置ずれ / 多天体 / エネルギー軸)
  test_closure.py       注入-回収クロージャ(★本丸、装置を parametrize で追加可能)
  test_tsmap.py         TS マップ localization / 複数 dataset 同時フィット
  test_cosipy_dataif.py DataIF の軸・形が契約どおりか / ImageDeconvolution+YAML が回るか
```

## インストールと実行

ライブラリとして pip で入ります。

```bash
pip install .                 # または: pip install /path/to/Toy3MLPlugin
pip install ".[cosipy,dev]"   # 目標5(cosipy)と notebook/テスト実行まで含める
```

```python
import Toy3MLPlugin
# 3ML プラグイン基底、TS マップ、cosipy DataIF などがそのまま import できる
from Toy3MLPlugin import ForwardFoldingLike, ts_map_1d, build_toy_scan_dataif
```

テストと notebook:

```bash
pytest -q                                   # 13 tests, all green
jupyter lab examples/                        # 実行済み notebook を開く(図つき)
```

依存: `numpy`, `scipy`, `pyyaml`, `astromodels>=2.5`, `threeML>=2.5`(必須)。
目標5の画像再構成のみ `cosipy>=0.4`(optional extra `[cosipy]`)が必要。

## あなたの装置を載せる手順の例

1. `Observation` を継承し、装置固有の補助データを足す
   (例: `class MyObs(Observation): attitude=...`)。
2. `ForwardFoldingLike` を継承し、`_load_response` と `_response_for_source` を実装。
3. `tests/test_response.py` にならって、自分の応答のサニティテストを書く
   (「1点源を FOV に置くと期待した場所にシグナルが立つ」)。
4. `tests/test_closure.py` の `INSTRUMENTS` に自分のファクトリを1行足す。
   → 既存の注入-回収テストがそのまま自分の装置にも流れます。

> クロージャテストが緑になったら、その装置の前方畳み込みは「正しく閉じている」。
> そこからが本番(実データのレスポンス、localization、複数軌道の同時フィット…)。
