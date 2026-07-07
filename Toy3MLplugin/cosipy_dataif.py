"""
Toy3MLplugin.cosipy_dataif — cosipy の画像再構成に差し込む「データインタフェース」のトイ実装。

画像再構成そのもの(RL/MAP-RL)は **cosipy を使い、ここでは自前で作らない**。
必要なのは cosipy.image_deconvolution.ImageDeconvolutionDataInterfaceBase を継承した
DataIF を1つ用意すること ── これがあれば cosipy の RichardsonLucyBasic /
MAP_RichardsonLucy がそのまま回る。

トイ装置(スキャンするコリメータ)の 2D 版:
  * モデル空間 = healpix 天球 x エネルギー(AllSkyImageModel と同じ軸)
  * データ空間 = 時間ビン(スキャンのライトカーブ)
  * レスポンス R[pix, Ei, Time] = 各時刻の指向方向と画素の角距離に応じたコリメータ応答 x 露出
    (漏れ込みで太った視野 = ガウス幅 fov_sigma)

cosipy の DataIF_COSI_DC2 と同じ骨格(load / tensordot による expectation・T_product・
exposure_map)。DC2 の CDS レスポンスを、トイのスキャンレスポンスに置き換えただけ。

これは「自分の装置の DataIF をどう書くか」の最小お手本。実装すべきは5つの
抽象メソッド(calc_source_expectation / calc_bkg_expectation / calc_T_product /
calc_bkg_model_product / calc_log_likelihood)だけ。
詳細は、
https://github.com/cositools/cosi-data-challenges/blob/main/image_deconvolution/README.md
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
import healpy as hp
from histpy import Histogram, Axes, Axis, HealpixAxis

from cosipy.image_deconvolution import ImageDeconvolutionDataInterfaceBase


def _plain(x):
    """Quantity なら値だけ、それ以外は ndarray にして返す(ユニット起因の事故を避ける)。"""
    if isinstance(x, u.Quantity):
        return np.asarray(x.value)
    return np.asarray(x)


class DataIF_ToyScan(ImageDeconvolutionDataInterfaceBase):
    """スキャン型トイ装置の cosipy データインタフェース。"""

    def __init__(self, name=None):
        super().__init__(name)
        self._image_response = None   # histpy.Histogram, axes = (lb, Ei, Time)

    @classmethod
    def load(cls, name, event, bkg_models, response, model_axes, data_axes):
        """
        event        : histpy.Histogram (axes = data_axes = (Time,))
        bkg_models   : {name: histpy.Histogram(data_axes)}
        response     : histpy.Histogram (axes = (lb, Ei, Time))
        model_axes   : histpy.Axes  (lb, Ei)
        data_axes    : histpy.Axes  (Time,)
        """
        new = cls(name)
        new._event = event
        new._bkg_models = dict(bkg_models)
        new._summed_bkg_models = {k: float(np.sum(_plain(v.contents)))
                                  for k, v in new._bkg_models.items()}
        new._image_response = response
        new._model_axes = model_axes
        new._data_axes = data_axes

        # exposure_map = Σ_time R[pix, Ei, t]  (RL の分母 R^T·1 に相当)。DC2 に倣い pixarea を掛ける。
        expmap = np.sum(_plain(response.contents), axis=2) * model_axes['lb'].pixarea().value
        new._exposure_map = Histogram(model_axes, contents=expmap, copy_contents=False)
        return new

    # --- R·model : モデル(lb, Ei) を response の (lb, Ei) 軸と縮約 → データ空間(Time) ---
    def calc_source_expectation(self, model):
        expectation = np.tensordot(_plain(model.contents), _plain(self._image_response.contents),
                                   axes=((0, 1), (0, 1)))   # -> (n_time,)
        return Histogram(self.data_axes, contents=expectation, copy_contents=False)

    # --- 背景 : Σ norm · template(データ空間)。背景なしなら 0 配列 ---
    def calc_bkg_expectation(self, dict_bkg_norm):
        keys = self.keys_bkg_models()
        if not keys:
            expectation = np.zeros(_plain(self._event.contents).shape)
        else:
            expectation = sum(_plain(self.bkg_model(key).contents) * dict_bkg_norm[key]
                              for key in keys)
        return Histogram(self.data_axes, contents=expectation, copy_contents=False)

    # --- R^T·H : データ空間 H(Time) を response の Time 軸と縮約 → モデル空間(lb, Ei) ---
    def calc_T_product(self, dataspace_histogram):
        tprod = np.tensordot(_plain(dataspace_histogram.contents),
                             _plain(self._image_response.contents),
                             axes=((0,), (2,)))              # -> (npix, n_e)
        return Histogram(self.model_axes, contents=tprod, copy_contents=False)

    # --- Σ_i B_i H_i ---
    def calc_bkg_model_product(self, key, dataspace_histogram):
        return float(np.sum(_plain(self.bkg_model(key).contents)
                            * _plain(dataspace_histogram.contents)))

    # --- Poisson 対数尤度 ---
    def calc_log_likelihood(self, expectation):
        mu = np.clip(_plain(expectation.contents), 1e-12, None)
        d = _plain(self._event.contents)
        return float(np.sum(d * np.log(mu) - mu))


# ----------------------------------------------------------------------
# トイのジオメトリ+レスポンスを組み立て、観測(Poisson)を合成して DataIF を返す。
# 複数「軌道」を作れば cosipy の DataInterfaceCollection にそのまま渡せる。
# ----------------------------------------------------------------------
def _pixel_lb(nside):
    """ring スキームの画素中心 (l, b) [deg]。"""
    l, b = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    return l, b


def _angsep_deg(l1, b1, l2, b2):
    """(l1,b1) と (l2,b2) の角距離 [deg]。すべて deg。"""
    r = np.radians
    v = (np.cos(r(b1)) * np.cos(r(l1)), np.cos(r(b1)) * np.sin(r(l1)), np.sin(r(b1)))
    w = (np.cos(r(b2)) * np.cos(r(l2)), np.cos(r(b2)) * np.sin(r(l2)), np.sin(r(b2)))
    dot = np.clip(v[0] * w[0] + v[1] * w[1] + v[2] * w[2], -1, 1)
    return np.degrees(np.arccos(dot))


def build_toy_scan_dataif(
    name,
    nside,
    truth_map,               # (npix,) 真のフラックス。events 合成に使う
    scan_l,                  # (n_t,) 各時刻の指向の銀経 [deg] (b=0 を掃く)
    dwell=100.0,             # 時間ビンあたり露出
    fov_sigma=8.0,           # コリメータのガウス幅 [deg]
    energy_edges=np.array([2.0, 10.0]),
    bkg_level=0.0,           # 平坦な instrumental 背景(データ空間)
    seed=0,
):
    """トイ scan の DataIF_ToyScan を1つ作って返す(events は Poisson で合成)。"""
    npix = hp.nside2npix(nside)
    n_t = len(scan_l)
    lb_axis = HealpixAxis(nside=nside, scheme="ring", coordsys="galactic", label="lb")
    e_axis = Axis(edges=energy_edges * u.keV, label="Ei", scale="log")
    t_axis = Axis(edges=np.arange(n_t + 1) * 1.0, label="Time")
    model_axes = Axes([lb_axis, e_axis], copy_axes=False)
    data_axes = Axes([t_axis], copy_axes=False)

    # レスポンス R[pix, Ei, Time]
    pl, pb = _pixel_lb(nside)
    R = np.zeros((npix, 1, n_t))
    # ここは、今は手で書いているが、ちゃんとした解析では、
    # threemlで作っているresponseの畳み込みをそのまま使えると
    # 同じものを作らなくてよい.
    for t, lt in enumerate(scan_l):
        sep = _angsep_deg(pl, pb, lt, 0.0)                       # (npix,)
        R[:, 0, t] = np.exp(-0.5 * (sep / fov_sigma) ** 2) * dwell
    response_axes = Axes([lb_axis, e_axis, t_axis], copy_axes=False)
    response = Histogram(response_axes, contents=R, copy_contents=False)

    # 期待カウント → Poisson で観測を合成
    truth = np.asarray(truth_map, dtype=float).reshape(npix, 1)
    src_expectation = np.tensordot(truth, R, axes=((0, 1), (0, 1)))  # (n_t,)
    bkg_template = np.full(n_t, float(bkg_level))
    expected = src_expectation + bkg_template
    counts = np.random.default_rng(seed).poisson(np.clip(expected, 0, None)).astype(float)

    event = Histogram(data_axes, contents=counts, copy_contents=False)
    bkg_models = {}
    if bkg_level > 0:
        bkg_models["inst"] = Histogram(data_axes, contents=bkg_template, copy_contents=False)

    return DataIF_ToyScan.load(name, event, bkg_models, response, model_axes, data_axes)
