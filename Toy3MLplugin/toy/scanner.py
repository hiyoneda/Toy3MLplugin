"""
toy.scanner — トイ装置「スキャンするコリメータ」の 3ML プラグイン。

レスポンスの中身(装置固有応答×軌道の畳み込み)は Toy3MLplugin.instrument 側で「顕に」作る。
このプラグインは、出来上がった畳み込み済みレスポンス R[time, sky, e] を受け取り、
点源1個ぶんの応答(= その位置の sky 列)を切り出すだけ。

    _load_response          … 観測から畳み込み済みレスポンスと sky グリッドを取り出す
    _response_for_source    … 点源の位置に対応する sky 列 R[:, k, :] を返す

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import astropy.units as u

from Toy3MLplugin.base import ForwardFoldingLike
from Toy3MLplugin.observation import Observation

# 軌道生成/畳み込みは instrument モジュールから使う(後方互換の再エクスポート)。
from Toy3MLplugin.instrument import make_scan_orbit, make_folded_response  # noqa: F401


def make_scan_geometry(scan_min=0.0, scan_max=30.0, n_t=300, dwell=1.0):
    """後方互換: スキャン軌道の (l, exposure) を返す薄いラッパ。
    新しいコードは Toy3MLplugin.instrument.make_scan_orbit を推奨。"""
    orbit = make_scan_orbit(scan_min, scan_max, n_t=n_t, dwell=dwell)
    return orbit["l"], orbit["exposure"]


@dataclass
class ToyScanObservation(Observation):
    """トイ観測。畳み込み済みレスポンスと sky グリッドを保持する。

    counts          : データ空間(時間ビン)の観測カウント (n_t,)
    exposure        : 時間ビンごとの露出 [s] (n_t,)  ※参考情報(露出はレスポンスに畳み込み済み)
    folded_response : astropy Quantity R[n_t, n_sky, n_e]  (Toy3MLplugin.instrument.fold_response の出力)
    sky_l           : モデル空間(sky)の銀経グリッド [deg] (n_sky,)
    """

    folded_response: object = None
    sky_l: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        if self.folded_response is None or self.sky_l is None:
            raise ValueError("folded_response と sky_l が必要です "
                             "(Toy3MLplugin.instrument.make_folded_response で作れます)")
        self.sky_l = np.asarray(self.sky_l, dtype=float).ravel()


class ToyScanLike(ForwardFoldingLike):
    """畳み込み済みレスポンスを使うトイ装置プラグイン。"""

    def _load_response(self):
        obs: ToyScanObservation = self._obs
        R = obs.folded_response
        # 単位つき(cm^2 s)なら値だけ取り出す(尤度計算は無単位の数値空間で行う)。
        self._R = R.to_value(u.cm ** 2 * u.s) if hasattr(R, "to_value") else np.asarray(R)
        self._sky_l = obs.sky_l
        if self._R.shape[0] != self._data.size:
            raise ValueError(f"folded_response の時間軸 {self._R.shape[0]} が "
                             f"counts の長さ {self._data.size} と一致しません")

    def _response_for_source(self, source):
        # 点源の銀経 l に最も近い sky 列を切り出す(点源応答 = レスポンスのスライス)。
        l0 = source.position.get_l()
        k = int(np.argmin(np.abs(self._sky_l - l0)))
        return self._R[:, k, :]                      # (n_t, n_e)


def toy_observation(counts, R, sky_l, orbit):
    """counts と (R, sky_l, orbit) から ToyScanObservation を作る薄いヘルパ。"""
    return ToyScanObservation(counts=counts, exposure=orbit["exposure"],
                              folded_response=R, sky_l=sky_l)
