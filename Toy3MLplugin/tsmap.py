"""
Toy3MLplugin.tsmap — 1次元 TS マップ(localization)。

localization は「フラックスは各位置で高速に profile、位置探しは外側のグリッド」
という二段構え(前回の説明どおり)。ここが「外側のグリッド」。

各グリッド位置 l に試験点源を置き、フラックスだけを最尤にして logL を測る。
    TS(l) = 2 * ( logL[位置 l に源あり(flux profiled)] - logL[源なし] )
TS(l) の山が源の位置 = localization。sqrt(TS) がおおよその検出有意度。

3ML にも TS 計算はあるが、ここでは「TS が何を測っているか」を透明にするため、
プラグインの前方モデル(get_log_like)の上に素朴に実装する。
内側のフラックス profile は1パラメータの凹最適化なので scipy で十分速い。
"""

from __future__ import annotations

import numpy as np
from astromodels import Model, PointSource, Powerlaw
from scipy.optimize import minimize_scalar


def _loglike_with_test_source(plugin, base_sources, l, b, flux, index=-2.0):
    """base_sources(固定) + 位置(l,b)の試験源(K=flux) を置いたときの logL。"""
    srcs = list(base_sources)
    test = PointSource("__test__", l=l, b=b,
                       spectral_shape=Powerlaw(K=max(flux, 1e-30), index=index))
    test.spectrum.main.Powerlaw.index.free = False
    test.position.l.free = False
    test.position.b.free = False
    plugin.set_model(Model(*srcs, test))
    return plugin.get_log_like()


def _profile_flux(plugin, base_sources, l, b, k_max=1e3):
    """位置(l,b)固定で、試験源のフラックスだけ最尤化した logL を返す。"""
    def neg(logK):
        return -_loglike_with_test_source(plugin, base_sources, l, b, np.exp(logK))
    res = minimize_scalar(neg, bounds=(np.log(1e-8), np.log(k_max)), method="bounded")
    return -res.fun


def ts_map_1d(plugin, grid_l, base_sources=None, b=0.0):
    """1次元 TS マップを返す。

    plugin        : 観測データを積んだ ForwardFoldingLike(背景があれば nuisance 込み)
    grid_l        : 走査する位置グリッド [deg]
    base_sources  : 既知源(固定)のリスト。None なら「源なし(背景のみ)」が null。

    returns: (grid_l, TS[grid_l])
    """
    base_sources = list(base_sources or [])

    # null: 既知源のみ(試験源なし)。源が無ければ背景のみ(空 Model は作れないので
    # ダミー源をフラックス0で置いて背景だけの期待にする)。
    if base_sources:
        plugin.set_model(Model(*base_sources))
        logL_null = plugin.get_log_like()
    else:
        logL_null = _loglike_with_test_source(plugin, [], grid_l[0], b, flux=0.0)

    ts = np.empty(len(grid_l))
    for i, l in enumerate(grid_l):
        logL_src = _profile_flux(plugin, base_sources, l, b)
        ts[i] = 2.0 * (logL_src - logL_null)
    return np.asarray(grid_l), ts
