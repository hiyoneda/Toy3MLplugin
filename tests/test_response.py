"""
応答 _response_for_source のサニティテスト。

トイ装置の物理的な期待:
  - 天体をスキャン範囲の中に置くと、その "transit 時刻" にライトカーブのピークが立つ。
  - 天体の位置(スキャン座標 l)をずらすと、ピーク位置も同じだけずれる。
  - shape 契約 (n_data_bins, n_e) を守る。

NinjaSat 版を書いたら、これに相当する「1点源を FOV に置くと transit にバンプ」
テストを必ず用意する ── それが直交スリット応答の正しさの最小保証になる。
"""

import numpy as np
from astromodels import Model, PointSource, Powerlaw

from Toy3MLplugin import make_folded_response
from Toy3MLplugin.toy.scanner import ToyScanLike, toy_observation


def _plugin_with_source_at(l_deg, energy_edges=np.array([2.0, 10.0])):
    R, sky_l, orbit = make_folded_response((0, 30), n_t=400, dwell=1.0,
                                           energy_edges=energy_edges, fov_sigma_deg=3.0)
    obs = toy_observation(counts=np.zeros(orbit["l"].size), R=R, sky_l=sky_l, orbit=orbit)
    plugin = ToyScanLike("toy", obs, energy_edges)
    src = PointSource("S", l=l_deg, b=0.0, spectral_shape=Powerlaw(K=1e-2, index=-2.0))
    plugin.set_model(Model(src))
    return plugin, orbit["l"]


def test_response_shape(energy_edges=np.array([2.0, 6.0, 10.0])):
    plugin, scan = _plugin_with_source_at(5.0, energy_edges)
    src = list(plugin._model.point_sources.values())[0]
    R = plugin._response_for_source(src)
    assert R.shape == (scan.size, energy_edges.size - 1)


def test_peak_at_transit():
    """天体を l=5 に置くと、スキャン角が 5 に最も近い時間ビンでカウント最大。"""
    plugin, scan = _plugin_with_source_at(5.0)
    mu = plugin.expected_counts()
    peak_bin = np.argmax(mu)
    assert abs(scan[peak_bin] - 5.0) < 0.2  # ビン間隔程度の精度でピーク一致


def test_peak_shifts_with_position():
    """位置をずらすと transit 時刻(ピーク)も同じ向きにずれる。"""
    p1, scan = _plugin_with_source_at(5.0)
    p2, _ = _plugin_with_source_at(15.0)
    peak1 = scan[np.argmax(p1.expected_counts())]
    peak2 = scan[np.argmax(p2.expected_counts())]
    assert peak2 > peak1
    assert np.isclose(peak2 - peak1, 10.0, atol=0.3)


def test_two_sources_two_bumps():
    """2天体を離して置くと、ライトカーブに2つのバンプが立つ(多天体の重ね合わせ)。"""
    R, sky_l, orbit = make_folded_response((0, 30), n_t=400, dwell=1.0,
                                           energy_edges=np.array([2.0, 10.0]), fov_sigma_deg=2.0)
    obs = toy_observation(counts=np.zeros(orbit["l"].size), R=R, sky_l=sky_l, orbit=orbit)
    plugin = ToyScanLike("toy", obs, np.array([2.0, 10.0]))
    a = PointSource("A", l=6.0, b=0.0, spectral_shape=Powerlaw(K=1e-2, index=-2.0))
    b = PointSource("B", l=20.0, b=0.0, spectral_shape=Powerlaw(K=1e-2, index=-2.0))
    plugin.set_model(Model(a, b))
    mu = plugin.expected_counts()

    # 局所ピークを数える(素朴に、両隣より大きい点)
    is_peak = (mu[1:-1] > mu[:-2]) & (mu[1:-1] > mu[2:]) & (mu[1:-1] > mu.max() * 0.1)
    assert is_peak.sum() == 2
