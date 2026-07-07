"""
注入-回収(injection-recovery)クロージャテスト ── このライブラリの本丸。

真値 → データ(Poisson) → 3ML JointLikelihood でフィット → 真値が誤差内で戻るか。
前方畳み込み解析が正しいことを担保する決定版テスト。

parametrize の狙い:
  いまは instrument="toy" だけだが、学生が NinjaSat / SMILE プラグインを書いたら
  下の INSTRUMENTS にファクトリを1つ足すだけで、同じクロージャテストが
  その装置にも流れる。共通コアのテスト1本で全装置をカバーする、という設計。
"""

import numpy as np
import pytest
from astromodels import Model, PointSource, Powerlaw

from Toy3MLplugin import simulate_counts
from threeML import DataList, JointLikelihood

# ----------------------------------------------------------------------
# 装置ごとの "真の plugin(データ合成用)" と "解析用 plugin" を作るファクトリ。
# 新装置はここに1エントリ足すだけ。
# ----------------------------------------------------------------------
from Toy3MLplugin import make_folded_response
from Toy3MLplugin.toy.scanner import ToyScanLike, toy_observation


def _toy_factory(counts, with_background):
    from Toy3MLplugin import BackgroundComponent
    E = np.array([2.0, 10.0])
    # 装置固有応答 × 軌道 の畳み込み(sim/解析で同じ R を使う; 決定論的)。
    R, sky_l, orbit = make_folded_response((0, 30), n_t=500, dwell=200.0,
                                           energy_edges=E, fov_sigma_deg=3.0)
    n_t = orbit["l"].size
    obs = toy_observation(counts=counts if counts is not None else np.zeros(n_t),
                          R=R, sky_l=sky_l, orbit=orbit)
    backgrounds = None
    if with_background:
        backgrounds = [BackgroundComponent("inst", np.ones(n_t), value=8.0)]
    return ToyScanLike("inst", obs, E, backgrounds=backgrounds)


INSTRUMENTS = {
    "toy": _toy_factory,
    # "ninjasat": _ninjasat_factory,   # 学生が書いたら追加
    # "smile":    _smile_factory,
}


def _truth_model(K_A, K_B):
    a = PointSource("SRC_A", l=2.0, b=0.0, spectral_shape=Powerlaw(K=K_A, index=-2.0))
    b = PointSource("SRC_B", l=13.0, b=0.0, spectral_shape=Powerlaw(K=K_B, index=-2.0))
    for s in (a, b):
        s.spectrum.main.Powerlaw.index.free = False
        s.position.l.free = False
        s.position.b.free = False
    return Model(a, b)


@pytest.mark.parametrize("instrument", list(INSTRUMENTS))
@pytest.mark.parametrize("with_background", [False, True])
def test_injection_recovery(instrument, with_background):
    factory = INSTRUMENTS[instrument]
    rng = np.random.default_rng(1234)

    K_A_true, K_B_true = 1.0, 0.5

    # --- 1) 真のモデルからデータを合成 ---
    truth = _truth_model(K_A_true, K_B_true)
    sim_plugin = factory(counts=None, with_background=with_background)
    data = simulate_counts(sim_plugin, truth, rng)

    # --- 2) 観測データを積んだ解析用 plugin を作る ---
    plugin = factory(counts=data, with_background=with_background)

    # --- 3) K をずらした初期モデルからフィット ---
    start = _truth_model(3.0e-3, 3.0e-3)
    jl = JointLikelihood(start, DataList(plugin))
    jl.set_minimizer("minuit")
    jl.fit(quiet=True, compute_covariance=True)

    fit_KA = start.SRC_A.spectrum.main.Powerlaw.K.value
    fit_KB = start.SRC_B.spectrum.main.Powerlaw.K.value

    # --- 4) 真値が相対誤差内で戻ることを確認 ---
    assert fit_KA == pytest.approx(K_A_true, rel=0.15), f"{instrument}: K_A off"
    assert fit_KB == pytest.approx(K_B_true, rel=0.20), f"{instrument}: K_B off"


@pytest.mark.parametrize("instrument", list(INSTRUMENTS))
def test_source_confusion(instrument):
    """近接2源でも縮退せずそれぞれの flux を分離できるか(スキャン装置で特に効く)。"""
    factory = INSTRUMENTS[instrument]
    rng = np.random.default_rng(77)

    # 近接(l=6 と l=9、fov_sigma=3 なので裾が重なる)
    a = PointSource("SRC_A", l=6.0, b=0.0, spectral_shape=Powerlaw(K=1.0, index=-2.0))
    b = PointSource("SRC_B", l=9.0, b=0.0, spectral_shape=Powerlaw(K=0.6, index=-2.0))
    for s in (a, b):
        s.spectrum.main.Powerlaw.index.free = False
        s.position.l.free = False
        s.position.b.free = False
    truth = Model(a, b)

    sim_plugin = factory(counts=None, with_background=False)
    data = simulate_counts(sim_plugin, truth, rng)
    plugin = factory(counts=data, with_background=False)

    a2 = PointSource("SRC_A", l=6.0, b=0.0, spectral_shape=Powerlaw(K=0.3, index=-2.0))
    b2 = PointSource("SRC_B", l=9.0, b=0.0, spectral_shape=Powerlaw(K=0.3, index=-2.0))
    for s in (a2, b2):
        s.spectrum.main.Powerlaw.index.free = False
        s.position.l.free = False
        s.position.b.free = False
    start = Model(a2, b2)

    jl = JointLikelihood(start, DataList(plugin))
    jl.set_minimizer("minuit")
    jl.fit(quiet=True, compute_covariance=False)

    assert start.SRC_A.spectrum.main.Powerlaw.K.value == pytest.approx(1.0, rel=0.25)
    assert start.SRC_B.spectrum.main.Powerlaw.K.value == pytest.approx(0.6, rel=0.30)
