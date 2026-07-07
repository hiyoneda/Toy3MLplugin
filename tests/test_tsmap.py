"""
TS マップ(localization)と、複数 dataset 同時フィットのテスト。
"""

import numpy as np
import pytest
from astromodels import Model, PointSource, Powerlaw

from Toy3MLplugin import simulate_counts, BackgroundComponent
from Toy3MLplugin.tsmap import ts_map_1d
from Toy3MLplugin import make_folded_response
from Toy3MLplugin.toy.scanner import ToyScanLike, toy_observation
from threeML import DataList, JointLikelihood

E_EDGES = np.array([2.0, 10.0])
SIGMA = 3.0


def _fixed(name, l, K):
    s = PointSource(name, l=l, b=0.0, spectral_shape=Powerlaw(K=K, index=-2.0))
    s.spectrum.main.Powerlaw.index.free = False
    s.position.l.free = False
    s.position.b.free = False
    return s


def _bkg(n):
    return [BackgroundComponent("inst", np.ones(n), value=4.0)]


def _toy(counts, lo, hi, dwell):
    R, sky_l, orbit = make_folded_response((lo, hi), n_t=400, dwell=dwell,
                                           energy_edges=E_EDGES, fov_sigma_deg=SIGMA)
    n_t = orbit["l"].size
    obs = toy_observation(counts=counts if counts is not None else np.zeros(n_t),
                          R=R, sky_l=sky_l, orbit=orbit)
    return obs, n_t


def test_tsmap_localizes_single_source():
    rng = np.random.default_rng(2)
    truth = Model(_fixed("S", 12.0, 0.5))
    obs0, n_t = _toy(None, 0, 30, 80.0)
    sim = ToyScanLike("t", obs0, E_EDGES, backgrounds=_bkg(n_t))
    data = simulate_counts(sim, truth, rng)
    obs1, _ = _toy(data, 0, 30, 80.0)
    plugin = ToyScanLike("t", obs1, E_EDGES, backgrounds=_bkg(n_t))

    grid = np.arange(0, 30.01, 1.0)
    gl, ts = ts_map_1d(plugin, grid)
    assert abs(gl[np.argmax(ts)] - 12.0) <= 1.0     # ピークが真の位置(グリッド刻み内)
    assert ts.max() > 25                            # 明るい源なので有意(>5σ)


def test_multi_dataset_joint_fit():
    """カバレッジの違う複数軌道を DataList で束ねて同時フィット → 両源を回収。"""
    def two(KA, KB):
        a = _fixed("SRC_A", 6.0, KA)
        b = _fixed("SRC_B", 18.0, KB)
        return Model(a, b)

    truth = two(1.0, 0.5)

    def orbit(name, lo, hi, dwell, seed):
        obs0, n_t = _toy(None, lo, hi, dwell)
        sim = ToyScanLike(name, obs0, E_EDGES, backgrounds=_bkg(n_t))
        data = simulate_counts(sim, truth, np.random.default_rng(seed))
        obs1, _ = _toy(data, lo, hi, dwell)
        return ToyScanLike(name, obs1, E_EDGES, backgrounds=_bkg(n_t))

    orbits = [orbit("orbit1", 0, 22, 120.0, 1),
              orbit("orbit2", 8, 30, 120.0, 2)]

    start = two(0.2, 0.2)
    jl = JointLikelihood(start, DataList(*orbits))
    jl.set_minimizer("minuit")
    jl.fit(quiet=True)

    assert start.SRC_A.spectrum.main.Powerlaw.K.value == pytest.approx(1.0, rel=0.15)
    assert start.SRC_B.spectrum.main.Powerlaw.K.value == pytest.approx(0.5, rel=0.20)
