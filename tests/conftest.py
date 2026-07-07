import logging as _logging; _logging.disable(_logging.DEBUG)
import os as _os; _os.environ.setdefault("ASTROMODELS_LOG_LEVEL", "WARNING")
"""
共有 fixture。トイ装置の観測ジオメトリと、真のモデルをここで用意する。

学生へ: 自分の装置プラグインを書いたら、ここに mock fixture を1つ足して
test_closure.py の parametrize に載せるだけで、同じクロージャテストが
自分の装置にも流れる ── というのがこのテスト設計の狙い。
"""

import numpy as np
import pytest
from astromodels import Model, PointSource, Powerlaw

from Toy3MLplugin import BackgroundComponent
from Toy3MLplugin import make_folded_response
from Toy3MLplugin.toy.scanner import ToyScanLike, toy_observation


@pytest.fixture
def rng():
    return np.random.default_rng(20260703)


@pytest.fixture
def energy_edges():
    # トイは 1 バンド運用(NinjaSat の n_e=1 と同じ退化ケース)。
    return np.array([2.0, 10.0])


@pytest.fixture
def geometry(energy_edges):
    # 装置固有応答 × 軌道 を畳み込んだレスポンスを一度だけ作って使い回す。
    R, sky_l, orbit = make_folded_response((0, 30), n_t=400, dwell=2.0,
                                           energy_edges=energy_edges, fov_sigma_deg=3.0)
    return R, sky_l, orbit


def _truth_model():
    """真の2天体モデル(K が真値、index 固定)。"""
    a = PointSource("SRC_A", l=2.0, b=0.0, spectral_shape=Powerlaw(K=1.0e-2, index=-2.0))
    b = PointSource("SRC_B", l=11.0, b=0.0, spectral_shape=Powerlaw(K=5.0e-3, index=-2.0))
    for s in (a, b):
        s.spectrum.main.Powerlaw.index.free = False
        s.position.l.free = False
        s.position.b.free = False
    return Model(a, b)


@pytest.fixture
def truth_model():
    return _truth_model()


@pytest.fixture
def truth_K():
    return {"SRC_A": 1.0e-2, "SRC_B": 5.0e-3}


def _fresh_model():
    """フィット開始用のモデル。構造は真と同じだが K を故意にずらす。"""
    m = _truth_model()
    m.SRC_A.spectrum.main.Powerlaw.K.value = 3.0e-3
    m.SRC_B.spectrum.main.Powerlaw.K.value = 3.0e-3
    return m


@pytest.fixture
def fresh_model():
    return _fresh_model()


def _make_toy_plugin(counts, geometry, energy_edges, with_background):
    R, sky_l, orbit = geometry
    obs = toy_observation(counts=counts, R=R, sky_l=sky_l, orbit=orbit)
    backgrounds = None
    if with_background:
        flat = np.ones(orbit["l"].size)
        backgrounds = [BackgroundComponent("inst", template=flat, value=5.0)]
    return ToyScanLike("toy", obs, energy_edges, backgrounds=backgrounds)


@pytest.fixture
def make_toy_plugin(geometry, energy_edges):
    """counts と背景の有無を渡すとトイ plugin を返すファクトリ。"""
    def _factory(counts, with_background=False):
        return _make_toy_plugin(counts, geometry, energy_edges, with_background)
    return _factory
