"""
トイ cosipy DataIF の構造テスト(cosipy が無ければ skip)。

画像再構成の厳密な検証は cosipy 側で各自行う前提。ここでは
「DataIF が cosipy の契約どおりの軸・形の Histogram を返すか」だけ確認する。
"""

import numpy as np
import pytest

pytest.importorskip("cosipy")

import astropy.units as u
import healpy as hp
from cosipy.image_deconvolution import AllSkyImageModel, ImageDeconvolution
from Toy3MLplugin.cosipy_dataif import build_toy_scan_dataif

NSIDE = 4


def _truth():
    npix = hp.nside2npix(NSIDE)
    l, b = hp.pix2ang(NSIDE, np.arange(npix), lonlat=True)
    t = np.zeros(npix)
    t[np.argmin((l - 10) ** 2 + b ** 2)] = 1.0e-2
    return t, l, b


def test_dataif_method_axes_and_shapes():
    truth, l, b = _truth()
    d = build_toy_scan_dataif("t", NSIDE, truth, np.linspace(0, 90, 40),
                              dwell=300.0, fov_sigma=12.0, seed=0)
    model = AllSkyImageModel(nside=NSIDE, energy_edges=np.array([2., 10.]) * u.keV)
    model[:] = 1e-4 * model.unit

    se = d.calc_source_expectation(model)
    assert [a.label for a in se.axes] == ["Time"]
    assert se.contents.shape == (40,)

    tp = d.calc_T_product(d.event)
    assert [a.label for a in tp.axes] == ["lb", "Ei"]
    assert tp.contents.shape == (hp.nside2npix(NSIDE), 1)

    assert d.exposure_map.contents.shape == (hp.nside2npix(NSIDE), 1)
    assert np.isfinite(d.calc_log_likelihood(se))


def test_cosipy_imagedeconvolution_runs_on_toy_dataif():
    """cosipy の ImageDeconvolution(正式な入口)+ YAML がトイ DataIF 上で回る。"""
    import os
    truth, l, b = _truth()
    ds = [build_toy_scan_dataif("A", NSIDE, truth, np.linspace(0, 90, 60),
                                dwell=400.0, fov_sigma=10.0, bkg_level=0.5, seed=1)]
    yml = os.path.join(os.path.dirname(__file__), "..", "examples",
                       "imagedeconvolution_toy.yml")
    imdec = ImageDeconvolution()
    imdec.set_dataset(ds)
    imdec.read_parameterfile(os.path.abspath(yml))
    imdec.initialize()
    imdec.run_deconvolution()
    rec = np.asarray(imdec.results[-1]["model"].contents).ravel()
    assert np.all(np.isfinite(rec)) and rec.max() > 0
