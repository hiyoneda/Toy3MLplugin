"""
Toy3MLplugin.instrument — レスポンス生成を「顕」にするモジュール。

解析の前段を明示的に分ける:

    衛星軌道(pointing履歴)  +  装置固有の応答(pointing非依存)
        ──────────────  fold_response  ──────────────►  畳み込み済みレスポンス
                                                          (これを ToyScanLike に入れる)

- (コリメータ系の場合）装置固有の応答 R0[theta, e] は「視線からの角距離 theta 
  とエネルギーに対する実効面積xコリメータ透過」。
  pointing にも軌道にもよらない、装置そのものの性質。
- 衛星軌道 orbit は、時刻ごとの指向 (l, b) と露出など。デモでは FITS で保存/読込する。
- fold_response が両者を畳み込み、データ空間(時間) × モデル空間(sky) × エネルギー の
  レスポンス R[t, sky, e] を返す。これが前方畳み込みの前方作用素そのもの。

いまは R0 / R を numpy or astropy(単位つき) の array で扱う。
今後は FITS なり HDF5 でレスポンスを配布/保存するといい.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.io import fits


# ---------------------------------------------------------------------------
# 装置固有の応答 R0(theta, e)  ── pointing 非依存
# ---------------------------------------------------------------------------
def intrinsic_response(
    theta_grid_deg: np.ndarray,
    energy_edges: np.ndarray,
    fov_sigma_deg: float = 3.0,
    eff_area_cm2: float = 1.0,
):
    """装置固有の応答 R0[theta, e] を返す(astropy Quantity, 単位 cm^2)。

    この例では「視線からの角距離 theta のガウス型コリメータ透過 × 一定の実効面積」。
    漏れ込みで視野が太る度合いが fov_sigma_deg。実装を差し替えれば
    エネルギー依存の実効面積 A_eff(e) やビーム形状も表現できる。

    NOTE: いまは numpy/astropy array。
    """
    theta = np.asarray(theta_grid_deg, dtype=float)
    n_e = len(np.asarray(energy_edges)) - 1
    transmission = np.exp(-0.5 * (theta / fov_sigma_deg) ** 2)          # [n_theta]
    R0 = np.outer(transmission, np.full(n_e, float(eff_area_cm2)))       # [n_theta, n_e]
    return R0 * u.cm ** 2, theta


# ---------------------------------------------------------------------------
# 衛星軌道(pointing履歴)  ── デモでは FITS で保存/読込
# ---------------------------------------------------------------------------
def make_scan_orbit(scan_min=0.0, scan_max=30.0, n_t=500, dwell=1.0, b=0.0):
    """一定速度で銀経を掃くスキャン軌道を作る(簡略化した pointing 履歴)。

    returns: dict(time [s], l [deg], b [deg], exposure [s])  各 (n_t,)
    """
    l = np.linspace(scan_min, scan_max, n_t)
    time = np.arange(n_t) * dwell
    return {
        "time": time,
        "l": l,
        "b": np.full(n_t, float(b)),
        "exposure": np.full(n_t, float(dwell)),
    }


def save_orbit_fits(path, time, l, b, exposure):
    """軌道(pointing履歴)を FITS のバイナリテーブルとして保存する(デモ)。"""
    cols = [
        fits.Column(name="TIME", format="D", unit="s", array=np.asarray(time)),
        fits.Column(name="L", format="D", unit="deg", array=np.asarray(l)),
        fits.Column(name="B", format="D", unit="deg", array=np.asarray(b)),
        fits.Column(name="EXPOSURE", format="D", unit="s", array=np.asarray(exposure)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="ORBIT")
    hdu.header["EXTNAME"] = "ORBIT"
    hdu.writeto(path, overwrite=True)


def load_orbit_fits(path):
    """save_orbit_fits で書いた軌道 FITS を読み込む → dict。"""
    with fits.open(path) as h:
        d = h["ORBIT"].data
        return {
            "time": np.array(d["TIME"], dtype=float),
            "l": np.array(d["L"], dtype=float),
            "b": np.array(d["B"], dtype=float),
            "exposure": np.array(d["EXPOSURE"], dtype=float),
        }


# ---------------------------------------------------------------------------
# 畳み込み: 軌道 x 装置固有応答 → データ空間xモデル空間xエネルギーのレスポンス
# ---------------------------------------------------------------------------
def _angsep_deg(l1, b1, l2, b2):
    """(l1,b1) と (l2,b2) の角距離 [deg]。"""
    r = np.radians
    v = (np.cos(r(b1)) * np.cos(r(l1)), np.cos(r(b1)) * np.sin(r(l1)), np.sin(r(b1)))
    w = (np.cos(r(b2)) * np.cos(r(l2)), np.cos(r(b2)) * np.sin(r(l2)), np.sin(r(b2)))
    dot = np.clip(v[0] * w[0] + v[1] * w[1] + v[2] * w[2], -1, 1)
    return np.degrees(np.arccos(dot))


def fold_response(orbit, R0, theta_grid, sky_l, sky_b=0.0):
    """衛星軌道 orbit と 装置固有応答 R0(theta,e) を畳み込む。

    R[t, k, e] = R0( angsep(sky_k, pointing_t), e ) * exposure_t     単位 [cm^2 s]

    orbit      : dict(l, b, exposure) (make_scan_orbit / load_orbit_fits の出力)
    R0         : astropy Quantity [n_theta, n_e] (intrinsic_response の出力)
    theta_grid : R0 の theta グリッド [deg]
    sky_l      : モデル空間(sky)の銀経グリッド [deg]  (n_sky,)
    sky_b      : sky の銀緯(トイは一定)

    returns: astropy Quantity R[n_t, n_sky, n_e]  単位 cm^2 s
             (これを ToyScanObservation.folded_response に渡す)

    NOTE: 出力はいま astropy array。
    """
    l_t = np.asarray(orbit["l"], float)
    b_t = np.asarray(orbit["b"], float)
    exp_t = np.asarray(orbit["exposure"], float)
    sky_l = np.asarray(sky_l, float)

    R0_val = R0.to_value(u.cm ** 2)                 # [n_theta, n_e]
    n_t, n_sky, n_e = len(l_t), len(sky_l), R0_val.shape[1]

    R = np.zeros((n_t, n_sky, n_e))
    for t in range(n_t):
        sep = _angsep_deg(sky_l, sky_b, l_t[t], b_t[t])          # [n_sky]
        for e in range(n_e):
            R[t, :, e] = np.interp(sep, theta_grid, R0_val[:, e]) * exp_t[t]
    return R * (u.cm ** 2 * u.s)


# ---------------------------------------------------------------------------
# 便利関数: 上の3ステップをまとめてトイ観測の材料を作る
# ---------------------------------------------------------------------------
def make_folded_response(scan_range, n_t, dwell, energy_edges,
                         fov_sigma_deg=3.0, sky_step=0.25, orbit=None):
    """軌道生成 → 装置固有応答 → 畳み込み をまとめて実行し、(R, sky_l, orbit) を返す。

    orbit を渡せばそれを使う(FITS から読んだ軌道など)。渡さなければ scan_range から作る。
    """
    if orbit is None:
        orbit = make_scan_orbit(scan_range[0], scan_range[1], n_t=n_t, dwell=dwell)
    span = scan_range[1] - scan_range[0]
    theta_grid = np.linspace(0.0, span + 6 * fov_sigma_deg, 400)
    R0, theta = intrinsic_response(theta_grid, energy_edges, fov_sigma_deg=fov_sigma_deg)
    sky_l = np.round(np.arange(scan_range[0], scan_range[1] + 1e-9, sky_step), 4)
    R = fold_response(orbit, R0, theta, sky_l)
    return R, sky_l, orbit
