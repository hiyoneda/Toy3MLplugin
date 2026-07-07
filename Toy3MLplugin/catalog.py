"""
Toy3MLplugin.catalog — YAML の点源リスト → astromodels.Model。

Fermi は XML でソースリストを与えるが、ここでは手編集しやすい薄い YAML 層を
かぶせる。中身の多天体モデルは astromodels.Model そのもの ── 位置・スペクトル形・
free/fixed・prior まで全部持っていて、3ML がそのままフィットする。
つまり「多天体同時フィットの枠組み」を自作する必要はなく、
このローダの責務は "YAML を読んで Model を組む" 一点に閉じている。

YAML 例 (examples/sources.yaml):

    sources:
      - name: SRC_A
        position: {l: 2.0, b: 4.0}
        spectrum:
          shape: Powerlaw
          params: {K: 1.0e-3, index: -2.0}
          free: [K]                 # norm だけ自由、index は固定
    diffuse: []                     # 背景は装置側で template を作って渡す

拡張する場合: shape を増やすときは _SHAPES に astromodels の関数を足すだけ.
              要テストだが、xspec model も使うことが可能なはず
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import yaml
from astromodels import Model, PointSource, Powerlaw, Gaussian, Cutoff_powerlaw

# 使えるスペクトル形。ここに astromodels の Function1D を足せば増やせる。
_SHAPES = {
    "Powerlaw": Powerlaw,
    "Cutoff_powerlaw": Cutoff_powerlaw,
    "Gaussian": Gaussian,
}


def _build_spectrum(spec: dict):
    shape_name = spec["shape"]
    if shape_name not in _SHAPES:
        raise KeyError(f"unknown spectrum shape '{shape_name}'. known: {list(_SHAPES)}")
    func = _SHAPES[shape_name]()

    params = spec.get("params", {})
    for pname, pval in params.items():
        getattr(func, pname).value = float(pval)

    # free リストにある名前だけ自由、残りは固定にする(明示的で事故が少ない)。
    free_names = set(spec.get("free", []))
    for pname in func.parameters:
        getattr(func, pname).free = pname in free_names

    return func


def load_sources(path: str) -> Tuple[Model, List[str]]:
    """YAML を読み、(astromodels.Model, diffuse名リスト) を返す。

    diffuse は、ここでは箱だけを作っておく。
    threemlは、diffuse modelもあるので、必要に応じて拡張していく。
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    sources = []
    for s in cfg.get("sources", []):
        pos = s["position"]
        ps = PointSource(
            s["name"],
            l=float(pos["l"]),
            b=float(pos["b"]),
            spectral_shape=_build_spectrum(s["spectrum"]),
        )
        sources.append(ps)

    if not sources:
        raise ValueError(f"no point sources found in {path}")

    model = Model(*sources)
    diffuse_names = [d["name"] for d in cfg.get("diffuse", [])]
    return model, diffuse_names
