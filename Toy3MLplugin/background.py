"""
Toy3MLplugin.background — 背景成分。

物理的には2系統ある:
  - diffuse   : 空の広がった放射。本来レスポンスを通って畳まれるべき。
                → セットアップ時に一度だけ空マップを fold して template を作る。
  - instrumental : 検出器フレームの背景。最初からデータ空間に直接足す。

だが実行時はどちらも「データ空間テンプレ × 自由 norm」で加算するだけ。
経路を統一しておくと速いし、norm を nuisance parameter にすれば 3ML が
勝手にフィット/プロファイルしてくれる。物理の違いは template の作り方に閉じ込める。
"""

from __future__ import annotations

import numpy as np
from astromodels import Parameter


class BackgroundComponent:
    def __init__(
        self,
        name: str,
        template: np.ndarray,
        value: float = 1.0,
        bounds: tuple = (0.0, 1e3),
        free: bool = True,
    ):
        self.name = name
        self.template = np.asarray(template, dtype=float).ravel()
        # norm は astromodels の Parameter。3ML はこれを nuisance として最適化する。
        self.norm = Parameter(
            f"bkg_{name}_norm",
            value,
            min_value=bounds[0],
            max_value=bounds[1],
            free=free,
            desc=f"normalization of background component '{name}'",
        )
