"""
Toy3MLplugin.observation — 観測データのコンテナ。

基底 Observation は「データ空間のカウント」と「露出」だけを持つ最小構造。

実際のケースでは以下のようになるだろう.
  NinjaSat-2 : 軌道ごとのスキャンデータ。
               長いライトカーブを軌道で区切った複数 Observation を DataList に
               並べて同時フィット、という Fermi 的運用になる。
  SMILE-3    : ETCCのデータ空間を定義したうえで、ここに data arrayが入る.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    """データ空間カウントと露出を持つ最小コンテナ。"""

    counts: np.ndarray                 # データ空間の観測カウント (n_data_bins,)
    exposure: np.ndarray               # データビンごとの露出 [s] (n_data_bins,) or scalar
                                       # exposure は毎回いるわけではないと思う.

    def __post_init__(self):
        self.counts = np.asarray(self.counts, dtype=float).ravel()
        self.exposure = np.broadcast_to(
            np.asarray(self.exposure, dtype=float), self.counts.shape
        ).copy()
