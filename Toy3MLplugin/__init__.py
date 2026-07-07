"""
Toy3MLplugin — 前方畳み込み解析の共通コア(教材テンプレート)。

シンプルな1次元スキャンのデータ形式で、以下が一式動く:
  3ML プラグイン / 多天体同時フィット / 複数 dataset / TS マップ(localization)。
画像再構成は cosipy を使う(自前で作らない)。cosipy に差し込む DataIF のトイ実装だけ
用意してある(Toy3MLplugin.cosipy_dataif、cosipy が入っている環境でのみ import 可能)。
"""

from .base import ForwardFoldingLike
from .background import BackgroundComponent
from .observation import Observation
from .catalog import load_sources
from .simulate import poisson_realize, simulate_counts
from .tsmap import ts_map_1d
from .instrument import (
    intrinsic_response, make_scan_orbit, save_orbit_fits, load_orbit_fits,
    fold_response, make_folded_response,
)

__all__ = [
    "ForwardFoldingLike",
    "BackgroundComponent",
    "Observation",
    "load_sources",
    "poisson_realize",
    "simulate_counts",
    "ts_map_1d",
    "intrinsic_response",
    "make_scan_orbit",
    "save_orbit_fits",
    "load_orbit_fits",
    "fold_response",
    "make_folded_response",
]

# cosipy DataIF は cosipy が入っている環境でのみ提供(必須依存にはしない)。
try:  # pragma: no cover
    from .cosipy_dataif import DataIF_ToyScan, build_toy_scan_dataif
    __all__ += ["DataIF_ToyScan", "build_toy_scan_dataif"]
except Exception:
    pass
