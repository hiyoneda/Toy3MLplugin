"""
Toy3MLplugin.base — 前方畳み込み(forward folding)プラグインの共通基底。

このファイルが「型」の中心。装置ごとに変化する重要なサブクラスは、以下の2メソッド:

    _load_response(self)                    ... 応答に必要なデータを obs から取り出す
    _response_for_source(self, source)      ... 点源1個 → データ空間の応答行列 R[data_bin, e]

尤度・多天体の重ね合わせ・背景・3ML連携は、すべてこのbase classが面倒を見る。
NinjaSat-2(スキャン→ライトカーブ)も SMILE-3(広視野撮像)も、
"データ空間の軸が何か" と "R の中身" が違うだけで、この骨格は共通。

    mu[data_bin] = Σ_source  R_src[data_bin, e] @ F_src[e]  +  Σ_bkg  norm · template[data_bin]

観測カウントと mu を Poisson 尤度で比べる。それだけ。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional

import numpy as np
from astromodels import Model, PointSource, Parameter
from threeML.plugin_prototype import PluginPrototype

from .background import BackgroundComponent
from .observation import Observation


def _rename_parameter(p: Parameter, new_name: str) -> Parameter:
    """astromodels の Parameter は name を再代入できないので、
    同じ属性で名前だけ変えた Parameter を作り直す。"""
    return Parameter(
        new_name,
        p.value,
        min_value=p.min_value,
        max_value=p.max_value,
        free=p.free,
        desc=getattr(p, "description", ""),
    )


class ForwardFoldingLike(PluginPrototype):
    """ビン化・多天体の前方畳み込み + Poisson 尤度を実装した 3ML プラグイン基底。

    3ML の PluginPrototype が要求する抽象メソッドは set_model / get_log_like /
    inner_fit の3つ。ここで全部埋める。そのうえで装置固有の2メソッドを
    新たに抽象化して、サブクラスに委ねる。
    """

    def __init__(
        self,
        name: str,
        observation: Observation,
        energy_edges: np.ndarray,
        backgrounds: Optional[List[BackgroundComponent]] = None,
    ):
        # --- データ空間(簡単のため1次元に潰したベクトル) ---
        self._obs: Observation = observation
        self._data: np.ndarray = np.asarray(observation.counts, dtype=float).ravel()

        # --- エネルギービン境界。ここは、astropy の 単位付きにするなど、高級にしていってもよい.
        self._e_edges: np.ndarray = np.asarray(energy_edges, dtype=float)
        assert self._e_edges.ndim == 1 and self._e_edges.size >= 2

        # --- 背景成分(データ空間テンプレ × 自由 norm) ---
        self._backgrounds: List[BackgroundComponent] = list(backgrounds or [])
        for b in self._backgrounds:
            if b.template.size != self._data.size:
                raise ValueError(
                    f"background '{b.name}' template size {b.template.size} "
                    f"!= data size {self._data.size}"
                )

        self._model: Optional[Model] = None
        self._resp_cache: Dict[tuple, np.ndarray] = {}

        # 装置固有の準備(応答データの取り込み)
        self._load_response()

        # 背景 norm を 3ML の nuisance parameter として公開すると、
        # JointLikelihood がスペクトルと一緒にフィット/プロファイルしてくれる。
        # 3ML は「nuisance 名にプラグイン名が含まれること」を要求する
        # (複数 Observation を並べたとき norm 同士が衝突しないように)。
        # → プラグイン名を接頭辞にした Parameter に作り直す。
        for b in self._backgrounds:
            b.norm = _rename_parameter(b.norm, f"{name}_{b.norm.name}")
        nuisance = {b.norm.name: b.norm for b in self._backgrounds}
        super().__init__(name, nuisance)

    # ------------------------------------------------------------------
    # 装置ごとに実装する2メソッド
    # ------------------------------------------------------------------
    @abstractmethod
    def _load_response(self) -> None:
        """self._obs から応答計算に必要な補助データを取り出して属性に持たせる。

        例: NinjaSat なら姿勢履歴・スリット角度応答・有効面積。
        """

    @abstractmethod
    def _response_for_source(self, source: PointSource) -> np.ndarray:
        """位置固定の点源1個に対する応答行列 R を返す。

        戻り値 shape は必ず (n_data_bins, n_e)。
        R[i, e] は「エネルギービン e の photon flux 1単位が、データビン i に
        何カウント落ちるか」を表す(有効面積×露出×装置応答をすべて畳んだもの)。
        位置固定なら R はモデルのスペクトルパラメータに依存しないので、
        基底側で位置キーごとにキャッシュされる。
        """

    # ------------------------------------------------------------------
    # 以下すべて完全共通(サブクラスは触らない)
    # ------------------------------------------------------------------
    def set_model(self, model: Model) -> None:
        self._model = model
        # 位置を固定運用しているなら実質一度きりしか埋まらない。
        # 位置を自由にする場合（点源の発見など）では、_response_for_source が位置に依存するため
        # キーが変わって再計算される(下の _expected 参照)。
        self._resp_cache.clear()

    def _flux_in_bins(self, source: PointSource) -> np.ndarray:
        """astromodels のスペクトル関数をエネルギービンで積分して photon flux ベクトルに。

        いまは中点xビン幅の矩形近似。精度が要るなら, scipyを使った積分などに差し替える
        """
        centers = 0.5 * (self._e_edges[:-1] + self._e_edges[1:])
        widths = np.diff(self._e_edges)
        return np.asarray(source(centers), dtype=float) * widths  # (n_e,)

    def _expected(self) -> np.ndarray:
        """データ空間の期待カウント mu を組み立てる。"""
        mu = np.zeros(self._data.size, dtype=float)

        # --- 点源の重ね合わせ(多天体同時フィットの本体) ---
        for src in self._model.point_sources.values():
            key = (round(src.position.get_l(), 6), round(src.position.get_b(), 6))
            R = self._resp_cache.get(key)
            if R is None:
                R = self._response_for_source(src)
                if R.shape != (self._data.size, self._e_edges.size - 1):
                    raise ValueError(
                        f"_response_for_source returned {R.shape}, "
                        f"expected {(self._data.size, self._e_edges.size - 1)}"
                    )
                self._resp_cache[key] = R
            # response と source model との畳み込み.
            # SMILE-3だと、ここの計算が重い可能性が高いので、
            # GPUやdeep learningを使った高速化の検討価値あり.
            mu += R @ self._flux_in_bins(src)

        # --- 背景成分 ---
        for b in self._backgrounds:
            mu += b.norm.value * b.template

        return mu

    def get_log_like(self) -> float:
        """Poisson の対数尤度(定数項 ln(k!) は最適化に無関係なので省略)。"""
        mu = np.clip(self._expected(), 1e-12, None)
        return float(np.sum(self._data * np.log(mu) - mu))

    def inner_fit(self) -> float:
        # nuisance(背景 norm)のプロファイルを内側で回すフックだが、
        # 3ML は nuisance も外側の minimizer に渡すので、ここは get_log_like で足りる。
        return self.get_log_like()

    def get_number_of_data_points(self) -> int:
        return int(self._data.size)

    # 便利メソッド(テストや可視化用)
    def expected_counts(self) -> np.ndarray:
        return self._expected()
