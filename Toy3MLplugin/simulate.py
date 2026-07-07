"""
Toy3MLplugin.simulate — 注入-回収(injection-recovery)クロージャテストの道具。

真のモデル+レスポンスから期待カウントを作り、Poisson でばらつかせて
"観測データ" を合成する。これを使って「真値 → データ → フィット → 真値が戻るか」
を確認するのが、前方畳み込み解析の正しさを担保する定番テスト。
"""

from __future__ import annotations

import numpy as np


def poisson_realize(expected: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """期待カウント → Poisson サンプル。"""
    return rng.poisson(np.clip(np.asarray(expected, dtype=float), 0, None)).astype(float)


def simulate_counts(plugin, model, rng: np.random.Generator) -> np.ndarray:
    """真のモデルを差し込んだ plugin の期待カウントを Poisson 実現する。

    plugin は set_model 済みでなくてよい(ここで真モデルを set する)。
    背景成分があれば、その norm.value も期待値に含まれる。
    """
    plugin.set_model(model)
    expected = plugin.expected_counts()
    return poisson_realize(expected, rng)
