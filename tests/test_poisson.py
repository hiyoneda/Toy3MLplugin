"""
Poisson 尤度カーネルの単体テスト。

get_log_like が返す値が、定数項を除いた Poisson 対数尤度と一致することを
scipy.stats.poisson.logpmf の総和と突き合わせて確認する。
"""

import numpy as np
from scipy.stats import poisson


def test_loglike_matches_scipy(make_toy_plugin, truth_model):
    plugin = make_toy_plugin(counts=np.zeros(400))  # counts は後で差し替える
    plugin.set_model(truth_model)
    mu = plugin.expected_counts()

    # 適当な観測を1つ作って、両者の対数尤度の "差" が定数(ln k! の総和)に
    # なることを確認する(get_log_like は ln k! を省いているため)。
    rng = np.random.default_rng(0)
    data = rng.poisson(mu).astype(float)

    plugin._data = data  # テスト用に観測を差し込む
    ours = plugin.get_log_like()

    mu_clip = np.clip(mu, 1e-12, None)
    scipy_full = np.sum(poisson.logpmf(data, mu_clip))
    const = np.sum(-poisson.logpmf(0, 0) * 0)  # 0、明示のためのダミー

    # ours = Σ(k ln mu - mu)、scipy_full = Σ(k ln mu - mu - ln k!)
    ln_factorial = np.sum([np.sum(np.log(np.arange(1, k + 1))) for k in data.astype(int)])
    assert np.isclose(ours, scipy_full + ln_factorial, rtol=1e-9, atol=1e-6)


def test_loglike_maximized_at_truth(make_toy_plugin, truth_model):
    """真の期待値そのものをデータにすると、尤度は近傍で最大になっているはず。"""
    plugin = make_toy_plugin(counts=np.zeros(400))
    plugin.set_model(truth_model)
    mu = plugin.expected_counts()
    plugin._data = mu.copy()  # 期待値=データ(連続近似)

    ll0 = plugin.get_log_like()

    # norm を少し動かすと尤度が下がることを確認(K を 1.2倍)
    truth_model.SRC_A.spectrum.main.Powerlaw.K.value *= 1.2
    ll1 = plugin.get_log_like()
    assert ll1 < ll0
