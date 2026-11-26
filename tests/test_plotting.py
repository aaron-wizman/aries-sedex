import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from aries_sedex import OptionArbitrageModel, plotting  # noqa: E402


@pytest.fixture
def fitted(bs_slice):
    m = OptionArbitrageModel(bs_slice, spot=1000.0, maturity=7 / 365)
    m.set_rates(0.04, 0.01)
    m.run(rates=(0.04, 0.01), sedex_kwargs={"delta": 0.02})
    return m


def test_density_window_is_centered(fitted):
    a, b = plotting.density_window(fitted)
    assert np.isclose(0.5 * (a + b), fitted.sedex_result.forward)


def test_smile_extends_beyond_quotes(fitted):
    k = fitted.aries_result.quotes.index.values
    s = plotting.smile_strikes(fitted)
    g = fitted.sedex_result.grid
    assert s.min() < k.min() and s.max() > k.max()
    assert s.min() >= g[0] and s.max() <= g[-1]


def test_figures_build(fitted):
    for fn in (plotting.plot_density, plotting.plot_smile, plotting.plot_iv_outside_bidask,
               plotting.plot_aries, plotting.plot_overview):
        fig = fn(fitted)
        assert fig.axes
        matplotlib.pyplot.close(fig)
