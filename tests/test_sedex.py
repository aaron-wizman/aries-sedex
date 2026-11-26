import os

import numpy as np
import pytest

from aries_sedex import OptionArbitrageModel

SPOT, T, R, Q, SIGMA = 1000.0, 7 / 365, 0.04, 0.01, 0.20


@pytest.fixture
def fitted(bs_slice):
    m = OptionArbitrageModel(bs_slice, spot=SPOT, maturity=T)
    m.set_rates(R, Q)
    m.build_call_quotes()
    m.detect_arbitrage()
    m.extract_density(delta=0.02)          # coarse grid: fast test
    return m


def test_constraints_hold(fitted):
    res = fitted.sedex_result
    info = res.solve_info
    assert info["status_final"] == "optimal"
    assert info["max_bidask_violation_ticks"] <= 0.01
    assert abs(res.masses.sum() - 1) < 1e-5 and abs(res.mean() - res.forward) < 1e-5
    kept = fitted.aries_result.quotes
    price = res.call_prices(kept.index.values)
    tick = 0.05 / SPOT
    assert np.all(price >= kept["bid"].values - 0.01 * tick) and np.all(price <= kept["ask"].values + 0.01 * tick)


def test_density_close_to_lognormal(fitted):
    res = fitted.sedex_result
    v = SIGMA * np.sqrt(T)
    m = (R - Q) * T - 0.5 * v * v
    x = res.grid
    true = np.exp(-(np.log(x) - m) ** 2 / (2 * v * v)) / (x * v * np.sqrt(2 * np.pi))
    assert np.sum(np.abs(res.masses - true * res.step)) < 0.1


def test_strikes_are_grid_points(fitted):
    res = fitted.sedex_result
    for k in fitted.aries_result.quotes.index.values:
        assert np.min(np.abs(res.grid - k)) < 1e-12


def test_default_entropy_weight(fitted):
    res = fitted.sedex_result
    assert np.isclose(res.lambda2, 12 * np.pi * 0.02**3 / res.forward**3, rtol=1e-12)


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1 to run (about 30 seconds)")
def test_example_dataset_full(example_dir):
    import json
    meta = json.loads((example_dir / "synthetic_1dte.json").read_text())
    m = OptionArbitrageModel.from_csv(example_dir / "synthetic_1dte.csv", spot=meta["spot"],
                                      maturity=meta["maturity_years"])
    res = m.run()
    assert res.solve_info["status_final"] == "optimal"
    assert res.solve_info["max_bidask_violation_ticks"] <= 0.01
