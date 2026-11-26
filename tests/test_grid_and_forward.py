import numpy as np
import pytest

from aries_sedex import call_equivalent_quotes, compatible_grid, estimate_rates, normalized_calls_and_puts


def test_grid_contains_every_strike():
    spot = 4987.65
    strikes = np.arange(4800, 5200 + 1, 5) / spot
    grid, step, m = compatible_grid(strikes, spot, 1e-4, 0.9, 1.1, 5.0)
    assert step <= 1e-4 and np.isclose(step * m, 5.0 / spot)
    for k in strikes:
        assert np.min(np.abs(grid - k)) < 1e-12


def test_grid_rejects_strikes_off_the_lattice():
    spot = 5000.0
    with pytest.raises(ValueError):
        compatible_grid(np.array([4997.5]) / spot, spot, 1e-4, 0.9, 1.1, 5.0)


def test_parity_recovers_forward(bs_slice):
    spot, T, r, q = 1000.0, 7 / 365, 0.04, 0.01
    calls, puts = normalized_calls_and_puts(bs_slice, spot)
    est = estimate_rates(calls, puts, T, atm_offset=5.0 / spot)
    assert est.method == "parity"
    assert abs(est.forward - np.exp((r - q) * T)) < 2e-5


def test_zero_rates_below_threshold(bs_slice):
    calls, puts = normalized_calls_and_puts(bs_slice, 1000.0)
    est = estimate_rates(calls, puts, 0.5 / 365, atm_offset=0.005)
    assert (est.r, est.q, est.method) == (0.0, 0.0, "zero")


def test_call_equivalent_quotes_use_parity(bs_slice):
    spot, T, r, q = 1000.0, 7 / 365, 0.04, 0.01
    calls, puts = normalized_calls_and_puts(bs_slice, spot)
    cq = call_equivalent_quotes(calls, puts, r, q, T)
    F = np.exp((r - q) * T)
    low = cq[cq.index < F]
    assert (low["source"] == "put").all() and (cq[cq.index >= F]["source"] == "call").all()
    k = low.index[0]
    expected = puts.loc[k, "bid"] + np.exp(-q * T) - k * np.exp(-r * T)
    assert np.isclose(low.loc[k, "bid"], expected)
