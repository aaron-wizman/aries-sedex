import numpy as np

from aries_sedex import bsm_price, implied_volatility


def test_round_trip_short_maturities():
    rng = np.random.default_rng(0)
    for T in (1 / 365, 1 / (365 * 24), 7 / 365):
        K = np.exp(rng.uniform(-0.10, 0.06, 400))
        sigma = rng.uniform(0.05, 1.0, 400)
        flags = rng.choice(["c", "p"], 400)
        price = bsm_price(flags, 1.0, K, T, 0.045, 0.013, sigma)
        F = np.exp((0.045 - 0.013) * T)
        intrinsic = np.exp(-0.045 * T) * np.where(flags == "c", np.maximum(F - K, 0), np.maximum(K - F, 0))
        ok = price - intrinsic > 1e-7          # time value above 1e-7 of the spot
        iv = implied_volatility(price[ok], 1.0, K[ok], T, 0.045, 0.013, flags[ok])
        assert np.max(np.abs(iv - sigma[ok])) < 1e-9


def test_put_call_parity_gives_same_volatility():
    T, r, q, K = 1 / 365, 0.04, 0.01, np.array([0.97, 1.0, 1.02])
    call = bsm_price("c", 1.0, K, T, r, q, 0.15)
    put = bsm_price("p", 1.0, K, T, r, q, 0.15)
    assert np.allclose(implied_volatility(call, 1.0, K, T, r, q, "c"), 0.15, atol=1e-10)
    assert np.allclose(implied_volatility(put, 1.0, K, T, r, q, "p"), 0.15, atol=1e-10)


def test_prices_outside_bounds_return_nan():
    T, r, q = 1 / 365, 0.04, 0.01
    iv = implied_volatility([0.0, -1e-6, 1.5], 1.0, [1.05, 0.95, 1.0], T, r, q, "c")
    assert np.isnan(iv).all()
