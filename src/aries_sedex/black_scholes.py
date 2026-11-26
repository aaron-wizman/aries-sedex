"""Black-Scholes-Merton prices and implied volatilities.

The implied volatility is obtained by Brent's method on the price of the
out-of-the-money option (an in-the-money price is first converted by subtracting
the intrinsic value of the forward), which keeps the root search well conditioned
for the very short maturities studied in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

_FLAGS = {"c": True, "call": True, "p": False, "put": False}

#: Relative tolerance of the root search (the smallest value accepted by ``brentq``).
_RTOL = 4.0 * np.finfo(float).eps


def _is_call(flag) -> bool:
    try:
        return _FLAGS[str(flag).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown option flag {flag!r}: use 'c' or 'p'.") from exc


def _undiscounted_price(forward: float, strike: float, total_vol: float, is_call: bool) -> float:
    """Undiscounted Black price with total volatility ``total_vol = sigma * sqrt(T)``."""
    d1 = (np.log(forward / strike) + 0.5 * total_vol * total_vol) / total_vol
    d2 = d1 - total_vol
    if is_call:
        return forward * ndtr(d1) - strike * ndtr(d2)
    return strike * ndtr(-d2) - forward * ndtr(-d1)


def bsm_price(flag, spot, strike, maturity, rate, dividend_yield, sigma):
    """Black-Scholes-Merton price of European calls or puts.

    Parameters
    ----------
    flag : {"c", "p"} or array of flags
    spot, strike : float or array
    maturity : float
        Time to expiry in years.
    rate, dividend_yield : float
        Continuously compounded interest rate and dividend yield.
    sigma : float or array
        Annualized volatility.

    Returns
    -------
    ndarray
    """
    flag, spot, strike, sigma = np.broadcast_arrays(
        np.asarray(flag, dtype=object), np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float), np.asarray(sigma, dtype=float))
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    total_vol = sigma * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol
    call = np.array([_is_call(f) for f in flag.ravel()]).reshape(flag.shape)
    undiscounted = np.where(call,
                            forward * ndtr(d1) - strike * ndtr(d2),
                            strike * ndtr(-d2) - forward * ndtr(-d1))
    return np.exp(-rate * maturity) * undiscounted


def implied_volatility(price, spot, strike, maturity, rate, dividend_yield, flag="c",
                       max_total_vol: float = 50.0):
    """Black-Scholes-Merton implied volatility.

    Parameters
    ----------
    price : float or array
        Option prices (same units as ``spot`` and ``strike``).
    spot, strike : float or array
    maturity : float
        Time to expiry in years.
    rate, dividend_yield : float
    flag : {"c", "p"} or array of flags
    max_total_vol : float
        Largest total volatility ``sigma * sqrt(T)`` explored by the root search.

    Returns
    -------
    ndarray
        Implied volatilities; ``nan`` where the price is outside the no-arbitrage
        bounds of the Black-Scholes-Merton model (at or below the intrinsic value
        of the forward, or at or above its upper bound).
    """
    price, spot, strike, flag = np.broadcast_arrays(
        np.asarray(price, dtype=float), np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float), np.asarray(flag, dtype=object))
    growth = np.exp(rate * maturity)
    sqrt_t = np.sqrt(maturity)
    out = np.full(price.shape, np.nan)
    for idx in np.ndindex(price.shape):
        forward = float(spot[idx]) * np.exp((rate - dividend_yield) * maturity)
        k = float(strike[idx])
        target = float(price[idx]) * growth
        is_call = _is_call(flag[idx])
        # convert an in-the-money price into the price of the out-of-the-money option
        if is_call and k < forward:
            target, is_call = target - (forward - k), False
        elif not is_call and k > forward:
            target, is_call = target - (k - forward), True
        upper = forward if is_call else k
        if not (np.isfinite(target) and 0.0 < target < upper):
            continue

        def excess(total_vol):
            return _undiscounted_price(forward, k, total_vol, is_call) - target

        lo, hi = 1e-12, 1.0
        while excess(hi) <= 0.0:
            lo, hi = hi, 2.0 * hi
            if hi > max_total_vol:
                break
        else:
            out[idx] = brentq(excess, lo, hi, xtol=1e-300, rtol=_RTOL, maxiter=500) / sqrt_t
    return out
