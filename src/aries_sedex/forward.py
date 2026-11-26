"""Interest rate and dividend yield implied by put-call parity, and call-equivalent quotes.

All quantities are normalized by the spot price (spot = 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.optimize as sco


@dataclass
class RatesEstimate:
    """Result of :func:`estimate_rates`.

    Attributes
    ----------
    r, q : float
        Interest rate and dividend yield (continuously compounded, annualized).
    forward : float
        Normalized forward price ``exp((r - q) T)``.
    n_strikes : int
        Number of strikes quoted for both calls and puts (0 when the parity fit is skipped).
    method : str
        ``"parity"`` (weighted least squares) or ``"zero"`` (maturity below the threshold).
    """

    r: float
    q: float
    forward: float
    n_strikes: int
    method: str


def _parity_objective(params, strikes, mid, weights):
    slope, intercept = params
    model_val = intercept + slope * strikes
    return np.sum(weights * (model_val - mid) ** 2)


def estimate_rates(calls: pd.DataFrame, puts: pd.DataFrame, maturity: float, atm_offset: float,
                   alpha: float = 0.5, r0: float = 0.05, q0: float = 0.02,
                   rate_bounds: Tuple[float, float] = (0.0, 0.15),
                   dividend_bounds: Tuple[float, float] = (0.0, 0.10),
                   zero_rate_maturity: float = 1.0 / 365.0) -> RatesEstimate:
    """Estimate (r, q) from put-call parity by weighted least squares.

    On the strikes quoted for both calls and puts, the parity mid
    ``0.5 [(C_bid - P_ask) + (C_ask - P_bid)]`` is regressed on the strike,
    ``C - P = exp(-q T) - K exp(-r T)``, with weights that combine the inverse of
    the total spread and the proximity to the money:

    ``w = (1 - alpha) w_spread / max(w_spread) + alpha w_atm / max(w_atm)``, with
    ``w_spread = 1 / spread`` and ``w_atm = 1 / (|K - 1| + atm_offset)``.

    Parameters
    ----------
    calls, puts : DataFrame
        Normalized quotes indexed by strike (see :func:`aries_sedex.quotes.normalized_calls_and_puts`).
    maturity : float
        Time to expiry in years.
    atm_offset : float
        Offset of the moneyness weight; the paper uses one strike increment over the spot.
    alpha : float
        Weight of the moneyness term (0: spread only, 1: moneyness only).
    r0, q0 : float
        Starting values of the optimizer.
    rate_bounds, dividend_bounds : (float, float)
        Bounds on r and q.
    zero_rate_maturity : float
        Below this maturity the fit is skipped and r = q = 0.

    Notes
    -----
    At short maturities the slope ``-exp(-r T)`` is weakly identified, so r and q
    taken separately are imprecise, while the forward ``exp((r - q) T)`` is well
    determined near the money. A user who prefers other values can bypass this
    estimate (see :meth:`aries_sedex.OptionArbitrageModel.set_rates`).
    """
    T = maturity
    if T < zero_rate_maturity:
        return RatesEstimate(0.0, 0.0, 1.0, 0, "zero")

    strikes = np.sort(np.intersect1d(calls.index.values, puts.index.values))
    if strikes.size < 2:
        raise ValueError("Put-call parity needs at least two strikes quoted for both calls and puts.")
    cal_calls, cal_puts = calls.loc[strikes], puts.loc[strikes]
    call_b, call_a = cal_calls["bid"].values, cal_calls["ask"].values
    put_b, put_a = cal_puts["bid"].values, cal_puts["ask"].values

    mid = 0.5 * ((call_b - put_a) + (call_a - put_b))
    spread = (call_a - call_b) + (put_a - put_b)
    if np.any(spread <= 0):
        raise ValueError("Put-call parity weights need a positive total spread at every strike.")
    raw_w_spread = 1.0 / spread
    raw_w_atm = 1.0 / (np.abs(strikes - 1.0) + atm_offset)
    weights = (1 - alpha) * raw_w_spread / np.max(raw_w_spread) + alpha * raw_w_atm / np.max(raw_w_atm)

    (r_inf, r_sup), (q_inf, q_sup) = rate_bounds, dividend_bounds
    x0 = [-np.exp(-r0 * T), np.exp(-q0 * T)]
    bounds = [(-np.exp(-r_inf * T), -np.exp(-r_sup * T)), (np.exp(-q_sup * T), np.exp(-q_inf * T))]
    res = sco.minimize(_parity_objective, x0=x0, bounds=bounds, args=(strikes, mid, weights), method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("Put-call parity fit failed: " + str(res.message))

    slope, intercept = res.x
    r, q = (-1 / T) * np.log(-slope), (-1 / T) * np.log(intercept)
    return RatesEstimate(float(r), float(q), float(np.exp((r - q) * T)), int(strikes.size), "parity")


def call_equivalent_quotes(calls: pd.DataFrame, puts: pd.DataFrame, r: float, q: float,
                           maturity: float) -> pd.DataFrame:
    """Out-of-the-money quotes expressed as call quotes.

    Puts with strike below the forward ``F = exp((r - q) T)`` are converted by
    put-call parity, ``C = P + exp(-q T) - K exp(-r T)`` (bid and ask shifted by the
    same amount, sizes unchanged); calls with strike at or above the forward are kept.

    Returns
    -------
    DataFrame indexed by normalized strike with the columns
    ``ask, bid, ask_size, bid_size, source`` (``source`` is ``"put"`` or ``"call"``).
    """
    T = maturity
    forward = np.exp((r - q) * T)

    otm_puts = puts[puts.index < forward].copy()
    shift = pd.Series(np.exp(-q * T) - otm_puts.index.to_numpy() * np.exp(-r * T), index=otm_puts.index)
    put_to_call = otm_puts.copy()
    put_to_call[["ask", "bid"]] = put_to_call[["ask", "bid"]].add(shift, axis=0)
    put_to_call["source"] = "put"

    otm_calls = calls[calls.index >= forward].copy()
    otm_calls["source"] = "call"
    if put_to_call.empty and otm_calls.empty:
        raise ValueError("No out-of-the-money quote.")
    return pd.concat([put_to_call, otm_calls], axis=0).sort_index()
