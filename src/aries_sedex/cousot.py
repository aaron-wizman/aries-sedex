"""Semi-static no-arbitrage conditions on bid and ask call prices (Cousot-type tests).

These pairwise and triplewise conditions are necessary for the absence of
arbitrage. They do not use the quoted sizes and are reported as a diagnostic;
the exact test is ARIES (:mod:`aries_sedex.aries`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CousotReport:
    """Violations found by :func:`cousot_check`.

    Attributes
    ----------
    intrinsic : DataFrame
        Asks at or below the intrinsic value ``exp(-q T) - K exp(-r T)``, or at or below zero
        (columns ``condition, strike, value``).
    vertical : DataFrame
        Pairs ``K_i < K_j`` with ``ask(K_i) - bid(K_j) <= eps`` (columns ``strike_i, strike_j, value``).
    butterfly : DataFrame
        Triplets ``K_i < K_j < K_k`` with
        ``(ask(K_i) - bid(K_j)) / (K_j - K_i) - (bid(K_j) - ask(K_k)) / (K_k - K_j) <= eps``
        (columns ``strike_i, strike_j, strike_k, value``).
    """

    intrinsic: pd.DataFrame
    vertical: pd.DataFrame
    butterfly: pd.DataFrame

    @property
    def n_violations(self) -> int:
        return len(self.intrinsic) + len(self.vertical) + len(self.butterfly)

    def summary(self) -> dict:
        return {"intrinsic": len(self.intrinsic), "vertical": len(self.vertical),
                "butterfly": len(self.butterfly)}


def cousot_check(strikes, bids, asks, r: float, q: float, maturity: float, eps: float = 1e-8) -> CousotReport:
    """Check the semi-static conditions on normalized call quotes.

    Parameters
    ----------
    strikes, bids, asks : array_like
        Normalized strikes (increasing) and call bid and ask prices (spot = 1).
    r, q : float
        Interest rate and dividend yield.
    maturity : float
        Time to expiry in years.
    eps : float
        Numerical tolerance: a spread whose value is at most ``eps`` is reported.
    """
    K = np.asarray(strikes, dtype=float)
    b = np.asarray(bids, dtype=float)
    a = np.asarray(asks, dtype=float)
    if np.any(np.diff(K) <= 0):
        raise ValueError("Strikes must be strictly increasing.")
    T = maturity
    discounted_spot = np.exp(-q * T)

    rows = []
    for i in np.flatnonzero(discounted_spot - a >= K * np.exp(-r * T) - eps):
        rows.append(("intrinsic value", K[i], discounted_spot - a[i] - K[i] * np.exp(-r * T)))
    for i in np.flatnonzero(a <= eps):
        rows.append(("nonpositive ask", K[i], a[i]))
    intrinsic = pd.DataFrame(rows, columns=["condition", "strike", "value"])

    # vertical spreads: ask(K_i) - bid(K_j) for i < j
    vs = a[:, None] - b[None, :]
    i_idx, j_idx = np.nonzero(np.triu(vs <= eps, k=1))
    vertical = pd.DataFrame({"strike_i": K[i_idx], "strike_j": K[j_idx], "value": vs[i_idx, j_idx]})

    # butterfly spreads, one middle strike at a time
    frames = []
    for j in range(1, K.size - 1):
        left = (a[:j] - b[j]) / (K[j] - K[:j])
        right = (b[j] - a[j + 1:]) / (K[j + 1:] - K[j])
        cbs = left[:, None] - right[None, :]
        ii, kk = np.nonzero(cbs <= eps)
        if ii.size:
            frames.append(pd.DataFrame({"strike_i": K[ii], "strike_j": K[j],
                                        "strike_k": K[j + 1 + kk], "value": cbs[ii, kk]}))
    butterfly = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=["strike_i", "strike_j", "strike_k", "value"]))
    return CousotReport(intrinsic, vertical, butterfly)
