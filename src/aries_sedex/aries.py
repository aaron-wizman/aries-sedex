"""ARIES: iterative removal of the quotes that support a static arbitrage.

At each iteration ARIES solves the executable arbitrage detection problem: a
linear program over the quantities sold at the bid (``lambda``), bought at the ask
(``mu``), a position ``kappa`` in the underlying and a cash amount ``alpha``,

    maximize    lambda . bid - mu . ask - kappa exp(-q T) - alpha
    subject to  sum_j (mu_j - lambda_j) (s - K_j)^+ + alpha exp(r T) + kappa s >= 0   at s = K_1, ..., K_n,
                sum_j (mu_j - lambda_j) + kappa >= 0                                  (s -> infinity),
                0 <= lambda <= bid_size,   0 <= mu <= ask_size,   alpha >= 0,

where prices and strikes are normalized by the spot. Two cases arise.

* Positive optimal value: a strong arbitrage. At least one size bound is
  saturated; the saturated quote with the smallest size is removed.
* Zero optimal value: either no arbitrage, or an arbitrage with zero cost and a
  nonnegative payoff that is positive in some state. The optimal portfolio is
  scaled until a size bound is saturated; if the scaled portfolio has a cost of at
  most ``eps_cost`` and a payoff above ``eps_cost`` in some state, the saturated
  quote with the smallest size is removed. Otherwise the procedure stops.

Only one side (bid or ask) of one strike is removed per iteration. The default
solver is the interior point method of HiGHS without crossover, which returns a
point in the relative interior of the optimal face: it is nonzero whenever an
arbitrage with zero cost exists, whereas a vertex solution may be the null portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: HiGHS interior point, no crossover, duality-gap tolerance 1e-10.
DEFAULT_HIGHS_OPTIONS = {"highs_options": {"solver": "ipm", "run_crossover": "off",
                                           "ipm_optimality_tolerance": 1e-10}}

REMOVAL_COLUMNS = ["iteration", "kind", "profit", "strike", "side", "size"]


@dataclass
class AriesResult:
    """Output of :func:`aries`.

    Attributes
    ----------
    quotes : DataFrame
        Quotes kept by ARIES (both sizes positive), same columns as the input.
    removals : DataFrame
        One row per removed quote: ``iteration``, ``kind`` (``"strong"`` or ``"standard"``),
        ``profit`` (optimal value of the linear program, normalized units), ``strike``
        (normalized), ``side`` (``"bid"`` or ``"ask"``) and ``size`` (size before removal).
    n_iterations : int
        Number of linear programs solved.
    """

    quotes: pd.DataFrame
    removals: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=REMOVAL_COLUMNS))
    n_iterations: int = 0

    @property
    def n_removed_strikes(self) -> int:
        """Number of distinct strikes with at least one removed side."""
        return int(self.removals["strike"].nunique()) if len(self.removals) else 0


def _saturated(x, Q, sat_rtol):
    """A size bound is saturated if ``|Q - x| <= sat_rtol * Q``. Quotes with ``Q = 0`` are never saturated."""
    return np.isclose(x, Q, rtol=sat_rtol, atol=0.0) & (Q > 0)


def _pick_smallest_from_saturated(sat_bid, sat_ask, BS, AS):
    """Saturated quote with the smallest size, as ``(local index, side, size)``, or ``None``."""
    cand_sizes, cand_sides, cand_idx = [], [], []
    if np.any(sat_bid):
        idx = np.where(sat_bid)[0]
        cand_sizes.append(BS[idx])
        cand_sides += ["bid"] * idx.size
        cand_idx += list(idx)
    if np.any(sat_ask):
        idx = np.where(sat_ask)[0]
        cand_sizes.append(AS[idx])
        cand_sides += ["ask"] * idx.size
        cand_idx += list(idx)
    if len(cand_sizes) == 0:
        return None
    sizes = np.concatenate(cand_sizes)
    k = int(np.argmin(sizes))
    return cand_idx[k], cand_sides[k], float(sizes[k])


def _scaling_factor(lam_v, mu_v, BS, AS, eps_lp):
    """Largest factor ``s`` such that ``s * lambda <= bid_size`` and ``s * mu <= ask_size``, or ``None``."""
    pos_mu = mu_v > eps_lp
    pos_lam = lam_v > eps_lp
    s_candidates = []
    if np.any(pos_mu):
        s_candidates.append(float(np.min(AS[pos_mu] / mu_v[pos_mu])))
    if np.any(pos_lam):
        s_candidates.append(float(np.min(BS[pos_lam] / lam_v[pos_lam])))
    if len(s_candidates) == 0:
        return None
    return float(np.min(s_candidates))


def _check_cvxpy_for_highs_options(solver_opts):
    if solver_opts and "highs_options" in solver_opts:
        version = tuple(int(x) for x in cp.__version__.split(".")[:3] if x.isdigit())
        if version < (1, 6, 5):
            raise RuntimeError(f"ARIES: cvxpy {cp.__version__} cannot pass HiGHS options; "
                               f"cvxpy >= 1.6.5 is required.")


def aries(quotes: pd.DataFrame, r: float, q: float, maturity: float, eps_lp: float = 1e-8,
          eps_cost: float = 1e-8, sat_rtol: float = 1e-4, solver: str = "HIGHS",
          solver_opts: Optional[dict] = None) -> AriesResult:
    """Remove the quotes that support a static arbitrage.

    Parameters
    ----------
    quotes : DataFrame
        Normalized call quotes indexed by strike, with the columns ``bid, ask, bid_size, ask_size``
        (see :func:`aries_sedex.forward.call_equivalent_quotes`).
    r, q : float
        Interest rate and dividend yield.
    maturity : float
        Time to expiry in years.
    eps_lp : float
        Threshold on the optimal value: above it the arbitrage is strong; in absolute value
        below it the optimal value is zero. Also the threshold on a positive quantity.
    eps_cost : float
        Tolerance on the cost and on the payoff of the scaled portfolio (zero-value case).
    sat_rtol : float
        Relative tolerance of the saturation test ``|Q - x| <= sat_rtol * Q``.
    solver : str
        CVXPY solver name. ``"HIGHS"`` (default) uses :data:`DEFAULT_HIGHS_OPTIONS` unless
        ``solver_opts`` is given.
    solver_opts : dict, optional
        Keyword arguments passed to ``cvxpy.Problem.solve``.

    Raises
    ------
    RuntimeError
        If a linear program fails or returns a status other than ``"optimal"``.
    """
    df_opt = quotes.copy()
    r, q, T = float(r), float(q), float(maturity)
    new_spot = 1.0

    # quotes without any size cannot be traded
    df_opt = df_opt[~((df_opt.bid_size == 0) & (df_opt.ask_size == 0))].copy()
    if df_opt.empty:
        raise ValueError("ARIES: no quote with a positive size.")

    bids = df_opt["bid"].copy()
    asks = df_opt["ask"].copy()
    bid_size = df_opt["bid_size"].copy()
    ask_size = df_opt["ask_size"].copy()
    removals = []

    if solver == "HIGHS" and solver_opts is None:
        solver_opts = DEFAULT_HIGHS_OPTIONS
    _check_cvxpy_for_highs_options(solver_opts)

    max_iter = 2 * len(df_opt.index)
    it = 0
    stopped = False
    while it < max_iter:
        it += 1

        active = ~((bid_size == 0) & (ask_size == 0))
        if not active.any():
            logger.info("No active quote left.")
            stopped = True
            break

        strikes = bid_size.index[active]
        n = len(strikes)
        B = bids.loc[strikes].to_numpy(float)
        A = asks.loc[strikes].to_numpy(float)
        BS = bid_size.loc[strikes].to_numpy(float)
        AS = ask_size.loc[strikes].to_numpy(float)
        Kvals = strikes.to_numpy(float)

        # payoff_matrix[i, j] = (K_i - K_j)^+ : payoff of the call of strike K_j in state s = K_i
        payoff_matrix = np.maximum(Kvals[:, None] - Kvals[None, :], 0.0)

        lambda_ = cp.Variable(n, nonneg=True)
        mu = cp.Variable(n, nonneg=True)
        kappa = cp.Variable()
        alpha = cp.Variable(nonneg=True)

        disc_q = float(np.exp(-q * T))
        grow_r = float(np.exp(r * T))
        profit0 = lambda_ @ B - mu @ A - kappa * new_spot * disc_q - alpha
        payoffs = payoff_matrix @ (mu - lambda_) + grow_r * alpha + kappa * Kvals

        cons = [
            payoffs >= 0,
            cp.sum(mu - lambda_) + kappa >= 0,
            lambda_ <= BS,
            mu <= AS,
        ]
        prob = cp.Problem(cp.Maximize(profit0), cons)

        try:
            val = prob.solve(solver=solver, verbose=False, **(solver_opts or {}))
        except (cp.error.SolverError, ValueError) as e:
            raise RuntimeError(f"ARIES: solver failure at iteration {it} ({e})") from e
        if prob.status != cp.OPTIMAL:
            raise RuntimeError(f"ARIES: solver returned status '{prob.status}' at iteration {it}")

        lam_v = np.array(lambda_.value, dtype=float)
        mu_v = np.array(mu.value, dtype=float)
        kappa_v = float(kappa.value if kappa.value is not None else 0.0)
        alpha_v = float(alpha.value if alpha.value is not None else 0.0)
        profit_opt = float(val)

        sat_bid = _saturated(lam_v, BS, sat_rtol)
        sat_ask = _saturated(mu_v, AS, sat_rtol)

        if profit_opt > eps_lp:
            # strong arbitrage; an interior-point optimum approaches the size bounds
            # without reaching them: if none is saturated, scale the portfolio first
            kind = "strong"
            choice = _pick_smallest_from_saturated(sat_bid, sat_ask, BS, AS)
            if choice is None:
                s = _scaling_factor(lam_v, mu_v, BS, AS, eps_lp)
                if s is not None:
                    choice = _pick_smallest_from_saturated(_saturated(lam_v * s, BS, sat_rtol),
                                                           _saturated(mu_v * s, AS, sat_rtol), BS, AS)
            if choice is None:
                raise RuntimeError(f"ARIES: strong arbitrage without saturated size at iteration {it}")

        elif abs(profit_opt) <= eps_lp:
            # zero optimal value: arbitrage with zero cost, or no arbitrage
            kind = "standard"
            choice = _pick_smallest_from_saturated(sat_bid, sat_ask, BS, AS)
            if choice is None:
                s = _scaling_factor(lam_v, mu_v, BS, AS, eps_lp)
                if s is None:
                    logger.info(f"No arbitrage (optimal value {profit_opt:.3e}, null portfolio).")
                    stopped = True
                    break
                # the scaled portfolio must be an arbitrage: cost <= eps_cost and
                # payoff > eps_cost in some state (s = 0, s = K_j or s -> infinity)
                cost_scaled = -s * profit_opt
                pay_scaled = s * max(float(np.max(payoffs.value)), grow_r * alpha_v,
                                     float(np.sum(mu_v - lam_v) + kappa_v))
                if cost_scaled > eps_cost or pay_scaled <= eps_cost:
                    logger.info(f"No arbitrage (optimal value {profit_opt:.3e}; scaled portfolio: "
                                f"cost {cost_scaled:.3e}, payoff {pay_scaled:.3e}).")
                    stopped = True
                    break
                choice = _pick_smallest_from_saturated(_saturated(lam_v * s, BS, sat_rtol),
                                                       _saturated(mu_v * s, AS, sat_rtol), BS, AS)
            if choice is None:
                raise RuntimeError(f"ARIES: arbitrage without saturated size after scaling at iteration {it}")

        else:
            # the null portfolio is feasible with value 0: a negative value is a solver inaccuracy
            raise RuntimeError(f"ARIES: negative optimal value ({profit_opt:.3e}) at iteration {it}")

        idx_local, side, size_before = choice
        g_idx = strikes[idx_local]
        if side == "bid":
            bid_size.loc[g_idx] = 0.0
        else:
            ask_size.loc[g_idx] = 0.0
        removals.append({"iteration": it, "kind": kind, "profit": profit_opt,
                         "strike": float(g_idx), "side": side, "size": float(size_before)})
        logger.info(f"{kind.capitalize()} arbitrage (optimal value {profit_opt:.3e}): removed {side} "
                    f"at K/S0 = {float(g_idx):.6g} (size {size_before:.6g}).")

    if not stopped:
        logger.warning(f"ARIES: iteration limit ({max_iter}) reached.")

    df_opt["bid_size"] = bid_size
    df_opt["ask_size"] = ask_size
    kept = df_opt[(df_opt.bid_size > 0) & (df_opt.ask_size > 0)].copy()
    removals = pd.DataFrame(removals, columns=REMOVAL_COLUMNS)
    logger.info(f"ARIES: {len(kept)} strikes kept, {len(removals)} quote(s) removed.")
    return AriesResult(quotes=kept, removals=removals, n_iterations=it)
