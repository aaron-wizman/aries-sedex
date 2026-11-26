"""SEDEx: extraction of a discrete risk-neutral density from arbitrage-free quotes.

The density is a vector of nonnegative masses ``p_j`` on a uniform grid ``s_j`` of
the normalized state (spot = 1) that contains every strike. SEDEx solves

    minimize    lambda_1 sum_j (p_{j+1} - p_j)^2 + lambda_2 sum_j p_j log p_j
    subject to  bid_i <= exp(-r T) sum_j p_j (s_j - K_i)^+ <= ask_i   for every strike K_i,
                sum_j p_j = 1,   sum_j p_j s_j = F = exp((r - q) T),   p >= 0.

Grid. The target mesh is ``Delta s = sigma_ATM sqrt(2 pi T) delta``, where
``sigma_ATM`` is the implied volatility of the mid quote at the first strike at or
above the forward. The mesh is then reduced to a divisor of the strike lattice (see
:func:`aries_sedex.grid.compatible_grid`). The support is
``[min(a_1, a_2), max(b_1, b_2)]`` with

    a_1 = K_min - (kappa_1 - 1)(K_max - K_min),   a_2 = exp(-kappa_2 sigma_ATM sqrt(T)),
    b_1 = K_max + (kappa_1 - 1)(K_max - K_min),   b_2 = exp(+kappa_2 sigma_ATM sqrt(T)).

Weights. With ``lambda_1`` given, the default entropy weight equalizes the first
variations of the two terms at the forward:
``lambda_2 = lambda_1 (Delta s)^3 / ((sqrt(2 pi) / 6) (F sigma_ATM sqrt(T))^3)``,
which equals ``12 pi delta^3 lambda_1 / F^3`` for the automatic mesh.

Acceptance. The solution must have masses above ``-check_tol``, a total mass and
a mean within ``check_tol`` of 1 and F, and model prices within ``check_ticks``
ticks of the bid-ask interval. If the first solve is not optimal or fails these
checks, the problem is solved again with stricter settings (``retry_opts``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import cvxpy as cp
import numpy as np
import pandas as pd

from .black_scholes import implied_volatility
from .grid import compatible_grid

logger = logging.getLogger(__name__)

#: Retry settings of the Clarabel solver.
DEFAULT_CLARABEL_RETRY = {"max_iter": 1000, "tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10,
                          "tol_feas": 1e-10, "time_limit": 120}

REGULARIZATIONS = ("hybrid", "smoothness", "entropy")


def call_payoff_matrix(strikes, states) -> np.ndarray:
    """Matrix ``M[i, j] = (s_j - K_i)^+``."""
    k = np.asarray(strikes, dtype=float).reshape((-1, 1))
    s = np.asarray(states, dtype=float).reshape((1, -1))
    return np.maximum(s - k, 0)


def put_payoff_matrix(strikes, states) -> np.ndarray:
    """Matrix ``M[i, j] = (K_i - s_j)^+``."""
    k = np.asarray(strikes, dtype=float).reshape((-1, 1))
    s = np.asarray(states, dtype=float).reshape((1, -1))
    return np.maximum(k - s, 0)


@dataclass
class SedexResult:
    """Output of :func:`sedex`. States and prices are normalized by the spot.

    Attributes
    ----------
    grid : ndarray
        Grid points ``s_j``.
    masses : ndarray
        Probability masses ``p_j`` (solver output).
    step : float
        Mesh size of the grid.
    refinement : int
        Number of grid intervals per strike increment.
    target_step : float
        Target mesh size before the reduction to a divisor of the strike lattice.
    sigma_atm : float
        Implied volatility used for the mesh and the support.
    atm_strike : float
        Strike at which ``sigma_atm`` is computed.
    forward : float
        Normalized forward ``exp((r - q) T)``.
    r, q, maturity, spot : float
    support : (float, float)
        Preliminary support bounds before the grid is built.
    support_rules : dict
        Rule (``"kappa1"`` or ``"kappa2"``) that sets each support bound.
    regularization : str
    lambda1, lambda2 : float or None
        Weights of the smoothness and entropy terms (``None`` when the term is absent).
    solve_info : dict
        Solver statuses, retry flag and acceptance checks.
    """

    grid: np.ndarray
    masses: np.ndarray
    step: float
    refinement: int
    target_step: float
    sigma_atm: float
    atm_strike: float
    forward: float
    r: float
    q: float
    maturity: float
    spot: float
    support: Tuple[float, float]
    support_rules: dict
    regularization: str
    lambda1: Optional[float]
    lambda2: Optional[float]
    solve_info: dict = field(default_factory=dict)

    @property
    def density(self) -> np.ndarray:
        """Density values ``p_j / step`` with respect to the normalized state."""
        return self.masses / self.step

    @property
    def cdf(self) -> np.ndarray:
        """Cumulative masses."""
        return np.cumsum(self.masses)

    def quantile(self, u) -> np.ndarray:
        """Smallest grid point whose cumulative mass reaches ``u``."""
        idx = np.searchsorted(self.cdf, np.asarray(u, dtype=float))
        return self.grid[np.minimum(idx, self.grid.size - 1)]

    def mean(self) -> float:
        return float(self.masses @ self.grid)

    def call_prices(self, strikes) -> np.ndarray:
        """Normalized call prices ``exp(-r T) sum_j p_j (s_j - K)^+``."""
        return np.exp(-self.r * self.maturity) * (call_payoff_matrix(strikes, self.grid) @ self.masses)

    def put_prices(self, strikes) -> np.ndarray:
        """Normalized put prices ``exp(-r T) sum_j p_j (K - s_j)^+``."""
        return np.exp(-self.r * self.maturity) * (put_payoff_matrix(strikes, self.grid) @ self.masses)

    def to_frame(self, in_index_points: bool = False) -> pd.DataFrame:
        """Grid, masses and density as a table (states in index points if requested)."""
        scale = self.spot if in_index_points else 1.0
        return pd.DataFrame({"state": self.grid * scale, "mass": self.masses,
                             "density": self.density / scale})


def sedex(quotes: pd.DataFrame, r: float, q: float, maturity: float, spot: float, *,
          strike_lattice: float = 5.0, delta: float = 0.005, kappa1: float = 1.55,
          kappa2: float = 10.0, regularization: str = "hybrid", lambda1: float = 1.0,
          lambda2: Optional[float] = None, lambda2_scale: float = 1.0,
          target_step: Optional[float] = None, lower_truncation: bool = True,
          solver: str = "CLARABEL", solver_opts: Optional[dict] = None,
          retry_opts: Union[str, dict, None] = "default", check_tick: float = 0.05,
          check_ticks: float = 0.01, check_tol: float = 1e-5) -> SedexResult:
    """Extract the risk-neutral density from arbitrage-free call quotes.

    Parameters
    ----------
    quotes : DataFrame
        Normalized call quotes indexed by strike with the columns ``bid`` and ``ask``,
        free of arbitrage (output of ARIES).
    r, q : float
        Interest rate and dividend yield.
    maturity : float
        Time to expiry in years.
    spot : float
        Spot price in index points (to place the strikes on the lattice).
    strike_lattice : float
        Strike increment of the exchange, in index points.
    delta : float
        Mesh parameter: ``Delta s = sigma_ATM sqrt(2 pi T) delta``.
    kappa1, kappa2 : float
        Support parameters (extension beyond the quoted strikes; number of ATM standard deviations).
    regularization : {"hybrid", "smoothness", "entropy"}
        Objective: smoothness plus entropy (SEDEx), smoothness only, or entropy only.
    lambda1 : float
        Weight of the smoothness term (hybrid objective).
    lambda2 : float, optional
        Weight of the entropy term. Default: the rule described in the module docstring.
    lambda2_scale : float
        Multiplier applied to the default ``lambda2`` (ignored when ``lambda2`` is given).
    target_step : float, optional
        Target mesh size; overrides the rule based on ``delta``.
    lower_truncation : bool
        If ``False``, the grid starts at the first grid point ``s_1 = step``.
    solver : str
        CVXPY solver supporting the exponential cone (default ``"CLARABEL"``).
    solver_opts : dict, optional
        Keyword arguments of the first solve.
    retry_opts : "default", dict or None
        Settings of the second solve; ``"default"`` uses :data:`DEFAULT_CLARABEL_RETRY` with
        Clarabel and no retry with other solvers; ``None`` disables the retry.
    check_tick : float
        Tick size (index points) used to express the bid-ask check.
    check_ticks : float
        Largest accepted violation of the bid-ask interval, in ticks.
    check_tol : float
        Tolerance of the checks on the masses, the total mass and the mean.

    Raises
    ------
    RuntimeError
        If the problem cannot be solved or the solution fails the checks.
    """
    if regularization not in REGULARIZATIONS:
        raise ValueError(f"Unknown regularization {regularization!r}; use one of {REGULARIZATIONS}.")
    strikes, asks, bids = quotes.index.values, quotes["ask"].values, quotes["bid"].values
    r, q, T = float(r), float(q), float(maturity)
    new_spot = 1.0
    forward_price = new_spot * np.exp((r - q) * T)

    above = quotes[quotes.index >= forward_price]
    if above.empty:
        raise ValueError("SEDEx needs a quote at or above the forward to set the ATM volatility.")
    atm_quote = above.iloc[0][["bid", "ask"]]
    atm_mid = atm_quote.mean()
    atm_vol = float(implied_volatility(atm_mid, new_spot, atm_quote.name, T, r, q, "c"))
    if not np.isfinite(atm_vol):
        raise ValueError("The ATM mid price has no implied volatility.")

    if target_step is None:
        target_step = (atm_vol * np.sqrt(T)) * np.sqrt(2 * np.pi) * delta

    s_min, s_max = strikes[0], strikes[-1]
    f = kappa1
    lower1 = f * s_min - (f - 1) * s_max
    upper1 = f * s_max - (f - 1) * s_min
    lower2 = new_spot * np.exp(-atm_vol * np.sqrt(T) * kappa2)
    upper2 = new_spot * np.exp(+atm_vol * np.sqrt(T) * kappa2)
    lower, upper = min(lower1, lower2), max(upper1, upper2)
    if not lower_truncation:
        lower = 0.0
    support_rules = {"lower": "kappa1" if lower1 <= lower2 else "kappa2",
                     "upper": "kappa1" if upper1 >= upper2 else "kappa2"}

    xr, grid_step, refinement = compatible_grid(strikes, spot, target_step, lower, upper, strike_lattice)
    logger.info(f"SEDEx grid: step {grid_step:.3e} (strike lattice / {refinement}), {len(xr)} points.")

    M = call_payoff_matrix(strikes, xr)
    M_comb = np.exp(-r * T) * np.vstack([M, -M])
    bounds_comb = np.hstack([asks, -bids])

    x = cp.Variable(len(xr), nonneg=True)
    lam1, lam2 = None, None
    if regularization == "smoothness":
        objective = cp.Minimize(cp.sum_squares(x[1:] - x[:-1]))
    elif regularization == "entropy":
        objective = cp.Minimize(-cp.sum(cp.entr(x)))
    else:
        lam1 = float(lambda1)
        if lambda2 is None:
            s_tot = atm_vol * np.sqrt(T)
            lam2 = lam1 * target_step**3 / (np.sqrt(2 * np.pi) / 6 * (forward_price * s_tot) ** 3)
            lam2 = lam2 * lambda2_scale
        else:
            lam2 = float(lambda2)
        objective = cp.Minimize(lam1 * cp.sum_squares(x[1:] - x[:-1]) - lam2 * cp.sum(cp.entr(x)))

    cons = [
        M_comb @ x <= bounds_comb,
        cp.sum(x) == 1.0,
        cp.sum(cp.multiply(x, xr)) == forward_price,
    ]
    prob = cp.Problem(objective, cons)

    tick = check_tick / spot

    def _check():
        p = np.asarray(x.value, dtype=float)
        price = np.exp(-r * T) * (M @ p)
        viol = float(np.max(np.maximum(np.maximum(bids - price, price - asks), 0.0))) / tick
        checks = {"min_mass": float(p.min()), "mass_error": float(abs(p.sum() - 1.0)),
                  "mean_error": float(abs(p @ xr - forward_price)), "max_bidask_violation_ticks": viol}
        failed = [name for name, ok in [("negative masses", checks["min_mass"] >= -check_tol),
                                        ("total mass", checks["mass_error"] <= check_tol),
                                        ("mean", checks["mean_error"] <= check_tol),
                                        ("bid-ask", viol <= check_ticks)] if not ok]
        return checks, failed

    if isinstance(retry_opts, str):
        if retry_opts != "default":
            raise ValueError("retry_opts must be 'default', a dict or None.")
        retry_opts = DEFAULT_CLARABEL_RETRY if solver == "CLARABEL" else None

    t0 = time.perf_counter()
    status_first, retried, failed_first = None, False, None
    try:
        prob.solve(solver=solver, verbose=False, **(solver_opts or {}))
        status_first = prob.status
    except cp.error.SolverError as e:
        status_first = f"solver error ({e})"
    usable = x.value is not None and status_first in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    if usable:
        _, failed_first = _check()
    if (status_first != cp.OPTIMAL or not usable or failed_first) and retry_opts is not None:
        retried = True
        try:
            prob.solve(solver=solver, verbose=False, **retry_opts)
        except cp.error.SolverError as e:
            raise RuntimeError(f"SEDEx failed ({status_first}; retry: solver error {e})") from e
    status_final = prob.status
    solve_time = time.perf_counter() - t0
    if x.value is None or status_final not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SEDEx infeasible ({status_first}" + (f"; retry: {status_final})" if retried else ")"))
    checks, failed = _check()
    solve_info = {"status_first": status_first, "status_final": status_final, "retried": retried,
                  "failed_checks_first": ", ".join(failed_first) if failed_first else "",
                  **checks, "solve_time_s": solve_time}
    viol = checks["max_bidask_violation_ticks"]
    if failed:
        raise RuntimeError(f"SEDEx solution rejected ({status_first}" + (f"; retry: {status_final}" if retried else "")
                           + f"): checks failed on {', '.join(failed)} (max bid-ask violation {viol:.3g} ticks)")
    if retried:
        logger.warning(f"SEDEx: '{status_first}'" + (f" failing the checks on {', '.join(failed_first)}" if failed_first else "")
                       + f" at first, accepted after retry ('{status_final}', max bid-ask violation {viol:.2e} ticks)")

    return SedexResult(grid=xr, masses=np.asarray(x.value, dtype=float), step=grid_step, refinement=refinement,
                       target_step=float(target_step), sigma_atm=atm_vol, atm_strike=float(atm_quote.name),
                       forward=float(forward_price), r=r, q=q, maturity=T, spot=float(spot),
                       support=(float(lower), float(upper)), support_rules=support_rules,
                       regularization=regularization, lambda1=lam1, lambda2=lam2, solve_info=solve_info)
