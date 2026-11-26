"""End-to-end pipeline on one slice of quotes: rates, call-equivalent quotes, ARIES, SEDEx."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from .aries import AriesResult, aries
from .black_scholes import implied_volatility
from .cousot import CousotReport, cousot_check
from .forward import RatesEstimate, call_equivalent_quotes, estimate_rates
from .quotes import load_quotes, normalized_calls_and_puts
from .sedex import SedexResult, sedex

logger = logging.getLogger(__name__)


class OptionArbitrageModel:
    """Arbitrage removal (ARIES) and density extraction (SEDEx) on one slice of quotes.

    Typical use::

        model = OptionArbitrageModel(quotes, spot=5187.36, maturity=1 / 365)
        model.calibrate_forward()          # or model.set_rates(r, q)
        model.build_call_quotes()
        model.detect_arbitrage()
        density = model.extract_density()

    or ``model.run()`` for the four steps with default settings.

    Parameters
    ----------
    quotes : DataFrame
        Quotes in the input format (see :mod:`aries_sedex.quotes`).
    spot : float
        Spot price of the underlying, in index points.
    maturity : float
        Time to expiry in years (see :func:`aries_sedex.quotes.year_fraction`).
    strike_lattice : float
        Strike increment of the exchange, in index points. Every strike must be a
        multiple of it; it also sets the moneyness offset of the parity weights.
    positive_bids_only : bool
        Drop the quotes with a zero bid (default).

    Attributes
    ----------
    calls, puts : DataFrame
        Normalized quotes (strikes and prices divided by the spot).
    r, q : float
        Interest rate and dividend yield, once set.
    call_quotes : DataFrame
        Call-equivalent out-of-the-money quotes (after :meth:`build_call_quotes`).
    aries_result : AriesResult
    sedex_result : SedexResult
    """

    def __init__(self, quotes: pd.DataFrame, spot: float, maturity: float, strike_lattice: float = 5.0,
                 positive_bids_only: bool = True):
        if not maturity > 0:
            raise ValueError("The maturity must be positive.")
        self.spot = float(spot)
        self.maturity = float(maturity)
        self.strike_lattice = float(strike_lattice)
        self.calls, self.puts = normalized_calls_and_puts(quotes, self.spot, positive_bids_only)
        self.r: Optional[float] = None
        self.q: Optional[float] = None
        self.rates: Optional[RatesEstimate] = None
        self.call_quotes: Optional[pd.DataFrame] = None
        self.aries_result: Optional[AriesResult] = None
        self.sedex_result: Optional[SedexResult] = None

    @classmethod
    def from_csv(cls, path: Union[str, Path], spot: float, maturity: float, **kwargs) -> "OptionArbitrageModel":
        """Build the model from a CSV file in the input format."""
        return cls(load_quotes(path), spot, maturity, **kwargs)

    # ------------------------------------------------------------------ rates
    @property
    def forward(self) -> float:
        """Normalized forward ``exp((r - q) T)``."""
        self._require(self.r is not None, "set the rates first (calibrate_forward or set_rates)")
        return float(np.exp((self.r - self.q) * self.maturity))

    def calibrate_forward(self, alpha: float = 0.5, r0: float = 0.05, q0: float = 0.02,
                          rate_bounds: Tuple[float, float] = (0.0, 0.15),
                          dividend_bounds: Tuple[float, float] = (0.0, 0.10),
                          zero_rate_maturity: float = 1.0 / 365.0,
                          atm_offset: Optional[float] = None) -> Tuple[float, float]:
        """Estimate (r, q) from put-call parity (see :func:`aries_sedex.forward.estimate_rates`).

        ``atm_offset`` defaults to one strike increment over the spot.
        """
        if atm_offset is None:
            atm_offset = self.strike_lattice / self.spot
        self.rates = estimate_rates(self.calls, self.puts, self.maturity, atm_offset, alpha=alpha, r0=r0, q0=q0,
                                    rate_bounds=rate_bounds, dividend_bounds=dividend_bounds,
                                    zero_rate_maturity=zero_rate_maturity)
        self.r, self.q = self.rates.r, self.rates.q
        logger.info(f"Rates ({self.rates.method}): r = {self.r:.4%}, q = {self.q:.4%}.")
        self._reset_downstream()
        return self.r, self.q

    def set_rates(self, r: float, q: float) -> None:
        """Use given rates instead of the parity estimate."""
        self.r, self.q = float(r), float(q)
        self.rates = RatesEstimate(self.r, self.q, self.forward, 0, "user")
        self._reset_downstream()

    # ------------------------------------------------------------------ quotes
    def build_call_quotes(self) -> pd.DataFrame:
        """Out-of-the-money quotes expressed as calls (puts below the forward converted by parity)."""
        self._require(self.r is not None, "set the rates first (calibrate_forward or set_rates)")
        self.call_quotes = call_equivalent_quotes(self.calls, self.puts, self.r, self.q, self.maturity)
        self.aries_result, self.sedex_result = None, None
        return self.call_quotes

    def cousot_check(self, stage: str = "raw", eps: float = 1e-8) -> CousotReport:
        """Semi-static conditions on the call-equivalent quotes, before (``"raw"``) or after (``"cleaned"``) ARIES."""
        if stage == "raw":
            self._require(self.call_quotes is not None, "run build_call_quotes first")
            df = self.call_quotes
        elif stage == "cleaned":
            self._require(self.aries_result is not None, "run detect_arbitrage first")
            df = self.aries_result.quotes
        else:
            raise ValueError("stage must be 'raw' or 'cleaned'.")
        return cousot_check(df.index.values, df["bid"].values, df["ask"].values, self.r, self.q, self.maturity, eps)

    # ------------------------------------------------------------------ ARIES, SEDEx
    def detect_arbitrage(self, **kwargs) -> AriesResult:
        """Run ARIES on the call-equivalent quotes; keyword arguments go to :func:`aries_sedex.aries.aries`."""
        self._require(self.call_quotes is not None, "run build_call_quotes first")
        self.aries_result = aries(self.call_quotes, self.r, self.q, self.maturity, **kwargs)
        self.sedex_result = None
        return self.aries_result

    def extract_density(self, **kwargs) -> SedexResult:
        """Run SEDEx on the quotes kept by ARIES; keyword arguments go to :func:`aries_sedex.sedex.sedex`."""
        self._require(self.aries_result is not None, "run detect_arbitrage first")
        kwargs.setdefault("strike_lattice", self.strike_lattice)
        self.sedex_result = sedex(self.aries_result.quotes, self.r, self.q, self.maturity, self.spot, **kwargs)
        return self.sedex_result

    def run(self, rates: Optional[Tuple[float, float]] = None, forward_kwargs: Optional[dict] = None,
            aries_kwargs: Optional[dict] = None, sedex_kwargs: Optional[dict] = None) -> SedexResult:
        """Rates (estimated, or ``rates = (r, q)``), call-equivalent quotes, ARIES and SEDEx."""
        if rates is None:
            self.calibrate_forward(**(forward_kwargs or {}))
        else:
            self.set_rates(*rates)
        self.build_call_quotes()
        self.detect_arbitrage(**(aries_kwargs or {}))
        return self.extract_density(**(sedex_kwargs or {}))

    # ------------------------------------------------------------------ diagnostics
    def removed_quotes(self) -> pd.DataFrame:
        """Quotes removed by ARIES, in index points, with their original option type."""
        self._require(self.aries_result is not None, "run detect_arbitrage first")
        rem = self.aries_result.removals.copy()
        if rem.empty:
            return rem.assign(strike_points=[], option_type=[], price=[])
        cq = self.call_quotes
        rem["strike_points"] = rem["strike"] * self.spot
        rem["option_type"] = ["P" if cq.loc[k, "source"] == "put" else "C" for k in rem["strike"]]
        table = {"put": self.puts, "call": self.calls}
        rem["price"] = [table[cq.loc[k, "source"]].loc[k, side] * self.spot
                        for k, side in zip(rem["strike"], rem["side"])]
        return rem

    def quote_table(self) -> pd.DataFrame:
        """Out-of-the-money quotes with implied volatilities and, after SEDEx, model prices.

        Columns: ``strike`` (normalized), ``strike_points``, ``option_type``, ``bid``, ``ask``
        (original option, index points), ``bid_size``, ``ask_size``, ``kept`` (kept by ARIES),
        ``iv_bid``, ``iv_ask``, ``iv_mid`` and, when the density is available, ``model_price``
        (index points) and ``iv_model``.
        """
        self._require(self.call_quotes is not None, "run build_call_quotes first")
        cq = self.call_quotes
        rows = []
        for k, src in zip(cq.index.values, cq["source"].values):
            original = self.puts if src == "put" else self.calls
            rows.append({"strike": k, "option_type": "P" if src == "put" else "C",
                         "bid": original.loc[k, "bid"], "ask": original.loc[k, "ask"],
                         "bid_size": original.loc[k, "bid_size"], "ask_size": original.loc[k, "ask_size"]})
        tab = pd.DataFrame(rows)
        flags = np.where(tab["option_type"] == "P", "p", "c")
        args = (1.0, tab["strike"].values, self.maturity, self.r, self.q, flags)
        tab["iv_bid"] = implied_volatility(tab["bid"].values, *args)
        tab["iv_ask"] = implied_volatility(tab["ask"].values, *args)
        tab["iv_mid"] = implied_volatility(0.5 * (tab["bid"].values + tab["ask"].values), *args)
        tab["kept"] = (tab["strike"].isin(self.aries_result.quotes.index) if self.aries_result is not None
                       else True)
        if self.sedex_result is not None:
            res = self.sedex_result
            model = np.where(tab["option_type"] == "P", res.put_prices(tab["strike"].values),
                             res.call_prices(tab["strike"].values))
            tab["iv_model"] = implied_volatility(model, *args)
            tab["model_price"] = model * self.spot
        tab["strike_points"] = tab["strike"] * self.spot
        for col in ("bid", "ask"):
            tab[col] = tab[col] * self.spot
        return tab

    def model_smile(self, strikes) -> pd.DataFrame:
        """SEDEx implied volatilities at given normalized strikes (out-of-the-money option at each strike)."""
        self._require(self.sedex_result is not None, "run extract_density first")
        k = np.asarray(strikes, dtype=float)
        res = self.sedex_result
        is_put = k < res.forward
        price = np.where(is_put, res.put_prices(k), res.call_prices(k))
        iv = implied_volatility(price, 1.0, k, self.maturity, self.r, self.q, np.where(is_put, "p", "c"))
        return pd.DataFrame({"strike": k, "price": price, "iv": iv})

    # ------------------------------------------------------------------ helpers
    def _reset_downstream(self):
        self.call_quotes, self.aries_result, self.sedex_result = None, None, None

    @staticmethod
    def _require(condition: bool, message: str):
        if not condition:
            raise RuntimeError(message)
