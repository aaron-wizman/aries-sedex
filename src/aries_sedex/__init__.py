"""ARIES and SEDEx: arbitrage removal and density extraction for short-dated options.

Reference: G. Wizman, G. Turinici and A. Merran, "From Arbitrage Removal to Density
Extraction: A Model-Free Framework for Short-Dated Options", arXiv:2605.22792.
"""

from .aries import AriesResult, aries
from .black_scholes import bsm_price, implied_volatility
from .cousot import CousotReport, cousot_check
from .forward import RatesEstimate, call_equivalent_quotes, estimate_rates
from .grid import compatible_grid
from .model import OptionArbitrageModel
from .quotes import load_quotes, normalized_calls_and_puts, validate_quotes, year_fraction
from .sedex import SedexResult, sedex

__version__ = "1.0.0"

__all__ = [
    "AriesResult", "CousotReport", "OptionArbitrageModel", "RatesEstimate", "SedexResult",
    "aries", "bsm_price", "call_equivalent_quotes", "compatible_grid", "cousot_check",
    "estimate_rates", "implied_volatility", "load_quotes", "normalized_calls_and_puts",
    "sedex", "validate_quotes", "year_fraction",
]
