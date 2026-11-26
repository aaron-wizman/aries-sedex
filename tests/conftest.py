import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
try:
    import aries_sedex  # noqa: F401
except ImportError:  # package not installed: use the source tree
    sys.path.insert(0, str(ROOT / "src"))

from aries_sedex import bsm_price  # noqa: E402


def bs_quotes(spot=1000.0, maturity=7 / 365, r=0.04, q=0.01, sigma=0.20, strikes=None, half_spread=0.10,
              tick=0.05, size=100):
    """Calls and puts priced by Black-Scholes, quoted on a tick grid around the model price."""
    if strikes is None:
        strikes = np.arange(900.0, 1100.0 + 1, 5.0)
    rows = []
    for flag, opt in (("c", "C"), ("p", "P")):
        fair = bsm_price(flag, spot, strikes, maturity, r, q, sigma)
        bid = np.floor((fair - half_spread) / tick) * tick
        ask = np.ceil((fair + half_spread) / tick) * tick
        for k, b, a in zip(strikes, bid, ask):
            rows.append({"strike": k, "option_type": opt, "bid": round(max(b, 0.0), 2), "ask": round(a, 2),
                         "bid_size": size, "ask_size": size})
    return pd.DataFrame(rows)


@pytest.fixture
def bs_slice():
    return bs_quotes()


@pytest.fixture
def example_dir():
    return ROOT / "examples" / "data"
