"""Input format of the option quotes.

A slice of quotes (one underlying, one expiry, one quote time) is a table with
one row per listed option and the columns

=============  ==============================================================
``strike``     strike price, in index points
``option_type`` ``"C"`` (call) or ``"P"`` (put)
``bid``        best bid price, in index points
``ask``        best ask price, in index points
``bid_size``   number of contracts available at the bid
``ask_size``   number of contracts available at the ask
=============  ==============================================================

The spot price and the time to expiry are passed separately. Any liquidity
filter (open interest, volume) is assumed to have been applied beforehand.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("strike", "option_type", "bid", "ask", "bid_size", "ask_size")

#: Column order of the normalized call and put tables used internally.
QUOTE_COLUMNS = ["ask", "bid", "ask_size", "bid_size"]

SECONDS_PER_YEAR = 365 * 24 * 3600


def year_fraction(quote_time: Union[str, datetime, pd.Timestamp],
                  expiry_time: Union[str, datetime, pd.Timestamp]) -> float:
    """Time to expiry in years (actual seconds / 365 days)."""
    seconds = (pd.Timestamp(expiry_time) - pd.Timestamp(quote_time)).total_seconds()
    if seconds <= 0:
        raise ValueError("The expiry time must be later than the quote time.")
    return seconds / SECONDS_PER_YEAR


def validate_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Check the input table and return a clean copy.

    Option types are mapped to ``"C"``/``"P"``; strikes, prices and sizes are cast
    to float. A ``ValueError`` is raised on a missing column, an unknown option
    type, a duplicated (strike, option type) pair, a non-finite value, a negative
    price or size, or a bid above the ask.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in quotes.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}.")
    df = quotes.loc[:, list(REQUIRED_COLUMNS)].copy()

    types = df["option_type"].astype(str).str.strip().str.upper()
    df["option_type"] = types.map({"C": "C", "CALL": "C", "P": "P", "PUT": "P"})
    if df["option_type"].isna().any():
        raise ValueError("option_type must be 'C' or 'P'.")

    numeric = ["strike", "bid", "ask", "bid_size", "ask_size"]
    df[numeric] = df[numeric].astype(float)
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("Strikes, prices and sizes must be finite numbers.")
    if (df["strike"] <= 0).any():
        raise ValueError("Strikes must be positive.")
    if (df[["bid", "ask", "bid_size", "ask_size"]] < 0).to_numpy().any():
        raise ValueError("Prices and sizes must be nonnegative.")
    if (df["bid"] > df["ask"]).any():
        raise ValueError("A bid is above its ask.")
    if df.duplicated(["strike", "option_type"]).any():
        raise ValueError("Each (strike, option_type) pair must appear once.")
    return df.reset_index(drop=True)


def load_quotes(path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV file in the input format and validate it."""
    return validate_quotes(pd.read_csv(path))


def normalized_calls_and_puts(quotes: pd.DataFrame, spot: float,
                              positive_bids_only: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the quotes into calls and puts normalized by the spot.

    Strikes and prices are divided by ``spot``; sizes are unchanged. Each table is
    indexed by the normalized strike, sorted in increasing order, with the columns
    ``ask, bid, ask_size, bid_size``.

    Parameters
    ----------
    quotes : DataFrame in the input format
    spot : float
        Spot price of the underlying, in index points.
    positive_bids_only : bool
        Drop the quotes whose bid is zero (default).
    """
    if not spot > 0:
        raise ValueError("The spot price must be positive.")
    df = validate_quotes(quotes)
    if positive_bids_only:
        df = df[df["bid"] > 0]
    tables = []
    for option_type in ("C", "P"):
        table = df[df["option_type"] == option_type].reset_index(drop=True)
        table = table[["strike"] + QUOTE_COLUMNS].copy()
        table[["strike", "ask", "bid"]] = table[["strike", "ask", "bid"]] / spot
        table = table.set_index("strike").sort_index()
        tables.append(table)
    calls, puts = tables
    if calls.empty and puts.empty:
        raise ValueError("No quote left after filtering.")
    return calls, puts
