"""Matplotlib figures for one slice: density, smile, bid-ask check and ARIES removals.

Model outputs are drawn as points, never joined by lines, since a line would
interpolate between the values actually computed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .model import OptionArbitrageModel

BID_STYLE = dict(marker="v", s=22, color="#1f4e9c", linewidths=0.4, edgecolors="black", zorder=3)
ASK_STYLE = dict(marker="^", s=22, color="#c0392b", linewidths=0.4, edgecolors="black", zorder=3)
MODEL_COLOR = "#e67e22"
EXTRAPOLATION_COLOR = "#f5c48a"
DENSITY_COLOR = "#b0002a"
REMOVED_STYLE = dict(marker="o", s=70, facecolors="none", edgecolors="black", linewidths=1.0, zorder=4)

STYLE = {"font.family": "serif", "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
         "legend.fontsize": 9, "xtick.direction": "out", "ytick.direction": "out"}


def _axes(ax, figsize=(7.0, 4.2)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    return ax.figure, ax


def density_window(model: OptionArbitrageModel, quantiles: Tuple[float, float] = (0.01, 0.99),
                   margin: float = 0.1) -> Tuple[float, float]:
    """Window of the density plot, centered on the forward.

    The half-width is the larger distance from the forward to the two quantiles of the
    extracted law, increased by ``margin`` (relative).
    """
    res = model.sedex_result
    lo, hi = res.quantile(quantiles)
    half = (1.0 + margin) * max(res.forward - lo, hi - res.forward)
    return res.forward - half, res.forward + half


def plot_density(model: OptionArbitrageModel, ax: Optional[plt.Axes] = None,
                 xlim: Optional[Tuple[float, float]] = None, quantiles: Tuple[float, float] = (0.01, 0.99),
                 margin: float = 0.1, true_density=None):
    """SEDEx density on the grid points of a window of the normalized state.

    The window is ``xlim`` if given, otherwise :func:`density_window` (centered on the forward).
    ``true_density`` (optional) is a function of the normalized state, drawn as small grey points.
    """
    res = model.sedex_result
    a, b = xlim if xlim is not None else density_window(model, quantiles, margin)
    sel = (res.grid >= a) & (res.grid <= b)
    with plt.rc_context(STYLE):
        fig, ax = _axes(ax)
        if true_density is not None:
            ax.scatter(res.grid[sel], true_density(res.grid[sel]), s=2, color="0.55", label="True density", zorder=2)
        ax.scatter(res.grid[sel], res.density[sel], s=5, color=DENSITY_COLOR, label="SEDEx", zorder=3)
        ax.set_xlim(a, b)
        ax.set_xlabel(r"$S_T / S_0$")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    return fig


def smile_strikes(model: OptionArbitrageModel, extrapolation: Tuple[float, float] = (0.10, 0.25)) -> np.ndarray:
    """Grid points at which the SEDEx smile is drawn.

    The quoted range ``[K_min, K_max]`` of the strikes kept by ARIES is extended on each side by
    ``extrapolation`` times its width. The extension stops halfway between the extreme strike and
    the bound of the support: near that bound the call prices of the discrete density go to zero
    and the smile bends down.
    """
    res = model.sedex_result
    k = model.aries_result.quotes.index.values
    k_min, k_max = k.min(), k.max()
    width = k_max - k_min
    lo = max(k_min - extrapolation[0] * width, k_min - 0.5 * (k_min - res.grid[0]))
    hi = min(k_max + extrapolation[1] * width, k_max + 0.5 * (res.grid[-1] - k_max))
    g = res.grid
    return g[(g >= lo) & (g <= hi)]


def plot_smile(model: OptionArbitrageModel, ax: Optional[plt.Axes] = None, model_strikes: Optional[Sequence] = None,
               extrapolation: Tuple[float, float] = (0.10, 0.25)):
    """Bid and ask implied volatilities of the quotes kept by ARIES and the SEDEx smile.

    The SEDEx smile is evaluated at ``model_strikes`` (normalized); by default at the grid points
    given by :func:`smile_strikes`, which extend beyond the quoted strikes. Points outside the
    range of the kept strikes (extrapolation) are drawn in a lighter color.
    """
    tab = model.quote_table()
    kept = tab[tab["kept"]]
    if model_strikes is None:
        model_strikes = smile_strikes(model, extrapolation)
    smile = model.model_smile(model_strikes)
    inside = (smile["strike"] >= kept["strike"].min()) & (smile["strike"] <= kept["strike"].max())
    with plt.rc_context(STYLE):
        fig, ax = _axes(ax)
        ax.scatter(smile["strike"][inside], 100 * smile["iv"][inside], s=2, color=MODEL_COLOR, label="SEDEx", zorder=2)
        if (~inside).any():
            ax.scatter(smile["strike"][~inside], 100 * smile["iv"][~inside], s=2, color=EXTRAPOLATION_COLOR,
                       label="SEDEx, beyond the quoted strikes", zorder=2)
        ax.scatter(kept["strike"], 100 * kept["iv_bid"], label="Bid", **BID_STYLE)
        ax.scatter(kept["strike"], 100 * kept["iv_ask"], label="Ask", **ASK_STYLE)
        ax.set_xlabel(r"$K / S_0$")
        ax.set_ylabel("Implied volatility (%)")
        ax.legend(frameon=False)
    return fig


def plot_iv_outside_bidask(model: OptionArbitrageModel, ax: Optional[plt.Axes] = None):
    """SEDEx implied volatility outside the bid-ask implied volatility interval, in volatility points.

    At each strike kept by ARIES: SEDEx IV minus bid IV when it is below the bid IV, SEDEx IV
    minus ask IV when it is above the ask IV; strikes inside the interval are not drawn. Since the
    SEDEx prices lie inside every bid-ask interval, the remaining values are of the order of the
    solver tolerance.
    """
    tab = model.quote_table()
    kept = tab[tab["kept"]]
    below = kept[kept["iv_model"] < kept["iv_bid"]]
    above = kept[kept["iv_model"] > kept["iv_ask"]]
    dev_below = 100 * (below["iv_model"] - below["iv_bid"])
    dev_above = 100 * (above["iv_model"] - above["iv_ask"])
    largest = max([dev_below.abs().max() if len(below) else 0.0, dev_above.abs().max() if len(above) else 0.0])
    exponent = int(np.floor(np.log10(largest))) if 0.0 < largest < 0.1 else 0
    scale = 10.0 ** exponent
    unit = "vol. points" if exponent == 0 else rf"$10^{{{exponent}}}$ vol. points"
    with plt.rc_context(STYLE):
        fig, ax = _axes(ax)
        ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
        ax.scatter(below["strike"], dev_below / scale, label="Below the bid", **BID_STYLE)
        ax.scatter(above["strike"], dev_above / scale, label="Above the ask", **ASK_STYLE)
        ax.set_xlim(kept["strike"].min() - 0.005, kept["strike"].max() + 0.005)
        if below.empty and above.empty:
            ax.text(0.5, 0.5, "SEDEx IV inside every bid-ask IV interval", transform=ax.transAxes,
                    ha="center", va="center")
        ax.set_xlabel(r"$K / S_0$")
        ax.set_ylabel(f"SEDEx IV outside the bid-ask IV interval\n({unit})")
        ax.legend(frameon=False)
    return fig


def plot_aries(model: OptionArbitrageModel, ax: Optional[plt.Axes] = None, annotate: bool = True):
    """All out-of-the-money quotes relative to the SEDEx price, with the quotes removed by ARIES circled.

    The vertical axis is ``quote - SEDEx price`` in index points: the quotes kept by ARIES
    bracket the SEDEx price (bids at or below zero, asks at or above zero).
    """
    tab = model.quote_table().set_index("strike")
    rem = model.removed_quotes()
    with plt.rc_context(STYLE):
        fig, ax = _axes(ax)
        ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
        ax.scatter(tab.index, tab["bid"] - tab["model_price"], label="Bid - SEDEx", **BID_STYLE)
        ax.scatter(tab.index, tab["ask"] - tab["model_price"], label="Ask - SEDEx", **ASK_STYLE)
        if len(rem):
            rem = rem.sort_values("strike")
            k = rem["strike"].to_numpy()
            dev = np.array([tab.loc[s, side] - tab.loc[s, "model_price"] for s, side in zip(k, rem["side"])])
            for kind, marker in (("strong", "o"), ("standard", "s")):
                sel = (rem["kind"] == kind).to_numpy()
                if sel.any():
                    ax.scatter(k[sel], dev[sel], label=f"Removed by ARIES ({kind})", **{**REMOVED_STYLE, "marker": marker})
            if annotate:
                for i, (s, d) in enumerate(zip(k, dev)):
                    ax.annotate(f"{s * model.spot:.0f}", (s, d), xytext=(-14, 10) if i % 2 == 0 else (4, -16),
                                textcoords="offset points", fontsize=8)
        ax.set_xlabel(r"$K / S_0$")
        ax.set_ylabel("Quote minus SEDEx price (index points)")
        ax.legend(frameon=False, loc="lower left")
    return fig


def plot_overview(model: OptionArbitrageModel, xlim: Optional[Tuple[float, float]] = None):
    """Three panels: density, smile, SEDEx IV outside the bid-ask IV interval."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
        plot_density(model, ax=axes[0], xlim=xlim)
        plot_smile(model, ax=axes[1])
        plot_iv_outside_bidask(model, ax=axes[2])
        for ax, label in zip(axes, "abc"):
            ax.set_title(f"({label})", loc="left")
        fig.tight_layout()
    return fig
