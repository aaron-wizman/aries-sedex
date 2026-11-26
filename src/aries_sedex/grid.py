"""State grid compatible with the strike lattice."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compatible_grid(strikes, spot: float, step_target: float, lower0: float, upper0: float,
                    strike_lattice: float = 5.0) -> Tuple[np.ndarray, float, int]:
    """Uniform grid of the normalized state that contains every strike.

    With ``eta = strike_lattice / spot`` the mesh of the normalized strike lattice,
    the refinement factor is ``m = min{l >= 1 : eta / l <= step_target}`` and the mesh
    is ``step = eta / m``. The grid points are ``s_j = j * step`` for the integers ``j``
    from ``max(1, floor(lower0 / step))`` to ``ceil(upper0 / step)``; a strike
    ``K = n * eta`` is the grid point ``j = n * m``.

    Parameters
    ----------
    strikes : array_like
        Normalized strikes ``K / S0``.
    spot : float
        Spot price ``S0`` in index points.
    step_target : float
        Target mesh size (normalized units).
    lower0, upper0 : float
        Preliminary bounds of the support (normalized units).
    strike_lattice : float
        Strike increment of the exchange, in index points.

    Returns
    -------
    grid : ndarray
        Grid points (normalized).
    step : float
        Mesh size.
    m : int
        Refinement factor.
    """
    k_lat = np.asarray(strikes, dtype=float) * spot / strike_lattice
    n_lat = np.rint(k_lat)
    if np.max(np.abs(k_lat - n_lat)) > 1e-6:
        raise ValueError(f"Strikes are not on the {strike_lattice:g}-point strike lattice: "
                         f"the grid cannot contain them.")
    eta = strike_lattice / spot
    m = int(np.ceil(eta / step_target))
    step = eta / m
    j_lo = max(1, int(np.floor(lower0 / step)))
    j_hi = int(np.ceil(upper0 / step))
    j = np.arange(j_lo, j_hi + 1)
    grid = (j * strike_lattice / m) / spot
    j_k = (n_lat * m).astype(np.int64)
    if np.any(j_k < j_lo) or np.any(j_k > j_hi):
        raise ValueError("Some strikes lie outside the support of the grid.")
    return grid, step, m
