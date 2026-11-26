"""ARIES and SEDEx on the synthetic 1DTE slice shipped in examples/data.

Run from the repository root:

    python examples/run_example.py

The figures are written to examples/figures and the extracted density to
examples/output/density_synthetic_1dte.csv.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    import aries_sedex  # noqa: F401
except ImportError:  # package not installed: use the source tree
    sys.path.insert(0, str(HERE.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from aries_sedex import OptionArbitrageModel  # noqa: E402
from aries_sedex import plotting  # noqa: E402


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    meta = json.loads((HERE / "data" / "synthetic_1dte.json").read_text())
    spot, maturity = meta["spot"], meta["maturity_years"]

    model = OptionArbitrageModel.from_csv(HERE / "data" / "synthetic_1dte.csv", spot=spot, maturity=maturity,
                                          strike_lattice=meta["strike_lattice"])
    print(f"Quotes with a positive bid: {len(model.calls)} calls, {len(model.puts)} puts; spot {spot}, "
          f"T = {maturity:.6f} years")

    # 1. rates from put-call parity (or model.set_rates(r, q) to use your own values)
    r, q = model.calibrate_forward()
    print(f"\nPut-call parity on {model.rates.n_strikes} strikes: r = {r:.4%}, q = {q:.4%}, "
          f"forward = {model.forward * spot:.3f}")

    # 2. out-of-the-money quotes as calls
    quotes = model.build_call_quotes()
    print(f"Out-of-the-money quotes: {len(quotes)} strikes "
          f"({(quotes['source'] == 'put').sum()} puts converted, {(quotes['source'] == 'call').sum()} calls)")
    print(f"Cousot-type conditions before ARIES: {model.cousot_check('raw').summary()}")

    # 3. ARIES
    t0 = time.perf_counter()
    model.detect_arbitrage()
    t_aries = time.perf_counter() - t0
    removed = model.removed_quotes()
    print(f"\nARIES: {len(removed)} quote(s) removed in {t_aries:.2f} s, "
          f"{len(model.aries_result.quotes)} strikes kept")
    if len(removed):
        print(removed[["iteration", "kind", "strike_points", "option_type", "side", "price", "size"]]
              .rename(columns={"strike_points": "strike"}).to_string(index=False))
    print(f"Cousot-type conditions after ARIES: {model.cousot_check('cleaned').summary()}")

    # 4. SEDEx
    t0 = time.perf_counter()
    res = model.extract_density()
    t_sedex = time.perf_counter() - t0
    info = res.solve_info
    print(f"\nSEDEx: {len(res.grid)} grid points, step {res.step * spot:.4f} index points "
          f"(strike lattice / {res.refinement}), sigma_ATM = {res.sigma_atm:.4f}, lambda_2 = {res.lambda2:.3e}")
    print(f"  solver status {info['status_final']} (retry: {info['retried']}), {t_sedex:.1f} s")
    print(f"  checks: min mass {info['min_mass']:.2e}, mass error {info['mass_error']:.2e}, "
          f"mean error {info['mean_error']:.2e}, max bid-ask violation {info['max_bidask_violation_ticks']:.2e} ticks")
    qs = res.quantile([0.01, 0.05, 0.5, 0.95, 0.99]) * spot
    print("  quantiles of S_T (1%, 5%, 50%, 95%, 99%): " + ", ".join(f"{v:.1f}" for v in qs))
    tab = model.quote_table()
    kept = tab[tab["kept"]]
    outside = np.maximum(np.maximum(kept["bid"] - kept["model_price"], kept["model_price"] - kept["ask"]), 0.0)
    print(f"  largest distance of a SEDEx price to its bid-ask interval: {outside.max():.1e} index points "
          f"({len(kept)} strikes)")

    out = HERE / "output"
    out.mkdir(exist_ok=True)
    res.to_frame(in_index_points=True).to_csv(out / "density_synthetic_1dte.csv", index=False)

    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    for name, fig in [("density", plotting.plot_density(model)), ("smile", plotting.plot_smile(model)),
                      ("iv_outside_bidask", plotting.plot_iv_outside_bidask(model)),
                      ("aries", plotting.plot_aries(model)), ("overview", plotting.plot_overview(model))]:
        fig.savefig(figs / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"\nFigures written to {figs}")


if __name__ == "__main__":
    main()
