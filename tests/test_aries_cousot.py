import numpy as np

from aries_sedex import OptionArbitrageModel

from conftest import bs_quotes

SPOT, T, R, Q = 1000.0, 7 / 365, 0.04, 0.01


def model_from(quotes):
    m = OptionArbitrageModel(quotes, spot=SPOT, maturity=T)
    m.set_rates(R, Q)
    m.build_call_quotes()
    return m


def test_no_arbitrage_on_model_quotes(bs_slice):
    m = model_from(bs_slice)
    assert m.cousot_check("raw").n_violations == 0
    res = m.detect_arbitrage()
    assert res.removals.empty and len(res.quotes) == len(m.call_quotes)


def test_strong_arbitrage_is_removed(bs_slice):
    df = bs_slice.copy()
    # call bid at 1040 above the call ask at 1035: crossed vertical spread
    i = df.index[(df.strike == 1040) & (df.option_type == "C")][0]
    j = df.index[(df.strike == 1035) & (df.option_type == "C")][0]
    df.loc[i, ["bid", "ask", "bid_size"]] = [df.loc[j, "ask"] + 0.10, df.loc[j, "ask"] + 0.30, 3]
    m = model_from(df)
    assert m.cousot_check("raw").n_violations > 0
    res = m.detect_arbitrage()
    assert len(res.removals) == 1
    row = res.removals.iloc[0]
    assert row["kind"] == "strong" and row["side"] == "bid" and np.isclose(row["strike"] * SPOT, 1040)
    assert m.cousot_check("cleaned").n_violations == 0


def test_zero_cost_arbitrage_is_standard():
    strikes = np.arange(900.0, 1100.0 + 1, 5.0)
    df = bs_quotes(strikes=strikes, half_spread=0.10)
    calls = (df.option_type == "C") & (df.strike >= 1080)
    df.loc[calls, ["bid", "ask"]] = [0.05, 0.10]                   # far wing at the minimum quotes
    i = df.index[(df.strike == 1090) & (df.option_type == "C")][0]
    df.loc[i, ["bid", "ask", "bid_size"]] = [0.10, 0.15, 2]        # bid equal to the asks of lower strikes
    m = model_from(df)
    res = m.detect_arbitrage()
    assert list(res.removals["kind"]) == ["standard"]
    assert np.isclose(res.removals["strike"].iloc[0] * SPOT, 1090) and res.removals["side"].iloc[0] == "bid"


def test_example_dataset_aries(example_dir):
    import json
    meta = json.loads((example_dir / "synthetic_1dte.json").read_text())
    m = OptionArbitrageModel.from_csv(example_dir / "synthetic_1dte.csv", spot=meta["spot"],
                                      maturity=meta["maturity_years"])
    m.calibrate_forward()
    m.build_call_quotes()
    res = m.detect_arbitrage()
    assert sorted(res.removals["kind"]) == ["standard"] * 3 + ["strong"] * 3
    assert m.cousot_check("cleaned").n_violations == 0
