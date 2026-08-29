#!/usr/bin/env python3
"""End-to-end risk report.

Runs the full pipeline: build a portfolio, estimate VaR by every method
(including EVT for the far tail and a conditional-volatility filtered historical
simulation), backtest each model walk-forward, run the supervisory tests for both
VaR and Expected Shortfall, and — with real data — a stress-test suite.

By default it uses the synthetic GARCH generator so the script runs anywhere
with no network. To use real Argentine equities instead:

    pip install yfinance
    python var_analysis.py --real
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

# Windows consoles default to cp1252, which cannot encode the report's dashes and
# arrows. Force UTF-8 and never crash on an odd glyph.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from varengine import (
    Portfolio,
    compare_methods,
    evt_var,
    filtered_historical_var,
    fit_factor_model,
    historical_var,
    parametric_var,
    reverse_stress,
    run_full_backtest,
    run_stress_suite,
    simulate_market,
)
from varengine.plots import risk_dashboard

TICKERS_AR = ["GGAL.BA", "YPFD.BA", "PAMP.BA", "TXAR.BA"]
RULE = "-" * 78


def banner(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true", help="download real market data")
    ap.add_argument("--confidence", type=float, default=0.99)
    ap.add_argument("--window", type=int, default=250, help="backtest training window")
    ap.add_argument("--value", type=float, default=1_000_000.0)
    ap.add_argument("--out", default="docs/risk_report.png")
    args = ap.parse_args()

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    # ---------------------------------------------------------------- data
    if args.real:
        from varengine import load_market_data

        print(f"Downloading {', '.join(TICKERS_AR)} ...")
        prices = load_market_data(TICKERS_AR, start="2019-01-01")
    else:
        prices = simulate_market(
            ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"], n_days=1800, seed=42
        )
        print("Using synthetic GARCH(1,1) data with Student-t innovations.")
        print("Run with --real for live market prices.")

    book = Portfolio(prices, value=args.value)

    banner("PORTFOLIO")
    print(f"Assets      : {', '.join(book.tickers)}")
    print(f"Weights     : {', '.join(f'{w:.1%}' for w in book.w)}")
    print(f"Value       : {book.value:,.0f}")
    print(f"Period      : {prices.index[0].date()} to {prices.index[-1].date()}")
    print()
    for k, v in book.summary().items():
        label = k.replace("_", " ").capitalize()
        print(f"{label:<24}{v:>12,.4f}" if isinstance(v, float) else f"{label:<24}{v:>12}")

    # ---------------------------------------------------------------- risk decomposition
    banner("RISK DECOMPOSITION — component VaR (99%, parametric normal)")
    cv = book.component_var(args.confidence, distribution="normal")
    print(f"{'asset':<12}{'marginal':>12}{'component':>12}{'share':>9}{'incremental':>13}")
    for asset, row in cv.iterrows():
        print(f"{asset:<12}{row['marginal']:>12.3%}{row['component']:>12.3%}"
              f"{row['component_pct']:>9.1%}{row['incremental']:>13.3%}")
    print("\nComponents sum to the portfolio VaR; 'incremental' is what you would")
    print("shed by dropping the position entirely (negative -> it is a diversifier).")

    # ---------------------------------------------------------------- VaR
    banner(f"VALUE AT RISK — {args.confidence:.0%} confidence, 1-day horizon")
    comparison = compare_methods(
        book.asset_returns, book.w, args.confidence, portfolio_value=book.value
    )
    for method, row in comparison.iterrows():
        print(f"{method:<26} VaR {row['VaR']:>7.3%}  ES {row['ES']:>7.3%}   "
              f"{row['VaR_amount']:>12,.0f}")
        if row["note"]:
            print(f"{'':<26} {row['note']}")

    uncond = comparison[~comparison.index.str.startswith("filtered")]["VaR"]
    spread = uncond.max() / uncond.min() - 1
    print(f"\nSpread across the unconditional estimators: {spread:.1%}")
    print("The normal-assumption methods cluster at the bottom — the signature of")
    print("fat tails. Filtered historical simulation is not on this axis: it scales")
    print("by *today's* volatility, so it sits lower when the sample ends calm. That")
    print("conditioning is the point, and it shows up in the backtest below.")

    banner("FAR TAIL — Extreme Value Theory (peaks-over-threshold GPD)")
    for conf in (args.confidence, 0.995, 0.999):
        try:
            res = evt_var(book.returns, conf, portfolio_value=book.value)
            hist = historical_var(book.returns, conf).var
            print(f"  {conf:>6.1%}   EVT VaR {res.var:>7.3%}   ES {res.expected_shortfall:>7.3%}"
                  f"   (historical VaR {hist:>7.3%})")
        except ValueError as e:
            print(f"  {conf:>6.1%}   skipped: {e}")
    evt_note = evt_var(book.returns, 0.99).note
    print(f"\n  {evt_note}")

    # ---------------------------------------------------------------- factor model
    if args.real:
        banner("FACTOR MODEL — Fama-French 5 + momentum")
        try:
            fm = fit_factor_model(book.returns, source="fama_french")
            print(fm)
            print("\nAlpha and risk that do not trace back to a known factor are")
            print("what a discretionary manager is actually paid for.")
        except Exception as e:
            print(f"  skipped ({e})")
    else:
        print("\n(Factor model needs real equity returns — run with --real.)")

    banner("MULTI-DAY HORIZON (square-root-of-time)")
    for days in (1, 5, 10):
        res = historical_var(book.returns, args.confidence, horizon_days=days,
                             portfolio_value=book.value)
        print(f"{days:>3}-day   VaR {res.var:>7.3%}   {res.var_amount:>12,.0f}")
    print("\nThis rule assumes i.i.d. returns. Volatility clusters, so it")
    print("understates risk in turbulent periods. Basel permits it anyway.")

    # ---------------------------------------------------------------- backtest
    banner(f"WALK-FORWARD BACKTEST — {args.window}-day rolling window")
    print("VaR tests (Kupiec, Christoffersen) AND the ES test (Acerbi-Szekely),")
    print("because FRTB made ES the measure a model is actually judged on.\n")

    frames = {}
    for method, dist, label in [
        ("historical", "normal", "Historical simulation"),
        ("parametric", "normal", "Parametric (normal)"),
        ("parametric", "t", "Parametric (Student-t)"),
        ("fhs", "ewma", "Filtered historical (EWMA)"),
    ]:
        out = run_full_backtest(
            book.returns, window=args.window, confidence=args.confidence,
            method=method, distribution=dist,
        )
        basel = out["basel"]
        frames[label] = out["frame"]
        print(f"{label}")
        print(f"  {out['n_exceptions']} exceptions in {out['n_days']} days "
              f"= {out['exception_rate']:.2%} (expected {out['expected_rate']:.2%})")
        print(f"  {out['kupiec']}")
        print(f"  {out['christoffersen']}")
        print(f"  {out['es_test']}")
        print(f"     {out['es_test'].detail}")
        print(f"  Basel zone: {basel['zone']}  "
              f"(capital multiplier {basel['capital_multiplier']:.2f}) — {basel['reading']}\n")

    # ---------------------------------------------------------------- stress testing
    stress_df = None
    if args.real:
        banner("STRESS TESTING")
        try:
            stress_df = run_stress_suite(book)
            for scenario, row in stress_df.iterrows():
                extra = ""
                if pd.notna(row.get("max_drawdown_pct")):
                    extra = f"   worst day {row['worst_day_pct']:.1%}, max DD {row['max_drawdown_pct']:.1%}"
                print(f"  {scenario:<28} {row['pnl_pct']:>8.2%}  "
                      f"({row['pnl_amount']:>12,.0f}){extra}")
            print()
            rev = reverse_stress(book, target_loss_pct=0.10, direction="risk_off")
            print(f"  Reverse stress — {rev.scenario}:")
            print(f"     {rev.detail}")
        except (ImportError, ValueError) as e:
            print(f"  Stress suite unavailable: {e}")
    else:
        print("\n(Stress testing needs real risk-factor data — run with --real.)")

    # ---------------------------------------------------------------- report
    var_levels = {
        "historical": historical_var(book.returns, args.confidence).var,
        "normal": parametric_var(book.returns, args.confidence, distribution="normal").var,
        "student-t": parametric_var(book.returns, args.confidence, distribution="t").var,
        "fhs-ewma": filtered_historical_var(book.returns, args.confidence, vol="ewma").var,
    }
    path = risk_dashboard(
        frames["Filtered historical (EWMA)"], book.returns, var_levels, comparison,
        args.out, stress_df=stress_df,
    )

    banner("OUTPUT")
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
