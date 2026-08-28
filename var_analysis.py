#!/usr/bin/env python3
"""End-to-end risk report.

Runs the full pipeline: build a portfolio, estimate VaR by every method,
backtest the model walk-forward, and run the supervisory tests.

By default it uses the synthetic GARCH generator so the script runs anywhere
with no network. To use real Argentine equities instead:

    pip install yfinance
    python var_analysis.py --real
"""

from __future__ import annotations

import argparse

import pandas as pd

from varengine import (
    Portfolio,
    compare_methods,
    historical_var,
    parametric_var,
    run_full_backtest,
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

    print("\nRisk contribution by asset:")
    for asset, share in book.marginal_var_contribution().items():
        print(f"  {asset:<12}{share:>8.1%}")

    # ---------------------------------------------------------------- VaR
    banner(f"VALUE AT RISK — {args.confidence:.0%} confidence, 1-day horizon")
    comparison = compare_methods(
        book.asset_returns, book.w, args.confidence, portfolio_value=book.value
    )
    for method, row in comparison.iterrows():
        line = (
            f"{method:<22} VaR {row['VaR']:>7.3%}  ES {row['ES']:>7.3%}   "
            f"{row['VaR_amount']:>12,.0f}"
        )
        print(line)
        if row["note"]:
            print(f"{'':<22} {row['note']}")

    spread = comparison["VaR"].max() / comparison["VaR"].min() - 1
    print(f"\nSpread between the most and least conservative estimate: {spread:.1%}")
    print("The methods disagree because they assume different tail behaviour.")

    banner("MULTI-DAY HORIZON (square-root-of-time)")
    for days in (1, 5, 10):
        res = historical_var(book.returns, args.confidence, horizon_days=days,
                             portfolio_value=book.value)
        print(f"{days:>3}-day   VaR {res.var:>7.3%}   {res.var_amount:>12,.0f}")
    print("\nThis rule assumes i.i.d. returns. Volatility clusters, so it")
    print("understates risk in turbulent periods. Basel permits it anyway.")

    # ---------------------------------------------------------------- backtest
    banner(f"WALK-FORWARD BACKTEST — {args.window}-day rolling window")
    for method, dist, label in [
        ("historical", "normal", "Historical simulation"),
        ("parametric", "normal", "Parametric (normal)"),
        ("parametric", "t", "Parametric (Student-t)"),
    ]:
        out = run_full_backtest(
            book.returns, window=args.window, confidence=args.confidence,
            method=method, distribution=dist,
        )
        basel = out["basel"]
        print(f"\n{label}")
        print(f"  {out['n_exceptions']} exceptions in {out['n_days']} days "
              f"= {out['exception_rate']:.2%} (expected {out['expected_rate']:.2%})")
        print(f"  {out['kupiec']}")
        print(f"     {out['kupiec'].detail}")
        print(f"  {out['christoffersen']}")
        print(f"     {out['christoffersen'].detail}")
        print(f"  {out['conditional_coverage']}")
        print(f"  Basel zone: {basel['zone']}  "
              f"(capital multiplier {basel['capital_multiplier']:.2f}) — {basel['reading']}")

        if label.startswith("Historical"):
            best = out

    # ---------------------------------------------------------------- report
    var_levels = {
        "historical": historical_var(book.returns, args.confidence).var,
        "normal": parametric_var(book.returns, args.confidence, distribution="normal").var,
        "student-t": parametric_var(book.returns, args.confidence, distribution="t").var,
    }
    path = risk_dashboard(best["frame"], book.returns, var_levels, comparison, args.out)

    banner("OUTPUT")
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
