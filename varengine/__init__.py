"""varengine — Value at Risk estimation and regulatory backtesting.

Public API. Import from here rather than the submodules:

    from varengine import Portfolio, compare_methods, run_full_backtest

Layout:
    data.py       market data loading; GARCH(1,1) simulator with t innovations
    portfolio.py  weights, returns, covariance, risk decomposition
    var.py        historical / parametric / Monte Carlo VaR and ES
    backtest.py   Kupiec, Christoffersen, conditional coverage, Basel zones
    plots.py      report figures (import as ``varengine.plots``)
"""

from __future__ import annotations

from .backtest import (
    BacktestResult,
    basel_traffic_light,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    rolling_var_backtest,
    run_full_backtest,
)
from .data import GarchParams, load_market_data, simulate_market
from .portfolio import Portfolio, log_returns, simple_returns
from .var import (
    VaRResult,
    compare_methods,
    expected_shortfall,
    historical_var,
    monte_carlo_var,
    parametric_var,
    scale_horizon,
)

__version__ = "0.1.0"

__all__ = [
    # data
    "GarchParams",
    "load_market_data",
    "simulate_market",
    # portfolio
    "Portfolio",
    "log_returns",
    "simple_returns",
    # var
    "VaRResult",
    "compare_methods",
    "expected_shortfall",
    "historical_var",
    "monte_carlo_var",
    "parametric_var",
    "scale_horizon",
    # backtest
    "BacktestResult",
    "basel_traffic_light",
    "christoffersen_independence",
    "conditional_coverage",
    "kupiec_pof",
    "rolling_var_backtest",
    "run_full_backtest",
]
