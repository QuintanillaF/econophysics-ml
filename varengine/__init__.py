"""varengine — Value at Risk estimation and regulatory backtesting.

Public API. Import from here rather than the submodules:

    from varengine import Portfolio, compare_methods, run_full_backtest

Layout:
    data.py        market data loading; GARCH(1,1) simulator with t innovations
    portfolio.py   weights, returns, covariance, risk decomposition, component VaR
    volatility.py  EWMA and GARCH(1,1) conditional-volatility models
    var.py         historical / parametric / Monte Carlo / filtered-historical VaR and ES
    evt.py         extreme value theory (peaks-over-threshold GPD) for the far tail
    backtest.py    Kupiec, Christoffersen, conditional coverage, ES test, Basel zones
    stress.py      historical-replay, hypothetical and reverse stress testing
    factors.py     Fama-French / style-ETF factor model and risk/return decomposition
    diagnostics.py PSR, deflated Sharpe, probability of backtest overfitting, purged CV
    plots.py       report figures (import as ``varengine.plots``)
"""

from __future__ import annotations

from .backtest import (
    BacktestResult,
    basel_traffic_light,
    christoffersen_independence,
    conditional_coverage,
    es_backtest_acerbi_szekely,
    kupiec_pof,
    rolling_var_backtest,
    run_full_backtest,
)
from .data import GarchParams, load_market_data, simulate_market
from .diagnostics import (
    PurgedKFold,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from .evt import GPDFit, evt_expected_shortfall, evt_var, fit_gpd
from .factors import FactorModel, fit_factor_model, load_fama_french, style_factors
from .portfolio import Portfolio, ledoit_wolf_cov, log_returns, simple_returns
from .stress import (
    HISTORICAL_SCENARIOS,
    SHOCK_LIBRARY,
    StressResult,
    factor_betas,
    historical_stress,
    parametric_stress,
    reverse_stress,
    run_stress_suite,
)
from .var import (
    VaRResult,
    compare_methods,
    expected_shortfall,
    filtered_historical_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
    scale_horizon,
)
from .volatility import VolModel, ewma_model, ewma_volatility, garch11_fit

__version__ = "0.2.0"

__all__ = [
    # data
    "GarchParams",
    "load_market_data",
    "simulate_market",
    # portfolio
    "Portfolio",
    "ledoit_wolf_cov",
    "log_returns",
    "simple_returns",
    # volatility
    "VolModel",
    "ewma_model",
    "ewma_volatility",
    "garch11_fit",
    # var
    "VaRResult",
    "compare_methods",
    "expected_shortfall",
    "filtered_historical_var",
    "historical_var",
    "monte_carlo_var",
    "parametric_var",
    "scale_horizon",
    # evt
    "GPDFit",
    "evt_expected_shortfall",
    "evt_var",
    "fit_gpd",
    # factor model
    "FactorModel",
    "fit_factor_model",
    "load_fama_french",
    "style_factors",
    # backtest diagnostics
    "PurgedKFold",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    # backtest
    "BacktestResult",
    "basel_traffic_light",
    "christoffersen_independence",
    "conditional_coverage",
    "es_backtest_acerbi_szekely",
    "kupiec_pof",
    "rolling_var_backtest",
    "run_full_backtest",
    # stress
    "HISTORICAL_SCENARIOS",
    "SHOCK_LIBRARY",
    "StressResult",
    "factor_betas",
    "historical_stress",
    "parametric_stress",
    "reverse_stress",
    "run_stress_suite",
]
