"""Portfolio construction and return computation.

Keeping this separate from the VaR estimators matters more than it looks: every
risk number downstream inherits whatever convention is fixed here, so the
choices are stated explicitly rather than buried in a one-line ``pct_change()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Portfolio", "log_returns", "simple_returns"]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Continuously compounded returns, ``ln(P_t / P_{t-1})``.

    Log returns are additive across time, which is what makes horizon scaling
    coherent. They are *not* additive across assets, so a portfolio of log
    returns is an approximation — acceptable for daily data, where returns are
    small, and the convention most risk systems use.
    """
    if (np.asarray(prices) <= 0).any():
        raise ValueError("prices must be strictly positive to take logs")
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Arithmetic returns, ``P_t / P_{t-1} - 1``. Additive across assets."""
    return prices.pct_change().dropna()


@dataclass
class Portfolio:
    """A fixed-weight portfolio over a price history.

    Weights are held constant, which implicitly assumes daily rebalancing back
    to target. That is the standard assumption for market-risk reporting; a
    buy-and-hold book would drift, and its risk profile with it.
    """

    prices: pd.DataFrame
    weights: dict[str, float] | np.ndarray | None = None
    value: float = 1_000_000.0
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.prices.empty:
            raise ValueError("prices is empty")
        if self.value <= 0:
            raise ValueError("portfolio value must be positive")

        n = self.prices.shape[1]
        if self.weights is None:
            w = np.full(n, 1.0 / n)
        elif isinstance(self.weights, dict):
            missing = set(self.prices.columns) - set(self.weights)
            if missing:
                raise ValueError(f"no weight given for {sorted(missing)}")
            w = np.array([self.weights[c] for c in self.prices.columns], dtype=float)
        else:
            w = np.asarray(self.weights, dtype=float)
            if w.shape != (n,):
                raise ValueError(f"expected {n} weights, got {w.shape[0]}")

        total = w.sum()
        if not np.isclose(total, 1.0):
            if np.isclose(total, 0.0):
                raise ValueError("weights sum to zero")
            w = w / total  # normalise silently; long-short books can sum oddly
        self._weights = w

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def w(self) -> np.ndarray:
        return self._weights

    @property
    def asset_returns(self) -> pd.DataFrame:
        return log_returns(self.prices)

    @property
    def returns(self) -> pd.Series:
        """Daily portfolio return series."""
        r = self.asset_returns @ self._weights
        r.name = "portfolio"
        return r

    @property
    def pnl(self) -> pd.Series:
        """Daily profit and loss in currency units."""
        return self.returns * self.value

    @property
    def cov(self) -> np.ndarray:
        """Sample covariance matrix of asset returns (daily)."""
        return np.asarray(self.asset_returns.cov())

    @property
    def volatility(self) -> float:
        """Daily portfolio volatility from the covariance matrix.

        Equal to ``sqrt(w' Sigma w)``. This is the analytic route; it agrees
        with the standard deviation of the realised return series up to the
        log-vs-simple approximation.
        """
        return float(np.sqrt(self._weights @ self.cov @ self._weights))

    def marginal_var_contribution(self) -> pd.Series:
        """Each asset's share of portfolio variance.

        Component i contributes ``w_i * (Sigma w)_i / (w' Sigma w)``. The shares
        sum to one, which makes this a clean way to answer "where is the risk
        actually coming from?" — usually the first question after the headline
        VaR number.
        """
        sigma_w = self.cov @ self._weights
        contrib = self._weights * sigma_w
        total = contrib.sum()
        if np.isclose(total, 0.0):
            raise ValueError("portfolio variance is zero; cannot decompose")
        return pd.Series(contrib / total, index=self.prices.columns, name="risk_share")

    def summary(self) -> dict[str, float]:
        r = self.returns
        return {
            "observations": len(r),
            "annualised_return": float(r.mean() * 252),
            "annualised_volatility": float(r.std(ddof=1) * np.sqrt(252)),
            "skewness": float(r.skew()),
            "excess_kurtosis": float(r.kurtosis()),
            "worst_day": float(r.min()),
            "best_day": float(r.max()),
        }
