"""Portfolio construction and return computation.

Keeping this separate from the VaR estimators matters more than it looks: every
risk number downstream inherits whatever convention is fixed here, so the
choices are stated explicitly rather than buried in a one-line ``pct_change()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["Portfolio", "ledoit_wolf_cov", "log_returns", "simple_returns"]


def ledoit_wolf_cov(asset_returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage estimate of the covariance matrix.

    The sample covariance is unbiased but noisy: with N assets and T
    observations it has ``N(N+1)/2`` free parameters, and when N is not small
    relative to T the extreme eigenvalues are badly estimated — which is exactly
    what a risk model leans on. Shrinkage pulls the sample matrix toward a
    structured target (here a scaled identity), trading a little bias for a large
    variance reduction. The shrinkage intensity is chosen analytically to
    minimise expected error, so there is nothing to tune.

    Falls back to the sample covariance for a single asset (nothing to shrink).
    """
    r = asset_returns.to_numpy(dtype=float)
    if r.shape[1] < 2:
        return np.atleast_2d(np.cov(r, rowvar=False))
    from sklearn.covariance import LedoitWolf

    return LedoitWolf().fit(r).covariance_


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
    shrink: bool = False  # use Ledoit-Wolf covariance for .cov and downstream risk
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
        """Covariance matrix of asset returns (daily).

        Ledoit-Wolf shrinkage if ``shrink=True`` was passed, otherwise the plain
        sample covariance.
        """
        if self.shrink:
            return self.cov_ledoit_wolf
        return np.asarray(self.asset_returns.cov())

    @property
    def cov_ledoit_wolf(self) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance, regardless of the ``shrink`` flag."""
        return ledoit_wolf_cov(self.asset_returns)

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

    def component_var(
        self, confidence: float = 0.99, distribution: str = "normal"
    ) -> pd.DataFrame:
        """Decompose the parametric VaR into per-asset contributions.

        Answers the question that always follows the headline number — *which
        position is the risk actually coming from?* — in VaR units rather than
        variance units.

        - **marginal**: ``d VaR / d w_i`` — the extra VaR from a marginal increase
          in position i. Equal to ``(Sigma w)_i / sigma_p * z``.
        - **component**: ``w_i * marginal_i``. The components sum exactly to the
          portfolio VaR (for a zero-mean approximation), so ``component_pct`` is a
          clean risk budget.
        - **incremental**: ``VaR(book) - VaR(book without i, renormalised)`` — the
          VaR you would shed by removing the position entirely. Unlike the
          component version this captures the non-linear diversification effect.

        The mean term is dropped from the marginal/component figures (standard for
        a 1-day horizon where drift is negligible) but kept in the total and in
        the incremental figures.
        """
        if distribution not in ("normal", "t"):
            raise ValueError("distribution must be 'normal' or 't'")
        cov = self.cov
        w = self._weights
        sigma_p = float(np.sqrt(w @ cov @ w))
        if np.isclose(sigma_p, 0.0):
            raise ValueError("portfolio volatility is zero; cannot decompose")

        r_port = self.returns
        if distribution == "normal":
            z = float(stats.norm.ppf(1.0 - confidence))
        else:
            nu, _loc, _scale = stats.t.fit(r_port.to_numpy())
            nu = max(float(nu), 2.05)
            z = float(stats.t.ppf(1.0 - confidence, nu) / np.sqrt(nu / (nu - 2.0)))

        marginal = -(cov @ w) / sigma_p * z          # positive loss units
        component = w * marginal
        total_component = component.sum()

        # Incremental: drop each asset, renormalise the rest, recompute VaR.
        incremental = np.empty(len(w))
        mu_p = float(r_port.mean())
        var_full = -(mu_p + z * sigma_p)
        for i in range(len(w)):
            keep = np.arange(len(w)) != i
            if not keep.any() or np.isclose(w[keep].sum(), 0.0):
                incremental[i] = var_full
                continue
            w_sub = w[keep] / w[keep].sum()
            cov_sub = cov[np.ix_(keep, keep)]
            sigma_sub = float(np.sqrt(w_sub @ cov_sub @ w_sub))
            mu_sub = float((self.asset_returns.iloc[:, keep] @ w_sub).mean())
            var_sub = -(mu_sub + z * sigma_sub)
            incremental[i] = var_full - var_sub

        return pd.DataFrame(
            {
                "marginal": marginal,
                "component": component,
                "component_pct": component / total_component,
                "incremental": incremental,
            },
            index=self.prices.columns,
        )

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
