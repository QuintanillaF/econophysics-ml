"""Value at Risk and Expected Shortfall estimators.

Sign convention, fixed once and used everywhere: **VaR is reported as a positive
number representing a loss**. A 1-day 99% VaR of 0.031 means "on 1 day in 100 we
expect to lose more than 3.1% of portfolio value". Returns themselves stay
signed as usual, so a loss is a negative return; the estimators negate the lower
quantile on the way out.

Three estimators are implemented because they disagree, and the disagreement is
the interesting part:

- **Historical** makes no distributional assumption. It reuses the empirical
  distribution, so fat tails and skew come along for free — but it can only
  produce losses it has already seen, and it weights a crash from two years ago
  the same as yesterday.

- **Parametric** assumes a shape (normal or Student-t) and needs only a mean and
  a covariance. Fast, smooth, and analytically decomposable — but under the
  normal assumption it systematically understates tail risk, because real
  returns have far more mass in the extremes than a Gaussian allows.

- **Monte Carlo** simulates forward from an assumed joint distribution. It
  handles non-linear instruments the other two cannot, at the cost of
  simulation error and a much heavier dependence on the assumed model.

Expected Shortfall is included alongside because Basel III / FRTB replaced VaR
with ES at 97.5% as the regulatory measure for market risk. VaR tells you the
threshold; ES tells you the average loss once you are past it. VaR is also not
sub-additive in general — it can say a diversified book is riskier than its
parts — while ES is coherent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "VaRResult",
    "compare_methods",
    "expected_shortfall",
    "historical_var",
    "monte_carlo_var",
    "parametric_var",
    "scale_horizon",
]

Method = Literal["historical", "parametric", "monte_carlo"]


@dataclass(frozen=True)
class VaRResult:
    """A VaR estimate together with everything needed to interpret it."""

    var: float
    expected_shortfall: float
    confidence: float
    method: str
    horizon_days: int = 1
    portfolio_value: float | None = None
    n_observations: int | None = None
    note: str = ""

    @property
    def var_amount(self) -> float | None:
        """VaR expressed in currency units, if a portfolio value was supplied."""
        return None if self.portfolio_value is None else self.var * self.portfolio_value

    @property
    def es_amount(self) -> float | None:
        if self.portfolio_value is None:
            return None
        return self.expected_shortfall * self.portfolio_value

    def __str__(self) -> str:
        head = (
            f"{self.method} | {self.confidence:.1%} | {self.horizon_days}d  "
            f"VaR {self.var:>7.3%}   ES {self.expected_shortfall:>7.3%}"
        )
        if self.portfolio_value is not None:
            head += f"   ({self.var_amount:,.0f} / {self.es_amount:,.0f})"
        return head


def _validate(returns: pd.Series | np.ndarray, confidence: float) -> np.ndarray:
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0.5, 1); got {confidence}")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError(
            f"need at least 30 observations for a meaningful estimate; got {r.size}"
        )
    return r


def scale_horizon(var_1d: float, horizon_days: int) -> float:
    """Scale a 1-day VaR to a longer horizon by the square root of time.

    This rests on returns being independent and identically distributed. They
    are not: volatility clusters, so a 10-day VaR built this way understates
    risk in turbulent periods and overstates it in calm ones. Basel permits the
    approximation, which is the only reason it is so widespread. Treat it as a
    convention, not a result.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    return var_1d * np.sqrt(horizon_days)


def historical_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
) -> VaRResult:
    """Empirical-quantile VaR.

    Takes the ``(1 - confidence)`` quantile of the realised return distribution
    and reports it as a positive loss. No distributional assumption is made.

    The binding limitation is sample size in the tail. At 99% confidence, 250
    observations place only 2.5 of them beyond the threshold, so the estimate is
    driven by two or three data points and is correspondingly unstable.
    """
    r = _validate(returns, confidence)
    q = np.quantile(r, 1.0 - confidence)
    var = float(-q)
    es = float(-r[r <= q].mean()) if (r <= q).any() else var

    tail_n = int((r <= q).sum())
    note = ""
    if tail_n < 5:
        note = (
            f"only {tail_n} observation(s) beyond the threshold — "
            "the tail estimate rests on very few points"
        )

    if horizon_days > 1:
        var, es = scale_horizon(var, horizon_days), scale_horizon(es, horizon_days)

    return VaRResult(
        var=var,
        expected_shortfall=es,
        confidence=confidence,
        method="historical",
        horizon_days=horizon_days,
        portfolio_value=portfolio_value,
        n_observations=r.size,
        note=note,
    )


def parametric_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
    distribution: Literal["normal", "t"] = "normal",
    min_nu: float = 2.05,
) -> VaRResult:
    """Variance-covariance VaR under an assumed distribution.

    Normal:   ``VaR = -(mu + z_a * sigma)`` with ``z_a`` the standard normal
              quantile at ``1 - confidence``.
    Student-t: same shape, but the quantile comes from a t distribution whose
              degrees of freedom are fitted by maximum likelihood and whose
              scale is adjusted so the fitted variance matches the sample.

    The normal version is the one banks used before 2008 and the one that failed
    then. Fitting a t is a cheap and substantial improvement: with the degrees of
    freedom typical of daily equity data (3-6), the 99% quantile sits materially
    further out than the Gaussian equivalent.

    ``min_nu`` floors the fitted degrees of freedom. Maximum-likelihood fitting
    of nu is unstable on short windows and can return values below 2, where the
    variance does not exist and the rescaling below is undefined. Clipping keeps
    a rolling backtest running and errs toward the fatter tail — the conservative
    direction for a risk number — but the clip is recorded in ``note`` rather
    than hidden, because a window that hits the floor is telling you the sample
    is too heavy-tailed for this estimator to be trusted.
    """
    r = _validate(returns, confidence)
    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    alpha = 1.0 - confidence
    note = ""

    if min_nu <= 2.0:
        raise ValueError("min_nu must exceed 2 or the variance is undefined")

    if distribution == "normal":
        z = stats.norm.ppf(alpha)
        var = -(mu + z * sigma)
        # ES for a normal: mu - sigma * phi(z) / alpha
        es = -(mu - sigma * stats.norm.pdf(z) / alpha)
        label = "parametric-normal"
    elif distribution == "t":
        nu_hat, _loc, _scale = stats.t.fit(r)
        nu = max(float(nu_hat), min_nu)
        # Rescale so the t has the sample standard deviation.
        std_t = np.sqrt(nu / (nu - 2.0))
        t_q = stats.t.ppf(alpha, nu) / std_t
        var = -(mu + t_q * sigma)
        # ES for a standardised t, rescaled to the sample sigma.
        pdf_q = stats.t.pdf(stats.t.ppf(alpha, nu), nu)
        es_std = -(pdf_q / alpha) * (nu + stats.t.ppf(alpha, nu) ** 2) / (nu - 1.0)
        es = -(mu + (es_std / std_t) * sigma)
        label = "parametric-t"
        note = f"fitted nu = {nu:.2f} (lower means fatter tails)"
        if nu_hat < min_nu:
            note += (
                f"; MLE returned {nu_hat:.2f} and was clipped to the floor — "
                "treat this window's estimate with caution"
            )
    else:
        raise ValueError("distribution must be 'normal' or 't'")

    if horizon_days > 1:
        var, es = scale_horizon(var, horizon_days), scale_horizon(es, horizon_days)

    return VaRResult(
        var=float(var),
        expected_shortfall=float(es),
        confidence=confidence,
        method=label,
        horizon_days=horizon_days,
        portfolio_value=portfolio_value,
        n_observations=r.size,
        note=note,
    )


def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    n_simulations: int = 100_000,
    portfolio_value: float | None = None,
    distribution: Literal["normal", "t"] = "normal",
    nu: float = 5.0,
    seed: int | None = 42,
) -> VaRResult:
    """Monte Carlo VaR from a fitted multivariate distribution.

    Estimates the mean vector and covariance matrix from the asset returns,
    draws ``n_simulations`` joint paths over the horizon, aggregates to
    portfolio level, and reads off the empirical quantile.

    Under a multivariate normal with a single-day horizon this converges to the
    parametric answer — that agreement is worth verifying, since it is a free
    check that the plumbing is right. Its real value appears when the
    distribution is non-normal, the horizon is multi-day, or the book contains
    instruments whose payoff is not linear in the risk factors.

    The Student-t path uses a Gaussian copula with t marginals, so cross-asset
    dependence stays linear while each margin carries fat tails.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0.5, 1); got {confidence}")
    if n_simulations < 1000:
        raise ValueError("use at least 1000 simulations for a usable tail estimate")
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    w = np.asarray(weights, dtype=float)
    if w.shape[0] != asset_returns.shape[1]:
        raise ValueError(
            f"weights has {w.shape[0]} entries but there are "
            f"{asset_returns.shape[1]} assets"
        )

    rng = np.random.default_rng(seed)
    mu = np.asarray(asset_returns.mean())
    cov = np.asarray(asset_returns.cov())

    # Nearest-PSD guard: sample covariance can be slightly indefinite from
    # numerical error, which would break the Cholesky factorisation.
    eigvals, eigvecs = np.linalg.eigh(cov)
    if (eigvals < 0).any():
        eigvals = np.clip(eigvals, 1e-14, None)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(len(mu)))

    k = len(mu)
    note = ""
    if distribution == "normal":
        z = rng.standard_normal((n_simulations, horizon_days, k))
        label = "monte-carlo-normal"
    elif distribution == "t":
        if nu <= 2:
            raise ValueError("nu must exceed 2")
        z = rng.standard_t(nu, size=(n_simulations, horizon_days, k))
        z /= np.sqrt(nu / (nu - 2.0))  # standardise to unit variance
        label = "monte-carlo-t"
        note = f"t innovations with nu = {nu}"
    else:
        raise ValueError("distribution must be 'normal' or 't'")

    shocks = z @ chol.T + mu  # (n_sim, horizon, k)
    port_paths = shocks @ w  # (n_sim, horizon)
    sims = port_paths.sum(axis=1)  # log returns add across time

    q = np.quantile(sims, 1.0 - confidence)
    var = float(-q)
    es = float(-sims[sims <= q].mean())

    return VaRResult(
        var=var,
        expected_shortfall=es,
        confidence=confidence,
        method=label,
        horizon_days=horizon_days,
        portfolio_value=portfolio_value,
        n_observations=int(asset_returns.shape[0]),
        note=note or f"{n_simulations:,} simulations",
    )


def expected_shortfall(
    returns: pd.Series | np.ndarray, confidence: float = 0.975
) -> float:
    """Standalone empirical Expected Shortfall (also called CVaR).

    The mean loss conditional on breaching the VaR threshold. Basel III sets the
    regulatory confidence level at 97.5% for ES, which is calibrated to be
    roughly comparable in severity to 99% VaR under a normal distribution — and
    strictly more conservative when tails are fat.
    """
    r = _validate(returns, confidence)
    q = np.quantile(r, 1.0 - confidence)
    tail = r[r <= q]
    return float(-tail.mean()) if tail.size else float(-q)


def compare_methods(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Run every estimator on the same data and tabulate the results.

    Reading the table is the point of the exercise. Normal-based estimates
    clustering below the historical and t-based ones is the signature of fat
    tails, and it is the single most common way a production risk model
    understates what it is supposed to measure.
    """
    port = asset_returns @ np.asarray(weights, dtype=float)

    results = [
        historical_var(port, confidence, horizon_days, portfolio_value),
        parametric_var(port, confidence, horizon_days, portfolio_value, "normal"),
        parametric_var(port, confidence, horizon_days, portfolio_value, "t"),
        monte_carlo_var(
            asset_returns, weights, confidence, horizon_days,
            portfolio_value=portfolio_value, distribution="normal", seed=seed,
        ),
        monte_carlo_var(
            asset_returns, weights, confidence, horizon_days,
            portfolio_value=portfolio_value, distribution="t", seed=seed,
        ),
    ]

    rows = []
    for res in results:
        rows.append(
            {
                "method": res.method,
                "VaR": res.var,
                "ES": res.expected_shortfall,
                "VaR_amount": res.var_amount,
                "ES_amount": res.es_amount,
                "note": res.note,
            }
        )
    return pd.DataFrame(rows).set_index("method")
