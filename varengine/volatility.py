"""Conditional volatility models.

Every estimator in ``var.py`` assumes the return distribution is stable over the
estimation window. It is not — volatility clusters, so a static 250-day sigma is
too high after a calm stretch and too low right when a shock hits. That single
assumption is what a walk-forward backtest exposes as clustered exceptions
(Christoffersen) and, often, as excess exceptions (Kupiec).

Two conditional models, in increasing order of sophistication and cost:

- **EWMA** (RiskMetrics): ``sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r2_{t-1}``.
  One parameter, fixed by convention at ``lambda = 0.94`` for daily data. No
  fitting, so it is cheap enough to run inside a rolling backtest. It is a GARCH
  with ``omega = 0`` and ``alpha + beta = 1`` (an IGARCH), which is why it has no
  mean reversion in the variance.

- **GARCH(1,1)**: ``sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}``.
  Three parameters fitted by maximum likelihood. Mean-reverts to a long-run
  variance ``omega / (1 - alpha - beta)``, so its multi-day forecasts pull toward
  the unconditional level instead of staying flat. The cost is an optimisation
  per fit, which makes a refit-every-day backtest slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

__all__ = ["VolModel", "ewma_volatility", "garch11_fit"]


def _clean(returns: pd.Series | np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 50:
        raise ValueError(
            f"need at least 50 observations to estimate conditional volatility; got {r.size}"
        )
    return r


def ewma_volatility(
    returns: pd.Series | np.ndarray, lam: float = 0.94
) -> pd.Series | np.ndarray:
    """Exponentially weighted conditional volatility (RiskMetrics).

    Returns the one-step-ahead sigma for each date: ``sigma_t`` uses information
    up to ``t - 1``, so the series is directly usable as a VaR scale without
    look-ahead. The seed is the sample standard deviation of the first 20
    observations.

    ``lam`` is the decay: 0.94 is the RiskMetrics daily convention (a ~33-day
    effective memory), 0.97 the monthly one. Lower means faster reaction and a
    noisier estimate.
    """
    if not 0.80 <= lam < 1.0:
        raise ValueError(f"lambda must lie in [0.80, 1); got {lam}")
    r = np.asarray(returns, dtype=float)
    mask = np.isfinite(r)
    r_clean = r[mask]
    if r_clean.size < 30:
        raise ValueError(f"need at least 30 finite observations; got {r_clean.size}")

    var = np.empty(r_clean.size)
    seed = float(np.var(r_clean[: min(20, r_clean.size)], ddof=1))
    prev = seed
    for i in range(r_clean.size):
        var[i] = prev
        prev = lam * prev + (1.0 - lam) * r_clean[i] ** 2

    sigma_clean = np.sqrt(var)

    if isinstance(returns, pd.Series):
        out = pd.Series(np.nan, index=returns.index, name="ewma_sigma")
        out.loc[returns.index[mask]] = sigma_clean
        return out.ffill().bfill()
    full = np.full(r.size, np.nan)
    full[mask] = sigma_clean
    return full


@dataclass
class VolModel:
    """A fitted conditional-volatility model.

    ``sigma`` is the in-sample one-step-ahead volatility path. ``forecast(h)``
    projects the volatility ``h`` days out from the end of the sample.
    """

    kind: str
    sigma: pd.Series
    params: dict[str, float]
    last_variance: float
    last_shock2: float
    _long_run_var: float = field(default=np.nan)

    @property
    def persistence(self) -> float:
        return self.params.get("alpha", 0.0) + self.params.get("beta", 0.0)

    @property
    def annualized_vol(self) -> float:
        """Current conditional volatility, annualised (252 trading days)."""
        return float(np.sqrt(self.next_variance()) * np.sqrt(252))

    def next_variance(self) -> float:
        """One-step-ahead conditional variance from the last observation."""
        if self.kind == "ewma":
            lam = self.params["lambda"]
            return lam * self.last_variance + (1.0 - lam) * self.last_shock2
        w, a, b = self.params["omega"], self.params["alpha"], self.params["beta"]
        return w + a * self.last_shock2 + b * self.last_variance

    def forecast(self, horizon_days: int = 1) -> float:
        """Volatility (not variance) forecast ``horizon_days`` ahead.

        For GARCH the multi-step variance mean-reverts to the long-run level:
        ``E[sigma2_{t+k}] = sigma2_inf + (alpha + beta)^(k-1) * (sigma2_{t+1} - sigma2_inf)``.
        EWMA has no mean reversion, so every horizon returns the same sigma
        scaled by ``sqrt(horizon)`` — the square-root-of-time rule, which is
        exactly why EWMA multi-day VaR inherits that rule's flaws.
        """
        if horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")
        v1 = self.next_variance()
        if self.kind == "ewma":
            return float(np.sqrt(v1 * horizon_days))
        p = self.persistence
        v_inf = self._long_run_var
        total = 0.0
        for k in range(horizon_days):
            total += v_inf + (p**k) * (v1 - v_inf)
        return float(np.sqrt(total))


def _garch_recursion(params: np.ndarray, r2: np.ndarray, seed_var: float) -> np.ndarray:
    omega, alpha, beta = params
    n = r2.size
    var = np.empty(n)
    prev = seed_var
    for i in range(n):
        var[i] = prev
        prev = omega + alpha * r2[i] + beta * prev
    return var


def garch11_fit(
    returns: pd.Series | np.ndarray, dist: str = "normal"
) -> VolModel:
    """Fit a GARCH(1,1) by maximum likelihood.

    The mean is treated as a constant and removed first (daily equity drift is
    negligible next to the volatility, but leaving it in biases the variance
    slightly). Parameters are found by minimising the negative log-likelihood
    under a normal or Student-t innovation, with ``alpha + beta < 1`` enforced so
    the variance process is stationary.

    ``dist="t"`` fits the degrees of freedom jointly. It is the honest choice for
    daily returns — even after GARCH removes the volatility clustering, the
    standardised residuals still have fat tails — but ``"normal"`` is faster and
    usually enough for the volatility path itself.
    """
    r = _clean(returns)
    mu = float(r.mean())
    eps = r - mu
    r2 = eps**2
    sample_var = float(eps.var(ddof=1))

    def neg_loglik(theta: np.ndarray) -> float:
        if dist == "t":
            omega, alpha, beta, nu = theta
            if nu <= 2.05:
                return 1e10
        else:
            omega, alpha, beta = theta
            nu = None
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        var = _garch_recursion(np.array([omega, alpha, beta]), r2, sample_var)
        if np.any(var <= 0) or not np.all(np.isfinite(var)):
            return 1e10
        if dist == "t":
            scale = np.sqrt(var * (nu - 2.0) / nu)
            ll = stats.t.logpdf(eps, df=nu, scale=scale)
        else:
            ll = -0.5 * (np.log(2 * np.pi) + np.log(var) + r2 / var)
        return -float(np.sum(ll))

    # Start from variance targeting: omega = (1 - persistence) * sample_var.
    if dist == "t":
        x0 = [0.1 * sample_var, 0.08, 0.90, 6.0]
        bounds = [(1e-12, sample_var), (1e-6, 0.5), (1e-6, 0.999), (2.1, 50.0)]
    else:
        x0 = [0.1 * sample_var, 0.08, 0.90]
        bounds = [(1e-12, sample_var), (1e-6, 0.5), (1e-6, 0.999)]

    res = optimize.minimize(
        neg_loglik, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-11},
    )
    theta = res.x
    omega, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
    persistence = alpha + beta
    long_run_var = omega / (1.0 - persistence) if persistence < 1 else sample_var

    var_path = _garch_recursion(np.array([omega, alpha, beta]), r2, sample_var)
    sigma_path = np.sqrt(var_path)

    params = {"omega": omega, "alpha": alpha, "beta": beta}
    if dist == "t":
        params["nu"] = float(theta[3])

    if isinstance(returns, pd.Series):
        idx = returns.index[np.isfinite(np.asarray(returns, dtype=float))]
        sigma = pd.Series(sigma_path, index=idx, name="garch_sigma")
    else:
        sigma = pd.Series(sigma_path, name="garch_sigma")

    return VolModel(
        kind="garch",
        sigma=sigma,
        params=params,
        last_variance=float(var_path[-1]),
        last_shock2=float(r2[-1]),
        _long_run_var=float(long_run_var),
    )


def ewma_model(returns: pd.Series | np.ndarray, lam: float = 0.94) -> VolModel:
    """Wrap :func:`ewma_volatility` in a :class:`VolModel` for a uniform interface."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sigma = ewma_volatility(returns, lam)
    sigma_arr = np.asarray(sigma if not isinstance(sigma, pd.Series) else sigma.dropna())
    return VolModel(
        kind="ewma",
        sigma=sigma if isinstance(sigma, pd.Series) else pd.Series(sigma_arr),
        params={"lambda": lam},
        last_variance=float(sigma_arr[-1] ** 2),
        last_shock2=float(r[-1] ** 2),
    )


__all__.append("ewma_model")
