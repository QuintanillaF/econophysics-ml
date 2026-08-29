"""Extreme Value Theory for the tail beyond the VaR threshold.

Historical simulation cannot produce a loss it has never seen, and at 99.5% or
99.9% confidence there are almost no observations left to estimate from — 1000
days give you 5 beyond the 99.5% point. EVT sidesteps this. The
Pickands-Balkema-de Haan theorem says that for a wide class of distributions, the
exceedances over a high enough threshold converge to a **Generalised Pareto
Distribution (GPD)** regardless of the parent distribution. So instead of
trusting the empirical tail, you fit a two-parameter GPD to the handful of
exceedances and read the extreme quantiles off the fitted model.

The shape parameter ``xi`` is the whole story:

- ``xi > 0`` — heavy tail (power law). The tail index is ``1/xi``; moments above
  ``1/xi`` do not exist. Daily equity returns typically sit at ``xi ≈ 0.2-0.3``.
- ``xi = 0`` — exponential tail (the Gumbel case; a normal distribution is here).
- ``xi < 0`` — bounded tail (there is a hard maximum loss).

The cost is the threshold choice: too low and the GPD approximation is biased,
too high and there are too few points to fit. The default 95th percentile is the
common compromise; the fit records how many exceedances it used so you can judge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .var import VaRResult, _validate

__all__ = ["GPDFit", "evt_expected_shortfall", "evt_var", "fit_gpd"]


@dataclass(frozen=True)
class GPDFit:
    """A Generalised Pareto fit to the losses beyond a threshold."""

    xi: float          # shape; > 0 means a heavy (power-law) tail
    beta: float        # scale
    threshold: float   # u, the loss level above which the GPD is fitted
    n: int             # total observations
    n_exceed: int      # observations above the threshold

    @property
    def tail_index(self) -> float:
        """``1 / xi`` — moments of order >= this do not exist. inf if xi <= 0."""
        return float("inf") if self.xi <= 0 else 1.0 / self.xi

    def __str__(self) -> str:
        return (
            f"GPD  xi={self.xi:+.3f}  beta={self.beta:.4f}  "
            f"u={self.threshold:.3%}  ({self.n_exceed}/{self.n} exceedances)"
        )


def fit_gpd(
    returns: pd.Series | np.ndarray, threshold_q: float = 0.95
) -> GPDFit:
    """Fit a GPD to the lower-tail losses via Peaks-Over-Threshold.

    Losses are ``-returns``; the threshold ``u`` is the ``threshold_q`` quantile
    of the losses, and the GPD is fitted by maximum likelihood to the amounts by
    which losses exceed ``u``.
    """
    if not 0.80 <= threshold_q < 0.99:
        raise ValueError(f"threshold_q must lie in [0.80, 0.99); got {threshold_q}")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 250:
        raise ValueError(
            f"EVT needs a long sample to have enough tail points; got {r.size} (want >= 250)"
        )

    losses = -r
    u = float(np.quantile(losses, threshold_q))
    exceed = losses[losses > u] - u
    if exceed.size < 10:
        raise ValueError(
            f"only {exceed.size} exceedances above the {threshold_q:.0%} threshold — "
            "lower threshold_q or supply more data"
        )

    xi, _loc, beta = stats.genpareto.fit(exceed, floc=0.0)

    return GPDFit(
        xi=float(xi),
        beta=float(beta),
        threshold=u,
        n=int(r.size),
        n_exceed=int(exceed.size),
    )


def _evt_quantile(fit: GPDFit, confidence: float) -> float:
    """Loss level (positive) at ``confidence``, from the fitted GPD."""
    p_exceed = fit.n_exceed / fit.n
    if 1.0 - confidence >= p_exceed:
        raise ValueError(
            f"confidence {confidence:.4f} is inside the threshold "
            f"({1 - p_exceed:.4f}); EVT only speaks about the tail beyond it"
        )
    ratio = (fit.n / fit.n_exceed) * (1.0 - confidence)
    if abs(fit.xi) < 1e-8:
        return fit.threshold + fit.beta * (-np.log(ratio))
    return fit.threshold + (fit.beta / fit.xi) * (ratio ** (-fit.xi) - 1.0)


def evt_var(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.99,
    threshold_q: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float | None = None,
) -> VaRResult:
    """VaR from a GPD fit to the tail. Meant for confidence >= 99%."""
    r = _validate(returns, confidence)
    fit = fit_gpd(r, threshold_q)
    var = _evt_quantile(fit, confidence)
    es = evt_expected_shortfall(r, confidence, threshold_q)

    if horizon_days > 1:
        scale = np.sqrt(horizon_days)
        var, es = var * scale, es * scale

    note = f"{fit}; tail index 1/xi = {fit.tail_index:.1f}"
    if fit.xi <= 0:
        note += " (xi <= 0: tail not heavier than exponential)"

    return VaRResult(
        var=float(var),
        expected_shortfall=float(es),
        confidence=confidence,
        method="evt-pot",
        horizon_days=horizon_days,
        portfolio_value=portfolio_value,
        n_observations=fit.n,
        note=note,
    )


def evt_expected_shortfall(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.975,
    threshold_q: float = 0.95,
) -> float:
    """Expected Shortfall from the GPD fit.

    Closed form once the GPD is fitted: ``ES = VaR / (1 - xi) + (beta - xi*u) / (1 - xi)``,
    valid for ``xi < 1`` (for ``xi >= 1`` the mean of the tail is infinite and ES
    is undefined — which is itself a useful warning).
    """
    r = _validate(returns, confidence)
    fit = fit_gpd(r, threshold_q)
    if fit.xi >= 1.0:
        raise ValueError(
            f"fitted xi = {fit.xi:.2f} >= 1: the tail mean is infinite, ES undefined"
        )
    var = _evt_quantile(fit, confidence)
    return float(var / (1.0 - fit.xi) + (fit.beta - fit.xi * fit.threshold) / (1.0 - fit.xi))
