"""Backtesting: does the VaR model actually work?

Producing a VaR number is easy. Demonstrating that it is calibrated is the part
that separates a risk model from a spreadsheet, and it is what regulators
require. The logic is a hypothesis test on the exception sequence — the days
where the realised loss exceeded the forecast.

Three tests, each catching a different failure:

**Kupiec (unconditional coverage)** — are there roughly the right *number* of
exceptions? A 99% model over 500 days should breach about 5 times. Twenty
breaches means the model understates risk; zero means it wastes capital.

**Christoffersen (independence)** — are the exceptions *spread out*? A model can
have exactly the right count and still be badly broken if all of them arrive in
the same week. Clustered breaches mean the model is not reacting to volatility,
which is precisely when you need it to.

**Conditional coverage** — the two combined. This is the one to quote, because a
model must pass both to be usable.

The Basel traffic-light zone is included as well: it is the supervisory rule
that converts an exception count over 250 trading days into a capital penalty,
and it is what a risk team is actually judged on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "BacktestResult",
    "basel_traffic_light",
    "christoffersen_independence",
    "conditional_coverage",
    "kupiec_pof",
    "rolling_var_backtest",
    "run_full_backtest",
]


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a single hypothesis test on the exception sequence."""

    test: str
    statistic: float
    p_value: float
    critical_value: float
    reject: bool
    detail: str = ""

    @property
    def verdict(self) -> str:
        return "REJECT model" if self.reject else "model not rejected"

    def __str__(self) -> str:
        return (
            f"{self.test:<26} LR={self.statistic:>7.3f}  p={self.p_value:>6.4f}  "
            f"{self.verdict}"
        )


def _exception_series(
    returns: pd.Series | np.ndarray, var_forecasts: pd.Series | np.ndarray
) -> np.ndarray:
    """Boolean array: True where the realised loss exceeded the VaR forecast.

    ``var_forecasts`` are positive losses, ``returns`` are signed, so an
    exception is ``return < -VaR``.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_forecasts, dtype=float)
    if r.shape != v.shape:
        raise ValueError(f"shape mismatch: returns {r.shape} vs forecasts {v.shape}")
    if (v < 0).any():
        raise ValueError(
            "VaR forecasts must be positive losses; a negative value suggests "
            "the sign convention has been inverted somewhere upstream"
        )
    return r < -v


def kupiec_pof(
    exceptions: np.ndarray, confidence: float = 0.99, alpha: float = 0.05
) -> BacktestResult:
    """Kupiec proportion-of-failures test (unconditional coverage).

    Under the null that the model is correctly calibrated, exceptions are
    Bernoulli with probability ``p = 1 - confidence``. The likelihood ratio

        LR = -2 ln[ (1-p)^(T-N) p^N  /  (1-N/T)^(T-N) (N/T)^N ]

    is asymptotically chi-squared with one degree of freedom.
    """
    x = np.asarray(exceptions, dtype=bool)
    T, N = x.size, int(x.sum())
    if T == 0:
        raise ValueError("empty exception series")

    p = 1.0 - confidence
    pi_hat = N / T

    if N == 0:
        lr = -2.0 * (T * np.log(1.0 - p))
    elif N == T:
        lr = -2.0 * (T * np.log(p))
    else:
        ll_null = (T - N) * np.log(1.0 - p) + N * np.log(p)
        ll_alt = (T - N) * np.log(1.0 - pi_hat) + N * np.log(pi_hat)
        lr = -2.0 * (ll_null - ll_alt)

    lr = float(max(lr, 0.0))
    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))
    crit = float(stats.chi2.ppf(1.0 - alpha, df=1))

    direction = "too many" if pi_hat > p else "too few"
    return BacktestResult(
        test="Kupiec (coverage)",
        statistic=lr,
        p_value=p_value,
        critical_value=crit,
        reject=lr > crit,
        detail=(
            f"{N} exceptions in {T} days = {pi_hat:.2%} observed vs {p:.2%} "
            f"expected ({direction}; {T * p:.1f} expected)"
        ),
    )


def christoffersen_independence(
    exceptions: np.ndarray, alpha: float = 0.05
) -> BacktestResult:
    """Christoffersen test for independence of exceptions.

    Fits a first-order Markov chain to the exception sequence and tests whether
    the probability of a breach depends on whether yesterday was a breach. Under
    the null of independence the transition probabilities are equal, and the
    likelihood ratio is chi-squared with one degree of freedom.

    Rejection means the breaches cluster — the classic symptom of a model using
    a static volatility estimate against returns that exhibit volatility
    clustering.
    """
    x = np.asarray(exceptions, dtype=int)
    if x.size < 2:
        raise ValueError("need at least two observations")

    prev, curr = x[:-1], x[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    # Degenerate cases: no exceptions, or none following an exception. The test
    # has nothing to detect, so report a non-rejection rather than divide by zero.
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return BacktestResult(
            test="Christoffersen (indep.)",
            statistic=0.0,
            p_value=1.0,
            critical_value=float(stats.chi2.ppf(1.0 - alpha, df=1)),
            reject=False,
            detail=(
                f"transitions n00={n00} n01={n01} n10={n10} n11={n11} — "
                "too few to test independence"
            ),
        )

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _safe_log(v: float) -> float:
        return np.log(v) if v > 0 else 0.0

    ll_null = (n00 + n10) * _safe_log(1.0 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (
        n00 * _safe_log(1.0 - pi01)
        + n01 * _safe_log(pi01)
        + n10 * _safe_log(1.0 - pi11)
        + n11 * _safe_log(pi11)
    )
    lr = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))
    crit = float(stats.chi2.ppf(1.0 - alpha, df=1))

    return BacktestResult(
        test="Christoffersen (indep.)",
        statistic=lr,
        p_value=p_value,
        critical_value=crit,
        reject=lr > crit,
        detail=(
            f"P(breach | breach yesterday) = {pi11:.2%} vs "
            f"P(breach | calm yesterday) = {pi01:.2%}"
        ),
    )


def conditional_coverage(
    exceptions: np.ndarray, confidence: float = 0.99, alpha: float = 0.05
) -> BacktestResult:
    """Joint test of correct coverage and independence.

    ``LR_cc = LR_pof + LR_ind``, chi-squared with two degrees of freedom. This is
    the headline result: a model that fails here is not fit for use regardless of
    how it performs on either component alone.
    """
    pof = kupiec_pof(exceptions, confidence, alpha)
    ind = christoffersen_independence(exceptions, alpha)

    lr = pof.statistic + ind.statistic
    p_value = float(1.0 - stats.chi2.cdf(lr, df=2))
    crit = float(stats.chi2.ppf(1.0 - alpha, df=2))

    return BacktestResult(
        test="Conditional coverage",
        statistic=lr,
        p_value=p_value,
        critical_value=crit,
        reject=lr > crit,
        detail=f"combines coverage (LR={pof.statistic:.2f}) and independence (LR={ind.statistic:.2f})",
    )


def basel_traffic_light(n_exceptions: int, n_days: int = 250) -> dict[str, object]:
    """Basel supervisory traffic-light zone for a 99% 1-day model.

    Thresholds are defined for a 250-day window; other window lengths are scaled
    proportionally, which is an approximation but keeps the comparison honest.

    The multiplier increment is added to the base capital factor of 3, so a red
    zone costs roughly a third more regulatory capital — the reason a risk team
    cares about this table more than about any p-value.
    """
    if n_days <= 0:
        raise ValueError("n_days must be positive")
    if n_exceptions < 0:
        raise ValueError("n_exceptions cannot be negative")

    scale = n_days / 250.0
    green_max = 4 * scale
    red_min = 10 * scale

    if n_exceptions <= green_max:
        zone, increment = "GREEN", 0.00
        reading = "model accepted; no capital penalty"
    elif n_exceptions < red_min:
        # Yellow-zone increments run from 0.40 (5 exceptions) to 0.85 (9).
        table = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
        scaled = round(n_exceptions / scale)
        zone, increment = "YELLOW", table.get(min(max(scaled, 5), 9), 0.65)
        reading = "under review; capital multiplier increased"
    else:
        zone, increment = "RED", 1.00
        reading = "model rejected; maximum penalty and mandatory remediation"

    return {
        "zone": zone,
        "exceptions": n_exceptions,
        "days": n_days,
        "multiplier_increment": increment,
        "capital_multiplier": 3.0 + increment,
        "reading": reading,
    }


def rolling_var_backtest(
    returns: pd.Series,
    window: int = 250,
    confidence: float = 0.99,
    method: str = "historical",
    distribution: str = "normal",
) -> pd.DataFrame:
    """Walk-forward backtest with an expanding-free, fixed-length rolling window.

    At each date the VaR is estimated using *only* the preceding ``window``
    observations, then compared against the return that actually followed. This
    is the crucial discipline: estimating VaR on the full sample and testing on
    the same data leaks future information and will make any model look
    excellent.

    Returns a frame indexed by date with the forecast, the realised return, and
    an exception flag.
    """
    from .var import historical_var, parametric_var  # local import avoids a cycle

    if window < 30:
        raise ValueError("window must be at least 30 observations")
    if len(returns) <= window:
        raise ValueError(
            f"need more than {window} observations to backtest; got {len(returns)}"
        )

    r = returns.dropna()
    dates, forecasts, realised = [], [], []

    for i in range(window, len(r)):
        train = r.iloc[i - window : i]
        if method == "historical":
            est = historical_var(train, confidence)
        elif method == "parametric":
            est = parametric_var(train, confidence, distribution=distribution)
        else:
            raise ValueError("method must be 'historical' or 'parametric'")

        dates.append(r.index[i])
        forecasts.append(est.var)
        realised.append(float(r.iloc[i]))

    out = pd.DataFrame(
        {"var_forecast": forecasts, "realised_return": realised}, index=pd.Index(dates, name="date")
    )
    out["exception"] = out["realised_return"] < -out["var_forecast"]
    return out


def run_full_backtest(
    returns: pd.Series,
    window: int = 250,
    confidence: float = 0.99,
    method: str = "historical",
    distribution: str = "normal",
) -> dict[str, object]:
    """Walk-forward backtest plus every test, returned as one bundle."""
    bt = rolling_var_backtest(returns, window, confidence, method, distribution)
    exc = bt["exception"].to_numpy()

    return {
        "frame": bt,
        "n_days": len(bt),
        "n_exceptions": int(exc.sum()),
        "exception_rate": float(exc.mean()),
        "expected_rate": 1.0 - confidence,
        "kupiec": kupiec_pof(exc, confidence),
        "christoffersen": christoffersen_independence(exc),
        "conditional_coverage": conditional_coverage(exc, confidence),
        "basel": basel_traffic_light(int(exc.sum()), len(bt)),
    }
