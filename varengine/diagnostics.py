"""Backtest-overfitting diagnostics.

A backtest is a claim, and the claim is almost always inflated. Every parameter
you tried, every start date you nudged, every feature you added and kept because
it "helped" — all of it is selection, and selection turns noise into a Sharpe
ratio. These tools put a number on how much of an observed result is likely to be
that.

- **Probabilistic Sharpe Ratio (PSR)** — the probability that the *true* Sharpe
  exceeds a benchmark, given the estimate, the sample length, and the return
  distribution's skew and kurtosis. A Sharpe of 2.0 on 40 observations of
  negatively-skewed returns is not the same evidence as 2.0 on 2000 near-normal
  ones, and PSR says so.

- **Deflated Sharpe Ratio (DSR)** — PSR where the benchmark is not zero but the
  Sharpe you would expect to see *by chance* after trying ``n_trials``
  configurations. Under the null that no configuration has skill, the best of N
  trials still looks good; the DSR asks whether the winner beats that bar.

- **Probability of Backtest Overfitting (PBO)** — via combinatorial symmetric
  cross-validation. Given the P&L of many trials, it repeatedly splits the
  timeline in half, picks the in-sample winner, and checks where that winner
  lands out-of-sample. If the in-sample winner is regularly below the
  out-of-sample median, the selection procedure itself is overfitting.

- **PurgedKFold** — cross-validation for labelled financial data. When a label
  spans several days (a triple-barrier outcome, a forward return), a training
  point next to the test set leaks information into it. Purging drops the
  overlapping training points; the embargo drops a buffer after the test fold.

References: Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*;
Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest
Overfitting*; López de Prado (2018), *Advances in Financial Machine Learning*,
ch. 7 and 11–12.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "PurgedKFold",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
]

_EULER_MASCHERONI = 0.5772156649015329
_PERIODS_PER_YEAR = 252


def _sharpe(returns: np.ndarray, periods: int = _PERIODS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    sr_benchmark: float = 0.0,
    periods: int = _PERIODS_PER_YEAR,
) -> float:
    """P(true annualised Sharpe > ``sr_benchmark``) given the sample.

    Uses the non-normal standard error of the Sharpe estimator (Mertens /
    Lo): ``se(SR) = sqrt((1 - skew·SR + (kurt-1)/4·SR²) / (n-1))`` with SR and
    the benchmark expressed *per period* inside the formula. Returns a
    probability in [0, 1].
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 20:
        raise ValueError(f"need at least 20 observations; got {n}")

    sd = r.std(ddof=1)
    if sd == 0:
        return 0.5
    sr = r.mean() / sd                       # per-period Sharpe
    sr_star = sr_benchmark / np.sqrt(periods)  # benchmark, per period
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # non-excess

    se = np.sqrt((1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1))
    if se <= 0:
        return 1.0 if sr > sr_star else 0.0
    return float(stats.norm.cdf((sr - sr_star) / se))


def expected_max_sharpe(
    n_trials: int,
    trials_sr_std: float = 1.0,
    periods: int = _PERIODS_PER_YEAR,
) -> float:
    """Annualised Sharpe you'd expect from the *best* of ``n_trials`` under H0.

    ``E[max] ≈ trials_sr_std · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]``
    with γ the Euler-Mascheroni constant. ``trials_sr_std`` is the standard
    deviation of the per-period Sharpe ratios across the trials (how much the
    search space spreads results); default 1.0 is a conservative placeholder.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    g = _EULER_MASCHERONI
    z = (1 - g) * stats.norm.ppf(1 - 1.0 / n_trials) + g * stats.norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    return float(trials_sr_std * z * np.sqrt(periods))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int = 20,
    trials_sr_std: float | None = None,
    periods: int = _PERIODS_PER_YEAR,
) -> dict[str, float | str]:
    """Deflated Sharpe Ratio: PSR against the by-chance benchmark of ``n_trials``.

    ``trials_sr_std`` is the spread of per-period Sharpe ratios across the
    configurations that were tried; if not supplied, a rough default of
    ``0.5 / sqrt(periods)`` per-period (≈ 0.5 annualised) is used, which is
    typical for a moderate hyper-parameter search.

    Returns a dict with the annualised Sharpe, the deflation threshold, the PSR
    and DSR probabilities, the moments used, and a one-word verdict.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sr_ann = _sharpe(r, periods)

    if trials_sr_std is None:
        trials_sr_std = 0.5 / np.sqrt(periods)
    sr_threshold = expected_max_sharpe(n_trials, trials_sr_std, periods)

    psr = probabilistic_sharpe_ratio(r, 0.0, periods)
    dsr = probabilistic_sharpe_ratio(r, sr_threshold, periods)

    verdict = (
        "credible" if dsr >= 0.95
        else "inconclusive" if dsr >= 0.50
        else "likely overfit"
    )
    return {
        "sr_annual": round(sr_ann, 3),
        "sr_threshold": round(sr_threshold, 3),
        "psr": round(psr, 4),
        "dsr": round(dsr, 4),
        "n_trials": n_trials,
        "skew": round(float(stats.skew(r)), 3),
        "kurtosis": round(float(stats.kurtosis(r, fisher=False)), 3),
        "verdict": verdict,
    }


def probability_of_backtest_overfitting(
    pnl_matrix: pd.DataFrame | np.ndarray,
    n_splits: int = 16,
) -> dict[str, float | int]:
    """PBO via combinatorial symmetric cross-validation (Bailey et al. 2017).

    ``pnl_matrix`` is ``T × N``: T time steps, one column of per-step P&L per
    trial/configuration. The timeline is cut into ``n_splits`` equal blocks; for
    every way of choosing half of them as in-sample (the rest out-of-sample):

    1. rank the trials by in-sample Sharpe, take the best,
    2. find that trial's out-of-sample rank, mapped to ``w ∈ (0, 1)``,
    3. logit ``λ = ln(w / (1 - w))``.

    PBO is the fraction of splits where ``λ < 0`` — the in-sample winner landed
    in the *worse* half out-of-sample. ~0.5 means the selection carries no
    information; low means the winner generalises.
    """
    M = np.asarray(pnl_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("pnl_matrix must be 2-D (T x N)")
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least 2 trials to assess overfitting")
    if n_splits % 2 != 0 or n_splits < 4:
        raise ValueError("n_splits must be even and >= 4")
    if T < n_splits * 2:
        raise ValueError(f"need at least {n_splits * 2} rows; got {T}")

    # Even blocks of consecutive rows.
    bounds = np.linspace(0, T, n_splits + 1, dtype=int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]

    def _sr(x: np.ndarray) -> np.ndarray:
        sd = x.std(axis=0, ddof=1)
        sd[sd == 0] = np.nan
        return np.nan_to_num(x.mean(axis=0) / sd)

    logits = []
    for is_idx in combinations(range(n_splits), n_splits // 2):
        oos_idx = [b for b in range(n_splits) if b not in is_idx]
        is_rows = np.concatenate([blocks[b] for b in is_idx])
        oos_rows = np.concatenate([blocks[b] for b in oos_idx])

        is_sr = _sr(M[is_rows])
        oos_sr = _sr(M[oos_rows])

        best = int(np.argmax(is_sr))
        # rank of the IS-best among OOS Sharpes: 1 = worst, N = best
        rank = float(stats.rankdata(oos_sr)[best])
        w = rank / (N + 1)
        logits.append(np.log(w / (1.0 - w)))

    logits = np.asarray(logits)
    return {
        "pbo": round(float((logits < 0).mean()), 4),
        "n_combinations": comb(n_splits, n_splits // 2),
        "median_logit": round(float(np.median(logits)), 4),
        "n_trials": N,
    }


class PurgedKFold:
    """K-fold cross-validation with purging and an embargo.

    Standard K-fold leaks when labels overlap in time: a training observation
    whose label is realised over ``[t, t + horizon]`` shares information with a
    test observation at ``t + 1``. This splitter drops any training index within
    ``horizon`` before the test fold (purge) and within ``embargo`` after it.

    ``horizon`` and ``embargo`` are in observations. Yields ``(train_idx,
    test_idx)`` arrays, sklearn-style.
    """

    def __init__(self, n_splits: int = 5, horizon: int = 5, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if horizon < 0 or embargo < 0:
            raise ValueError("horizon and embargo must be non-negative")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo

    def split(self, X):
        n = len(X)
        indices = np.arange(n)
        fold_bounds = np.linspace(0, n, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            start, stop = fold_bounds[i], fold_bounds[i + 1]
            test_idx = indices[start:stop]

            lo = max(0, start - self.horizon)
            hi = min(n, stop + self.embargo)
            blocked = np.zeros(n, dtype=bool)
            blocked[lo:hi] = True
            train_idx = indices[~blocked]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
