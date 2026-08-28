# var-risk-engine

Value at Risk estimation and regulatory backtesting for equity portfolios.

Three VaR methodologies, Expected Shortfall, and the hypothesis tests
supervisors use to decide whether a risk model may be relied upon — implemented
from first principles, tested against closed-form results, and demonstrated on a
portfolio with realistic fat tails.

![Risk report](../docs/risk_report.png)

---

## The finding

The headline result is not a number, it is a disagreement.

Run against a portfolio whose returns exhibit volatility clustering and excess
kurtosis of 4.9 — properties every real equity series has — the estimators
diverge by 15%:

| Method | 1-day 99% VaR | Expected Shortfall |
|---|---|---|
| Historical simulation | 1.905% | 2.690% |
| Parametric — Student-t | 1.865% | 2.515% |
| Monte Carlo — Student-t | 1.774% | 2.218% |
| Monte Carlo — normal | 1.649% | 1.880% |
| Parametric — normal | 1.652% | 1.894% |

The normal-assumption methods sit consistently at the bottom. That gap is not a
rounding difference — it is the model failing to price the tail it was built to
measure, and it is what a walk-forward backtest exposes:

| Model | Exceptions / 1549 days | Kupiec test | Basel zone |
|---|---|---|---|
| Historical simulation | 23 (1.48%) | not rejected | 🟢 GREEN |
| Parametric — normal | 25 (1.61%) | **REJECTED** (p = 0.026) | 🟡 YELLOW |
| Parametric — Student-t | 24 (1.55%) | **REJECTED** (p = 0.044) | 🟢 GREEN |

A yellow zone raises the regulatory capital multiplier from 3.00 to 3.40. The
Gaussian assumption is not an academic simplification; it has a price.

---

> Part of the [`econophysics-ml`](../README.md) repository. `varengine/` is the one
> importable package; run everything from the repo root.

## Install

```bash
pip install -r requirements.txt        # from the repo root
pip install -e ".[dev]"                 # optional: editable install + pytest/ruff
```

`yfinance` (already in `requirements.txt`) is only needed for `--real`.

## Run

```bash
python var_analysis.py            # synthetic GARCH data, no network needed
python var_analysis.py --real     # Argentine equities via Yahoo Finance
```

```python
from varengine import Portfolio, simulate_market, compare_methods, run_full_backtest

prices = simulate_market(["A", "B", "C"], n_days=1500)
book = Portfolio(prices, value=1_000_000)

print(compare_methods(book.asset_returns, book.w, confidence=0.99))

bt = run_full_backtest(book.returns, window=250, confidence=0.99)
print(bt["kupiec"])          # unconditional coverage
print(bt["christoffersen"])  # independence of exceptions
print(bt["basel"]["zone"])   # supervisory traffic light
```

---

## What is implemented

**Estimators** — historical simulation, parametric variance-covariance under
normal and Student-t, and Monte Carlo with a Gaussian copula. Expected Shortfall
throughout, since Basel III / FRTB replaced VaR with ES at 97.5% as the
regulatory measure: VaR gives you the threshold, ES gives you the average loss
past it, and unlike VaR, ES is sub-additive.

**Backtesting** — walk-forward with a rolling window, so each forecast uses only
information available on the day it was made. Then:

- **Kupiec proportion-of-failures** — is the *number* of exceptions right?
- **Christoffersen independence** — are they *spread out*, or clustered?
- **Conditional coverage** — the joint test, and the one to quote.
- **Basel traffic light** — the supervisory rule that turns an exception count
  into a capital penalty.

The independence test matters more than it first appears. A model can produce
exactly the expected number of breaches and still be unusable if they all arrive
in the same week — the signature of a static volatility estimate meeting returns
that cluster. The test suite includes a case constructed to demonstrate exactly
this: two exception sequences with identical counts, where Kupiec cannot tell
them apart and Christoffersen rejects one.

**Portfolio analytics** — analytic volatility from the covariance matrix, and
marginal risk contributions answering the question that always follows the
headline number: where is the risk actually coming from?

---

## Known limitations

Stated plainly, because a risk model whose limitations are not documented is
worse than no model.

**Square-root-of-time scaling** assumes returns are independent and identically
distributed. They are not. A 10-day VaR built this way understates risk in
turbulent periods and overstates it in calm ones. Basel permits it, which is the
only reason it is standard.

**Historical simulation cannot produce a loss it has never seen.** It also
weights a crash from two years ago identically to yesterday. Exponentially
weighted historical simulation would address the second problem; extreme value
theory would address the first.

**No volatility model.** All estimators here assume the return distribution is
stable over the estimation window. It is not — that is the whole reason
volatility clustering exists. A GARCH-based conditional VaR is the natural
extension and would likely resolve the coverage failures above.

**Linear instruments only.** Options and other convex payoffs need full
revaluation or a delta-gamma approximation. Monte Carlo is the right foundation
for that; the machinery is here, the instrument layer is not.

**Sample covariance is noisy** with many assets relative to observations.
Ledoit-Wolf shrinkage would help.

**Student-t degrees of freedom are unstable** when fitted by maximum likelihood
on short windows — the fit can return values below 2, where the variance does
not exist. The estimator clips at 2.05 and records the clip in its output rather
than failing silently or crashing a backtest.

---

## Tests

```bash
python -m pytest              # 62 tests
python -m pytest --cov=varengine
```

Tests assert mathematical properties, not snapshots of previous runs. Where a
closed form exists it is checked directly: normal VaR against the analytic
quantile, empirical VaR against the true quantile on 200,000 draws, Monte Carlo
convergence to the parametric answer under matching assumptions, and the
square-root-of-time identity. Structural invariants — ES never below VaR, VaR
monotone in confidence, risk contributions summing to one, diversification
reducing volatility — are checked because a violation means something is wrong
regardless of what the numbers look like.

---

## Layout

```
varengine/
    data.py        market data loading; GARCH(1,1) simulator with t innovations
    portfolio.py   weights, returns, covariance, risk decomposition
    var.py         historical / parametric / Monte Carlo VaR and ES
    backtest.py    Kupiec, Christoffersen, conditional coverage, Basel zones
    plots.py       report figures
tests/test_varengine.py   62 tests
var_analysis.py           end-to-end analysis (repo root)
```

---

## References

Kupiec (1995), *Techniques for verifying the accuracy of risk measurement
models*. Christoffersen (1998), *Evaluating interval forecasts*. Basel Committee
(1996), *Supervisory framework for the use of backtesting*. Basel Committee
(2019), *Minimum capital requirements for market risk* (FRTB).

MIT licensed.
