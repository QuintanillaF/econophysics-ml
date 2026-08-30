# var-risk-engine

Value at Risk estimation and regulatory backtesting for equity portfolios.

Five VaR methodologies (historical, parametric normal/Student-t, Monte Carlo,
and volatility-filtered historical simulation), Expected Shortfall, extreme value
theory for the far tail, stress testing, a Fama-French factor model, and the
backtest-overfitting diagnostics (PSR, deflated Sharpe, PBO) — plus the
hypothesis tests supervisors use to decide whether a risk model may be relied
upon, for both VaR and ES. Implemented from first principles, tested against
closed-form results, and demonstrated on a portfolio with realistic fat tails.

![Risk report](../docs/risk_report.png)

---

## The finding

The headline is not a number, it is a disagreement — and then a second, subtler
one.

Run against a portfolio whose returns exhibit volatility clustering and excess
kurtosis of 4.9 — properties every real equity series has — the unconditional
estimators diverge by 15%:

| Method | 1-day 99% VaR | Expected Shortfall |
|---|---|---|
| Historical simulation | 1.905% | 2.690% |
| Parametric — Student-t | 1.865% | 2.515% |
| Monte Carlo — Student-t | 1.774% | 2.218% |
| Monte Carlo — normal | 1.649% | 1.880% |
| Parametric — normal | 1.652% | 1.894% |
| Filtered historical (EWMA) | 1.060% | 1.360% |

The normal-assumption methods sit consistently at the bottom — the model failing
to price the tail it was built to measure. The walk-forward backtest over 1549
days exposes it, and exposes something else:

| Model | Exceptions | Kupiec | ES test (Acerbi-Székely) | Basel zone |
|---|---|---|---|---|
| Historical simulation | 23 (1.48%) | not rejected (p = 0.074) | **REJECTED** (Z2 = −0.74, p = 0.018) | GREEN |
| Parametric — normal | 25 (1.61%) | **REJECTED** (p = 0.026) | **REJECTED** (Z2 = −0.93, p = 0.008) | YELLOW |
| Parametric — Student-t | 24 (1.55%) | **REJECTED** (p = 0.044) | **REJECTED** (Z2 = −0.54, p = 0.040) | GREEN |
| **Filtered historical (EWMA)** | **17 (1.10%)** | **not rejected** (p = 0.70) | **not rejected** (Z2 = −0.22, p = 0.23) | GREEN |

Two findings:

1. The Gaussian assumption has a price. Parametric-normal VaR is rejected by
   Kupiec and lands in the yellow zone — the capital multiplier goes from 3.00 to
   3.40.

2. Passing Kupiec is not enough. Historical simulation clears the exception count
   and stays green, but **fails the ES backtest** — the measure FRTB actually
   uses. When a breach happens, the loss is worse than its ES predicted.

Conditioning on volatility fixes both: EWMA-filtered historical simulation nails
the exception rate at 1.10%, passes every test, and stays green.

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
normal and Student-t, Monte Carlo with a Gaussian copula, and filtered historical
simulation on an EWMA or GARCH(1,1) volatility. Expected Shortfall throughout,
since Basel III / FRTB replaced VaR with ES at 97.5% as the regulatory measure.

**Conditional volatility** (`volatility.py`) — EWMA (RiskMetrics, λ = 0.94) and a
GARCH(1,1) fitted by maximum likelihood (normal or t innovations), with
mean-reverting multi-day variance forecasts.

**Extreme value theory** (`evt.py`) — peaks-over-threshold: a Generalised Pareto
fit to the exceedances gives VaR and ES at 99.5% / 99.9%, where the empirical tail
runs out of data. The shape parameter `xi` is reported as the tail index `1/xi`.

**Portfolio analytics** (`portfolio.py`) — analytic volatility, Ledoit-Wolf
shrinkage covariance (`shrink=True`), and component / marginal / incremental VaR
that decomposes the headline number by position.

**Backtesting** (`backtest.py`) — walk-forward, then:

- **Kupiec proportion-of-failures** — is the *number* of exceptions right?
- **Christoffersen independence** — are they *spread out*, or clustered? (A model
  can have the exact expected count and still be unusable if the breaches arrive
  in one week — the test suite has a constructed case where Kupiec cannot tell two
  such sequences apart and Christoffersen rejects one.)
- **Conditional coverage** — the joint VaR test.
- **Acerbi-Székely ES test (Z2)** — is the *Expected Shortfall* calibrated? A
  model can pass Kupiec and fail this.
- **Basel traffic light** — the supervisory rule that turns an exception count
  into a capital penalty.

**Stress testing** (`stress.py`) — historical replay of crisis windows (GFC,
COVID, 2022 rate shock, and the crypto blow-ups: Terra/UST, 3AC, FTX),
hypothetical factor shocks, and reverse stress (fix the loss, solve for the
moves). The book is mapped to tradable risk factors by regression; asset-level
replay is used when the history is available.

**Factor model** (`factors.py`) — regress the portfolio's excess return on
Fama-French 5 + momentum (Ken French data) or style-ETF factors. Reports betas
with t-stats, annualised alpha and its t-stat, R², a variance decomposition per
factor with the idiosyncratic remainder, and a return attribution.

**Backtest diagnostics** (`diagnostics.py`) — Probabilistic and Deflated Sharpe
ratios (Bailey & López de Prado), Probability of Backtest Overfitting via
combinatorial symmetric cross-validation, and a `PurgedKFold` splitter with
purge and embargo for labelled financial data.

---

## Known limitations

Stated plainly, because a risk model whose limitations are not documented is
worse than no model.

**Square-root-of-time scaling** assumes returns are i.i.d. They are not. A 10-day
VaR built this way understates risk in turbulent periods. Basel permits it, which
is the only reason it is standard. The GARCH forecast mean-reverts and is a little
better; EWMA still inherits the flaw.

**GARCH-filtered rolling backtests are slow** — one MLE per window. The rolling
backtest defaults to EWMA-FHS (no fit); GARCH-FHS is a point estimate.

**Linear instruments only.** Options and other convex payoffs need full
revaluation or a delta-gamma approximation. The Monte Carlo and stress machinery
is the right foundation; the instrument layer is not built.

**Stress betas assume linear, stable factor exposure** — fine for a first-order
stress on a cash book, not for a book with optionality.

**The credit-spread proxy** in the macro layer is ETF-based (HYG vs IEF), not a
true option-adjusted spread (that needs FRED).

**Student-t degrees of freedom are unstable** on short windows — the fit can
return values below 2. The estimator clips at 2.05 and records the clip.

---

## Tests

```bash
python -m pytest              # 86 tests
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
    portfolio.py   weights, returns, covariance (+ Ledoit-Wolf), component VaR
    volatility.py  EWMA and GARCH(1,1) conditional-volatility models
    var.py         historical / parametric / Monte Carlo / filtered-historical VaR and ES
    evt.py         extreme value theory (peaks-over-threshold, GPD)
    backtest.py    Kupiec, Christoffersen, conditional coverage, ES test, Basel zones
    stress.py      historical-replay, hypothetical and reverse stress testing
    factors.py     Fama-French / style-ETF factor model
    diagnostics.py PSR, deflated Sharpe, PBO, purged cross-validation
    plots.py       report figures
tests/test_varengine.py   97 tests
var_analysis.py           end-to-end analysis (repo root)
```

---

## References

Kupiec (1995), *Techniques for verifying the accuracy of risk measurement
models*. Christoffersen (1998), *Evaluating interval forecasts*. Acerbi & Székely
(2014), *Back-testing Expected Shortfall*. Barone-Adesi, Giannopoulos & Vosper
(1999), *VaR without correlations …* (filtered historical simulation). McNeil &
Frey (2000), *Estimation of tail-related risk measures … an extreme value
approach*. Ledoit & Wolf (2004), *Honey, I shrunk the sample covariance matrix*.
J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*. Basel Committee
(1996), *Supervisory framework for the use of backtesting*; (2019), *Minimum
capital requirements for market risk* (FRTB).

MIT licensed.
