"""Stress testing: what the portfolio loses in a scenario, not a distribution.

VaR and Expected Shortfall answer "how bad is a normal-ish bad day?". They say
nothing about a specific event — a rate shock, a credit freeze, a 2008 — because
those live in the part of the distribution where there is no data to fit. Stress
testing fills that gap by asking a different question: *given this exact set of
market moves, what happens to the book?*

Three flavours, all of which a supervisor expects to see:

- **Historical replay** — take the actual market moves from a past crisis window
  and apply them to today's positions. No model, no distribution; the only
  assumption is that the portfolio's exposures are what they are now.

- **Hypothetical / parametric** — a hand-specified set of risk-factor moves
  ("equities -30%, credit -15%, oil -20%, gold +8%"). Flexible, and the only way
  to stress a scenario that has never happened.

- **Reverse stress** — fix the loss first ("what would cost us 10% of the book?")
  and solve for the market moves that produce it. Often more revealing than
  forward stress because it surfaces the scenario you were not worrying about.

The portfolio is mapped to a small set of tradable risk factors by regression
(``factor_betas``). This assumes exposures are linear and stable — fine for a
first-order stress on a cash equity/crypto book, not for options. When every
asset in the book has price history covering the scenario window, historical
replay uses the real asset returns instead and the linearity assumption drops
away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import Portfolio

__all__ = [
    "HISTORICAL_SCENARIOS",
    "SHOCK_LIBRARY",
    "StressResult",
    "factor_betas",
    "historical_stress",
    "parametric_stress",
    "reverse_stress",
    "run_stress_suite",
]

# Tradable proxies for the risk factors a cross-asset book is exposed to.
FACTOR_TICKERS: dict[str, str] = {
    "equity": "SPY",
    "rates": "IEF",          # 7-10y Treasuries: price up = yields down
    "credit": "HYG",         # high-yield credit
    "usd": "DX-Y.NYB",       # dollar index
    "gold": "GLD",
    "oil": "CL=F",
    "crypto": "BTC-USD",
}

# Crisis windows for historical replay (inclusive start, inclusive end).
HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "GFC 2008 (Lehman)":        ("2008-09-02", "2008-11-20"),
    "China / August 2015":      ("2015-08-10", "2015-08-25"),
    "Volmageddon Feb 2018":     ("2018-01-26", "2018-02-08"),
    "Q4 2018 selloff":          ("2018-10-01", "2018-12-24"),
    "COVID crash Mar 2020":     ("2020-02-19", "2020-03-23"),
    "2022 rate shock":          ("2022-01-03", "2022-10-14"),
    # Crypto-specific dislocations (replayed through BTC-USD, or asset-level).
    "Terra / UST May 2022":     ("2022-05-05", "2022-05-16"),
    "3AC deleveraging Jun 2022": ("2022-06-10", "2022-06-19"),
    "FTX collapse Nov 2022":    ("2022-11-05", "2022-11-14"),
}

# Hypothetical scenarios as risk-factor total returns over the horizon.
SHOCK_LIBRARY: dict[str, dict[str, float]] = {
    "equity_crash":       {"equity": -0.30, "credit": -0.15, "oil": -0.20, "gold": 0.08, "crypto": -0.40},
    "rates_shock":        {"rates": -0.08, "equity": -0.10, "credit": -0.06, "usd": 0.04},
    "risk_off":           {"equity": -0.15, "credit": -0.08, "usd": 0.05, "gold": 0.05, "crypto": -0.20},
    "stagflation":        {"equity": -0.12, "rates": -0.05, "oil": 0.25, "gold": 0.10},
    "crypto_winter":      {"crypto": -0.55, "equity": -0.08},
    "stablecoin_depeg":   {"crypto": -0.45, "equity": -0.05},
    "exchange_insolvency": {"crypto": -0.30, "credit": -0.05, "equity": -0.04},
}

_FACTOR_CACHE: dict[tuple, pd.DataFrame] = {}


def _load_factor_prices(start: str, end: str | None = None) -> pd.DataFrame:
    """Download the risk-factor proxy prices (cached in-process)."""
    key = (start, end)
    if key in _FACTOR_CACHE:
        return _FACTOR_CACHE[key]
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "stress testing needs yfinance for the risk-factor proxies; "
            "install it or pass factor_returns explicitly"
        ) from exc

    raw = yf.download(
        list(FACTOR_TICKERS.values()), start=start, end=end,
        auto_adjust=True, progress=False,
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = pd.DataFrame(prices).ffill()
    inv = {v: k for k, v in FACTOR_TICKERS.items()}
    prices = prices.rename(columns=inv)
    _FACTOR_CACHE[key] = prices
    return prices


def _factor_returns(start: str = "2007-01-01", end: str | None = None) -> pd.DataFrame:
    prices = _load_factor_prices(start, end)
    # WTI (CL=F) printed a negative settle on 2020-04-20; simple returns keep the
    # panel finite where a log return would not.
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    return rets.clip(-0.9, 9.0).dropna(how="all")


def _as_returns(portfolio: Portfolio | pd.Series) -> pd.Series:
    if isinstance(portfolio, Portfolio):
        return portfolio.returns
    return pd.Series(portfolio).dropna()


@dataclass(frozen=True)
class StressResult:
    """Outcome of one stress scenario."""

    scenario: str
    kind: str                       # "historical" | "hypothetical" | "reverse"
    pnl_pct: float                  # portfolio return under the scenario
    pnl_amount: float | None
    worst_day_pct: float | None     # only for historical replay
    max_drawdown_pct: float | None  # only for historical replay
    factor_moves: dict[str, float]
    detail: str = ""

    def __str__(self) -> str:
        amt = f"  ({self.pnl_amount:,.0f})" if self.pnl_amount is not None else ""
        return f"{self.scenario:<26} {self.pnl_pct:>8.2%}{amt}   {self.detail}"


def factor_betas(
    portfolio: Portfolio | pd.Series,
    factor_returns: pd.DataFrame | None = None,
    lookback: int = 500,
) -> pd.Series:
    """OLS betas of the portfolio return on the risk-factor returns.

    Uses the most recent ``lookback`` overlapping days. The intercept is fitted
    and discarded (it is the idiosyncratic drift, not a factor exposure). Factors
    with no data over the window (e.g. ``crypto`` for a pre-2015 sample) are
    dropped.
    """
    r_port = _as_returns(portfolio)
    if factor_returns is None:
        start = (r_port.index[0] if hasattr(r_port.index[0], "date") else None)
        factor_returns = _factor_returns(
            start=str(start.date()) if start is not None else "2015-01-01"
        )

    df = pd.concat([r_port.rename("port"), factor_returns], axis=1, join="inner").dropna()
    df = df.tail(lookback)
    if len(df) < 60:
        raise ValueError(f"only {len(df)} overlapping days with the factor set; need >= 60")

    y = df["port"].to_numpy()
    factors = [c for c in factor_returns.columns if c in df.columns]
    X = np.column_stack([np.ones(len(df)), df[factors].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(coef[1:], index=factors, name="beta")


def _pnl_from_moves(betas: pd.Series, moves: dict[str, float]) -> float:
    return float(sum(betas.get(f, 0.0) * m for f, m in moves.items()))


def historical_stress(
    portfolio: Portfolio,
    scenario: str,
    factor_returns: pd.DataFrame | None = None,
) -> StressResult:
    """Replay a crisis window against the current book.

    Tries an exact asset-level replay first: if every ticker in the portfolio has
    price history spanning the window, apply the real asset returns and report
    the full P&L path (worst day, max drawdown). Otherwise fall back to a
    factor-based replay using the portfolio's betas and the factors' realised
    moves over the window.
    """
    if scenario not in HISTORICAL_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; see HISTORICAL_SCENARIOS")
    start, end = HISTORICAL_SCENARIOS[scenario]

    # --- attempt asset-level replay -------------------------------------------
    asset_path = _asset_level_replay(portfolio, start, end)
    if asset_path is not None:
        cum = (1.0 + asset_path).prod() - 1.0
        equity = (1.0 + asset_path).cumprod()
        dd = float((equity / equity.cummax() - 1.0).min())
        return StressResult(
            scenario=scenario,
            kind="historical",
            pnl_pct=float(cum),
            pnl_amount=float(cum * portfolio.value),
            worst_day_pct=float(asset_path.min()),
            max_drawdown_pct=dd,
            factor_moves={},
            detail=f"asset-level replay, {len(asset_path)} days ({start}..{end})",
        )

    # --- factor-based replay -------------------------------------------------
    fr = factor_returns if factor_returns is not None else _factor_returns(start="2007-01-01")
    betas = factor_betas(portfolio, fr)
    window = fr.loc[start:end]
    if window.empty:
        raise ValueError(f"no factor data in {start}..{end}")
    moves = {f: float((1.0 + window[f]).prod() - 1.0) for f in betas.index if f in window}
    pnl = _pnl_from_moves(betas, moves)
    return StressResult(
        scenario=scenario,
        kind="historical",
        pnl_pct=pnl,
        pnl_amount=pnl * portfolio.value,
        worst_day_pct=None,
        max_drawdown_pct=None,
        factor_moves=moves,
        detail=f"factor replay via betas ({start}..{end})",
    )


def _asset_level_replay(
    portfolio: Portfolio, start: str, end: str
) -> pd.Series | None:
    """Portfolio return path over the window from real asset prices, or None."""
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover
        return None
    tickers = list(portfolio.prices.columns)
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    except Exception:  # pragma: no cover - network
        return None
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = pd.DataFrame(prices)
    if len(tickers) == 1:
        prices.columns = tickers
    prices = prices.dropna(how="all").ffill()
    if prices.empty or prices.isna().any().any() or len(prices) < 5:
        return None
    if set(prices.columns) != set(tickers):
        return None
    if (prices[tickers] <= 0).any().any():
        return None
    rets = np.log(prices[tickers] / prices[tickers].shift(1)).dropna()
    return rets @ portfolio.w


def parametric_stress(
    portfolio: Portfolio,
    shock: str | dict[str, float],
    factor_returns: pd.DataFrame | None = None,
) -> StressResult:
    """Apply a hypothetical set of risk-factor moves to the book."""
    if isinstance(shock, str):
        if shock not in SHOCK_LIBRARY:
            raise ValueError(f"unknown shock {shock!r}; see SHOCK_LIBRARY")
        name, moves = shock, SHOCK_LIBRARY[shock]
    else:
        name, moves = "custom", dict(shock)

    betas = factor_betas(portfolio, factor_returns)
    pnl = _pnl_from_moves(betas, moves)
    return StressResult(
        scenario=name,
        kind="hypothetical",
        pnl_pct=pnl,
        pnl_amount=pnl * portfolio.value,
        worst_day_pct=None,
        max_drawdown_pct=None,
        factor_moves=dict(moves),
        detail=", ".join(f"{k} {v:+.0%}" for k, v in moves.items()),
    )


def reverse_stress(
    portfolio: Portfolio,
    target_loss_pct: float = 0.10,
    direction: str = "risk_off",
    factor_returns: pd.DataFrame | None = None,
) -> StressResult:
    """Scale a shock direction until the book loses ``target_loss_pct``.

    ``direction`` is a key of :data:`SHOCK_LIBRARY` (used as a unit direction in
    factor space). Returns the scaled factor moves that produce the target loss —
    i.e. "for us to lose 10%, this is what the market has to do".
    """
    if not 0 < target_loss_pct < 1:
        raise ValueError("target_loss_pct must lie in (0, 1)")
    if direction not in SHOCK_LIBRARY:
        raise ValueError(f"unknown direction {direction!r}; see SHOCK_LIBRARY")

    base = SHOCK_LIBRARY[direction]
    betas = factor_betas(portfolio, factor_returns)
    unit_pnl = _pnl_from_moves(betas, base)
    if unit_pnl >= 0:
        raise ValueError(
            f"the '{direction}' direction does not lose money for this book "
            f"(unit P&L {unit_pnl:+.2%}); pick another direction"
        )

    scale = -target_loss_pct / unit_pnl
    scaled = {k: v * scale for k, v in base.items()}
    return StressResult(
        scenario=f"reverse: -{target_loss_pct:.0%} via {direction}",
        kind="reverse",
        pnl_pct=-target_loss_pct,
        pnl_amount=-target_loss_pct * portfolio.value,
        worst_day_pct=None,
        max_drawdown_pct=None,
        factor_moves=scaled,
        detail="required moves: " + ", ".join(f"{k} {v:+.1%}" for k, v in scaled.items()),
    )


def run_stress_suite(
    portfolio: Portfolio,
    factor_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Every historical and hypothetical scenario, tabulated by P&L."""
    rows: list[dict] = []
    fr = factor_returns
    if fr is None:
        try:
            fr = _factor_returns(start="2007-01-01")
        except ImportError:
            fr = None

    for name in HISTORICAL_SCENARIOS:
        try:
            res = historical_stress(portfolio, name, fr)
            rows.append(_row(res))
        except (ValueError, KeyError):
            continue
    for name in SHOCK_LIBRARY:
        try:
            res = parametric_stress(portfolio, name, fr)
            rows.append(_row(res))
        except (ValueError, KeyError):
            continue

    return pd.DataFrame(rows).set_index("scenario").sort_values("pnl_pct")


def _row(res: StressResult) -> dict:
    return {
        "scenario": res.scenario,
        "kind": res.kind,
        "pnl_pct": res.pnl_pct,
        "pnl_amount": res.pnl_amount,
        "worst_day_pct": res.worst_day_pct,
        "max_drawdown_pct": res.max_drawdown_pct,
        "detail": res.detail,
    }
