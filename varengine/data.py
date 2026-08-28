"""
Data loading for the risk engine.

Two sources are supported:

``load_market_data``   pulls real price history from Yahoo Finance. This is what
                       you use in practice; it needs network access and the
                       optional ``yfinance`` dependency.

``simulate_market``    generates synthetic prices from a GARCH(1,1) process with
                       Student-t innovations. This exists for two reasons: the
                       test suite must run offline and deterministically, and a
                       synthetic series has *known* parameters, so you can check
                       whether an estimator recovers what it should.

The synthetic generator is not a toy. Real equity returns show volatility
clustering (calm periods and turbulent periods arrive in runs) and fat tails
(extreme moves are far more common than a Gaussian predicts). GARCH with t
innovations reproduces both, which is precisely what makes normal-assumption
VaR fail in backtesting — the effect this project is built to expose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["GarchParams", "load_market_data", "simulate_market"]


@dataclass(frozen=True)
class GarchParams:
    """GARCH(1,1) parameters: sigma2_t = omega + alpha*e2_{t-1} + beta*sigma2_{t-1}.

    Persistence is ``alpha + beta``; it must stay below 1 for the variance
    process to be stationary. Values near 0.97-0.99 are typical for daily
    equity data and produce the long, slowly-decaying volatility episodes you
    see in real markets.
    """

    omega: float = 2.0e-6
    alpha: float = 0.09
    beta: float = 0.89
    nu: float = 4.5  # Student-t degrees of freedom; lower = fatter tails
    mu: float = 3.0e-4  # daily drift

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0 or self.omega <= 0:
            raise ValueError("omega must be positive; alpha and beta non-negative")
        if self.alpha + self.beta >= 1:
            raise ValueError(
                f"alpha + beta = {self.alpha + self.beta:.3f} >= 1: "
                "the variance process is not stationary"
            )
        if self.nu <= 2:
            raise ValueError("nu must exceed 2 or the variance is undefined")

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        """Unconditional daily volatility implied by the parameters."""
        return float(np.sqrt(self.omega / (1.0 - self.persistence)))


def simulate_market(
    tickers: list[str],
    n_days: int = 1500,
    params: GarchParams | None = None,
    correlation: float = 0.55,
    seed: int = 42,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Simulate correlated price series with volatility clustering and fat tails.

    Each asset follows its own GARCH(1,1) variance process, but the standardised
    innovations are correlated across assets — so the series co-move, and
    volatility spikes tend to arrive together, as they do in a real crisis.

    Returns a DataFrame of prices indexed by business day.
    """
    if n_days < 2:
        raise ValueError("n_days must be at least 2")
    if not -1.0 < correlation < 1.0:
        raise ValueError("correlation must lie strictly between -1 and 1")

    p = params or GarchParams()
    rng = np.random.default_rng(seed)
    k = len(tickers)

    # Equicorrelation matrix, Cholesky factor for inducing cross-sectional
    # dependence in the standardised shocks.
    corr = np.full((k, k), correlation)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    # Student-t shocks scaled to unit variance: Var(t_nu) = nu / (nu - 2).
    raw = rng.standard_t(p.nu, size=(n_days, k)) / np.sqrt(p.nu / (p.nu - 2.0))
    shocks = raw @ chol.T

    returns = np.empty((n_days, k))
    sigma2 = np.full(k, p.long_run_vol**2)
    eps_prev = np.zeros(k)

    for t in range(n_days):
        sigma2 = p.omega + p.alpha * eps_prev**2 + p.beta * sigma2
        eps = np.sqrt(sigma2) * shocks[t]
        returns[t] = p.mu + eps
        eps_prev = eps

    # Slightly different starting prices keep the portfolio weights meaningful.
    p0 = 100.0 * (1.0 + 0.1 * np.arange(k))
    prices = p0 * np.exp(np.cumsum(returns, axis=0))

    idx = pd.bdate_range(start=start, periods=n_days, name="date")
    return pd.DataFrame(prices, index=idx, columns=tickers)


def load_market_data(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Requires ``pip install yfinance`` and network access. For Argentine equities
    use the ``.BA`` suffix, e.g. ``["GGAL.BA", "YPFD.BA", "PAMP.BA"]``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "load_market_data needs yfinance. Install it with "
            "`pip install 'var-risk-engine[data]'`, or use simulate_market() "
            "to work offline."
        ) from exc

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = pd.DataFrame(prices)
    if len(tickers) == 1:
        prices.columns = tickers

    prices = prices.dropna(how="all").ffill().dropna()
    if prices.empty:
        raise ValueError(f"No price data returned for {tickers}")

    prices.index.name = "date"
    return prices
