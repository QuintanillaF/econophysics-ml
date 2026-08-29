"""Equity factor model — where the return and the risk actually come from.

A portfolio's return is rarely skill. Most of it is exposure: to the market, to
small caps, to cheap stocks, to recent winners, to quality, to low volatility.
These exposures are priced, spanned by cheap index products, and a risk desk
decomposes every equity book into them before it looks at anything else. What is
left after the exposures — the intercept — is *alpha*, and the question is
whether it survives.

Two factor sets:

- **Fama-French** — the academic factors (`Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`,
  plus momentum), downloaded from Ken French's data library. Long-short,
  dollar-neutral, going back to 1963.

- **Style ETFs** — a practitioner stand-in built from tradable products
  (`VLUE`, `MTUM`, `QUAL`, `USMV`, `IWM` versus `SPY`). Works for any recent
  window and needs only price data.

``fit_factor_model`` regresses the portfolio's excess return on the factors and
reports the betas (with t-stats), the annualised alpha, the R², and two
decompositions: how much of the *variance* each factor explains (with the
idiosyncratic remainder) and how much of the *return* each exposure contributed.

Reference: Fama & French (1993, 2015); Carhart (1997) for momentum.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["FactorModel", "fit_factor_model", "load_fama_french", "style_factors"]

_FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
_FF_5 = f"{_FF_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_FF_3 = f"{_FF_BASE}/F-F_Research_Data_Factors_daily_CSV.zip"
_FF_MOM = f"{_FF_BASE}/F-F_Momentum_Factor_daily_CSV.zip"

STYLE_ETFS = {
    "MKT": "SPY",
    "SIZE": "IWM",       # small caps
    "VALUE": "VLUE",
    "MOMENTUM": "MTUM",
    "QUALITY": "QUAL",
    "LOWVOL": "USMV",
}

_FF_CACHE: dict[tuple, pd.DataFrame] = {}


def _fetch_ff_zip(url: str, value_cols: list[str]) -> pd.DataFrame:
    """Download a Ken French daily CSV zip and parse the daily rows."""
    import requests

    resp = requests.get(url, timeout=25)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    text = zf.read(zf.namelist()[0]).decode("latin-1")

    rows: list[tuple] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 1 + len(value_cols):
            continue
        date_tok = parts[0]
        if not (date_tok.isdigit() and len(date_tok) == 8):
            continue  # skip preamble, headers, annual block
        try:
            vals = [float(p) for p in parts[1 : 1 + len(value_cols)]]
        except ValueError:
            continue
        rows.append((pd.Timestamp(date_tok), *vals))

    df = pd.DataFrame(rows, columns=["date", *value_cols]).set_index("date")
    return df / 100.0  # percent -> decimal


def load_fama_french(model: str = "5", momentum: bool = True) -> pd.DataFrame:
    """Daily Fama-French factors as decimal returns, indexed by date.

    ``model="5"`` gives ``Mkt-RF, SMB, HML, RMW, CMA, RF``; ``model="3"`` gives
    ``Mkt-RF, SMB, HML, RF``. With ``momentum=True`` a ``Mom`` column is joined.
    Cached in-process. Needs network.
    """
    key = (model, momentum)
    if key in _FF_CACHE:
        return _FF_CACHE[key]

    if model == "5":
        ff = _fetch_ff_zip(_FF_5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    elif model == "3":
        ff = _fetch_ff_zip(_FF_3, ["Mkt-RF", "SMB", "HML", "RF"])
    else:
        raise ValueError("model must be '3' or '5'")

    if momentum:
        mom = _fetch_ff_zip(_FF_MOM, ["Mom"])
        ff = ff.join(mom, how="inner")

    _FF_CACHE[key] = ff
    return ff


def style_factors(start: str = "2014-01-01", end: str | None = None) -> pd.DataFrame:
    """Practitioner style factors from tradable ETFs.

    ``MKT`` is the market (SPY) return; the rest are *tilts* — the style ETF
    minus SPY — so ``VALUE > 0`` on a day cheap stocks beat the market. No
    risk-free column, so the alpha from a fit against these is a gross alpha.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("style_factors needs yfinance") from exc

    raw = yf.download(
        list(STYLE_ETFS.values()), start=start, end=end,
        auto_adjust=True, progress=False,
    )
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px = pd.DataFrame(px).ffill().dropna()
    rets = np.log(px / px.shift(1)).dropna()

    out = pd.DataFrame(index=rets.index)
    out["MKT"] = rets["SPY"]
    for name, ticker in STYLE_ETFS.items():
        if name == "MKT":
            continue
        out[name] = rets[ticker] - rets["SPY"]
    return out


@dataclass
class FactorModel:
    """Result of regressing a portfolio's excess return on a factor set."""

    betas: pd.Series
    tstats: pd.Series
    alpha_annual: float
    alpha_tstat: float
    r_squared: float
    n_obs: int
    risk_contribution: pd.Series       # fraction of variance, per factor + 'idiosyncratic'
    return_contribution: pd.Series      # annualised, per factor
    residual_vol_annual: float
    source: str

    @property
    def idiosyncratic_share(self) -> float:
        return float(self.risk_contribution.get("idiosyncratic", np.nan))

    def __str__(self) -> str:
        lines = [
            f"Factor model ({self.source}) — {self.n_obs} obs, R² = {self.r_squared:.2f}",
            f"  alpha (annual) {self.alpha_annual:>+7.2%}  (t = {self.alpha_tstat:+.2f})",
            f"  {'factor':<12}{'beta':>8}{'t':>7}{'risk %':>9}{'ret contr.':>12}",
        ]
        for f in self.betas.index:
            lines.append(
                f"  {f:<12}{self.betas[f]:>8.2f}{self.tstats[f]:>7.2f}"
                f"{self.risk_contribution.get(f, np.nan):>9.1%}"
                f"{self.return_contribution.get(f, np.nan):>+12.2%}"
            )
        lines.append(f"  {'idiosyncratic':<12}{'':>8}{'':>7}{self.idiosyncratic_share:>9.1%}")
        return "\n".join(lines)


def fit_factor_model(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame | None = None,
    model: str = "5",
    momentum: bool = True,
    source: str = "fama_french",
) -> FactorModel:
    """Regress ``portfolio_returns`` on a factor set and decompose the result.

    ``source="fama_french"`` downloads the academic factors and regresses the
    portfolio's return in *excess of the risk-free rate*; ``source="style"`` uses
    the ETF tilts and regresses the raw return (gross alpha). Pass ``factors``
    explicitly to skip the download.
    """
    if factors is None:
        if source == "fama_french":
            factors = load_fama_french(model, momentum)
        elif source == "style":
            start = str(portfolio_returns.index[0].date()) if len(portfolio_returns) else "2014-01-01"
            factors = style_factors(start=start)
        else:
            raise ValueError("source must be 'fama_french' or 'style'")

    has_rf = "RF" in factors.columns
    fcols = [c for c in factors.columns if c != "RF"]

    df = pd.concat(
        [portfolio_returns.rename("port"), factors], axis=1, join="inner"
    ).dropna()
    if len(df) < 60:
        raise ValueError(f"only {len(df)} overlapping days with the factor set; need >= 60")

    y = df["port"].to_numpy()
    if has_rf:
        y = y - df["RF"].to_numpy()
    F = df[fcols].to_numpy()
    X = np.column_stack([np.ones(len(df)), F])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    dof = n - k
    sigma2 = float(resid @ resid / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    tstats_all = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)

    alpha_daily = float(coef[0])
    betas = pd.Series(coef[1:], index=fcols, name="beta")
    tstats = pd.Series(tstats_all[1:], index=fcols, name="t")

    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - (resid @ resid) / tss if tss > 0 else 0.0

    # Variance decomposition: Var(r) ≈ b' Σ_f b + Var(ε).
    sigma_f = df[fcols].cov().to_numpy()
    b = betas.to_numpy()
    contrib = b * (sigma_f @ b)                  # sums to systematic variance
    var_resid = float(resid.var(ddof=1))
    var_total = float(contrib.sum() + var_resid)
    risk_contribution = pd.Series(
        {f: c / var_total for f, c in zip(fcols, contrib)}, name="risk_share"
    )
    risk_contribution["idiosyncratic"] = var_resid / var_total

    fmeans = df[fcols].mean().to_numpy()
    return_contribution = pd.Series(
        {f: bb * mm * 252 for f, bb, mm in zip(fcols, b, fmeans)}, name="ret_contr"
    )

    return FactorModel(
        betas=betas,
        tstats=tstats,
        alpha_annual=alpha_daily * 252,
        alpha_tstat=float(tstats_all[0]),
        r_squared=float(r_squared),
        n_obs=int(n),
        risk_contribution=risk_contribution,
        return_contribution=return_contribution,
        residual_vol_annual=float(np.sqrt(var_resid * 252)),
        source=source,
    )
