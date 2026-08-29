"""Plotting helpers for the risk report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "plot_backtest",
    "plot_method_comparison",
    "plot_return_distribution",
    "plot_stress",
    "risk_dashboard",
]

INK = "#131a1f"
PLOT = "#1b4a5a"
BREACH = "#a8322a"
CALM = "#5a6b70"
GRID = "#d5dbd9"


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_backtest(bt: pd.DataFrame, ax=None, title: str = "Walk-forward backtest"):
    """Realised returns against the VaR forecast, with breaches highlighted."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))

    ax.plot(bt.index, bt["realised_return"], lw=0.7, color=CALM,
            alpha=0.8, label="realised return")
    ax.plot(bt.index, -bt["var_forecast"], lw=1.4, color=PLOT,
            label="VaR threshold (99%)")

    exc = bt[bt["exception"]]
    ax.scatter(exc.index, exc["realised_return"], s=26, color=BREACH,
               zorder=5, label=f"exceptions ({len(exc)})", edgecolors="white", linewidths=0.5)

    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_title(title, fontsize=11, color=INK, loc="left", weight="bold")
    ax.set_ylabel("daily return")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
    _style(ax)
    return ax


def plot_return_distribution(returns: pd.Series, var_levels: dict[str, float], ax=None):
    """Return histogram with a fitted normal and each method's VaR threshold."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    r = returns.dropna()
    ax.hist(r, bins=90, density=True, color=PLOT, alpha=0.30, edgecolor="none",
            label="realised")

    xs = np.linspace(r.min(), r.max(), 500)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=INK, lw=1.3,
            ls="--", label="fitted normal")

    palette = [BREACH, "#b8791a", "#2f7d5d", "#6a4c93"]
    for (name, v), colour in zip(var_levels.items(), palette):
        ax.axvline(-v, color=colour, lw=1.5, alpha=0.9, label=f"{name}: {v:.2%}")

    ax.set_xlim(r.quantile(0.001), r.quantile(0.999))
    ax.set_title("Return distribution and VaR thresholds", fontsize=11,
                 color=INK, loc="left", weight="bold")
    ax.set_xlabel("daily return")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return ax


def plot_method_comparison(df: pd.DataFrame, ax=None):
    """Horizontal bars comparing VaR and ES across estimators."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    y = np.arange(len(df))
    ax.barh(y - 0.19, df["VaR"], height=0.36, color=PLOT, label="VaR")
    ax.barh(y + 0.19, df["ES"], height=0.36, color=BREACH, alpha=0.85, label="ES")

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("loss as fraction of portfolio value")
    ax.set_title("VaR and Expected Shortfall by method (99%)", fontsize=11,
                 color=INK, loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return ax


def plot_stress(stress_df: pd.DataFrame, ax=None):
    """Horizontal bars of portfolio P&L under each stress scenario."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    df = stress_df.sort_values("pnl_pct")
    y = np.arange(len(df))
    colours = [BREACH if v < 0 else "#2f7d5d" for v in df["pnl_pct"]]
    ax.barh(y, df["pnl_pct"] * 100, color=colours, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("portfolio P&L (%)")
    ax.set_title("Stress scenarios", fontsize=11, color=INK, loc="left", weight="bold")
    _style(ax)
    return ax


def risk_dashboard(bt, returns, var_levels, comparison, path="risk_report.png",
                   stress_df=None):
    """Assemble the panels into a single report image.

    Three panels by default (backtest, return distribution, method comparison);
    a fourth stress panel is added when ``stress_df`` is supplied.
    """
    rows = 3 if stress_df is not None else 2
    fig = plt.figure(figsize=(13, 4.5 * rows))
    gs = fig.add_gridspec(rows, 2, hspace=0.34, wspace=0.22)

    plot_backtest(bt, ax=fig.add_subplot(gs[0, :]))
    plot_return_distribution(returns, var_levels, ax=fig.add_subplot(gs[1, 0]))
    plot_method_comparison(comparison, ax=fig.add_subplot(gs[1, 1]))
    if stress_df is not None:
        plot_stress(stress_df, ax=fig.add_subplot(gs[2, :]))

    fig.suptitle("Portfolio market-risk report", fontsize=14, weight="bold",
                 color=INK, x=0.007, ha="left", y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
