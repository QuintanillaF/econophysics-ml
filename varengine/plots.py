"""Plotting helpers for the risk report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["plot_backtest", "plot_method_comparison", "plot_return_distribution", "risk_dashboard"]

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


def risk_dashboard(bt, returns, var_levels, comparison, path="risk_report.png"):
    """Assemble the three panels into a single report image."""
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.22)

    plot_backtest(bt, ax=fig.add_subplot(gs[0, :]))
    plot_return_distribution(returns, var_levels, ax=fig.add_subplot(gs[1, 0]))
    plot_method_comparison(comparison, ax=fig.add_subplot(gs[1, 1]))

    fig.suptitle("Portfolio market-risk report", fontsize=14, weight="bold",
                 color=INK, x=0.007, ha="left", y=0.985)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
