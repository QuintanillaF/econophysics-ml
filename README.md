# econophysics-ml
Machine learning and econophysics models applied to financial time series
[README.md](https://github.com/user-attachments/files/27326310/README.md)
# Econophysics + ML Trading System

A Python-based quantitative trading framework that combines **econophysics** concepts with **machine learning** to analyze and model financial markets. Developed as part of a physics undergraduate research background.

> **Disclaimer:** This project is for educational and research purposes only. Backtested results do not guarantee future performance.

---

##What is Econophysics?

Econophysics applies concepts from statistical physics to model financial systems. Instead of assuming markets are efficient and returns are Gaussian, it uses tools like:

- **Hurst exponent** — measures long-range memory and persistence in time series
- **Tsallis entropy / q-statistics** — models heavy-tailed distributions in returns
- **Fractal dimension** — quantifies the complexity of price trajectories
- **LPPLS (Log-Periodic Power Law Singularity)** — Sornette's bubble detection model
- **Random Matrix Theory (RMT)** — separates real correlations from noise in portfolios

---

##Project Structure

```
econophysics-ml/
│
├── bot_trading_ml_econofisica.py       # Short-term ML trading bot
└── econofisica_mediano_largo_plazo.py  # Medium/long-term econophysics analysis
```

---

##Module 1: ML Trading Bot (`bot_trading_ml_econofisica.py`)

A full pipeline that extracts econophysics-based features, trains a classifier, and runs a realistic backtest.

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────┐
│  FEATURES       │───▶│  ML MODEL        │────▶│ BACKTEST   │
│  (Econophysics) │     │  RandomForest /  │     │ ENGINE     │
│                 │     │  XGBoost         │     │            │
│ • Hurst rolling │     │                  │     │ • PnL      │
│ • Vol. regime   │     │  Predicts:       │     │ • Sharpe   │
│ • Tsallis q     │     │  +1 (long)       │     │ • Drawdown │
│ • Momentum      │     │  -1 (short)      │     │ • Win rate │
│ • RSI, BB       │     │   0 (neutral)    │     │            │
└─────────────────┘     └──────────────────┘     └────────────┘
```

### Features used

| Feature | Type | Description |
|---|---|---|
| `hurst_rolling` | Econophysics | Rolling Hurst exponent (R/S method) |
| `fractal_dim` | Econophysics | Fractal dimension = 2 - Hurst |
| `tsallis_q` | Econophysics | Heavy-tail estimation via local kurtosis |
| `regimen_vol` | Econophysics | Volatility regime (calm=0 / stress=1) |
| `autocorr_5` | Econophysics | 5-day autocorrelation (memory signal) |
| `rsi_14/28` | Technical | Relative Strength Index |
| `bb_pos` | Technical | Bollinger Band position |
| `mom_5/10/21/63` | Technical | Multi-horizon momentum |

### Labeling methods

- **Triple Barrier** (López de Prado) — realistic: take profit / stop loss / time limit
- **Fixed Horizon** — simpler: classify return over next N days

### Validation

Walk-forward (rolling) validation — the only statistically correct method for financial time series.

### Backtest features

- Transaction costs and slippage
- Confidence threshold (only trade when model is sure)
- Fractional Kelly position sizing
- Global stop-loss
- Benchmark comparison (Buy & Hold)



## Module 2: Medium/Long-Term Analysis (`econofisica_mediano_largo_plazo.py`)

A deeper econophysics toolkit for multi-year analysis and portfolio construction.

### Modules

| Module | Description |
|---|---|
| `DataLoader` | Downloads up to 10 years of historical data |
| `AnalisisHurst` | Robust Hurst + DFA (Detrended Fluctuation Analysis) |
| `DetectorBurbuja` | Sornette's LPPLS bubble detection model |
| `AnalisisCiclos` | FFT-based cycle and seasonality analysis |
| `RMT_Correlaciones` | Random Matrix Theory for portfolio correlation filtering |
| `OptimizadorRobust` | Entropy-based optimization with drawdown constraints |
| `CoIntegracion` | Long-term cointegration analysis between assets |

---

## Installation

```bash
pip install yfinance numpy scipy pandas matplotlib seaborn scikit-learn xgboost statsmodels ta
```

---


## References

- Mantegna & Stanley — *An Introduction to Econophysics* (Cambridge, 2000)
- López de Prado — *Advances in Financial Machine Learning* (Wiley, 2018)
- Sornette — *Why Stock Markets Crash* (Princeton, 2003)
- Tsallis — *Introduction to Nonextensive Statistical Mechanics* (Springer, 2009)

---

## Author

**Francisco Nahuel Quintanilla**
Physics undergraduate — Universidad Nacional del Sur, Bahía Blanca, Argentina
[LinkedIn](https://www.linkedin.com/in/francisco-quintanilla-b40367386/)
