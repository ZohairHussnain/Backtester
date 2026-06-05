# What BackTester Is

BackTester is a quantitative trading research and execution system. It started as a C++20 daily-bar backtesting engine, and has grown into a full ML-driven trading pipeline with walk-forward validation, a Python production pipeline, and IBKR paper trading integration.

## What it does

### C++ Backtesting Engine

- **Simulates trading strategies on historical data.** Walks through daily bars, generates buy/sell signals, executes against a portfolio with cash, positions, fees, and slippage, and records equity curves and trade logs.

- **Models realistic execution costs.** Entry at next-day open with slippage. Commissions per-share with min/max bounds. Position sizing accounts for costs iteratively.

- **Manages risk at the portfolio level.** Risk-per-trade and max-position-fraction constraints. Mark-to-market equity for sizing. Supports multiple concurrent positions with a configurable cap.

- **Supports swing trade mechanics.** Stop-loss, take-profit targets, and maximum holding period. Exits checked before entries each bar.

- **Runs single-ticker and multi-ticker backtests.** `Backtest` for single-ticker. `MultiAssetBacktest` for multi-ticker with a shared calendar, exit-before-entry ordering, and one equity point per date.

- **Exports ML training data.** FeatureEngine computes 11 technical features from past/current bars only. LabelEngine generates two label types (target-stop and median-return) using future data as targets only. MLDataExporter joins features and labels by date+ticker.

### Python ML Pipeline

- **Walk-forward training.** Expanding-window yearly folds with per-ticker row-based purge/embargo. No label overlap across fold boundaries. Class-balanced training. Pre-committed model export (no test-set snooping).

- **Three models trained per fold.** Logistic Regression (with StandardScaler), XGBoost, LightGBM. Logistic Regression selected as the production model based on research (highest AUC, best calibration).

- **101-stock universe.** Technology, healthcare, financials, consumer, industrials, energy, utilities, communications. Downloaded via yfinance.

- **Research validated.** Cross-sectional leave-one-out AUC ~0.665 across all 7 sectors. Model generalizes — it learns a reusable swing-trading pattern, not stock-specific behavior. Adversarial validation confirmed signal alpha over random selection.

### Production Pipeline

- **Single daily command.** `python run_daily.py` downloads data, computes features, generates predictions, ranks candidates, sizes positions, and produces orders.

- **Three execution modes.** `sim` (simulated fills), `ibkr_dry_run` (connect IBKR, log intended orders, submit nothing), `ibkr_paper` (submit MOO orders to IBKR paper account).

- **IBKR paper trading integration.** Paper-only port (4002) hardcoded. Account prefix verification. Pre-flight checks compare local vs broker state. Duplicate order protection. MOO (market-on-open) orders only.

- **Order lifecycle management.** SIGNAL -> PENDING -> SUBMITTED -> FILLED/CANCELLED/REJECTED. Portfolio state updated only from confirmed fills via FillReconciler. Atomic JSON saves with backup.

## What it is not

- **Not a live trading system yet.** IBKR integration is paper-only. Live port (4001) is blocked at import time. Switching to live requires a code change visible in git.

- **Not a data provider.** Downloads historical data via yfinance. Does not stream real-time data.

- **Not a high-frequency system.** Operates on daily bars. Signals generated after close, orders submitted before open. No intraday logic.

- **Not a portfolio optimizer.** Runs a single pre-committed configuration. Does not search parameter space or do walk-forward optimization.

- **Not multi-asset-class.** US equities only. No futures, options, forex, or crypto.

## Current performance (research, not live)

- Model: Logistic Regression on label_median_return
- Universe: 101 US stocks across 8 sectors
- AUC: ~0.60 (pooled walk-forward, out-of-sample)
- Cross-sectional: leave-one-out AUC ~0.665 across all tickers
- Top-pick excess return: +1.16% over universe average per 20-day trade
- Signal alpha over random selection: +7-8% CAGR, +1.4 Sharpe

## Test coverage

203 assertions across 15 test sections: Trade, Portfolio, Transaction Costs, Strategies, Metrics, TradeMetrics, FeatureEngine, LabelEngine, PredictionLoader, MLDataExporter, Backtest Integration, MultiAssetBacktest, End-to-End Smoke, Signal/Entry Timing, Portfolio Mark-to-Market.
