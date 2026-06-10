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

- **Three execution modes.** `sim` (simulated fills), `ibkr_dry_run` (connect IBKR, log intended orders, submit nothing), `ibkr_paper` (submit live orders to the IBKR paper account). By default orders are pre-market MOO (market-on-open); `--market-hours` submits immediate MKT orders during regular trading hours instead.

- **IBKR paper trading integration.** Paper-only port (4002) hardcoded. Account prefix verification. The strategy runs as a configured sub-allocation (`IBKR_PAPER_STRATEGY_CAPITAL`) inside the larger paper account, so pre-flight uses a one-sided equity guard (broker ≥ local) plus buying-power sufficiency rather than requiring equal balances; local positions must be backed at the broker (unrelated broker holdings only warn). All orders are whole shares (fractional floored). A stale-price guard blocks submission if the latest bar is older than a configurable number of NYSE trading sessions. Duplicate order protection.

- **Order lifecycle management.** SIGNAL -> PENDING -> SUBMITTED -> FILLED/PARTIALLY_FILLED/CANCELLED/REJECTED. Fills are matched to orders on IBKR's stable `perm_id`, applied at the exact broker price/commission (partial fills accumulated), and portfolio state is updated only from confirmed fills. Reconciliation is idempotent and crash-safe — re-running it never double-counts. Atomic JSON saves with backup.

## What it is not

- **Not a live trading system yet.** IBKR integration is paper-only. Live port (4001) is blocked at import time. Switching to live requires a code change visible in git.

- **Not a data provider.** Downloads historical data via yfinance. Does not stream real-time data.

- **Not a high-frequency system.** Operates on daily bars. Signals are generated from the previous close; orders are submitted pre-market (MOO) or, with `--market-hours`, as immediate market orders during regular hours. There is no intraday signal logic — the decision is still one daily-bar signal per name.

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
