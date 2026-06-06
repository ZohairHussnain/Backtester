# BackTester

BackTester is a quantitative trading research and execution system. It began as a C++20 daily-bar backtesting engine and has grown into a full ML-driven trading pipeline: a C++ backtester that exports labeled training data, a Python walk-forward ML training pipeline, a daily production pipeline, and IBKR **paper** trading integration.

> **Status:** Research + paper trading only. This is **not** a live trading system. The IBKR live port (4001) is blocked at import time; switching to live requires a code change visible in git. See [ABOUT.md](ABOUT.md) for the full "what it is / what it is not" breakdown.

## Components

### 1. C++ Backtesting Engine (`*.h`, `Source.cpp`)

Header-only modules with a thin `Source.cpp` `main()`:

- `Core.h` — `Signal`, `Regimes`, `Trade`, `Position`, `Portfolio` (risk-based sizing, mark-to-market, slippage, fees)
- `Strategies.h` — `MomentumStrategy`, `BuyAndHoldStrategy`, `MLProbabilityStrategy`, `FixedPriceStrategy`
- `Backtest.h` — single-ticker backtester (signal from bar i-1, entry at bar i open)
- `MultiAssetBacktest.h` — multi-ticker with a shared calendar and exit-before-entry ordering
- `FeatureEngine.h` — 11 technical features from past/current bars only (requires 200 bars history)
- `LabelEngine.h` — two labels: `label_target_stop` and `label_median_return`
- `MLDataExporter.h` — exports `date,ticker,features...,labels` with no label-derived feature columns
- `Metrics.h` — CAGR, Sharpe, max drawdown, trade metrics

Running the backtester produces console metrics, an equity curve / trade log, and `ml_dataset.csv` for the Python pipeline.

### 2. Python ML Pipeline (`python/`)

- **Walk-forward training** (`walk_forward_train.py`) — expanding-window yearly folds, per-ticker row-based purge/embargo (20 trading days), class-balanced models, pre-committed model export (no test-AUC snooping).
- **Three models per fold** — Logistic Regression (StandardScaler pipeline), XGBoost, LightGBM. Logistic Regression is the production model.
- **101-stock universe** across 10 sectors, downloaded via yfinance (`expand_universe.py`).

### 3. Production Pipeline (`python/`)

A single daily command runs the whole flow:

- `run_daily.py` — downloads data, computes features, predicts, ranks, sizes, and produces orders.
- `config.py` — all thresholds, risk limits, model selection, and paths centralized here.
- Modules: `feature_engine.py`, `predictor.py`, `order_generator.py`, `portfolio.py`, `reporter.py`, `paper_trade_executor.py`.

### 4. Execution / Broker Architecture (`python/execution/`)

- `order.py` — `Order`, `Fill`, `OrderStatus`, `Action`
- `order_manager.py` — signal-to-order conversion with lifecycle tracking
- `broker_adapter.py` — abstract `BrokerAdapter` + `SimulatedBroker`
- `ibkr_broker.py` — paper-only IBKR via `ib_insync` (port 4002 hardcoded, paper account verified)
- `fill_reconciler.py` — matches broker fills to orders, handles partials
- `portfolio_manager.py` — state updated **only** from confirmed fills

## Requirements

**C++:**
- C++20 compiler, CMake 3.20+
- [vcpkg](https://github.com/microsoft/vcpkg) with `cpr` and `nlohmann-json`

**Python:**
- Python 3.10+
- `pip install -r python/requirements.txt` (pandas, numpy, scikit-learn, xgboost, lightgbm, joblib, yfinance, ib_insync)
- For paper trading: IBKR Gateway running in paper mode with the API enabled on port 4002

## Build & Run (C++)

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build
.\build\Debug\BackTester.exe     # runs backtest + exports ml_dataset.csv
```

## Tests (C++)

```powershell
cmake --build build --target BackTesterTests
.\build\Debug\BackTesterTests.exe
```

203 assertions across 15 sections (Trade, Portfolio, Transaction Costs, Strategies, Metrics, TradeMetrics, FeatureEngine, LabelEngine, PredictionLoader, MLDataExporter, Backtest Integration, MultiAssetBacktest, End-to-End Smoke, Signal/Entry Timing, Portfolio Mark-to-Market). Zero-dependency assert harness.

## Python Pipeline

```powershell
cd python
pip install -r requirements.txt

# Train models (walk-forward, ~5 min on 101 tickers)
python walk_forward_train.py

# Daily production pipeline
python run_daily.py                                           # sim mode (default)
python run_daily.py --mode ibkr_dry_run                       # connect IBKR, log orders, no submission
python run_daily.py --mode ibkr_paper --confirm-paper-orders  # submit MOO to IBKR paper
python run_daily.py --reconcile-only                          # fetch fills, update state

# Download / update price data
python download_data.py           # default tickers
python expand_universe.py         # 101-stock universe

# IBKR connection test (places no orders)
python test_ibkr_connection.py
```

## Signal Timing (critical for correctness)

```
Backtest.h:     signal = generate(time_series, i-1)     ->  entry at time_series[i].open
MultiAsset:     signal = generate(data, bar_index-1)    ->  entry at day->open
Predictions:    probability from calendar[index-1]       ->  entry at current day->open
LabelEngine:    features at bar i                        ->  entry_price = days[i+1].open
```

All paths: signal from the previous bar, entry at the current bar open. Verified by 4 timing-specific tests.

## Key Safety Properties

- Entry timing: signal from bar D, entry at bar D+1 open (matches LabelEngine)
- No look-ahead bias (audited multiple times)
- Walk-forward purge removes the last 20 trading rows per ticker before each test year
- Portfolio state updated only from confirmed fills (not from orders or signals)
- IBKR paper port (4002) hardcoded; live port blocked at import time
- Atomic portfolio saves with backup and corrupt-file recovery
- `ibkr_paper` requires `--confirm-paper-orders`
- MOO orders blocked outside 04:00–09:25 ET (requires two override flags)
- Pre-flight checks compare local vs broker equity and positions before submission

## Current Performance (research, not live)

- Model: Logistic Regression on `label_median_return`
- Universe: 101 US stocks across 8 sectors
- AUC: ~0.60 (pooled walk-forward, out-of-sample); leave-one-out ~0.665 cross-sectional
- Top-pick excess return: +1.16% over universe average per 20-day trade
- Signal alpha over random selection: +7–8% CAGR, +1.4 Sharpe

These are research results. Past backtested performance does not guarantee future results.

## Documentation

- [ABOUT.md](ABOUT.md) — what the project is and is not, in detail
- [CLAUDE.md](CLAUDE.md) — build/run/test commands and architecture for contributors and tooling

## Suggested Next Steps

1. Run `run_daily.py` in `sim` mode for ~2 weeks and confirm stable, sensible output.
2. Run `--mode ibkr_dry_run` against IBKR Gateway paper to validate connectivity and order generation without submitting.
3. Only then run `--mode ibkr_paper --confirm-paper-orders` and monitor fills/reconciliation daily.
4. Keep retraining walk-forward as new data arrives; never reuse test-set metrics for model selection.
