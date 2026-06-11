# CLAUDE.md

Guidance for Claude Code (and other tooling/contributors) working in this repository.

## What this project is

A quantitative trading research + execution system in two layers:

1. **C++20 backtesting engine** (repo root) — daily-bar backtester that computes features/labels and exports an ML training dataset.
2. **Python ML + production pipeline** (`python/`) — walk-forward training, a daily signal/order pipeline, and **paper-only** IBKR execution.

Research + **paper** trading only. Not a live trading system. See [ABOUT.md](ABOUT.md) and [README.md](README.md) for scope; [RUNBOOK.md](RUNBOOK.md) for the end-to-end operating guide.

## Build / run / test

### C++ engine (repo root)

```powershell
# Build (needs vcpkg with cpr + nlohmann-json)
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build

# Run the backtest + export output/ml_dataset.csv
.\build\Debug\BackTester.exe

# Tests (zero-dependency assert harness in Tests.cpp; ~203 assertions, all must pass)
cmake --build build --target BackTesterTests
.\build\Debug\BackTesterTests.exe
```

CMake targets: `BackTester` (from `Source.cpp`) and `BackTesterTests` (from `Tests.cpp`). C++20, no compiler extensions.

### Python pipeline (`python/`)

```powershell
cd python
pip install -r requirements.txt

# Train models (walk-forward, ~5 min on 101 tickers)
python walk_forward_train.py

# Daily production pipeline
python run_daily.py                                           # sim mode (default)
python run_daily.py --mode ibkr_dry_run                       # connect IBKR, log orders, submit nothing
python run_daily.py --mode ibkr_paper --confirm-paper-orders  # submit pre-market MOO to IBKR paper
python run_daily.py --mode ibkr_paper --confirm-paper-orders --market-hours  # immediate MKT during RTH
python run_daily.py --reconcile-only                          # fetch fills, update state (idempotent)

# Sync machines on one paper account (read-only pull; --apply rebuilds local paper ledger)
python ibkr_sync.py                                           # print broker state + diff vs local
python ibkr_sync.py --apply --confirm                         # rebuild portfolio_state.paper.json from broker

# Data
python download_data.py           # default tickers / incremental refresh
python expand_universe.py         # full 101-stock universe

# IBKR connectivity smoke test (places no orders)
python test_ibkr_connection.py

# Python test suites (zero-dependency assert harness, temp files only)
python test_state_isolation.py    # sim/paper state isolation, migration, reset
python test_ibkr_execution.py     # preflight, integer shares, reconciliation, stale-price guard
python test_ibkr_sync.py          # broker-truth snapshot, position diff, paper-ledger rebuild
```

There is no pytest setup. Tests are plain scripts with a `check(cond, msg)` harness that exits non-zero on any failure. Run them directly. Match this style when adding tests.

## Architecture

### C++ (header-only modules + thin `main`)

- `Core.h` — `Signal`, `Regimes`, `Trade`, `Position`, `Portfolio` (risk sizing, mark-to-market, slippage, fees)
- `Strategies.h` — `MomentumStrategy`, `BuyAndHoldStrategy`, `MLProbabilityStrategy`, `FixedPriceStrategy`
- `Backtest.h` / `MultiAssetBacktest.h` — single- and multi-ticker backtesters
- `FeatureEngine.h` — 11 technical features from past/current bars only (needs 200 bars)
- `LabelEngine.h` — labels `label_target_stop`, `label_median_return`
- `MLDataExporter.h` — writes `date,ticker,features...,labels` (no label-derived feature columns)
- `Metrics.h`, `Day.h`, `PredictionLoader.h` — metrics, bar type, prediction I/O
- `Source.cpp` (run), `Tests.cpp` (tests)

### Python (`python/`)

- `config.py` — **all** thresholds, risk limits, model choice, paths, and IBKR settings. Change behavior here, not by editing modules.
- `run_daily.py` — the daily orchestrator (modes: `sim`, `ibkr_dry_run`, `ibkr_paper`, plus `--reconcile-only`).
- `ibkr_sync.py` — read-only pull of live broker state (open orders/positions/account/fills) to sync machines sharing one paper account; `--apply --confirm` rebuilds `portfolio_state.paper.json` from broker truth. Never submits orders.
- `feature_engine.py`, `predictor.py`, `order_generator.py`, `portfolio.py`, `reporter.py`, `download_data.py`.
- `execution/`:
  - `order.py` — `Order`, `Fill`, `OrderStatus`, `Action` (orders carry `broker_order_id` + `perm_id`)
  - `order_manager.py` — signal→order conversion, lifecycle logging
  - `broker_adapter.py` — abstract `BrokerAdapter` + `SimulatedBroker`
  - `ibkr_broker.py` — paper-only IBKR via `ib_insync`
  - `fill_reconciler.py` — matches broker fills to orders (perm_id → broker_order_id → ticker+action), accumulates partials
  - `portfolio.py` / `portfolio_manager.py` — ledger; state mutated only from confirmed fills

## Critical invariants — do not break

These are load-bearing. Changing them needs deliberate review.

- **Signal/entry timing.** Signal from bar `D`, entry at bar `D+1` open, across all paths (Backtest, MultiAsset, predictions, LabelEngine). No look-ahead. Guarded by C++ timing tests.
- **Walk-forward hygiene.** Expanding-window yearly folds with a per-ticker 20-trading-day purge/embargo. Never select the model from test-set AUC; the production model (`logistic`) is pre-committed.
- **No label leakage.** Feature columns derive only from past/current bars; `FORBIDDEN_FEATURE_COLS` must never enter the feature set.
- **State only from fills.** Portfolio positions/cash change only via confirmed fills (`Portfolio.apply_buy_fill` / `apply_sell_fill` at the exact broker price/commission), never from orders or signals. Atomic saves (`.tmp` → rename, with `.bak`).
- **Reconciliation is idempotent + crash-safe.** Fills match on stable `perm_id`; processed `execId`s live in portfolio state (`processed_fill_ids`) and commit atomically with cash/positions; `fills.csv` is an audit log written only after the save. Re-running `--reconcile-only` never double-counts.
- **State isolation.** `sim` writes `portfolio_state.sim.json`; `ibkr_dry_run`/`ibkr_paper` write `portfolio_state.paper.json`. Unknown modes fall back to sim, never paper. A sim run must never touch paper state. (`state_file_for_mode`)
- **Exit decisioning mirrors the backtest.** Every BUY fill stamps `stop_price = entry*(1-STOP_LOSS_PCT)` and `target_price = entry*(1+TARGET_PROFIT_PCT)` off the (blended) fill price (`apply_buy_fill`); `backfill_exit_levels` retrofits any legacy position stored with zeroed levels. `run_daily.determine_exits` is the single exit authority: it scans completed daily bars **strictly after** the entry date (D+1 onward — never the entry bar), flags `stop_hit`/`target_hit` on `low<=stop`/`high>=target`, breaks a same-bar both-hit tie by open-distance (stop wins ties, matching `LabelEngine.compute_target_stop`), and falls back to `max_hold_exit` at `MAX_HOLD_DAYS`. Exits act at the next open (MOO/MKT SELL). The reason flows through `order_generator` into the SELL order and trade record. Keep this aligned with `LabelEngine`.

## IBKR paper safety (hard rules)

- **Live trading is disabled.** Port **4002** (paper) is hardcoded in `execution/ibkr_broker.py`; **4001 (live) must stay blocked**. `PAPER_TRADING_ONLY` is enforced at import time. Do not add CLI/env overrides for the port.
- **Confirmation flags are required and must not be removed.** `ibkr_paper` needs `--confirm-paper-orders`. Submitting outside the active window (04:00–09:25 ET for MOO, 09:30–16:00 ET for `--market-hours`) needs BOTH `--override-time-check` and `--i-understand-time-risk`.
- **Sub-allocation capital.** The strategy sizes off `IBKR_PAPER_STRATEGY_CAPITAL` (default $10k), NOT the broker's total paper balance (auto-funded ~$1M+). Pre-flight equity is **one-sided** (broker ≥ local — a surplus is expected; broker *below* local = stale/corrupt local state) plus a buying-power check. Do **not** reintroduce a symmetric equity-equality check.
- **Position safety.** Local positions must be backed at the broker (hard fail); unrelated broker-only holdings only warn. Do not weaken this.
- **Integer shares.** IBKR rejects fractional orders. All IBKR-bound orders are floored to whole shares via `floor_shares_for_ibkr` / `build_ibkr_order`; sub-1 orders are skipped. Backtest/sim/research sizing stays fractional.
- **Stale-price guard.** `ibkr_paper` submission is blocked when the latest bar is older than `MAX_DATA_STALENESS_TRADING_DAYS` NYSE sessions (weekend/holiday aware via `pandas_market_calendars`). Override one run with `--allow-stale-data`.

## Conventions

- Centralize tunables in `config.py` (Python) / constants at the top of the relevant header (C++).
- Tests are zero-dependency assert scripts; new tests should follow the existing `check()` harness and use temp files — never touch real `output/` state.
- Keep changes small and testable. Run both Python suites and the C++ tests before considering a change done.
- Console output targets a Windows code page; prefer ASCII in printed strings (use `--`, not `—`).

## Output files (`output/`)

`ml_dataset.csv`, `predictions*.csv`, `ml_metrics.csv`, `orders.csv`, `orders_lifecycle.csv` (incl. `broker_order_id`/`perm_id`), `fills.csv` (audit), `ibkr_order_log.csv`, `portfolio_state.{sim,paper}.json` (+`.bak`, incl. `processed_fill_ids`), `daily_report.html`, `run_log.csv`. Models live in `python/models/*.joblib`.
