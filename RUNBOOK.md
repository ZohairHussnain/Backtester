# End-to-End Runbook

This guide walks the full system from a fresh clone to IBKR paper orders, in order. Every command is meant to be run from the path shown in its code block.

> **Safety:** This is research + **paper** trading only. The IBKR live port (4001) is blocked at import time. Nothing here places a real-money order. Do not skip the sim → dry-run → paper progression.

---

## 0. Prerequisites (one time)

**C++ toolchain**
- A C++20 compiler (MSVC / Visual Studio 2022 recommended on Windows)
- CMake 3.20+
- [vcpkg](https://github.com/microsoft/vcpkg) installed, with the `VCPKG_ROOT` path known

**Python**
- Python 3.10+ on `PATH`

**For paper trading only (steps 7–9)**
- IBKR account with paper trading enabled
- IBKR **Gateway** (recommended) or TWS, logged into the **paper** account
- API enabled: Configure → API → Settings → "Enable ActiveX and Socket Clients", socket port **4002** (Gateway paper), "Allow connections from localhost only" checked

---

## 1. Build the C++ engine

From the project root (`BackTester/`):

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

Replace `C:/path/to/vcpkg` with your vcpkg install path. This pulls `cpr` and `nlohmann-json` via vcpkg.

**Verify the build** by running the test suite (203 assertions, all should pass):

```powershell
cmake --build build --target BackTesterTests
.\build\Debug\BackTesterTests.exe
```

Do not proceed if any assertion fails — the tests guard the signal-timing and look-ahead-bias invariants the whole system depends on.

---

## 2. Set up the Python environment

```powershell
cd python
pip install -r requirements.txt
```

This installs pandas, numpy, scikit-learn, xgboost, lightgbm, joblib, yfinance, and ib_insync.

All thresholds, risk limits, model choice, and paths live in `python/config.py`. The defaults are pre-committed (threshold 0.60, top-N 2, max 2 positions, $10,000 starting capital). You normally do **not** edit this.

---

## 3. Download price data

The universe is defined in `ticker_data/universe.txt`. Download / refresh OHLCV for it:

```powershell
cd python
python expand_universe.py     # downloads the full 101-stock universe across 10 sectors
```

For a smaller default set, or to incrementally refresh existing data, use:

```powershell
python download_data.py       # default tickers / incremental update
```

Data lands in `ticker_data/<TICKER>.json`. Re-run any time to update to the latest bars.

---

## 4. Export the ML dataset (C++)

The C++ engine computes the 11 features and the two labels and writes the training dataset:

```powershell
# from project root
.\build\Debug\BackTester.exe
```

This produces `output/ml_dataset.csv` with columns
`date,ticker,features...,label_target_stop,label_median_return,forward_return`.
Feature columns are derived only from past/current bars; label columns are never used as features.

---

## 5. Train models (walk-forward)

```powershell
cd python
python walk_forward_train.py      # ~5 min on 101 tickers
```

What it does:
- Reads `output/ml_dataset.csv`
- Runs expanding-window yearly folds with a per-ticker 20-trading-day purge/embargo (no label overlap across fold boundaries)
- Trains Logistic Regression, XGBoost, and LightGBM per fold (class-balanced)
- Exports the **pre-committed** model (`logistic`) — model choice is not made from test-set AUC

Outputs:
- `python/models/*.joblib` — serialized models per fold
- `output/predictions.csv`, `output/predictions_all.csv` — predictions
- `output/ml_metrics.csv` — per-model per-fold metrics

After this, `python/models/` must contain at least one fold's logistic model — the daily pipeline loads the latest fold.

---

## 6. Run the daily pipeline in SIM mode

This is the safe default and the first thing you run end-to-end. No broker connection.

```powershell
cd python
python run_daily.py               # equivalent to --mode sim
```

The 7-step flow it prints:
1. Update prices
2. Compute latest features for the universe
3. Generate predictions (count above threshold 0.60)
4. Check exits on open positions
5. Generate sized, risk-checked orders → `output/orders.csv`
6. Execute with **simulated** fills (close × (1 + slippage))
7. Save state + write `output/daily_report.html`

State is persisted atomically to `output/portfolio_state.json` (with a `.bak` backup). The run is logged to `output/run_log.csv`. A duplicate run on the same calendar day is detected and skipped.

**Recommendation: run sim daily for ~2 weeks** and confirm the report, orders, and portfolio state look sensible before touching IBKR.

---

## 7. Verify IBKR connectivity (read-only)

Start IBKR Gateway in **paper** mode (see Prerequisites), then:

```powershell
cd python
python test_ibkr_connection.py
```

This connects read-only, prints the account (must start with `DU`/`DF` for paper), positions, and open orders, then disconnects. **It places no orders.** If this fails, fix the Gateway/API setup before going further.

---

## 8. IBKR dry run (connect, log orders, submit nothing)

```powershell
cd python
python run_daily.py --mode ibkr_dry_run
```

Runs the full signal/order flow and connects to IBKR, but logs intended orders to `output/ibkr_order_log.csv` instead of submitting. Portfolio state is **not** mutated. Outside the 04:00–09:25 ET window it warns but still proceeds (since nothing is submitted). Review the logged orders against the sim output.

---

## 9. IBKR paper orders (submits MOO orders)

Only after sim + dry-run look correct. MOO (market-on-open) orders must be submitted in the pre-market window.

```powershell
cd python
python run_daily.py --mode ibkr_paper --confirm-paper-orders
```

Guardrails enforced, in order:
- `--confirm-paper-orders` is **required**; without it the run aborts.
- Hard-blocked outside **04:00–09:25 ET** unless you pass *both* `--override-time-check` and `--i-understand-time-risk`.
- **Pre-flight checks** (the IBKR paper account is funded far above the strategy ledger, so equity is *not* required to match):
  - Equity mismatch between broker and the strategy ledger is **warning-only**.
  - **Buying-power sufficiency is the hard gate**: today's intended BUY notional (computed from *whole* shares) must fit within `IBKR_BUYING_POWER_SAFETY_FRAC` (default 95%) of the broker's available funds.
  - **Positions (option B):** the strategy owns only the tickers in `portfolio_state.paper.json`. Unrelated broker positions are ignored (logged). It is a **hard failure** if an owned ticker is missing at the broker or its broker share count disagrees with the ledger. If positions can't be fetched, the run blocks (fail-safe).
- **Whole shares only:** every IBKR order is floored to an integer; anything that rounds below one share is skipped.
- Per-ticker duplicate checks against `output/orders_lifecycle.csv` and live IBKR open orders.
- No shorting: SELL only for tickers you actually hold; BUY skipped if already holding.

The strategy trades `IBKR_PAPER_STRATEGY_CAPITAL` (default $10,000 in `config.py`), **not** the full paper balance. Submitted orders are logged to `output/orders_lifecycle.csv` (written immediately per order, so a crash mid-run still leaves a reconcilable record) and `output/ibkr_order_log.csv`. MOO orders fill at the open, so the run usually reports no immediate fills.

### After market open: reconcile fills

```powershell
cd python
python run_daily.py --reconcile-only
```

Fetches fills from IBKR, matches them to submitted orders **by broker order id**, and updates `portfolio_state.paper.json`. **Portfolio state is only ever mutated from confirmed fills** — never from orders or signals. Fills are de-duplicated by broker execution id (recorded in `output/fills.csv`), so re-running `--reconcile-only` is **idempotent** — it never double-counts. Run this after the open each day you submitted orders; running it again later is safe.

---

## Manual test checklist (IBKR paper)

Run these in order. Each step is safe; only step 3 submits orders, and only into the paper account.

**1. Connectivity (read-only, places nothing):**
```powershell
cd python
python test_ibkr_connection.py
```
Expect: `SMOKE TEST PASSED`, account starting with `DU`/`DF`, and a printed `BuyingPower`. If `Connection refused`, start IBKR Gateway in paper mode with API on port 4002.

**2. Dry run (connects, logs orders, submits nothing):**
```powershell
python run_daily.py --mode ibkr_dry_run
```
Expect: each intended order printed as `[DRY RUN] BUY <int> <TICKER> MOO` with **whole** share counts. `portfolio_state.paper.json` is **not** mutated.

**3. Submit paper orders (pre-market 04:00–09:25 ET):**
```powershell
python run_daily.py --mode ibkr_paper --confirm-paper-orders
```
Expect: the structured pre-flight block (broker equity / buying power / available funds, local strategy equity, configured capital, intended notional), then `OK: buying power sufficient`, then `SUBMITTED:` lines. A 99% equity gap should print as a `NOTE … Not blocking`, **not** a failure.

**4. After the open: reconcile fills (idempotent):**
```powershell
python run_daily.py --reconcile-only
```
Expect: `RECONCILED: BUY <int> <TICKER> @ $<price>` for each fill, then updated cash. Run it a second time — it should report `No new fills found` (or reconcile nothing) and leave state unchanged.

### Files to inspect

| File | What "good" looks like |
|------|------------------------|
| `output/orders.csv` | today's generated orders (research sizing; may be fractional) |
| `output/orders_lifecycle.csv` | one row per submitted order with a non-empty `broker_order_id` and `SUBMITTED`/`FILLED` status |
| `output/ibkr_order_log.csv` | a `DRY_RUN`/`Submitted`/`REJECTED` line per order; share counts are integers |
| `output/fills.csv` | one row per confirmed fill; no duplicate `fill_id`s after re-reconciling |
| `output/portfolio_state.paper.json` | cash/positions reflect only **confirmed fills**; whole-share positions |
| `output/daily_report.html` | matches the state file (cash, positions, trades) |

---

## Daily operating loop (once live on paper)

| When | Command | Purpose |
|------|---------|---------|
| After close (prep) | `python download_data.py` | refresh price data |
| After close | `.\build\Debug\BackTester.exe` then `python walk_forward_train.py` | (periodically) refresh dataset + retrain |
| Pre-market 04:00–09:25 ET | `python run_daily.py --mode ibkr_paper --confirm-paper-orders` | submit MOO orders |
| After open | `python run_daily.py --reconcile-only` | fetch fills, update state |

You do not need to retrain every day — retrain periodically as new data accumulates, and never reuse test-set metrics for model selection.

---

## Output files reference

All under `output/`:

| File | Written by | Contents |
|------|-----------|----------|
| `ml_dataset.csv` | C++ `BackTester.exe` | features + labels for training |
| `predictions.csv` / `predictions_all.csv` | `walk_forward_train.py` | model predictions |
| `ml_metrics.csv` | `walk_forward_train.py` | per-fold evaluation metrics |
| `orders.csv` | `run_daily.py` (order generator) | today's generated orders |
| `orders_lifecycle.csv` | `run_daily.py` (OrderManager) | submitted-order lifecycle log |
| `fills.csv` | FillReconciler | confirmed fills (idempotency source; deduped by `fill_id`) |
| `ibkr_order_log.csv` | IBKRBroker | every IBKR submit/dry-run/reject |
| `portfolio_state.sim.json` (+`.bak`) | Portfolio (sim) | sim ledger — isolated from paper |
| `portfolio_state.paper.json` (+`.bak`) | Portfolio (ibkr dry-run/paper) | paper ledger — mutated only from confirmed fills |
| `daily_report.html` | Reporter | human-readable daily report |
| `run_log.csv` | `run_daily.py` | one row per pipeline run |

`python/models/*.joblib` holds the trained models.

---

## Troubleshooting

- **`No features computed. Aborting.`** — price data is missing or stale; run `python download_data.py` (need ≥200 bars per ticker).
- **`No model found` / predictor load error** — run `walk_forward_train.py`; confirm `python/models/` has `.joblib` files.
- **`Connection refused` to IBKR** — Gateway not running, not logged in, or API/port 4002 not enabled. Re-run `test_ibkr_connection.py`.
- **Pre-flight blocked: insufficient buying power** — intended BUY notional exceeds 95% of broker available funds; lower `IBKR_PAPER_STRATEGY_CAPITAL`/position count in `config.py`, or check the account is funded.
- **Pre-flight blocked: position mismatch (paper)** — an owned ticker disagrees with the broker; run `python run_daily.py --reconcile-only` to sync, then re-submit. (A 99% *equity* gap is expected and only warns.)
- **`Pipeline already ran today`** — duplicate-run guard; use `--reconcile-only` to fetch fills without regenerating orders.
- **Blocked outside MOO window** — submit between 04:00–09:25 ET, or (rarely) pass both override flags deliberately.

See also: [README.md](README.md), [ABOUT.md](ABOUT.md), [CLAUDE.md](CLAUDE.md).
