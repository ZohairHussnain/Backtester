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

All thresholds, risk limits, model choice, and paths live in `python/config.py`. The defaults are pre-committed (threshold 0.60, $10,000 starting capital). You normally do **not** edit this. For IBKR paper, `IBKR_PAPER_STRATEGY_CAPITAL` (default `$10,000`) is the capital the strategy sizes off — the strategy runs as a sub-allocation inside the larger IBKR paper account, so this is intentionally **not** the broker's total balance.

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
- **Pre-flight checks** (sub-allocation aware — the strategy runs as a `$IBKR_PAPER_STRATEGY_CAPITAL` envelope inside a much larger paper account):
  - Equity guard is **one-sided** — broker equity must be **≥** local strategy equity. A large broker surplus is expected and fine; broker *below* local signals stale/corrupt local state and blocks.
  - **Buying-power sufficiency** — broker buying power must cover the intended BUY notional.
  - **Position check** — every *local* position must be backed at the broker (hard block); unrelated *broker-only* holdings are warnings, not blockers.
  - Logs broker equity / buying power / cash, local equity / cash, configured capital, and intended order notional.
- **Integer shares** — all IBKR-bound orders are floored to whole shares (IBKR rejects fractional); orders that floor to < 1 share are skipped.
- Per-ticker duplicate checks against `output/orders_lifecycle.csv` and live IBKR open orders.
- No shorting: SELL only for tickers you actually hold; BUY skipped if already holding.

Submitted orders are logged to `output/orders_lifecycle.csv` (now including `broker_order_id` and `perm_id`) and `output/ibkr_order_log.csv`. MOO orders fill at the open, so the run usually reports no immediate fills.

### After market open: reconcile fills

```powershell
cd python
python run_daily.py --reconcile-only
```

Fetches fills from IBKR and matches them to submitted orders via the FillReconciler — on **`perm_id`** (IBKR's stable cross-session id), falling back to `broker_order_id` then a unique ticker+action match. Fills are applied to `portfolio_state.paper.json` at the **exact broker fill price and commission** (no simulated slippage/fee), with partial fills accumulated. **Portfolio state is only ever mutated from confirmed fills** — never from orders or signals.

Reconciliation is **idempotent and crash-safe**: each applied execution id is recorded inside the portfolio state (`processed_fill_ids`) and committed atomically with cash/positions, and `fills.csv` is written only *after* that save. Re-running `--reconcile-only` (or re-running after a crash mid-reconcile) never double-counts a fill. Run it after the open each day you submitted orders.

---

## 9a. IBKR paper verification checklist (run once, in order)

A safe, ordered pass to confirm the paper integration end-to-end. Stop at the first step that misbehaves.

```powershell
cd python

# 1. Read-only connectivity. Places NO orders. Account must start with DU/DF.
python test_ibkr_connection.py

# 2. Full flow, connects to IBKR, logs intended orders, submits NOTHING.
#    Paper state is NOT mutated.
python run_daily.py --mode ibkr_dry_run

# 3. Submit MOO orders to paper (pre-market 04:00-09:25 ET).
#    Requires the confirmation flag.
python run_daily.py --mode ibkr_paper --confirm-paper-orders

# 4. After the market opens, fetch fills and update the paper ledger.
#    Safe to re-run: idempotent.
python run_daily.py --reconcile-only
```

**What to inspect after each step** (all under `output/`):

| After | File | Look for |
|-------|------|----------|
| 2 (dry run) | `orders.csv` | today's generated orders (sizing/ranking). Fractional `shares` here is fine — they are floored for IBKR downstream. |
| 2 (dry run) | `ibkr_order_log.csv` | one `DRY_RUN` row per order with **whole-share** quantities; any `REJECTED` rows (e.g. shares floored to < 1). |
| 2 (dry run) | `portfolio_state.paper.json` | **unchanged** — dry run must not mutate it. |
| 3 (paper) | console | the pre-flight block: broker equity ≫ local, buying power ≥ intended notional, "Positions backed at broker", "OK" lines. |
| 3 (paper) | `orders_lifecycle.csv` | one row per submitted order with `status=SUBMITTED`, a non-empty **`broker_order_id`** and **`perm_id`**, integer `shares`. |
| 3 (paper) | `ibkr_order_log.csv` | matching `Submitted ... MOO permId=...` rows. |
| 3 (paper) | `daily_report.html` | the order summary matches what was submitted (integer shares). |
| 4 (reconcile) | console | `RECONCILED:` lines for each fill; on a second run, `No NEW fills to apply (all already reconciled).` |
| 4 (reconcile) | `fills.csv` | one row per execution (`fill_id`/execId, `perm_id`, price, commission); no duplicate `fill_id`s after re-running. |
| 4 (reconcile) | `portfolio_state.paper.json` | cash debited/credited at the **actual** fill price; BUY positions added, SELL positions reduced/removed; `processed_fill_ids` populated. Re-running step 4 leaves cash and positions **identical**. |

> Idempotency check: run step 4 twice. The second run must report no new fills and leave `portfolio_state.paper.json` byte-for-byte equivalent in cash/positions. If a fill ever appears twice in `fills.csv` *and* moves cash twice, stop and investigate.

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
| `orders_lifecycle.csv` | `run_daily.py` (OrderManager) | submitted-order lifecycle log; includes `broker_order_id` + `perm_id` |
| `fills.csv` | FillReconciler | confirmed fills (audit log; written after state is saved) |
| `ibkr_order_log.csv` | IBKRBroker | every IBKR submit/dry-run/reject |
| `portfolio_state.paper.json` (+`.bak`) | Portfolio | paper/dry-run ledger: cash, positions, trade history, `processed_fill_ids` |
| `portfolio_state.sim.json` (+`.bak`) | Portfolio | sim-mode ledger (isolated from paper) |
| `daily_report.html` | Reporter | human-readable daily report |
| `run_log.csv` | `run_daily.py` | one row per pipeline run |

`python/models/*.joblib` holds the trained models.

---

## Troubleshooting

- **`No features computed. Aborting.`** — price data is missing or stale; run `python download_data.py` (need ≥200 bars per ticker).
- **`No model found` / predictor load error** — run `walk_forward_train.py`; confirm `python/models/` has `.joblib` files.
- **`Connection refused` to IBKR** — Gateway not running, not logged in, or API/port 4002 not enabled. Re-run `test_ibkr_connection.py`.
- **Pre-flight blocked (paper)** — broker equity *below* local, buying power below intended notional, or a *local* position missing at the broker. Run `python run_daily.py --reconcile-only` to sync, then re-submit. (A large broker surplus and unrelated broker-only holdings are expected and do **not** block.)
- **`Pipeline already ran today`** — duplicate-run guard; use `--reconcile-only` to fetch fills without regenerating orders.
- **Blocked outside MOO window** — submit between 04:00–09:25 ET, or (rarely) pass both override flags deliberately.

See also: [README.md](README.md), [ABOUT.md](ABOUT.md), [CLAUDE.md](CLAUDE.md).
