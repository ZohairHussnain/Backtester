"""
Daily production pipeline.

Usage:
    python run_daily.py                              # default: sim mode
    python run_daily.py --mode sim                   # simulated fills
    python run_daily.py --mode ibkr_dry_run          # connect IBKR, log orders, no submission
    python run_daily.py --mode ibkr_paper --confirm-paper-orders  # submit MOO to IBKR paper
    python run_daily.py --reconcile-only             # fetch fills, update state, no new orders

Safety (ibkr_paper):
    - Requires --confirm-paper-orders to submit any order.
    - Hard-blocked outside 04:00-09:25 ET unless BOTH --override-time-check
      AND --i-understand-time-risk are passed.
    - Pre-flight checks compare local portfolio state vs IBKR broker state.
    - Mismatches block submission until reconciled.
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    UNIVERSE_FILE, MODELS_DIR, PORTFOLIO_STATE_FILE,
    PORTFOLIO_STATE_FILE_SIM, PORTFOLIO_STATE_FILE_PAPER,
    state_file_for_mode,
    ORDERS_FILE, OUTPUT_DIR, STARTING_CAPITAL, THRESHOLD,
    SLIPPAGE, FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT,
)
from download_data import update_prices
from feature_engine import get_latest_features, load_prices
from predictor import Predictor
from portfolio import Portfolio
from order_generator import OrderGenerator
from reporter import Reporter

from execution.order import Order, Action, OrderStatus
from execution.order_manager import OrderManager
from execution.fill_reconciler import FillReconciler

ET = ZoneInfo("America/New_York")
RUN_LOG = OUTPUT_DIR / "run_log.csv"

# Pre-market MOO submission window (ET)
MOO_WINDOW_START_HOUR = 4
MOO_WINDOW_START_MIN = 0
MOO_WINDOW_END_HOUR = 9
MOO_WINDOW_END_MIN = 25

# Tolerance for local vs broker equity mismatch (fraction)
EQUITY_MISMATCH_TOLERANCE = 0.10


# ======================================================================
# Logging
# ======================================================================

def log_run(mode: str, status: str, message: str) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RUN_LOG.exists()
    with open(RUN_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "mode", "status", "message"])
        writer.writerow([datetime.now().isoformat(), mode, status, message])


# ======================================================================
# Timezone-safe market timing
# ======================================================================

def get_et_now() -> datetime:
    return datetime.now(ET)


def is_in_moo_window() -> bool:
    """Check if current ET time is within the MOO order submission window."""
    now = get_et_now()
    start = now.replace(hour=MOO_WINDOW_START_HOUR, minute=MOO_WINDOW_START_MIN, second=0)
    end = now.replace(hour=MOO_WINDOW_END_HOUR, minute=MOO_WINDOW_END_MIN, second=0)
    return start <= now <= end


def check_time_safety(args) -> None:
    """Hard-block ibkr_paper outside MOO window unless both override flags are set."""
    now_et = get_et_now()
    in_window = is_in_moo_window()
    time_str = now_et.strftime("%H:%M:%S ET")

    if in_window:
        print(f"  Time check: {time_str} -- within MOO window (04:00-09:25 ET). OK.")
        return

    print(f"  Time check: {time_str} -- OUTSIDE MOO window (04:00-09:25 ET).")

    if args.mode == "ibkr_dry_run":
        print(f"  WARNING: Outside MOO window. Dry-run proceeds (no orders submitted).")
        return

    if args.mode == "ibkr_paper":
        if args.override_time_check and args.i_understand_time_risk:
            print(f"  OVERRIDE: Both time-override flags set. Proceeding despite timing.")
            log_run(args.mode, "WARNING", f"time override at {time_str}")
            return
        elif args.override_time_check:
            print(f"  BLOCKED: --override-time-check alone is insufficient.")
            print(f"  You must also pass --i-understand-time-risk to submit outside MOO window.")
            log_run(args.mode, "BLOCKED", f"missing --i-understand-time-risk at {time_str}")
            sys.exit(1)
        else:
            print(f"  BLOCKED: MOO orders must be submitted between 04:00-09:25 ET.")
            print(f"  To override, pass BOTH --override-time-check AND --i-understand-time-risk.")
            log_run(args.mode, "BLOCKED", f"outside MOO window at {time_str}")
            sys.exit(1)


# ======================================================================
# Pre-flight sanity checks (ibkr_paper only)
# ======================================================================

def preflight_checks(broker, portfolio: Portfolio) -> bool:
    """Compare local state vs IBKR broker state. Return True if safe to proceed."""
    print("\n  Pre-flight sanity checks...")
    all_ok = True

    # 1. Account value comparison
    try:
        acct = broker.get_account_state()
        broker_equity = acct.get("NetLiquidation", 0)
        broker_cash = acct.get("TotalCashValue", 0)
        local_equity = portfolio.equity
        local_cash = portfolio.cash

        print(f"    Broker equity: ${broker_equity:,.2f}  |  Local equity: ${local_equity:,.2f}")
        print(f"    Broker cash:   ${broker_cash:,.2f}  |  Local cash:   ${local_cash:,.2f}")

        if broker_equity > 0 and local_equity > 0:
            diff = abs(broker_equity - local_equity) / max(broker_equity, local_equity)
            if diff > EQUITY_MISMATCH_TOLERANCE:
                print(f"    FAIL: Equity mismatch {diff:.1%} exceeds {EQUITY_MISMATCH_TOLERANCE:.0%} tolerance.")
                print(f"    Run --reconcile-only to sync state before submitting orders.")
                all_ok = False
            else:
                print(f"    OK: Equity difference {diff:.1%} within tolerance.")
    except Exception as e:
        print(f"    WARNING: Could not fetch account state ({e}). Skipping equity check.")

    # 2. Position comparison
    try:
        broker_positions = broker.get_positions()
        local_positions = set(portfolio.open_positions.keys())
        broker_tickers = set(broker_positions.keys())

        broker_only = broker_tickers - local_positions
        local_only = local_positions - broker_tickers

        if broker_only:
            print(f"    FAIL: Broker has positions not in local state: {broker_only}")
            print(f"    Run --reconcile-only to sync before submitting.")
            all_ok = False
        if local_only:
            print(f"    FAIL: Local state has positions not at broker: {local_only}")
            print(f"    Run --reconcile-only to sync before submitting.")
            all_ok = False
        if not broker_only and not local_only:
            print(f"    OK: Positions match ({len(local_positions)} open).")
    except Exception as e:
        print(f"    WARNING: Could not fetch positions ({e}). Skipping position check.")

    # 3. Open order check
    try:
        broker_orders = broker.get_open_orders()
        if broker_orders:
            known_tickers = set()
            unknown = []
            for bo in broker_orders:
                known_tickers.add(bo.get("ticker", ""))
                # Check if this was submitted by us (hard to verify without order_id mapping)
                unknown.append(f"{bo.get('action','')} {bo.get('shares','')} {bo.get('ticker','')}")
            if unknown:
                print(f"    WARNING: {len(broker_orders)} open orders at broker: {unknown}")
                print(f"    These may conflict with today's submissions.")
        else:
            print(f"    OK: No open orders at broker.")
    except Exception as e:
        print(f"    WARNING: Could not fetch open orders ({e}).")

    return all_ok


def print_order_summary(orders, portfolio: Portfolio, broker) -> None:
    """Print final order summary before submission."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("  IBKR PAPER ORDER SUMMARY")
    print("=" * 60)

    try:
        acct = broker.get_account_state()
        print(f"  Account:       {broker.ib.managedAccounts()[0] if broker.ib.managedAccounts() else '?'}")
        print(f"  Broker equity: ${acct.get('NetLiquidation', 0):,.2f}")
        print(f"  Broker cash:   ${acct.get('TotalCashValue', 0):,.2f}")
    except Exception:
        pass

    print(f"  Local cash:    ${portfolio.cash:,.2f}")
    print(f"  Local equity:  ${portfolio.equity:,.2f}")
    print(f"  Open positions: {len(portfolio.open_positions)}")

    if orders.empty:
        print(f"\n  No orders to submit.")
        return

    total_value = 0.0
    print(f"\n  Orders to submit:")
    print(f"  {'Action':<6} {'Ticker':<8} {'Shares':>8} {'Est.Value':>12} {'Stop':>8} {'Target':>8} {'Prob':>6}")
    print(f"  {'-'*58}")

    for _, o in orders.iterrows():
        shares = o.get("shares", 0)
        # Estimate value from latest close
        try:
            prices = load_prices(o["ticker"])
            price = prices.iloc[-1]["close"]
        except Exception:
            price = 0
        est_val = shares * price
        total_value += est_val if o.get("action") == "BUY" else 0

        print(f"  {o.get('action',''):<6} {o.get('ticker',''):<8} {shares:>8.2f} "
              f"${est_val:>11,.2f} ${o.get('stop_price',0):>7.2f} "
              f"${o.get('target_price',0):>7.2f} {o.get('probability',0):>5.4f}")

    print(f"\n  Total estimated new exposure: ${total_value:,.2f}")
    print(f"  Current positions + new:      {len(portfolio.open_positions) + len(orders[orders['action']=='BUY'])}")
    print("=" * 60)


# ======================================================================
# Broker factory
# ======================================================================

def create_broker(mode: str):
    if mode == "sim":
        from execution.broker_adapter import SimulatedBroker
        return SimulatedBroker()
    elif mode == "ibkr_dry_run":
        from execution.ibkr_broker import IBKRBroker
        broker = IBKRBroker(dry_run=True)
        try:
            broker.connect()
        except Exception as e:
            print(f"  WARNING: IBKR connection failed ({e}). Running in offline dry-run.")
        return broker
    elif mode == "ibkr_paper":
        from execution.ibkr_broker import IBKRBroker, PAPER_TRADING_ONLY
        if not PAPER_TRADING_ONLY:
            print("ERROR: PAPER_TRADING_ONLY is False. Aborting.")
            sys.exit(1)
        broker = IBKRBroker(dry_run=False)
        broker.connect()
        return broker
    else:
        print(f"ERROR: Unknown mode '{mode}'")
        sys.exit(1)


# ======================================================================
# Signal + order generation (shared)
# ======================================================================

def generate_signals_and_orders(portfolio: Portfolio, today: str):
    print("\n[1/7] Updating prices...")
    try:
        update_prices(UNIVERSE_FILE)
    except Exception as e:
        print(f"  WARNING: Price update failed ({e}). Using cached data.")

    print("\n[2/7] Computing features...")
    tickers = [line.strip() for line in open(UNIVERSE_FILE) if line.strip()]
    print(f"  Universe: {len(tickers)} tickers")
    features = get_latest_features(tickers)
    if features.empty:
        print("ERROR: No features computed. Aborting.")
        sys.exit(1)
    print(f"  Features for {len(features)} tickers on {features['date'].iloc[0]}")

    print("\n[3/7] Generating predictions...")
    predictor = Predictor(MODELS_DIR)
    predictor.load_latest_model()
    predictions = predictor.predict(features)
    n_above = (predictions["probability"] >= THRESHOLD).sum()
    print(f"  {len(predictions)} predictions, {n_above} above threshold {THRESHOLD}")

    print("\n[4/7] Checking exits...")
    exit_tickers = portfolio.check_exits(today)
    if exit_tickers:
        print(f"  Exit signals: {exit_tickers}")
    else:
        print(f"  No exits needed.")

    print("\n[5/7] Generating orders...")
    generator = OrderGenerator()
    orders = generator.generate_orders(predictions, portfolio.get_state(), exit_tickers)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_csv(ORDERS_FILE, index=False)
    print(f"  {len(orders)} orders saved to {ORDERS_FILE}")

    return orders, predictions, exit_tickers


# ======================================================================
# Mode: sim
# ======================================================================

def run_sim(today: str):
    portfolio = Portfolio(state_file_for_mode("sim"), STARTING_CAPITAL)
    is_duplicate = check_duplicate(portfolio, today)
    orders, predictions, _ = generate_signals_and_orders(portfolio, today)

    print("\n[6/7] Executing (simulated fills)...")
    if not is_duplicate and not orders.empty:
        _apply_sim_orders(orders, portfolio, today)
    elif is_duplicate:
        print("  SKIPPING: duplicate run detected.")
    else:
        print("  No orders to execute.")

    _save_and_report(portfolio, orders, predictions, today)


def _apply_sim_orders(orders, portfolio: Portfolio, today: str) -> None:
    import pandas as pd
    for _, order in orders[orders["action"] == "SELL"].iterrows():
        ticker = order["ticker"]
        if ticker not in portfolio.open_positions:
            continue
        try:
            pos = portfolio.open_positions[ticker]
            portfolio.record_exit(ticker, pos["entry_price"], today, order.get("reason", "exit"))
            print(f"  EXITED {ticker}")
        except Exception as e:
            print(f"  FAILED EXIT {ticker}: {e}")

    for _, order in orders[orders["action"] == "BUY"].iterrows():
        ticker = order["ticker"]
        if ticker in portfolio.open_positions:
            continue
        try:
            prices = load_prices(ticker)
            latest_close = prices.iloc[-1]["close"]
            entry_price = latest_close * (1 + SLIPPAGE)
            shares = order["shares"]
            fee = max(min(shares * FEE_PER_SHARE, FEE_CAP_PCT * shares * entry_price), FEE_MIN)
            portfolio.record_entry(ticker, shares, entry_price,
                order.get("stop_price", 0), order.get("target_price", 0), today, fee)
            print(f"  ENTERED {ticker}: {shares:.2f} shares @ ${entry_price:.2f}")
        except Exception as e:
            print(f"  FAILED ENTRY {ticker}: {e}")


# ======================================================================
# Mode: ibkr_dry_run
# ======================================================================

def run_ibkr_dry_run(today: str):
    portfolio = Portfolio(state_file_for_mode("ibkr_dry_run"), STARTING_CAPITAL)
    orders, predictions, _ = generate_signals_and_orders(portfolio, today)

    print("\n[6/7] IBKR DRY RUN -- NO ORDERS SUBMITTED...")
    broker = create_broker("ibkr_dry_run")

    for _, order_row in orders.iterrows():
        action = Action.BUY if order_row["action"] == "BUY" else Action.SELL
        order = Order.create(
            date=today, ticker=order_row["ticker"], action=action,
            shares=order_row["shares"],
            stop_price=order_row.get("stop_price", 0),
            target_price=order_row.get("target_price", 0),
            reason=order_row.get("reason", ""),
            probability=order_row.get("probability", 0),
        )
        try:
            broker.submit_order(order)
        except Exception as e:
            print(f"    DRY RUN skip {order.ticker}: {e}")

    if hasattr(broker, "disconnect"):
        broker.disconnect()

    _save_and_report(portfolio, orders, predictions, today, update_state=False)


# ======================================================================
# Mode: ibkr_paper
# ======================================================================

def run_ibkr_paper(today: str):
    portfolio = Portfolio(state_file_for_mode("ibkr_paper"), STARTING_CAPITAL)
    is_duplicate = check_duplicate(portfolio, today)

    if is_duplicate:
        print("  Duplicate run detected. Skipping order generation.")
        print("  Use --reconcile-only to fetch fills.")
        return

    orders, predictions, _ = generate_signals_and_orders(portfolio, today)

    if orders.empty:
        print("\n[6/7] No orders to submit.")
        _save_and_report(portfolio, orders, predictions, today)
        return

    print("\n[6/7] Submitting to IBKR paper...")
    broker = create_broker("ibkr_paper")

    # --- Pre-flight checks ---
    safe = preflight_checks(broker, portfolio)
    if not safe:
        print("\n  PRE-FLIGHT FAILED. Orders will NOT be submitted.")
        print("  Fix the issues above and re-run, or use --reconcile-only first.")
        log_run("ibkr_paper", "BLOCKED", "preflight failed")
        broker.disconnect()
        _save_and_report(portfolio, orders, predictions, today, update_state=False)
        return

    # --- Final order summary ---
    print_order_summary(orders, portfolio, broker)

    # --- Submit orders ---
    reconciler = FillReconciler(OUTPUT_DIR / "fills.csv")
    om = OrderManager(OUTPUT_DIR / "orders_lifecycle.csv")

    submitted_orders = []
    for _, order_row in orders.iterrows():
        action = Action.BUY if order_row["action"] == "BUY" else Action.SELL
        order = Order.create(
            date=today, ticker=order_row["ticker"], action=action,
            shares=order_row["shares"],
            stop_price=order_row.get("stop_price", 0),
            target_price=order_row.get("target_price", 0),
            reason=order_row.get("reason", ""),
            probability=order_row.get("probability", 0),
        )

        # Duplicate checks
        if _has_pending_order(order.ticker, today):
            print(f"    SKIP {order.ticker}: pending order exists for today")
            continue

        # Check IBKR open orders for this ticker
        has_ibkr_dup = False
        try:
            ibkr_open = broker.get_open_orders()
            if any(oo.get("ticker") == order.ticker for oo in ibkr_open):
                print(f"    SKIP {order.ticker}: open IBKR order already exists")
                has_ibkr_dup = True
        except Exception:
            pass

        if has_ibkr_dup:
            continue

        try:
            broker.submit_order(order)
            submitted_orders.append(order)
            print(f"    SUBMITTED: {order.action.value} {order.shares:.0f} {order.ticker}")
        except Exception as e:
            print(f"    REJECTED {order.ticker}: {e}")

    om.log_orders(submitted_orders)

    # Check for immediate fills
    print("\n  Checking for immediate fills...")
    broker_fills = broker.get_fills()
    if broker_fills:
        fills = reconciler.reconcile(broker_fills, submitted_orders)
        for fill in fills:
            try:
                if fill.action == Action.BUY:
                    portfolio.record_entry(
                        fill.ticker, fill.shares_filled, fill.fill_price,
                        0, 0, today, fill.commission)
                else:
                    portfolio.record_exit(fill.ticker, fill.fill_price, today, "ibkr_fill")
                print(f"    FILLED: {fill.action.value} {fill.shares_filled:.0f} "
                      f"{fill.ticker} @ ${fill.fill_price:.2f}")
            except Exception as e:
                print(f"    FILL ERROR {fill.ticker}: {e}")
    else:
        print("  No immediate fills (MOO orders fill at market open).")
        print("  Run --reconcile-only after market open to fetch fills.")

    broker.disconnect()
    _save_and_report(portfolio, orders, predictions, today)


# ======================================================================
# Mode: reconcile-only
# ======================================================================

def run_reconcile_only(today: str):
    print("\n  Reconcile-only mode: fetching fills from IBKR...")

    # Reconciliation applies IBKR fills, so it operates on PAPER state.
    portfolio = Portfolio(PORTFOLIO_STATE_FILE_PAPER, STARTING_CAPITAL)
    broker = create_broker("ibkr_dry_run")

    if not hasattr(broker, '_connected') or not broker._connected:
        print("  Cannot connect to IBKR. Nothing to reconcile.")
        return

    reconciler = FillReconciler(OUTPUT_DIR / "fills.csv")
    broker_fills = broker.get_fills()

    if not broker_fills:
        print("  No new fills found.")
        broker.disconnect()
        return

    print(f"  Found {len(broker_fills)} fills.")

    lifecycle_path = OUTPUT_DIR / "orders_lifecycle.csv"
    pending_orders = []
    if lifecycle_path.exists():
        import pandas as pd
        df = pd.read_csv(lifecycle_path)
        for _, row in df[df["status"].isin(["SUBMITTED", "PENDING"])].iterrows():
            pending_orders.append(Order(
                order_id=row["order_id"], date=row["date"], ticker=row["ticker"],
                action=Action(row["action"]), shares=row["shares"],
                order_type=row.get("order_type", "MOO"),
                stop_price=row.get("stop_price", 0),
                target_price=row.get("target_price", 0),
                status=OrderStatus(row["status"]), reason=row.get("reason", ""),
            ))

    fills = reconciler.reconcile(broker_fills, pending_orders)
    for fill in fills:
        try:
            if fill.action == Action.BUY:
                portfolio.record_entry(
                    fill.ticker, fill.shares_filled, fill.fill_price,
                    0, 0, today, fill.commission)
            else:
                portfolio.record_exit(fill.ticker, fill.fill_price, today, "ibkr_fill")
            print(f"    RECONCILED: {fill.action.value} {fill.shares_filled:.0f} "
                  f"{fill.ticker} @ ${fill.fill_price:.2f}")
        except Exception as e:
            print(f"    RECONCILE ERROR {fill.ticker}: {e}")

    portfolio.save()
    broker.disconnect()
    print(f"  Portfolio updated. Cash: ${portfolio.cash:,.2f}")


# ======================================================================
# Helpers
# ======================================================================

def check_duplicate(portfolio: Portfolio, today: str) -> bool:
    last = portfolio.get_state().get("last_updated", "")
    if last and last[:10] == today:
        print(f"  WARNING: Pipeline already ran today ({last}).")
        return True
    return False


def migrate_legacy_sim_state(legacy: Path = PORTFOLIO_STATE_FILE,
                             sim: Path = PORTFOLIO_STATE_FILE_SIM) -> bool:
    """One-time migration of pre-Phase-1 state.

    If a legacy portfolio_state.json exists and no sim state file does yet,
    copy it to the sim file. Those positions were created by sim fills, so they
    belong to sim -- NOT to paper. Paper state is intentionally left untouched
    so it starts clean. Returns True if a migration was performed.
    """
    if legacy.exists() and not sim.exists():
        sim.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, sim)
        print(f"  Migrated legacy state '{legacy.name}' -> '{sim.name}' (sim only; paper untouched).")
        return True
    return False


def reset_sim_state(sim: Path = PORTFOLIO_STATE_FILE_SIM) -> list:
    """Delete the sim state file and its sidecars so the next sim run starts
    fresh. Does NOT touch paper state. Returns the list of removed paths.
    """
    removed = []
    for p in (sim, Path(str(sim) + ".bak"), Path(str(sim) + ".tmp")):
        if p.exists():
            p.unlink()
            removed.append(p)
    return removed


def _has_pending_order(ticker: str, today: str) -> bool:
    """Check orders_lifecycle.csv for existing SUBMITTED/PENDING order for same ticker+date."""
    lifecycle_path = OUTPUT_DIR / "orders_lifecycle.csv"
    if not lifecycle_path.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(lifecycle_path)
        if "ticker" in df.columns and "date" in df.columns and "status" in df.columns:
            match = df[(df["ticker"] == ticker) & (df["date"] == today) &
                       (df["status"].isin(["SUBMITTED", "PENDING", "FILLED"]))]
            return len(match) > 0
    except Exception:
        pass
    return False


def _save_and_report(portfolio: Portfolio, orders, predictions, today: str,
                     update_state: bool = True) -> None:
    print("\n[7/7] Saving state and generating report...")
    if update_state:
        portfolio.save()

    reporter = Reporter()
    report_path = reporter.generate_report(portfolio.get_state(), orders, predictions)
    print(f"  Report: {report_path}")

    print("\n" + "=" * 60)
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Equity: ${portfolio.equity:,.2f}")
    print(f"  Open positions: {len(portfolio.open_positions)}")
    for ticker, pos in portfolio.open_positions.items():
        print(f"    {ticker}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.2f} "
              f"(entered {pos['entry_date']})")
    print(f"  Closed trades: {len(portfolio.get_state().get('trade_history', []))}")
    print("=" * 60)


# ======================================================================
# Main
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Daily trading pipeline")
    parser.add_argument("--mode", choices=["sim", "ibkr_dry_run", "ibkr_paper"],
                       default="sim", help="Execution mode (default: sim)")
    parser.add_argument("--confirm-paper-orders", action="store_true",
                       help="Required flag to submit IBKR paper orders")
    parser.add_argument("--reconcile-only", action="store_true",
                       help="Fetch fills from IBKR, update portfolio, no new orders")
    parser.add_argument("--override-time-check", action="store_true",
                       help="Allow ibkr_paper outside MOO window (requires --i-understand-time-risk)")
    parser.add_argument("--i-understand-time-risk", action="store_true",
                       help="Second confirmation for time override")
    parser.add_argument("--reset-sim-state", action="store_true",
                       help="Delete the sim portfolio state (and sidecars) and exit. Paper state untouched.")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Reset sim state and exit (does not touch paper state) ---
    if args.reset_sim_state:
        removed = reset_sim_state()
        if removed:
            print(f"Reset sim state. Removed: {[p.name for p in removed]}")
        else:
            print("Reset sim state. Nothing to remove (already clean).")
        return

    # --- One-time migration of legacy state into the sim file ---
    migrate_legacy_sim_state()

    today = datetime.now().strftime("%Y-%m-%d")
    now_et = get_et_now()

    print("=" * 60)
    print(f"  DAILY PIPELINE -- {today} {now_et.strftime('%H:%M:%S ET')}")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    # --- Reconcile-only ---
    if args.reconcile_only:
        log_run(args.mode, "START", "reconcile-only")
        run_reconcile_only(today)
        log_run(args.mode, "COMPLETE", "reconcile-only")
        print("Pipeline complete (reconcile-only).")
        return

    # --- Safety checks for ibkr_paper ---
    if args.mode == "ibkr_paper":
        if not args.confirm_paper_orders:
            print("ERROR: ibkr_paper mode requires --confirm-paper-orders flag.")
            print("\nUsage: python run_daily.py --mode ibkr_paper --confirm-paper-orders")
            log_run(args.mode, "BLOCKED", "missing --confirm-paper-orders")
            sys.exit(1)

    # --- Time check ---
    if args.mode in ("ibkr_dry_run", "ibkr_paper"):
        check_time_safety(args)

    log_run(args.mode, "START", f"mode={args.mode}")

    try:
        if args.mode == "sim":
            run_sim(today)
        elif args.mode == "ibkr_dry_run":
            run_ibkr_dry_run(today)
        elif args.mode == "ibkr_paper":
            run_ibkr_paper(today)

        log_run(args.mode, "COMPLETE", "success")
    except Exception as e:
        log_run(args.mode, "ERROR", str(e))
        print(f"\nPIPELINE ERROR: {e}")
        raise

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
