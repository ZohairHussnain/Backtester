"""
Daily production pipeline.

Usage:
    python run_daily.py                              # default: sim mode
    python run_daily.py --mode sim                   # simulated fills
    python run_daily.py --mode ibkr_dry_run          # connect IBKR, log orders, no submission
    python run_daily.py --mode ibkr_paper --confirm-paper-orders  # submit MOO to IBKR paper
    python run_daily.py --mode ibkr_paper --confirm-paper-orders --market-hours
                                                     # submit immediate MKT orders during RTH
    python run_daily.py --reconcile-only             # fetch fills, update state, no new orders

Order timing:
    - Default: pre-market MOO (market-on-open, TIF=OPG). Submission window is
      04:00-09:25 ET. These fill at the next open.
    - --market-hours: immediate MKT orders (TIF=DAY) for running DURING regular
      trading hours (09:30-16:00 ET). These fill right away, so reconcile in the
      same run instead of waiting for the open.

Safety (ibkr_paper):
    - Requires --confirm-paper-orders to submit any order.
    - Hard-blocked outside the active submission window (04:00-09:25 ET for MOO,
      09:30-16:00 ET for --market-hours) unless BOTH --override-time-check AND
      --i-understand-time-risk are passed.
    - Pre-flight checks compare local portfolio state vs IBKR broker state.
    - Mismatches block submission until reconciled.
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    UNIVERSE_FILE, MODELS_DIR, PORTFOLIO_STATE_FILE,
    PORTFOLIO_STATE_FILE_SIM, PORTFOLIO_STATE_FILE_PAPER,
    state_file_for_mode,
    ORDERS_FILE, OUTPUT_DIR, STARTING_CAPITAL, IBKR_PAPER_STRATEGY_CAPITAL,
    THRESHOLD, SLIPPAGE, FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT,
    MAX_DATA_STALENESS_TRADING_DAYS, MAX_HOLD_DAYS,
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

# Regular trading hours window (ET) — used by --market-hours for immediate MKT orders.
RTH_START_HOUR = 9
RTH_START_MIN = 30
RTH_END_HOUR = 16
RTH_END_MIN = 0

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


def is_in_rth() -> bool:
    """Check if current ET time is within regular trading hours (09:30-16:00 ET)."""
    now = get_et_now()
    start = now.replace(hour=RTH_START_HOUR, minute=RTH_START_MIN, second=0)
    end = now.replace(hour=RTH_END_HOUR, minute=RTH_END_MIN, second=0)
    return start <= now <= end


def check_time_safety(args) -> None:
    """Hard-block ibkr_paper outside the active submission window unless both
    override flags are set.

    The active window depends on the order type:
      - default (MOO):     04:00-09:25 ET pre-market
      - --market-hours (MKT): 09:30-16:00 ET regular trading hours
    """
    now_et = get_et_now()
    time_str = now_et.strftime("%H:%M:%S ET")

    if getattr(args, "market_hours", False):
        in_window = is_in_rth()
        window_label = "RTH"
        window_desc = "09:30-16:00 ET"
        order_desc = "MKT orders"
    else:
        in_window = is_in_moo_window()
        window_label = "MOO"
        window_desc = "04:00-09:25 ET"
        order_desc = "MOO orders"

    if in_window:
        print(f"  Time check: {time_str} -- within {window_label} window ({window_desc}). OK.")
        return

    print(f"  Time check: {time_str} -- OUTSIDE {window_label} window ({window_desc}).")

    if args.mode == "ibkr_dry_run":
        print(f"  WARNING: Outside {window_label} window. Dry-run proceeds (no orders submitted).")
        return

    if args.mode == "ibkr_paper":
        if args.override_time_check and args.i_understand_time_risk:
            print(f"  OVERRIDE: Both time-override flags set. Proceeding despite timing.")
            log_run(args.mode, "WARNING", f"time override at {time_str}")
            return
        elif args.override_time_check:
            print(f"  BLOCKED: --override-time-check alone is insufficient.")
            print(f"  You must also pass --i-understand-time-risk to submit outside the {window_label} window.")
            log_run(args.mode, "BLOCKED", f"missing --i-understand-time-risk at {time_str}")
            sys.exit(1)
        else:
            print(f"  BLOCKED: {order_desc} must be submitted between {window_desc}.")
            print(f"  To override, pass BOTH --override-time-check AND --i-understand-time-risk.")
            log_run(args.mode, "BLOCKED", f"outside {window_label} window at {time_str}")
            sys.exit(1)


# ======================================================================
# Pre-flight sanity checks (ibkr_paper only)
# ======================================================================

def _latest_close(ticker: str) -> float:
    """Latest close for a ticker, or 0.0 if unavailable.

    Single price-lookup seam: every place that needs a current price routes
    through here so behavior (and tests) have one point to stub.
    """
    try:
        prices = load_prices(ticker)
        return float(prices.iloc[-1]["close"])
    except Exception:
        return 0.0


def intended_buy_notional(orders) -> float:
    """Estimate total cash needed for today's BUY orders from latest closes.

    Uses integer share quantities (IBKR floors fractional), so the estimate
    matches what will actually be submitted.
    """
    total = 0.0
    if orders is None or orders.empty:
        return 0.0
    import math
    for _, o in orders.iterrows():
        if o.get("action") != "BUY":
            continue
        shares = math.floor(o.get("shares", 0))
        if shares < 1:
            continue
        total += shares * _latest_close(o["ticker"])
    return total


def preflight_checks(broker, portfolio: Portfolio, orders=None) -> bool:
    """Compare local state vs IBKR broker state. Return True if safe to proceed.

    The strategy runs as a SUB-ALLOCATION inside a larger IBKR paper account, so
    broker equity is expected to be MUCH larger than local strategy equity. The
    checks here are therefore:
      1. Equity guard is ONE-SIDED: broker must be >= local (broker below local
         signals corrupt/stale local state). A broker surplus is fine.
      2. Buying-power sufficiency: broker must have enough buying power for the
         intended BUY notional.
      3. Position check: every LOCAL position must be backed at the broker
         (hard fail). Broker-only positions are unrelated account holdings and
         are reported as warnings, never blockers.
    """
    print("\n  Pre-flight sanity checks...")
    all_ok = True

    # 1. Account value comparison (one-sided) + buying power sufficiency
    try:
        acct = broker.get_account_state()
        broker_equity = acct.get("NetLiquidation", 0)
        broker_cash = acct.get("TotalCashValue", 0)
        broker_bp = acct.get("BuyingPower", 0)
        local_equity = portfolio.equity
        local_cash = portfolio.cash
        order_notional = intended_buy_notional(orders)

        print(f"    Broker equity:        ${broker_equity:,.2f}")
        print(f"    Broker buying power:  ${broker_bp:,.2f}")
        print(f"    Broker cash:          ${broker_cash:,.2f}")
        print(f"    Local strategy equity:${local_equity:,.2f}")
        print(f"    Local strategy cash:  ${local_cash:,.2f}")
        print(f"    Configured capital:   ${IBKR_PAPER_STRATEGY_CAPITAL:,.2f}")
        print(f"    Intended BUY notional:${order_notional:,.2f}")

        # One-sided equity guard: broker below local => stale/corrupt local state.
        if broker_equity > 0 and local_equity > 0:
            if broker_equity < local_equity * (1 - EQUITY_MISMATCH_TOLERANCE):
                shortfall = (local_equity - broker_equity) / local_equity
                print(f"    FAIL: Broker equity is {shortfall:.1%} BELOW local equity.")
                print(f"    This signals corrupt/stale local state. Run --reconcile-only first.")
                all_ok = False
            else:
                print(f"    OK: Broker equity >= local strategy equity (sub-allocation).")

        # Buying-power sufficiency for today's BUYs.
        if order_notional > 0:
            if broker_bp > 0 and broker_bp < order_notional:
                print(f"    FAIL: Buying power ${broker_bp:,.2f} < intended "
                      f"BUY notional ${order_notional:,.2f}.")
                all_ok = False
            else:
                print(f"    OK: Buying power covers intended BUY notional.")
    except Exception as e:
        print(f"    WARNING: Could not fetch account state ({e}). Skipping equity/BP check.")

    # 2. Position comparison (local-backed is a hard requirement; broker-only is OK)
    try:
        broker_positions = broker.get_positions()
        local_positions = set(portfolio.open_positions.keys())
        broker_tickers = set(broker_positions.keys())

        broker_only = broker_tickers - local_positions
        local_only = local_positions - broker_tickers

        if local_only:
            print(f"    FAIL: Local state has positions not at broker: {local_only}")
            print(f"    Run --reconcile-only to sync before submitting.")
            all_ok = False

        # Shared tickers: broker must hold at least the local share count.
        for ticker in (local_positions & broker_tickers):
            local_shares = portfolio.open_positions[ticker].get("shares", 0)
            broker_shares = broker_positions[ticker].get("shares", 0)
            if broker_shares < local_shares:
                print(f"    FAIL: {ticker} broker shares {broker_shares} < local {local_shares}.")
                all_ok = False

        if broker_only:
            print(f"    WARNING: Broker holds positions not tracked by the strategy "
                  f"(unrelated account holdings): {broker_only}")

        if not local_only and not (local_positions & broker_tickers):
            print(f"    OK: No local positions to verify ({len(local_positions)} tracked).")
        elif not local_only:
            print(f"    OK: All {len(local_positions)} local positions backed at broker.")
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
        # Show the floored (integer) quantity that will actually be submitted.
        shares = floor_shares_for_ibkr(o.get("shares", 0))
        if shares < 1:
            continue
        # Estimate value from latest close
        est_val = shares * _latest_close(o["ticker"])
        total_value += est_val if o.get("action") == "BUY" else 0

        print(f"  {o.get('action',''):<6} {o.get('ticker',''):<8} {shares:>8d} "
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

def _nyse_holidays_between(start, end):
    """NYSE holiday dates in [start, end] as np.datetime64, or [] if the
    pandas_market_calendars package is not installed (weekday-only fallback)."""
    try:
        import pandas_market_calendars as mcal
        import pandas as pd
        cal = mcal.get_calendar("XNYS")
        hol = cal.holidays().holidays  # array of datetime64[ns]
        s = pd.Timestamp(start).to_datetime64()
        e = pd.Timestamp(end).to_datetime64()
        return [h for h in hol if s <= h <= e]
    except Exception:
        return []


def trading_days_between(latest_date: str, today: str) -> int:
    """Count NYSE trading sessions strictly after `latest_date`, up to and
    including `today`. Weekends are always excluded; market holidays are too
    when pandas_market_calendars is available. Returns -1 if dates won't parse.

    Examples (no holidays): Mon->Tue = 1; Fri->Mon = 1; Fri->Tue = 2; same day = 0.
    """
    import numpy as np
    try:
        latest = datetime.strptime(str(latest_date)[:10], "%Y-%m-%d").date()
        now = datetime.strptime(str(today)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return -1
    # busday_count is [begin inclusive, end exclusive); we want (latest, now].
    begin = latest + timedelta(days=1)
    end = now + timedelta(days=1)
    holidays = _nyse_holidays_between(begin, now)
    return int(np.busday_count(begin, end, holidays=holidays))


def check_data_freshness(latest_date: str, today: str, mode: str,
                         allow_stale: bool = False) -> None:
    """Guard against trading on stale prices.

    The daily price update is best-effort (yfinance errors are swallowed and the
    run continues on cached data), so a network/yfinance outage can silently
    leave the latest bar days old. This measures the age of the freshest bar the
    pipeline is about to trade on in TRADING SESSIONS (weekend/holiday aware) and,
    for ibkr_paper, BLOCKS submission when it exceeds MAX_DATA_STALENESS_TRADING_DAYS.
    Other modes warn only. Pass --allow-stale-data to override the block.
    """
    gap = trading_days_between(latest_date, today)
    if gap < 0:
        print(f"  WARNING: Could not parse data date '{latest_date}' for staleness check.")
        return

    if gap <= MAX_DATA_STALENESS_TRADING_DAYS:
        print(f"  Data freshness: latest bar {str(latest_date)[:10]} "
              f"({gap} trading session(s) old, threshold "
              f"{MAX_DATA_STALENESS_TRADING_DAYS}). OK.")
        return

    bang = "!" * 56
    print(f"  {bang}")
    print(f"  WARNING: STALE DATA -- latest bar is {str(latest_date)[:10]}, "
          f"{gap} trading sessions old (threshold {MAX_DATA_STALENESS_TRADING_DAYS}).")
    print(f"  The daily price update likely failed (check yfinance / network).")
    print(f"  {bang}")

    if mode == "ibkr_paper" and not allow_stale:
        log_run(mode, "BLOCKED", f"stale data: latest {str(latest_date)[:10]} ({gap} sessions)")
        print(f"  BLOCKED: refusing to submit paper orders on stale data.")
        print(f"  Fix the price data and re-run, or pass --allow-stale-data to override.")
        sys.exit(1)


def determine_exits(portfolio: Portfolio, today: str) -> dict:
    """Decide which open positions to exit today and why.

    Returns {ticker: reason} where reason is one of:
      - "stop_hit"      a bar's low traded down to/through the stop level
      - "target_hit"    a bar's high traded up to/through the target level
      - "max_hold_exit" held >= MAX_HOLD_DAYS NYSE trading sessions

    Price exits take precedence over the time stop. For each position we scan the
    completed daily bars STRICTLY AFTER the entry date, up to and including today
    (D+1 onward — never the entry bar itself, matching the backtest's
    signal->next-open timing). A bar is a stop if low <= stop_price and a target
    if high >= target_price. If a single bar trips BOTH, the tie is broken exactly
    like LabelEngine.compute_target_stop: whichever level is closer to that bar's
    open wins, stop on an exact tie. The first triggering bar decides. Exits are
    acted on at the next open (a MOO/MKT SELL), so this is a daily-check,
    next-open exit — it does not assume an intrabar fill price.
    """
    exits = {}
    for ticker, pos in portfolio.open_positions.items():
        entry_date = str(pos.get("entry_date", ""))[:10]
        stop = pos.get("stop_price", 0) or 0.0
        target = pos.get("target_price", 0) or 0.0
        reason = None

        if stop > 0 and target > 0:
            try:
                prices = load_prices(ticker)
                bars = prices[(prices["date"] > entry_date) & (prices["date"] <= today)]
                for _, bar in bars.iterrows():
                    low = float(bar["low"])
                    high = float(bar["high"])
                    op = float(bar["open"])
                    stop_hit = low <= stop
                    target_hit = high >= target
                    if stop_hit and target_hit:
                        reason = "stop_hit" if abs(op - stop) <= abs(op - target) else "target_hit"
                        break
                    if stop_hit:
                        reason = "stop_hit"
                        break
                    if target_hit:
                        reason = "target_hit"
                        break
            except Exception as e:
                print(f"  WARNING: exit price scan failed for {ticker}: {e}")

        if reason is None:
            # Time-based max hold, counted in NYSE TRADING SESSIONS (not calendar
            # days) so it matches the backtest/LabelEngine horizon, which is in
            # bars. trading_days_between counts sessions strictly after the entry
            # date up to and including today (weekend/holiday aware); it returns
            # -1 when the date won't parse.
            held_sessions = trading_days_between(entry_date, today)
            if held_sessions < 0:
                print(f"  WARNING: bad entry_date for {ticker}, forcing exit")
                reason = "max_hold_exit"
            elif held_sessions >= MAX_HOLD_DAYS:
                reason = "max_hold_exit"

        if reason:
            exits[ticker] = reason
    return exits


def generate_signals_and_orders(portfolio: Portfolio, today: str,
                                mode: str = "sim", allow_stale: bool = False):
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

    # Block trading on stale prices (ibkr_paper) / warn otherwise.
    check_data_freshness(str(features["date"].max()), today, mode, allow_stale)

    print("\n[3/7] Generating predictions...")
    predictor = Predictor(MODELS_DIR)
    predictor.load_latest_model()
    predictions = predictor.predict(features)
    n_above = (predictions["probability"] >= THRESHOLD).sum()
    print(f"  {len(predictions)} predictions, {n_above} above threshold {THRESHOLD}")

    print("\n[4/7] Checking exits...")
    backfilled = portfolio.backfill_exit_levels()
    if backfilled:
        print(f"  Backfilled stop/target on {len(backfilled)} pre-Phase-B "
              f"position(s) from entry price: {backfilled}")
    exits = determine_exits(portfolio, today)
    if exits:
        for t, r in exits.items():
            print(f"  EXIT {t}: {r}")
    else:
        print(f"  No exits needed.")

    print("\n[5/7] Generating orders...")
    generator = OrderGenerator()
    orders = generator.generate_orders(predictions, portfolio.get_state(), exits)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_csv(ORDERS_FILE, index=False)
    print(f"  {len(orders)} orders saved to {ORDERS_FILE}")

    return orders, predictions, exits


# ======================================================================
# Mode: sim
# ======================================================================

def run_sim(today: str):
    portfolio = Portfolio(state_file_for_mode("sim"), STARTING_CAPITAL)
    is_duplicate = check_duplicate(portfolio, today)
    orders, predictions, _ = generate_signals_and_orders(portfolio, today, "sim")

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
# IBKR integer-share enforcement
# ======================================================================

def floor_shares_for_ibkr(shares) -> int:
    """Floor a (possibly fractional) share quantity to whole shares.

    IBKR does not accept fractional share orders, so every IBKR-bound order
    must be a whole number. Floor rounding only; never rounds up.
    """
    import math
    try:
        return int(math.floor(float(shares)))
    except (TypeError, ValueError):
        return 0


def build_ibkr_order(order_row, today: str, order_type: str = "MOO"):
    """Build an integer-share Order for IBKR, or None if it should be skipped.

    Shares are floored to whole units. Orders that floor to < 1 share return
    None so the caller can skip and log them. This is the single place where
    IBKR-bound order quantities are made integer — both dry-run and paper go
    through here, so the lifecycle log, summary, and submission all agree.

    order_type selects the IBKR time-in-force at submission: "MOO" (pre-market
    market-on-open) or "MKT" (immediate market order during trading hours).
    """
    shares = floor_shares_for_ibkr(order_row["shares"])
    if shares < 1:
        return None
    action = Action.BUY if order_row["action"] == "BUY" else Action.SELL
    return Order.create(
        date=today, ticker=order_row["ticker"], action=action,
        shares=shares,
        stop_price=order_row.get("stop_price", 0),
        target_price=order_row.get("target_price", 0),
        reason=order_row.get("reason", ""),
        probability=order_row.get("probability", 0),
        order_type=order_type,
    )


# ======================================================================
# Mode: ibkr_dry_run
# ======================================================================

def run_ibkr_dry_run(today: str, order_type: str = "MOO", allow_stale: bool = False):
    portfolio = Portfolio(state_file_for_mode("ibkr_dry_run"), IBKR_PAPER_STRATEGY_CAPITAL)
    orders, predictions, _ = generate_signals_and_orders(
        portfolio, today, "ibkr_dry_run", allow_stale)

    print(f"\n[6/7] IBKR DRY RUN ({order_type}) -- NO ORDERS SUBMITTED...")
    broker = create_broker("ibkr_dry_run")

    for _, order_row in orders.iterrows():
        order = build_ibkr_order(order_row, today, order_type)
        if order is None:
            print(f"    DRY RUN skip {order_row['ticker']}: "
                  f"{order_row['shares']} shares floors to < 1")
            continue
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

def run_ibkr_paper(today: str, order_type: str = "MOO", allow_stale: bool = False,
                   force_resubmit: bool = False):
    portfolio = Portfolio(state_file_for_mode("ibkr_paper"), IBKR_PAPER_STRATEGY_CAPITAL)
    is_duplicate = check_duplicate(portfolio, today)

    if is_duplicate and not force_resubmit:
        print("  Duplicate run detected. Skipping order generation.")
        print("  Use --reconcile-only to fetch fills, or --force-resubmit to submit")
        print("  again (e.g. after manually cancelling at the broker). Live broker")
        print("  open-order/holding checks prevent duplicate orders.")
        return
    if is_duplicate and force_resubmit:
        print("  Duplicate run, but --force-resubmit set: proceeding.")
        print("  Live broker checks will skip any ticker already working or held.")

    orders, predictions, _ = generate_signals_and_orders(
        portfolio, today, "ibkr_paper", allow_stale)

    if orders.empty:
        print("\n[6/7] No orders to submit.")
        _save_and_report(portfolio, orders, predictions, today)
        return

    print("\n[6/7] Submitting to IBKR paper...")
    broker = create_broker("ibkr_paper")

    # --- Pre-flight checks ---
    safe = preflight_checks(broker, portfolio, orders)
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

    # Duplicate guard: the BROKER is the source of truth for what is currently
    # working. orders_lifecycle.csv is append-only and never learns about
    # cancellations/fills, so it must NOT gate submission (a cancelled-then-
    # resubmitted order would be wrongly skipped). Fetch live open orders once.
    try:
        open_order_tickers = {oo.get("ticker") for oo in broker.get_open_orders()}
    except Exception:
        open_order_tickers = set()

    submitted_orders = []
    for _, order_row in orders.iterrows():
        order = build_ibkr_order(order_row, today, order_type)
        if order is None:
            print(f"    SKIP {order_row['ticker']}: "
                  f"{order_row['shares']} shares floors to < 1")
            continue

        # Skip only if the broker already has a live working order for this ticker.
        if order.ticker in open_order_tickers:
            print(f"    SKIP {order.ticker}: open IBKR order already exists")
            continue

        try:
            # submit_order also re-checks live open orders / holdings and rejects
            # genuine duplicates or shorts against current broker state.
            broker.submit_order(order)  # logs its own SUBMITTED line with the broker id
            submitted_orders.append(order)
            open_order_tickers.add(order.ticker)
        except Exception as e:
            print(f"    REJECTED {order.ticker}: {e}")

    om.log_orders(submitted_orders)

    # Immediate-fill handling.
    #
    # MOO orders CANNOT fill before the open, so we never reconcile them here:
    # ib.fills() returns the account's whole recent execution history, and
    # applying any of it now would attach stale fills to today's pending orders.
    # MOO fills are picked up by --reconcile-only after the open (matched on
    # perm_id). Only MKT orders (run during RTH) can fill immediately.
    applied = []
    if order_type == "MKT":
        print("\n  Checking for immediate fills...")
        exit_reasons = {o.ticker: o.reason for o in submitted_orders
                        if o.action == Action.SELL and o.reason}
        broker_fills = broker.get_fills()
        fills = reconciler.reconcile(broker_fills, submitted_orders) if broker_fills else []
        # Apply SELLs before BUYs (see run_reconcile_only): a rotation SELL frees
        # cash its paired BUY may need, and local cash updates only from fills.
        fills = sorted(fills, key=lambda f: 0 if f.action == Action.SELL else 1)
        for fill in fills:
            try:
                if apply_fill(portfolio, fill, today, exit_reasons):
                    applied.append(fill)
                    print(f"    FILLED: {fill.action.value} {fill.shares_filled:.0f} "
                          f"{fill.ticker} @ ${fill.fill_price:.2f}")
            except Exception as e:
                print(f"    FILL ERROR {fill.ticker}: {e}")
        if not applied:
            print("  No immediate fills yet. MKT orders fill within seconds during RTH.")
        print("  Run --reconcile-only shortly to fetch any remaining fills.")
    else:
        print("\n  MOO orders submitted. They fill at the market open.")
        print("  Run --reconcile-only after the open to fetch fills.")

    broker.disconnect()
    # Save state (commits processed_fill_ids atomically) BEFORE writing the
    # fills audit log, so a crash can never leave an audited-but-unapplied fill.
    _save_and_report(portfolio, orders, predictions, today)
    reconciler.log_fills(applied)


# ======================================================================
# Mode: reconcile-only
# ======================================================================

def run_reconcile_only(today: str):
    print("\n  Reconcile-only mode: fetching fills from IBKR...")

    # Reconciliation applies IBKR fills, so it operates on PAPER state.
    portfolio = Portfolio(PORTFOLIO_STATE_FILE_PAPER, IBKR_PAPER_STRATEGY_CAPITAL)
    broker = create_broker("ibkr_dry_run")

    if not hasattr(broker, '_connected') or not broker._connected:
        print("  Cannot connect to IBKR. Nothing to reconcile.")
        return

    # Ensure every position carries Phase B exit levels. Backfill is pure
    # metadata (it never generates or submits an exit), it is idempotent, and it
    # is not time-gated, so --reconcile-only is the natural place to retrofit
    # legacy positions that were filled before stop/target stamping existed.
    backfilled = portfolio.backfill_exit_levels()
    if backfilled:
        print(f"  Backfilled stop/target on {len(backfilled)} position(s) "
              f"from entry price: {backfilled}")

    reconciler = FillReconciler(OUTPUT_DIR / "fills.csv")
    broker_fills = broker.get_fills()

    if not broker_fills:
        print("  No new fills found.")
        if backfilled:
            portfolio.save()
            print("  Saved backfilled exit levels.")
        broker.disconnect()
        return

    print(f"  Found {len(broker_fills)} fills.")

    lifecycle_path = OUTPUT_DIR / "orders_lifecycle.csv"
    pending_orders = load_pending_orders_from_lifecycle(lifecycle_path)
    exit_reasons = {o.ticker: o.reason for o in pending_orders
                    if o.action == Action.SELL and o.reason}

    fills = reconciler.reconcile(broker_fills, pending_orders)
    # Apply SELLs before BUYs: a same-day rotation/replacement frees cash on the
    # SELL that the paired BUY may need, and local cash updates only from fills.
    fills = sorted(fills, key=lambda f: 0 if f.action == Action.SELL else 1)
    applied = []
    for fill in fills:
        try:
            if apply_fill(portfolio, fill, today, exit_reasons):
                applied.append(fill)
                print(f"    RECONCILED: {fill.action.value} {fill.shares_filled:.0f} "
                      f"{fill.ticker} @ ${fill.fill_price:.2f}")
            else:
                print(f"    SKIP {fill.ticker}: fill {fill.fill_id} already applied")
        except Exception as e:
            print(f"    RECONCILE ERROR {fill.ticker}: {e}")

    if not applied:
        print("  No NEW fills to apply (all already reconciled).")

    # Save state (commits processed_fill_ids atomically) BEFORE the audit log.
    portfolio.save()
    reconciler.log_fills(applied)
    broker.disconnect()
    print(f"  Portfolio updated. Cash: ${portfolio.cash:,.2f}")


# ======================================================================
# Helpers
# ======================================================================

def apply_fill(portfolio: Portfolio, fill, today: str,
               exit_reasons: dict = None) -> bool:
    """Apply a confirmed broker fill to the strategy ledger.

    Returns True if the fill was applied, False if it was already processed
    (idempotent skip). Uses the EXACT broker fill price and commission (no
    simulated slippage/fee) and supports partial fills. The fill id is marked
    processed in the same Portfolio state that records the fill's effect, so the
    mark and the effect are committed together on the next atomic save().

    exit_reasons: optional {ticker: reason} built from the pending SELL orders.
    A SELL is booked into trade_history with WHY it exited
    (stop_hit / target_hit / max_hold_exit) instead of a generic label. Falls
    back to "ibkr_fill" when the reason is unknown (e.g. a manual or unmatched
    sell with no originating order).
    """
    if portfolio.is_fill_processed(fill.fill_id):
        return False
    fill_date = fill.filled_at[:10] if fill.filled_at else today
    if fill.action == Action.BUY:
        portfolio.apply_buy_fill(
            fill.ticker, fill.shares_filled, fill.fill_price,
            fill.commission, fill_date)
    else:
        reason = (exit_reasons or {}).get(fill.ticker, "ibkr_fill")
        portfolio.apply_sell_fill(
            fill.ticker, fill.shares_filled, fill.fill_price,
            fill.commission, fill_date, reason)
    portfolio.mark_fill_processed(fill.fill_id)
    return True


def load_pending_orders_from_lifecycle(lifecycle_path: Path) -> list:
    """Rebuild pending Order objects from orders_lifecycle.csv, INCLUDING the
    broker_order_id / perm_id bridge fields needed to match IBKR fills.
    """
    pending = []
    if not lifecycle_path.exists():
        return pending
    import pandas as pd
    df = pd.read_csv(lifecycle_path)
    open_statuses = ["SUBMITTED", "PENDING", "PARTIALLY_FILLED"]
    for _, row in df[df["status"].isin(open_statuses)].iterrows():
        def _opt(col):
            val = row.get(col)
            if val is None or (isinstance(val, float) and val != val):  # NaN
                return None
            return str(val)
        pending.append(Order(
            order_id=row["order_id"], date=row["date"], ticker=row["ticker"],
            action=Action(row["action"]), shares=row["shares"],
            order_type=row.get("order_type", "MOO"),
            stop_price=row.get("stop_price", 0),
            target_price=row.get("target_price", 0),
            status=OrderStatus(row["status"]), reason=row.get("reason", ""),
            broker_order_id=_opt("broker_order_id"),
            perm_id=_opt("perm_id"),
        ))
    return pending


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
    parser.add_argument("--market-hours", action="store_true",
                       help="Submit immediate MKT orders (TIF=DAY) for running during "
                            "regular trading hours (09:30-16:00 ET) instead of pre-market "
                            "MOO orders. Changes the allowed submission window to RTH.")
    parser.add_argument("--allow-stale-data", action="store_true",
                       help="Override the stale-price guard and submit paper orders even "
                            "when the latest bar is older than MAX_DATA_STALENESS_TRADING_DAYS.")
    parser.add_argument("--force-resubmit", action="store_true",
                       help="Re-run ibkr_paper the same day even if it already ran today. "
                            "Safe: live broker open-order/holding checks prevent duplicate "
                            "orders. Use after manually cancelling orders at the broker.")
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

    order_type = "MKT" if args.market_hours else "MOO"

    try:
        if args.mode == "sim":
            run_sim(today)
        elif args.mode == "ibkr_dry_run":
            run_ibkr_dry_run(today, order_type, args.allow_stale_data)
        elif args.mode == "ibkr_paper":
            run_ibkr_paper(today, order_type, args.allow_stale_data, args.force_resubmit)

        log_run(args.mode, "COMPLETE", "success")
    except Exception as e:
        log_run(args.mode, "ERROR", str(e))
        print(f"\nPIPELINE ERROR: {e}")
        raise

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
