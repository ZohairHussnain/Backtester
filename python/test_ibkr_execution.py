"""IBKR execution tests: preflight (sub-allocation), integer shares,
reconciliation matching, partial fills, idempotency, and crash/re-run safety.

Zero-dependency assert harness (mirrors test_state_isolation.py).
Run:  python test_ibkr_execution.py

Uses temp files and in-memory fakes only. Never connects to IBKR and never
reads or writes the real output/ state files.
"""

import sys
import tempfile
import types
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import run_daily
from portfolio import Portfolio
from execution.order import Order, OrderStatus, Action, Fill
from execution.fill_reconciler import FillReconciler

_passed = 0
_failed = 0


def check(cond: bool, msg: str) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS: {msg}")
    else:
        _failed += 1
        print(f"  FAIL: {msg}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBroker:
    """Minimal stand-in for IBKRBroker used by preflight_checks."""
    def __init__(self, account=None, positions=None, open_orders=None):
        self._account = account or {}
        self._positions = positions or {}
        self._open_orders = open_orders or []

    def get_account_state(self):
        return self._account

    def get_positions(self):
        return self._positions

    def get_open_orders(self):
        return self._open_orders


class FakeReconcileBroker:
    """Connected broker that reports no new fills (for reconcile-only tests)."""
    _connected = True

    def get_fills(self):
        return []

    def disconnect(self):
        pass


def _fake_prices(price):
    """Return a load_prices-compatible frame whose last close is `price`."""
    def _loader(ticker):
        return pd.DataFrame({"close": [price]})
    return _loader


def _fake_prices_by_ticker(mapping, default=50.0):
    """load_prices loader returning a per-ticker close (default for the rest)."""
    def _loader(ticker):
        return pd.DataFrame({"close": [mapping.get(ticker, default)]})
    return _loader


def _preds(rows, date="2026-06-12"):
    """Predictions frame from (ticker, probability) tuples."""
    return pd.DataFrame([{"date": date, "ticker": t, "probability": p}
                         for (t, p) in rows])


def _full_state(holdings, cash=10000.0):
    """portfolio_state with the given tickers held (10 sh @ $50 each)."""
    return {"cash": cash, "open_positions": {
        t: {"shares": 10, "entry_price": 50.0, "entry_date": "2026-06-01",
            "stop_price": 47.5, "target_price": 55.0, "entry_fee": 1.0}
        for t in holdings}}


def _save_rotation(og):
    return (og.ROTATION_ENABLED, og.ROTATION_MIN_PROB_IMPROVEMENT,
            og.ROTATION_ONLY_WHEN_FULL, og.ROTATION_MAX_PER_DAY, og.load_prices)


def _restore_rotation(og, saved):
    (og.ROTATION_ENABLED, og.ROTATION_MIN_PROB_IMPROVEMENT,
     og.ROTATION_ONLY_WHEN_FULL, og.ROTATION_MAX_PER_DAY, og.load_prices) = saved


def _set_rotation(og, enabled=True, improvement=0.05, only_full=True,
                  max_per_day=1, prices=None):
    og.ROTATION_ENABLED = enabled
    og.ROTATION_MIN_PROB_IMPROVEMENT = improvement
    og.ROTATION_ONLY_WHEN_FULL = only_full
    og.ROTATION_MAX_PER_DAY = max_per_day
    if prices is not None:
        og.load_prices = prices


def _fake_bars(bars):
    """Return a load_prices loader yielding the given OHLC bars.

    `bars` is a list of (date, open, high, low) tuples. Close defaults to open.
    """
    def _loader(ticker):
        return pd.DataFrame([
            {"date": d, "open": o, "high": h, "low": l, "close": o}
            for (d, o, h, l) in bars
        ])
    return _loader


def _orders_df(rows):
    return pd.DataFrame(rows)


def _order(ticker, action, shares, perm_id=None, broker_order_id=None, order_id="x"):
    return Order(
        order_id=order_id, date="2026-06-10", ticker=ticker,
        action=Action(action), shares=shares, order_type="MOO",
        stop_price=0, target_price=0, status=OrderStatus.SUBMITTED,
        reason="", broker_order_id=broker_order_id, perm_id=perm_id,
    )


def _broker_fill(ticker, action, shares, price, fill_id, perm_id="", order_id="0",
                 commission=1.0):
    return {
        "fill_id": fill_id, "perm_id": perm_id, "order_id": order_id,
        "ticker": ticker, "action": action, "shares_filled": shares,
        "fill_price": price, "commission": commission, "filled_at": "2026-06-10T09:30:00",
    }


def _fixed_et(hour, minute=0):
    """Pin run_daily.get_et_now to a fixed ET wall-clock time."""
    def _now():
        return datetime(2026, 6, 10, hour, minute, 0, tzinfo=run_daily.ET)
    return _now


def _args(mode="ibkr_paper", market_hours=False,
          override_time_check=False, i_understand_time_risk=False):
    return types.SimpleNamespace(
        mode=mode, market_hours=market_hours,
        override_time_check=override_time_check,
        i_understand_time_risk=i_understand_time_risk,
    )


# ---------------------------------------------------------------------------
# Phase 2: preflight under a large sub-allocation account
# ---------------------------------------------------------------------------

def test_preflight_broker_equity_much_larger_passes():
    print("\n[A] broker equity >> local strategy equity passes")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    broker = FakeBroker(account={
        "NetLiquidation": 1_480_000.0, "TotalCashValue": 1_480_000.0,
        "BuyingPower": 2_000_000.0,
    })
    ok = run_daily.preflight_checks(broker, p, orders=None)
    check(ok is True, "1.48M broker vs 10k local does NOT fail equity check")


def test_preflight_broker_below_local_fails():
    print("\n[B] broker equity below local strategy equity fails (stale state)")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    broker = FakeBroker(account={
        "NetLiquidation": 5000.0, "TotalCashValue": 5000.0, "BuyingPower": 5000.0,
    })
    ok = run_daily.preflight_checks(broker, p, orders=None)
    check(ok is False, "broker 5k < local 10k fails the one-sided guard")


def test_preflight_insufficient_buying_power_fails():
    print("\n[C] insufficient buying power fails")
    run_daily.load_prices = _fake_prices(50.0)
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    broker = FakeBroker(account={
        "NetLiquidation": 1_480_000.0, "TotalCashValue": 1000.0, "BuyingPower": 1000.0,
    })
    orders = _orders_df([{"ticker": "AAA", "action": "BUY", "shares": 100,
                          "stop_price": 0, "target_price": 0}])  # 100 * 50 = 5000
    ok = run_daily.preflight_checks(broker, p, orders=orders)
    check(ok is False, "BP 1000 < intended notional 5000 fails")


def test_preflight_local_only_position_fails():
    print("\n[D] local position not at broker fails")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p.apply_buy_fill("AAA", 10, 50.0, 1.0, "2026-06-10")
    broker = FakeBroker(account={"NetLiquidation": 1_480_000.0, "BuyingPower": 2e6},
                        positions={})  # broker has no AAA
    ok = run_daily.preflight_checks(broker, p, orders=None)
    check(ok is False, "local-only position AAA blocks submission")


def test_preflight_broker_only_position_is_warning():
    print("\n[E] unrelated broker-only position is a warning, not a blocker")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    broker = FakeBroker(account={"NetLiquidation": 1_480_000.0, "BuyingPower": 2e6},
                        positions={"ZZZ": {"shares": 500, "avg_price": 12.0}})
    ok = run_daily.preflight_checks(broker, p, orders=None)
    check(ok is True, "unrelated broker holding ZZZ does NOT block")


# ---------------------------------------------------------------------------
# Phase 3: integer-share enforcement
# ---------------------------------------------------------------------------

def test_floor_shares():
    print("\n[F] floor_shares_for_ibkr floors and never rounds up")
    check(run_daily.floor_shares_for_ibkr(10.99) == 10, "10.99 -> 10")
    check(run_daily.floor_shares_for_ibkr(1.0) == 1, "1.0 -> 1")
    check(run_daily.floor_shares_for_ibkr(0.7) == 0, "0.7 -> 0")
    check(isinstance(run_daily.floor_shares_for_ibkr(5.5), int), "returns int")


def test_build_ibkr_order_is_integer_or_skipped():
    print("\n[G] build_ibkr_order yields whole shares or None")
    o = run_daily.build_ibkr_order(
        {"ticker": "AAA", "action": "BUY", "shares": 12.8,
         "stop_price": 1, "target_price": 2}, "2026-06-10")
    check(o is not None and o.shares == 12, "12.8 -> integer 12 Order")
    check(float(o.shares).is_integer(), "Order.shares is a whole number")
    skip = run_daily.build_ibkr_order(
        {"ticker": "BBB", "action": "BUY", "shares": 0.4,
         "stop_price": 1, "target_price": 2}, "2026-06-10")
    check(skip is None, "0.4 shares -> skipped (None)")


# ---------------------------------------------------------------------------
# Phase 4: reconciliation matching
# ---------------------------------------------------------------------------

def test_reconcile_matches_on_permid_when_orderid_zero():
    print("\n[H] fills match on perm_id even when orderId is 0")
    r = FillReconciler(None)
    order = _order("AAA", "BUY", 10, perm_id="P100", broker_order_id="55")
    bf = _broker_fill("AAA", "BUY", 10, 50.0, fill_id="E1", perm_id="P100", order_id="0")
    fills = r.reconcile([bf], [order])
    check(len(fills) == 1, "one fill matched via perm_id")
    check(order.status == OrderStatus.FILLED, "order marked FILLED")


def test_reconcile_unmatched_is_skipped():
    print("\n[I] a fill with no matching order is skipped, not misapplied")
    r = FillReconciler(None)
    order = _order("AAA", "BUY", 10, perm_id="P100")
    bf = _broker_fill("ZZZ", "SELL", 5, 9.0, fill_id="E9", perm_id="P999")
    fills = r.reconcile([bf], [order])
    check(len(fills) == 0, "unmatched fill produced no Fill")


def test_stale_same_ticker_fill_not_matched():
    print("\n[I2] a stale same-ticker fill (different perm_id) is NOT matched")
    r = FillReconciler(None)
    order = _order("NFLX", "BUY", 9, perm_id="P_today", broker_order_id="26")
    # ib.fills() returns an old NFLX execution from a previous run: same
    # ticker+action, but a different order's ids (orderId came back as 0).
    stale = _broker_fill("NFLX", "BUY", 9, 86.46, fill_id="OLD1",
                         perm_id="P_old", order_id="0")
    fills = r.reconcile([stale], [order])
    check(len(fills) == 0, "stale NFLX fill does not attach to today's NFLX order")
    check(order.status != OrderStatus.FILLED, "today's order not marked filled by a stale fill")


def test_partial_fills_accumulate():
    print("\n[J] partial fills accumulate to FILLED")
    r = FillReconciler(None)
    order = _order("AAA", "BUY", 10, perm_id="P100")
    f1 = _broker_fill("AAA", "BUY", 6, 50.0, fill_id="E1", perm_id="P100")
    f2 = _broker_fill("AAA", "BUY", 4, 50.0, fill_id="E2", perm_id="P100")
    fills = r.reconcile([f1, f2], [order])
    check(len(fills) == 2, "both executions returned")
    check(order.fill_shares == 10, "cumulative fill shares == 10")
    check(order.status == OrderStatus.FILLED, "order FILLED after both partials")


def test_partial_buy_then_sell_in_portfolio():
    print("\n[K] portfolio applies partial buy fills then a partial sell")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p.apply_buy_fill("AAA", 6, 50.0, 1.0, "2026-06-10")
    p.apply_buy_fill("AAA", 4, 52.0, 1.0, "2026-06-10")  # averages in
    pos = p.open_positions["AAA"]
    check(pos["shares"] == 10, "position accumulated to 10 shares")
    check(abs(pos["entry_price"] - (6*50 + 4*52) / 10) < 1e-9, "avg entry price correct")
    p.apply_sell_fill("AAA", 4, 55.0, 1.0, "2026-06-11")
    check(p.open_positions["AAA"]["shares"] == 6, "partial sell leaves 6 shares")
    p.apply_sell_fill("AAA", 6, 55.0, 1.0, "2026-06-11")
    check("AAA" not in p.open_positions, "final sell closes the position")
    check(len(p.get_state()["trade_history"]) == 2, "two exit trades recorded")


# ---------------------------------------------------------------------------
# Phase 4: idempotency / crash-safety
# ---------------------------------------------------------------------------

def test_duplicate_reconciliation_is_idempotent():
    print("\n[L] applying the same fill twice is a no-op")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    fill = Fill("E1", "o1", "AAA", Action.BUY, 10, 50.0, 1.0, "2026-06-10T09:30:00")
    first = run_daily.apply_fill(p, fill, "2026-06-10")
    cash_after_first = p.cash
    second = run_daily.apply_fill(p, fill, "2026-06-10")
    check(first is True, "first application returns True")
    check(second is False, "second application is skipped (False)")
    check(p.cash == cash_after_first, "cash unchanged on duplicate apply")
    check(p.open_positions["AAA"]["shares"] == 10, "position not doubled")


def test_processed_ids_survive_reload():
    print("\n[M] processed fill ids persist across save/reload (re-run safety)")
    with tempfile.TemporaryDirectory() as d:
        state = Path(d) / "portfolio_state.paper.json"
        p = Portfolio(state, 10000.0)
        fill = Fill("E42", "o1", "AAA", Action.BUY, 5, 20.0, 1.0, "2026-06-10T09:30:00")
        run_daily.apply_fill(p, fill, "2026-06-10")
        p.save()

        p2 = Portfolio(state, 10000.0)
        check(p2.is_fill_processed("E42") is True, "reloaded state remembers E42")
        again = run_daily.apply_fill(p2, fill, "2026-06-10")
        check(again is False, "re-run after restart does not re-apply E42")


def test_reconcile_full_flow_idempotent_via_reconciler_and_portfolio():
    print("\n[N] full reconcile path is idempotent across two runs")
    with tempfile.TemporaryDirectory() as d:
        state = Path(d) / "portfolio_state.paper.json"
        fills_log = Path(d) / "fills.csv"
        order = _order("AAA", "BUY", 10, perm_id="P7", broker_order_id="3")
        bf = _broker_fill("AAA", "BUY", 10, 50.0, fill_id="EX7", perm_id="P7", order_id="0")

        # Run 1
        p = Portfolio(state, 10000.0)
        r = FillReconciler(fills_log)
        matched = r.reconcile([bf], [order])
        applied = [f for f in matched if run_daily.apply_fill(p, f, "2026-06-10")]
        p.save()
        r.log_fills(applied)
        cash1 = p.cash

        # Run 2 (same broker fill returned again)
        p2 = Portfolio(state, 10000.0)
        r2 = FillReconciler(fills_log)
        matched2 = r2.reconcile([bf], [order])
        applied2 = [f for f in matched2 if run_daily.apply_fill(p2, f, "2026-06-10")]
        p2.save()
        r2.log_fills(applied2)

        check(len(applied) == 1, "run 1 applied the fill")
        check(len(applied2) == 0, "run 2 applied nothing (idempotent)")
        check(p2.cash == cash1, "cash identical after re-run")
        check(p2.open_positions["AAA"]["shares"] == 10, "position not double-counted")


# ---------------------------------------------------------------------------
# Phase 5: dry-run / fills-only mutation invariants
# ---------------------------------------------------------------------------

def test_reconcile_does_not_mutate_portfolio():
    print("\n[O] reconciler matching never mutates the ledger")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    cash_before = p.cash
    r = FillReconciler(None)
    order = _order("AAA", "BUY", 10, perm_id="P1")
    r.reconcile([_broker_fill("AAA", "BUY", 10, 50.0, "E1", perm_id="P1")], [order])
    check(p.cash == cash_before, "matching fills did not touch cash")
    check(len(p.open_positions) == 0, "matching fills did not add positions")


def test_paper_state_only_mutates_from_fills():
    print("\n[P] building IBKR orders does not mutate paper state")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    cash_before = p.cash
    run_daily.build_ibkr_order(
        {"ticker": "AAA", "action": "BUY", "shares": 12.8,
         "stop_price": 1, "target_price": 2}, "2026-06-10")
    check(p.cash == cash_before, "order generation left cash untouched")
    check(len(p.open_positions) == 0, "order generation added no positions")


# ---------------------------------------------------------------------------
# Market-hours flag: order type + submission window
# ---------------------------------------------------------------------------

def test_order_type_plumbing():
    print("\n[Q] build_ibkr_order carries the requested order type")
    default = run_daily.build_ibkr_order(
        {"ticker": "AAA", "action": "BUY", "shares": 5,
         "stop_price": 1, "target_price": 2}, "2026-06-10")
    check(default.order_type == "MOO", "default order type is MOO")
    mkt = run_daily.build_ibkr_order(
        {"ticker": "AAA", "action": "BUY", "shares": 5,
         "stop_price": 1, "target_price": 2}, "2026-06-10", "MKT")
    check(mkt.order_type == "MKT", "explicit MKT order type is carried through")


def test_window_detection():
    print("\n[R] RTH vs MOO window detection")
    saved = run_daily.get_et_now
    try:
        run_daily.get_et_now = _fixed_et(10, 0)   # 10:00 ET
        check(run_daily.is_in_rth() is True, "10:00 ET is within RTH")
        check(run_daily.is_in_moo_window() is False, "10:00 ET is outside MOO window")
        run_daily.get_et_now = _fixed_et(5, 0)     # 05:00 ET
        check(run_daily.is_in_moo_window() is True, "05:00 ET is within MOO window")
        check(run_daily.is_in_rth() is False, "05:00 ET is outside RTH")
    finally:
        run_daily.get_et_now = saved


def test_time_safety_market_hours_during_rth_proceeds():
    print("\n[S] --market-hours during RTH is allowed without overrides")
    saved_now, saved_log = run_daily.get_et_now, run_daily.log_run
    try:
        run_daily.get_et_now = _fixed_et(10, 0)
        run_daily.log_run = lambda *a, **k: None
        ok = True
        try:
            run_daily.check_time_safety(_args(market_hours=True))
        except SystemExit:
            ok = False
        check(ok is True, "market-hours run at 10:00 ET proceeds (no block)")
    finally:
        run_daily.get_et_now, run_daily.log_run = saved_now, saved_log


def test_time_safety_moo_during_rth_blocks():
    print("\n[T] default MOO during RTH is blocked (wrong window)")
    saved_now, saved_log = run_daily.get_et_now, run_daily.log_run
    try:
        run_daily.get_et_now = _fixed_et(10, 0)
        run_daily.log_run = lambda *a, **k: None
        blocked = False
        try:
            run_daily.check_time_safety(_args(market_hours=False))
        except SystemExit:
            blocked = True
        check(blocked is True, "MOO run at 10:00 ET is blocked (outside 04:00-09:25)")
    finally:
        run_daily.get_et_now, run_daily.log_run = saved_now, saved_log


def test_time_safety_overrides_proceed_outside_window():
    print("\n[U] both override flags proceed outside the window")
    saved_now, saved_log = run_daily.get_et_now, run_daily.log_run
    try:
        run_daily.get_et_now = _fixed_et(20, 0)  # 20:00 ET, outside both windows
        run_daily.log_run = lambda *a, **k: None
        ok = True
        try:
            run_daily.check_time_safety(_args(
                market_hours=True, override_time_check=True, i_understand_time_risk=True))
        except SystemExit:
            ok = False
        check(ok is True, "both override flags allow submission outside RTH")
    finally:
        run_daily.get_et_now, run_daily.log_run = saved_now, saved_log


# ---------------------------------------------------------------------------
# Stale-price guard
# ---------------------------------------------------------------------------

def _exits(fn, *args, **kwargs):
    """Call fn; return True if it raised SystemExit, False otherwise."""
    saved_log = run_daily.log_run
    run_daily.log_run = lambda *a, **k: None
    try:
        fn(*args, **kwargs)
        return False
    except SystemExit:
        return True
    finally:
        run_daily.log_run = saved_log


def test_trading_day_counting():
    print("\n[V0] trading-session counting is weekend-aware")
    # 2026-06-05 Fri, 2026-06-08 Mon, 2026-06-09 Tue, 2026-06-10 Wed
    check(run_daily.trading_days_between("2026-06-08", "2026-06-09") == 1, "Mon->Tue = 1 session")
    check(run_daily.trading_days_between("2026-06-05", "2026-06-08") == 1, "Fri->Mon = 1 (weekend skipped)")
    check(run_daily.trading_days_between("2026-06-05", "2026-06-09") == 2, "Fri->Tue = 2 sessions")
    check(run_daily.trading_days_between("2026-06-09", "2026-06-09") == 0, "same day = 0")
    check(run_daily.trading_days_between("bad", "2026-06-10") == -1, "unparseable -> -1")


def test_holiday_aware_counting():
    print("\n[V0b] NYSE holidays are excluded from session counting")
    # Thanksgiving 2025-11-27 (Thu) is an NYSE holiday; Wed 26 and Fri 28 trade.
    # Weekday-only counting would give 2 (Thu+Fri); holiday-aware gives 1.
    check(run_daily.trading_days_between("2025-11-26", "2025-11-28") == 1,
          "Thanksgiving Thu excluded: Wed->Fri = 1 session")
    # Christmas 2025-12-25 (Thu) holiday; Wed 24 (half day) and Fri 26 trade.
    check(run_daily.trading_days_between("2025-12-24", "2025-12-26") == 1,
          "Christmas Thu excluded: Wed->Fri = 1 session")


def test_fresh_data_passes():
    print("\n[V] fresh data passes the staleness guard")
    blocked = _exits(run_daily.check_data_freshness, "2026-06-09", "2026-06-10",
                     "ibkr_paper", False)
    check(blocked is False, "1-session-old bar does not block ibkr_paper")


def test_weekend_gap_not_stale():
    print("\n[V1] a Friday bar read on Monday is NOT stale")
    blocked = _exits(run_daily.check_data_freshness, "2026-06-05", "2026-06-08",
                     "ibkr_paper", False)
    check(blocked is False, "Fri->Mon (1 session) does not block despite 3 calendar days")


def test_stale_data_blocks_paper():
    print("\n[W] stale data blocks ibkr_paper")
    blocked = _exits(run_daily.check_data_freshness, "2026-05-01", "2026-06-10",
                     "ibkr_paper", False)
    check(blocked is True, "40-day-old bar blocks ibkr_paper submission")


def test_stale_data_warns_only_in_sim():
    print("\n[X] stale data warns but does not block non-paper modes")
    blocked = _exits(run_daily.check_data_freshness, "2026-05-01", "2026-06-10",
                     "sim", False)
    check(blocked is False, "sim mode warns only, never blocks")
    blocked_dry = _exits(run_daily.check_data_freshness, "2026-05-01", "2026-06-10",
                         "ibkr_dry_run", False)
    check(blocked_dry is False, "dry-run warns only, never blocks")


def test_allow_stale_override():
    print("\n[Y] --allow-stale-data overrides the paper block")
    blocked = _exits(run_daily.check_data_freshness, "2026-05-01", "2026-06-10",
                     "ibkr_paper", True)
    check(blocked is False, "allow_stale=True lets stale paper run proceed")


def test_unparseable_date_does_not_block():
    print("\n[Z] an unparseable data date warns, does not crash/block")
    blocked = _exits(run_daily.check_data_freshness, "not-a-date", "2026-06-10",
                     "ibkr_paper", False)
    check(blocked is False, "bad date string is handled gracefully")


# ---------------------------------------------------------------------------
# Phase B: stop/target exit levels + daily-check exits
# ---------------------------------------------------------------------------

def test_buy_fill_derives_stop_and_target():
    print("\n[B1] a BUY fill stamps stop/target from the fill price")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p.apply_buy_fill("AAA", 10, 100.0, 1.0, "2026-06-10")
    pos = p.open_positions["AAA"]
    check(abs(pos["stop_price"] - 95.0) < 1e-6, "stop = entry * (1 - 0.05) = 95.0")
    check(abs(pos["target_price"] - 110.0) < 1e-6, "target = entry * (1 + 0.10) = 110.0")


def test_partial_buy_reblends_stop_target():
    print("\n[B2] averaging-in re-derives stop/target off the blended entry")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p.apply_buy_fill("AAA", 6, 50.0, 1.0, "2026-06-10")
    p.apply_buy_fill("AAA", 4, 52.0, 1.0, "2026-06-10")  # blended entry = 50.8
    pos = p.open_positions["AAA"]
    check(abs(pos["entry_price"] - 50.8) < 1e-9, "blended entry is 50.8")
    check(abs(pos["stop_price"] - 50.8 * 0.95) < 1e-6, "stop re-derived off blended entry")
    check(abs(pos["target_price"] - 50.8 * 1.10) < 1e-6, "target re-derived off blended entry")


def test_backfill_exit_levels():
    print("\n[B3] backfill_exit_levels fills legacy zero stop/target from entry")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p.get_state()["open_positions"]["OLD"] = {
        "shares": 5, "entry_price": 200.0, "entry_date": "2026-06-01",
        "stop_price": 0.0, "target_price": 0.0, "entry_fee": 1.0,
    }
    updated = p.backfill_exit_levels()
    check(updated == ["OLD"], "OLD reported as backfilled")
    pos = p.open_positions["OLD"]
    check(abs(pos["stop_price"] - 190.0) < 1e-6, "stop backfilled to 200*0.95=190")
    check(abs(pos["target_price"] - 220.0) < 1e-6, "target backfilled to 200*1.10=220")
    # Idempotent: a second pass finds nothing to do.
    check(p.backfill_exit_levels() == [], "second backfill is a no-op")


def test_determine_exits_stop_hit():
    print("\n[B4] determine_exits flags a stop when a bar low pierces it")
    saved = run_daily.load_prices
    try:
        p = Portfolio(Path(tempfile.mktemp()), 10000.0)
        p.apply_buy_fill("AAA", 10, 100.0, 1.0, "2026-06-10")  # stop 95, target 110
        run_daily.load_prices = _fake_bars([("2026-06-11", 99.0, 100.0, 94.0)])
        exits = run_daily.determine_exits(p, "2026-06-11")
        check(exits.get("AAA") == "stop_hit", "low 94 <= stop 95 -> stop_hit")
    finally:
        run_daily.load_prices = saved


def test_determine_exits_target_hit():
    print("\n[B5] determine_exits flags a target when a bar high reaches it")
    saved = run_daily.load_prices
    try:
        p = Portfolio(Path(tempfile.mktemp()), 10000.0)
        p.apply_buy_fill("AAA", 10, 100.0, 1.0, "2026-06-10")  # stop 95, target 110
        run_daily.load_prices = _fake_bars([("2026-06-11", 101.0, 111.0, 100.0)])
        exits = run_daily.determine_exits(p, "2026-06-11")
        check(exits.get("AAA") == "target_hit", "high 111 >= target 110 -> target_hit")
    finally:
        run_daily.load_prices = saved


def test_determine_exits_tie_break_stop_wins():
    print("\n[B6] same-bar stop+target ties break by open distance (stop on exact tie)")
    saved = run_daily.load_prices
    try:
        p = Portfolio(Path(tempfile.mktemp()), 10000.0)
        p.apply_buy_fill("AAA", 10, 100.0, 1.0, "2026-06-10")  # stop 95, target 110
        # Bar trips BOTH (low<=95 and high>=110). Open 96 is closer to stop.
        run_daily.load_prices = _fake_bars([("2026-06-11", 96.0, 111.0, 94.0)])
        check(run_daily.determine_exits(p, "2026-06-11").get("AAA") == "stop_hit",
              "open 96 closer to stop -> stop_hit")
        # Open 109 is closer to target.
        run_daily.load_prices = _fake_bars([("2026-06-11", 109.0, 111.0, 94.0)])
        check(run_daily.determine_exits(p, "2026-06-11").get("AAA") == "target_hit",
              "open 109 closer to target -> target_hit")
        # Exact tie (open equidistant) -> stop wins.
        run_daily.load_prices = _fake_bars([("2026-06-11", 102.5, 111.0, 94.0)])
        check(run_daily.determine_exits(p, "2026-06-11").get("AAA") == "stop_hit",
              "equidistant open -> stop wins the tie")
    finally:
        run_daily.load_prices = saved


def test_determine_exits_skips_entry_bar():
    print("\n[B7] the entry bar itself never triggers an exit (D+1 onward)")
    saved = run_daily.load_prices
    try:
        p = Portfolio(Path(tempfile.mktemp()), 10000.0)
        p.apply_buy_fill("AAA", 10, 100.0, 1.0, "2026-06-10")  # stop 95
        # A piercing bar dated on the ENTRY day must be ignored.
        run_daily.load_prices = _fake_bars([("2026-06-10", 96.0, 100.0, 90.0)])
        exits = run_daily.determine_exits(p, "2026-06-10")
        check("AAA" not in exits, "entry-day bar does not trigger an exit")
    finally:
        run_daily.load_prices = saved


def test_determine_exits_max_hold_fallback():
    print("\n[B8] no price hit but past max-hold -> max_hold_exit")
    saved = run_daily.load_prices
    try:
        p = Portfolio(Path(tempfile.mktemp()), 10000.0)
        p.apply_buy_fill("AAA", 1, 100.0, 1.0, "2026-05-01")  # >20 cal days before today
        run_daily.load_prices = _fake_bars([("2026-05-04", 100.0, 101.0, 99.0)])  # no hit
        exits = run_daily.determine_exits(p, "2026-06-10")
        check(exits.get("AAA") == "max_hold_exit", "held > 20 days -> max_hold_exit")
    finally:
        run_daily.load_prices = saved


def test_determine_exits_no_exit():
    print("\n[B9] fresh position, no price hit -> no exit")
    saved = run_daily.load_prices
    try:
        p = Portfolio(Path(tempfile.mktemp()), 10000.0)
        p.apply_buy_fill("AAA", 10, 100.0, 1.0, "2026-06-10")
        run_daily.load_prices = _fake_bars([("2026-06-11", 100.0, 101.0, 99.0)])
        exits = run_daily.determine_exits(p, "2026-06-11")
        check("AAA" not in exits, "no stop/target hit and within max-hold -> no exit")
    finally:
        run_daily.load_prices = saved


def test_exit_order_carries_reason():
    print("\n[B10] order_generator threads the exit reason into the SELL order")
    from order_generator import OrderGenerator
    preds = pd.DataFrame({"date": ["2026-06-11"], "ticker": ["ZZZ"], "probability": [0.1]})
    state = {"cash": 1000.0, "open_positions": {
        "AAA": {"shares": 10, "entry_price": 100.0, "entry_date": "2026-06-10",
                "stop_price": 95.0, "target_price": 110.0, "entry_fee": 1.0}}}
    orders = OrderGenerator().generate_orders(preds, state, {"AAA": "stop_hit"})
    sells = orders[orders["action"] == "SELL"]
    check(len(sells) == 1, "one SELL order generated for the exit")
    check(sells.iloc[0]["reason"] == "stop_hit", "SELL order reason == stop_hit")
    # Legacy list input still works and defaults the reason.
    orders2 = OrderGenerator().generate_orders(preds, state, ["AAA"])
    check(orders2[orders2["action"] == "SELL"].iloc[0]["reason"] == "max_hold_exit",
          "list input defaults reason to max_hold_exit")


def test_reconcile_only_backfills_legacy_levels():
    print("\n[B11] --reconcile-only backfills legacy zero stop/target and saves")
    saved_broker = run_daily.create_broker
    saved_paper = run_daily.PORTFOLIO_STATE_FILE_PAPER
    with tempfile.TemporaryDirectory() as d:
        state = Path(d) / "portfolio_state.paper.json"
        p = Portfolio(state, 10000.0)
        p.get_state()["open_positions"]["OLD"] = {
            "shares": 5, "entry_price": 200.0, "entry_date": "2026-06-01",
            "stop_price": 0.0, "target_price": 0.0, "entry_fee": 1.0}
        p.save()
        try:
            run_daily.PORTFOLIO_STATE_FILE_PAPER = state
            run_daily.create_broker = lambda mode: FakeReconcileBroker()
            run_daily.run_reconcile_only("2026-06-11")
        finally:
            run_daily.create_broker = saved_broker
            run_daily.PORTFOLIO_STATE_FILE_PAPER = saved_paper
        pos = Portfolio(state, 10000.0).open_positions["OLD"]
        check(abs(pos["stop_price"] - 190.0) < 1e-6, "reconcile-only persisted stop=190")
        check(abs(pos["target_price"] - 220.0) < 1e-6, "reconcile-only persisted target=220")


def test_max_hold_uses_trading_days():
    print("\n[B12] max-hold counts trading sessions, not calendar days")
    from datetime import date, timedelta
    saved = run_daily.load_prices
    try:
        entry = "2026-06-10"
        # Flat bar within the stop/target band -> only the time stop can fire.
        run_daily.load_prices = _fake_bars([("2026-06-11", 100.0, 101.0, 99.0)])

        def find_today(n):
            d = date(2026, 6, 11)
            for _ in range(150):
                t = d.isoformat()
                if run_daily.trading_days_between(entry, t) == n:
                    return t
                d += timedelta(days=1)
            raise AssertionError(f"no date with {n} sessions")

        t19 = find_today(19)   # one session short of the 20-session threshold
        t20 = find_today(20)   # exactly at the threshold

        # At 19 trading sessions, MORE than 20 calendar days have elapsed, so a
        # calendar-day rule WOULD exit here. A trading-day rule must not.
        cal_days = (datetime.strptime(t19, "%Y-%m-%d")
                    - datetime.strptime(entry, "%Y-%m-%d")).days
        check(cal_days >= 20, f"19 sessions spans >= 20 calendar days (got {cal_days})")

        p = Portfolio(Path(tempfile.mktemp()), 10000.0)
        p.apply_buy_fill("AAA", 1, 100.0, 1.0, entry)
        check("AAA" not in run_daily.determine_exits(p, t19),
              f"19 trading sessions ({t19}) does NOT trigger max-hold")
        check(run_daily.determine_exits(p, t20).get("AAA") == "max_hold_exit",
              f"20 trading sessions ({t20}) triggers max_hold_exit")
    finally:
        run_daily.load_prices = saved


def test_sell_fill_records_exit_reason():
    print("\n[B13] a SELL fill books its exit reason into trade_history")
    for reason in ("stop_hit", "target_hit", "max_hold_exit"):
        p = Portfolio(Path(tempfile.mktemp()), 10000.0)
        p.apply_buy_fill("AAA", 10, 100.0, 1.0, "2026-06-10")
        sell = Fill("S1", "o1", "AAA", Action.SELL, 10, 105.0, 1.0, "2026-06-12T09:30:00")
        applied = run_daily.apply_fill(p, sell, "2026-06-12", {"AAA": reason})
        th = p.get_state()["trade_history"]
        check(applied is True, f"{reason}: sell applied")
        check(len(th) == 1 and th[0]["reason"] == reason,
              f"trade_history reason == {reason}")
        check("AAA" not in p.open_positions, f"{reason}: position closed")


def test_sell_fill_reason_defaults_when_unknown():
    print("\n[B13b] a SELL with no known reason falls back to 'ibkr_fill'")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p.apply_buy_fill("AAA", 10, 100.0, 1.0, "2026-06-10")
    sell = Fill("S2", "o1", "AAA", Action.SELL, 10, 105.0, 1.0, "2026-06-12T09:30:00")
    run_daily.apply_fill(p, sell, "2026-06-12")  # no exit_reasons map
    th = p.get_state()["trade_history"]
    check(th[0]["reason"] == "ibkr_fill", "unknown-reason sell defaults to ibkr_fill")


def test_reconcile_sell_propagates_reason_end_to_end():
    print("\n[B14] reconcile path maps a SELL order's reason into trade_history")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p.apply_buy_fill("AAA", 9, 80.0, 1.0, "2026-06-10")
    # SELL order as load_pending_orders_from_lifecycle would produce it.
    sell_order = Order(
        order_id="o1", date="2026-06-12", ticker="AAA", action=Action.SELL,
        shares=9, order_type="MOO", stop_price=0, target_price=0,
        status=OrderStatus.SUBMITTED, reason="stop_hit",
        broker_order_id="55", perm_id="P9")
    # Mirror the call-site mapping used by run_reconcile_only / run_ibkr_paper.
    exit_reasons = {o.ticker: o.reason for o in [sell_order]
                    if o.action == Action.SELL and o.reason}
    r = FillReconciler(None)
    bf = _broker_fill("AAA", "SELL", 9, 79.0, fill_id="EX", perm_id="P9", order_id="0")
    fills = r.reconcile([bf], [sell_order])
    for f in fills:
        run_daily.apply_fill(p, f, "2026-06-12", exit_reasons)
    th = p.get_state()["trade_history"]
    check(len(th) == 1 and th[0]["reason"] == "stop_hit",
          "reconciled SELL booked as stop_hit end-to-end")


def test_buy_fill_unaffected_by_exit_reasons():
    print("\n[B15] BUY fills ignore exit_reasons (no regression)")
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    buy = Fill("E1", "o1", "AAA", Action.BUY, 10, 50.0, 1.0, "2026-06-10T09:30:00")
    applied = run_daily.apply_fill(p, buy, "2026-06-10", {"AAA": "stop_hit"})
    pos = p.open_positions["AAA"]
    check(applied is True, "buy applied")
    check(pos["shares"] == 10, "buy added 10 shares")
    check(abs(pos["stop_price"] - 47.5) < 1e-6, "buy still stamps stop from fill price (50*0.95)")
    check(len(p.get_state()["trade_history"]) == 0, "buy writes no trade record")


# ---------------------------------------------------------------------------
# Rotation / replacement exit
# ---------------------------------------------------------------------------

def test_rotation_disabled_by_default():
    print("\n[RO1] rotation is OFF by default -> full book produces no orders")
    import config
    import order_generator as og
    check(config.ROTATION_ENABLED is False, "shipped default ROTATION_ENABLED is False")
    saved = _save_rotation(og)
    try:
        og.ROTATION_ENABLED = False
        og.load_prices = _fake_prices_by_ticker({})
        gen = og.OrderGenerator()
        gen.max_positions = 2
        preds = _preds([("AAA", 0.50), ("BBB", 0.55), ("CCC", 0.90)])
        orders = gen.generate_orders(preds, _full_state(["AAA", "BBB"]), {})
        check(orders.empty, "full book + rotation disabled -> no orders (regression)")
    finally:
        _restore_rotation(og, saved)


def test_rotation_triggers_when_full():
    print("\n[RO2] full book + strong candidate -> SELL weakest, BUY candidate")
    import order_generator as og
    saved = _save_rotation(og)
    try:
        _set_rotation(og, enabled=True, improvement=0.05, only_full=True,
                      max_per_day=1, prices=_fake_prices_by_ticker({}))
        gen = og.OrderGenerator()
        gen.max_positions = 2
        preds = _preds([("AAA", 0.50), ("BBB", 0.55), ("CCC", 0.90)])
        orders = gen.generate_orders(preds, _full_state(["AAA", "BBB"]), {}) \
            .reset_index(drop=True)
        sells = orders[orders["action"] == "SELL"]
        buys = orders[orders["action"] == "BUY"]
        check(len(sells) == 1 and sells.iloc[0]["ticker"] == "AAA",
              "weakest holding AAA (prob 0.50) is sold")
        check(sells.iloc[0]["reason"] == "rotation_exit", "SELL reason is rotation_exit")
        check(len(buys) == 1 and buys.iloc[0]["ticker"] == "CCC",
              "stronger candidate CCC is bought")
        check(str(buys.iloc[0]["reason"]).startswith("rotation_in"),
              "BUY reason records the rotation")
        sell_idx = orders.index[orders["action"] == "SELL"][0]
        buy_idx = orders.index[orders["action"] == "BUY"][0]
        check(sell_idx < buy_idx, "SELL row precedes BUY row (submission order)")
    finally:
        _restore_rotation(og, saved)


def test_rotation_below_improvement_does_nothing():
    print("\n[RO3] candidate not enough stronger -> no rotation (anti-churn)")
    import order_generator as og
    saved = _save_rotation(og)
    try:
        _set_rotation(og, enabled=True, improvement=0.05, only_full=True,
                      prices=_fake_prices_by_ticker({}))
        gen = og.OrderGenerator()
        gen.max_positions = 2
        # weakest = AAA 0.50; need >= 0.55; candidate CCC only 0.53.
        preds = _preds([("AAA", 0.50), ("BBB", 0.55), ("CCC", 0.53)])
        orders = gen.generate_orders(preds, _full_state(["AAA", "BBB"]), {})
        check(orders.empty, "0.53 < 0.50 + 0.05 -> no rotation")
    finally:
        _restore_rotation(og, saved)


def test_rotation_excludes_position_already_exiting():
    print("\n[RO4] never rotates out a position already exiting that day")
    import order_generator as og
    saved = _save_rotation(og)
    try:
        # only_full=False so rotation runs even with a free slot (AAA exiting).
        _set_rotation(og, enabled=True, improvement=0.05, only_full=False,
                      max_per_day=1, prices=_fake_prices_by_ticker({}))
        gen = og.OrderGenerator()
        gen.max_positions = 10
        preds = _preds([("AAA", 0.40), ("BBB", 0.55), ("CCC", 0.60), ("DDD", 0.90)])
        state = _full_state(["AAA", "BBB", "CCC"])
        orders = gen.generate_orders(preds, state, {"AAA": "stop_hit"})
        sells = orders[orders["action"] == "SELL"]
        aaa = sells[sells["ticker"] == "AAA"]
        check(aaa.iloc[0]["reason"] == "stop_hit",
              "AAA leaves via its stop, not rotation")
        check(sells[(sells["ticker"] == "AAA") &
                    (sells["reason"] == "rotation_exit")].empty,
              "AAA is never rotation_exit despite being weakest (it is exiting)")
        rot = sells[sells["reason"] == "rotation_exit"]
        check(len(rot) == 1 and rot.iloc[0]["ticker"] == "BBB",
              "rotation sells BBB (weakest ELIGIBLE), not the exiting AAA")
    finally:
        _restore_rotation(og, saved)


def test_rotation_atomic_no_naked_sell():
    print("\n[RO5] if the replacement BUY can't be placed, emit NEITHER order")
    import order_generator as og
    saved = _save_rotation(og)
    try:
        # CCC priced so high the BUY floors to < 1 share -> rotation must abort.
        _set_rotation(og, enabled=True, improvement=0.05, only_full=True,
                      prices=_fake_prices_by_ticker({"CCC": 100000.0}, default=50.0))
        gen = og.OrderGenerator()
        gen.max_positions = 2
        preds = _preds([("AAA", 0.50), ("BBB", 0.55), ("CCC", 0.90)])
        orders = gen.generate_orders(preds, _full_state(["AAA", "BBB"], cash=1000.0), {})
        check(orders.empty, "no viable BUY -> no SELL either (no naked sell)")
    finally:
        _restore_rotation(og, saved)


def test_rotation_respects_max_per_day():
    print("\n[RO6] rotations are capped at ROTATION_MAX_PER_DAY")
    import order_generator as og
    saved = _save_rotation(og)
    try:
        _set_rotation(og, enabled=True, improvement=0.05, only_full=True,
                      max_per_day=2, prices=_fake_prices_by_ticker({}))
        gen = og.OrderGenerator()
        gen.max_positions = 3
        # 3 weak holdings, 3 strong candidates -> all 3 would qualify; cap = 2.
        preds = _preds([("AAA", 0.40), ("BBB", 0.45), ("CCC", 0.50),
                        ("DDD", 0.95), ("EEE", 0.92), ("FFF", 0.91)])
        state = _full_state(["AAA", "BBB", "CCC"])
        orders = gen.generate_orders(preds, state, {})
        rot = orders[(orders["action"] == "SELL") & (orders["reason"] == "rotation_exit")]
        check(len(rot) == 2, "exactly 2 rotation_exit SELLs (capped), not 3")
    finally:
        _restore_rotation(og, saved)


def test_rotation_excludes_holding_without_prediction():
    print("\n[RO7] a holding with no prediction today is not force-rotated")
    import order_generator as og
    saved = _save_rotation(og)
    try:
        _set_rotation(og, enabled=True, improvement=0.05, only_full=True,
                      prices=_fake_prices_by_ticker({}))
        gen = og.OrderGenerator()
        gen.max_positions = 2
        # AAA held but has NO prediction row; BBB held with 0.55; CCC candidate 0.90.
        preds = _preds([("BBB", 0.55), ("CCC", 0.90)])
        orders = gen.generate_orders(preds, _full_state(["AAA", "BBB"]), {})
        rot = orders[(orders["action"] == "SELL") & (orders["reason"] == "rotation_exit")]
        check(len(rot) == 1 and rot.iloc[0]["ticker"] == "BBB",
              "rotates the ranked holding BBB, never the unscored AAA")
    finally:
        _restore_rotation(og, saved)


def test_reconcile_applies_sells_before_buys():
    print("\n[R1] reconciliation applies SELL fills before BUY fills")
    # Tight cash: a BUY applied before its paired SELL would fail; SELL-first works.
    p = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p.apply_buy_fill("AAA", 10, 50.0, 1.0, "2026-06-10")
    p.get_state()["cash"] = 100.0
    buy = Fill("B1", "o1", "CCC", Action.BUY, 8, 50.0, 1.0, "2026-06-12T09:30:01")
    sell = Fill("S1", "o2", "AAA", Action.SELL, 10, 50.0, 1.0, "2026-06-12T09:30:00")

    # Production sort key (mirrors run_reconcile_only / run_ibkr_paper).
    ordered = sorted([buy, sell], key=lambda f: 0 if f.action == Action.SELL else 1)
    exit_reasons = {"AAA": "rotation_exit"}
    applied = [run_daily.apply_fill(p, f, "2026-06-12", exit_reasons) for f in ordered]
    check(all(applied), "both fills applied when SELL precedes BUY")
    check("AAA" not in p.open_positions, "AAA sold")
    check("CCC" in p.open_positions, "CCC bought from freed cash")
    check(p.get_state()["trade_history"][0]["reason"] == "rotation_exit",
          "rotation SELL booked as rotation_exit")

    # Prove the unsorted (BUY-first) order would have failed -> why we sort.
    p2 = Portfolio(Path(tempfile.mktemp()), 10000.0)
    p2.apply_buy_fill("AAA", 10, 50.0, 1.0, "2026-06-10")
    p2.get_state()["cash"] = 100.0
    raised = False
    try:
        run_daily.apply_fill(p2, buy, "2026-06-12")  # BUY first, tight cash
    except ValueError:
        raised = True
    check(raised, "BUY-before-SELL raises Insufficient cash (the reason for the sort)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  IBKR EXECUTION TESTS")
    print("=" * 60)

    test_preflight_broker_equity_much_larger_passes()
    test_preflight_broker_below_local_fails()
    test_preflight_insufficient_buying_power_fails()
    test_preflight_local_only_position_fails()
    test_preflight_broker_only_position_is_warning()
    test_floor_shares()
    test_build_ibkr_order_is_integer_or_skipped()
    test_reconcile_matches_on_permid_when_orderid_zero()
    test_reconcile_unmatched_is_skipped()
    test_stale_same_ticker_fill_not_matched()
    test_partial_fills_accumulate()
    test_partial_buy_then_sell_in_portfolio()
    test_duplicate_reconciliation_is_idempotent()
    test_processed_ids_survive_reload()
    test_reconcile_full_flow_idempotent_via_reconciler_and_portfolio()
    test_reconcile_does_not_mutate_portfolio()
    test_paper_state_only_mutates_from_fills()
    test_order_type_plumbing()
    test_window_detection()
    test_time_safety_market_hours_during_rth_proceeds()
    test_time_safety_moo_during_rth_blocks()
    test_time_safety_overrides_proceed_outside_window()
    test_trading_day_counting()
    test_holiday_aware_counting()
    test_fresh_data_passes()
    test_weekend_gap_not_stale()
    test_stale_data_blocks_paper()
    test_stale_data_warns_only_in_sim()
    test_allow_stale_override()
    test_unparseable_date_does_not_block()
    test_buy_fill_derives_stop_and_target()
    test_partial_buy_reblends_stop_target()
    test_backfill_exit_levels()
    test_determine_exits_stop_hit()
    test_determine_exits_target_hit()
    test_determine_exits_tie_break_stop_wins()
    test_determine_exits_skips_entry_bar()
    test_determine_exits_max_hold_fallback()
    test_determine_exits_no_exit()
    test_exit_order_carries_reason()
    test_reconcile_only_backfills_legacy_levels()
    test_max_hold_uses_trading_days()
    test_sell_fill_records_exit_reason()
    test_sell_fill_reason_defaults_when_unknown()
    test_reconcile_sell_propagates_reason_end_to_end()
    test_buy_fill_unaffected_by_exit_reasons()
    test_rotation_disabled_by_default()
    test_rotation_triggers_when_full()
    test_rotation_below_improvement_does_nothing()
    test_rotation_excludes_position_already_exiting()
    test_rotation_atomic_no_naked_sell()
    test_rotation_respects_max_per_day()
    test_rotation_excludes_holding_without_prediction()
    test_reconcile_applies_sells_before_buys()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
