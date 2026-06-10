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


def _fake_prices(price):
    """Return a load_prices-compatible frame whose last close is `price`."""
    def _loader(ticker):
        return pd.DataFrame({"close": [price]})
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
    test_fresh_data_passes()
    test_weekend_gap_not_stale()
    test_stale_data_blocks_paper()
    test_stale_data_warns_only_in_sim()
    test_allow_stale_override()
    test_unparseable_date_does_not_block()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
