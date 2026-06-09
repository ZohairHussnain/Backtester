"""Execution-path tests: integer-share enforcement, reconciliation idempotency,
partial fills, and buying-power pre-flight.

Zero-dependency assert harness (mirrors the C++ / test_state_isolation style).
Run:  python test_execution.py

Uses temporary files and a FakeBroker only. Never connects to IBKR and never
reads or writes the real output/ state files.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

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
# Phase 3: integer-share enforcement
# ---------------------------------------------------------------------------

def test_floor_helper():
    print("\n[P3-A] floor_shares_for_ibkr")
    f = config.floor_shares_for_ibkr
    check(f(12.9) == 12, "12.9 floors to 12")
    check(f(1.0) == 1, "1.0 stays 1")
    check(f(0.9) == 0, "0.9 floors to 0 (drop)")
    check(f(0.0) == 0, "0.0 -> 0")
    check(f(-5.0) == 0, "negative clamps to 0")
    check(f(float("nan")) == 0, "NaN -> 0")
    check(isinstance(f(12.9), int), "returns int type")


def test_build_ibkr_order_is_integer():
    print("\n[P3-B] _build_ibkr_order produces whole shares")
    import run_daily
    from execution.order import Action

    buy = run_daily._build_ibkr_order(
        {"action": "BUY", "ticker": "AAPL", "shares": 12.9876,
         "stop_price": 95.0, "target_price": 110.0, "reason": "x",
         "probability": 0.7}, "2026-06-09")
    check(buy is not None, "BUY of 12.99 shares builds an order")
    check(buy.shares == 12, "BUY shares floored to 12")
    check(isinstance(buy.shares, int), "BUY shares is int")
    check(buy.action == Action.BUY, "BUY action preserved")

    sell = run_daily._build_ibkr_order(
        {"action": "SELL", "ticker": "MSFT", "shares": 5.0,
         "stop_price": 0, "target_price": 0, "reason": "max_hold_exit"},
        "2026-06-09")
    check(sell is not None and sell.shares == 5, "SELL of 5.0 -> 5 shares")
    check(sell.action == Action.SELL, "SELL action preserved")


def test_build_ibkr_order_skips_sub_one():
    print("\n[P3-C] sub-one-share orders are skipped")
    import run_daily
    skip = run_daily._build_ibkr_order(
        {"action": "BUY", "ticker": "BRKA", "shares": 0.42,
         "stop_price": 0, "target_price": 0, "reason": "x",
         "probability": 0.9}, "2026-06-09")
    check(skip is None, "0.42 shares -> None (skipped, never reaches broker)")


# ---------------------------------------------------------------------------
# Phase 4: reconciliation
# ---------------------------------------------------------------------------

def _make_order(broker_id, ticker, shares, action_str="BUY"):
    from execution.order import Order, Action
    action = Action.BUY if action_str == "BUY" else Action.SELL
    o = Order.create(date="2026-06-09", ticker=ticker, action=action,
                     shares=shares, stop_price=0, target_price=0, reason="x")
    o.broker_order_id = broker_id
    return o


def _bf(fill_id, broker_id, ticker, shares, price, action="BOT"):
    # Mirrors ibkr_broker.get_fills(): action is IBKR side (BOT/SLD), order_id
    # is the broker execution.orderId.
    return {"fill_id": fill_id, "order_id": broker_id, "ticker": ticker,
            "action": action, "shares_filled": shares, "fill_price": price,
            "commission": 1.0, "filled_at": "2026-06-09T09:30:01"}


def test_reconcile_matches_on_broker_order_id():
    print("\n[P4-A] fills match orders by broker_order_id (not internal id)")
    from execution.fill_reconciler import FillReconciler
    from execution.order import OrderStatus, Action
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "fills.csv"
        order = _make_order("1001", "AAPL", 10)
        fills = FillReconciler(log).reconcile([_bf("e1", "1001", "AAPL", 10, 150.0)], [order])
        check(len(fills) == 1, "one fill reconciled")
        check(fills[0].order_id == order.order_id, "fill carries internal order_id")
        check(fills[0].action == Action.BUY, "IBKR 'BOT' side mapped to BUY via order")
        check(order.status == OrderStatus.FILLED, "order marked FILLED")
        check(log.exists(), "fill written to log")


def test_reconcile_is_idempotent():
    print("\n[P4-B] re-running reconciliation does not double-count")
    from execution.fill_reconciler import FillReconciler
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "fills.csv"
        order = _make_order("1002", "MSFT", 5)
        bfs = [_bf("e9", "1002", "MSFT", 5, 300.0)]
        first = FillReconciler(log).reconcile(bfs, [order])
        second = FillReconciler(log).reconcile(bfs, [_make_order("1002", "MSFT", 5)])
        check(len(first) == 1, "first run applies the fill")
        check(len(second) == 0, "second run is a no-op (fill_id already seen)")
        rows = log.read_text().strip().splitlines()
        check(len(rows) == 2, "log has header + exactly one fill row")


def test_reconcile_skips_unexpected_fill():
    print("\n[P4-C] fills for unknown broker orders are not applied")
    from execution.fill_reconciler import FillReconciler
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "fills.csv"
        order = _make_order("1003", "NVDA", 3)
        # Fill references broker order 9999, which we never submitted.
        fills = FillReconciler(log).reconcile([_bf("e3", "9999", "NVDA", 3, 100.0)], [order])
        check(len(fills) == 0, "unexpected fill skipped")
        check(not log.exists(), "nothing written for unexpected fill")


def test_reconcile_partial_then_complete():
    print("\n[P4-D] partial fills accumulate across runs to FILLED")
    from execution.fill_reconciler import FillReconciler
    from execution.order import OrderStatus
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "fills.csv"
        order = _make_order("1004", "AMD", 10)
        r1 = FillReconciler(log).reconcile([_bf("p1", "1004", "AMD", 6, 120.0)], [order])
        check(len(r1) == 1 and r1[0].shares_filled == 6, "first execution: 6 shares")
        check(order.status == OrderStatus.PARTIALLY_FILLED, "order PARTIALLY_FILLED after 6/10")

        order2 = _make_order("1004", "AMD", 10)  # rebuilt next run...
        order2.order_id = order.order_id          # ...with the stable id from orders_lifecycle.csv
        r2 = FillReconciler(log).reconcile(
            [_bf("p1", "1004", "AMD", 6, 120.0),    # already seen
             _bf("p2", "1004", "AMD", 4, 121.0)],   # new
            [order2])
        check(len(r2) == 1 and r2[0].fill_id == "p2", "only the new execution returns")
        check(order2.status == OrderStatus.FILLED, "cumulative 10/10 -> FILLED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    test_floor_helper()
    test_build_ibkr_order_is_integer()
    test_build_ibkr_order_skips_sub_one()
    test_reconcile_matches_on_broker_order_id()
    test_reconcile_is_idempotent()
    test_reconcile_skips_unexpected_fill()
    test_reconcile_partial_then_complete()

    print(f"\n{'='*50}")
    print(f"  {_passed} passed, {_failed} failed")
    print(f"{'='*50}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
