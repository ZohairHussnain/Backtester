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
# Phase 2: buying-power pre-flight (option B positions)
# ---------------------------------------------------------------------------

class _FakeBroker:
    def __init__(self, account, positions, open_orders=None):
        self._account = account
        self._positions = positions
        self._open = open_orders or []

    def get_account_state(self):
        return self._account

    def get_positions(self):
        return self._positions

    def get_open_orders(self):
        return self._open


class _FakePortfolio:
    def __init__(self, cash, positions=None):
        self._cash = cash
        self._pos = positions or {}

    @property
    def cash(self):
        return self._cash

    @property
    def equity(self):
        return self._cash + sum(p["shares"] * p["entry_price"] for p in self._pos.values())

    @property
    def open_positions(self):
        return self._pos


def _orders_df(rows):
    import pandas as pd
    cols = ["date", "ticker", "action", "shares", "stop_price", "target_price", "probability"]
    return pd.DataFrame(rows, columns=cols)


def _preflight(broker, portfolio, orders, prices):
    """Run preflight with _latest_close monkeypatched to a price dict."""
    import run_daily
    orig = run_daily._latest_close
    run_daily._latest_close = lambda t: float(prices.get(t, 0.0))
    try:
        return run_daily.preflight_checks(broker, portfolio, orders)
    finally:
        run_daily._latest_close = orig


def test_preflight_huge_broker_equity_not_blocked():
    print("\n[P2-A] broker equity >> local does not block")
    broker = _FakeBroker({"NetLiquidation": 1_000_000, "TotalCashValue": 1_000_000,
                          "BuyingPower": 4_000_000}, positions={})
    pf = _FakePortfolio(cash=10_000)
    orders = _orders_df([["2026-06-09", "AAPL", "BUY", 10.0, 95, 110, 0.7]])
    ok = _preflight(broker, pf, orders, {"AAPL": 150.0})  # notional 1500
    check(ok is True, "passes despite 99% equity gap (buying power ample)")


def test_preflight_insufficient_buying_power():
    print("\n[P2-B] insufficient buying power blocks")
    broker = _FakeBroker({"NetLiquidation": 2_000, "TotalCashValue": 1_000,
                          "BuyingPower": 1_000}, positions={})
    pf = _FakePortfolio(cash=10_000)
    orders = _orders_df([["2026-06-09", "AAPL", "BUY", 100.0, 95, 110, 0.7]])
    ok = _preflight(broker, pf, orders, {"AAPL": 150.0})  # notional 15000 >> 950
    check(ok is False, "blocks when intended notional exceeds 95% of available funds")


def test_preflight_strategy_ticker_mismatch_blocks():
    print("\n[P2-C] share mismatch on an owned ticker blocks")
    broker = _FakeBroker({"NetLiquidation": 1_000_000, "TotalCashValue": 1_000_000,
                          "BuyingPower": 4_000_000},
                         positions={"AAPL": {"shares": 5, "avg_price": 150}})
    pf = _FakePortfolio(cash=9_000, positions={"AAPL": {"shares": 10, "entry_price": 150}})
    orders = _orders_df([])  # no new orders; just reconciling
    ok = _preflight(broker, pf, orders, {})
    check(ok is False, "local 10 vs broker 5 shares on AAPL blocks")


def test_preflight_unrelated_broker_position_ignored():
    print("\n[P2-D] unrelated broker positions are ignored (option B)")
    broker = _FakeBroker({"NetLiquidation": 1_000_000, "TotalCashValue": 1_000_000,
                          "BuyingPower": 4_000_000},
                         positions={"AAPL": {"shares": 10, "avg_price": 150},
                                    "TSLA": {"shares": 99, "avg_price": 200}})  # not ours
    pf = _FakePortfolio(cash=8_500, positions={"AAPL": {"shares": 10, "entry_price": 150}})
    orders = _orders_df([["2026-06-09", "MSFT", "BUY", 5.0, 95, 110, 0.7]])
    ok = _preflight(broker, pf, orders, {"MSFT": 300.0})  # notional 1500
    check(ok is True, "TSLA (broker-only) ignored; owned AAPL matches -> passes")


def test_preflight_strategy_ticker_missing_at_broker_blocks():
    print("\n[P2-E] owned ticker absent at broker blocks")
    broker = _FakeBroker({"NetLiquidation": 1_000_000, "TotalCashValue": 1_000_000,
                          "BuyingPower": 4_000_000}, positions={})
    pf = _FakePortfolio(cash=8_500, positions={"AAPL": {"shares": 10, "entry_price": 150}})
    ok = _preflight(broker, pf, _orders_df([]), {})
    check(ok is False, "strategy holds AAPL but broker has none -> blocks")


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
    test_preflight_huge_broker_equity_not_blocked()
    test_preflight_insufficient_buying_power()
    test_preflight_strategy_ticker_mismatch_blocks()
    test_preflight_unrelated_broker_position_ignored()
    test_preflight_strategy_ticker_missing_at_broker_blocks()

    print(f"\n{'='*50}")
    print(f"  {_passed} passed, {_failed} failed")
    print(f"{'='*50}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
