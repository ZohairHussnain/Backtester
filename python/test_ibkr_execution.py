"""IBKR execution tests: preflight (sub-allocation), integer shares,
reconciliation matching, partial fills, idempotency, and crash/re-run safety.

Zero-dependency assert harness (mirrors test_state_isolation.py).
Run:  python test_ibkr_execution.py

Uses temp files and in-memory fakes only. Never connects to IBKR and never
reads or writes the real output/ state files.
"""

import sys
import tempfile
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

    print("\n" + "=" * 60)
    print(f"  RESULTS: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
