"""Tests for ibkr_sync: snapshot shaping, position diffing, and rebuilding the
local paper ledger from broker truth.

Zero-dependency assert harness (mirrors test_ibkr_execution / test_state_isolation).
Run:  python test_ibkr_sync.py

Never connects to IBKR. Uses a fake broker dict and a temp paper-state file; the
real output/ state is never read or written.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ibkr_sync
from portfolio import Portfolio

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


def _temp_portfolio(positions=None, cash=10000.0, trade_history=None):
    """A Portfolio backed by a temp file, seeded with given positions/history."""
    tmp = Path(tempfile.mkdtemp()) / "portfolio_state.paper.json"
    p = Portfolio(tmp, cash)
    p.state["cash"] = cash
    p.state["open_positions"] = positions or {}
    p.state["trade_history"] = trade_history or []
    return p


# ---------------------------------------------------------------------------
# build_snapshot
# ---------------------------------------------------------------------------

def test_build_snapshot_shape():
    print("\n[S-A] build_snapshot shapes broker data")
    snap = ibkr_sync.build_snapshot(
        open_orders=[{"ticker": "AAPL", "action": "BUY", "shares": 10,
                      "order_type": "MOO", "broker_order_id": "5", "status": "Submitted"}],
        positions={"MSFT": {"shares": 8.0, "avg_price": 420.5}},
        account={"NetLiquidation": 1_000_000.0, "BuyingPower": 4_000_000.0},
        fills=[{"ticker": "MSFT", "action": "BUY", "shares_filled": 8,
                "fill_price": 420.5, "filled_at": "2026-06-10T14:30:00"}],
        captured_at="2026-06-10T15:00:00",
    )
    check(snap["source"] == "ibkr_paper", "source tagged ibkr_paper")
    check(snap["captured_at"] == "2026-06-10T15:00:00", "captured_at preserved")
    check(snap["positions"]["MSFT"]["shares"] == 8.0, "position shares carried")
    check(snap["positions"]["MSFT"]["avg_price"] == 420.5, "position avg_price carried")
    check(len(snap["open_orders"]) == 1, "open order carried")
    check(len(snap["recent_fills"]) == 1, "fill carried")


# ---------------------------------------------------------------------------
# diff_positions
# ---------------------------------------------------------------------------

def test_diff_in_sync():
    print("\n[S-B] diff: identical positions are in sync")
    local = {"AAPL": {"shares": 10.0}, "MSFT": {"shares": 5.0}}
    broker = {"AAPL": {"shares": 10.0}, "MSFT": {"shares": 5.0}}
    d = ibkr_sync.diff_positions(local, broker)
    check(set(d["matched"]) == {"AAPL", "MSFT"}, "both matched")
    check(not d["only_local"] and not d["only_broker"], "no exclusives")
    check(not d["share_mismatch"], "no share mismatch")


def test_diff_only_broker():
    print("\n[S-C] diff: position opened on another machine shows only_broker")
    local = {"AAPL": {"shares": 10.0}}
    broker = {"AAPL": {"shares": 10.0}, "NVDA": {"shares": 3.0}}
    d = ibkr_sync.diff_positions(local, broker)
    check(d["only_broker"] == ["NVDA"], "NVDA seen only at broker")
    check(d["matched"] == ["AAPL"], "AAPL matched")


def test_diff_only_local():
    print("\n[S-D] diff: position closed elsewhere shows only_local")
    local = {"AAPL": {"shares": 10.0}, "TSLA": {"shares": 4.0}}
    broker = {"AAPL": {"shares": 10.0}}
    d = ibkr_sync.diff_positions(local, broker)
    check(d["only_local"] == ["TSLA"], "TSLA only local (closed at broker)")


def test_diff_share_mismatch():
    print("\n[S-E] diff: differing share counts are flagged")
    local = {"AAPL": {"shares": 10.0}}
    broker = {"AAPL": {"shares": 7.0}}
    d = ibkr_sync.diff_positions(local, broker)
    check(len(d["share_mismatch"]) == 1, "one mismatch")
    m = d["share_mismatch"][0]
    check(m["ticker"] == "AAPL" and m["local_shares"] == 10.0
          and m["broker_shares"] == 7.0, "mismatch records both counts")
    check(not d["matched"], "not counted as matched")


# ---------------------------------------------------------------------------
# apply_broker_positions
# ---------------------------------------------------------------------------

def test_apply_adds_broker_only_position():
    print("\n[S-F] apply: broker-only position is added to local ledger")
    p = _temp_portfolio(positions={})
    broker = {"NVDA": {"shares": 3.0, "avg_price": 100.0}}
    summary = ibkr_sync.apply_broker_positions(p, broker, "2026-06-10")
    check("NVDA" in p.open_positions, "NVDA added")
    check(p.open_positions["NVDA"]["shares"] == 3.0, "shares from broker")
    check(p.open_positions["NVDA"]["entry_price"] == 100.0, "entry_price = broker avg")
    check(summary["added"] == ["NVDA"], "summary records add")
    # cash = CAP + realized(0) - cost(300)
    check(abs(p.cash - (ibkr_sync.IBKR_PAPER_STRATEGY_CAPITAL - 300.0)) < 1e-6,
          "cash = capital - cost basis")


def test_apply_removes_stale_local_position():
    print("\n[S-G] apply: local position not at broker is removed")
    p = _temp_portfolio(positions={
        "TSLA": {"shares": 4.0, "entry_price": 200.0, "entry_date": "2026-06-01",
                 "stop_price": 190.0, "target_price": 220.0, "entry_fee": 1.0},
    })
    broker = {}  # broker holds nothing
    summary = ibkr_sync.apply_broker_positions(p, broker, "2026-06-10")
    check("TSLA" not in p.open_positions, "TSLA removed")
    check(summary["removed"] == ["TSLA"], "summary records removal")
    check(abs(p.cash - ibkr_sync.IBKR_PAPER_STRATEGY_CAPITAL) < 1e-6,
          "cash back to full capital (no open cost, no realized)")


def test_apply_preserves_local_metadata():
    print("\n[S-H] apply: stop/target/entry_date kept for tickers held by both")
    p = _temp_portfolio(positions={
        "AAPL": {"shares": 10.0, "entry_price": 150.0, "entry_date": "2026-06-02",
                 "stop_price": 142.0, "target_price": 165.0, "entry_fee": 2.0},
    })
    # Broker reports a corrected share count + avg cost for the same ticker.
    broker = {"AAPL": {"shares": 12.0, "avg_price": 151.0}}
    ibkr_sync.apply_broker_positions(p, broker, "2026-06-10")
    pos = p.open_positions["AAPL"]
    check(pos["shares"] == 12.0, "share count corrected from broker")
    check(pos["entry_price"] == 151.0, "avg cost corrected from broker")
    check(pos["entry_date"] == "2026-06-02", "entry_date preserved")
    check(pos["stop_price"] == 142.0, "stop_price preserved")
    check(pos["target_price"] == 165.0, "target_price preserved")


def test_apply_cash_includes_realized_pnl():
    print("\n[S-I] apply: derived cash includes local realized P&L")
    p = _temp_portfolio(
        positions={},
        trade_history=[{"ticker": "X", "pnl": 250.0}, {"ticker": "Y", "pnl": -50.0}],
    )
    broker = {"AAPL": {"shares": 2.0, "avg_price": 100.0}}
    summary = ibkr_sync.apply_broker_positions(p, broker, "2026-06-10")
    check(abs(summary["realized_pnl"] - 200.0) < 1e-6, "realized P&L summed (250-50)")
    # cash = CAP + 200 - 200(cost)
    expected = ibkr_sync.IBKR_PAPER_STRATEGY_CAPITAL + 200.0 - 200.0
    check(abs(p.cash - expected) < 1e-6, "cash = capital + realized - cost basis")
    # equity self-consistency: cash + cost basis == capital + realized
    equity = p.cash + 2.0 * 100.0
    check(abs(equity - (ibkr_sync.IBKR_PAPER_STRATEGY_CAPITAL + 200.0)) < 1e-6,
          "equity == capital + realized P&L")


def test_apply_ignores_flat_broker_position():
    print("\n[S-J] apply: a zero-share broker position is not added")
    p = _temp_portfolio(positions={})
    broker = {"ZERO": {"shares": 0.0, "avg_price": 0.0},
              "AAPL": {"shares": 5.0, "avg_price": 100.0}}
    ibkr_sync.apply_broker_positions(p, broker, "2026-06-10")
    check("ZERO" not in p.open_positions, "flat position skipped")
    check("AAPL" in p.open_positions, "non-flat position kept")


def test_apply_save_roundtrip():
    print("\n[S-K] apply: saved paper state reloads with synced positions")
    p = _temp_portfolio(positions={})
    broker = {"AAPL": {"shares": 5.0, "avg_price": 100.0}}
    ibkr_sync.apply_broker_positions(p, broker, "2026-06-10")
    p.save()
    reloaded = Portfolio(p.state_file, ibkr_sync.IBKR_PAPER_STRATEGY_CAPITAL)
    check("AAPL" in reloaded.open_positions, "position persisted")
    check(reloaded.open_positions["AAPL"]["shares"] == 5.0, "shares persisted")
    check(abs(reloaded.cash - p.cash) < 1e-6, "cash persisted")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    test_build_snapshot_shape()
    test_diff_in_sync()
    test_diff_only_broker()
    test_diff_only_local()
    test_diff_share_mismatch()
    test_apply_adds_broker_only_position()
    test_apply_removes_stale_local_position()
    test_apply_preserves_local_metadata()
    test_apply_cash_includes_realized_pnl()
    test_apply_ignores_flat_broker_position()
    test_apply_save_roundtrip()

    print(f"\n{'='*50}")
    print(f"  {_passed} passed, {_failed} failed")
    print(f"{'='*50}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
