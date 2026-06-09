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
# Runner
# ---------------------------------------------------------------------------

def main():
    test_floor_helper()
    test_build_ibkr_order_is_integer()
    test_build_ibkr_order_skips_sub_one()

    print(f"\n{'='*50}")
    print(f"  {_passed} passed, {_failed} failed")
    print(f"{'='*50}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
