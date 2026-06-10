"""
ibkr_sync.py -- pull live broker state from IBKR paper and (optionally) sync the
local strategy ledger to it.

WHY THIS EXISTS
---------------
The strategy is run from more than one machine against the SAME IBKR paper
account. Each machine keeps its own local ledger (portfolio_state.paper.json)
plus per-machine logs (orders_lifecycle.csv, fills.csv). When laptop A submits
orders and they fill, laptop B's local ledger knows nothing about it.

`run_daily.py --reconcile-only` does NOT fix this: it matches broker fills
against the LOCAL orders_lifecycle.csv (by perm_id / broker_order_id). Laptop B
never recorded laptop A's orders, so those fills are "unmatched" and skipped.

The IBKR account itself is the single shared source of truth. This tool pulls
that truth -- open orders, positions, account summary, recent fills -- so both
machines can see the same picture, and can rebuild their local paper ledger from
the broker's authoritative positions.

SAFETY
------
- Connects through IBKRBroker(dry_run=True), which opens a READ-ONLY API session
  on the hardcoded paper port (4002). This tool can never place or cancel an
  order; it only reads.
- The default action is read-only: print a snapshot, write output/ibkr_snapshot.json,
  and show how local paper state differs from the broker. Nothing is mutated.
- `--apply` rewrites portfolio_state.paper.json open positions to match the
  broker. It requires `--confirm`, makes an atomic backed-up save, and never
  touches sim state.

USAGE
-----
    python ibkr_sync.py                  # pull broker truth, print + diff vs local
    python ibkr_sync.py --json           # same, machine-readable JSON only
    python ibkr_sync.py --apply --confirm # rebuild local paper positions from broker
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PORTFOLIO_STATE_FILE_PAPER,
    IBKR_PAPER_STRATEGY_CAPITAL,
    OUTPUT_DIR,
)
from portfolio import Portfolio

SNAPSHOT_FILE = OUTPUT_DIR / "ibkr_snapshot.json"

# Share-count tolerance below which two positions are considered identical.
SHARE_EPS = 1e-6


# ======================================================================
# Pure helpers (no IBKR connection -- unit-testable with a fake broker)
# ======================================================================

def build_snapshot(open_orders: list, positions: dict, account: dict,
                   fills: list, captured_at: str) -> dict:
    """Assemble a plain-dict snapshot of broker truth.

    Pure function: takes already-fetched broker data and shapes it into the
    structure written to ibkr_snapshot.json. Keeping this separate from the live
    fetch makes it testable without an IBKR connection.
    """
    return {
        "captured_at": captured_at,
        "source": "ibkr_paper",
        "account": dict(account or {}),
        "positions": {
            ticker: {
                "shares": float(p.get("shares", 0)),
                "avg_price": float(p.get("avg_price", 0)),
            }
            for ticker, p in (positions or {}).items()
        },
        "open_orders": list(open_orders or []),
        "recent_fills": list(fills or []),
    }


def diff_positions(local_positions: dict, broker_positions: dict) -> dict:
    """Compare local ledger positions against broker positions.

    Returns a structured diff:
      - only_local:  tickers the strategy thinks it holds but the broker does not
                     (e.g. a fill on another laptop that closed the position).
      - only_broker: tickers the broker holds but this ledger does not
                     (e.g. an entry submitted from another laptop).
      - share_mismatch: tickers held by both but with different share counts.
      - matched:     tickers that agree within SHARE_EPS.
    """
    local = local_positions or {}
    broker = broker_positions or {}
    local_t = set(local.keys())
    broker_t = set(broker.keys())

    share_mismatch = []
    matched = []
    for ticker in sorted(local_t & broker_t):
        ls = float(local[ticker].get("shares", 0))
        bs = float(broker[ticker].get("shares", 0))
        if abs(ls - bs) > SHARE_EPS:
            share_mismatch.append({
                "ticker": ticker, "local_shares": ls, "broker_shares": bs,
            })
        else:
            matched.append(ticker)

    return {
        "only_local": sorted(local_t - broker_t),
        "only_broker": sorted(broker_t - local_t),
        "share_mismatch": share_mismatch,
        "matched": matched,
    }


def realized_pnl(portfolio: Portfolio) -> float:
    """Sum realized P&L from the local trade history."""
    total = 0.0
    for trade in portfolio.get_state().get("trade_history", []):
        try:
            total += float(trade.get("pnl", 0))
        except (TypeError, ValueError):
            continue
    return total


def apply_broker_positions(portfolio: Portfolio, broker_positions: dict,
                           default_date: str) -> dict:
    """Rebuild the local paper ledger's open positions from broker truth.

    The broker is authoritative for WHAT is held and at WHAT average cost. It
    does NOT know the strategy's stop/target or original entry date, so for a
    ticker already tracked locally we PRESERVE those fields and only correct the
    share count and average cost. Broker-only positions are added with the run
    date and zero stop/target (the operator should review them).

    Cash is reconstructed from the strategy capital envelope, NOT read from the
    broker (the paper account is auto-funded at ~$1M and is shared with unrelated
    funds). We keep the ledger self-consistent:

        equity == STRATEGY_CAPITAL + realized_pnl
        cash   == STRATEGY_CAPITAL + realized_pnl - cost_basis_of_open_positions

    so that marking open positions at cost reproduces capital + realized gains.

    Returns a summary dict of what changed. Does NOT save; the caller decides.
    """
    state = portfolio.get_state()
    old_positions = dict(state.get("open_positions", {}))
    new_positions = {}
    cost_basis = 0.0

    for ticker, bp in (broker_positions or {}).items():
        shares = float(bp.get("shares", 0))
        if abs(shares) <= SHARE_EPS:
            continue  # broker reports flat -> not an open position
        avg_price = float(bp.get("avg_price", 0))
        prev = old_positions.get(ticker, {})
        new_positions[ticker] = {
            "shares": shares,
            "entry_price": avg_price,
            "entry_date": prev.get("entry_date", default_date),
            "stop_price": prev.get("stop_price", 0),
            "target_price": prev.get("target_price", 0),
            "entry_fee": prev.get("entry_fee", 0.0),
        }
        cost_basis += shares * avg_price

    realized = realized_pnl(portfolio)
    new_cash = IBKR_PAPER_STRATEGY_CAPITAL + realized - cost_basis

    state["open_positions"] = new_positions
    state["cash"] = new_cash

    return {
        "removed": sorted(set(old_positions) - set(new_positions)),
        "added": sorted(set(new_positions) - set(old_positions)),
        "kept": sorted(set(new_positions) & set(old_positions)),
        "realized_pnl": realized,
        "cost_basis": cost_basis,
        "new_cash": new_cash,
    }


# ======================================================================
# Printing
# ======================================================================

def print_snapshot(snapshot: dict) -> None:
    acct = snapshot.get("account", {})
    positions = snapshot.get("positions", {})
    open_orders = snapshot.get("open_orders", [])
    fills = snapshot.get("recent_fills", [])

    print("=" * 64)
    print(f"  IBKR PAPER SNAPSHOT -- {snapshot.get('captured_at', '')}")
    print("=" * 64)

    print("\n  Account:")
    if acct:
        for tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                    "GrossPositionValue", "MaintMarginReq"):
            if tag in acct:
                print(f"    {tag:<18} ${acct[tag]:>14,.2f}")
    else:
        print("    (unavailable)")

    print(f"\n  Positions ({len(positions)}):")
    if positions:
        print(f"    {'Ticker':<8} {'Shares':>12} {'AvgCost':>12}")
        for ticker in sorted(positions):
            p = positions[ticker]
            print(f"    {ticker:<8} {p['shares']:>12.4f} ${p['avg_price']:>11,.2f}")
    else:
        print("    (none)")

    print(f"\n  Open orders ({len(open_orders)}):")
    if open_orders:
        for oo in open_orders:
            print(f"    {oo.get('action',''):<5} {oo.get('shares','')} "
                  f"{oo.get('ticker',''):<8} {oo.get('order_type',''):<6} "
                  f"id={oo.get('broker_order_id','')} ({oo.get('status','')})")
    else:
        print("    (none)")

    print(f"\n  Recent fills ({len(fills)}):")
    if fills:
        for f in fills[:25]:
            print(f"    {f.get('filled_at',''):<26} {f.get('action',''):<5} "
                  f"{f.get('shares_filled','')} {f.get('ticker',''):<8} "
                  f"@ ${float(f.get('fill_price',0)):,.2f}")
        if len(fills) > 25:
            print(f"    ... and {len(fills) - 25} more")
    else:
        print("    (none)")


def print_diff(diff: dict) -> None:
    print("\n" + "-" * 64)
    print("  LOCAL PAPER LEDGER vs BROKER")
    print("-" * 64)

    if diff["matched"]:
        print(f"  In sync ({len(diff['matched'])}): {', '.join(diff['matched'])}")

    out_of_sync = (diff["only_local"] or diff["only_broker"]
                   or diff["share_mismatch"])
    if not out_of_sync:
        print("  OK: local paper ledger matches the broker. Nothing to sync.")
        return

    if diff["only_broker"]:
        print(f"\n  At broker but NOT in local ledger (likely traded on another "
              f"machine):")
        for t in diff["only_broker"]:
            print(f"    + {t}")
    if diff["only_local"]:
        print(f"\n  In local ledger but NOT at broker (closed elsewhere, or never "
              f"filled):")
        for t in diff["only_local"]:
            print(f"    - {t}")
    if diff["share_mismatch"]:
        print(f"\n  Share-count mismatches:")
        for m in diff["share_mismatch"]:
            print(f"    ~ {m['ticker']}: local {m['local_shares']:.4f} "
                  f"vs broker {m['broker_shares']:.4f}")

    print("\n  To rebuild this machine's paper ledger from the broker, run:")
    print("    python ibkr_sync.py --apply --confirm")


# ======================================================================
# Live fetch + CLI
# ======================================================================

def fetch_broker_snapshot(captured_at: str) -> dict:
    """Connect read-only to IBKR paper and pull a full snapshot. Returns None on
    connection failure (the caller reports it)."""
    from execution.ibkr_broker import IBKRBroker

    broker = IBKRBroker(dry_run=True)  # dry_run=True => readonly API session
    try:
        broker.connect()
    except Exception as e:
        print(f"  ERROR: could not connect to IBKR paper ({e}).")
        print(f"  Is IBKR Gateway/TWS running with the API enabled on the paper port?")
        return None

    try:
        open_orders = _safe(broker.get_open_orders, [])
        positions = _safe(broker.get_positions, {})
        account = _safe(broker.get_account_state, {})
        fills = _safe(broker.get_fills, [])
    finally:
        broker.disconnect()

    return build_snapshot(open_orders, positions, account, fills, captured_at)


def _safe(fn, default):
    try:
        return fn()
    except Exception as e:
        print(f"  WARNING: {getattr(fn, '__name__', 'fetch')} failed ({e}).")
        return default


def write_snapshot(snapshot: dict, path: Path = SNAPSHOT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    tmp.replace(path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Pull IBKR paper broker state and sync the local paper ledger.")
    p.add_argument("--apply", action="store_true",
                   help="Rebuild local paper positions from broker truth "
                        "(requires --confirm).")
    p.add_argument("--confirm", action="store_true",
                   help="Confirm the --apply write to portfolio_state.paper.json.")
    p.add_argument("--json", action="store_true",
                   help="Print the snapshot as JSON only (still writes the file).")
    return p.parse_args()


def main():
    args = parse_args()
    captured_at = datetime.now().isoformat()

    snapshot = fetch_broker_snapshot(captured_at)
    if snapshot is None:
        sys.exit(1)

    write_snapshot(snapshot)

    if args.json:
        print(json.dumps(snapshot, indent=2, default=str))
    else:
        print_snapshot(snapshot)
        print(f"\n  Snapshot written to {SNAPSHOT_FILE}")

    # Compare against the local paper ledger.
    portfolio = Portfolio(PORTFOLIO_STATE_FILE_PAPER, IBKR_PAPER_STRATEGY_CAPITAL)
    diff = diff_positions(portfolio.open_positions, snapshot["positions"])

    if not args.json:
        print_diff(diff)

    # Optional: rebuild local paper positions from the broker.
    if args.apply:
        if not args.confirm:
            print("\n  --apply requires --confirm. No changes made.")
            sys.exit(1)
        summary = apply_broker_positions(
            portfolio, snapshot["positions"], captured_at[:10])
        portfolio.save()
        print("\n  APPLIED broker positions to local paper ledger:")
        if summary["added"]:
            print(f"    added:   {', '.join(summary['added'])}")
        if summary["removed"]:
            print(f"    removed: {', '.join(summary['removed'])}")
        if summary["kept"]:
            print(f"    updated: {', '.join(summary['kept'])}")
        print(f"    realized P&L (local history): ${summary['realized_pnl']:,.2f}")
        print(f"    open cost basis:              ${summary['cost_basis']:,.2f}")
        print(f"    cash set to:                  ${summary['new_cash']:,.2f}")
        print(f"\n  Saved {PORTFOLIO_STATE_FILE_PAPER.name}.")
        print("  NOTE: stop/target/entry-date for broker-only positions default to")
        print("  today / 0. Review them before the next run if exit logic matters.")


if __name__ == "__main__":
    main()
