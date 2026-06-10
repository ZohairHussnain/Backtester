"""FillReconciler: match broker fills to orders, produce Fill records.

Matching strategy (in priority order), because IBKR's per-session orderId is
unreliable (often 0 for executions queried in a later API session):
  1. perm_id        — IBKR's stable cross-session order id
  2. broker_order_id — the orderId captured at submit time
  3. (ticker, action) — last-resort match against a single pending order

Idempotency is NOT owned here. The reconciler only de-duplicates identical
execution ids WITHIN a single fetch. Cross-run idempotency is owned by the
Portfolio (processed_fill_ids, committed atomically with state), so a crash can
never leave a fill marked "seen" but unapplied. Callers must apply fills, save
the portfolio, and only then call log_fills() to append the audit record.
"""

import csv
from pathlib import Path
from typing import Optional

from execution.order import Order, OrderStatus, Fill, Action


class FillReconciler:
    def __init__(self, fills_log_path: Path = None):
        self.log_path = fills_log_path

    def _match_order(self, bf: dict, pending_orders: list[Order]) -> Optional[Order]:
        """Find the local Order a broker fill belongs to."""
        perm_id = str(bf.get("perm_id") or "")
        order_id = str(bf.get("order_id") or "")

        if perm_id:
            for o in pending_orders:
                if o.perm_id and str(o.perm_id) == perm_id:
                    return o
        if order_id and order_id != "0":
            for o in pending_orders:
                if o.broker_order_id and str(o.broker_order_id) == order_id:
                    return o
        # Last resort: a single pending order matching ticker + action.
        ticker = bf.get("ticker")
        action = bf.get("action")
        candidates = [o for o in pending_orders
                      if o.ticker == ticker and o.action.value == action]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def reconcile(self, broker_fills: list[dict],
                  pending_orders: list[Order]) -> list[Fill]:
        """Match broker fills to orders. Returns confirmed Fill objects.

        - De-duplicates identical execution ids within this fetch.
        - Accumulates multiple executions against one order (partial fills):
          an order is FILLED only once cumulative shares >= order.shares.
        Does NOT write any log or mutate any ledger.
        """
        fills = []
        seen_in_batch = set()
        cumulative = {id(o): 0.0 for o in pending_orders}

        for bf in broker_fills:
            fill_id = bf.get("fill_id", "")
            if fill_id and fill_id in seen_in_batch:
                continue

            order = self._match_order(bf, pending_orders)
            if order is None:
                print(f"  WARNING: Unmatched fill {fill_id} "
                      f"({bf.get('action')} {bf.get('ticker')}) -- skipping")
                continue

            fill = Fill(
                fill_id=fill_id,
                order_id=order.order_id,
                ticker=bf.get("ticker", order.ticker),
                action=Action(bf.get("action", order.action.value)),
                shares_filled=bf.get("shares_filled", order.shares),
                fill_price=bf.get("fill_price", 0),
                commission=bf.get("commission", 0),
                filled_at=bf.get("filled_at", ""),
            )
            fills.append(fill)
            if fill_id:
                seen_in_batch.add(fill_id)

            cumulative[id(order)] += fill.shares_filled
            if cumulative[id(order)] >= order.shares:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            order.filled_at = fill.filled_at
            order.fill_price = fill.fill_price
            order.fill_shares = cumulative[id(order)]

        return fills

    def log_fills(self, fills: list[Fill]) -> None:
        """Append fills to the audit log. Call AFTER the portfolio is saved."""
        if not self.log_path or not fills:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fills[0].to_dict().keys())
            if write_header:
                writer.writeheader()
            for fill in fills:
                writer.writerow(fill.to_dict())
