"""FillReconciler: match broker fills to orders, produce Fill records.

Join key: a broker fill is matched to one of our orders by BROKER order id
(IBKR's execution.orderId == Order.broker_order_id), NOT the internal uuid
order_id. Matching on the internal id can never succeed and silently drops
every fill.

Idempotency: already-processed executions are identified by their broker
execution id (fill_id) recorded in the fills log. Re-running reconciliation is
a safe no-op -- a fill is applied at most once.

Partial fills: a single order may fill across several executions. Order status
is FILLED only once cumulative filled shares (across all runs, read back from
the fills log) reach the ordered quantity; otherwise PARTIALLY_FILLED.
"""

import csv
from pathlib import Path
from typing import Optional

from execution.order import Order, OrderStatus, Fill, Action


class FillReconciler:
    def __init__(self, fills_log_path: Path = None):
        self.log_path = fills_log_path

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(self, broker_fills: list[dict],
                  pending_orders: list[Order]) -> list[Fill]:
        """Match broker fills to orders. Returns only NEW (unseen) Fill objects.

        Fills whose broker order id matches none of pending_orders are reported
        as unexpected and skipped (never applied to the portfolio). Fills whose
        execution id was already processed are skipped (idempotent).
        """
        # Index our orders by the real broker join key.
        order_map: dict[str, Order] = {}
        for o in pending_orders:
            if o.broker_order_id:
                order_map[str(o.broker_order_id)] = o

        seen_fill_ids, cum_by_order_id = self._load_processed()

        new_fills: list[Fill] = []
        new_shares_by_order_id: dict[str, float] = {}

        for bf in broker_fills:
            fill_id = str(bf.get("fill_id", ""))
            if not fill_id:
                print("  WARNING: broker fill with no execution id -- skipping")
                continue
            if fill_id in seen_fill_ids:
                continue  # already applied -- idempotent skip

            broker_id = str(bf.get("order_id", ""))
            order = order_map.get(broker_id)
            if order is None:
                print(f"  WARNING: Unexpected fill {fill_id} for broker order "
                      f"{broker_id} (not one of ours) -- skipping")
                continue

            # Trust our own order's action; IBKR reports side as BOT/SLD which
            # is not a valid Action value.
            fill = Fill(
                fill_id=fill_id,
                order_id=order.order_id,
                ticker=bf.get("ticker", order.ticker),
                action=order.action,
                shares_filled=float(bf.get("shares_filled", order.shares)),
                fill_price=float(bf.get("fill_price", 0) or 0),
                commission=float(bf.get("commission", 0) or 0),
                filled_at=bf.get("filled_at", ""),
            )
            new_fills.append(fill)
            seen_fill_ids.add(fill_id)
            new_shares_by_order_id[order.order_id] = (
                new_shares_by_order_id.get(order.order_id, 0.0) + fill.shares_filled)

        # Update order status from cumulative filled shares (prior runs + this).
        for order in order_map.values():
            filled = (cum_by_order_id.get(order.order_id, 0.0)
                      + new_shares_by_order_id.get(order.order_id, 0.0))
            if filled <= 0:
                continue
            order.filled_at = new_fills[-1].filled_at if new_fills else order.filled_at
            if filled >= order.shares:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

        self.log_fills(new_fills)
        return new_fills

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_processed(self) -> tuple[set, dict]:
        """Read the fills log: set of seen execution ids and cumulative filled
        shares per internal order_id. Empty if the log does not exist yet."""
        seen: set = set()
        cum: dict = {}
        if not self.log_path or not Path(self.log_path).exists():
            return seen, cum
        try:
            with open(self.log_path, newline="") as f:
                for row in csv.DictReader(f):
                    fid = str(row.get("fill_id", ""))
                    if fid:
                        seen.add(fid)
                    oid = str(row.get("order_id", ""))
                    try:
                        cum[oid] = cum.get(oid, 0.0) + float(row.get("shares_filled", 0) or 0)
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            print(f"  WARNING: could not read fills log {self.log_path} ({e}); "
                  f"treating as empty (may risk re-applying fills).")
        return seen, cum

    def log_fills(self, fills: list[Fill]) -> None:
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
