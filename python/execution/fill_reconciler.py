"""FillReconciler: match broker fills to orders, produce Fill records."""

import csv
from pathlib import Path
from typing import Optional

from execution.order import Order, OrderStatus, Fill, Action


class FillReconciler:
    def __init__(self, fills_log_path: Path = None):
        self.log_path = fills_log_path

    def reconcile(self, broker_fills: list[dict],
                  pending_orders: list[Order]) -> list[Fill]:
        """Match broker fills to orders. Returns confirmed Fill objects."""
        fills = []
        order_map = {o.order_id: o for o in pending_orders}

        for bf in broker_fills:
            order_id = bf.get("order_id")
            if not order_id or order_id not in order_map:
                print(f"  WARNING: Unexpected fill for order {order_id} — skipping")
                continue

            order = order_map[order_id]
            fill = Fill(
                fill_id=bf.get("fill_id", ""),
                order_id=order_id,
                ticker=bf.get("ticker", order.ticker),
                action=Action(bf.get("action", order.action.value)),
                shares_filled=bf.get("shares_filled", order.shares),
                fill_price=bf.get("fill_price", 0),
                commission=bf.get("commission", 0),
                filled_at=bf.get("filled_at", ""),
            )
            fills.append(fill)

            # Update order status
            if fill.shares_filled >= order.shares:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            order.filled_at = fill.filled_at
            order.fill_price = fill.fill_price
            order.fill_shares = fill.shares_filled

        self.log_fills(fills)
        return fills

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
