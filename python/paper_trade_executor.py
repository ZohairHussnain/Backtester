"""Paper trade executor — dry-run mode only. No live orders."""

import pandas as pd

from config import DRY_RUN, MAX_POSITIONS
from portfolio import Portfolio


class PaperTradeExecutor:
    def __init__(self, dry_run: bool = DRY_RUN):
        self.dry_run = dry_run

    def execute(self, orders: pd.DataFrame, portfolio: Portfolio) -> None:
        if orders.empty:
            print("  No orders to execute.")
            return

        for _, order in orders.iterrows():
            valid, msg = self.validate_order(order, portfolio)
            if not valid:
                print(f"  REJECTED: {order['ticker']} {order['action']} — {msg}")
                continue

            if self.dry_run:
                self._log_dry_run(order)
            else:
                self._execute_live(order, portfolio)

    def validate_order(self, order: dict, portfolio: Portfolio) -> tuple[bool, str]:
        ticker = order.get("ticker", "")
        action = order.get("action", "")
        shares = order.get("shares", 0)

        if not ticker or not isinstance(ticker, str):
            return False, "invalid ticker"
        if action not in ("BUY", "SELL"):
            return False, f"invalid action: {action}"
        if shares is None or shares != shares:  # NaN check
            return False, "invalid shares (NaN)"
        if action == "BUY" and shares <= 0:
            return False, f"non-positive shares: {shares}"
        if action == "BUY" and len(portfolio.open_positions) >= MAX_POSITIONS:
            return False, f"max positions ({MAX_POSITIONS}) reached"
        if action == "BUY" and ticker in portfolio.open_positions:
            return False, f"already holding {ticker}"
        if action == "SELL" and ticker not in portfolio.open_positions:
            return False, f"not holding {ticker}"
        return True, "ok"

    def _log_dry_run(self, order: dict) -> None:
        action = order["action"]
        ticker = order["ticker"]
        shares = order.get("shares", "?")
        prob = order.get("probability", "?")
        reason = order.get("reason", "")

        if action == "BUY":
            stop = order.get("stop_price", "?")
            target = order.get("target_price", "?")
            print(f"  [DRY RUN] BUY {shares:.2f} shares {ticker} "
                  f"(prob={prob}, stop=${stop}, target=${target}) — {reason}")
        else:
            print(f"  [DRY RUN] SELL {shares} shares {ticker} — {reason}")

    def _execute_live(self, order: dict, portfolio: Portfolio) -> None:
        raise NotImplementedError("Live execution not implemented. Set DRY_RUN=True.")
