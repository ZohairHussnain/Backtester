"""BrokerAdapter: abstract interface + SimulatedBroker for replay."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import uuid

from execution.order import Order, OrderStatus, Action, Fill

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SLIPPAGE, FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT


class BrokerAdapter(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit order, return broker_order_id."""

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order."""

    @abstractmethod
    def get_fills(self, since: Optional[str] = None) -> list[dict]:
        """Get fills since a timestamp."""

    @abstractmethod
    def get_positions(self) -> dict[str, dict]:
        """Get current broker positions."""

    @abstractmethod
    def get_account_state(self) -> dict:
        """Get cash, equity, buying power."""


class SimulatedBroker(BrokerAdapter):
    """Fills orders immediately at a given price + slippage.

    For replay, the caller provides the fill price (typically the entry_price
    from the order, which already includes slippage from the normalized 100.0
    base price).
    """

    def __init__(self):
        self.pending_fills: list[Fill] = []
        self._submitted: dict[str, Order] = {}

    def submit_order(self, order: Order) -> str:
        """Simulate immediate fill at order's entry price."""
        broker_id = f"SIM-{uuid.uuid4().hex[:6]}"
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now().isoformat()
        order.broker_order_id = broker_id
        self._submitted[broker_id] = order

        # Immediate fill
        fill_price = order.stop_price  # placeholder, overridden by caller
        if order.action == Action.BUY:
            # Entry price already computed by OrderManager (latest_close * (1+slippage))
            fill_price = order.shares  # we need to pass price externally
            # Actually: for the normalized simulation, fill_price = entry_price
            # which is set externally. We just create the fill.
            fill_price = 100.0 * (1 + SLIPPAGE)  # normalized entry
        else:
            fill_price = 100.0  # exits use forward_return, handled by caller

        commission = max(
            min(order.shares * FEE_PER_SHARE, FEE_CAP_PCT * order.shares * fill_price),
            FEE_MIN)

        fill = Fill(
            fill_id=f"FILL-{uuid.uuid4().hex[:6]}",
            order_id=order.order_id,
            ticker=order.ticker,
            action=order.action,
            shares_filled=order.shares,
            fill_price=fill_price,
            commission=commission,
            filled_at=datetime.now().isoformat(),
        )
        self.pending_fills.append(fill)

        order.status = OrderStatus.FILLED
        order.filled_at = fill.filled_at
        order.fill_price = fill.fill_price
        order.fill_shares = fill.shares_filled
        return broker_id

    def get_fills(self, since=None) -> list[dict]:
        fills = [f.to_dict() for f in self.pending_fills]
        self.pending_fills.clear()
        return fills

    def cancel_order(self, broker_order_id: str) -> bool:
        if broker_order_id in self._submitted:
            self._submitted[broker_order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_positions(self) -> dict:
        return {}

    def get_account_state(self) -> dict:
        return {}
