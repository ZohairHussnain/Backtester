"""Order and Fill data structures with lifecycle state machine."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    order_id: str
    date: str
    ticker: str
    action: Action
    shares: float
    order_type: str  # "MOO" (market-on-open)
    stop_price: float
    target_price: float
    status: OrderStatus
    reason: str
    probability: float = 0.0
    rank: int = 0
    created_at: str = ""
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    fill_price: Optional[float] = None
    fill_shares: Optional[float] = None
    broker_order_id: Optional[str] = None
    perm_id: Optional[str] = None
    entry_fee: float = 0.0

    @staticmethod
    def create(date: str, ticker: str, action: Action, shares: float,
               stop_price: float, target_price: float, reason: str,
               probability: float = 0.0, rank: int = 0) -> "Order":
        return Order(
            order_id=str(uuid.uuid4())[:8],
            date=date, ticker=ticker, action=action, shares=shares,
            order_type="MOO", stop_price=stop_price, target_price=target_price,
            status=OrderStatus.PENDING, reason=reason,
            probability=probability, rank=rank,
            created_at=datetime.now().isoformat(),
        )

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id, "date": self.date, "ticker": self.ticker,
            "action": self.action.value, "shares": self.shares,
            "order_type": self.order_type, "stop_price": self.stop_price,
            "target_price": self.target_price, "status": self.status.value,
            "reason": self.reason, "probability": self.probability,
            "rank": self.rank, "created_at": self.created_at,
            "submitted_at": self.submitted_at, "filled_at": self.filled_at,
            "fill_price": self.fill_price, "fill_shares": self.fill_shares,
            "broker_order_id": self.broker_order_id, "perm_id": self.perm_id,
            "entry_fee": self.entry_fee,
        }


@dataclass
class Fill:
    fill_id: str
    order_id: str
    ticker: str
    action: Action
    shares_filled: float
    fill_price: float
    commission: float
    filled_at: str

    def to_dict(self) -> dict:
        return {
            "fill_id": self.fill_id, "order_id": self.order_id,
            "ticker": self.ticker, "action": self.action.value,
            "shares_filled": self.shares_filled, "fill_price": self.fill_price,
            "commission": self.commission, "filled_at": self.filled_at,
        }
