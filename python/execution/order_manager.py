"""OrderManager: convert signals to orders, enforce limits, track lifecycle."""

import csv
from pathlib import Path

import pandas as pd

from execution.order import Order, OrderStatus, Action, Fill

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    THRESHOLD, TOP_N, MAX_POSITIONS,
    RISK_PER_TRADE, MAX_POSITION_FRAC,
    STOP_LOSS_PCT, TARGET_PROFIT_PCT,
    SLIPPAGE, FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT,
    MAX_PORTFOLIO_RISK_PCT,
)


class OrderManager:
    def __init__(self, orders_log_path: Path = None,
                 threshold: float = None, top_n: int = None, max_positions: int = None):
        self.orders: list[Order] = []
        self.log_path = orders_log_path
        self.threshold = threshold if threshold is not None else THRESHOLD
        self.top_n = top_n if top_n is not None else TOP_N
        self.max_positions = max_positions if max_positions is not None else MAX_POSITIONS

    def generate_orders(self, signals: pd.DataFrame,
                        portfolio_state: dict,
                        exit_tickers: list[str] = None,
                        entry_prices: dict[str, float] = None,
                        equity_override: float = None) -> list[Order]:
        """Convert signals into sized, validated orders.

        entry_prices: ticker -> latest close price for position sizing.
        equity_override: if provided, use this as portfolio equity for risk sizing
                        instead of computing from positions at entry price.
        """
        today = signals["date"].max() if len(signals) > 0 else ""
        orders = []

        # --- Exit orders ---
        open_positions = portfolio_state.get("open_positions", {})
        for ticker in (exit_tickers or []):
            if ticker not in open_positions:
                continue
            pos = open_positions[ticker]
            orders.append(Order.create(
                date=today, ticker=ticker, action=Action.SELL,
                shares=pos["shares"], stop_price=0, target_price=0,
                reason="max_hold_exit",
            ))

        # --- Entry orders ---
        valid_exits = set(t for t in (exit_tickers or []) if t in open_positions)
        n_remaining = len(open_positions) - len(valid_exits)
        slots = self.max_positions - max(n_remaining, 0)
        n_buys = 0

        if slots > 0 and len(signals) > 0:
            candidates = signals[signals["probability"] >= self.threshold].copy()
            candidates = candidates[~candidates["ticker"].isin(open_positions.keys())]
            candidates = candidates.nlargest(self.top_n, "probability")

            cash = portfolio_state.get("cash", 0)
            if equity_override is not None:
                equity = equity_override
            else:
                equity = cash
                for _, pos in open_positions.items():
                    equity += pos["shares"] * pos["entry_price"]

            for rank, (_, cand) in enumerate(candidates.iterrows(), 1):
                if n_buys >= slots:
                    break

                ticker = cand["ticker"]
                prob = cand["probability"]

                if entry_prices and ticker in entry_prices:
                    latest_close = entry_prices[ticker]
                else:
                    continue  # no price, skip

                if latest_close <= 0 or latest_close != latest_close:
                    continue

                entry_price = latest_close * (1 + SLIPPAGE)
                stop_price = entry_price * (1 - STOP_LOSS_PCT)
                target_price = entry_price * (1 + TARGET_PROFIT_PCT)

                # Position sizing
                stop_dist = entry_price * STOP_LOSS_PCT
                shares_risk = (equity * RISK_PER_TRADE) / stop_dist if stop_dist > 0 else 0
                shares_cash = (cash * MAX_POSITION_FRAC) / entry_price
                shares = min(shares_risk, shares_cash)

                if shares <= 0:
                    continue

                # Portfolio risk check
                pos_risk = shares * entry_price * STOP_LOSS_PCT
                if pos_risk > equity * MAX_PORTFOLIO_RISK_PCT:
                    shares = (equity * MAX_PORTFOLIO_RISK_PCT) / (entry_price * STOP_LOSS_PCT)

                fee = max(min(shares * FEE_PER_SHARE, FEE_CAP_PCT * shares * entry_price), FEE_MIN)
                cost = shares * entry_price + fee
                if cost > cash * MAX_POSITION_FRAC or shares <= 0:
                    continue

                order = Order.create(
                    date=today, ticker=ticker, action=Action.BUY,
                    shares=shares, stop_price=stop_price, target_price=target_price,
                    reason=f"signal_prob={prob:.4f}_rank={rank}",
                    probability=prob, rank=rank,
                )
                order.entry_fee = fee
                orders.append(order)
                cash -= cost
                n_buys += 1

        self.orders.extend(orders)
        return orders

    def log_orders(self, orders: list[Order]) -> None:
        """Append orders to the CSV log."""
        if not self.log_path or not orders:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=orders[0].to_dict().keys())
            if write_header:
                writer.writeheader()
            for o in orders:
                writer.writerow(o.to_dict())
