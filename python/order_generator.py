"""Order generation with ranking, sizing, and safety checks."""

import numpy as np
import pandas as pd

from config import (
    THRESHOLD, TOP_N, MAX_POSITIONS,
    RISK_PER_TRADE, MAX_POSITION_FRAC,
    STOP_LOSS_PCT, TARGET_PROFIT_PCT,
    SLIPPAGE, FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT,
    MAX_PORTFOLIO_RISK_PCT,
)
from feature_engine import load_prices


class OrderGenerator:
    def __init__(self):
        self.threshold = THRESHOLD
        self.top_n = TOP_N
        self.max_positions = MAX_POSITIONS

    def generate_orders(self, predictions: pd.DataFrame,
                        portfolio_state: dict,
                        exit_tickers: list[str] = None) -> pd.DataFrame:
        """Generate entry and exit orders.

        Returns DataFrame with columns:
            date, ticker, probability, rank, action, shares,
            entry_type, stop_price, target_price, reason
        """
        orders = []
        today = predictions["date"].max() if len(predictions) > 0 else ""

        # --- Exit orders ---
        for ticker in (exit_tickers or []):
            pos = portfolio_state.get("open_positions", {}).get(ticker)
            if pos:
                orders.append({
                    "date": today, "ticker": ticker, "probability": None,
                    "rank": None, "action": "SELL",
                    "shares": pos["shares"], "entry_type": "MOO",
                    "stop_price": None, "target_price": None,
                    "reason": "max_hold_exit",
                })

        # --- Entry orders ---
        open_positions = portfolio_state.get("open_positions", {})
        # Count positions that will remain after exits
        valid_exits = [t for t in (exit_tickers or []) if t in open_positions]
        n_remaining = len(open_positions) - len(valid_exits)
        slots_available = self.max_positions - max(n_remaining, 0)

        if slots_available <= 0:
            if orders:
                return pd.DataFrame(orders)
            return pd.DataFrame(columns=[
                "date", "ticker", "probability", "rank", "action",
                "shares", "entry_type", "stop_price", "target_price", "reason",
            ])

        # Filter and rank
        candidates = predictions[predictions["probability"] >= self.threshold].copy()
        candidates = candidates[~candidates["ticker"].isin(open_positions.keys())]

        if candidates.empty:
            if orders:
                return pd.DataFrame(orders)
            return pd.DataFrame(columns=[
                "date", "ticker", "probability", "rank", "action",
                "shares", "entry_type", "stop_price", "target_price", "reason",
            ])

        candidates = candidates.nlargest(self.top_n, "probability")

        cash = portfolio_state.get("cash", 0)
        equity = cash
        for _, pos in open_positions.items():
            equity += pos["shares"] * pos["entry_price"]

        for rank, (_, cand) in enumerate(candidates.iterrows(), 1):
            if len([o for o in orders if o["action"] == "BUY"]) >= slots_available:
                break

            ticker = cand["ticker"]
            prob = cand["probability"]

            # Safety checks
            if prob != prob:  # NaN
                continue
            if ticker in open_positions:
                continue

            # Get latest close for entry price estimate
            try:
                prices = load_prices(ticker)
                latest_close = prices.iloc[-1]["close"]
            except Exception:
                continue

            if latest_close <= 0 or latest_close != latest_close:
                continue

            entry_price = latest_close * (1 + SLIPPAGE)
            stop_price = entry_price * (1 - STOP_LOSS_PCT)
            target_price = entry_price * (1 + TARGET_PROFIT_PCT)

            shares = self._size_position(entry_price, equity, cash)
            if shares <= 0:
                print(f"    SKIP {ticker}: position sized to zero (equity=${equity:.2f}, cash=${cash:.2f})")
                continue

            # Portfolio risk check
            position_risk = shares * entry_price * STOP_LOSS_PCT
            if position_risk > equity * MAX_PORTFOLIO_RISK_PCT:
                shares = (equity * MAX_PORTFOLIO_RISK_PCT) / (entry_price * STOP_LOSS_PCT)

            fee = max(min(shares * FEE_PER_SHARE, FEE_CAP_PCT * shares * entry_price), FEE_MIN)
            cost = shares * entry_price + fee
            if cost > cash * MAX_POSITION_FRAC:
                continue

            orders.append({
                "date": today, "ticker": ticker, "probability": round(prob, 6),
                "rank": rank, "action": "BUY",
                "shares": round(shares, 4), "entry_type": "MOO",
                "stop_price": round(stop_price, 2),
                "target_price": round(target_price, 2),
                "reason": f"signal_prob={prob:.4f}_rank={rank}",
            })
            cash -= cost  # reduce available cash for next position

        return pd.DataFrame(orders)

    def _size_position(self, entry_price: float, equity: float,
                       cash: float) -> float:
        risk_dollars = equity * RISK_PER_TRADE
        stop_distance = entry_price * STOP_LOSS_PCT
        if stop_distance <= 0:
            return 0.0
        shares_by_risk = risk_dollars / stop_distance
        shares_by_cash = (cash * MAX_POSITION_FRAC) / entry_price
        return min(shares_by_risk, shares_by_cash)
