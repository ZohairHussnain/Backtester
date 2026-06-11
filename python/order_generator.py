"""Order generation with ranking, sizing, and safety checks."""

import numpy as np
import pandas as pd

import math

from config import (
    THRESHOLD, TOP_N, MAX_POSITIONS,
    RISK_PER_TRADE, MAX_POSITION_FRAC,
    STOP_LOSS_PCT, TARGET_PROFIT_PCT,
    SLIPPAGE, FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT,
    MAX_PORTFOLIO_RISK_PCT,
    ROTATION_ENABLED, ROTATION_MIN_PROB_IMPROVEMENT,
    ROTATION_ONLY_WHEN_FULL, ROTATION_MAX_PER_DAY,
)
from feature_engine import load_prices

_EMPTY_COLS = [
    "date", "ticker", "probability", "rank", "action",
    "shares", "entry_type", "stop_price", "target_price", "reason",
]


class OrderGenerator:
    def __init__(self):
        self.threshold = THRESHOLD
        self.top_n = TOP_N
        self.max_positions = MAX_POSITIONS

    def generate_orders(self, predictions: pd.DataFrame,
                        portfolio_state: dict,
                        exit_tickers=None) -> pd.DataFrame:
        """Generate entry and exit orders.

        exit_tickers may be a {ticker: reason} dict (from determine_exits) or a
        plain list of tickers (legacy; defaults the reason to "max_hold_exit").

        Returns DataFrame with columns:
            date, ticker, probability, rank, action, shares,
            entry_type, stop_price, target_price, reason
        """
        orders = []
        today = predictions["date"].max() if len(predictions) > 0 else ""

        # Normalize to {ticker: reason}.
        if isinstance(exit_tickers, dict):
            exit_reasons = dict(exit_tickers)
        else:
            exit_reasons = {t: "max_hold_exit" for t in (exit_tickers or [])}

        open_positions = portfolio_state.get("open_positions", {})

        # --- Exit orders ---
        for ticker, reason in exit_reasons.items():
            pos = open_positions.get(ticker)
            if pos:
                orders.append({
                    "date": today, "ticker": ticker, "probability": None,
                    "rank": None, "action": "SELL",
                    "shares": pos["shares"], "entry_type": "MOO",
                    "stop_price": None, "target_price": None,
                    "reason": reason,
                })

        # Slots remaining after the day's exits.
        valid_exits = [t for t in exit_reasons if t in open_positions]
        n_remaining = len(open_positions) - len(valid_exits)
        slots_available = self.max_positions - max(n_remaining, 0)

        cash = portfolio_state.get("cash", 0)
        equity = cash
        for _, pos in open_positions.items():
            equity += pos["shares"] * pos["entry_price"]

        # Candidate pool (>= threshold, not held), ranked. Shared by rotation
        # and normal entries.
        candidates = predictions[predictions["probability"] >= self.threshold].copy()
        candidates = candidates[~candidates["ticker"].isin(open_positions.keys())]
        candidates = candidates.nlargest(self.top_n, "probability")

        bought = set()       # candidate tickers already BUY'd (avoid double-buy)
        rotated_out = set()  # holdings already SOLD via rotation this run

        # --- Rotation / replacement (optional, gated) ---
        if ROTATION_ENABLED and (slots_available <= 0 or not ROTATION_ONLY_WHEN_FULL):
            cash = self._apply_rotations(
                orders, predictions, candidates, open_positions, exit_reasons,
                bought, rotated_out, today, equity, cash)

        # --- Normal entry orders ---
        if slots_available > 0 and not candidates.empty:
            n_buys = 0
            for rank, (_, cand) in enumerate(candidates.iterrows(), 1):
                if n_buys >= slots_available:
                    break
                ticker = cand["ticker"]
                prob = cand["probability"]
                if prob != prob:  # NaN
                    continue
                if ticker in open_positions or ticker in bought:
                    continue
                built = self._build_buy_order(ticker, prob, rank, today, equity, cash)
                if built is None:
                    continue
                order, cost = built
                orders.append(order)
                cash -= cost  # reduce available cash for next position
                n_buys += 1

        if orders:
            return pd.DataFrame(orders)
        return pd.DataFrame(columns=_EMPTY_COLS)

    def _build_buy_order(self, ticker, prob, rank, today, equity, avail_cash,
                         require_whole_share: bool = False):
        """Build a sized BUY order dict + its cash cost, or None if it can't be
        placed safely.

        avail_cash bounds both the position size and the per-order cash cap. When
        require_whole_share is True, an order that floors to < 1 share is rejected
        (IBKR-safe) — used by rotation so it never emits a BUY that can't pair
        with its SELL.
        """
        try:
            prices = load_prices(ticker)
            latest_close = prices.iloc[-1]["close"]
        except Exception:
            return None

        if latest_close <= 0 or latest_close != latest_close:
            return None

        entry_price = latest_close * (1 + SLIPPAGE)
        stop_price = entry_price * (1 - STOP_LOSS_PCT)
        target_price = entry_price * (1 + TARGET_PROFIT_PCT)

        shares = self._size_position(entry_price, equity, avail_cash)
        if shares <= 0:
            print(f"    SKIP {ticker}: position sized to zero "
                  f"(equity=${equity:.2f}, cash=${avail_cash:.2f})")
            return None

        # Portfolio risk check
        position_risk = shares * entry_price * STOP_LOSS_PCT
        if position_risk > equity * MAX_PORTFOLIO_RISK_PCT:
            shares = (equity * MAX_PORTFOLIO_RISK_PCT) / (entry_price * STOP_LOSS_PCT)

        fee = max(min(shares * FEE_PER_SHARE, FEE_CAP_PCT * shares * entry_price), FEE_MIN)
        cost = shares * entry_price + fee
        if cost > avail_cash * MAX_POSITION_FRAC:
            return None
        if require_whole_share and math.floor(shares) < 1:
            return None

        order = {
            "date": today, "ticker": ticker, "probability": round(prob, 6),
            "rank": rank, "action": "BUY",
            "shares": round(shares, 4), "entry_type": "MOO",
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "reason": f"signal_prob={prob:.4f}_rank={rank}",
        }
        return order, cost

    def _latest_close_safe(self, ticker) -> float:
        try:
            return float(load_prices(ticker).iloc[-1]["close"])
        except Exception:
            return 0.0

    def _apply_rotations(self, orders, predictions, candidates, open_positions,
                         exit_reasons, bought, rotated_out, today, equity, cash):
        """Replace the weakest holding(s) with materially stronger candidates.

        For each candidate (highest probability first), find the weakest ELIGIBLE
        holding by today's prediction probability and, if the candidate beats it
        by >= ROTATION_MIN_PROB_IMPROVEMENT, emit a SELL (reason="rotation_exit")
        for the holding and a BUY for the candidate — but only if the replacement
        BUY can be built safely (atomic: both orders or neither, never a naked
        SELL). Eligible holdings exclude any ticker already exiting today
        (stop/target/max-hold), any already rotated out, and — conservatively —
        any holding with no prediction row today (a missing signal must not force
        a sale). Capped at ROTATION_MAX_PER_DAY. Returns the updated cash.
        """
        pred_prob = dict(zip(predictions["ticker"], predictions["probability"]))
        rotations_done = 0

        for rank, (_, cand) in enumerate(candidates.iterrows(), 1):
            if rotations_done >= ROTATION_MAX_PER_DAY:
                break
            cand_ticker = cand["ticker"]
            p_new = cand["probability"]
            if p_new != p_new:  # NaN
                continue
            if cand_ticker in open_positions or cand_ticker in bought:
                continue

            # Weakest eligible holding by today's probability.
            weakest, p_weak = None, None
            for held in open_positions:
                if held in exit_reasons or held in rotated_out:
                    continue
                hp = pred_prob.get(held)
                if hp is None or hp != hp:  # no prediction / NaN -> exclude
                    continue
                if p_weak is None or hp < p_weak:
                    weakest, p_weak = held, hp

            if weakest is None:
                break  # nothing eligible to rotate out

            # Anti-churn gate. Candidates are sorted desc, so if the strongest
            # candidate can't clear the bar, none can.
            if p_new < p_weak + ROTATION_MIN_PROB_IMPROVEMENT:
                break

            # Pre-validate the replacement BUY using cash + estimated proceeds
            # from the rotation SELL. Abort the whole rotation if it can't be
            # built safely (no naked SELL).
            weak_pos = open_positions[weakest]
            proceeds = weak_pos["shares"] * self._latest_close_safe(weakest)
            effective_cash = cash + max(proceeds, 0.0)
            built = self._build_buy_order(
                cand_ticker, p_new, rank, today, equity, effective_cash,
                require_whole_share=True)
            if built is None:
                continue  # this candidate can't be placed; try the next one

            buy_order, cost = built
            buy_order["reason"] = (f"rotation_in_prob={p_new:.4f}_"
                                   f"replacing={weakest}@{p_weak:.4f}")

            # Emit SELL first (lower row index), then the paired BUY.
            orders.append({
                "date": today, "ticker": weakest, "probability": round(p_weak, 6),
                "rank": None, "action": "SELL", "shares": weak_pos["shares"],
                "entry_type": "MOO", "stop_price": None, "target_price": None,
                "reason": "rotation_exit",
            })
            orders.append(buy_order)

            rotated_out.add(weakest)
            bought.add(cand_ticker)
            cash = cash + proceeds - cost  # reflect freed + spent capital
            rotations_done += 1

        return cash

    def _size_position(self, entry_price: float, equity: float,
                       cash: float) -> float:
        risk_dollars = equity * RISK_PER_TRADE
        stop_distance = entry_price * STOP_LOSS_PCT
        if stop_distance <= 0:
            return 0.0
        shares_by_risk = risk_dollars / stop_distance
        shares_by_cash = (cash * MAX_POSITION_FRAC) / entry_price
        return min(shares_by_risk, shares_by_cash)
