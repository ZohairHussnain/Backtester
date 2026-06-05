"""Portfolio state management with JSON persistence."""

import json
from datetime import datetime
from pathlib import Path

from config import (
    STARTING_CAPITAL, MAX_HOLD_DAYS, SLIPPAGE,
    FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT,
    PORTFOLIO_STATE_FILE,
)


class Portfolio:
    def __init__(self, state_file: Path = PORTFOLIO_STATE_FILE,
                 starting_capital: float = STARTING_CAPITAL):
        self.state_file = state_file
        self.state = self._load_or_init(starting_capital)

    def _load_or_init(self, starting_capital: float) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
            print(f"Loaded portfolio state from {self.state_file}")
            print(f"  Cash: ${state['cash']:.2f}, "
                  f"Positions: {len(state.get('open_positions', {}))}")
            return state
        return {
            "cash": starting_capital,
            "starting_capital": starting_capital,
            "open_positions": {},
            "trade_history": [],
            "last_updated": datetime.now().isoformat(),
        }

    def save(self) -> None:
        self.state["last_updated"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def get_state(self) -> dict:
        return self.state

    @property
    def cash(self) -> float:
        return self.state["cash"]

    @property
    def open_positions(self) -> dict:
        return self.state.get("open_positions", {})

    @property
    def equity(self) -> float:
        """Cash + estimated value of open positions at entry price (conservative)."""
        total = self.cash
        for _, pos in self.open_positions.items():
            total += pos["shares"] * pos["entry_price"]
        return total

    def record_entry(self, ticker: str, shares: float, entry_price: float,
                     stop_price: float, target_price: float,
                     date: str, fee: float) -> None:
        if ticker in self.open_positions:
            raise ValueError(f"Already holding {ticker}")

        cost = shares * entry_price + fee
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need ${cost:.2f}, have ${self.cash:.2f}")

        self.state["cash"] -= cost
        self.state["open_positions"][ticker] = {
            "shares": shares,
            "entry_price": entry_price,
            "entry_date": date,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_fee": fee,
        }

    def record_exit(self, ticker: str, exit_price: float,
                    date: str, reason: str) -> None:
        if ticker not in self.open_positions:
            raise ValueError(f"Not holding {ticker}")

        pos = self.open_positions[ticker]
        exec_price = exit_price * (1 - SLIPPAGE)
        value = pos["shares"] * exec_price
        fee = max(min(pos["shares"] * FEE_PER_SHARE, FEE_CAP_PCT * value), FEE_MIN)
        pnl = (exec_price - pos["entry_price"]) * pos["shares"] - fee - pos["entry_fee"]

        self.state["cash"] += value - fee
        self.state["trade_history"].append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "exit_date": date,
            "entry_price": pos["entry_price"],
            "exit_price": exec_price,
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
            "return_pct": round(exec_price / pos["entry_price"] - 1, 6),
            "reason": reason,
        })
        del self.state["open_positions"][ticker]

    def check_exits(self, today: str) -> list[str]:
        """Return tickers that should be exited (max hold exceeded)."""
        exits = []
        for ticker, pos in self.open_positions.items():
            # Simple date comparison — counts calendar days
            entry = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
            now = datetime.strptime(today, "%Y-%m-%d")
            days_held = (now - entry).days
            if days_held >= MAX_HOLD_DAYS:
                exits.append(ticker)
        return exits
