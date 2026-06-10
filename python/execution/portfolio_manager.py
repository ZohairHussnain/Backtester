"""PortfolioManager: state updated ONLY from confirmed fills."""

import json
import os
import shutil
from pathlib import Path

from execution.order import Fill, Action

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import STARTING_CAPITAL, SLIPPAGE


class PortfolioManager:
    def __init__(self, state_file: Path, starting_capital: float = STARTING_CAPITAL):
        self.state_file = state_file
        self.state = self._load_or_init(starting_capital)

    def _load_or_init(self, starting_capital: float) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                if "cash" in state and "open_positions" in state:
                    return state
            except (json.JSONDecodeError, ValueError):
                backup = Path(str(self.state_file) + ".bak")
                if backup.exists():
                    with open(backup) as f:
                        return json.load(f)
        return {
            "cash": starting_capital,
            "starting_capital": starting_capital,
            "open_positions": {},
            "trade_history": [],
        }

    def apply_fill(self, fill: Fill, forward_return: float = 0.0,
                   stop_price: float = 0.0, target_price: float = 0.0,
                   entry_fee: float = 0.0) -> None:
        """Update portfolio state from a confirmed fill.
        This is the ONLY way to mutate positions."""
        if fill.action == Action.BUY:
            cost = fill.shares_filled * fill.fill_price + fill.commission
            if cost > self.state["cash"]:
                raise ValueError(f"Insufficient cash for fill: need ${cost:.2f}, have ${self.state['cash']:.2f}")
            self.state["cash"] -= cost
            self.state["open_positions"][fill.ticker] = {
                "shares": fill.shares_filled,
                "entry_price": fill.fill_price,
                "entry_date": fill.filled_at[:10] if fill.filled_at else "",
                "stop_price": stop_price,
                "target_price": target_price,
                "entry_fee": entry_fee,
                "forward_return": forward_return,
            }
        elif fill.action == Action.SELL:
            if fill.ticker not in self.state["open_positions"]:
                raise ValueError(f"Cannot sell {fill.ticker}: not in portfolio")
            pos = self.state["open_positions"][fill.ticker]
            value = fill.shares_filled * fill.fill_price
            self.state["cash"] += value - fill.commission
            self.state["trade_history"].append({
                "ticker": fill.ticker,
                "entry_date": pos.get("entry_date", ""),
                "exit_date": fill.filled_at[:10] if fill.filled_at else "",
                "entry_price": pos["entry_price"],
                "exit_price": fill.fill_price,
                "shares": fill.shares_filled,
                "pnl": round((fill.fill_price - pos["entry_price"]) * fill.shares_filled
                             - fill.commission - pos.get("entry_fee", 0), 6),
                "return_pct": round(fill.fill_price / pos["entry_price"] - 1, 6),
                "reason": "max_hold_exit",
            })
            del self.state["open_positions"][fill.ticker]

    def get_state(self) -> dict:
        return self.state

    @property
    def cash(self) -> float:
        return self.state["cash"]

    @property
    def open_positions(self) -> dict:
        return self.state.get("open_positions", {})

    def equity(self, mark_prices: dict[str, float] = None) -> float:
        total = self.cash
        for ticker, pos in self.open_positions.items():
            price = (mark_prices or {}).get(ticker, pos["entry_price"])
            total += pos["shares"] * price
        return total

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists():
            shutil.copy2(self.state_file, str(self.state_file) + ".bak")
        tmp = Path(str(self.state_file) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.state_file)
