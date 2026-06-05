"""Portfolio state management with JSON persistence.

Safety properties:
- Atomic saves via temp file + rename
- record_entry validates before mutating state
- Cash cannot go negative
- Duplicate positions rejected
- Trade history capped at 1000 entries
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from config import (
    STARTING_CAPITAL, MAX_HOLD_DAYS, SLIPPAGE,
    FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT,
    PORTFOLIO_STATE_FILE, MAX_POSITIONS,
)

MAX_TRADE_HISTORY = 1000


class Portfolio:
    def __init__(self, state_file: Path = PORTFOLIO_STATE_FILE,
                 starting_capital: float = STARTING_CAPITAL):
        self.state_file = state_file
        self.state = self._load_or_init(starting_capital)

    def _load_or_init(self, starting_capital: float) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                if "cash" not in state or "open_positions" not in state:
                    raise ValueError("Missing required fields")
                print(f"Loaded portfolio state from {self.state_file}")
                print(f"  Cash: ${state['cash']:.2f}, "
                      f"Positions: {len(state.get('open_positions', {}))}")
                return state
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Corrupt file — load backup if exists
                backup = Path(str(self.state_file) + ".bak")
                if backup.exists():
                    print(f"  WARNING: State file corrupt ({e}), loading backup")
                    with open(backup) as f:
                        return json.load(f)
                print(f"  WARNING: State file corrupt ({e}), starting fresh")

        return {
            "cash": starting_capital,
            "starting_capital": starting_capital,
            "open_positions": {},
            "trade_history": [],
            "last_updated": "",  # empty = never run before
        }

    def save(self) -> None:
        """Atomic save: write to temp file, then rename."""
        self.state["last_updated"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Trim trade history if too long
        if len(self.state.get("trade_history", [])) > MAX_TRADE_HISTORY:
            self.state["trade_history"] = self.state["trade_history"][-MAX_TRADE_HISTORY:]

        # Backup existing file
        if self.state_file.exists():
            backup = Path(str(self.state_file) + ".bak")
            shutil.copy2(self.state_file, backup)

        # Write to temp file, then atomic rename
        tmp_path = Path(str(self.state_file) + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.state_file)

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
        total = self.cash
        for _, pos in self.open_positions.items():
            total += pos["shares"] * pos["entry_price"]
        return total

    def record_entry(self, ticker: str, shares: float, entry_price: float,
                     stop_price: float, target_price: float,
                     date: str, fee: float) -> None:
        """Record a new position. Validates ALL constraints before mutating state."""
        # --- Validate before any mutation ---
        if ticker in self.open_positions:
            raise ValueError(f"Already holding {ticker}")
        if len(self.open_positions) >= MAX_POSITIONS:
            raise ValueError(f"Max positions ({MAX_POSITIONS}) reached")
        if shares <= 0:
            raise ValueError(f"Invalid shares: {shares}")
        if entry_price <= 0:
            raise ValueError(f"Invalid entry price: {entry_price}")

        cost = shares * entry_price + fee
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need ${cost:.2f}, have ${self.cash:.2f}")

        new_cash = self.cash - cost
        if new_cash < 0:
            raise ValueError(f"Cash would go negative: ${new_cash:.2f}")

        # --- All checks passed — mutate atomically ---
        self.state["cash"] = new_cash
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
        """Record position exit. Validates before mutating."""
        if ticker not in self.open_positions:
            raise ValueError(f"Not holding {ticker}")

        pos = self.open_positions[ticker]
        exec_price = exit_price * (1 - SLIPPAGE)
        value = pos["shares"] * exec_price
        fee = max(min(pos["shares"] * FEE_PER_SHARE, FEE_CAP_PCT * value), FEE_MIN)
        pnl = (exec_price - pos["entry_price"]) * pos["shares"] - fee - pos["entry_fee"]

        # Build trade record before mutating
        trade = {
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "exit_date": date,
            "entry_price": pos["entry_price"],
            "exit_price": exec_price,
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
            "return_pct": round(exec_price / pos["entry_price"] - 1, 6),
            "reason": reason,
        }

        # --- Mutate atomically ---
        self.state["cash"] += value - fee
        self.state["trade_history"].append(trade)
        del self.state["open_positions"][ticker]

    def check_exits(self, today: str) -> list[str]:
        """Return tickers that should be exited (max hold exceeded)."""
        try:
            now = datetime.strptime(today, "%Y-%m-%d")
        except ValueError:
            print(f"  WARNING: Invalid date format for exit check: {today}")
            return []

        exits = []
        for ticker, pos in self.open_positions.items():
            try:
                entry = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
                days_held = (now - entry).days
                if days_held >= MAX_HOLD_DAYS:
                    exits.append(ticker)
            except (ValueError, KeyError):
                print(f"  WARNING: Bad entry_date for {ticker}, forcing exit")
                exits.append(ticker)
        return exits
