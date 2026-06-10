"""
IBKRBroker: Paper-only IBKR integration via ib_insync.

SAFETY:
  - Port 4002 is HARDCODED. No override from CLI or environment.
  - PAPER_TRADING_ONLY = True. Raises RuntimeError if False.
  - DRY_RUN mode logs intended orders without submitting.
  - No short selling. BUY only for entries, SELL only for existing positions.
  - Shares floored to integer unless ALLOW_FRACTIONAL = True.
  - Duplicate orders are detected and rejected.

RECONCILIATION:
  IBKRBroker does NOT update portfolio_state.json.
  It returns fills to the caller. PortfolioManager updates state
  through FillReconciler only.
"""

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from ib_insync import IB, Stock, MarketOrder, LimitOrder, util

from execution.order import Order, OrderStatus, Action, Fill
from execution.broker_adapter import BrokerAdapter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# HARD SAFETY CONSTANTS — DO NOT CHANGE WITHOUT CODE REVIEW
# ---------------------------------------------------------------------------

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002        # Gateway paper trading. NEVER use 4001 (live).
IBKR_CLIENT_ID = 11
PAPER_TRADING_ONLY = True
ALLOW_FRACTIONAL = False
DRY_RUN = True           # If True, log orders but do not submit to IBKR.

IBKR_ORDER_LOG = OUTPUT_DIR / "ibkr_order_log.csv"

# ---------------------------------------------------------------------------
# Safety checks at import time
# ---------------------------------------------------------------------------

if not PAPER_TRADING_ONLY:
    raise RuntimeError(
        "PAPER_TRADING_ONLY is False. This is not allowed. "
        "Live trading requires a separate code review and deployment."
    )

if IBKR_PORT not in (4002, 7497):
    raise RuntimeError(
        f"IBKR_PORT={IBKR_PORT} is not a paper trading port. "
        f"Allowed: 4002 (Gateway paper), 7497 (TWS paper). "
        f"NEVER use 4001 or 7496."
    )


class IBKRBroker(BrokerAdapter):
    def __init__(self, dry_run: bool = DRY_RUN):
        self.ib = IB()
        self.dry_run = dry_run
        self._connected = False
        self._order_log: list[dict] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to IBKR Gateway/TWS paper trading."""
        if self._connected:
            return

        print(f"  Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT} "
              f"(client={IBKR_CLIENT_ID}, paper={'YES' if PAPER_TRADING_ONLY else 'NO'})...")

        try:
            self.ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID,
                           timeout=10, readonly=self.dry_run)
            self._connected = True

            # Verify paper account
            accounts = self.ib.managedAccounts()
            if accounts:
                acct = accounts[0]
                # IBKR paper accounts typically start with 'DU' or 'DF'
                if not (acct.startswith("DU") or acct.startswith("DF")):
                    self.disconnect()
                    raise RuntimeError(
                        f"Account {acct} does not look like a paper account. "
                        f"Paper accounts start with DU or DF. Disconnecting."
                    )
                print(f"  Connected. Account: {acct} (paper)")
            else:
                print(f"  Connected. No account info available.")

        except ConnectionRefusedError:
            raise RuntimeError(
                f"Connection refused at {IBKR_HOST}:{IBKR_PORT}. "
                f"Is IBKR Gateway running with API enabled?"
            )
        except Exception as e:
            raise RuntimeError(f"IBKR connection failed: {e}")

    def disconnect(self) -> None:
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            print("  Disconnected from IBKR.")

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to IBKR. Call connect() first.")

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Submit a MOO order to IBKR paper trading.

        Returns broker_order_id as string.
        Raises on failure. Does NOT update portfolio state.
        """
        self._ensure_connected()

        # Safety: no shorting
        if order.action == Action.SELL:
            positions = self.get_positions()
            if order.ticker not in positions:
                self._log_order(order, "REJECTED", "", "No position to sell (short not allowed)")
                order.status = OrderStatus.REJECTED
                raise ValueError(f"Cannot sell {order.ticker}: no position (shorting not allowed)")

        # Duplicate check: IBKR open orders
        open_orders = self.get_open_orders()
        for oo in open_orders:
            if oo.get("ticker") == order.ticker and oo.get("action") == order.action.value:
                self._log_order(order, "REJECTED", "", f"Duplicate: existing IBKR order for {order.ticker}")
                order.status = OrderStatus.REJECTED
                raise ValueError(f"Duplicate order for {order.ticker}: already has open IBKR order")

        # Duplicate check: IBKR positions (for BUY)
        if order.action == Action.BUY:
            positions = self.get_positions()
            if order.ticker in positions and positions[order.ticker].get("shares", 0) > 0:
                self._log_order(order, "REJECTED", "", f"Already holding {order.ticker}")
                order.status = OrderStatus.REJECTED
                raise ValueError(f"Already holding {order.ticker} at IBKR")

        # Fractional share handling
        shares = order.shares
        if not ALLOW_FRACTIONAL:
            shares = math.floor(shares)
            if shares < 1:
                self._log_order(order, "REJECTED", "", f"Shares rounded to {shares} (< 1)")
                order.status = OrderStatus.REJECTED
                raise ValueError(f"Order for {order.ticker}: {order.shares:.2f} shares rounds to 0")

        # Build IBKR contract and order
        contract = Stock(order.ticker, "SMART", "USD")
        try:
            self.ib.qualifyContracts(contract)
        except Exception as e:
            self._log_order(order, "REJECTED", "", f"Contract qualification failed: {e}")
            order.status = OrderStatus.REJECTED
            raise ValueError(f"Invalid contract for {order.ticker}: {e}")

        side = "BUY" if order.action == Action.BUY else "SELL"
        ib_order = MarketOrder(side, shares)
        # Time-in-force from the order type: MKT = immediate market order valid
        # for the day (run during RTH); otherwise market-on-open (pre-market).
        if order.order_type == "MKT":
            ib_order.tif = "DAY"
        else:
            ib_order.tif = "OPG"
        ib_order.outsideRth = False  # No extended hours
        type_label = order.order_type

        if self.dry_run:
            self._log_order(order, "DRY_RUN", "DRY-0",
                            f"Would submit {side} {shares} {order.ticker} {type_label}")
            print(f"    [DRY RUN] {side} {shares} {order.ticker} {type_label} "
                  f"(prob={order.probability:.4f})")
            order.status = OrderStatus.PENDING
            return "DRY-0"

        # Submit
        try:
            trade = self.ib.placeOrder(contract, ib_order)
            self.ib.sleep(1)  # wait for acknowledgment

            broker_id = str(trade.order.orderId)
            # permId is the STABLE cross-session id IBKR assigns once the order
            # is acknowledged. orderId can come back as 0 for fills queried in a
            # later API session, so permId is what reconciliation matches on.
            perm_id = str(trade.order.permId) if trade.order.permId else ""
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now().isoformat()
            order.broker_order_id = broker_id
            order.perm_id = perm_id

            status = trade.orderStatus.status
            self._log_order(order, status, broker_id,
                          f"Submitted {side} {shares} {order.ticker} {type_label} permId={perm_id}")
            print(f"    SUBMITTED: {side} {shares} {order.ticker} {type_label} "
                  f"-> orderId={broker_id} ({status})")
            return broker_id

        except Exception as e:
            self._log_order(order, "ERROR", "", str(e))
            order.status = OrderStatus.REJECTED
            raise RuntimeError(f"IBKR order submission failed for {order.ticker}: {e}")

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, broker_order_id: str) -> bool:
        self._ensure_connected()
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == broker_order_id:
                self.ib.cancelOrder(trade.order)
                self.ib.sleep(1)
                return True
        return False

    def get_open_orders(self) -> list[dict]:
        self._ensure_connected()
        result = []
        for trade in self.ib.openTrades():
            result.append({
                "broker_order_id": str(trade.order.orderId),
                "ticker": trade.contract.symbol,
                "action": trade.order.action,
                "shares": trade.order.totalQuantity,
                "order_type": trade.order.orderType,
                "status": trade.orderStatus.status,
            })
        return result

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, dict]:
        self._ensure_connected()
        result = {}
        for pos in self.ib.positions():
            if pos.contract.secType == "STK":
                result[pos.contract.symbol] = {
                    "shares": pos.position,
                    "avg_price": pos.avgCost,
                }
        return result

    def get_account_state(self) -> dict:
        self._ensure_connected()
        summary = {}
        for item in self.ib.accountSummary():
            if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                           "GrossPositionValue", "MaintMarginReq"):
                summary[item.tag] = float(item.value)
        return summary

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_side(side: str) -> str:
        """Map IBKR execution side codes to our Action values.

        IBKR reports executions as 'BOT' (bought) / 'SLD' (sold), not
        'BUY'/'SELL'. Reconciliation builds Action(...) from this, so it must
        be normalized or Action('BOT') would raise.
        """
        s = (side or "").upper()
        if s in ("BOT", "BUY"):
            return "BUY"
        if s in ("SLD", "SELL"):
            return "SELL"
        return s

    def get_fills(self, since: Optional[str] = None) -> list[dict]:
        self._ensure_connected()
        result = []
        for fill in self.ib.fills():
            filled_at = fill.time.isoformat() if fill.time else ""
            if since and filled_at < since:
                continue
            execu = fill.execution
            result.append({
                "fill_id": str(execu.execId),
                # permId is stable across API sessions; orderId may be 0 here.
                "perm_id": str(execu.permId) if getattr(execu, "permId", 0) else "",
                "order_id": str(execu.orderId),
                "ticker": fill.contract.symbol,
                "action": self._normalize_side(execu.side),
                "shares_filled": execu.shares,
                "fill_price": execu.price,
                "commission": fill.commissionReport.commission if fill.commissionReport else 0,
                "filled_at": filled_at,
            })
        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_order(self, order: Order, status: str, broker_id: str, message: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "order_id": order.order_id,
            "ticker": order.ticker,
            "action": order.action.value,
            "shares": order.shares,
            "order_type": order.order_type,
            "status": status,
            "broker_order_id": broker_id,
            "message": message,
        }
        self._order_log.append(entry)

        # Append to CSV
        IBKR_ORDER_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not IBKR_ORDER_LOG.exists()
        with open(IBKR_ORDER_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(entry)
