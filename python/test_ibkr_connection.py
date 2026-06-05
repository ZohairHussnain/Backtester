"""
IBKR Connection Smoke Test

Tests connectivity to IBKR Gateway paper trading.
Places NO orders. Read-only.

Prerequisites:
  1. Install: pip install ib_insync
  2. Start IBKR Gateway (or TWS) in paper trading mode
  3. Enable API: Configure -> API -> Settings:
     - Enable ActiveX and Socket Clients: checked
     - Socket port: 4002 (Gateway paper) or 7497 (TWS paper)
     - Allow connections from localhost only: checked
  4. Run this script: python test_ibkr_connection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    print("=" * 60)
    print("  IBKR CONNECTION SMOKE TEST")
    print("  Paper trading only. No orders will be placed.")
    print("=" * 60)

    # Import here to catch import errors early
    try:
        from ib_insync import IB
    except ImportError:
        print("\nERROR: ib_insync not installed.")
        print("Run: pip install ib_insync")
        sys.exit(1)

    from execution.ibkr_broker import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID

    print(f"\n  Host:      {IBKR_HOST}")
    print(f"  Port:      {IBKR_PORT}")
    print(f"  Client ID: {IBKR_CLIENT_ID}")
    print(f"  Paper:     YES (enforced)")

    # Connect
    print(f"\n[1/5] Connecting...")
    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID,
                   timeout=10, readonly=True)
    except ConnectionRefusedError:
        print(f"\n  FAILED: Connection refused at {IBKR_HOST}:{IBKR_PORT}")
        print(f"  Is IBKR Gateway running? Is API enabled?")
        print(f"\n  Setup steps:")
        print(f"    1. Open IBKR Gateway (or TWS)")
        print(f"    2. Log in with paper trading credentials")
        print(f"    3. Go to Configure -> API -> Settings")
        print(f"    4. Enable 'Enable ActiveX and Socket Clients'")
        print(f"    5. Set Socket port to {IBKR_PORT}")
        print(f"    6. Check 'Allow connections from localhost only'")
        print(f"    7. Re-run this script")
        sys.exit(1)
    except Exception as e:
        print(f"\n  FAILED: {e}")
        sys.exit(1)

    print(f"  Connected successfully.")

    # Account info
    print(f"\n[2/5] Account summary...")
    try:
        accounts = ib.managedAccounts()
        print(f"  Accounts: {accounts}")
        if accounts:
            acct = accounts[0]
            if acct.startswith("DU") or acct.startswith("DF"):
                print(f"  Account type: PAPER (confirmed)")
            else:
                print(f"  WARNING: Account {acct} may not be paper!")

        for item in ib.accountSummary():
            if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                           "GrossPositionValue"):
                print(f"  {item.tag}: ${float(item.value):,.2f}")
    except Exception as e:
        print(f"  Account summary failed: {e}")

    # Positions
    print(f"\n[3/5] Open positions...")
    try:
        positions = ib.positions()
        if positions:
            for pos in positions:
                if pos.contract.secType == "STK":
                    print(f"  {pos.contract.symbol}: {pos.position} shares "
                          f"@ ${pos.avgCost:.2f}")
        else:
            print(f"  No open positions.")
    except Exception as e:
        print(f"  Positions failed: {e}")

    # Open orders
    print(f"\n[4/5] Open orders...")
    try:
        trades = ib.openTrades()
        if trades:
            for trade in trades:
                print(f"  {trade.order.action} {trade.order.totalQuantity} "
                      f"{trade.contract.symbol} ({trade.orderStatus.status})")
        else:
            print(f"  No open orders.")
    except Exception as e:
        print(f"  Open orders failed: {e}")

    # Disconnect
    print(f"\n[5/5] Disconnecting...")
    ib.disconnect()
    print(f"  Disconnected.")

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST PASSED")
    print(f"  IBKR Gateway paper trading is accessible.")
    print(f"  No orders were placed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
