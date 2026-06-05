"""
Daily production pipeline. Run after market close.

Usage:
    python run_daily.py

Workflow:
    1. Update latest OHLCV prices
    2. Compute features for all tickers
    3. Load trained model and generate predictions
    4. Load portfolio state
    5. Check for exits (max hold exceeded)
    6. Generate entry orders (ranked by probability)
    7. Execute orders (dry run — no live trades)
    8. Save portfolio state
    9. Generate daily HTML report
"""

import sys
from datetime import datetime
from pathlib import Path

# Ensure python/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    UNIVERSE_FILE, MODELS_DIR, PORTFOLIO_STATE_FILE,
    ORDERS_FILE, OUTPUT_DIR, DRY_RUN, STARTING_CAPITAL,
)
from download_data import update_prices
from feature_engine import get_latest_features
from predictor import Predictor
from portfolio import Portfolio
from order_generator import OrderGenerator
from paper_trade_executor import PaperTradeExecutor
from reporter import Reporter


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        print(f"ERROR: Universe file not found: {UNIVERSE_FILE}")
        sys.exit(1)
    tickers = [line.strip() for line in open(UNIVERSE_FILE) if line.strip()]
    print(f"Universe: {len(tickers)} tickers")
    return tickers


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print("=" * 60)
    print(f"  DAILY PIPELINE — {today}")
    print(f"  Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print("=" * 60)

    # 1. Update prices
    print("\n[1/8] Updating prices...")
    update_prices(UNIVERSE_FILE)

    # 2. Compute features
    print("\n[2/8] Computing features...")
    tickers = load_universe()
    features = get_latest_features(tickers)
    if features.empty:
        print("ERROR: No features computed. Aborting.")
        sys.exit(1)
    print(f"  Features for {len(features)} tickers on {features['date'].iloc[0]}")

    # 3. Load model and predict
    print("\n[3/8] Generating predictions...")
    predictor = Predictor(MODELS_DIR)
    predictor.load_latest_model()
    predictions = predictor.predict(features)
    n_above = (predictions["probability"] >= 0.60).sum()
    print(f"  {len(predictions)} predictions, {n_above} above threshold 0.60")

    # 4. Load portfolio
    print("\n[4/8] Loading portfolio state...")
    portfolio = Portfolio(PORTFOLIO_STATE_FILE, STARTING_CAPITAL)

    # 5. Check exits
    print("\n[5/8] Checking exits...")
    exit_tickers = portfolio.check_exits(today)
    if exit_tickers:
        print(f"  Exit signals: {exit_tickers}")
    else:
        print(f"  No exits needed.")

    # 6. Generate orders
    print("\n[6/8] Generating orders...")
    generator = OrderGenerator()
    orders = generator.generate_orders(predictions, portfolio.get_state(), exit_tickers)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_csv(ORDERS_FILE, index=False)
    print(f"  {len(orders)} orders saved to {ORDERS_FILE}")

    # 7. Execute (dry run)
    print("\n[7/8] Executing orders...")
    executor = PaperTradeExecutor(dry_run=DRY_RUN)
    executor.execute(orders, portfolio)

    # Apply to portfolio state (even in dry run, to track paper positions)
    for _, order in orders.iterrows():
        if order["action"] == "SELL" and order["ticker"] in portfolio.open_positions:
            # Use entry price as approximate exit (no real-time price in dry run)
            pos = portfolio.open_positions[order["ticker"]]
            portfolio.record_exit(order["ticker"], pos["entry_price"], today, order["reason"])
        elif order["action"] == "BUY" and order["ticker"] not in portfolio.open_positions:
            from feature_engine import load_prices
            try:
                prices = load_prices(order["ticker"])
                latest_close = prices.iloc[-1]["close"]
                from config import SLIPPAGE, FEE_PER_SHARE, FEE_MIN, FEE_CAP_PCT
                entry_price = latest_close * (1 + SLIPPAGE)
                shares = order["shares"]
                fee = max(min(shares * FEE_PER_SHARE, FEE_CAP_PCT * shares * entry_price), FEE_MIN)
                portfolio.record_entry(
                    order["ticker"], shares, entry_price,
                    order["stop_price"], order["target_price"],
                    today, fee
                )
            except Exception as e:
                print(f"  Failed to record entry for {order['ticker']}: {e}")

    # 8. Save state and report
    print("\n[8/8] Saving state and generating report...")
    portfolio.save()

    reporter = Reporter()
    report_path = reporter.generate_report(
        portfolio.get_state(), orders, predictions)
    print(f"  Report saved to {report_path}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Equity: ${portfolio.equity:,.2f}")
    print(f"  Open positions: {len(portfolio.open_positions)}")
    for ticker, pos in portfolio.open_positions.items():
        print(f"    {ticker}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.2f}")
    print(f"  Closed trades: {len(portfolio.get_state().get('trade_history', []))}")
    print("=" * 60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
