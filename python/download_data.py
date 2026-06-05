"""Download full historical OHLCV data for tickers and save in the format expected by the C++ engine."""

import json
from pathlib import Path
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "SPY"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ticker_data"

def download_ticker(ticker: str):
    print(f"Downloading {ticker}...", end=" ")
    t = yf.Ticker(ticker)
    df = t.history(period="max", auto_adjust=False)

    if df.empty:
        print("FAILED (no data)")
        return

    records = []
    for date, row in df.iterrows():
        records.append({
            "Date": date.strftime("%Y-%m-%dT00:00:00.000"),
            "Open": round(row["Open"], 6),
            "High": round(row["High"], 6),
            "Low": round(row["Low"], 6),
            "Close": round(row["Close"], 6),
            "Volume": int(row["Volume"]),
            "OpenInterest": 0,
        })

    path = OUTPUT_DIR / f"{ticker}.json"
    with open(path, "w") as f:
        json.dump(records, f)

    print(f"{len(records)} rows, {records[0]['Date'][:10]} to {records[-1]['Date'][:10]}")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for ticker in TICKERS:
        download_ticker(ticker)
    print("Done.")

if __name__ == "__main__":
    main()
