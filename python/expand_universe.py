"""
Research Phase 8: Universe Expansion

Download 100+ stocks across sectors, export features+labels via C++,
train the same pipeline, and test cross-sectional stock selection.
"""

import json
from pathlib import Path
import yfinance as yf

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ticker_data"

# Broad universe: ~100 stocks across sectors, including winners, mediocre, and losers.
UNIVERSE = {
    # Technology
    "tech": ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AVGO", "ORCL", "CRM", "ADBE", "AMD",
             "INTC", "CSCO", "IBM", "TXN", "QCOM", "MU", "HPQ", "DELL"],
    # Healthcare
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
                   "GILD", "CVS", "CI", "MDT", "BIIB"],
    # Financials
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
                   "MET", "PRU", "ALL"],
    # Consumer Discretionary
    "consumer_disc": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX",
                      "F", "GM", "MAR"],
    # Consumer Staples
    "consumer_staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MO", "PM", "KHC"],
    # Industrials
    "industrials": ["CAT", "BA", "HON", "UPS", "GE", "MMM", "DE", "LMT", "RTX", "FDX"],
    # Energy
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "HAL"],
    # Utilities
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE"],
    # Communications
    "communications": ["DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS"],
    # ETFs for benchmarking
    "etf": ["SPY", "QQQ"],
}


def download_ticker(ticker: str) -> bool:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="max", auto_adjust=False)
        if df.empty or len(df) < 500:
            print(f"  {ticker}: SKIP (only {len(df)} rows)")
            return False

        records = []
        for date, row in df.iterrows():
            if row["Close"] is None or row["Open"] is None:
                continue
            try:
                c = float(row["Close"])
                o = float(row["Open"])
                if c != c or o != o:  # NaN check
                    continue
            except (ValueError, TypeError):
                continue
            records.append({
                "Date": date.strftime("%Y-%m-%dT00:00:00.000"),
                "Open": round(float(row["Open"]), 6),
                "High": round(float(row["High"]), 6),
                "Low": round(float(row["Low"]), 6),
                "Close": round(float(row["Close"]), 6),
                "Volume": int(row["Volume"]) if row["Volume"] == row["Volume"] else 0,
                "OpenInterest": 0,
            })

        if len(records) < 500:
            print(f"  {ticker}: SKIP (only {len(records)} clean rows)")
            return False

        path = OUTPUT_DIR / f"{ticker}.json"
        with open(path, "w") as f:
            json.dump(records, f)
        first = records[0]["Date"][:10]
        last = records[-1]["Date"][:10]
        print(f"  {ticker}: {len(records)} rows ({first} to {last})")
        return True
    except Exception as e:
        print(f"  {ticker}: ERROR ({e})")
        return False


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_tickers = []
    for sector, tickers in UNIVERSE.items():
        all_tickers.extend(tickers)

    all_tickers = sorted(set(all_tickers))
    print(f"Downloading {len(all_tickers)} tickers...")

    success = []
    for ticker in all_tickers:
        if download_ticker(ticker):
            success.append(ticker)

    print(f"\nDownloaded {len(success)}/{len(all_tickers)} tickers successfully.")
    print(f"Tickers: {success}")

    # Save ticker list
    list_path = OUTPUT_DIR / "universe.txt"
    with open(list_path, "w") as f:
        for t in success:
            f.write(t + "\n")
    print(f"Ticker list saved to {list_path}")


if __name__ == "__main__":
    main()
