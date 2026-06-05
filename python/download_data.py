"""Download OHLCV data for tickers. Supports full download and incremental update."""

import json
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ticker_data"
UNIVERSE_FILE = OUTPUT_DIR / "universe.txt"


def _clean_record(date, row) -> dict | None:
    try:
        c = float(row["Close"])
        o = float(row["Open"])
        if c != c or o != o:  # NaN
            return None
    except (ValueError, TypeError):
        return None
    return {
        "Date": date.strftime("%Y-%m-%dT00:00:00.000"),
        "Open": round(float(row["Open"]), 6),
        "High": round(float(row["High"]), 6),
        "Low": round(float(row["Low"]), 6),
        "Close": round(float(row["Close"]), 6),
        "Volume": int(row["Volume"]) if row["Volume"] == row["Volume"] else 0,
        "OpenInterest": 0,
    }


def download_ticker(ticker: str, period: str = "max") -> bool:
    """Download full history for a ticker."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=False)
        if df.empty or len(df) < 200:
            return False

        records = [r for r in (_clean_record(d, row) for d, row in df.iterrows()) if r]
        if len(records) < 200:
            return False

        path = OUTPUT_DIR / f"{ticker}.json"
        with open(path, "w") as f:
            json.dump(records, f)
        return True
    except Exception:
        return False


def update_prices(universe_file: Path = UNIVERSE_FILE) -> list[str]:
    """Download latest data for all tickers in universe. Returns updated tickers."""
    if not universe_file.exists():
        print(f"No universe file at {universe_file}")
        return []

    tickers = [line.strip() for line in open(universe_file) if line.strip()]
    print(f"Updating prices for {len(tickers)} tickers...")

    updated = []
    for ticker in tickers:
        path = OUTPUT_DIR / f"{ticker}.json"
        if path.exists():
            # Incremental: download last 5 days, merge
            try:
                t = yf.Ticker(ticker)
                df = t.history(period="5d", auto_adjust=False)
                if df.empty:
                    continue

                with open(path) as f:
                    existing = json.load(f)

                existing_dates = {r["Date"][:10] for r in existing}
                new_records = []
                for date, row in df.iterrows():
                    rec = _clean_record(date, row)
                    if rec and rec["Date"][:10] not in existing_dates:
                        new_records.append(rec)

                if new_records:
                    existing.extend(new_records)
                    existing.sort(key=lambda r: r["Date"])
                    with open(path, "w") as f:
                        json.dump(existing, f)
                    updated.append(ticker)
            except Exception:
                pass
        else:
            # Full download for new tickers
            if download_ticker(ticker):
                updated.append(ticker)

    print(f"Updated {len(updated)} tickers: {updated[:10]}{'...' if len(updated) > 10 else ''}")
    return updated


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if UNIVERSE_FILE.exists():
        update_prices()
    else:
        default = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "SPY"]
        for ticker in default:
            print(f"Downloading {ticker}...", end=" ")
            if download_ticker(ticker):
                print("OK")
            else:
                print("FAILED")


if __name__ == "__main__":
    main()
