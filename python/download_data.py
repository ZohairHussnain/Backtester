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


def _session_complete(session_date, last_ts) -> bool:
    """True if the NYSE regular session for session_date has ended.

    Compares the latest intraday bar to the scheduled market close (handles
    half-days via pandas_market_calendars). Falls back to a regular-hours
    15:59 ET check when the calendar package is unavailable. Used to make sure
    we only synthesize a daily bar from a session that is actually OVER -- never
    from one still trading.
    """
    try:
        import pandas as pd
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar("XNYS").schedule(
            start_date=session_date, end_date=session_date)
        if sched.empty:
            return False  # not a trading day per the calendar
        close = sched.iloc[0]["market_close"]  # tz-aware (UTC)
        # The last regular 1m bar is stamped the minute before the close.
        return last_ts.tz_convert("UTC") >= (close - pd.Timedelta(minutes=1))
    except Exception:
        # Fallback: regular close 16:00 ET -> last regular bar 15:59 (local time).
        return last_ts.strftime("%H:%M") >= "15:59"


def _synthesize_daily_from_intraday(ticker: str) -> dict | None:
    """Rebuild the most recent daily bar from 1-minute data.

    Yahoo's DAILY endpoint sometimes returns the latest (already-closed) session
    with a NaN Close that hasn't been consolidated yet, even though the intraday
    data for that session is complete. This reconstructs that one bar from the
    minute series (Open=first, High=max, Low=min, Close=last, Volume=sum) so the
    pipeline doesn't stall a day behind. Returns a _clean_record-style dict, or
    None if intraday data is missing/incomplete (session still trading) -- it
    NEVER fabricates a bar from an unfinished session.
    """
    try:
        m = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
    except Exception:
        return None
    if m.empty:
        return None

    last_ts = m.index[-1]
    if not _session_complete(last_ts.date(), last_ts):
        return None

    try:
        o = float(m["Open"].iloc[0])
        h = float(m["High"].max())
        lo = float(m["Low"].min())
        c = float(m["Close"].iloc[-1])
        vol = m["Volume"].sum()
        v = int(vol) if vol == vol else 0  # NaN guard
    except (ValueError, TypeError):
        return None
    if o != o or c != c or h != h or lo != lo:  # any NaN -> reject
        return None

    return {
        "Date": last_ts.strftime("%Y-%m-%dT00:00:00.000"),
        "Open": round(o, 6), "High": round(h, 6), "Low": round(lo, 6),
        "Close": round(c, 6), "Volume": v, "OpenInterest": 0,
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

                # Fallback: the daily endpoint may return the latest (already
                # closed) session with a NaN Close that Yahoo hasn't consolidated
                # yet, so _clean_record drops it and we stall a day behind. If the
                # most recent daily bar is missing for exactly that reason, rebuild
                # it from intraday data (only if that session has actually ended).
                queued = {r["Date"][:10] for r in new_records}
                last_daily_date = df.index[-1].strftime("%Y-%m-%d")
                last_close = df["Close"].iloc[-1]
                if (last_close != last_close  # NaN
                        and last_daily_date not in existing_dates
                        and last_daily_date not in queued):
                    syn = _synthesize_daily_from_intraday(ticker)
                    if (syn and syn["Date"][:10] not in existing_dates
                            and syn["Date"][:10] not in queued):
                        new_records.append(syn)

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
