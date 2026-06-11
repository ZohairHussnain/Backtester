"""Tests for download_data: the intraday-fallback that rebuilds the latest
daily bar when Yahoo's daily endpoint returns it with a NaN (unconsolidated)
Close even though the session has closed.

Zero-dependency assert harness (mirrors test_ibkr_execution.py). Uses an
in-memory fake yfinance and temp files only -- never hits the network or real
ticker_data/.

Run:  python test_download_data.py
"""

import json
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import download_data as dd

_passed = 0
_failed = 0


def check(cond: bool, msg: str) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS: {msg}")
    else:
        _failed += 1
        print(f"  FAIL: {msg}")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _idx_et(times):
    return pd.to_datetime(times).tz_localize("America/New_York")


class _FakeYF:
    """Stand-in for the yfinance module. .Ticker(t).history(...) returns the
    canned daily frame, or the intraday frame when interval='1m'."""
    def __init__(self, daily=None, intraday=None):
        self._daily = daily
        self._intraday = intraday
        self.intraday_calls = 0

    def Ticker(self, ticker):
        outer = self

        class _T:
            def history(_s, period, interval=None, auto_adjust=False):
                if interval == "1m":
                    outer.intraday_calls += 1
                    return outer._intraday if outer._intraday is not None else pd.DataFrame()
                return outer._daily if outer._daily is not None else pd.DataFrame()

        return _T()


def _complete_intraday():
    # A full, CLOSED regular session (2025-01-02 was a normal NYSE day).
    idx = _idx_et(["2025-01-02 09:30", "2025-01-02 12:00", "2025-01-02 15:59"])
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0], "High": [101.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0], "Close": [100.5, 102.0, 103.25],
        "Volume": [1000, 2000, 1500],
    }, index=idx)


def _incomplete_intraday():
    # Session still trading (last bar 11:00 ET).
    idx = _idx_et(["2025-01-02 09:30", "2025-01-02 11:00"])
    return pd.DataFrame({
        "Open": [100.0, 101.0], "High": [101.0, 103.0],
        "Low": [99.0, 100.0], "Close": [100.5, 102.0], "Volume": [1000, 2000],
    }, index=idx)


def _daily_nan_close():
    # Latest daily row (2025-01-02) has an unconsolidated NaN Close.
    idx = _idx_et(["2024-12-31 00:00", "2025-01-02 00:00"])
    return pd.DataFrame({
        "Open": [98.0, 100.0], "High": [99.0, 104.0], "Low": [97.0, 99.0],
        "Close": [98.5, np.nan], "Volume": [5000, 4500],
    }, index=idx)


def _daily_all_present():
    idx = _idx_et(["2024-12-31 00:00", "2025-01-02 00:00"])
    return pd.DataFrame({
        "Open": [98.0, 100.0], "High": [99.0, 104.0], "Low": [97.0, 99.0],
        "Close": [98.5, 103.25], "Volume": [5000, 4500],
    }, index=idx)


def _rec(d, close):
    return {"Date": f"{d}T00:00:00.000", "Open": 1.0, "High": 1.0,
            "Low": 1.0, "Close": close, "Volume": 1, "OpenInterest": 0}


# ---------------------------------------------------------------------------
# _session_complete
# ---------------------------------------------------------------------------

def test_session_complete_regular_day():
    print("\n[D1] _session_complete: ended session True, mid-session False")
    done = _idx_et(["2025-01-02 15:59"])[0]
    mid = _idx_et(["2025-01-02 11:00"])[0]
    check(bool(dd._session_complete(date(2025, 1, 2), done)),
          "15:59 ET on a regular day -> session complete")
    check(not dd._session_complete(date(2025, 1, 2), mid),
          "11:00 ET -> session not complete")


# ---------------------------------------------------------------------------
# _synthesize_daily_from_intraday
# ---------------------------------------------------------------------------

def test_synthesize_from_complete_intraday():
    print("\n[D2] synthesize a daily bar from a complete intraday session")
    saved = dd.yf
    try:
        dd.yf = _FakeYF(intraday=_complete_intraday())
        rec = dd._synthesize_daily_from_intraday("AAA")
        check(rec is not None, "record built from complete intraday")
        check(rec["Date"][:10] == "2025-01-02", "date is the session date")
        check(rec["Open"] == 100.0, "Open = first bar open")
        check(rec["High"] == 104.0, "High = session max high")
        check(rec["Low"] == 99.0, "Low = session min low")
        check(rec["Close"] == 103.25, "Close = last bar close (the missing value)")
        check(rec["Volume"] == 4500, "Volume = session sum")
    finally:
        dd.yf = saved


def test_synthesize_skips_incomplete_session():
    print("\n[D3] never synthesize from a session still trading")
    saved = dd.yf
    try:
        dd.yf = _FakeYF(intraday=_incomplete_intraday())
        check(dd._synthesize_daily_from_intraday("AAA") is None,
              "mid-session intraday -> None (no fabricated close)")
    finally:
        dd.yf = saved


def test_synthesize_empty_intraday():
    print("\n[D4] no intraday data -> None")
    saved = dd.yf
    try:
        dd.yf = _FakeYF(intraday=pd.DataFrame())
        check(dd._synthesize_daily_from_intraday("AAA") is None,
              "empty intraday frame -> None")
    finally:
        dd.yf = saved


# ---------------------------------------------------------------------------
# update_prices integration
# ---------------------------------------------------------------------------

def test_update_prices_fills_nan_close_from_intraday():
    print("\n[D5] update_prices fills a NaN-close daily bar from intraday")
    saved_yf, saved_out = dd.yf, dd.OUTPUT_DIR
    with tempfile.TemporaryDirectory() as d:
        try:
            dd.OUTPUT_DIR = Path(d)
            uni = Path(d) / "universe.txt"
            uni.write_text("AAA\n")
            (Path(d) / "AAA.json").write_text(json.dumps([_rec("2024-12-31", 98.5)]))
            dd.yf = _FakeYF(daily=_daily_nan_close(), intraday=_complete_intraday())

            updated = dd.update_prices(uni)
            check(updated == ["AAA"], "AAA updated via the intraday fallback")
            check(dd.yf.intraday_calls >= 1, "intraday endpoint was consulted")
            data = json.loads((Path(d) / "AAA.json").read_text())
            dates = [r["Date"][:10] for r in data]
            check("2025-01-02" in dates, "synthesized 2025-01-02 bar persisted")
            row = next(r for r in data if r["Date"][:10] == "2025-01-02")
            check(row["Close"] == 103.25, "persisted Close came from intraday (103.25)")
        finally:
            dd.yf, dd.OUTPUT_DIR = saved_yf, saved_out


def test_update_prices_no_intraday_when_close_present():
    print("\n[D6] healthy daily Close -> normal merge, intraday NOT consulted")
    saved_yf, saved_out = dd.yf, dd.OUTPUT_DIR
    with tempfile.TemporaryDirectory() as d:
        try:
            dd.OUTPUT_DIR = Path(d)
            uni = Path(d) / "universe.txt"
            uni.write_text("AAA\n")
            (Path(d) / "AAA.json").write_text(json.dumps([_rec("2024-12-31", 98.5)]))
            dd.yf = _FakeYF(daily=_daily_all_present(), intraday=_complete_intraday())

            updated = dd.update_prices(uni)
            check(updated == ["AAA"], "normal daily merge adds the new bar")
            check(dd.yf.intraday_calls == 0,
                  "intraday fallback NOT triggered when daily Close is present")
            data = json.loads((Path(d) / "AAA.json").read_text())
            row = next(r for r in data if r["Date"][:10] == "2025-01-02")
            check(row["Close"] == 103.25, "daily Close persisted normally")
        finally:
            dd.yf, dd.OUTPUT_DIR = saved_yf, saved_out


def test_update_prices_no_synth_when_session_open():
    print("\n[D7] NaN close + session still open -> no bar added (waits)")
    saved_yf, saved_out = dd.yf, dd.OUTPUT_DIR
    with tempfile.TemporaryDirectory() as d:
        try:
            dd.OUTPUT_DIR = Path(d)
            uni = Path(d) / "universe.txt"
            uni.write_text("AAA\n")
            (Path(d) / "AAA.json").write_text(json.dumps([_rec("2024-12-31", 98.5)]))
            # Daily NaN close, but intraday shows the session is still trading.
            dd.yf = _FakeYF(daily=_daily_nan_close(), intraday=_incomplete_intraday())

            updated = dd.update_prices(uni)
            check(updated == [], "no update: refuses to synthesize a live session")
            data = json.loads((Path(d) / "AAA.json").read_text())
            check([r["Date"][:10] for r in data] == ["2024-12-31"],
                  "file unchanged (still only the prior bar)")
        finally:
            dd.yf, dd.OUTPUT_DIR = saved_yf, saved_out


def main():
    print("=" * 60)
    print("  DOWNLOAD DATA TESTS")
    print("=" * 60)

    test_session_complete_regular_day()
    test_synthesize_from_complete_intraday()
    test_synthesize_skips_incomplete_session()
    test_synthesize_empty_intraday()
    test_update_prices_fills_nan_close_from_intraday()
    test_update_prices_no_intraday_when_close_present()
    test_update_prices_no_synth_when_session_open()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
