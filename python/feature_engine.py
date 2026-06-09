"""Python port of C++ FeatureEngine.h — computes the same 11 features.

Must produce identical output to the C++ version for the same input data.
All features use only current and past bars (no look-ahead).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_COLS, MINIMUM_HISTORY, TICKER_DATA_DIR


def load_prices(ticker: str, data_dir: Path = TICKER_DATA_DIR) -> pd.DataFrame:
    """Load OHLCV data from ticker JSON file."""
    path = data_dir / f"{ticker}.json"
    if not path.exists():
        raise FileNotFoundError(f"No data file for {ticker}: {path}")

    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    # Use close as adjusted_close (matching C++ fallback when Adjusted Close is absent)
    df["adj_close"] = df["close"]
    df = df.sort_values("date").reset_index(drop=True)
    df["ticker"] = ticker
    return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def _adjusted_return(adj_close: np.ndarray, i: int, lookback: int) -> float:
    past = adj_close[i - lookback]
    if past == 0.0:
        return 0.0
    return adj_close[i] / past - 1.0


def _average(values: np.ndarray, end: int, window: int) -> float:
    return values[end + 1 - window: end + 1].mean()


def _rsi_14(adj_close: np.ndarray, end: int) -> float:
    gains = 0.0
    losses = 0.0
    for i in range(end + 1 - 14, end + 1):
        change = adj_close[i] - adj_close[i - 1]
        if change > 0:
            gains += change
        else:
            losses += -change
    avg_gain = gains / 14.0
    avg_loss = losses / 14.0
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_14_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, end: int) -> float:
    tr_sum = 0.0
    for i in range(end + 1 - 14, end + 1):
        prev_close = close[i - 1]
        hl = high[i] - low[i]
        hpc = abs(high[i] - prev_close)
        lpc = abs(low[i] - prev_close)
        tr_sum += max(hl, hpc, lpc)
    atr = tr_sum / 14.0
    return atr / close[end] if close[end] != 0.0 else 0.0


def _rolling_vol_20(adj_close: np.ndarray, end: int) -> float:
    rets = []
    for i in range(end + 1 - 20, end + 1):
        past = adj_close[i - 1]
        rets.append(adj_close[i] / past - 1.0 if past != 0.0 else 0.0)
    rets = np.array(rets)
    mean = rets.mean()
    sq_diff = ((rets - mean) ** 2).sum()
    return np.sqrt(sq_diff / len(rets))


def compute_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 11 features for a single ticker's price history.

    Input must be sorted by date with columns:
        date, ticker, open, high, low, close, adj_close, volume

    Returns rows starting from index MINIMUM_HISTORY-1 (bar 199).
    """
    if len(df) < MINIMUM_HISTORY:
        return pd.DataFrame()

    adj = df["adj_close"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    vol = df["volume"].values.astype(float)

    rows = []
    for i in range(MINIMUM_HISTORY - 1, len(df)):
        ma20 = _average(adj, i, 20)
        ma50 = _average(adj, i, 50)
        ma200 = _average(adj, i, 200)
        avg_vol20 = _average(vol, i, 20)

        rows.append({
            "date": df.iloc[i]["date"],
            "ticker": df.iloc[i]["ticker"],
            "ret_1d": _adjusted_return(adj, i, 1),
            "ret_5d": _adjusted_return(adj, i, 5),
            "ret_10d": _adjusted_return(adj, i, 10),
            "ret_20d": _adjusted_return(adj, i, 20),
            "rsi_14": _rsi_14(adj, i),
            "atr_14_pct": _atr_14_pct(high, low, close, i),
            "volume_ratio_20": vol[i] / avg_vol20 if avg_vol20 != 0 else 0.0,
            "dist_ma20": adj[i] / ma20 - 1.0 if ma20 != 0 else 0.0,
            "dist_ma50": adj[i] / ma50 - 1.0 if ma50 != 0 else 0.0,
            "dist_ma200": adj[i] / ma200 - 1.0 if ma200 != 0 else 0.0,
            "rolling_vol_20": _rolling_vol_20(adj, i),
        })

    return pd.DataFrame(rows)


def compute_features(tickers: list[str], data_dir: Path = TICKER_DATA_DIR) -> pd.DataFrame:
    """Compute features for all tickers. Returns combined DataFrame."""
    all_features = []
    for ticker in tickers:
        try:
            prices = load_prices(ticker, data_dir)
            features = compute_features_for_ticker(prices)
            if len(features) > 0:
                all_features.append(features)
        except Exception as e:
            print(f"  {ticker}: SKIP ({e})")
    if not all_features:
        return pd.DataFrame()
    return pd.concat(all_features, ignore_index=True)


def get_latest_features(tickers: list[str], data_dir: Path = TICKER_DATA_DIR) -> pd.DataFrame:
    """Get only the most recent feature row per ticker (today's features)."""
    all_features = compute_features(tickers, data_dir)
    if all_features.empty:
        return all_features
    # Take the last row per ticker
    return all_features.groupby("ticker").tail(1).reset_index(drop=True)
