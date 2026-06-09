"""
Backtest comparison: Does better AUC translate to better trading?

Runs C++ MultiAssetBacktest with different prediction sets and thresholds,
then compares trading performance.

Experiments:
  1. Momentum baseline (no ML)
  2. Old target_stop label predictions at threshold 0.55, 0.60, 0.65
  3. New median_return label predictions at threshold 0.55, 0.60, 0.65

All experiments use the same portfolio settings and date range.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PREDICTIONS_PATH = PROJECT_DIR / "output" / "predictions.csv"
PREDICTIONS_ALL_PATH = PROJECT_DIR / "output" / "predictions_all.csv"
TRADES_PATH = PROJECT_DIR / "output" / "trades.csv"
EQUITY_PATH = PROJECT_DIR / "output" / "equity.dat"


def generate_target_stop_predictions():
    """Re-train with target_stop label and save predictions separately."""
    print("Generating target_stop predictions...")

    df = pd.read_csv(PROJECT_DIR / "output" / "ml_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "rsi_14", "atr_14_pct", "volume_ratio_20",
        "dist_ma20", "dist_ma50", "dist_ma200",
        "rolling_vol_20",
    ]
    df = df.dropna(subset=feature_cols + ["label_target_stop"])

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[feature_cols]
    y = df["label_target_stop"]
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())

    all_preds = []
    for i in range(4, len(years)):
        test_year = years[i]
        train_mask = df["year"] < test_year
        test_mask = df["year"] == test_year

        # Purge last 20 rows
        tickers = sorted(df["ticker"].unique())
        purge = set()
        for ticker in tickers:
            tp = df.index[(df["ticker"] == ticker) & train_mask]
            if len(tp) > 20:
                purge.update(tp[-20:])
            else:
                purge.update(tp)
        train_mask = train_mask & ~df.index.isin(purge)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]

        model = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        preds = df.loc[test_mask, ["date", "ticker"]].copy()
        preds["probability"] = proba
        all_preds.append(preds)

    result = pd.concat(all_preds, ignore_index=True)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    out_path = PROJECT_DIR / "output" / "predictions_target_stop.csv"
    result.to_csv(out_path, index=False)
    print(f"  Saved {len(result)} predictions to {out_path}")
    return out_path


def load_equity(path):
    """Load equity.dat (index value format)."""
    data = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data.append(float(parts[1]))
    return data


def compute_metrics(equity, trades_df):
    """Compute trading metrics from equity curve and trades."""
    if not equity or len(equity) < 2:
        return {"CAGR": 0, "Sharpe": 0, "MaxDD": 0, "Trades": 0,
                "WinRate": 0, "ProfitFactor": 0, "AvgReturn": 0, "Exposure": 0}

    eq = np.array(equity)
    days = len(eq)
    years = days / 252.0

    # CAGR
    cagr = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0 if eq[0] > 0 and eq[-1] > 0 else 0

    # Sharpe
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = ((np.mean(rets) - 0.04/252) / np.std(rets)) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Trade metrics
    n_trades = len(trades_df)
    if n_trades > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / n_trades
        total_win = wins["pnl"].sum() if len(wins) > 0 else 0
        total_loss = -losses["pnl"].sum() if len(losses) > 0 else 0
        pf = total_win / total_loss if total_loss > 0 else float("inf") if total_win > 0 else 0
        avg_ret = trades_df["return_pct"].mean()

        # Exposure: fraction of days in a position
        if "entry_index" in trades_df.columns and "exit_index" in trades_df.columns:
            bars_held = (trades_df["exit_index"] - trades_df["entry_index"]).sum()
            exposure = bars_held / days if days > 0 else 0
        else:
            exposure = 0
    else:
        win_rate = pf = avg_ret = exposure = 0

    return {
        "CAGR": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 4),
        "MaxDD": round(max_dd * 100, 2),
        "Trades": n_trades,
        "WinRate": round(win_rate * 100, 1),
        "ProfitFactor": round(pf, 3),
        "AvgReturn": round(avg_ret * 100, 3),
        "Exposure": round(exposure * 100, 1),
    }


def run_backtest_with_predictions(pred_path, buy_threshold, experiment_name):
    """Write a small C++ runner isn't needed — we can manipulate predictions.csv
    and re-run the existing executable. But the current main() uses MomentumStrategy,
    not predictions. Instead, we'll simulate by filtering predictions and computing
    metrics from the walk-forward predictions + forward returns directly."""
    pass


def simulate_from_predictions(pred_path, threshold, df_dataset, label_name):
    """Simulate trading: buy when probability >= threshold, hold for 20 days.
    Uses forward_return from the dataset as the actual trade outcome.
    This avoids needing to re-run C++ for each experiment.
    """
    preds = pd.read_csv(pred_path)
    preds["date"] = pd.to_datetime(preds["date"])

    # Filter to MODERN regime (2020+)
    preds = preds[preds["date"] >= "2020-01-01"].copy()

    # Merge with forward_return from dataset
    dataset = df_dataset[["date", "ticker", "forward_return"]].copy()
    merged = preds.merge(dataset, on=["date", "ticker"], how="inner")

    # Signal: probability >= threshold
    signals = merged[merged["probability"] >= threshold].copy()

    if len(signals) == 0:
        return {
            "experiment": f"{label_name}_t{threshold}",
            "threshold": threshold,
            "trades": 0, "mean_return": 0, "median_return": 0,
            "win_rate": 0, "sharpe_trades": 0, "total_return": 0,
        }

    rets = signals["forward_return"]
    wins = (rets > 0).sum()

    result = {
        "experiment": f"{label_name}_t{threshold}",
        "threshold": threshold,
        "trades": len(signals),
        "mean_return": round(rets.mean() * 100, 3),
        "median_return": round(rets.median() * 100, 3),
        "win_rate": round(wins / len(signals) * 100, 1),
        "sharpe_trades": round(rets.mean() / rets.std() * np.sqrt(252/20), 4) if rets.std() > 0 else 0,
        "total_return": round((1 + rets).prod() - 1, 4),
    }
    return result


def main():
    print("=" * 70)
    print("  BACKTEST COMPARISON: Does Better AUC -> Better Trading?")
    print("=" * 70)

    # Load dataset for forward returns
    df = pd.read_csv(PROJECT_DIR / "output" / "ml_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Generate target_stop predictions (logistic, same pipeline)
    ts_path = generate_target_stop_predictions()

    # median_return predictions already at predictions.csv
    mr_path = PREDICTIONS_PATH

    results = []
    thresholds = [0.50, 0.55, 0.60, 0.65]

    # Baseline: always buy (threshold=0, every day is a signal)
    # Use forward_return of all dates in 2020+
    modern = df[df["date"] >= "2020-01-01"]
    baseline_rets = modern["forward_return"].dropna()
    if len(baseline_rets) > 0:
        results.append({
            "experiment": "buy_always",
            "threshold": 0.0,
            "trades": len(baseline_rets),
            "mean_return": round(baseline_rets.mean() * 100, 3),
            "median_return": round(baseline_rets.median() * 100, 3),
            "win_rate": round((baseline_rets > 0).mean() * 100, 1),
            "sharpe_trades": round(baseline_rets.mean() / baseline_rets.std() * np.sqrt(252/20), 4) if baseline_rets.std() > 0 else 0,
            "total_return": round((1 + baseline_rets).prod() - 1, 4),
        })

    # Target-stop predictions at various thresholds
    for t in thresholds:
        results.append(simulate_from_predictions(ts_path, t, df, "target_stop"))

    # Median-return predictions at various thresholds
    for t in thresholds:
        results.append(simulate_from_predictions(mr_path, t, df, "median_return"))

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS (MODERN regime, 2020+)")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print(f"\n{'Experiment':<25} {'Thresh':>6} {'Trades':>7} {'MeanRet%':>9} {'MedRet%':>8} "
          f"{'WinRate%':>9} {'Sharpe':>8} {'TotalRet':>10}")
    print("-" * 93)
    for _, r in results_df.iterrows():
        print(f"{r['experiment']:<25} {r['threshold']:>6.2f} {r['trades']:>7} "
              f"{r['mean_return']:>9.3f} {r['median_return']:>8.3f} "
              f"{r['win_rate']:>9.1f} {r['sharpe_trades']:>8.4f} "
              f"{r['total_return']:>10.4f}")

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Compare median_return at 0.55 vs target_stop at 0.55
    mr_55 = results_df[results_df["experiment"] == "median_return_t0.55"]
    ts_55 = results_df[results_df["experiment"] == "target_stop_t0.55"]
    baseline = results_df[results_df["experiment"] == "buy_always"]

    if len(mr_55) > 0 and len(ts_55) > 0:
        print(f"\n  median_return (t=0.55): mean_ret={mr_55.iloc[0]['mean_return']:.3f}%, "
              f"win_rate={mr_55.iloc[0]['win_rate']:.1f}%, sharpe={mr_55.iloc[0]['sharpe_trades']:.4f}")
        print(f"  target_stop (t=0.55):   mean_ret={ts_55.iloc[0]['mean_return']:.3f}%, "
              f"win_rate={ts_55.iloc[0]['win_rate']:.1f}%, sharpe={ts_55.iloc[0]['sharpe_trades']:.4f}")
    if len(baseline) > 0:
        print(f"  buy_always:             mean_ret={baseline.iloc[0]['mean_return']:.3f}%, "
              f"win_rate={baseline.iloc[0]['win_rate']:.1f}%, sharpe={baseline.iloc[0]['sharpe_trades']:.4f}")

    print(f"\n  Does better AUC translate to better trading?")
    if len(mr_55) > 0 and len(ts_55) > 0:
        if mr_55.iloc[0]["mean_return"] > ts_55.iloc[0]["mean_return"]:
            print(f"    YES - median_return label produces higher mean trade return.")
        else:
            print(f"    NO - target_stop label still produces higher mean trade return.")

    if len(mr_55) > 0 and len(baseline) > 0:
        if mr_55.iloc[0]["mean_return"] > baseline.iloc[0]["mean_return"]:
            print(f"    YES - ML signal beats buy-always baseline.")
        else:
            print(f"    NO - ML signal does not beat buy-always baseline.")

    # Save
    out_path = PROJECT_DIR / "output" / "backtest_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
