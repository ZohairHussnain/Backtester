"""
Research Phases 4-6: Top-N Ranking, Probability Calibration, Portfolio Backtesting

Uses existing out-of-sample predictions. No retraining. No new models/features/labels.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PREDICTIONS_ALL = PROJECT_DIR / "output" / "predictions_all.csv"
DATASET_PATH = PROJECT_DIR / "output" / "ml_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "output"

EXPORT_MODEL = "logistic"
STARTING_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01
MAX_POSITION_FRAC = 0.40
STOP_LOSS_PCT = 0.05
TARGET_PROFIT_PCT = 0.10
MAX_HOLD_DAYS = 20
SLIPPAGE = 0.0005
FEE_PER_SHARE = 0.0035
FEE_MIN = 0.35
FEE_CAP_PCT = 0.01


def load_data():
    preds = pd.read_csv(PREDICTIONS_ALL)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model"] == EXPORT_MODEL].copy()

    ds = pd.read_csv(DATASET_PATH)
    ds["date"] = pd.to_datetime(ds["date"])

    merged = preds.merge(
        ds[["date", "ticker", "forward_return", "label_median_return"]],
        on=["date", "ticker"], how="inner"
    )
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
    merged["year"] = merged["date"].dt.year
    # Filter to 2015+ for modern evaluation
    merged = merged[merged["year"] >= 2015].reset_index(drop=True)
    return merged


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ======================================================================
# PHASE 4: TOP-N RANKING
# ======================================================================

def phase4_topn(df):
    section("PHASE 4: TOP-N RANKING RESEARCH")

    thresholds = [0.50, 0.55, 0.60]
    top_ns = [1, 2, 3, 5, 999]  # 999 = all above threshold

    results = []
    for thresh in thresholds:
        above = df[df["probability"] >= thresh].copy()

        for top_n in top_ns:
            label = f"top{top_n}" if top_n < 999 else "all"

            # Per-date ranking: take top N by probability.
            # Use date string as groupby key to preserve the date column.
            above["_date_str"] = above["date"].astype(str)
            selected = (above.groupby("_date_str", group_keys=False)
                        .apply(lambda g: g.nlargest(top_n, "probability"))
                        .reset_index(drop=True))
            above.drop(columns=["_date_str"], inplace=True)
            selected.drop(columns=["_date_str"], inplace=True, errors="ignore")

            if len(selected) == 0:
                results.append({
                    "threshold": thresh, "top_n": label, "trades": 0,
                    "mean_ret": 0, "median_ret": 0, "win_rate": 0,
                    "sharpe": 0, "avg_prob": 0, "signal_days": 0,
                })
                continue

            rets = selected["forward_return"]
            wins = (rets > 0).sum()
            sharpe = (rets.mean() / rets.std() * np.sqrt(252/MAX_HOLD_DAYS)
                      if rets.std() > 0 else 0)
            signal_days = selected["date"].nunique()

            results.append({
                "threshold": thresh, "top_n": label,
                "trades": len(selected),
                "mean_ret": round(rets.mean() * 100, 3),
                "median_ret": round(rets.median() * 100, 3),
                "win_rate": round(wins / len(selected) * 100, 1),
                "sharpe": round(sharpe, 4),
                "avg_prob": round(selected["probability"].mean(), 4),
                "signal_days": signal_days,
            })

    results_df = pd.DataFrame(results)
    print(f"\n{'Thresh':>6} {'TopN':>5} {'Trades':>7} {'MeanR%':>8} {'MedR%':>7} "
          f"{'WinR%':>6} {'Sharpe':>8} {'AvgProb':>8} {'Days':>5}")
    print("-" * 72)
    for _, r in results_df.iterrows():
        print(f"{r['threshold']:>6.2f} {r['top_n']:>5} {r['trades']:>7} "
              f"{r['mean_ret']:>8.3f} {r['median_ret']:>7.3f} "
              f"{r['win_rate']:>6.1f} {r['sharpe']:>8.4f} "
              f"{r['avg_prob']:>8.4f} {r['signal_days']:>5}")

    # Best by year for top picks
    print(f"\nTop-1 @ 0.55 by year:")
    above55 = df[df["probability"] >= 0.55].copy()
    above55["_date_str"] = above55["date"].astype(str)
    top1 = (above55.groupby("_date_str", group_keys=False)
            .apply(lambda g: g.nlargest(1, "probability"))
            .reset_index(drop=True))
    above55.drop(columns=["_date_str"], inplace=True)
    top1.drop(columns=["_date_str"], inplace=True, errors="ignore")
    if len(top1) > 0:
        yearly = top1.groupby("year")["forward_return"].agg(["mean", "count", "std"])
        yearly["sharpe"] = yearly["mean"] / yearly["std"] * np.sqrt(252/MAX_HOLD_DAYS)
        print(yearly.round(4).to_string())

    return results_df


# ======================================================================
# PHASE 5: PROBABILITY CALIBRATION
# ======================================================================

def phase5_calibration(df):
    section("PHASE 5: PROBABILITY CALIBRATION")

    bins = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
            (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]

    print(f"\n{'Bin':<12} {'Count':>6} {'AvgPred':>8} {'ActRate':>8} {'Gap':>8} "
          f"{'MeanRet%':>9} {'MedRet%':>8} {'WinRate%':>9} {'StdRet%':>8}")
    print("-" * 87)

    bin_preds, bin_actuals = [], []
    for lo, hi in bins:
        mask = (df["probability"] >= lo) & (df["probability"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue

        avg_pred = sub["probability"].mean()
        act_rate = sub["label_median_return"].mean()
        gap = act_rate - avg_pred
        mean_ret = sub["forward_return"].mean() * 100
        med_ret = sub["forward_return"].median() * 100
        win_rate = (sub["forward_return"] > 0).mean() * 100
        std_ret = sub["forward_return"].std() * 100

        label = f"[{lo:.2f},{hi:.2f})"
        print(f"{label:<12} {len(sub):>6} {avg_pred:>8.4f} {act_rate:>8.4f} {gap:>+8.4f} "
              f"{mean_ret:>9.3f} {med_ret:>8.3f} {win_rate:>9.1f} {std_ret:>8.3f}")

        bin_preds.append(avg_pred)
        bin_actuals.append(act_rate)

    # Overall calibration metrics
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(df["label_median_return"], df["probability"])
    print(f"\nBrier score: {brier:.4f}")

    # Expected Calibration Error
    if bin_preds:
        bin_preds = np.array(bin_preds)
        bin_actuals = np.array(bin_actuals)
        ece = np.mean(np.abs(bin_actuals - bin_preds))
        print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # Monotonicity check
    print(f"\nMonotonicity check (does higher prob -> higher return?):")
    for i in range(len(bins)):
        lo, hi = bins[i]
        mask = (df["probability"] >= lo) & (df["probability"] < hi)
        sub = df[mask]
        if len(sub) > 0:
            print(f"  [{lo:.2f},{hi:.2f}): mean_ret={sub['forward_return'].mean()*100:+.3f}%, "
                  f"n={len(sub)}")

    # Calibration plot
    try:
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(
            df["label_median_return"], df["probability"],
            n_bins=8, strategy="quantile")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(mean_pred, frac_pos, "s-", label=f"{EXPORT_MODEL}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve (out-of-sample)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = OUTPUT_DIR / "calibration_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nCalibration plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"\nCalibration plot failed: {e}")


# ======================================================================
# PHASE 6: PORTFOLIO-LEVEL BACKTESTING
# ======================================================================

def simulate_portfolio(df, threshold, top_n, max_positions, starting_capital):
    """Portfolio simulator using forward_return as the realized trade outcome.

    Each trade: enter at normalized price, exit after MAX_HOLD_DAYS at
    entry_price * (1 + forward_return), with slippage and fees on both sides.
    Position sizing uses risk-per-trade and max-position-fraction of cash.
    """
    dates = sorted(df["date"].unique())

    cash = starting_capital
    positions = {}  # ticker -> {shares, entry_price, entry_date, entry_idx, fwd_ret, entry_fee}
    equity_curve = []
    trades = []
    daily_dates = []

    for d_idx, date in enumerate(dates):
        # --- Exits: close positions held for MAX_HOLD_DAYS ---
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            bars_held = d_idx - pos["entry_idx"]

            if bars_held >= MAX_HOLD_DAYS:
                # Exit price = entry * (1 + forward_return) * (1 - slippage)
                exit_price = pos["entry_price"] * (1 + pos["fwd_ret"]) * (1 - SLIPPAGE)
                value = pos["shares"] * exit_price
                fee = max(min(pos["shares"] * FEE_PER_SHARE, FEE_CAP_PCT * value), FEE_MIN)
                cash += value - fee
                pnl = (exit_price - pos["entry_price"]) * pos["shares"] - fee - pos["entry_fee"]
                trades.append({
                    "ticker": ticker, "entry_date": pos["entry_date"],
                    "exit_date": date, "entry_price": pos["entry_price"],
                    "exit_price": exit_price, "shares": pos["shares"],
                    "pnl": pnl, "return_pct": exit_price / pos["entry_price"] - 1,
                    "bars_held": bars_held, "exit_reason": "max_hold",
                })
                del positions[ticker]

        # --- Entries: use previous day's prediction, enter today ---
        if d_idx > 0:
            prev_date = dates[d_idx - 1]
            prev_data = df[df["date"] == prev_date]

            candidates = prev_data[prev_data["probability"] >= threshold].copy()
            candidates = candidates[~candidates["ticker"].isin(positions.keys())]
            candidates = candidates.nlargest(top_n, "probability")

            for _, cand in candidates.iterrows():
                if len(positions) >= max_positions:
                    break

                entry_price = 100.0 * (1 + SLIPPAGE)  # normalized entry + slippage

                portfolio_value = cash
                for p in positions.values():
                    # Mark existing positions at estimated current value
                    hold_frac = min((d_idx - p["entry_idx"]) / MAX_HOLD_DAYS, 1.0)
                    current_val = p["entry_price"] * (1 + p["fwd_ret"] * hold_frac)
                    portfolio_value += p["shares"] * current_val

                risk_dollars = portfolio_value * RISK_PER_TRADE
                stop_distance = entry_price * STOP_LOSS_PCT
                if stop_distance <= 0:
                    continue

                shares_by_risk = risk_dollars / stop_distance
                shares_by_cash = (cash * MAX_POSITION_FRAC) / entry_price
                shares = min(shares_by_risk, shares_by_cash)

                fee = max(min(shares * FEE_PER_SHARE, FEE_CAP_PCT * shares * entry_price), FEE_MIN)
                cost = shares * entry_price + fee

                if cost > cash * MAX_POSITION_FRAC or shares <= 0:
                    continue

                cash -= cost
                positions[cand["ticker"]] = {
                    "shares": shares, "entry_price": entry_price,
                    "entry_date": date, "entry_idx": d_idx,
                    "fwd_ret": cand["forward_return"],
                    "entry_fee": fee,
                }

        # --- Mark to market ---
        equity = cash
        for ticker, pos in positions.items():
            hold_frac = min((d_idx - pos["entry_idx"]) / MAX_HOLD_DAYS, 1.0)
            current_val = pos["entry_price"] * (1 + pos["fwd_ret"] * hold_frac)
            equity += pos["shares"] * current_val
        equity_curve.append(equity)
        daily_dates.append(date)

    # Force-close remaining positions
    for ticker in list(positions.keys()):
        pos = positions[ticker]
        exit_price = pos["entry_price"] * (1 + pos["fwd_ret"]) * (1 - SLIPPAGE)
        value = pos["shares"] * exit_price
        fee = max(min(pos["shares"] * FEE_PER_SHARE, FEE_CAP_PCT * value), FEE_MIN)
        cash += value - fee
        pnl = (exit_price - pos["entry_price"]) * pos["shares"] - fee - pos["entry_fee"]
        trades.append({
            "ticker": ticker, "entry_date": pos["entry_date"],
            "exit_date": daily_dates[-1], "entry_price": pos["entry_price"],
            "exit_price": exit_price, "shares": pos["shares"],
            "pnl": pnl, "return_pct": exit_price / pos["entry_price"] - 1,
            "bars_held": len(dates) - pos["entry_idx"],
            "exit_reason": "final_close",
        })

    return pd.DataFrame(trades), np.array(equity_curve), daily_dates


def compute_portfolio_metrics(trades_df, equity, dates, starting_capital):
    if len(equity) < 2:
        return {}

    days = len(equity)
    years = days / 252.0

    # Returns
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]

    # CAGR
    final = equity[-1]
    cagr = (final / starting_capital) ** (1 / years) - 1 if starting_capital > 0 and final > 0 else 0

    # Sharpe
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) - 0.04/252) / np.std(rets) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = (np.mean(rets) - 0.04/252) / np.std(downside) * np.sqrt(252)
    else:
        sortino = 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Volatility
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0

    # Trade metrics
    n_trades = len(trades_df)
    if n_trades > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(wins) / n_trades
        pf = wins["pnl"].sum() / (-losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() < 0 else float("inf")
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
        avg_hold = trades_df["bars_held"].mean()
    else:
        win_rate = pf = avg_win = avg_loss = avg_hold = 0

    # Yearly returns
    date_series = pd.Series(equity, index=pd.DatetimeIndex(dates))
    yearly = date_series.resample("YE").last().pct_change().dropna()

    return {
        "CAGR%": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4),
        "MaxDD%": round(max_dd * 100, 2),
        "Vol%": round(vol * 100, 2),
        "TotalRet%": round((final / starting_capital - 1) * 100, 2),
        "Trades": n_trades,
        "WinRate%": round(win_rate * 100, 1),
        "ProfitFactor": round(pf, 3),
        "AvgWin": round(avg_win, 2),
        "AvgLoss": round(avg_loss, 2),
        "AvgHold": round(avg_hold, 1),
        "FinalEquity": round(final, 2),
        "BestYear%": round(yearly.max() * 100, 2) if len(yearly) > 0 else 0,
        "WorstYear%": round(yearly.min() * 100, 2) if len(yearly) > 0 else 0,
    }


def spy_buy_and_hold(df, starting_capital):
    """Compute SPY buy-and-hold benchmark from forward returns."""
    spy = df[df["ticker"] == "SPY"].sort_values("date").drop_duplicates("date")
    if len(spy) == 0:
        return None, None, None

    # Reconstruct price from cumulative returns
    price = np.cumprod(1 + spy["forward_return"].clip(-0.5, 5).fillna(0).values / MAX_HOLD_DAYS)
    price = price / price[0] * starting_capital
    return None, price, spy["date"].values


def phase6_portfolio(df):
    section("PHASE 6: PORTFOLIO-LEVEL BACKTESTING")

    configs = [
        {"threshold": 0.55, "top_n": 1, "max_pos": 1, "label": "t55_top1_pos1"},
        {"threshold": 0.55, "top_n": 2, "max_pos": 2, "label": "t55_top2_pos2"},
        {"threshold": 0.55, "top_n": 3, "max_pos": 3, "label": "t55_top3_pos3"},
        {"threshold": 0.60, "top_n": 1, "max_pos": 1, "label": "t60_top1_pos1"},
        {"threshold": 0.60, "top_n": 2, "max_pos": 2, "label": "t60_top2_pos2"},
        {"threshold": 0.50, "top_n": 3, "max_pos": 3, "label": "t50_top3_pos3"},
        {"threshold": 0.55, "top_n": 999, "max_pos": 3, "label": "t55_all_pos3"},
    ]

    all_metrics = []
    for cfg in configs:
        trades_df, equity, dates = simulate_portfolio(
            df, cfg["threshold"], cfg["top_n"], cfg["max_pos"], STARTING_CAPITAL)
        metrics = compute_portfolio_metrics(trades_df, equity, dates, STARTING_CAPITAL)
        metrics["config"] = cfg["label"]
        all_metrics.append(metrics)

    results_df = pd.DataFrame(all_metrics)

    print(f"\n{'Config':<18} {'CAGR%':>7} {'Sharpe':>7} {'Sort':>7} {'MaxDD%':>7} "
          f"{'Trades':>6} {'WR%':>5} {'PF':>6} {'Final$':>9}")
    print("-" * 78)
    for _, r in results_df.iterrows():
        print(f"{r['config']:<18} {r['CAGR%']:>7.2f} {r['Sharpe']:>7.4f} "
              f"{r['Sortino']:>7.4f} {r['MaxDD%']:>7.2f} "
              f"{r['Trades']:>6} {r['WinRate%']:>5.1f} "
              f"{r['ProfitFactor']:>6.3f} {r['FinalEquity']:>9.2f}")

    # Full detail
    print(f"\nFull metrics:")
    for _, r in results_df.iterrows():
        print(f"\n  {r['config']}:")
        for k, v in r.items():
            if k != "config":
                print(f"    {k}: {v}")

    return results_df


# ======================================================================
# FINAL RECOMMENDATION
# ======================================================================

def final_recommendation(topn_df, portfolio_df, df):
    section("FINAL RECOMMENDATION")

    # Best config by Sharpe
    if len(portfolio_df) > 0:
        best = portfolio_df.loc[portfolio_df["Sharpe"].idxmax()]
        print(f"\n  Best portfolio config by Sharpe: {best['config']}")
        print(f"    CAGR: {best['CAGR%']:.2f}%")
        print(f"    Sharpe: {best['Sharpe']:.4f}")
        print(f"    Sortino: {best['Sortino']:.4f}")
        print(f"    Max Drawdown: {best['MaxDD%']:.2f}%")
        print(f"    Trades: {int(best['Trades'])}")
        print(f"    Win Rate: {best['WinRate%']:.1f}%")
        print(f"    Final Equity: ${best['FinalEquity']:.2f}")

    # Trade-level best from Phase 4
    if len(topn_df) > 0:
        viable = topn_df[(topn_df["trades"] >= 50) & (topn_df["sharpe"] > 0)]
        if len(viable) > 0:
            best_trade = viable.loc[viable["sharpe"].idxmax()]
            print(f"\n  Best trade-level config (>=50 trades):")
            print(f"    {best_trade['threshold']:.2f} threshold, top {best_trade['top_n']}")
            print(f"    Trade Sharpe: {best_trade['sharpe']:.4f}")
            print(f"    Mean Return: {best_trade['mean_ret']:.3f}%")
            print(f"    Win Rate: {best_trade['win_rate']:.1f}%")

    print(f"\n  Ready for simulated paper trading?")
    if len(portfolio_df) > 0 and portfolio_df["Sharpe"].max() > 0:
        print(f"    The model produces positive risk-adjusted returns after costs.")
        print(f"    Suitable for paper trading with position limits.")
    else:
        print(f"    Insufficient evidence for paper trading.")

    print(f"\n  Before IBKR paper execution:")
    print(f"    - Implement real-time prediction pipeline")
    print(f"    - Add order execution with market-on-open orders")
    print(f"    - Add position tracking and reconciliation")
    print(f"    - Set hard daily loss limit")
    print(f"    - Run paper for minimum 3 months before live")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  RESEARCH PHASES 4-6: Portfolio Construction & Evaluation")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} predictions ({EXPORT_MODEL}), {df['date'].dt.year.min()}-{df['date'].dt.year.max()}")
    print(f"Tickers: {sorted(df['ticker'].unique())}")

    topn_df = phase4_topn(df)
    phase5_calibration(df)
    portfolio_df = phase6_portfolio(df)
    final_recommendation(topn_df, portfolio_df, df)

    # Save
    topn_df.to_csv(OUTPUT_DIR / "topn_results.csv", index=False)
    portfolio_df.to_csv(OUTPUT_DIR / "portfolio_results.csv", index=False)
    print(f"\nResults saved to output/topn_results.csv and output/portfolio_results.csv")


if __name__ == "__main__":
    main()
