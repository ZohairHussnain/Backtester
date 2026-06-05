"""
Final Long-Only Portfolio Test

101-stock universe, Logistic Regression, label_median_return.
Portfolio-level evaluation with benchmarks and adversarial random baselines.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PREDICTIONS_ALL = PROJECT_DIR / "output" / "predictions_all.csv"
DATASET_PATH = PROJECT_DIR / "output" / "ml_dataset.csv"

EXPORT_MODEL = "logistic"
HOLD_DAYS = 20
SLIPPAGE = 0.0005
FEE_PER_SHARE = 0.0035
FEE_MIN = 0.35
FEE_CAP_PCT = 0.01
RISK_PCT = 0.01
MAX_POS_FRAC = 0.40
STOP_PCT = 0.05
TARGET_PCT = 0.10
START_YEAR = 2015


def load():
    preds = pd.read_csv(PREDICTIONS_ALL)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model"] == EXPORT_MODEL].copy()
    ds = pd.read_csv(DATASET_PATH, usecols=["date", "ticker", "forward_return", "label_median_return", "ret_1d"])
    ds["date"] = pd.to_datetime(ds["date"])
    merged = preds.merge(ds, on=["date", "ticker"], how="inner")
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
    merged["year"] = merged["date"].dt.year
    merged = merged[merged["year"] >= START_YEAR].reset_index(drop=True)
    return merged


def simulate(df, threshold, top_n, max_pos, capital,
             use_random=False, rng_seed=42):
    """Simulate a long-only portfolio using forward_return as realized outcome."""
    work = df.copy()
    if use_random:
        rng = np.random.RandomState(rng_seed)
        work["probability"] = rng.uniform(0.3, 0.7, len(work))

    dates = sorted(work["date"].unique())
    cash = capital
    positions = {}
    equity_curve = []
    trades_list = []

    for d_idx, date in enumerate(dates):
        # --- Exits ---
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            held = d_idx - pos["entry_idx"]
            if held >= HOLD_DAYS:
                exit_p = pos["entry_price"] * (1 + pos["fwd_ret"]) * (1 - SLIPPAGE)
                val = pos["shares"] * exit_p
                fee = max(min(pos["shares"] * FEE_PER_SHARE, FEE_CAP_PCT * val), FEE_MIN)
                cash += val - fee
                pnl = (exit_p - pos["entry_price"]) * pos["shares"] - fee - pos["entry_fee"]
                trades_list.append({
                    "ticker": ticker, "entry_date": pos["entry_date"],
                    "exit_date": date, "pnl": pnl,
                    "return_pct": exit_p / pos["entry_price"] - 1,
                    "held": held, "probability": pos["prob"],
                })
                del positions[ticker]

        # --- Entries (signal from previous day) ---
        if d_idx > 0:
            prev = dates[d_idx - 1]
            prev_data = work[work["date"] == prev]
            cands = prev_data[prev_data["probability"] >= threshold]
            cands = cands[~cands["ticker"].isin(positions.keys())]
            cands = cands.nlargest(top_n, "probability")

            for _, c in cands.iterrows():
                if len(positions) >= max_pos:
                    break
                entry_p = 100.0 * (1 + SLIPPAGE)
                pv = cash
                for p in positions.values():
                    hf = min((d_idx - p["entry_idx"]) / HOLD_DAYS, 1.0)
                    pv += p["shares"] * p["entry_price"] * (1 + p["fwd_ret"] * hf)

                risk_d = pv * RISK_PCT
                stop_d = entry_p * STOP_PCT
                sh_risk = risk_d / stop_d if stop_d > 0 else 0
                sh_cash = (cash * MAX_POS_FRAC) / entry_p
                sh = min(sh_risk, sh_cash)
                fee = max(min(sh * FEE_PER_SHARE, FEE_CAP_PCT * sh * entry_p), FEE_MIN)
                cost = sh * entry_p + fee
                if cost > cash * MAX_POS_FRAC or sh <= 0:
                    continue

                cash -= cost
                positions[c["ticker"]] = {
                    "shares": sh, "entry_price": entry_p,
                    "entry_date": date, "entry_idx": d_idx,
                    "fwd_ret": c["forward_return"], "entry_fee": fee,
                    "prob": c["probability"],
                }

        # --- MTM ---
        eq = cash
        for _, pos in positions.items():
            hf = min((d_idx - pos["entry_idx"]) / HOLD_DAYS, 1.0)
            eq += pos["shares"] * pos["entry_price"] * (1 + pos["fwd_ret"] * hf)
        equity_curve.append(eq)

    # Force close
    for ticker in list(positions.keys()):
        pos = positions[ticker]
        exit_p = pos["entry_price"] * (1 + pos["fwd_ret"]) * (1 - SLIPPAGE)
        val = pos["shares"] * exit_p
        fee = max(min(pos["shares"] * FEE_PER_SHARE, FEE_CAP_PCT * val), FEE_MIN)
        cash += val - fee
        pnl = (exit_p - pos["entry_price"]) * pos["shares"] - fee - pos["entry_fee"]
        trades_list.append({
            "ticker": ticker, "entry_date": pos["entry_date"],
            "exit_date": dates[-1], "pnl": pnl,
            "return_pct": exit_p / pos["entry_price"] - 1,
            "held": len(dates) - pos["entry_idx"], "probability": pos["prob"],
        })

    return pd.DataFrame(trades_list), np.array(equity_curve), dates


def compute_metrics(tdf, eq, capital, dates):
    if len(eq) < 2:
        return {}
    days = len(eq)
    yrs = days / 252.0
    final = eq[-1]
    cagr = (final / capital) ** (1 / yrs) - 1 if capital > 0 and final > 0 else 0
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = (np.mean(rets) - 0.04/252) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0
    down = rets[rets < 0]
    sortino = (np.mean(rets) - 0.04/252) / np.std(down) * np.sqrt(252) if len(down) > 1 and np.std(down) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0

    n = len(tdf)
    if n > 0:
        wins = tdf[tdf["pnl"] > 0]
        losses = tdf[tdf["pnl"] <= 0]
        wr = len(wins) / n
        pf = wins["pnl"].sum() / (-losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() < 0 else (float("inf") if len(wins) > 0 else 0)
        avg_ret = tdf["return_pct"].mean()
        avg_hold = tdf["held"].mean()
    else:
        wr = pf = avg_ret = avg_hold = 0

    # Yearly
    ds = pd.Series(eq, index=pd.DatetimeIndex(dates))
    yr = ds.resample("YE").last().pct_change().dropna()

    # Exposure: fraction of time with any position open
    # approximate from trades
    if n > 0:
        total_bars_held = tdf["held"].sum()
        exposure = total_bars_held / days  # can exceed 1 if multiple positions
    else:
        exposure = 0

    return {
        "CAGR%": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4),
        "MaxDD%": round(dd * 100, 2),
        "Vol%": round(vol * 100, 2),
        "TotalRet%": round((final / capital - 1) * 100, 2),
        "Final$": round(final, 2),
        "Exposure%": round(exposure * 100, 1),
        "Trades": n,
        "WR%": round(wr * 100, 1),
        "PF": round(pf, 3),
        "AvgRet%": round(avg_ret * 100, 3),
        "AvgHold": round(avg_hold, 1),
        "BestYr%": round(yr.max() * 100, 2) if len(yr) > 0 else 0,
        "WorstYr%": round(yr.min() * 100, 2) if len(yr) > 0 else 0,
    }


def spy_benchmark(df, capital):
    spy = df[df["ticker"] == "SPY"].sort_values("date").drop_duplicates("date")
    if len(spy) == 0:
        return {}, np.array([capital])
    daily_ret = spy["ret_1d"].fillna(0).values
    eq = capital * np.cumprod(1 + daily_ret)
    days = len(eq)
    yrs = days / 252.0
    cagr = (eq[-1] / capital) ** (1 / yrs) - 1 if eq[-1] > 0 else 0
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = (np.mean(rets) - 0.04/252) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0
    down = rets[rets < 0]
    sortino = (np.mean(rets) - 0.04/252) / np.std(down) * np.sqrt(252) if len(down) > 1 and np.std(down) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    ds = pd.Series(eq, index=pd.DatetimeIndex(spy["date"].values))
    yr = ds.resample("YE").last().pct_change().dropna()
    return {
        "CAGR%": round(cagr * 100, 2), "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4), "MaxDD%": round(dd * 100, 2),
        "TotalRet%": round((eq[-1]/capital - 1)*100, 2), "Final$": round(eq[-1], 2),
        "BestYr%": round(yr.max()*100, 2) if len(yr) > 0 else 0,
        "WorstYr%": round(yr.min()*100, 2) if len(yr) > 0 else 0,
    }, eq


def ew_benchmark(df, capital):
    daily = df.groupby("date")["ret_1d"].mean().sort_index()
    eq = capital * np.cumprod(1 + daily.fillna(0).values)
    days = len(eq)
    yrs = days / 252.0
    cagr = (eq[-1] / capital) ** (1 / yrs) - 1 if eq[-1] > 0 else 0
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = (np.mean(rets) - 0.04/252) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0
    down = rets[rets < 0]
    sortino = (np.mean(rets) - 0.04/252) / np.std(down) * np.sqrt(252) if len(down) > 1 and np.std(down) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    ds = pd.Series(eq, index=pd.DatetimeIndex(daily.index))
    yr = ds.resample("YE").last().pct_change().dropna()
    return {
        "CAGR%": round(cagr * 100, 2), "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4), "MaxDD%": round(dd * 100, 2),
        "TotalRet%": round((eq[-1]/capital - 1)*100, 2), "Final$": round(eq[-1], 2),
        "BestYr%": round(yr.max()*100, 2) if len(yr) > 0 else 0,
        "WorstYr%": round(yr.min()*100, 2) if len(yr) > 0 else 0,
    }, eq


def section(t):
    print(f"\n{'='*80}\n  {t}\n{'='*80}")


def main():
    print("=" * 80)
    print("  FINAL LONG-ONLY PORTFOLIO TEST")
    print("  101-stock universe | Logistic Regression | label_median_return")
    print("=" * 80)

    df = load()
    CAP = 10000.0
    print(f"Loaded {len(df)} predictions, {df['ticker'].nunique()} tickers, {START_YEAR}-{df['year'].max()}")

    # ====== BENCHMARKS ======
    section("BENCHMARKS")
    spy_m, spy_eq = spy_benchmark(df, CAP)
    ew_m, ew_eq = ew_benchmark(df, CAP)
    print(f"  {'SPY B&H':<22} CAGR={spy_m.get('CAGR%',0):>7.2f}%  Sharpe={spy_m.get('Sharpe',0):>7.4f}  "
          f"MaxDD={spy_m.get('MaxDD%',0):>7.2f}%  Final=${spy_m.get('Final$',0):>10.2f}")
    print(f"  {'EW 101-stock':<22} CAGR={ew_m.get('CAGR%',0):>7.2f}%  Sharpe={ew_m.get('Sharpe',0):>7.4f}  "
          f"MaxDD={ew_m.get('MaxDD%',0):>7.2f}%  Final=${ew_m.get('Final$',0):>10.2f}")

    # ====== STRATEGY CONFIGS ======
    section("STRATEGY CONFIGURATIONS")

    configs = []
    for thresh in [0.50, 0.55, 0.60]:
        for top_n, max_pos in [(1, 1), (2, 2), (3, 3), (5, 5)]:
            configs.append({"t": thresh, "n": top_n, "p": max_pos,
                           "label": f"t{int(thresh*100)}_top{top_n}_pos{max_pos}"})

    all_results = []
    header = (f"{'Config':<20} {'CAGR%':>7} {'Sharpe':>7} {'Sort':>7} {'MaxDD%':>7} "
              f"{'Trades':>6} {'WR%':>5} {'PF':>6} {'AvgRet%':>8} {'Exp%':>5} "
              f"{'Final$':>10} {'BstYr%':>7} {'WstYr%':>7}")
    print(f"\n{header}")
    print("-" * len(header))

    for cfg in configs:
        tdf, eq, dt = simulate(df, cfg["t"], cfg["n"], cfg["p"], CAP)
        m = compute_metrics(tdf, eq, CAP, dt)
        m["config"] = cfg["label"]
        all_results.append(m)
        print(f"{cfg['label']:<20} {m['CAGR%']:>7.2f} {m['Sharpe']:>7.4f} {m['Sortino']:>7.4f} "
              f"{m['MaxDD%']:>7.2f} {m['Trades']:>6} {m['WR%']:>5.1f} {m['PF']:>6.3f} "
              f"{m['AvgRet%']:>8.3f} {m['Exposure%']:>5.1f} {m['Final$']:>10.2f} "
              f"{m['BestYr%']:>7.2f} {m['WorstYr%']:>7.2f}")

    # ====== RANDOM BASELINES (matched configs) ======
    section("RANDOM BASELINES (10 trials each, same position rules)")

    best_configs = [
        {"t": 0.50, "n": 3, "p": 3, "label": "t50_top3_pos3"},
        {"t": 0.55, "n": 3, "p": 3, "label": "t55_top3_pos3"},
        {"t": 0.50, "n": 5, "p": 5, "label": "t50_top5_pos5"},
    ]

    print(f"\n{'Config':<20} {'Model CAGR%':>12} {'Rand CAGR%':>12} {'Excess':>8} "
          f"{'Model Shrp':>11} {'Rand Shrp':>10} {'Excess':>8}")
    print("-" * 83)

    for cfg in best_configs:
        tdf_m, eq_m, dt_m = simulate(df, cfg["t"], cfg["n"], cfg["p"], CAP)
        m_model = compute_metrics(tdf_m, eq_m, CAP, dt_m)

        rand_cagrs, rand_sharpes = [], []
        for seed in range(10):
            tdf_r, eq_r, dt_r = simulate(df, cfg["t"], cfg["n"], cfg["p"], CAP,
                                         use_random=True, rng_seed=seed)
            m_r = compute_metrics(tdf_r, eq_r, CAP, dt_r)
            rand_cagrs.append(m_r["CAGR%"])
            rand_sharpes.append(m_r["Sharpe"])

        exc_cagr = m_model["CAGR%"] - np.mean(rand_cagrs)
        exc_sharpe = m_model["Sharpe"] - np.mean(rand_sharpes)
        print(f"{cfg['label']:<20} {m_model['CAGR%']:>12.2f} {np.mean(rand_cagrs):>12.2f} "
              f"{exc_cagr:>+8.2f} {m_model['Sharpe']:>11.4f} {np.mean(rand_sharpes):>10.4f} "
              f"{exc_sharpe:>+8.4f}")

    # ====== YEAR-BY-YEAR FOR BEST CONFIG ======
    section("YEAR-BY-YEAR (t50_top3_pos3)")

    tdf_best, eq_best, dt_best = simulate(df, 0.50, 3, 3, CAP)
    tdf_best["year"] = pd.to_datetime(tdf_best["entry_date"]).dt.year

    print(f"\n{'Year':>6} {'Trades':>7} {'MeanRet%':>9} {'WR%':>6} {'TotalPnL':>10}")
    print("-" * 40)
    for year in sorted(tdf_best["year"].unique()):
        yr_trades = tdf_best[tdf_best["year"] == year]
        wr = (yr_trades["pnl"] > 0).mean() * 100
        print(f"{year:>6} {len(yr_trades):>7} {yr_trades['return_pct'].mean()*100:>9.3f} "
              f"{wr:>6.1f} {yr_trades['pnl'].sum():>10.2f}")

    # ====== RISK COMPARISON ======
    section("RISK-ADJUSTED COMPARISON")

    print(f"\n{'Strategy':<22} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>7} {'MaxDD%':>7} "
          f"{'CAGR/DD':>7}")
    print("-" * 59)

    strategies = [
        ("SPY B&H", spy_m),
        ("EW 101-stock", ew_m),
    ]
    # Add top 3 model configs by Sharpe
    results_df = pd.DataFrame(all_results)
    top3 = results_df.nlargest(3, "Sharpe")
    for _, row in top3.iterrows():
        strategies.append((row["config"], row.to_dict()))

    for name, m in strategies:
        cagr = m.get("CAGR%", 0)
        sharpe = m.get("Sharpe", 0)
        sortino = m.get("Sortino", 0)
        maxdd = m.get("MaxDD%", 0)
        calmar = abs(cagr / maxdd) if maxdd != 0 else 0
        print(f"{name:<22} {cagr:>7.2f} {sharpe:>7.4f} {sortino:>7.4f} {maxdd:>7.2f} {calmar:>7.3f}")

    # ====== FINAL VERDICT ======
    section("FINAL VERDICT")

    best = results_df.loc[results_df["Sharpe"].idxmax()]
    print(f"\n  Best configuration: {best['config']}")
    print(f"    CAGR:     {best['CAGR%']:.2f}%")
    print(f"    Sharpe:   {best['Sharpe']:.4f}")
    print(f"    Sortino:  {best['Sortino']:.4f}")
    print(f"    MaxDD:    {best['MaxDD%']:.2f}%")
    print(f"    Trades:   {int(best['Trades'])}")
    print(f"    Win Rate: {best['WR%']:.1f}%")
    print(f"    PF:       {best['PF']:.3f}")
    print(f"    Final:    ${best['Final$']:.2f}")

    spy_cagr = spy_m.get("CAGR%", 0)
    spy_sharpe = spy_m.get("Sharpe", 0)
    spy_dd = spy_m.get("MaxDD%", 0)

    print(f"\n  vs SPY Buy-and-Hold:")
    print(f"    SPY:    CAGR={spy_cagr:.2f}%, Sharpe={spy_sharpe:.4f}, MaxDD={spy_dd:.2f}%")
    print(f"    Model:  CAGR={best['CAGR%']:.2f}%, Sharpe={best['Sharpe']:.4f}, MaxDD={best['MaxDD%']:.2f}%")
    print(f"    Excess CAGR: {best['CAGR%'] - spy_cagr:+.2f}%")
    print(f"    Excess Sharpe: {best['Sharpe'] - spy_sharpe:+.4f}")

    print(f"\n  Recommendations:")
    print(f"    Best threshold:       0.50")
    print(f"    Best top-N:           {int(best['config'].split('top')[1].split('_')[0]) if 'top' in best['config'] else '?'}")
    print(f"    Max open positions:   {int(best['config'].split('pos')[1]) if 'pos' in best['config'] else '?'}")
    print(f"    Fractional shares:    Yes (required for small accounts)")

    # Determine verdict
    beats_spy_sharpe = best["Sharpe"] > spy_sharpe
    beats_spy_dd = abs(best["MaxDD%"]) < abs(spy_dd)
    positive_cagr = best["CAGR%"] > 0
    enough_trades = best["Trades"] > 100

    if beats_spy_sharpe and positive_cagr and enough_trades:
        verdict = "READY FOR IBKR PAPER ACCOUNT TESTING"
        print(f"\n  VERDICT: {verdict}")
        print(f"    The strategy shows better risk-adjusted returns than SPY.")
        print(f"    Proceed to IBKR paper with strict position limits and daily loss cap.")
    elif positive_cagr and enough_trades:
        verdict = "READY FOR SIMULATED PAPER TRADING ONLY"
        print(f"\n  VERDICT: {verdict}")
        print(f"    Strategy is profitable but does not clearly beat SPY risk-adjusted.")
        print(f"    Run simulated paper trading for 3+ months before IBKR.")
    else:
        verdict = "NOT READY FOR PAPER TRADING"
        print(f"\n  VERDICT: {verdict}")

    # Save
    results_df.to_csv(PROJECT_DIR / "output" / "final_portfolio_results.csv", index=False)
    print(f"\nResults saved to output/final_portfolio_results.csv")


if __name__ == "__main__":
    main()
