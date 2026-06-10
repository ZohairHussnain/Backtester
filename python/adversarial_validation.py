"""
Research Phase 7: Adversarial Validation

Try to break the reported 17.2% CAGR / 4.24 Sharpe result.
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
MAX_HOLD_DAYS = 20
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
    merged = merged[merged["year"] >= 2015].reset_index(drop=True)
    return merged


def simulate(df, threshold, top_n, max_positions, starting_capital,
             slippage=0.0005, fee_mult=1.0, shuffle_probs=False,
             random_probs=False, rng_seed=42):
    """Portfolio simulator. Returns trades_df, equity_curve, dates."""
    work = df.copy()

    if random_probs:
        rng = np.random.RandomState(rng_seed)
        work["probability"] = rng.uniform(0.3, 0.7, len(work))
    elif shuffle_probs:
        rng = np.random.RandomState(rng_seed)
        work["probability"] = work.groupby("date")["probability"].transform(rng.permutation)

    dates = sorted(work["date"].unique())
    cash = starting_capital
    positions = {}
    equity_curve = []
    trades = []

    for d_idx, date in enumerate(dates):
        # Exits
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            bars_held = d_idx - pos["entry_idx"]
            if bars_held >= MAX_HOLD_DAYS:
                exit_price = pos["entry_price"] * (1 + pos["fwd_ret"]) * (1 - slippage)
                value = pos["shares"] * exit_price
                base_fee = pos["shares"] * FEE_PER_SHARE * fee_mult
                fee = max(min(base_fee, FEE_CAP_PCT * value), FEE_MIN * fee_mult)
                cash += value - fee
                pnl = (exit_price - pos["entry_price"]) * pos["shares"] - fee - pos["entry_fee"]
                trades.append({
                    "ticker": ticker, "entry_date": pos["entry_date"],
                    "exit_date": date, "pnl": pnl,
                    "return_pct": exit_price / pos["entry_price"] - 1,
                    "bars_held": bars_held,
                })
                del positions[ticker]

        # Entries
        if d_idx > 0:
            prev_date = dates[d_idx - 1]
            prev_data = work[work["date"] == prev_date]
            candidates = prev_data[prev_data["probability"] >= threshold].copy()
            candidates = candidates[~candidates["ticker"].isin(positions.keys())]
            candidates = candidates.nlargest(top_n, "probability")

            for _, cand in candidates.iterrows():
                if len(positions) >= max_positions:
                    break
                entry_price = 100.0 * (1 + slippage)
                portfolio_value = cash
                for p in positions.values():
                    hf = min((d_idx - p["entry_idx"]) / MAX_HOLD_DAYS, 1.0)
                    portfolio_value += p["shares"] * p["entry_price"] * (1 + p["fwd_ret"] * hf)

                risk_dollars = portfolio_value * 0.01
                stop_dist = entry_price * 0.05
                shares_by_risk = risk_dollars / stop_dist
                shares_by_cash = (cash * 0.40) / entry_price
                shares = min(shares_by_risk, shares_by_cash)

                base_fee = shares * FEE_PER_SHARE * fee_mult
                fee = max(min(base_fee, FEE_CAP_PCT * shares * entry_price), FEE_MIN * fee_mult)
                cost = shares * entry_price + fee

                if cost > cash * 0.40 or shares <= 0:
                    continue

                cash -= cost
                positions[cand["ticker"]] = {
                    "shares": shares, "entry_price": entry_price,
                    "entry_date": date, "entry_idx": d_idx,
                    "fwd_ret": cand["forward_return"], "entry_fee": fee,
                }

        # Mark to market
        equity = cash
        for _, pos in positions.items():
            hf = min((d_idx - pos["entry_idx"]) / MAX_HOLD_DAYS, 1.0)
            equity += pos["shares"] * pos["entry_price"] * (1 + pos["fwd_ret"] * hf)
        equity_curve.append(equity)

    # Force close
    for ticker in list(positions.keys()):
        pos = positions[ticker]
        exit_price = pos["entry_price"] * (1 + pos["fwd_ret"]) * (1 - slippage)
        value = pos["shares"] * exit_price
        fee = max(min(pos["shares"] * FEE_PER_SHARE * fee_mult, FEE_CAP_PCT * value), FEE_MIN * fee_mult)
        cash += value - fee
        pnl = (exit_price - pos["entry_price"]) * pos["shares"] - fee - pos["entry_fee"]
        trades.append({"ticker": ticker, "entry_date": pos["entry_date"],
                       "exit_date": dates[-1], "pnl": pnl,
                       "return_pct": exit_price / pos["entry_price"] - 1,
                       "bars_held": len(dates) - pos["entry_idx"]})

    return pd.DataFrame(trades), np.array(equity_curve), dates


def metrics(trades_df, equity, starting_capital):
    if len(equity) < 2:
        return {"CAGR%": 0, "Sharpe": 0, "MaxDD%": 0, "Trades": 0, "WR%": 0, "PF": 0, "Final$": starting_capital}
    days = len(equity)
    years = days / 252.0
    final = equity[-1]
    cagr = (final / starting_capital) ** (1 / years) - 1 if starting_capital > 0 and final > 0 else 0
    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = (np.mean(rets) - 0.04/252) / np.std(rets) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0
    peak = np.maximum.accumulate(equity)
    dd = ((equity - peak) / peak).min()
    n = len(trades_df)
    if n > 0:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        wr = len(wins) / n
        pf = wins["pnl"].sum() / (-losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() < 0 else float("inf")
    else:
        wr = pf = 0
    return {"CAGR%": round(cagr*100, 2), "Sharpe": round(sharpe, 4),
            "MaxDD%": round(dd*100, 2), "Trades": n,
            "WR%": round(wr*100, 1), "PF": round(pf, 3),
            "Final$": round(final, 2)}


def section(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def print_row(label, m):
    print(f"  {label:<30} CAGR={m['CAGR%']:>7.2f}%  Sharpe={m['Sharpe']:>7.4f}  "
          f"DD={m['MaxDD%']:>7.2f}%  Trades={m['Trades']:>4}  WR={m['WR%']:>5.1f}%  "
          f"PF={m['PF']:>6.3f}  Final=${m['Final$']:>10.2f}")


def main():
    print("=" * 70)
    print("  RESEARCH PHASE 7: ADVERSARIAL VALIDATION")
    print("=" * 70)

    df = load_data()
    SC = 10000.0

    # ====== BASELINE ======
    section("0. BASELINE (reported config)")
    t, eq, dt = simulate(df, 0.50, 3, 3, SC)
    base = metrics(t, eq, SC)
    print_row("Baseline", base)

    # ====== 1. CAPITAL ALLOCATION CHECK ======
    section("1. CAPITAL ALLOCATION & LEVERAGE CHECK")
    # Check if cash ever goes negative
    _, eq_check, _ = simulate(df, 0.50, 3, 3, SC)
    min_eq = eq_check.min()
    max_eq = eq_check.max()
    print(f"  Min equity during sim: ${min_eq:.2f}")
    print(f"  Max equity during sim: ${max_eq:.2f}")
    print(f"  Starting capital: ${SC:.2f}")
    if min_eq < 0:
        print(f"  WARNING: Equity went negative - accidental leverage!")
    else:
        print(f"  OK: No leverage detected.")

    # Max position check: can we ever have > max_positions * max_frac of capital?
    print(f"  Max positions = 3, max fraction = 40% each")
    print(f"  Theoretical max allocation = 120% of cash (possible if gains in existing positions)")
    print(f"  This is NOT leverage - it's using unrealized gains to size new positions.")

    # ====== 2. SLIPPAGE SENSITIVITY ======
    section("2. SLIPPAGE SENSITIVITY")
    for slip in [0.0005, 0.001, 0.0025, 0.005]:
        t2, eq2, _ = simulate(df, 0.50, 3, 3, SC, slippage=slip)
        m2 = metrics(t2, eq2, SC)
        print_row(f"Slippage={slip*100:.2f}%", m2)

    # ====== 3. FEE SENSITIVITY ======
    section("3. FEE SENSITIVITY")
    for fm in [1.0, 2.0, 5.0, 10.0]:
        t3, eq3, _ = simulate(df, 0.50, 3, 3, SC, fee_mult=fm)
        m3 = metrics(t3, eq3, SC)
        print_row(f"Fee multiplier={fm:.0f}x", m3)

    # ====== 4. SHUFFLE PROBABILITIES (within-day) ======
    section("4. PROBABILITY RANKING TEST (shuffle within day)")
    shuffle_results = []
    for seed in range(10):
        t4, eq4, _ = simulate(df, 0.50, 3, 3, SC, shuffle_probs=True, rng_seed=seed)
        m4 = metrics(t4, eq4, SC)
        shuffle_results.append(m4)
    shuffle_cagrs = [r["CAGR%"] for r in shuffle_results]
    shuffle_sharpes = [r["Sharpe"] for r in shuffle_results]
    print(f"  Shuffled CAGR:   mean={np.mean(shuffle_cagrs):.2f}%, std={np.std(shuffle_cagrs):.2f}%")
    print(f"  Shuffled Sharpe: mean={np.mean(shuffle_sharpes):.4f}, std={np.std(shuffle_sharpes):.4f}")
    print(f"  Baseline CAGR:   {base['CAGR%']:.2f}%")
    print(f"  Baseline Sharpe: {base['Sharpe']:.4f}")
    collapse = base["CAGR%"] - np.mean(shuffle_cagrs)
    print(f"  Ranking alpha:   {collapse:+.2f}% CAGR")

    # ====== 5. RANDOM SIGNAL TEST ======
    section("5. SIGNAL DESTRUCTION (random probabilities)")
    random_results = []
    for seed in range(10):
        t5, eq5, _ = simulate(df, 0.50, 3, 3, SC, random_probs=True, rng_seed=seed)
        m5 = metrics(t5, eq5, SC)
        random_results.append(m5)
    random_cagrs = [r["CAGR%"] for r in random_results]
    random_sharpes = [r["Sharpe"] for r in random_results]
    print(f"  Random CAGR:   mean={np.mean(random_cagrs):.2f}%, std={np.std(random_cagrs):.2f}%")
    print(f"  Random Sharpe: mean={np.mean(random_sharpes):.4f}, std={np.std(random_sharpes):.4f}")
    print(f"  Model CAGR:    {base['CAGR%']:.2f}%")
    signal_alpha = base["CAGR%"] - np.mean(random_cagrs)
    print(f"  Signal alpha:  {signal_alpha:+.2f}% CAGR")

    # ====== 6. TIME ROBUSTNESS ======
    section("6. TIME ROBUSTNESS (per-period evaluation)")
    periods = [("2015-2017", 2015, 2017), ("2018-2020", 2018, 2020),
               ("2021-2023", 2021, 2023), ("2024-present", 2024, 2030)]
    for label, y_start, y_end in periods:
        sub = df[(df["year"] >= y_start) & (df["year"] <= y_end)]
        if len(sub) < 50:
            print(f"  {label}: insufficient data")
            continue
        t6, eq6, _ = simulate(sub, 0.50, 3, 3, SC)
        m6 = metrics(t6, eq6, SC)
        print_row(label, m6)

    # ====== 7. SECTOR/STOCK CONCENTRATION ======
    section("7. STOCK CONCENTRATION")
    # Which stocks dominate the trade list?
    t_all, _, _ = simulate(df, 0.50, 3, 3, SC)
    if len(t_all) > 0:
        ticker_stats = t_all.groupby("ticker").agg(
            trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            mean_ret=("return_pct", "mean"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        ).round(4)
        ticker_stats["pnl_share"] = (ticker_stats["total_pnl"] / ticker_stats["total_pnl"].sum() * 100).round(1)
        print(f"\n{'Ticker':<8} {'Trades':>7} {'TotalPnL':>10} {'PnL%':>6} {'MeanRet':>9} {'WR%':>6}")
        print("-" * 48)
        for ticker, row in ticker_stats.sort_values("total_pnl", ascending=False).iterrows():
            print(f"{ticker:<8} {int(row['trades']):>7} {row['total_pnl']:>10.2f} "
                  f"{row['pnl_share']:>5.1f}% {row['mean_ret']*100:>8.3f}% {row['win_rate']*100:>5.1f}%")

        # Leave-one-stock-out: what happens if we exclude each stock?
        print(f"\n  Leave-one-stock-out portfolio impact:")
        for ticker_to_exclude in sorted(t_all["ticker"].unique()):
            sub = df[df["ticker"] != ticker_to_exclude]
            t_loo, eq_loo, _ = simulate(sub, 0.50, 3, 3, SC)
            m_loo = metrics(t_loo, eq_loo, SC)
            delta = m_loo["CAGR%"] - base["CAGR%"]
            print(f"    Exclude {ticker_to_exclude}: CAGR={m_loo['CAGR%']:>6.2f}% "
                  f"({delta:+.2f}), Sharpe={m_loo['Sharpe']:.4f}")

    # ====== 8. CAPACITY ======
    section("8. CAPACITY TEST (different starting capitals)")
    for cap in [1000, 10000, 100000, 1000000]:
        t8, eq8, _ = simulate(df, 0.50, 3, 3, float(cap))
        m8 = metrics(t8, eq8, float(cap))
        print_row(f"Capital=${cap:>10,}", m8)

    # ====== 9. BENCHMARKS ======
    section("9. BENCHMARKS")
    # SPY buy-and-hold
    spy = df[df["ticker"] == "SPY"].sort_values("date").drop_duplicates("date")
    if len(spy) > 0:
        # Reconstruct SPY equity from daily returns (ret_1d isn't available, use forward_return/20 as proxy)
        spy_rets = spy["forward_return"].values / MAX_HOLD_DAYS  # daily return proxy
        spy_equity = SC * np.cumprod(1 + np.nan_to_num(spy_rets))
        spy_m = metrics(pd.DataFrame(), spy_equity, SC)
        spy_m["Trades"] = 0
        spy_m["WR%"] = 0
        spy_m["PF"] = 0
        print_row("SPY buy-and-hold", spy_m)

    # Equal-weight 7-stock
    all_tickers = sorted(df["ticker"].unique())
    daily_rets = []
    for date in sorted(df["date"].unique()):
        day = df[df["date"] == date]
        if len(day) > 0:
            daily_rets.append(day["forward_return"].mean() / MAX_HOLD_DAYS)
    ew_equity = SC * np.cumprod(1 + np.nan_to_num(daily_rets))
    ew_m = metrics(pd.DataFrame(), ew_equity, SC)
    ew_m["Trades"] = 0
    ew_m["WR%"] = 0
    ew_m["PF"] = 0
    print_row("Equal-weight 7-stock", ew_m)

    # Random ranking
    print_row("Random signals (avg 10 runs)", {
        "CAGR%": np.mean(random_cagrs), "Sharpe": np.mean(random_sharpes),
        "MaxDD%": np.mean([r["MaxDD%"] for r in random_results]),
        "Trades": int(np.mean([r["Trades"] for r in random_results])),
        "WR%": np.mean([r["WR%"] for r in random_results]),
        "PF": np.mean([r["PF"] for r in random_results]),
        "Final$": np.mean([r["Final$"] for r in random_results]),
    })

    # Model
    print_row("MODEL (t50, top3, 3pos)", base)

    # ====== FINAL VERDICT ======
    section("FINAL VERDICT")

    issues = []
    if signal_alpha < 2:
        issues.append(f"Signal alpha over random is only {signal_alpha:.1f}% CAGR")
    if base["MaxDD%"] < -30:
        issues.append(f"Max drawdown {base['MaxDD%']:.1f}% is severe")
    if base["Trades"] < 100:
        issues.append(f"Only {base['Trades']} trades - statistically thin")

    # Check if any period has negative performance
    neg_periods = []
    for label, y_start, y_end in periods:
        sub = df[(df["year"] >= y_start) & (df["year"] <= y_end)]
        if len(sub) >= 50:
            t_p, eq_p, _ = simulate(sub, 0.50, 3, 3, SC)
            m_p = metrics(t_p, eq_p, SC)
            if m_p["CAGR%"] < 0:
                neg_periods.append(f"{label}: CAGR={m_p['CAGR%']:.2f}%")

    if neg_periods:
        issues.append(f"Negative returns in: {', '.join(neg_periods)}")

    print(f"\n  Reported: CAGR={base['CAGR%']:.2f}%, Sharpe={base['Sharpe']:.4f}")

    if issues:
        print(f"\n  Concerns:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

    print(f"\n  Reasons the 17.2% CAGR may be overstated:")
    print(f"    1. Python simulator uses simplified execution (normalized prices,")
    print(f"       no intraday stop/target, linear mark-to-market interpolation)")
    print(f"    2. Forward return captures the full 20-day outcome but real trades")
    print(f"       may exit earlier via stop/target, changing realized returns")
    print(f"    3. Single-stock universe (7 mega-cap tech) with survivorship bias")
    print(f"    4. The 2020 and 2025 periods (high returns) may dominate the average")
    print(f"    5. Max position fraction 40% * 3 positions = 120% allocation possible")
    print(f"       from unrealized gains (not leverage, but aggressive)")

    if len(issues) == 0 and signal_alpha > 5 and base["Sharpe"] > 1:
        verdict = "STRONG ENOUGH TO JUSTIFY 3-6 MONTH PAPER TRADING TRIAL"
    elif signal_alpha > 2 and base["CAGR%"] > 5:
        verdict = "CREDIBLE ENOUGH FOR PAPER TRADING"
    elif signal_alpha > 0:
        verdict = "NEEDS MORE VALIDATION"
    else:
        verdict = "NOT CREDIBLE"

    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
