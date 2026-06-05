"""
Research Phase 8: Universe Expansion — Cross-Sectional Stock Selection Test

Main question: Can the model identify stocks that outperform the broader universe?
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


def load():
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
    merged = merged[merged["year"] >= 2005].reset_index(drop=True)
    return merged


def section(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def main():
    print("=" * 70)
    print("  RESEARCH PHASE 8: Cross-Sectional Stock Selection (101 stocks)")
    print("=" * 70)

    df = load()
    n_tickers = df["ticker"].nunique()
    n_dates = df["date"].nunique()
    print(f"Loaded {len(df)} predictions, {n_tickers} tickers, {n_dates} dates")
    print(f"Years: {df['year'].min()}-{df['year'].max()}")

    # ====== 1. PER-DATE PERCENTILE RANKING ======
    section("1. PERCENTILE RANKING: Top vs Bottom Returns")

    # For each date, rank stocks by probability, then compare returns of top vs bottom
    results = []
    for date, group in df.groupby("date"):
        if len(group) < 20:  # need enough stocks to rank
            continue

        group = group.sort_values("probability", ascending=False)
        n = len(group)

        for pct_label, top_k in [("top_1%", max(1, n//100)), ("top_5%", max(1, n//20)),
                                   ("top_10%", max(1, n//10)), ("top_20%", max(1, n//5)),
                                   ("bottom_20%", max(1, n//5))]:
            if pct_label.startswith("top"):
                sel = group.head(top_k)
            else:
                sel = group.tail(top_k)

            results.append({
                "date": date, "bucket": pct_label,
                "mean_ret": sel["forward_return"].mean(),
                "count": len(sel),
            })

    results_df = pd.DataFrame(results)

    # Universe average per date
    univ_avg = df.groupby("date")["forward_return"].mean().reset_index()
    univ_avg.columns = ["date", "universe_avg"]

    print(f"\nMean 20-day return by probability bucket (2005-present):")
    print(f"{'Bucket':<15} {'MeanRet%':>9} {'ExcessVsUniv%':>14} {'Count/Day':>10}")
    print("-" * 50)

    for bucket in ["top_1%", "top_5%", "top_10%", "top_20%", "bottom_20%"]:
        sub = results_df[results_df["bucket"] == bucket]
        sub = sub.merge(univ_avg, on="date")
        mean_ret = sub["mean_ret"].mean() * 100
        excess = (sub["mean_ret"] - sub["universe_avg"]).mean() * 100
        avg_count = sub["count"].mean()
        print(f"{bucket:<15} {mean_ret:>9.3f} {excess:>+14.3f} {avg_count:>10.1f}")

    # Universe average
    overall_univ = df.groupby("date")["forward_return"].mean().mean() * 100
    print(f"{'universe_avg':<15} {overall_univ:>9.3f} {0:>+14.3f}")

    # ====== 2. LONG-SHORT SPREAD ======
    section("2. LONG-SHORT SPREAD (top 20% minus bottom 20%)")

    spread_data = []
    for date in df["date"].unique():
        group = df[df["date"] == date].sort_values("probability", ascending=False)
        if len(group) < 10:
            continue
        n = len(group)
        top20 = group.head(max(1, n // 5))["forward_return"].mean()
        bot20 = group.tail(max(1, n // 5))["forward_return"].mean()
        spread_data.append({"date": date, "top20": top20, "bot20": bot20,
                           "spread": top20 - bot20, "year": date.year})

    spread_df = pd.DataFrame(spread_data)
    if len(spread_df) > 0:
        mean_spread = spread_df["spread"].mean() * 100
        hit_rate = (spread_df["spread"] > 0).mean() * 100
        t_stat = (spread_df["spread"].mean() / (spread_df["spread"].std() / np.sqrt(len(spread_df))))
        sharpe = spread_df["spread"].mean() / spread_df["spread"].std() * np.sqrt(252 / HOLD_DAYS)
        print(f"  Mean spread (top20 - bot20): {mean_spread:+.3f}% per 20 days")
        print(f"  Hit rate (spread > 0):       {hit_rate:.1f}%")
        print(f"  t-statistic:                 {t_stat:.2f}")
        print(f"  Annualized Sharpe of spread: {sharpe:.4f}")

        # By year
        print(f"\n  Spread by year:")
        yearly = spread_df.groupby("year")["spread"].agg(["mean", "count", "std"])
        yearly["sharpe"] = yearly["mean"] / yearly["std"] * np.sqrt(252/HOLD_DAYS)
        for year, row in yearly.iterrows():
            print(f"    {year}: spread={row['mean']*100:+.3f}%, sharpe={row['sharpe']:.4f}, n={int(row['count'])}")

    # ====== 3. RANDOM BASELINE ======
    section("3. RANDOM BASELINE (shuffle rankings 50 times)")

    random_spreads = []
    rng = np.random.RandomState(42)
    for trial in range(50):
        trial_spreads = []
        for date in df["date"].unique():
            group = df[df["date"] == date].copy()
            if len(group) < 10:
                continue
            n = len(group)
            group["random_prob"] = rng.uniform(0, 1, n)
            group = group.sort_values("random_prob", ascending=False)
            top20 = group.head(max(1, n // 5))["forward_return"].mean()
            bot20 = group.tail(max(1, n // 5))["forward_return"].mean()
            trial_spreads.append(top20 - bot20)
        random_spreads.append(np.mean(trial_spreads) * 100)

    print(f"  Random spread: mean={np.mean(random_spreads):+.3f}%, std={np.std(random_spreads):.3f}%")
    print(f"  Model spread:  {mean_spread:+.3f}%")
    print(f"  Model vs random: {mean_spread - np.mean(random_spreads):+.3f}%")
    z_score = (mean_spread - np.mean(random_spreads)) / max(np.std(random_spreads), 1e-6)
    print(f"  Z-score: {z_score:.2f}")

    # ====== 4. PORTFOLIO SIMULATION (top-N per day) ======
    section("4. PORTFOLIO SIMULATION (top-N selection, 2015+)")

    df_2015 = df[df["year"] >= 2015]
    configs = [
        ("top_1", 1), ("top_3", 3), ("top_5", 5), ("top_10", 10),
        ("top_20", 20), ("random_5", 5),
    ]

    print(f"\n{'Config':<12} {'Trades':>7} {'MeanRet%':>9} {'ExVsUniv%':>10} "
          f"{'WR%':>6} {'Sharpe':>8} {'HitRate':>8}")
    print("-" * 68)

    for label, top_k in configs:
        is_random = "random" in label
        all_rets = []
        all_excess = []

        for date, group in df_2015.groupby("date"):
            if len(group) < 10:
                continue

            if is_random:
                sel = group.sample(n=min(top_k, len(group)), random_state=42)
            else:
                sel = group.nlargest(top_k, "probability")

            univ_mean = group["forward_return"].mean()
            for _, row in sel.iterrows():
                all_rets.append(row["forward_return"])
                all_excess.append(row["forward_return"] - univ_mean)

        rets = np.array(all_rets)
        excess = np.array(all_excess)
        if len(rets) == 0:
            continue

        mean_ret = rets.mean() * 100
        mean_excess = excess.mean() * 100
        wr = (rets > 0).mean() * 100
        sharpe = rets.mean() / rets.std() * np.sqrt(252/HOLD_DAYS) if rets.std() > 0 else 0
        hit_rate = (excess > 0).mean() * 100

        print(f"{label:<12} {len(rets):>7} {mean_ret:>9.3f} {mean_excess:>+10.3f} "
              f"{wr:>6.1f} {sharpe:>8.4f} {hit_rate:>7.1f}%")

    # ====== 5. SECTOR BREAKDOWN ======
    section("5. SECTOR PERFORMANCE (model top-5 per day, 2015+)")

    sector_map = {}
    universe_def = {
        "tech": ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AVGO", "ORCL", "CRM", "ADBE", "AMD",
                 "INTC", "CSCO", "IBM", "TXN", "QCOM", "MU", "HPQ", "DELL"],
        "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
                       "GILD", "CVS", "CI", "MDT", "BIIB"],
        "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
                      "MET", "PRU", "ALL"],
        "cons_disc": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "F", "GM", "MAR"],
        "cons_staples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MO", "PM", "KHC"],
        "industrials": ["CAT", "BA", "HON", "UPS", "GE", "MMM", "DE", "LMT", "RTX", "FDX"],
        "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "HAL"],
        "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE"],
        "comms": ["DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS"],
        "etf": ["SPY", "QQQ"],
    }
    for sector, tickers in universe_def.items():
        for t in tickers:
            sector_map[t] = sector

    # Collect top-5 selections
    top5_selections = []
    for date, group in df_2015.groupby("date"):
        if len(group) < 10:
            continue
        sel = group.nlargest(5, "probability")
        for _, row in sel.iterrows():
            top5_selections.append({
                "ticker": row["ticker"],
                "sector": sector_map.get(row["ticker"], "unknown"),
                "forward_return": row["forward_return"],
            })

    sel_df = pd.DataFrame(top5_selections)
    if len(sel_df) > 0:
        sector_stats = sel_df.groupby("sector").agg(
            count=("forward_return", "count"),
            mean_ret=("forward_return", "mean"),
            win_rate=("forward_return", lambda x: (x > 0).mean()),
        ).round(4)
        sector_stats["alloc%"] = (sector_stats["count"] / sector_stats["count"].sum() * 100).round(1)
        sector_stats = sector_stats.sort_values("count", ascending=False)

        print(f"\n{'Sector':<15} {'Trades':>7} {'Alloc%':>7} {'MeanRet%':>9} {'WR%':>6}")
        print("-" * 46)
        for sector, row in sector_stats.iterrows():
            print(f"{sector:<15} {int(row['count']):>7} {row['alloc%']:>6.1f}% "
                  f"{row['mean_ret']*100:>8.3f}% {row['win_rate']*100:>5.1f}%")

    # ====== VERDICT ======
    section("VERDICT")

    print(f"\n  Universe: {n_tickers} stocks across {len(universe_def)} sectors")
    print(f"  Model AUC: 0.60 (101-stock pooled)")
    print(f"  Long-short spread: {mean_spread:+.3f}% per 20 days")
    print(f"  Spread t-stat: {t_stat:.2f}")
    print(f"  Spread vs random: {mean_spread - np.mean(random_spreads):+.3f}%")

    if t_stat > 2 and mean_spread > np.mean(random_spreads) + 2 * np.std(random_spreads):
        print(f"\n  CONCLUSION: Signal SEPARATES winners from losers cross-sectionally.")
        print(f"  The model has genuine stock selection ability.")
    elif t_stat > 1.5:
        print(f"\n  CONCLUSION: Signal shows WEAK but detectable stock selection ability.")
        print(f"  Statistical significance is marginal.")
    else:
        print(f"\n  CONCLUSION: Signal does NOT reliably separate winners from losers.")
        print(f"  Cross-sectional stock selection is not demonstrated.")


if __name__ == "__main__":
    main()
