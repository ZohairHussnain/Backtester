"""
Research Phase 3: Cross-Sectional Validation

Tests whether the median_return signal generalizes across stocks or is
AAPL-specific. Two experiments:

A) Pool: Train on all tickers, test on all tickers (walk-forward by year).
B) Leave-one-out: Train on 6 tickers, test on the 7th (walk-forward by year).
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_DIR / "output" / "ml_dataset.csv"

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_14", "atr_14_pct", "volume_ratio_20",
    "dist_ma20", "dist_ma50", "dist_ma200",
    "rolling_vol_20",
]
TARGET = "label_median_return"
MIN_TRAIN_YEARS = 4
PURGE_ROWS = 20


def load():
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    df["year"] = df["date"].dt.year
    return df


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")),
    ])


def purge_indices(df, pre_test_mask, tickers):
    purge = set()
    for ticker in tickers:
        tp = df.index[(df["ticker"] == ticker) & pre_test_mask]
        if len(tp) <= PURGE_ROWS:
            purge.update(tp)
        else:
            purge.update(tp[-PURGE_ROWS:])
    return purge


def evaluate(y_true, proba):
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan
    y_pred = (proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, proba)
    return auc, f1, brier


# ======================================================================
# A) Pooled: train on all, test on all (walk-forward by year)
# ======================================================================

def experiment_pooled(df):
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: Pooled (train all, test all, walk-forward by year)")
    print("=" * 70)

    years = sorted(df["year"].unique())
    tickers = sorted(df["ticker"].unique())

    all_results = []

    test_years = [y for y in years if y >= 2015]
    for test_year in test_years:
        pre_test_mask = df["year"] < test_year
        test_mask = df["year"] == test_year

        if pre_test_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        purge = purge_indices(df, pre_test_mask, tickers)
        train_mask = pre_test_mask & ~df.index.isin(purge)

        if train_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, FEATURE_COLS]
        y_train = df.loc[train_mask, TARGET]
        X_test = df.loc[test_mask, FEATURE_COLS]
        y_test = df.loc[test_mask, TARGET]

        model = make_model()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        # Overall
        auc, f1, brier = evaluate(y_test, proba)

        # Per-ticker
        test_df = df.loc[test_mask].copy()
        test_df["proba"] = proba

        for ticker in tickers:
            tmask = test_df["ticker"] == ticker
            if tmask.sum() < 10:
                continue
            t_auc, t_f1, t_brier = evaluate(
                test_df.loc[tmask, TARGET], test_df.loc[tmask, "proba"])
            all_results.append({
                "test_year": test_year, "ticker": ticker,
                "auc": t_auc, "f1": t_f1, "brier": t_brier,
                "n_test": int(tmask.sum()),
            })

    results_df = pd.DataFrame(all_results)

    # Summary per ticker
    print(f"\n{'Ticker':<8} {'Mean AUC':>10} {'Std AUC':>10} {'Mean F1':>10} {'Folds':>6}")
    print("-" * 46)
    for ticker in sorted(results_df["ticker"].unique()):
        sub = results_df[results_df["ticker"] == ticker]
        print(f"{ticker:<8} {sub['auc'].mean():>10.4f} {sub['auc'].std():>10.4f} "
              f"{sub['f1'].mean():>10.4f} {len(sub):>6}")

    # Overall
    print(f"\n{'ALL':<8} {results_df['auc'].mean():>10.4f} {results_df['auc'].std():>10.4f} "
          f"{results_df['f1'].mean():>10.4f} {len(results_df):>6}")

    return results_df


# ======================================================================
# B) Leave-one-out: train on 6, test on 1 (walk-forward by year)
# ======================================================================

def experiment_leave_one_out(df):
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Leave-One-Stock-Out (train 6, test 1)")
    print("=" * 70)

    years = sorted(df["year"].unique())
    tickers = sorted(df["ticker"].unique())
    test_years = [y for y in years if y >= 2015]

    all_results = []

    for held_out_ticker in tickers:
        train_tickers = [t for t in tickers if t != held_out_ticker]
        ticker_results = []

        for test_year in test_years:
            # Train: all tickers except held-out, year < test_year
            train_mask = (df["year"] < test_year) & (df["ticker"].isin(train_tickers))
            # Test: held-out ticker only, year == test_year
            test_mask = (df["year"] == test_year) & (df["ticker"] == held_out_ticker)

            if train_mask.sum() < 100 or test_mask.sum() < 10:
                continue

            purge = purge_indices(df, train_mask, train_tickers)
            train_clean = train_mask & ~df.index.isin(purge)
            if train_clean.sum() == 0:
                continue

            X_train = df.loc[train_clean, FEATURE_COLS]
            y_train = df.loc[train_clean, TARGET]
            X_test = df.loc[test_mask, FEATURE_COLS]
            y_test = df.loc[test_mask, TARGET]

            model = make_model()
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]

            auc, f1, brier = evaluate(y_test, proba)
            ticker_results.append({
                "held_out": held_out_ticker, "test_year": test_year,
                "auc": auc, "f1": f1, "brier": brier,
                "n_train": int(train_clean.sum()), "n_test": int(test_mask.sum()),
            })

        all_results.extend(ticker_results)

    results_df = pd.DataFrame(all_results)

    # Summary per held-out ticker
    print(f"\n{'Held Out':<8} {'Mean AUC':>10} {'Std AUC':>10} {'Mean F1':>10} {'Folds':>6}")
    print("-" * 46)
    for ticker in tickers:
        sub = results_df[results_df["held_out"] == ticker]
        if len(sub) > 0:
            print(f"{ticker:<8} {sub['auc'].mean():>10.4f} {sub['auc'].std():>10.4f} "
                  f"{sub['f1'].mean():>10.4f} {len(sub):>6}")

    overall = results_df["auc"].mean()
    print(f"\n{'ALL':<8} {results_df['auc'].mean():>10.4f} {results_df['auc'].std():>10.4f} "
          f"{results_df['f1'].mean():>10.4f} {len(results_df):>6}")

    return results_df


# ======================================================================
# Trade performance per stock
# ======================================================================

def trade_performance(df, pooled_results):
    print("\n" + "=" * 70)
    print("  TRADE PERFORMANCE BY STOCK (pooled model, t=0.55, 2020+)")
    print("=" * 70)

    tickers = sorted(df["ticker"].unique())
    years = sorted(df["year"].unique())
    test_years = [y for y in years if y >= 2020]

    all_preds = []
    for test_year in test_years:
        pre_test_mask = df["year"] < test_year
        test_mask = df["year"] == test_year
        if pre_test_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        purge = purge_indices(df, pre_test_mask, tickers)
        train_mask = pre_test_mask & ~df.index.isin(purge)
        if train_mask.sum() == 0:
            continue

        model = make_model()
        model.fit(df.loc[train_mask, FEATURE_COLS], df.loc[train_mask, TARGET])
        proba = model.predict_proba(df.loc[test_mask, FEATURE_COLS])[:, 1]

        preds = df.loc[test_mask, ["date", "ticker", "forward_return"]].copy()
        preds["proba"] = proba
        all_preds.append(preds)

    all_preds_df = pd.concat(all_preds, ignore_index=True)
    signals = all_preds_df[all_preds_df["proba"] >= 0.55]

    print(f"\n{'Ticker':<8} {'Trades':>7} {'MeanRet%':>9} {'WinRate%':>9} {'Sharpe':>8}")
    print("-" * 43)
    for ticker in tickers:
        sub = signals[signals["ticker"] == ticker]
        if len(sub) == 0:
            print(f"{ticker:<8} {'---':>7}")
            continue
        rets = sub["forward_return"]
        wins = (rets > 0).sum()
        sharpe = rets.mean() / rets.std() * np.sqrt(252/20) if rets.std() > 0 else 0
        print(f"{ticker:<8} {len(sub):>7} {rets.mean()*100:>9.3f} "
              f"{wins/len(sub)*100:>9.1f} {sharpe:>8.4f}")

    # Overall
    all_rets = signals["forward_return"]
    if len(all_rets) > 0:
        wins = (all_rets > 0).sum()
        sharpe = all_rets.mean() / all_rets.std() * np.sqrt(252/20) if all_rets.std() > 0 else 0
        print(f"\n{'ALL':<8} {len(all_rets):>7} {all_rets.mean()*100:>9.3f} "
              f"{wins/len(all_rets)*100:>9.1f} {sharpe:>8.4f}")


# ======================================================================
# Verdict
# ======================================================================

def verdict(pooled_df, loo_df):
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    pooled_auc = pooled_df["auc"].mean()
    loo_auc = loo_df["auc"].mean()

    # Per-ticker AUCs in leave-one-out
    ticker_aucs = loo_df.groupby("held_out")["auc"].mean()
    above_05 = (ticker_aucs > 0.52).sum()
    total = len(ticker_aucs)

    print(f"\n  Pooled mean AUC:          {pooled_auc:.4f}")
    print(f"  Leave-one-out mean AUC:   {loo_auc:.4f}")
    print(f"  Tickers with LOO AUC > 0.52: {above_05}/{total}")

    print(f"\n  Per-ticker LOO AUC:")
    for ticker in sorted(ticker_aucs.index):
        marker = " ***" if ticker_aucs[ticker] > 0.55 else ""
        print(f"    {ticker}: {ticker_aucs[ticker]:.4f}{marker}")

    if loo_auc > 0.55 and above_05 >= total * 0.7:
        conclusion = "SIGNAL STRONGLY GENERALIZES"
    elif loo_auc > 0.52 and above_05 >= total * 0.5:
        conclusion = "SIGNAL PARTIALLY GENERALIZES"
    else:
        conclusion = "SIGNAL DOES NOT GENERALIZE"

    print(f"\n  Conclusion: {conclusion}")

    print(f"\n  Interpretation:")
    if loo_auc > 0.55:
        print(f"    The model trained on OTHER stocks can predict the held-out stock.")
        print(f"    This means the features capture a reusable swing-trading pattern,")
        print(f"    not stock-specific behavior.")
    elif loo_auc > 0.52:
        print(f"    There is weak but detectable cross-sectional transfer.")
        print(f"    The signal is partially reusable but may benefit from per-stock tuning.")
    else:
        print(f"    The model is learning stock-specific patterns that do not transfer.")
        print(f"    Cross-sectional training may be diluting per-stock signal.")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("  RESEARCH PHASE 3: Cross-Sectional Validation")
    print("=" * 70)

    df = load()
    print(f"Loaded {len(df)} rows, {sorted(df['ticker'].unique())}")
    print(f"Rows per ticker:")
    for ticker in sorted(df["ticker"].unique()):
        n = len(df[df["ticker"] == ticker])
        yr_range = f"{df[df['ticker']==ticker]['year'].min()}-{df[df['ticker']==ticker]['year'].max()}"
        print(f"  {ticker}: {n} rows ({yr_range})")

    pooled_df = experiment_pooled(df)
    loo_df = experiment_leave_one_out(df)
    trade_performance(df, pooled_df)
    verdict(pooled_df, loo_df)

    # Save
    out_path = PROJECT_DIR / "output" / "cross_sectional_results.csv"
    combined = pd.concat([
        pooled_df.assign(experiment="pooled"),
        loo_df.rename(columns={"held_out": "ticker"}).assign(experiment="leave_one_out"),
    ], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
