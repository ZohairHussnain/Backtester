"""
Research Phase 2: Label Quality Comparison

Compares two label definitions:
  A) "target_stop" — existing label: did price hit +target% before -stop% within horizon?
  B) "median_return" — future 20-day return above trailing 252-day rolling median

Trains the same three models (Logistic, XGBoost, LightGBM) on each label using
walk-forward yearly folds with purge, then compares AUC, precision, recall, F1,
calibration, and regime stability.

No backtester changes. No new models. Pure label quality investigation.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_DIR / "output" / "ml_dataset.csv"

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_14", "atr_14_pct", "volume_ratio_20",
    "dist_ma20", "dist_ma50", "dist_ma200",
    "rolling_vol_20",
]

MIN_TRAIN_YEARS = 4
PURGE_ROWS = 20
FORWARD_DAYS = 20
ROLLING_MEDIAN_WINDOW = 252


# ======================================================================
# Data loading and label construction
# ======================================================================

def load_and_label(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + ["label"])

    # Rename existing label
    df.rename(columns={"label": "label_target_stop"}, inplace=True)

    # Compute forward 20-day return from adjusted close proxy.
    # We don't have raw close in the CSV, but we can reconstruct forward return
    # from the ret_1d column: cumulative sum of daily log-returns over the next
    # 20 days. However, we only have CURRENT ret_1d, not FUTURE ret_1d.
    #
    # We need to compute the forward return directly from the time series.
    # Since the CSV only has features (backward-looking), we must derive the
    # forward return from price data. The simplest approach: use the adjusted
    # close implied by cumulating ret_1d backwards to get a price series, then
    # compute forward returns.
    #
    # Actually, ret_1d = adjusted_close[i] / adjusted_close[i-1] - 1.
    # So adjusted_close[i] = adjusted_close[i-1] * (1 + ret_1d[i]).
    # We can reconstruct a price index from ret_1d.

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_df = df.loc[mask].copy()

        # Reconstruct price index from ret_1d
        # Start at 1.0, compound forward
        price = np.cumprod(1 + ticker_df["ret_1d"].values)

        # Forward 20-day return
        fwd_ret = np.full(len(price), np.nan)
        for i in range(len(price) - FORWARD_DAYS):
            fwd_ret[i] = price[i + FORWARD_DAYS] / price[i] - 1.0

        # Rolling median of forward return (trailing 252-day window)
        fwd_series = pd.Series(fwd_ret, index=ticker_df.index)
        rolling_median = fwd_series.rolling(ROLLING_MEDIAN_WINDOW, min_periods=60).median()

        # Label: 1 if forward return > trailing rolling median
        label_median = (fwd_series > rolling_median).astype(float)
        label_median[fwd_series.isna() | rolling_median.isna()] = np.nan

        df.loc[mask, "fwd_20d_return"] = fwd_ret
        df.loc[mask, "rolling_median_fwd"] = rolling_median.values
        df.loc[mask, "label_median_return"] = label_median.values

    # Drop rows where either label is missing
    df = df.dropna(subset=["label_target_stop", "label_median_return"]).reset_index(drop=True)
    df["label_median_return"] = df["label_median_return"].astype(int)
    df["label_target_stop"] = df["label_target_stop"].astype(int)

    return df


def print_label_stats(df: pd.DataFrame):
    print("\n--- Label Statistics ---")
    for label_name in ["label_target_stop", "label_median_return"]:
        y = df[label_name]
        pos_rate = y.mean()
        print(f"\n  {label_name}:")
        print(f"    Positive rate: {pos_rate:.3f} ({int(y.sum())}/{len(y)})")

        # Rate by decade
        df["decade"] = (df["date"].dt.year // 10) * 10
        print(f"    Rate by decade:")
        for decade, group in df.groupby("decade"):
            print(f"      {decade}s: {group[label_name].mean():.3f} (n={len(group)})")

        # Autocorrelation
        same = (y == y.shift(1)).mean()
        random_baseline = pos_rate**2 + (1 - pos_rate)**2
        print(f"    Autocorrelation P(t==t-1): {same:.4f} (random={random_baseline:.4f})")

        # Yearly std
        yearly = df.groupby(df["date"].dt.year)[label_name].mean()
        print(f"    Yearly rate CV: {yearly.std()/yearly.mean():.3f}")

    df.drop(columns=["decade"], inplace=True, errors="ignore")


# ======================================================================
# Walk-forward folds (same as walk_forward_train.py)
# ======================================================================

def build_folds(df: pd.DataFrame) -> list[dict]:
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    tickers = sorted(df["ticker"].unique())

    folds = []
    for i in range(MIN_TRAIN_YEARS, len(years)):
        test_year = years[i]
        test_mask = df["year"] == test_year
        pre_test_mask = df["year"] < test_year

        if pre_test_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        purge_indices = set()
        for ticker in tickers:
            ticker_pre = df.index[(df["ticker"] == ticker) & pre_test_mask]
            if len(ticker_pre) <= PURGE_ROWS:
                purge_indices.update(ticker_pre)
            else:
                purge_indices.update(ticker_pre[-PURGE_ROWS:])

        train_mask = pre_test_mask & ~df.index.isin(purge_indices)
        if train_mask.sum() == 0:
            continue

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        assert df.loc[train_idx, "date"].max() < df.loc[test_idx, "date"].min()
        assert len(set(train_idx) & set(test_idx)) == 0

        folds.append({
            "fold": len(folds) + 1,
            "test_year": test_year,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    return folds


# ======================================================================
# Training and evaluation
# ======================================================================

def get_models(scale_pos_weight: float) -> dict:
    return {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")),
        ]),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", verbosity=0, use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, verbose=-1,
            scale_pos_weight=scale_pos_weight,
        ),
    }


def evaluate_label(df, folds, label_col, label_name):
    """Run walk-forward evaluation for one label definition."""
    X = df[FEATURE_COLS]
    y = df[label_col]

    all_metrics = []
    all_probas = []  # for calibration

    for fold_info in folds:
        fold = fold_info["fold"]
        test_year = fold_info["test_year"]
        train_idx = fold_info["train_idx"]
        test_idx = fold_info["test_idx"]

        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]

        num_pos = int(y_train.sum())
        num_neg = len(y_train) - num_pos
        if num_pos == 0:
            continue
        spw = num_neg / num_pos

        for model_name, model in get_models(spw).items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= 0.5).astype(int)

            try:
                auc = roc_auc_score(y_test, proba)
            except ValueError:
                auc = np.nan

            metrics = {
                "label": label_name,
                "model": model_name,
                "fold": fold,
                "test_year": test_year,
                "auc": auc,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "brier": brier_score_loss(y_test, proba),
                "pos_rate_train": num_pos / len(y_train),
                "pos_rate_test": y_test.mean(),
            }
            all_metrics.append(metrics)
            all_probas.append({
                "label": label_name,
                "model": model_name,
                "fold": fold,
                "y_test": y_test.values,
                "proba": proba,
            })

    return pd.DataFrame(all_metrics), all_probas


# ======================================================================
# Reporting
# ======================================================================

def print_comparison(metrics_a, metrics_b):
    print("\n" + "=" * 70)
    print("  LABEL COMPARISON REPORT")
    print("=" * 70)

    combined = pd.concat([metrics_a, metrics_b], ignore_index=True)

    # Overall summary by label
    print("\n--- Overall Mean Metrics by Label ---")
    summary = combined.groupby("label").agg(
        mean_auc=("auc", "mean"),
        std_auc=("auc", "std"),
        mean_f1=("f1", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_brier=("brier", "mean"),
        folds=("fold", "count"),
    ).round(4)
    print(summary.to_string())

    # By label and model
    print("\n--- Mean Metrics by Label x Model ---")
    detail = combined.groupby(["label", "model"]).agg(
        mean_auc=("auc", "mean"),
        std_auc=("auc", "std"),
        mean_f1=("f1", "mean"),
        mean_brier=("brier", "mean"),
    ).round(4)
    print(detail.to_string())

    # Regime stability: yearly AUC std
    print("\n--- Regime Stability (yearly AUC std, lower is more stable) ---")
    yearly_std = combined.groupby(["label", "model"]).agg(
        auc_std=("auc", "std"),
        auc_min=("auc", "min"),
        auc_max=("auc", "max"),
    ).round(4)
    print(yearly_std.to_string())

    # Label rate stability
    print("\n--- Label Rate Stability Across Test Folds ---")
    for label_name in combined["label"].unique():
        subset = combined[combined["label"] == label_name]
        # one row per fold per model — take first model to avoid duplicates
        fold_rates = subset.drop_duplicates(subset=["fold"])[["fold", "test_year", "pos_rate_test"]]
        cv = fold_rates["pos_rate_test"].std() / fold_rates["pos_rate_test"].mean()
        print(f"  {label_name}:")
        print(f"    Test positive rate: mean={fold_rates['pos_rate_test'].mean():.3f}, "
              f"std={fold_rates['pos_rate_test'].std():.3f}, CV={cv:.3f}")
        print(f"    Range: {fold_rates['pos_rate_test'].min():.3f} to {fold_rates['pos_rate_test'].max():.3f}")

    # Head-to-head: per fold, which label wins?
    print("\n--- Head-to-Head: Which Label Wins Per Fold? (XGBoost) ---")
    xgb_a = metrics_a[metrics_a["model"] == "xgboost"].set_index("fold")["auc"]
    xgb_b = metrics_b[metrics_b["model"] == "xgboost"].set_index("fold")["auc"]
    common_folds = xgb_a.index.intersection(xgb_b.index)
    wins_a = (xgb_a.loc[common_folds] > xgb_b.loc[common_folds]).sum()
    wins_b = (xgb_b.loc[common_folds] > xgb_a.loc[common_folds]).sum()
    ties = (xgb_a.loc[common_folds] == xgb_b.loc[common_folds]).sum()
    print(f"  {metrics_a['label'].iloc[0]} wins: {wins_a}")
    print(f"  {metrics_b['label'].iloc[0]} wins: {wins_b}")
    print(f"  Ties: {ties}")
    print(f"  Total folds compared: {len(common_folds)}")

    # Recent performance (2020+)
    print("\n--- Recent Performance (2020+) ---")
    recent = combined[combined["test_year"] >= 2020]
    if len(recent) > 0:
        recent_summary = recent.groupby(["label", "model"]).agg(
            mean_auc=("auc", "mean"),
            mean_f1=("f1", "mean"),
        ).round(4)
        print(recent_summary.to_string())


def print_calibration(probas_a, probas_b):
    print("\n--- Calibration Analysis ---")
    for label_name, probas_list in [("target_stop", probas_a), ("median_return", probas_b)]:
        # Pool all XGBoost predictions
        xgb_probs = [p for p in probas_list if p["model"] == "xgboost"]
        if not xgb_probs:
            continue
        all_y = np.concatenate([p["y_test"] for p in xgb_probs])
        all_p = np.concatenate([p["proba"] for p in xgb_probs])

        try:
            fraction_pos, mean_predicted = calibration_curve(all_y, all_p, n_bins=5, strategy="quantile")
            print(f"\n  {label_name} (XGBoost, pooled across folds):")
            print(f"  {'Bin':>5} {'Predicted':>12} {'Actual':>12} {'Gap':>8}")
            print(f"  " + "-" * 39)
            for i in range(len(fraction_pos)):
                gap = fraction_pos[i] - mean_predicted[i]
                print(f"  {i+1:>5} {mean_predicted[i]:>12.4f} {fraction_pos[i]:>12.4f} {gap:>+8.4f}")
        except Exception as e:
            print(f"  {label_name}: calibration failed ({e})")


def print_verdict(metrics_a, metrics_b):
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    auc_a = metrics_a.groupby("model")["auc"].mean().mean()
    auc_b = metrics_b.groupby("model")["auc"].mean().mean()
    std_a = metrics_a.groupby("model")["auc"].std().mean()
    std_b = metrics_b.groupby("model")["auc"].std().mean()

    label_a = metrics_a["label"].iloc[0]
    label_b = metrics_b["label"].iloc[0]

    print(f"\n  {label_a}: mean AUC = {auc_a:.4f}, mean AUC std = {std_a:.4f}")
    print(f"  {label_b}: mean AUC = {auc_b:.4f}, mean AUC std = {std_b:.4f}")

    print(f"\n  Which label is easier to predict?")
    if auc_b > auc_a + 0.01:
        print(f"    -> {label_b} (higher AUC)")
    elif auc_a > auc_b + 0.01:
        print(f"    -> {label_a} (higher AUC)")
    else:
        print(f"    -> Similar (AUC difference < 0.01)")

    print(f"\n  Which label is more stable across regimes?")
    if std_b < std_a - 0.005:
        print(f"    -> {label_b} (lower AUC variance across folds)")
    elif std_a < std_b - 0.005:
        print(f"    -> {label_a} (lower AUC variance across folds)")
    else:
        print(f"    -> Similar stability")

    # Rate stability
    for metrics, name in [(metrics_a, label_a), (metrics_b, label_b)]:
        rates = metrics.drop_duplicates("fold")["pos_rate_test"]
        cv = rates.std() / rates.mean()
        print(f"\n  {name} positive-rate CV: {cv:.3f}")

    print(f"\n  Recommendation:")
    if auc_b > auc_a:
        print(f"    Switch to {label_b} for training.")
    else:
        print(f"    Keep {label_a} or investigate further.")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("  RESEARCH PHASE 2: Label Quality Comparison")
    print("=" * 70)

    print("\nLoading data and constructing labels...")
    df = load_and_label(INPUT_PATH)
    print(f"Rows with both labels: {len(df)}")

    print_label_stats(df)

    print("\nBuilding walk-forward folds...")
    folds = build_folds(df)
    print(f"Folds: {len(folds)}")

    print("\n--- Evaluating label: target_stop ---")
    metrics_a, probas_a = evaluate_label(df, folds, "label_target_stop", "target_stop")

    print("\n--- Evaluating label: median_return ---")
    metrics_b, probas_b = evaluate_label(df, folds, "label_median_return", "median_return")

    print_comparison(metrics_a, metrics_b)
    print_calibration(probas_a, probas_b)
    print_verdict(metrics_a, metrics_b)

    # Save detailed metrics
    combined = pd.concat([metrics_a, metrics_b], ignore_index=True)
    out_path = PROJECT_DIR / "output" / "label_comparison_metrics.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nDetailed metrics saved to {out_path}")


if __name__ == "__main__":
    main()
