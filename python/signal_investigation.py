"""
Quantitative research investigation: Why is the ML signal weak?

Analyzes the ml_dataset.csv to diagnose label quality, feature quality,
predictive power, data quality, baselines, and regime dependence.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR.parent / "output" / "ml_dataset.csv"

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_14", "atr_14_pct", "volume_ratio_20",
    "dist_ma20", "dist_ma50", "dist_ma200",
    "rolling_vol_20",
]

def load():
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    return df

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

# ======================================================================
# 1. DATA QUALITY
# ======================================================================

def data_quality(df):
    section("1. DATA QUALITY")

    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Tickers: {sorted(df['ticker'].unique())}")
    print(f"Years: {df['date'].dt.year.nunique()}")

    # Class balance
    counts = df["label"].value_counts()
    print(f"\nClass balance:")
    print(f"  label=0: {counts.get(0, 0)} ({counts.get(0,0)/len(df)*100:.1f}%)")
    print(f"  label=1: {counts.get(1, 0)} ({counts.get(1,0)/len(df)*100:.1f}%)")
    print(f"  Ratio (neg/pos): {counts.get(0,1)/max(counts.get(1,1),1):.2f}")

    # Missing values
    print(f"\nMissing values per column:")
    missing = df[FEATURE_COLS + ["label"]].isnull().sum()
    for col in missing.index:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]}")
    if missing.sum() == 0:
        print(f"  None")

    # Feature distributions
    print(f"\nFeature distributions:")
    print(f"{'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Skew':>8} {'Kurt':>8}")
    print("-" * 78)
    for col in FEATURE_COLS:
        s = df[col]
        print(f"{col:<20} {s.mean():>10.4f} {s.std():>10.4f} {s.min():>10.4f} {s.max():>10.4f} "
              f"{s.skew():>8.2f} {s.kurtosis():>8.2f}")

    # Outlier check (>5 std from mean)
    print(f"\nOutliers (>5 std from mean):")
    for col in FEATURE_COLS:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        n_outliers = (z > 5).sum()
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} rows ({n_outliers/len(df)*100:.2f}%)")

    # Label leakage confirmation
    print(f"\nLabel leakage check:")
    print(f"  Columns in dataset: {list(df.columns)}")
    suspect = [c for c in df.columns if c not in FEATURE_COLS + ["date", "ticker", "label", "year"]]
    if suspect:
        print(f"  SUSPECT COLUMNS: {suspect}")
    else:
        print(f"  No suspect columns found. Feature set is clean.")

# ======================================================================
# 2. LABEL QUALITY ANALYSIS
# ======================================================================

def label_quality(df):
    section("2. LABEL QUALITY ANALYSIS")

    y = df["label"]

    # Label autocorrelation (are labels clustered in time?)
    print("Label autocorrelation (consecutive label agreement):")
    same_as_prev = (y == y.shift(1)).mean()
    print(f"  P(label_t == label_t-1): {same_as_prev:.4f}")
    print(f"  Random baseline would be: {(y.mean()**2 + (1-y.mean())**2):.4f}")

    # Label stability across decades
    df["decade"] = (df["date"].dt.year // 10) * 10
    print(f"\nLabel rate by decade:")
    for decade, group in df.groupby("decade"):
        rate = group["label"].mean()
        n = len(group)
        print(f"  {decade}s: {rate:.3f} (n={n})")

    # Label rate by year
    df["year"] = df["date"].dt.year
    yearly_rates = df.groupby("year")["label"].mean()
    print(f"\nLabel rate by year (selected):")
    for year in [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]:
        if year in yearly_rates.index:
            print(f"  {year}: {yearly_rates[year]:.3f}")

    # Label rate volatility
    print(f"\nLabel rate statistics across years:")
    print(f"  Mean: {yearly_rates.mean():.3f}")
    print(f"  Std: {yearly_rates.std():.3f}")
    print(f"  Min: {yearly_rates.min():.3f} ({yearly_rates.idxmin()})")
    print(f"  Max: {yearly_rates.max():.3f} ({yearly_rates.idxmax()})")

    # The core question: is the label definition too noisy?
    print(f"\nLabel noise diagnostic:")
    print(f"  If label_rate varies wildly across years, the target is regime-dependent.")
    print(f"  Coefficient of variation: {yearly_rates.std()/yearly_rates.mean():.3f}")

# ======================================================================
# 3. FEATURE-TARGET RELATIONSHIPS
# ======================================================================

def feature_target_analysis(df):
    section("3. FEATURE-TARGET RELATIONSHIPS")

    X = df[FEATURE_COLS]
    y = df["label"]

    # Point-biserial correlations
    print("Point-biserial correlation with label:")
    print(f"{'Feature':<20} {'Corr':>8} {'p-value':>12} {'Abs Corr':>10}")
    print("-" * 52)
    corrs = []
    for col in FEATURE_COLS:
        r, p = stats.pointbiserialr(y, X[col])
        corrs.append((col, r, p, abs(r)))
    corrs.sort(key=lambda x: -x[3])
    for col, r, p, absr in corrs:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{col:<20} {r:>8.4f} {p:>12.2e} {absr:>10.4f} {sig}")

    # Mutual information (discretized)
    print(f"\nMutual information with label:")
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    X_disc = discretizer.fit_transform(X)
    print(f"{'Feature':<20} {'MI':>10}")
    print("-" * 32)
    mis = []
    for i, col in enumerate(FEATURE_COLS):
        mi = mutual_info_score(y, X_disc[:, i])
        mis.append((col, mi))
    mis.sort(key=lambda x: -x[1])
    for col, mi in mis:
        print(f"{col:<20} {mi:>10.6f}")

    # Single-feature AUCs
    print(f"\nSingle-feature AUC (logistic regression per feature):")
    print(f"{'Feature':<20} {'AUC':>8} {'Lift vs 0.5':>12}")
    print("-" * 42)
    aucs = []
    for col in FEATURE_COLS:
        x_single = X[[col]].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_single)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(x_scaled, y)
        proba = lr.predict_proba(x_scaled)[:, 1]
        auc = roc_auc_score(y, proba)
        aucs.append((col, auc))
    aucs.sort(key=lambda x: -abs(x[1] - 0.5))
    for col, auc in aucs:
        lift = auc - 0.5
        print(f"{col:<20} {auc:>8.4f} {lift:>+12.4f}")

    # Feature redundancy (correlation matrix)
    print(f"\nFeature-feature correlations (|r| > 0.7):")
    corr_matrix = X.corr()
    high_corr = []
    for i in range(len(FEATURE_COLS)):
        for j in range(i+1, len(FEATURE_COLS)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                high_corr.append((FEATURE_COLS[i], FEATURE_COLS[j], r))
    high_corr.sort(key=lambda x: -abs(x[2]))
    if high_corr:
        for f1, f2, r in high_corr:
            print(f"  {f1} <-> {f2}: {r:.4f}")
    else:
        print("  None")

# ======================================================================
# 4. PERMUTATION IMPORTANCE (walk-forward)
# ======================================================================

def permutation_importance_analysis(df):
    section("4. PERMUTATION IMPORTANCE (last fold)")

    # Use the last fold: train on everything before 2024, test on 2024
    df["year"] = df["date"].dt.year
    train = df[df["year"] < 2024]
    test = df[df["year"] == 2024]

    if len(test) == 0 or len(train) == 0:
        print("Not enough data for permutation importance")
        return

    X_train = train[FEATURE_COLS]
    y_train = train["label"]
    X_test = test[FEATURE_COLS].copy()
    y_test = test["label"]

    num_pos = int(y_train.sum())
    num_neg = len(y_train) - num_pos
    spw = num_neg / max(num_pos, 1)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train, y_train)

    base_proba = model.predict_proba(X_test)[:, 1]
    base_auc = roc_auc_score(y_test, base_proba)
    print(f"Baseline AUC on 2024 test: {base_auc:.4f}")

    print(f"\n{'Feature':<20} {'Shuffled AUC':>14} {'AUC Drop':>10}")
    print("-" * 46)
    importances = []
    rng = np.random.RandomState(42)
    for col in FEATURE_COLS:
        X_shuffled = X_test.copy()
        X_shuffled[col] = rng.permutation(X_shuffled[col].values)
        shuffled_proba = model.predict_proba(X_shuffled)[:, 1]
        shuffled_auc = roc_auc_score(y_test, shuffled_proba)
        drop = base_auc - shuffled_auc
        importances.append((col, shuffled_auc, drop))

    importances.sort(key=lambda x: -x[2])
    for col, sauc, drop in importances:
        print(f"{col:<20} {sauc:>14.4f} {drop:>+10.4f}")

# ======================================================================
# 5. BASELINE COMPARISON (walk-forward)
# ======================================================================

def baseline_comparison(df):
    section("5. BASELINE COMPARISON (expanding window, yearly folds)")

    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())

    results = {
        "always_0": [], "logistic": [], "xgboost": [], "lightgbm": [],
    }

    # Use last 10 years as test folds for speed
    test_years = [y for y in years if y >= 2015]

    for test_year in test_years:
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]
        if len(train) < 100 or len(test) < 10:
            continue

        X_train = train[FEATURE_COLS]
        y_train = train["label"]
        X_test = test[FEATURE_COLS]
        y_test = test["label"]

        num_pos = int(y_train.sum())
        num_neg = len(y_train) - num_pos
        spw = num_neg / max(num_pos, 1)

        # Always-negative
        auc_0 = 0.5  # by definition
        results["always_0"].append(auc_0)

        # Logistic
        lr = Pipeline([("scaler", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        lr.fit(X_train, y_train)
        results["logistic"].append(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))

        # XGBoost
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            scale_pos_weight=spw, eval_metric="logloss", verbosity=0)
        xgb.fit(X_train, y_train)
        results["xgboost"].append(roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))

        # LightGBM
        lgb = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             scale_pos_weight=spw, verbose=-1)
        lgb.fit(X_train, y_train)
        results["lightgbm"].append(roc_auc_score(y_test, lgb.predict_proba(X_test)[:, 1]))

    print(f"Walk-forward AUC (test years {test_years[0]}-{test_years[-1]}):")
    print(f"{'Model':<15} {'Mean AUC':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'Folds':>6}")
    print("-" * 57)
    for name, aucs in results.items():
        if aucs:
            arr = np.array(aucs)
            print(f"{name:<15} {arr.mean():>10.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f} {len(arr):>6}")

    # Per-year breakdown
    print(f"\nPer-year AUC:")
    print(f"{'Year':<8}", end="")
    for name in ["logistic", "xgboost", "lightgbm"]:
        print(f" {name:>10}", end="")
    print()
    print("-" * 40)
    for i, year in enumerate(test_years):
        if i < len(results["logistic"]):
            print(f"{year:<8}", end="")
            for name in ["logistic", "xgboost", "lightgbm"]:
                print(f" {results[name][i]:>10.4f}", end="")
            print()

# ======================================================================
# 6. REGIME ANALYSIS
# ======================================================================

def regime_analysis(df):
    section("6. REGIME ANALYSIS")

    df["year"] = df["date"].dt.year
    df["rolling_vol_60"] = df["ret_1d"].rolling(60, min_periods=30).std()
    df["ret_60d"] = df.groupby("ticker")["ret_1d"].transform(
        lambda x: x.rolling(60, min_periods=30).sum()
    )

    # Define regimes
    vol_median = df["rolling_vol_60"].median()
    ret_median = df["ret_60d"].median()

    df["regime_vol"] = np.where(df["rolling_vol_60"] > vol_median, "high_vol", "low_vol")
    df["regime_trend"] = np.where(df["ret_60d"] > ret_median, "bull", "bear")

    print("Label rate by regime:")
    print(f"{'Regime':<15} {'Label Rate':>12} {'Count':>8} {'Pct of Data':>12}")
    print("-" * 49)
    for regime_col in ["regime_vol", "regime_trend"]:
        for regime_val, group in df.groupby(regime_col):
            rate = group["label"].mean()
            n = len(group)
            print(f"{regime_val:<15} {rate:>12.3f} {n:>8} {n/len(df)*100:>11.1f}%")

    # Train on pre-2020, test on regimes in 2020+
    print(f"\nXGBoost AUC by regime (train <2020, test 2020+):")
    train = df[df["year"] < 2020].dropna(subset=["rolling_vol_60", "ret_60d"])
    test = df[df["year"] >= 2020].dropna(subset=["rolling_vol_60", "ret_60d"])

    if len(train) > 100 and len(test) > 10:
        X_train = train[FEATURE_COLS]
        y_train = train["label"]
        num_pos = int(y_train.sum())
        spw = (len(y_train) - num_pos) / max(num_pos, 1)

        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=spw, eval_metric="logloss", verbosity=0)
        model.fit(X_train, y_train)

        test["proba"] = model.predict_proba(test[FEATURE_COLS])[:, 1]

        print(f"{'Regime':<15} {'AUC':>8} {'Count':>8}")
        print("-" * 33)
        for regime_col in ["regime_vol", "regime_trend"]:
            for regime_val, group in test.groupby(regime_col):
                if len(group) > 20 and group["label"].nunique() == 2:
                    auc = roc_auc_score(group["label"], group["proba"])
                    print(f"{regime_val:<15} {auc:>8.4f} {len(group):>8}")

    # Cleanup
    df.drop(columns=["rolling_vol_60", "ret_60d", "regime_vol", "regime_trend"],
            inplace=True, errors="ignore")

# ======================================================================
# 7. MAIN
# ======================================================================

def main():
    print("=" * 60)
    print("  SIGNAL INVESTIGATION: Why is AUC ~0.48?")
    print("=" * 60)

    df = load()

    data_quality(df)
    label_quality(df)
    feature_target_analysis(df)
    permutation_importance_analysis(df)
    baseline_comparison(df)
    regime_analysis(df)

    section("INVESTIGATION COMPLETE")
    print("Review output above. Key questions:")
    print("  - Which features have near-zero correlation with label?")
    print("  - Is the label definition stable across regimes?")
    print("  - Do any features have single-feature AUC > 0.53?")
    print("  - Are features highly correlated with each other (redundant)?")
    print("  - Does the model perform differently in bull vs bear regimes?")

if __name__ == "__main__":
    main()
