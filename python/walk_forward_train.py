"""
Walk-forward ML training pipeline for BackTester.

Reads:   ../output/ml_dataset.csv   (exported by C++ FeatureEngine + LabelEngine)
Writes:  ../output/predictions.csv  (date,ticker,probability -- consumed by C++ PredictionLoader)
         ../output/predictions_all.csv  (full detail with model and fold columns)
         ../output/ml_metrics.csv   (per-model per-fold evaluation metrics)
         models/*.joblib            (serialized model files)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_DIR / "output" / "ml_dataset.csv"
PREDICTIONS_PATH = PROJECT_DIR / "output" / "predictions.csv"
PREDICTIONS_ALL_PATH = PROJECT_DIR / "output" / "predictions_all.csv"
METRICS_PATH = PROJECT_DIR / "output" / "ml_metrics.csv"
MODELS_DIR = SCRIPT_DIR / "models"

FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_14", "atr_14_pct", "volume_ratio_20",
    "dist_ma20", "dist_ma50", "dist_ma200",
    "rolling_vol_20",
]

# Which label column to train on. Must be in ml_dataset.csv but NOT in FEATURE_COLS.
TARGET_COLUMN = "label_median_return"

# Which model's predictions to export to predictions.csv.
# Pre-committed choice -- NOT selected using test-set performance.
EXPORT_MODEL = "logistic"

MIN_TRAIN_YEARS = 4
PURGE_TRADING_DAYS = 20  # must equal max_horizon_days used in LabelEngine

# Columns that must never appear in features (labels, targets, future-derived).
FORBIDDEN_FEATURE_COLS = {
    "label", "label_target_stop", "label_median_return",
    "forward_return", "date", "ticker", "year",
}


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found. Run the C++ exporter first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COLUMN])

    # Safety: verify no forbidden column is in features
    leaked = set(FEATURE_COLS) & FORBIDDEN_FEATURE_COLS
    assert not leaked, f"FEATURE_COLS contains forbidden columns: {leaked}"

    print(f"Loaded {len(df)} rows from {path}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Tickers: {sorted(df['ticker'].unique())}")
    print(f"  Target column: {TARGET_COLUMN}")
    print(f"  Export model: {EXPORT_MODEL}")
    print(f"  Label distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 2. Walk-forward folds
# ---------------------------------------------------------------------------

def build_folds(df: pd.DataFrame) -> list[dict]:
    """Build expanding-window walk-forward folds with per-ticker row-based purge."""
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    tickers = sorted(df["ticker"].unique())

    if len(years) < MIN_TRAIN_YEARS + 1:
        print(f"ERROR: Need at least {MIN_TRAIN_YEARS + 1} years of data, got {len(years)}")
        sys.exit(1)

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
            if len(ticker_pre) <= PURGE_TRADING_DAYS:
                purge_indices.update(ticker_pre)
            else:
                purge_indices.update(ticker_pre[-PURGE_TRADING_DAYS:])

        train_mask = pre_test_mask & ~df.index.isin(purge_indices)
        if train_mask.sum() == 0:
            continue

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        max_train_date = df.loc[train_idx, "date"].max()
        min_test_date = df.loc[test_idx, "date"].min()
        assert max_train_date < min_test_date
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(train_idx) & purge_indices) == 0

        folds.append({
            "fold": len(folds) + 1,
            "test_year": test_year,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    print(f"\nBuilt {len(folds)} walk-forward folds "
          f"(purge={PURGE_TRADING_DAYS} rows/ticker)")
    return folds


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------

def get_models(scale_pos_weight: float) -> dict:
    """Build models with class-imbalance handling.
    scale_pos_weight = num_negative / num_positive, computed from y_train only.
    """
    return {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, solver="lbfgs", class_weight="balanced",
            )),
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


# ---------------------------------------------------------------------------
# 4. Train & evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, folds: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_predictions = []
    all_metrics = []

    X = df[FEATURE_COLS]
    y = df[TARGET_COLUMN]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for fold_info in folds:
        fold = fold_info["fold"]
        test_year = fold_info["test_year"]
        train_idx = fold_info["train_idx"]
        test_idx = fold_info["test_idx"]

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        # Class weights from training labels only.
        num_positive = int(y_train.sum())
        num_negative = int(len(y_train) - num_positive)
        assert num_positive + num_negative == len(y_train)

        if num_positive == 0:
            print(f"\n--- Fold {fold} (test year {test_year}) --- SKIPPED: no positives")
            continue

        scale_pos_weight = num_negative / num_positive

        print(f"\n--- Fold {fold} (test year {test_year}) --- "
              f"{num_negative} neg / {num_positive} pos "
              f"({num_positive/len(y_train):.1%}+, spw={scale_pos_weight:.2f})")

        for model_name, model in get_models(scale_pos_weight).items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]

            joblib.dump(model, MODELS_DIR / f"{model_name}_fold{fold}.joblib")

            fold_preds = df.loc[test_idx, ["date", "ticker"]].copy()
            fold_preds["probability"] = proba
            fold_preds["model"] = model_name
            fold_preds["fold"] = fold
            all_predictions.append(fold_preds)

            try:
                auc = roc_auc_score(y_test, proba)
            except ValueError:
                auc = float("nan")

            y_pred = (proba >= 0.5).astype(int)
            metrics = {
                "model": model_name,
                "fold": fold,
                "test_year": test_year,
                "auc": round(auc, 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "brier_score": round(brier_score_loss(y_test, proba), 4),
            }
            all_metrics.append(metrics)
            marker = " <-- EXPORT" if model_name == EXPORT_MODEL else ""
            print(f"  {model_name:12s}  AUC={metrics['auc']:.4f}  "
                  f"F1={metrics['f1']:.4f}  Brier={metrics['brier_score']:.4f}{marker}")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    return predictions_df, metrics_df


# ---------------------------------------------------------------------------
# 5. Export predictions (pre-committed model, no test-set snooping)
# ---------------------------------------------------------------------------

def export_predictions(predictions_df: pd.DataFrame, metrics_df: pd.DataFrame):
    """Export predictions from the pre-committed EXPORT_MODEL only.
    No test-set AUC is used for model selection.
    """
    export_preds = predictions_df[predictions_df["model"] == EXPORT_MODEL].copy()
    export_preds["date"] = export_preds["date"].dt.strftime("%Y-%m-%d")

    # predictions.csv -- C++-compatible
    export_preds[["date", "ticker", "probability"]].to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nSaved {len(export_preds)} predictions ({EXPORT_MODEL}) to {PREDICTIONS_PATH}")

    # predictions_all.csv -- all models for analysis
    all_preds = predictions_df.copy()
    all_preds["date"] = all_preds["date"].dt.strftime("%Y-%m-%d")
    all_preds.to_csv(PREDICTIONS_ALL_PATH, index=False)
    print(f"Saved {len(all_preds)} predictions (all models) to {PREDICTIONS_ALL_PATH}")

    # ml_metrics.csv
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"Saved {len(metrics_df)} metric rows to {METRICS_PATH}")


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

def print_summary(metrics_df: pd.DataFrame):
    print("\n========== SUMMARY ==========")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Export model: {EXPORT_MODEL}")
    summary = metrics_df.groupby("model").agg(
        mean_auc=("auc", "mean"),
        mean_f1=("f1", "mean"),
        mean_brier=("brier_score", "mean"),
        folds=("fold", "count"),
    ).round(4)
    print(summary.to_string())

    # Highlight the export model
    export_metrics = metrics_df[metrics_df["model"] == EXPORT_MODEL]
    if len(export_metrics) > 0:
        print(f"\n{EXPORT_MODEL} (exported):")
        print(f"  Mean AUC:   {export_metrics['auc'].mean():.4f}")
        print(f"  Mean F1:    {export_metrics['f1'].mean():.4f}")
        print(f"  Mean Brier: {export_metrics['brier_score'].mean():.4f}")
    print("=============================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("Walk-Forward ML Training Pipeline")
    print(f"Target: {TARGET_COLUMN} | Export: {EXPORT_MODEL}")
    print("=" * 50)

    df = load_data(INPUT_PATH)
    folds = build_folds(df)
    predictions_all, metrics_df = train_and_evaluate(df, folds)
    export_predictions(predictions_all, metrics_df)
    print_summary(metrics_df)


if __name__ == "__main__":
    main()
