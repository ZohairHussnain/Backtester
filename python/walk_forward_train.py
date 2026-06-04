"""
Walk-forward ML training pipeline for BackTester.

Reads:   ../output/ml_dataset.csv   (exported by C++ FeatureEngine + LabelEngine)
Writes:  ../output/predictions.csv  (date,ticker,probability — consumed by C++ PredictionLoader)
         ../output/predictions_all.csv  (full detail with model and fold columns)
         ../output/ml_metrics.csv   (per-model per-fold evaluation metrics)
         models/*.joblib            (serialized model files)
"""

import os
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Paths
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

MIN_TRAIN_YEARS = 4  # require at least 4 years of training data before first test fold


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found. Run the C++ exporter first.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS + ["label"])

    print(f"Loaded {len(df)} rows from {path}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Tickers: {sorted(df['ticker'].unique())}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# 2. Walk-forward folds
# ---------------------------------------------------------------------------

def build_folds(df: pd.DataFrame) -> list[dict]:
    """Build expanding-window walk-forward folds. Each test set is one calendar year."""
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())

    if len(years) < MIN_TRAIN_YEARS + 1:
        print(f"ERROR: Need at least {MIN_TRAIN_YEARS + 1} years of data, got {len(years)}")
        sys.exit(1)

    folds = []
    for i in range(MIN_TRAIN_YEARS, len(years)):
        test_year = years[i]
        train_mask = df["year"] < test_year
        test_mask = df["year"] == test_year

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        folds.append({
            "fold": len(folds) + 1,
            "test_year": test_year,
            "train_idx": df.index[train_mask],
            "test_idx": df.index[test_mask],
        })

    print(f"\nBuilt {len(folds)} walk-forward folds:")
    for f in folds:
        print(f"  Fold {f['fold']}: train={len(f['train_idx'])} rows, "
              f"test={len(f['test_idx'])} rows (year {f['test_year']})")
    return folds


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------

def get_models() -> dict:
    return {
        "logistic": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", verbosity=0, use_label_encoder=False,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, verbose=-1,
        ),
    }


# ---------------------------------------------------------------------------
# 4. Train & evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, folds: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all_predictions_df, metrics_df)."""
    all_predictions = []
    all_metrics = []

    X = df[FEATURE_COLS]
    y = df["label"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for fold_info in folds:
        fold = fold_info["fold"]
        test_year = fold_info["test_year"]
        train_idx = fold_info["train_idx"]
        test_idx = fold_info["test_idx"]

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        print(f"\n--- Fold {fold} (test year {test_year}) ---")

        for model_name, model in get_models().items():
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]

            # Save model
            model_path = MODELS_DIR / f"{model_name}_fold{fold}.joblib"
            joblib.dump(model, model_path)

            # Predictions
            fold_preds = df.loc[test_idx, ["date", "ticker"]].copy()
            fold_preds["probability"] = proba
            fold_preds["model"] = model_name
            fold_preds["fold"] = fold
            all_predictions.append(fold_preds)

            # Metrics
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
            print(f"  {model_name:12s}  AUC={metrics['auc']:.4f}  "
                  f"F1={metrics['f1']:.4f}  Brier={metrics['brier_score']:.4f}")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    return predictions_df, metrics_df


# ---------------------------------------------------------------------------
# 5. Select best model per fold & save
# ---------------------------------------------------------------------------

def select_best_predictions(predictions_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """For each fold, pick the model with the highest AUC."""
    best_per_fold = metrics_df.loc[metrics_df.groupby("fold")["auc"].idxmax()]
    print("\n--- Best model per fold ---")
    for _, row in best_per_fold.iterrows():
        print(f"  Fold {int(row['fold'])}: {row['model']} (AUC={row['auc']:.4f})")

    best_rows = []
    for _, row in best_per_fold.iterrows():
        mask = (predictions_df["fold"] == row["fold"]) & (predictions_df["model"] == row["model"])
        best_rows.append(predictions_df.loc[mask])

    best_df = pd.concat(best_rows, ignore_index=True)
    return best_df[["date", "ticker", "probability"]]


def save_outputs(
    predictions_all: pd.DataFrame,
    best_predictions: pd.DataFrame,
    metrics_df: pd.DataFrame,
):
    # predictions.csv — C++-compatible (date,ticker,probability)
    best_predictions = best_predictions.copy()
    best_predictions["date"] = best_predictions["date"].dt.strftime("%Y-%m-%d")
    best_predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nSaved {len(best_predictions)} predictions to {PREDICTIONS_PATH}")

    # predictions_all.csv — full detail
    predictions_all = predictions_all.copy()
    predictions_all["date"] = predictions_all["date"].dt.strftime("%Y-%m-%d")
    predictions_all.to_csv(PREDICTIONS_ALL_PATH, index=False)
    print(f"Saved {len(predictions_all)} predictions (all models) to {PREDICTIONS_ALL_PATH}")

    # ml_metrics.csv
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"Saved {len(metrics_df)} metric rows to {METRICS_PATH}")


# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

def print_summary(metrics_df: pd.DataFrame):
    print("\n========== SUMMARY ==========")
    summary = metrics_df.groupby("model").agg(
        mean_auc=("auc", "mean"),
        mean_f1=("f1", "mean"),
        mean_brier=("brier_score", "mean"),
        folds=("fold", "count"),
    ).round(4)
    print(summary.to_string())
    print("=============================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("Walk-Forward ML Training Pipeline")
    print("=" * 50)

    df = load_data(INPUT_PATH)
    folds = build_folds(df)
    predictions_all, metrics_df = train_and_evaluate(df, folds)
    best_predictions = select_best_predictions(predictions_all, metrics_df)
    save_outputs(predictions_all, best_predictions, metrics_df)
    print_summary(metrics_df)


if __name__ == "__main__":
    main()
