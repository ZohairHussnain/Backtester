"""Model loading and inference for production predictions."""

import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLS, EXPORT_MODEL, MODELS_DIR


class Predictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.model_path = None
        self.fold = None

    def load_latest_model(self) -> None:
        """Load the most recent fold's model. Raises on failure."""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        pattern = re.compile(rf"^{EXPORT_MODEL}_fold(\d+)\.joblib$")
        candidates = []
        for f in self.models_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                candidates.append((int(m.group(1)), f))

        if not candidates:
            raise FileNotFoundError(f"No {EXPORT_MODEL} models found in {self.models_dir}")

        candidates.sort(key=lambda x: x[0])
        self.fold, self.model_path = candidates[-1]

        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_path}: {e}") from e

        print(f"Loaded model: {self.model_path.name} (fold {self.fold})")

    def validate_feature_schema(self, feature_df: pd.DataFrame) -> None:
        """Raise ValueError if feature columns don't match expected schema."""
        missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Generate probabilities for all rows in feature_df.

        Returns DataFrame with columns: date, ticker, probability.
        Drops rows with NaN features (logs which tickers were dropped).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_latest_model() first.")

        self.validate_feature_schema(feature_df)
        X = feature_df[FEATURE_COLS]

        # Check for NaN/inf
        bad_mask = X.isnull().any(axis=1) | np.isinf(X).any(axis=1)
        if bad_mask.any():
            dropped = feature_df.loc[bad_mask, "ticker"].tolist()
            print(f"  WARNING: Dropping {len(dropped)} rows with bad features: {dropped}")
            X = X[~bad_mask]
            feature_df = feature_df[~bad_mask]

        if X.empty:
            return pd.DataFrame(columns=["date", "ticker", "probability"])

        try:
            proba = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            raise RuntimeError(f"predict_proba failed: {e}") from e

        result = feature_df[["date", "ticker"]].copy()
        result["probability"] = proba
        return result.reset_index(drop=True)

    def get_metadata(self) -> dict:
        return {
            "model_type": EXPORT_MODEL,
            "model_path": str(self.model_path) if self.model_path else None,
            "fold": self.fold,
            "feature_count": len(FEATURE_COLS),
            "feature_cols": FEATURE_COLS,
            "loaded_at": datetime.now().isoformat(),
        }
