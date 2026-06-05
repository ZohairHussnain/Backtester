"""Centralized configuration for the production trading pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TICKER_DATA_DIR = PROJECT_DIR / "ticker_data"
OUTPUT_DIR = PROJECT_DIR / "output"
MODELS_DIR = SCRIPT_DIR / "models"

UNIVERSE_FILE = TICKER_DATA_DIR / "universe.txt"
PORTFOLIO_STATE_FILE = OUTPUT_DIR / "portfolio_state.json"
ORDERS_FILE = OUTPUT_DIR / "orders.csv"
DAILY_REPORT_FILE = OUTPUT_DIR / "daily_report.html"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

EXPORT_MODEL = "logistic"
TARGET_COLUMN = "label_median_return"

# Features — must match training exactly. Order matters.
FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "rsi_14", "atr_14_pct", "volume_ratio_20",
    "dist_ma20", "dist_ma50", "dist_ma200",
    "rolling_vol_20",
]

# Columns that must NEVER be used as features.
FORBIDDEN_FEATURE_COLS = {
    "label", "label_target_stop", "label_median_return",
    "forward_return", "date", "ticker", "year",
}

# Minimum bars of history required to compute all features.
MINIMUM_HISTORY = 200

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

THRESHOLD = 0.60
TOP_N = 2
MAX_POSITIONS = 2
STARTING_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01
MAX_POSITION_FRAC = 0.40
STOP_LOSS_PCT = 0.05
TARGET_PROFIT_PCT = 0.10
MAX_HOLD_DAYS = 20

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

SLIPPAGE = 0.0005
FEE_PER_SHARE = 0.0035
FEE_MIN = 0.35
FEE_CAP_PCT = 0.01

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

MAX_PORTFOLIO_RISK_PCT = 0.02
DRY_RUN = True
