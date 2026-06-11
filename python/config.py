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

# Portfolio state files.
#   PORTFOLIO_STATE_FILE        — legacy path (pre Phase 1). Kept for migration
#                                 and as the default for the Portfolio class.
#   PORTFOLIO_STATE_FILE_SIM    — sim mode only. Fully isolated test state.
#   PORTFOLIO_STATE_FILE_PAPER  — IBKR dry-run / paper. The source of truth that
#                                 must never be contaminated by sim testing.
PORTFOLIO_STATE_FILE = OUTPUT_DIR / "portfolio_state.json"
PORTFOLIO_STATE_FILE_SIM = OUTPUT_DIR / "portfolio_state.sim.json"
PORTFOLIO_STATE_FILE_PAPER = OUTPUT_DIR / "portfolio_state.paper.json"

# Which state file each run mode reads/writes. Separating sim from paper means
# a sim run can no longer add positions to — or advance the duplicate-run
# timestamp of — the state that IBKR paper mode trusts.
STATE_FILE_BY_MODE = {
    "sim": PORTFOLIO_STATE_FILE_SIM,
    "ibkr_dry_run": PORTFOLIO_STATE_FILE_PAPER,
    "ibkr_paper": PORTFOLIO_STATE_FILE_PAPER,
}


def state_file_for_mode(mode: str):
    """Return the portfolio state file a given run mode must use.

    Unknown modes fall back to the isolated sim file so that an accidental
    new mode can never write to paper state.
    """
    return STATE_FILE_BY_MODE.get(mode, PORTFOLIO_STATE_FILE_SIM)


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
TOP_N = 5
MAX_POSITIONS = 10
STARTING_CAPITAL = 10000.0

# IBKR paper trading runs the strategy as a SUB-ALLOCATION inside a much larger
# IBKR paper account (auto-funded at ~$1M+). The strategy must size off this
# capital, NOT the full broker balance. Local portfolio_state.paper.json is the
# strategy's own cash-envelope ledger; broker NetLiquidation includes unrelated
# funds. Pre-flight therefore does NOT require broker == local equity.
IBKR_PAPER_STRATEGY_CAPITAL = 10000.0
RISK_PER_TRADE = 0.004      # position size = equity * (RISK_PER_TRADE / STOP_LOSS_PCT) = ~8% of equity
MAX_POSITION_FRAC = 0.30    # per-order cash cap; loose enough to fill the 10th position as cash depletes
STOP_LOSS_PCT = 0.05
TARGET_PROFIT_PCT = 0.10
MAX_HOLD_DAYS = 20

# ---------------------------------------------------------------------------
# Rotation / replacement exit
# ---------------------------------------------------------------------------
# When the book is full, optionally replace the weakest current holding with a
# materially stronger new candidate. "Strength" is the per-stock model
# PREDICTION PROBABILITY for today (NOT AUC, which is a model-level metric).
# Disabled by default; enable only after a dry-run review. Do not tune yet.
ROTATION_ENABLED = True
# A candidate must beat the weakest holding's probability by at least this much
# to justify the turnover (anti-churn guard).
ROTATION_MIN_PROB_IMPROVEMENT = 0.05
# If True, only consider rotation when MAX_POSITIONS is already full (no free
# slot). If False, rotation may also run while slots remain.
ROTATION_ONLY_WHEN_FULL = True
# Cap rotations per day to limit turnover.
ROTATION_MAX_PER_DAY = 1

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

# Maximum age (in TRADING SESSIONS) of the latest price bar the pipeline trades
# on. Counted as NYSE sessions strictly after the latest bar up to and including
# today, so weekends never inflate it (a Friday bar read Monday = 1 session) and
# market holidays are excluded when pandas_market_calendars is installed
# (weekday-only fallback otherwise). If the freshest bar is older than this, the
# daily price update likely failed (yfinance down / network) and ibkr_paper
# submission is blocked. 3 sessions tolerates a one-session data delay plus a
# normal lag. Override a single run with --allow-stale-data.
MAX_DATA_STALENESS_TRADING_DAYS = 3
