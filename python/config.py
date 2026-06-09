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
RISK_PER_TRADE = 0.004      # position size = equity * (RISK_PER_TRADE / STOP_LOSS_PCT) = ~8% of equity
MAX_POSITION_FRAC = 0.30    # per-order cash cap; loose enough to fill the 10th position as cash depletes
STOP_LOSS_PCT = 0.05
TARGET_PROFIT_PCT = 0.10
MAX_HOLD_DAYS = 20

# ---------------------------------------------------------------------------
# IBKR paper trading
# ---------------------------------------------------------------------------

# The IBKR paper account is funded far above what this strategy should trade
# (e.g. ~$1,000,000). The strategy ledger (portfolio_state.paper.json) is sized
# independently from the broker balance: we use the broker ONLY for execution
# and position verification, never to size positions off the full account.
IBKR_PAPER_STRATEGY_CAPITAL = 10000.0

# Pre-flight commits at most this fraction of broker available funds. Acts as a
# buying-power sufficiency gate (not an equity-equality gate).
IBKR_BUYING_POWER_SAFETY_FRAC = 0.95

# IBKR rejects fractional share orders, so every IBKR-bound order is floored to
# a whole number and dropped if it rounds below one share.
ALLOW_FRACTIONAL_IBKR = False


def floor_shares_for_ibkr(shares: float) -> int:
    """Floor a (possibly fractional) share count to whole shares for IBKR.

    Single source of truth for integer-share enforcement. Returns 0 for any
    quantity that rounds below one share so the caller can skip the order.
    Negative inputs clamp to 0 (no shorting via sizing bugs).
    """
    import math
    if shares != shares or shares <= 0:  # NaN or non-positive
        return 0
    return int(math.floor(shares))


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
