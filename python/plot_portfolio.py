"""Plot a snapshot of the current portfolio.

Three panels on one figure:
  1. Allocation     -- each position's current market value + cash, as a share
                       of total equity (horizontal bar).
  2. Per-position   -- unrealized return vs entry for each holding, drawn against
     P&L             the stop (-STOP_LOSS_PCT) and target (+TARGET_PROFIT_PCT)
                       bands so you can see how close each is to an exit.
  3. Equity summary -- headline totals (equity, cash, invested, unrealized /
                       realized P&L, return on starting capital).

Positions are valued at the LATEST LOCAL price bar (ticker_data via
load_prices), so this runs fully offline -- no IBKR connection needed. If a
ticker has no price data the entry price is used and the row is flagged stale.

Usage:
    python plot_portfolio.py                      # paper state -> output/portfolio_chart.png
    python plot_portfolio.py --mode sim           # graph sim state instead
    python plot_portfolio.py --out my_chart.png   # custom output path
    python plot_portfolio.py --show               # also open an interactive window
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
# Default to a non-interactive backend (save-only). --show switches to a GUI
# backend; decided before pyplot picks one.
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    state_file_for_mode, OUTPUT_DIR,
    STOP_LOSS_PCT, TARGET_PROFIT_PCT,
)
from feature_engine import load_prices

GAIN = "#1b9e4b"
LOSS = "#d62728"
CASH = "#7f7f7f"
POS = "#3b6fb0"
GRID = "#dddddd"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_state(mode: str):
    """Load a portfolio state JSON for the given mode (sim/paper)."""
    path = state_file_for_mode(mode)
    if not path.exists():
        print(f"ERROR: no state file for mode '{mode}': {path}")
        print("Run the pipeline first, or pass --mode sim/ibkr_paper.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f), path


def latest_price_and_date(ticker: str):
    """Most recent local close and its bar date, or (None, None) if unavailable."""
    try:
        prices = load_prices(ticker)
        last = prices.iloc[-1]
        return float(last["close"]), str(last["date"])[:10]
    except Exception:
        return None, None


def build_rows(positions: dict):
    """Per-position valuation rows using the latest local bar."""
    rows = []
    for ticker, pos in positions.items():
        shares = float(pos.get("shares", 0))
        entry = float(pos.get("entry_price", 0) or 0)
        stop = float(pos.get("stop_price", 0) or 0)
        target = float(pos.get("target_price", 0) or 0)
        price, pdate = latest_price_and_date(ticker)
        stale = price is None
        if stale:
            price = entry
        cur_val = shares * price
        cost = shares * entry
        pnl = cur_val - cost
        pnl_pct = (price / entry - 1.0) if entry > 0 else 0.0
        rows.append({
            "ticker": ticker, "shares": shares, "entry": entry, "price": price,
            "stop": stop, "target": target, "cur_val": cur_val, "cost": cost,
            "pnl": pnl, "pnl_pct": pnl_pct, "date": pdate, "stale": stale,
        })
    rows.sort(key=lambda r: r["cur_val"], reverse=True)
    return rows


def summarize(state: dict, rows: list):
    cash = float(state.get("cash", 0))
    invested_cost = sum(r["cost"] for r in rows)
    invested_val = sum(r["cur_val"] for r in rows)
    equity = cash + invested_val
    start = float(state.get("starting_capital", equity) or equity)
    unreal = invested_val - invested_cost
    realized = sum(float(t.get("pnl", 0)) for t in state.get("trade_history", []))
    as_of = max((r["date"] for r in rows if r["date"]), default=None)
    return {
        "cash": cash, "invested_cost": invested_cost, "invested_val": invested_val,
        "equity": equity, "start": start, "unreal": unreal, "realized": realized,
        "total_ret": (equity / start - 1.0) if start > 0 else 0.0,
        "as_of": as_of, "n_positions": len(rows),
    }


def compute_metrics(state: dict, rows: list, s: dict) -> dict:
    """Snapshot quant metrics: exposure, concentration, risk budget, and
    (when there are closed trades) realized performance stats.

    Risk/reward are measured from the CURRENT price to each position's stop /
    target -- i.e. the dollars given up if every stop filled, vs gained if every
    target filled -- so it reflects live risk, not the risk planned at entry.
    """
    eq = s["equity"]
    inv = s["invested_val"]
    n = len(rows)
    m = {"n_positions": n}

    # --- Exposure ---
    m["gross_exposure"] = inv
    m["gross_exposure_pct"] = (inv / eq) if eq else 0.0
    m["cash_pct"] = (s["cash"] / eq) if eq else 0.0

    # --- Concentration (over the invested basket) ---
    weights = [r["cur_val"] / inv for r in rows] if inv > 0 else []
    hhi = sum(w * w for w in weights)
    m["hhi"] = hhi
    m["effective_n"] = (1.0 / hhi) if hhi > 0 else 0.0
    m["largest_weight"] = max((r["cur_val"] / eq for r in rows), default=0.0)

    # --- Risk budget: current price -> stop / target ---
    risk = sum(r["shares"] * (r["price"] - r["stop"])
               for r in rows if r["stop"] > 0)
    reward = sum(r["shares"] * (r["target"] - r["price"])
                 for r in rows if r["target"] > 0)
    m["risk_to_stops"] = risk
    m["risk_to_stops_pct"] = (risk / eq) if eq else 0.0
    m["reward_to_targets"] = reward
    m["reward_to_targets_pct"] = (reward / eq) if eq else 0.0
    m["reward_risk"] = (reward / risk) if risk > 0 else 0.0

    # --- Open-position performance ---
    m["unrealized"] = s["unreal"]
    m["open_win_rate"] = (sum(1 for r in rows if r["pnl"] > 0) / n) if n else 0.0
    m["avg_pos_return"] = (sum(r["pnl_pct"] for r in rows) / n) if n else 0.0
    m["best"] = max(rows, key=lambda r: r["pnl_pct"], default=None)
    m["worst"] = min(rows, key=lambda r: r["pnl_pct"], default=None)

    # --- Realized performance (closed trades) ---
    th = state.get("trade_history", [])
    pnls = [float(t.get("pnl", 0)) for t in th]
    rets = [float(t.get("return_pct", 0)) for t in th]
    m["n_closed"] = len(th)
    if th:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_win = sum(wins)
        gross_loss = -sum(losses)
        m["realized_win_rate"] = len(wins) / len(pnls)
        m["profit_factor"] = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        m["avg_win"] = (gross_win / len(wins)) if wins else 0.0
        m["avg_loss"] = (sum(losses) / len(losses)) if losses else 0.0
        m["avg_trade_ret"] = sum(rets) / len(rets)
        m["expectancy"] = sum(pnls) / len(pnls)
    return m


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot_allocation(ax, rows, cash):
    labels = [r["ticker"] for r in rows] + ["CASH"]
    values = [r["cur_val"] for r in rows] + [cash]
    colors = [POS] * len(rows) + [CASH]
    total = sum(values) or 1.0
    y = range(len(labels))
    ax.barh(list(y), values, color=colors, edgecolor="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + total * 0.01, i, f"${v:,.0f}  ({v / total:.1%})",
                va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.28 if values else 1)
    ax.set_xlabel("Market value ($)")
    ax.set_title("Allocation (current value + cash)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_pnl(ax, rows):
    if not rows:
        ax.text(0.5, 0.5, "No open positions", ha="center", va="center")
        ax.axis("off")
        return
    tickers = [r["ticker"] for r in rows]
    pcts = [r["pnl_pct"] * 100 for r in rows]
    x = range(len(rows))
    colors = [GAIN if p >= 0 else LOSS for p in pcts]

    # Stop / target reference bands (uniform pct from entry).
    ax.axhspan(-STOP_LOSS_PCT * 100, TARGET_PROFIT_PCT * 100,
               color="#eef6ee", zorder=0)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axhline(TARGET_PROFIT_PCT * 100, color=GAIN, linestyle="--", linewidth=1.0)
    ax.axhline(-STOP_LOSS_PCT * 100, color=LOSS, linestyle="--", linewidth=1.0)
    ax.text(len(rows) - 0.4, TARGET_PROFIT_PCT * 100, " target",
            color=GAIN, va="bottom", ha="right", fontsize=8)
    ax.text(len(rows) - 0.4, -STOP_LOSS_PCT * 100, " stop",
            color=LOSS, va="top", ha="right", fontsize=8)

    ax.bar(list(x), pcts, color=colors, edgecolor="white", zorder=2)
    for i, r in enumerate(rows):
        p = r["pnl_pct"] * 100
        va = "bottom" if p >= 0 else "top"
        off = 0.25 if p >= 0 else -0.25
        flag = "*" if r["stale"] else ""
        ax.text(i, p + off, f"${r['pnl']:+,.0f}{flag}\n@${r['price']:,.2f}",
                ha="center", va=va, fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Unrealized return vs entry (%)")
    lo = min(pcts + [-STOP_LOSS_PCT * 100]) - 3
    hi = max(pcts + [TARGET_PROFIT_PCT * 100]) + 3
    ax.set_ylim(lo, hi)
    ax.set_title("Per-position P&L vs stop / target", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_summary(ax, s):
    ax.axis("off")
    unreal_c = GAIN if s["unreal"] >= 0 else LOSS
    ret_c = GAIN if s["total_ret"] >= 0 else LOSS
    invested_pct = (s["invested_val"] / s["equity"]) if s["equity"] else 0
    cash_pct = (s["cash"] / s["equity"]) if s["equity"] else 0

    lines = [
        ("Total equity", f"${s['equity']:,.2f}", "black"),
        ("Starting capital", f"${s['start']:,.2f}", "black"),
        ("Total return", f"{s['total_ret']:+.2%}", ret_c),
        ("", "", "black"),
        ("Cash", f"${s['cash']:,.2f}  ({cash_pct:.0%})", "black"),
        ("Invested (mkt value)", f"${s['invested_val']:,.2f}  ({invested_pct:.0%})", "black"),
        ("Invested (cost basis)", f"${s['invested_cost']:,.2f}", "black"),
        ("", "", "black"),
        ("Unrealized P&L", f"${s['unreal']:+,.2f}", unreal_c),
        ("Realized P&L", f"${s['realized']:+,.2f}",
         GAIN if s["realized"] >= 0 else LOSS),
        ("Open positions", f"{s['n_positions']}", "black"),
    ]
    y = 0.97
    ax.text(0.0, y, "Equity summary", fontweight="bold", fontsize=12,
            transform=ax.transAxes)
    y -= 0.11
    for label, value, color in lines:
        if label:
            ax.text(0.0, y, label, fontsize=10, transform=ax.transAxes,
                    color="#444444")
            ax.text(1.0, y, value, fontsize=10, transform=ax.transAxes,
                    color=color, ha="right", fontweight="bold")
        y -= 0.083


def _fmt_pf(pf):
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def plot_metrics(ax, m):
    ax.axis("off")
    rr_c = GAIN if m["reward_risk"] >= 1 else LOSS

    left = [
        ("Gross exposure", f"${m['gross_exposure']:,.0f}  ({m['gross_exposure_pct']:.0%})", "black"),
        ("Cash weight", f"{m['cash_pct']:.0%}", "black"),
        ("Positions", f"{m['n_positions']}  (eff. {m['effective_n']:.1f})", "black"),
        ("Largest weight", f"{m['largest_weight']:.1%} of equity", "black"),
        ("Concentration (HHI)", f"{m['hhi']:.3f}", "black"),
        ("Risk to stops", f"${m['risk_to_stops']:,.0f}  ({m['risk_to_stops_pct']:.2%})", LOSS),
        ("Reward to targets", f"${m['reward_to_targets']:,.0f}  ({m['reward_to_targets_pct']:.2%})", GAIN),
        ("Reward : Risk", f"{m['reward_risk']:.2f} : 1", rr_c),
    ]

    best, worst = m["best"], m["worst"]
    right = [
        ("Unrealized P&L", f"${m['unrealized']:+,.2f}", GAIN if m["unrealized"] >= 0 else LOSS),
        ("Open win rate", f"{m['open_win_rate']:.0%}", "black"),
        ("Avg position return", f"{m['avg_pos_return']:+.2%}", "black"),
        ("Best", f"{best['ticker']} {best['pnl_pct']:+.1%}" if best else "--", GAIN),
        ("Worst", f"{worst['ticker']} {worst['pnl_pct']:+.1%}" if worst else "--", LOSS),
    ]
    if m["n_closed"]:
        right += [
            ("Closed trades", f"{m['n_closed']}", "black"),
            ("Realized win rate", f"{m['realized_win_rate']:.0%}", "black"),
            ("Profit factor", _fmt_pf(m["profit_factor"]), "black"),
            ("Expectancy / trade", f"${m['expectancy']:+,.2f}", "black"),
        ]
    else:
        right += [("Closed trades", "0 (no realized stats yet)", "#888888")]

    ax.text(0.0, 1.0, "Quant metrics", fontweight="bold", fontsize=12,
            transform=ax.transAxes)

    def _col(items, x_label, x_val):
        y = 0.86
        for label, value, color in items:
            ax.text(x_label, y, label, fontsize=9.5, color="#444444",
                    transform=ax.transAxes)
            ax.text(x_val, y, value, fontsize=9.5, color=color, ha="right",
                    fontweight="bold", transform=ax.transAxes)
            y -= 0.105

    _col(left, 0.0, 0.46)
    _col(right, 0.54, 1.0)


def make_figure(state, rows, summary, mode, show_metrics=False, metrics=None):
    if show_metrics:
        fig = plt.figure(figsize=(13, 11.5))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.05, 0.95],
                      hspace=0.42, wspace=0.22)
    else:
        fig = plt.figure(figsize=(13, 9))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.1],
                      hspace=0.32, wspace=0.22)
    plot_allocation(fig.add_subplot(gs[0, 0]), rows, summary["cash"])
    plot_summary(fig.add_subplot(gs[0, 1]), summary)
    plot_pnl(fig.add_subplot(gs[1, :]), rows)
    if show_metrics:
        plot_metrics(fig.add_subplot(gs[2, :]), metrics)

    as_of = summary["as_of"] or "n/a"
    stale_note = "   (* = no price data, valued at entry)" if any(
        r["stale"] for r in rows) else ""
    fig.suptitle(
        f"Portfolio snapshot -- {mode}   |   as of {as_of}   |   "
        f"equity ${summary['equity']:,.0f}   ({summary['total_ret']:+.2%}){stale_note}",
        fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97, bottom=0.07)
    return fig


# ----------------------------------------------------------------------
# Console
# ----------------------------------------------------------------------

def print_table(rows, summary):
    print(f"\n  {'Ticker':<7}{'Shares':>8}{'Entry':>10}{'Price':>10}"
          f"{'Value':>12}{'P&L':>11}{'P&L%':>8}")
    print(f"  {'-' * 64}")
    for r in rows:
        flag = "*" if r["stale"] else " "
        print(f"  {r['ticker']:<7}{r['shares']:>8.0f}{r['entry']:>10.2f}"
              f"{r['price']:>9.2f}{flag}{r['cur_val']:>12,.2f}"
              f"{r['pnl']:>+11,.2f}{r['pnl_pct']:>+8.2%}")
    print(f"  {'-' * 64}")
    print(f"  Cash ${summary['cash']:,.2f}   Equity ${summary['equity']:,.2f}   "
          f"Unrealized ${summary['unreal']:+,.2f}   "
          f"Return {summary['total_ret']:+.2%}")


def print_metrics(m):
    print(f"\n  Quant metrics")
    print(f"  {'-' * 64}")
    print(f"  Gross exposure      ${m['gross_exposure']:,.0f} ({m['gross_exposure_pct']:.0%})"
          f"      Cash weight        {m['cash_pct']:.0%}")
    print(f"  Positions           {m['n_positions']} (eff. {m['effective_n']:.1f})"
          f"            Largest weight     {m['largest_weight']:.1%} of equity")
    print(f"  Concentration HHI   {m['hhi']:.3f}"
          f"                 Open win rate      {m['open_win_rate']:.0%}")
    print(f"  Risk to stops       ${m['risk_to_stops']:,.0f} ({m['risk_to_stops_pct']:.2%})"
          f"      Reward to targets  ${m['reward_to_targets']:,.0f} ({m['reward_to_targets_pct']:.2%})")
    print(f"  Reward : Risk       {m['reward_risk']:.2f} : 1"
          f"             Avg position ret   {m['avg_pos_return']:+.2%}")
    if m["best"] and m["worst"]:
        print(f"  Best {m['best']['ticker']} {m['best']['pnl_pct']:+.1%}"
              f"              Worst {m['worst']['ticker']} {m['worst']['pnl_pct']:+.1%}")
    if m["n_closed"]:
        print(f"  Closed trades       {m['n_closed']}"
              f"                  Realized win rate  {m['realized_win_rate']:.0%}")
        print(f"  Profit factor       {_fmt_pf(m['profit_factor'])}"
              f"                Expectancy/trade   ${m['expectancy']:+,.2f}")
    else:
        print(f"  Closed trades       0 (no realized stats yet)")
    print(f"  {'-' * 64}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot the current portfolio snapshot")
    p.add_argument("--mode", default="ibkr_paper",
                   choices=["sim", "ibkr_paper", "ibkr_dry_run"],
                   help="Which state file to graph (default: ibkr_paper)")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: output/portfolio_chart.png)")
    p.add_argument("--show", action="store_true",
                   help="Open an interactive window in addition to saving")
    p.add_argument("--metrics", action="store_true",
                   help="Compute and display quant metrics (exposure, concentration, "
                        "risk-to-stops / reward-to-targets, win rate, realized stats) "
                        "in the console and as an extra panel on the figure")
    return p.parse_args()


def main():
    args = parse_args()
    state, path = load_state(args.mode)
    rows = build_rows(state.get("open_positions", {}))
    summary = summarize(state, rows)

    print(f"Loaded {args.mode} state: {path}")
    print(f"  {summary['n_positions']} positions, cash ${summary['cash']:,.2f}, "
          f"as of {summary['as_of'] or 'n/a'}")
    print_table(rows, summary)

    metrics = None
    if args.metrics:
        metrics = compute_metrics(state, rows, summary)
        print_metrics(metrics)

    fig = make_figure(state, rows, summary, args.mode,
                      show_metrics=args.metrics, metrics=metrics)
    out = Path(args.out) if args.out else (OUTPUT_DIR / "portfolio_chart.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\n  Saved chart -> {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
