"""Daily HTML report generator with XSS-safe output."""

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import DAILY_REPORT_FILE, EXPORT_MODEL, THRESHOLD, TOP_N, MAX_POSITIONS


def _esc(val) -> str:
    return html.escape(str(val))


class Reporter:
    def generate_report(self, portfolio_state: dict,
                        orders: pd.DataFrame,
                        predictions: pd.DataFrame,
                        output_path: Path = DAILY_REPORT_FILE) -> str:
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cash = portfolio_state.get("cash", 0)
        positions = portfolio_state.get("open_positions", {})
        history = portfolio_state.get("trade_history", [])
        starting = portfolio_state.get("starting_capital", 10000)

        equity = cash
        for _, pos in positions.items():
            equity += pos["shares"] * pos["entry_price"]

        total_return = (equity / starting - 1) * 100 if starting > 0 else 0
        realized_pnl = sum(t.get("pnl", 0) for t in history)
        ret_class = "positive" if total_return >= 0 else "negative"
        pnl_class = "positive" if realized_pnl >= 0 else "negative"

        # Top predictions
        top_preds = ""
        if len(predictions) > 0:
            top = predictions.nlargest(10, "probability")
            for _, r in top.iterrows():
                top_preds += (f"<tr><td>{_esc(r['ticker'])}</td>"
                             f"<td>{r['probability']:.4f}</td></tr>\n")

        # Open positions
        pos_rows = ""
        for ticker, pos in positions.items():
            pos_rows += (f"<tr><td>{_esc(ticker)}</td>"
                        f"<td>{pos['shares']:.2f}</td>"
                        f"<td>${pos['entry_price']:.2f}</td>"
                        f"<td>{_esc(pos['entry_date'])}</td>"
                        f"<td>${pos.get('stop_price',0):.2f}</td>"
                        f"<td>${pos.get('target_price',0):.2f}</td></tr>\n")

        # Orders
        order_rows = ""
        if len(orders) > 0:
            for _, o in orders.iterrows():
                order_rows += (f"<tr><td>{_esc(o.get('action',''))}</td>"
                              f"<td>{_esc(o.get('ticker',''))}</td>"
                              f"<td>{o.get('shares','')}</td>"
                              f"<td>{o.get('probability','')}</td>"
                              f"<td>{_esc(o.get('reason',''))}</td></tr>\n")

        # Recent trades
        trade_rows = ""
        for t in history[-10:]:
            pnl = t.get("pnl", 0)
            pnl_cls = "positive" if pnl >= 0 else "negative"
            trade_rows += (f"<tr><td>{_esc(t.get('ticker',''))}</td>"
                          f"<td>{_esc(t.get('entry_date',''))}</td>"
                          f"<td>{_esc(t.get('exit_date',''))}</td>"
                          f"<td>${t.get('entry_price',0):.2f}</td>"
                          f"<td>${t.get('exit_price',0):.2f}</td>"
                          f"<td class='{pnl_cls}'>${pnl:.2f}</td>"
                          f"<td>{_esc(t.get('reason',''))}</td></tr>\n")

        html_content = f"""<!DOCTYPE html>
<html><head><title>Daily Report {_esc(today)}</title>
<style>
  body {{ font-family: monospace; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1, h2 {{ color: #00d4ff; }}
  table {{ border-collapse: collapse; margin: 10px 0; }}
  th, td {{ border: 1px solid #333; padding: 6px 12px; text-align: right; }}
  th {{ background: #16213e; }}
  .positive {{ color: #00ff88; }}
  .negative {{ color: #ff4444; }}
  .summary {{ font-size: 1.2em; margin: 10px 0; }}
  .meta {{ color: #888; font-size: 0.9em; }}
</style></head><body>
<h1>Daily Trading Report</h1>
<p>{_esc(today)}</p>
<p class="meta">Model: {_esc(EXPORT_MODEL)} | Threshold: {THRESHOLD} | Top-N: {TOP_N} | Max Positions: {MAX_POSITIONS} | Mode: DRY RUN</p>

<h2>Portfolio Summary</h2>
<div class="summary">
  <p>Cash: <b>${cash:,.2f}</b></p>
  <p>Equity: <b>${equity:,.2f}</b></p>
  <p>Total Return: <b class="{ret_class}">{total_return:+.2f}%</b></p>
  <p>Open Positions: <b>{len(positions)}</b></p>
  <p>Realized PnL: <b class="{pnl_class}">${realized_pnl:,.2f}</b></p>
  <p>Total Trades: <b>{len(history)}</b></p>
</div>

<h2>Open Positions</h2>
<table><tr><th>Ticker</th><th>Shares</th><th>Entry$</th><th>Entry Date</th><th>Stop</th><th>Target</th></tr>
{pos_rows if pos_rows else '<tr><td colspan="6">No open positions</td></tr>'}
</table>

<h2>Today\'s Orders</h2>
<table><tr><th>Action</th><th>Ticker</th><th>Shares</th><th>Probability</th><th>Reason</th></tr>
{order_rows if order_rows else '<tr><td colspan="5">No orders</td></tr>'}
</table>

<h2>Top Signals</h2>
<table><tr><th>Ticker</th><th>Probability</th></tr>
{top_preds if top_preds else '<tr><td colspan="2">No predictions</td></tr>'}
</table>

<h2>Recent Trades (last 10)</h2>
<table><tr><th>Ticker</th><th>Entry</th><th>Exit</th><th>Entry$</th><th>Exit$</th><th>PnL</th><th>Reason</th></tr>
{trade_rows if trade_rows else '<tr><td colspan="7">No closed trades</td></tr>'}
</table>

</body></html>"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        return str(output_path)
