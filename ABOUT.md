# What BackTester Is

BackTester is a local, offline stock strategy backtesting engine written in C++20. You give it historical daily OHLCV price data as JSON files, pick a strategy and a date range, and it simulates trading that strategy day by day -- tracking a portfolio with cash, positions, fees, and slippage -- then reports how it would have performed.

## What it does

- **Simulates trading strategies on historical data.** It walks through daily bars in order, generates buy/sell signals from a strategy, executes them against a portfolio, and records the equity value after each day. At the end you get an equity curve, a list of closed trades, and performance metrics.

- **Models realistic execution costs.** Every buy fills at a slippage-adjusted price above the close. Every sell fills below. Commissions are per-share with a minimum and a cap. Position sizing accounts for these costs iteratively -- it won't let fees push you past your risk budget.

- **Manages risk at the portfolio level.** Each trade risks a configurable percentage of portfolio value. Position size is the smaller of what the risk budget allows and what the cash allocation cap allows. The portfolio can hold multiple concurrent positions across different tickers, with a configurable cap on how many.

- **Supports swing trade mechanics.** Trades can have stop-loss prices, take-profit targets, and a maximum holding period. On each bar the engine checks stops and targets against the day's low and high before evaluating strategy signals. If a trade exceeds the max hold period it exits at the close.

- **Runs single-ticker and multi-ticker backtests.** The single-ticker engine (`Backtest`) is simpler and iterates one time series. The multi-ticker engine (`MultiAssetBacktest`) builds a shared calendar across all tickers, processes all exits before entries on each date, and records one portfolio value per date. This means you can test strategies that allocate across multiple stocks from a single cash pool.

- **Supports ML-driven entries.** The multi-ticker engine can rank entry candidates by probability scores loaded from an external CSV (produced by a model trained elsewhere). On each date it enters the highest-probability candidates first, up to the position limit.

- **Exports training data for ML models.** The feature engine computes technical indicators (momentum returns, RSI, ATR, volume ratio, moving average distances, rolling volatility) from historical bars. The label engine looks forward from each date and asks: did price hit the profit target before the stop loss within N days? The exporter joins features and labels by date and writes a CSV that can be fed to a classifier.

- **Computes standard performance metrics.** Maximum drawdown, CAGR, annualized Sharpe ratio (with a 4% risk-free rate), plus trade-level stats: total trades, win rate, expectancy, average win/loss, profit factor, total fees.

- **Plots the equity curve.** If gnuplot is on your PATH, it opens a chart window automatically.

## What it is not

- **Not a live trading system.** There is no broker connection, no order management, no real-time data feed, no execution engine. It reads static JSON files from disk and simulates trades in a loop. It cannot place or manage real orders.

- **Not a data provider.** It does not download, scrape, or stream market data. You must supply your own JSON files in `ticker_data/`. The `cpr` HTTP library is listed as a dependency but is not used for data fetching in the current implementation.

- **Not a machine learning framework.** It exports features and labels to CSV and can import prediction probabilities from CSV, but it does not train, evaluate, or serve models. The ML workflow assumes you train externally (Python, R, etc.) and bring predictions back as a CSV file.

- **Not configurable at runtime.** There are no command-line arguments, no config files, no GUI. To change the ticker, strategy, date range, risk parameters, or anything else, you edit `main()` in `Source.cpp` and recompile.

- **Not a portfolio optimizer.** It does not search for optimal parameters, run walk-forward analysis, or do combinatorial strategy selection. It runs one configuration and reports the result. If you want to compare strategies or sweep parameters, you write the loop yourself.

- **Not an event-driven or tick-level simulator.** It operates on daily bars only. Intraday price action within a bar is not modeled -- when both stop and target are breached on the same day, it uses a heuristic (distance from open) to decide which hit first. It does not simulate order books, partial fills, or market microstructure.

- **Not multi-asset-class.** It assumes equity-like instruments with OHLCV bars and prices in a single currency. There is no support for futures, options, forex margin, crypto, or instruments that require different position/margin models.

- **Not a risk analytics platform.** It computes a small fixed set of metrics. There is no Value-at-Risk, no Monte Carlo simulation, no correlation analysis, no benchmark comparison, no rolling-window statistics.

- **Not tested.** There is no test suite -- no unit tests, no integration tests, no regression tests. Correctness is verified by inspection and by comparing output to expected values manually.
