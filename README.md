# BackTester

BackTester is a small C++20 stock strategy backtesting project. It loads local historical price data from JSON files, runs a simple momentum strategy, tracks portfolio equity and closed trades, and prints summary performance metrics.

## Features

- Local JSON price data loading from `ticker_data/`
- Momentum strategy based on a configurable lookback window
- Portfolio simulation with cash, shares, slippage, and transaction fees
- Regime-based backtests:
  - `STRESS`: 2000-01-01 to 2009-12-31
  - `EASY`: 2010-01-01 to 2019-12-31
  - `MODERN`: 2020-01-01 to 2025-12-31
- Performance metrics:
  - Maximum drawdown
  - CAGR
  - Sharpe ratio
- Trade metrics:
  - Total trades
  - Expectancy
  - Win rate
  - Average trade PnL
  - Average win/loss
  - Profit factor
  - Total fees
- Output files:
  - `output/equity.dat`
  - `output/trades.csv`

## Project Structure

```text
.
|-- Source.cpp              # Main application, strategy, portfolio, backtest, and metrics code
|-- CMakeLists.txt          # CMake build configuration
|-- vcpkg.json              # vcpkg dependency manifest
|-- ticker_data/            # Local historical price data
|   |-- AAPL.json
|   |-- MSFT.json
|   |-- NVDA.json
|   `-- SPY.json
|-- output/                 # Generated equity and trade output
`-- notes/                  # Development notes and older snippets
```

## Requirements

- C++20 compiler
- CMake 3.20 or newer
- vcpkg
- Dependencies from `vcpkg.json`:
  - `cpr`
  - `nlohmann-json`
- Optional: `gnuplot` available on your `PATH` if you want the equity curve plot window to open automatically

`cpr` is listed as a dependency and included in the source. The current implementation loads local files from `ticker_data/` rather than downloading prices at runtime.

## Build

From the project root:

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

Replace `C:/path/to/vcpkg` with your local vcpkg installation path.

If you use Visual Studio with vcpkg integration enabled, you can also open the project and build the `BackTester` target directly.

## Run

After building with CMake:

```powershell
.\build\Debug\BackTester.exe
```

Depending on your generator and configuration, the executable may also be under a different build subdirectory such as `build\Release`.

The current `main()` configuration runs:

- Ticker: `AAPL`
- Strategy: `MomentumStrategy(20)`
- Starting cash: `$10,000`
- Regime: `MODERN` (`2020-01-01` to `2025-12-31`)

## Input Data Format

Ticker files are expected in `ticker_data/` and named as:

```text
ticker_data/<TICKER>.json
```

Each file should contain a JSON array. Each row must include at least:

```json
{
  "Date": "2020-01-02T00:00:00.000",
  "Close": 75.0875
}
```

The loader sorts rows by date before running the backtest.

## Output

Running the program prints portfolio and trade metrics to the console.

It also writes:

- `output/equity.dat`: equity curve data in `index value` format for gnuplot
- `output/trades.csv`: closed trade history with entry/exit dates, prices, shares, fees, PnL, and return percentage

If `gnuplot` is installed and available on `PATH`, the program opens a plot of the equity curve.

## Changing the Backtest

The current project does not expose command-line arguments. To change the run, edit `main()` in `Source.cpp`.

Common changes:

```cpp
Backtest b("MSFT", opt);
MomentumStrategy strat(50);
b.run_backtest(p3, strat, Regimes::EASY);
```

Available local tickers are currently:

- `AAPL`
- `MSFT`
- `NVDA`
- `SPY`

## Notes

- `MeanReversionStrategy` is declared but not implemented yet.
- The commented section in `main()` shows an earlier interactive ticker/regime selection flow.
- The `notes/` directory contains development notes and older data-loading snippets.
