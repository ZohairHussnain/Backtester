#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include "Day.h"
#include "Core.h"
#include "Strategies.h"
#include "Backtest.h"
#include "MultiAssetBacktest.h"
#include "Metrics.h"
#include "FeatureEngine.h"
#include "LabelEngine.h"
#include "MLDataExporter.h"
#include "PredictionLoader.h"

using json = nlohmann::json;
using response = cpr::Response;

int main() {
	int opt = 1;
	std::string ticker = "AAPL";
	std::string rr_ticker = "RR";
	Regimes regime{Regimes::MODERN};

	// Portfolio risk controls:
	// - risk_per_trade = 0.01 risks about 1% of portfolio value if the stop is hit.
	// - max_position_fraction = 0.40 caps a single position at 40% of available cash.
	// - max_open_positions caps how many tickers can be held at once.
	double starting_cash = 10000.0;
	double risk_per_trade = 0.01;
	double max_position_fraction = 0.40;
	int max_open_positions = 5;
	// Strategy setup:
	// MomentumStrategy(20) compares recent price action against a 20-day lookback.
	MomentumStrategy strat(20);
	//FixedPriceStrategy strat(2.30, 2.75);

	// Swing trade management:
	// - max_hold_days exits after this many bars in a trade.
	// - stop_loss_pct sets stop_price below entry.
	// - target_profit_pct sets target_price above entry.
	// If stop and target are both hit on the same day, the backtest assumes stop first.
	int max_hold_days = 20;
	double stop_loss_pct = 0.05;
	double target_profit_pct = 0.10;

	std::cout << "Running true multi-stock swing backtest for " << ticker << " and " << rr_ticker << "\n";
	std::cout << "Starting cash: " << starting_cash << "\n";
	std::cout << "Risk per trade: " << risk_per_trade * 100.0 << "%\n";
	std::cout << "Max position fraction: " << max_position_fraction * 100.0 << "%\n";
	std::cout << "Max open positions: " << max_open_positions << "\n";
	std::cout << "Regime: MODERN (2020-01-01 to 2026-05-31)\n";
	std::cout << "Stop loss: " << stop_loss_pct * 100.0 << "%, target: " << target_profit_pct * 100.0 << "%, max hold: " << max_hold_days << " days\n";

	// True multi-stock backtest:
	// MultiAssetBacktest builds one shared calendar across all tickers and records
	// one portfolio equity value per date.
	Portfolio multi_portfolio(starting_cash, risk_per_trade, max_position_fraction, max_open_positions);
	MultiAssetBacktest multi({ ticker, rr_ticker }, multi_portfolio);
	multi.run_with_strategy(strat, regime, max_hold_days, stop_loss_pct, target_profit_pct);
	Metrics multi_metrics(multi.get_equity_curve());
	multi_metrics.print_metrics(multi.get_equity_curve());
	TradeMetrics::print(multi.get_portfolio().get_trades());
	std::cout << "Closed trades: " << multi.get_portfolio().get_trades().size() << "\n";
	TradeMetrics::save_csv(multi.get_portfolio().get_trades(), "output/trades.csv");

	// ML dataset export:
	// Features use only past/current data; labels use future OHLC for the same dates.
	// The exporter joins rows by date + ticker and writes only rows with both sides.
	// Exports both label_target_stop and label_median_return for each row.
	{
		Backtest b_export(ticker, 1);
		auto features = FeatureEngine::generate(ticker, b_export.get_time_series());
		auto labels = LabelEngine::generate(ticker, b_export.get_time_series(), target_profit_pct, stop_loss_pct, max_hold_days);
		size_t ml_rows = MLDataExporter::save_csv(features, labels);
		std::cout << "ML rows exported: " << ml_rows << " to output/ml_dataset.csv\n";
	}

	// Prediction import example:
	// PredictionLoader predictions("output/predictions.csv");
	// auto probability = predictions.get_probability("2024-01-05", ticker);
	// if (probability.has_value()) {
	// 	std::cout << "Prediction probability: " << *probability << "\n";
	// }

	// ML-ranked multi-stock example:
	// PredictionLoader predictions("output/predictions.csv");
	// MultiAssetBacktest ml_multi({ ticker, rr_ticker }, Portfolio(starting_cash, risk_per_trade, max_position_fraction, max_open_positions));
	// ml_multi.run_with_predictions(predictions, 0.60, regime, max_hold_days, stop_loss_pct, target_profit_pct);
	// Metrics ml_metrics(ml_multi.get_equity_curve());
	// ml_metrics.print_metrics(ml_multi.get_equity_curve());


	return 0;
}
