#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Core.h"
#include "Day.h"
#include "PredictionLoader.h"
#include "Strategies.h"

class MultiAssetBacktest {
private:
	struct EntryCandidate {
		std::string ticker;
		double probability;
		double price;
		std::string date;
	};

	std::vector<std::string> tickers;
	std::map<std::string, std::vector<Day>> data;
	Portfolio portfolio;
	std::vector<double> equity_curve;

	nlohmann::json load_json(const std::string& ticker) {
		namespace fs = std::filesystem;
		std::string filename = "ticker_data/" + ticker + ".json";
		if (!fs::exists(filename)) {
			throw std::runtime_error("JSON not found: " + filename);
		}

		std::ifstream file(filename);
		if (!file.is_open()) {
			throw std::runtime_error("File exists but cannot be opened: " + filename);
		}

		nlohmann::json j;
		file >> j;
		return j;
	}

	std::vector<Day> load_days(const std::string& ticker) {
		nlohmann::json rows = load_json(ticker);
		if (!rows.is_array()) {
			throw std::runtime_error("Expected JSON array at root for ticker: " + ticker);
		}

		std::vector<Day> days;
		days.reserve(rows.size());
		for (const auto& row : rows) {
			std::string full_date = row.at("Date").get<std::string>();
			std::string date = full_date.substr(0, 10);
			double open = row.at("Open").get<double>();
			double high = row.at("High").get<double>();
			double low = row.at("Low").get<double>();
			double close = row.at("Close").get<double>();
			double adjusted_close = close;

			if (row.contains("Adjusted Close")) {
				adjusted_close = row.at("Adjusted Close").get<double>();
			}
			else if (row.contains("Adj Close")) {
				adjusted_close = row.at("Adj Close").get<double>();
			}

			if (!row.contains("Volume")) {
				throw std::runtime_error("Missing required Volume field for " + ticker + " on date: " + date);
			}
			double volume = row.at("Volume").get<double>();

			days.emplace_back(date, open, high, low, close, adjusted_close, volume);
		}

		std::sort(days.begin(), days.end(),
			[](const Day& a, const Day& b) {
				return a.date < b.date;
			});
		return days;
	}

	std::vector<std::string> build_calendar() const {
		std::set<std::string> unique_dates;
		for (const auto& [ticker, days] : data) {
			for (const auto& day : days) {
				unique_dates.insert(day.date);
			}
		}
		return std::vector<std::string>(unique_dates.begin(), unique_dates.end());
	}

	std::map<std::string, double> build_latest_prices(const std::string& date) const {
		std::map<std::string, double> latest_prices;
		for (const auto& ticker : tickers) {
			double close = get_latest_close(ticker, date);
			if (std::isfinite(close) && close > 0.0) {
				latest_prices[ticker] = close;
			}
		}
		return latest_prices;
	}

	std::optional<size_t> get_bar_index(const std::string& ticker, const std::string& date) const {
		auto data_it = data.find(ticker);
		if (data_it == data.end()) {
			return std::nullopt;
		}

		const auto& days = data_it->second;
		auto it = std::lower_bound(days.begin(), days.end(), date,
			[](const Day& day, const std::string& value) {
				return day.date < value;
			});

		if (it == days.end() || it->date != date) {
			return std::nullopt;
		}

		return static_cast<size_t>(std::distance(days.begin(), it));
	}

	static bool date_in_range(const std::string& date, const std::string& start_date, const std::string& end_date) {
		return (start_date.empty() || date >= start_date) && (end_date.empty() || date <= end_date);
	}

	static std::pair<std::string, std::string> regime_range(Regimes regime) {
		if (regime == Regimes::STRESS) {
			return { "2000-01-01", "2009-12-31" };
		}
		if (regime == Regimes::EASY) {
			return { "2010-01-01", "2019-12-31" };
		}
		return { "2020-01-01", "2026-05-31" };
	}

public:
	MultiAssetBacktest(const std::vector<std::string>& tickers, const Portfolio& portfolio)
		: tickers(tickers), portfolio(portfolio) {
		for (const auto& ticker : tickers) {
			data[ticker] = load_days(ticker);
		}
	}

	MultiAssetBacktest(const std::vector<std::string>& tickers, std::map<std::string, std::vector<Day>> data, const Portfolio& portfolio)
		: tickers(tickers), data(std::move(data)), portfolio(portfolio) {
	}

	bool has_bar(const std::string& ticker, const std::string& date) const {
		return get_bar(ticker, date) != nullptr;
	}

	const Day* get_bar(const std::string& ticker, const std::string& date) const {
		auto data_it = data.find(ticker);
		if (data_it == data.end()) {
			return nullptr;
		}

		const auto& days = data_it->second;
		auto it = std::lower_bound(days.begin(), days.end(), date,
			[](const Day& day, const std::string& value) {
				return day.date < value;
			});

		if (it != days.end() && it->date == date) {
			return &(*it);
		}
		return nullptr;
	}

	double get_latest_close(const std::string& ticker, const std::string& date) const {
		auto data_it = data.find(ticker);
		if (data_it == data.end()) {
			return std::numeric_limits<double>::quiet_NaN();
		}

		const auto& days = data_it->second;
		auto it = std::upper_bound(days.begin(), days.end(), date,
			[](const std::string& value, const Day& day) {
				return value < day.date;
			});

		if (it == days.begin()) {
			return std::numeric_limits<double>::quiet_NaN();
		}

		--it;
		return it->close;
	}

	void run_with_predictions(
		const PredictionLoader& predictions,
		double buy_threshold,
		int max_hold_days,
		double stop_loss_pct,
		double target_profit_pct,
		const std::string& start_date = "",
		const std::string& end_date = ""
	) {
		equity_curve.clear();
		auto calendar = build_calendar();

		// A shared calendar is required for a true multi-stock backtest: all exits,
		// entries, and mark-to-market values must be processed once per date so one
		// Portfolio produces exactly one equity curve.
		for (size_t calendar_index = 0; calendar_index < calendar.size(); ++calendar_index) {
			const std::string& date = calendar[calendar_index];
			if (!date_in_range(date, start_date, end_date)) {
				continue;
			}
			auto latest_prices = build_latest_prices(date);
			portfolio.set_latest_prices(latest_prices);

			for (const auto& ticker : tickers) {
				if (!portfolio.in_position(ticker)) {
					continue;
				}

				const Day* day = get_bar(ticker, date);
				bool exited = false;
				if (day != nullptr) {
					double stop_price = portfolio.current_stop_price(ticker);
					double target_price = portfolio.current_target_price(ticker);

					if (stop_price > 0.0 && day->low <= stop_price) {
						portfolio.sell(ticker, stop_price, date, static_cast<int>(calendar_index), "stop_loss");
						exited = true;
					}
					else if (std::isfinite(target_price) && day->high >= target_price) {
						portfolio.sell(ticker, target_price, date, static_cast<int>(calendar_index), "take_profit");
						exited = true;
					}
				}

				if (!exited && max_hold_days > 0 && portfolio.current_entry_index(ticker) >= 0 &&
					static_cast<int>(calendar_index) - portfolio.current_entry_index(ticker) >= max_hold_days) {
					auto price_it = latest_prices.find(ticker);
					if (price_it != latest_prices.end()) {
						portfolio.sell(ticker, price_it->second, date, static_cast<int>(calendar_index), "max_hold");
					}
				}
			}

			// Prediction for date D is computed from features known after D's close.
			// Entry must happen at D+1's open. We are processing calendar date
			// `date` (D+1), so we look up the prediction for the previous calendar
			// date (D) and enter at today's open.
			std::vector<EntryCandidate> candidates;
			if (calendar_index > 0) {
				const std::string& signal_date = calendar[calendar_index - 1];
				for (const auto& ticker : tickers) {
					const Day* day = get_bar(ticker, date);
					if (day == nullptr || portfolio.in_position(ticker)) {
						continue;
					}

					auto probability = predictions.get_probability(signal_date, ticker);
					if (!probability.has_value() || *probability < buy_threshold) {
						continue;
					}

					candidates.push_back(EntryCandidate{ ticker, *probability, day->open, date });
				}
			}

			std::sort(candidates.begin(), candidates.end(),
				[](const EntryCandidate& a, const EntryCandidate& b) {
					return a.probability > b.probability;
				});

			for (const auto& candidate : candidates) {
				if (portfolio.open_position_count() >= portfolio.max_open_position_count()) {
					break;
				}

				double stop_price = stop_loss_pct > 0.0 ? candidate.price * (1.0 - stop_loss_pct) : 0.0;
				double target_price = target_profit_pct > 0.0 ? candidate.price * (1.0 + target_profit_pct) : std::numeric_limits<double>::infinity();
				portfolio.buy(candidate.ticker, candidate.price, candidate.date, static_cast<int>(calendar_index), stop_price, target_price);
			}

			equity_curve.push_back(portfolio.value(latest_prices));
		}
	}

	void run_with_predictions(
		const PredictionLoader& predictions,
		double buy_threshold,
		Regimes regime,
		int max_hold_days,
		double stop_loss_pct,
		double target_profit_pct
	) {
		auto [start_date, end_date] = regime_range(regime);
		run_with_predictions(predictions, buy_threshold, max_hold_days, stop_loss_pct, target_profit_pct, start_date, end_date);
	}

	void run_with_strategy(
		Strategy& strategy,
		int max_hold_days,
		double stop_loss_pct,
		double target_profit_pct,
		const std::string& start_date = "",
		const std::string& end_date = ""
	) {
		equity_curve.clear();
		strategy.reset();
		auto calendar = build_calendar();

		// This is the same shared-calendar model as the prediction path. The key
		// point is that all tickers are processed for one date before one portfolio
		// value is recorded.
		for (size_t calendar_index = 0; calendar_index < calendar.size(); ++calendar_index) {
			const std::string& date = calendar[calendar_index];
			if (!date_in_range(date, start_date, end_date)) {
				continue;
			}
			auto latest_prices = build_latest_prices(date);
			portfolio.set_latest_prices(latest_prices);

			for (const auto& ticker : tickers) {
				if (!portfolio.in_position(ticker)) {
					continue;
				}

				const Day* day = get_bar(ticker, date);
				bool exited = false;
				if (day != nullptr) {
					double stop_price = portfolio.current_stop_price(ticker);
					double target_price = portfolio.current_target_price(ticker);

					if (stop_price > 0.0 && day->low <= stop_price) {
						portfolio.sell(ticker, stop_price, date, static_cast<int>(calendar_index), "stop_loss");
						exited = true;
					}
					else if (std::isfinite(target_price) && day->high >= target_price) {
						portfolio.sell(ticker, target_price, date, static_cast<int>(calendar_index), "take_profit");
						exited = true;
					}
				}

				if (!exited && max_hold_days > 0 && portfolio.current_entry_index(ticker) >= 0 &&
					static_cast<int>(calendar_index) - portfolio.current_entry_index(ticker) >= max_hold_days) {
					auto price_it = latest_prices.find(ticker);
					if (price_it != latest_prices.end()) {
						portfolio.sell(ticker, price_it->second, date, static_cast<int>(calendar_index), "max_hold");
						exited = true;
					}
				}

				auto bar_index = get_bar_index(ticker, date);
				if (!exited && day != nullptr && bar_index.has_value() && *bar_index > 0 &&
					strategy.generate(data.at(ticker), *bar_index - 1, true) == Signal::SELL) {
					portfolio.sell(ticker, day->close, date, static_cast<int>(calendar_index), "strategy_sell");
				}
			}

			// Collect entry candidates, then sort alphabetically so results
			// are deterministic regardless of the order tickers were passed in.
			std::vector<EntryCandidate> strategy_candidates;
			for (const auto& ticker : tickers) {
				const Day* day = get_bar(ticker, date);
				auto bar_index = get_bar_index(ticker, date);
				if (day == nullptr || !bar_index.has_value() || *bar_index == 0 || portfolio.in_position(ticker)) {
					continue;
				}

				// Signal from previous bar, execute at current bar (matches Backtest.h timing)
				if (strategy.generate(data.at(ticker), *bar_index - 1, false) != Signal::BUY) {
					continue;
				}

				// Enter at open (matches LabelEngine's entry_price = next-day open)
				strategy_candidates.push_back(EntryCandidate{ ticker, 0.0, day->open, date });
			}

			std::sort(strategy_candidates.begin(), strategy_candidates.end(),
				[](const EntryCandidate& a, const EntryCandidate& b) {
					return a.ticker < b.ticker;
				});

			for (const auto& candidate : strategy_candidates) {
				if (portfolio.open_position_count() >= portfolio.max_open_position_count()) {
					break;
				}

				double stop_price = stop_loss_pct > 0.0 ? candidate.price * (1.0 - stop_loss_pct) : 0.0;
				double target_price = target_profit_pct > 0.0 ? candidate.price * (1.0 + target_profit_pct) : std::numeric_limits<double>::infinity();
				portfolio.buy(candidate.ticker, candidate.price, candidate.date, static_cast<int>(calendar_index), stop_price, target_price);
			}

			equity_curve.push_back(portfolio.value(latest_prices));
		}
	}

	void run_with_strategy(
		Strategy& strategy,
		Regimes regime,
		int max_hold_days,
		double stop_loss_pct,
		double target_profit_pct
	) {
		auto [start_date, end_date] = regime_range(regime);
		run_with_strategy(strategy, max_hold_days, stop_loss_pct, target_profit_pct, start_date, end_date);
	}

	const std::vector<double>& get_equity_curve() const {
		return equity_curve;
	}

	const Portfolio& get_portfolio() const {
		return portfolio;
	}
};
