#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Core.h"
#include "Day.h"
#include "Strategies.h"

class Backtest {
private:
	std::string ticker;
	std::vector<Day> time_series;
	std::vector<double> equity_curve;

	nlohmann::json load_json(std::string ticker) {
		namespace fs = std::filesystem;
		std::string filename = "ticker_data/" + ticker + ".json";
		if (fs::exists(filename)) {
			std::ifstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("File exists but cannot be opened: " + filename);
			}

			nlohmann::json j;
			file >> j;
			return j;
		}
		else
		{
			throw std::runtime_error("JSON not found: " + filename);
		}
	}
	bool verify_true(const std::vector<Day>& time_series) {
		for (size_t i = 0; i + 1 < time_series.size(); i++) {
			if (time_series[i].date > time_series[i + 1].date)
				return false;
		}
		return true;
	}
	void init_time_series(std::string ticker) {
		nlohmann::json data = load_json(ticker);
		std::string last_refresh{};

		if (!data.is_array()) {
			throw std::runtime_error("Expected JSON array at root");
		}

		time_series.reserve(data.size());


		for (const auto& row : data) {
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
				throw std::runtime_error("Missing required Volume field for date: " + date);
			}
			double volume = row.at("Volume").get<double>();

			time_series.emplace_back(date, open, high, low, close, adjusted_close, volume);
		}
		std::sort(time_series.begin(), time_series.end(),
			[](const Day& a, const Day& b) {
				return a.date < b.date;
			});
		if (!verify_true(time_series))
			throw std::runtime_error("Time series is not sorted in ascending order");
	}
	bool in_regime(const std::string& date, Regimes regime) {
		if (regime == Regimes::STRESS) {
			return date >= "2000-01-01" && date <= "2009-12-31";
		}
		if (regime == Regimes::EASY) {
			return date >= "2010-01-01" && date <= "2019-12-31";
		}
		return date >= "2020-01-01" && date <= "2026-05-31";
	}
public:
	Backtest(std::string ticker, int opt) : ticker(ticker) { init_time_series(ticker); }
	Backtest(std::string ticker, std::vector<Day> days) : ticker(ticker), time_series(std::move(days)) {}
	void run_backtest(Portfolio& p, Strategy& s, Regimes regime, int max_hold_days, double stop_loss_pct = 0.0, double target_profit_pct = 0.0) {
		s.reset();
		Signal signal{};
		double price{};
		for (size_t i = 2; i < time_series.size(); i++) {
			if (!in_regime(time_series[i].date, regime))     continue;
			if (!in_regime(time_series[i - 1].date, regime)) continue;
			signal = s.generate(time_series, i - 1, p.in_position(ticker));
			price = time_series[i].close;
			bool exited_today = false;

			if (p.in_position(ticker)) {
				const Day& day = time_series[i];
				int index = static_cast<int>(i);
				double stop_price = p.current_stop_price(ticker);
				double target_price = p.current_target_price(ticker);

				if (stop_price > 0.0 && day.low <= stop_price) {
					p.sell(ticker, stop_price, day.date, index, "stop_loss");
					exited_today = true;
				}
				else if (std::isfinite(target_price) && day.high >= target_price) {
					p.sell(ticker, target_price, day.date, index, "take_profit");
					exited_today = true;
				}
				else if (max_hold_days > 0 && p.current_entry_index(ticker) >= 0 && index - p.current_entry_index(ticker) >= max_hold_days) {
					p.sell(ticker, price, day.date, index, "max_hold");
					exited_today = true;
				}
				else if (signal == Signal::SELL) {
					p.sell(ticker, price, day.date, index, "strategy_sell");
					exited_today = true;
					//std::cout << "SELL @" << price << std::endl;
				}
			}

			if (signal == Signal::BUY && !p.in_position(ticker) && !exited_today) {
				double stop_price = stop_loss_pct > 0.0 ? price * (1.0 - stop_loss_pct) : 0.0;
				double target_price = target_profit_pct > 0.0 ? price * (1.0 + target_profit_pct) : std::numeric_limits<double>::infinity();
				p.buy(ticker, price, time_series[i].date, static_cast<int>(i), stop_price, target_price);
				//std::cout << "BUY @" << price << std::endl;
			}
			equity_curve.push_back(p.value({ { ticker, price } }));
		}
	}
	void print_equity_curve() {
		for (const auto& x : equity_curve) {
			std::cout << x << std::endl;
		}
	}
	const std::vector<double>& get_equity_curve() const {
		return this->equity_curve;
	}
	const std::vector<Day>& get_time_series() const {
		return this->time_series;
	}
	void clear() {
		equity_curve.clear();
	}
};
