#define NOMINMAX
#include "backtester.h"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <stdexcept>
#include <iostream>

using json = nlohmann::json;
using response = cpr::Response;

class Backtest {
private:
	std::vector<Day> time_series;
	std::vector<double> equity_curve;

	json load_json(std::string ticker) {
		namespace fs = std::filesystem;
		std::string filename = "ticker_data/" + ticker + ".json";
		if (fs::exists(filename)) {
			std::ifstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("File exists but cannot be opened: " + filename);
			}
			json j;
			file >> j;
			return j;
		}
		else {
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
		json data = load_json(ticker);
		std::string last_refresh{};

		if (!data.is_array()) {
			throw std::runtime_error("Expected JSON array at root");
		}

		time_series.reserve(data.size());

		for (const auto& row : data) {
			std::string full_date = row.at("Date").get<std::string>();
			std::string date = full_date.substr(0, 10);
			double close = row.at("Close").get<double>();
			time_series.emplace_back(date, close);
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
		return date >= "2020-01-01" && date <= "2025-12-31";
	}
public:
	Backtest(std::string ticker, int opt) { init_time_series(ticker); }
	void run_backtest(Portfolio& p, const Strategy& s, Regimes regime) {
		Signal signal{};
		double price{};
		for (size_t i = 2; i < time_series.size(); i++) {
			if (!in_regime(time_series[i].date, regime))     continue;
			if (!in_regime(time_series[i - 1].date, regime)) continue;
			if (!in_regime(time_series[i - 2].date, regime)) continue;
			signal = s.generate(time_series, i - 1);
			price = time_series[i].close;

			if (signal == Signal::BUY && !p.in_position()) {
				p.buy(price, time_series[i].date);
			}
			if (signal == Signal::SELL && p.in_position()) {
				p.sell(price, time_series[i].date);
			}
			equity_curve.push_back(p.value(price));
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
	void clear() {
		equity_curve.clear();
	}
};

int main() {
	int opt = 1;
	Regimes regime{Regimes::MODERN};

	MomentumStrategy strat(20);
	Backtest b("AAPL", opt);

	Portfolio p3(10000);
	b.clear();
	b.run_backtest(p3, strat, Regimes::MODERN);
	Metrics m3(b.get_equity_curve());
	m3.print_metrics(b.get_equity_curve());
	TradeMetrics::print(p3.get_trades());
	TradeMetrics::save_csv(p3.get_trades(), "output/trades.csv");

	return 0;
}
