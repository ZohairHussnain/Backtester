#define NOMINMAX
#include <iostream>
#include <string>
#include <vector>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <cmath>
#include <numeric>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>

#include "Day.h"
#include "FeatureEngine.h"
#include "LabelEngine.h"
#include "MLDataExporter.h"
#include "PredictionLoader.h"

using json = nlohmann::json;
using response = cpr::Response;

enum class PositionState {
	IN_CASH,
	IN_POSITION
};
enum class Signal {
	BUY,
	SELL,
	HOLD
};
enum class Regimes {
	STRESS,
	EASY,
	MODERN
};
class Trade {
public:
	std::string entry_date;
	std::string exit_date;
	double entry_price;
	double exit_price;
	double shares;
	double fees;
	double stop_price;
	double target_price;
	int entry_index;
	int exit_index;
	std::string exit_reason;

	double pnl() const {
		return (exit_price - entry_price) * shares - fees;
	}
	double return_pct() const {
		return (exit_price - entry_price) / entry_price;
	}
};

struct Position {
	std::string ticker;
	double shares;
	Trade trade;
};

class Portfolio {
private:
	double cash;
	std::unordered_map<std::string, Position> positions;
	std::vector<Trade> closed_trades;
	double risk_per_trade;
	double max_position_fraction;
	int max_open_positions;
	double slippage = 0.0005;
	double calculate_transaction_costs(double price, double shares) const {
		double trade_value = price * shares;

		double commission = shares * 0.0035;
		commission = std::max(commission, 0.35); //min cost $0.35 
		commission = std::min(commission, 0.01 * trade_value); //max cost 1%
		return commission;
	}
	double estimated_portfolio_value() const {
		double total = cash;
		for (const auto& [ticker, position] : positions) {
			total += position.shares * position.trade.entry_price;
		}
		return total;
	}
public:
	Portfolio() : cash(10000), risk_per_trade(0.01), max_position_fraction(0.40), max_open_positions(5) {}
	Portfolio(double cash) : cash(cash), risk_per_trade(0.01), max_position_fraction(0.40), max_open_positions(5) {}
	Portfolio(double cash, double risk_per_trade, double max_position_fraction)
		: cash(cash), risk_per_trade(risk_per_trade), max_position_fraction(max_position_fraction), max_open_positions(5) {
	}
	Portfolio(double cash, double risk_per_trade, double max_position_fraction, int max_open_positions)
		: cash(cash), risk_per_trade(risk_per_trade), max_position_fraction(max_position_fraction), max_open_positions(max_open_positions) {
	}
	bool in_position(const std::string& ticker) const {
		return positions.find(ticker) != positions.end();
	}
	int open_position_count() const {
		return static_cast<int>(positions.size());
	}
	int max_open_position_count() const {
		return max_open_positions;
	}
	double current_stop_price(const std::string& ticker) const {
		auto it = positions.find(ticker);
		return it != positions.end() ? it->second.trade.stop_price : 0.0;
	}
	double current_target_price(const std::string& ticker) const {
		auto it = positions.find(ticker);
		return it != positions.end() ? it->second.trade.target_price : std::numeric_limits<double>::infinity();
	}
	int current_entry_index(const std::string& ticker) const {
		auto it = positions.find(ticker);
		return it != positions.end() ? it->second.trade.entry_index : -1;
	}
	void buy(const std::string& ticker, double price, const std::string& date, int index, double stop_price, double target_price) {
		if (in_position(ticker) || open_position_count() >= max_open_positions || price <= 0 || risk_per_trade <= 0 || max_position_fraction <= 0)
			return;
		double execution_price = price * (1.0 + slippage);
		// Rescale stop/target so percentages are relative to the actual execution
		// price rather than the raw price the caller used to compute them.
		if (price > 0.0 && stop_price > 0.0) {
			double stop_pct = 1.0 - stop_price / price;
			stop_price = execution_price * (1.0 - stop_pct);
		}
		if (price > 0.0 && std::isfinite(target_price)) {
			double target_pct = target_price / price - 1.0;
			target_price = execution_price * (1.0 + target_pct);
		}
		if (stop_price >= execution_price)
			return;

		double portfolio_value = estimated_portfolio_value();
		double account_risk_dollars = portfolio_value * risk_per_trade;
		double stop_distance = execution_price - stop_price;
		double shares_by_risk = account_risk_dollars / stop_distance;
		double shares_by_cash = (cash * max_position_fraction) / execution_price;
		double sh = std::min(shares_by_risk, shares_by_cash);
		for (int i = 0; i < 10; ++i) {
			double fees = calculate_transaction_costs(execution_price, sh);
			double stop_execution_price = stop_price * (1.0 - slippage);
			double exit_fees = calculate_transaction_costs(stop_execution_price, sh);
			double risk_with_costs = (execution_price - stop_execution_price) * sh + fees + exit_fees;
			double cash_budget = cash * max_position_fraction;
			double refined_by_risk = risk_with_costs > 0.0 ? sh * (account_risk_dollars / risk_with_costs) : 0.0;
			double refined_by_cash = (cash_budget - fees) / execution_price;
			double refined = std::min({ sh, refined_by_risk, refined_by_cash });
			if (refined <= 0) return;
			if (std::abs(refined - sh) < 1e-9) break;
			sh = refined;
		}
		double fees = calculate_transaction_costs(execution_price, sh);
		if (sh <= 0 || sh * execution_price + fees > cash * max_position_fraction) return;

		this->cash -= sh * execution_price + fees;

		Trade trade{ date, "", execution_price, 0.0, sh, fees, stop_price, target_price, index, -1, "" };
		positions.emplace(ticker, Position{ ticker, sh, trade });
	}
	void sell(const std::string& ticker, double price, const std::string& date, int index, const std::string& exit_reason) {
		auto it = positions.find(ticker);
		if (it == positions.end() || price <= 0)
			return;
		double execution_price = price * (1.0 - slippage);

		Position& position = it->second;
		double value = position.shares * execution_price;
		double fees = calculate_transaction_costs(execution_price, position.shares);

		this->cash += value - fees;
		

		position.trade.exit_date = date;
		position.trade.exit_price = execution_price;
		position.trade.fees += fees;
		position.trade.exit_index = index;
		position.trade.exit_reason = exit_reason;
		
		closed_trades.push_back(position.trade);
		positions.erase(it);
	}
	double value(const std::map<std::string, double>& latest_prices) const {
		double total = this->cash;
		for (const auto& [ticker, position] : positions) {
			auto price_it = latest_prices.find(ticker);
			double mark_price = price_it != latest_prices.end() ? price_it->second : position.trade.entry_price;
			total += position.shares * mark_price;
		}
		return total;
	}
	const std::vector<Trade>& get_trades() const {
		return closed_trades;
	}
};

class Strategy {
public:
	virtual Signal generate(const std::vector<Day>& time_series, size_t i) = 0;
	// Position-aware strategies can override this overload. Existing strategies keep
	// the original interface and are adapted here, which keeps this change minimal.
	virtual Signal generate(const std::vector<Day>& time_series, size_t i, bool in_position) {
		return generate(time_series, i);
	}
	virtual void reset() {}
	virtual ~Strategy() = default;
};
class MomentumStrategy : public Strategy {
private:
	size_t lookback;
public:
	explicit MomentumStrategy(size_t lookback = 20)
		: lookback(lookback) {
	}
	Signal generate(const std::vector<Day>& time_series, size_t i) override {
		if (i < lookback)
			return Signal::HOLD;

		double now = time_series[i].close;
		double past = time_series[i - lookback].close;

		if (now > past) return Signal::BUY;
		if (now < past) return Signal::SELL;
		return Signal::HOLD;
	}
};
class BuyAndHoldStrategy : public Strategy {
private:
	bool has_bought = false;
public:
	void reset() override {
		has_bought = false;
	}
	Signal generate(const std::vector<Day>& time_series, size_t i) override {
		if (time_series.empty() || i >= time_series.size())
			return Signal::HOLD;

		if (!has_bought) {
			has_bought = true;
			return Signal::BUY;
		}

		return Signal::HOLD;
	}
};
class MLProbabilityStrategy : public Strategy {
private:
	std::string ticker;
	const PredictionLoader& predictions;
	double buy_threshold;
	std::optional<double> sell_threshold;

public:
	MLProbabilityStrategy(
		const std::string& ticker,
		const PredictionLoader& predictions,
		double buy_threshold,
		std::optional<double> sell_threshold = std::nullopt
	)
		: ticker(ticker), predictions(predictions), buy_threshold(buy_threshold), sell_threshold(sell_threshold) {
	}

	Signal generate(const std::vector<Day>& time_series, size_t i) override {
		return generate(time_series, i, false);
	}

	Signal generate(const std::vector<Day>& time_series, size_t i, bool in_position) override {
		if (time_series.empty() || i >= time_series.size()) {
			return Signal::HOLD;
		}

		auto probability = predictions.get_probability(time_series[i].date, ticker);
		if (!probability.has_value()) {
			return Signal::HOLD;
		}

		if (!in_position && *probability >= buy_threshold) {
			return Signal::BUY;
		}
		if (in_position && sell_threshold.has_value() && *probability <= *sell_threshold) {
			return Signal::SELL;
		}

		return Signal::HOLD;
	}
};
class FixedPriceStrategy : public Strategy {
private:
	double buy_price;
	double sell_price;
public:
	FixedPriceStrategy(double buy_price, double sell_price)
		: buy_price(buy_price), sell_price(sell_price) {
	}

	Signal generate(const std::vector<Day>& time_series, size_t i) override {
		if (time_series.empty() || i >= time_series.size())
			return Signal::HOLD;

		const Day& day = time_series[i];

		if (day.low <= buy_price) {
			return Signal::BUY;
		}

		if (day.high >= sell_price) {
			return Signal::SELL;
		}

		return Signal::HOLD;
	}
};
class MeanReversionStrategy : public Strategy {

};



class Backtest {
private:
	std::string ticker;
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
		json data = load_json(ticker);
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

	json load_json(const std::string& ticker) {
		namespace fs = std::filesystem;
		std::string filename = "ticker_data/" + ticker + ".json";
		if (!fs::exists(filename)) {
			throw std::runtime_error("JSON not found: " + filename);
		}

		std::ifstream file(filename);
		if (!file.is_open()) {
			throw std::runtime_error("File exists but cannot be opened: " + filename);
		}

		json j;
		file >> j;
		return j;
	}

	std::vector<Day> load_days(const std::string& ticker) {
		json rows = load_json(ticker);
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

			std::vector<EntryCandidate> candidates;
			for (const auto& ticker : tickers) {
				const Day* day = get_bar(ticker, date);
				if (day == nullptr || portfolio.in_position(ticker)) {
					continue;
				}

				auto probability = predictions.get_probability(date, ticker);
				if (!probability.has_value() || *probability < buy_threshold) {
					continue;
				}

				candidates.push_back(EntryCandidate{ ticker, *probability, day->close, date });
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
				if (!exited && day != nullptr && bar_index.has_value() &&
					strategy.generate(data.at(ticker), *bar_index, true) == Signal::SELL) {
					portfolio.sell(ticker, day->close, date, static_cast<int>(calendar_index), "strategy_sell");
				}
			}

			// Collect entry candidates, then sort alphabetically so results
			// are deterministic regardless of the order tickers were passed in.
			std::vector<EntryCandidate> strategy_candidates;
			for (const auto& ticker : tickers) {
				const Day* day = get_bar(ticker, date);
				auto bar_index = get_bar_index(ticker, date);
				if (day == nullptr || !bar_index.has_value() || portfolio.in_position(ticker)) {
					continue;
				}

				if (strategy.generate(data.at(ticker), *bar_index, false) != Signal::BUY) {
					continue;
				}

				strategy_candidates.push_back(EntryCandidate{ ticker, 0.0, day->close, date });
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

class Metrics {
private:
	std::vector<double> returns{};
	double drawdown{};
	double CAGR{};
	double sharpe{};
	int trading_days_override = 0;
	void calc_daily_returns(const std::vector<double>& equity_curve) {
		returns.clear();
		for (size_t i = 1; i < equity_curve.size(); i++) {
			if (equity_curve[i - 1] == 0) continue;
			returns.push_back((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]);
		}
	}
	void calc_max_drawdown(const std::vector<double>& equity_curve) {
		if (equity_curve.empty()) return;
		drawdown = 0.0;
		double peak_equity = equity_curve[0];
		for (size_t i = 0; i < equity_curve.size(); i++) {
			if (equity_curve[i] > peak_equity)
				peak_equity = equity_curve[i];
			if (peak_equity <= 0) continue;
			double d = (equity_curve[i] - peak_equity) / peak_equity;
			if (d < drawdown)
				drawdown = d;
		}
	}
	void calc_CAGR(const std::vector<double>& equity_curve) {
		if (equity_curve.size() < 2) {
			CAGR = 0.0;
			return;
		}
		double initial = equity_curve.front();
		double final_val = equity_curve.back();

		if (initial <= 0 || final_val <= 0) {
			CAGR = 0.0;
			return;
		}

		double days = trading_days_override > 0
			? static_cast<double>(trading_days_override)
			: static_cast<double>(equity_curve.size());
		double years = days / 252.0;

		CAGR = std::pow(final_val / initial, 1.0 / years) - 1.0;
	}
	void calc_sharpe() {
		if (returns.size() < 2) {
			sharpe = 0.0;
			return;
		}

		double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

		double variance = 0.0;
		for (double r : returns) {
			variance += (r - mean) * (r - mean);
		}
		variance /= (returns.size() - 1);

		double std = std::sqrt(variance);

		if (std == 0.0) {
			sharpe = 0.0;
			return;
		}

		constexpr double daily_risk_free = 0.04 / 252.0;
		sharpe = ((mean - daily_risk_free) / std) * std::sqrt(252.0);
	}
	void saveVector(const std::vector<double>& v) {
		std::ofstream file("output/equity.dat");
		for (size_t i = 0; i < v.size(); ++i)
			file << i << " " << v[i] << "\n";
		file.close();
		system(
			"gnuplot -persist -e \""
			"set title 'Equity Curve';"
			"set xlabel 'Trading Days';"
			"set ylabel 'Account Value';"
			"set grid;"
			"plot 'output/equity.dat' using 1:2 with lines lw 2 title 'Equity'\""
		);
	}
public:
	Metrics(const std::vector<double>& equity_curve) {
		calc_daily_returns(equity_curve);
		calc_max_drawdown(equity_curve);
		calc_CAGR(equity_curve);
		calc_sharpe();
	}
	Metrics(const std::vector<double>& equity_curve, int trading_days)
		: trading_days_override(trading_days) {
		calc_daily_returns(equity_curve);
		calc_max_drawdown(equity_curve);
		calc_CAGR(equity_curve);
		calc_sharpe();
	}
	double getCAGR() const { return this->CAGR; }
	double max_drawdown() const { return this->drawdown; }
	const std::vector<double>& daily_returns() const { return this->returns; }
	double get_sharpe() const { return this->sharpe; }
	void print_metrics(const std::vector<double>& equity_curve) {
		auto drawdown = max_drawdown();
		std::cout << "MAX DRAWDOWN:" << drawdown << std::endl;
		auto cagr = getCAGR();
		std::cout << "CAGR:" << cagr << std::endl;
		auto sharpe = get_sharpe();
		std::cout << "SHARPE: " << sharpe << std::endl;
		saveVector(equity_curve);
	}
};

class TradeMetrics {
public:
	static void print(const std::vector<Trade>& trades) {
		if (trades.empty()) {
			std::cout << "No trades executed";
			return;
		}

		int wins{ 0 };
		int losses{ 0 };

		double total_pnl{ 0.0 };
		double total_win{ 0.0 };
		double total_loss{ 0.0 };
		double total_fees{ 0.0 };

		for (const auto& t : trades) {
			double pnl = t.pnl();
			total_pnl += pnl;
			total_fees += t.fees;

			if (pnl > 0) {
				wins++;
				total_win += pnl;
			}
			else {
				losses++;
				total_loss += -pnl;
			}
		}

		int total_trades{ wins + losses };

		double win_rate = static_cast<double>(wins) / total_trades;
		double avg_pnl = total_pnl / total_trades;
		double avg_win = wins > 0 ? total_win / wins : 0.0;
		double avg_loss = losses > 0 ? total_loss / losses : 0.0;
		double profit_factor = (total_loss > 0) ? total_win / total_loss : (total_win > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);

		double expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss;//not sure of this

		std::cout << "TRADES: " << total_trades << "\n";
		std::cout << "EV: " << expectancy << "\n";
		std::cout << "WIN RATE: " << win_rate * 100 << "%\n";
		std::cout << "AVG TRADE PNL: " << avg_pnl << "\n";
		std::cout << "AVG WIN: " << avg_win << "\n";
		std::cout << "AVG LOSS: " << avg_loss << "\n";
		std::cout << "PROFIT FACTOR: " << profit_factor << "\n";
		std::cout << "TOTAL FEES: " << total_fees << "\n";
	}
	static void save_csv(const std::vector<Trade>& trades, const std::string& path) {
		std::ofstream f(path);
		f << "entry_date,exit_date,entry_price,exit_price,shares,fees,stop_price,target_price,entry_index,exit_index,exit_reason,pnl,return_pct\n";
		for (const auto& t : trades) {
			f << t.entry_date << ","
			  << t.exit_date << ","
			  << t.entry_price << ","
			  << t.exit_price << ","
			  << t.shares << ","
			  << t.fees << ","
			  << t.stop_price << ","
			  << t.target_price << ","
			  << t.entry_index << ","
			  << t.exit_index << ","
			  << t.exit_reason << ","
			  << t.pnl() << ","
			  << t.return_pct() << "\n";
		}
	}
};

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
	//auto features = FeatureEngine::generate(ticker, b.get_time_series());
	//auto labels = LabelEngine::generate(ticker, b.get_time_series(), target_profit_pct, stop_loss_pct, max_hold_days);
	//size_t ml_rows = MLDataExporter::save_csv(features, labels);
	//std::cout << "ML rows exported: " << ml_rows << " to output/ml_dataset.csv\n";

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

