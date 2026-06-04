#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

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
	friend class PortfolioTestAccess;
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
	std::map<std::string, double> latest_prices_;
	double estimated_portfolio_value() const {
		double total = cash;
		for (const auto& [ticker, position] : positions) {
			auto it = latest_prices_.find(ticker);
			double mark = (it != latest_prices_.end()) ? it->second : position.trade.entry_price;
			total += position.shares * mark;
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
	void set_latest_prices(const std::map<std::string, double>& prices) {
		latest_prices_ = prices;
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
