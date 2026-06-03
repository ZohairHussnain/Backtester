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

using json = nlohmann::json;
using response = cpr::Response;

struct Day {
	std::string date;
	double open;
	double high;
	double low;
	double close;
	double adjusted_close;
	double volume;
	Day(std::string date, double open, double high, double low, double close, double adjusted_close, double volume)
		: date(date), open(open), high(high), low(low), close(close), adjusted_close(adjusted_close), volume(volume) {
	}
};
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

class Portfolio {
private:
	double cash;
	double shares;
	std::optional<Trade> open_trade;
	std::vector<Trade> closed_trades;
	double slippage = 0.0005;
	double calculate_transaction_costs(double price, double shares) const {
		double trade_value = price * shares;

		double commission = shares * 0.0035;
		commission = std::max(commission, 0.35); //min cost $0.35 
		commission = std::min(commission, 0.01 * trade_value); //max cost 1%
		return commission;
	}
public:
	Portfolio() : cash(10000), shares(0) {}
	Portfolio(double cash) : cash(cash), shares(0) {}
	bool in_position() const {
		return open_trade.has_value();
	}
	double current_stop_price() const {
		return open_trade ? open_trade->stop_price : 0.0;
	}
	double current_target_price() const {
		return open_trade ? open_trade->target_price : std::numeric_limits<double>::infinity();
	}
	int current_entry_index() const {
		return open_trade ? open_trade->entry_index : -1;
	}
	void buy(double price, const std::string& date, int index, double stop_price, double target_price) {
		if (in_position() || price <= 0)
			return;
		double execution_price = price * (1.0 + slippage);

		double sh = cash / execution_price;
		for (int i = 0; i < 10; ++i) {
			double fees = calculate_transaction_costs(execution_price, sh);
			double refined = (cash - fees) / execution_price;
			if (refined <= 0) return;
			if (std::abs(refined - sh) < 1e-9) break;
			sh = refined;
		}
		double fees = calculate_transaction_costs(execution_price, sh);
		if (sh <= 0 || fees > cash) return;

		this->shares = sh;
		this->cash = 0.0;

		open_trade = Trade{ date, "", execution_price, 0.0, sh, fees, stop_price, target_price, index, -1, "" };
	}
	void sell(double price, const std::string& date, int index, const std::string& exit_reason) {
		if (!in_position() || price <= 0)
			return;
		double execution_price = price * (1.0 - slippage);

		double value = shares * execution_price;
		double fees = calculate_transaction_costs(execution_price, shares);

		this->cash = value - fees;
		this->shares = 0.0;
		

		open_trade->exit_date = date;
		open_trade->exit_price = execution_price;
		open_trade->fees += fees;
		open_trade->exit_index = index;
		open_trade->exit_reason = exit_reason;
		
		closed_trades.push_back(*open_trade);
		open_trade.reset();
	}
	double value(double price) const {
		if (!in_position())
			return this->cash;
		else
			return this->shares * price;
	}
	const std::vector<Trade>& get_trades() const {
		return closed_trades;
	}
};

class Strategy {
public:
	virtual Signal generate(const std::vector<Day>& time_series, size_t i) = 0;
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

		double now = time_series[i - 1].close;
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
	Backtest(std::string ticker, int opt) { init_time_series(ticker); }
	void run_backtest(Portfolio& p, Strategy& s, Regimes regime, int max_hold_days, double stop_loss_pct = 0.0, double target_profit_pct = 0.0) {
		s.reset();
		Signal signal{};
		double price{};
		for (size_t i = 2; i < time_series.size(); i++) {
			if (!in_regime(time_series[i].date, regime))     continue;
			if (!in_regime(time_series[i - 1].date, regime)) continue;
			if (!in_regime(time_series[i - 2].date, regime)) continue;
			signal = s.generate(time_series, i - 1);
			price = time_series[i].close;
			bool exited_today = false;

			if (p.in_position()) {
				const Day& day = time_series[i];
				int index = static_cast<int>(i);
				double stop_price = p.current_stop_price();
				double target_price = p.current_target_price();

				if (stop_price > 0.0 && day.low <= stop_price) {
					p.sell(stop_price, day.date, index, "stop_loss");
					exited_today = true;
				}
				else if (std::isfinite(target_price) && day.high >= target_price) {
					p.sell(target_price, day.date, index, "take_profit");
					exited_today = true;
				}
				else if (max_hold_days > 0 && p.current_entry_index() >= 0 && index - p.current_entry_index() >= max_hold_days) {
					p.sell(price, day.date, index, "max_hold");
					exited_today = true;
				}
				else if (signal == Signal::SELL) {
					p.sell(price, day.date, index, "strategy_sell");
					exited_today = true;
					//std::cout << "SELL @" << price << std::endl;
				}
			}

			if (signal == Signal::BUY && !p.in_position() && !exited_today) {
				double stop_price = stop_loss_pct > 0.0 ? price * (1.0 - stop_loss_pct) : 0.0;
				double target_price = target_profit_pct > 0.0 ? price * (1.0 + target_profit_pct) : std::numeric_limits<double>::infinity();
				p.buy(price, time_series[i].date, static_cast<int>(i), stop_price, target_price);
				//std::cout << "BUY @" << price << std::endl;
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

class Metrics {
private:
	std::vector<double> returns{};
	double drawdown{};
	double CAGR{};
	double sharpe{};
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
		double final = equity_curve.back();

		if (initial <= 0 || final <= 0) {
			CAGR = 0.0;
			return;
		}

		double days = static_cast<double>(equity_curve.size());
		double years = days / 252.0;

		CAGR = std::pow(final / initial, 1.0 / years) - 1.0;
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
	int reg = 3;
	Regimes regime{Regimes::MODERN};
	/*
	std::cout << "Select ticker:\n";
	std::cout << "1. AAPL\n";
	std::cout << "Custom Search(Limited to 100 trading days):\n";
	std::cin >> opt;
	
	std::cout << "Select Regime:\n";
	std::cout << "1: 2000�2009 stress test\n2: 2010�2019 easy regime\n3: 2020�2025 modern\n";
	std::cin >> reg;
	if(reg==1) regime = Regimes::STRESS;
	if(reg==2) regime = Regimes::EASY;
	if(reg==3) regime = Regimes::MODERN;
	*/


	//Portfolio p1(10000);
	MomentumStrategy strat(20);
	//FixedPriceStrategy strat(2.30, 2.75);
	Backtest b("AAPL", opt);
	//b.run_backtest(p1,strat, Regimes::STRESS, 20);
	//Metrics m1(b.get_equity_curve());
	//m1.print_metrics(b.get_equity_curve());
	//
	//
	//Portfolio p2(10000);
	//b.clear();
	//b.run_backtest(p2,strat, Regimes::EASY, 20);
	//Metrics m2(b.get_equity_curve());
	//m2.print_metrics(b.get_equity_curve());
	//TradeMetrics::print(p2.get_trades());

	Portfolio p3(10000);
	b.clear();
	b.run_backtest(p3,strat, Regimes::MODERN, 20);
	Metrics m3(b.get_equity_curve());
	m3.print_metrics(b.get_equity_curve());
	TradeMetrics::print(p3.get_trades());
	TradeMetrics::save_csv(p3.get_trades(), "output/trades.csv");
	

	return 0;
}

