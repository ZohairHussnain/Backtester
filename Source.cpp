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
using json = nlohmann::json;
using response = cpr::Response;

struct Day {
	std::string date;
	double close;
	Day(std::string date, double close) : date(date), close(close) {}
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
Signal generate_signal(const std::vector<Day>& time_series, size_t i) {
	if (i <= 1)
		return Signal::HOLD;
	if (time_series[i - 1].close > time_series[i - 2].close)
		return Signal::BUY;
	else if (time_series[i - 1].close < time_series[i - 2].close)
		return Signal::SELL;
	return Signal::HOLD;
}
class Portfolio {
private:
	double cash;
	double shares;
	PositionState state;
	double slippage = 0.0005;
	double calculate_transaction_costs(double price, double shares) const {
		double trade_value = price * shares;

		double commission = shares * 0.0035;
		commission = std::max(commission, 0.35); //min cost $0.35 
		commission = std::min(commission, 0.01 * trade_value); //max cost 1%
		return commission;
	}

public:
	Portfolio() : cash(10000), shares(0), state(PositionState::IN_CASH) {}
	Portfolio(double cash) : cash(cash), shares(0), state(PositionState::IN_CASH) {}
	bool in_position() const {
		return state == PositionState::IN_POSITION;
	}
	void buy(double price) {
		if (this->state == PositionState::IN_POSITION || price <= 0)
			return;
		double execution_price = price * (1.0 + slippage);

		double tentative_shares = cash / execution_price;
		double fees = calculate_transaction_costs(execution_price, tentative_shares);

		double investable_cash = cash - fees;

		if (investable_cash <= 0)
			return;

		

		this->shares = investable_cash / execution_price;
		this->cash = 0.0;
		this->state = PositionState::IN_POSITION;
	}
	void sell(double price) {
		if (this->state == PositionState::IN_CASH || price <= 0)
			return;

		double execution_price = price * (1.0 - slippage);

		double value = shares * execution_price;
		double fees = calculate_transaction_costs(execution_price, shares);

		this->cash = value - fees;
		this->shares = 0.0;
		this->state = PositionState::IN_CASH;
	}
	double value(double price) const {
		if (this->state == PositionState::IN_CASH)
			return this->cash;
		else
			return this->shares * price;
	}

};
class Backtest {
private:
	std::vector<Day> time_series;
	std::vector<double> equity_curve;
	json fetch_data(std::string ticker) {
		std::string function{ "TIME_SERIES_DAILY" };
		std::string API_KEY = "78K9C0K0JIKZFLXB";
		std::string url = "https://www.alphavantage.co/query?function=";
		std::string endpoint = url + function + "&symbol=" + ticker + "&apikey=" + API_KEY;
		cpr::Response r = cpr::Get(cpr::Url(endpoint));
		json data = json::parse(r.text);
		std::ofstream save(ticker + ".json");
		save << data.dump(4);
		return data;
	}

	json load_json_if_exists(std::string ticker) {
		namespace fs = std::filesystem;
		std::string filename = ticker + ".json";
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
			return fetch_data(ticker);
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
		json data = load_json_if_exists(ticker);
		std::string last_refresh = data.at("Meta Data").at("3. Last Refreshed").get<std::string>();
		std::cout << "Latest date: " << last_refresh << std::endl;

		json daily = data.at("Time Series (Daily)");


		for (auto it = daily.begin(); it != daily.end(); ++it) {
			std::string date = it.key();
			json daily_data = it.value();

			double close = std::stod(daily_data.at("4. close").get<std::string>());

			time_series.push_back(Day(date, close));
			//std::cout << date << " -> " << close << std::endl;
		}
		std::sort(time_series.begin(), time_series.end(),
			[](const Day& a, const Day& b) {
				return a.date < b.date;
			});
		if (!verify_true(time_series))
		{
			exit(1);
		}
	}
public:
	Backtest(std::string ticker) { init_time_series(ticker); }
	void run_backtest(Portfolio& p) {
		Signal signal{};
		double price{};
		for (size_t i = 2; i < time_series.size(); i++) {
			signal = generate_signal(time_series, i - 1);
			price = time_series[i].close;

			if (signal == Signal::BUY && !p.in_position()) {
				p.buy(price);
				//std::cout << "BUY @" << price << std::endl;
			}
			if (signal == Signal::SELL && p.in_position()) {
				p.sell(price);
				//std::cout << "SELL @" << price << std::endl;
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

		sharpe = (mean / std) * std::sqrt(252.0);
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

	
};


void saveVector(const std::vector<double>& v) {
	std::ofstream file("equity.dat");
	for (size_t i = 0; i < v.size(); ++i)
		file << i << " " << v[i] << "\n";
	file.close();
	system(
		"gnuplot -persist -e \""
		"set title 'Equity Curve';"
		"set xlabel 'Trade Number';"
		"set ylabel 'Account Value';"
		"set grid;"
		"plot 'equity.dat' using 1:2 with lines lw 2 title 'Equity'\""
	);
}

int main() {
	
	Portfolio p(10000);
	Backtest b("SPY");
	b.run_backtest(p);
	//b.print_equity_curve();
	Metrics m(b.get_equity_curve());
	auto returns = m.daily_returns();
	/*for (auto x : returns) {
		std::cout << x << std::endl;
	}*/
	auto drawdown = m.max_drawdown();
	std::cout << "MAX DRAWDOWN:" << drawdown << std::endl;
	auto cagr = m.getCAGR();
	std::cout << "CAGR:" << cagr << std::endl;
	auto sharpe = m.get_sharpe();
	std::cout << "SHARPE: " << sharpe << std::endl;

	saveVector(b.get_equity_curve());
	
	return 0;
}