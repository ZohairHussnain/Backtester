#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "Core.h"

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
