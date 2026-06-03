#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "Day.h"

struct FeatureRow {
	std::string date;
	std::string ticker;
	double ret_1d;
	double ret_5d;
	double ret_10d;
	double ret_20d;
	double rsi_14;
	double atr_14_pct;
	double volume_ratio_20;
	double dist_ma20;
	double dist_ma50;
	double dist_ma200;
	double rolling_vol_20;
};

class FeatureEngine {
private:
	static double adjusted_return(const std::vector<Day>& days, size_t i, size_t lookback) {
		double past = days[i - lookback].adjusted_close;
		if (past == 0.0) {
			return 0.0;
		}
		return days[i].adjusted_close / past - 1.0;
	}

	static double average_adjusted_close(const std::vector<Day>& days, size_t end, size_t window) {
		double sum = 0.0;
		for (size_t i = end + 1 - window; i <= end; ++i) {
			sum += days[i].adjusted_close;
		}
		return sum / static_cast<double>(window);
	}

	static double average_volume(const std::vector<Day>& days, size_t end, size_t window) {
		double sum = 0.0;
		for (size_t i = end + 1 - window; i <= end; ++i) {
			sum += days[i].volume;
		}
		return sum / static_cast<double>(window);
	}

	static double rsi_14(const std::vector<Day>& days, size_t end) {
		double gains = 0.0;
		double losses = 0.0;

		for (size_t i = end + 1 - 14; i <= end; ++i) {
			double change = days[i].adjusted_close - days[i - 1].adjusted_close;
			if (change > 0.0) {
				gains += change;
			}
			else {
				losses += -change;
			}
		}

		double avg_gain = gains / 14.0;
		double avg_loss = losses / 14.0;
		if (avg_loss == 0.0) {
			return avg_gain == 0.0 ? 50.0 : 100.0;
		}

		double rs = avg_gain / avg_loss;
		return 100.0 - (100.0 / (1.0 + rs));
	}

	static double atr_14_pct(const std::vector<Day>& days, size_t end) {
		double tr_sum = 0.0;

		for (size_t i = end + 1 - 14; i <= end; ++i) {
			double previous_close = days[i - 1].close;
			double high_low = days[i].high - days[i].low;
			double high_prev_close = std::abs(days[i].high - previous_close);
			double low_prev_close = std::abs(days[i].low - previous_close);
			tr_sum += std::max({ high_low, high_prev_close, low_prev_close });
		}

		double atr = tr_sum / 14.0;
		return days[end].close != 0.0 ? atr / days[end].close : 0.0;
	}

	static double rolling_vol_20(const std::vector<Day>& days, size_t end) {
		std::vector<double> returns;
		returns.reserve(20);

		for (size_t i = end + 1 - 20; i <= end; ++i) {
			returns.push_back(adjusted_return(days, i, 1));
		}

		double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / static_cast<double>(returns.size());
		double squared_diff_sum = 0.0;
		for (double ret : returns) {
			double diff = ret - mean;
			squared_diff_sum += diff * diff;
		}

		return std::sqrt(squared_diff_sum / static_cast<double>(returns.size()));
	}

public:
	static std::vector<FeatureRow> generate(const std::string& ticker, const std::vector<Day>& days) {
		std::vector<FeatureRow> features;
		constexpr size_t min_history = 200;
		if (days.size() < min_history) {
			return features;
		}

		features.reserve(days.size() - min_history + 1);
		for (size_t i = min_history - 1; i < days.size(); ++i) {
			double ma20 = average_adjusted_close(days, i, 20);
			double ma50 = average_adjusted_close(days, i, 50);
			double ma200 = average_adjusted_close(days, i, 200);
			double avg_volume20 = average_volume(days, i, 20);
			double adjusted_close = days[i].adjusted_close;

			features.push_back(FeatureRow{
				days[i].date,
				ticker,
				adjusted_return(days, i, 1),
				adjusted_return(days, i, 5),
				adjusted_return(days, i, 10),
				adjusted_return(days, i, 20),
				rsi_14(days, i),
				atr_14_pct(days, i),
				avg_volume20 != 0.0 ? days[i].volume / avg_volume20 : 0.0,
				ma20 != 0.0 ? adjusted_close / ma20 - 1.0 : 0.0,
				ma50 != 0.0 ? adjusted_close / ma50 - 1.0 : 0.0,
				ma200 != 0.0 ? adjusted_close / ma200 - 1.0 : 0.0,
				rolling_vol_20(days, i)
			});
		}

		return features;
	}
};


