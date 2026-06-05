#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "Day.h"
#include "FeatureEngine.h"

struct LabelRow {
	std::string date;
	std::string ticker;
	int label_target_stop;
	int label_median_return;
	double forward_return;
};

class LabelEngine {
private:
	// Compute target-stop label: did price hit +target% before -stop% within horizon?
	static int compute_target_stop(
		const std::vector<Day>& days, size_t i,
		double entry_price, double target_pct, double stop_pct, size_t max_horizon_days
	) {
		if (entry_price <= 0.0) return 0;

		double target_price = entry_price * (1.0 + target_pct);
		double stop_price = entry_price * (1.0 - stop_pct);

		for (size_t j = i + 1; j <= i + max_horizon_days; ++j) {
			bool stop_hit = days[j].low <= stop_price;
			bool target_hit = days[j].high >= target_price;

			if (stop_hit && target_hit) {
				double dist_to_stop = std::abs(days[j].open - stop_price);
				double dist_to_target = std::abs(days[j].open - target_price);
				return dist_to_stop <= dist_to_target ? 0 : 1;
			}
			if (stop_hit) return 0;
			if (target_hit) return 1;
		}

		// Timeout: label based on whether position is profitable at horizon close.
		double exit_close = days[i + max_horizon_days].close;
		return exit_close > entry_price ? 1 : 0;
	}

public:
	static constexpr size_t median_window = 252;

	static std::vector<LabelRow> generate(
		const std::string& ticker,
		const std::vector<Day>& days,
		double target_pct,
		double stop_pct,
		size_t max_horizon_days
	) {
		std::vector<LabelRow> labels;
		if (days.size() < FeatureEngine::minimum_history || target_pct <= 0.0 || stop_pct <= 0.0 || max_horizon_days == 0) {
			return labels;
		}

		size_t first_index = FeatureEngine::minimum_history - 1;
		if (first_index + max_horizon_days >= days.size()) {
			return labels;
		}

		size_t last_index = days.size() - max_horizon_days - 1;

		// Precompute forward returns for median-return label.
		// forward_return[i] = close[i + max_horizon_days] / close[i] - 1
		std::vector<double> forward_returns(days.size(), 0.0);
		for (size_t i = 0; i + max_horizon_days < days.size(); ++i) {
			if (days[i].close > 0.0) {
				forward_returns[i] = days[i + max_horizon_days].close / days[i].close - 1.0;
			}
		}

		// Compute trailing rolling median of forward returns.
		// For each bar i, median of forward_returns[i-median_window+1 .. i].
		// Uses only past/current forward returns (no future leakage in the median itself).
		std::vector<double> rolling_medians(days.size(), std::numeric_limits<double>::quiet_NaN());
		for (size_t i = first_index; i <= last_index; ++i) {
			size_t window_start = (i >= median_window) ? i - median_window + 1 : 0;
			// Only use entries where forward_return was valid (had enough future bars)
			std::vector<double> window_vals;
			for (size_t k = window_start; k <= i; ++k) {
				if (k + max_horizon_days < days.size()) {
					window_vals.push_back(forward_returns[k]);
				}
			}
			if (window_vals.size() >= 60) {
				std::sort(window_vals.begin(), window_vals.end());
				size_t n = window_vals.size();
				rolling_medians[i] = (n % 2 == 0)
					? (window_vals[n / 2 - 1] + window_vals[n / 2]) / 2.0
					: window_vals[n / 2];
			}
		}

		labels.reserve(last_index - first_index + 1);

		for (size_t i = first_index; i <= last_index; ++i) {
			double entry_price = days[i + 1].open;
			double fwd_ret = forward_returns[i];

			int label_ts = compute_target_stop(days, i, entry_price, target_pct, stop_pct, max_horizon_days);

			int label_mr = 0;
			if (!std::isnan(rolling_medians[i])) {
				label_mr = fwd_ret > rolling_medians[i] ? 1 : 0;
			} else {
				// Not enough history for rolling median — use simple sign
				label_mr = fwd_ret > 0.0 ? 1 : 0;
			}

			labels.push_back(LabelRow{
				days[i].date,
				ticker,
				label_ts,
				label_mr,
				fwd_ret,
			});
		}

		return labels;
	}
};
