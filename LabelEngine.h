#pragma once

#include <string>
#include <vector>

#include "Day.h"
#include "FeatureEngine.h"

struct LabelRow {
	std::string date;
	std::string ticker;
	int label;
	double entry_price;
	double target_price;
	double stop_price;
};

class LabelEngine {
public:
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
		labels.reserve(last_index - first_index + 1);

		for (size_t i = first_index; i <= last_index; ++i) {
			double entry_price = days[i + 1].open;
			double target_price = 0.0;
			double stop_price = 0.0;
			int label = 0;

			if (entry_price > 0.0) {
				target_price = entry_price * (1.0 + target_pct);
				stop_price = entry_price * (1.0 - stop_pct);

				for (size_t j = i + 1; j <= i + max_horizon_days; ++j) {
					bool stop_hit = days[j].low <= stop_price;
					bool target_hit = days[j].high >= target_price;

					if (stop_hit) {
						label = 0;
						break;
					}
					if (target_hit) {
						label = 1;
						break;
					}
				}
			}

			labels.push_back(LabelRow{
				days[i].date,
				ticker,
				label,
				entry_price,
				target_price,
				stop_price
			});
		}

		return labels;
	}
};
