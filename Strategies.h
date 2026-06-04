#pragma once

#include <optional>
#include <string>
#include <vector>

#include "Core.h"
#include "Day.h"
#include "PredictionLoader.h"

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
