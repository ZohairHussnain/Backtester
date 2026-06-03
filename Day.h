#pragma once

#include <string>

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
