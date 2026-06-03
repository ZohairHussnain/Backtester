#pragma once

#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

class PredictionLoader {
private:
	std::unordered_map<std::string, double> probabilities;

	static std::string key(const std::string& date, const std::string& ticker) {
		return date + "|" + ticker;
	}

	static bool parse_line(const std::string& line, std::string& date, std::string& ticker, double& probability) {
		size_t first_comma = line.find(',');
		if (first_comma == std::string::npos) {
			return false;
		}

		size_t second_comma = line.find(',', first_comma + 1);
		if (second_comma == std::string::npos) {
			return false;
		}

		date = line.substr(0, first_comma);
		ticker = line.substr(first_comma + 1, second_comma - first_comma - 1);
		std::string probability_text = line.substr(second_comma + 1);
		if (date.empty() || ticker.empty() || probability_text.empty()) {
			return false;
		}

		try {
			probability = std::stod(probability_text);
		}
		catch (...) {
			return false;
		}

		return true;
	}

public:
	PredictionLoader() = default;

	explicit PredictionLoader(const std::string& path) {
		load(path);
	}

	void load(const std::string& path) {
		std::ifstream file(path);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open prediction CSV: " + path);
		}

		probabilities.clear();

		std::string line;
		bool first_line = true;
		while (std::getline(file, line)) {
			if (line.empty()) {
				continue;
			}

			if (first_line) {
				first_line = false;
				if (line == "date,ticker,probability") {
					continue;
				}
			}

			std::string date;
			std::string ticker;
			double probability = 0.0;
			if (!parse_line(line, date, ticker, probability)) {
				continue;
			}

			probabilities[key(date, ticker)] = probability;
		}
	}

	std::optional<double> get_probability(const std::string& date, const std::string& ticker) const {
		auto it = probabilities.find(key(date, ticker));
		if (it == probabilities.end()) {
			return std::nullopt;
		}
		return it->second;
	}
};
