#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "FeatureEngine.h"
#include "LabelEngine.h"

class MLDataExporter {
private:
	static std::string key(const std::string& date, const std::string& ticker) {
		return date + "|" + ticker;
	}

public:
	struct LabelPair {
		int label_target_stop;
		int label_median_return;
		double forward_return;
	};

	static size_t save_csv(
		const std::vector<FeatureRow>& features,
		const std::vector<LabelRow>& labels,
		const std::string& path = "output/ml_dataset.csv"
	) {
		namespace fs = std::filesystem;
		fs::path output_path(path);
		if (output_path.has_parent_path()) {
			fs::create_directories(output_path.parent_path());
		}

		std::unordered_map<std::string, LabelPair> labels_by_key;
		labels_by_key.reserve(labels.size());
		for (const auto& label : labels) {
			labels_by_key[key(label.date, label.ticker)] = {
				label.label_target_stop,
				label.label_median_return,
				label.forward_return,
			};
		}

		std::ofstream file(path);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open ML dataset output file: " + path);
		}

		file << "date,ticker,"
			 << "ret_1d,ret_5d,ret_10d,ret_20d,rsi_14,atr_14_pct,"
			 << "volume_ratio_20,dist_ma20,dist_ma50,dist_ma200,rolling_vol_20,"
			 << "label_target_stop,label_median_return,forward_return\n";

		size_t rows_written = 0;
		for (const auto& feature : features) {
			auto it = labels_by_key.find(key(feature.date, feature.ticker));
			if (it == labels_by_key.end()) {
				continue;
			}

			file << feature.date << ","
				 << feature.ticker << ","
				 << feature.ret_1d << ","
				 << feature.ret_5d << ","
				 << feature.ret_10d << ","
				 << feature.ret_20d << ","
				 << feature.rsi_14 << ","
				 << feature.atr_14_pct << ","
				 << feature.volume_ratio_20 << ","
				 << feature.dist_ma20 << ","
				 << feature.dist_ma50 << ","
				 << feature.dist_ma200 << ","
				 << feature.rolling_vol_20 << ","
				 << it->second.label_target_stop << ","
				 << it->second.label_median_return << ","
				 << it->second.forward_return << "\n";
			++rows_written;
		}

		return rows_written;
	}
};
