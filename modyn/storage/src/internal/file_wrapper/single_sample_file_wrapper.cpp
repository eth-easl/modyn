#include "internal/file_wrapper/single_sample_file_wrapper.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>

using namespace storage;

int64_t SingleSampleFileWrapper::get_number_of_samples() {
  if (file_path_.find(file_wrapper_config_["file_extension"].as<std::string>()) == std::string::npos) {
    return 0;
  }
  return 1;
}

int64_t SingleSampleFileWrapper::get_label(int64_t index) {
  const auto label_file_extension = file_wrapper_config_["label_file_extension"].as<std::string>();
  auto label_path = std::filesystem::path(file_path_).replace_extension(label_file_extension);
  std::vector<unsigned char> label = filesystem_wrapper_->get(label_path);
  if (!label.empty()) {
    auto label_str = std::string(reinterpret_cast<char*>(label.data()), label.size());
    return std::stoi(label_str);
  }
  SPDLOG_ERROR("Label file not found for file {}", file_path_);
  return -1;
}

std::vector<int64_t> SingleSampleFileWrapper::get_all_labels() { return std::vector<int64_t>{get_label(0)}; }

std::vector<unsigned char> SingleSampleFileWrapper::get_sample(int64_t index) {
  return filesystem_wrapper_->get(file_path_);
}

std::vector<std::vector<unsigned char>> SingleSampleFileWrapper::get_samples(int64_t start, int64_t end) {
  return std::vector<std::vector<unsigned char>>{get_sample(0)};
}

std::vector<std::vector<unsigned char>> SingleSampleFileWrapper::get_samples_from_indices(
    const std::vector<int64_t>& indices) {  // NOLINT (misc-unused-parameters)
  return std::vector<std::vector<unsigned char>>{get_sample(0)};
}

void SingleSampleFileWrapper::validate_file_extension() {
  const auto file_extension = file_wrapper_config_["file_extension"].as<std::string>();
}

void SingleSampleFileWrapper::delete_samples(const std::vector<int64_t>& indices) {  // NOLINT (misc-unused-parameters)
  filesystem_wrapper_->remove(file_path_);
}