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
  if (get_number_of_samples() == 0) {
    throw std::runtime_error("File has wrong file extension.");
  }
  if (index != 0) {
    throw std::runtime_error("SingleSampleFileWrapper contains only one sample.");
  }
  if (!file_wrapper_config_["label_file_extension"]) {
    throw std::runtime_error("No label file extension defined.");
  }
  const auto label_file_extension = file_wrapper_config_["label_file_extension"].as<std::string>();
  auto label_path = std::filesystem::path(file_path_).replace_extension(label_file_extension);
  std::vector<unsigned char> label = filesystem_wrapper_->get(label_path);
  if (!label.empty()) {
    auto label_str = std::string(reinterpret_cast<char*>(label.data()), label.size());
    return std::stoi(label_str);
  }
  throw std::runtime_error("Label file not found.");
}

std::vector<int64_t> SingleSampleFileWrapper::get_all_labels() { return std::vector<int64_t>{get_label(0)}; }

std::vector<unsigned char> SingleSampleFileWrapper::get_sample(int64_t index) {
  if (get_number_of_samples() == 0) {
    throw std::runtime_error("File has wrong file extension.");
  }
  if (index != 0) {
    throw std::runtime_error("SingleSampleFileWrapper contains only one sample.");
  }
  return filesystem_wrapper_->get(file_path_);
}

std::vector<std::vector<unsigned char>> SingleSampleFileWrapper::get_samples(int64_t start, int64_t end) {
  if (get_number_of_samples() == 0) {
    throw std::runtime_error("File has wrong file extension.");
  }
  if (start != 0 || end != 1) {
    throw std::runtime_error("SingleSampleFileWrapper contains only one sample.");
  }
  return std::vector<std::vector<unsigned char>>{get_sample(0)};
}

std::vector<std::vector<unsigned char>> SingleSampleFileWrapper::get_samples_from_indices(
    const std::vector<int64_t>& indices) {
  if (get_number_of_samples() == 0) {
    throw std::runtime_error("File has wrong file extension.");
  }
  if (indices.size() != 1) {
    throw std::runtime_error("SingleSampleFileWrapper contains only one sample.");
  }
  return std::vector<std::vector<unsigned char>>{get_sample(0)};
}

void SingleSampleFileWrapper::validate_file_extension() {
  if (!file_wrapper_config_["file_extension"]) {
    throw std::runtime_error("file_extension must be specified in the file wrapper config.");
  }
  const auto file_extension = file_wrapper_config_["file_extension"].as<std::string>();
  if (file_path_.find(file_extension) == std::string::npos) {
    throw std::runtime_error("File has wrong file extension.");
  }
}