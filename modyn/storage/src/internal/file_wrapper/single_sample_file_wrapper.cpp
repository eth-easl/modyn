#include "internal/file_wrapper/single_sample_file_wrapper.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>

#include "modyn/utils/utils.hpp"

using namespace modyn::storage;

uint64_t SingleSampleFileWrapper::get_number_of_samples() {
  ASSERT(file_wrapper_config_["file_extension"], "File wrapper configuration does not contain a file extension");
  const auto file_extension = file_wrapper_config_["file_extension"].as<std::string>();
  if (file_path_.find(file_extension) == std::string::npos) {
    return 0;
  }
  return 1;
}

int64_t SingleSampleFileWrapper::get_label(uint64_t /* index */) {
  ASSERT(file_wrapper_config_["file_extension"], "File wrapper configuration does not contain a label file extension");
  const auto label_file_extension = file_wrapper_config_["label_file_extension"].as<std::string>();
  auto label_path = std::filesystem::path(file_path_).replace_extension(label_file_extension);
  ASSERT(filesystem_wrapper_->exists(label_path), fmt::format("Label file does not exist: {}", label_path.string()));
  std::vector<unsigned char> label = filesystem_wrapper_->get(label_path);
  if (!label.empty()) {
    auto label_str = std::string(reinterpret_cast<char*>(label.data()), label.size());
    return std::stoi(label_str);
  }
  FAIL(fmt::format("Label file is empty: {}", label_path.string()));
  return -1;
}

std::vector<int64_t> SingleSampleFileWrapper::get_all_labels() { return std::vector<int64_t>{get_label(0)}; }

std::vector<unsigned char> SingleSampleFileWrapper::get_sample(uint64_t index) {
  ASSERT(index == 0,
         fmt::format("Single sample file wrappers can only access the first sample. file_path = {}, index = {}",
                     file_path_, index));
  return filesystem_wrapper_->get(file_path_);
}

std::vector<std::vector<unsigned char>> SingleSampleFileWrapper::get_samples(uint64_t start, uint64_t end) {
  ASSERT(
      start == 0 && end == 1,
      fmt::format("Single sample file wrappers can only access the first sample. file_path = {}, start = {}, end = {}",
                  file_path_, start, end));
  return std::vector<std::vector<unsigned char>>{get_sample(0)};
}

std::vector<std::vector<unsigned char>> SingleSampleFileWrapper::get_samples_from_indices(
    const std::vector<uint64_t>& indices) {
  ASSERT(indices.size() == 1 && indices[0] == 0,
         fmt::format("Single sample file wrappers can only access the first sample. file_path = {}, indices.size() = "
                     "{}, indices = [{}]",
                     file_path_, indices.size(), fmt::join(indices, ", ")));
  return std::vector<std::vector<unsigned char>>{get_sample(0)};
}

/*
 * get_targets for single sample file wrappers.
 * Reads from a separate target file whose extension is specified in the configuration as "target_file_extension".
 */
std::vector<std::vector<unsigned char>> SingleSampleFileWrapper::get_targets(uint64_t start, uint64_t end) {
  ASSERT(
      start == 0 && end == 1,
      fmt::format(
          "Single sample file wrappers can only access the first sample target. file_path = {}, start = {}, end = {}",
          file_path_, start, end));
  return std::vector<std::vector<unsigned char>>{get_target(0)};
}

/*
 * get_targets_from_indices for single sample file wrappers.
 */
std::vector<std::vector<unsigned char>> SingleSampleFileWrapper::get_targets_from_indices(
    const std::vector<uint64_t>& indices) {
  ASSERT(indices.size() == 1 && indices[0] == 0,
         fmt::format("Single sample file wrappers can only access the first sample target. file_path = {}, "
                     "indices.size() = {}, indices = [{}]",
                     file_path_, indices.size(), fmt::join(indices, ", ")));
  return std::vector<std::vector<unsigned char>>{get_target(0)};
}

void SingleSampleFileWrapper::validate_file_extension() {
  ASSERT(file_wrapper_config_["file_extension"], "File wrapper configuration does not contain a file extension");
  const auto file_extension = file_wrapper_config_["file_extension"].as<std::string>();
  if (file_path_.find(file_extension) == std::string::npos) {
    FAIL(fmt::format("File extension {} does not match file path {}", file_extension, file_path_));
  }
}

void SingleSampleFileWrapper::delete_samples(const std::vector<uint64_t>& /* indices */) {
  // The file will be deleted at a higher level
}

FileWrapperType SingleSampleFileWrapper::get_type() { return FileWrapperType::SINGLE_SAMPLE; }

/*
 * get_target for single sample file wrappers.
 * Reads from a separate target file whose extension is specified in the configuration as "target_file_extension".
 */
std::vector<unsigned char> SingleSampleFileWrapper::get_target(uint64_t index) {
  ASSERT(index == 0,
         fmt::format("Single sample file wrappers can only access the first sample. file_path = {}, index = {}",
                     file_path_, index));
  ASSERT(file_wrapper_config_["target_file_extension"],
         "File wrapper configuration does not contain a target file extension");
  const auto target_file_extension = file_wrapper_config_["target_file_extension"].as<std::string>();
  auto target_path = std::filesystem::path(file_path_).replace_extension(target_file_extension);
  ASSERT(filesystem_wrapper_->exists(target_path), fmt::format("Target file does not exist: {}", target_path.string()));
  return filesystem_wrapper_->get(target_path);
}
