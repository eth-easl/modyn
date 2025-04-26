#pragma once

#include <spdlog/spdlog.h>

#include <cstddef>
#include <fstream>
#include <iostream>

#include "internal/file_wrapper/file_wrapper.hpp"
#include "modyn/utils/utils.hpp"

namespace modyn::storage {
class BinaryFileWrapper : public FileWrapper {
 public:
  BinaryFileWrapper(const std::string& path, const YAML::Node& fw_config,
                    std::shared_ptr<FilesystemWrapper> filesystem_wrapper)
      : FileWrapper(path, fw_config, std::move(filesystem_wrapper)) {
    ASSERT(filesystem_wrapper_ != nullptr, "Filesystem wrapper cannot be null.");
    ASSERT(fw_config["record_size"], "record_size must be specified in the file wrapper config.");
    record_size_ = fw_config["record_size"].as<uint64_t>();


    if (!file_wrapper_config_["has_labels"] || file_wrapper_config_["has_labels"].as<bool>()) {
      has_labels_ = true;
      ASSERT(fw_config["label_size"], "label_size must be specified in the file wrapper config.");
      label_size_ = fw_config["label_size"].as<uint64_t>();
    } else {
      has_labels_ = false;
      label_size_ = 0;  // No labels exist
    }


    if (fw_config["has_targets"] && fw_config["has_targets"].as<bool>()) {
      has_targets_ = true;
      ASSERT(fw_config["target_size"], "target_size must be specified in the file wrapper config.");
      target_size_ = fw_config["target_size"].as<uint64_t>();
    } else {
      has_targets_ = false;
      target_size_ = 0;
    }

    // Adjust sample_size_ to account for target size.
    sample_size_ = record_size_ - label_size_ - target_size_;

    validate_file_extension();
    file_size_ = filesystem_wrapper_->get_file_size(path);
    little_endian_ = fw_config["byteorder"].as<std::string>() == "little";

    ASSERT(static_cast<int64_t>(record_size_ - label_size_ - target_size_) >= 1,
           "Each record must have at least 1 byte of data other than the label and target.");
    ASSERT(file_size_ % record_size_ == 0, "File size must be a multiple of the record size.");

    stream_ = filesystem_wrapper_->get_stream(path);
  }

  uint64_t get_number_of_samples() override;
  int64_t get_label(uint64_t index) override;
  std::vector<int64_t> get_all_labels() override;
  std::vector<unsigned char> get_sample(uint64_t index) override;
  // ADDED: New function to retrieve target data for a given record.
  std::vector<unsigned char> get_target(uint64_t index) override;
  std::vector<std::vector<unsigned char>> get_samples(uint64_t start, uint64_t end) override;
  std::vector<std::vector<unsigned char>> get_targets(uint64_t start, uint64_t end) override;
  std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<uint64_t>& indices) override;
  std::vector<std::vector<unsigned char>> get_targets_from_indices(const std::vector<uint64_t>& indices) override;
  void validate_file_extension() override;
  void delete_samples(const std::vector<uint64_t>& indices) override;
  void set_file_path(const std::string& path) override;
  FileWrapperType get_type() override;

  ~BinaryFileWrapper() override {
    if (stream_->is_open()) {
      stream_->close();
    }
  }
  BinaryFileWrapper(const BinaryFileWrapper&) = default;
  BinaryFileWrapper& operator=(const BinaryFileWrapper&) = default;
  BinaryFileWrapper(BinaryFileWrapper&&) = default;
  BinaryFileWrapper& operator=(BinaryFileWrapper&&) = default;

 private:
  static void validate_request_indices(uint64_t total_samples, const std::vector<uint64_t>& indices);
  static int64_t int_from_bytes_little_endian(const unsigned char* begin, const unsigned char* end);
  static int64_t int_from_bytes_big_endian(const unsigned char* begin, const unsigned char* end);
  std::ifstream* get_stream();
  uint64_t record_size_;
  uint64_t label_size_;
  uint64_t target_size_;
  uint64_t file_size_;

  uint64_t sample_size_;
  bool little_endian_;
  std::shared_ptr<std::ifstream> stream_;
  bool has_labels_;
  bool has_targets_;
  friend class BinaryFileWrapperTest;  // let gtest access private members
};
}  // namespace modyn::storage
