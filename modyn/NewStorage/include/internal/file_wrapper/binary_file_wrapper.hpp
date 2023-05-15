#pragma once

#include <spdlog/spdlog.h>

#include <cstddef>
#include <iostream>

#include "internal/file_wrapper/abstract_file_wrapper.hpp"

namespace storage {
class BinaryFileWrapper : public AbstractFileWrapper {  // NOLINT
 private:
  int64_t record_size_;
  int64_t label_size_;
  int64_t file_size_;
  int64_t sample_size_;
  static void validate_request_indices(int total_samples, const std::vector<int64_t>* indices) {
    for (int indice : *indices) {  // NOLINT (we want to iterate over the indices)
      if (indice < 0 || indice > (total_samples - 1)) {
        SPDLOG_ERROR("Requested index {} is out of bounds.", indice);
        throw std::out_of_range("Requested index is out of bounds.");
      }
    }
  }
  static int int_from_bytes(const unsigned char* begin, const unsigned char* end);

 public:
  BinaryFileWrapper(const std::string& path, const YAML::Node& fw_config,  // NOLINT
                    AbstractFilesystemWrapper* fs_wrapper)
      : AbstractFileWrapper(path, fw_config, fs_wrapper) {
    if (!fw_config["record_size"]) {
      throw std::runtime_error("record_size_must be specified in the file wrapper config.");
    }
    this->record_size_ = fw_config["record_size"].as<int>();
    if (!fw_config["label_size"]) {
      throw std::runtime_error("label_size must be specified in the file wrapper config.");
    }
    this->label_size_ = fw_config["label_size"].as<int>();
    this->sample_size_ = this->record_size_ - this->label_size_;

    if (this->record_size_ - this->label_size_ < 1) {
      throw std::runtime_error(
          "Each record must have at least 1 byte of data "
          "other than the label.");
    }

    this->validate_file_extension();
    this->file_size_ = fs_wrapper->get_file_size(path);

    if (this->file_size_ % this->record_size_ != 0) {
      throw std::runtime_error("File size must be a multiple of the record size.");
    }
  }
  int get_number_of_samples() override;
  int get_label(int index) override;
  std::vector<int>* get_all_labels() override;
  std::vector<std::vector<unsigned char>>* get_samples(int64_t start, int64_t end) override;
  std::vector<unsigned char>* get_sample(int64_t index) override;
  std::vector<std::vector<unsigned char>>* get_samples_from_indices(std::vector<int64_t>* indices) override;
  void validate_file_extension() override;
  std::string get_name() override { return "BIN"; }
  ~BinaryFileWrapper() override = default;
};
}  // namespace storage
