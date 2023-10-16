#pragma once

#include <spdlog/spdlog.h>

#include <cstddef>
#include <iostream>

#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/utils/utils.hpp"

namespace storage::file_wrapper {
class BinaryFileWrapper : public storage::file_wrapper::FileWrapper {
 private:
  int64_t record_size_;
  int64_t label_size_;
  int64_t file_size_;
  int64_t sample_size_;
  static void validate_request_indices(int64_t total_samples, const std::vector<int64_t>& indices);
  static int64_t int_from_bytes(const unsigned char* begin, const unsigned char* end);

 public:
  BinaryFileWrapper(const std::string& path, const YAML::Node& fw_config,
                    std::shared_ptr<storage::filesystem_wrapper::FilesystemWrapper> filesystem_wrapper)
      : storage::file_wrapper::FileWrapper(path, fw_config, std::move(filesystem_wrapper)) {
    ASSERT(filesystem_wrapper_ != nullptr, "Filesystem wrapper cannot be null.");

    if (!fw_config["record_size"]) {
      FAIL("record_size_must be specified in the file wrapper config.");
    }
    record_size_ = fw_config["record_size"].as<int64_t>();
    if (!fw_config["label_size"]) {
      FAIL("label_size must be specified in the file wrapper config.");
    }
    label_size_ = fw_config["label_size"].as<int64_t>();
    sample_size_ = record_size_ - label_size_;

    if (record_size_ - label_size_ < 1) {
      FAIL(
          "Each record must have at least 1 byte of data "
          "other than the label.");
    }

    validate_file_extension();
    file_size_ = filesystem_wrapper_->get_file_size(path);

    if (file_size_ % record_size_ != 0) {
      FAIL("File size must be a multiple of the record size.");
    }
  }
  int64_t get_number_of_samples() override;
  int64_t get_label(int64_t index) override;
  std::vector<int64_t> get_all_labels() override;
  std::vector<std::vector<unsigned char>> get_samples(int64_t start, int64_t end) override;
  std::vector<unsigned char> get_sample(int64_t index) override;
  std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<int64_t>& indices) override;
  void validate_file_extension() override;
  void delete_samples(const std::vector<int64_t>& indices) override;
  void set_file_path(const std::string& path) override;
  FileWrapperType get_type() override;
  ~BinaryFileWrapper() = default;
};
}  // namespace storage::file_wrapper
