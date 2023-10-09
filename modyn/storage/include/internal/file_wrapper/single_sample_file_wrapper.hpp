#pragma once

#include <cstddef>

#include "internal/file_wrapper/file_wrapper.hpp"

namespace storage {
class SingleSampleFileWrapper : public FileWrapper {
 public:
  SingleSampleFileWrapper(const std::string& path, const YAML::Node& fw_config,
                          std::shared_ptr<FilesystemWrapper> filesystem_wrapper)
      : FileWrapper(path, fw_config, std::move(filesystem_wrapper)) {
    validate_file_extension();
  }
  int64_t get_number_of_samples() override;
  int64_t get_label(int64_t index) override;
  std::vector<int64_t> get_all_labels() override;
  std::vector<std::vector<unsigned char>> get_samples(int64_t start, int64_t end) override;
  std::vector<unsigned char> get_sample(int64_t index) override;
  std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<int64_t>& indices) override;
  void validate_file_extension() override;
  FileWrapperType get_type() override;
  void set_file_path(const std::string& path) override { file_path_ = path; }
  void delete_samples(const std::vector<int64_t>& indices) override;
  ~SingleSampleFileWrapper() override = default;
};
}  // namespace storage
