#pragma once

#include <cstddef>

#include "internal/file_wrapper/file_wrapper.hpp"

namespace modyn::storage {

class SingleSampleFileWrapper : public FileWrapper {
 public:
  SingleSampleFileWrapper(const std::string& path, const YAML::Node& fw_config,
                          std::shared_ptr<FilesystemWrapper> filesystem_wrapper)
      : FileWrapper(path, fw_config, std::move(filesystem_wrapper)) {
    validate_file_extension();
  }
  uint64_t get_number_of_samples() override;
  int64_t get_label(uint64_t index) override;
  std::vector<int64_t> get_all_labels() override;
  std::vector<unsigned char> get_sample(uint64_t index) override;
  std::vector<std::vector<unsigned char>> get_samples(uint64_t start, uint64_t end) override;
  std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<uint64_t>& indices,
                                                                   bool generative) override;
  void validate_file_extension() override;
  void delete_samples(const std::vector<uint64_t>& indices) override;
  void set_file_path(const std::string& path) override { file_path_ = path; }
  FileWrapperType get_type() override;
};
}  // namespace modyn::storage
