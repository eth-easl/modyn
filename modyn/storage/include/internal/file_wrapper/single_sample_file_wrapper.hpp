#pragma once

#include <cstddef>
#include <optional>

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
  std::vector<unsigned char> get_target(uint64_t index) override;
  std::vector<std::vector<unsigned char>> get_samples(uint64_t start, uint64_t end) override;
  std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<uint64_t>& indices) override;
  std::vector<std::vector<unsigned char>> get_targets(uint64_t start, uint64_t end) override;
  std::vector<std::vector<unsigned char>> get_targets_from_indices(const std::vector<uint64_t>& indices) override;
  void validate_file_extension() override;
  void delete_samples(const std::vector<uint64_t>& indices) override;
  void set_file_path(const std::string& path) override { file_path_ = path; }
  FileWrapperType get_type() override;

  bool has_labels() const override { return has_labels_; }
  bool has_targets() const override { return has_targets_; }

 private:
  bool has_labels_ = true;
  bool has_targets_ = false;
};

}  // namespace modyn::storage
