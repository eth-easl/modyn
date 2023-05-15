#pragma once

#include <cstddef>

#include "internal/file_wrapper/abstract_file_wrapper.hpp"

namespace storage {
class SingleSampleFileWrapper : public AbstractFileWrapper {  // NOLINT
 public:
  SingleSampleFileWrapper(const std::string &path, const YAML::Node &fw_config, AbstractFilesystemWrapper* fs_wrapper)
      : AbstractFileWrapper(path, fw_config, fs_wrapper) {
    this->validate_file_extension();
  }
  int get_number_of_samples() override;
  int get_label(int index) override;
  std::vector<int>* get_all_labels() override;
  std::vector<std::vector<unsigned char>>* get_samples(int64_t start, int64_t end) override;
  std::vector<unsigned char>* get_sample(int64_t index) override;
  std::vector<std::vector<unsigned char>>* get_samples_from_indices(std::vector<int64_t>* indices) override;
  void validate_file_extension() override;
  std::string get_name() override { return "SINGLE_SAMPLE"; }
  ~SingleSampleFileWrapper() override = default;
};
}  // namespace storage
