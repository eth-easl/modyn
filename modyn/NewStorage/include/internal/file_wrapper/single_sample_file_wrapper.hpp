#pragma once

#include <cstddef>

#include "internal/file_wrapper/abstract_file_wrapper.hpp"

namespace storage {
class SingleSampleFileWrapper : public AbstractFileWrapper {  // NOLINT
 public:
  SingleSampleFileWrapper(std::string path, const YAML::Node fw_config, AbstractFilesystemWrapper* fs_wrapper)
      : AbstractFileWrapper(std::move(path), fw_config, fs_wrapper) {
    this->validate_file_extension();
  }
  int get_number_of_samples() override;
  int get_label(int index) override;
  std::vector<int>* get_all_labels() override;
  std::vector<std::vector<unsigned char>>* get_samples(int start, int end) override;
  std::vector<unsigned char>* get_sample(int index) override;
  std::vector<std::vector<unsigned char>>* get_samples_from_indices(std::vector<int>* indices) override;
  void validate_file_extension() override;
  std::string get_name() override { return "SINGLE_SAMPLE"; }
  ~SingleSampleFileWrapper() override = default;
};
}  // namespace storage
