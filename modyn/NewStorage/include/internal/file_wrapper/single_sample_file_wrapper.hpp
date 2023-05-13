#pragma once

#include <cstddef>

#include "internal/file_wrapper/abstract_file_wrapper.hpp"

namespace storage {
class SingleSampleFileWrapper : public AbstractFileWrapper {
 public:
  SingleSampleFileWrapper(std::string path, YAML::Node fw_config,
                          AbstractFilesystemWrapper* fs_wrapper)
      : AbstractFileWrapper(path, fw_config, fs_wrapper) {
    this->validate_file_extension();
  }
  int get_number_of_samples();
  int get_label(int index);
  std::vector<int>* get_all_labels();
  std::vector<std::vector<unsigned char>>* get_samples(int start, int end);
  std::vector<unsigned char>* get_sample(int index);
  std::vector<std::vector<unsigned char>>* get_samples_from_indices(std::vector<int>* indices);
  void validate_file_extension();
  std::string get_name() { return "SINGLE_SAMPLE"; }
  ~SingleSampleFileWrapper() {}
};
}  // namespace storage
