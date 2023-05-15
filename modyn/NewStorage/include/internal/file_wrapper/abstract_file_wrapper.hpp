#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "internal/filesystem_wrapper/abstract_filesystem_wrapper.hpp"

namespace storage {
class AbstractFileWrapper {  // NOLINT
 protected:
  std::string file_path_;
  YAML::Node file_wrapper_config_;
  AbstractFilesystemWrapper* filesystem_wrapper_;

 public:
  AbstractFileWrapper(std::string path, const YAML::Node& fw_config, AbstractFilesystemWrapper* fs_wrapper)
      : file_path_(std::move(path)), file_wrapper_config_(fw_config), filesystem_wrapper_(fs_wrapper) {}
  virtual int get_number_of_samples() = 0;
  virtual std::vector<std::vector<unsigned char>>* get_samples(int64_t start, int64_t end) = 0;
  virtual int get_label(int index) = 0;
  virtual std::vector<int>* get_all_labels() = 0;
  virtual std::vector<unsigned char>* get_sample(int64_t index) = 0;
  virtual std::vector<std::vector<unsigned char>>* get_samples_from_indices(std::vector<int64_t>* indices) = 0;
  virtual std::string get_name() = 0;
  virtual void validate_file_extension() = 0;
  virtual ~AbstractFileWrapper() {}  // NOLINT
};
}  // namespace storage
