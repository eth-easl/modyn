#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "internal/filesystem_wrapper/abstract_filesystem_wrapper.hpp"

namespace storage {
class AbstractFileWrapper {  // NOLINT
 protected:
  std::string file_path_;
  YAML::Node file_wrapper_config_;
  std::shared_ptr<AbstractFilesystemWrapper> filesystem_wrapper_;

 public:
  AbstractFileWrapper(std::string path, const YAML::Node& fw_config,
                      std::shared_ptr<AbstractFilesystemWrapper>& fs_wrapper)
      : file_path_(std::move(path)), file_wrapper_config_(fw_config), filesystem_wrapper_(std::move(fs_wrapper)) {}
  virtual int64_t get_number_of_samples() = 0;
  virtual std::vector<std::vector<unsigned char>> get_samples(int64_t start, int64_t end) = 0;
  virtual int64_t get_label(int64_t index) = 0;
  virtual std::vector<int64_t> get_all_labels() = 0;
  virtual std::vector<unsigned char> get_sample(int64_t index) = 0;
  virtual std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<int64_t>& indices) = 0;
  virtual std::string get_name() = 0;
  virtual void validate_file_extension() = 0;
  virtual ~AbstractFileWrapper() {}  // NOLINT
  AbstractFileWrapper(const AbstractFileWrapper& other) = default;
};
}  // namespace storage
