#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"

namespace storage {

enum FileWrapperType { SINGLE_SAMPLE, BINARY };

class FileWrapper {  // NOLINT
 protected:
  std::string file_path_;
  YAML::Node file_wrapper_config_;
  std::shared_ptr<FilesystemWrapper> filesystem_wrapper_;

 public:
  FileWrapper(std::string path, const YAML::Node& fw_config, std::shared_ptr<FilesystemWrapper> filesystem_wrapper)
      : file_path_{std::move(path)},
        file_wrapper_config_{fw_config},
        filesystem_wrapper_{std::move(filesystem_wrapper)} {}
  virtual int64_t get_number_of_samples() = 0;
  virtual std::vector<std::vector<unsigned char>> get_samples(int64_t start, int64_t end) = 0;
  virtual int64_t get_label(int64_t index) = 0;
  virtual std::vector<int64_t> get_all_labels() = 0;
  virtual std::vector<unsigned char> get_sample(int64_t index) = 0;
  virtual std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<int64_t>& indices) = 0;
  virtual FileWrapperType get_type() = 0;
  virtual void validate_file_extension() = 0;
  virtual ~FileWrapper() {}  // NOLINT
  FileWrapper(const FileWrapper& other) = default;
};
}  // namespace storage
