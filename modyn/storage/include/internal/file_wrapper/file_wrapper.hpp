#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"

namespace modyn::storage {

enum FileWrapperType : uint8_t { INVALID_FW, SINGLE_SAMPLE, BINARY, CSV };

class FileWrapper {
 public:
  FileWrapper(std::string path, const YAML::Node& fw_config, std::shared_ptr<FilesystemWrapper> filesystem_wrapper)
      : file_path_{std::move(path)},
        file_wrapper_config_{fw_config},
        filesystem_wrapper_{std::move(filesystem_wrapper)} {}
  virtual uint64_t get_number_of_samples() = 0;
  virtual int64_t get_label(uint64_t index) = 0;
  virtual std::vector<int64_t> get_all_labels() = 0;
  virtual std::vector<unsigned char> get_sample(uint64_t index) = 0;
  virtual std::vector<std::vector<unsigned char>> get_samples(uint64_t start, uint64_t end) = 0;
  virtual std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<uint64_t>& indices,
                                                                           bool include_labels) = 0;
  virtual void validate_file_extension() = 0;
  virtual void delete_samples(const std::vector<uint64_t>& indices) = 0;
  virtual void set_file_path(const std::string& path) = 0;
  virtual FileWrapperType get_type() = 0;
  static FileWrapperType get_file_wrapper_type(const std::string& type) {
    static const std::unordered_map<std::string, FileWrapperType> FILE_WRAPPER_TYPE_MAP = {
        {"SingleSampleFileWrapper", FileWrapperType::SINGLE_SAMPLE},
        {"BinaryFileWrapper", FileWrapperType::BINARY},
        {"CsvFileWrapper", FileWrapperType::CSV}};
    return FILE_WRAPPER_TYPE_MAP.at(type);
  }
  virtual ~FileWrapper() = default;
  FileWrapper(const FileWrapper&) = default;
  FileWrapper& operator=(const FileWrapper&) = default;
  FileWrapper(FileWrapper&&) = default;
  FileWrapper& operator=(FileWrapper&&) = default;

 protected:
  std::string file_path_;
  YAML::Node file_wrapper_config_;
  std::shared_ptr<FilesystemWrapper> filesystem_wrapper_;
};
}  // namespace modyn::storage
