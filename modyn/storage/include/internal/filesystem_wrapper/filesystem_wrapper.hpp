#pragma once

#include <spdlog/spdlog.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace modyn::storage {

enum FilesystemWrapperType { INVALID_FSW, LOCAL };

class FilesystemWrapper {
 public:
  FilesystemWrapper() = default;
  virtual std::vector<unsigned char> get(const std::string& path) = 0;
  virtual bool exists(const std::string& path) = 0;
  virtual std::vector<std::string> list(const std::string& path, bool recursive, std::string extension) = 0;
  virtual bool is_directory(const std::string& path) = 0;
  virtual bool is_file(const std::string& path) = 0;
  virtual uint64_t get_file_size(const std::string& path) = 0;
  virtual int64_t get_modified_time(const std::string& path) = 0;
  virtual bool is_valid_path(const std::string& path) = 0;
  virtual std::shared_ptr<std::ifstream> get_stream(const std::string& path) = 0;
  virtual FilesystemWrapperType get_type() = 0;
  virtual bool remove(const std::string& path) = 0;
  static FilesystemWrapperType get_filesystem_wrapper_type(const std::string& type) {
    static const std::unordered_map<std::string, FilesystemWrapperType> FILESYSTEM_WRAPPER_TYPE_MAP = {
        {"LocalFilesystemWrapper", FilesystemWrapperType::LOCAL},
    };
    return FILESYSTEM_WRAPPER_TYPE_MAP.at(type);
  }
  virtual ~FilesystemWrapper() = default;
  FilesystemWrapper(const FilesystemWrapper&) = default;
  FilesystemWrapper& operator=(const FilesystemWrapper&) = default;
  FilesystemWrapper(FilesystemWrapper&&) = default;
  FilesystemWrapper& operator=(FilesystemWrapper&&) = default;
};
}  // namespace modyn::storage
