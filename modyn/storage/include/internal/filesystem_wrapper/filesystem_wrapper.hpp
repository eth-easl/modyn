#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace storage {

enum FilesystemWrapperType { LOCAL };

class FilesystemWrapper {  // NOLINT
 protected:
  std::string base_path_;

 public:
  explicit FilesystemWrapper(std::string path) : base_path_{std::move(path)} {}
  virtual std::vector<unsigned char> get(const std::string& path) = 0;
  virtual bool exists(const std::string& path) = 0;
  virtual std::vector<std::string> list(const std::string& path, bool recursive) = 0;
  virtual bool is_directory(const std::string& path) = 0;
  virtual bool is_file(const std::string& path) = 0;
  virtual int64_t get_file_size(const std::string& path) = 0;
  virtual int64_t get_modified_time(const std::string& path) = 0;
  virtual std::string join(const std::vector<std::string>& paths) = 0;
  virtual bool is_valid_path(const std::string& path) = 0;
  virtual FilesystemWrapperType get_type() = 0;
  virtual bool remove(const std::string& path) = 0;
  static FilesystemWrapperType get_filesystem_wrapper_type(const std::string& type) {
    static const std::unordered_map<std::string, FilesystemWrapperType> FILESYSTEM_WRAPPER_TYPE_MAP = {
        {"local", FilesystemWrapperType::LOCAL},
    };
    return FILESYSTEM_WRAPPER_TYPE_MAP.at(type);
  }
  virtual ~FilesystemWrapper() {}  // NOLINT
};
}  // namespace storage
