#pragma once

#include <string>
#include <vector>

namespace storage {
class AbstractFilesystemWrapper {  // NOLINT
 protected:
  std::string base_path_;

 public:
  explicit AbstractFilesystemWrapper(std::string path) : base_path_{std::move(path)} {}
  virtual std::vector<unsigned char> get(const std::string& path) = 0;
  virtual bool exists(const std::string& path) = 0;
  virtual std::vector<std::string> list(const std::string& path, bool recursive) = 0;
  virtual bool is_directory(const std::string& path) = 0;
  virtual bool is_file(const std::string& path) = 0;
  virtual int64_t get_file_size(const std::string& path) = 0;
  virtual int64_t get_modified_time(const std::string& path) = 0;
  virtual int64_t get_created_time(const std::string& path) = 0;
  virtual std::string join(const std::vector<std::string>& paths) = 0;
  virtual bool is_valid_path(const std::string& path) = 0;
  virtual std::string get_name() = 0;
  virtual ~AbstractFilesystemWrapper() {}  // NOLINT
};
}  // namespace storage
