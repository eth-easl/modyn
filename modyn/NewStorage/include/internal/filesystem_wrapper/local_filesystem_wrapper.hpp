#pragma once

#include "internal/filesystem_wrapper/abstract_filesystem_wrapper.hpp"

namespace storage {
class LocalFilesystemWrapper : public AbstractFilesystemWrapper {  // NOLINT
 public:
  explicit LocalFilesystemWrapper(const std::string &path) : AbstractFilesystemWrapper(path) {}
  std::vector<unsigned char>* get(std::string path) override;
  bool exists(std::string path) override;
  std::vector<std::string>* list(std::string path, bool recursive) override;  // NOLINT
  bool is_directory(std::string path) override;
  bool is_file(std::string path) override;
  int64_t get_file_size(std::string path) override;
  int64_t get_modified_time(std::string path) override;
  int64_t get_created_time(std::string path) override;
  std::string join(std::vector<std::string> paths) override;
  bool is_valid_path(std::string path) override;
  std::string get_name() final { return "LOCAL"; }
  ~LocalFilesystemWrapper() override = default;
};
}  // namespace storage
