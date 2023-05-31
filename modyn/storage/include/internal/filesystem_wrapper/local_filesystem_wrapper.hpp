#pragma once

#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"

namespace storage {
class LocalFilesystemWrapper : public FilesystemWrapper {  // NOLINT
 public:
  explicit LocalFilesystemWrapper(const std::string& path) : FilesystemWrapper(path) {}
  std::vector<unsigned char> get(const std::string& path) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list(const std::string& path, bool recursive) override;  // NOLINT
  bool is_directory(const std::string& path) override;
  bool is_file(const std::string& path) override;
  int64_t get_file_size(const std::string& path) override;
  int64_t get_modified_time(const std::string& path) override;
  std::string join(const std::vector<std::string>& paths) override;
  bool is_valid_path(const std::string& path) override;
  FilesystemWrapperType get_type() final { return FilesystemWrapperType::LOCAL; }
  bool remove(const std::string& path) override;
  ~LocalFilesystemWrapper() override = default;
};
}  // namespace storage
