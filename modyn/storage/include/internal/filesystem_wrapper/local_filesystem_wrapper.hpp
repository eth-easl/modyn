#pragma once

#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"

namespace modyn::storage {
class LocalFilesystemWrapper : public FilesystemWrapper {
 public:
  LocalFilesystemWrapper() = default;
  std::vector<unsigned char> get(const std::string& path) override;
  bool exists(const std::string& path) override;
  std::vector<std::string> list(const std::string& path, bool recursive) override;
  bool is_directory(const std::string& path) override;
  bool is_file(const std::string& path) override;
  uint64_t get_file_size(const std::string& path) override;
  int64_t get_modified_time(const std::string& path) override;
  bool is_valid_path(const std::string& path) override;
  std::ifstream& get_stream(const std::string& path) override;
  FilesystemWrapperType get_type() override;
  bool remove(const std::string& path) override;
};
}  // namespace modyn::storage
