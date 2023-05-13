#pragma once

#include "internal/filesystem_wrapper/abstract_filesystem_wrapper.hpp"

namespace storage {
class LocalFilesystemWrapper : public AbstractFilesystemWrapper {
 public:
  LocalFilesystemWrapper(std::string base_path) : AbstractFilesystemWrapper(base_path) {}
  std::vector<unsigned char>* get(std::string path);
  bool exists(std::string path);
  std::vector<std::string>* list(std::string path, bool recursive = false);
  bool is_directory(std::string path);
  bool is_file(std::string path);
  int get_file_size(std::string path);
  int get_modified_time(std::string path);
  int get_created_time(std::string path);
  std::string join(std::vector<std::string> paths);
  bool is_valid_path(std::string path);
  std::string get_name() { return "LOCAL"; }
};
}  // namespace storage
