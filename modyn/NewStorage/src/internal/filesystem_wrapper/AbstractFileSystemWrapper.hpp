#ifndef ABSTRACT_FILESYSTEM_WRAPPER_H
#define ABSTRACT_FILESYSTEM_WRAPPER_H

#include <string>
#include <vector>

namespace storage {
class AbstractFilesystemWrapper {
protected:
  std::string base_path;

public:
  AbstractFilesystemWrapper(std::string base_path) {
    this->base_path = base_path;
  }
  virtual std::vector<unsigned char> *get(std::string path) = 0;
  virtual bool exists(std::string path) = 0;
  virtual std::vector<std::string> *list(std::string path,
                                         bool recursive = false) = 0;
  virtual bool is_directory(std::string path) = 0;
  virtual bool is_file(std::string path) = 0;
  virtual int get_file_size(std::string path) = 0;
  virtual int get_modified_time(std::string path) = 0;
  virtual int get_created_time(std::string path) = 0;
  virtual std::string join(std::vector<std::string> paths) = 0;
  virtual bool is_valid_path(std::string path) = 0;
  virtual std::string get_name() = 0;
};
} // namespace storage

#endif