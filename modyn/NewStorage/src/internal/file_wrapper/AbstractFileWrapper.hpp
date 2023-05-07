#ifndef ABSTRACT_FILE_WRAPPER_HPP
#define ABSTRACT_FILE_WRAPPER_HPP

#include "../filesystem_wrapper/AbstractFilesystemWrapper.hpp"
#include <string>
#include <yaml-cpp/yaml.h>

namespace storage {
class AbstractFileWrapper {
protected:
  std::string path;
  YAML::Node file_wrapper_config;
  AbstractFilesystemWrapper *filesystem_wrapper;

public:
  AbstractFileWrapper(std::string path, YAML::Node file_wrapper_config,
                      AbstractFilesystemWrapper *filesystem_wrapper) {
    this->path = path;
    this->file_wrapper_config = file_wrapper_config;
    this->filesystem_wrapper = filesystem_wrapper;
  }
  virtual int get_number_of_samples() = 0;
  virtual std::vector<std::vector<unsigned char>> *get_samples(int start,
                                                               int end) = 0;
  virtual int get_label(int index) = 0;
  virtual std::vector<int> *get_all_labels() = 0;
  virtual std::vector<unsigned char> *get_sample(int index) = 0;
  virtual std::vector<std::vector<unsigned char>> *
  get_samples_from_indices(std::vector<int> *indices) = 0;
};
} // namespace storage

#endif