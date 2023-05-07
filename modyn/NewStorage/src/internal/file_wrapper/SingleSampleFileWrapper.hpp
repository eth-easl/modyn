#ifndef SINGLE_SAMPLE_FILE_WRAPPER_HPP
#define SINGLE_SAMPLE_FILE_WRAPPER_HPP

#include "AbstractFileWrapper.hpp"
#include <cstddef>

namespace storage {
class SingleSampleFileWrapper : public AbstractFileWrapper {
public:
  SingleSampleFileWrapper(std::string path, YAML::Node file_wrapper_config,
                          AbstractFilesystemWrapper *filesystem_wrapper)
      : AbstractFileWrapper(path, file_wrapper_config, filesystem_wrapper) {}
  int get_number_of_samples();
  int get_label(int index);
  std::vector<int> *get_all_labels();
  std::vector<std::vector<unsigned char>> *get_samples(int start, int end);
  std::vector<unsigned char> *get_sample(int index);
  std::vector<std::vector<unsigned char>> *
  get_samples_from_indices(std::vector<int> *indices);
};
} // namespace storage

#endif