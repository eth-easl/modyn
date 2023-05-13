#pragma once

#include <cstddef>
#include <iostream>

#include "internal/file_wrapper/abstract_file_wrapper.hpp"

namespace storage {
class BinaryFileWrapper : public AbstractFileWrapper {
 private:
  int record_size;
  int label_size;
  int file_size;
  int sample_size;
  void validate_request_indices(int total_samples, std::vector<int>* indices);
  int int_from_bytes(unsigned char* begin, unsigned char* end);

 public:
  BinaryFileWrapper(std::string path, YAML::Node fw_config, AbstractFilesystemWrapper* fs_wrapper)
      : AbstractFileWrapper(path, fw_config, fs_wrapper) {
    if (!fw_config["record_size"]) {
      throw std::runtime_error("record_size must be specified in the file wrapper config.");
    }
    this->record_size = fw_config["record_size"].as<int>();
    if (!fw_config["label_size"]) {
      throw std::runtime_error("label_size must be specified in the file wrapper config.");
    }
    this->label_size = fw_config["label_size"].as<int>();
    this->sample_size = this->record_size - this->label_size;

    if (this->record_size - this->label_size < 1) {
      throw std::runtime_error(
          "Each record must have at least 1 byte of data "
          "other than the label.");
    }

    this->validate_file_extension();
    this->file_size = fs_wrapper->get_file_size(path);

    if (this->file_size % this->record_size != 0) {
      throw std::runtime_error("File size must be a multiple of the record size.");
    }
  }
  int get_number_of_samples();
  int get_label(int index);
  std::vector<int>* get_all_labels();
  std::vector<std::vector<unsigned char>>* get_samples(int start, int end);
  std::vector<unsigned char>* get_sample(int index);
  std::vector<std::vector<unsigned char>>* get_samples_from_indices(std::vector<int>* indices);
  void validate_file_extension();
  std::string get_name() { return "BIN"; }
  ~BinaryFileWrapper() {}
};
}  // namespace storage
