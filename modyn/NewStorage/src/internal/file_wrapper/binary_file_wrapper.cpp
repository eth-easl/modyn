#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <fstream>
#include <numeric>
#include <vector>

using namespace storage;

int BinaryFileWrapper::int_from_bytes(unsigned char* begin, unsigned char* end) {
  int value = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  value = std::accumulate(begin, end, 0, [](int acc, unsigned char x) { return (acc << 8) | x; });
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  value = std::accumulate(begin, end, 0, [](int acc, unsigned char x) { return (acc << 8) | x; });
#else
#error "Unknown byte order"
#endif
  return value;
}

int BinaryFileWrapper::get_number_of_samples() { return this->file_size / this->record_size; }

void BinaryFileWrapper::validate_file_extension() {
  std::string extension = this->file_path.substr(this->file_path.find_last_of(".") + 1);
  if (extension != "bin") {
    throw std::invalid_argument("Binary file wrapper only supports .bin files.");
  }
}

void BinaryFileWrapper::validate_request_indices(int total_samples, std::vector<int>* indices) {
  for (unsigned long i = 0; i < indices->size(); i++) {
    if (indices->at(i) < 0 || indices->at(i) > (total_samples - 1)) {
      throw std::runtime_error("Requested index is out of bounds.");
    }
  }
}

int BinaryFileWrapper::get_label(int index) {
  int record_start = index * this->record_size;
  unsigned char* data = this->filesystem_wrapper->get(this->file_path)->data();
  unsigned char* label_begin = data + record_start;
  unsigned char* label_end = label_begin + this->label_size;
  return int_from_bytes(label_begin, label_end);
}

std::vector<int>* BinaryFileWrapper::get_all_labels() {
  int num_samples = this->get_number_of_samples();
  std::vector<int>* labels = new std::vector<int>();
  labels->reserve(num_samples);
  unsigned char* data = this->filesystem_wrapper->get(this->file_path)->data();
  for (int i = 0; i < num_samples; i++) {
    unsigned char* label_begin = data + (i * this->record_size);
    unsigned char* label_end = label_begin + this->label_size;
    int label = int_from_bytes(label_begin, label_end);
    labels->push_back(label);
  }
  return labels;
}

std::vector<std::vector<unsigned char>>* BinaryFileWrapper::get_samples(int start, int end) {
  std::vector<int> indices = {start, end};
  this->validate_request_indices(this->get_number_of_samples(), &indices);
  int num_samples = end - start;
  int record_start = start * this->record_size;
  int record_end = end * this->record_size;
  unsigned char* data = this->filesystem_wrapper->get(this->file_path)->data();
  std::vector<std::vector<unsigned char>>* samples = new std::vector<std::vector<unsigned char>>;
  samples->reserve(num_samples);
  for (int i = record_start; i < record_end; i += this->record_size) {
    unsigned char* sample_begin = data + i + this->label_size;
    unsigned char* sample_end = sample_begin + this->sample_size;
    std::vector<unsigned char> sample(sample_begin, sample_end);
    samples->push_back(sample);
  }
  return samples;
}

std::vector<unsigned char>* BinaryFileWrapper::get_sample(int index) {
  std::vector<int> indices = {index};
  this->validate_request_indices(this->get_number_of_samples(), &indices);
  int record_start = index * this->record_size;
  unsigned char* data = this->filesystem_wrapper->get(this->file_path)->data();
  unsigned char* sample_begin = data + record_start + this->label_size;
  unsigned char* sample_end = sample_begin + this->sample_size;
  std::vector<unsigned char>* sample = new std::vector<unsigned char>(sample_begin, sample_end);
  return sample;
}

std::vector<std::vector<unsigned char>>* BinaryFileWrapper::get_samples_from_indices(std::vector<int>* indices) {
  this->validate_request_indices(this->get_number_of_samples(), indices);
  std::vector<std::vector<unsigned char>>* samples = new std::vector<std::vector<unsigned char>>;
  samples->reserve(indices->size());
  unsigned char* data = this->filesystem_wrapper->get(this->file_path)->data();
  for (unsigned long i = 0; i < indices->size(); i++) {
    int index = indices->at(i);
    int record_start = index * this->record_size;
    unsigned char* sample_begin = data + record_start + this->label_size;
    unsigned char* sample_end = sample_begin + this->sample_size;
    std::vector<unsigned char> sample(sample_begin, sample_end);
    samples->push_back(sample);
  }
  return samples;
}
