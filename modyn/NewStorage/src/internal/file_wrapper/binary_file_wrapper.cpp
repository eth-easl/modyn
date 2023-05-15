#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <fstream>
#include <numeric>
#include <vector>

using namespace storage;

int BinaryFileWrapper::int_from_bytes(const unsigned char* begin, const unsigned char* end) {
  int value = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  value = std::accumulate(begin, end, 0, [](int acc, unsigned char other) {
    return (static_cast<unsigned int>(acc) << 8) | other;  // NOLINT
  });
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  value = std::accumulate(begin, end, 0, [](int acc, unsigned char other) {
    return (static_cast<unsigned int>(acc) << 8) | other;  // NOLINT
  });
#else
#error "Unknown byte order"
#endif
  return value;
}

int BinaryFileWrapper::get_number_of_samples() { return this->file_size_ / this->record_size_; }

void BinaryFileWrapper::validate_file_extension() {
  const std::string extension = this->file_path_.substr(this->file_path_.find_last_of('.') + 1);
  if (extension != "bin") {
    throw std::invalid_argument("Binary file wrapper only supports .bin files.");
  }
}

int BinaryFileWrapper::get_label(int index) {
  const int record_start = index * this->record_size_;
  unsigned char* data = this->filesystem_wrapper_->get(this->file_path_)->data();
  unsigned char* label_begin = data + record_start;
  unsigned char* label_end = label_begin + this->label_size_;
  return int_from_bytes(label_begin, label_end);
}

std::vector<int>* BinaryFileWrapper::get_all_labels() {
  const int64_t num_samples = this->get_number_of_samples();
  auto* labels = new std::vector<int>();
  labels->reserve(num_samples);
  unsigned char* data = this->filesystem_wrapper_->get(this->file_path_)->data();
  for (int64_t i = 0; i < num_samples; i++) {
    unsigned char* label_begin = data + (i * this->record_size_);
    unsigned char* label_end = label_begin + this->label_size_;
    const int label = int_from_bytes(label_begin, label_end);
    labels->push_back(label);
  }
  return labels;
}

std::vector<std::vector<unsigned char>>* BinaryFileWrapper::get_samples(int64_t start, int64_t end) {
  const std::vector<int64_t> indices = {start, end};
  BinaryFileWrapper::validate_request_indices(this->get_number_of_samples(), &indices);
  const int64_t num_samples = end - start;
  const int64_t record_start = start * this->record_size_;
  const int64_t record_end = end * this->record_size_;
  unsigned char* data = this->filesystem_wrapper_->get(this->file_path_)->data();
  auto* samples = new std::vector<std::vector<unsigned char>>;
  samples->reserve(num_samples);
  for (int64_t i = record_start; i < record_end; i += this->record_size_) {
    unsigned char* sample_begin = data + i + this->label_size_;
    unsigned char* sample_end = sample_begin + this->sample_size_;
    const std::vector<unsigned char> sample(sample_begin, sample_end);
    samples->push_back(sample);
  }
  return samples;
}

std::vector<unsigned char>* BinaryFileWrapper::get_sample(int64_t index) {
  const std::vector<int64_t> indices = {index};
  BinaryFileWrapper::validate_request_indices(this->get_number_of_samples(), &indices);
  const int64_t record_start = index * this->record_size_;
  unsigned char* data = this->filesystem_wrapper_->get(this->file_path_)->data();
  unsigned char* sample_begin = data + record_start + this->label_size_;
  unsigned char* sample_end = sample_begin + this->sample_size_;
  auto* sample = new std::vector<unsigned char>(sample_begin, sample_end);
  return sample;
}

std::vector<std::vector<unsigned char>>* BinaryFileWrapper::get_samples_from_indices(std::vector<int64_t>* indices) {
  BinaryFileWrapper::validate_request_indices(this->get_number_of_samples(), indices);
  auto* samples = new std::vector<std::vector<unsigned char>>;
  samples->reserve(indices->size());
  unsigned char* data = this->filesystem_wrapper_->get(this->file_path_)->data();
  for (const int64_t index : *indices) {
    const int64_t record_start = index * this->record_size_;
    unsigned char* sample_begin = data + record_start + this->label_size_;
    unsigned char* sample_end = sample_begin + this->sample_size_;
    const std::vector<unsigned char> sample(sample_begin, sample_end);
    samples->push_back(sample);
  }
  return samples;
}
