#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <fstream>
#include <numeric>
#include <vector>

using namespace storage;

int64_t BinaryFileWrapper::int_from_bytes(const unsigned char* begin, const unsigned char* end) {
  int64_t value = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  value = std::accumulate(begin, end, 0, [](uint64_t acc, unsigned char other) { return (acc << 8u) | other; });
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  value = std::accumulate(begin, end, 0, [](uint64_t acc, unsigned char other) { return (acc << 8u) | other; });
#else
#error "Unknown byte order"
#endif
  return value;
}

int64_t BinaryFileWrapper::get_number_of_samples() { return file_size_ / record_size_; }

void BinaryFileWrapper::validate_file_extension() {
  const std::string extension = file_path_.substr(file_path_.find_last_of('.') + 1);
  if (extension != "bin") {
    throw std::invalid_argument("Binary file wrapper only supports .bin files.");
  }
}

int64_t BinaryFileWrapper::get_label(int64_t index) {
  const int64_t record_start = index * record_size_;
  std::vector<unsigned char> data_vec = filesystem_wrapper_->get(file_path_);
  unsigned char* data = data_vec.data();
  unsigned char* label_begin = data + record_start;
  unsigned char* label_end = label_begin + label_size_;
  return int_from_bytes(label_begin, label_end);
}

std::vector<int64_t> BinaryFileWrapper::get_all_labels() {
  const int64_t num_samples = get_number_of_samples();
  std::vector<int64_t> labels = std::vector<int64_t>();
  labels.reserve(num_samples);
  std::vector<unsigned char> data_vec = filesystem_wrapper_->get(file_path_);
  unsigned char* data = data_vec.data();
  for (int64_t i = 0; i < num_samples; i++) {
    unsigned char* label_begin = data + (i * record_size_);
    unsigned char* label_end = label_begin + label_size_;
    labels.push_back(int_from_bytes(label_begin, label_end));
  }
  return labels;
}

std::vector<std::vector<unsigned char>> BinaryFileWrapper::get_samples(int64_t start, int64_t end) {
  const std::vector<int64_t> indices = {start, end};
  BinaryFileWrapper::validate_request_indices(get_number_of_samples(), indices);
  const int64_t num_samples = end - start + 1;
  const int64_t record_start = start * record_size_;
  const int64_t record_end = record_start + num_samples * record_size_;
  std::vector<unsigned char> data_vec = filesystem_wrapper_->get(file_path_);
  unsigned char* data = data_vec.data();
  std::vector<std::vector<unsigned char>> samples = std::vector<std::vector<unsigned char>>(num_samples);
  for (int64_t i = record_start; i < record_end; i += record_size_) {
    unsigned char* sample_begin = data + i + label_size_;
    unsigned char* sample_end = sample_begin + sample_size_;
    const std::vector<unsigned char> sample(sample_begin, sample_end);
    samples[(i - record_start) / record_size_] = sample;
  }
  return samples;
}

std::vector<unsigned char> BinaryFileWrapper::get_sample(int64_t index) {
  const std::vector<int64_t> indices = {index};
  BinaryFileWrapper::validate_request_indices(get_number_of_samples(), indices);
  const int64_t record_start = index * record_size_;
  std::vector<unsigned char> data_vec = filesystem_wrapper_->get(file_path_);
  unsigned char* data = data_vec.data();
  unsigned char* sample_begin = data + record_start + label_size_;
  unsigned char* sample_end = sample_begin + sample_size_;
  return {sample_begin, sample_end};
}

std::vector<std::vector<unsigned char>> BinaryFileWrapper::get_samples_from_indices(
    const std::vector<int64_t>& indices) {
  BinaryFileWrapper::validate_request_indices(get_number_of_samples(), indices);
  std::vector<std::vector<unsigned char>> samples = std::vector<std::vector<unsigned char>>();
  samples.reserve(indices.size());
  std::vector<unsigned char> data_vec = filesystem_wrapper_->get(file_path_);
  unsigned char* data = data_vec.data();
  for (const int64_t index : indices) {
    const int64_t record_start = index * record_size_;
    unsigned char* sample_begin = data + record_start + label_size_;
    unsigned char* sample_end = sample_begin + sample_size_;
    const std::vector<unsigned char> sample(sample_begin, sample_end);
    samples.push_back(sample);
  }
  return samples;
}
