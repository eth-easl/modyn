#include "internal/file_wrapper/binary_file_wrapper.hpp"

#include <fstream>
#include <numeric>
#include <vector>

using namespace modyn::storage;

/*
 * Transforms a vector of bytes into an int64_t.
 *
 * Handles both big and little endian machines.
 *
 * @param begin The beginning of the vector.
 * @param end The end of the vector.
 */
int64_t BinaryFileWrapper::int_from_bytes(const unsigned char* begin, const unsigned char* end) {
  int64_t value = 0;

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  value = std::accumulate(begin, end, 0LL, [](uint64_t acc, unsigned char byte) { return (acc << 8u) | byte; });
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  const std::reverse_iterator<const unsigned char*> rbegin(end);
  const std::reverse_iterator<const unsigned char*> rend(begin);
  value = std::accumulate(rbegin, rend, 0LL, [](uint64_t acc, unsigned char byte) { return (acc << 8u) | byte; });
#else
#error "Unknown byte order"
#endif
  return value;
}

int64_t BinaryFileWrapper::get_number_of_samples() { return file_size_ / record_size_; }

void BinaryFileWrapper::validate_file_extension() {
  const std::string extension = file_path_.substr(file_path_.find_last_of('.') + 1);
  if (extension != "bin") {
    SPDLOG_ERROR("Binary file wrapper only supports .bin files.");
  }
}

/*
 * Offset calculation to retrieve the label of a sample.
 *
 * @param index The index of the sample.
 */
int64_t BinaryFileWrapper::get_label(int64_t index) {
  ASSERT(index >= 0 && index < get_number_of_samples(), "Invalid index");

  const int64_t record_start = index * record_size_;
  std::ifstream& stream = filesystem_wrapper_->get_stream(file_path_);

  stream.seekg(record_start, std::ios::beg);

  std::vector<unsigned char> label_vec(label_size_);
  stream.read(reinterpret_cast<char*>(label_vec.data()), label_size_);

  return int_from_bytes(label_vec.data(), label_vec.data() + label_size_) - '0';
}

/*
 * Offset calculation to retrieve all the labels of a sample.
 */
std::vector<int64_t> BinaryFileWrapper::get_all_labels() {
  const int64_t num_samples = get_number_of_samples();
  std::vector<int64_t> labels = std::vector<int64_t>();
  labels.reserve(num_samples);

  std::ifstream& stream = filesystem_wrapper_->get_stream(file_path_);

  for (int64_t i = 0; i < num_samples; i++) {
    stream.seekg(i * record_size_, std::ios::beg);

    std::vector<unsigned char> label_vec(label_size_);
    stream.read(reinterpret_cast<char*>(label_vec.data()), label_size_);

    labels.push_back(int_from_bytes(label_vec.data(), label_vec.data() + label_size_) - '0');
  }

  return labels;
}

/*
 * Offset calculation to retrieve the data of a sample interval.
 *
 * @param start The start index of the sample interval.
 * @param end The end index of the sample interval.
 */
std::vector<std::vector<unsigned char>> BinaryFileWrapper::get_samples(int64_t start, int64_t end) {
  ASSERT(start >= 0 && end >= start && end <= get_number_of_samples(), "Invalid indices");

  const int64_t num_samples = end - start + 1;

  std::ifstream& stream = filesystem_wrapper_->get_stream(file_path_);

  std::vector<std::vector<unsigned char>> samples(num_samples);
  int64_t record_start;
  for (int64_t index = 0; index < num_samples; index++) {
    record_start = (start + index) * record_size_;
    stream.seekg(record_start + label_size_, std::ios::beg);

    std::vector<unsigned char> sample_vec(sample_size_);
    stream.read(reinterpret_cast<char*>(sample_vec.data()), sample_size_);

    samples[index] = sample_vec;
  }

  return samples;
}

/*
 * Offset calculation to retrieve the data of a sample.
 *
 * @param index The index of the sample.
 */
std::vector<unsigned char> BinaryFileWrapper::get_sample(int64_t index) {
  ASSERT(index >= 0 && index < get_number_of_samples(), "Invalid index");

  const int64_t record_start = index * record_size_;

  std::ifstream& stream = filesystem_wrapper_->get_stream(file_path_);

  stream.seekg(record_start + label_size_, std::ios::beg);

  std::vector<unsigned char> sample_vec(sample_size_);
  stream.read(reinterpret_cast<char*>(sample_vec.data()), sample_size_);

  return sample_vec;
}

/*
 * Offset calculation to retrieve the data of a sample interval.
 *
 * @param indices The indices of the sample interval.
 */
std::vector<std::vector<unsigned char>> BinaryFileWrapper::get_samples_from_indices(
    const std::vector<int64_t>& indices) {
  ASSERT(std::all_of(indices.begin(), indices.end(),
                     [&](int64_t index) { return index >= 0 && index < get_number_of_samples(); }),
         "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  samples.reserve(indices.size());

  std::ifstream& stream = filesystem_wrapper_->get_stream(file_path_);
  int64_t record_start = 0;
  for (const int64_t index : indices) {
    record_start = index * record_size_;

    stream.seekg(record_start + label_size_, std::ios::beg);

    std::vector<unsigned char> sample_vec(sample_size_);
    stream.read(reinterpret_cast<char*>(sample_vec.data()), sample_size_);

    samples.push_back(sample_vec);
  }

  return samples;
}

/*
 * Delete the samples at the given index list. The indices are zero based.
 *
 * We do not support deleting samples from binary files.
 * We can only delete the entire file which is done when every sample is deleted.
 * This is done to avoid the overhead of updating the file after every deletion.
 *
 * See DeleteData in the storage grpc servicer for more details.
 *
 * @param indices The indices of the samples to delete.
 */
void BinaryFileWrapper::delete_samples(const std::vector<int64_t>& /*indices*/) {}

/*
 * Set the file path of the file wrapper.
 *
 * @param path The new file path.
 */
void BinaryFileWrapper::set_file_path(const std::string& path) {
  file_path_ = path;
  file_size_ = filesystem_wrapper_->get_file_size(path);

  if (file_size_ % record_size_ != 0) {
    FAIL("File size must be a multiple of the record size.");
  }
}

FileWrapperType BinaryFileWrapper::get_type() { return FileWrapperType::BINARY; }