#include "trigger_sample_storage.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <vector>

#include "modyn/utils/utils.hpp"

// NOLINTBEGIN(modernize-avoid-c-arrays)
// We need to use these c-style arrays to interface with ctypes
namespace modyn::common::trigger_sample_storage {

/**
 * @brief Get the number of samples in file
 *
 * @param filename File to read
 * @return int Amount of samples
 */
int64_t get_num_samples_in_file_impl(const char* filename) {
  std::ifstream file = open_file(filename);
  read_magic(file);
  return read_data_size_from_header(file);
}

/**
 * @brief Get a specified amount of samples from matching files, starting at a specified index
 *
 * @param folder Directory to search in
 * @param size Return pointer for array size
 * @param pattern Pattern to match. Should match at start of filename
 * @param start_index Index of the first sample to store
 * @param worker_subset_size Total amount of samples to store
 * @return void* Array of samples
 */
void* get_worker_samples_impl(const char* folder, int64_t* size, const char* pattern, const int64_t start_index,
                              const int64_t worker_subset_size) {
  const std::vector<std::string> matching_files = get_matching_files(folder, pattern);
  int64_t samples = 0;
  int64_t current_index = 0;

  std::vector<char> char_vector;

  for (const std::string& filename : matching_files) {
    if (current_index >= start_index + worker_subset_size) {
      // We have already retrieved all the samples for the worker
      break;
    }
    const int64_t num_samples = get_num_samples_in_file_impl(filename.c_str());
    if (current_index + num_samples <= start_index) {
      // The samples in the file are before the samples for the worker
      current_index += num_samples;
      continue;
    }
    const int64_t start = (start_index >= current_index) ? start_index - current_index : 0;
    if (current_index + num_samples < start_index + worker_subset_size) {
      // The head of samples for the worker are in the file, either partially from
      // start_index - current_index to the end of the file if start_index> current_index
      // or completely from 0 to the end of the file.
      // Because the end index is exclusive, we compare < instead of <= otherwise we would retrieve
      // one more sample than we should
      parse_file_subset(filename.c_str(), char_vector, samples, start, num_samples);
      samples += num_samples - start;
      current_index += num_samples;
      continue;
    }

    parse_file_subset(filename.c_str(), char_vector, samples, start, start_index + worker_subset_size - current_index);
    samples += start_index + worker_subset_size - current_index - start;
    break;
  }

  const uint64_t data_length = sizeof(char) * dtype_size * samples;
  void* data = malloc(data_length);  // NOLINT: required for ctypes
  ASSERT(data != nullptr, "Could not allocate memory!");

  std::memcpy(static_cast<char*>(data), char_vector.data(), data_length);

  *size = samples;
  return data;
}

/**
 * @brief Get all the samples from matching files
 *
 * @param folder Directory to search in
 * @param size Return pointer for array size
 * @param pattern Pattern to match. Should match at start of filename
 * @return void* Array of samples
 */
void* get_all_samples_impl(const char* folder, int64_t* size, const char* pattern) {
  const std::vector<std::string> matching_files = get_matching_files(folder, pattern);
  std::vector<char> char_vector;

  int64_t samples = 0;

  for (const std::string& filename : matching_files) {
    std::ifstream file = open_file(filename.c_str());
    read_magic(file);
    const int64_t samples_in_file = read_data_size_from_header(file);

    char_vector.resize(static_cast<uint64_t>(dtype_size * (samples + samples_in_file)));
    file.read(char_vector.data() + dtype_size * samples, dtype_size * samples_in_file);
    samples += samples_in_file;

    file.close();
  }

  const uint64_t data_length = sizeof(char) * dtype_size * samples;
  void* data = malloc(data_length);  // NOLINT: required for ctypes

  std::memcpy(static_cast<char*>(data), char_vector.data(), data_length);

  *size = samples;
  return data;
}

/**
 * @brief Read samples from file
 *
 * @param filename File to read from
 * @param size Return pointer for array size
 * @return void* Array of samples
 */
void* parse_file_impl(const char* filename, int64_t* size) {
  std::ifstream file = open_file(filename);
  read_magic(file);
  const int64_t samples = read_data_size_from_header(file);

  *size = samples;

  const uint64_t data_length = sizeof(char) * dtype_size * samples;
  void* data = malloc(data_length);  // NOLINT: required for ctypes
  ASSERT(data != nullptr, "Could not allocate memory");

  file.read(static_cast<char*>(data), static_cast<int64_t>(data_length));
  file.close();

  return data;
}

/**
 * @brief Write samples to multiple files using async
 *
 * @param filenames Files to write to
 * @param data Array of samples to write
 * @param data_lengths Amount of samples to write to each file
 * @param headers Headers for the files
 * @param header_length Length of the headers
 * @param num_files Amount of files
 */
void write_files_impl(const char* filenames[], const void* data, int64_t data_lengths[], const char* headers[],
                      int64_t header_length, uint64_t num_files) {
  std::vector<std::future<void>> futures;
  int64_t data_offset = 0;

  for (uint64_t i = 0; i < num_files; ++i) {
    futures.push_back(std::async(std::launch::async, write_file, filenames[i], data, data_offset, data_lengths[i],
                                 headers[i], header_length));
    data_offset += data_lengths[i];
  }

  for (auto& future : futures) {
    future.get();
  }
}

/**
 * @brief Release memory of array
 *
 * @param data Array to free the memory of
 */
void release_data_impl(void* data) { std::free(data); }  // NOLINT: required for ctypes

/**
 * @brief Write samples to file
 *
 * @param filename File to write to
 * @param data Array of samples to write
 * @param data_offset Offset in array to write
 * @param data_length Length of the array
 * @param header File header to write
 * @param header_length Length of the header
 */
void write_file(const char* filename, const void* data, int64_t data_offset, const int64_t data_length,
                const char* header, const int64_t header_length) {
  std::ofstream file = open_file_write(filename);

  file.write(header, header_length);
  file.write(static_cast<const char*>(data) + dtype_size * data_offset, dtype_size * data_length);

  file.close();
}

/**
 * @brief Read subset of samples from file
 *
 * @param filename File to read from
 * @param char_vector Vector to append to
 * @param samples Amount of samples in vector
 * @param start_index Start index of new samples
 * @param end_index End index of new samples
 * @return true File read succesfully
 * @return false end_index exceeds samples of file
 */
bool parse_file_subset(const char* filename, std::vector<char>& char_vector, const int64_t samples,
                       const int64_t start_index, const int64_t end_index) {
  std::ifstream file = open_file(filename);
  read_magic(file);
  const int64_t samples_in_file = read_data_size_from_header(file);

  if (end_index > samples_in_file) {
    return false;
  }

  const int64_t offset = static_cast<int64_t>(start_index) * dtype_size;
  const int64_t num_bytes = (end_index - start_index) * dtype_size;

  file.seekg(offset, std::ios::cur);
  char_vector.resize(dtype_size * (samples + num_bytes));
  file.read(char_vector.data() + dtype_size * samples, dtype_size * num_bytes);

  file.close();
  return true;
}

/**
 * @brief Retrieve all matching filenames in a directory
 *
 * @param folder Directory to search in
 * @param pattern Pattern to match. Should match at start of filename
 * @return std::vector<std::string> Vector of matching filenames
 */
std::vector<std::string> get_matching_files(const char* folder, const char* pattern) {
  std::vector<std::string> matching_files;

  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    if (std::filesystem::is_regular_file(entry)) {
      const std::string filename = entry.path().filename().string();
      if (filename.starts_with(pattern)) {
        matching_files.push_back(std::string(folder) + '/' + filename);
      }
    }
  }

  std::sort(matching_files.begin(), matching_files.end());

  return matching_files;
}

/**
 * @brief Read array file header. Should be called after reading magic bytes.
 *
 * @param file File to read from
 * @return int64_t Data size
 */
int64_t read_data_size_from_header(std::ifstream& file) {
  std::array<char, 2> header_chars = {};
  file.read(header_chars.data(), 2);
  // double cast required by clang-tidy
  auto header_length = static_cast<uint64_t>(static_cast<unsigned char>(header_chars[1]));
  header_length <<= 8u;
  header_length += header_chars[0];

  std::string buffer(header_length, ' ');
  file.read(buffer.data(), static_cast<int64_t>(header_length));

  // Find the location of the shape and convert to int64_t
  // We search for `(` to skip the rest of the header, which we don't use.
  // We convert the characters base 10 into a number.
  return std::strtol(&buffer[buffer.find_last_of('(') + 1], nullptr, 10);
}

/**
 * @brief Read the magic NumPy bytes
 *
 * @param file File to read from
 * @return int64_t Major NumPy version
 */
int64_t read_magic(std::ifstream& file) {
  const std::vector<unsigned char> magic_bytes = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  char byte = 0;
  for (const unsigned char magic_byte : magic_bytes) {
    if (!file.get(byte) || byte != static_cast<char>(magic_byte)) {
      FAIL("Trigger Sample Storage tried to open a non-valid NumPy file.");
      return -1;
    }
  }

  file.get(byte);
  // double cast required by clang-tidy
  const auto major_version = static_cast<int64_t>(static_cast<unsigned char>(byte));
  file.get(byte);  // minor version is ignored

  return major_version;
}

/**
 * @brief Open a file for reading bytes
 *
 * @param filename File to open
 * @return std::ifstream Opened file stream
 */
std::ifstream open_file(const char* filename) {
  const std::filesystem::path file_path(filename);
  std::ifstream file(file_path, std::ios::binary);
  ASSERT(file.is_open(), "Trigger Sample Storage failed to open a file for reading.");

  return file;
}

/**
 * @brief Open a file for writing bytes
 *
 * @param filename File to open
 * @return std::ofstream Opened file stream
 */
std::ofstream open_file_write(const char* filename) {
  const std::filesystem::path file_path(filename);
  std::ofstream file(file_path, std::ios::binary);
  ASSERT(file.is_open(), "Trigger Sample Storage failed to open a file for writing.");

  return file;
}
}  // namespace modyn::common::trigger_sample_storage

// NOLINTEND(modernize-avoid-c-arrays)
