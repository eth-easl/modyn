#include "trigger_storage_cpp.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "modyn/utils/utils.hpp"

namespace modyn::common::trigger_storage_cpp {

const int DTYPE_SIZE = 16;

/**
 * @brief Get a specified amount of samples from matching files, starting at a specified index
 *
 * @param folder Directory to search in
 * @param pattern Pattern to match. Should match at start of filename
 * @param array Array to store the samples in
 * @param start_index Index of the first sample to store
 * @param worker_subset_size Total amount of samples to store
 * @return uint64_t Amount of samples
 */
uint64_t get_worker_samples_impl(const char* folder, const char* pattern, void* array, const uint64_t start_index,
                                 const std::size_t worker_subset_size) {
  std::vector<std::string> matching_files = get_matching_files(folder, pattern);
  std::size_t array_offset = 0;
  std::size_t current_index = 0;

  for (const std::string& filename : matching_files) {
    if (current_index >= start_index + worker_subset_size) {
      // We have already retrieved all the samples for the worker
      break;
    }
    std::size_t num_samples = get_num_samples_in_file_impl(filename.c_str());
    if (current_index + num_samples <= start_index) {
      // The samples in the file are before the samples for the worker
      current_index += num_samples;
      continue;
    }
    std::size_t start = (start_index >= current_index) ? start_index - current_index : 0;
    if (current_index + num_samples < start_index + worker_subset_size) {
      // The head of samples for the worker are in the file, either partially from
      // start_index - current_index to the end of the file if start_index> current_index
      //  or completely from 0 to the end of the file.
      // Because the end index is exclusive, we compare < instead of <= otherwise we would retrieve
      // one more sample than we should
      parse_file_subset_impl(filename.c_str(), array, array_offset, start, num_samples);
      array_offset += num_samples - start;
      current_index += num_samples;
      continue;
    }

    parse_file_subset_impl(filename.c_str(), array, array_offset, start,
                           start_index + worker_subset_size - current_index);
    array_offset += start_index + worker_subset_size - current_index - start;
    break;
  }
  return array_offset;
}

/**
 * @brief Get all the samples from matching files
 *
 * @param folder Directory to search in
 * @param pattern Pattern to match. Should match at start of filename
 * @param array Array to store the samples in
 * @return uint64_t Amount of samples
 */
uint64_t get_all_samples_impl(const char* folder, const char* pattern, void* array) {
  std::vector<std::string> matching_files = get_matching_files(folder, pattern);
  std::size_t array_offset = 0;

  for (const std::string& filename : matching_files) {
    array_offset += parse_file_impl(filename.c_str(), array, array_offset);
  }
  return array_offset;
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
      std::string filename = entry.path().filename().string();
      if (filename.rfind(pattern, 0) == 0) {
        matching_files.push_back(std::string(folder) + '/' + filename);
      }
    }
  }

  std::sort(matching_files.begin(), matching_files.end());

  return matching_files;
}

/**
 * @brief Get the number of samples in file
 *
 * @param filename File to read
 * @return uint64_t Amount of samples
 */
uint64_t get_num_samples_in_file_impl(const char* filename) {
  std::ifstream file = open_file(filename);
  read_magic(file);
  return read_data_size_from_header(file);
}

/**
 * @brief Read samples from file
 *
 * @param filename File to read from
 * @param array Array to write to
 * @return uint64_t Amount of samples read
 */
uint64_t parse_file_impl(const char* filename, void* array, const uint64_t array_offset) {
  std::ifstream file = open_file(filename);
  read_magic(file);
  std::size_t samples = read_data_size_from_header(file);

  file.read((char*)array + DTYPE_SIZE * array_offset, DTYPE_SIZE * samples);
  file.close();

  return samples;
}

/**
 * @brief Read subset of samples from file
 *
 * @param filename File to read from
 * @param array  Array to write to
 * @param start_index Start index of samples
 * @param end_index End index of samples
 * @return true File read succesfully
 * @return false end_index exceeds samples of file
 */
bool parse_file_subset_impl(const char* filename, void* array, const uint64_t array_offset, const uint64_t start_index,
                            const uint64_t end_index) {
  std::ifstream file = open_file(filename);
  read_magic(file);
  std::size_t samples = read_data_size_from_header(file);

  if (end_index > samples) {
    return false;
  }

  std::size_t offset = start_index * DTYPE_SIZE;
  std::size_t num_bytes = (end_index - start_index) * DTYPE_SIZE;

  file.seekg(offset, std::ios::cur);
  file.read((char*)array + DTYPE_SIZE * array_offset, DTYPE_SIZE * num_bytes);
  file.close();

  return true;
}

/**
 * @brief Write samples to file
 *
 * @param filename File to write to
 * @param array Samples to write
 * @param array_length Length of the array
 * @param header File header to write
 * @param header_length Length of the header
 */
void write_file_impl(const char* filename, void* array, const std::size_t data_length, const char* header,
                     const std::size_t header_length) {
  std::ofstream file = open_file_write(filename);

  file.write(header, header_length);
  file.write((char*)array, data_length);

  file.close();
}

/**
 * @brief Read array file header
 *
 * @param file File to read from
 * @return std::size_t Data size
 */
std::size_t read_data_size_from_header(std::ifstream& file) {
  char header_chars[2];
  file.read(header_chars, 2);
  int header_length = header_chars[1];
  header_length <<= 8;
  header_length += header_chars[0];

  std::string buffer(header_length, ' ');
  file.read(&buffer[0], header_length);

  // Find the location of the shape and convert to int
  return std::strtol(&buffer[buffer.find_last_of('(') + 1], nullptr, 10);
}

/**
 * @brief Read the magic NumPy bytes
 *
 * @param file File to read from
 * @return int Major NumPy version
 */
int read_magic(std::ifstream& file) {
  std::vector<unsigned char> magic_bytes = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  char byte;
  for (char magic_byte : magic_bytes) {
    if (!file.get(byte) || byte != magic_byte) {
      std::cerr << "Not a valid NumPy file." << std::endl;
      return -1;
    }
  }

  file.get(byte);
  int major_version = static_cast<int>(byte);
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
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
  }
  return file;
}

/**
 * @brief Open a file for writing bytes
 *
 * @param filename File to open
 * @return std::ofstream Opened file stream
 */
std::ofstream open_file_write(const char* filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
  }
  return file;
}
}  // namespace modyn::common::trigger_storage_cpp
