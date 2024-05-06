#include "local_storage_backend.hpp"

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
namespace modyn::selector::local_storage_backend {

/**
 * @brief Read samples from multiple files using async
 *
 * @param filenames Files to read from
 * @param data Array of samples to write to
 * @param data_lengths Amount of samples in every file
 * @param data_offsets Amount of samples to skip in every file
 * @param num_files Number of files
 */
void parse_files_impl(const char* filenames[], void* data, int64_t data_lengths[], int64_t data_offsets[],
                      uint64_t num_files) {
  std::vector<std::future<void>> futures;
  int64_t current_offset = 0;

  for (uint64_t i = 0; i < num_files; ++i) {
    futures.push_back(std::async(std::launch::async, parse_file, filenames[i], data, current_offset, data_lengths[i],
                                 data_offsets[i]));
    current_offset += data_lengths[i];
  }

  for (auto& future : futures) {
    future.get();
  }
}

/**
 * @brief Read samples from file
 *
 * @param filename File to read from
 * @param data Array of samples to write to
 * @param data_offset Offset in array to write to
 * @param file_length Length of file
 * @param file_offset Amount of samples to skip in file
 */
void parse_file(const char* filename, void* data, int64_t data_offset, int64_t file_length, int64_t file_offset) {
  std::ifstream file = open_file(filename);

  file.seekg(dtype_size * file_offset, std::ios::cur);
  file.read(static_cast<char*>(data) + dtype_size * data_offset, dtype_size * file_length);
  file.close();
}

/**
 * @brief Write samples to multiple files using async
 *
 * @param filenames Files to write to
 * @param data Array of samples to write
 * @param data_lengths Amount of samples to write to each file
 * @param num_files Amount of files
 */
void write_files_impl(const char* filenames[], const void* data, int64_t data_lengths[], uint64_t num_files) {
  std::vector<std::future<void>> futures;
  int64_t data_offset = 0;

  for (uint64_t i = 0; i < num_files; ++i) {
    futures.push_back(std::async(std::launch::async, write_file, filenames[i], data, data_offset, data_lengths[i]));
    data_offset += data_lengths[i];
  }

  for (auto& future : futures) {
    future.get();
  }
}

/**
 * @brief Write samples to file
 *
 * @param filename File to write to
 * @param data Array of samples to write
 * @param data_offset Offset in array to write
 * @param data_length Length of the array
 */
void write_file(const char* filename, const void* data, int64_t data_offset, const int64_t data_length) {
  std::ofstream file = open_file_write(filename);

  file.write(static_cast<const char*>(data) + dtype_size * data_offset, dtype_size * data_length);

  file.close();
}

/**
 * @brief Open a file for reading bytes
 *
 * @param filename File to open
 * @return std::ifstream Opened file stream
 */
std::ifstream open_file(const char* filename) {
  const std::filesystem::path file_path = filename;
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    FAIL("Trigger Sample Storage failed to open a file for reading.");
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
  const std::filesystem::path file_path = filename;
  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    FAIL("Trigger Sample Storage failed to open a file for writing.");
  }
  return file;
}
}  // namespace modyn::selector::local_storage_backend

// NOLINTEND(modernize-avoid-c-arrays)
