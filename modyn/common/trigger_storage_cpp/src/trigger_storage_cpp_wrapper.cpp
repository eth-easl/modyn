#include "trigger_storage_cpp_wrapper.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "trigger_storage_cpp.hpp"

extern "C" {
void* get_worker_samples(const char* folder, uint64_t* size, const char* pattern, const uint64_t start_index,
                         const size_t worker_subset_size) {
  return modyn::common::trigger_storage_cpp::get_worker_samples_impl(folder, size, pattern, start_index,
                                                                     worker_subset_size);
}

void* get_all_samples(const char* folder, uint64_t* size, const char* pattern) {
  return modyn::common::trigger_storage_cpp::get_all_samples_impl(folder, size, pattern);
}

uint64_t get_num_samples_in_file(const char* filename) {
  return modyn::common::trigger_storage_cpp::get_num_samples_in_file_impl(filename);
}

uint64_t parse_file(const char* filename, void* array, const uint64_t array_offset) {
  return modyn::common::trigger_storage_cpp::parse_file_impl(filename, array, array_offset);
}

void write_file(const char* filename, const void* array, size_t array_offset, const size_t array_length,
                const char* header, const size_t header_length) {
  return modyn::common::trigger_storage_cpp::write_file_impl(filename, array, array_offset, array_length, header,
                                                             header_length);
}
void write_files(const char* filenames[], const void* array, std::size_t data_lengths[], const char* headers[],
                 std::size_t header_length, std::size_t num_files) {
  return modyn::common::trigger_storage_cpp::write_files_impl(filenames, array, data_lengths, headers, header_length,
                                                              num_files);
}

void* parse_file(const char* filename, uint64_t* size) {
  return modyn::common::trigger_storage_cpp::parse_file_impl(filename, size);
}
void release_array(void* array) { return modyn::common::trigger_storage_cpp::release_array_impl(array); }
}
