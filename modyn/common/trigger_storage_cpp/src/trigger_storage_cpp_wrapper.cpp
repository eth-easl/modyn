#include "trigger_storage_cpp_wrapper.hpp"

#include <cstddef>
#include <cstdint>

#include "trigger_storage_cpp.hpp"

extern "C" {
uint64_t get_worker_samples(const char* folder, const char* pattern, void* array, const uint64_t start_index,
                            const size_t worker_subset_size) {
  return modyn::common::trigger_storage_cpp::get_worker_samples_impl(folder, pattern, array, start_index,
                                                                     worker_subset_size);
}

uint64_t get_all_samples(const char* folder, const char* pattern, void* array) {
  return modyn::common::trigger_storage_cpp::get_all_samples_impl(folder, pattern, array);
}

uint64_t get_num_samples_in_file(const char* filename) {
  return modyn::common::trigger_storage_cpp::get_num_samples_in_file_impl(filename);
}

uint64_t parse_file(const char* filename, void* array, const uint64_t array_offset) {
  return modyn::common::trigger_storage_cpp::parse_file_impl(filename, array, array_offset);
}
bool parse_file_subset(const char* filename, void* array, const uint64_t array_offset, uint64_t start_index,
                       uint64_t end_index) {
  return modyn::common::trigger_storage_cpp::parse_file_subset_impl(filename, array, array_offset, start_index,
                                                                    end_index);
}

void write_file(const char* filename, void* array, const size_t array_length, const char* header,
                const size_t header_length) {
  return modyn::common::trigger_storage_cpp::write_file_impl(filename, array, array_length, header, header_length);
}
}
