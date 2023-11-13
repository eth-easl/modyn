#include "trigger_sample_storage_wrapper.hpp"

#include <cstddef>
#include <cstdint>

#include "trigger_sample_storage.hpp"

extern "C" {
uint64_t get_num_samples_in_file(const char* filename) {
  return modyn::common::trigger_sample_storage::get_num_samples_in_file_impl(filename);
}

void* get_worker_samples(const char* folder, uint64_t* size, const char* pattern, uint64_t start_index,
                         uint64_t worker_subset_size) {
  return modyn::common::trigger_sample_storage::get_worker_samples_impl(folder, size, pattern, start_index,
                                                                        worker_subset_size);
}
void* get_all_samples(const char* folder, uint64_t* size, const char* pattern) {
  return modyn::common::trigger_sample_storage::get_all_samples_impl(folder, size, pattern);
}

void write_file(const char* filename, const void* data, uint64_t data_offset, uint64_t data_length, const char* header,
                uint64_t header_length) {
  return modyn::common::trigger_sample_storage::write_file_impl(filename, data, data_offset, data_length, header,
                                                                header_length);
}
void write_files(const char* filenames[], const void* data, std::uint64_t data_lengths[], const char* headers[],
                 std::uint64_t header_length, std::uint64_t num_files) {
  return modyn::common::trigger_sample_storage::write_files_impl(filenames, data, data_lengths, headers, header_length,
                                                                 num_files);
}

void* parse_file(const char* filename, uint64_t* size) {
  return modyn::common::trigger_sample_storage::parse_file_impl(filename, size);
}

void release_data(void* data) { return modyn::common::trigger_sample_storage::release_data_impl(data); }
}
