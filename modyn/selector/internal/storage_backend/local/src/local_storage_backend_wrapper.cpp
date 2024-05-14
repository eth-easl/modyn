#include "local_storage_backend_wrapper.hpp"

#include <cstddef>
#include <cstdint>

#include "local_storage_backend.hpp"

extern "C" {

void write_files(const char* filenames[], const void* data, int64_t data_lengths[], std::uint64_t num_files) {
  return modyn::selector::local_storage_backend::write_files_impl(filenames, data, data_lengths, num_files);
}

void parse_files(const char* filenames[], void* data, int64_t data_lengths[], int64_t data_offsets[],
                 uint64_t num_files) {
  return modyn::selector::local_storage_backend::parse_files_impl(filenames, data, data_lengths, data_offsets,
                                                                  num_files);
}
}
