#pragma once
#include <cstdint>
#include <fstream>
#include <vector>

// NOLINTBEGIN(modernize-avoid-c-arrays)

namespace modyn::common::local_storage_backend {

const int64_t dtype_size = 8;

void parse_files_impl(const char* filenames[], void* data, int64_t data_lengths[], int64_t data_offsets[],
                      uint64_t num_files);

void write_files_impl(const char* filenames[], const void* data, int64_t data_lengths[], std::uint64_t num_files);

void parse_file(const char* filename, void* data, int64_t data_offset, int64_t file_length, int64_t file_offset);
void write_file(const char* filename, const void* data, int64_t data_offset, int64_t data_length);

std::ifstream open_file(const char* filename);
std::ofstream open_file_write(const char* filename);
}  // namespace modyn::common::local_storage_backend

// NOLINTEND(modernize-avoid-c-arrays)
