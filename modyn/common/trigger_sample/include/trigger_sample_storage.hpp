#pragma once
#include <cstdint>
#include <fstream>
#include <vector>

// NOLINTBEGIN(modernize-avoid-c-arrays)

namespace modyn::common::trigger_sample_storage {

const long dtype_size = 16;

long get_num_samples_in_file_impl(const char* filename);

void* get_worker_samples_impl(const char* folder, long* size, const char* pattern, long start_index,
                              long worker_subset_size);
void* get_all_samples_impl(const char* folder, long* size, const char* pattern);

void* parse_file_impl(const char* filename, long* size);

void write_file_impl(const char* filename, const void* data, long data_offset, long data_length, const char* header,
                     long header_length);
void write_files_impl(const char* filenames[], const void* data, long data_lengths[], const char* headers[],
                      long header_length, std::uint64_t num_files);

void release_data_impl(void* data);

bool parse_file_subset(const char* filename, std ::vector<char>& char_vector, long samples, long start_index,
                       long end_index);
std::vector<std::string> get_matching_files(const char* folder, const char* pattern);

long read_data_size_from_header(std::ifstream& file);
long read_magic(std::ifstream& file);

std::ifstream open_file(const char* filename);
std::ofstream open_file_write(const char* filename);
}  // namespace modyn::common::trigger_sample_storage

// NOLINTEND(modernize-avoid-c-arrays)
