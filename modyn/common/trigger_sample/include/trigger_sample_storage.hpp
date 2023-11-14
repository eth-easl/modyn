#pragma once
#include <cstdint>
#include <fstream>
#include <vector>

// NOLINTBEGIN(modernize-avoid-c-arrays)

namespace modyn::common::trigger_sample_storage {

const int64_t dtype_size = 16;

int64_t get_num_samples_in_file_impl(const char* filename);

void* get_worker_samples_impl(const char* folder, int64_t* size, const char* pattern, int64_t start_index,
                              int64_t worker_subset_size);
void* get_all_samples_impl(const char* folder, int64_t* size, const char* pattern);

void* parse_file_impl(const char* filename, int64_t* size);

void write_files_impl(const char* filenames[], const void* data, int64_t data_lengths[], const char* headers[],
                      int64_t header_length, std::uint64_t num_files);

void release_data_impl(void* data);

void write_file(const char* filename, const void* data, int64_t data_offset, int64_t data_length, const char* header,
                int64_t header_length);
bool parse_file_subset(const char* filename, std ::vector<char>& char_vector, int64_t samples, int64_t start_index,
                       int64_t end_index);
std::vector<std::string> get_matching_files(const char* folder, const char* pattern);

int64_t read_data_size_from_header(std::ifstream& file);
int64_t read_magic(std::ifstream& file);

std::ifstream open_file(const char* filename);
std::ofstream open_file_write(const char* filename);
}  // namespace modyn::common::trigger_sample_storage

// NOLINTEND(modernize-avoid-c-arrays)
