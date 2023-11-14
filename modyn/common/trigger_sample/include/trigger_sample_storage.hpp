#pragma once
#include <cstdint>
#include <fstream>
#include <vector>

// NOLINTBEGIN(modernize-avoid-c-arrays)

namespace modyn::common::trigger_sample_storage {

const int DTYPE_SIZE = 16;

uint64_t get_num_samples_in_file_impl(const char* filename);

void* get_worker_samples_impl(const char* folder, uint64_t* size, const char* pattern, uint64_t start_index,
                              uint64_t worker_subset_size);
void* get_all_samples_impl(const char* folder, uint64_t* size, const char* pattern);

void* parse_file_impl(const char* filename, uint64_t* size);

void write_file_impl(const char* filename, const void* data, uint64_t data_offset, uint64_t data_length,
                     const char* header, uint64_t header_length);
void write_files_impl(const char* filenames[], const void* data, std::uint64_t data_lengths[], const char* headers[],
                      std::uint64_t header_length, std::uint64_t num_files);

void release_data_impl(void* data);

bool parse_file_subset(const char* filename, std ::vector<char>& char_vector, uint64_t samples, uint64_t start_index,
                       uint64_t end_index);
std::vector<std::string> get_matching_files(const char* folder, const char* pattern);

std::uint64_t read_data_size_from_header(std::ifstream& file);
int read_magic(std::ifstream& file);

std::ifstream open_file(const char* filename);
std::ofstream open_file_write(const char* filename);
}  // namespace modyn::common::trigger_sample_storage

// NOLINTEND(modernize-avoid-c-arrays)
