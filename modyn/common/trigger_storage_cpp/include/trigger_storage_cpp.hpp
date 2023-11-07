#pragma once
#include <cstdint>
#include <fstream>
#include <vector>

namespace modyn::common::trigger_storage_cpp {

uint64_t get_num_samples_in_file_impl(const char* filename);

void* get_worker_samples_impl(const char* folder, uint64_t* size, const char* pattern, const uint64_t start_index,
                              const size_t worker_subset_size);
void* get_all_samples_impl(const char* folder, uint64_t* size, const char* pattern);

void* parse_file_impl(const char* filename, uint64_t* size);

void write_file_impl(const char* filename, const void* data, size_t data_offset, const size_t data_length,
                     const char* header, const size_t header_length);
void write_files_impl(const char* filenames[], const void* data, std::size_t data_lengths[], const char* headers[],
                      std::size_t header_length, std::size_t num_files);

void release_data_impl(void* data);

bool parse_file_subset(const char* filename, std ::vector<char>& char_vector, const uint64_t samples,
                       uint64_t start_index, uint64_t end_index);
std::vector<std::string> get_matching_files(const char* folder, const char* pattern);

std::size_t read_data_size_from_header(std::ifstream& file);
int read_magic(std::ifstream& file);

std::ifstream open_file(const char* filename);
std::ofstream open_file_write(const char* filename);
}  // namespace modyn::common::trigger_storage_cpp
