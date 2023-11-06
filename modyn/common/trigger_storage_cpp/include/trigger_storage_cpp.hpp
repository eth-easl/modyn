#pragma once
#include <cstdint>
#include <fstream>
#include <vector>

namespace modyn::common::trigger_storage_cpp {

void* get_worker_samples_impl(const char* folder, uint64_t* size, const char* pattern, const uint64_t start_index,
                              const size_t worker_subset_size);
void* get_all_samples_impl(const char* folder, uint64_t* size, const char* pattern);
uint64_t get_num_samples_in_file_impl(const char* filename);
uint64_t parse_file_impl(const char* filename, void* array, const uint64_t array_offset);
bool parse_file_subset(const char* filename, std ::vector<char>& char_vector, const uint64_t samples,
                       uint64_t start_index, uint64_t end_index);
void write_file_impl(const char* filename, void* array, const size_t array_length, const char* header,
                     const size_t header_length);
void* parse_file_direct_impl(const char* filename, uint64_t* size);
void release_array_impl(void* array);

std::vector<std::string> get_matching_files(const char* folder, const char* pattern);
std::size_t read_data_size_from_header(std::ifstream& file);
int read_magic(std::ifstream& file);
std::ifstream open_file(const char* filename);
std::ofstream open_file_write(const char* filename);
}  // namespace modyn::common::trigger_storage_cpp
