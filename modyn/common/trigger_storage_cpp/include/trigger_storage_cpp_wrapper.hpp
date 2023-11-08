#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" {

uint64_t get_num_samples_in_file(const char* filename);

void* get_worker_samples(const char* folder, uint64_t* size, const char* pattern, uint64_t start_index,
                         size_t worker_subset_size);
void* get_all_samples(const char* folder, uint64_t* size, const char* pattern);

void* parse_file(const char* filename, uint64_t* size);

void write_file(const char* filename, const void* data, size_t data_offset, size_t data_length, const char* header,
                size_t header_length);
void write_files(const char* filenames[], const void* data, std::size_t data_lengths[], const char* headers[],
                 std::size_t header_length, std::size_t num_files);

void release_data(void* data);
}
