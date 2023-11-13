#pragma once
#include <cstdint>

extern "C" {

uint64_t get_num_samples_in_file(const char* filename);

void* get_worker_samples(const char* folder, uint64_t* size, const char* pattern, uint64_t start_index,
                         uint64_t worker_subset_size);
void* get_all_samples(const char* folder, uint64_t* size, const char* pattern);

void* parse_file(const char* filename, uint64_t* size);

void write_file(const char* filename, const void* data, uint64_t data_offset, uint64_t data_length, const char* header,
                uint64_t header_length);
void write_files(const char* filenames[], const void* data, std::uint64_t data_lengths[], const char* headers[],
                 std::uint64_t header_length, std::uint64_t num_files);

void release_data(void* data);
}
