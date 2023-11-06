#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" {

void* get_worker_samples(const char* folder, uint64_t* size, const char* pattern, const uint64_t start_index,
                         const size_t worker_subset_size);
void* get_all_samples(const char* folder, uint64_t* size, const char* pattern);
uint64_t get_num_samples_in_file(const char* filename);
uint64_t parse_file(const char* filename, void* array, const uint64_t array_offset);
void write_file(const char* filename, void* array, const size_t array_length, const char* header,
                const size_t header_length);
void* parse_file_direct(const char* filename, uint64_t* size);
void release_array(void* array);
}
