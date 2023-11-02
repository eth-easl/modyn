#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
uint64_t get_worker_samples(const char* folder, const char* pattern, void* array, const uint64_t start_index,
                            const size_t worker_subset_size);
uint64_t get_all_samples(const char* folder, const char* pattern, void* array);
uint64_t get_num_samples_in_file(const char* filename);
uint64_t parse_file(const char* filename, void* array, const uint64_t array_offset);
bool parse_file_subset(const char* filename, void* array, const uint64_t array_offset, uint64_t start_index,
                       uint64_t end_index);
void write_file(const char* filename, void* array, const size_t array_length, const char* header,
                const size_t header_length);
}
