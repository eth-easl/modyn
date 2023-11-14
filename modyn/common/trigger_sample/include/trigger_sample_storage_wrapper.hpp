#pragma once
#include <cstdint>

extern "C" {

int64_t get_num_samples_in_file(const char* filename);

void* get_worker_samples(const char* folder, int64_t* size, const char* pattern, int64_t start_index,
                         int64_t worker_subset_size);
void* get_all_samples(const char* folder, int64_t* size, const char* pattern);

void* parse_file(const char* filename, int64_t* size);

void write_file(const char* filename, const void* data, int64_t data_offset, int64_t data_length, const char* header,
                int64_t header_length);
void write_files(const char* filenames[], const void* data, int64_t data_lengths[], const char* headers[],
                 int64_t header_length, std::uint64_t num_files);

void release_data(void* data);
}
