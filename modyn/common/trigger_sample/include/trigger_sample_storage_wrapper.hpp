#pragma once
#include <cstdint>

extern "C" {

long get_num_samples_in_file(const char* filename);

void* get_worker_samples(const char* folder, long* size, const char* pattern, long start_index,
                         long worker_subset_size);
void* get_all_samples(const char* folder, long* size, const char* pattern);

void* parse_file(const char* filename, long* size);

void write_file(const char* filename, const void* data, long data_offset, long data_length, const char* header,
                long header_length);
void write_files(const char* filenames[], const void* data, long data_lengths[], const char* headers[],
                 long header_length, std::uint64_t num_files);

void release_data(void* data);
}
