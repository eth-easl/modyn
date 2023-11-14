#pragma once
#include <cstdint>

extern "C" {

int get_num_samples_in_file(const char* filename);

void* get_worker_samples(const char* folder, int* size, const char* pattern, int start_index, int worker_subset_size);
void* get_all_samples(const char* folder, int* size, const char* pattern);

void* parse_file(const char* filename, int* size);

void write_file(const char* filename, const void* data, int data_offset, int data_length, const char* header,
                int header_length);
void write_files(const char* filenames[], const void* data, int data_lengths[], const char* headers[],
                 int header_length, std::uint64_t num_files);

void release_data(void* data);
}
