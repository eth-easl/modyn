#pragma once
#include <cstdint>

extern "C" {

void parse_files(const char* filenames[], void* data, int64_t data_lengths[], int64_t data_offsets[],
                 uint64_t num_files);

void write_files(const char* filenames[], const void* data, int64_t data_lengths[], uint64_t num_files);
}
