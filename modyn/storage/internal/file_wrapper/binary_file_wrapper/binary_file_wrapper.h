#ifndef BINARY_FILE_WRAPPER_H
#define BINARY_FILE_WRAPPER_H

#include <cstddef>

struct IntVector {
    int* data;
    size_t size;
};

struct CharVector {
    char* data;
    size_t size;
};

extern "C" bool validate_request_indices(int total_samples, IntVector* indices);
extern "C" int get_label_native(const char* filename, int index, int record_size, int label_size);
extern "C" int get_label(unsigned char *data, int index, int record_size, int label_size);
extern "C" IntVector* get_all_labels_native(const char* filename, double num_samples, int record_size, int label_size);
extern "C" IntVector* get_all_labels(unsigned char *data, double num_samples, int record_size, int label_size);
extern "C" CharVector* get_samples_from_indices_native(const char* filename, IntVector* indices, int record_size, int label_size);
extern "C" CharVector* get_samples_from_indices(unsigned char *data, IntVector* indices, int record_size, int label_size);


int int_from_bytes(unsigned char *begin, unsigned char *end);
bool validate_request_indices(int total_samples, IntVector *indices);
std::vector<unsigned char> get_data_from_file(const char *filename);

#endif