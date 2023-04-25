#include "binary_file_wrapper.h"
#include <fstream>
#include <numeric>
#include <vector>
#include <iostream>

using namespace std;

std::vector<unsigned char> get_data_from_file(const char *filename)
{
    std::ifstream input_file(filename);
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());

    return data;
}

bool validate_request_indices(int total_samples, IntVector *indices)
{
    for (int i = 0; i < indices->size; i++)
    {
        if (indices->data[i] < 0 || indices->data[i] > (total_samples - 1))
        {
            return false;
        }
    }
    return true;
}

int int_from_bytes(unsigned char *begin, unsigned char *end)
{
    int value = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    value = std::accumulate(begin, end, 0,
                            [](int acc, unsigned char x)
                            { return (acc << 8) | x; });
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    value = *reinterpret_cast<int *>(begin);
#else
#error "Unknown byte order"
#endif
    return value;
}

int get_label(unsigned char *data, int index, int record_size, int label_size)
{
    int record_start = index * record_size;
    unsigned char *label_begin = data + record_start;
    unsigned char *label_end = label_begin + label_size;

    int label = int_from_bytes(label_begin, label_end);
    return label;
}

int get_label_native(const char *filename, int index, int record_size, int label_size)
{
    std::vector<unsigned char> data = get_data_from_file(filename);
    return get_label(data.data(), index, record_size, label_size);
}

IntVector *get_all_labels(unsigned char *data, double num_samples, int record_size, int label_size)
{
    cout << "num_samples: " << num_samples << endl;
    IntVector *labels = new IntVector;
    labels->size = num_samples;
    cout << "labels->size: " << labels->size << endl;
    for (int idx = 0; idx < num_samples; idx++)
    {
        unsigned char *label_begin = data + (idx * record_size);
        unsigned char *label_end = label_begin + label_size;
        labels->data[idx] = int_from_bytes(label_begin, label_end);
    }
    return labels;
}

IntVector *get_all_labels_native(const char *filename, double num_samples, int record_size, int label_size)
{
    std::vector<unsigned char> data = get_data_from_file(filename);
    return get_all_labels(data.data(), num_samples, record_size, label_size);
}

CharVector *get_samples_from_indices(unsigned char *data, IntVector *indices, int record_size, int label_size)
{
    int sample_size = record_size - label_size;
    CharVector *samples = new CharVector;
    samples->size = indices->size;
    samples->data = new char[samples->size * sample_size];
    for (int idx = 0; idx < indices->size; idx++)
    {
        unsigned char *sample_begin = data + (indices->data[idx] * record_size) + label_size;
        memcpy(samples->data + (idx * sample_size), sample_begin, sample_size);
    }
    return samples;
}

CharVector *get_samples_from_indices_native(const char *filename, IntVector *indices, int record_size, int label_size)
{
    std::vector<unsigned char> data = get_data_from_file(filename);
    return get_samples_from_indices(data.data(), indices, record_size, label_size);
}
