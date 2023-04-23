#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

void _validate_request_indices(int total_samples, std::vector<int> indices)
{
    for (int idx : indices)
    {
        if (idx < 0 || idx > (total_samples - 1))
        {
            throw std::out_of_range("Indices are out of range. Indices should be between 0 and " + std::to_string(total_samples));
        }
    }
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

std::vector<int> get_all_labels(unsigned char *data, double num_samples, int record_size, int label_size)
{
    std::vector<int> labels(num_samples);
    for (int idx = 0; idx < num_samples; idx++)
    {
        unsigned char *label_begin = data + (idx * record_size);
        unsigned char *label_end = label_begin + label_size;
        labels[idx] = int_from_bytes(label_begin, label_end);
    }
    return labels;
}

std::vector<unsigned char> get_data_from_file(std::string filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return std::vector<unsigned char>();
    }
    std::vector<unsigned char> contents;
    char c;
    while (file.get(c))
    {
        contents.push_back(c);
    }
    file.close();
    return contents;
}
