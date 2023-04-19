#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;

void _validate_request_indices(int total_samples, std::vector<int> indices) {
    bool invalid_indices = false;
    for (int idx : indices) {
        if (idx < 0 || idx > (total_samples - 1)) {
            invalid_indices = true;
            break;
        }
    }
    if (invalid_indices) {
        throw std::out_of_range("Indices are out of range. Indices should be between 0 and " + std::to_string(total_samples));
    }
}

int int_from_bytes(std::vector<unsigned char> bytes, std::string byteorder) {
    int value = 0;
    if (byteorder == "big") {
        value = std::accumulate(bytes.begin(), bytes.end(), 0, 
                                [](int acc, unsigned char x) { return (acc << 8) | x; });
    } else if (byteorder == "little") {
        value = std::accumulate(bytes.rbegin(), bytes.rend(), 0, 
                                [](int acc, unsigned char x) { return (acc << 8) | x; });
    }
    return value;
}

int get_label(std::vector<unsigned char> data, int index, int record_size, int label_size, std::string byteorder) {
    int record_start = index * record_size;

    std::vector<unsigned char> label_bytes = std::vector<unsigned char>(data.begin() + record_start, data.begin() + record_start + label_size);

    int label = int_from_bytes(label_bytes, byteorder);
    return label;
}

std::vector<int> get_all_labels(std::vector<unsigned char> data, double num_samples, int record_size, int label_size, std::string byteorder) {
    std::vector<int> labels(num_samples);
    for (int idx = 0; idx < num_samples; idx++) {
        std::vector<unsigned char> label_bytes(data.begin() + (idx * record_size), 
                                                data.begin() + (idx * record_size) + label_size);
        labels[idx] = int_from_bytes(label_bytes, byteorder);
    }
    return labels;
}

std::vector<unsigned char> get_data_from_file(std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return std::vector<unsigned char>();
    }
    std::vector<unsigned char> contents;
    char c;
    while (file.get(c)) {
        contents.push_back(c);
    }
    file.close();
    return contents;
}

int main(int argc, char* argv[]) {
    std::string arg = argv[1];
    int num_iterations = std::stoi(arg);
    arg = argv[2];
    double num_samples = std::stod(arg);
    std::string byteorder = argv[3];
    std::string filename = argv[4];

    cout << "num_samples: " << num_samples << endl;

    // Time the get_label function
    int total_time = 0;

    std::vector<unsigned char> data = get_data_from_file(filename);
    for (int i = 0; i < num_iterations; i++) {

        // Start timer
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> labels = get_all_labels(data, num_samples, 8, 4, byteorder);
        // Stop timer
        auto stop = std::chrono::high_resolution_clock::now();
        if (labels.size() != num_samples) {
            std::cerr << "Error: labels.size() != num_samples" << std::endl;
        }
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        total_time += duration.count();
    }

    // Print time in nanoseconds averaged over num_iterations
    std::cout << "C++ microbenchmark time: " << num_iterations << " loops, best of 1: " << total_time / num_iterations << " msec per loop" << std::endl;
    return 0;
}