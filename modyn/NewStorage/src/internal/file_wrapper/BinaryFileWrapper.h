#ifndef BINARY_FILE_WRAPPER_H
#define BINARY_FILE_WRAPPER_H

#include "AbstractFileWrapper.h"
#include <cstddef>

namespace storage
{
    class BinaryFileWrapper : public AbstractFileWrapper
    {
    private:
        std::string byteorder;
        int record_size;
        int label_size;
        int file_size;
        int sample_size;
        void validate_file_extension();
        void validate_request_indices(int total_samples, std::vector<int> *indices);
        int int_from_bytes(unsigned char *begin, unsigned char *end);

    public:
        BinaryFileWrapper(std::string path, YAML::Node file_wrapper_config, AbstractFileSystemWrapper *filesystem_wrapper) : AbstractFileWrapper(path, file_wrapper_config, filesystem_wrapper)
        {
            this->byteorder = file_wrapper_config["byteorder"].as<std::string>();
            this->record_size = file_wrapper_config["record_size"].as<int>();
            this->label_size = file_wrapper_config["label_size"].as<int>();
            this->sample_size = this->record_size - this->label_size;

            if (this->record_size - this->label_size < 1)
            {
                throw std::runtime_error("Each record must have at least 1 byte of data other than the label.");
            }

            this->validate_file_extension();
            this->file_size = filesystem_wrapper->get_file_size(path);

            if (this->file_size % this->record_size != 0)
            {
                throw std::runtime_error("File size must be a multiple of the record size.");
            }
        }
        int get_number_of_samples();
        int get_label(int index);
        std::vector<std::vector<int>> *get_all_labels();
        std::vector<std::vector<unsigned char>> *get_samples(int start, int end);
        std::vector<unsigned char>* get_sample(int index);
        std::vector<std::vector<unsigned char>> *get_samples_from_indices(std::vector<int> *indices);
    };
}

#endif