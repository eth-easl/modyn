#include <string>
#include <yaml-cpp/yaml.h>
#include "modyn/NewStorage/src/internal/filesystem_wrapper/abstract_file_system_wrapper.h"

namespace storage {
    class AbstractFileWrapper {
        protected:
            std::string path;
            YAML::Node file_wrapper_config;
            AbstractFileSystemWrapper* file_system_wrapper;
        AbstractFileWrapper(std::string path, YAML::Node file_wrapper_config, AbstractFileSystemWrapper* file_system_wrapper) {
            this->path = path;
            this->file_wrapper_config = file_wrapper_config;
            this->file_system_wrapper = file_system_wrapper;
        }
        virtual int get_number_of_samples() = 0;
        virtual std::vector<std::vector<unsigned char>>* get_samples(int start, int end) = 0;
        virtual int get_label(int index) = 0;
        virtual std::vector<std::vector<int>>* get_all_labels() = 0;
        virtual unsigned char get_sample(int index) = 0;
        virtual std::vector<std::vector<unsigned char>>* get_samples_from_indices(std::vector<int>* indices) = 0;
    };
}