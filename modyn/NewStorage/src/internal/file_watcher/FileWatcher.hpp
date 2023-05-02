#ifndef FILE_WATCHER_HPP
#define FILE_WATCHER_HPP

#include <yaml-cpp/yaml.h>
#include <atomic>
#include <string>
#include <vector>
#include "../file_wrapper/AbstractFileWrapper.hpp"
#include "../filesystem_wrapper/AbstractFilesystemWrapper.hpp"

namespace storage {
    class FileWatcher {
        private:
            YAML::Node config;
            int dataset_id;
            int insertion_threads;
            bool is_test;
            bool disable_multithreading;
            std::atomic<bool> is_running;
            bool file_unknown(std::string file_path);
            void handle_file_paths(
                std::vector<std::string> file_paths,
                std::string data_file_extension,
                AbstractFileWrapper* file_wrapper,
                AbstractFilesystemWrapper* filesystem_wrapper,
                int timestamp
            );
            void update_files_in_directory(
                AbstractFileWrapper* file_wrapper,
                AbstractFilesystemWrapper* filesystem_wrapper,
                std::string directory_path,
                int timestamp
            );
            void seek_dataset();
            void seek();
            void get_datasets();
        public:
            FileWatcher(YAML::Node config, int dataset_id, int insertion_threads, bool is_test = false) {
                this->config = config;
                this->dataset_id = dataset_id;
                this->insertion_threads = insertion_threads;
                this->is_test = is_test;
                this->disable_multithreading = insertion_threads <= 1;
                this->is_running = true;
            }
            void run();
    };
}

#endif