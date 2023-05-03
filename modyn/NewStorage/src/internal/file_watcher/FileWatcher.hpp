#ifndef FILE_WATCHER_HPP
#define FILE_WATCHER_HPP

#include "../file_wrapper/AbstractFileWrapper.hpp"
#include "../filesystem_wrapper/AbstractFilesystemWrapper.hpp"
#include <atomic>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace storage {
class FileWatcher {
private:
  YAML::Node config;
  int dataset_id;
  int insertion_threads;
  bool is_test;
  bool disable_multithreading;
  std::atomic<bool> is_running;
  int sample_dbinsertion_batchsize = 1000000;
  bool file_unknown(std::string file_path);
  void handle_file_paths(std::vector<std::string> file_paths,
                         std::string data_file_extension,
                         AbstractFileWrapper *file_wrapper,
                         AbstractFilesystemWrapper *filesystem_wrapper,
                         int timestamp);
  void update_files_in_directory(AbstractFileWrapper *file_wrapper,
                                 AbstractFilesystemWrapper *filesystem_wrapper,
                                 std::string directory_path, int timestamp);
  void seek_dataset();
  void seek();
  void get_datasets();
  void postgres_copy_insertion(
      int process_id, int dataset_id,
      std::vector<std::vector<std::tuple<int, int, long, long>>> *file_data);
  void fallback_copy_insertion(
      int process_id, int dataset_id,
      std::vector<std::vector<std::tuple<int, int, long, long>>> *file_data);

public:
  FileWatcher(YAML::Node config, int dataset_id, std::atomic<bool> *is_running,
              bool is_test) {
    this->config = config;
    this->dataset_id = dataset_id;
    this->insertion_threads = config["storage"]["insertion_threads"].as<int>();
    this->is_test = is_test;
    this->disable_multithreading = insertion_threads <= 1;
    this->is_running = is_running;
    if (config["storage"]["sample_dbinsertion_batchsize"]) {
      this->sample_dbinsertion_batchsize =
          config["storage"]["sample_dbinsertion_batchsize"].as<int>();
    }
  }
  void run();
};
} // namespace storage

#endif