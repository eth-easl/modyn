#ifndef FILE_WATCHER_HPP
#define FILE_WATCHER_HPP

#include "../database/StorageDatabaseConnection.hpp"
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
  std::string config_path;
  long long dataset_id;
  int insertion_threads;
  bool is_test;
  bool disable_multithreading;
  int sample_dbinsertion_batchsize = 1000000;
  StorageDatabaseConnection *storage_database_connection;
  void handle_file_paths(std::vector<std::string> file_paths,
                         std::string data_file_extension,
                         std::string file_wrapper_type,
                         AbstractFilesystemWrapper *filesystem_wrapper,
                         int timestamp);
  void update_files_in_directory(AbstractFilesystemWrapper *filesystem_wrapper,
                                 std::string directory_path, int timestamp);
  void seek_dataset();
  void seek();
  bool checkValidFile(std::string file_path, std::string data_file_extension,
                      bool ignore_last_timestamp, int timestamp,
                      AbstractFilesystemWrapper *filesystem_wrapper);
  void postgres_copy_insertion(
      std::vector<std::tuple<long long, long long, int, int>> file_frame,
      soci::session *sql);
  void fallback_insertion(
      std::vector<std::tuple<long long, long long, int, int>> file_frame,
      soci::session *sql);

public:
  FileWatcher(YAML::Node config, long long dataset_id,
              std::atomic<bool> *is_running, bool is_test,
              std::string config_path) {
    this->config = config;
    this->config_path = config_path;
    this->dataset_id = dataset_id;
    this->insertion_threads = config["storage"]["insertion_threads"].as<int>();
    this->is_test = is_test;
    this->disable_multithreading = insertion_threads <= 1;
    if (config["storage"]["sample_dbinsertion_batchsize"]) {
      this->sample_dbinsertion_batchsize =
          config["storage"]["sample_dbinsertion_batchsize"].as<int>();
    }
    this->storage_database_connection = new StorageDatabaseConnection(config);
  }
  void run();
};
} // namespace storage

#endif