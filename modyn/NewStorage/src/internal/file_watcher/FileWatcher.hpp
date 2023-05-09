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
  std::string config_file;
  long long dataset_id;
  int insertion_threads;
  bool is_test;
  bool disable_multithreading;
  int sample_dbinsertion_batchsize = 1000000;
  StorageDatabaseConnection *storage_database_connection;
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
  std::string FileWatcher::extract_file_paths_per_thread_to_file(
      int i, int files_per_thread, std::vector<std::string> file_paths);

public:
  FileWatcher(std::string config_file, long long dataset_id, bool is_test) {
    this->config = YAML::LoadFile(config_file);
    ;
    this->config_file = config_file;
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
  void handle_file_paths(std::vector<std::string> file_paths,
                         std::string data_file_extension,
                         std::string file_wrapper_type,
                         AbstractFilesystemWrapper *filesystem_wrapper,
                         int timestamp);
};
} // namespace storage

#endif