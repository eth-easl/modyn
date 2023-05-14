#pragma once

#include <yaml-cpp/yaml.h>

#include <atomic>
#include <map>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "file_watcher.hpp"
#include "internal/database/storage_database_connection.hpp"

namespace storage {
class FileWatchdog {
 private:
  YAML::Node config;
  std::string config_file;
  std::unordered_map<long long, std::tuple<std::thread, int, std::shared_ptr<std::atomic<bool>>>>
      file_watcher_processes;
  std::shared_ptr<std::atomic<bool>> stop_file_watchdog;

 public:
  FileWatchdog(std::string config_file, std::shared_ptr<std::atomic<bool>> stop_file_watchdog) {
    this->config_file = config_file;
    this->config = YAML::LoadFile(config_file);
    this->file_watcher_processes =
        std::unordered_map<long long, std::tuple<std::thread, int, std::shared_ptr<std::atomic<bool>>>>();
    this->stop_file_watchdog = stop_file_watchdog;
  }
  void watch_file_watcher_processes(StorageDatabaseConnection* storage_database_connection);
  void start_file_watcher_process(long long dataset_id, int retries);
  void stop_file_watcher_process(long long dataset_id, bool is_test = false);
  void run();
  std::vector<long long> get_running_file_watcher_processes();
};
}  // namespace storage
