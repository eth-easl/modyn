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
  YAML::Node config_;
  std::unordered_map<int64_t, std::thread> file_watcher_processes_;
  std::unordered_map<int64_t, int16_t> file_watcher_process_retries_;
  std::unordered_map<int64_t, std::atomic<bool>> file_watcher_process_stop_flags_;
  std::atomic<bool>* stop_file_watchdog_;

 public:
  FileWatchdog(
      const YAML::Node& config,
      std::atomic<bool>* stop_file_watchdog)  // NOLINT // clang-tidy thinks we dont initialize the unordered maps
      : config_{config}, stop_file_watchdog_(stop_file_watchdog) {
    file_watcher_processes_ = std::unordered_map<int64_t, std::thread>();
    file_watcher_process_retries_ = std::unordered_map<int64_t, int16_t>();
    file_watcher_process_stop_flags_ = std::unordered_map<int64_t, std::atomic<bool>>();
  }
  void watch_file_watcher_processes(StorageDatabaseConnection* storage_database_connection);
  void start_file_watcher_process(int64_t dataset_id, int16_t retries);
  void stop_file_watcher_process(int64_t dataset_id, bool is_test = false);
  void run();
  std::vector<int64_t> get_running_file_watcher_processes();
};
}  // namespace storage
