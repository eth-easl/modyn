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
  std::string config_file_;
  std::unordered_map<int64_t, std::tuple<std::thread, int, std::shared_ptr<std::atomic<bool>>>> file_watcher_processes_;
  std::shared_ptr<std::atomic<bool>> stop_file_watchdog_;

 public:
  FileWatchdog(const std::string& config_file, std::shared_ptr<std::atomic<bool>> stop_file_watchdog)  // NOLINT
      : config_file_(config_file), stop_file_watchdog_(std::move(stop_file_watchdog)) {
    config_ = YAML::LoadFile(config_file);
    file_watcher_processes_ =
        std::unordered_map<int64_t, std::tuple<std::thread, int, std::shared_ptr<std::atomic<bool>>>>();
  }
  void watch_file_watcher_processes(StorageDatabaseConnection* storage_database_connection);
  void start_file_watcher_process(int64_t dataset_id, int retries);
  void stop_file_watcher_process(int64_t dataset_id, bool is_test = false);
  void run();
  std::vector<int64_t> get_running_file_watcher_processes();
};
}  // namespace storage
