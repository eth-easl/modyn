#pragma once

#include <yaml-cpp/yaml.h>

#include <atomic>
#include <map>
#include <thread>
#include <tuple>
#include <vector>

#include "file_watcher.hpp"
#include "internal/database/storage_database_connection.hpp"
#include "internal/utils/utils.hpp"

namespace storage::file_watcher {
class FileWatcherWatchdog {
 public:
  FileWatcherWatchdog(const YAML::Node& config, std::atomic<bool>* stop_file_watcher_watchdog)
      : config_{config},
        file_watcher_threads_{std::map<int64_t, std::thread>()},
        file_watcher_dataset_retries_{std::map<int64_t, int16_t>()},
        file_watcher_thread_stop_flags_{std::map<int64_t, std::atomic<bool>>()},
        stop_file_watcher_watchdog_{stop_file_watcher_watchdog},
        storage_database_connection_{storage::database::StorageDatabaseConnection(config_)} {
    if (stop_file_watcher_watchdog_ == nullptr) {
      FAIL("stop_file_watcher_watchdog_ is nullptr.");
    }
  }
  void watch_file_watcher_threads();
  void start_file_watcher_thread(int64_t dataset_id, int16_t retries);
  void stop_file_watcher_thread(int64_t dataset_id);
  void run();
  std::vector<int64_t> get_running_file_watcher_threads();

 private:
  YAML::Node config_;
  std::map<int64_t, std::thread> file_watcher_threads_;
  std::map<int64_t, int16_t> file_watcher_dataset_retries_;
  std::map<int64_t, std::atomic<bool>> file_watcher_thread_stop_flags_;
  // Used to stop the FileWatcherWatchdog thread from storage main thread
  std::atomic<bool>* stop_file_watcher_watchdog_;
  storage::database::StorageDatabaseConnection storage_database_connection_;
};
}  // namespace storage::file_watcher
