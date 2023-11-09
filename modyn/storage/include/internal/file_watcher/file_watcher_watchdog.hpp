#pragma once

#include <yaml-cpp/yaml.h>

#include <atomic>
#include <map>
#include <thread>
#include <tuple>
#include <vector>

#include "file_watcher.hpp"
#include "internal/database/storage_database_connection.hpp"
#include "modyn/utils/utils.hpp"

namespace modyn::storage {

class FileWatcherWatchdog {
 public:
  FileWatcherWatchdog(const YAML::Node& config, std::atomic<bool>* stop_file_watcher_watchdog,
                      std::atomic<bool>* request_storage_shutdown)
      : config_{config},
        stop_file_watcher_watchdog_{stop_file_watcher_watchdog},
        request_storage_shutdown_{request_storage_shutdown},
        storage_database_connection_{StorageDatabaseConnection(config_)} {
    if (stop_file_watcher_watchdog_ == nullptr) {
      FAIL("stop_file_watcher_watchdog_ is nullptr.");
    }

    if (config_["storage"]["file_watcher_watchdog_sleep_time_s"]) {
      file_watcher_watchdog_sleep_time_s_ = config_["storage"]["file_watcher_watchdog_sleep_time_s"].as<int64_t>();
    }

    ASSERT(config_["storage"]["insertion_threads"], "Config does not contain insertion_threads");
  }
  void watch_file_watcher_threads();
  void start_file_watcher_thread(int64_t dataset_id);
  void stop_file_watcher_thread(int64_t dataset_id);
  void run();
  void stop() {
    stop_file_watcher_watchdog_->store(true);
    request_storage_shutdown_->store(true);
  }
  std::vector<int64_t> get_running_file_watcher_threads();

 private:
  void stop_and_clear_all_file_watcher_threads();
  YAML::Node config_;
  int64_t file_watcher_watchdog_sleep_time_s_ = 3;
  std::map<int64_t, FileWatcher> file_watchers_ = {};
  std::map<int64_t, std::thread> file_watcher_threads_ = {};
  std::map<int64_t, int16_t> file_watcher_dataset_retries_ = {};
  std::map<int64_t, std::atomic<bool>> file_watcher_thread_stop_flags_ = {};
  // Used to stop the FileWatcherWatchdog thread from storage main thread
  std::atomic<bool>* stop_file_watcher_watchdog_;
  std::atomic<bool>* request_storage_shutdown_;
  StorageDatabaseConnection storage_database_connection_;
};
}  // namespace modyn::storage
