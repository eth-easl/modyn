#include "internal/file_watcher/file_watcher_watchdog.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <map>
#include <unordered_set>

#include "soci/soci.h"

using namespace modyn::storage;

/*
 * Start a new FileWatcher thread for the given dataset
 *
 * Also add the FileWatcher thread to the map of FileWatcher threads, we propegate the retries value to the map
 * that way we can keep track of how many retries are left for a given dataset
 */
void FileWatcherWatchdog::start_file_watcher_thread(int64_t dataset_id) {
  // Start a new child thread of a FileWatcher
  file_watcher_thread_stop_flags_.emplace(dataset_id, false);
  std::unique_ptr<FileWatcher> file_watcher =
      std::make_unique<FileWatcher>(config_, dataset_id, &file_watcher_thread_stop_flags_[dataset_id],
                                    config_["storage"]["insertion_threads"].as<int16_t>());
  if (file_watcher == nullptr || file_watcher_thread_stop_flags_[dataset_id].load()) {
    SPDLOG_ERROR("Failed to create FileWatcher for dataset {}", dataset_id);
    return;
  }
  std::thread th(&FileWatcher::run, std::move(file_watcher));
  file_watcher_threads_[dataset_id] = std::move(th);
}

/*
 * Stop a FileWatcher thread for the given dataset
 *
 * Also remove the FileWatcher thread from the map of FileWatcher threads
 */
void FileWatcherWatchdog::stop_file_watcher_thread(int64_t dataset_id) {
  if (file_watcher_threads_.contains(dataset_id)) {
    // Set the stop flag for the FileWatcher thread
    file_watcher_thread_stop_flags_[dataset_id].store(true);
    // Wait for the FileWatcher thread to stop
    if (file_watcher_threads_[dataset_id].joinable()) {
      file_watcher_threads_[dataset_id].join();
    }
    auto file_watcher_thread_it = file_watcher_threads_.find(dataset_id);
    if (file_watcher_thread_it == file_watcher_threads_.end()) {
      SPDLOG_ERROR("FileWatcher thread for dataset {} not found", dataset_id);
    } else {
      file_watcher_threads_.erase(file_watcher_thread_it);
    }

    auto file_watcher_dataset_retries_it = file_watcher_dataset_retries_.find(dataset_id);
    if (file_watcher_dataset_retries_it == file_watcher_dataset_retries_.end()) {
      SPDLOG_ERROR("FileWatcher thread retries for dataset {} not found", dataset_id);
    } else {
      file_watcher_dataset_retries_.erase(file_watcher_dataset_retries_it);
    }

    auto file_watcher_thread_stop_flags_it = file_watcher_thread_stop_flags_.find(dataset_id);
    if (file_watcher_thread_stop_flags_it == file_watcher_thread_stop_flags_.end()) {
      SPDLOG_ERROR("FileWatcher thread stop flag for dataset {} not found", dataset_id);
    } else {
      file_watcher_thread_stop_flags_.erase(file_watcher_thread_stop_flags_it);
    }
  } else {
    SPDLOG_ERROR("FileWatcher thread for dataset {} not found", dataset_id);
  }
}

void FileWatcherWatchdog::stop_and_clear_all_file_watcher_threads() {
  for (auto& file_watcher_thread_flag : file_watcher_thread_stop_flags_) {
    file_watcher_thread_flag.second.store(true);
  }
  for (auto& file_watcher_thread : file_watcher_threads_) {
    if (file_watcher_thread.second.joinable()) {
      file_watcher_thread.second.join();
    }
  }
  file_watcher_threads_.clear();
  file_watcher_dataset_retries_.clear();
  file_watcher_thread_stop_flags_.clear();
}

/*
 * Watch the FileWatcher threads and start/stop them as needed
 */
void FileWatcherWatchdog::watch_file_watcher_threads() {
  soci::session session = storage_database_connection_.get_session();

  int64_t number_of_datasets = 0;
  session << "SELECT COUNT(dataset_id) FROM datasets", soci::into(number_of_datasets);

  if (number_of_datasets == 0) {
    if (file_watcher_threads_.empty()) {
      // There are no FileWatcher threads running, nothing to do
      return;
    }
    // There are no datasets in the database, stop all FileWatcher threads
    stop_and_clear_all_file_watcher_threads();
    return;
  }

  std::vector<int64_t> dataset_ids_vector(number_of_datasets);
  session << "SELECT dataset_id FROM datasets", soci::into(dataset_ids_vector);

  std::unordered_set<int64_t> dataset_ids(dataset_ids_vector.begin(), dataset_ids_vector.end());

  const std::vector<int64_t> running_file_watcher_threads = get_running_file_watcher_threads();
  for (const auto& dataset_id : running_file_watcher_threads) {
    if (!dataset_ids.contains(dataset_id)) {
      // There is a FileWatcher thread running for a dataset that was deleted
      // from the database. Stop the thread.
      stop_file_watcher_thread(dataset_id);
    }
  }

  for (const auto& dataset_id : dataset_ids) {
    if (file_watcher_dataset_retries_[dataset_id] > 2) {
      if (file_watcher_dataset_retries_[dataset_id] == 3) {
        SPDLOG_ERROR("FileWatcher thread for dataset {} failed to start 3 times, not trying again", dataset_id);
        file_watcher_dataset_retries_[dataset_id] += 1;
      }
      // There have been more than 3 restart attempts for this dataset, we are not going to try again
    } else if (!file_watcher_threads_.contains(dataset_id)) {
      // There is no FileWatcher thread registered for this dataset. Start one.
      if (!file_watcher_dataset_retries_.contains(dataset_id)) {
        file_watcher_dataset_retries_[dataset_id] = 0;
      }
      start_file_watcher_thread(dataset_id);
    } else if (!file_watcher_threads_[dataset_id].joinable()) {
      // The FileWatcher thread is not running. (Re)start it.
      start_file_watcher_thread(dataset_id);
      file_watcher_dataset_retries_[dataset_id] += 1;
    }
  }
}

void FileWatcherWatchdog::run() {
  while (true) {
    if (stop_file_watcher_watchdog_->load()) {
      break;
    }
    try {
      watch_file_watcher_threads();
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Exception in FileWatcherWatchdog::run(): {}", e.what());
      stop();
    }
    std::this_thread::sleep_for(std::chrono::seconds(file_watcher_watchdog_sleep_time_s_));
  }
  for (auto& file_watcher_thread_flag : file_watcher_thread_stop_flags_) {
    file_watcher_thread_flag.second.store(true);
  }
  for (auto& file_watcher_thread : file_watcher_threads_) {
    if (file_watcher_thread.second.joinable()) {
      file_watcher_thread.second.join();
    }
  }
}

std::vector<int64_t> FileWatcherWatchdog::get_running_file_watcher_threads() {
  std::vector<int64_t> running_file_watcher_threads = {};
  for (const auto& pair : file_watcher_threads_) {
    if (pair.second.joinable()) {
      running_file_watcher_threads.push_back(pair.first);
    }
  }
  return running_file_watcher_threads;
}