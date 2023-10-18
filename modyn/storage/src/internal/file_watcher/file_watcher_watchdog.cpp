#include "internal/file_watcher/file_watcher_watchdog.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <map>

#include "soci/soci.h"

using namespace storage::file_watcher;

/*
 * Start a new FileWatcher thread for the given dataset
 *
 * Also add the FileWatcher thread to the map of FileWatcher threads, we propegate the retries value to the map
 * that way we can keep track of how many retries are left for a given dataset
 *
 * @param dataset_id The id of the dataset to start a FileWatcher thread for
 * @param retries The number of retries left for the FileWatcher thread
 */
void FileWatcherWatchdog::start_file_watcher_thread(int64_t dataset_id, int16_t retries) {
  SPDLOG_INFO("Starting FileWatcher thread for dataset {}", dataset_id);
  // Start a new child thread of a FileWatcher
  file_watcher_thread_stop_flags_.emplace(dataset_id, false);
  std::unique_ptr<FileWatcher> file_watcher =
      std::make_unique<FileWatcher>(config_, dataset_id, &file_watcher_thread_stop_flags_[dataset_id],
                                    config_["storage"]["insertion_threads"].as<int16_t>());
  std::thread th(&FileWatcher::run, std::move(file_watcher));
  file_watcher_threads_[dataset_id] = std::move(th);
  file_watcher_dataset_retries_[dataset_id] = retries;
}

/*
 * Stop a FileWatcher thread for the given dataset
 *
 * Also remove the FileWatcher thread from the map of FileWatcher threads
 *
 * @param dataset_id The id of the dataset to start a FileWatcher thread for
 */
void FileWatcherWatchdog::stop_file_watcher_thread(int64_t dataset_id) {
  SPDLOG_INFO("Stopping FileWatcher thread for dataset {}", dataset_id);
  if (file_watcher_threads_.contains(dataset_id)) {
    // Set the stop flag for the FileWatcher thread
    file_watcher_thread_stop_flags_[dataset_id].store(true);
    // Wait for the FileWatcher thread to stop
    if (file_watcher_threads_[dataset_id].joinable()) {
      file_watcher_threads_[dataset_id].join();
    }
    auto file_watcher_thread_it = file_watcher_threads_.find(dataset_id);
    file_watcher_threads_.erase(file_watcher_thread_it);

    auto file_watcher_dataset_retries_it = file_watcher_dataset_retries_.find(dataset_id);
    file_watcher_dataset_retries_.erase(file_watcher_dataset_retries_it);

    auto file_watcher_thread_stop_flags_it = file_watcher_thread_stop_flags_.find(dataset_id);
    file_watcher_thread_stop_flags_.erase(file_watcher_thread_stop_flags_it);
  } else {
    SPDLOG_ERROR("FileWatcher thread for dataset {} not found", dataset_id);
  }
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
    for (auto& file_watcher_thread_flag : file_watcher_thread_stop_flags_) {
      file_watcher_thread_flag.second.store(true);
    }
    for (auto& file_watcher_thread : file_watcher_threads_) {
      file_watcher_thread.second.join();
    }
    file_watcher_threads_.clear();
    file_watcher_dataset_retries_.clear();
    file_watcher_thread_stop_flags_.clear();
    return;
  }

  std::vector<int64_t> dataset_ids(number_of_datasets);
  session << "SELECT dataset_id FROM datasets", soci::into(dataset_ids);

  std::vector<int64_t> const running_file_watcher_threads = get_running_file_watcher_threads();
  for (const auto& dataset_id : running_file_watcher_threads) {
    if (std::find(dataset_ids.begin(), dataset_ids.end(), dataset_id) == dataset_ids.end()) {
      // There is a FileWatcher thread running for a dataset that was deleted
      // from the database. Stop the thread.
      stop_file_watcher_thread(dataset_id);
    }
  }

  for (const auto& dataset_id : dataset_ids) {
    if (file_watcher_dataset_retries_[dataset_id] > 2) {
      // There have been more than 3 restart attempts for this dataset, we are not going to try again
    } else if (!file_watcher_threads_.contains(dataset_id)) {
      // There is no FileWatcher thread registered for this dataset. Start one.
      start_file_watcher_thread(dataset_id, 0);
    } else if (!file_watcher_threads_[dataset_id].joinable()) {
      // The FileWatcher thread is not running. Start it.
      start_file_watcher_thread(dataset_id, file_watcher_dataset_retries_[dataset_id]);
      file_watcher_dataset_retries_[dataset_id] += 1;
    }
  }
}

void FileWatcherWatchdog::run() {
  SPDLOG_INFO("FileWatchdog running");

  while (true) {
    if (stop_file_watcher_watchdog_->load()) {
      break;
    }
    watch_file_watcher_threads();
    std::this_thread::sleep_for(std::chrono::seconds(file_watcher_watchdog_sleep_time_s_));
  }
  for (auto& file_watcher_thread_flag : file_watcher_thread_stop_flags_) {
    file_watcher_thread_flag.second.store(true);
  }
  for (auto& file_watcher_thread : file_watcher_threads_) {
    file_watcher_thread.second.join();
  }
  stop_file_watcher_watchdog_->store(true);
}

std::vector<int64_t> FileWatcherWatchdog::get_running_file_watcher_threads() {
  std::vector<int64_t> running_file_watcher_threads;
  for (const auto& pair : file_watcher_threads_) {
    if (pair.second.joinable()) {
      running_file_watcher_threads.push_back(pair.first);
    }
  }
  return running_file_watcher_threads;
}