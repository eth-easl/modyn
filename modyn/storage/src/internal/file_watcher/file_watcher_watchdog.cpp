#include "internal/file_watcher/file_watcher_watchdog.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>

#include "soci/soci.h"

using namespace storage;

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
  std::shared_ptr<FileWatcher> file_watcher =
      std::make_shared<FileWatcher>(config_, dataset_id, &file_watcher_thread_stop_flags_[dataset_id],
                                    config_["storage"]["insertion_threads"].as<int16_t>());
  std::thread th(&FileWatcher::run, file_watcher);
  file_watcher_threads_[dataset_id] = std::move(th);
  file_watcher_dataset_retries_[dataset_id] = retries;
}

/*
 * Stop a FileWatcher thread for the given dataset
 *
 * Also remove the FileWatcher thread from the map of FileWatcher threads
 *
 * In case of a test we don't want to remove the FileWatcher thread from the map, this way we can fake kill the thread
 *
 * @param dataset_id The id of the dataset to start a FileWatcher thread for
 * @param is_test Whether or not this method use is a test
 */
void FileWatchdog::stop_file_watcher_thread(int64_t dataset_id, bool is_test) {
  SPDLOG_INFO("Stopping FileWatcher thread for dataset {}", dataset_id);
  if (file_watcher_threads_.count(dataset_id) == 1) {
    // Set the stop flag for the FileWatcher thread
    file_watcher_thread_stop_flags_[dataset_id].store(true);
    // Wait for the FileWatcher thread to stop
    if (file_watcher_threads_[dataset_id].joinable()) {
      file_watcher_threads_[dataset_id].join();
    }
    if (!is_test) {
      // Remove the FileWatcher thread from the map, unless this is a test (we want to be able to fake kill the thread
      // to test the watchdog)
      std::unordered_map<int64_t, std::thread>::iterator file_watcher_thread_it;
      file_watcher_thread_it = file_watcher_threads_.find(dataset_id);
      file_watcher_threads_.erase(file_watcher_thread_it);

      std::unordered_map<int64_t, int16_t>::iterator file_watcher_dataset_retries_it;
      file_watcher_dataset_retries_it = file_watcher_dataset_retries_.find(dataset_id);
      file_watcher_dataset_retries_.erase(file_watcher_dataset_retries_it);

      std::unordered_map<int64_t, std::atomic<bool>>::iterator file_watcher_thread_stop_flags_it;
      file_watcher_thread_stop_flags_it = file_watcher_thread_stop_flags_.find(dataset_id);
      file_watcher_thread_stop_flags_.erase(file_watcher_thread_stop_flags_it);
    }
  } else {
    SPDLOG_ERROR("FileWatcher thread for dataset {} not found", dataset_id);
  }
}

/*
 * Watch the FileWatcher threads and start/stop them as needed
 */
void FileWatchdog::watch_file_watcher_threads() {
  if (storage_database_connection_ == nullptr) {
    SPDLOG_ERROR("StorageDatabaseConnection is null");
    throw std::runtime_error("StorageDatabaseConnection is null");
  }
  soci::session session = storage_database_connection_->get_session();
  int64_t number_of_datasets = 0;
  session << "SELECT COUNT(dataset_id) FROM datasets", soci::into(number_of_datasets);
  if (number_of_datasets == 0) {
    // There are no datasets in the database. Stop all FileWatcher threads.
    std::vector<int64_t> running_file_watcher_threads = get_running_file_watcher_threads();
    for (const auto& dataset_id : running_file_watcher_threads) {
      stop_file_watcher_thread(dataset_id);
    }
    return;
  }
  std::vector<int64_t> dataset_ids = std::vector<int64_t>(number_of_datasets);
  session << "SELECT dataset_id FROM datasets", soci::into(dataset_ids);

  std::vector<int64_t> running_file_watcher_threads = get_running_file_watcher_threads();
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
    } else if (!std::map::contains(file_watcher_threads_, dataset_id)) {
      // There is no FileWatcher thread registered for this dataset. Start one.
      start_file_watcher_thread(dataset_id, 0);
    } else if (!file_watcher_threads_[dataset_id].joinable()) {
      // The FileWatcher thread is not running. Start it.
      start_file_watcher_thread(dataset_id, file_watcher_dataset_retries_[dataset_id]);
      file_watcher_dataset_retries_[dataset_id] += 1;
    }
  }
}

void FileWatchdog::run() {
  SPDLOG_INFO("FileWatchdog running");

  while (true) {
    if (stop_file_watcher_watchdog_->load()) {
      break;
    }
    watch_file_watcher_threads();
    // Wait for 3 seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }
  for (auto& file_watcher_thread_flag : file_watcher_thread_stop_flags_) {
    file_watcher_thread_flag.second.store(true);
  }
  for (auto& file_watcher_thread : file_watcher_threads_) {
    file_watcher_thread.second.join();
  }
}

std::vector<int64_t> FileWatchdog::get_running_file_watcher_threads() {
  std::vector<int64_t> running_file_watcher_threads;
  for (const auto& pair : file_watcher_threads_) {
    if (pair.second.joinable()) {
      running_file_watcher_threads.push_back(pair.first);
    }
  }
  return running_file_watcher_threads;
}