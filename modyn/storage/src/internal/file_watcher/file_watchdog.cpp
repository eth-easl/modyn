#include "internal/file_watcher/file_watchdog.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>

#include "internal/database/storage_database_connection.hpp"
#include "soci/soci.h"

using namespace storage;

/*
 * Start a new FileWatcher process for the given dataset
 *
 * Also add the FileWatcher process to the map of FileWatcher processes, we propegate the retries value to the map
 * that way we can keep track of how many retries are left for a given dataset
 *
 * @param dataset_id The id of the dataset to start a FileWatcher process for
 * @param retries The number of retries left for the FileWatcher process
 */
void FileWatchdog::start_file_watcher_process(int64_t dataset_id, int16_t retries) {
  SPDLOG_INFO("Starting FileWatcher process for dataset {}", dataset_id);
  // Start a new child process of a FileWatcher
  file_watcher_process_stop_flags_.emplace(dataset_id, false);
  std::shared_ptr<FileWatcher> file_watcher =
      std::make_shared<FileWatcher>(config_, dataset_id, &file_watcher_process_stop_flags_[dataset_id], config_["storage"]["insertion_threads"].as<int16_t>());
  std::thread th(&FileWatcher::run, file_watcher);
  file_watcher_processes_[dataset_id] = std::move(th);
  file_watcher_process_retries_[dataset_id] = retries;
}

/*
 * Stop a FileWatcher process for the given dataset
 *
 * Also remove the FileWatcher process from the map of FileWatcher processes
 *
 * In case of a test we don't want to remove the FileWatcher process from the map, this way we can fake kill the thread
 *
 * @param dataset_id The id of the dataset to start a FileWatcher process for
 * @param is_test Whether or not this method use is a test
 */
void FileWatchdog::stop_file_watcher_process(int64_t dataset_id, bool is_test) {
  SPDLOG_INFO("Stopping FileWatcher process for dataset {}", dataset_id);
  if (file_watcher_processes_.count(dataset_id) == 1) {
    // Set the stop flag for the FileWatcher process
    file_watcher_process_stop_flags_[dataset_id].store(true);
    // Wait for the FileWatcher process to stop
    if (file_watcher_processes_[dataset_id].joinable()) {
      file_watcher_processes_[dataset_id].join();
    }
    if (!is_test) {
      // Remove the FileWatcher process from the map, unless this is a test (we want to be able to fake kill the thread
      // to test the watchdog)
      std::unordered_map<int64_t, std::thread>::iterator file_watcher_process_it;
      file_watcher_process_it = file_watcher_processes_.find(dataset_id);
      file_watcher_processes_.erase(file_watcher_process_it);

      std::unordered_map<int64_t, int16_t>::iterator file_watcher_process_retries_it;
      file_watcher_process_retries_it = file_watcher_process_retries_.find(dataset_id);
      file_watcher_process_retries_.erase(file_watcher_process_retries_it);

      std::unordered_map<int64_t, std::atomic<bool>>::iterator file_watcher_process_stop_flags_it;
      file_watcher_process_stop_flags_it = file_watcher_process_stop_flags_.find(dataset_id);
      file_watcher_process_stop_flags_.erase(file_watcher_process_stop_flags_it);
    }
  } else {
    SPDLOG_ERROR("FileWatcher process for dataset {} not found", dataset_id);
  }
}

/*
 * Watch the FileWatcher processes and start/stop them as needed
 *
 * @param storage_database_connection The StorageDatabaseConnection object to use for database queries
 */
void FileWatchdog::watch_file_watcher_processes(  // NOLINT (readability-convert-member-functions-to-static)
    StorageDatabaseConnection* storage_database_connection) {
  if (storage_database_connection == nullptr) {
    SPDLOG_ERROR("StorageDatabaseConnection is null");
    throw std::runtime_error("StorageDatabaseConnection is null");
  }
  soci::session session = storage_database_connection->get_session();
  int64_t number_of_datasets = 0;
  session << "SELECT COUNT(dataset_id) FROM datasets", soci::into(number_of_datasets);
  if (number_of_datasets == 0) {
    // There are no datasets in the database. Stop all FileWatcher processes.
    try {
      std::vector<int64_t> running_file_watcher_processes = get_running_file_watcher_processes();
      for (const auto& dataset_id : running_file_watcher_processes) {
        stop_file_watcher_process(dataset_id);
      }
    } catch (const std::runtime_error& e) {
      SPDLOG_ERROR("Error stopping FileWatcher process: {}", e.what());
    }
    return;
  }
  std::vector<int64_t> dataset_ids = std::vector<int64_t>(number_of_datasets);
  session << "SELECT dataset_id FROM datasets", soci::into(dataset_ids);

  std::vector<int64_t> running_file_watcher_processes = get_running_file_watcher_processes();
  for (const auto& dataset_id : running_file_watcher_processes) {
    if (std::find(dataset_ids.begin(), dataset_ids.end(), dataset_id) == dataset_ids.end()) {
      // There is a FileWatcher process running for a dataset that was deleted
      // from the database. Stop the process.
      try {
        stop_file_watcher_process(dataset_id);
      } catch (const std::runtime_error& e) {
        SPDLOG_ERROR("Error stopping FileWatcher process: {}", e.what());
      }
    }
  }

  for (const auto& dataset_id : dataset_ids) {
    if (file_watcher_processes_.count(  // NOLINT - cppcheck doesn't like the count() method but we need it here
            dataset_id) == 0) {
      // There is no FileWatcher process registered for this dataset. Start one.
      start_file_watcher_process(dataset_id, 0);
    } else if (file_watcher_process_retries_[dataset_id] > 2) {
      // There have been more than 3 restart attempts for this process. Stop it.
      try {
        stop_file_watcher_process(dataset_id);
      } catch (const std::runtime_error& e) {
        SPDLOG_ERROR("Error stopping FileWatcher process: {}. Trying again in the next iteration.", e.what());
      }
    } else if (!file_watcher_processes_[dataset_id].joinable()) {
      // The FileWatcher process is not running. Start it.
      start_file_watcher_process(dataset_id, file_watcher_process_retries_[dataset_id]);
      file_watcher_process_retries_[dataset_id] += 1;
    }
  }
  session.close();
}

void FileWatchdog::run() {
  StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  storage_database_connection.create_tables();

  SPDLOG_INFO("FileWatchdog running");

  while (true) {
    if (stop_file_watchdog_->load()) {
      break;
    }
    watch_file_watcher_processes(&storage_database_connection);
    // Wait for 3 seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  for (auto& file_watcher_process_flag : file_watcher_process_stop_flags_) {
    file_watcher_process_flag.second.store(true);
  }
  for (auto& file_watcher_process : file_watcher_processes_) {
    file_watcher_process.second.join();
  }
}

std::vector<int64_t> FileWatchdog::get_running_file_watcher_processes() {
  std::vector<int64_t> running_file_watcher_processes;
  for (const auto& pair : file_watcher_processes_) {
    if (pair.second.joinable()) {
      running_file_watcher_processes.push_back(pair.first);
    }
  }
  return running_file_watcher_processes;
}