#include "internal/file_watcher/file_watchdog.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>

#include "internal/database/storage_database_connection.hpp"
#include "soci/soci.h"

using namespace storage;

void FileWatchdog::start_file_watcher_process(int64_t dataset_id, int16_t retries) {
  // Start a new child process of a FileWatcher
  std::atomic<bool> stop_file_watcher = false;
  const FileWatcher file_watcher = FileWatcher(config_, dataset_id, &stop_file_watcher);
  std::thread th(&FileWatcher::run, file_watcher);
  file_watcher_processes_[dataset_id] = std::tuple(std::move(th), retries, &stop_file_watcher);
}

void FileWatchdog::stop_file_watcher_process(int64_t dataset_id, bool is_test) {
  if (file_watcher_processes_.count(dataset_id) == 1) {
    // Set the stop flag for the FileWatcher process
    std::get<2>(file_watcher_processes_[dataset_id])->store(true);
    // Wait for the FileWatcher process to stop
    if (std::get<0>(file_watcher_processes_[dataset_id]).joinable()) {
      std::get<0>(file_watcher_processes_[dataset_id]).join();
    }
    if (!is_test) {
      // Remove the FileWatcher process from the map, unless this is a test (we want to be able to fake kill the thread
      // to test the watchdog)
      std::unordered_map<int64_t, std::tuple<std::thread, int16_t, std::atomic<bool>*>>::iterator it;
      it = file_watcher_processes_.find(dataset_id);
      file_watcher_processes_.erase(it);
    }
  } else {
    throw std::runtime_error("FileWatcher process not found");
  }
}

void FileWatchdog::watch_file_watcher_processes(StorageDatabaseConnection* storage_database_connection) {
  soci::session session = storage_database_connection->get_session();
  int64_t number_of_datasets = 0;
  session << "SELECT COUNT(dataset_id) FROM datasets", soci::into(number_of_datasets);
  if (number_of_datasets == 0) {
    // There are no datasets in the database. Stop all FileWatcher processes.
    for (const auto& pair : file_watcher_processes_) {
      stop_file_watcher_process(pair.first);
    }
    return;
  }
  std::vector<int64_t> dataset_ids = std::vector<int64_t>(number_of_datasets);
  session << "SELECT dataset_id FROM datasets", soci::into(dataset_ids);

  int64_t dataset_id;
  for (const auto& pair : file_watcher_processes_) {
    dataset_id = pair.first;
    if (std::find(dataset_ids.begin(), dataset_ids.end(), dataset_id) == dataset_ids.end()) {
      // There is a FileWatcher process running for a dataset that was deleted
      // from the database. Stop the process.
      stop_file_watcher_process(dataset_id);
    }
  }

  for (const auto& dataset_id : dataset_ids) {
    if (std::get<2>(file_watcher_processes_[dataset_id]) == nullptr) {
      // There is no FileWatcher process registered for this dataset. Start one.
      start_file_watcher_process(dataset_id, 0);
    } else if (std::get<1>(file_watcher_processes_[dataset_id]) > 2) {
      // There have been more than 3 restart attempts for this process. Stop it.
      stop_file_watcher_process(dataset_id);
    } else if (!std::get<0>(file_watcher_processes_[dataset_id]).joinable()) {
      // The FileWatcher process is not running. Start it.
      start_file_watcher_process(dataset_id, std::get<1>(file_watcher_processes_[dataset_id]));
      std::get<1>(file_watcher_processes_[dataset_id]) += 1;
    }
  }
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
  for (auto& file_watcher_process : file_watcher_processes_) {
    std::get<2>(file_watcher_process.second)->store(true);
  }
}

std::vector<int64_t> FileWatchdog::get_running_file_watcher_processes() {
  std::vector<int64_t> running_file_watcher_processes;
  for (const auto& pair : file_watcher_processes_) {
    if (std::get<0>(pair.second).joinable()) {
      running_file_watcher_processes.push_back(pair.first);
    }
  }
  return running_file_watcher_processes;
}