#include "internal/file_watcher/file_watchdog.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>

#include "internal/database/storage_database_connection.hpp"
#include "soci/soci.h"

using namespace storage;

void FileWatchdog::start_file_watcher_process(long long dataset_id) {
  SPDLOG_INFO("Start FileWatcher process for dataset {}", dataset_id);
  // Start a new child process of a FileWatcher
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher *file_watcher = new FileWatcher(this->config_file, dataset_id, false, stop_file_watcher);
  std::thread th(&FileWatcher::run, file_watcher);
  this->file_watcher_processes[dataset_id] = std::tuple(std::move(th), 0, stop_file_watcher);
}

void FileWatchdog::stop_file_watcher_process(long long dataset_id) {
  SPDLOG_INFO("Stop FileWatcher process for dataset {}", dataset_id);
  if (this->file_watcher_processes.count(dataset_id) == 1) {
    // Set the stop flag for the FileWatcher process
    std::get<2>(this->file_watcher_processes[dataset_id]).get()->store(true);
    // Wait for the FileWatcher process to stop
    std::get<0>(this->file_watcher_processes[dataset_id]).join();
    std::unordered_map<long long, std::tuple<std::thread, int, std::shared_ptr<std::atomic<bool>>>>::iterator it;
    it = this->file_watcher_processes.find(dataset_id);
    this->file_watcher_processes.erase(it);
  } else {
    throw std::runtime_error("FileWatcher process not found");
  }
}

void FileWatchdog::watch_file_watcher_processes(StorageDatabaseConnection* storage_database_connection) {
  SPDLOG_INFO("Watch FileWatcher processes");
  soci::session* sql = storage_database_connection->get_session();
  int number_of_datasets = 0;
  *sql << "SELECT COUNT(dataset_id) FROM datasets", soci::into(number_of_datasets);
  SPDLOG_INFO("Number of datasets: {}", number_of_datasets);
  if (number_of_datasets == 0) {
    // There are no datasets in the database. Stop all FileWatcher processes.
    for (const auto& pair : this->file_watcher_processes) {
      this->stop_file_watcher_process(pair.first);
    }
    return;
  }
  std::vector<long long> dataset_ids = std::vector<long long>(number_of_datasets);
  *sql << "SELECT dataset_id FROM datasets", soci::into(dataset_ids);

  SPDLOG_INFO("Number of FileWatcher processes: {}", this->file_watcher_processes.size());
  SPDLOG_INFO("Number of datasets: {}", dataset_ids.size());
  long long dataset_id;
  for (const auto& pair : this->file_watcher_processes) {
    dataset_id = pair.first;
    if (std::find(dataset_ids.begin(), dataset_ids.end(), dataset_id) == dataset_ids.end()) {
      // There is a FileWatcher process running for a dataset that was deleted
      // from the database. Stop the process.
      this->stop_file_watcher_process(dataset_id);
    }
  }

  for (const auto& dataset_id : dataset_ids) {
    SPDLOG_INFO("Dataset ID: {}", dataset_id);
    if (this->file_watcher_processes.find(dataset_id) == this->file_watcher_processes.end()) {
      // There is no FileWatcher process running for this dataset. Start one.
      this->start_file_watcher_process(dataset_id);
    }

    if (std::get<1>(this->file_watcher_processes[dataset_id]) > 3) {
      // There have been more than 3 restart attempts for this process. Stop it.
      this->stop_file_watcher_process(dataset_id);
    } else if (std::get<0>(this->file_watcher_processes[dataset_id]).joinable()) {
      // The process is not running. Start it.
      this->start_file_watcher_process(dataset_id);
      std::get<1>(this->file_watcher_processes[dataset_id])++;
    } else {
      // The process is running. Reset the restart attempts counter.
      std::get<1>(this->file_watcher_processes[dataset_id]) = 0;
    }
  }
}

void FileWatchdog::run() {
  StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(this->config);
  storage_database_connection.create_tables();

  SPDLOG_INFO("FileWatchdog running");

  while (true) {
    if (this->stop_file_watchdog.get()->load()) {
      break;
    }
    this->watch_file_watcher_processes(&storage_database_connection);
    // Wait for 3 seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  for (auto& file_watcher_process : this->file_watcher_processes) {
    std::get<2>(file_watcher_process.second).get()->store(true);
  }
}

std::vector<long long> FileWatchdog::get_running_file_watcher_processes() {
  std::vector<long long> running_file_watcher_processes;
  for (const auto& pair : this->file_watcher_processes) {
    if (std::get<0>(pair.second).joinable()) {
      running_file_watcher_processes.push_back(pair.first);
    }
  }
  return running_file_watcher_processes;
}